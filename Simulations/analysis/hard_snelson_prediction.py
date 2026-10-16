#!/usr/bin/env python3
"""Reproduce the frozen Snelson fit and draw hard-block predictions separately.

The historical Snelson release saved its figure and manifest but no per-model
parameters.  This script reruns that pinned training protocol, checks its final
objectives against the committed manifest, and writes a separate prediction
figure and numerical record.  It never writes into the historical checkout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

import hard_block_prediction_campaign as hard
import real_predictor_campaign as real


def predict(model, name, x_train, x_target, block_size, sn, *, train=False):
    import jax
    import jax.numpy as jnp

    if name == "SGPR":
        mean, variance = sn.prediction_for_plot(model, x_train, x_target, name)
        return mean, variance, {"partition": "none", "minimum_raw_residual": None}

    components = sn.gpflow_predictive_components(model, x_train, x_target)
    if name == "T-SGPR":
        x, z, ell = sn.current_structure_inputs(model)
        blocks = [np.asarray(indices, dtype=np.int32) for coeff, indices in
                  sn.sbt_terms_host(x, z, ell, block_size,
                                    split_mode="boundary") if coeff > 0]
        source = "prediction-time-inducing-anchor"
    else:
        blocks = [np.asarray(indices, dtype=np.int32)
                  for indices in sn.model_blocks(model)]
        source = "fitted-blocks"
        _, _, ell = sn.current_structure_inputs(model)

    params = {"log_ell": jnp.log(jnp.asarray(ell))}
    representatives = hard.block_representatives(
        components, blocks, params, kernel_name="rbf")
    if train:
        assignment = np.empty(len(x_train), dtype=np.int32)
        for block_id, indices in enumerate(blocks):
            assignment[indices] = block_id
    else:
        assignment = hard.assign_test_points(
            components, representatives, test_chunk=512)
    correction, timings = hard.hard_correction(
        components, blocks, name, assignment, test_chunk=512,
        eig_cutoff_scale=10.0)
    residual = np.array(jax.device_get(components["d_test"]),
                        dtype=np.float64, copy=True).reshape(-1)
    residual -= correction
    variance = (np.asarray(jax.device_get(components["qvar"])).reshape(-1)
                + residual + float(jax.device_get(components["s2"])))
    if not np.isfinite(variance).all() or np.min(variance) <= 0:
        raise ValueError(f"invalid hard variance for {name}")
    if np.min(residual) < -1e-7:
        raise ValueError(f"negative conditional residual for {name}")
    mean = np.asarray(jax.device_get(components["mean"])).reshape(-1)
    return mean, variance, {
        "partition": source,
        "block_sizes": [len(indices) for indices in blocks],
        "representative_indices": representatives.tolist(),
        "assigned_counts": np.bincount(assignment, minlength=len(blocks)).tolist(),
        "minimum_raw_residual": float(np.min(residual)),
        **timings,
    }


def draw(sn, output: Path, x_train, y_train, x_plot, models, losses,
         exact_sgpr_loss, exact_model, predicted):
    fig, axes = sn.plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 4]})
    ax = axes[0]
    sn.add_exact_sgpr_boundary(ax, exact_sgpr_loss, len(x_train))
    for name, values in losses.items():
        ax.plot(np.arange(len(values)), np.maximum(values / len(x_train), 1e-12),
                color=sn.method_color(name), label=name, linewidth=1.5)
    ax.set(xlabel="iteration", ylabel="negative lower bound / N")
    ax.set_yscale("log")
    ax.legend(fontsize=8.0, framealpha=0.8, loc="upper right")

    ax = axes[1]
    ax.plot(x_train[:, 0], y_train[:, 0], "xk", markersize=4, alpha=0.75)
    if exact_model is not None:
        sn.plot_exact_gp(ax, exact_model, x_plot)
    for name, model in models.items():
        mean = predicted[name]["mean"]
        variance = predicted[name]["variance"]
        color = sn.method_color(name)
        ax.plot(x_plot[:, 0], mean, color=color,
                linewidth=sn.method_linewidth(name), label=name)
        spread = 2.0 * np.sqrt(variance)
        for bound in (mean - spread, mean + spread):
            ax.plot(x_plot[:, 0], bound, color=color,
                    linewidth=1.0,
                    linestyle=sn.UNCERTAINTY_DASHES, alpha=0.78)
        z = model.inducing_variable.Z.numpy()
        ax.plot(z[:, 0], np.full(len(z), 1.78), "o",
                color=color, markersize=5)
    ax.set(xlabel="x", ylabel="y")
    ax.set_xlim(float(x_plot[0, 0]), float(x_plot[-1, 0]))
    ax.set_ylim(-3.0, 2.0)
    sn.add_parameter_text(ax, models, exact_model=exact_model)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", dpi=300, pad_inches=0.05)
    sn.plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-manifest", type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    real.guard_separate(output, [source])
    expected_path = (args.expected_manifest.resolve() if args.expected_manifest
                     else source / "Report/Results/snelson_sgpr_t_bt_sbt_manifest.json")
    expected = json.loads(expected_path.read_text())
    commit = expected["code"]["commit"]
    import subprocess
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError(f"Snelson source commit {actual} differs from {commit}")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source / "Simulations"))
    from sgpr.snelson import run_snelson as sn

    original = sn.save_figure
    original_summary = sn.write_summary
    original_table = sn.write_latex_table
    original_manifest = sn.write_release_manifest
    hard_results = {}
    fit_objectives = {}

    def capture(x_train, y_train, x_plot, models, losses,
                exact_sgpr_loss=None, report_pdf=None, exact_model=None):
        original(x_train, y_train, x_plot, models, losses,
                 exact_sgpr_loss=exact_sgpr_loss,
                 report_pdf=output / "snelson_sgpr_t_bt_sbt.pdf",
                 exact_model=exact_model)
        block_size = sn.block_size_from_n_blocks(len(x_train), sn.N_BLOCKS)
        plotted = {}
        for name, model in models.items():
            mean, variance, info = predict(
                model, name, x_train, x_plot, block_size, sn)
            train_mean, train_variance, train_info = predict(
                model, name, x_train, x_train, block_size, sn, train=True)
            rmse, nll = sn.pr.gaussian_rmse_nll(
                train_mean, train_variance, np.asarray(y_train).reshape(-1))
            plotted[name] = {"mean": mean, "variance": variance}
            hard_results[name] = {
                "in_sample_rmse": float(rmse), "in_sample_nll": float(nll),
                "plot_diagnostics": info, "train_diagnostics": train_info,
                "kernel_variance": float(model.kernel.variance.numpy()),
                "lengthscales": np.asarray(
                    model.kernel.lengthscales.numpy()).reshape(-1).tolist(),
                "noise_variance": float(model.likelihood.variance.numpy()),
                "inducing_points": np.asarray(
                    model.inducing_variable.Z.numpy()).tolist(),
            }
            fit_objectives[name] = float(losses[name][-1] / len(x_train))
        fit_objectives["E-SGPR"] = float(exact_sgpr_loss / len(x_train))
        draw(sn, output / "snelson_hard_prediction.pdf", x_train, y_train,
             x_plot, models, losses, exact_sgpr_loss, exact_model, plotted)

    def write_summary(*values, **keywords):
        values = list(values)
        values[4] = output / "snelson_summary_sgpr_t_bt_sbt.csv"
        return original_summary(*values, **keywords)

    def write_table(*values, **keywords):
        values = list(values)
        values[4] = output / "snelson_table_sgpr_t_bt_sbt.tex"
        return original_table(*values, **keywords)

    def write_manifest(*values, **keywords):
        values = list(values)
        values[0] = output / "snelson_sgpr_t_bt_sbt_manifest.json"
        values[-1] = (output / "snelson_sgpr_t_bt_sbt.pdf",
                      output / "snelson_summary_sgpr_t_bt_sbt.csv",
                      output / "snelson_table_sgpr_t_bt_sbt.tex")
        return original_manifest(*values, **keywords)

    sn.save_figure = capture
    sn.write_summary = write_summary
    sn.write_latex_table = write_table
    sn.write_release_manifest = write_manifest
    previous_argv = sys.argv
    try:
        sys.argv = [str(source / "Simulations/sgpr/snelson/run_snelson.py")]
        sn.main()
    finally:
        sys.argv = previous_argv
        sn.save_figure = original
        sn.write_summary = original_summary
        sn.write_latex_table = original_table
        sn.write_release_manifest = original_manifest

    generated = json.loads((output / "snelson_sgpr_t_bt_sbt_manifest.json").read_text())
    fit_objectives["True GP"] = generated["final_negative_bound_per_datum"]["True GP"]
    deviations = {name: abs(value - expected["final_negative_bound_per_datum"][name])
                  for name, value in fit_objectives.items()}
    fit_matched = all(delta <= 1e-4 for delta in deviations.values())
    result = {
        "implementation": "snelson-hard-pic-prediction-v1",
        "historical_commit": commit,
        "expected_manifest_sha256": real.sha256(expected_path),
        "historical_figure_sha256": real.sha256(
            output / "snelson_sgpr_t_bt_sbt.pdf"),
        "hard_figure_sha256": real.sha256(
            output / "snelson_hard_prediction.pdf"),
        "fit_objective_deviations": deviations,
        "fit_reproduced_within_1e-4_per_datum": fit_matched,
        "methods": hard_results,
        "caveat": "Snelson has no held-out split; RMSE/NLL are in-sample",
    }
    real.atomic_json(output / "snelson_hard_prediction.json", result)
    print("SNELSON_HARD_FIT_REPRODUCED=" + str(fit_matched), flush=True)
    print(f"SNELSON_HARD_PDF={output / 'snelson_hard_prediction.pdf'}", flush=True)
    return 0 if fit_matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
