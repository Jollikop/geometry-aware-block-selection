#!/usr/bin/env python3
"""Exploratory SBT-hyperparameter transplant into SGPR on Snelson.

SBT first optimizes kernel parameters, noise and inducing locations under its
own block bound.  At the fitted SBT state we evaluate both the all-block SBT
predictor and the standard sparse SGPR predictor.  We then freeze SBT's kernel
and noise parameters and optimize only its inducing locations under the SGPR
trace bound.  The output is deliberately separate from official paper figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "mpl_cache"),
)

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parents[1]

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from common import predictors as pr  # noqa: E402
from common import terms as tm  # noqa: E402
from common.data_split import (  # noqa: E402
    apply_standardizer,
    fit_standardizer,
    load_snelson,
    to_jax,
)
from sgpr import benchmark as bench  # noqa: E402
from sgpr import jax_collapsed as jm  # noqa: E402
from sgpr.analysis.explore_fixed_theta_bounds import (  # noqa: E402
    make_method_components,
    scalar,
    train_method,
)
from sgpr.analysis.explore_hyperparameter_transplant import (  # noqa: E402
    transplant_to_sgpr,
)
from sgpr.analysis.explore_sbt_guided_sgpr import (  # noqa: E402
    dummy_terms,
    params_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--n-blocks", type=int, default=10)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--transplant-steps", type=int, default=60)
    parser.add_argument(
        "--refresh-schedule",
        default="2,3,5,20,30",
        help="comma-separated SBT L-BFGS phase lengths",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            HERE.parent
            / "exploratory_results"
            / "snelson_sbt_transplant"
        ),
    )
    return parser.parse_args()


def parse_schedule(value: str, total: int) -> list[int]:
    phases = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not phases or any(item <= 0 for item in phases):
        raise ValueError("refresh schedule must contain positive integers")
    if sum(phases) != total:
        raise ValueError(
            f"refresh schedule sums to {sum(phases)}, expected {total}"
        )
    return phases


def raw_prediction(prediction, standardizer):
    mean, variance = prediction
    y_mean = float(standardizer["y_mean"])
    y_std = float(standardizer["y_std"])
    mean = np.asarray(jax.device_get(mean)).reshape(-1) * y_std + y_mean
    variance = np.asarray(jax.device_get(variance)).reshape(-1) * y_std**2
    return mean, np.maximum(variance, 1e-12)


def metrics(mean, variance, y):
    y = np.asarray(y).reshape(-1)
    rmse = float(np.sqrt(np.mean((y - mean) ** 2)))
    nll = float(
        np.mean(0.5 * (np.log(2.0 * np.pi * variance) + (y - mean) ** 2 / variance))
    )
    return rmse, nll


def interval(ax, x, mean, variance, *, color, label, linestyle="-", fill=True):
    std = np.sqrt(variance)
    if fill:
        ax.fill_between(
            x, mean - 1.96 * std, mean + 1.96 * std,
            color=color, alpha=0.16, linewidth=0,
        )
    else:
        ax.plot(x, mean - 1.96 * std, color=color, linewidth=1.2, linestyle=linestyle)
        ax.plot(x, mean + 1.96 * std, color=color, linewidth=1.2, linestyle=linestyle)
    ax.plot(x, mean, color=color, linewidth=2.0, linestyle=linestyle, label=label)


def main() -> None:
    args = parse_args()
    schedule = parse_schedule(args.refresh_schedule, args.steps)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y_raw = load_snelson()
    standardizer = fit_standardizer(x_raw, y_raw)
    x_scaled, y_scaled = apply_standardizer(x_raw, y_raw, standardizer)
    x_train, y_train = to_jax(x_scaled, y_scaled)
    y_train = y_train.reshape(-1)

    block_size = int(np.ceil(len(x_raw) / args.n_blocks))
    initial = bench.init_params(x_train, args.m, args.seed, kernel_name="rbf")
    losses = {}
    builders = {}
    for method in ("SGPR", "SBT-SGPR"):
        losses[method], builders[method] = make_method_components(
            method,
            x_train,
            y_train,
            "rbf",
            block_size,
            args.seed,
            args.m,
        )

    print("Training direct SGPR ...", flush=True)
    sgpr, sgpr_curve, sgpr_info, _, sgpr_search_s, sgpr_wall_s = train_method(
        "SGPR",
        initial,
        losses=losses,
        builders=builders,
        steps=args.steps,
        refresh_schedule=schedule,
    )

    print("Training SBT donor ...", flush=True)
    sbt, sbt_curve, sbt_info, sbt_terms, sbt_search_s, sbt_wall_s = train_method(
        "SBT-SGPR",
        initial,
        losses=losses,
        builders=builders,
        steps=args.steps,
        refresh_schedule=schedule,
    )
    blocks = tm.unpack_terms(sbt_terms)
    if len(blocks) != args.n_blocks:
        raise RuntimeError(f"expected {args.n_blocks} blocks, found {len(blocks)}")

    print("Transplanting SBT theta into SGPR and optimizing only Z ...", flush=True)
    transplanted, transplant_curve, transplant_info, transplant_wall_s = (
        transplant_to_sgpr(sbt, losses["SGPR"], args.transplant_steps)
    )

    print("Fine-tuning all SBT parameters under the SGPR objective ...", flush=True)
    finetuned, finetune_curve, finetune_info, _, _, finetune_wall_s = train_method(
        "SGPR",
        sbt,
        losses=losses,
        builders=builders,
        steps=args.transplant_steps,
        refresh_schedule=[args.transplant_steps],
    )

    x_plot_raw = np.linspace(-2.0, 8.0, 500).reshape(-1, 1)
    x_plot_scaled = (x_plot_raw - standardizer["x_mean"]) / standardizer["x_std"]
    x_plot = jnp.asarray(x_plot_scaled, dtype=jnp.float64)

    predictions = {
        "SGPR direct": raw_prediction(
            jm.predict_jax(sgpr, x_train, y_train, x_plot, kernel_name="rbf"),
            standardizer,
        ),
        "SBT complete": raw_prediction(
            jm.predict_collapsed(
                sbt,
                x_train,
                y_train,
                x_plot,
                terms=sbt_terms,
                kernel_name="rbf",
                predictor='full', blocks=blocks,
            ),
            standardizer,
        ),
        "SBT as sparse SGPR": raw_prediction(
            jm.predict_jax(sbt, x_train, y_train, x_plot, kernel_name="rbf"),
            standardizer,
        ),
        "SBT -> SGPR": raw_prediction(
            jm.predict_jax(
                transplanted, x_train, y_train, x_plot, kernel_name="rbf"
            ),
            standardizer,
        ),
        "SBT -> SGPR full fine-tune": raw_prediction(
            jm.predict_jax(
                finetuned, x_train, y_train, x_plot, kernel_name="rbf"
            ),
            standardizer,
        ),
    }

    train_predictions = {
        "SGPR direct": raw_prediction(
            jm.predict_jax(sgpr, x_train, y_train, x_train, kernel_name="rbf"),
            standardizer,
        ),
        "SBT complete": raw_prediction(
            jm.predict_collapsed(
                sbt,
                x_train,
                y_train,
                x_train,
                terms=sbt_terms,
                kernel_name="rbf",
                predictor='full', blocks=blocks,
            ),
            standardizer,
        ),
        "SBT as sparse SGPR": raw_prediction(
            jm.predict_jax(sbt, x_train, y_train, x_train, kernel_name="rbf"),
            standardizer,
        ),
        "SBT -> SGPR": raw_prediction(
            jm.predict_jax(
                transplanted, x_train, y_train, x_train, kernel_name="rbf"
            ),
            standardizer,
        ),
        "SBT -> SGPR full fine-tune": raw_prediction(
            jm.predict_jax(
                finetuned, x_train, y_train, x_train, kernel_name="rbf"
            ),
            standardizer,
        ),
    }

    n = len(x_raw)
    sbt_own_elbo = -scalar(losses["SBT-SGPR"](sbt, *sbt_terms)) / n
    rows = []
    for name, params, elbo_loss, timing in (
        (
            "SGPR direct", sgpr, losses["SGPR"],
            sgpr_search_s + sgpr_wall_s,
        ),
        (
            "SBT complete", sbt, losses["SGPR"],
            sbt_search_s + sbt_wall_s,
        ),
        (
            "SBT as sparse SGPR", sbt, losses["SGPR"],
            sbt_search_s + sbt_wall_s,
        ),
        (
            "SBT -> SGPR", transplanted, losses["SGPR"],
            sbt_search_s + sbt_wall_s + transplant_wall_s,
        ),
        (
            "SBT -> SGPR full fine-tune", finetuned, losses["SGPR"],
            sbt_search_s + sbt_wall_s + finetune_wall_s,
        ),
    ):
        rmse, nll = metrics(*train_predictions[name], y_raw)
        rows.append(
            {
                "variant": name,
                "rmse_in_sample": rmse,
                "nll_in_sample": nll,
                "sgpr_elbo_per_datum": -scalar(
                    elbo_loss(params, *dummy_terms())
                ) / n,
                "sbt_elbo_per_datum": (
                    sbt_own_elbo
                    if name in ("SBT complete", "SBT as sparse SGPR")
                    else ""
                ),
                "total_training_wall_s": timing,
            }
        )

    csv_path = args.out_dir / "snelson_sbt_transplant_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(args.out_dir / "params_sgpr_direct.npz", **params_numpy(sgpr))
    np.savez_compressed(args.out_dir / "params_sbt_donor.npz", **params_numpy(sbt))
    np.savez_compressed(
        args.out_dir / "params_sbt_to_sgpr.npz", **params_numpy(transplanted)
    )
    np.savez_compressed(
        args.out_dir / "params_sbt_to_sgpr_full_finetune.npz",
        **params_numpy(finetuned),
    )
    np.savez_compressed(
        args.out_dir / "sbt_blocks.npz",
        **{f"block_{index}": block for index, block in enumerate(blocks)},
    )
    np.savez_compressed(
        args.out_dir / "optimization_curves.npz",
        sgpr=np.asarray(sgpr_curve),
        sbt=np.asarray(sbt_curve),
        transplant=np.asarray(transplant_curve),
        full_finetune=np.asarray(finetune_curve),
    )

    colors = {
        "SGPR direct": "#377EB8",
        "SBT complete": "#D62728",
        "SBT as sparse SGPR": "#E69F00",
        "SBT -> SGPR": "#168A68",
        "SBT -> SGPR full fine-tune": "#7B3294",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharex=True, sharey=True)
    x_plot_flat = x_plot_raw.reshape(-1)
    points_x = x_raw.reshape(-1)
    points_y = y_raw.reshape(-1)

    for ax in axes:
        ax.scatter(points_x, points_y, s=12, color="#333333", alpha=0.62, zorder=5)
        ax.set_xlim(-2.0, 8.0)
        ax.set_xlabel("Input")
        ax.grid(alpha=0.18, linewidth=0.7)
    axes[0].set_ylabel("Target")

    interval(
        axes[0], x_plot_flat, *predictions["SGPR direct"],
        color=colors["SGPR direct"], label="SGPR direct",
    )
    interval(
        axes[0], x_plot_flat, *predictions["SBT complete"],
        color=colors["SBT complete"], label="SBT complete (all blocks)",
    )
    interval(
        axes[0], x_plot_flat, *predictions["SBT as sparse SGPR"],
        color=colors["SBT as sparse SGPR"],
        label="SBT parameters, sparse variance", linestyle="--", fill=False,
    )
    metrics_by_name = {row["variant"]: row for row in rows}
    axes[0].set_title(
        "Before transplant\n"
        f"NLL: complete {metrics_by_name['SBT complete']['nll_in_sample']:.3f}, "
        f"sparse {metrics_by_name['SBT as sparse SGPR']['nll_in_sample']:.3f}"
    )

    interval(
        axes[1], x_plot_flat, *predictions["SGPR direct"],
        color=colors["SGPR direct"], label="SGPR direct",
    )
    interval(
        axes[1], x_plot_flat, *predictions["SBT -> SGPR"],
        color=colors["SBT -> SGPR"], label="SBT -> SGPR (Z only)",
    )
    interval(
        axes[1], x_plot_flat, *predictions["SBT -> SGPR full fine-tune"],
        color=colors["SBT -> SGPR full fine-tune"],
        label="SBT -> SGPR (full fine-tune)",
    )
    axes[1].set_title(
        "SGPR training initialized from SBT\n"
        f"NLL: direct {metrics_by_name['SGPR direct']['nll_in_sample']:.3f}, "
        f"Z-only {metrics_by_name['SBT -> SGPR']['nll_in_sample']:.3f}, "
        "full "
        f"{metrics_by_name['SBT -> SGPR full fine-tune']['nll_in_sample']:.3f}"
    )

    y_floor = min(ax.get_ylim()[0] for ax in axes)
    for ax, entries in (
        (axes[0], ((sgpr, colors["SGPR direct"]), (sbt, colors["SBT complete"]))),
        (
            axes[1],
            (
                (sgpr, colors["SGPR direct"]),
                (transplanted, colors["SBT -> SGPR"]),
                (finetuned, colors["SBT -> SGPR full fine-tune"]),
            ),
        ),
    ):
        for offset, (params, color) in enumerate(entries):
            z_raw = (
                np.asarray(jax.device_get(params["z"])).reshape(-1)
                * standardizer["x_std"][0]
                + standardizer["x_mean"][0]
            )
            ax.scatter(
                z_raw,
                np.full_like(z_raw, y_floor + 0.10 + 0.12 * offset),
                marker="|",
                s=115,
                linewidths=2.0,
                color=color,
                clip_on=False,
                zorder=7,
            )

    for ax in axes:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "Snelson SBT hyperparameter transplant (exploratory, in-sample diagnostic)",
        fontsize=11,
    )
    fig.tight_layout()
    png_path = args.out_dir / "snelson_sbt_transplant.png"
    pdf_path = args.out_dir / "snelson_sbt_transplant.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "protocol": {
            "dataset": "snelson",
            "evaluation": "in_sample_diagnostic",
            "kernel": "rbf",
            "seed": args.seed,
            "m": args.m,
            "n_blocks": args.n_blocks,
            "block_size": block_size,
            "donor_steps": args.steps,
            "refresh_schedule": schedule,
            "transplant_steps": args.transplant_steps,
            "transplant_frozen": ["log_ell", "log_sf2", "log_s2"],
            "transplant_optimized": ["z"],
            "full_finetune_initialized_from": "SBT-SGPR",
            "full_finetune_optimized": ["log_ell", "log_sf2", "log_s2", "z"],
        },
        "optimizer": {
            "sgpr": sgpr_info,
            "sbt": sbt_info,
            "transplant": transplant_info,
            "full_finetune": finetune_info,
        },
        "metrics": rows,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=float) + "\n"
    )

    print("\nvariant                      RMSE       NLL    SGPR ELBO/N")
    for row in rows:
        print(
            f"{row['variant']:<27} "
            f"{row['rmse_in_sample']:8.4f} "
            f"{row['nll_in_sample']:9.4f} "
            f"{row['sgpr_elbo_per_datum']:+12.4f}"
        )
    print(f"\nWrote: {png_path}")
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
