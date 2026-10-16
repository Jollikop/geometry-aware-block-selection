#!/usr/bin/env python3
"""Test structured bounds as hyperparameter donors for a final SGPR model.

Each donor is first trained with its own collapsed bound, jointly optimizing
kernel parameters, Gaussian noise and inducing locations.  We then report both
its bound-matched predictive NLL and the standard sparse SGPR NLL at exactly
the same fitted parameters.

For the transplant, the donor's kernel/noise parameters are frozen, its fitted
inducing locations initialize a new SGPR optimization, and only Z is optimized
under the SGPR trace bound.  The collapsed SGPR posterior is recomputed
analytically after every Z update.  This isolates whether a structured bound
finds useful hyperparameters that SGPR can retain after adapting its inducing
locations to the SGPR objective.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

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
import numpy as np

jax.config.update("jax_enable_x64", True)

from common import kernels as kn  # noqa: E402
from common import predictors as pr  # noqa: E402
from common import terms as tm  # noqa: E402
from common.provenance import code_state, runtime_manifest  # noqa: E402
from sgpr import benchmark as bench  # noqa: E402
from sgpr import jax_collapsed as jm  # noqa: E402
from sgpr import protocol  # noqa: E402
from sgpr.analysis.explore_fixed_theta_bounds import (  # noqa: E402
    COLORS,
    METHODS,
    make_method_components,
    parse_schedule,
    scalar,
    train_method,
)
from sgpr.analysis.explore_sbt_guided_sgpr import (  # noqa: E402
    dummy_terms,
    params_numpy,
    run_lbfgs,
    save_curve,
    sem,
    timed,
    write_rows,
)
from sgpr.run_benchmark import (  # noqa: E402
    BENCHMARK_DATASETS,
    load_benchmark_dataset,
)

IMPLEMENTATION = "sgpr-hyperparameter-transplant-v7"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="kin40k", choices=BENCHMARK_DATASETS)
    parser.add_argument("--kernel", default="rbf", choices=bench.KERNEL_ORDER)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument(
        "--donor-schedule",
        type=parse_schedule,
        default=list(protocol.LBFGS_REFRESH_SCHEDULE),
    )
    parser.add_argument("--donor-steps", type=int, default=protocol.LBFGS_STEPS)
    parser.add_argument(
        "--transplant-steps", type=int, default=protocol.TRANSPLANT_STEPS
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--donor-runs-dir",
        type=Path,
        help="load the already fitted official donor params/blocks from this "
             "runs tree instead of retraining them",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def cache_matches(path, args, seed):
    """Accept only a complete transplant produced by this exact protocol."""
    try:
        payload = json.loads(path.read_text())
        config = payload.get("config") or {}
        rows = payload.get("rows") or []
        current_code = code_state(SIM_ROOT.parent)
        split_seed = seed if args.split_seed is None else int(args.split_seed)
        expected = {
            "dataset": args.dataset,
            "kernel": kn.canonical_kernel_name(args.kernel),
            "M": int(args.m),
            "seed": int(seed),
            "split_seed": int(split_seed),
            "n_train": int(args.n_train),
            "n_test": int(args.n_test),
            "block_size": int(args.block_size),
            "transplant_steps": int(args.transplant_steps),
            "convergence_window_steps": protocol.CONVERGENCE_WINDOW_STEPS,
            "convergence_obj_n_tol": protocol.CONVERGENCE_OBJ_N_TOL,
            "convergence_grad_max_n_tol": protocol.CONVERGENCE_GRAD_MAX_N_TOL,
            "convergence_patience": protocol.CONVERGENCE_PATIENCE,
            "optimizer_backend": "scipy-l-bfgs-b",
            "scipy_lbfgsb_ftol": protocol.SCIPY_LBFGSB_FTOL,
            "scipy_lbfgsb_gtol": protocol.SCIPY_LBFGSB_GTOL,
            "scipy_lbfgsb_maxcor": protocol.SCIPY_LBFGSB_MAXCOR,
            "scipy_lbfgsb_maxfun": protocol.SCIPY_LBFGSB_MAXFUN,
            "scipy_lbfgsb_maxls": protocol.SCIPY_LBFGSB_MAXLS,
            "donor_runs_dir": (
                str(args.donor_runs_dir.resolve())
                if args.donor_runs_dir else None
            ),
        }
        return bool(
            payload.get("implementation") == IMPLEMENTATION
            and all(config.get(key) == value for key, value in expected.items())
            and len(rows) == len(METHODS)
            and {row.get("donor_method") for row in rows} == set(METHODS)
            and all(
                row.get("transplant_convergence_reached") is True
                and int(row.get("transplant_max_steps", -1))
                    == int(args.transplant_steps)
                and float(row.get(
                    "transplant_final_grad_max_per_datum", math.inf
                )) <= protocol.CONVERGENCE_GRAD_MAX_N_TOL
                for row in rows
            )
            and (payload.get("code") or {}).get("source_dirty") is False
            and (payload.get("code") or {}).get("commit")
                == current_code.get("commit")
            and current_code.get("source_dirty") is False
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def diagonal_predict_jax(params, x, y, xs, *, kernel_name):
    """Bound-matched predictive covariance for the T-SGPR singleton cover."""
    c = jm._collapsed_predictive_components(
        params, x, y, xs, kernel_name, 1e-8
    )
    residual_cross = (
        c["kernel_fn"](c["xs"], c["x"]) - c["a_test"].T @ c["a_train"]
    )
    correction = jnp.sum(
        jnp.square(residual_cross)
        / jnp.maximum(c["d_train"] + c["s2"], 1e-12)[None, :],
        axis=1,
    )
    residual = jnp.maximum(c["d_test"] - correction, 0.0)
    variance = jnp.maximum(c["qvar"] + residual + c["s2"], 1e-10)
    return c["mean"], variance


def diagonal_metrics_jax(params, x, y, xs, ys, *, kernel_name):
    mu, variance = diagonal_predict_jax(
        params, x, y, xs, kernel_name=kernel_name
    )
    return pr.gaussian_rmse_nll_device(mu, variance, ys)


def complete_metrics(method, params, terms, data, kernel_name):
    """Return the predictive metrics of the donor's full structured model.

    Here "full" means all components retained by that approximation: every
    singleton for T-SGPR and every saved residual block for BT/SBT.  It is not
    the dense E-SGPR full-cross endpoint.
    """
    x_train, y_train, x_test, y_test, _ = data
    if method == "SGPR":
        call = lambda: jm.metrics_jax(
            params, x_train, y_train, x_test, y_test,
            exact=False, kernel_name=kernel_name,
        )
        predictor = "standard_sparse"
    elif method == "T-SGPR":
        call = lambda: diagonal_metrics_jax(
            params, x_train, y_train, x_test, y_test,
            kernel_name=kernel_name,
        )
        predictor = "all_singletons"
    else:
        blocks = tm.unpack_terms(terms)
        call = lambda: jm.metrics_collapsed(
            params, x_train, y_train, x_test, y_test,
            predictor="block", blocks=blocks, kernel_name=kernel_name,
        )
        predictor = f"all_{len(blocks)}_blocks"
    metrics, elapsed = timed(call, warmup=1, repeats=1)
    return scalar(metrics[0]), scalar(metrics[1]), elapsed, predictor


def sparse_metrics(params, data, kernel_name):
    x_train, y_train, x_test, y_test, _ = data
    metrics, elapsed = timed(
        lambda: jm.metrics_jax(
            params, x_train, y_train, x_test, y_test,
            exact=False, kernel_name=kernel_name,
        )
    )
    return scalar(metrics[0]), scalar(metrics[1]), elapsed


def transplant_to_sgpr(donor, trace_loss, steps, n_points):
    """Freeze donor theta/noise and optimize donor Z under the SGPR bound."""
    frozen = {key: value for key, value in donor.items() if key != "z"}

    def lift(z_params):
        full = dict(frozen)
        full["z"] = z_params["z"]
        return full

    def z_loss(z_params, coeffs, idx, mask):
        return trace_loss(lift(z_params), coeffs, idx, mask)

    trained_z, curve, info, _, wall_s = run_lbfgs(
        z_loss,
        {"z": copy.deepcopy(donor["z"])},
        terms=dummy_terms(),
        steps=steps,
        term_fn=None,
        schedule=[steps],
        convergence_n_points=n_points,
    )
    return lift(trained_z), curve, info, wall_s


def load_official_donor(root, kernel, dataset, m, method, seed):
    """Load one frozen donor and its exact fitted partition."""
    run_dir = root / kernel / dataset / f"M{m}" / method / f"seed{seed}"
    metrics_path = run_dir / "metrics.json"
    params_path = run_dir / "params.npz"
    if not metrics_path.exists() or not params_path.exists():
        raise FileNotFoundError(f"incomplete official donor: {run_dir}")
    source_metrics = json.loads(metrics_path.read_text())
    if source_metrics.get("implementation") != bench.IMPLEMENTATION:
        raise ValueError(
            f"incompatible donor implementation at {run_dir}: "
            f"{source_metrics.get('implementation')!r}"
        )
    if source_metrics.get("convergence_passed") is not True:
        raise ValueError(f"official donor did not pass convergence: {run_dir}")
    expected = {
        "kernel": kernel, "dataset": dataset, "M": int(m),
        "method": method, "seed": int(seed),
    }
    for key, value in expected.items():
        if source_metrics.get(key) != value:
            raise ValueError(
                f"donor identity mismatch at {run_dir}: {key}="
                f"{source_metrics.get(key)!r}, expected {value!r}"
            )
    loaded = np.load(params_path)
    params = {key: jnp.asarray(loaded[key]) for key in loaded.files}
    if method in ("RBT-SGPR", "SBT-SGPR"):
        block_path = run_dir / "final_blocks.npz"
        if not block_path.exists():
            raise FileNotFoundError(f"missing fitted donor blocks: {block_path}")
        packed = np.load(block_path)
        terms = tuple(jnp.asarray(packed[key]) for key in ("coeffs", "idx", "mask"))
    else:
        terms = dummy_terms()
    return params, terms, source_metrics, run_dir


def run_seed(args, seed):
    seed_dir = args.out_dir / f"seed{seed}"
    metrics_path = seed_dir / "metrics.json"
    if metrics_path.exists() and not args.force and cache_matches(
        metrics_path, args, seed
    ):
        print(f"seed{seed}: cached")
        return
    if metrics_path.exists() and not args.force:
        print(f"seed{seed}: stale/incomplete cache; rerun")
    seed_dir.mkdir(parents=True, exist_ok=True)

    split_seed = seed if args.split_seed is None else int(args.split_seed)
    data = load_benchmark_dataset(
        args.dataset, split_seed, args.n_train, args.n_test
    )
    x_train, y_train, _, _, metadata = data
    n = int(x_train.shape[0])
    kernel_name = kn.canonical_kernel_name(args.kernel)
    initial = bench.init_params(x_train, args.m, seed, kernel_name=kernel_name)

    losses, builders = {}, {}
    for method in METHODS:
        losses[method], builders[method] = make_method_components(
            method, x_train, y_train, kernel_name,
            args.block_size, seed, args.m,
        )

    donors = {}
    donor_terms = {}
    donor_training = {}
    if args.donor_runs_dir:
        print(f"seed{seed}: load official hyperparameter donors")
        for method in METHODS:
            params, terms, source, run_dir = load_official_donor(
                args.donor_runs_dir, kernel_name, args.dataset, args.m,
                method, seed,
            )
            donors[method], donor_terms[method] = params, terms
            donor_training[method] = {
                "donor_optimizer_s": source.get("optimizer_s"),
                "donor_compile_s": source.get("compile_s"),
                "donor_search_s": source.get("search_s"),
                "donor_wall_s": source.get("total_s"),
                "donor_config_id": source.get("config_id"),
                "donor_run_dir": str(run_dir),
            }
    else:
        print(f"seed{seed}: train hyperparameter donors")
        for method in METHODS:
            print(f"  -> donor {method}")
            params, curve, info, terms, search_s, wall_s = train_method(
                method,
                initial,
                losses=losses,
                builders=builders,
                steps=args.donor_steps,
                refresh_schedule=args.donor_schedule,
            )
            donors[method] = params
            donor_terms[method] = terms
            donor_training[method] = {
                "donor_optimizer_s": info["optimizer_s"],
                "donor_compile_s": info["compile_s"],
                "donor_search_s": search_s + info["search_s"],
                "donor_wall_s": search_s + wall_s,
                "donor_config_id": None,
                "donor_run_dir": None,
            }
            save_curve(seed_dir / f"curve_donor_{method}.npz", curve)
            np.savez_compressed(
                seed_dir / f"params_donor_{method}.npz", **params_numpy(params)
            )

    rows = []
    trace_loss = losses["SGPR"]
    print(f"seed{seed}: transplant each donor into SGPR")
    for method in METHODS:
        donor = donors[method]
        terms = donor_terms[method]
        complete_rmse, complete_nll, complete_s, predictor = complete_metrics(
            method, donor, terms, data, kernel_name
        )
        donor_sparse_rmse, donor_sparse_nll, donor_sparse_s = sparse_metrics(
            donor, data, kernel_name
        )
        donor_own_elbo = -scalar(losses[method](donor, *terms)) / n
        donor_sgpr_elbo = -scalar(trace_loss(donor, *dummy_terms())) / n

        print(f"  -> {method} theta; optimize SGPR Z")
        transplanted, curve, info, wall_s = transplant_to_sgpr(
            donor, trace_loss, args.transplant_steps, n
        )
        transplant_rmse, transplant_nll, transplant_pred_s = sparse_metrics(
            transplanted, data, kernel_name
        )
        transplant_elbo = (
            -scalar(trace_loss(transplanted, *dummy_terms())) / n
        )
        z_shift = scalar(jnp.sqrt(jnp.mean(
            (transplanted["z"] - donor["z"]) ** 2
        )))

        row = {
            "donor_method": method,
            "donor_complete_predictor": predictor,
            "donor_complete_rmse": complete_rmse,
            "donor_complete_nll": complete_nll,
            "donor_complete_prediction_s": complete_s,
            "donor_sparse_rmse": donor_sparse_rmse,
            "donor_sparse_nll": donor_sparse_nll,
            "donor_sparse_prediction_s": donor_sparse_s,
            "donor_own_elbo_n": donor_own_elbo,
            "donor_sgpr_elbo_n": donor_sgpr_elbo,
            "transplant_sgpr_rmse": transplant_rmse,
            "transplant_sgpr_nll": transplant_nll,
            "transplant_sgpr_elbo_n": transplant_elbo,
            "transplant_prediction_s": transplant_pred_s,
            "transplant_optimizer_s": info["optimizer_s"],
            "transplant_compile_s": info["compile_s"],
            "transplant_wall_s": wall_s,
            "transplant_steps": int(info["actual_steps"]),
            "transplant_max_steps": int(args.transplant_steps),
            "transplant_convergence_reached": bool(
                info["convergence_reached"]
            ),
            "transplant_final_grad_l2_per_datum": (
                float(info["convergence_checks"][-1]["grad_l2"]) / n
                if info.get("convergence_checks") else float("nan")
            ),
            "transplant_final_grad_max_per_datum": (
                float(info["convergence_checks"][-1]["grad_max"]) / n
                if info.get("convergence_checks") else float("nan")
            ),
            "transplant_z_rms_shift": z_shift,
            "delta_sgpr_elbo_n": transplant_elbo - donor_sgpr_elbo,
            "delta_sparse_nll": transplant_nll - donor_sparse_nll,
            "noise_var": math.exp(scalar(donor["log_s2"])),
            "signal_var": math.exp(scalar(donor["log_sf2"])),
            "ell_mean": float(np.exp(params_numpy(donor)["log_ell"]).mean()),
        }
        row.update(donor_training[method])
        row.update({
            "seed": seed,
            "split_seed": split_seed,
            "dataset": args.dataset,
            "kernel": kernel_name,
            "M": args.m,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "d": int(metadata["d"]),
            "block_size": args.block_size,
            "optimizer": "lbfgs",
            "optimizer_backend": "scipy-l-bfgs-b",
            "implementation": IMPLEMENTATION,
        })
        rows.append(row)
        save_curve(seed_dir / f"curve_transplant_{method}.npz", curve)
        np.savez_compressed(
            seed_dir / f"params_transplant_{method}.npz",
            **params_numpy(transplanted),
        )

    payload = {
        "implementation": IMPLEMENTATION,
        "config": {
            "dataset": args.dataset,
            "kernel": kernel_name,
            "M": args.m,
            "seed": seed,
            "split_seed": split_seed,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "block_size": args.block_size,
            "donor_steps": args.donor_steps,
            "donor_schedule": args.donor_schedule,
            "transplant_steps": args.transplant_steps,
            "convergence_window_steps": protocol.CONVERGENCE_WINDOW_STEPS,
            "convergence_obj_n_tol": protocol.CONVERGENCE_OBJ_N_TOL,
            "convergence_grad_max_n_tol": protocol.CONVERGENCE_GRAD_MAX_N_TOL,
            "convergence_patience": protocol.CONVERGENCE_PATIENCE,
            "optimizer_backend": "scipy-l-bfgs-b",
            "scipy_lbfgsb_ftol": protocol.SCIPY_LBFGSB_FTOL,
            "scipy_lbfgsb_gtol": protocol.SCIPY_LBFGSB_GTOL,
            "scipy_lbfgsb_maxcor": protocol.SCIPY_LBFGSB_MAXCOR,
            "scipy_lbfgsb_maxfun": protocol.SCIPY_LBFGSB_MAXFUN,
            "scipy_lbfgsb_maxls": protocol.SCIPY_LBFGSB_MAXLS,
            "transplant": "freeze each donor theta/noise; optimize donor Z with SGPR",
            "donor_runs_dir": (
                str(args.donor_runs_dir.resolve()) if args.donor_runs_dir else None
            ),
        },
        "code": code_state(SIM_ROOT.parent),
        "runtime": runtime_manifest(),
        "rows": rows,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    write_rows(seed_dir / "metrics.csv", rows)
    print(f"seed{seed}: wrote {metrics_path}")


def write_tex(path, summary, kernel, n_train, n_test, m):
    """The paper's second kin40k table: every donor, made sparse.

    Each row is one donor bound. The first pair of columns is the donor at its
    own fitted parameters under the predictor its bound models; the second is
    the same parameters read through the Titsias sparse predictor; the third is
    after re-optimising only Z under the SGPR trace bound. The last column is
    what that re-optimisation bought.
    """
    def cell(row, field, digits=3):
        mean = row.get(f"{field}_mean")
        sem_value = row.get(f"{field}_sem")
        if mean is None or not np.isfinite(float(mean)):
            return "--"
        if sem_value is None or not np.isfinite(float(sem_value)):
            return f"{float(mean):.{digits}f}"
        return f"{float(mean):.{digits}f} ({float(sem_value):.{digits}f})"

    by_method = {row["donor_method"]: row for row in summary}
    n_seeds = max((row["n_seeds"] for row in summary), default=0)
    lines = [
        "% Generated by "
        "Simulations/sgpr/analysis/explore_hyperparameter_transplant.py.",
        "\\begin{table}[H]",
        f"\\caption{{Kin40k hyperparameter transplant, \\({n_train}/{n_test}\\) "
        f"train/test points and \\(M={m}\\). Every donor is trained under its "
        "own collapsed bound with L-BFGS. Its kernel and noise are then frozen "
        "and only the inducing locations are re-optimised under the SGPR trace "
        "bound, so the final model is an ordinary sparse GP and its predictive "
        "is the Titsias sparse predictor. Entries are mean (standard error) "
        f"over {n_seeds} seeds.}}",
        f"\\label{{tab:kin40k-transplant-{kernel}}}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        "& \\multicolumn{2}{c}{Donor, own predictor} "
        "& \\multicolumn{2}{c}{Donor, read as sparse} "
        "& \\multicolumn{2}{c}{After transplant} & \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Donor bound & RMSE & NLL & RMSE & NLL & RMSE & NLL "
        "& \\(\\Delta\\) ELBO/N \\\\",
        "\\midrule",
    ]
    for method in METHODS:
        row = by_method.get(method)
        if row is None:
            continue
        cells = [
            method,
            cell(row, "donor_complete_rmse"), cell(row, "donor_complete_nll"),
            cell(row, "donor_sparse_rmse"), cell(row, "donor_sparse_nll"),
            cell(row, "transplant_sgpr_rmse"), cell(row, "transplant_sgpr_nll"),
            cell(row, "delta_sgpr_elbo_n", 4),
        ]
        if method == "SBT-SGPR":
            cells = [f"\\textbf{{{value}}}" for value in cells]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


def summarize(out_dir):
    records, configs = [], []
    for path in sorted(out_dir.glob("seed*/metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("implementation") == IMPLEMENTATION:
            records.extend(payload["rows"])
            configs.append(payload.get("config", {}))
    if not records:
        raise SystemExit(f"no {IMPLEMENTATION} results under {out_dir}")
    # Every seed of one campaign shares the protocol; the table's caption has
    # to state it, so refuse to summarise a directory holding two of them.
    protocol = {
        key: {str(cfg.get(key)) for cfg in configs}
        for key in ("kernel", "n_train", "n_test", "M")
    }
    mixed = sorted(key for key, values in protocol.items() if len(values) > 1)
    if mixed:
        raise SystemExit(
            f"{out_dir} mixes {mixed}; summarise one protocol at a time"
        )

    fields = (
        "donor_complete_rmse", "donor_complete_nll",
        "donor_sparse_rmse", "donor_sparse_nll",
        "donor_own_elbo_n", "donor_sgpr_elbo_n",
        "transplant_sgpr_rmse", "transplant_sgpr_nll",
        "transplant_sgpr_elbo_n", "delta_sgpr_elbo_n",
        "delta_sparse_nll", "donor_wall_s", "transplant_wall_s",
    )
    summary = []
    for method in METHODS:
        subset = [row for row in records if row["donor_method"] == method]
        if not subset:
            continue
        item = {"donor_method": method, "n_seeds": len(subset)}
        for field in fields:
            values = [float(row[field]) for row in subset]
            item[f"{field}_mean"] = float(np.mean(values))
            item[f"{field}_sem"] = sem(values)
        summary.append(item)

    write_rows(out_dir / "all_results.csv", records)
    write_rows(out_dir / "summary.csv", summary)
    write_tex(
        out_dir / "hyperparameter_transplant.tex", summary,
        kernel=next(iter(protocol["kernel"])),
        n_train=next(iter(protocol["n_train"])),
        n_test=next(iter(protocol["n_test"])),
        m=next(iter(protocol["M"])),
    )

    import matplotlib.pyplot as plt

    by_method = {row["donor_method"]: row for row in summary}
    x = np.arange(len(METHODS))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    comparisons = (
        ("nll", "NLL", False),
        ("rmse", "RMSE", False),
    )
    for axis, (suffix, title, _) in zip(axes[:2], comparisons):
        series = (
            ("donor_complete", "Donor complete", -width),
            ("donor_sparse", "Donor as sparse SGPR", 0.0),
            ("transplant_sgpr", "After SGPR transplant", width),
        )
        for prefix, label, offset in series:
            means = [by_method[m][f"{prefix}_{suffix}_mean"] for m in METHODS]
            errors = [by_method[m][f"{prefix}_{suffix}_sem"] for m in METHODS]
            errors = [0.0 if not np.isfinite(v) else v for v in errors]
            axis.bar(x + offset, means, width, yerr=errors, capsize=2, label=label)
        axis.set_title(title + " (lower is better)")
        axis.set_xticks(x, METHODS, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)

    axis = axes[2]
    before = [by_method[m]["donor_sgpr_elbo_n_mean"] for m in METHODS]
    after = [by_method[m]["transplant_sgpr_elbo_n_mean"] for m in METHODS]
    axis.bar(x - width / 2, before, width, label="Before transplant")
    axis.bar(x + width / 2, after, width, label="After transplant")
    axis.set_title("SGPR ELBO / N (higher is better)")
    axis.set_xticks(x, METHODS, rotation=20, ha="right")
    axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "hyperparameter_transplant.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(out_dir / "hyperparameter_transplant.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'hyperparameter_transplant.pdf'}")


def main():
    args = parse_args()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(args.out_dir)
        return
    for seed in args.seeds:
        run_seed(args, int(seed))
    if not args.no_summary:
        summarize(args.out_dir)


if __name__ == "__main__":
    main()
