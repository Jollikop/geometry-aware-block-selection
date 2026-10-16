#!/usr/bin/env python3
"""Explore whether SBT supplies better inducing-point initialisations to SGPR.

Every seed starts from exactly the same parameters and fits five controlled
variants on the same 5,000-point split:

* ``SGPR-direct``: SGPR for the target-stage budget;
* ``SGPR-direct-budget``: SGPR for the complete SBT + target budget;
* ``SBT-to-SGPR``: SBT parameters evaluated with the standard SGPR predictor;
* ``SBT-Z-to-SGPR``: only SBT's inducing locations initialise an SGPR fit;
* ``SBT-all-to-SGPR``: all SBT parameters initialise an SGPR fit.

The two fine-tuning fits start new L-BFGS calls, so their curvature history is
reset when the objective changes. The headline predictor is always SGPR. A
local SBT predictor is recorded only as a covariance diagnostic; its mean must
agree numerically with the SGPR mean at the same parameters.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

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
from common.provenance import code_state, runtime_manifest  # noqa: E402
from sgpr import benchmark as bench  # noqa: E402
from common import terms as tm  # noqa: E402
from sgpr import jax_collapsed as jm  # noqa: E402
from sgpr import protocol  # noqa: E402
from sgpr.run_benchmark import (  # noqa: E402
    BENCHMARK_DATASETS,
    load_benchmark_dataset,
)

IMPLEMENTATION = "sbt-guided-sgpr-exploratory-v1"
VARIANT_ORDER = (
    "SGPR-direct",
    "SGPR-direct-budget",
    "SBT-to-SGPR",
    "SBT-Z-to-SGPR",
    "SBT-all-to-SGPR",
)
DISPLAY_NAMES = {
    "SGPR-direct": "SGPR direct",
    "SGPR-direct-budget": "SGPR direct\n(equal budget)",
    "SBT-to-SGPR": "SBT -> SGPR\n(no fine-tune)",
    "SBT-Z-to-SGPR": "SBT Z -> SGPR",
    "SBT-all-to-SGPR": "SBT all -> SGPR",
}


def parse_schedule(value: str) -> list[int]:
    phases = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not phases or any(item <= 0 for item in phases):
        raise argparse.ArgumentTypeError("schedule must contain positive integers")
    return phases


def parse_args() -> argparse.Namespace:
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
        "--sbt-schedule",
        type=parse_schedule,
        default=parse_schedule("15,15,20,25,25"),
    )
    parser.add_argument("--target-steps", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="separate exploratory result directory",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="write seed artifacts only (used by parallel Slurm tasks)",
    )
    return parser.parse_args()


def dummy_terms():
    return (
        jnp.zeros((1,), dtype=jnp.float64),
        jnp.zeros((1, 1), dtype=jnp.int32),
        jnp.zeros((1, 1), dtype=jnp.float64),
    )


def scalar(value) -> float:
    return float(np.asarray(jax.device_get(value)))


def timed(callable_, *, warmup: int = 1, repeats: int = 3):
    result = callable_()
    jax.block_until_ready(result)
    for _ in range(max(0, warmup - 1)):
        result = callable_()
        jax.block_until_ready(result)
    elapsed = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = callable_()
        jax.block_until_ready(result)
        elapsed.append(time.perf_counter() - start)
    return result, float(np.median(elapsed))


def residual_trace(params, x, kernel_name: str, jitter: float = 1e-8):
    """Compute tr(Kff-Qff) in O(NM) memory/time after kernel products."""
    x = jnp.asarray(x)
    kernel_name = kn.canonical_kernel_name(kernel_name)
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    z = params["z"]
    warp = params.get("warp")
    kuu = kn.kernel(z, z, ell, sf2, kernel_name, warp)
    kuu = kuu + jitter * jnp.eye(z.shape[0], dtype=x.dtype)
    kuf = kn.kernel(z, x, ell, sf2, kernel_name, warp)
    chol = jnp.linalg.cholesky(kuu)
    projected = jax.scipy.linalg.solve_triangular(chol, kuf, lower=True)
    diagonal = kn.kernel_diag(x, ell, sf2, kernel_name, warp)
    return jnp.sum(jnp.maximum(diagonal - jnp.sum(projected**2, axis=0), 0.0))


def coverage_distance(params, x):
    """Mean Euclidean distance from each training point to its nearest Z."""
    x = jnp.asarray(x)
    z = jnp.asarray(params["z"])
    x2 = jnp.sum(x * x, axis=1, keepdims=True)
    z2 = jnp.sum(z * z, axis=1)[None, :]
    distance2 = jnp.maximum(x2 + z2 - 2.0 * (x @ z.T), 0.0)
    return jnp.mean(jnp.sqrt(jnp.min(distance2, axis=1)))


def run_lbfgs(
    loss,
    params0,
    *,
    terms,
    steps,
    term_fn=None,
    schedule=None,
    convergence_n_points=None,
):
    start = time.perf_counter()
    params, curve, info = jm.train_collapsed(
        loss,
        copy.deepcopy(params0),
        terms0=terms,
        term_fn=term_fn,
        steps=int(steps),
        refresh_every=int(steps) + 1,
        optimizer="lbfgs",
        lr=0.0,
        noise_lr=None,
        grad_clip=0.0,
        train_z=True,
        log_every=0,
        log_cb=None,
        refresh_schedule=schedule,
        convergence_abs_tol=(
            None if convergence_n_points is None
            else protocol.CONVERGENCE_OBJ_N_TOL * int(convergence_n_points)
        ),
        convergence_window_steps=protocol.CONVERGENCE_WINDOW_STEPS,
        convergence_patience=protocol.CONVERGENCE_PATIENCE,
        convergence_grad_max_tol=(
            None if convergence_n_points is None
            else protocol.CONVERGENCE_GRAD_MAX_N_TOL
                * int(convergence_n_points)
        ),
        objective_scale=(
            1.0 if convergence_n_points is None
            else float(convergence_n_points)
        ),
    )
    wall_s = time.perf_counter() - start
    final_terms = info.pop("final_terms")
    return params, np.asarray(curve), info, final_terms, wall_s


def params_numpy(params):
    return {key: np.asarray(jax.device_get(value)) for key, value in params.items()}


def evaluate(
    name,
    params,
    *,
    x_train,
    y_train,
    x_test,
    y_test,
    kernel_name,
    trace_loss,
    block_loss,
    sbt_builder,
    initial_params,
    stage_info,
):
    n = int(x_train.shape[0])
    terms = sbt_builder(params)
    jax.block_until_ready(terms[0])

    (rmse, nll), pred_s = timed(
        lambda: jm.metrics_jax(
            params,
            x_train,
            y_train,
            x_test,
            y_test,
            exact=False,
            kernel_name=kernel_name,
        )
    )
    (local_rmse, local_nll), local_pred_s = timed(
        lambda: jm.metrics_collapsed(
            params,
            x_train,
            y_train,
            x_test,
            y_test,
            predictor="full",
            blocks=tm.unpack_terms(terms),
            kernel_name=kernel_name,
        )
    )
    sparse_prediction = jm.predict_jax(
        params, x_train, y_train, x_test, exact=False, kernel_name=kernel_name
    )
    local_prediction = jm.predict_collapsed(
        params,
        x_train,
        y_train,
        x_test,
        predictor="full",
        blocks=tm.unpack_terms(terms),
        kernel_name=kernel_name,
    )
    mean_difference = jnp.max(jnp.abs(sparse_prediction[0] - local_prediction[0]))
    trace = residual_trace(params, x_train, kernel_name)
    coverage = coverage_distance(params, x_train)
    z_shift = jnp.sqrt(jnp.mean((params["z"] - initial_params["z"]) ** 2))

    row = {
        "variant": name,
        "rmse": scalar(rmse),
        "nll": scalar(nll),
        "local_rmse": scalar(local_rmse),
        "local_nll": scalar(local_nll),
        "mean_max_abs_difference": scalar(mean_difference),
        "sgpr_elbo_n": -scalar(trace_loss(params, *dummy_terms())) / n,
        "sbt_elbo_n": -scalar(block_loss(params, *terms)) / n,
        "residual_trace_n": scalar(trace) / n,
        "coverage_distance": scalar(coverage),
        "z_rms_shift": scalar(z_shift),
        "noise_var": math.exp(scalar(params["log_s2"])),
        "signal_var": math.exp(scalar(params["log_sf2"])),
        "ell_mean": float(np.exp(params_numpy(params)["log_ell"]).mean()),
        "prediction_s": pred_s,
        "local_prediction_s": local_pred_s,
    }
    row.update(stage_info)
    return row


def write_rows(path: Path, rows: list[dict]):
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_curve(path: Path, curve):
    curve = np.asarray(curve)
    if curve.size:
        np.savez_compressed(path, step=curve[:, 0], loss=curve[:, 1])


def plot_seed(seed_dir: Path, n_train: int):
    import matplotlib.pyplot as plt

    curves = {}
    for path in seed_dir.glob("curve_*.npz"):
        archive = np.load(path)
        curves[path.stem.removeprefix("curve_")] = (
            archive["step"], archive["loss"] / n_train
        )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    if "sbt" in curves:
        axes[0].plot(*curves["sbt"], color="#E31A1C", lw=1.8)
    axes[0].set_title("SBT pretraining objective")
    axes[0].set_xlabel("L-BFGS step")
    axes[0].set_ylabel("negative bound / N")

    styles = {
        "direct": ("SGPR direct", "#222222"),
        "direct_budget": ("SGPR direct (equal budget)", "#777777"),
        "z_finetune": ("SBT Z -> SGPR", "#377EB8"),
        "all_finetune": ("SBT all -> SGPR", "#E31A1C"),
    }
    for key, (label, color) in styles.items():
        if key in curves:
            axes[1].plot(*curves[key], label=label, color=color, lw=1.7)
    axes[1].set_title("SGPR target objective")
    axes[1].set_xlabel("L-BFGS step")
    axes[1].set_ylabel("negative SGPR bound / N")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(seed_dir / "convergence.png", dpi=180, bbox_inches="tight")
    fig.savefig(seed_dir / "convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def run_seed(args, seed: int):
    seed_dir = args.out_dir / f"seed{seed}"
    result_path = seed_dir / "metrics.json"
    if result_path.exists() and not args.force:
        print(f"seed{seed}: cached at {result_path}")
        return
    seed_dir.mkdir(parents=True, exist_ok=True)

    split_seed = seed if args.split_seed is None else int(args.split_seed)
    data = load_benchmark_dataset(
        args.dataset, split_seed, args.n_train, args.n_test
    )
    x_train, y_train, x_test, y_test, meta = data
    kernel_name = kn.canonical_kernel_name(args.kernel)
    initial = bench.init_params(x_train, args.m, seed, kernel_name=kernel_name)
    trace_loss = jm.make_loss(
        x_train, y_train, dense_mode="trace", kernel_name=kernel_name
    )
    block_loss = jm.make_loss(
        x_train, y_train, dense_mode=None, kernel_name=kernel_name
    )
    selector_args = SimpleNamespace(kernel=kernel_name, seed=seed)
    sbt_builder = bench.make_term_fn(
        "SBT-SGPR",
        x_train,
        args.block_size,
        selector_args,
        1e-8,
        structure_seed=seed,
        m=args.m,
    )
    start = time.perf_counter()
    initial_terms = sbt_builder(initial)
    jax.block_until_ready(initial_terms[0])
    initial_search_s = time.perf_counter() - start

    print(f"seed{seed}: SGPR direct ({args.target_steps} steps)")
    direct, direct_curve, direct_info, _, direct_wall = run_lbfgs(
        trace_loss,
        initial,
        terms=dummy_terms(),
        steps=args.target_steps,
        schedule=[args.target_steps],
    )
    total_budget_steps = sum(args.sbt_schedule) + args.target_steps
    print(f"seed{seed}: SGPR direct equal-budget ({total_budget_steps} steps)")
    direct_budget, budget_curve, budget_info, _, budget_wall = run_lbfgs(
        trace_loss,
        initial,
        terms=dummy_terms(),
        steps=total_budget_steps,
        schedule=[total_budget_steps],
    )
    print(f"seed{seed}: SBT pretraining ({sum(args.sbt_schedule)} steps)")
    sbt, sbt_curve, sbt_info, _, sbt_wall = run_lbfgs(
        block_loss,
        initial,
        terms=initial_terms,
        steps=sum(args.sbt_schedule),
        term_fn=sbt_builder,
        schedule=args.sbt_schedule,
    )

    z_initial = copy.deepcopy(initial)
    z_initial["z"] = sbt["z"]
    print(f"seed{seed}: SBT Z -> SGPR ({args.target_steps} target steps)")
    z_finetune, z_curve, z_info, _, z_wall = run_lbfgs(
        trace_loss,
        z_initial,
        terms=dummy_terms(),
        steps=args.target_steps,
        schedule=[args.target_steps],
    )
    print(f"seed{seed}: SBT all -> SGPR ({args.target_steps} target steps)")
    all_finetune, all_curve, all_info, _, all_wall = run_lbfgs(
        trace_loss,
        sbt,
        terms=dummy_terms(),
        steps=args.target_steps,
        schedule=[args.target_steps],
    )

    common = dict(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        kernel_name=kernel_name,
        trace_loss=trace_loss,
        block_loss=block_loss,
        sbt_builder=sbt_builder,
        initial_params=initial,
    )
    rows = [
        evaluate(
            "SGPR-direct", direct, **common,
            stage_info={
                "sbt_pretrain_s": 0.0,
                "sbt_compile_s": 0.0,
                "target_optimizer_s": direct_info["optimizer_s"],
                "target_compile_s": direct_info["compile_s"],
                "search_s": 0.0,
                "total_wall_s": direct_wall,
                "target_steps": args.target_steps,
                "pretrain_steps": 0,
            },
        ),
        evaluate(
            "SGPR-direct-budget", direct_budget, **common,
            stage_info={
                "sbt_pretrain_s": 0.0,
                "sbt_compile_s": 0.0,
                "target_optimizer_s": budget_info["optimizer_s"],
                "target_compile_s": budget_info["compile_s"],
                "search_s": 0.0,
                "total_wall_s": budget_wall,
                "target_steps": total_budget_steps,
                "pretrain_steps": 0,
            },
        ),
        evaluate(
            "SBT-to-SGPR", sbt, **common,
            stage_info={
                "sbt_pretrain_s": sbt_info["optimizer_s"],
                "sbt_compile_s": sbt_info["compile_s"],
                "target_optimizer_s": 0.0,
                "target_compile_s": 0.0,
                "search_s": initial_search_s + sbt_info["search_s"],
                "total_wall_s": initial_search_s + sbt_wall,
                "target_steps": 0,
                "pretrain_steps": sum(args.sbt_schedule),
            },
        ),
        evaluate(
            "SBT-Z-to-SGPR", z_finetune, **common,
            stage_info={
                "sbt_pretrain_s": sbt_info["optimizer_s"],
                "sbt_compile_s": sbt_info["compile_s"],
                "target_optimizer_s": z_info["optimizer_s"],
                "target_compile_s": z_info["compile_s"],
                "search_s": initial_search_s + sbt_info["search_s"],
                "total_wall_s": initial_search_s + sbt_wall + z_wall,
                "target_steps": args.target_steps,
                "pretrain_steps": sum(args.sbt_schedule),
            },
        ),
        evaluate(
            "SBT-all-to-SGPR", all_finetune, **common,
            stage_info={
                "sbt_pretrain_s": sbt_info["optimizer_s"],
                "sbt_compile_s": sbt_info["compile_s"],
                "target_optimizer_s": all_info["optimizer_s"],
                "target_compile_s": all_info["compile_s"],
                "search_s": initial_search_s + sbt_info["search_s"],
                "total_wall_s": initial_search_s + sbt_wall + all_wall,
                "target_steps": args.target_steps,
                "pretrain_steps": sum(args.sbt_schedule),
            },
        ),
    ]
    for row in rows:
        row.update({
            "seed": seed,
            "split_seed": split_seed,
            "dataset": args.dataset,
            "kernel": kernel_name,
            "M": args.m,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "d": int(meta["d"]),
            "block_size": args.block_size,
            "optimizer": "lbfgs",
            "implementation": IMPLEMENTATION,
        })
        if row["mean_max_abs_difference"] > 1e-8:
            raise RuntimeError(
                f"{row['variant']}: SBT and SGPR predictive means differ by "
                f"{row['mean_max_abs_difference']:.3e}"
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
            "sbt_schedule": args.sbt_schedule,
            "target_steps": args.target_steps,
        },
        "code": code_state(SIM_ROOT.parent),
        "runtime": runtime_manifest(),
        "rows": rows,
    }
    result_path.write_text(json.dumps(payload, indent=2))
    write_rows(seed_dir / "metrics.csv", rows)
    parameter_sets = {
        "SGPR-direct": direct,
        "SGPR-direct-budget": direct_budget,
        "SBT-to-SGPR": sbt,
        "SBT-Z-to-SGPR": z_finetune,
        "SBT-all-to-SGPR": all_finetune,
    }
    for name, params in parameter_sets.items():
        np.savez_compressed(seed_dir / f"params_{name}.npz", **params_numpy(params))
    save_curve(seed_dir / "curve_direct.npz", direct_curve)
    save_curve(seed_dir / "curve_direct_budget.npz", budget_curve)
    save_curve(seed_dir / "curve_sbt.npz", sbt_curve)
    save_curve(seed_dir / "curve_z_finetune.npz", z_curve)
    save_curve(seed_dir / "curve_all_finetune.npz", all_curve)
    plot_seed(seed_dir, args.n_train)
    print(f"seed{seed}: wrote {result_path}")


def sem(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(values.size))


def summarize(out_dir: Path):
    records = []
    for path in sorted(out_dir.glob("seed*/metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("implementation") != IMPLEMENTATION:
            continue
        records.extend(payload["rows"])
    if not records:
        raise SystemExit(f"no {IMPLEMENTATION} results under {out_dir}")

    metrics = (
        "nll",
        "rmse",
        "sgpr_elbo_n",
        "sbt_elbo_n",
        "residual_trace_n",
        "coverage_distance",
        "total_wall_s",
    )
    summary = []
    for variant in VARIANT_ORDER:
        subset = [row for row in records if row["variant"] == variant]
        if not subset:
            continue
        row = {"variant": variant, "n_seeds": len(subset)}
        for metric in metrics:
            values = [float(item[metric]) for item in subset]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sem"] = sem(values)
        summary.append(row)

    write_rows(out_dir / "all_results.csv", records)
    write_rows(out_dir / "summary.csv", summary)

    import matplotlib.pyplot as plt

    panels = (
        ("nll", "Test NLL", False),
        ("rmse", "Test RMSE", False),
        ("sgpr_elbo_n", "Final SGPR ELBO / N", True),
        ("residual_trace_n", "Residual trace / N", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4))
    x = np.arange(len(summary))
    colors = ["#222222", "#777777", "#E31A1C", "#377EB8", "#C51B7D"]
    labels = [DISPLAY_NAMES[row["variant"]] for row in summary]
    for axis, (metric, title, higher_is_better) in zip(axes.ravel(), panels):
        means = [row[f"{metric}_mean"] for row in summary]
        errors = [row[f"{metric}_sem"] for row in summary]
        errors = [0.0 if not np.isfinite(value) else value for value in errors]
        axis.bar(x, means, yerr=errors, color=colors[:len(summary)], capsize=3)
        axis.set_title(title)
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.2)
        if higher_is_better:
            axis.set_ylabel("higher is better")
    fig.suptitle("SBT-guided SGPR, standard SGPR prediction", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "sbt_guided_sgpr_summary.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(out_dir / "sbt_guided_sgpr_summary.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"results: {len(records)} rows from "
          f"{len({row['seed'] for row in records})} seed(s)")
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'sbt_guided_sgpr_summary.pdf'}")


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
