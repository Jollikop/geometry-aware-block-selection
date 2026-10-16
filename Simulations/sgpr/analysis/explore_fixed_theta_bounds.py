#!/usr/bin/env python3
"""Separate bound effects from inducing-location effects in collapsed SGPR.

For each seed this exploratory experiment has two stages:

1. Joint training: SGPR, T-SGPR, RBT-SGPR and SBT-SGPR optimize all kernel,
   noise and inducing-location parameters from the same initialization.
2. Fixed-theta training: kernel and noise parameters are frozen to the jointly
   trained SGPR solution. Every method restarts from the same initial inducing
   locations and optimizes Z only under its own bound.

Every fitted state is cross-evaluated under all four bounds. Headline RMSE/NLL
always use the standard SGPR predictor, so the fixed-theta stage isolates the
effect of the inducing locations on the common sparse posterior mean.
"""
from __future__ import annotations

import argparse
import copy
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
from common.plot_style import method_color  # noqa: E402
from common.provenance import code_state, runtime_manifest  # noqa: E402
from sgpr import benchmark as bench  # noqa: E402
from sgpr import jax_collapsed as jm  # noqa: E402
from sgpr import protocol  # noqa: E402
from sgpr.analysis.explore_sbt_guided_sgpr import (  # noqa: E402
    coverage_distance,
    dummy_terms,
    params_numpy,
    residual_trace,
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

IMPLEMENTATION = "sgpr-fixed-theta-bound-comparison-v1"
METHODS = ("SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR")


def elbo_column(method: str) -> str:
    """Column name holding one method's ELBO/N at another method's parameters."""
    return f"elbo_{method.lower().replace('-', '_')}_n"
LOSS_MODES = {
    "SGPR": "trace",
    "T-SGPR": "diag",
    "RBT-SGPR": None,
    "SBT-SGPR": None,
}
# The palette has one owner, common.plot_style. A local copy here drifted to a
# different hue for RBT-SGPR and propagated into the transplant figure, which
# imports COLORS from this module.
COLORS = {method: method_color(method) for method in METHODS}


def parse_schedule(value: str) -> list[int]:
    phases = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not phases or any(phase <= 0 for phase in phases):
        raise argparse.ArgumentTypeError("schedule must contain positive integers")
    return phases


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
        "--refresh-schedule",
        type=parse_schedule,
        default=list(protocol.LBFGS_REFRESH_SCHEDULE),
    )
    parser.add_argument("--steps", type=int, default=protocol.LBFGS_STEPS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    return parser.parse_args()


def scalar(value) -> float:
    return float(np.asarray(jax.device_get(value)))


def lift_z(z_params, frozen):
    params = dict(frozen)
    params["z"] = z_params["z"]
    return params


def make_method_components(method, x, y, kernel_name, block_size, seed, m):
    loss = jm.make_loss(
        x,
        y,
        dense_mode=LOSS_MODES[method],
        kernel_name=kernel_name,
    )
    if method not in ("RBT-SGPR", "SBT-SGPR"):
        return loss, None
    selector_args = SimpleNamespace(kernel=kernel_name, seed=seed)
    builder = bench.make_term_fn(
        method,
        x,
        block_size,
        selector_args,
        1e-8,
        structure_seed=seed,
        m=m,
    )
    return loss, builder


def train_method(
    method,
    params0,
    *,
    losses,
    builders,
    steps,
    refresh_schedule,
    frozen=None,
):
    base_loss = losses[method]
    base_builder = builders[method]
    if frozen is None:
        train_params = copy.deepcopy(params0)
        loss = base_loss
        builder = base_builder
        to_full = lambda params: params
    else:
        train_params = {"z": copy.deepcopy(params0["z"])}

        def to_full(params):
            return lift_z(params, frozen)

        def loss(params, coeffs, idx, mask):
            return base_loss(to_full(params), coeffs, idx, mask)

        if base_builder is None:
            builder = None
        else:
            def builder(params):
                return base_builder(to_full(params))

            if getattr(base_builder, "_host_refresh", False):
                builder._host_refresh = True

    search_start = time.perf_counter()
    terms0 = dummy_terms() if builder is None else builder(train_params)
    jax.block_until_ready(terms0[0])
    initial_search_s = 0.0 if builder is None else time.perf_counter() - search_start
    schedule = refresh_schedule if method == "SBT-SGPR" else [steps]
    trained, curve, info, final_terms, wall_s = run_lbfgs(
        loss,
        train_params,
        terms=terms0,
        steps=steps,
        term_fn=builder if method == "SBT-SGPR" else None,
        schedule=schedule,
    )
    full = to_full(trained)
    return full, curve, info, final_terms, initial_search_s, wall_s


def cross_elbos(params, *, losses, builders, n):
    values = {}
    for method in METHODS:
        builder = builders[method]
        terms = dummy_terms() if builder is None else builder(params)
        jax.block_until_ready(terms[0])
        values[elbo_column(method)] = (
            -scalar(losses[method](params, *terms)) / n
        )
    return values


def evaluate(
    stage,
    method,
    params,
    *,
    data,
    kernel_name,
    losses,
    builders,
    initial,
    final_terms,
    timing,
):
    x_train, y_train, x_test, y_test, _ = data
    n = int(x_train.shape[0])
    (rmse, nll), prediction_s = timed(
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
    own_elbo = -scalar(losses[method](params, *final_terms)) / n
    row = {
        "stage": stage,
        "method": method,
        "rmse": scalar(rmse),
        "nll": scalar(nll),
        "own_elbo_n": own_elbo,
        "prediction_s": prediction_s,
        "residual_trace_n": scalar(
            residual_trace(params, x_train, kernel_name)
        ) / n,
        "coverage_distance": scalar(coverage_distance(params, x_train)),
        "z_rms_shift": scalar(
            jnp.sqrt(jnp.mean((params["z"] - initial["z"]) ** 2))
        ),
        "noise_var": math.exp(scalar(params["log_s2"])),
        "signal_var": math.exp(scalar(params["log_sf2"])),
        "ell_mean": float(np.exp(params_numpy(params)["log_ell"]).mean()),
    }
    row.update(cross_elbos(params, losses=losses, builders=builders, n=n))
    row.update(timing)
    return row


def plot_seed(seed_dir, n_train):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13.0, 5.8), sharey="row")
    for row, stage in enumerate(("joint", "fixed_theta")):
        for col, method in enumerate(METHODS):
            path = seed_dir / f"curve_{stage}_{method}.npz"
            axis = axes[row, col]
            if path.exists():
                archive = np.load(path)
                axis.plot(
                    archive["step"],
                    archive["loss"] / n_train,
                    color=COLORS[method],
                    lw=1.7,
                )
            axis.set_title(method)
            axis.set_xlabel("L-BFGS step")
            axis.grid(alpha=0.2)
        axes[row, 0].set_ylabel(
            "joint negative bound / N" if stage == "joint"
            else "fixed-theta negative bound / N"
        )
    fig.tight_layout()
    fig.savefig(seed_dir / "convergence.png", dpi=180, bbox_inches="tight")
    fig.savefig(seed_dir / "convergence.pdf", bbox_inches="tight")
    plt.close(fig)


def run_seed(args, seed):
    seed_dir = args.out_dir / f"seed{seed}"
    metrics_path = seed_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"seed{seed}: cached")
        return
    seed_dir.mkdir(parents=True, exist_ok=True)

    split_seed = seed if args.split_seed is None else int(args.split_seed)
    data = load_benchmark_dataset(
        args.dataset, split_seed, args.n_train, args.n_test
    )
    x_train, y_train, _, _, metadata = data
    kernel_name = kn.canonical_kernel_name(args.kernel)
    initial = bench.init_params(x_train, args.m, seed, kernel_name=kernel_name)

    losses, builders = {}, {}
    for method in METHODS:
        losses[method], builders[method] = make_method_components(
            method,
            x_train,
            y_train,
            kernel_name,
            args.block_size,
            seed,
            args.m,
        )

    rows = []
    joint_params = {}
    print(f"seed{seed}: joint training")
    for method in METHODS:
        print(f"  -> {method}")
        params, curve, info, final_terms, initial_search_s, wall_s = train_method(
            method,
            initial,
            losses=losses,
            builders=builders,
            steps=args.steps,
            refresh_schedule=args.refresh_schedule,
        )
        joint_params[method] = params
        timing = {
            "optimizer_s": info["optimizer_s"],
            "compile_s": info["compile_s"],
            "search_s": initial_search_s + info["search_s"],
            "total_wall_s": initial_search_s + wall_s,
            "refreshes": info["refreshes"],
        }
        rows.append(evaluate(
            "joint", method, params, data=data, kernel_name=kernel_name,
            losses=losses, builders=builders, initial=initial,
            final_terms=final_terms, timing=timing,
        ))
        save_curve(seed_dir / f"curve_joint_{method}.npz", curve)
        np.savez_compressed(
            seed_dir / f"params_joint_{method}.npz", **params_numpy(params)
        )

    reference = joint_params["SGPR"]
    frozen = {key: value for key, value in reference.items() if key != "z"}
    print(f"seed{seed}: fixed SGPR theta, Z-only training")
    for method in METHODS:
        print(f"  -> {method}")
        params, curve, info, final_terms, initial_search_s, wall_s = train_method(
            method,
            initial,
            losses=losses,
            builders=builders,
            steps=args.steps,
            refresh_schedule=args.refresh_schedule,
            frozen=frozen,
        )
        timing = {
            "optimizer_s": info["optimizer_s"],
            "compile_s": info["compile_s"],
            "search_s": initial_search_s + info["search_s"],
            "total_wall_s": initial_search_s + wall_s,
            "refreshes": info["refreshes"],
        }
        rows.append(evaluate(
            "fixed_theta", method, params, data=data, kernel_name=kernel_name,
            losses=losses, builders=builders, initial=initial,
            final_terms=final_terms, timing=timing,
        ))
        save_curve(seed_dir / f"curve_fixed_theta_{method}.npz", curve)
        np.savez_compressed(
            seed_dir / f"params_fixed_theta_{method}.npz", **params_numpy(params)
        )

    for row in rows:
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
            "theta_reference": "joint-SGPR" if row["stage"] == "fixed_theta"
            else "self",
            "implementation": IMPLEMENTATION,
        })

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
            "steps": args.steps,
            "refresh_schedule": args.refresh_schedule,
            "fixed_theta_source": "joint-SGPR",
        },
        "code": code_state(SIM_ROOT.parent),
        "runtime": runtime_manifest(),
        "rows": rows,
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    write_rows(seed_dir / "metrics.csv", rows)
    plot_seed(seed_dir, args.n_train)
    print(f"seed{seed}: wrote {metrics_path}")


def summarize(out_dir):
    records = []
    for path in sorted(out_dir.glob("seed*/metrics.json")):
        payload = json.loads(path.read_text())
        if payload.get("implementation") == IMPLEMENTATION:
            records.extend(payload["rows"])
    if not records:
        raise SystemExit(f"no {IMPLEMENTATION} results under {out_dir}")

    # Derived from METHODS, not written out: cross_elbos builds the same names
    # from the same tuple, and a hardcoded copy silently desynced at the
    # BT -> RBT rename and made every summarise call raise KeyError.
    fields = (
        "rmse", "nll", "own_elbo_n",
        *(elbo_column(method) for method in METHODS),
        "residual_trace_n", "coverage_distance", "total_wall_s",
    )
    summary = []
    for stage in ("joint", "fixed_theta"):
        for method in METHODS:
            subset = [
                row for row in records
                if row["stage"] == stage and row["method"] == method
            ]
            if not subset:
                continue
            result = {"stage": stage, "method": method, "n_seeds": len(subset)}
            for field in fields:
                values = [float(row[field]) for row in subset]
                result[f"{field}_mean"] = float(np.mean(values))
                result[f"{field}_sem"] = sem(values)
            summary.append(result)
    write_rows(out_dir / "all_results.csv", records)
    write_rows(out_dir / "summary.csv", summary)

    import matplotlib.pyplot as plt

    panels = (
        ("elbo_sgpr_n", "SGPR ELBO / N", True),
        ("own_elbo_n", "Own ELBO / N", True),
        ("nll", "SGPR-predictor NLL", False),
        ("rmse", "SGPR-predictor RMSE", False),
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.0))
    x = np.arange(len(METHODS))
    for row_index, stage in enumerate(("joint", "fixed_theta")):
        stage_rows = [row for row in summary if row["stage"] == stage]
        by_method = {row["method"]: row for row in stage_rows}
        for axis, (field, title, higher) in zip(axes[row_index], panels):
            means = [by_method[method][f"{field}_mean"] for method in METHODS]
            errors = [by_method[method][f"{field}_sem"] for method in METHODS]
            errors = [0.0 if not np.isfinite(value) else value for value in errors]
            axis.bar(
                x, means, yerr=errors,
                color=[COLORS[method] for method in METHODS], capsize=3,
            )
            axis.set_xticks(x, METHODS, rotation=20, ha="right", fontsize=8)
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.2)
            if higher:
                axis.set_ylabel("higher is better")
        axes[row_index, 0].text(
            -0.32, 0.5,
            "Joint training" if stage == "joint" else "Fixed SGPR theta; Z only",
            transform=axes[row_index, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "fixed_theta_bound_comparison.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(out_dir / "fixed_theta_bound_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'fixed_theta_bound_comparison.pdf'}")


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
