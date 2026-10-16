#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official driver of the collapsed benchmark.

A thin layer over ``sgpr.benchmark``: it resolves datasets, fingerprints the
configuration, runs, and persists. Splitting, standardisation, optimisation,
prediction and metrics all happen in the benchmark core and in ``common``.

Default methods: SGPR, T-SGPR, RBT-SGPR, SBT-SGPR. Dense diagnostics
(Exact-GP, E-SGPR, METIS-SGPR) are opt-in via ``--methods``.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "mpl_cache"),
)

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent

import jax
jax.config.update("jax_enable_x64", True)

from common import data_split as jd             # noqa: E402
from common import datasets as dm               # noqa: E402
from common import datasets_uci as dr           # noqa: E402
from common.provenance import (                 # noqa: E402
    SGPR_RELEASE_PACKAGES,
    config_fingerprint,
    release_runtime_matches,
    runtime_manifest,
    sha256_file,
)
from common.kernels_nonstationary import (      # noqa: E402
    stationary_to_gated_log_ell,
)
from sgpr import benchmark as bench             # noqa: E402
from sgpr import protocol                       # noqa: E402

DEFAULT_RUNS_DIR = HERE / "runs_v23"
DEFAULT_LOGS_DIR = HERE / "logs_v23"
DEFAULT_TABLES_DIR = HERE / "tables_v18"
DATA_DIR = dr.DATA_DIR                         # Simulations/dataset/sgpr/
KIN40K_NPZ = jd.DATASET_DIR / "kin40k.npz"     # Simulations/dataset/kin40k.npz
# Refresh schedules are optimizer-aware.
#
# Adam needs many first-order steps, so its schedule is long. SciPy L-BFGS-B
# converges on the smooth, full-batch collapsed objective in far fewer
# iterations, and its curvature history is
# RESET at every structure refresh (the objective changes). So an intelligent
# L-BFGS schedule must balance three things:
#   1. refresh OFTEN early, when Z/theta (hence the residual geometry) move fast;
#   2. give each phase ENOUGH steps for L-BFGS to rebuild curvature and settle
#      on the fixed structure (too-frequent refreshes keep resetting the memory
#      and cripple the quasi-Newton step);
#   3. END on a long post-refresh phase so the frozen structure matches the
#      FINAL geometry and the convergence diagnostic is evaluated without an
#      objective change inside its window.
# The phase semantics: phase 1 uses the initial structure; every later phase
# refreshes BEFORE optimising. A K-phase schedule => K-1 refreshes, and the
# last phase always runs on a freshly refreshed structure.
ADAM_REFRESH_SCHEDULE = "40,60,100,400,600"          # 1200 first-order steps
LBFGS_REFRESH_SCHEDULE = ",".join(
    str(value) for value in protocol.LBFGS_REFRESH_SCHEDULE
)                                                      # 30000-iteration cap
DEFAULT_OPTIMIZER = "lbfgs"
DEFAULT_REFRESH_SCHEDULE = (
    LBFGS_REFRESH_SCHEDULE if DEFAULT_OPTIMIZER == "lbfgs"
    else ADAM_REFRESH_SCHEDULE
)
DEFAULT_STEPS = sum(int(x) for x in DEFAULT_REFRESH_SCHEDULE.split(","))
DEFAULT_REFRESH_EVERY = 200
DEFAULT_LOG_EVERY = 100


def default_refresh_schedule_for(optimizer):
    """Optimizer-aware default phase schedule (intelligent refresh placement)."""
    return (
        LBFGS_REFRESH_SCHEDULE if str(optimizer).lower() == "lbfgs"
        else ADAM_REFRESH_SCHEDULE
    )
SGPR_KERNEL_ORDER = list(bench.KERNEL_ORDER)
SGPR_METHOD_ORDER = bench.DEFAULT_METHOD_ORDER
# Priority of loaders: kin40k npz > datasets_sgpr (dr) > shared modern registry
# (dm: the nine datasets declared by the SVGP paper campaign).
# The modern loaders can carry angular (periodic) features; those columns are
# kept unstandardised via angle_cols.
_MODERN_ONLY = tuple(k for k in dm.DATASET_KEYS
                     if k != "kin40k" and k not in dr.DATASET_KEYS)
BENCHMARK_DATASETS = ("kin40k", *dr.DATASET_KEYS, *_MODERN_ONLY)


def dataset_signature(dataset):
    if dataset == "kin40k":
        dm.load_raw("kin40k", data_dir=jd.DATASET_DIR)
        path = KIN40K_NPZ
    elif dataset in dr.DATASET_KEYS:
        path = dr.cache_path(dataset, DATA_DIR)
    else:                                        # shared modern registry
        dm.load_raw(dataset)                     # ensure the cache exists
        path = dm.raw_cache_path(dataset)
    return sha256_file(path)


def dataset_source(dataset):
    if dataset == "kin40k":
        return dm.dataset_source("kin40k")
    if dataset in dr.DATASET_KEYS:
        return f"UCI:{dataset}"
    return dm.dataset_source(dataset)


def cache_matches(path, config_id, *, require_params=False,
                  require_blocks=False, require_convergence=False):
    """Reuse only a run with the exact same experimental configuration."""
    try:
        row = json.loads(path.read_text())
        config = row.get("config") or {}
        valid = (
            row.get("implementation") == bench.IMPLEMENTATION
            and row.get("config_id") == config_id
            and config_fingerprint(config) == config_id
        )
        if require_params:
            valid = valid and (path.parent / "params.npz").exists()
        if require_blocks:
            valid = valid and (path.parent / "final_blocks.npz").exists()
        if require_convergence:
            valid = valid and row.get("convergence_passed") is True
        return valid
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def load_benchmark_dataset(dataset, split_seed, n_train, n_test,
                           split_rng=jd.DEFAULT_SPLIT_RNG):
    if dataset == "kin40k":
        dm.load_raw("kin40k", data_dir=jd.DATASET_DIR)
        return jd.load_npz(
            KIN40K_NPZ, split_seed, n_train, n_test, split_rng=split_rng
        )
    if dataset in dr.DATASET_KEYS:
        X, y, _ = dr.load_raw(dataset, data_dir=DATA_DIR)   # host file IO only
        return jd.split_standardize(
            X, y, split_seed, n_train, n_test, split_rng=split_rng
        )
    # shared modern registry (co2, traffic, 3droad, ...). Angular
    # (periodic) feature columns are kept unstandardised via angle_cols.
    X, y = dm.load_raw(dataset)
    return jd.split_standardize(
        X,
        y,
        split_seed,
        n_train,
        n_test,
        angle_cols=dm.angle_cols_for(dataset),
        split_rng=split_rng,
    )


def resolve_split_seed(args, model_seed):
    """Resolve the data split independently from model initialization."""
    return int(model_seed if args.split_seed is None else args.split_seed)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(BENCHMARK_DATASETS))
    ap.add_argument("--methods", nargs="*", default=SGPR_METHOD_ORDER)
    ap.add_argument("--kernel", default=None,
                    help="single kernel alias; overrides the default rbf run "
                         "unless --kernels is passed.")
    ap.add_argument("--kernels", nargs="*", default=None,
                    help=f"kernel sweep; available kernels: "
                         f"{' '.join(SGPR_KERNEL_ORDER)}")
    ap.add_argument("--m", nargs="*", type=int, default=[256, 512])
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    ap.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="train/test split seed; default: use each run's model seed",
    )
    ap.add_argument(
        "--split-rng",
        choices=jd.SUPPORTED_SPLIT_RNGS,
        default=jd.DEFAULT_SPLIT_RNG,
        help="deterministic split implementation; legacy JAX is archival only",
    )
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument(
        "--n-blocks",
        type=int,
        default=20,
        help="BT/SBT block count -> B = ceil(N/n_blocks).",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="explicit BT/SBT block size B; overrides --n-blocks.",
    )
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help="optimisation steps, equal for every method.")
    ap.add_argument("--refresh-every", type=int,
                    default=DEFAULT_REFRESH_EVERY)
    ap.add_argument("--refreshes", type=int, default=None,
                    help="target number of structure refreshes; overrides "
                         "--refresh-every for adaptive structured methods.")
    ap.add_argument("--refresh-schedule", default=None,
                    help="comma-separated phase lengths, e.g. 2,3,5,20,30; "
                         "overrides --steps/--refreshes/--refresh-every. If "
                         "omitted, an optimizer-aware default is used: "
                         f"L-BFGS={LBFGS_REFRESH_SCHEDULE}, "
                         f"Adam/Lion={ADAM_REFRESH_SCHEDULE}.")
    ap.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    ap.add_argument("--optimizer", choices=["adam", "lion", "lbfgs"],
                    default=DEFAULT_OPTIMIZER,
                    help="collapsed objective is smooth/full-batch -> L-BFGS "
                         "(SciPy L-BFGS-B with JAX gradients) converges in fewer steps "
                         "and is the default here; adam/lion are first-order "
                         "alternatives. Note: under L-BFGS the line search sets "
                         "the step, so --lr and --noise-lr are ignored.")
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument(
        "--noise-lr",
        type=float,
        default=None,
        help="optional Adam/Lion learning rate for log_s2 only; defaults to --lr.",
    )
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--no-live", action="store_true",
                    help="disable the live per-step print (keeps the per-run "
                         "search/optimizer/per-gradient timing summary).")
    ap.add_argument(
        "--logdet-diagnostics",
        dest="logdet_diagnostics",
        action="store_true",
        default=True,
        help="log the residual log-determinant captured-mass diagnostic "
             "(theory measures) per run. On by default.",
    )
    ap.add_argument(
        "--no-logdet-diagnostics",
        dest="logdet_diagnostics",
        action="store_false",
        help="disable the captured-mass diagnostic.",
    )
    ap.add_argument(
        "--logdet-max-n",
        type=int,
        default=6000,
        help="skip the O(N^2/N^3) dense captured-mass diagnostic when the "
             "training size exceeds this cap (keeps large runs cheap).",
    )
    ap.add_argument(
        "--full-predictor-max-n",
        type=int,
        default=bench.DEFAULT_FULL_PREDICTOR_MAX_N,
        help="refuse a run whose reported predictor is Eq. (12) above this "
             "training size. A guard rail, not a fallback: exceeding it stops "
             "the run rather than reporting a different estimator.",
    )
    ap.add_argument(
        "--save-block-contributions",
        action="store_true",
        help="also store the full [n_test, n_blocks] Eq. (12) decomposition, "
             "not just its per-block means and the blocks-included curve.",
    )
    ap.add_argument(
        "--no-train-inducing",
        action="store_true",
        help="freeze Z at its initial value and optimise only the "
             "hyperparameters.",
    )
    ap.add_argument(
        "--initialization-strategy",
        choices=list(bench.initz.INITIALIZATION_STRATEGIES),
        default=protocol.INITIALIZATION_POLICY[0],
        help="primary inducing-point initialization; release convergence "
             "runs use the full fingerprinted fallback policy",
    )
    ap.add_argument(
        "--warm-start-runs-dir",
        type=Path,
        default=None,
        help="for Gibbs, initialise each non-exact method from the matching "
             "certified RBF run in this result tree",
    )
    ap.add_argument(
        "--save-params",
        action="store_true",
        help="save final params.npz and structured final_blocks.npz artefacts",
    )
    ap.add_argument(
        "--require-convergence",
        action="store_true",
        help="fail a non-converged run after saving its diagnostic curve; "
             "failed results are never accepted from cache",
    )
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                    help="root of the per-run result tree.")
    ap.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR,
                    help="directory holding the shared result CSV.")
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR,
                    help="directory for the aggregate table.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--no-tables", action="store_true",
                    help="skip final aggregate table build; useful for "
                         "parallel workers.")
    return ap.parse_args()


def load_rbf_gibbs_warm_start(root, dataset, m, method, seed):
    """Load and exactly embed a certified stationary donor into Gibbs."""
    run_dir = root / "rbf" / dataset / f"M{m}" / method / f"seed{seed}"
    metrics_path = run_dir / "metrics.json"
    params_path = run_dir / "params.npz"
    if not metrics_path.is_file() or not params_path.is_file():
        raise FileNotFoundError(f"missing certified RBF warm start: {run_dir}")
    metrics = json.loads(metrics_path.read_text())
    if (
        metrics.get("implementation") != bench.IMPLEMENTATION
        or metrics.get("convergence_passed") is not True
    ):
        raise ValueError(f"RBF warm start is not certified v23: {run_dir}")
    with np.load(params_path, allow_pickle=False) as archive:
        params = {key: jax.numpy.asarray(archive[key]) for key in archive.files}
    params["log_ell"] = stationary_to_gated_log_ell(params["log_ell"])
    d = int(params["z"].shape[1])
    params["warp"] = jax.numpy.zeros((d, d), dtype=params["z"].dtype)
    return params, metrics, params_path


def main():
    args = parse_args()
    runtime = runtime_manifest()
    if not release_runtime_matches(runtime, SGPR_RELEASE_PACKAGES):
        raise SystemExit(
            "runtime package versions do not match Simulations/requirements.txt"
        )
    args.live = not args.no_live
    unknown_datasets = sorted(set(args.datasets) - set(BENCHMARK_DATASETS))
    if unknown_datasets:
        raise SystemExit(
            f"unknown datasets {unknown_datasets}; choose from "
            f"{list(BENCHMARK_DATASETS)}"
        )
    kernel_inputs = (
        args.kernels if args.kernels is not None
        else ([args.kernel] if args.kernel else ["rbf"])
    )
    try:
        args.kernels = [bench.canonical_kernel_name(k) for k in kernel_inputs]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    # Optimizer-aware default refresh schedule (only when the user did not pass
    # one explicitly): L-BFGS gets the short, curvature-friendly schedule.
    if args.refresh_schedule is None:
        args.refresh_schedule = default_refresh_schedule_for(args.optimizer)
    try:
        bench.configure_refresh_protocol(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    bad = [m for m in args.methods if m not in bench.METHODS]
    if bad:
        raise SystemExit(
            f"unknown methods {bad}; choose from {bench.METHOD_ORDER}"
        )

    if args.tables_only:
        ids = set()
        for kernel_name in args.kernels:
            table_args = copy.copy(args)
            table_args.kernel = kernel_name
            for dataset in args.datasets:
                table_args.data_source = dataset_source(dataset)
                if dataset != "kin40k":
                    dr.load_raw(dataset, data_dir=DATA_DIR)
                signature = dataset_signature(dataset)
                for method in args.methods:
                    for m in args.m:
                        for seed in args.seeds:
                            config_args = copy.copy(table_args)
                            config_args.model_seed = int(seed)
                            config_args.split_seed = resolve_split_seed(
                                config_args, seed
                            )
                            ids.add(bench.make_run_config(
                                method, m, seed, dataset, config_args, signature
                            )[1])
        bench.build_tables(
            args.runs_dir, args.tables_dir, args.datasets, args.m, config_ids=ids
        )
        return
    n_runs = (len(args.kernels) * len(args.datasets) * len(args.m)
              * len(args.methods) * len(args.seeds))
    print(f"Device: {jax.devices()[0]} (float64)  planned runs: {n_runs}")
    done, failures = 0, 0
    config_ids = set()
    data_cache = {}
    for kernel_name in args.kernels:
        for dataset in args.datasets:
            if dataset in dr.DATASET_KEYS:
                dr.load_raw(dataset, data_dir=DATA_DIR)
            signature = dataset_signature(dataset)
            for m in args.m:
                for method in args.methods:
                    for seed in args.seeds:
                        done += 1
                        tag = (f"[{done}/{n_runs}] {kernel_name} {dataset} "
                               f"M{m} {method} seed{seed}")
                        run_args = copy.copy(args)
                        run_args.kernel = kernel_name
                        run_args.data_source = dataset_source(dataset)
                        run_args.model_seed = int(seed)
                        run_args.split_seed = resolve_split_seed(run_args, seed)
                        warm_params = None
                        if (
                            kernel_name == "gibbs"
                            and method != "Exact-GP"
                            and args.warm_start_runs_dir is not None
                        ):
                            warm_params, warm_metrics, warm_path = (
                                load_rbf_gibbs_warm_start(
                                    args.warm_start_runs_dir,
                                    dataset,
                                    m,
                                    method,
                                    seed,
                                )
                            )
                            run_args.warm_start_donor = warm_metrics.get(
                                "config_id"
                            )
                            run_args.warm_start_params_sha256 = sha256_file(
                                warm_path
                            )
                        config, config_id = bench.make_run_config(
                            method, m, seed, dataset, run_args, signature
                        )
                        config_ids.add(config_id)
                        out = (args.runs_dir / kernel_name / dataset / f"M{m}"
                               / method / f"seed{seed}")
                        metrics = out / "metrics.json"
                        if not args.force and cache_matches(
                            metrics,
                            config_id,
                            require_params=bool(args.save_params),
                            require_blocks=(
                                bool(args.save_params)
                                and bench.METHODS[method] == "terms"
                            ),
                            require_convergence=bool(args.require_convergence),
                        ):
                            print(f"{tag}  (cached)")
                            continue
                        data_key = (
                            dataset, int(run_args.split_seed),
                            int(args.n_train), int(args.n_test),
                            str(run_args.split_rng),
                        )
                        if data_key not in data_cache:
                            data_cache[data_key] = load_benchmark_dataset(
                                dataset,
                                run_args.split_seed,
                                args.n_train,
                                args.n_test,
                                split_rng=run_args.split_rng,
                            )
                        data = data_cache[data_key]
                        strategies = (
                            (args.initialization_strategy,)
                            if method == "Exact-GP" or not args.require_convergence
                            else protocol.INITIALIZATION_POLICY
                        )
                        completed_attempts = []
                        attempt_errors = []
                        for strategy in strategies:
                            attempt_args = copy.copy(run_args)
                            attempt_args.seed = seed
                            attempt_args.initialization_strategy = strategy
                            if (
                                warm_params is not None
                                and strategy == strategies[0]
                            ):
                                attempt_args._initial_params = warm_params
                            attempt_args._run_config = config
                            attempt_args._config_id = config_id
                            try:
                                attempt = bench.train_and_eval(
                                    method, m, seed, dataset, data, attempt_args
                                )
                            except Exception as exc:  # noqa: BLE001
                                attempt_errors.append({
                                    "strategy": strategy,
                                    "error": f"{type(exc).__name__}: {exc}",
                                })
                                print(
                                    f"{tag}  {strategy} FAILED: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                                continue
                            attempt_res, attempt_curve, attempt_artifacts = attempt
                            completed_attempts.append(attempt)
                            if attempt_res.get("convergence_passed") is True:
                                break

                        if not completed_attempts:
                            failures += 1
                            continue
                        certified = [
                            attempt for attempt in completed_attempts
                            if attempt[0].get("convergence_passed") is True
                        ]
                        selected = min(
                            certified or completed_attempts,
                            key=lambda attempt: float(attempt[0]["obj_n"]),
                        )
                        res, curve, artifacts = selected
                        res["restart_count"] = (
                            len(completed_attempts) + len(attempt_errors) - 1
                        )
                        res["optimization_attempts"] = [
                            {
                                "strategy": attempt[0].get(
                                    "initialization_strategy"
                                ),
                                "convergence_passed": bool(attempt[0].get(
                                    "convergence_passed", False
                                )),
                                "obj_n": float(attempt[0]["obj_n"]),
                                "steps": int(attempt[0]["steps"]),
                                "final_grad_max_per_datum": float(attempt[0][
                                    "final_grad_max_per_datum"
                                ]),
                            }
                            for attempt in completed_attempts
                        ] + attempt_errors
                        bench.save_run(
                            args.runs_dir,
                            res,
                            curve,
                            res["n_train"],
                            artifacts=artifacts if args.save_params else None,
                        )
                        bench.append_row(args.logs_dir, res)
                        print(f"{tag}  rmse={res['rmse']:.4f} "
                              f"nll={res['nll']:.4f} "
                              f"obj/N={res['obj_n']:+.4f} "
                              f"elbo/N={res['elbo_n']:+.4f} "
                              f"search={res['search_s']:.1f}s "
                              f"opt={res['optimizer_s']:.1f}s "
                              f"grad={res['grad_ms']:.2f}ms")
                        if args.require_convergence and not res["convergence_passed"]:
                            reasons = ",".join(
                                res.get("convergence_failure_reasons") or
                                ["unknown"]
                            )
                            print(
                                f"{tag}  FAILED CONVERGENCE: reasons={reasons} "
                                f"tail/N={res['convergence_tail_range_obj_n']:.3e} "
                                f"(tol={res['convergence_obj_n_tol']:.3e}) "
                                f"grad_max/N={res['final_grad_max_per_datum']:.3e} "
                                f"(tol={res['convergence_grad_max_n_tol']:.3e}) "
                                f"optimizer_reached="
                                f"{res['optimizer_convergence_reached']}"
                            )
                            failures += 1
    if not args.no_tables:
        bench.build_tables(
            args.runs_dir, args.tables_dir, args.datasets, args.m, config_ids=config_ids
        )
    if failures:
        raise SystemExit(f"{failures} run(s) failed")
    print("Done.")


if __name__ == "__main__":
    main()
