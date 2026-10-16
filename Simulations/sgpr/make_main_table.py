#!/usr/bin/env python3
"""Validate the official Kin40k SGPR runs and build the paper table.

One row per (kernel, method). The headline NLL uses the residual covariance
represented by the fitted bound; the standard sparse NLL is saved separately.

Kernels are grouped in the order given by ``--kernels``: RBF first and
Gibbs second, the remaining two after them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SIM_ROOT = Path(__file__).resolve().parent.parent

from common.provenance import (  # noqa: E402
    SGPR_RELEASE_PACKAGES,
    config_fingerprint,
    release_runtime_matches,
)


from sgpr.benchmark import (  # noqa: E402  (owner of the id)
    IMPLEMENTATION,
    METHOD_PREDICTOR,
)
from sgpr import protocol  # noqa: E402
GROUPS = (
    (
        "Dense covariance or dense search",
        ("Exact-GP", "E-SGPR", "METIS-SGPR"),
    ),
    (
        "No dense residual matrix",
        ("SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR"),
    ),
)
METHODS = tuple(method for _, methods in GROUPS for method in methods)
KERNEL_LABELS = {
    "rbf": "RBF",
    "gibbs": "Gibbs (non-stationary)",
    "matern32": "Mat\\'ern-3/2",
    "matern52": "Mat\\'ern-5/2",
    "amplitude-rbf": "Amplitude-RBF",
    "matern-periodic": "Mat\\'ern-5/2 $\\times$ periodic",
}
EXPECTED_SEEDS = tuple(range(10))
DEFAULT_KERNELS = ("rbf", "gibbs", "matern52", "amplitude-rbf")

# Everything that must be identical across the whole table. The kernel and the
# inducing budget are deliberately NOT here: they vary by row and are checked
# against the requested grid instead.
EXPECTED_CONFIG = {
    "implementation": IMPLEMENTATION,
    "dataset": "kin40k",
    "n_train": 5000,
    "n_test": 2000,
    "block_size": 500,
    "steps": protocol.LBFGS_STEPS,
    "refresh_schedule": list(protocol.LBFGS_REFRESH_SCHEDULE),
    "refresh_every": protocol.LBFGS_STEPS + 1,
    "target_refreshes": None,
    "log_every": 100,
    "optimizer": "lbfgs",
    "optimizer_backend": "scipy-l-bfgs-b",
    "scipy_lbfgsb_ftol": protocol.SCIPY_LBFGSB_FTOL,
    "scipy_lbfgsb_gtol": protocol.SCIPY_LBFGSB_GTOL,
    "scipy_lbfgsb_maxcor": protocol.SCIPY_LBFGSB_MAXCOR,
    "scipy_lbfgsb_maxfun": protocol.SCIPY_LBFGSB_MAXFUN,
    "scipy_lbfgsb_maxls": protocol.SCIPY_LBFGSB_MAXLS,
    "lr": 0.02,
    "noise_lr": None,
    "grad_clip": 10.0,
    "train_inducing": True,
    "initialization_policy": list(protocol.INITIALIZATION_POLICY),
    "initialization_nested_max_m": protocol.INITIALIZATION_NESTED_MAX_M,
    "warmup_max_cycles": protocol.WARMUP_MAX_CYCLES,
    "warmup_theta_steps": protocol.WARMUP_THETA_STEPS,
    "warmup_reselect_obj_n_tol": protocol.WARMUP_RESELECT_OBJ_N_TOL,
    "require_convergence": True,
    "convergence_window_steps": protocol.CONVERGENCE_WINDOW_STEPS,
    "convergence_obj_n_tol": protocol.CONVERGENCE_OBJ_N_TOL,
    "convergence_grad_max_n_tol": protocol.CONVERGENCE_GRAD_MAX_N_TOL,
    "convergence_patience": protocol.CONVERGENCE_PATIENCE,
    "data_source": "uci_datasets-1.1.0:kin40k",
}

#: Metrics aggregated over seeds. ``pred_s_*`` are prediction wall times.
AGGREGATED = (
    "obj_n", "rmse", "nll", "search_s", "search_build_ms",
    "pred_s_sparse", "pred_s_bound",
)


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=project / "Simulations/sgpr/runs_v22",
        help="tree holding <kernel>/kin40k/M<M>/<method>/seed<k>/metrics.json",
    )
    parser.add_argument("--kernels", nargs="*", default=list(DEFAULT_KERNELS))
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=project / "Report/Results/kin40k_sgpr_main_summary.csv",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=project / "Report/tables/kin40k_sgpr_main.tex",
    )
    return parser.parse_args()


def load_runs(runs_dir: Path, kernel: str, m: int) -> dict[str, list[dict]]:
    by_method = {method: [] for method in METHODS}
    for path in (runs_dir / kernel).rglob("metrics.json"):
        row = json.loads(path.read_text())
        if row.get("method") not in by_method:
            continue
        if row.get("implementation") != IMPLEMENTATION:
            continue
        if row.get("convergence_passed") is not True:
            continue
        if row.get("optimizer_convergence_reached") is not True:
            continue
        if float(row.get("final_grad_max_per_datum", math.inf)) \
                > protocol.CONVERGENCE_GRAD_MAX_N_TOL:
            continue
        if row.get("initialization_strategy") not in (
            *protocol.INITIALIZATION_POLICY, "not-applicable"
        ):
            continue
        if not isinstance(row.get("optimization_attempts"), list):
            continue
        if not (1 <= int(row.get("steps", -1)) <= protocol.LBFGS_STEPS):
            continue
        if int(row.get("max_steps", -1)) != protocol.LBFGS_STEPS:
            continue
        config = row.get("config", {})
        if row.get("config_id") != config_fingerprint(config):
            continue
        if any(config.get(key) != value for key, value in EXPECTED_CONFIG.items()):
            continue
        if config.get("kernel") != kernel or int(config.get("M", -1)) != int(m):
            continue
        if (
            config.get("method") != row.get("method")
            or int(config.get("seed", -1)) != int(row.get("seed", -2))
            or config.get("kernel") != row.get("kernel")
            or int(config.get("M", -1)) != int(row.get("M", -2))
            or config.get("dataset") != row.get("dataset")
        ):
            continue
        if not str(config.get("data_signature", "")).startswith("sha256:"):
            continue
        if not config.get("data_source"):
            continue
        commit = str(config.get("code_commit", ""))
        if len(commit) != 40 or config.get("code_source_dirty") is not False:
            continue
        runtime = row.get("runtime")
        devices = runtime.get("jax_devices") if isinstance(runtime, dict) else None
        if not release_runtime_matches(runtime, SGPR_RELEASE_PACKAGES):
            continue
        if not isinstance(devices, list) or not any(
            device.get("platform") == "gpu"
            for device in devices if isinstance(device, dict)
        ):
            continue
        # The predictor is a property of the method, declared once in
        # common.predictors. A row that reports a different one is not part of
        # this table -- that is the whole point of the column being comparable.
        expected_predictor = METHOD_PREDICTOR[row["method"]]
        if row.get("reported_predictor") != expected_predictor:
            continue
        if config.get("reported_predictor") != expected_predictor:
            continue
        headline_key = None if expected_predictor == "exact" else (
            "nll_sparse" if expected_predictor == "sparse" else "nll_bound"
        )
        if headline_key is not None:
            try:
                if abs(float(row["nll"]) - float(row[headline_key])) > 1e-10:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
        by_method[row["method"]].append(row)

    signatures = set()
    commits = set()
    for method, rows in by_method.items():
        seeds = sorted(int(row["seed"]) for row in rows)
        if seeds != list(EXPECTED_SEEDS):
            raise SystemExit(
                f"{method}: expected seeds {list(EXPECTED_SEEDS)}, found {seeds}"
            )
        signatures.update(row["config"]["data_signature"] for row in rows)
        commits.update(row["config"]["code_commit"] for row in rows)
    if len(signatures) != 1:
        raise SystemExit(f"expected one dataset signature, found {sorted(signatures)}")
    if len(commits) != 1:
        raise SystemExit(f"expected one code commit, found {sorted(commits)}")
    return by_method


def aggregate(rows: list[dict], predictor: str) -> dict[str, float]:
    result = {"n_seeds": len(rows)}
    for metric in AGGREGATED:
        # Search construction is undefined for methods that do not build a
        # partition and is serialized as NaN.  Treat those entries as missing:
        # Python 3.12.13's statistics.stdev otherwise crashes internally while
        # trying to reduce a sample containing NaN.
        values = [
            value
            for row in rows
            if row.get(metric) is not None
            for value in (float(row[metric]),)
            if math.isfinite(value)
        ]
        if not values:
            result[f"{metric}_mean"] = float("nan")
            result[f"{metric}_std"] = float("nan")
            continue
        result[f"{metric}_mean"] = statistics.mean(values)
        result[f"{metric}_std"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
    # The time the REPORTED predictor took, whichever one that is.
    result["pred_ms_mean"] = 1e3 * result[
        "pred_s_bound_mean" if predictor not in {"sparse", "exact"}
        else "pred_s_sparse_mean"
    ]
    return result


def write_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "kernel", "group", "method", "reported_predictor", "n_seeds",
        "obj_n_mean", "obj_n_std",
        "rmse_mean", "rmse_std",
        "nll_mean", "nll_std",
        "pred_ms_mean",
        "pred_s_sparse_mean", "pred_s_bound_mean",
        "search_s_mean", "search_s_std",
        "search_build_ms_mean", "search_build_ms_std",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for kernel, per_method in summary.items():
            for group, methods in GROUPS:
                for method in methods:
                    writer.writerow({
                        "kernel": kernel,
                        "group": group,
                        "method": method,
                        "reported_predictor": METHOD_PREDICTOR[method],
                        **per_method[method],
                    })


def cell(values: dict[str, float], metric: str, digits: int) -> str:
    """Format an already-aggregated ``mean (std)`` table cell.

    Not ``common.stats.mean_std``: that one takes a raw sample and returns a
    pair. ``aggregate`` has already reduced the seeds here, so this only
    renders. The two share a concept, not a signature.
    """
    return (f"{values[f'{metric}_mean']:.{digits}f} "
            f"({values[f'{metric}_std']:.{digits}f})")


def write_tex(path: Path, summary: dict, kernels) -> None:
    lines = [
        "% Generated by Simulations/sgpr/make_main_table.py.",
        "\\begin{table*}[t]",
        "\\caption{Kin40k collapsed SGPR, \\(5000/2000\\) train/test points and "
        "\\(M=256\\). Entries are mean (standard deviation) over ten seeds. "
        "Each method reports the predictive covariance its own bound models: "
        "the sparse, diagonal, fitted-block, or dense residual predictor. "
        "\\emph{Pred.} is the wall time of "
        "one full test-set prediction under that estimator.}",
        "\\label{tab:kin40k-sgpr}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}llrrrr@{}}",
        "\\toprule",
        "Method & Pred. cov. & Obj./N & RMSE & NLL & Pred. (ms) \\\\",
    ]
    predictor_label = {
        "sparse": "sparse", "diagonal": "diagonal residual",
        "block": "all fitted blocks", "full": "dense residual", "exact": "exact",
    }
    for kernel in kernels:
        per_method = summary[kernel]
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{KERNEL_LABELS.get(kernel, kernel)}}}}} \\\\"
        )
        for group_index, (group, methods) in enumerate(GROUPS):
            lines.append(
                f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{group}}}}} \\\\"
            )
            for method in methods:
                values = per_method[method]
                cells = [
                    method,
                    predictor_label[METHOD_PREDICTOR[method]],
                    cell(values, "obj_n", 3),
                    cell(values, "rmse", 3),
                    cell(values, "nll", 3),
                    f"{values['pred_ms_mean']:.1f}",
                ]
                if method == "SBT-SGPR":
                    cells = [f"\\textbf{{{value}}}" for value in cells]
                lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    build = [
        f"{KERNEL_LABELS.get(kernel, kernel)}: METIS "
        f"{summary[kernel]['METIS-SGPR']['search_build_ms_mean']:.0f} ms, SBT "
        f"{summary[kernel]['SBT-SGPR']['search_build_ms_mean']:.2f} ms"
        for kernel in kernels
    ]
    lines.extend([
        "\\multicolumn{6}{@{}p{0.98\\textwidth}@{}}{\\textbf{Partition construction:}\\enspace "
        + ";\\enspace ".join(build) + "} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    summary, total = {}, 0
    for kernel in args.kernels:
        by_method = load_runs(args.runs_dir, kernel, args.m)
        summary[kernel] = {
            method: aggregate(rows, METHOD_PREDICTOR[method])
            for method, rows in by_method.items()
        }
        count = sum(len(rows) for rows in by_method.values())
        total += count
        print(f"  {kernel:10s} {count}/{len(METHODS) * len(EXPECTED_SEEDS)} runs")
    write_csv(args.csv_out, summary)
    write_tex(args.tex_out, summary, args.kernels)
    print(f"validated {total}/"
          f"{len(args.kernels) * len(METHODS) * len(EXPECTED_SEEDS)} runs")
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.tex_out}")


if __name__ == "__main__":
    main()
