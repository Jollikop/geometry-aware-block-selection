#!/usr/bin/env python3
"""Tell apart the two reasons an M-curve fails to settle.

A curve that keeps moving at the largest M looks the same in a figure whether
the optimiser stopped short or the bound is still loose, but the two call for
opposite remedies -- a longer schedule in the first case, a larger M or a
different dataset in the second -- so guessing between them is expensive. The
fitted hyperparameters separate them, and every campaign already stores them:
``noise_sd``, ``signal_var`` and the lengthscale range are merged into
``metrics.json`` by the runner.

Three failures are distinguished, and each is reported with the evidence that
identifies it rather than as a bare verdict.

*Optimiser stalled.* ``noise_sd`` still sits at its initial ``sqrt(0.1)`` and
the lengthscales at their initial 0.7. Nothing moved: the schedule, the
learning rate or the gradient path is at fault, and no amount of extra M will
help.

*Near-deterministic target.* ``noise_sd`` collapses towards zero. This is the
regime that retired bike, and it has a signature no other failure shares: the
objective grows without settling *because there is no finite optimum to settle
at*, since the likelihood rewards sigma^2 -> 0 without limit. Distinguishing
degrees matters -- a genuinely low-noise dataset like a road elevation map is
merely hard, while a leaked target is unusable -- so the residual RMSE is
printed next to the noise. If RMSE tracks ``noise_sd`` down together, the
target is being reconstructed, not predicted.

*Bound still loose.* The hyperparameters are healthy and stable across seeds,
but the objective still improves materially between the last two M. Nothing is
broken; the inducing budget is simply below what the dataset needs, which is a
finding to report rather than a bug to fix.

Read-only: it opens ``metrics.json`` files and writes nothing, so it is safe to
point at a campaign that is still running.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from svgp.analysis.paper_status import matches_protocol

HERE = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = HERE / "runs_gain_v10" / "paper_grid_v10"

# Initial values set by ``jax_svgp.init_params``. A fitted value still sitting
# on one of these is the fingerprint of a parameter that never received a
# gradient, which is a different disease from one that moved to a bad place.
INIT_NOISE_SD = math.sqrt(0.1)
INIT_ELL = 0.7
UNMOVED_RELATIVE_TOLERANCE = 0.02

# Below this fitted noise the likelihood has no interior optimum worth trusting;
# 0.05 in standardised units is 0.25% of the target variance.
NOISE_FLOOR = 0.05
# Relative improvement between the last two M below which the curve is settled.
PLATEAU_TOLERANCE = 0.01

FIELDS = ("last_epoch_stochastic_objective_estimate_per_datum", "rmse", "nll",
          "noise_sd", "signal_var", "ell_min", "ell_max")


def finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def read_runs(root: Path, run_tag: str | None):
    """Every complete metrics.json under ``root``, keyed for aggregation."""
    runs = []
    rejected = 0
    for path in sorted(root.glob("**/metrics.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not all(key in row for key in ("kernel", "dataset", "M", "method",
                                          "seed")):
            continue
        if run_tag and not matches_protocol(row, run_tag):
            rejected += 1
            continue
        runs.append(row)
    return runs, rejected


def summarise(rows):
    """Median over seeds of each field, plus the seed spread of the objective.

    The median rather than the mean, because a single diverged seed would
    otherwise decide the verdict for the other fifteen -- and a diverged seed
    is exactly what this script exists to find, so it must be visible as a
    spread rather than folded into the centre.
    """
    out = {"n_seeds": len({int(row["seed"]) for row in rows})}
    for field in FIELDS:
        values = np.array([finite(row.get(field)) for row in rows])
        values = values[np.isfinite(values)]
        out[field] = float(np.median(values)) if values.size else float("nan")
    objective = np.array([
        finite(row.get("last_epoch_stochastic_objective_estimate_per_datum"))
        for row in rows])
    objective = objective[np.isfinite(objective)]
    out["objective_spread"] = (
        float(np.percentile(objective, 75) - np.percentile(objective, 25))
        if objective.size >= 4 else float("nan"))
    return out


def unmoved(summary) -> bool:
    """True when both noise and lengthscale are still at their initial value."""
    noise, ell = summary["noise_sd"], summary["ell_min"]
    if not (math.isfinite(noise) and math.isfinite(ell)):
        return False
    return (abs(noise - INIT_NOISE_SD) <= UNMOVED_RELATIVE_TOLERANCE * INIT_NOISE_SD
            and abs(ell - INIT_ELL) <= UNMOVED_RELATIVE_TOLERANCE * INIT_ELL)


def plateau_gap(by_m) -> float:
    """Relative objective improvement between the two largest M, or NaN.

    Normalised by the total improvement over the whole M range, so the answer
    is 'what fraction of the gain is still arriving at the end' and does not
    depend on the units of the objective.
    """
    ms = sorted(by_m)
    if len(ms) < 3:
        return float("nan")
    values = [by_m[m]["last_epoch_stochastic_objective_estimate_per_datum"]
              for m in ms]
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    total = abs(values[-1] - values[0])
    if total < 1e-12:
        return 0.0
    return abs(values[-1] - values[-2]) / total


def verdict(by_m) -> tuple[str, str]:
    """A label and the measurement that earned it."""
    ms = sorted(by_m)
    last = by_m[ms[-1]]
    if unmoved(last):
        return ("OPTIMISER STALLED",
                f"noise_sd={last['noise_sd']:.4f} and "
                f"ell_min={last['ell_min']:.4f} are still at their initial "
                f"{INIT_NOISE_SD:.4f} and {INIT_ELL}")
    if last["noise_sd"] < NOISE_FLOOR:
        return ("NOISE AT THE FLOOR",
                f"noise_sd={last['noise_sd']:.4f} < {NOISE_FLOOR}, "
                f"RMSE={last['rmse']:.4f}: the target is near deterministic, so "
                "the objective grows because there is no finite optimum to "
                "settle at")
    gap = plateau_gap(by_m)
    if math.isfinite(gap) and gap > PLATEAU_TOLERANCE:
        return ("BOUND STILL LOOSE",
                f"{100 * gap:.1f}% of the total gain still arrives between "
                f"M={ms[-2]} and M={ms[-1]}: healthy hyperparameters, inducing "
                "budget too small")
    return ("CONVERGED",
            f"noise_sd={last['noise_sd']:.4f}, coda del guadagno "
            f"{100 * gap:.1f}%" if math.isfinite(gap) else
            f"noise_sd={last['noise_sd']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--run-tag", default=None,
                        help="keep only runs matching this official protocol")
    parser.add_argument("--kernels", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    args = parser.parse_args()

    root = Path(args.runs_dir)
    if not root.exists():
        raise SystemExit(f"{root} non esiste")
    runs, rejected = read_runs(root, args.run_tag)
    if not runs:
        raise SystemExit(f"no complete metrics.json under {root}")
    print(f"{len(runs)} run lette da {root}"
          + (f" ({rejected} scartate dal filtro protocollo)" if rejected else ""))

    keep_kernels = set(args.kernels) if args.kernels else None
    keep_datasets = set(args.datasets) if args.datasets else None
    keep_methods = set(args.methods) if args.methods else None

    grouped = defaultdict(list)
    for row in runs:
        key = (row["kernel"], row["dataset"], row["method"], int(row["M"]))
        if keep_kernels and row["kernel"] not in keep_kernels:
            continue
        if keep_datasets and row["dataset"] not in keep_datasets:
            continue
        if keep_methods and row["method"] not in keep_methods:
            continue
        grouped[key].append(row)

    cells = {key: summarise(rows) for key, rows in grouped.items()}
    verdicts = []
    for kernel, dataset, method in sorted({key[:3] for key in cells}):
        by_m = {key[3]: value for key, value in cells.items()
                if key[:3] == (kernel, dataset, method)}
        print(f"\n=== {kernel} / {dataset} / {method} ===")
        print(f"{'M':>6}{'semi':>6}{'obiettivo/N':>13}{'IQR semi':>10}"
              f"{'RMSE':>9}{'NLL':>9}{'noise_sd':>10}{'sig_var':>9}"
              f"{'ell_min':>9}{'ell_max':>9}")
        for m in sorted(by_m):
            s = by_m[m]
            print(f"{m:>6}{s['n_seeds']:>6}"
                  f"{s['last_epoch_stochastic_objective_estimate_per_datum']:>13.4f}"
                  f"{s['objective_spread']:>10.4f}{s['rmse']:>9.4f}"
                  f"{s['nll']:>9.4f}{s['noise_sd']:>10.4f}"
                  f"{s['signal_var']:>9.4f}{s['ell_min']:>9.4f}"
                  f"{s['ell_max']:>9.4f}")
        label, evidence = verdict(by_m)
        print(f"  -> {label}: {evidence}")
        verdicts.append((kernel, dataset, method, label))

    print("\n=== riepilogo ===")
    for label in ("OPTIMISER STALLED", "NOISE AT THE FLOOR",
                  "BOUND STILL LOOSE", "CONVERGED"):
        hits = [f"{k}/{d}/{m}" for k, d, m, l in verdicts if l == label]
        if hits:
            print(f"{label:>22}: {', '.join(hits)}")


if __name__ == "__main__":
    main()
