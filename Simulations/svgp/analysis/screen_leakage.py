#!/usr/bin/env python3
"""Screen datasets for target leakage before spending compute on them.

``bike`` cost a full campaign column before anyone noticed its target was
``log(casual + registered)`` with both columns present as features. The failure
mode is silent: training looks like non-convergence (sigma^2 -> 0, diverging
ELBO) rather than like a broken dataset, so it is diagnosed late and expensively.

Four statistics, cheap enough to run on every dataset before every campaign:

``R2lin``     linear R^2 of y on X. Catches an exact linear combination.
``R2rank``    the same on rank-transformed columns and target. Catches
              ``y = monotone(linear combination)``, the class ``bike`` belongs
              to, without enumerating pairs -- which matters when d is large.
``max|rho|``  largest absolute Spearman correlation of a single column.
``pairs``     largest absolute Spearman correlation of a column *sum*, over all
              pairs. Skipped above ``--max-pair-dim`` columns, where the O(d^2)
              scan stops being worth it and ``R2rank`` covers the same ground.

A statistic near 1 means the target is reconstructible from the features and the
dataset measures nothing. Near 1 is not the same as high: a legitimate strong
signal (a time series against its own trend) sits at 0.99, deterministic leakage
sits at 1.0000. The threshold only decides what gets a second look; the
judgement is yours, and the reason belongs in ``DATASETS[key]['note']``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent

from svgp import plans  # noqa: E402
from common import datasets as dm  # noqa: E402


def r2(features: np.ndarray, target: np.ndarray) -> float:
    design = np.c_[features, np.ones(len(features))]
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coef
    total = ((target - target.mean()) ** 2).sum()
    return float(1.0 - (residual ** 2).sum() / total) if total > 0 else float("nan")


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0.0:                       # constant column: undefined
        return 0.0
    value = spearmanr(a, b).statistic
    return 0.0 if not np.isfinite(value) else abs(float(value))


def screen(key: str, data_dir: Path, max_pair_dim: int, block: int):
    X, y = dm.load_raw(key, data_dir=data_dir)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[finite], y[finite]
    n, d = X.shape

    lin = r2(X, y)
    rank = r2(np.apply_along_axis(rankdata, 0, X), rankdata(y))
    single = max((safe_spearman(X[:, j], y), j) for j in range(d))
    pair = (float("nan"), -1, -1)
    if d <= max_pair_dim:
        pair = max((safe_spearman(X[:, i] + X[:, j], y), i, j)
                   for i in range(d) for j in range(i + 1, d))

    constant = [j for j in range(d) if np.ptp(X[:, j]) == 0.0]
    tiny = [j for j in range(d)
            if 0.0 < X[:, j].std() < 1e-8 * max(1.0, abs(X[:, j].mean()))]
    return {
        "key": key, "n": n, "d": d,
        "blocks": (n * 4 // 5) // max(1, block),
        "R2lin": lin, "R2rank": rank,
        "max_rho": single[0], "max_rho_col": single[1],
        "pair_rho": pair[0], "pair_cols": (pair[1], pair[2]),
        "constant_cols": constant, "near_constant_cols": tiny,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="default: every dataset of the live grid plan")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--threshold", type=float, default=0.98)
    ap.add_argument("--max-pair-dim", type=int, default=30)
    ap.add_argument("--block", type=int, default=500,
                    help="block size used to report the resulting block count")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else dm.DATA_DIR
    keys = args.datasets or list(plans.GRID_DATASETS)

    print(f"{'dataset':18}{'N':>9}{'d':>5}{'blocks':>8}"
          f"{'R2lin':>9}{'R2rank':>9}{'max|rho|':>10}{'pairs':>9}")
    flagged, failed = [], []
    for key in keys:
        try:
            r = screen(key, data_dir, args.max_pair_dim, args.block)
        except Exception as exc:                              # noqa: BLE001
            print(f"{key:18}  {type(exc).__name__}: {exc}")
            failed.append(key)
            continue
        worst = max(v for v in (r["R2lin"], r["R2rank"], r["max_rho"],
                                r["pair_rho"]) if np.isfinite(v))
        mark = "   <<< SUSPECT" if worst > args.threshold else ""
        pair_text = (f"{r['pair_rho']:9.4f}" if np.isfinite(r["pair_rho"])
                     else f"{'-':>9}")
        print(f"{key:18}{r['n']:9}{r['d']:5}{r['blocks']:8}"
              f"{r['R2lin']:9.4f}{r['R2rank']:9.4f}{r['max_rho']:10.4f}"
              f"{pair_text}{mark}")
        if mark:
            flagged.append(r)
        if r["constant_cols"]:
            print(f"{'':18}  constant columns: {r['constant_cols']}")
        if r["near_constant_cols"]:
            # std ~ 0 is divided by std + 1e-12 in split_standardize, so a
            # near-constant column is amplified enormously instead of zeroed.
            print(f"{'':18}  NEAR-constant columns (unstable under "
                  f"standardisation): {r['near_constant_cols']}")
        if r["blocks"] < 10:
            print(f"{'':18}  only {r['blocks']} blocks at B={args.block}: "
                  "too few to separate the block methods")

    if flagged:
        print("\nsuspects:")
        for r in flagged:
            print(f"  {r['key']}: strongest single column is {r['max_rho_col']}"
                  + (f", strongest pair is {r['pair_cols']}"
                     if np.isfinite(r["pair_rho"]) else "")
                  + ". Check whether the target is a function of these before "
                    "running anything.")
    if failed:
        raise SystemExit(f"\nnot loadable: {', '.join(failed)}")


if __name__ == "__main__":
    main()
