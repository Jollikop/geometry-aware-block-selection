#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Appendix figure: how fast each fitted residual model reaches its endpoint.

Each method reports the covariance represented by its own bound: T-SGPR uses a
diagonal residual model, while RBT-SGPR and SBT-SGPR use independently
factorised block models.  For a fitted block model the correction is

    sum_b d_*b (D_bb + s2 I)^-1 d_b* = sum_b c_b(x*).

The plotted partial sums run from the standard sparse predictor to each
method's complete fitted residual predictor.  They are truncations of the
block approximation, not partial evaluations of the dense Eq. (12) predictor.

A cover that concentrates the correction in its leading blocks reaches the dense
reference sooner. That is the whole claim about anchoring, and this figure is
where it is visible: T-SGPR spreads the mass over singletons, RBT-SGPR over
random blocks, SBT-SGPR over inducing-anchored ones.

Reads a finished runs tree and plots ``block_decomposition.npz``. Retrains
nothing, so it runs on a laptop.

    python -m sgpr.plot_block_convergence --runs-dir sgpr/runs_v18/rbf/kin40k/M256
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import stats as st
from common.plot_style import (
    method_color,
    method_linewidth,
    method_marker,
    method_markersize,
)

#: The three methods whose covers differ. SGPR has no residual model and E-SGPR
#: has no cover, so neither has a curve to draw.
METHODS = ("T-SGPR", "RBT-SGPR", "SBT-SGPR")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True,
                        help="tree holding <method>/seed<k>/block_decomposition.npz")
    parser.add_argument("--out-base", type=Path, required=True,
                        help="output path without extension")
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--error-mode", default="sem",
                        choices=st.UNCERTAINTY_MODES)
    parser.add_argument("--formats", nargs="*", default=["pdf", "png"])
    parser.add_argument("--require-seeds", type=int, default=0,
                        help="refuse to plot a method with fewer seeds than this")
    return parser.parse_args()


def read_method(runs_dir: Path, method: str):
    """Stack the blocks-included curves of every seed of one method.

    Seeds may end up with a different block count when N is not divisible by the
    block size, so curves are read onto a common axis in *fraction of the cover*
    and the absolute count is reported separately.
    """
    curves, captured, points, counts, seeds = [], [], [], [], []
    for path in sorted(runs_dir.glob(f"{method}/seed*/block_decomposition.npz")):
        with np.load(path) as archive:
            nll = np.asarray(archive["nll_vs_blocks"], dtype=np.float64)
            frac = np.asarray(archive["captured_fraction"], dtype=np.float64)
            pts = np.asarray(archive["points_included"], dtype=np.float64)
        metrics_path = path.with_name("metrics.json")
        if metrics_path.exists():
            row = json.loads(metrics_path.read_text())
            expected = "diagonal" if method == "T-SGPR" else "block"
            if row.get("reported_predictor") != expected:
                continue
            seeds.append(int(row.get("seed", -1)))
        curves.append(nll)
        captured.append(frac)
        points.append(pts)
        counts.append(nll.size - 1)
    if not curves:
        return None
    width = min(len(c) for c in curves)
    return {
        "nll": np.stack([c[:width] for c in curves]),
        "captured": np.stack([c[:width] for c in captured]),
        "points": np.stack([p[:width] for p in points]),
        "n_blocks": int(min(counts)),
        "seeds": sorted(seeds),
    }


def draw(data, out_base: Path, formats, error_mode):
    """Two panels: the NLL path, and the share of the correction captured."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    for method, entry in data.items():
        # Points, not blocks: T-SGPR splits the correction over singletons and
        # the block methods over blocks of B, so "eight units" means two very
        # different amounts of conditioning. Points make the axis comparable
        # and read directly as cost.
        blocks = np.maximum(np.nanmean(entry["points"], axis=0), 0.7)
        style = dict(
            color=method_color(method),
            linewidth=method_linewidth(method),
            marker=method_marker(method),
            markersize=method_markersize(method),
            markevery=max(1, blocks.size // 8),
            label=method,
        )
        for ax, key, in ((axes[0], "nll"), (axes[1], "captured")):
            mean = np.nanmean(entry[key], axis=0)
            spread = st.uncertainty(entry[key], error_mode, axis=0)
            ax.plot(blocks, mean, **style)
            ax.fill_between(blocks, mean - spread, mean + spread,
                            color=method_color(method), alpha=0.15, linewidth=0)
            style = {k: v for k, v in style.items() if k != "label"}

    axes[0].set_xlabel("training points conditioned on")
    axes[0].set_xscale("symlog", linthresh=1.0)
    axes[0].set_ylabel("test NLL")
    axes[0].set_title("Sparse to fitted residual predictor", fontsize=9)
    axes[1].set_xlabel("training points conditioned on")
    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set_ylabel("share of fitted residual correction")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].axhline(1.0, color="0.55", linewidth=0.8, linestyle=(0, (4, 3)))
    axes[1].set_title("How fast each model reaches its endpoint", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2, linewidth=0.6)
    axes[0].legend(fontsize=8, framealpha=0.85)

    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in formats:
        path = out_base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=220 if suffix == "png" else None,
                    bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def write_summary(data, path: Path, error_mode):
    """The plotted numbers, so the figure can be read without opening the PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "method", "blocks_included", "points_included", "n_seeds",
            "nll_mean", f"nll_{error_mode}",
            "captured_mean", f"captured_{error_mode}",
        ])
        for method, entry in data.items():
            n_seeds = entry["nll"].shape[0]
            nll_mean = np.nanmean(entry["nll"], axis=0)
            nll_spread = st.uncertainty(entry["nll"], error_mode, axis=0)
            cap_mean = np.nanmean(entry["captured"], axis=0)
            cap_spread = st.uncertainty(entry["captured"], error_mode, axis=0)
            points_mean = np.nanmean(entry["points"], axis=0)
            for j in range(nll_mean.size):
                writer.writerow([
                    method, j, int(points_mean[j]), n_seeds,
                    f"{nll_mean[j]:.6f}", f"{nll_spread[j]:.6f}",
                    f"{cap_mean[j]:.6f}", f"{cap_spread[j]:.6f}",
                ])


def main():
    args = parse_args()
    data = {}
    for method in args.methods:
        entry = read_method(args.runs_dir, method)
        if entry is None:
            raise SystemExit(
                f"no block decomposition for {method} under {args.runs_dir}. "
                "Run the benchmark first; every structured method "
                "writes block_decomposition.npz beside its metrics."
            )
        if args.require_seeds and len(entry["seeds"]) < args.require_seeds:
            raise SystemExit(
                f"{method}: {len(entry['seeds'])} seeds, "
                f"{args.require_seeds} required"
            )
        data[method] = entry
        cap = np.nanmean(entry["captured"], axis=0)
        pts = np.nanmean(entry["points"], axis=0)
        half = int(np.searchsorted(cap, 0.5))
        print(f"  {method:10s} {entry['nll'].shape[0]} seeds, "
              f"{entry['n_blocks']} blocks, "
              f"half the correction at {int(pts[min(half, pts.size - 1)])} points")

    written = draw(data, args.out_base, args.formats, args.error_mode)
    summary = args.out_base.with_name(args.out_base.name + "_summary.csv")
    write_summary(data, summary, args.error_mode)
    for path in written:
        print(f"wrote {path}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
