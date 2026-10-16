#!/usr/bin/env python3
"""Render certified RBF–Pol M128 ablation curves without refitting models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "Simulations"))
from common.plot_style import method_color, SAME_BATCH_COLOR  # noqa: E402

CURVES = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP", "SBT-SVGP-same-batch")
LABELS = ("SVGP", "T-SVGP", "(R)BT-SVGP", "(S)BT-SVGP",
          "(S)BT-SVGP: same batch")
EPOCHS = tuple(range(20, 341, 20)) + (350,)
METRICS = ("objective", "rmse", "nll")


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load(root: Path) -> dict:
    certificate = json.loads((root / "BLOCK_DIAG_ABLATION_CERTIFICATE.json").read_text())
    aggregate = root / "epochs_rbf_pol_M128_block_diag_summary.csv"
    if (certificate["implementation"] != "rbf-pol-block-diag-ablation-v1"
            or certificate["prediction_count"] != 900
            or digest(aggregate) != certificate["aggregate_sha256"]
            or len(certificate["prediction_files"]) != 900):
        raise ValueError("incomplete or changed ablation certificate")
    for relative, expected in certificate["prediction_files"].items():
        path = root / relative
        if digest(path) != expected:
            raise ValueError(f"certified prediction changed: {path}")
    rows = {}
    with aggregate.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["curve"], row["metric"], int(row["epoch"])
            if (key in rows or key[0] not in CURVES or key[1] not in METRICS
                    or key[2] not in EPOCHS or int(row["n_seeds"]) != 10):
                raise ValueError(f"invalid aggregate cell: {key}")
            value = float(row["mean"]), float(row["sem"])
            if not all(math.isfinite(v) for v in value):
                raise ValueError(f"nonfinite aggregate cell: {key}")
            rows[key] = value
    if len(rows) != len(CURVES) * len(METRICS) * len(EPOCHS):
        raise ValueError("aggregate grid incomplete")
    return rows


def draw(rows: dict, metric: str, target: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    colors = tuple(method_color(curve) for curve in CURVES[:-1]) + (SAME_BATCH_COLOR,)
    for curve, label, color in zip(CURVES, LABELS, colors):
        x = np.asarray(EPOCHS)
        values = np.asarray([rows[curve, metric, epoch] for epoch in EPOCHS])
        y, sem = values[:, 0], values[:, 1]
        if metric == "objective":
            y = -y
        ax.plot(x, y, color=color, lw=1.15,
                ls="--" if curve == CURVES[-1] else "-", label=label)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=.13, lw=0)
    ax.set_xlim(0, 350)
    ax.set_xticks((0, 100, 200, 300))
    ax.set_xlabel("Epoch")
    ax.set_ylabel({"objective": "Negative stochastic\nobjective / N",
                   "rmse": "Test RMSE", "nll": "Test NLL"}[metric])
    ax.grid(alpha=.2, lw=.4)
    ax.tick_params(length=2, pad=2)
    ax.text(.97, .90, r"$\downarrow$ lower is better", ha="right",
            transform=ax.transAxes, fontsize=6.4)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=6.4, frameon=False, handlelength=1.7)
    fig.subplots_adjust(left=.20, right=.98, top=.98, bottom=.40)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="downloaded certified campaign root")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "Report" / "figures")
    args = parser.parse_args()
    rows = load(args.input.resolve())
    for metric in METRICS:
        target = args.output_dir / f"epochs_rbf_pol_M128_{metric}_block_diag.pdf"
        draw(rows, metric, target)
        print(target)


if __name__ == "__main__":
    main()
