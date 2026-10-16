#!/usr/bin/env python3
"""Generate the official tables and figures from a validated final suite."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-final-suite")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.plot_style import (
    method_color, method_linewidth, method_marker, method_markersize,
)
from common.stats import uncertainty


HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent
KERNELS = ("rbf", "gibbs", "matern52", "amplitude-rbf")
KERNEL_LABEL = {
    "rbf": "RBF", "matern52": "Matern-5/2",
    "amplitude-rbf": "Amplitude-RBF", "gibbs": "Gibbs/Paciorek",
}
DATASETS = (
    "pol", "superconductivity", "kin40k", "protein",
    "elevators", "traffic", "electricity", "3droad",
)
MAIN_DATASETS = DATASETS[:4]
APPENDIX_DATASETS = DATASETS[4:]
METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")


def run(*parts: object) -> None:
    command = [str(part) for part in parts]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def make_inducing(root: Path, out: Path) -> None:
    plotter = SIM_ROOT / "svgp" / "plot_inducing_curves.py"
    stationary = root / "svgp" / "stationary" / "runs"
    extended = root / "svgp" / "extended" / "runs"
    for kernel in KERNELS:
        for section, datasets in (
            ("main", MAIN_DATASETS), ("appendix", APPENDIX_DATASETS),
        ):
            run(
                sys.executable, plotter,
                "--runs-dir", stationary,
                "--additional-runs-dir", extended,
                "--out-dir", out,
                "--plan", "custom",
                "--kernels", kernel,
                "--datasets", *datasets,
                "--pairs", *(f"{kernel}:{dataset}" for dataset in datasets),
                "--m", "64", "128", "256", "512",
                "--methods", *METHODS,
                "--metrics", "rmse", "nll",
                "last_epoch_stochastic_objective_estimate_per_datum",
                "--seeds", *map(str, range(10)),
                "--require-complete-seeds",
                "--columns-per-figure", "4",
                "--figure-prefix", f"svgp_{kernel}_{section}",
                "--formats", "pdf", "png",
            )


def make_sgpr_table(root: Path, out: Path) -> None:
    run(
        sys.executable, SIM_ROOT / "sgpr" / "make_main_table.py",
        "--runs-dir", root / "sgpr" / "runs",
        "--kernels", *KERNELS,
        "--m", "256",
        "--csv-out", out / "sgpr_kin40k_main.csv",
        "--tex-out", out / "sgpr_kin40k_main.tex",
    )


def metric_from_history(row: dict, method: str, metric: str) -> float:
    if method != "SVGP":
        bound = row.get(f"{metric}_reported", row.get(f"{metric}_full", ""))
        try:
            value = float(bound)
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            pass
    return float(row[metric])


def make_convergence(root: Path, out: Path) -> None:
    runs = root / "svgp" / "convergence" / "runs"
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.4), sharex="col")
    summary = []
    for column, kernel in enumerate(KERNELS):
        for method in METHODS:
            history_path = (
                runs / kernel / "kin40k" / "M256" / method / "seed0"
                / "history.csv"
            )
            with history_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            epochs = np.asarray([int(row["epoch"]) for row in rows])
            for ridx, metric in enumerate(("nll", "rmse")):
                values = np.asarray([
                    metric_from_history(row, method, metric) for row in rows
                ])
                axes[ridx, column].plot(
                    epochs, values, color=method_color(method),
                    linewidth=method_linewidth(method),
                    marker=method_marker(method),
                    markersize=method_markersize(method),
                    markevery=max(1, len(epochs) // 8), label=method,
                )
                for epoch, value in zip(epochs, values):
                    summary.append({
                        "kernel": kernel, "method": method,
                        "metric": metric, "epoch": int(epoch),
                        "value": float(value),
                    })
        axes[0, column].set_title(KERNEL_LABEL[kernel])
        axes[1, column].set_xlabel("Epoch")
        for row in range(2):
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Test NLL")
    axes[1, 0].set_ylabel("Test RMSE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "svgp_kin40k_convergence.pdf", bbox_inches="tight")
    fig.savefig(out / "svgp_kin40k_convergence.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)
    with (out / "svgp_kin40k_convergence.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0])
        writer.writeheader()
        writer.writerows(summary)


def source_root(root: Path, kernel: str) -> Path:
    section = "stationary" if kernel in {"rbf", "matern52"} else "extended"
    return root / "svgp" / section / "runs"


def make_transplant(root: Path, out: Path) -> None:
    target = root / "svgp" / "transplant" / "runs"
    rows = []
    for kernel in KERNELS:
        source = source_root(root, kernel)
        for dataset in DATASETS:
            for m in (64, 128, 256, 512):
                for method in METHODS:
                    target_method = f"{method}-to-SVGP"
                    for seed in range(10):
                        before = load(
                            source / kernel / dataset / f"M{m}" / method
                            / f"seed{seed}" / "metrics.json"
                        )
                        after = load(
                            target / kernel / dataset / f"M{m}" / target_method
                            / f"seed{seed}" / "metrics.json"
                        )
                        rows.append({
                            "kernel": kernel, "dataset": dataset, "M": m,
                            "donor_method": method, "seed": seed,
                            "before_predictor": before["reported_predictor"],
                            "before_nll": float(before["nll"]),
                            "after_nll": float(after["nll"]),
                            "delta_nll": float(after["nll"] - before["nll"]),
                            "before_rmse": float(before["rmse"]),
                            "after_rmse": float(after["rmse"]),
                            "delta_rmse": float(after["rmse"] - before["rmse"]),
                        })
    fields = list(rows[0])
    with (out / "svgp_sparse_transplant_all.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.0), sharex=True)
    x = np.arange(len(DATASETS))
    for col, kernel in enumerate(KERNELS):
        for method in METHODS:
            subset = [row for row in rows if row["kernel"] == kernel
                      and row["donor_method"] == method]
            for ridx, metric in enumerate(("delta_nll", "delta_rmse")):
                means, errors = [], []
                for dataset in DATASETS:
                    values = np.asarray([
                        row[metric] for row in subset
                        if row["dataset"] == dataset
                    ])
                    means.append(float(np.mean(values)))
                    errors.append(float(uncertainty(values, "sem")))
                axes[ridx, col].errorbar(
                    x, means, yerr=errors, color=method_color(method),
                    linewidth=method_linewidth(method),
                    marker=method_marker(method),
                    markersize=method_markersize(method), capsize=2,
                    label=method,
                )
        axes[0, col].set_title(KERNEL_LABEL[kernel])
        axes[1, col].set_xticks(x, DATASETS, rotation=38, ha="right")
        for ridx in range(2):
            axes[ridx, col].axhline(0, color="0.5", linewidth=0.8)
            axes[ridx, col].grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("NLL after - before")
    axes[1, 0].set_ylabel("RMSE after - before")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out / "svgp_sparse_transplant_delta.pdf", bbox_inches="tight")
    fig.savefig(out / "svgp_sparse_transplant_delta.png", dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    out = (args.out or root / "figures").expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    make_sgpr_table(root, out)
    make_inducing(root, out)
    make_convergence(root, out)
    make_transplant(root, out)
    print(f"official figures -> {out}")


if __name__ == "__main__":
    main()
