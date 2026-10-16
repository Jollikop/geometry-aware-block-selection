#!/usr/bin/env python3
"""Plot SVGP metrics against the inducing-point budget.

Each column is one dataset/kernel pair, each row is one metric, and every
method is a curve over M.  Error bars summarize repeated seeds.  The script
reads complete ``metrics.json`` files directly, so partial and running jobs are
ignored without relying on the shared CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

from svgp import plans  # noqa: E402
from common import stats as st  # noqa: E402
from svgp.analysis.paper_status import matches_protocol


from common.plot_style import (  # noqa: E402
    MAIN_SVGP_METHODS,
    canonical_method,
    main_plot_methods,
    method_color,
    method_linewidth,
    method_marker,
    method_markersize,
)
from common.provenance import sha256_file  # noqa: E402

DEFAULT_RUNS_DIR = HERE / "runs_v10" / "main_report"

METHOD_LABELS = {
    "SVGP": "SVGP",
    "T-SVGP": "T-SVGP",
    "RBT-SVGP": "RBT-SVGP",
    "SBT-SVGP": "SBT-SVGP",
}

# Publication year appended to each legend entry, taken from the reference
# PDFs under References/. Set an entry to None to drop the year.
#
#   SVGP      Hensman, Fusi & Lawrence, "Gaussian Processes for Big Data",
#             UAI 2013 (uncollapsed stochastic bound with the trace term).
#   T-SVGP    Titsias, "New Bounds for Sparse Variational Gaussian
#             Processes", arXiv:2502.08730, 2025. The concurrent Bui, Ashman
#             & Turner, "Tighter sparse variational Gaussian processes",
#             arXiv:2502.04750, is also 2025, so the label is unaffected by
#             which of the two the diagonal correction is attributed to.
#   RBT-SVGP  random block partition baseline for the block bound.
#   SBT-SVGP  this work, 2026.
METHOD_YEARS = {
    "SVGP": 2013,
    "T-SVGP": 2025,
    "RBT-SVGP": None,
    "SBT-SVGP": 2026,
}


def method_label(method: str) -> str:
    """Legend entry: method name plus the year of the paper it comes from."""
    name = METHOD_LABELS.get(method, method)
    year = METHOD_YEARS.get(method)
    return f"{name} ({year})" if year else name


def dataset_dimensions(rows) -> dict[str, int]:
    """Input dimensionality per dataset, read from the stored metrics."""
    dims: dict[str, int] = {}
    for row in rows:
        dataset = str(row.get("dataset", ""))
        value = row.get("d", (row.get("config") or {}).get("d"))
        try:
            dims[dataset] = int(value)
        except (TypeError, ValueError):
            continue
    return dims

METRIC_LABELS = {
    "rmse": "Test RMSE",
    "nll": "Test NLL",
    "nll_full": "Eq. (12) NLL",
    "nll_anchor_full_cross": "Anchor full-cross NLL",
    "nll_full_cross": "Dense full-cross NLL",
    "last_epoch_stochastic_objective_estimate_per_datum": (
        "Last-epoch stochastic\nobjective estimate / N"
    ),
    "neg_last_epoch_stochastic_objective_estimate_per_datum": (
        "Negative last-epoch\nstochastic objective / N"
    ),
    "search_s": "Search time (s)",
    "optimizer_s": "Optimizer time (s)",
    "pred_ms": "Predict time (ms)",
}

# The stochastic objective is a lower bound: larger is better, the opposite of
# RMSE and NLL. Plotting its negative makes every panel of the figure read the
# same way -- down is better -- which is what the "lower is better" arrow in
# the first column states. Sign handling is confined to `oriented` and
# `display_metric`, so the stored metrics keep their original convention.
NEGATED_METRICS = {"last_epoch_stochastic_objective_estimate_per_datum"}
LOWER_IS_BETTER_ARROW = "lower is better"
# Placement of the direction marker, in axes fractions of the first column.
# The oblique arrow runs down-right in the lower-left corner, which the
# descending curves leave empty; move these if a future panel fills it.
ARROW_TAIL = (0.07, 0.30)
ARROW_TIP = (0.22, 0.11)
ARROW_COLOR = "0.30"
ARROW_FONTSIZE = 11.0
# Block banner in the stacked figure: one sub-figure per kernel.
BLOCK_TITLE_FONTSIZE = 17.0
BLOCK_GAP = 0.045


def display_metric(metric: str, negate: bool = True) -> str:
    """Name under which a metric is labelled and written to the summary CSV."""
    if negate and metric in NEGATED_METRICS:
        return f"neg_{metric}"
    return metric


def oriented(values, metric: str, negate: bool = True):
    """Flip the sign of bound-like metrics so that lower is always better."""
    if negate and metric in NEGATED_METRICS:
        return -np.asarray(values)
    return values

KERNEL_LABELS = {
    "rbf": "RBF",
    "matern32": "Matern-3/2",
    "matern52": "Matern-5/2",
    "matern-periodic": "Matern + periodic",
    "periodic-angular": "periodic-angular",
    "gibbs": "Gibbs",
    "amplitude-rbf": "Amplitude-RBF",
}

MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*")
TITLE_FONTSIZE = 14.5
AXIS_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 13


def finite_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def metric_value(row, metric):
    if metric == "nll_anchor_full_cross":
        return row.get(metric, row.get("nll_full"))
    if metric == "rmse_anchor_full_cross":
        return row.get(metric, row.get("rmse_full"))
    return row.get(metric)


def parse_pair(value: str) -> tuple[str, str]:
    """Parse a reviewer-facing ``kernel:dataset`` pair."""
    try:
        kernel, dataset = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{value!r} must use the form kernel:dataset"
        ) from exc
    if not kernel or not dataset:
        raise argparse.ArgumentTypeError(
            f"{value!r} must use the form kernel:dataset"
        )
    return kernel, dataset


def read_metrics(
    roots: Path | list[Path], run_tag: str | None = None,
    plan: str = "report",
) -> list[dict]:
    rows = []
    rejected = 0
    roots = [roots] if isinstance(roots, Path) else list(roots)
    paths = sorted(
        path for root in roots for path in root.glob("**/metrics.json")
    )
    for path in paths:
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if run_tag and not matches_protocol(row, run_tag, plan):
            rejected += 1
            continue
        if all(key in row for key in ("kernel", "dataset", "M", "method", "seed")):
            row["method"] = canonical_method(row["method"])
            rows.append(row)
    if run_tag:
        print(
            f"official protocol filter: kept={len(rows)} rejected={rejected} "
            f"run_tag={run_tag}"
        )
    return rows


def available_pairs(rows, methods, seeds, kernels, datasets, ms,
                    preferred_order=None):
    keep_methods = set(methods) if methods else None
    keep_seeds = {int(seed) for seed in seeds} if seeds else None
    keep_kernels = set(kernels) if kernels else None
    keep_datasets = set(datasets) if datasets else None
    keep_ms = {int(m) for m in ms} if ms else None
    pairs = set()
    for row in rows:
        try:
            method = str(row["method"])
            seed = int(row["seed"])
            kernel = str(row["kernel"])
            dataset = str(row["dataset"])
            m = int(row["M"])
        except (KeyError, TypeError, ValueError):
            continue
        if keep_methods and method not in keep_methods:
            continue
        if keep_seeds and seed not in keep_seeds:
            continue
        if keep_kernels and kernel not in keep_kernels:
            continue
        if keep_datasets and dataset not in keep_datasets:
            continue
        if keep_ms and m not in keep_ms:
            continue
        pairs.add((kernel, dataset))
    ordered = []
    if preferred_order:
        ordered = [pair for pair in preferred_order if pair in pairs]
        return ordered
    return sorted(pairs, key=lambda pair: (pair[1], pair[0]))


def values_for(rows, pair, m, method, metric, seeds):
    kernel, dataset = pair
    keep_seeds = {int(seed) for seed in seeds} if seeds else None
    # One value per seed prevents a stale duplicate CSV-like row from changing
    # the uncertainty estimate. Complete metrics in the run tree are canonical.
    by_seed = {}
    for row in rows:
        try:
            row_seed = int(row.get("seed"))
            row_m = int(row.get("M"))
        except (TypeError, ValueError):
            continue
        if str(row.get("kernel")) != kernel or str(row.get("dataset")) != dataset:
            continue
        if row_m != int(m) or str(row.get("method")) != method:
            continue
        if keep_seeds and row_seed not in keep_seeds:
            continue
        value = finite_float(metric_value(row, metric))
        if math.isfinite(value):
            by_seed[row_seed] = value
    values = np.asarray([by_seed[key] for key in sorted(by_seed)], dtype=float)
    return group_seeds(values)


# Optional seed grouping, set once from --seed-group-size. Applied here rather
# than at each call site so that every figure, every error bar and the summary
# CSV all see the same sample: the only thing that changes between the two
# averaging strategies is what this function returns.
SEED_GROUP_SIZE = 0


def group_seeds(values):
    """Average equal-sized seed groups, per ``--seed-group-size``.

    Applied here rather than at each call site so every figure, every error bar
    and the summary CSV see the same sample.
    """
    return st.group_seeds(values, SEED_GROUP_SIZE)


def summary_row(pair, m, method, metric, values):
    kernel, dataset = pair
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    mean = float(np.mean(values))
    ci95 = st.uncertainty(values, "ci95")
    return {
        "kernel": kernel,
        "dataset": dataset,
        "M": int(m),
        "method": method,
        "metric": metric,
        "n_seeds": int(values.size),
        "mean": mean,
        "std": std,
        "var": float(np.var(values, ddof=1)) if values.size > 1 else 0.0,
        "sem": std / math.sqrt(values.size),
        "ci95": ci95,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
    }


def save_summary(path: Path, rows):
    fields = [
        "kernel", "dataset", "M", "method", "metric",
        "n_seeds", "mean", "std", "var", "sem",
        "ci95", "ci95_low", "ci95_high",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_chunk(rows, pairs, methods, metrics, ms, seeds, error_mode, plan,
               out_base: Path, formats, require_complete_seeds=False,
               negate_objective=True):
    nrows, ncols = len(metrics), len(pairs)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(4.2, 4.0 * ncols), max(3.0, 2.45 * nrows + 0.9)),
        squeeze=False,
        sharex=False,
    )
    summary = []
    dims = dataset_dimensions(rows)

    for column, pair in enumerate(pairs):
        kernel, dataset = pair
        pair_ms = plans.m_values_for_pair(plan, kernel, dataset, ms)
        x = np.asarray(pair_ms, dtype=float)
        for row_index, metric in enumerate(metrics):
            ax = axes[row_index][column]
            for method_index, method in enumerate(methods):
                means = np.full(len(pair_ms), np.nan)
                errors = np.full(len(pair_ms), np.nan)
                for m_index, m in enumerate(pair_ms):
                    values = values_for(rows, pair, m, method, metric, seeds)
                    if not values.size:
                        continue
                    if require_complete_seeds and values.size != len(seeds):
                        kernel, dataset = pair
                        raise SystemExit(
                            "incomplete paper group: "
                            f"{kernel}:{dataset} M={m} method={method} "
                            f"metric={metric} has {values.size}/{len(seeds)} seeds"
                        )
                    values = oriented(values, metric, negate_objective)
                    means[m_index] = float(np.mean(values))
                    errors[m_index] = st.uncertainty(values, error_mode)
                    summary.append(summary_row(
                        pair, m, method,
                        display_metric(metric, negate_objective), values,
                    ))
                valid = np.isfinite(means)
                if not np.any(valid):
                    continue
                color = method_color(method)
                error_linewidth = 1.05 if method.startswith("SBT-") else 0.8
                ax.errorbar(
                    x[valid],
                    means[valid],
                    yerr=errors[valid],
                    color=color,
                    ecolor=color,
                    marker=method_marker(
                        method, MARKERS[method_index % len(MARKERS)]
                    ),
                    markersize=method_markersize(method),
                    linewidth=method_linewidth(method),
                    elinewidth=error_linewidth,
                    capthick=error_linewidth,
                    capsize=2.0,
                    alpha=1.0 if method.startswith("SBT-") else 0.82,
                    label=method_label(method),
                    zorder=3,
                )

            if row_index == 0:
                kernel_label = KERNEL_LABELS.get(kernel, kernel)
                dim = dims.get(dataset)
                head = f"{dataset} (d={dim})" if dim else dataset
                ax.set_title(
                    f"{head}\n{kernel_label}",
                    fontsize=TITLE_FONTSIZE,
                    pad=7,
                )
            if column == 0:
                shown = display_metric(metric, negate_objective)
                ax.set_ylabel(
                    METRIC_LABELS.get(shown, shown),
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
                # One marker per metric, first panel only: with the objective
                # negated every row is minimised, so a single statement of
                # direction covers the whole figure. It sits in the lower-left
                # corner, which the descending curves leave free.
                ax.annotate(
                    "",
                    xy=ARROW_TIP, xytext=ARROW_TAIL,
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-|>,head_width=0.28,head_length=0.55",
                        color=ARROW_COLOR, linewidth=1.6,
                    ),
                    zorder=5,
                )
                ax.text(
                    ARROW_TIP[0] + 0.045, ARROW_TIP[1] + 0.035,
                    LOWER_IS_BETTER_ARROW,
                    transform=ax.transAxes,
                    fontsize=ARROW_FONTSIZE, color=ARROW_COLOR,
                    ha="left", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor="none", alpha=0.75),
                )
            if row_index == nrows - 1:
                ax.set_xlabel(
                    "Number of inducing points, M",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(x)
            ax.set_xlim(x[0] / 1.04, x[-1] * 1.04)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.grid(True, axis="both", alpha=0.22, linewidth=0.7)
            ax.tick_params(axis="both", labelsize=8.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

    handles = [
        Line2D(
            [0], [0],
            color=method_color(method),
            marker=method_marker(method, MARKERS[index % len(MARKERS)]),
            linewidth=method_linewidth(method),
            markersize=method_markersize(method),
            label=method_label(method),
        )
        for index, method in enumerate(methods)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        markerscale=1.25,
        handlelength=2.4,
        handletextpad=0.7,
        columnspacing=1.8,
        borderpad=0.7,
        bbox_to_anchor=(0.5, 0.07),
    )
    fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        target = out_base.with_suffix(f".{extension}")
        fig.savefig(target, dpi=190, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {target}")
    plt.close(fig)
    return summary


def plot_stacked(rows, datasets, kernels, methods, metrics, ms, seeds,
                 error_mode, plan, out_base: Path, formats,
                 negate_objective=True):
    """One figure with a block of metric rows per kernel, datasets as columns.

    Used for the main-paper figure: RBF on top, Matern-3/2 underneath, so the
    two kernels are compared on the same datasets and the same axes without
    flipping between two separate images.
    """
    nrows, ncols = len(kernels) * len(metrics), len(datasets)
    fig = plt.figure(
        figsize=(max(4.2, 4.0 * ncols), max(3.0, 2.35 * nrows + 1.4)),
        constrained_layout=True,
    )
    # One sub-figure per kernel: each carries its own banner and its own
    # column headers, so the two blocks read as two separate panels of the
    # same figure instead of six undifferentiated rows.
    subfigs = fig.subfigures(len(kernels), 1, hspace=BLOCK_GAP)
    if len(kernels) == 1:
        subfigs = [subfigs]
    block_axes = []
    for block, kernel in enumerate(kernels):
        subfig = subfigs[block]
        subfig.suptitle(
            KERNEL_LABELS.get(kernel, kernel),
            fontsize=BLOCK_TITLE_FONTSIZE, fontweight="bold",
        )
        block_axes.append(
            subfig.subplots(len(metrics), ncols, squeeze=False)
        )
    summary = []
    dims = dataset_dimensions(rows)

    for block, kernel in enumerate(kernels):
        for metric_index, metric in enumerate(metrics):
            for column, dataset in enumerate(datasets):
                ax = block_axes[block][metric_index][column]
                pair = (kernel, dataset)
                pair_ms = plans.m_values_for_pair(plan, kernel, dataset, ms)
                x = np.asarray(pair_ms, dtype=float)
                for method_index, method in enumerate(methods):
                    means = np.full(len(pair_ms), np.nan)
                    errors = np.full(len(pair_ms), np.nan)
                    for m_index, m in enumerate(pair_ms):
                        values = values_for(
                            rows, pair, m, method, metric, seeds
                        )
                        if not values.size:
                            continue
                        values = oriented(values, metric, negate_objective)
                        means[m_index] = float(np.mean(values))
                        errors[m_index] = st.uncertainty(values, error_mode)
                        summary.append(summary_row(
                            pair, m, method,
                            display_metric(metric, negate_objective), values,
                        ))
                    valid = np.isfinite(means)
                    if not np.any(valid):
                        continue
                    color = method_color(method)
                    width = 1.05 if method.startswith("SBT-") else 0.8
                    ax.errorbar(
                        x[valid], means[valid], yerr=errors[valid],
                        color=color, ecolor=color,
                        marker=method_marker(
                            method, MARKERS[method_index % len(MARKERS)]
                        ),
                        markersize=method_markersize(method),
                        linewidth=method_linewidth(method),
                        elinewidth=width, capthick=width, capsize=2.0,
                        alpha=1.0 if method.startswith("SBT-") else 0.82,
                        label=method_label(method), zorder=3,
                    )

                if metric_index == 0:
                    dim = dims.get(dataset)
                    head = f"{dataset} (d={dim})" if dim else dataset
                    ax.set_title(head, fontsize=TITLE_FONTSIZE, pad=7)
                if column == 0:
                    shown = display_metric(metric, negate_objective)
                    label = METRIC_LABELS.get(shown, shown)
                    ax.set_ylabel(label, fontsize=AXIS_LABEL_FONTSIZE)
                    # The direction marker states a fact about the whole
                    # figure, so it is drawn once per metric in the top block
                    # only; repeating it under Matern would be noise.
                    if block == 0:
                        ax.annotate(
                            "", xy=ARROW_TIP, xytext=ARROW_TAIL,
                            xycoords="axes fraction",
                            textcoords="axes fraction",
                            arrowprops=dict(
                                arrowstyle=(
                                    "-|>,head_width=0.28,head_length=0.55"
                                ),
                                color=ARROW_COLOR, linewidth=1.6,
                            ),
                            zorder=5,
                        )
                        ax.text(
                            ARROW_TIP[0] + 0.045, ARROW_TIP[1] + 0.035,
                            LOWER_IS_BETTER_ARROW, transform=ax.transAxes,
                            fontsize=ARROW_FONTSIZE, color=ARROW_COLOR,
                            ha="left", va="center", zorder=5,
                            bbox=dict(boxstyle="round,pad=0.22",
                                      facecolor="white", edgecolor="none",
                                      alpha=0.75),
                        )
                if metric_index == len(metrics) - 1:
                    ax.set_xlabel(
                        "Number of inducing points, M",
                        fontsize=AXIS_LABEL_FONTSIZE,
                    )
                ax.set_xscale("log", base=2)
                ax.set_xticks(x)
                ax.set_xlim(x[0] / 1.04, x[-1] * 1.04)
                ax.xaxis.set_major_formatter(ScalarFormatter())
                ax.grid(True, axis="both", alpha=0.22, linewidth=0.7)
                ax.tick_params(axis="both", labelsize=8.5)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.8)

    handles = [
        Line2D(
            [0], [0], color=method_color(method),
            marker=method_marker(method, MARKERS[index % len(MARKERS)]),
            linewidth=method_linewidth(method),
            markersize=method_markersize(method),
            label=method_label(method),
        )
        for index, method in enumerate(methods)
    ]
    fig.legend(
        handles=handles, ncol=min(4, len(handles)),
        frameon=True, fontsize=LEGEND_FONTSIZE, markerscale=1.25,
        handlelength=2.4, handletextpad=0.7, columnspacing=1.8,
        borderpad=0.7, loc="outside lower center",
    )
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        target = out_base.with_suffix(f".{extension}")
        fig.savefig(target, dpi=190, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {target}")
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument(
        "--additional-runs-dir", action="append", default=[],
        help="additional metrics tree; repeat to combine separately frozen "
             "kernel campaigns without copying their results",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="when set, include only metrics matching this official run tag",
    )
    parser.add_argument("--out-dir", default=str(HERE / "figures_v10" / "main_report" / "inducing_curves"))
    parser.add_argument("--plan", default="report")
    parser.add_argument("--kernels", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--m", nargs="*", type=int, default=[64, 128, 256, 512])
    parser.add_argument(
        "--methods", nargs="*", default=list(MAIN_SVGP_METHODS)
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[
            "rmse",
            "nll",
            "last_epoch_stochastic_objective_estimate_per_datum",
        ],
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument(
        "--seed-group-size", type=int, default=0,
        help="average the seeds in consecutive groups of this size and treat "
             "the group means as the sample. 0 disables. The curves are "
             "unchanged -- with equal groups the mean of means is the mean -- "
             "so this only alters the error bars, and it alters them for the "
             "worse: fewer degrees of freedom for the same information.")
    parser.add_argument(
        "--error",
        choices=("std", "var", "sem", "ci95"),
        default="sem",
        help="error-bar half-width; sem is one standard error of the "
             "across-seed mean, while ci95 is a two-sided Student-t interval",
    )
    parser.add_argument(
        "--require-complete-seeds",
        action="store_true",
        help="fail instead of drawing any method/M group missing a requested seed",
    )
    parser.add_argument(
        "--raw-objective",
        action="store_true",
        help="plot the stochastic objective with its native sign (larger is "
             "better) instead of its negative; the default negates it so that "
             "every panel is lower-is-better and the arrow applies throughout",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        type=parse_pair,
        default=None,
        help="exact ordered kernel:dataset pairs; useful for selecting the "
             "four main-paper panels after the complete run",
    )
    parser.add_argument("--columns-per-figure", type=int, default=4)
    parser.add_argument(
        "--stack-kernels",
        nargs="*",
        default=None,
        help="produce ONE figure with a block of metric rows per kernel, in "
             "the given order (e.g. --stack-kernels rbf matern32), with the "
             "datasets of --datasets as columns. Overrides the per-pair "
             "layout and --columns-per-figure.",
    )
    parser.add_argument("--figure-prefix", default="inducing_curves")
    parser.add_argument("--formats", nargs="*", choices=("png", "pdf"), default=["png", "pdf"])
    args = parser.parse_args()
    global SEED_GROUP_SIZE
    SEED_GROUP_SIZE = int(args.seed_group_size)

    if args.columns_per_figure < 1:
        raise SystemExit("--columns-per-figure must be >= 1")
    ms = sorted(set(args.m))
    if not ms or any(m <= 0 for m in ms):
        raise SystemExit("--m requires positive inducing-point counts")
    roots = [Path(args.runs_dir)] + [
        Path(value) for value in args.additional_runs_dir
    ]
    rows = read_metrics(roots, args.run_tag, args.plan)
    args.methods = main_plot_methods(args.methods)

    if args.stack_kernels:
        if not args.datasets:
            raise SystemExit("--stack-kernels requires --datasets")
        out_dir = Path(args.out_dir)
        summary = plot_stacked(
            rows, list(args.datasets), list(args.stack_kernels),
            args.methods, args.metrics, ms, args.seeds, args.error,
            args.plan, out_dir / args.figure_prefix, args.formats,
            negate_objective=not args.raw_objective,
        )
        if not summary:
            raise SystemExit("no metrics matched the requested kernels/datasets")
        save_summary(out_dir / f"{args.figure_prefix}_summary.csv", summary)
        return
    preferred_order = None
    if args.plan != "custom":
        preferred_order = plans.build_plan(args.plan)
    filter_ms = ms
    if args.plan == "report":
        filter_ms = sorted({
            m
            for kernel, dataset in preferred_order
            for m in plans.m_values_for_pair(args.plan, kernel, dataset)
        })
    pairs = available_pairs(
        rows, args.methods, args.seeds, args.kernels, args.datasets, filter_ms,
        preferred_order=preferred_order,
    )
    if args.pairs:
        available = set(pairs)
        missing = [pair for pair in args.pairs if pair not in available]
        if missing:
            formatted = ", ".join(f"{k}:{d}" for k, d in missing)
            raise SystemExit(f"missing complete metrics for pairs: {formatted}")
        pairs = list(dict.fromkeys(args.pairs))
    if not pairs:
        raise SystemExit("no complete matching metrics.json files found")

    out_dir = Path(args.out_dir)
    all_summary = []
    for start in range(0, len(pairs), args.columns_per_figure):
        chunk = pairs[start:start + args.columns_per_figure]
        index = start // args.columns_per_figure + 1
        if len(pairs) <= args.columns_per_figure:
            filename = args.figure_prefix
        else:
            filename = f"{args.figure_prefix}_{index:02d}"
        out_base = out_dir / filename
        all_summary.extend(plot_chunk(
            rows,
            chunk,
            args.methods,
            args.metrics,
            ms,
            args.seeds,
            args.error,
            args.plan,
            out_base,
            args.formats,
            require_complete_seeds=args.require_complete_seeds,
            negate_objective=not args.raw_objective,
        ))
    summary_path = out_dir / f"{args.figure_prefix}_summary.csv"
    save_summary(summary_path, all_summary)
    selected = {
        (kernel, dataset, int(m), method, int(seed))
        for kernel, dataset in pairs
        for m in plans.m_values_for_pair(args.plan, kernel, dataset, ms)
        for method in args.methods
        for seed in args.seeds
    }
    relevant = [
        row for row in rows
        if (
            str(row.get("kernel")),
            str(row.get("dataset")),
            int(row.get("M")),
            str(row.get("method")),
            int(row.get("seed")),
        ) in selected
    ]
    if len(relevant) != len(selected):
        raise SystemExit(
            f"manifest expected {len(selected)} unique results, found {len(relevant)}"
        )
    commits = {
        str(row["config"]["code_commit"]) for row in relevant
    }
    if len(commits) != 1:
        raise SystemExit(
            f"a release figure requires one code commit, found {sorted(commits)}"
        )
    artifacts = sorted(out_dir.glob(f"{args.figure_prefix}*.pdf"))
    manifest = {
        "implementation": "svgp-residual-gain-double-batch-jax-v8",
        "run_tag": args.run_tag,
        "pairs": [f"{kernel}:{dataset}" for kernel, dataset in pairs],
        "methods": list(args.methods),
        "seeds": [int(seed) for seed in args.seeds],
        "error_bar": "sample SEM",
        "sample_standard_deviation_ddof": 1,
        "result_count": len(relevant),
        "config_ids": sorted({str(row["config_id"]) for row in relevant}),
        "data_signatures": sorted({
            str(row["config"]["data_signature"]) for row in relevant
        }),
        "code_commit": next(iter(commits)),
        "artifacts": {
            path.name: sha256_file(path) for path in artifacts
        },
    }
    manifest_path = out_dir / f"{args.figure_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")
    print(f"made {math.ceil(len(pairs) / args.columns_per_figure)} figures for {len(pairs)} dataset/kernel pairs")


if __name__ == "__main__":
    main()
