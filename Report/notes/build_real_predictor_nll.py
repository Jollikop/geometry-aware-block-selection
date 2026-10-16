"""Paper-style NLL comparison from paired frozen prediction records.

This reuses the official Matplotlib method palette, markers and figure sizing.
It reads only prediction sidecars and matching frozen metrics, and writes a
standalone plot plus an includable AISTATS table; it never changes the freeze.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "Simulations"))
from common.plot_style import method_color, method_marker  # noqa: E402
from analysis.paper_labels import paper_label  # noqa: E402

DEFAULT_RESULTS = PROJECT / "output/results/real_predictor_main_20261002_v2"
DEFAULT_OUT = PROJECT / "output/pdf/real_predictor_main_20261002_v2"
SGPR_METHODS = ("SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR", "METIS-SGPR")
SVGP_METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")
MS = (64, 128, 256, 512)


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values: list[float]) -> tuple[float, float]:
    if len(values) != 10:
        raise ValueError(f"expected ten seeds, found {len(values)}")
    return mean(values), stdev(values) / 10**0.5


def load_pairs(root: Path) -> tuple[dict, dict, float]:
    plan = json.loads((root / "plan.json").read_text())
    if len(plan["runs"]) != 370:
        raise ValueError("unexpected real-predictor plan size")
    paired = defaultdict(dict)
    max_discarded = 0.0
    for item in plan["runs"]:
        predicted = json.loads((root / item["output_relative"] / "prediction.json").read_text())
        if predicted["identity"]["evaluator_sha256"] != plan["evaluator_sha256"]:
            raise ValueError(f"evaluator changed: {item['output_relative']}")
        if item["family"] == "sgpr":
            frozen_path = (root / "original/sgpr"
                           / Path(item["relative"]).relative_to("runs")
                           / "metrics.json")
        else:
            frozen_path = (root / "original/svgp" / item["kernel"] / "pol"
                           / f"M{item['M']}" / item["method"]
                           / f"seed{item['seed']}" / "metrics.json")
        frozen = json.loads(frozen_path.read_text())
        if digest(frozen_path) not in predicted["identity"]["inputs"].values():
            raise ValueError(f"frozen input differs: {frozen_path}")
        if frozen["config_id"] != predicted["identity"]["config_id"]:
            raise ValueError(f"checkpoint differs: {frozen_path}")
        if abs(frozen["rmse"] - predicted["rmse"]) > 1e-5:
            raise ValueError(f"predictive mean changed: {frozen_path}")
        diagnostics = predicted["diagnostics"]
        if diagnostics["min_latent_variance"] < -1e-7:
            raise ValueError(f"invalid predictive variance: {frozen_path}")
        max_discarded = max(max_discarded,
                            diagnostics["max_discarded_cross_energy_fraction"])
        key = (item["family"], item["kernel"], item["M"], item["method"])
        if item["seed"] in paired[key]:
            raise ValueError(f"duplicate seed: {key}")
        paired[key][item["seed"]] = (float(frozen["nll"]), float(predicted["nll"]))
    if len(paired) != 5 + 2 * 4 * 4 or max_discarded > 1e-7:
        raise ValueError("incomplete comparison or unstable spectral cutoff")
    if any(set(rows) != set(range(10)) for rows in paired.values()):
        raise ValueError("incomplete ten-seed comparison")
    exact = {}
    for seed in range(10):
        path = (root / "original/sgpr/rbf/kin40k/M512/Exact-GP"
                / f"seed{seed}" / "metrics.json")
        exact[seed] = float(json.loads(path.read_text())["nll"])
    return paired, exact, max_discarded


def cell(rows: dict, index: int) -> tuple[float, float]:
    return summary([rows[seed][index] for seed in range(10)])


def compact(value: float, se: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value):.3f}({round(se * 1000)})".replace("0.", ".", 1)


def make_table(path: Path, paired: dict, exact: dict) -> None:
    labels = {
        "SGPR": "SGPR", "T-SGPR": "T-SGPR",
        "RBT-SGPR": "(R)BT-SGPR", "SBT-SGPR": "(S)BT-SGPR",
        "METIS-SGPR": "(METIS)BT-SGPR$^{\\dagger}$",
    }
    lines = [
        "% Paired, read-only recomputation from the frozen RBF-Kin40k M=512 runs.",
        r"\begin{table}[!ht]",
        r"\caption{RBF--Kin40k, $M=512$: test NLL (mean (SE) over ten paired seeds; "
        r"lower is better). Approx. uses the frozen diagonal/block predictor; "
        r"dense uses the full $\mathbf{D}_{\mathbf{ff}}$. Bold marks the best of the four "
        r"principal methods; $^\dagger$ denotes comparators.}",
        r"\label{tab:kin40k-sgpr-real-nll}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        "Method & Approx. NLL & Dense NLL " + r"\\",
        r"\midrule",
    ]
    for method in SGPR_METHODS:
        if method == "METIS-SGPR":
            lines.append(r"\midrule")
        rows = paired[("sgpr", "rbf", 512, method)]
        old, old_se = cell(rows, 0)
        real, real_se = cell(rows, 1)
        formatted = [compact(old, old_se), compact(real, real_se)]
        if method == "SBT-SGPR":
            formatted = [r"\textbf{" + value + "}" for value in formatted]
        lines.append(f"{labels[method]} & {formatted[0]} & {formatted[1]} " + r"\\")
    exact_mean, exact_se = summary([exact[seed] for seed in range(10)])
    lines.extend([
        r"\midrule",
        "Exact GP$^{\\dagger}$ & --- & " + compact(exact_mean, exact_se) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines))


def paper_axis(ax) -> None:
    ax.grid(alpha=.18, lw=.4)
    ax.tick_params(length=2, pad=1)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    for spine in ax.spines.values():
        spine.set_linewidth(.5)


def make_plot(path: Path, paired: dict, max_discarded: float) -> None:
    # Match Report/notes/build_focused_svgp_page.py and the shared paper palette.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig = plt.figure(figsize=(6.75, 3.75))
    grid = fig.add_gridspec(1, 2, left=.095, right=.99, top=.79,
                            bottom=.235, wspace=.23)
    for ax, kernel, title in ((fig.add_subplot(grid[0, 0]), "rbf", "RBF - Pol | SVGP"),
                              (fig.add_subplot(grid[0, 1]), "amplitude-rbf",
                               "Amplitude-RBF - Pol | SVGP")):
        for method in SVGP_METHODS:
            color = method_color(method)
            marker = method_marker(method)
            variants = (1,) if method == "SVGP" else (0, 1)
            for index in variants:
                points = [cell(paired[("svgp", kernel, m, method)], index) for m in MS]
                full = index == 1
                ax.errorbar(MS, [v[0] for v in points], yerr=[v[1] for v in points],
                            color=color, marker=marker,
                            linestyle="-" if full else (0, (4, 2.5)),
                            markerfacecolor=color if full else "white",
                            markeredgewidth=.65 if full else .45,
                            lw=1 if full else .65,
                            markersize=3.3 if method == "SBT-SVGP" else 3,
                            elinewidth=.5 if full else .35, capsize=1.2)
        ax.set_xlim(40, 536)
        ax.set_xticks(MS)
        ax.get_xticklabels()[0].set_horizontalalignment("right")
        ax.set_xlabel("Inducing points $M$ (linear scale)")
        ax.set_ylabel("Test NLL")
        ax.set_title(title, fontsize=9, pad=4)
        paper_axis(ax)
        if kernel == "rbf":
            ax.text(.02, .07, r"$\downarrow$ lower is better",
                    transform=ax.transAxes, fontsize=6.5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.2))

    styles = [
        Line2D([], [], color=".25", linestyle=(0, (4, 2.5)), marker="o",
               markerfacecolor="white", lw=.65, markersize=3,
               label="Diagonal/block approximation"),
        Line2D([], [], color=".25", linestyle="-", marker="o", lw=1,
               markersize=3, label="Real predictor (full $D_{ff}$)"),
    ]
    fig.legend(handles=styles, loc="upper center", bbox_to_anchor=(.5, .98),
               ncol=2, frameon=False, fontsize=7.6, handlelength=2.6,
               columnspacing=1.4)
    handles = [Line2D([], [], color=method_color(method),
                      marker=method_marker(method), lw=1, markersize=3,
                      label=paper_label(method)) for method in SVGP_METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .055),
               ncol=2, frameon=False, fontsize=7.8,
               handlelength=1.6, columnspacing=2)
    fig.text(.5, .018,
             "Mean +/- SEM over ten paired seeds; predictive means and RMSE are unchanged.",
             ha="center", fontsize=7.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    paired, exact, max_discarded = load_pairs(args.results_root)
    args.out.mkdir(parents=True, exist_ok=True)
    table = args.out / "kin40k_sgpr_real_predictor_nll.tex"
    plot = args.out / "pol_real_predictor_nll.pdf"
    make_table(table, paired, exact)
    make_plot(plot, paired, max_discarded)
    print(f"AUDIT=370/370 paired; max discarded cross energy={max_discarded:.3e}")
    print(f"TABLE={table}")
    print(f"PLOT={plot}")


if __name__ == "__main__":
    main()
