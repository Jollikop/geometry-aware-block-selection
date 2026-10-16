"""Compare certified block-diagonal and full-residual predictors on paired fits.

This is a prediction-only diagnostic. It reads the saved Pol and Kin40k
evaluations and never modifies training checkpoints or frozen results.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
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

HARD_DEFAULT = PROJECT / "output/results/hard_pic_predict_full_v1_20261003"
DENSE_DEFAULT = PROJECT / "output/results/real_predictor_main_20261002_v2"
OUT_DEFAULT = PROJECT / "Report/figures/hard_pic_dense_diagnostic"
SGPR_METHODS = ("SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR", "METIS-SGPR")
SVGP_METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")
MS = (64, 128, 256, 512)
KERNELS = ("rbf", "amplitude-rbf")


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def mean_se(values: list[float]) -> tuple[float, float]:
    if len(values) != 10 or not all(math.isfinite(x) for x in values):
        raise ValueError(f"expected ten finite paired seeds, got {len(values)}")
    return mean(values), stdev(values) / math.sqrt(10)


def load_old_validator():
    path = PROJECT / "Report/notes/build_real_predictor_nll.py"
    spec = importlib.util.spec_from_file_location("dense_prediction_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pairs(hard_root: Path, dense_root: Path):
    old = load_old_validator()
    dense_pairs, exact, discarded = old.load_pairs(dense_root)
    plan_path = hard_root / "plan.json"
    certificate_path = hard_root / "HARD_PIC_PREDICTION_CERTIFICATE.json"
    plan = json.loads(plan_path.read_text())
    certificate = json.loads(certificate_path.read_text())
    if (certificate["count"] != 3880
            or certificate["plan_sha256"] != digest(plan_path)
            or len(certificate["predictions"]) != 3880):
        raise ValueError("hard prediction certificate or plan is incomplete")
    if plan["hard_evaluator_sha256"] is None:
        raise ValueError("hard evaluator hash missing")

    paired = defaultdict(dict)
    max_rmse_change = 0.0
    for family, kernels, methods, ms, dataset in (
        ("sgpr", ("rbf",), SGPR_METHODS[1:], (512,), "kin40k"),
        ("svgp", KERNELS, SVGP_METHODS[1:], MS, "pol"),
    ):
        for kernel in kernels:
            for m in ms:
                for method in methods:
                    for seed in range(10):
                        relative = (f"{family}/{kernel}/{dataset}/M{m}/"
                                    f"{method}/seed{seed}")
                        hard_path = hard_root / "hard" / relative / "prediction.json"
                        dense_path = dense_root / relative / "prediction.json"
                        if digest(hard_path) != certificate["predictions"][relative]:
                            raise ValueError(f"hard result differs from certificate: {relative}")
                        hard = json.loads(hard_path.read_text())
                        dense = json.loads(dense_path.read_text())
                        hi, di = hard["identity"], dense["identity"]
                        if (hard["valid"] is not True
                                or hi["implementation"] != certificate["implementation"]
                                or hi["plan_sha256"] != certificate["plan_sha256"]
                                or hi["evaluator_sha256"] != plan["hard_evaluator_sha256"]
                                or di["implementation"] != "real-dff-variational-predictor-v1"
                                or hi["config_id"] != di["config_id"]
                                or hi["inputs"] != di["inputs"]
                                or hi["source_commit"] != di["source_commit"]):
                            raise ValueError(f"non-paired or invalid predictions: {relative}")
                        rmse_change = abs(float(hard["rmse"]) - float(dense["rmse"]))
                        max_rmse_change = max(max_rmse_change, rmse_change)
                        if rmse_change > 1e-5:
                            raise ValueError(f"predictive means differ: {relative}")
                        pair = (float(hard["nll"]), float(dense["nll"]))
                        if not all(map(math.isfinite, pair)):
                            raise ValueError(f"nonfinite NLL: {relative}")
                        key = (family, kernel, m, method)
                        paired[key][seed] = pair
                        if not math.isclose(pair[1], dense_pairs[key][seed][1],
                                            abs_tol=1e-12):
                            raise ValueError(f"dense validator disagrees: {relative}")
    if len(paired) != 28 or sum(map(len, paired.values())) != 280:
        raise ValueError("incomplete hard/dense paired grid")
    if any(set(rows) != set(range(10)) for rows in paired.values()):
        raise ValueError("incomplete ten-seed comparison")
    return paired, dense_pairs, exact, {
        "hard_certificate_sha256": digest(certificate_path),
        "hard_plan_sha256": digest(plan_path),
        "dense_plan_sha256": digest(dense_root / "plan.json"),
        "paired_predictions": 280,
        "max_rmse_change": max_rmse_change,
        "dense_max_discarded_cross_energy_fraction": discarded,
    }


def compact(value: float, se: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value):.3f}({round(se * 1000)})".replace("0.", ".", 1)


def make_table(path: Path, paired: dict, dense_pairs: dict, exact: dict) -> None:
    labels = {"SGPR": "SGPR", "T-SGPR": "T-SGPR",
              "RBT-SGPR": "(R)BT-SGPR", "SBT-SGPR": "(S)BT-SGPR",
              "METIS-SGPR": "(METIS)BT-SGPR$^{\\dagger}$"}
    lines = [
        "% Paired prediction-only comparison on frozen RBF-Kin40k M=512 states.",
        r"\begin{table}[!ht]",
        r"\caption{RBF--Kin40k, $M=512$: test NLL (mean (SE) over ten paired "
        r"seeds; lower is better). Block diag. is the predictor in the main "
        r"results; full uses the dense $\mathbf D_{\mathbf{ff}}$ in "
        r"Eq.~\eqref{eq:bui-predictive-variance}. Bold marks the best of "
        r"the four principal methods; $^\dagger$ denotes comparators.}",
        r"\label{tab:kin40k-sgpr-real-nll}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrr@{}}",
        r"\toprule",
        "Method & Block diag. NLL & Full-$D_{ff}$ NLL " + r"\\",
        r"\midrule",
    ]
    for method in SGPR_METHODS:
        if method == "METIS-SGPR":
            lines.append(r"\midrule")
        rows = (dense_pairs if method == "SGPR" else paired)[
            ("sgpr", "rbf", 512, method)]
        first = mean_se([rows[s][0 if method != "SGPR" else 1]
                         for s in range(10)])
        second = mean_se([rows[s][1] for s in range(10)])
        cells = [compact(*first), compact(*second)]
        if method == "SBT-SGPR":
            cells = [r"\textbf{" + value + "}" for value in cells]
        lines.append(f"{labels[method]} & {cells[0]} & {cells[1]} " + r"\\")
    exact_cell = compact(*mean_se([exact[s] for s in range(10)]))
    lines.extend([r"\midrule",
                  "Exact GP$^{\\dagger}$ & --- & " + exact_cell + r" \\",
                  r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def paper_axis(ax) -> None:
    ax.grid(alpha=.18, lw=.4)
    ax.tick_params(length=2, pad=1)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    for spine in ax.spines.values():
        spine.set_linewidth(.5)


def make_plot(path: Path, paired: dict, dense_pairs: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42,
                         "ps.fonttype": 42})
    fig = plt.figure(figsize=(6.75, 3.75))
    grid = fig.add_gridspec(1, 2, left=.095, right=.99, top=.79,
                            bottom=.235, wspace=.23)
    for ax, kernel, title in ((fig.add_subplot(grid[0, 0]), "rbf", "RBF - Pol"),
                              (fig.add_subplot(grid[0, 1]), "amplitude-rbf",
                               "Amplitude-RBF - Pol")):
        for method in SVGP_METHODS:
            color = method_color(method)
            marker = method_marker(method)
            variants = (1,) if method == "SVGP" else (0, 1)
            for index in variants:
                source = dense_pairs if method == "SVGP" else paired
                points = [mean_se([source[("svgp", kernel, m, method)][s][index]
                                   for s in range(10)]) for m in MS]
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
               label="Block diag. predictor"),
        Line2D([], [], color=".25", linestyle="-", marker="o", lw=1,
               markersize=3, label="Full residual $D_{ff}$"),
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
    parser.add_argument("--hard-root", type=Path, default=HARD_DEFAULT)
    parser.add_argument("--dense-root", type=Path, default=DENSE_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    paired, dense_pairs, exact, audit = load_pairs(args.hard_root, args.dense_root)
    gaps = {}
    for family in ("sgpr", "svgp"):
        vals = [abs(mean_se([rows[s][0] for s in range(10)])[0]
                    - mean_se([rows[s][1] for s in range(10)])[0])
                for key, rows in paired.items() if key[0] == family]
        gaps[family] = max(vals)
    args.out.mkdir(parents=True, exist_ok=True)
    table = args.out / "kin40k_sgpr_hard_vs_dense_nll.tex"
    plot = args.out / "pol_hard_vs_dense_nll.pdf"
    make_table(table, paired, dense_pairs, exact)
    make_plot(plot, paired, dense_pairs)
    audit["maximum_absolute_mean_nll_gap"] = gaps
    audit["outputs"] = {p.name: digest(p) for p in (table, plot)}
    (args.out / "HARD_PIC_DENSE_DIAGNOSTIC.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"PAIRED={audit['paired_predictions']}/280 "
          f"MAX_SGPR_GAP={gaps['sgpr']:.6f} "
          f"MAX_SVGP_GAP={gaps['svgp']:.6f} "
          f"MAX_RMSE_CHANGE={audit['max_rmse_change']:.3e}")
    print(f"TABLE={table}\nPLOT={plot}")


if __name__ == "__main__":
    main()
