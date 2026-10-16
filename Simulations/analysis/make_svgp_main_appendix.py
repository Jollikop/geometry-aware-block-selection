"""Paper-style full-predictor SVGP main/appendix; no transplant comparisons."""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.make_svgp_preview import BUDGETS, DATASETS, METHODS, digest, finite, read_campaign
from common.provenance import config_fingerprint
from common.plot_style import method_color, method_marker
from analysis.paper_labels import paper_label, SVGP_LABELS

KERNELS = ("rbf", "matern52", "amplitude-rbf", "gibbs")
LABELS = {"rbf": "RBF", "matern52": "Matern-5/2", "amplitude-rbf": "Amplitude-RBF", "gibbs": "Gibbs"}
METRICS = ("rmse", "nll", "negative_objective")


def aggregate(rows):
    result = []
    for k, d, method, m, metric in itertools.product(KERNELS, DATASETS, METHODS, BUDGETS, METRICS):
        stage = "stationary" if k in ("rbf", "matern52") else "extended"
        values = []
        for seed in range(10):
            key = (stage, k, d, m, method, seed)
            source_metric = "objective" if metric == "negative_objective" else metric
            value = rows.get(key, {}).get(source_metric)
            if not finite(value):
                raise ValueError(f"Missing or invalid {metric}: {key}")
            values.append(-float(value) if metric == "negative_objective" else float(value))
        result.append(dict(kernel=k, dataset=d, method=method, M=m, metric=metric, n=10,
                           mean=float(np.mean(values)), sem=float(np.std(values, ddof=1)/np.sqrt(10))))
    return result


def draw(lookup, kernel, datasets):
    fig, axes = plt.subplots(3, 4, figsize=(16, 9.2))
    for col, dataset in enumerate(datasets):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for method in METHODS:
                points = [lookup[kernel, dataset, method, m, metric] for m in BUDGETS]
                ax.errorbar(BUDGETS, [p["mean"] for p in points],
                            yerr=[p["sem"] for p in points], color=method_color(method),
                            marker=method_marker(method), markersize=4.8,
                            linewidth=1.4, linestyle="-", elinewidth=0.7, capsize=2)
            # Keep the actual spacing between inducing-point budgets visible.
            # On a log2 axis 64, 128, 256 and 512 are artificially equispaced.
            ax.set_xlim(40, 536)
            ax.set_xticks(BUDGETS, [str(m) for m in BUDGETS])
            ax.minorticks_off()
            ax.set_xlabel("Inducing points M")
            ax.set_ylabel({"rmse": "Test RMSE", "nll": "Test NLL",
                           "negative_objective": "Negative last-epoch stochastic\nobjective estimate / N"}[metric])
            ax.set_title(dataset)
            ax.grid(alpha=0.18)
            if col == 0:
                ax.annotate(
                    "", xy=(0.22, 0.11), xytext=(0.07, 0.30),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.55",
                                    color="0.30", linewidth=1.4), zorder=5,
                )
                ax.text(0.265, 0.145, "lower is better", transform=ax.transAxes,
                        fontsize=9, color="0.30", ha="left", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                                  edgecolor="none", alpha=0.8))
    handles = [Line2D([], [], color=method_color(m), marker=method_marker(m),
                      linewidth=1.4, linestyle="-", label=paper_label(m)) for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.028), ncol=4, frameon=False)
    fig.suptitle(LABELS[kernel] + " | SVGP full predictors", fontsize=15)
    fig.text(0.5, 0.01, "Mean +/- SEM across 10 seeds. Lower is better for every row; the stochastic objective is negated.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.96))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--donor-predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, predictions, out = [p.expanduser().resolve() for p in (args.root, args.donor_predictions, args.out)]
    for source in (root, predictions):
        if not source.is_dir() or out == source or source in out.parents or out in source.parents:
            raise SystemExit("Input directories must exist; output must be disjoint")
    if out.exists():
        raise SystemExit("Output exists: choose a new directory")
    print("Verifying all 5120 full-predictor runs...", flush=True)
    rows, issues, sources = read_campaign(root, [predictions])
    rows = {k: r for k, r in rows.items() if k[0] in ("stationary", "extended")}
    relevant = lambda path: "/svgp/stationary/" in str(path) or "/svgp/extended/" in str(path)
    fatal = [r for r in issues if relevant(r["path"]) and r["reason"] != "missing metrics or wrong predictor"]
    if fatal:
        raise SystemExit(json.dumps(fatal[:20], indent=2))
    records = aggregate(rows)
    if len(rows) != 5120:
        raise SystemExit(f"Expected 5120 full-predictor runs, found {len(rows)}")
    hashes = {Path(r["path"]): r["sha256"] for r in sources if relevant(r["path"])}
    hashes[root / "EXPECTED_HEAD"] = digest(root / "EXPECTED_HEAD")
    for row in rows.values():
        if config_fingerprint(row["config"]) != row["config_id"]:
            raise SystemExit(f"Configuration fingerprint mismatch: {row['run']}")
        for name in ("params.npz", "curve.npz"):
            path = root / row["run"] / name
            hashes[path] = digest(path)
        if row["method"] in ("RBT-SVGP", "SBT-SVGP"):
            path = root / row["run"] / "final_blocks.npz"
            hashes[path] = digest(path)
        path = Path(row["prediction_source"])
        if path.name == "prediction.json":
            for relative, expected in json.loads(path.read_text())["identity"]["inputs"].items():
                path_in = (root / relative).resolve()
                if root not in path_in.parents:
                    raise SystemExit("Prediction input outside training root")
                if path_in in hashes and hashes[path_in] != expected:
                    raise SystemExit(f"Conflicting provenance: {path_in}")
                hashes[path_in] = expected
    lookup = {(r["kernel"], r["dataset"], r["method"], r["M"], r["metric"]): r for r in records}
    out.mkdir(parents=True)
    with (out / "svgp_full_aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for section, datasets in (("main", DATASETS[:4]), ("appendix", DATASETS[4:])):
        with PdfPages(out / f"svgp_{section}.pdf") as pdf:
            for kernel in KERNELS:
                fig = draw(lookup, kernel, datasets)
                fig.savefig(out / f"svgp_{section}_{kernel}.pdf")
                fig.savefig(out / f"svgp_{section}_{kernel}.png", dpi=150)
                pdf.savefig(fig)
                plt.close(fig)
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise SystemExit(f"Input changed; output NOT verified: {path}")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), complete_runs=5120,
                    kernels=KERNELS, main_datasets=DATASETS[:4], appendix_datasets=DATASETS[4:],
                    methods=METHODS, display_labels=SVGP_LABELS, seeds=list(range(10)), inducing_points=BUDGETS,
                    predictor="own full predictor for each method", transplant_included=False,
                    timing_included=False, full_paper_freeze_certified=True,
                    inducing_axis="linear (budget-proportional spacing)",
                    objective_display="negative last-epoch stochastic objective estimate / N",
                    objective_transform="multiply saved objective by -1; SEM unchanged",
                    source_hashes={str(p): h for p, h in hashes.items()},
                    script_hashes={p.name: digest(p) for p in (
                        Path(__file__), Path(__file__).with_name("make_svgp_preview.py"),
                        Path(__file__).with_name("paper_labels.py"),
                        Path(__file__).resolve().parents[1] / "common/plot_style.py")},
                    outputs={p.name: digest(p) for p in out.iterdir() if p.is_file()})
    (out / "SVGP_FULL_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Complete: 5120/5120. Main + appendix: {out}")


if __name__ == "__main__":
    main()
