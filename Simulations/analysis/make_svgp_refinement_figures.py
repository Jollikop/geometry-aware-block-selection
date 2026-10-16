"""Compare saved donor-as-sparse predictions with trained sparse transplants.

Only common seeds contribute. Missing donor-as-sparse predictions stay missing.
This isolates predictor choice, but changes both Z and q during refinement.
"""
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
from analysis.make_svgp_complete_figures import (
    BUDGETS, DATASETS, KERNELS, METHODS, LINE_WIDTH, digest, finite,
    read_campaign, validate_rows, config_fingerprint, method_color, method_marker,
)


def collect_pairs(rows, root):
    pairs = []
    for key, after in sorted(rows.items()):
        stage, k, d, m, method, seed = key
        if stage != "transplant":
            continue
        donor_stage = "stationary" if k in ("rbf", "matern52") else "extended"
        before = rows[(donor_stage, k, d, m, method, seed)]
        metrics_path = root / before["run"] / "metrics.json"
        direct = None
        if method == "SVGP":
            direct = {v: before[v] for v in ("rmse", "nll")}
        elif metrics_path.is_file():
            record = json.loads(metrics_path.read_text())
            if record.get("config_id") != before["config_id"] or record.get("config") != before["config"]:
                raise ValueError(f"Old sparse metrics do not match donor: {metrics_path}")
            if all(finite(record.get(v + "_sparse")) for v in ("rmse", "nll")):
                direct = {v: float(record[v + "_sparse"]) for v in ("rmse", "nll")}
        if direct is None:
            continue
        for metric in ("rmse", "nll"):
            pairs.append(dict(kernel=k, dataset=d, M=m, method=method, seed=seed,
                              metric=metric, no_refinement=direct[metric],
                              refined=after[metric], delta=after[metric] - direct[metric],
                              refinement_epochs=after["config"].get("epochs"),
                              refinement_phases=after["config"].get("phases")))
    return pairs


def summarize(pairs):
    lookup = {}
    for k, d, method, m, metric in itertools.product(KERNELS, DATASETS, METHODS, BUDGETS, ("rmse", "nll")):
        points = [r for r in pairs if (r["kernel"], r["dataset"], r["method"], r["M"], r["metric"]) == (k, d, method, m, metric)]
        n = len(points)
        row = dict(kernel=k, dataset=d, method=method, M=m, metric=metric, n=n,
                   seeds=",".join(str(r["seed"]) for r in points))
        for kind in ("no_refinement", "refined", "delta"):
            values = [p[kind] for p in points]
            row[kind] = float(np.mean(values)) if n else float("nan")
            row[kind + "_sem"] = float(np.std(values, ddof=1)/np.sqrt(n)) if n > 1 else float("nan")
        lookup[k, d, method, m, metric] = row
    return lookup


def draw(lookup, kernel, datasets, delta=False):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.8))
    for col, dataset in enumerate(datasets):
        for row_idx, metric in enumerate(("rmse", "nll")):
            ax = axes[row_idx, col]
            count_lines, any_data = [], False
            for method in METHODS:
                stats = [lookup[kernel, dataset, method, m, metric] for m in BUDGETS]
                ns = [v["n"] for v in stats]
                count_lines.append(method + ": " + "/".join(map(str, ns)))
                for variant in (("delta",) if delta else ("no_refinement", "refined")):
                    ys = [v[variant] for v in stats]
                    any_data |= any(np.isfinite(ys))
                    refined = variant == "refined"
                    ax.errorbar(BUDGETS, ys, yerr=[v[variant + "_sem"] for v in stats],
                                color=method_color(method), linewidth=LINE_WIDTH,
                                marker=method_marker(method), markersize=4.5,
                                linestyle="--" if refined else "-",
                                markerfacecolor="white" if refined else method_color(method),
                                elinewidth=0.7, capsize=2)
            if delta:
                ax.axhline(0, color="0.3", linewidth=0.8)
            if not any_data:
                ax.text(0.5, 0.5, "No donor-as-sparse predictions", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9)
            ax.set_title(dataset)
            ax.set_ylabel(("Delta " if delta else "Test ") + metric.upper())
            ax.set_xscale("log", base=2)
            ax.set_xlim(56, 585)
            ax.set_xticks(BUDGETS, [str(m) for m in BUDGETS])
            ax.minorticks_off()
            ax.set_xlabel("Inducing points M")
            ax.grid(alpha=0.18)
            if row_idx == 1:
                ax.text(0.5, -0.29, "n at M=64/128/256/512\n" + "\n".join(count_lines),
                        transform=ax.transAxes, ha="center", va="top", fontsize=8)
    handles = [Line2D([], [], color=method_color(m), marker=method_marker(m),
                      linewidth=LINE_WIDTH, label=m + " donor") for m in METHODS]
    if not delta:
        handles += [Line2D([], [], color="0.25", linestyle="-", label="Sparse, no extra training"),
                    Line2D([], [], color="0.25", linestyle="--", label="Sparse, after training")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
               ncol=3 if not delta else 4, frameon=False)
    title = "Does post-donor training help?" if not delta else "Effect of post-donor training: after minus before"
    fig.suptitle(f"{kernel} | {title}\nSame sparse SVGP predictor; available paired seeds only", fontsize=14)
    fig.text(0.5, 0.012, "Mean +/- SEM. " + ("Negative favors extra training. " if delta else "Lower is better. ")
             + "Incomplete coverage; not an official full-grid comparison.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 0.93), h_pad=1.5)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prediction-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    predictions = [p.resolve() for p in args.prediction_roots]
    for p in [root, *predictions]:
        if not p.is_dir() or p == out or p in out.parents or out in p.parents:
            raise SystemExit("Inputs must exist and output must be disjoint")
    if out.exists():
        raise SystemExit("Choose a new output directory")
    rows, issues, sources = read_campaign(root, predictions)
    fatal = [r for r in issues if r["reason"] != "missing metrics or wrong predictor"]
    if fatal:
        raise SystemExit(json.dumps(fatal[:20], indent=2))
    selected, _, hashes = validate_rows(rows, root, predictions)
    hashes.update({Path(s["path"]): s["sha256"] for s in sources})
    for row in selected:
        if config_fingerprint(row["config"]) != row["config_id"]:
            raise SystemExit(f"Config mismatch: {row['run']}")
        path = Path(row["prediction_source"])
        if path.name == "prediction.json":
            hashes.update({(root / p).resolve(): h for p, h in json.loads(path.read_text())["identity"]["inputs"].items()})
    pairs = collect_pairs(rows, root)
    lookup = summarize(pairs)
    out.mkdir(parents=True)
    for name, records in (("paired_refinement", pairs), ("aggregate_refinement", list(lookup.values()))):
        with (out / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    for delta in (False, True):
        name = "refinement_gain" if delta else "no_training_vs_refined"
        with PdfPages(out / (name + "_all.pdf")) as pdf:
            for kernel in KERNELS:
                for group, datasets in enumerate((DATASETS[:4], DATASETS[4:]), 1):
                    fig = draw(lookup, kernel, datasets, delta)
                    fig.savefig(out / f"{name}_{kernel}_{group}.pdf")
                    fig.savefig(out / f"{name}_{kernel}_{group}.png", dpi=150)
                    pdf.savefig(fig)
                    plt.close(fig)
    for path, expected in hashes.items():
        if root not in path.parents and not any(p in path.parents for p in predictions):
            raise SystemExit(f"Unexpected input path: {path}")
        if digest(path) != expected:
            raise SystemExit(f"Input changed; output not validated: {path}")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), paired_runs=len(pairs)//2,
                    target_runs=5120, full_grid=False, interpretation="SVGP sparse prediction before/after refinement",
                    source_hashes={str(p): h for p, h in hashes.items()},
                    script_hash=digest(Path(__file__)), reader_notes=issues)
    (out / "REFINEMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Paired refinement comparison: {len(pairs)//2}/5120. Output: {out}")


if __name__ == "__main__":
    main()
