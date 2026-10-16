"""Read-only accuracy aggregation: full donor vs saved sparse SVGP transplant.

Requires the entire 10-seed grid. Does not certify optimization, timing, or the
whole paper freeze. All products are written to a new, disjoint directory.
"""
import argparse
import csv
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.plot_style import method_color, method_marker
from common.provenance import config_fingerprint
from analysis.paper_labels import paper_label
from analysis.make_svgp_preview import (
    BUDGETS, DATASETS, KERNELS, METHODS, digest, finite, paired_deltas, read_campaign,
)

KINDS = ("complete", "transplant", "combined", "paired_delta")
LINE_WIDTH = 1.4
TITLES = {"complete": "Donor with its full predictor",
          "transplant": "After transplant: saved SVGP posterior",
          "combined": "Full predictor vs sparse SVGP transplant",
          "paired_delta": "Paired difference: transplant minus full predictor"}


def validate_rows(rows, root, prediction_roots):
    """Require both 5120 grids and verify donor lineage, not just matching names."""
    selected, checkpoint_hashes = [], {}
    paired = paired_deltas(rows)
    for k, d, m, method, seed in itertools.product(KERNELS, DATASETS, BUDGETS, METHODS, range(10)):
        donor_stage = "stationary" if k in ("rbf", "matern52") else "extended"
        donor_key = (donor_stage, k, d, m, method, seed)
        after_key = ("transplant", k, d, m, method, seed)
        for key in (donor_key, after_key):
            row = rows.get(key, {})
            if not all(finite(row.get(v)) for v in ("rmse", "nll")):
                raise ValueError(f"Missing/invalid accuracy: {key}")
            selected.append(row)
        before, after = rows[donor_key], rows[after_key]
        if after_key not in paired:
            raise ValueError(f"Different or missing split/preprocessing provenance: {after_key}")
        source = Path(after["prediction_source"]).resolve()
        if not any(p in source.parents for p in prediction_roots):
            raise ValueError(f"Transplant must have verified prediction sidecar: {after_key}")
        cfg = after["config"]
        if cfg.get("donor_method") != method:
            raise ValueError(f"Wrong transplant donor: {after_key}")
        path = (root / before["run"] / "params.npz").resolve()
        if root not in path.parents:
            raise ValueError("Checkpoint outside training root")
        if path not in checkpoint_hashes:
            checkpoint_hashes[path] = digest(path)
        if cfg.get("donor_params_sha256") != checkpoint_hashes[path]:
            raise ValueError(f"Donor checkpoint changed: {after_key}")
    return selected, paired, checkpoint_hashes


def aggregate(rows, paired):
    result = []
    for kind in ("complete", "transplant", "paired_delta"):
        for k, d, method, m, metric in itertools.product(KERNELS, DATASETS, METHODS, BUDGETS, ("rmse", "nll")):
            stage = ("stationary" if k in ("rbf", "matern52") else "extended") if kind == "complete" else "transplant"
            source = paired if kind == "paired_delta" else rows
            values = [float(source[(stage, k, d, m, method, s)][metric]) for s in range(10)]
            result.append(dict(kind=kind, kernel=k, dataset=d, method=method, M=m,
                               metric=metric, n=10, mean=float(np.mean(values)),
                               sem=float(np.std(values, ddof=1) / np.sqrt(10))))
    return result


def draw_page(lookup, kind, kernel, datasets, *, transplant_only_solid=False, direction_arrows=False):
    fig, axes = plt.subplots(2, len(datasets), figsize=(16, 6.8), squeeze=False)
    for col, dataset in enumerate(datasets):
        for row, metric in enumerate(("rmse", "nll")):
            ax = axes[row, col]
            for method in METHODS:
                for variant in (("complete", "transplant") if kind == "combined" else (kind,)):
                    points = [lookup[(variant, kernel, dataset, method, m, metric)] for m in BUDGETS]
                    color = method_color(method)
                    dashed = variant == "transplant" and not (kind == "transplant" and transplant_only_solid)
                    ax.errorbar(BUDGETS, [r["mean"] for r in points],
                                yerr=[r["sem"] for r in points], color=color,
                                linestyle="--" if dashed else "-", marker=method_marker(method),
                                markerfacecolor="white" if dashed else color,
                                markersize=5, linewidth=LINE_WIDTH,
                                elinewidth=0.75, capsize=2, alpha=0.95)
            if kind == "paired_delta":
                ax.axhline(0, color="0.4", linewidth=0.8, zorder=0)
            ax.set_xscale("log", base=2)
            ax.set_xlim(56, 585)
            ax.set_xticks(BUDGETS, labels=[str(m) for m in BUDGETS])
            ax.minorticks_off()
            ax.set_xlabel("Inducing points M")
            ax.set_ylabel(("Delta " if kind == "paired_delta" else "Test ") + metric.upper())
            ax.set_title(dataset)
            ax.grid(alpha=0.18)
            if direction_arrows and col == 0 and kind != "paired_delta":
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
    handles = [Line2D([], [], color=method_color(m), marker=method_marker(m), linewidth=LINE_WIDTH,
                      label=paper_label(m) + (" donor" if kind == "transplant" else "")) for m in METHODS]
    if kind == "combined":
        handles += [Line2D([], [], color="0.25", linestyle="-", label="Full predictor"),
                    Line2D([], [], color="0.25", linestyle="--", marker="o",
                           markerfacecolor="white", label="SVGP transplant")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.04),
               ncol=4 if kind == "combined" else len(handles), frameon=False)
    note = "Mean +/- SEM across the same 10 seeds; lower is better."
    if kind == "paired_delta":
        note = "Mean +/- SEM of per-seed differences; negative favors transplant."
    if kind == "transplant":
        note += " Colors identify the donor method."
    fig.text(0.5, 0.016, note, ha="center", fontsize=9)
    fig.suptitle(f"{kernel} | {TITLES[kind]}", fontsize=15)
    fig.tight_layout(rect=(0, 0.115, 1, 0.945))
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--prediction-roots", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--paper-four", action="store_true",
                   help="only four PDFs: transplant and full-vs-transplant, each main/appendix; RMSE/NLL only")
    args = p.parse_args()
    root, out = args.root.expanduser().resolve(), args.out.expanduser().resolve()
    prediction_roots = [x.expanduser().resolve() for x in args.prediction_roots]
    for path in [root, *prediction_roots]:
        if not path.is_dir():
            raise SystemExit(f"Input directory missing: {path}")
        if out == path or path in out.parents or out in path.parents:
            raise SystemExit("Output must be separate from all input trees")
    if out.exists():
        raise SystemExit("Output already exists; choose a new directory")
    print("Reading metrics and checking prediction/checkpoint hashes...", flush=True)
    rows, issues, sources = read_campaign(root, prediction_roots)
    # Missing old metrics may legitimately be superseded by validated sidecars.
    # Never silently ignore a corrupt/stale supplied sidecar or training record.
    fatal = [r for r in issues if r["reason"] != "missing metrics or wrong predictor"]
    if fatal:
        raise SystemExit(json.dumps(fatal[:20], indent=2))
    selected, paired, checkpoint_hashes = validate_rows(rows, root, prediction_roots)
    source_hashes = {Path(s["path"]): s["sha256"] for s in sources}
    source_hashes.update(checkpoint_hashes)
    for row in selected:
        if config_fingerprint(row["config"]) != row["config_id"]:
            raise SystemExit(f"Configuration fingerprint mismatch: {row['run']}")
        path = Path(row["prediction_source"])
        if path.name == "prediction.json":
            for relative, expected in json.loads(path.read_text())["identity"]["inputs"].items():
                input_path = (root / relative).resolve()
                if root not in input_path.parents:
                    raise SystemExit(f"Prediction input outside training root: {relative}")
                if input_path in source_hashes and source_hashes[input_path] != expected:
                    raise SystemExit(f"Conflicting input identities: {input_path}")
                source_hashes[input_path] = expected
    aggregates = aggregate(rows, paired)
    lookup = {(r["kind"], r["kernel"], r["dataset"], r["method"], r["M"], r["metric"]): r for r in aggregates}
    out.mkdir(parents=True)
    with (out / "aggregate_accuracy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    with (out / "paired_seed_accuracy.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel", "dataset", "M", "method", "seed", "delta_rmse", "delta_nll"])
        for key, row in sorted(paired.items()):
            writer.writerow([*key[1:], row["rmse"], row["nll"]])
    if args.paper_four:
        for kind, name in (("transplant", "transplant"), ("combined", "all")):
            for section, datasets in (("main", DATASETS[:4]), ("appendix", DATASETS[4:])):
                with PdfPages(out / f"{name}_{section}.pdf") as pdf:
                    for kernel in ("rbf", "matern52", "amplitude-rbf", "gibbs"):
                        fig = draw_page(lookup, kind, kernel, datasets,
                                        transplant_only_solid=True, direction_arrows=True)
                        fig.savefig(out / f"{name}_{section}_{kernel}.png", dpi=150)
                        pdf.savefig(fig)
                        plt.close(fig)
    else:
        for kind in KINDS:
            with PdfPages(out / f"{kind}_all.pdf") as pdf:
                for kernel in KERNELS:
                    for group, datasets in enumerate((DATASETS[:4], DATASETS[4:]), 1):
                        fig = draw_page(lookup, kind, kernel, datasets)
                        name = f"{kind}_{kernel}_{group}"
                        fig.savefig(out / f"{name}.pdf")
                        fig.savefig(out / f"{name}.png", dpi=150)
                        pdf.savefig(fig)
                        plt.close(fig)
    # This manifest is published only after a final input-stability check.
    for path, expected in source_hashes.items():
        if digest(path) != expected:
            raise SystemExit(f"Inputs changed during aggregation; output NOT verified: {path}")
    outputs = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    scope="SVGP predictive accuracy: stationary, extended, transplant",
                    validated_accuracy_rows=len(selected), paired_seeds=len(paired),
                    full_campaign_release_certified=False, timing_included=False,
                    layout="paper-four" if args.paper_four else "all-families",
                    metrics_plotted=["rmse", "nll"],
                    comparison="donor full predictor vs trained sparse SVGP transplant, not sparse before/after refinement",
                    root=str(root), prediction_roots=list(map(str, prediction_roots)),
                    source_hashes={str(p): h for p, h in source_hashes.items()},
                    script_hashes={p.name: digest(p) for p in (
                        Path(__file__), Path(__file__).with_name("make_svgp_preview.py"),
                        Path(__file__).with_name("paper_labels.py"),
                        Path(__file__).resolve().parents[1] / "common/plot_style.py")},
                    reused_legacy_prediction_rows=sum(Path(r["prediction_source"]).name == "metrics.json" for r in selected),
                    outputs=outputs, reader_notes=issues)
    (out / "ACCURACY_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Accuracy verified: {len(selected)}/10240; paired: {len(paired)}/5120. Output: {out}")


if __name__ == "__main__":
    main()
