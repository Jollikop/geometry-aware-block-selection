"""Aggregate complete, provenance-checked transplant predictions without writes to training."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from make_svgp_preview import (
    BUDGETS, DATASETS, KERNELS, METHODS, digest, finite, read_campaign,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root, predictions, out = (v.expanduser().resolve() for v in (a.root, a.predictions, a.out))
    for source in (root, predictions):
        if source == out or source in out.parents or out in source.parents:
            raise SystemExit("Output must be separate from input trees")
    rows, issues, sources = read_campaign(root, [predictions])
    selected = []
    for k in KERNELS:
        for d in DATASETS:
            for m in BUDGETS:
                for method in METHODS:
                    for seed in range(10):
                        key = ("transplant", k, d, m, method, seed)
                        row = rows.get(key, {})
                        path = Path(row.get("prediction_source", "none"))
                        if predictions not in path.parents or not all(finite(row.get(v)) for v in ("rmse", "nll")):
                            raise SystemExit(f"Missing or unverified transplant prediction: {key}")
                        selected.append(row)
    out.mkdir(parents=True, exist_ok=False)
    aggregates = []
    for k in KERNELS:
        for d in DATASETS:
            for method in METHODS:
                for m in BUDGETS:
                    for metric in ("rmse", "nll"):
                        values = [rows[("transplant", k, d, m, method, seed)][metric] for seed in range(10)]
                        aggregates.append(dict(kernel=k, dataset=d, method=method, M=m,
                                               metric=metric, n=10, mean=float(np.mean(values)),
                                               sem=float(np.std(values, ddof=1)/np.sqrt(10))))
    with (out / "aggregate.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(aggregates[0]))
        w.writeheader()
        w.writerows(aggregates)
    lookup = {(r["kernel"], r["dataset"], r["method"], r["M"], r["metric"]): r for r in aggregates}
    with PdfPages(out / "transplant_all.pdf") as combined:
        for k in KERNELS:
            for group, datasets in enumerate((DATASETS[:4], DATASETS[4:]), 1):
                fig, axes = plt.subplots(2, 4, figsize=(15, 6), squeeze=False)
                for col, d in enumerate(datasets):
                    for r, metric in enumerate(("rmse", "nll")):
                        ax = axes[r, col]
                        for method in METHODS:
                            stats = [lookup[k, d, method, m, metric] for m in BUDGETS]
                            ax.errorbar(BUDGETS, [v["mean"] for v in stats],
                                        yerr=[v["sem"] for v in stats], marker="o", capsize=2,
                                        label=method + " donor")
                        ax.set_xticks(BUDGETS)
                        ax.set_xlabel("Inducing points M")
                        ax.set_ylabel("Test " + metric.upper())
                        ax.set_title(d)
                        ax.grid(alpha=0.2)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
                fig.suptitle(f"{k}: after transplant, SVGP predictor | mean +/- SEM, 10 seeds")
                fig.tight_layout(rect=(0, 0.065, 1, 0.94))
                fig.savefig(out / f"transplant_{k}_{group}.pdf")
                combined.savefig(fig)
                plt.close(fig)
    manifest = dict(scope="transplant predictive accuracy only", verified_predictions=len(selected),
                    root=str(root), predictions=str(predictions), sources=sources,
                    selected_predictions=[r["prediction_source"] for r in selected],
                    reader_issues=issues, aggregator_sha256=digest(Path(__file__)),
                    note="Not full campaign release certification; concurrent timing excluded")
    (out / "AGGREGATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"Verified {len(selected)}/5120 transplant predictions. PDFs: {out}")


if __name__ == "__main__":
    main()
