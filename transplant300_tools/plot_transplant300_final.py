#!/usr/bin/env python3
"""Certify and plot the complete 300-epoch SVGP transplant campaign."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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

SIM = Path(__file__).resolve().parents[2] / "Simulations"
sys.path.insert(0, str(SIM))
from analysis.make_svgp_complete_figures import config_fingerprint, method_color, method_marker
from analysis.paper_labels import paper_label

METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")
KERNELS = ("rbf", "matern52", "amplitude-rbf", "gibbs")
KERNEL_LABELS = {"rbf": "RBF", "matern52": "Matern-5/2",
                 "amplitude-rbf": "Amplitude-RBF", "gibbs": "Gibbs"}
DATASETS = tuple("pol superconductivity kin40k protein elevators traffic electricity 3droad".split())
BUDGETS = (64, 128, 256, 512)
METRICS = ("rmse", "nll")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def expected_grid():
    return set(itertools.product(KERNELS, DATASETS, BUDGETS, METHODS, range(10)))


def relative_run(path: Path, root: Path) -> Path:
    rel = path.relative_to(root)
    if rel.parts[-1] != "prediction.json":
        raise ValueError(path)
    return rel.parent


def validate_inputs(identity, export_files, old_training, training_remote):
    for raw, claimed in identity["inputs"].items():
        raw = str(raw)
        if training_remote + "/" in raw:
            suffix = raw.split(training_remote + "/", 1)[1]
            key = "training300/" + suffix
            if export_files.get(key) != claimed:
                raise ValueError(f"300-epoch training input is not bound to export manifest: {key}")
        elif "/aistats2026_five_v18_518db753c8a1_20260828_012117/" in raw:
            suffix = raw.split("/aistats2026_five_v18_518db753c8a1_20260828_012117/", 1)[1]
            local = old_training / suffix
            if not local.is_file() or digest(local) != claimed:
                raise ValueError(f"150-epoch lineage input mismatch: {suffix}")


def load_and_certify(bundle: Path, old_training: Path):
    export = bundle / "export"
    old_predictions = bundle / "predictions300"
    recovery = bundle / "predictions300_recovery"
    manifest_path = export / "EXPORT_MANIFEST.json"
    verification_path = export / "LOCAL_VERIFICATION.json"
    archive = export / "transplant300_final.tar.gz"
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads(verification_path.read_text())
    if not verification.get("integrity_verified") or verification.get("final_params") != 5120:
        raise ValueError("the 300-epoch training archive was not integrity-verified")
    if digest(archive).removeprefix("sha256:") != verification["archive_sha256"]:
        raise ValueError("training archive checksum mismatch")
    export_files = manifest["files"]
    training_remote = manifest["training_root"].rstrip("/")

    plan_path = recovery / "recovery_plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["training"].rstrip("/") != training_remote:
        raise ValueError("recovery plan targets a different training freeze")
    states = []
    for worker in range(5):
        state_path = recovery / "state" / f"worker{worker}.json"
        state = json.loads(state_path.read_text())
        if not (state.get("finished") and state.get("complete") and not state.get("failed")):
            raise ValueError(f"incomplete recovery worker: {worker}")
        states.append(state)
    if sum(int(s["completed"]) for s in states) != 1000:
        raise ValueError("recovery workers do not account for exactly 1000 predictions")

    old_paths = sorted(old_predictions.glob("svgp/transplant/runs/*/*/M*/*/seed*/prediction.json"))
    new_paths = sorted(recovery.glob("svgp/transplant/runs/*/*/M*/*/seed*/prediction.json"))
    if (len(old_paths), len(new_paths)) != (4120, 1000):
        raise ValueError(f"expected 4120 + 1000 predictions, found {len(old_paths)} + {len(new_paths)}")
    old_runs = {str(relative_run(p, old_predictions)): p for p in old_paths}
    new_runs = {str(relative_run(p, recovery)): p for p in new_paths}
    if set(old_runs) != set(plan["saved"]):
        raise ValueError("old prediction inventory differs from the recovery plan")
    if set(new_runs) != set(plan["missing"]):
        raise ValueError("recovery prediction inventory differs from the recovery plan")
    if set(old_runs) & set(new_runs):
        raise ValueError("old and recovery predictions overlap")

    lookup, source_hashes = {}, {}
    for source_name, root, paths in (("original", old_predictions, old_paths),
                                     ("recovery", recovery, new_paths)):
        for path in paths:
            run = relative_run(path, root)
            parts = run.parts
            kernel, dataset = parts[3], parts[4]
            budget = int(parts[5][1:])
            transplant_method = parts[6]
            seed = int(parts[7][4:])
            if not transplant_method.endswith("-to-SVGP"):
                raise ValueError(f"unexpected method path: {run}")
            donor = transplant_method[:-8]
            key = (kernel, dataset, budget, donor, seed)
            payload = json.loads(path.read_text())
            cfg, identity = payload["training_config"], payload["identity"]
            if key != (cfg["kernel"], cfg["dataset"], int(cfg["M"]), cfg["donor_method"], int(cfg["seed"])):
                raise ValueError(f"path/config identity mismatch: {run}")
            if cfg.get("epochs") != 300 or cfg.get("training_method") != "SVGP":
                raise ValueError(f"not a 300-epoch sparse transplant: {run}")
            if identity.get("predictor") != "sparse" or identity.get("config_id") != config_fingerprint(cfg):
                raise ValueError(f"prediction/config fingerprint mismatch: {run}")
            if identity.get("gpu_kinds") != ["NVIDIA A100-PCIE-40GB"]:
                raise ValueError(f"non-uniform prediction hardware: {run}")
            if not np.isfinite([payload.get("rmse"), payload.get("nll")]).all():
                raise ValueError(f"non-finite metric: {run}")
            actual = digest(path)
            if source_name == "original":
                manifest_key = "predictions300/" + str(path.relative_to(old_predictions))
                if export_files.get(manifest_key) != actual:
                    raise ValueError(f"original prediction checksum mismatch: {run}")
                if plan["saved"][str(run)] != actual.removeprefix("sha256:"):
                    raise ValueError(f"recovery-plan old checksum mismatch: {run}")
            validate_inputs(identity, export_files, old_training, training_remote)
            if key in lookup:
                raise ValueError(f"duplicate prediction: {key}")
            lookup[key] = payload
            source_hashes[str(path)] = actual
    if set(lookup) != expected_grid():
        missing = sorted(expected_grid() - set(lookup))
        extra = sorted(set(lookup) - expected_grid())
        raise ValueError(f"prediction grid mismatch; missing={missing[:3]} extra={extra[:3]}")
    certification_inputs = {
        str(manifest_path): digest(manifest_path),
        str(verification_path): digest(verification_path),
        str(archive): "sha256:" + verification["archive_sha256"],
        str(plan_path): digest(plan_path),
    }
    return lookup, source_hashes, certification_inputs


def aggregate(lookup):
    rows = []
    for kernel, dataset, budget, method, metric in itertools.product(
            KERNELS, DATASETS, BUDGETS, METHODS, METRICS):
        values = [float(lookup[kernel, dataset, budget, method, seed][metric]) for seed in range(10)]
        rows.append({"kernel": kernel, "dataset": dataset, "M": budget,
                     "donor": method, "metric": metric, "n": 10,
                     "mean": float(np.mean(values)),
                     "sem": float(np.std(values, ddof=1) / np.sqrt(10))})
    return rows


def draw(table, kernel, datasets):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2))
    for col, dataset in enumerate(datasets):
        for row, metric in enumerate(METRICS):
            ax = axes[row, col]
            for method in METHODS:
                points = [table[kernel, dataset, budget, method, metric] for budget in BUDGETS]
                ax.errorbar(BUDGETS, [p["mean"] for p in points],
                            yerr=[p["sem"] for p in points],
                            color=method_color(method), marker=method_marker(method),
                            linewidth=1.4, markersize=4.8, elinewidth=.7, capsize=2)
            ax.set_xlim(40, 536)
            ax.set_xticks(BUDGETS, [str(v) for v in BUDGETS])
            ax.set_xlabel("Inducing points M")
            ax.set_ylabel("Test " + metric.upper())
            ax.set_title(dataset)
            ax.grid(alpha=.18)
            if col == 0:
                ax.annotate("", xy=(.19, .10), xytext=(.06, .27), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=".3", linewidth=1.2))
                ax.text(.22, .12, "lower is better", transform=ax.transAxes, fontsize=8.5,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
    handles = [Line2D([], [], color=method_color(m), marker=method_marker(m),
                      linewidth=1.4, label=paper_label(m) + " donor") for m in METHODS]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035), ncol=4, frameon=False)
    fig.suptitle(KERNEL_LABELS[kernel] + " | Sparse SVGP after 300-epoch transplant", fontsize=15)
    fig.text(.5, .014, "Mean +/- SEM across the same 10 seeds. Standardized targets; lower is better.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.06, right=.99, top=.90, bottom=.18, hspace=.40, wspace=.34)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--old-training", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    bundle, old_training, out = args.bundle.resolve(), args.old_training.resolve(), args.out.resolve()
    for source in (bundle, old_training):
        if not source.is_dir() or source == out or source in out.parents or out in source.parents:
            raise SystemExit("inputs must exist and output must be disjoint")
    if out.exists():
        raise SystemExit("output already exists")
    lookup, source_hashes, certification_inputs = load_and_certify(bundle, old_training)
    records = aggregate(lookup)
    table = {(r["kernel"], r["dataset"], r["M"], r["donor"], r["metric"]): r for r in records}
    out.mkdir(parents=True)
    with (out / "svgp_transplant300_aggregate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for section, datasets in (("main", DATASETS[:4]), ("appendix", DATASETS[4:])):
        with PdfPages(out / f"svgp_transplant300_{section}.pdf") as pdf:
            for kernel in KERNELS:
                fig = draw(table, kernel, datasets)
                pdf.savefig(fig)
                fig.savefig(out / f"svgp_transplant300_{section}_{kernel}.png", dpi=160)
                plt.close(fig)
    outputs = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    certificate = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_certified": True,
        "training_params": 5120,
        "predictions": 5120,
        "old_predictions": 4120,
        "recovered_predictions": 1000,
        "complete_grid": True,
        "seeds_per_cell": 10,
        "kernels": KERNELS,
        "datasets": DATASETS,
        "inducing_points": BUDGETS,
        "inducing_axis": "linear (budget-proportional spacing)",
        "predictor": "standard sparse SVGP",
        "prediction_gpu": "NVIDIA A100-PCIE-40GB",
        "certification_inputs": certification_inputs,
        "prediction_source_hashes": source_hashes,
        "script_sha256": digest(Path(__file__)),
        "outputs": outputs,
    }
    (out / "SVGP_TRANSPLANT300_CERTIFICATE.json").write_text(json.dumps(certificate, indent=2))
    print(json.dumps({"release_certified": True, "predictions": len(lookup), "output": str(out)}))


if __name__ == "__main__":
    main()
