#!/usr/bin/env python3
"""Read-only, partial SVGP campaign figures; never a release/freeze generator.

Reads legacy metrics, training records and separately evaluated prediction
sidecars. Saves a new PNG gallery, per-run snapshot and coverage CSV. No JAX,
checkpoint fitting, prediction, paper editing or release-manifest writes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys

KERNELS = ("rbf", "gibbs", "matern52", "amplitude-rbf")
DATASETS = ("pol", "superconductivity", "kin40k", "protein",
            "elevators", "traffic", "electricity", "3droad")
METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")
BUDGETS = (64, 128, 256, 512)
OBJECTIVE = "last_epoch_stochastic_objective_estimate_per_datum"


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            h.update(data)
    return "sha256:" + h.hexdigest()


def key_from_relative(relative):
    p = relative.parts
    if len(p) != 8 or p[0] != "svgp" or p[2] != "runs":
        raise ValueError("unexpected run path")
    stage, kernel, dataset, m, method, seed = p[1], p[3], p[4], p[5], p[6], p[7]
    if stage not in ("stationary", "extended", "transplant"):
        raise ValueError("unknown stage")
    if stage == "transplant":
        if not method.endswith("-to-SVGP"):
            raise ValueError("not an SVGP transplant")
        method = method.removesuffix("-to-SVGP")
    m, seed = int(m.removeprefix("M")), int(seed.removeprefix("seed"))
    if kernel not in KERNELS or dataset not in DATASETS or method not in METHODS \
            or m not in BUDGETS or seed not in range(10):
        raise ValueError("outside campaign scope")
    if stage != "transplant" and stage != ("stationary" if kernel in ("rbf", "matern52") else "extended"):
        raise ValueError("kernel/stage mismatch")
    return stage, kernel, dataset, m, method, seed


def read_campaign(root, prediction_roots):
    rows, issues, sources = {}, [], []
    expected = (root / "EXPECTED_HEAD").read_text().strip()

    def load(path):
        data = path.read_bytes()
        obj = json.loads(data)
        sources.append({"path": str(path), "sha256": "sha256:" + hashlib.sha256(data).hexdigest()})
        return obj

    # Freeze the enumeration first. A newly published file appears next time.
    paths = sorted({p.parent for stage in ("stationary", "extended", "transplant")
                    for name in ("metrics.json", "training.json")
                    for p in (root / "svgp" / stage / "runs").glob(f"**/{name}")})
    for path in paths:
        try:
            key = key_from_relative(path.relative_to(root))
            training = load(path / "training.json") if (path / "training.json").is_file() else None
            metrics = load(path / "metrics.json") if (path / "metrics.json").is_file() else None
            record = training or metrics
            cfg = record["config"]
            if cfg.get("code_commit") != expected:
                raise ValueError("numerical reference does not match EXPECTED_HEAD")
            if not (path / "params.npz").is_file():
                raise ValueError("checkpoint missing")
            if training is not None and training.get("training_complete") is not True:
                raise ValueError("training not complete")
            stage, kernel, dataset, m, method, seed = key
            expected_method = method + "-to-SVGP" if stage == "transplant" else method
            for field, value in (("kernel", kernel), ("dataset", dataset), ("M", m),
                                 ("method", expected_method), ("seed", seed)):
                if cfg.get(field, record.get(field)) != value:
                    raise ValueError(f"path/config mismatch: {field}")
            row = {"stage": stage, "kernel": kernel, "dataset": dataset,
                   "M": m, "method": method, "seed": seed, "config": cfg,
                   "config_id": record["config_id"], "run": str(path.relative_to(root)),
                   "objective": record.get(OBJECTIVE), "prediction_source": "none"}
            if metrics and metrics.get("config_id") == record["config_id"]:
                wanted = "sparse" if stage == "transplant" or method == "SVGP" else (
                    "diagonal" if method == "T-SVGP" else "block")
                actual = metrics.get("reported_predictor", cfg.get("reported_predictor"))
                if actual == wanted and finite(metrics.get("rmse")) and finite(metrics.get("nll")):
                    row.update(rmse=float(metrics["rmse"]), nll=float(metrics["nll"]),
                               prediction_source=str(path / "metrics.json"))
                else:
                    issues.append({"path": str(path / "metrics.json"), "reason": "missing metrics or wrong predictor"})
            rows[key] = row
        except (ValueError, KeyError, TypeError, OSError) as exc:
            issues.append({"path": str(path), "reason": str(exc)})

    hash_cache = {}
    for prediction_root in prediction_roots:
        for path in sorted((prediction_root / "svgp").glob("**/prediction.json")):
            try:
                key = key_from_relative(path.parent.relative_to(prediction_root))
                row = rows[key]
                sidecar = load(path)
                identity = sidecar["identity"]
                if identity["config_id"] != row["config_id"] or sidecar["training_config"] != row["config"]:
                    raise ValueError("stale prediction configuration")
                wanted = "sparse" if key[0] == "transplant" or key[4] == "SVGP" else (
                    "diagonal" if key[4] == "T-SVGP" else "block")
                if identity["predictor"] != wanted:
                    raise ValueError("wrong prediction estimator")
                inputs = identity["inputs"]
                required = {row["run"] + "/params.npz", "EXPECTED_HEAD"}
                if not required <= inputs.keys():
                    raise ValueError("prediction has no checkpoint provenance")
                for relative, expected_hash in inputs.items():
                    source = (root / relative).resolve()
                    if root not in source.parents:
                        raise ValueError("input hash path outside root")
                    stat = source.stat()
                    stamp = (source, stat.st_size, stat.st_mtime_ns)
                    if stamp not in hash_cache:
                        hash_cache[stamp] = digest(source)
                    if hash_cache[stamp] != expected_hash:
                        raise ValueError(f"stale prediction input: {relative}")
                if not all(finite(sidecar.get(m)) for m in ("rmse", "nll")):
                    raise ValueError("nonfinite prediction metrics")
                # Explicit precedence: a verified sidecar supersedes old metrics;
                # last supplied prediction directory wins, never adds a seed twice.
                row.update(rmse=float(sidecar["rmse"]), nll=float(sidecar["nll"]),
                           prediction_source=str(path))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                issues.append({"path": str(path), "reason": str(exc)})
    return rows, issues, sources


def paired_deltas(rows):
    paired = {}
    for key, after in rows.items():
        stage, kernel, dataset, m, method, seed = key
        if stage != "transplant" or not all(finite(after.get(v)) for v in ("rmse", "nll")):
            continue
        donor_stage = "stationary" if kernel in ("rbf", "matern52") else "extended"
        before = rows.get((donor_stage, kernel, dataset, m, method, seed))
        if not before or not all(finite(before.get(v)) for v in ("rmse", "nll")):
            continue
        split_fields = ("train_indices_sha256", "test_indices_sha256", "preprocessing_protocol", "data_signature")
        if any(after["config"].get(v) is None or after["config"].get(v) != before["config"].get(v)
               for v in split_fields):
            continue
        paired[key] = {**after, **{v: after[v] - before[v] for v in ("rmse", "nll")}}
    return paired


def make_gallery(rows, root, out, stamp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common.plot_style import method_color, method_marker, method_linewidth

    coverage, images = [], []
    paired = paired_deltas(rows)
    def save(fig, name, title):
        fig.suptitle(f"PREVIEW — {title}\n{stamp} | partial seeds; not release validated", fontsize=13)
        fig.text(0.5, 0.01, "Counts list n at M=64/128/256/512 (target 10 each); bars = SEM (undefined for n=1). Missing is not zero.",
                 ha="center", fontsize=8)
        handles = [plt.Line2D([], [], color=method_color(m), marker=method_marker(m), label=m) for m in METHODS]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.10, 1, 0.90))
        fig.savefig(out / f"{name}_PREVIEW.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        images.append((f"{name}_PREVIEW.png", title))

    for kind in ("donor", "transplant", "paired_delta"):
        for kernel in KERNELS:
            stage = "transplant" if kind != "donor" else (
                "stationary" if kernel in ("rbf", "matern52") else "extended")
            source = paired if kind == "paired_delta" else rows
            metrics = ("rmse", "nll", "objective") if kind == "donor" else ("rmse", "nll")
            for section, datasets in (("main", DATASETS[:4]), ("appendix", DATASETS[4:])):
                fig, axes = plt.subplots(len(metrics), 4, figsize=(16, 3.1 * len(metrics)), squeeze=False)
                for col, dataset in enumerate(datasets):
                    for r, metric in enumerate(metrics):
                        ax = axes[r, col]
                        any_data = False
                        for im, method in enumerate(METHODS):
                            means, sems, counts = [], [], []
                            for m in BUDGETS:
                                values, seeds = [], []
                                for seed in range(10):
                                    row = source.get((stage, kernel, dataset, m, method, seed), {})
                                    if finite(row.get(metric)):
                                        values.append(float(row[metric]) * (-1 if metric == "objective" else 1))
                                        seeds.append(seed)
                                n = len(values)
                                counts.append(n)
                                means.append(float(np.mean(values)) if n else np.nan)
                                sems.append(float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan)
                                coverage.append({"figure_kind": kind, "kernel": kernel, "dataset": dataset,
                                                 "method": method, "M": m, "metric": metric,
                                                 "n": n, "seeds": ",".join(map(str, seeds)),
                                                 "mean": means[-1] if n else "", "sem": sems[-1] if n > 1 else ""})
                            if any(counts):
                                any_data = True
                                ax.plot(BUDGETS, means, color=method_color(method), marker=method_marker(method),
                                        linewidth=method_linewidth(method))
                                has_sem = np.isfinite(sems)
                                if has_sem.any():
                                    ax.errorbar(np.array(BUDGETS)[has_sem], np.array(means)[has_sem],
                                                yerr=np.array(sems)[has_sem], fmt="none", color=method_color(method), capsize=2)
                            # A separate count row avoids putting labels over curves.
                            ax.text(0.02, 0.02 + 0.047 * im, f"{method}: " + "/".join(map(str, counts)),
                                    transform=ax.transAxes, color=method_color(method), fontsize=6.5)
                        if not any_data:
                            ax.text(0.5, 0.55, "NO AVAILABLE DATA", ha="center", transform=ax.transAxes, color="0.5")
                        if kind == "paired_delta":
                            ax.axhline(0, color="0.6", linewidth=0.8)
                        label = "Negative stochastic objective / N" if metric == "objective" else metric.upper()
                        ax.set_ylabel(("After − donor " if kind == "paired_delta" else "") + label)
                        ax.set_xticks(BUDGETS)
                        ax.set_xlabel("M")
                        ax.set_title(dataset if r == 0 else "")
                        ax.grid(alpha=0.2)
                title = {"donor": "Donor, own predictor", "transplant": "After transplant, SVGP predictor",
                         "paired_delta": "After SVGP transplant minus donor own predictor (paired seeds)"}[kind]
                save(fig, f"svgp_{kind}_{kernel}_{section}", f"{title} — {kernel} — {section}")

    # Official convergence scope: use only actual saved prediction history,
    # never invent epoch-level predictions from final metrics or objective curves.
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), squeeze=False)
    history_rows = []
    history_sources = []
    for col, kernel in enumerate(KERNELS):
        for method in METHODS:
            path = root / "svgp/convergence/runs" / kernel / "kin40k/M256" / method / "seed0/history.csv"
            if not path.is_file():
                continue
            try:
                record_path = next((path.parent / n for n in ("training.json", "metrics.json")
                                    if (path.parent / n).is_file()), None)
                if record_path is None:
                    continue
                record = json.loads(record_path.read_text())
                if record.get("config", {}).get("code_commit") != (root / "EXPECTED_HEAD").read_text().strip():
                    continue
                with path.open(newline="") as stream:
                    history = list(csv.DictReader(stream))
                history_sources.append({"path": str(path), "sha256": digest(path),
                                        "record": str(record_path), "record_sha256": digest(record_path)})
                for r, metric in enumerate(("nll", "rmse")):
                    points = []
                    for row in history:
                        fields = (f"{metric}_reported", f"{metric}_full", metric) if method != "SVGP" else (metric,)
                        value = next((row.get(k) for k in fields if finite(row.get(k))), None)
                        if finite(value) and finite(row.get("epoch")):
                            points.append((float(row["epoch"]), float(value)))
                    if points:
                        points.sort()
                        axes[r, col].plot(*zip(*points), color=method_color(method), marker=method_marker(method))
                        history_rows.extend({"kernel": kernel, "method": method, "metric": metric,
                                             "epoch": epoch, "value": value, "source": str(path)} for epoch, value in points)
            except (OSError, ValueError):
                pass
        for r, metric in enumerate(("NLL", "RMSE")):
            axes[r, col].set(title=kernel, xlabel="Epoch", ylabel=metric)
            axes[r, col].grid(alpha=0.2)
            if not axes[r, col].lines:
                axes[r, col].text(0.5, 0.5, "NO PREDICTION HISTORY", ha="center", transform=axes[r, col].transAxes)
    save(fig, "svgp_kin40k_convergence", "Saved convergence prediction history — M256, seed0 only")
    (out / "convergence_inputs.json").write_text(json.dumps(history_sources, indent=2))
    for filename, data in (("coverage.csv", coverage), ("convergence_points.csv", history_rows)):
        if data:
            with (out / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prediction-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.expanduser().resolve(), args.out.expanduser().resolve()
    predictions = [p.expanduser().resolve() for p in args.prediction_roots]
    for source in [root, *predictions]:
        if not source.is_dir():
            parser.error(f"missing input directory: {source}")
        if out == source or source in out.parents or out in source.parents:
            parser.error("preview output must be disjoint from all input directories")
    out.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(out / ".matplotlib")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print("Reading completed records and checking prediction provenance...", flush=True)
    rows, issues, sources = read_campaign(root, predictions)
    snapshot = {"preview_only": True, "release_ready": False, "started_utc": stamp,
                "root": str(root), "prediction_roots": list(map(str, predictions)),
                "generator_sha256": digest(Path(__file__)), "rows": list(rows.values()),
                "issues": issues, "sources": sources,
                "note": "Live read, not a globally atomic freeze. No new prediction or training. Timing metrics omitted."}
    (out / "PREVIEW_SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2))
    print(f"Records: {len(rows)}; with predictions: {sum('rmse' in r for r in rows.values())}; issues: {len(issues)}", flush=True)
    images = make_gallery(rows, root, out, stamp)
    body = "\n".join(f'<h2>{html.escape(title)}</h2><a href="{name}"><img loading="lazy" src="{name}" style="max-width:100%"></a>'
                     for name, title in images)
    (out / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>SVGP PREVIEW</title>'
        '<h1>SVGP PREVIEW — incomplete, not release validated</h1>'
        f'<p>{stamp}. Missing points are not zero. Donor and transplant deltas use only available paired predictions; '
        'absolute transplant panels also include predictions without an evaluated donor. Counts can differ across methods.</p>'
        '<p>Training-only objectives are shown, but no missing RMSE/NLL is inferred. No timing comparisons. '
        'Convergence uses saved history only. SGPR illustrations and separate ablations are not regenerated.</p>' + body)
    print(f"{len(images)} PNG previews and gallery: {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
