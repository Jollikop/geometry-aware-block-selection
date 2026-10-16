#!/usr/bin/env python3
"""Read-only partition controls on certified frozen donor checkpoints."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import hard_block_prediction_campaign as hard
import real_predictor_campaign as real
import weighted_residual_campaign as campaign
import tsvgp_anchor_block_campaign as anchor

FIG4_KERNELS = ("rbf", "amplitude-rbf")
FIG4_DATASETS = ("pol", "superconductivity", "kin40k", "protein")


def select_runs(base, experiment, scope):
    rows = [copy.deepcopy(r) for r in base["runs"] if r["family"] == "svgp"]
    if experiment in ("E1", "E2"):
        method = "RBT-SVGP" if experiment == "E1" else "SBT-SVGP"
        rows = [r for r in rows if r["method"] == method]
    else:
        # Every configuration has a T donor in the historical source. Add its
        # matching standard SVGP donor; Gibbs SBT keeps its certified anchor root.
        originals = [copy.deepcopy(r) for r in rows if r["method"] == "T-SVGP"]
        for r in originals:
            r["method"] = "SVGP"
            r["relative"] = r["relative"].replace("/T-SVGP/", "/SVGP/")
            r["output_relative"] = r["output_relative"].replace("/T-SVGP/", "/SVGP/")
        rows.extend(originals)
    if scope == "fig4":
        rows = [r for r in rows if r["kernel"] in FIG4_KERNELS
                and r["dataset"] in FIG4_DATASETS]
    expected = (320 if scope == "fig4" else 1280) * (4 if experiment == "E3" else 1)
    if len(rows) != expected:
        raise ValueError(f"control grid has {len(rows)} rows, expected {expected}")
    keys = {(r["kernel"], r["dataset"], r["M"], r["method"], r["seed"]) for r in rows}
    if len(keys) != expected:
        raise ValueError("duplicate configurations in control grid")
    return rows


def prepare(args):
    baseline = args.baseline.resolve()
    output = args.output.resolve()
    base_path = baseline / "plan.json"
    base = json.loads(base_path.read_text())
    real.guard_separate(output, [baseline] + [Path(s["root"]) for s in base["sources"].values()]
                        + [Path(r["root"]) for r in base["runs"]])
    certificate_path = baseline / "HARD_PIC_PREDICTION_CERTIFICATE.json"
    certificate = json.loads(certificate_path.read_text())
    if certificate["plan_sha256"] != real.sha256(base_path):
        raise ValueError("baseline certificate names another plan")
    rows = select_runs(base, args.experiment, args.scope)
    for source in base["sources"].values():
        for name, digest in source["files"].items():
            if real.sha256(Path(source["root"]) / name) != digest:
                raise ValueError(f"historical source changed: {name}")
    for row in rows:
        run = Path(row["root"]) / row["relative"]
        record_path = campaign.record_path(run)
        record = json.loads(record_path.read_text())
        cfg = record["config"]
        if cfg["code_commit"] != base["expected_heads"]["svgp"]:
            raise ValueError("checkpoint commit differs from frozen source")
        if any(cfg[k] != row[k] for k in ("kernel", "dataset", "M", "seed")):
            raise ValueError("checkpoint configuration differs from planned row")
        if cfg.get("structural_block_size", cfg.get("batch_size", 500)) != 500:
            raise ValueError("unexpected frozen structural block size")
        row["frozen_inputs"] = campaign.verify_run_inputs(base, row, run, record_path)
        current = baseline / "hard" / row["output_relative"] / "prediction.json"
        if row["method"] != "SVGP":
            if certificate["predictions"].get(row["output_relative"]) != real.sha256(current):
                raise ValueError("baseline prediction is not certified")
            reference = json.loads(current.read_text())
            if reference["identity"]["inputs"] != row["frozen_inputs"]:
                raise ValueError("baseline/control do not use the same checkpoint")
            row["reference"] = str(current)
            row["reference_sha256"] = real.sha256(current)
    plan = copy.deepcopy(base)
    plan.update(runs=rows, implementation="partition-controls-plan-v1",
                experiment=args.experiment, scope=args.scope,
                baseline_plan=str(base_path), baseline_plan_sha256=real.sha256(base_path),
                baseline_certificate=str(certificate_path),
                baseline_certificate_sha256=real.sha256(certificate_path),
                control_evaluator_sha256=real.sha256(Path(__file__)),
                control_options=dict(partition_source={"E1": "anchor", "E2": "random",
                                                       "E3": "default"}[args.experiment],
                                     predictor="sparse" if args.experiment == "E3" else "hard",
                                     seed_pred_offset=10007),
                hard_evaluator_sha256=real.sha256(Path(hard.__file__)))
    # Historical source hashes remain unchanged; only evaluator hashes are new.
    for key, path in (("evaluator_sha256", campaign.__file__),
                      ("weighted_engine_sha256", campaign.weighted.__file__),
                      ("sgpr_loader_sha256", real.__file__),
                      ("svgp_loader_sha256", anchor.__file__)):
        plan[key] = real.sha256(Path(path))
    destination = output / "plan.json"
    if destination.exists():
        raise ValueError("output plan already exists; select a new output root")
    real.atomic_json(destination, plan)
    print(f"{args.experiment} {args.scope}: planned {len(rows)} predictions at {output}")


def load_plan(output):
    plan = hard.load_plan(output)
    if plan["control_evaluator_sha256"] != real.sha256(Path(__file__)):
        raise ValueError("control orchestration changed after planning")
    for label in ("baseline_plan", "baseline_certificate"):
        if real.sha256(Path(plan[label])) != plan[label + "_sha256"]:
            raise ValueError(f"baseline provenance changed: {label}")
    return plan


def worker(args):
    plan = load_plan(args.output)
    options = plan["control_options"]
    indices = list(range(args.worker, len(plan["runs"]), args.workers))
    failures, durations = [], []
    for position, index in enumerate(indices, 1):
        print(f"[{position}/{len(indices)}] {plan['experiment']} {plan['runs'][index]['output_relative']}", flush=True)
        start = time.perf_counter()
        result = subprocess.run([sys.executable, "-u", hard.__file__, "evaluate",
            "--output", str(args.output), "--index", str(index),
            "--partition-source", options["partition_source"],
            "--predictor", options["predictor"], "--seed-pred-offset", "10007",
            "--test-chunk", "512", "--eig-cutoff-scale", "10"], check=False)
        durations.append(dict(index=index, allocated_gpu_wall_seconds=time.perf_counter()-start))
        if result.returncode:
            failures.append(dict(index=index, exit_code=result.returncode))
        real.atomic_json(args.output / "state" / f"worker{args.worker}.json",
                         dict(assigned=len(indices), processed=position, failures=failures,
                              durations=durations, complete=position == len(indices)))
    return 1 if failures else 0


def summary_rows(plan, payloads):
    grouped = defaultdict(list)
    variant = {"E1": "RBT-donor_anchor", "E2": "SBT-donor_random",
               "E3": "direct-sparse-donor"}[plan["experiment"]]
    for row, p in payloads:
        if not p["valid"]:
            continue
        key = (row["kernel"], row["dataset"], row["M"], row["method"])
        for metric in ("rmse", "nll"):
            grouped[(key, metric)].append(p[metric])
        if "reference" in row:
            ref_path = Path(row["reference"])
            if real.sha256(ref_path) != row["reference_sha256"]:
                raise ValueError("paired reference changed")
            ref = json.loads(ref_path.read_text())
            grouped[(key, "nll_difference_from_current")].append(p["nll"]-ref["nll"])
    return [dict(experiment=plan["experiment"], variant=variant, kernel=key[0],
                 dataset=key[1], M=key[2], method=key[3], metric=metric,
                 n=len(values), mean=float(np.mean(values)),
                 sem=float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.0)
            for (key, metric), values in sorted(grouped.items())]


def certify(args):
    plan = load_plan(args.output)
    payloads, missing, rejected, digests = [], [], [], {}
    for row in plan["runs"]:
        path = args.output / "hard" / row["output_relative"] / "prediction.json"
        if not path.is_file():
            missing.append(row["output_relative"])
            continue
        p = json.loads(path.read_text())
        identity = p["identity"]
        if identity["plan_sha256"] != real.sha256(args.output / "plan.json") or (
                identity["evaluator_sha256"] != plan["hard_evaluator_sha256"]):
            raise ValueError("prediction evaluator/plan mismatch")
        run = Path(row["root"]) / row["relative"]
        inputs = campaign.verify_run_inputs(plan, row, run, campaign.record_path(run))
        if inputs != row["frozen_inputs"] or identity["inputs"] != inputs:
            raise ValueError("frozen inputs changed")
        if not p["valid"]:
            rejected.append(dict(run=row["output_relative"], reason=p["diagnostics"]["rejection_reason"]))
        payloads.append((row, p))
        digests[row["output_relative"]] = dict(sha256=real.sha256(path),
            partition_sha256=p["partition"]["sha256"], seed_pred=p["partition"].get("seed_pred"))
    for source in plan["sources"].values():
        for name, digest in source["files"].items():
            if real.sha256(Path(source["root"]) / name) != digest:
                raise ValueError(f"historical source changed: {name}")
    timings = [p["diagnostics"]["total_prediction_seconds"] for _, p in payloads]
    certificate = dict(implementation="partition-controls-certificate-v1",
        experiment=plan["experiment"], plan_sha256=real.sha256(args.output/"plan.json"),
        count=len(payloads), expected=len(plan["runs"]), valid_count=len(payloads)-len(rejected),
        rejected_count=len(rejected), rejected=rejected, missing=missing,
        min_raw_residual_variance=min((p["diagnostics"]["min_raw_residual"] for _, p in payloads), default=None),
        max_jitter=max((p["diagnostics"]["maximum_cholesky_jitter"] for _, p in payloads), default=0),
        predictions=digests, timing=dict(mean_seconds=float(np.mean(timings)) if timings else None,
            median_seconds=float(np.median(timings)) if timings else None,
            definition="synchronized end-to-end prediction wall time on one allocated GPU; includes load, compilation, search, transfers"),
        complete=not missing and not rejected,
        scope="prediction-only; frozen inputs verified before and after each evaluation")
    real.atomic_json(args.output / "PARTITION_CONTROL_CERTIFICATE.json", certificate)
    rows = summary_rows(plan, payloads)
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("experiment", "variant", "kernel", "dataset",
                                   "M", "method", "metric", "n", "mean", "sem"))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k:v for k,v in certificate.items() if k not in ("predictions", "missing", "rejected")}, indent=2))
    return 0 if certificate["complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--experiment", choices=("E1", "E2", "E3"), required=True)
    p.add_argument("--scope", choices=("fig4", "all"), default="fig4")
    for name in ("worker", "certify"):
        q = sub.add_parser(name)
        if name == "worker":
            q.add_argument("--worker", type=int, required=True)
            q.add_argument("--workers", type=int, default=5)
    for q in sub.choices.values():
        q.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    return {"plan": prepare, "worker": worker, "certify": certify}[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
