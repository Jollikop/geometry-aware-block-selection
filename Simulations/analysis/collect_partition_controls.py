#!/usr/bin/env python3
"""Collect certified new controls and their paired, per-seed NLL differences."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import real_predictor_campaign as real


def sparse_reference(row, identity, external_root):
    candidates = []
    if external_root:
        candidates.append(external_root/row["relative"]/"prediction.json")
    candidates.append(Path(row["root"])/row["relative"]/"metrics.json")
    for path in candidates:
        if not path.is_file():
            continue
        ref = json.loads(path.read_text())
        if ref.get("identity", {}).get("config_id", ref.get("config_id")) != identity["config_id"]:
            raise ValueError("standard SVGP reference names a different donor")
        if "identity" in ref:
            if ref["identity"].get("predictor") != "sparse":
                raise ValueError("standard SVGP reference is not sparse")
            for name, digest in identity["inputs"].items():
                if ref["identity"]["inputs"].get(name) != digest:
                    raise ValueError("standard SVGP reference checkpoint differs")
        if ref.get("nll") is not None:
            return float(ref["nll"]), str(path), real.sha256(path)
    return None, None, None


def collect(roots, output, sparse_reference_root=None):
    real.guard_separate(output, roots)
    if output.exists():
        raise ValueError("collection output exists; select a new directory")
    paired, inputs, grouped = [], {}, defaultdict(list)
    for root in roots:
        plan_path = root/"plan.json"
        plan = json.loads(plan_path.read_text())
        dense = plan.get("implementation") == "dense-cutoff-sweep-v1"
        cert_path = root/("DENSE_CUTOFF_CERTIFICATE.json" if dense else "PARTITION_CONTROL_CERTIFICATE.json")
        cert = json.loads(cert_path.read_text())
        if not cert["complete"] or cert["plan_sha256"] != real.sha256(plan_path):
            raise ValueError(f"control not complete/certified: {root}")
        inputs[str(root)] = dict(plan_sha256=real.sha256(plan_path), certificate_sha256=real.sha256(cert_path))
        experiment = "E4" if dense else plan["experiment"]
        for row in plan["runs"]:
            path = root/row["output_relative"]/"cutoff_sweep.json" if dense else (
                root/"hard"/row["output_relative"]/"prediction.json")
            certified = cert["predictions"][row["output_relative"]]
            expected_hash = certified if dense else certified["sha256"]
            if real.sha256(path) != expected_hash:
                raise ValueError("certified prediction changed")
            p = json.loads(path.read_text())
            if dense:
                ref_path = Path(row["reference"])
                if real.sha256(ref_path) != row["reference_sha256"]:
                    raise ValueError("dense paired reference changed")
                current = json.loads(ref_path.read_text())["nll"]
                variants = [(f"cutoff-{r['scale']:g}", r) for r in p["results"]]
                reference_name, reference_sha = str(ref_path), row["reference_sha256"]
                config_id = p["config_id"]
            else:
                if not p["valid"]:
                    raise ValueError("invalid prediction in a completed control certificate")
                identity = p["identity"]
                if identity["inputs"] != row["frozen_inputs"] or (
                        identity["plan_sha256"] != real.sha256(plan_path)):
                    raise ValueError("prediction/checkpoint provenance mismatch")
                for name,value in plan["control_options"].items():
                    if identity.get(name) != value:
                        raise ValueError("prediction variant differs from planned options")
                record_path = Path(row["root"])/row["relative"]/"training.json"
                if not record_path.is_file():
                    record_path = record_path.with_name("metrics.json")
                record = json.loads(record_path.read_text())
                if record["config_id"] != identity["config_id"] or record["config"]["block"] != 500:
                    raise ValueError("donor configuration or structural width differs")
                variant = {"E1":"RBT-donor_anchor", "E2":"SBT-donor_random", "E3":"direct-sparse-donor"}[experiment]
                variants = [(variant,p)]
                config_id = p["identity"]["config_id"]
                if "reference" in row:
                    ref_path = Path(row["reference"])
                    if real.sha256(ref_path) != row["reference_sha256"]:
                        raise ValueError("paired block reference changed")
                    current = json.loads(ref_path.read_text())["nll"]
                    reference_name, reference_sha = str(ref_path), row["reference_sha256"]
                else:
                    current, reference_name, reference_sha = sparse_reference(
                        row, p["identity"], sparse_reference_root)
            for variant,r in variants:
                if not r["valid"]:
                    continue
                key = (experiment,variant,row["kernel"],row.get("dataset", "kin40k" if row["family"]=="sgpr" else "pol"),row["M"],row["method"])
                delta = None if current is None else r["nll"]-current
                for metric,value in (("rmse",r["rmse"]),("nll",r["nll"]),
                                     ("nll_difference_from_current",delta)):
                    if value is not None:
                        grouped[(key,metric)].append(float(value))
                paired.append(dict(experiment=experiment,variant=variant,kernel=key[2],
                    dataset=key[3],M=row["M"],method=row["method"],seed=row["seed"],
                    config_id=config_id,rmse=r["rmse"],nll=r["nll"],current_nll=current,
                    nll_difference_from_current=delta,prediction_sha256=expected_hash,
                    reference=reference_name,reference_sha256=reference_sha))
    rows = []
    for (key,metric),values in sorted(grouped.items()):
        rows.append(dict(experiment=key[0],variant=key[1],kernel=key[2],dataset=key[3],
            M=key[4],method=key[5],metric=metric,n=len(values),mean=float(np.mean(values)),
            sem=float(np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else 0.0))
    output.mkdir(parents=True)
    for name,records in (("partition_controls_summary.csv",rows),("paired_per_seed.csv",paired)):
        with (output/name).open("w",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    real.atomic_json(output/"COLLECTION_CERTIFICATE.json",dict(inputs=inputs,
        collector_sha256=real.sha256(Path(__file__)),
        outputs={n:real.sha256(output/n) for n in ("partition_controls_summary.csv","paired_per_seed.csv")},
        per_seed_records=len(paired),missing_paired_references=sum(r["current_nll"] is None for r in paired),
        scope="new prediction-only controls; no original results changed"))
    print(f"Collected {len(paired)} per-seed records; {len(rows)} mean/SEM rows; missing paired references={sum(r['current_nll'] is None for r in paired)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots",nargs="+",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--sparse-reference-root",type=Path)
    args = parser.parse_args()
    collect([p.resolve() for p in args.roots],args.output.resolve(),args.sparse_reference_root)


if __name__ == "__main__":
    main()
