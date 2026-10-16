#!/usr/bin/env python3
"""Certify the separate, paired Gibbs-anchor 300-epoch transplant output."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    parent = campaign / "parent150"
    continued = campaign / "continued300"
    prediction = campaign / "predictions"
    plan_path = continued / "continuation_plan.json"
    plan = read_json(plan_path)
    if len(plan["runs"]) != 320 or plan["root"] != str(parent):
        raise ValueError("not the paired 320-run Gibbs-anchor plan")
    if plan["anchor_donor_root"] == str(parent):
        raise ValueError("donor and parent outputs are not separate")
    for relative, expected in plan["anchor_donor_inputs"].items():
        if sha(Path(plan["anchor_donor_root"]) / relative) != expected:
            raise ValueError(f"donor changed: {relative}")
    head = (continued / "EXPECTED_HEAD").read_text().strip()
    if head != (parent / "EXPECTED_HEAD").read_text().strip():
        raise ValueError("numerical reference changed")

    for worker in range(5):
        report = read_json(prediction / f"transplant_worker{worker}of5_evaluation_summary.json")
        if (report.get("expected") != 64 or report.get("successful") != 64
                or report.get("evaluation_complete") is not True
                or report.get("failed") or report.get("uniform_device_kind") is not True):
            raise ValueError(f"prediction worker {worker} incomplete")

    rows = []
    seen = set()
    plan_hash = sha(plan_path)
    for entry in plan["runs"]:
        relative = Path(entry["run"])
        if relative in seen:
            raise ValueError(f"duplicate run: {relative}")
        seen.add(relative)
        if relative.parts[:4] != ("svgp", "transplant", "runs", "gibbs"):
            raise ValueError(f"non-Gibbs run in scope: {relative}")
        parent_run = parent / relative
        continued_run = continued / relative
        prediction_path = prediction / relative / "prediction.json"
        parent_record = read_json(parent_run / "training.json")
        record = read_json(continued_run / "training.json")
        marker = read_json(continued_run / "COMPLETE.json")
        pred = read_json(prediction_path)
        cfg = record["config"]
        parent_cfg = parent_record["config"]
        if (parent_record.get("training_complete") is not True
                or parent_record.get("epochs") != 150
                or record.get("training_complete") is not True
                or record.get("epochs") != 300
                or cfg.get("donor_method") != "SBT-SVGP"
                or cfg.get("training_method") != "SVGP"
                or cfg.get("kernel") != "gibbs"
                or cfg.get("code_commit") != head
                or parent_cfg.get("donor_params_sha256") != cfg.get("donor_params_sha256")):
            raise ValueError(f"wrong training lineage: {relative}")
        if marker["identity"] != {
            "plan_sha256": plan_hash,
            "parent_config_id": entry["config_id"],
            "protocol": plan["protocol"],
        }:
            raise ValueError(f"wrong completion identity: {relative}")
        for filename, expected in marker["files"].items():
            if sha(continued_run / filename) != expected:
                raise ValueError(f"continuation file changed: {relative}/{filename}")
        for filename in ("epoch150.json", "epoch200.json", "epoch250.json", "epoch300.json"):
            checkpoint = read_json(continued_run / filename)
            if checkpoint.get("theta_unchanged") is not True:
                raise ValueError(f"hyperparameters changed: {relative}/{filename}")
        if (pred["identity"]["predictor"] != "sparse"
                or pred["identity"]["config_id"] != record["config_id"]
                or pred["training_config"] != cfg
                or pred.get("reference_check_points") != 8
                or not all(math.isfinite(float(pred[key])) for key in ("rmse", "nll"))):
            raise ValueError(f"uncertified sparse prediction: {relative}")
        for input_name, expected in pred["identity"]["inputs"].items():
            if sha(continued / input_name) != expected:
                raise ValueError(f"prediction input changed: {input_name}")
        rows.append({
            "kernel": "gibbs", "dataset": cfg["dataset"], "M": cfg["M"],
            "donor_method": "SBT-SVGP", "seed": cfg["seed"],
            "transplant_epoch": 300, "rmse": pred["rmse"],
            "nll": pred["nll"],
            "train_elbo_per_datum_epoch150": read_json(continued_run / "epoch150.json")["train_elbo_per_datum"],
            "train_elbo_per_datum_epoch300": read_json(continued_run / "epoch300.json")["train_elbo_per_datum"],
            "relative_run": str(relative),
        })
    if len(rows) != 320:
        raise ValueError("incomplete transplant grid")
    rows.sort(key=lambda row: (row["dataset"], row["M"], row["seed"]))
    aggregate = campaign / "gibbs_anchor_transplant300_aggregate.csv"
    temporary = aggregate.with_name(f".{aggregate.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, aggregate)
    atomic_json(campaign / "GIBBS_ANCHOR_TRANSPLANT_CERTIFICATE.json", {
        "release_certified": True,
        "scope": "Gibbs SBT-SVGP inducing-anchor donors to sparse SVGP; 150+150 epochs",
        "runs": 320,
        "prediction_model": "sparse SVGP at 300 epochs",
        "numerical_reference": head,
        "donor_root": plan["anchor_donor_root"],
        "plan_sha256": plan_hash,
        "aggregate_sha256": sha(aggregate),
        "notes": "Donor freeze unchanged; training and predictions are separate outputs.",
    })
    print("CERTIFIED 320/320 paired Gibbs-anchor transplants and sparse predictions")
    print(f"aggregate: {aggregate}")


if __name__ == "__main__":
    main()
