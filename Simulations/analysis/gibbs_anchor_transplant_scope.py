#!/usr/bin/env python3
"""Scope the established 150-to-300 transplant continuation to Gibbs anchors.

This file selects and audits runs; all training and prediction arithmetic stays
in the existing campaign implementation. Parent, continuation and prediction
outputs must be separate from the frozen donor tree.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


DATASETS = (
    "pol superconductivity kin40k protein elevators traffic electricity 3droad"
).split()
MS = (64, 128, 256, 512)
SEEDS = range(10)


def continuation_module(path: Path):
    spec = importlib.util.spec_from_file_location("transplant_continuation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def selected_runs(module):
    return [
        (relative, identity)
        for relative, identity in module.expected_runs()
        if identity["kernel"] == "gibbs"
        and identity["donor_method"] == "SBT-SVGP"
    ]


def audit_donors(module, donor_root: Path):
    inputs = {}
    for dataset in DATASETS:
        for m in MS:
            for seed in SEEDS:
                run = (
                    donor_root / "svgp/extended/runs/gibbs" / dataset
                    / f"M{m}" / "SBT-SVGP" / f"seed{seed}"
                )
                record_path = run / "training.json"
                params_path = run / "params.npz"
                record = json.loads(record_path.read_text())
                config = record["config"]
                expected = {
                    "kernel": "gibbs", "method": "SBT-SVGP", "dataset": dataset,
                    "M": m, "seed": seed, "sbt_geometry": "anchor",
                    "init_from_kernel": "rbf", "epochs": 300,
                }
                if record.get("training_complete") is not True or any(
                    config.get(key) != value for key, value in expected.items()
                ):
                    raise ValueError(f"not a complete Gibbs-anchor donor: {run}")
                inputs[str(record_path.relative_to(donor_root))] = module.sha(record_path)
                inputs[str(params_path.relative_to(donor_root))] = module.sha(params_path)
    if len(inputs) != 640:
        raise ValueError(f"expected 320 donor records and parameters, got {len(inputs)}")
    return inputs


def check_donors(module, plan):
    root = Path(plan["anchor_donor_root"])
    for relative, expected in plan["anchor_donor_inputs"].items():
        if module.sha(root / relative) != expected:
            raise ValueError(f"frozen anchor donor changed: {relative}")


def prepare(module, args):
    parent = args.parent.resolve()
    output = args.output.resolve()
    donor_root = args.donor_root.resolve()
    module.separate(parent, output)
    module.separate(donor_root, output)
    actual_donor = (parent / "svgp/extended/runs/gibbs").resolve()
    expected_donor = (donor_root / "svgp/extended/runs/gibbs").resolve()
    if actual_donor != expected_donor:
        raise ValueError("parent donor link does not point to the frozen anchor tree")
    donor_inputs = audit_donors(module, donor_root)
    selected = selected_runs(module)
    if len(selected) != 320:
        raise ValueError(f"expected 320 paired transplants, got {len(selected)}")
    original = module.expected_runs
    module.expected_runs = lambda: iter(selected)
    try:
        module.plan(SimpleNamespace(root=parent, output=output,
                                    data_dir=args.data_dir.resolve()))
    finally:
        module.expected_runs = original
    plan_path = output / "continuation_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["selection_rule"] = (
        "all eight datasets, four M values and ten seeds for Gibbs "
        "SBT-SVGP inducing-anchor donors; no test-metric selection"
    )
    plan["anchor_donor_root"] = str(donor_root)
    plan["anchor_donor_inputs"] = donor_inputs
    plan["runs_expected"] = 320
    plan["scope_script_sha256"] = module.sha(Path(__file__))
    module.write_json(plan_path, plan)
    # The later sparse-prediction auditor checks the donor through its output
    # root. Keep that path read-only and point it at the same audited donor.
    target = output / "svgp/extended/runs"
    target.mkdir(parents=True)
    (target / "gibbs").symlink_to(expected_donor, target_is_directory=True)
    (output / "runs.txt").write_text(
        "\n".join(row["run"] for row in plan["runs"]) + "\n"
    )
    print("ANCHOR TRANSPLANT PLAN: 320/320 paired runs; donor inputs audited", flush=True)


def worker(module, args):
    plan = module.read_plan(args.output.resolve())
    if len(plan["runs"]) != 320 or plan.get("runs_expected") != 320:
        raise ValueError("wrong transplant scope")
    if plan.get("scope_script_sha256") != module.sha(Path(__file__)):
        raise ValueError("scope script changed after planning")
    check_donors(module, plan)
    module.worker(SimpleNamespace(output=args.output.resolve(), worker=args.worker))


def prepare_prediction(module, args):
    import numpy as np

    output = args.output.resolve()
    plan = module.read_plan(output)
    if len(plan["runs"]) != 320:
        raise ValueError("wrong transplant scope")
    if plan.get("scope_script_sha256") != module.sha(Path(__file__)):
        raise ValueError("scope script changed after planning")
    check_donors(module, plan)
    parent = Path(plan["root"])
    for row in plan["runs"]:
        run = output / row["run"]
        if not module.finished(run, {
            "plan_sha256": module.sha(output / "continuation_plan.json"),
            "parent_config_id": row["config_id"],
            "protocol": module.PROTOCOL,
        }):
            raise ValueError(f"300-epoch transplant incomplete: {row['run']}")
        values = {}
        for source in [parent / row["run"] / "curve.npz",
                       *sorted(run.glob("curve_from_epoch*.npz"))]:
            with np.load(source, allow_pickle=False) as curve:
                for epoch, objective in zip(curve["epoch"], curve["stoch_obj_n"]):
                    values[int(epoch)] = float(objective)
        if not values or max(values) != 300:
            raise ValueError(f"incomplete objective curve: {row['run']}")
        target = run / "curve.npz"
        if not target.exists():
            temporary = run / f".curve.{os.getpid()}.npz"
            epochs = np.asarray(sorted(values), dtype=np.int32)
            np.savez_compressed(temporary, epoch=epochs,
                                stoch_obj_n=np.asarray([values[e] for e in epochs]))
            os.replace(temporary, target)
    print("PREDICTION INPUTS READY: 320/320 saved 300-epoch posteriors", flush=True)


def status(module, args):
    output = args.output.resolve()
    plan_path = output / "continuation_plan.json"
    if not plan_path.exists():
        print("Continuation plan pending")
        return
    plan = json.loads(plan_path.read_text())
    counts = {epoch: 0 for epoch in (200, 250, 300)}
    complete = 0
    for row in plan["runs"]:
        run = output / row["run"]
        for epoch in counts:
            counts[epoch] += int((run / f"epoch{epoch}.json").is_file())
        complete += int((run / "COMPLETE.json").is_file())
    print("Gibbs inducing-anchor transplant, 150 -> 300 epochs")
    for epoch, count in counts.items():
        print(f"epoch {epoch}: {count}/320")
    print(f"complete: {complete}/320")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "worker", "prepare-prediction", "status"))
    parser.add_argument("--continuation-tool", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--donor-root", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    module = continuation_module(args.continuation_tool.resolve())
    if args.mode == "plan":
        if not args.parent or not args.donor_root or not args.data_dir:
            parser.error("plan requires parent, donor-root and data-dir")
        prepare(module, args)
    elif args.mode == "worker":
        if args.worker is None or not 0 <= args.worker < 5:
            parser.error("worker must be in [0, 5)")
        worker(module, args)
    elif args.mode == "prepare-prediction":
        prepare_prediction(module, args)
    else:
        status(module, args)


if __name__ == "__main__":
    main()
