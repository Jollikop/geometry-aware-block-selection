#!/usr/bin/env python3
"""Paired comparison of SBT uniform-cycle runs with and without refresh."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(root: Path):
    rows = {}
    for path in root.rglob("metrics.json"):
        metric = json.loads(path.read_text())
        cfg = metric.get("config", {})
        if cfg.get("method") != "SBT-SVGP":
            continue
        key = (
            cfg.get("kernel"), cfg.get("dataset"), int(cfg.get("M")),
            int(cfg.get("seed")),
        )
        rows[key] = metric
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-runs", required=True)
    parser.add_argument("--refresh-runs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    control = load(Path(args.control_runs))
    refreshed = load(Path(args.refresh_runs))
    common = sorted(set(control) & set(refreshed))
    if not common:
        raise SystemExit("No paired SBT runs were found")

    rows = []
    grouped = defaultdict(list)
    for key in common:
        old, new = control[key], refreshed[key]
        old_cfg, new_cfg = old["config"], new["config"]
        for name, cfg in (("control", old_cfg), ("refresh", new_cfg)):
            if cfg.get("penalty_sampling") != "uniform":
                raise SystemExit(f"{name} run {key} is not uniform-cycle")
            if cfg.get("batch_coupling") != "split":
                raise SystemExit(f"{name} run {key} does not use split batches")
        if old_cfg.get("data_signature") != new_cfg.get("data_signature"):
            raise SystemExit(f"dataset signature mismatch for {key}")

        row = {
            "kernel": key[0], "dataset": key[1], "M": key[2], "seed": key[3],
            "control_refreshes": old.get("refreshes"),
            "refresh_refreshes": new.get("refreshes"),
            "post_swa_structure_rebuilds": new.get(
                "post_swa_structure_rebuilds", 0),
            "control_nll": float(old["nll"]),
            "refresh_nll": float(new["nll"]),
            "delta_nll": float(new["nll"]) - float(old["nll"]),
            "control_rmse": float(old["rmse"]),
            "refresh_rmse": float(new["rmse"]),
            "delta_rmse": float(new["rmse"]) - float(old["rmse"]),
        }
        rows.append(row)
        grouped[key[:3]].append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Negative deltas favour structure refresh.\n")
    print(f"{'kernel':10} {'dataset':18} {'M':>4} {'n':>3} "
          f"{'delta NLL':>12} {'delta RMSE':>12} {'wins':>7}")
    for key, values in sorted(grouped.items()):
        dnll = np.asarray([row["delta_nll"] for row in values])
        drmse = np.asarray([row["delta_rmse"] for row in values])
        print(f"{key[0]:10} {key[1]:18} {key[2]:4d} {len(values):3d} "
              f"{dnll.mean():+12.5f} {drmse.mean():+12.5f} "
              f"{np.mean(dnll < 0):7.2f}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
