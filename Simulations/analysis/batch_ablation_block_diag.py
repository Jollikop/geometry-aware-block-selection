#!/usr/bin/env python3
"""Reproduce the RBF-Pol batch ablation with certified epochwise NLL.

The numerical training modules are imported from a clean, commit-pinned
checkout. This analysis wrapper only snapshots parameters and fitted blocks
at the existing 20-epoch history points, then evaluates the paper's
block-diagonal predictor in separate files. It never edits the source fits.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from statistics import mean, stdev

import numpy as np

EPOCHS = tuple(range(20, 341, 20)) + (350,)
ARMS = (("SVGP", "split"), ("T-SVGP", "split"),
        ("RBT-SVGP", "split"), ("SBT-SVGP", "split"),
        ("SBT-SVGP", "coupled"))
IMPLEMENTATION = "rbf-pol-block-diag-ablation-v1"


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def source_commit(source_sim: Path) -> str:
    project = source_sim.parent
    commit = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain",
         "--untracked-files=all", "--", "Simulations"], text=True).strip()
    if dirty:
        raise ValueError("numerical source checkout is not clean")
    return commit


def run_dir(root: Path, method: str, coupling: str, seed: int) -> Path:
    return (root / "runs" / coupling / "rbf" / "pol" / "M128"
            / method / f"seed{seed}")


def read_manifest(path: Path, *, source_sim: Path) -> dict:
    payload = json.loads(path.read_text())
    if (payload["implementation"] != IMPLEMENTATION
            or payload["source_commit"] != source_commit(source_sim)
            or payload["snapshotter_sha256"] != sha(Path(__file__))
            or [int(row["epoch"]) for row in payload["snapshots"]] != list(EPOCHS)):
        raise ValueError(f"snapshot provenance mismatch: {path}")
    for row in payload["snapshots"]:
        folder = path.parent / "snapshots" / f"epoch{row['epoch']:03d}"
        if sha(folder / "params.npz") != row["params_sha256"]:
            raise ValueError(f"snapshot parameters changed: {folder}")
        if row["blocks_sha256"] is not None and (
                sha(folder / "blocks.npz") != row["blocks_sha256"]):
            raise ValueError(f"snapshot blocks changed: {folder}")
    return payload


def train(args: argparse.Namespace) -> None:
    source_sim = args.source_sim.resolve()
    commit = source_commit(source_sim)
    out = run_dir(args.output, args.method, args.coupling, args.seed)
    manifest_path = out / "snapshot_manifest.json"
    if manifest_path.is_file():
        read_manifest(manifest_path, source_sim=source_sim)
        print(f"TRAIN CACHED {args.method} {args.coupling} seed{args.seed}", flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"partial run needs manual inspection: {out}")
    sys.path.insert(0, str(source_sim))
    from svgp import engine as numerical  # noqa: E402
    from svgp import run_benchmark as benchmark  # noqa: E402
    import jax  # noqa: E402

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("training requires exactly one assigned GPU")
    captured = []
    original = numerical.train_svgp

    def record_train(engine, method, params, make_bank, metrics_fn, n, run_args,
                     **kwargs):
        latest_bank = [None]

        def track_bank(current):
            bank = make_bank(current)
            latest_bank[0] = bank
            return bank

        def record_metrics(current):
            index = len(captured)
            if index >= len(EPOCHS):
                raise ValueError("unexpected extra history evaluation")
            epoch = EPOCHS[index]
            folder = out / "snapshots" / f"epoch{epoch:03d}"
            atomic_npz(folder / "params.npz", {
                key: np.asarray(jax.device_get(value))
                for key, value in current.items()})
            blocks_saved = False
            if method in ("RBT-SVGP", "SBT-SVGP"):
                bank = latest_bank[0]
                if bank is None:
                    raise ValueError("metric callback preceded block construction")
                atomic_npz(folder / "blocks.npz", {
                    "idx": np.asarray(jax.device_get(bank["base_idx"])),
                    "mask": np.asarray(jax.device_get(bank["base_mask"])),
                })
                blocks_saved = True
            captured.append({"epoch": epoch, "blocks": blocks_saved})
            return metrics_fn(current)

        # The historical structured history callback computes a dense/full
        # prediction at every epoch. The new block-diagonal evaluation is a
        # separate pass, so only the sparse mean diagnostic is recorded here.
        kwargs["struct_metrics_fn"] = None
        return original(engine, method, params, track_bank, record_metrics,
                        n, run_args, **kwargs)

    numerical.train_svgp = record_train
    original_argv = sys.argv
    sys.argv = [str(source_sim / "svgp" / "run_benchmark.py"),
        "--plan", "custom", "--datasets", "pol", "--kernels", "rbf",
        "--methods", args.method, "--m", "128", "--seeds", str(args.seed),
        "--all-data", "--epochs", "350", "--block", "500",
        "--ll-batch-size", "1000", "--blocks-per-step", "2",
        "--refresh-every", "40", "--refresh-until", "200",
        "--importance-refresh-every", "10", "--importance-epsilon", "0.05",
        "--common-ll-batches", "--batch-coupling", args.coupling,
        "--penalty-sampling", "uniform", "--optimizer", "lion",
        "--lr", "0.003", "--z-lr", "0.003", "--lr-profile", "stable",
        "--q-optimizer", "natural", "--natural-lr", "0.05",
        "--sbt-geometry", "anchor", "--normalize-objective",
        "--record-history", "--save-params", "--prediction-reps", "0",
        "--run-tag", IMPLEMENTATION,
        "--runs-dir", str(args.output / "runs" / args.coupling),
        "--logs-dir", str(args.output / "logs" / args.coupling), "--no-live"]
    try:
        benchmark.main()
    finally:
        sys.argv = original_argv
        numerical.train_svgp = original
    if [row["epoch"] for row in captured] != list(EPOCHS):
        raise ValueError(f"history grid mismatch: {captured}")
    if args.method in ("RBT-SVGP", "SBT-SVGP") and not all(
            row["blocks"] for row in captured):
        raise ValueError("fitted block snapshot missing")
    training = out / "metrics.json"
    history = out / "history.csv"
    if not training.is_file() or not history.is_file():
        raise ValueError("training metrics or history missing")
    config = json.loads(training.read_text())["config"]
    if config.get("code_commit") != commit:
        raise ValueError("training record cites a different source commit")
    with history.open(newline="") as stream:
        history_epochs = [int(row["epoch"]) for row in csv.DictReader(stream)]
    if history_epochs != list(EPOCHS):
        raise ValueError(f"saved history epochs differ: {history_epochs}")
    rows = []
    for row in captured:
        epoch = row["epoch"]
        folder = out / "snapshots" / f"epoch{epoch:03d}"
        blocks = folder / "blocks.npz"
        rows.append({"epoch": epoch,
                     "params_sha256": sha(folder / "params.npz"),
                     "blocks_sha256": sha(blocks) if blocks.is_file() else None})
    atomic_json(manifest_path, {
        "implementation": IMPLEMENTATION,
        "snapshotter_sha256": sha(Path(__file__)),
        "source_commit": commit,
        "method": args.method, "coupling": args.coupling, "seed": args.seed,
        "training_metrics_sha256": sha(training),
        "history_sha256": sha(history), "snapshots": rows,
    })
    print(f"SNAPSHOTS {args.method} {args.coupling} seed{args.seed} "
          f"{len(rows)}/{len(EPOCHS)}", flush=True)


def evaluate(args: argparse.Namespace) -> None:
    source_sim = args.source_sim.resolve()
    out = run_dir(args.output, args.method, args.coupling, args.seed)
    manifest_path = out / "snapshot_manifest.json"
    manifest = read_manifest(manifest_path, source_sim=source_sim)
    sys.path.insert(0, str(source_sim))
    sys.path.insert(0, str(args.tools.resolve()))
    from svgp import run_benchmark as runner  # noqa: E402
    from svgp import engine  # noqa: E402
    from common import terms, predictors  # noqa: E402
    import hard_block_prediction_campaign as hard  # noqa: E402
    import tsvgp_anchor_block_campaign as anchor  # noqa: E402
    import weighted_residual_audit as audit  # noqa: E402
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("evaluation requires exactly one assigned GPU")
    training_path = out / "metrics.json"
    if sha(training_path) != manifest["training_metrics_sha256"]:
        raise ValueError("training metrics changed after snapshotting")
    training = json.loads(training_path.read_text())
    cfg = training["config"]
    if (cfg["method"] != args.method or cfg["seed"] != args.seed
            or cfg["code_commit"] != manifest["source_commit"]):
        raise ValueError("training identity mismatch")
    raw = runner.dm.raw_cache_path("pol", runner.DATASET_DIR)
    if runner.sha256_file(raw) != cfg["data_signature"]:
        raise ValueError("Pol dataset signature mismatch")
    X, y = runner.dm.load_raw("pol", data_dir=runner.DATASET_DIR)
    x, _, xs, ys, meta = runner.jd.split_standardize(
        X, y, cfg["split_seed"], cfg["n_train"], cfg["n_test"],
        angle_cols=runner.dm.angle_cols_for("pol"), split_rng=cfg["split_rng"])
    for key in ("train_indices_sha256", "test_indices_sha256",
                "preprocessing_protocol"):
        if meta[key] != cfg[key]:
            raise ValueError(f"data split changed: {key}")
    for entry in manifest["snapshots"]:
        epoch = entry["epoch"]
        target = out / "block_diag" / f"epoch{epoch:03d}.json"
        identity = {"implementation": IMPLEMENTATION,
                    "snapshot_manifest_sha256": sha(manifest_path),
                    "snapshotter_sha256": sha(Path(__file__)),
                    "hard_evaluator_sha256": sha(Path(hard.__file__)),
                    "source_commit": manifest["source_commit"],
                    "source_config_id": training["config_id"],
                    "epoch": epoch, "params_sha256": entry["params_sha256"],
                    "blocks_sha256": entry["blocks_sha256"]}
        if target.is_file():
            saved = json.loads(target.read_text())
            if saved.get("identity") != identity:
                raise ValueError(f"prediction identity mismatch: {target}")
            continue
        folder = out / "snapshots" / f"epoch{epoch:03d}"
        with np.load(folder / "params.npz") as archive:
            params = {key: jnp.asarray(archive[key]) for key in archive.files}
        components = engine._predictive_components(
            params, x, xs, "rbf", runner.JITTER)
        if args.method == "SVGP":
            correction = np.zeros(len(ys), dtype=np.float64)
            details = {"partition_source": "none", "assigned_counts": []}
        else:
            if args.method == "T-SVGP":
                blocks = anchor.anchor_blocks(np.asarray(x), params, 500, "rbf")
                block_source = "prediction-time-inducing-anchor"
            else:
                with np.load(folder / "blocks.npz") as archive:
                    blocks = engine.blocks_from_bank(archive["idx"], archive["mask"])
                block_source = "fitted-blocks-at-epoch"
            blocks = [np.asarray(block, dtype=np.int32) for block in blocks]
            terms.assert_partition(blocks, len(x))
            reps = hard.block_representatives(components, blocks, params,
                                              kernel_name="rbf")
            assignment = hard.assign_test_points(components, reps, test_chunk=512)
            correction, timing = hard.hard_correction(
                components, blocks, args.method, assignment,
                test_chunk=512, eig_cutoff_scale=10.0)
            details = {"partition_source": block_source,
                       "partition_sha256": anchor.partition_digest(blocks),
                       "assigned_counts": np.bincount(
                           assignment, minlength=len(blocks)).tolist(),
                       "timing": timing}
        dtest = np.asarray(jax.device_get(components["d_test"])).reshape(-1)
        qvar = np.asarray(jax.device_get(components["qvar"])).reshape(-1)
        pred_mean = np.asarray(jax.device_get(components["mean"])).reshape(-1)
        s2 = float(jax.device_get(components["s2"]))
        residual = dtest - correction
        variance = qvar + residual + s2
        if (not np.isfinite(variance).all() or np.min(variance) <= 0
                or np.min(residual) < -audit.NEGATIVE_TOLERANCE):
            raise ValueError(f"invalid block-diagonal variance at epoch {epoch}")
        rmse, nll = predictors.gaussian_rmse_nll(pred_mean, variance, ys)
        atomic_json(target, {"identity": identity,
                             "method": args.method, "coupling": args.coupling,
                             "seed": args.seed, "epoch": epoch,
                             "rmse": float(rmse), "nll": float(nll),
                             "minimum_residual_variance": float(np.min(residual)),
                             "minimum_observation_variance": float(np.min(variance)),
                             "details": details})
        print(f"PREDICTED {args.method} {args.coupling} seed{args.seed} "
              f"epoch={epoch} rmse={rmse:.6g} nll={nll:.6g}", flush=True)
    print(f"EVALUATED {args.method} {args.coupling} seed{args.seed} "
          f"{len(EPOCHS)}/{len(EPOCHS)}", flush=True)


def worker(args: argparse.Namespace) -> None:
    method, coupling = ARMS[args.worker]
    failures = []
    for seed in range(10):
        base = ["--source-sim", str(args.source_sim),
                "--output", str(args.output), "--tools", str(args.tools),
                "--method", method, "--coupling", coupling,
                "--seed", str(seed)]
        for phase in ("train", "evaluate"):
            code = subprocess.call([sys.executable, "-u", str(Path(__file__)),
                                    phase, *base])
            if code:
                failures.append({"seed": seed, "phase": phase, "exit": code})
                break
        atomic_json(args.output / "state" / f"worker{args.worker}.json",
                    {"method": method, "coupling": coupling,
                     "processed_seeds": seed + 1, "failures": failures})
    if failures:
        raise SystemExit(f"worker {args.worker}: {len(failures)} failed seeds")


def certify(args: argparse.Namespace) -> None:
    source_sim = args.source_sim.resolve()
    commit = source_commit(source_sim)
    by_curve = {}
    files = {}
    for method, coupling in ARMS:
        curve = method + ("-same-batch" if coupling == "coupled" else "")
        for seed in range(10):
            out = run_dir(args.output, method, coupling, seed)
            manifest = read_manifest(out / "snapshot_manifest.json",
                                     source_sim=source_sim)
            history = out / "history.csv"
            if sha(history) != manifest["history_sha256"]:
                raise ValueError(f"history changed: {history}")
            with history.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if [int(row["epoch"]) for row in rows] != list(EPOCHS):
                raise ValueError(f"history grid changed: {history}")
            for row in rows:
                epoch = int(row["epoch"])
                predicted = out / "block_diag" / f"epoch{epoch:03d}.json"
                payload = json.loads(predicted.read_text())
                if (payload["identity"]["snapshot_manifest_sha256"]
                        != sha(out / "snapshot_manifest.json")
                        or payload["identity"]["hard_evaluator_sha256"]
                        != sha(args.tools / "hard_block_prediction_campaign.py")
                        or payload["identity"]["source_commit"] != commit):
                    raise ValueError(f"prediction provenance mismatch: {predicted}")
                old_rmse = float(row["rmse"])
                if abs(old_rmse - float(payload["rmse"])) > 1e-7:
                    raise ValueError(f"predictive mean changed: {predicted}")
                values = {
                    "objective": float(row["stochastic_objective_estimate_per_datum"]),
                    "rmse": float(payload["rmse"]), "nll": float(payload["nll"]),
                }
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError(f"nonfinite metric: {predicted}")
                for metric, value in values.items():
                    by_curve.setdefault((curve, metric, epoch), []).append(value)
                files[str(predicted.relative_to(args.output))] = sha(predicted)
    if len(files) != 900 or len(by_curve) != 5 * 3 * len(EPOCHS):
        raise ValueError("incomplete ablation grid")
    rows = []
    for (curve, metric, epoch), values in sorted(by_curve.items()):
        if len(values) != 10:
            raise ValueError(f"missing seed: {curve} {metric} {epoch}")
        rows.append({"curve": curve, "metric": metric, "epoch": epoch,
                     "n_seeds": 10, "mean": mean(values),
                     "sem": stdev(values) / math.sqrt(10)})
    aggregate = args.output / "epochs_rbf_pol_M128_block_diag_summary.csv"
    with aggregate.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {"implementation": IMPLEMENTATION, "source_commit": commit,
              "snapshotter_sha256": sha(Path(__file__)),
              "hard_evaluator_sha256": sha(
                  args.tools / "hard_block_prediction_campaign.py"),
              "dataset": "pol", "kernel": "rbf", "M": 128,
              "seeds_per_arm": 10, "arms": list(ARMS), "epochs": list(EPOCHS),
              "prediction_files": files,
              "aggregate_sha256": sha(aggregate), "prediction_count": len(files),
              "training_and_prediction_separate": True,
              "frozen_campaign_unchanged": True}
    atomic_json(args.output / "BLOCK_DIAG_ABLATION_CERTIFICATE.json", record)
    print(f"CERTIFIED=900/900 ARMS=5 SEEDS=10 EPOCHS={len(EPOCHS)} "
          f"AGGREGATE={aggregate}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "worker", "certify"))
    parser.add_argument("--source-sim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tools", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--method", choices=tuple(sorted({m for m, _ in ARMS})))
    parser.add_argument("--coupling", choices=("split", "coupled"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command in ("train", "evaluate"):
        if (args.method, args.coupling) not in ARMS or args.seed not in range(10):
            parser.error("specify a paper arm and seed 0..9")
    if args.command == "worker" and args.worker not in range(5):
        parser.error("worker must be 0..4")
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "worker":
        worker(args)
    else:
        certify(args)


if __name__ == "__main__":
    main()
