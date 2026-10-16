#!/usr/bin/env python3
"""Read-only, PSD-safe T-SVGP anchor-block predictive proxy.

The diagonal scaling M_T is retained, while the symmetric inverse square
root of D_ff is approximated by the inverse square root of the one anchor
block assigned to each test point.  A kernel-distance shortlist of anchor
blocks is formed from their medoids; assignment maximizes the residual
projection score within that shortlist, without test labels.  Keeping test residual correlations
only within assigned blocks defines a positive-semidefinite block residual
kernel, unlike summing corrections over all blocks with the original
train-test cross-covariance.  This remains a proxy, not the exact predictive
covariance of the frozen variational process.  Raw variances are audited.

No frozen input is written.  One fresh process evaluates each checkpoint so
its pinned source checkout cannot be mixed with another Python module cache.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import real_predictor_campaign as real


IMPLEMENTATION = "tsvgp-anchor-block-shortlist-psd-proxy-v4"
KERNEL_STAGE = {
    "rbf": "stationary",
    "matern52": "stationary",
    "amplitude-rbf": "extended",
    "gibbs": "extended",
}
DATASETS = (
    "pol", "superconductivity", "kin40k", "protein",
    "elevators", "traffic", "electricity", "3droad",
)
M_VALUES = (64, 128, 256, 512)
SEEDS = tuple(range(10))
EXPECTED_RUNS = len(KERNEL_STAGE) * len(DATASETS) * len(M_VALUES) * len(SEEDS)
RAW_VARIANCE_TOLERANCE = 1e-7


def sha256(path: Path) -> str:
    return real.sha256(path)


def atomic_json(path: Path, payload: dict) -> None:
    real.atomic_json(path, payload)


def partition_digest(blocks: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for block in blocks:
        values = np.asarray(block, dtype="<i4").reshape(-1)
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values.tobytes())
    return "sha256:" + digest.hexdigest()


def build_plan(args) -> None:
    frozen = args.frozen_root.resolve()
    source = args.source_root.resolve()
    output = args.output.resolve()
    real.guard_separate(output, [frozen, source])
    expected = (frozen / "EXPECTED_HEAD").read_text().strip()
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise ValueError(f"source checkout {actual} != frozen commit {expected}")
    if not (source / "Simulations/svgp/engine.py").is_file():
        raise FileNotFoundError("pinned SVGP source is missing")
    runs = []
    for kernel, stage in KERNEL_STAGE.items():
        for dataset in DATASETS:
            for m in M_VALUES:
                for seed in SEEDS:
                    relative = (Path("svgp") / stage / "runs" / kernel / dataset
                                / f"M{m}" / "T-SVGP" / f"seed{seed}")
                    run = frozen / relative
                    record = (run / "training.json" if (run / "training.json").is_file()
                              else run / "metrics.json")
                    if not (record.is_file() and (run / "params.npz").is_file()):
                        raise FileNotFoundError(f"incomplete frozen checkpoint: {run}")
                    metadata = json.loads(record.read_text())
                    if metadata["config"]["code_commit"] != expected:
                        raise ValueError(f"checkpoint commit differs: {record}")
                    if metadata.get("training_complete") is not True and "rmse" not in metadata:
                        raise ValueError(f"incomplete training: {record}")
                    runs.append({
                        "family": "svgp", "root": str(frozen),
                        "source_sim": str(source / "Simulations"),
                        "relative": str(relative),
                        "output_relative": str(Path("svgp") / kernel / dataset
                                               / f"M{m}" / "T-SVGP" / f"seed{seed}"),
                        "kernel": kernel, "dataset": dataset,
                        "M": m, "seed": seed, "method": "T-SVGP",
                    })
    if len(runs) != EXPECTED_RUNS:
        raise ValueError(f"expected {EXPECTED_RUNS} runs, got {len(runs)}")
    plan = {
        "implementation": IMPLEMENTATION,
        "evaluator_sha256": sha256(Path(__file__)),
        "loader_sha256": sha256(Path(real.__file__)),
        "source_commit": expected,
        "frozen_root": str(frozen),
        "block_size": args.block_size,
        "runs": runs,
    }
    destination = output / "plan.json"
    if destination.exists():
        if json.loads(destination.read_text()) != plan:
            raise ValueError("different plan exists; choose a new output root")
    else:
        atomic_json(destination, plan)
    print(f"PLAN={destination} RUNS={len(runs)} POL={sum(r['dataset']=='pol' for r in runs)} "
          f"BLOCK={args.block_size} SOURCE={expected}", flush=True)


def anchor_blocks(x, params, block_size: int, kernel: str) -> list[np.ndarray]:
    from common import partitions, terms

    packed = partitions.sbt_terms_device(x, params, block_size, kernel)
    blocks = [np.asarray(block, dtype=np.int32) for block in terms.unpack_terms(packed)]
    terms.assert_partition(blocks, len(x))
    if max(map(len, blocks)) > block_size:
        raise ValueError("anchor partition exceeded its block budget")
    return blocks


def test_candidates(components: dict, blocks: list[np.ndarray], params: dict,
                    *, test_chunk: int, candidate_blocks: int) -> np.ndarray:
    """Shortlist anchor blocks by frozen-kernel distance to their medoids."""
    import jax
    import jax.numpy as jnp

    x = np.asarray(jax.device_get(components["x"]), dtype=np.float64)
    ell = np.asarray(jax.device_get(jnp.exp(params["log_ell"])), dtype=np.float64)
    ell = np.broadcast_to(ell, (x.shape[1],))
    medoids = []
    for block in blocks:
        cell = x[block]
        center = np.mean(cell, axis=0)
        offset = (cell - center) / ell
        medoids.append(int(block[np.argmin(np.sum(offset * offset, axis=1))]))
    medoids = np.asarray(medoids, dtype=np.int32)
    reps = components["x"][jnp.asarray(medoids)]
    rep_diag = (components["d_train"][jnp.asarray(medoids)]
                + jnp.sum(components["a_train"][:, jnp.asarray(medoids)] ** 2,
                          axis=0))
    width = min(candidate_blocks, len(blocks))
    chosen = np.empty((len(components["xs"]), width), dtype=np.int32)
    for start in range(0, len(chosen), test_chunk):
        stop = min(start + test_chunk, len(chosen))
        xs = components["xs"][start:stop]
        diag = (components["d_test"][start:stop]
                + jnp.sum(components["a_test"][:, start:stop] ** 2, axis=0))
        distances = np.asarray(jax.device_get(jnp.maximum(
            diag[:, None] + rep_diag[None, :]
            - 2.0 * components["kernel_fn"](xs, reps), 0.0)), dtype=np.float64)
        shortlisted = np.argpartition(
            distances, kth=width - 1, axis=1)[:, :width]
        order = np.argsort(
            np.take_along_axis(distances, shortlisted, axis=1),
            axis=1, kind="stable")
        chosen[start:stop] = np.take_along_axis(shortlisted, order, axis=1)
    return chosen


def evaluate_proxy(components: dict, blocks: list[np.ndarray], params: dict, *,
                   test_chunk: int, eig_cutoff_scale: float,
                   candidate_blocks: int) -> tuple[np.ndarray, dict]:
    """Block-root T correction, with raw latent-variance validity diagnostics."""
    import jax
    import jax.numpy as jnp

    if test_chunk < 1 or eig_cutoff_scale <= 0 or candidate_blocks < 1:
        raise ValueError("invalid numerical policy")
    n_test = int(components["xs"].shape[0])
    s2 = float(jax.device_get(components["s2"]))
    if not np.isfinite(s2) or s2 <= 0:
        raise ValueError("noise variance must be positive")
    base = np.asarray(jax.device_get(
        components["qvar"] + components["d_test"]), dtype=np.float64).reshape(-1)
    gain = np.asarray(jax.device_get(
        components["d_train"] / (components["d_train"] + s2)),
        dtype=np.float64).reshape(-1)
    begin = time.perf_counter()
    candidates = test_candidates(
        components, blocks, params, test_chunk=test_chunk,
        candidate_blocks=candidate_blocks)
    assignment_seconds = time.perf_counter() - begin
    target_lists = [[] for _ in blocks]
    for test_id, row in enumerate(candidates):
        for block_id in row:
            target_lists[int(block_id)].append(test_id)
    correction = np.zeros(n_test, dtype=np.float64)
    best_score = np.full(n_test, -np.inf, dtype=np.float64)
    assigned_block = np.full(n_test, -1, dtype=np.int32)
    smallest_eig = float("inf")
    discarded_energy = 0.0
    effective_rank = 0
    factored_blocks = 0
    factor_seconds = 0.0
    cross_seconds = 0.0
    for block_id, block in enumerate(blocks):
        if not target_lists[block_id]:
            continue
        factored_blocks += 1
        begin = time.perf_counter()
        idx = jnp.asarray(block)
        xb = components["x"][idx]
        ab = components["a_train"][:, idx]
        db = components["kernel_fn"](xb, xb) - ab.T @ ab
        db = 0.5 * (db + db.T)
        lam, vec = jnp.linalg.eigh(db)
        spectrum = np.asarray(jax.device_get(lam), dtype=np.float64)
        top = max(float(spectrum[-1]), 0.0)
        cutoff = float(np.finfo(np.float64).eps * len(block) * top
                       * eig_cutoff_scale)
        if spectrum[0] < -max(cutoff, 1e-10):
            raise ValueError(f"residual block not PSD: min={spectrum[0]:.6g}")
        keep = spectrum > cutoff
        effective_rank += int(np.count_nonzero(keep))
        smallest_eig = min(smallest_eig, float(spectrum[0]))
        inv_sqrt = jnp.asarray(np.where(
            keep, 1.0 / np.sqrt(np.maximum(spectrum, np.finfo(float).tiny)), 0.0
        ))
        block_gain = jnp.asarray(gain[block])
        jax.block_until_ready(vec)
        factor_seconds += time.perf_counter() - begin
        begin = time.perf_counter()
        targets = np.asarray(target_lists[block_id], dtype=np.int32)
        for start in range(0, len(targets), test_chunk):
            ids = targets[start:start + test_chunk]
            # Fixed shapes avoid one XLA compilation for every distinct
            # anchor block's final chunk size.
            padded_ids = np.resize(ids, test_chunk)
            test_ids = jnp.asarray(padded_ids)
            cross = (components["kernel_fn"](components["xs"][test_ids], xb)
                     - components["a_test"][:, test_ids].T @ ab)
            coeff = vec.T @ cross.T
            white = vec @ (inv_sqrt[:, None] * coeff)
            part = np.asarray(jax.device_get(
                jnp.sum(block_gain[:, None] * white * white, axis=0)
            ), dtype=np.float64)[:len(ids)]
            score = np.asarray(jax.device_get(
                jnp.sum(white * white, axis=0)
            ), dtype=np.float64)[:len(ids)]
            improved = score > best_score[ids]
            best_score[ids] = np.where(improved, score, best_score[ids])
            correction[ids] = np.where(improved, part, correction[ids])
            assigned_block[ids] = np.where(
                improved, block_id, assigned_block[ids])
            energy = np.asarray(jax.device_get(jnp.sum(
                jnp.where(jnp.asarray(keep)[:, None], 0.0, coeff * coeff), axis=0
            )), dtype=np.float64)
            total = np.asarray(jax.device_get(jnp.sum(coeff * coeff, axis=0)),
                               dtype=np.float64)
            discarded_energy = max(discarded_energy,
                                   float(np.max(energy / np.maximum(total, 1e-30))))
        cross_seconds += time.perf_counter() - begin
    if np.any(assigned_block < 0):
        raise ValueError("some test points were not assigned to a residual block")
    latent = base - correction
    diagnostics = {
        "min_raw_latent_variance": float(np.min(latent)),
        "negative_raw_latent_count": int(np.count_nonzero(latent < -RAW_VARIANCE_TOLERANCE)),
        "max_discarded_cross_energy_fraction": discarded_energy,
        "min_block_eigenvalue": smallest_eig,
        "effective_block_rank_sum": effective_rank,
        "n_blocks": len(blocks),
        "factored_blocks": factored_blocks,
        "candidate_blocks": int(candidates.shape[1]),
        "assignment_seconds": assignment_seconds,
        "test_block_counts": np.bincount(
            assigned_block, minlength=len(blocks)).tolist(),
        "max_projection_to_test_residual_ratio": float(np.max(
            best_score / np.maximum(np.asarray(
                jax.device_get(components["d_test"]), dtype=np.float64), 1e-30))),
        "factor_seconds": factor_seconds,
        "cross_seconds": cross_seconds,
        "max_correction_to_base_ratio": float(np.max(
            correction / np.maximum(base, 1e-30))),
    }
    return latent + s2, diagnostics


def load_components_all(item: dict, run: Path, record: dict, source_sim: Path):
    """Load any frozen SVGP dataset through the commit-pinned runner."""
    import jax.numpy as jnp

    sys.path.insert(0, str(source_sim))
    from svgp import run_benchmark as runner
    from svgp import engine

    cfg = record["config"]
    dataset = item["dataset"]
    if cfg["dataset"] != dataset or cfg["kernel"] != item["kernel"]:
        raise ValueError("checkpoint dataset or kernel differs from plan")
    if runner.config_fingerprint(cfg) != record["config_id"]:
        raise ValueError("SVGP configuration fingerprint mismatch")
    raw = runner.dm.raw_cache_path(dataset, runner.DATASET_DIR)
    if sha256(raw) != cfg["data_signature"]:
        raise ValueError("SVGP dataset hash mismatch")
    X, y = runner.dm.load_raw(dataset, data_dir=runner.DATASET_DIR)
    x, _, xs, ys, meta = runner.jd.split_standardize(
        X, y, cfg["split_seed"], cfg["n_train"], cfg["n_test"],
        angle_cols=runner.dm.angle_cols_for(dataset), split_rng=cfg["split_rng"])
    for key in ("train_indices_sha256", "test_indices_sha256",
                "preprocessing_protocol"):
        if meta[key] != cfg[key]:
            raise ValueError(f"SVGP split mismatch: {key}")
    with np.load(run / "params.npz") as archive:
        params = {key: jnp.asarray(archive[key]) for key in archive.files}
    components = engine._predictive_components(
        params, x, xs, cfg["kernel"], runner.JITTER)
    return components, params, ys, meta


def evaluate(args) -> int:
    import jax
    plan_path = args.plan.resolve()
    output = args.output.resolve()
    if plan_path.parent != output:
        raise ValueError("plan must live in its separate output root")
    plan = json.loads(plan_path.read_text())
    if plan["evaluator_sha256"] != sha256(Path(__file__)):
        raise ValueError("evaluator changed after plan creation")
    if plan["loader_sha256"] != sha256(Path(real.__file__)):
        raise ValueError("loader changed after plan creation")
    item = plan["runs"][args.index]
    frozen = Path(item["root"])
    source_sim = Path(item["source_sim"])
    real.guard_separate(output, [frozen, source_sim.parent])
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError(f"exactly one assigned GPU required: {devices}")
    run = frozen / item["relative"]
    record_path = run / "training.json"
    if not record_path.is_file():
        record_path = run / "metrics.json"
    params_path = run / "params.npz"
    expected_path = frozen / "EXPECTED_HEAD"
    record = json.loads(record_path.read_text())
    if record["config"]["code_commit"] != plan["source_commit"]:
        raise ValueError("checkpoint commit differs from plan")
    inputs = {str(path.relative_to(frozen)): sha256(path)
              for path in (record_path, params_path, expected_path)}
    identity = {
        "implementation": IMPLEMENTATION,
        "evaluator_sha256": plan["evaluator_sha256"],
        "loader_sha256": plan["loader_sha256"],
        "source_commit": plan["source_commit"],
        "config_id": record["config_id"],
        "inputs": inputs,
        "kernel": item["kernel"], "dataset": item["dataset"],
        "M": item["M"], "seed": item["seed"],
        "block_size": plan["block_size"],
        "test_chunk": args.test_chunk,
        "eig_cutoff_scale": args.eig_cutoff_scale,
        "candidate_blocks": args.candidate_blocks,
    }
    destination = output / item["output_relative"] / "prediction.json"
    if destination.exists():
        if json.loads(destination.read_text()).get("identity") == identity:
            print(f"CACHED {item['output_relative']}", flush=True)
            return 0
        raise ValueError(f"different result exists: {destination}")
    components, params, ys, meta = load_components_all(
        item, run, record, source_sim)
    from common import predictors, terms

    t0 = time.perf_counter()
    blocks = anchor_blocks(np.asarray(jax.device_get(components["x"])), params,
                           plan["block_size"], item["kernel"])
    terms.assert_partition(blocks, len(components["x"]))
    search_seconds = time.perf_counter() - t0
    variance, diagnostics = evaluate_proxy(
        components, blocks, params, test_chunk=args.test_chunk,
        eig_cutoff_scale=args.eig_cutoff_scale,
        candidate_blocks=args.candidate_blocks)
    mean = np.asarray(jax.device_get(components["mean"]), dtype=np.float64).reshape(-1)
    valid = (diagnostics["negative_raw_latent_count"] == 0
             and np.isfinite(variance).all()
             and np.min(variance) > 0)
    if valid:
        # Only microscopic roundoff is clipped; a materially negative latent
        # variance is an invalid proxy, never a reported Gaussian prediction.
        variance = np.maximum(variance, float(jax.device_get(components["s2"])))
        rmse, nll = predictors.gaussian_rmse_nll(mean, variance, ys)
    else:
        rmse, nll = None, None
    for path in (record_path, params_path, expected_path):
        if sha256(path) != inputs[str(path.relative_to(frozen))]:
            raise ValueError("frozen input changed during evaluation")
    result = {
        "identity": identity,
        "partition": {
            "selector": "inducing-anchor-from-frozen-T-SVGP-theta-and-Z",
            "sha256": partition_digest(blocks),
            "sizes": [len(block) for block in blocks],
            "search_seconds": search_seconds,
        },
        "valid": bool(valid), "rmse": None if rmse is None else float(rmse),
        "nll": None if nll is None else float(nll),
        "rmse_orig": None if rmse is None else float(rmse * meta["y_std"]),
        "nll_orig": None if nll is None else float(nll + np.log(meta["y_std"])),
        "diagnostics": diagnostics,
        "device_kind": devices[0].device_kind,
        "interpretation": "shortlisted one-anchor-block PSD proxy, not exact frozen variational predictor",
    }
    atomic_json(destination, result)
    print(f"SAVED {item['output_relative']} valid={valid} nll={nll} "
          f"min_latent={diagnostics['min_raw_latent_variance']:.6g} "
          f"blocks={len(blocks)} search={search_seconds:.2f}s", flush=True)
    return 0 if valid else 2


def selected_indices(plan: dict, scope: str) -> list[int]:
    if scope == "all":
        return list(range(len(plan["runs"])))
    return [i for i, item in enumerate(plan["runs"])
            if (item["dataset"] == "pol"
                and item["kernel"] in ("rbf", "amplitude-rbf"))
            == (scope == "pilot")]


def worker(args) -> int:
    plan_path = args.output / "plan.json"
    plan = json.loads(plan_path.read_text())
    selected = [i for i in selected_indices(plan, args.scope)
                if i % args.workers == args.worker]
    failed = []
    for position, index in enumerate(selected, 1):
        item = plan["runs"][index]
        print(f"[{position}/{len(selected)}] {item['output_relative']}", flush=True)
        outcome = subprocess.run([
            sys.executable, "-u", str(Path(__file__).resolve()), "evaluate",
            "--plan", str(plan_path), "--output", str(args.output),
            "--index", str(index), "--test-chunk", str(args.test_chunk),
            "--eig-cutoff-scale", str(args.eig_cutoff_scale),
            "--candidate-blocks", str(args.candidate_blocks),
        ], check=False)
        if outcome.returncode:
            failed.append({"index": index, "exit_code": outcome.returncode})
        atomic_json(args.output / "state" / f"{args.scope}_worker{args.worker}.json", {
            "scope": args.scope, "worker": args.worker,
            "assigned": len(selected), "processed": position,
            "failed": failed, "complete": position == len(selected),
        })
    return 1 if failed else 0


def audit(args) -> int:
    plan = json.loads((args.output / "plan.json").read_text())
    selected = selected_indices(plan, args.scope)
    complete = invalid = missing = 0
    paired = defaultdict(list)
    for index in selected:
        item = plan["runs"][index]
        path = args.output / item["output_relative"] / "prediction.json"
        if not path.is_file():
            missing += 1
            continue
        row = json.loads(path.read_text())
        complete += 1
        if row["valid"] is not True:
            invalid += 1
        if args.dense_reference and item["dataset"] == "pol" and row["valid"]:
            ref = (args.dense_reference / "svgp" / item["kernel"] / "pol"
                   / f"M{item['M']}" / "T-SVGP" / f"seed{item['seed']}"
                   / "prediction.json")
            if ref.is_file():
                dense = json.loads(ref.read_text())
                if dense["identity"]["config_id"] != row["identity"]["config_id"]:
                    raise ValueError(f"dense comparison checkpoint mismatch: {ref}")
                paired[(item["kernel"], item["M"])].append(
                    float(row["nll"] - dense["nll"]))
    print(f"SCOPE={args.scope} SAVED={complete}/{len(selected)} "
          f"INVALID={invalid} MISSING={missing}")
    for key, differences in sorted(paired.items()):
        print(f"DENSE_COMPARISON kernel={key[0]} M={key[1]} "
              f"paired={len(differences)}/10 mean_delta_NLL={np.mean(differences):+.6g} "
              f"mean_abs_delta={np.mean(np.abs(differences)):.6g} "
              f"max_abs_delta={np.max(np.abs(differences)):.6g}")
    ready = complete == len(selected) and invalid == 0 and missing == 0
    if args.scope == "pilot" and args.dense_reference:
        ready &= len(paired) == 8 and all(len(v) == 10 for v in paired.values())
    print("AUDIT_READY=" + ("yes" if ready else "no"))
    return 0 if ready else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--frozen-root", type=Path, required=True)
    plan_parser.add_argument("--source-root", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    plan_parser.add_argument("--block-size", type=int, default=500)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--plan", type=Path, required=True)
    eval_parser.add_argument("--index", type=int, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    work_parser = sub.add_parser("worker")
    work_parser.add_argument("--output", type=Path, required=True)
    work_parser.add_argument("--worker", type=int, required=True)
    work_parser.add_argument("--workers", type=int, default=5)
    work_parser.add_argument("--scope", choices=("pilot", "rest", "all"), required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--scope", choices=("pilot", "rest", "all"), required=True)
    audit_parser.add_argument("--dense-reference", type=Path)
    for command in (eval_parser, work_parser):
        command.add_argument("--test-chunk", type=int, default=128)
        command.add_argument("--eig-cutoff-scale", type=float, default=10.0)
        command.add_argument("--candidate-blocks", type=int, default=4)
    args = parser.parse_args()
    if args.command == "plan":
        if args.block_size < 1:
            raise ValueError("block size must be positive")
        build_plan(args)
        return 0
    if args.command == "evaluate":
        return evaluate(args)
    if args.command == "worker":
        return worker(args)
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
