#!/usr/bin/env python3
"""Read-only audit of clipped predictors and a coherent weighted-block proxy.

This script reads frozen checkpoints from a commit-pinned checkout and writes
only to a separate output root.  The proxy replaces the *joint* residual
kernel, not just its train/train matrix:

    D_tilde(x, x') = D(x, x') sum_r sqrt(w_r(x) w_r(x')).

Training weights are one-hot in the frozen partition.  Test weights are
inverse-power distances to the nearest training input in each block, using
the fitted kernel metric.  At distinct training inputs the continuous limit
of these weights is also one-hot.  Hence the training residual is precisely
block diagonal and the finite joint covariance is positive semidefinite.

This is a prediction-time surrogate, not the original variational predictor.
The frozen numerical code and its results are never modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import real_predictor_campaign as real
import tsvgp_anchor_block_campaign as anchor


IMPLEMENTATION = "weighted-block-residual-audit-v2"
METHODS = {"T-SGPR", "T-SVGP", "RBT-SGPR", "SBT-SGPR",
           "METIS-SGPR", "RBT-SVGP", "SBT-SVGP"}
NEGATIVE_TOLERANCE = 1e-7


def inverse_distance_weights(distances: np.ndarray, power: float) -> np.ndarray:
    """Return [test, block] partition-of-unity weights (including zero limits)."""
    h = np.asarray(distances, dtype=np.float64)
    if h.ndim != 2 or not h.size or not np.isfinite(h).all() or power <= 0:
        raise ValueError("expected finite nonempty [test, block] distances")
    h = np.maximum(h, 0.0)
    zero = h <= 1e-12
    has_zero = zero.any(axis=1)
    weights = np.zeros_like(h)
    if has_zero.any():
        weights[has_zero] = (zero[has_zero]
                             / zero[has_zero].sum(axis=1, keepdims=True))
    if (~has_zero).any():
        logits = -power * np.log(h[~has_zero])
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        weights[~has_zero] = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return weights


def cross_block_duplicate_groups(x: np.ndarray,
                                 blocks: list[np.ndarray]) -> int:
    """Count identical training inputs assigned to different frozen blocks."""
    labels = np.empty(len(x), dtype=np.int32)
    for block_id, block in enumerate(blocks):
        labels[block] = block_id
    _, inverse = np.unique(x, axis=0, return_inverse=True)
    smallest = np.full(int(inverse.max()) + 1, len(blocks), dtype=np.int32)
    largest = np.full(int(inverse.max()) + 1, -1, dtype=np.int32)
    np.minimum.at(smallest, inverse, labels)
    np.maximum.at(largest, inverse, labels)
    return int(np.count_nonzero(smallest != largest))


def pilot_indices(runs: list[dict]) -> list[int]:
    targets = {
        ("sgpr", "rbf", 512, "T-SGPR", 0),
        ("sgpr", "rbf", 512, "RBT-SGPR", 0),
        ("sgpr", "rbf", 512, "SBT-SGPR", 0),
        ("svgp", "rbf", 128, "T-SVGP", 0),
        ("svgp", "rbf", 128, "RBT-SVGP", 0),
        ("svgp", "rbf", 128, "SBT-SVGP", 0),
        ("svgp", "amplitude-rbf", 128, "T-SVGP", 0),
        ("svgp", "amplitude-rbf", 128, "RBT-SVGP", 0),
        ("svgp", "amplitude-rbf", 128, "SBT-SVGP", 0),
    }
    found = [i for i, row in enumerate(runs)
             if (row["family"], row["kernel"], row["M"], row["method"],
                 row["seed"]) in targets]
    if len(found) != len(targets):
        raise ValueError(f"pilot is incomplete: found {len(found)}/{len(targets)}")
    return found


def selected_indices(plan: dict, scope: str) -> list[int]:
    pilots = set(pilot_indices(plan["runs"]))
    if scope == "pilot":
        return sorted(pilots)
    if scope == "rest":
        return [i for i in range(len(plan["runs"])) if i not in pilots]
    return list(range(len(plan["runs"])))


def make_plan(args: argparse.Namespace) -> int:
    source = args.source_plan.resolve()
    output = args.output.resolve()
    parent = json.loads(source.read_text())
    runs = [row for row in parent["runs"] if row["method"] in METHODS]
    if len(runs) != 280:
        raise ValueError(f"expected 40 SGPR + 240 SVGP main runs, got {len(runs)}")
    roots = [Path(row["root"]) for row in runs]
    checkouts = [Path(row["source_sim"]).parent for row in runs]
    real.guard_separate(output, roots + checkouts)
    for family, expected_key in (("sgpr", "sgpr_expected_head"),
                                 ("svgp", "svgp_expected_head")):
        row = next(item for item in runs if item["family"] == family)
        expected = parent[expected_key]
        if (Path(row["root"]) / "EXPECTED_HEAD").read_text().strip() != expected:
            raise ValueError(f"{family} frozen root changed")
        checkout_head = subprocess.check_output(
            ["git", "-C", str(Path(row["source_sim"]).parent), "rev-parse", "HEAD"],
            text=True).strip()
        if checkout_head != expected:
            raise ValueError(f"{family} source checkout changed")
        dirty = subprocess.check_output(
            ["git", "-C", str(Path(row["source_sim"]).parent), "status",
             "--porcelain", "--untracked-files=no", "--", "Simulations"],
            text=True).strip()
        if dirty:
            raise ValueError(f"{family} source checkout has modified tracked code")
    payload = {
        "implementation": IMPLEMENTATION,
        "evaluator_sha256": real.sha256(Path(__file__)),
        "real_loader_sha256": real.sha256(Path(real.__file__)),
        "anchor_loader_sha256": real.sha256(Path(anchor.__file__)),
        "source_plan_sha256": real.sha256(source),
        "expected_heads": {k: parent[k] for k in
                           ("sgpr_expected_head", "svgp_expected_head")},
        "block_size_for_T": args.block_size,
        "weight_power": args.weight_power,
        "runs": runs,
    }
    path = output / "plan.json"
    if path.is_file():
        if json.loads(path.read_text()) != payload:
            raise ValueError("different audit plan exists at output root")
    else:
        real.atomic_json(path, payload)
    print(f"PLAN={path} RUNS={len(runs)} PILOT={len(pilot_indices(runs))}",
          flush=True)
    return 0


def correction_matrices(components: dict, blocks: list[np.ndarray], *,
                        method: str, test_chunk: int,
                        eig_cutoff_scale: float) -> tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, dict]:
    """Return current correction, block proxy contributions and distances."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    n_test = int(components["xs"].shape[0])
    n_blocks = len(blocks)
    s2 = float(jax.device_get(components["s2"]))
    if s2 <= 0 or not np.isfinite(s2):
        raise ValueError("positive noise variance required")
    d_train = np.asarray(jax.device_get(components["d_train"])).reshape(-1)
    d_test = np.asarray(jax.device_get(components["d_test"])).reshape(-1)
    a_train = components["a_train"]
    a_test = components["a_test"]
    train_kernel_diag = d_train + np.asarray(jax.device_get(
        jnp.sum(a_train * a_train, axis=0)))
    test_kernel_diag = d_test + np.asarray(jax.device_get(
        jnp.sum(a_test * a_test, axis=0)))
    current = np.zeros(n_test, dtype=np.float64)
    contributions = np.empty((n_test, n_blocks), dtype=np.float64)
    distances = np.empty((n_test, n_blocks), dtype=np.float64)
    minimum_eigenvalue = float("inf")
    maximum_discarded_cross = 0.0
    timings = {"factor_seconds": 0.0, "cross_seconds": 0.0}

    for r, block in enumerate(blocks):
        start = time.perf_counter()
        idx = jnp.asarray(block, dtype=jnp.int32)
        xb = components["x"][idx]
        ab = a_train[:, idx]
        if method.startswith("T-"):
            residual = components["kernel_fn"](xb, xb) - ab.T @ ab
            residual = 0.5 * (residual + residual.T)
            lam, vec = jnp.linalg.eigh(residual)
            spectrum = np.asarray(jax.device_get(lam))
            top = max(float(spectrum[-1]), 0.0)
            cutoff = np.finfo(np.float64).eps * len(block) * top * eig_cutoff_scale
            minimum_eigenvalue = min(minimum_eigenvalue, float(spectrum[0]))
            if spectrum[0] < -max(cutoff, 1e-10):
                raise ValueError(f"residual block {r} is not PSD")
            keep = spectrum > cutoff
            inv_sqrt = jnp.asarray(np.where(
                keep, 1.0 / np.sqrt(np.maximum(spectrum, np.finfo(float).tiny)),
                0.0))
            gain = jnp.asarray(d_train[block] / (d_train[block] + s2))
        else:
            residual = components["kernel_fn"](xb, xb) - ab.T @ ab
            residual = 0.5 * (residual + residual.T)
            chol = jnp.linalg.cholesky(residual + (s2 + 1e-8)
                                       * jnp.eye(len(block)))
            jax.block_until_ready(chol)
        timings["factor_seconds"] += time.perf_counter() - start

        start = time.perf_counter()
        for first in range(0, n_test, test_chunk):
            last = min(first + test_chunk, n_test)
            kcross = components["kernel_fn"](components["xs"][first:last], xb)
            cross = kcross - a_test[:, first:last].T @ ab
            local_dist = (test_kernel_diag[first:last, None]
                          + train_kernel_diag[block][None, :]
                          - 2.0 * np.asarray(jax.device_get(kcross)))
            distances[first:last, r] = np.maximum(local_dist, 0.0).min(axis=1)
            if method.startswith("T-"):
                current[first:last] += np.asarray(jax.device_get(jnp.sum(
                    cross * cross / jnp.asarray(d_train[block] + s2)[None, :],
                    axis=1)))
                coeff = vec.T @ cross.T
                whitened = vec @ (inv_sqrt[:, None] * coeff)
                part = jnp.sum(gain[:, None] * whitened * whitened, axis=0)
                discarded = np.asarray(jax.device_get(jnp.sum(
                    jnp.where(jnp.asarray(keep)[:, None], 0.0, coeff * coeff),
                    axis=0)))
                energy = np.asarray(jax.device_get(jnp.sum(coeff * coeff, axis=0)))
                maximum_discarded_cross = max(
                    maximum_discarded_cross,
                    float(np.max(discarded / np.maximum(energy, 1e-30))))
            else:
                solved = solve_triangular(chol, cross.T, lower=True)
                part = jnp.sum(solved * solved, axis=0)
            values = np.asarray(jax.device_get(part), dtype=np.float64)
            contributions[first:last, r] = values
            if not method.startswith("T-"):
                current[first:last] += values
        timings["cross_seconds"] += time.perf_counter() - start
    timings.update({
        "minimum_block_eigenvalue": minimum_eigenvalue if method.startswith("T-") else None,
        "maximum_discarded_cross_energy_fraction": maximum_discarded_cross
        if method.startswith("T-") else None,
    })
    return current, contributions, distances, timings


def evaluate(args: argparse.Namespace) -> int:
    import jax

    output = args.output.resolve()
    plan = json.loads((output / "plan.json").read_text())
    for label, path in (("evaluator_sha256", Path(__file__)),
                        ("real_loader_sha256", Path(real.__file__)),
                        ("anchor_loader_sha256", Path(anchor.__file__))):
        if plan[label] != real.sha256(path):
            raise ValueError(f"{label} differs from planned evaluator")
    row = plan["runs"][args.index]
    family = row["family"]
    frozen = Path(row["root"])
    source_sim = Path(row["source_sim"])
    real.guard_separate(output, [frozen, source_sim.parent])
    device = jax.devices()
    if len(device) != 1 or device[0].platform != "gpu":
        raise ValueError(f"one assigned GPU required, found {device}")
    run = frozen / row["relative"]
    record_path = run / "metrics.json"
    record = json.loads(record_path.read_text())
    expected = plan["expected_heads"][f"{family}_expected_head"]
    if record["config"]["code_commit"] != expected:
        raise ValueError("checkpoint code commit differs from frozen root")
    input_paths = [record_path, run / "params.npz", frozen / "EXPECTED_HEAD"]
    if row["method"].startswith(("RBT-", "SBT-", "METIS-")):
        input_paths.append(run / "final_blocks.npz")
    input_hashes = {str(path.relative_to(frozen)): real.sha256(path)
                    for path in input_paths}
    identity = {
        "implementation": IMPLEMENTATION,
        "evaluator_sha256": plan["evaluator_sha256"],
        "source_commit": expected,
        "config_id": record["config_id"],
        "inputs": input_hashes,
        "weight_power": plan["weight_power"],
        "T_block_size": plan["block_size_for_T"],
        "test_chunk": args.test_chunk,
        "eig_cutoff_scale": args.eig_cutoff_scale,
    }
    destination = output / row["output_relative"] / "audit.json"
    if destination.is_file():
        if json.loads(destination.read_text()).get("identity") == identity:
            print(f"CACHED {row['output_relative']}", flush=True)
            return 0
        raise ValueError(f"different audit exists at {destination}")
    components, frozen_blocks, ys, meta = real.load_components(
        row, run, record, source_sim)
    from common import predictors, terms
    if row["method"].startswith("T-"):
        with np.load(run / "params.npz") as archive:
            import jax.numpy as jnp
            params = {key: jnp.asarray(archive[key]) for key in archive.files}
        blocks = anchor.anchor_blocks(np.asarray(jax.device_get(components["x"])),
                                      params, plan["block_size_for_T"],
                                      row["kernel"])
        partition_type = "final-parameter-inducing-anchor-prediction-only"
    else:
        blocks = [np.asarray(b, dtype=np.int32) for b in frozen_blocks]
        partition_type = "frozen-fitted-blocks"
    terms.assert_partition(blocks, len(components["x"]))
    train_x = np.asarray(jax.device_get(components["x"]))
    duplicate_groups = cross_block_duplicate_groups(train_x, blocks)

    started = time.perf_counter()
    current, parts, distance, timings = correction_matrices(
        components, blocks, method=row["method"], test_chunk=args.test_chunk,
        eig_cutoff_scale=args.eig_cutoff_scale)
    weights = inverse_distance_weights(distance, plan["weight_power"])
    multiple_zeros = int(np.count_nonzero(
        np.sum(distance <= 1e-12, axis=1) > 1))
    weighted = np.sum(weights * parts, axis=1)
    d_test = np.asarray(jax.device_get(components["d_test"])).reshape(-1)
    qvar = np.asarray(jax.device_get(components["qvar"])).reshape(-1)
    s2 = float(jax.device_get(components["s2"]))
    mean = np.asarray(jax.device_get(components["mean"])).reshape(-1)
    current_raw = d_test - current
    weighted_raw = d_test - weighted
    current_var = qvar + np.maximum(current_raw, 0.0) + s2
    weighted_var = qvar + weighted_raw + s2
    current_rmse, current_nll = predictors.gaussian_rmse_nll(mean, current_var, ys)
    finite_joint_valid = (np.isfinite(weighted_var).all()
                          and np.min(weighted_var) > 0
                          and np.min(weighted_raw) >= -NEGATIVE_TOLERANCE
                          and np.max(np.abs(weights.sum(axis=1) - 1.0)) < 1e-10)
    input_kernel_compatible = (finite_joint_valid and multiple_zeros == 0
                               and duplicate_groups == 0)
    weighted_rmse = weighted_nll = None
    if finite_joint_valid:
        # Tiny roundoff is never used to excuse a material negative residual.
        weighted_var = qvar + np.maximum(weighted_raw, 0.0) + s2
        weighted_rmse, weighted_nll = predictors.gaussian_rmse_nll(
            mean, weighted_var, ys)
    for path in input_paths:
        if real.sha256(path) != input_hashes[str(path.relative_to(frozen))]:
            raise ValueError(f"frozen input changed during evaluation: {path}")
    clipped = current_raw < -NEGATIVE_TOLERANCE
    report = {
        "identity": identity,
        "family": family,
        "kernel": row["kernel"], "method": row["method"],
        "M": row["M"], "seed": row["seed"],
        "partition": {"type": partition_type,
                      "sha256": anchor.partition_digest(blocks),
                      "sizes": [len(b) for b in blocks],
                      "cross_block_duplicate_training_groups":
                      duplicate_groups},
        "weight_definition": "normalized inverse-power nearest-training-point kernel distance per block",
        "weight_power": plan["weight_power"],
        "n_test": int(len(ys)),
        "current": {"variant": "official-diagonal-clipped" if row["method"].startswith("T-")
                    else "official-all-block-clipped",
                    "nll_recomputed": float(current_nll),
                    "rmse_recomputed": float(current_rmse),
                    "reported_nll": record.get("nll"),
                    "reproduction_abs_nll_error": None if record.get("nll") is None
                    else abs(float(current_nll) - float(record["nll"])),
                    "clip_count": int(np.count_nonzero(clipped)),
                    "negative_raw_residual_count": int(np.count_nonzero(current_raw < 0)),
                    "min_raw_residual": float(np.min(current_raw)),
                    "max_clipped_residual_magnitude": float(np.max(
                        np.maximum(-current_raw, 0.0))),
                    "mean_clipped_residual_magnitude": float(np.mean(
                        np.maximum(-current_raw, 0.0)))},
        "weighted": {"finite_joint_valid": bool(finite_joint_valid),
                     "input_kernel_compatible": bool(input_kernel_compatible),
                     "nll": None if weighted_nll is None else float(weighted_nll),
                     "rmse": None if weighted_rmse is None else float(weighted_rmse),
                     "min_raw_residual": float(np.min(weighted_raw)),
                     "negative_raw_residual_count": int(np.count_nonzero(
                         weighted_raw < -NEGATIVE_TOLERANCE)),
                     "max_weight_sum_error": float(np.max(np.abs(
                         weights.sum(axis=1) - 1.0))),
                     "multiple_zero_distance_test_points": multiple_zeros,
                     "effective_blocks_mean": float(np.mean(
                         1.0 / np.sum(weights * weights, axis=1)))},
        "timing": {**timings, "total_seconds": time.perf_counter()-started},
        "device_kind": device[0].device_kind,
        "interpretation": (
            "PSD finite indexed Gaussian proxy; it represents a kernel on input "
            "locations compatible with the frozen block training covariance "
            "only if input_kernel_compatible is true; frozen variational "
            "parameters are unchanged"),
    }
    real.atomic_json(destination, report)
    print(f"SAVED {row['output_relative']} clipped={report['current']['clip_count']}/"
          f"{report['n_test']} finite_joint_valid={finite_joint_valid} "
          f"input_kernel_compatible={input_kernel_compatible} "
          f"delta_NLL={None if weighted_nll is None else weighted_nll-current_nll}",
          flush=True)
    return 0 if finite_joint_valid else 2


def worker(args: argparse.Namespace) -> int:
    plan = json.loads((args.output / "plan.json").read_text())
    indices = [index for position, index in
               enumerate(selected_indices(plan, args.scope))
               if position % args.workers == args.worker]
    failed = []
    for position, index in enumerate(indices, 1):
        print(f"[{position}/{len(indices)}] {plan['runs'][index]['output_relative']}",
              flush=True)
        result = subprocess.run([
            sys.executable, "-u", str(Path(__file__).resolve()), "evaluate",
            "--output", str(args.output), "--index", str(index),
            "--test-chunk", str(args.test_chunk),
            "--eig-cutoff-scale", str(args.eig_cutoff_scale),
        ], check=False)
        if result.returncode:
            failed.append({"index": index, "exit_code": result.returncode})
        real.atomic_json(args.output / "state" /
                         f"{args.scope}_worker{args.worker}.json", {
                             "assigned": len(indices), "processed": position,
                             "failed": failed, "complete": position == len(indices)})
    return 1 if failed else 0


def audit(args: argparse.Namespace) -> int:
    plan = json.loads((args.output / "plan.json").read_text())
    selected = selected_indices(plan, args.scope)
    grouped = {}
    saved = invalid = incompatible = mismatched = 0
    for index in selected:
        row = plan["runs"][index]
        path = args.output / row["output_relative"] / "audit.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        saved += 1
        invalid += not result["weighted"]["finite_joint_valid"]
        incompatible += not result["weighted"]["input_kernel_compatible"]
        mismatch = result["current"]["reproduction_abs_nll_error"]
        mismatched += mismatch is not None and mismatch > 1e-5
        bucket = grouped.setdefault((row["family"], row["method"]),
                                    {"runs": 0, "points": 0, "clipped": 0,
                                     "max_clip": 0.0, "nll_delta": []})
        bucket["runs"] += 1
        bucket["points"] += result["n_test"]
        bucket["clipped"] += result["current"]["clip_count"]
        bucket["max_clip"] = max(bucket["max_clip"],
                                 result["current"]["max_clipped_residual_magnitude"])
        if result["weighted"]["finite_joint_valid"]:
            bucket["nll_delta"].append(result["weighted"]["nll"]
                                       - result["current"]["nll_recomputed"])
    print(f"SCOPE={args.scope} SAVED={saved}/{len(selected)} "
          f"FINITE_JOINT_INVALID={invalid} "
          f"INPUT_KERNEL_INCOMPATIBLE={incompatible} "
          f"CURRENT_NLL_MISMATCH={mismatched}")
    for key, bucket in sorted(grouped.items()):
        deltas = bucket["nll_delta"]
        print(f"{key[0]} {key[1]} runs={bucket['runs']} "
              f"clip={bucket['clipped']}/{bucket['points']} "
              f"max_clip={bucket['max_clip']:.6g} "
              f"mean_delta_NLL={np.mean(deltas) if deltas else float('nan'):+.6g}")
    return 0 if saved == len(selected) and invalid == 0 and mismatched == 0 else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--source-plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--block-size", type=int, default=500)
    p.add_argument("--weight-power", type=float, default=2.0)
    e = sub.add_parser("evaluate")
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--index", type=int, required=True)
    w = sub.add_parser("worker")
    w.add_argument("--output", type=Path, required=True)
    w.add_argument("--scope", choices=("pilot", "rest", "all"), required=True)
    w.add_argument("--worker", type=int, required=True)
    w.add_argument("--workers", type=int, default=5)
    a = sub.add_parser("audit")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--scope", choices=("pilot", "rest", "all"), required=True)
    for command in (e, w):
        command.add_argument("--test-chunk", type=int, default=128)
        command.add_argument("--eig-cutoff-scale", type=float, default=10.0)
    args = parser.parse_args()
    if args.command == "plan":
        return make_plan(args)
    if args.command == "evaluate":
        return evaluate(args)
    if args.command == "worker":
        return worker(args)
    return audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
