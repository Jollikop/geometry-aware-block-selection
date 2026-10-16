#!/usr/bin/env python3
"""Prediction-only PIC/PITC-style hard-block campaign.

Each test point is assigned to one fitted block by kernel-induced distance to
one representative per block.  The residual cross-covariance is retained
only for that block.  This is a finite, PSD block-conditional surrogate:
training/test residuals use the same block gate and no variance clipping is
used.  Frozen checkpoints and training code are read-only.
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
import weighted_residual_audit as audit
import weighted_residual_campaign as campaign


IMPLEMENTATION = "hard-block-pic-prediction-v1"


def prediction_blocks(components, params, row, plan, frozen_blocks, run,
                      partition_source="default", seed_offset=10007):
    """Change only prediction geometry; never change donor parameters."""
    import jax
    import jax.numpy as jnp
    from common import terms

    seed_pred = None
    if partition_source == "anchor" or (
            partition_source == "default" and row["method"].startswith("T-")):
        blocks = anchor.anchor_blocks(np.asarray(jax.device_get(components["x"])),
                                     params, plan["block_size_for_T"], row["kernel"])
        source = "prediction-time-inducing-anchor"
    elif partition_source == "random":
        seed_pred = int(row["seed"]) + int(seed_offset)
        n = len(components["x"])
        order = jax.random.permutation(jax.random.PRNGKey(seed_pred),
                                       jnp.arange(n, dtype=jnp.int32))
        idx, mask, _ = terms.pack_blocks_device(order, n, plan["block_size_for_T"])
        blocks = [np.asarray(i[m > 0.5], dtype=np.int32)
                  for i, m in zip(np.asarray(jax.device_get(idx)),
                                  np.asarray(jax.device_get(mask)))]
        source = "prediction-time-uniform-random"
    elif partition_source == "default":
        blocks = ([np.asarray(b, dtype=np.int32) for b in frozen_blocks]
                  if frozen_blocks is not None else campaign.load_frozen_blocks(
                      run, row["family"]))
        source = "frozen-fitted-blocks"
    else:
        raise ValueError(f"unknown partition source: {partition_source}")
    terms.assert_partition(blocks, len(components["x"]))
    return blocks, source, seed_pred


def load_plan(output: Path) -> dict:
    plan = campaign.load_plan(output)
    expected = plan.get("hard_evaluator_sha256")
    if expected != real.sha256(Path(__file__)):
        raise ValueError("hard prediction evaluator changed after plan creation")
    return plan


def make_plan(args: argparse.Namespace) -> int:
    """Build the frozen source/run manifest and label it as hard prediction."""
    campaign.make_plan(args)
    destination = args.output.resolve() / "plan.json"
    payload = json.loads(destination.read_text())
    payload["implementation"] = "hard-pic-pitc-plan-v1"
    payload["hard_evaluator_sha256"] = real.sha256(Path(__file__))
    real.atomic_json(destination, payload)
    print(f"HARD_PLAN={destination} RUNS={len(payload['runs'])}", flush=True)
    return 0


def block_representatives(components: dict, blocks: list[np.ndarray], params,
                          *, kernel_name: str) -> np.ndarray:
    """Choose one training representative per block without labels."""
    import jax
    import jax.numpy as jnp

    x = np.asarray(jax.device_get(components["x"]), dtype=np.float64)
    ell = np.asarray(jax.device_get(jnp.exp(params["log_ell"])), dtype=float)
    ell = np.broadcast_to(ell.reshape(-1), (x.shape[1],))
    reps = []
    for block in blocks:
        values = x[block]
        center = values.mean(axis=0)
        scaled = (values - center) / np.maximum(ell, 1e-12)
        reps.append(int(block[np.argmin(np.sum(scaled * scaled, axis=1))]))
    return np.asarray(reps, dtype=np.int32)


def assign_test_points(components: dict, representatives: np.ndarray,
                       *, test_chunk: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    reps = jnp.asarray(representatives, dtype=jnp.int32)
    xr = components["x"][reps]
    rr = (components["d_train"][reps]
          + jnp.sum(components["a_train"][:, reps] ** 2, axis=0))
    result = np.empty(int(components["xs"].shape[0]), dtype=np.int32)
    for first in range(0, len(result), test_chunk):
        last = min(first + test_chunk, len(result))
        xs = components["xs"][first:last]
        diag = (components["d_test"][first:last]
                + jnp.sum(components["a_test"][:, first:last] ** 2, axis=0))
        distance = jnp.maximum(
            diag[:, None] + rr[None, :]
            - 2.0 * components["kernel_fn"](xs, xr), 0.0)
        result[first:last] = np.asarray(jax.device_get(
            jnp.argmin(distance, axis=1)), dtype=np.int32)
    return result


def hard_correction(components: dict, blocks: list[np.ndarray], method: str,
                    assignment: np.ndarray, *, test_chunk: int,
                    eig_cutoff_scale: float) -> tuple[np.ndarray, dict]:
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    n_test = int(len(assignment))
    s2 = float(jax.device_get(components["s2"]))
    d_train = np.asarray(jax.device_get(components["d_train"])).reshape(-1)
    correction = np.zeros(n_test, dtype=np.float64)
    factor_s = cross_s = 0.0
    min_eig = float("inf")
    max_discarded = 0.0
    max_cholesky_jitter = 0.0
    for block_id, block in enumerate(blocks):
        targets = np.flatnonzero(assignment == block_id)
        if not len(targets):
            continue
        t0 = time.perf_counter()
        idx = jnp.asarray(block, dtype=jnp.int32)
        xb = components["x"][idx]
        ab = components["a_train"][:, idx]
        residual = components["kernel_fn"](xb, xb) - ab.T @ ab
        residual = 0.5 * (residual + residual.T)
        if method.startswith("T-"):
            lam, vec = jnp.linalg.eigh(residual)
            spectrum = np.asarray(jax.device_get(lam), dtype=float)
            top = max(float(spectrum[-1]), 0.0)
            cutoff = np.finfo(np.float64).eps * len(block) * top * eig_cutoff_scale
            min_eig = min(min_eig, float(spectrum[0]))
            if spectrum[0] < -max(cutoff, 1e-10):
                raise ValueError("residual block is not PSD")
            keep = spectrum > cutoff
            inv_sqrt = jnp.asarray(np.where(
                keep, 1.0 / np.sqrt(np.maximum(spectrum, np.finfo(float).tiny)), 0.0))
            gain = jnp.asarray(d_train[block] / (d_train[block] + s2))
        else:
            matrix = residual + s2 * jnp.eye(len(block))
            chol = jnp.linalg.cholesky(matrix)
            if not np.isfinite(np.asarray(jax.device_get(chol))).all():
                jitter = 1e-8 * max(1.0, s2)
                chol = jnp.linalg.cholesky(matrix + jitter * jnp.eye(len(block)))
                if not np.isfinite(np.asarray(jax.device_get(chol))).all():
                    raise ValueError("residual block plus noise is not positive definite")
                max_cholesky_jitter = max(max_cholesky_jitter, jitter)
        factor_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        for first in range(0, len(targets), test_chunk):
            ids = targets[first:first + test_chunk]
            cross = (components["kernel_fn"](components["xs"][jnp.asarray(ids)], xb)
                     - components["a_test"][:, jnp.asarray(ids)].T @ ab)
            if method.startswith("T-"):
                coeff = vec.T @ cross.T
                white = vec @ (inv_sqrt[:, None] * coeff)
                part = jnp.sum(gain[:, None] * white * white, axis=0)
                discarded = np.asarray(jax.device_get(jnp.sum(
                    jnp.where(jnp.asarray(keep)[:, None], 0.0, coeff * coeff), axis=0)))
                total = np.asarray(jax.device_get(jnp.sum(coeff * coeff, axis=0)))
                max_discarded = max(max_discarded,
                    float(np.max(discarded / np.maximum(total, 1e-30))))
            else:
                solved = solve_triangular(chol, cross.T, lower=True)
                part = jnp.sum(solved * solved, axis=0)
            correction[ids] = np.asarray(jax.device_get(part), dtype=float)
        cross_s += time.perf_counter() - t0
    return correction, dict(factor_seconds=factor_s, cross_seconds=cross_s,
                            minimum_block_eigenvalue=(None if not method.startswith("T-")
                                                      else min_eig),
                            maximum_discarded_cross_energy_fraction=(
                                None if not method.startswith("T-") else max_discarded),
                            maximum_cholesky_jitter=max_cholesky_jitter)


def evaluate(args: argparse.Namespace) -> int:
    import jax
    import jax.numpy as jnp

    output = args.output.resolve()
    plan = load_plan(output)
    row = plan["runs"][args.index]
    frozen = Path(row["root"])
    source_sim = Path(row["source_sim"])
    real.guard_separate(output, [frozen, source_sim.parent])
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("one assigned GPU is required")
    for name, digest in plan["sources"][row["source_kind"]]["files"].items():
        if real.sha256(source_sim / name) != digest:
            raise ValueError(f"historical source changed: {name}")
    run = frozen / row["relative"]
    record_path = campaign.record_path(run)
    record = json.loads(record_path.read_text())
    expected = plan["expected_heads"]["sgpr" if row["family"] == "sgpr" else "svgp"]
    if record["config"]["code_commit"] != expected:
        raise ValueError("checkpoint source commit differs")
    inputs = campaign.verify_run_inputs(plan, row, run, record_path)
    if "frozen_inputs" in row and inputs != row["frozen_inputs"]:
        raise ValueError("frozen inputs differ from the control plan")
    identity = dict(implementation=IMPLEMENTATION,
                    evaluator_sha256=plan["hard_evaluator_sha256"],
                    plan_sha256=real.sha256(output / "plan.json"),
                    source_kind=row["source_kind"],
                    source_commit=expected, config_id=record["config_id"],
                    inputs=inputs, test_chunk=args.test_chunk,
                    eig_cutoff_scale=args.eig_cutoff_scale)
    partition_option = getattr(args, "partition_source", "default")
    predictor_option = getattr(args, "predictor", "hard")
    options = dict(partition_source=partition_option, predictor=predictor_option,
                   seed_pred_offset=getattr(args, "seed_pred_offset", 10007))
    if "control_options" in plan and options != plan["control_options"]:
        raise ValueError("prediction options differ from the frozen control plan")
    if partition_option != "default" or predictor_option != "hard":
        identity.update(partition_source=partition_option, predictor=predictor_option,
                        seed_pred_offset=getattr(args, "seed_pred_offset", 10007))
    destination = output / "hard" / row["output_relative"] / "prediction.json"
    if destination.is_file():
        if json.loads(destination.read_text()).get("identity") == identity:
            print(f"CACHED {row['output_relative']}", flush=True)
            return 0
        raise ValueError(f"different result exists: {destination}")
    prediction_start = time.perf_counter()
    if row["family"] == "sgpr":
        components, frozen_blocks, ys, meta = real.load_components(
            row, run, record, source_sim)
        with np.load(run / "params.npz") as archive:
            params = {key: jnp.asarray(archive[key]) for key in archive.files}
    else:
        components, params, ys, meta = anchor.load_components_all(
            row, run, record, source_sim)
        frozen_blocks = None
    jax.block_until_ready(components)
    components_seconds = time.perf_counter() - prediction_start
    from common import predictors
    t0 = time.perf_counter()
    if predictor_option == "sparse":
        blocks, reps = [], np.array([], dtype=np.int32)
        assignment = np.array([], dtype=np.int32)
        partition_source, seed_pred = "none-direct-sparse-donor", None
        correction = np.zeros(len(components["xs"]), dtype=np.float64)
        timings = dict(factor_seconds=0.0, cross_seconds=0.0,
                       maximum_cholesky_jitter=0.0)
        search_seconds = assignment_seconds = 0.0
    else:
        blocks, partition_source, seed_pred = prediction_blocks(
            components, params, row, plan, frozen_blocks, run, partition_option,
            getattr(args, "seed_pred_offset", 10007))
        search_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        reps = block_representatives(components, blocks, params,
                                     kernel_name=row["kernel"])
        assignment = assign_test_points(components, reps, test_chunk=args.test_chunk)
        assignment_seconds = time.perf_counter() - t0
        correction, timings = hard_correction(
            components, blocks, row["method"], assignment,
            test_chunk=args.test_chunk, eig_cutoff_scale=args.eig_cutoff_scale)
    d_test = np.asarray(jax.device_get(components["d_test"])).reshape(-1)
    qvar = np.asarray(jax.device_get(components["qvar"])).reshape(-1)
    mean = np.asarray(jax.device_get(components["mean"])).reshape(-1)
    s2 = float(jax.device_get(components["s2"]))
    raw = d_test - correction
    variance = qvar + raw + s2
    valid = bool(np.isfinite(variance).all() and np.min(variance) > 0
                 and np.min(raw) >= -audit.NEGATIVE_TOLERANCE)
    rmse = nll = None
    if valid:
        rmse, nll = predictors.gaussian_rmse_nll(mean, variance, ys)
    jax.block_until_ready(components)
    total_prediction_seconds = time.perf_counter() - prediction_start
    for name, digest in inputs.items():
        if real.sha256(frozen / name) != digest:
            raise ValueError(f"frozen input changed: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(identity=identity, family=row["family"], kernel=row["kernel"],
                   dataset=row["dataset"], method=row["method"], M=row["M"],
                   seed=row["seed"], valid=valid,
                   rmse=None if rmse is None else float(rmse),
                   nll=None if nll is None else float(nll),
                   rmse_orig=None if rmse is None else float(rmse * meta["y_std"]),
                   nll_orig=None if nll is None else float(nll + np.log(meta["y_std"])),
                   partition=dict(source=partition_source,
                                  selector="nearest block representative",
                                  representative_indices=reps.tolist(),
                                  assigned_counts=np.bincount(
                                      assignment, minlength=len(blocks)).tolist(),
                                  sha256=anchor.partition_digest(blocks),
                                  sizes=[len(b) for b in blocks], seed_pred=seed_pred),
                   diagnostics=dict(min_raw_residual=float(np.min(raw)),
                                    negative_raw_residual_count=int(np.count_nonzero(
                                        raw < -audit.NEGATIVE_TOLERANCE)),
                                    **timings, components_seconds=components_seconds,
                                    search_seconds=search_seconds,
                                    assignment_seconds=assignment_seconds,
                                    total_prediction_seconds=total_prediction_seconds,
                                    rejection_reason=(None if valid else
                                        "nonfinite/nonpositive variance or negative raw residual")),
                   device_kind=jax.devices()[0].device_kind,
                   interpretation="hard PIC/PITC-style block-conditioned finite Gaussian proxy")
    real.atomic_json(destination, payload)
    print(f"PREDICTED {row['output_relative']} valid={valid} nll={nll}", flush=True)
    return 0 if valid else 2


def worker(args: argparse.Namespace) -> int:
    plan = load_plan(args.output)
    indices = list(range(args.worker, len(plan["runs"]), args.workers))
    failures = []
    for position, index in enumerate(indices, 1):
        print(f"[{position}/{len(indices)}] {plan['runs'][index]['output_relative']}", flush=True)
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                                 "evaluate", "--output", str(args.output),
                                 "--index", str(index), "--test-chunk", str(args.test_chunk),
                                 "--eig-cutoff-scale", str(args.eig_cutoff_scale),
                                 "--partition-source", args.partition_source,
                                 "--seed-pred-offset", str(args.seed_pred_offset),
                                 "--predictor", args.predictor], check=False)
        if result.returncode:
            failures.append(dict(index=index, exit_code=result.returncode))
        real.atomic_json(args.output / "state" / f"worker{args.worker}.json",
                         dict(assigned=len(indices), processed=position, failures=failures,
                              complete=position == len(indices)))
    return 1 if failures else 0


def summary(args: argparse.Namespace) -> int:
    plan = load_plan(args.output)
    saved = invalid = 0
    family_counts = {"sgpr": 0, "svgp": 0}
    family_targets = {"sgpr": 0, "svgp": 0}
    for row in plan["runs"]:
        family_targets[row["family"]] += 1
        path = args.output / "hard" / row["output_relative"] / "prediction.json"
        if not path.is_file():
            continue
        saved += 1
        family_counts[row["family"]] += 1
        invalid += not json.loads(path.read_text())["valid"]
    print(f"SAVED={saved}/{len(plan['runs'])} INVALID={invalid} "
          + " ".join(f"{name.upper()}={family_counts[name]}/{family_targets[name]}"
                     for name in ("sgpr", "svgp")))
    return 0 if saved == len(plan["runs"]) and invalid == 0 else 2


def certify(args: argparse.Namespace) -> int:
    plan = load_plan(args.output)
    for kind, source in plan["sources"].items():
        root = Path(source["root"])
        for name, digest in source["files"].items():
            if real.sha256(root / name) != digest:
                raise ValueError(f"source file changed ({kind}): {name}")
    if real.sha256(Path(plan["gibbs_anchor_prediction_plan"])) != (
            plan["gibbs_anchor_prediction_plan_sha256"]):
        raise ValueError("Gibbs donor provenance plan changed")
    digests = {}
    for row in plan["runs"]:
        path = args.output / "hard" / row["output_relative"] / "prediction.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing prediction: {path}")
        payload = json.loads(path.read_text())
        if not payload.get("valid"):
            raise ValueError(f"invalid prediction: {path}")
        if payload["identity"].get("evaluator_sha256") != plan["hard_evaluator_sha256"]:
            raise ValueError(f"prediction evaluator differs: {path}")
        if payload["identity"].get("plan_sha256") != real.sha256(
                args.output / "plan.json"):
            raise ValueError(f"prediction plan differs: {path}")
        frozen = Path(row["root"])
        for name, digest in payload["identity"]["inputs"].items():
            if real.sha256(frozen / name) != digest:
                raise ValueError(f"frozen input changed: {name}")
        digests[row["output_relative"]] = real.sha256(path)
    certificate = dict(
        implementation=IMPLEMENTATION,
        plan_sha256=real.sha256(args.output / "plan.json"),
        predictions=digests,
        count=len(digests),
        scope="prediction-only; frozen checkpoints and training inputs unchanged",
        interpretation="hard PIC/PITC-style block-conditioned finite Gaussian proxy",
        caveat="not the exact dense variational predictive covariance",
    )
    destination = args.output / "HARD_PIC_PREDICTION_CERTIFICATE.json"
    if destination.is_file() and json.loads(destination.read_text()) != certificate:
        raise ValueError(f"different certificate already exists: {destination}")
    if not destination.is_file():
        real.atomic_json(destination, certificate)
    print(f"CERTIFIED={len(digests)} SHA256={real.sha256(destination)}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("plan")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--sgpr-root", required=True, type=Path)
    prepare.add_argument("--svgp-root", required=True, type=Path)
    prepare.add_argument("--gibbs-anchor-root", required=True, type=Path)
    prepare.add_argument("--sgpr-source", required=True, type=Path)
    prepare.add_argument("--svgp-source", required=True, type=Path)
    prepare.add_argument("--gibbs-anchor-prediction-plan", required=True,
                         type=Path)
    prepare.add_argument("--block-size", type=int, default=500)
    prepare.add_argument("--weight-power", type=float, default=2.0)
    for name in ("evaluate", "worker"):
        command = sub.add_parser(name)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--test-chunk", type=int, default=512)
        command.add_argument("--eig-cutoff-scale", type=float, default=10.0)
        command.add_argument("--partition-source", choices=("default", "anchor", "random"),
                             default="default")
        command.add_argument("--seed-pred-offset", type=int, default=10007)
        command.add_argument("--predictor", choices=("hard", "sparse"), default="hard")
        if name == "evaluate":
            command.add_argument("--index", required=True, type=int)
        else:
            command.add_argument("--worker", required=True, type=int)
            command.add_argument("--workers", type=int, default=5)
    for name in ("summary", "certify"):
        command = sub.add_parser(name)
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        return make_plan(args)
    if args.command == "evaluate":
        return evaluate(args)
    if args.command == "worker":
        return worker(args)
    if args.command == "summary":
        return summary(args)
    return certify(args)


if __name__ == "__main__":
    raise SystemExit(main())
