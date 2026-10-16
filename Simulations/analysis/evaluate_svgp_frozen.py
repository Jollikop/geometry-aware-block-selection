#!/usr/bin/env python3
"""Evaluate saved SVGP posteriors into a SEPARATE, provenance-bound directory.

Uses a frozen source checkout for kernels, data preprocessing, parameter loading
and reference predictions. Never calls training and never writes to the source
campaign. Each model runs in a fresh subprocess to release JAX device memory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DATASETS = "pol superconductivity kin40k protein elevators traffic electricity 3droad".split()
METHODS = "SVGP T-SVGP RBT-SVGP SBT-SVGP".split()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def separate_output(root, output):
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("output must be separate from (not inside/above) the frozen root")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    os.replace(tmp, path)


def streaming_block_reference(engine, pr, params, x, xs, blocks, kernel_name, jitter):
    """Exact historical block algebra, with projections formed one block at a time.

    Every fitted block contributes. Sum raw corrections before clipping once;
    summing already-clipped predictive variances would change the model.
    This deliberately uses the historical engine, not PreparedSVGP.
    """
    import numpy as np
    import jax
    base = engine._predictive_components(params, x[:1], xs, kernel_name, jitter)
    corrections = []
    for block in blocks:
        xb = x[np.asarray(block, dtype=np.int64)]
        components = engine._predictive_components(params, xb, xs, kernel_name, jitter)
        raw = pr.residual_contributions(components, predictor='block',
            blocks=[np.arange(len(xb), dtype=np.int64)], jitter=jitter)
        corrections.append(raw[:, 0])
    correction = np.stack(corrections, axis=1).sum(axis=1)
    mean = np.asarray(jax.device_get(base['mean'])).reshape(-1)
    variance = (np.asarray(jax.device_get(base['qvar'])).reshape(-1)
                + np.maximum(np.asarray(jax.device_get(base['d_test'])).reshape(-1) - correction, 0)
                + float(jax.device_get(base['s2'])))
    return mean, np.maximum(variance, pr.VARIANCE_FLOOR)


def evaluate(args):
    import numpy as np
    sys.path.insert(0, str(args.source_sim))
    from svgp import run_benchmark as runner
    from common import predictors as pr
    import jax
    import jax.numpy as jnp
    if getattr(args, "require_single_gpu", False):
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise ValueError(f"exactly one usable assigned GPU required, got {devices}")

    module_path = args.source_sim / "svgp/fast_prediction.py"
    spec = importlib.util.spec_from_file_location("evaluation_backend", module_path)
    backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backend)
    relative = Path(args.run)
    run = (args.root / relative).resolve()
    if args.root not in run.parents:
        raise ValueError("run outside frozen root")
    record_path = run / "training.json"
    if not record_path.is_file():
        record_path = run / "metrics.json"
    record = json.loads(record_path.read_text())
    cfg = record["config"]
    if runner.config_fingerprint(cfg) != record["config_id"]:
        raise ValueError("training configuration fingerprint mismatch")
    if cfg["code_commit"] != (args.root / "EXPECTED_HEAD").read_text().strip():
        raise ValueError("training numerical reference mismatch")
    if not (record.get("training_complete") is True or "rmse" in record):
        raise ValueError("training incomplete")
    method = cfg.get("training_method", cfg["method"])
    forced_predictor = getattr(args, "force_predictor", None)
    predictor = forced_predictor or runner.sv.METHOD_PREDICTOR[method]
    streaming_reference = getattr(args, 'block_reference_streaming', False)
    if streaming_reference and predictor != 'block':
        raise ValueError('streaming block reference is only valid for block prediction')
    if args.stage == "transplant" and (method != "SVGP" or predictor != "sparse"
                                       or cfg.get("donor_method") not in METHODS):
        raise ValueError("transplants must use their SAVED SVGP posterior")

    input_paths = [record_path, run / "params.npz", run / "curve.npz",
                   args.root / "EXPECTED_HEAD"]
    if predictor == "block":
        input_paths.append(run / "final_blocks.npz")
    donor = cfg.get("donor_method")
    if donor:
        stage = "stationary" if cfg["kernel"] in {"rbf", "matern52"} else "extended"
        path = (args.root / "svgp" / stage / "runs" / cfg["kernel"] / cfg["dataset"]
                / f"M{cfg['M']}" / donor / f"seed{cfg['seed']}" / "params.npz")
        if sha(path) != cfg.get("donor_params_sha256"):
            raise ValueError("donor checkpoint mismatch")
        input_paths.append(path)
    inputs = {str(p.relative_to(args.root)): sha(p) for p in input_paths}
    source_paths = sorted((args.source_sim / "common").glob("*.py"))
    source_paths += [args.source_sim / "svgp/engine.py",
                     args.source_sim / "svgp/run_benchmark.py"]
    source = {str(p.relative_to(args.source_sim)): sha(p) for p in source_paths}
    runtime = runner.runtime_manifest()
    identity = {
        "implementation": backend.IMPLEMENTATION,
        "backend_sha256": sha(module_path), "evaluator_sha256": sha(__file__),
        "source_modules": source, "inputs": inputs, "config_id": record["config_id"],
        "predictor": predictor, "forced_predictor": forced_predictor,
        "test_chunk": args.test_chunk,
        "train_chunk": args.train_chunk, "block_batch": args.block_batch,
        "factor_cache_mib": args.factor_cache_mib, "reps": args.reps,
        "reference_points": args.reference_points,
        "reference_policy": "frozen-sparse-predictor-training-input-invariance-v2"
        if predictor == "sparse" else ("frozen-blockwise-reference-v2" if streaming_reference
                                       else "frozen-full-training-reference-v1"),
        "measurement_context": getattr(args, "measurement_context", "unspecified"),
        "gpu_kinds": sorted({d.device_kind for d in jax.devices()}),
        "packages": runtime["packages"],
        "execution_env": {k: os.environ.get(k) for k in (
            "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_FLAGS", "JAX_PLATFORMS")},
    }
    dataset = cfg["dataset"]
    raw = runner.dm.raw_cache_path(dataset, runner.DATASET_DIR)
    if not raw.is_file() or sha(raw) != cfg["data_signature"]:
        raise ValueError("raw dataset missing or does not match frozen data hash")
    destination = args.output / relative / "prediction.json"
    if args.output not in destination.resolve().parents:
        raise ValueError("output symlink escapes the separate evaluation directory")
    if destination.exists():
        cached = json.loads(destination.read_text())
        if cached.get("identity") == identity:
            print(f"CACHED {relative}", flush=True)
            return
        raise ValueError("existing evaluation has a different identity; choose a NEW output root")

    X, y = runner.dm.load_raw(dataset, data_dir=runner.DATASET_DIR)
    x, _, xs, ys, meta = runner.jd.split_standardize(
        X, y, cfg["split_seed"], cfg["n_train"], cfg["n_test"],
        angle_cols=runner.dm.angle_cols_for(dataset), split_rng=cfg["split_rng"])
    for key in ("train_indices_sha256", "test_indices_sha256", "preprocessing_protocol"):
        if meta[key] != cfg[key]:
            raise ValueError(f"frozen preprocessing mismatch: {key}")
    params = runner._load_parameter_arrays(run / "params.npz")
    blocks = runner._load_saved_blocks(run) if predictor == "block" else None
    t0 = time.perf_counter()
    prepared = backend.PreparedSVGP(
        params, x, predictor=predictor, blocks=blocks, kernel_name=cfg["kernel"],
        jitter=runner.JITTER, test_chunk=args.test_chunk, train_chunk=args.train_chunk,
        block_batch=args.block_batch, factor_cache_bytes=args.factor_cache_mib << 20)
    prepare_s = time.perf_counter() - t0
    xs_device = jnp.asarray(xs)
    jax.block_until_ready(xs_device)
    t0 = time.perf_counter()
    mu, var = jax.block_until_ready(prepared(xs_device))
    compile_first_s = time.perf_counter() - t0
    if not bool(jnp.all(jnp.isfinite(mu)) & jnp.all(jnp.isfinite(var))):
        raise ValueError("nonfinite prepared predictions")
    # Check the actual frozen model, not just a synthetic unit-test fixture.
    check = np.linspace(0, len(xs) - 1, min(args.reference_points, len(xs)), dtype=int)
    # Sparse q(u) prediction is independent of the training input matrix.
    # The legacy reference constructs K(Z, X_train) before branching, although
    # the sparse branch never uses it.  Two differently sliced training inputs
    # both avoid that irrelevant allocation and explicitly certify invariance.
    reference_x = x[:1] if predictor == "sparse" else x
    if streaming_reference:
        ref_mu, ref_var = streaming_block_reference(
            runner.sv, pr, params, x, xs[check], blocks, cfg['kernel'], runner.JITTER)
    else:
        ref_mu, ref_var = runner.sv.predict_svgp(
            params, reference_x, xs[check], predictor=predictor, blocks=blocks,
            kernel_name=cfg["kernel"], jitter=runner.JITTER)
    if predictor == "sparse":
        invariant_mu, invariant_var = runner.sv.predict_svgp(
            params, x[-17:], xs[check], predictor=predictor, blocks=blocks,
            kernel_name=cfg["kernel"], jitter=runner.JITTER)
        np.testing.assert_allclose(ref_mu, invariant_mu, atol=2e-8, rtol=2e-7)
        np.testing.assert_allclose(ref_var, invariant_var, atol=2e-8, rtol=2e-7)
    got_mu, got_var = np.asarray(mu)[check], np.asarray(var)[check]
    np.testing.assert_allclose(got_mu, ref_mu, atol=2e-8, rtol=2e-7)
    np.testing.assert_allclose(got_var, ref_var, atol=2e-8, rtol=2e-7)
    timings = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        jax.block_until_ready(prepared(xs_device))
        timings.append(time.perf_counter() - t0)
    rmse, nll = pr.gaussian_rmse_nll(mu, var, ys)
    for p in input_paths:
        if sha(p) != inputs[str(p.relative_to(args.root))]:
            raise ValueError("input changed during evaluation; stop training before evaluating")
    for p in source_paths:
        if sha(p) != source[str(p.relative_to(args.source_sim))]:
            raise ValueError("source changed during evaluation")
    write_json(destination, {
        "identity": identity, "prediction_runtime": runtime,
        "training_config": cfg, "rmse": rmse, "nll": nll,
        "rmse_orig": rmse * float(meta["y_std"]),
        "nll_orig": nll + float(np.log(meta["y_std"])),
        "prepare_s": prepare_s, "compile_and_first_prediction_s": compile_first_s,
        "warm_prediction_ms": float(np.median(timings) * 1000),
        "timing_scope": "prepared model; device inputs/outputs; no metrics or host transfer",
        "factors_cached": prepared.factors_cached,
        "factor_bytes": prepared.factor_bytes,
        "reference_check_points": len(check),
        "reference_mean_max_abs": float(np.max(np.abs(got_mu - ref_mu))),
        "reference_variance_max_abs": float(np.max(np.abs(got_var - ref_var))),
    })
    print(f"PREDICTED {relative} rmse={rmse:.6g} nll={nll:.6g} "
          f"warm={np.median(timings)*1000:.2f}ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-sim", required=True, type=Path)
    parser.add_argument("--stage", choices=["stationary", "extended", "transplant"], default="transplant")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--measurement-context", choices=["unspecified", "concurrent-training"],
                        default="unspecified")
    parser.add_argument("--require-single-gpu", action="store_true")
    parser.add_argument('--block-reference-streaming', action='store_true',
                        help='check the exact historical block formula with bounded projection memory')
    parser.add_argument("--test-chunk", type=int, default=256)
    parser.add_argument("--train-chunk", type=int, default=512)
    parser.add_argument("--block-batch", type=int, default=4)
    parser.add_argument("--factor-cache-mib", type=int, default=128)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--reference-points", type=int, default=8)
    parser.add_argument("--force-predictor", choices=["sparse", "diagonal", "block", "full"])
    parser.add_argument("--runs-file", type=Path,
                        help="optional newline-delimited list of frozen run paths")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    args.source_sim = args.source_sim.resolve()
    separate_output(args.root, args.output)
    if min(args.test_chunk, args.train_chunk, args.block_batch, args.reps, args.reference_points) < 1:
        parser.error("chunk, batch, repetitions and reference points must be positive")
    if args.factor_cache_mib < 0 or not set(args.datasets) <= set(DATASETS):
        parser.error("invalid cache size or datasets")
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        parser.error("worker-index must be in [0, workers)")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds) \
            or not set(args.seeds) <= set(range(10)):
        parser.error("datasets/seeds must be unique; seeds must be in [0,9]")
    args.seeds = sorted(args.seeds)
    if args.run:
        evaluate(args)
        return
    kernels = {"stationary": ["rbf", "matern52"], "extended": ["amplitude-rbf", "gibbs"],
               "transplant": ["rbf", "matern52", "amplitude-rbf", "gibbs"]}[args.stage]
    if args.runs_file:
        runs = [Path(line.strip()) for line in args.runs_file.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    else:
        runs = [Path("svgp") / args.stage / "runs" / k / d / f"M{m}"
                / (method + "-to-SVGP" if args.stage == "transplant" else method) / f"seed{s}"
                for k in kernels for d in args.datasets for m in (64, 128, 256, 512)
                for method in METHODS for s in args.seeds]
    runs = [run for index, run in enumerate(runs) if index % args.workers == args.worker_index]
    for relative in runs:
        run = args.root / relative
        if not (run / "params.npz").is_file() or not (run / "curve.npz").is_file() \
                or not any((run / name).is_file() for name in ("training.json", "metrics.json")):
            raise SystemExit(f"Missing trained run: {relative}. Finish training first.")
    failures = []
    consecutive_failures = 0
    for i, relative in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {relative}", flush=True)
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 *sys.argv[1:], "--run", str(relative)])
        if result.returncode:
            failures.append(str(relative))
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("Stopping after three consecutive failures; completed predictions are preserved.", flush=True)
                failures.extend(str(p) for p in runs[i:])
                break
        else:
            consecutive_failures = 0
    kinds = set()
    for relative in runs:
        path = args.output / relative / "prediction.json"
        if path.is_file() and str(relative) not in failures:
            kinds.add(tuple(json.loads(path.read_text())["identity"]["gpu_kinds"]))
    uniform = len(kinds) == 1
    suffix = "" if args.seeds == list(range(10)) else "_seeds_" + "-".join(map(str, args.seeds))
    worker_suffix = f"_worker{args.worker_index}of{args.workers}" if args.workers > 1 else ""
    write_json(args.output / f"{args.stage}{suffix}{worker_suffix}_evaluation_summary.json", {
        "datasets": args.datasets, "seeds": args.seeds,
        "measurement_context": args.measurement_context,
        "expected": len(runs), "successful": len(runs) - len(failures),
        "evaluation_complete": not failures and uniform, "failed": failures,
        "uniform_device_kind": uniform, "device_kinds": sorted(kinds),
        "note": "Separate prediction evaluation, NOT a replacement training/review freeze."})
    raise SystemExit(1 if failures or not uniform else 0)


if __name__ == "__main__":
    main()
