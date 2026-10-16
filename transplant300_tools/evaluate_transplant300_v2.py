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
import importlib.metadata
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

    module_path = Path(__file__).resolve().with_name("fast_prediction.py")
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
    plan_path = args.root / 'continuation_plan.json'
    plan = json.loads(plan_path.read_text())
    parent_root = Path(plan['root']).resolve()
    separate_output(parent_root, args.output)
    if record.get('epochs') != 300 or record.get('training_protocol') != plan['protocol']:
        raise ValueError('expected the frozen 150-to-300 continuation')
    if Path(record['parent_root']).resolve() != parent_root:
        raise ValueError('parent root mismatch')
    if cfg["code_commit"] != (parent_root / "EXPECTED_HEAD").read_text().strip():
        raise ValueError("training numerical reference mismatch")
    if sha(parent_root / 'EXPECTED_HEAD') != plan['expected_head_sha256']:
        raise ValueError('parent numerical reference hash mismatch')
    for name, digest in plan['source'].items():
        if sha(args.source_sim / name) != digest:
            raise ValueError(f'continuation source changed: {name}')
    for name, version in plan['numeric_packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'numeric package mismatch: {name}')
    complete_path = run / 'COMPLETE.json'
    complete = json.loads(complete_path.read_text())
    if complete['identity'] != record['continuation_identity'] or complete['identity']['plan_sha256'] != sha(plan_path):
        raise ValueError('continuation identity mismatch')
    for name, digest in complete['files'].items():
        if Path(name).name != name or sha(run / name) != digest:
            raise ValueError(f'complete archive hash mismatch: {name}')
    if not (record.get("training_complete") is True or "rmse" in record):
        raise ValueError("training incomplete")
    method = cfg.get("training_method", cfg["method"])
    predictor = runner.sv.METHOD_PREDICTOR[method]
    if args.stage == "transplant" and (method != "SVGP" or predictor != "sparse"
                                       or cfg.get("donor_method") not in METHODS):
        raise ValueError("transplants must use their SAVED SVGP posterior")

    input_paths = [record_path, run / "params.npz", run / "deterministic_curve.npz",
                   complete_path, plan_path, parent_root / "EXPECTED_HEAD"]
    parent_params = parent_root / relative / 'params.npz'
    if sha(parent_params) != cfg['parent_params_sha256']:
        raise ValueError('epoch-150 parent checkpoint mismatch')
    input_paths.append(parent_params)
    if predictor == "block":
        input_paths.append(run / "final_blocks.npz")
    donor = cfg.get("donor_method")
    if donor:
        stage = "stationary" if cfg["kernel"] in {"rbf", "matern52"} else "extended"
        path = (parent_root / "svgp" / stage / "runs" / cfg["kernel"] / cfg["dataset"]
                / f"M{cfg['M']}" / donor / f"seed{cfg['seed']}" / "params.npz")
        if sha(path) != cfg.get("donor_params_sha256"):
            raise ValueError("donor checkpoint mismatch")
        input_paths.append(path)
    inputs = {str(p): sha(p) for p in input_paths}
    source_paths = sorted((args.source_sim / "common").glob("*.py"))
    source_paths += [args.source_sim / "svgp/engine.py",
                     args.source_sim / "svgp/run_benchmark.py"]
    source = {str(p.relative_to(args.source_sim)): sha(p) for p in source_paths}
    runtime = runner.runtime_manifest()
    identity = {
        "implementation": backend.IMPLEMENTATION,
        "backend_sha256": sha(module_path), "evaluator_sha256": sha(__file__),
        "source_modules": source, "inputs": inputs, "config_id": record["config_id"],
        "predictor": predictor, "test_chunk": args.test_chunk,
        "train_chunk": args.train_chunk, "block_batch": args.block_batch,
        "factor_cache_mib": args.factor_cache_mib, "reps": args.reps,
        "reference_points": args.reference_points,
        "measurement_context": getattr(args, "measurement_context", "unspecified"),
        "reference_policy": "frozen-sparse-predictor-training-input-invariance-v2",
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
    # Sparse q(u) predictions depend only on theta, Z, q(u), and x_test.
    # The frozen reference unnecessarily constructs K(Z, x_train), although
    # its sparse branch never uses that projection. Keep the reference code
    # unchanged and pass one training row to avoid that irrelevant allocation.
    # Do NOT apply this shortcut to diagonal/block/full predictors.
    if predictor != "sparse":
        raise ValueError("v2 reference shortcut is restricted to sparse transplants")
    ref_mu, ref_var = runner.sv.predict_svgp(
        params, x[:1], xs[check], predictor=predictor, blocks=blocks,
        kernel_name=cfg["kernel"], jitter=runner.JITTER)
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
        if sha(p) != inputs[str(p)]:
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
    parser.add_argument("--test-chunk", type=int, default=256)
    parser.add_argument("--train-chunk", type=int, default=512)
    parser.add_argument("--block-batch", type=int, default=4)
    parser.add_argument("--factor-cache-mib", type=int, default=128)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--reference-points", type=int, default=8)
    parser.add_argument("--run", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    args.source_sim = args.source_sim.resolve()
    separate_output(args.root, args.output)
    if min(args.test_chunk, args.train_chunk, args.block_batch, args.reps, args.reference_points) < 1:
        parser.error("chunk, batch, repetitions and reference points must be positive")
    if args.factor_cache_mib < 0 or not set(args.datasets) <= set(DATASETS):
        parser.error("invalid cache size or datasets")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds) \
            or not set(args.seeds) <= set(range(10)):
        parser.error("datasets/seeds must be unique; seeds must be in [0,9]")
    args.seeds = sorted(args.seeds)
    if args.run:
        evaluate(args)
        return
    kernels = {"stationary": ["rbf", "matern52"], "extended": ["amplitude-rbf", "gibbs"],
               "transplant": ["rbf", "matern52", "amplitude-rbf", "gibbs"]}[args.stage]
    runs = [Path("svgp") / args.stage / "runs" / k / d / f"M{m}"
            / (method + "-to-SVGP" if args.stage == "transplant" else method) / f"seed{s}"
            for k in kernels for d in args.datasets for m in (64, 128, 256, 512)
            for method in METHODS for s in args.seeds]
    for relative in runs:
        run = args.root / relative
        if not (run / "params.npz").is_file() or not (run / "COMPLETE.json").is_file() \
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
    write_json(args.output / f"{args.stage}{suffix}_evaluation_summary.json", {
        "datasets": args.datasets, "seeds": args.seeds,
        "measurement_context": args.measurement_context,
        "expected": len(runs), "successful": len(runs) - len(failures),
        "evaluation_complete": not failures and uniform, "failed": failures,
        "uniform_device_kind": uniform, "device_kinds": sorted(kinds),
        "note": "Separate prediction evaluation, NOT a replacement training/review freeze."})
    raise SystemExit(1 if failures or not uniform else 0)


if __name__ == "__main__":
    main()
