#!/usr/bin/env python3
"""Isolated 320-run Gibbs anchor ablation; never write to the donor freeze.

Numerics come from the clean historical execution checkout, not this working
tree. Training is deferred-prediction, with immutable copied RBF initializers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

SOURCE_HEAD = "b4a2ebcad2dba7b2e712b212a749460c5d3e54bf"
REFERENCE_HEAD = "518db753c8a18e5a3d6349da9289568df36cb13e"
DATASETS = "pol superconductivity kin40k protein elevators traffic electricity 3droad".split()
TAG = "gibbs_anchor_rbf_warm300_v1"
ARTIFACTS = ("training.json", "params.npz", "curve.npz", "final_blocks.npz")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()[:16]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def disjoint(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError(f"directories must be separate: {a}, {b}")


def source_hashes(source):
    source = Path(source)
    if git(source, "rev-parse", "HEAD") != SOURCE_HEAD:
        raise ValueError(f"use the historical execution checkout {SOURCE_HEAD}")
    if git(source, "status", "--porcelain", "--untracked-files=all", "--", "Simulations"):
        raise ValueError("historical Simulations source is dirty")
    changed = git(source, "diff", "--name-only", REFERENCE_HEAD, SOURCE_HEAD,
                  "--", "Simulations/common", "Simulations/svgp").splitlines()
    if any(not p.endswith("/Simulations/svgp/run_benchmark.py") for p in changed):
        raise ValueError("historical numerical modules differ from frozen reference")
    paths = sorted((source / "Simulations/common").glob("*.py"))
    paths += sorted((source / "Simulations/svgp").glob("*.py"))
    return {str(p.relative_to(source)): sha(p) for p in paths}


def expected_config(reference):
    config = dict(reference)
    # The only numerical intervention is the selector. A custom one-run plan
    # releases JAX memory between tasks; tag and plan are identity metadata.
    config.update(sbt_geometry="anchor", run_tag=TAG, plan="custom")
    return config


def prepare(args):
    root, baseline, source = args.root.resolve(), args.baseline.resolve(), args.source.resolve()
    disjoint(root, baseline)
    disjoint(root, source)
    if root.exists():
        raise ValueError("output already exists; refuse to overwrite or mix campaigns")
    modules = source_hashes(source)
    tasks, input_files, raw_data = [], {}, {}
    for dataset in DATASETS:
        for m in (64, 128, 256, 512):
            for seed in range(10):
                suffix = Path(dataset) / f"M{m}" / "SBT-SVGP" / f"seed{seed}"
                old = baseline / "svgp/extended/runs/gibbs" / suffix
                reference_path = old / "training.json"
                if not reference_path.is_file():
                    reference_path = old / "metrics.json"
                reference = read(reference_path)
                cfg = reference["config"]
                required = dict(kernel="gibbs", method="SBT-SVGP", dataset=dataset,
                                M=m, seed=seed, epochs=300, sbt_geometry="residual-probe",
                                init_from_kernel="rbf", refresh_every=40, refresh_until=81,
                                code_commit=REFERENCE_HEAD, code_source_dirty=False,
                                warp_gate_fraction=0.0, batch_coupling="split",
                                penalty_sampling="uniform", all_data=True)
                if any(cfg.get(k) != v for k, v in required.items()):
                    raise ValueError(f"unexpected baseline protocol: {reference_path}")
                if reference["config_id"] != fingerprint(cfg):
                    raise ValueError(f"baseline fingerprint mismatch: {reference_path}")
                warm = baseline / "svgp/stationary/runs/rbf" / suffix / "params.npz"
                warm_hash = sha(warm)
                if warm_hash != cfg["init_params_sha256"]:
                    raise ValueError(f"RBF initializer differs from original Gibbs run: {warm}")
                raw = args.datasets.resolve() / cfg["data_cache"]
                if str(raw) not in raw_data:
                    raw_data[str(raw)] = sha(raw)
                if raw_data[str(raw)] != cfg["data_signature"]:
                    raise ValueError(f"dataset changed: {raw}")
                warm_rel = str(Path("inputs/rbf") / suffix / "params.npz")
                ref_rel = str(Path("inputs/reference") / suffix / "training.json")
                input_files[warm_rel] = (warm, warm_hash)
                input_files[ref_rel] = (reference_path, sha(reference_path))
                tasks.append(dict(index=len(tasks), worker=seed % 5,
                                  run=str(Path("svgp/extended/runs/gibbs") / suffix),
                                  warm=warm_rel, reference=ref_rel,
                                  config=expected_config(cfg)))
    # Check everything before making any output; inputs are copies, not links.
    root.mkdir(parents=True)
    for relative, (path, digest) in input_files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha(target) != digest:
            raise ValueError(f"copy checksum mismatch: {target}")
    tool = root / "tools/gibbs_anchor_campaign.py"
    tool.parent.mkdir()
    shutil.copyfile(__file__, tool)
    for directory in ("slurm", "state", "logs"):
        (root / directory).mkdir()
    (root / "EXPECTED_HEAD").write_text(REFERENCE_HEAD + "\n")
    plan = dict(version=TAG, baseline=str(baseline), source=str(source),
                source_head=SOURCE_HEAD, numerical_reference=REFERENCE_HEAD,
                source_modules=modules, datasets=raw_data, python=str(args.python.absolute()),
                dataset_dir=str(args.datasets.resolve()), workers=5, tasks=tasks,
                inputs={k: v[1] for k, v in input_files.items()}, tool_sha256=sha(tool),
                prediction="deferred; evaluate into a separate output directory",
                scientific_change="SBT Gibbs selector residual-probe -> inducing anchor")
    write(root / "campaign.json", plan)
    (root / "campaign.sha256").write_text(sha(root / "campaign.json") + "\n")
    print(f"PREPARED {len(tasks)} runs; 5 shards of 64; baseline untouched\nROOT={root}")


def load_plan(root):
    if sha(root / "campaign.json") != (root / "campaign.sha256").read_text().strip():
        raise ValueError("campaign manifest changed")
    plan = read(root / "campaign.json")
    if sha(__file__) != plan["tool_sha256"]:
        raise ValueError("use the copied, frozen campaign tool")
    return plan


def command(plan, root, task):
    c = task["config"]
    cmd = [plan["python"], "-u", str(Path(plan["source"]) / "Simulations/svgp/run_benchmark.py"),
           "--plan", "custom", "--run-tag", TAG, "--kernels", "gibbs",
           "--datasets", c["dataset"], "--methods", "SBT-SVGP", "--m", str(c["M"]),
           "--seeds", str(c["seed"]), "--split-seed", str(c["split_seed"]),
           "--all-data", "--common-ll-batches", "--normalize-objective", "--save-params",
           "--defer-prediction", "--prediction-reps", "0", "--no-live",
           "--sbt-geometry", "anchor", "--init-from-kernel", "rbf",
           "--init-from-runs-dir", str(root / "inputs"),
           "--runs-dir", str(root / "svgp/extended/runs"),
           "--logs-dir", str(root / "logs" / f"worker{task['worker']}")]
    for name in ("epochs", "block", "ll_batch_size", "blocks_per_step", "refresh_every",
                 "refresh_until", "batch_coupling", "penalty_sampling", "optimizer",
                 "lr", "z_lr", "lr_profile", "lr_schedule", "q_optimizer", "natural_lr",
                 "warp_gate_fraction", "swa_fraction", "grad_clip", "residual_probes",
                 "importance_refresh_every", "importance_epsilon", "log_every",
                 "split_rng", "full_predictor_max_n"):
        cmd += ["--" + name.replace("_", "-"), str(c[name])]
    return cmd


def validate_task(root, task, *, deep=False):
    run = root / task["run"]
    if not (run / "training.json").exists():
        return "missing"
    try:
        row = read(run / "training.json")
        if row.get("config") != task["config"]:
            raise ValueError("configuration differs from paired protocol")
        if row.get("config_id") != fingerprint(task["config"]):
            raise ValueError("configuration hash mismatch")
        if row.get("training_complete") is not True or row.get("epochs") != 300:
            raise ValueError("training incomplete")
        if row.get("sbt_geometry") != "anchor" or row.get("execution_commit") != SOURCE_HEAD:
            raise ValueError("selector/execution mismatch")
        if not math.isfinite(row["last_epoch_stochastic_objective_estimate_per_datum"]):
            raise ValueError("nonfinite final objective")
        for name in ARTIFACTS:
            if not (run / name).is_file() or (run / name).stat().st_size == 0:
                raise ValueError(f"missing {name}")
        if deep:
            import numpy as np
            for name in ARTIFACTS[1:]:
                with np.load(run / name, allow_pickle=False) as arrays:
                    if not arrays.files or any(not np.isfinite(arrays[k]).all() for k in arrays.files):
                        raise ValueError(f"invalid/nonfinite arrays: {name}")
            with np.load(run / "curve.npz", allow_pickle=False) as curve:
                if not len(curve["epoch"]) or int(curve["epoch"][-1]) != 300:
                    raise ValueError("curve does not reach epoch 300")
        return "valid"
    except (ValueError, KeyError, OSError, TypeError) as exc:
        return f"invalid: {exc}"


def verify_inputs(root, plan):
    if source_hashes(plan["source"]) != plan["source_modules"]:
        raise ValueError("historical code changed")
    for name, digest in plan["inputs"].items():
        if sha(root / name) != digest:
            raise ValueError(f"copied input changed: {name}")
    for name, digest in plan["datasets"].items():
        if sha(name) != digest:
            raise ValueError(f"dataset changed: {name}")


def one(args):
    """Check the actual resolved runner configuration BEFORE any optimization."""
    root = args.root.resolve()
    plan = load_plan(root)
    task = plan["tasks"][args.index]
    sys.path.insert(0, str(Path(plan["source"]) / "Simulations"))
    from svgp import run_benchmark as runner
    original = runner.make_run_config

    def checked_config(*positional, **keywords):
        config, config_id = original(*positional, **keywords)
        if config != task["config"]:
            differences = {k: (task["config"].get(k), config.get(k))
                           for k in set(config) | set(task["config"])
                           if config.get(k) != task["config"].get(k)}
            print(f"resolved configuration differs BEFORE training: {differences}", file=sys.stderr)
            raise SystemExit(78)
        return config, config_id

    runner.make_run_config = checked_config
    sys.argv = command(plan, root, task)[2:]
    runner.main()


def worker(args):
    root = args.root.resolve()
    plan = load_plan(root)
    if args.worker not in range(5):
        raise ValueError("worker must be 0..4")
    with (root / "state" / f"worker{args.worker}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_inputs(root, plan)
        env = os.environ.copy()
        env.update(JAX_PLATFORMS="cuda", PYTHONDONTWRITEBYTECODE="1",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform",
                   SVGP_NUMERICAL_REFERENCE_HEAD=REFERENCE_HEAD,
                   SCALABLE_GP_DATASET_DIR=plan["dataset_dir"],
                   PYTHONPATH=str(Path(plan["source"]) / "Simulations"))
        # A real allocated CUDA operation, not merely an nvidia-smi listing.
        subprocess.run([plan["python"], "-c",
                        "import jax,jax.numpy as j; d=jax.devices(); print(d,flush=True); "
                        "assert len(d)==1 and d[0].platform=='gpu'; "
                        "(j.ones((32,32))@j.ones((32,32))).block_until_ready()"],
                       env=env, check=True)
        assigned = [t for t in plan["tasks"] if t["worker"] == args.worker]
        counts = dict(valid=0, failed=0, assigned=len(assigned))
        for i, task in enumerate(assigned, 1):
            print(f"[{i}/{len(assigned)}] {task['run']}", flush=True)
            state = validate_task(root, task)
            if state == "missing":
                result = subprocess.run(
                    [plan["python"], "-u", str(root / "tools/gibbs_anchor_campaign.py"),
                     "one", "--root", str(root), "--index", str(task["index"])],
                    env=env, cwd=plan["source"])
                state = validate_task(root, task, deep=True)
                if result.returncode:
                    state = f"invalid: process exit {result.returncode}; {state}"
                if result.returncode == 78:
                    write(root / "state" / f"worker{args.worker}.json",
                          dict(counts, finished=False, error=state, current=task["run"]))
                    raise SystemExit(78)
            # Existing invalid results are never overwritten automatically.
            counts["valid" if state == "valid" else "failed"] += 1
            print(state, flush=True)
            write(root / "state" / f"worker{args.worker}.json",
                  dict(counts, finished=i == len(assigned), current=task["run"]))
        if counts["failed"]:
            raise SystemExit(1)


def status(args):
    root = args.root.resolve()
    plan = load_plan(root)
    counts = Counter(validate_task(root, t).split(":")[0] for t in plan["tasks"])
    print(f"GIBBS SBT-SVGP ANCHOR | trained-valid={counts['valid']}/320 "
          f"missing={counts['missing']} invalid={counts['invalid']}")
    print("RBF warm start + 300 epochs; prediction deferred; NOT a final paper certificate")
    for k in range(5):
        p = root / "state" / f"worker{k}.json"
        print(f"worker {k}: {read(p) if p.exists() else 'not started / check Slurm'}")


def audit(args):
    root = args.root.resolve()
    plan = load_plan(root)
    verify_inputs(root, plan)
    errors, files = [], {}
    for task in plan["tasks"]:
        state = validate_task(root, task, deep=True)
        if state != "valid":
            errors.append(dict(run=task["run"], error=state))
        else:
            for name in ARTIFACTS:
                p = root / task["run"] / name
                files[str(p.relative_to(root))] = sha(p)
    report = dict(training_complete=len(errors) == 0, expected=320, errors=errors,
                  campaign_sha256=sha(root / "campaign.json"), artifacts=files,
                  prediction_complete=False, release_ready=False,
                  note="Training artifact audit only; no convergence or predictive claim")
    write(root / "training_audit.json", report)
    print(f"TRAINING_AUDIT: valid={320-len(errors)}/320; release_ready=no")
    if errors:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--datasets", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    for name in ("worker", "status", "audit", "one"):
        q = sub.add_parser(name)
        q.add_argument("--root", type=Path, required=True)
        if name == "worker":
            q.add_argument("--worker", type=int, required=True)
        if name == "one":
            q.add_argument("--index", type=int, required=True)
    p.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "worker": worker, "status": status, "audit": audit,
     "one": one}[args.mode](args)


if __name__ == "__main__":
    main()
