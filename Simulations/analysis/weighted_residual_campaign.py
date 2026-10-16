#!/usr/bin/env python3
"""Frozen, prediction-only weighted-block campaign.

The evaluated residual kernel is gated in both train/test directions:

    D_w(x, x') = D(x, x') sum_r sqrt(w_r(x) w_r(x')).

Training observations have one-hot weights in their fitted blocks. Test
weights are normalized inverse squared kernel distances to the nearest
training observation in each block. Thus the training residual is the frozen
block restriction and every finite joint covariance is PSD. When identical
training inputs occur in different blocks, this remains an indexed Gaussian
surrogate, not a kernel defined solely on input locations. Such runs are
recorded explicitly. Neither training code nor frozen checkpoints are written.
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
import weighted_residual_audit as weighted


IMPLEMENTATION = "weighted-frozen-block-prediction-v1"
METHODS_SGPR = ("T-SGPR", "RBT-SGPR", "SBT-SGPR", "METIS-SGPR")
METHODS_SVGP = ("T-SVGP", "RBT-SVGP", "SBT-SVGP")
SOURCE_FILES = ("common/kernels.py", "common/kernels_nonstationary.py",
                "common/data_split.py", "common/datasets.py",
                "common/partitions.py", "common/terms.py",
                "common/predictors.py", "svgp/engine.py",
                "svgp/run_benchmark.py")


def record_path(run: Path) -> Path:
    for name in ("training.json", "metrics.json"):
        path = run / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing training record: {run}")


def source_hashes(source_sim: Path, *, family: str) -> dict[str, str]:
    names = SOURCE_FILES if family == "svgp" else (
        "common/kernels.py", "common/kernels_nonstationary.py",
        "common/partitions.py", "common/terms.py",
        "common/predictors.py", "sgpr/jax_collapsed.py",
        "sgpr/run_benchmark.py")
    return {name: real.sha256(source_sim / name) for name in names}


def checked_checkout(path: Path, frozen: Path) -> str:
    expected = (frozen / "EXPECTED_HEAD").read_text().strip()
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError(f"checkout {actual} differs from frozen {expected}")
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain",
         "--untracked-files=no", "--", "Simulations"], text=True).strip()
    if dirty:
        raise ValueError("historical source has modified tracked simulation code")
    return expected


def make_plan(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    sgpr_root, svgp_root, gibbs_root = (
        args.sgpr_root.resolve(), args.svgp_root.resolve(),
        args.gibbs_anchor_root.resolve())
    sgpr_source, svgp_source = (args.sgpr_source.resolve(),
                                args.svgp_source.resolve())
    gibbs_plan_path = args.gibbs_anchor_prediction_plan.resolve()
    gibbs_plan = json.loads(gibbs_plan_path.read_text())
    gibbs_source = Path(gibbs_plan["source_sim"]).resolve()
    if Path(gibbs_plan["root"]).resolve() != gibbs_root:
        raise ValueError("Gibbs anchor prediction provenance names another donor root")
    real.guard_separate(output, [sgpr_root, svgp_root, gibbs_root,
                                 sgpr_source, svgp_source, gibbs_source,
                                 gibbs_plan_path.parent])
    expected_sgpr = checked_checkout(sgpr_source, sgpr_root)
    expected_svgp = checked_checkout(svgp_source, svgp_root)
    expected_gibbs = (gibbs_root / "EXPECTED_HEAD").read_text().strip()
    if expected_gibbs != expected_svgp:
        raise ValueError("Gibbs anchor was not built against SVGP frozen source")
    if gibbs_plan["expected"] != 320 or not gibbs_plan["training_audited"]:
        raise ValueError("Gibbs anchor training was not fully audited")
    for name, digest in gibbs_plan["source_files"].items():
        if real.sha256(gibbs_source / name) != digest:
            raise ValueError(f"Gibbs anchor source snapshot changed: {name}")
    if gibbs_plan["training_inputs"]["EXPECTED_HEAD"] != real.sha256(
            gibbs_root / "EXPECTED_HEAD"):
        raise ValueError("Gibbs anchor training head hash changed")

    runs: list[dict] = []
    for method in METHODS_SGPR:
        for seed in range(10):
            relative = (Path("runs/rbf/kin40k/M512") / method / f"seed{seed}")
            runs.append(dict(family="sgpr", root=str(sgpr_root),
                             source_sim=str(sgpr_source / "Simulations"),
                             relative=str(relative),
                             output_relative=str(Path("sgpr/rbf/kin40k/M512")
                                                 / method / f"seed{seed}"),
                             method=method, kernel="rbf", dataset="kin40k",
                             M=512, seed=seed, source_kind="historical_sgpr"))
    for kernel, stage in anchor.KERNEL_STAGE.items():
        for dataset in anchor.DATASETS:
            for m in anchor.M_VALUES:
                for method in METHODS_SVGP:
                    for seed in anchor.SEEDS:
                        use_anchor = kernel == "gibbs" and method == "SBT-SVGP"
                        root = gibbs_root if use_anchor else svgp_root
                        source_sim = (gibbs_source if use_anchor
                                      else svgp_source / "Simulations")
                        relative = (Path("svgp") / stage / "runs" / kernel
                                    / dataset / f"M{m}" / method / f"seed{seed}")
                        runs.append(dict(family="svgp", root=str(root),
                                         source_sim=str(source_sim),
                                         relative=str(relative),
                                         output_relative=str(Path("svgp") / kernel
                                                             / dataset / f"M{m}"
                                                             / method / f"seed{seed}"),
                                         method=method, kernel=kernel,
                                         dataset=dataset, M=m, seed=seed,
                                         source_kind=("gibbs_anchor" if use_anchor
                                                      else "historical_svgp")))
    if len(runs) != 3880:
        raise ValueError(f"expected 40 SGPR + 3840 SVGP, got {len(runs)}")
    counts = {"gibbs_anchor": 0, "historical_svgp": 0,
              "historical_sgpr": 0}
    for row in runs:
        counts[row["source_kind"]] += 1
        root = Path(row["root"])
        run = root / row["relative"]
        metadata = json.loads(record_path(run).read_text())
        cfg = metadata["config"]
        expected = expected_sgpr if row["family"] == "sgpr" else expected_svgp
        if (cfg["code_commit"] != expected or cfg["kernel"] != row["kernel"]
                or cfg["M"] != row["M"] or cfg["seed"] != row["seed"]):
            raise ValueError(f"frozen checkpoint config mismatch: {run}")
        if row["family"] == "svgp" and cfg["dataset"] != row["dataset"]:
            raise ValueError(f"frozen dataset mismatch: {run}")
        if row["source_kind"] == "gibbs_anchor" and not (
                cfg["sbt_geometry"] == "anchor" and cfg["epochs"] == 300
                and cfg["init_from_kernel"] == "rbf"):
            raise ValueError(f"not a 300-epoch anchor donor: {run}")
        needed = ("params.npz",) + (("final_blocks.npz",)
                                   if row["method"].startswith((
                                       "RBT-", "SBT-", "METIS-"))
                                   else ())
        if not all((run / name).is_file() for name in needed):
            raise FileNotFoundError(f"missing frozen parameters or blocks: {run}")

    payload = dict(
        implementation=IMPLEMENTATION,
        evaluator_sha256=real.sha256(Path(__file__)),
        weighted_engine_sha256=real.sha256(Path(weighted.__file__)),
        sgpr_loader_sha256=real.sha256(Path(real.__file__)),
        svgp_loader_sha256=real.sha256(Path(anchor.__file__)),
        expected_heads={"sgpr": expected_sgpr, "svgp": expected_svgp,
                        "gibbs_anchor": expected_gibbs},
        sources={kind: {"root": str(path),
                        "files": source_hashes(path, family=family)}
                 for kind, path, family in (
                     ("historical_sgpr", sgpr_source / "Simulations", "sgpr"),
                     ("historical_svgp", svgp_source / "Simulations", "svgp"),
                     ("gibbs_anchor", gibbs_source, "svgp"))},
        gibbs_anchor_prediction_plan=str(gibbs_plan_path),
        gibbs_anchor_prediction_plan_sha256=real.sha256(gibbs_plan_path),
        block_size_for_T=args.block_size,
        weight_power=args.weight_power,
        runs=runs,
    )
    destination = output / "plan.json"
    if destination.is_file():
        if json.loads(destination.read_text()) != payload:
            raise ValueError("different plan already exists: choose a new output root")
    else:
        real.atomic_json(destination, payload)
    print(f"PLAN={destination} RUNS={len(runs)} SOURCES={counts}", flush=True)
    return 0


def load_plan(output: Path) -> dict:
    plan = json.loads((output / "plan.json").read_text())
    for key, path in (("evaluator_sha256", Path(__file__)),
                      ("weighted_engine_sha256", Path(weighted.__file__)),
                      ("sgpr_loader_sha256", Path(real.__file__)),
                      ("svgp_loader_sha256", Path(anchor.__file__))):
        if plan[key] != real.sha256(path):
            raise ValueError(f"planned evaluator changed: {key}")
    return plan


def verify_run_inputs(plan: dict, row: dict, run: Path,
                      record: Path) -> dict[str, str]:
    frozen = Path(row["root"])
    paths = [record, run / "params.npz", frozen / "EXPECTED_HEAD"]
    if row["method"].startswith(("RBT-", "SBT-", "METIS-")):
        paths.append(run / "final_blocks.npz")
    hashes = {str(path.relative_to(frozen)): real.sha256(path) for path in paths}
    if row["source_kind"] == "gibbs_anchor":
        history = json.loads(Path(plan["gibbs_anchor_prediction_plan"]).read_text())
        if real.sha256(Path(plan["gibbs_anchor_prediction_plan"])) != (
                plan["gibbs_anchor_prediction_plan_sha256"]):
            raise ValueError("Gibbs anchor provenance plan changed")
        for name, digest in hashes.items():
            if history["training_inputs"].get(name) != digest:
                raise ValueError(f"Gibbs anchor certified input changed: {name}")
    return hashes


def load_frozen_blocks(run: Path, family: str) -> list[np.ndarray]:
    from common import terms

    with np.load(run / "final_blocks.npz") as archive:
        if family == "sgpr":
            import jax.numpy as jnp

            packed = tuple(jnp.asarray(archive[key])
                           for key in ("coeffs", "idx", "mask"))
            blocks = terms.unpack_terms(packed)
        else:
            blocks = [row[mask > 0.5] for row, mask in
                      zip(archive["idx"], archive["mask"])]
    return [np.asarray(block, dtype=np.int32) for block in blocks]


def evaluate(args: argparse.Namespace) -> int:
    import jax

    output = args.output.resolve()
    plan = load_plan(output)
    row = plan["runs"][args.index]
    frozen = Path(row["root"])
    source_sim = Path(row["source_sim"])
    real.guard_separate(output, [frozen, source_sim.parent])
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("one usable assigned GPU is required")
    for name, digest in plan["sources"][row["source_kind"]]["files"].items():
        if real.sha256(source_sim / name) != digest:
            raise ValueError(f"historical source changed: {name}")
    run = frozen / row["relative"]
    metadata_path = record_path(run)
    record = json.loads(metadata_path.read_text())
    expected = plan["expected_heads"]["sgpr" if row["family"] == "sgpr"
                                      else "svgp"]
    if record["config"]["code_commit"] != expected:
        raise ValueError("checkpoint commit differs from planned source")
    hashes = verify_run_inputs(plan, row, run, metadata_path)
    identity = dict(implementation=IMPLEMENTATION,
                    evaluator_sha256=plan["evaluator_sha256"],
                    weighted_engine_sha256=plan["weighted_engine_sha256"],
                    source_kind=row["source_kind"],
                    source_commit=expected, config_id=record["config_id"],
                    inputs=hashes, weight_power=plan["weight_power"],
                    T_block_size=plan["block_size_for_T"],
                    test_chunk=args.test_chunk,
                    eig_cutoff_scale=args.eig_cutoff_scale)
    destination = output / row["output_relative"] / "prediction.json"
    if destination.is_file():
        if json.loads(destination.read_text()).get("identity") == identity:
            print(f"CACHED {row['output_relative']}", flush=True)
            return 0
        raise ValueError(f"different prediction already exists: {destination}")

    if row["family"] == "sgpr":
        components, frozen_blocks, ys, meta = real.load_components(
            row, run, record, source_sim)
        params = None
    else:
        components, params, ys, meta = anchor.load_components_all(
            row, run, record, source_sim)
        frozen_blocks = None
    from common import predictors, terms

    begin = time.perf_counter()
    if row["method"].startswith("T-"):
        if params is None:
            import jax.numpy as jnp

            with np.load(run / "params.npz") as archive:
                params = {key: jnp.asarray(archive[key])
                          for key in archive.files}
        blocks = anchor.anchor_blocks(
            np.asarray(jax.device_get(components["x"])), params,
            plan["block_size_for_T"], row["kernel"])
        partition_source = "prediction-time-inducing-anchor"
    else:
        blocks = (load_frozen_blocks(run, row["family"])
                  if frozen_blocks is None else
                  [np.asarray(b, dtype=np.int32) for b in frozen_blocks])
        partition_source = "frozen-fitted-blocks"
    terms.assert_partition(blocks, len(components["x"]))
    search_seconds = time.perf_counter() - begin
    duplicates = weighted.cross_block_duplicate_groups(
        np.asarray(jax.device_get(components["x"])), blocks)
    begin = time.perf_counter()
    _, contributions, distances, timings = weighted.correction_matrices(
        components, blocks, method=row["method"], test_chunk=args.test_chunk,
        eig_cutoff_scale=args.eig_cutoff_scale)
    weights = weighted.inverse_distance_weights(distances, plan["weight_power"])
    s2 = float(jax.device_get(components["s2"]))
    d_test = np.asarray(jax.device_get(components["d_test"])).reshape(-1)
    qvar = np.asarray(jax.device_get(components["qvar"])).reshape(-1)
    mean = np.asarray(jax.device_get(components["mean"])).reshape(-1)
    raw_residual = d_test - np.sum(weights * contributions, axis=1)
    noise_variance = qvar + raw_residual + s2
    finite_joint_valid = bool(
        np.isfinite(noise_variance).all()
        and np.min(noise_variance) > 0
        and np.min(raw_residual) >= -weighted.NEGATIVE_TOLERANCE
        and np.max(np.abs(weights.sum(axis=1) - 1)) < 1e-10)
    multiple_zero = int(np.count_nonzero((distances <= 1e-12).sum(axis=1) > 1))
    input_kernel_compatible = bool(finite_joint_valid and duplicates == 0
                                   and multiple_zero == 0)
    rmse = nll = None
    if finite_joint_valid:
        noise_variance = qvar + np.maximum(raw_residual, 0.0) + s2
        rmse, nll = predictors.gaussian_rmse_nll(mean, noise_variance, ys)
    for name, digest in hashes.items():
        if real.sha256(frozen / name) != digest:
            raise ValueError(f"frozen input changed during prediction: {name}")
    payload = dict(
        identity=identity, family=row["family"], kernel=row["kernel"],
        dataset=row["dataset"], method=row["method"], M=row["M"],
        seed=row["seed"], valid=finite_joint_valid,
        input_kernel_compatible=input_kernel_compatible,
        rmse=None if rmse is None else float(rmse),
        nll=None if nll is None else float(nll),
        rmse_orig=None if rmse is None else float(rmse * meta["y_std"]),
        nll_orig=None if nll is None else float(nll + np.log(meta["y_std"])),
        partition=dict(source=partition_source,
                       sha256=anchor.partition_digest(blocks),
                       sizes=[len(block) for block in blocks],
                       cross_block_duplicate_training_groups=duplicates,
                       search_seconds=search_seconds),
        diagnostics=dict(
            min_raw_residual=float(np.min(raw_residual)),
            negative_raw_residual_count=int(np.count_nonzero(
                raw_residual < -weighted.NEGATIVE_TOLERANCE)),
            max_weight_sum_error=float(np.max(np.abs(weights.sum(axis=1)-1))),
            multiple_zero_distance_test_points=multiple_zero,
            effective_blocks_mean=float(np.mean(1/np.sum(weights*weights,
                                                         axis=1))),
            **timings),
        device_kind=jax.devices()[0].device_kind,
        prediction_seconds=time.perf_counter()-begin,
        interpretation=("PSD finite indexed Gaussian weighted-block proxy; "
                        "not the exact frozen variational predictive covariance"),
    )
    real.atomic_json(destination, payload)
    print(f"PREDICTED {row['output_relative']} valid={finite_joint_valid} "
          f"input_kernel_compatible={input_kernel_compatible} "
          f"nll={nll} elapsed={payload['prediction_seconds']:.1f}s", flush=True)
    return 0 if finite_joint_valid else 2


def worker(args: argparse.Namespace) -> int:
    plan = load_plan(args.output)
    assigned = [(index, row) for index, row in enumerate(plan["runs"])
                if index % args.workers == args.worker]
    failures = []
    for position, (index, row) in enumerate(assigned, 1):
        print(f"[{position}/{len(assigned)}] {row['output_relative']}", flush=True)
        result = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "evaluate",
             "--output", str(args.output), "--index", str(index),
             "--test-chunk", str(args.test_chunk),
             "--eig-cutoff-scale", str(args.eig_cutoff_scale)],
            check=False)
        if result.returncode:
            failures.append(dict(index=index, exit_code=result.returncode))
        real.atomic_json(args.output / "state" / f"worker{args.worker}.json",
                         dict(assigned=len(assigned), processed=position,
                              failures=failures,
                              complete=position == len(assigned)))
    return 1 if failures else 0


def summary(args: argparse.Namespace) -> int:
    plan = load_plan(args.output)
    grouped: dict[tuple[str, str, str], dict] = {}
    saved = invalid = incompatible = 0
    for row in plan["runs"]:
        path = args.output / row["output_relative"] / "prediction.json"
        if not path.is_file():
            continue
        record = json.loads(path.read_text())
        saved += 1
        invalid += not record["valid"]
        incompatible += not record["input_kernel_compatible"]
        key = (row["family"], row["kernel"], row["method"])
        bucket = grouped.setdefault(key, dict(saved=0, invalid=0))
        bucket["saved"] += 1
        bucket["invalid"] += not record["valid"]
    print(f"SAVED={saved}/{len(plan['runs'])} FINITE_JOINT_INVALID={invalid} "
          f"INPUT_KERNEL_INCOMPATIBLE={incompatible}")
    for key, value in sorted(grouped.items()):
        print(f"{key[0]} {key[1]} {key[2]} "
              f"saved={value['saved']} invalid={value['invalid']}")
    return 0 if saved == len(plan["runs"]) and invalid == 0 else 2


def certify(args: argparse.Namespace) -> int:
    if summary(args):
        raise ValueError("cannot certify an incomplete or numerically invalid campaign")
    plan = load_plan(args.output)
    if real.sha256(Path(plan["gibbs_anchor_prediction_plan"])) != (
            plan["gibbs_anchor_prediction_plan_sha256"]):
        raise ValueError("Gibbs anchor prediction provenance changed")
    for kind, source in plan["sources"].items():
        for name, digest in source["files"].items():
            if real.sha256(Path(source["root"]) / name) != digest:
                raise ValueError(f"frozen source changed: {kind}/{name}")
    certified = {}
    incompatible = 0
    for row in plan["runs"]:
        frozen = Path(row["root"])
        run = frozen / row["relative"]
        output = args.output / row["output_relative"] / "prediction.json"
        record = json.loads(output.read_text())
        if record["identity"]["config_id"] != json.loads(
                record_path(run).read_text())["config_id"]:
            raise ValueError(f"different training configuration: {run}")
        if not record["valid"]:
            raise ValueError(f"invalid prediction: {output}")
        for name, digest in record["identity"]["inputs"].items():
            if real.sha256(frozen / name) != digest:
                raise ValueError(f"frozen input changed: {name}")
        incompatible += not record["input_kernel_compatible"]
        certified[row["output_relative"]] = real.sha256(output)
    certificate = dict(implementation=IMPLEMENTATION,
                       plan_sha256=real.sha256(args.output / "plan.json"),
                       predictions=certified, count=len(certified),
                       finite_joint_valid=len(certified),
                       input_kernel_incompatible=incompatible,
                       scope="prediction-only, frozen training inputs unchanged",
                       caveat=("PSD finite indexed Gaussian proxy; runs with "
                               "duplicate inputs split across blocks do not "
                               "define an input-only GP kernel"))
    destination = args.output / "WEIGHTED_PREDICTION_CERTIFICATE.json"
    if destination.is_file():
        if json.loads(destination.read_text()) != certificate:
            raise ValueError("different certificate already exists")
    else:
        real.atomic_json(destination, certificate)
    print(f"CERTIFIED={len(certified)} INPUT_KERNEL_INCOMPATIBLE={incompatible} "
          f"SHA256={real.sha256(destination)}", flush=True)
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
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--output", required=True, type=Path)
    evaluate_parser.add_argument("--index", required=True, type=int)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--output", required=True, type=Path)
    worker_parser.add_argument("--worker", required=True, type=int)
    worker_parser.add_argument("--workers", type=int, default=5)
    for command in (evaluate_parser, worker_parser):
        command.add_argument("--test-chunk", type=int, default=512)
        command.add_argument("--eig-cutoff-scale", type=float, default=10.0)
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
