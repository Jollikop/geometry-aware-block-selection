#!/usr/bin/env python3
"""Read-only, provenance-bound dense variational predictions from frozen runs.

For T/BT this evaluates D_ff^{-1/2}(I-M)D_ff^{-1/2} with the *full*
training residual D_ff. It does not substitute a diagonal/block residual
matrix at test time. Each checkpoint is evaluated in a fresh process; all
results are written under a separate output root, never into a frozen run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


SGPR_METHODS = ("SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR", "METIS-SGPR")
SVGP_METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")
IMPLEMENTATION = "real-dff-variational-predictor-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return "sha256:" + digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    os.replace(temporary, path)


def guard_separate(output: Path, sources: list[Path]) -> None:
    output = output.resolve()
    for source in sources:
        source = source.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"output is not separate from source: {source}")


def build_plan(args) -> dict:
    sgpr_root = args.sgpr_root.resolve()
    svgp_root = args.svgp_root.resolve()
    source_sgpr = args.source_sgpr.resolve()
    source_svgp = args.source_svgp.resolve()
    output = args.output.resolve()
    guard_separate(output, [sgpr_root, svgp_root, source_sgpr, source_svgp])
    runs = []

    for method in SGPR_METHODS:
        for seed in range(10):
            relative = Path("runs/rbf/kin40k/M512") / method / f"seed{seed}"
            run = sgpr_root / relative
            needed = ["metrics.json", "params.npz"]
            if method in {"RBT-SGPR", "SBT-SGPR", "METIS-SGPR"}:
                needed.append("final_blocks.npz")
            if not all((run / name).is_file() for name in needed):
                raise FileNotFoundError(f"incomplete frozen run: {run}")
            runs.append({
                "family": "sgpr", "root": str(sgpr_root),
                "source_sim": str(source_sgpr / "Simulations"),
                "relative": str(relative),
                "output_relative": str(Path("sgpr/rbf/kin40k/M512") / method / f"seed{seed}"),
                "method": method, "kernel": "rbf", "M": 512, "seed": seed,
            })

    for kernel, stage in (("rbf", "stationary"), ("amplitude-rbf", "extended")):
        for m in (64, 128, 256, 512):
            for method in SVGP_METHODS:
                for seed in range(10):
                    relative = (Path("svgp") / stage / "runs" / kernel / "pol"
                                / f"M{m}" / method / f"seed{seed}")
                    run = svgp_root / relative
                    needed = ["metrics.json", "params.npz"]
                    if method in ("RBT-SVGP", "SBT-SVGP"):
                        needed.append("final_blocks.npz")
                    if not all((run / name).is_file() for name in needed):
                        raise FileNotFoundError(f"incomplete frozen run: {run}")
                    runs.append({
                        "family": "svgp", "root": str(svgp_root),
                        "source_sim": str(source_svgp / "Simulations"),
                        "relative": str(relative),
                        "output_relative": str(Path("svgp") / kernel / "pol"
                                               / f"M{m}" / method / f"seed{seed}"),
                        "method": method, "kernel": kernel, "M": m,
                        "seed": seed,
                    })

    plan = {
        "implementation": IMPLEMENTATION,
        "evaluator_sha256": sha256(Path(__file__)),
        "sgpr_expected_head": (sgpr_root / "EXPECTED_HEAD").read_text().strip(),
        "svgp_expected_head": (svgp_root / "EXPECTED_HEAD").read_text().strip(),
        "source_sgpr_head": subprocess.check_output(
            ["git", "-C", str(source_sgpr), "rev-parse", "HEAD"], text=True).strip(),
        "source_svgp_head": subprocess.check_output(
            ["git", "-C", str(source_svgp), "rev-parse", "HEAD"], text=True).strip(),
        "runs": runs,
    }
    if plan["source_sgpr_head"] != plan["sgpr_expected_head"]:
        raise ValueError("SGPR source checkout does not match the frozen commit")
    if plan["source_svgp_head"] != plan["svgp_expected_head"]:
        raise ValueError("SVGP source checkout does not match the frozen commit")
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise ValueError("different plan already exists; choose a fresh output root")
    else:
        atomic_json(plan_path, plan)
    print(f"PLAN={plan_path} SGPR={sum(r['family']=='sgpr' for r in runs)} "
          f"SVGP={sum(r['family']=='svgp' for r in runs)} TOTAL={len(runs)}", flush=True)
    return plan


def load_components(item: dict, run: Path, record: dict, source_sim: Path):
    import jax.numpy as jnp
    sys.path.insert(0, str(source_sim))
    cfg = record["config"]
    if item["family"] == "sgpr":
        from sgpr import run_benchmark as runner
        from sgpr import jax_collapsed as engine
        from common import terms

        if runner.config_fingerprint(cfg) != record["config_id"]:
            raise ValueError("SGPR configuration fingerprint mismatch")
        if runner.dataset_signature("kin40k") != cfg["data_signature"]:
            raise ValueError("SGPR dataset hash mismatch")
        x, y, xs, ys, meta = runner.load_benchmark_dataset(
            "kin40k", cfg["split_seed"], cfg["n_train"], cfg["n_test"],
            split_rng=cfg["split_rng"])
        with __import__("numpy").load(run / "params.npz") as archive:
            params = {key: jnp.asarray(archive[key]) for key in archive.files}
        components = engine.collapsed_predictive_components(
            params, x, y, xs, kernel_name=cfg["kernel"])
        blocks = None
        if item["method"] in ("RBT-SGPR", "SBT-SGPR", "METIS-SGPR"):
            with __import__("numpy").load(run / "final_blocks.npz") as archive:
                packed = tuple(jnp.asarray(archive[key])
                               for key in ("coeffs", "idx", "mask"))
            blocks = terms.unpack_terms(packed)
            terms.assert_partition(blocks, len(x))
    else:
        from svgp import run_benchmark as runner
        from svgp import engine
        import numpy as np

        if runner.config_fingerprint(cfg) != record["config_id"]:
            raise ValueError("SVGP configuration fingerprint mismatch")
        raw = runner.dm.raw_cache_path("pol", runner.DATASET_DIR)
        if sha256(raw) != cfg["data_signature"]:
            raise ValueError("SVGP dataset hash mismatch")
        X, y = runner.dm.load_raw("pol", data_dir=runner.DATASET_DIR)
        x, _, xs, ys, meta = runner.jd.split_standardize(
            X, y, cfg["split_seed"], cfg["n_train"], cfg["n_test"],
            angle_cols=runner.dm.angle_cols_for("pol"), split_rng=cfg["split_rng"])
        for key in ("train_indices_sha256", "test_indices_sha256",
                    "preprocessing_protocol"):
            if meta[key] != cfg[key]:
                raise ValueError(f"SVGP split mismatch: {key}")
        with np.load(run / "params.npz") as archive:
            params = {key: jnp.asarray(archive[key]) for key in archive.files}
        blocks = None
        if item["method"] in ("RBT-SVGP", "SBT-SVGP"):
            with np.load(run / "final_blocks.npz") as archive:
                block_idx = archive["idx"]
                block_mask = archive["mask"]
            blocks = [row[mask_row > 0.5] for row, mask_row in
                      zip(block_idx, block_mask)]
        if item["method"] in ("RBT-SVGP", "SBT-SVGP") and blocks is None:
            raise ValueError("fitted SVGP partition missing")
        components = engine._predictive_components(
            params, x, xs, cfg["kernel"], runner.JITTER)
    return components, blocks, ys, meta


def dense_variational_prediction(components, blocks, method: str,
                                 *, test_chunk: int, eig_cutoff_scale: float):
    """Return exact-D variance under the fitted diagonal/block M.

    A spectral pseudoinverse is used only for numerically null eigenvalues of
    D_ff. The truncation threshold and discarded test-cross energy are saved
    so a run cannot silently claim an unqualified exact inverse.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    n = int(components["x"].shape[0])
    n_test = int(components["xs"].shape[0])
    s2 = float(jax.device_get(components["s2"]))
    sparse = method in ("SGPR", "SVGP")
    t0 = time.perf_counter()
    if sparse:
        eigenvalues = eigenvectors = inv_sqrt = retained = None
        weights = indices = None
        eigen_info = {"computed": False}
    else:
        x = components["x"]
        a = components["a_train"]
        residual = components["kernel_fn"](x, x) - a.T @ a
        residual = 0.5 * (residual + residual.T)
        eigenvalues, eigenvectors = jnp.linalg.eigh(residual)
        eigenvalues_np = np.asarray(jax.device_get(eigenvalues))
        max_eig = float(max(eigenvalues_np[-1], 0.0))
        cutoff = float(np.finfo(np.float64).eps * n * max_eig
                       * eig_cutoff_scale)
        if float(eigenvalues_np[0]) < -cutoff:
            raise ValueError(f"D_ff has materially negative eigenvalue: "
                             f"{eigenvalues_np[0]:.6g} < {-cutoff:.6g}")
        retained = eigenvalues > cutoff
        inv_sqrt = jnp.where(retained, 1.0 / jnp.sqrt(
            jnp.maximum(eigenvalues, cutoff)), 0.0)
        eigen_info = {
            "computed": True, "n": n, "min_eigenvalue": float(eigenvalues_np[0]),
            "max_eigenvalue": max_eig, "cutoff": cutoff,
            "effective_rank": int(np.count_nonzero(eigenvalues_np > cutoff)),
        }
        weights = indices = None
        if method.startswith(("RBT-", "SBT-", "METIS-")):
            if not blocks:
                raise ValueError("block method has no fitted partition")
            block_indices = [np.asarray(b, dtype=np.int32).reshape(-1)
                             for b in blocks]
            order = np.concatenate(block_indices)
            if len(order) != n or len(np.unique(order)) != n:
                raise ValueError("fitted blocks do not partition the training set")
            width = max(map(len, block_indices))
            padded_idx = np.zeros((len(block_indices), width), dtype=np.int32)
            padded_weight = []
            for b, idx in enumerate(block_indices):
                padded_idx[b, :len(idx)] = idx
                db = residual[jnp.ix_(jnp.asarray(idx), jnp.asarray(idx))]
                lam, vec = jnp.linalg.eigh(db)
                lam = jnp.maximum(lam, 0.0)
                gain = lam / (lam + s2)
                wb = (vec * gain[None, :]) @ vec.T
                padded_weight.append(jnp.pad(
                    wb, ((0, width-len(idx)), (0, width-len(idx)))))
            indices = jnp.asarray(padded_idx)
            weights = jnp.stack(padded_weight)
            jax.block_until_ready(weights)
    prepare_s = time.perf_counter() - t0

    means, variances = [], []
    min_latent = float("inf")
    max_discarded_fraction = 0.0
    predict_start = time.perf_counter()
    for start in range(0, n_test, test_chunk):
        stop = min(start + test_chunk, n_test)
        qvar = components["qvar"][start:stop]
        d_test = components["d_test"][start:stop]
        if sparse:
            correction = jnp.zeros_like(d_test)
        else:
            cross = (components["kernel_fn"](
                components["xs"][start:stop], components["x"])
                - components["a_test"][:, start:stop].T
                  @ components["a_train"])
            coeff = eigenvectors.T @ cross.T
            discarded = jnp.sum(jnp.where(retained[:, None], 0.0,
                                          coeff * coeff), axis=0)
            total = jnp.sum(coeff * coeff, axis=0)
            max_discarded_fraction = max(
                max_discarded_fraction,
                float(jax.device_get(jnp.max(discarded / jnp.maximum(total, 1e-30)))))
            white = eigenvectors @ (inv_sqrt[:, None] * coeff)
            if method.startswith("T-"):
                diag = components["d_train"]
                gain = diag / (diag + s2)
                correction = jnp.sum(gain[:, None] * white**2, axis=0)
            else:
                part = white[indices]
                correction = jnp.einsum("bik,bij,bjk->k", part, weights, part)
        latent = qvar + d_test - correction
        min_latent = min(min_latent, float(jax.device_get(jnp.min(latent))))
        means.append(np.asarray(jax.device_get(
            components["mean"][start:stop])).reshape(-1))
        variances.append(np.asarray(jax.device_get(latent + s2)).reshape(-1))
    predict_s = time.perf_counter() - predict_start
    mean = np.concatenate(means)
    variance = np.concatenate(variances)
    if not (np.isfinite(mean).all() and np.isfinite(variance).all()):
        raise ValueError("nonfinite real predictor result")
    if min_latent < -1e-7:
        raise ValueError(f"materially negative latent variance: {min_latent:.6g}")
    if np.min(variance) <= 0:
        raise ValueError("nonpositive observation variance")
    return mean, variance, {
        "preparation_s": prepare_s, "prediction_s": predict_s,
        "min_latent_variance": min_latent,
        "max_discarded_cross_energy_fraction": max_discarded_fraction,
        "eigenspectrum": eigen_info,
    }


def evaluate(plan_path: Path, index: int, output: Path,
             *, test_chunk: int, eig_cutoff_scale: float) -> None:
    import jax
    import numpy as np

    plan = json.loads(plan_path.read_text())
    item = plan["runs"][index]
    source_sim = Path(item["source_sim"]).resolve()
    # This process handles one checkpoint only, so pinned modules cannot mix.
    sys.path.insert(0, str(source_sim))
    from common import predictors as pr
    root = Path(item["root"]).resolve()
    run = (root / item["relative"]).resolve()
    guard_separate(output, [root, source_sim.parent])
    if root not in run.parents:
        raise ValueError("run path escapes frozen root")
    if plan["evaluator_sha256"] != sha256(Path(__file__)):
        raise ValueError("evaluator changed after plan creation")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError(f"exactly one assigned GPU required, got {devices}")
    record_path = run / "training.json"
    if not record_path.is_file():
        record_path = run / "metrics.json"
    record = json.loads(record_path.read_text())
    cfg = record["config"]
    expected = (root / "EXPECTED_HEAD").read_text().strip()
    if cfg["code_commit"] != expected:
        raise ValueError("checkpoint code commit differs from frozen EXPECTED_HEAD")
    if item["family"] == "svgp" and not (
            record.get("training_complete") is True or "rmse" in record):
        raise ValueError("SVGP checkpoint is not a completed training result")
    predictor = "sparse" if item["method"] in ("SGPR", "SVGP") else (
        "diagonal" if item["method"].startswith("T-") else "block")
    required = [record_path, run / "params.npz", root / "EXPECTED_HEAD"]
    if predictor == "block":
        required.append(run / "final_blocks.npz")
    inputs = {str(p.relative_to(root)): sha256(p) for p in required}
    identity = {
        "implementation": IMPLEMENTATION, "evaluator_sha256": sha256(Path(__file__)),
        "inputs": inputs, "config_id": record["config_id"],
        "source_commit": expected, "method": item["method"],
        "kernel": item["kernel"], "M": item["M"], "seed": item["seed"],
        "predictor": predictor, "residual_matrix": "full-Dff-symmetric-eigenroot",
        "test_chunk": test_chunk, "eig_cutoff_scale": eig_cutoff_scale,
    }
    destination = output / item["output_relative"] / "prediction.json"
    if destination.is_file():
        if json.loads(destination.read_text()).get("identity") == identity:
            print(f"CACHED {item['output_relative']}", flush=True)
            return
        raise ValueError("different result already exists; choose a new output root")
    if destination.exists():
        raise ValueError("prediction destination exists but is not a file")
    components, blocks, ys, meta = load_components(item, run, record, source_sim)
    mean, variance, diagnostics = dense_variational_prediction(
        components, blocks, item["method"], test_chunk=test_chunk,
        eig_cutoff_scale=eig_cutoff_scale)
    rmse, nll = pr.gaussian_rmse_nll(mean, variance, ys)
    for path in required:
        if sha256(path) != inputs[str(path.relative_to(root))]:
            raise ValueError("frozen input changed during prediction")
    atomic_json(destination, {
        "identity": identity, "rmse": float(rmse), "nll": float(nll),
        "rmse_orig": float(rmse * meta["y_std"]),
        "nll_orig": float(nll + np.log(meta["y_std"])),
        "diagnostics": diagnostics,
        "donor_convergence_passed": record.get("convergence_passed"),
        "device_kind": devices[0].device_kind,
    })
    print(f"SAVED {item['output_relative']} rmse={rmse:.6f} nll={nll:.6f} "
          f"rank={diagnostics['eigenspectrum'].get('effective_rank', 'na')} "
          f"prepare={diagnostics['preparation_s']:.1f}s "
          f"predict={diagnostics['prediction_s']:.1f}s", flush=True)


def worker(args) -> int:
    plan_path = args.output / "plan.json"
    plan = json.loads(plan_path.read_text())
    assigned = [i for i in range(len(plan["runs"]))
                if i % args.workers == args.worker]
    failed = []
    for position, index in enumerate(assigned, 1):
        item = plan["runs"][index]
        print(f"[{position}/{len(assigned)}] {item['output_relative']}", flush=True)
        result = subprocess.run([
            sys.executable, "-u", str(Path(__file__).resolve()), "evaluate",
            "--plan", str(plan_path), "--index", str(index),
            "--output", str(args.output), "--test-chunk", str(args.test_chunk),
            "--eig-cutoff-scale", str(args.eig_cutoff_scale),
        ], check=False)
        if result.returncode:
            failed.append({"index": index, "run": item["output_relative"],
                           "exit_code": result.returncode})
        atomic_json(args.output / "state" / f"worker{args.worker}.json", {
            "worker": args.worker, "assigned": len(assigned), "processed": position,
            "failed": failed, "complete": position == len(assigned),
        })
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--sgpr-root", type=Path, required=True)
    plan.add_argument("--svgp-root", type=Path, required=True)
    plan.add_argument("--source-sgpr", type=Path, required=True)
    plan.add_argument("--source-svgp", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--plan", type=Path, required=True)
    evaluate_parser.add_argument("--index", type=int, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--worker", type=int, required=True)
    worker_parser.add_argument("--workers", type=int, default=5)
    worker_parser.add_argument("--output", type=Path, required=True)
    for command in (evaluate_parser, worker_parser):
        command.add_argument("--test-chunk", type=int, default=32)
        command.add_argument("--eig-cutoff-scale", type=float, default=10.0)
    args = parser.parse_args()
    if args.command == "plan":
        build_plan(args)
        return 0
    if args.command == "evaluate":
        plan_path = args.plan.resolve()
        if plan_path.parent != args.output.resolve():
            raise ValueError("plan must be inside the separate output root")
        evaluate(plan_path, args.index, args.output.resolve(),
                 test_chunk=args.test_chunk,
                 eig_cutoff_scale=args.eig_cutoff_scale)
        return 0
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
