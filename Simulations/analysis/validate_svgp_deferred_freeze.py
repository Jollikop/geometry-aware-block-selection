#!/usr/bin/env python3
"""Validate and optionally hash a split SVGP training/prediction freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


M_VALUES = (64, 128, 256, 512)
METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")
BLOCK_METHODS = {"RBT-SVGP", "SBT-SVGP"}


def fingerprint(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_record(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def valid_npz(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.infolist())
    except (OSError, zipfile.BadZipFile):
        return False


def record_errors(record: dict[str, object] | None, expected_head: str) -> list[str]:
    if not record:
        return ["missing or invalid JSON record"]
    config = record.get("config")
    if not isinstance(config, dict):
        return ["missing config"]
    errors = []
    if record.get("config_id") != fingerprint(config):
        errors.append("config fingerprint mismatch")
    if config.get("code_commit") != expected_head:
        errors.append("numerical reference commit mismatch")
    if record.get("training_complete") is not True and "rmse" not in record:
        errors.append("training not marked complete")
    return errors


def gpu_kind(record: dict[str, object]) -> str:
    runtime = record.get("prediction_runtime") or record.get("runtime") or {}
    devices = runtime.get("jax_devices", []) if isinstance(runtime, dict) else []
    kinds = {
        str(device.get("kind"))
        for device in devices
        if isinstance(device, dict) and device.get("platform") == "gpu"
    }
    return next(iter(kinds)) if len(kinds) == 1 else ""


def expected_paths(root: Path, stage: str, kernels: list[str], datasets: list[str]):
    for kernel in kernels:
        for dataset in datasets:
            for m in M_VALUES:
                for method in METHODS:
                    output_method = (
                        f"{method}-to-SVGP" if stage == "transplant" else method
                    )
                    for seed in range(10):
                        yield (
                            kernel,
                            dataset,
                            m,
                            method,
                            seed,
                            root / "svgp" / stage / "runs" / kernel / dataset
                            / f"M{m}" / output_method / f"seed{seed}",
                        )


def donor_params_path(
    root: Path,
    kernel: str,
    dataset: str,
    m: int,
    method: str,
    seed: int,
) -> Path:
    stage = "stationary" if kernel in {"rbf", "matern52"} else "extended"
    return (
        root / "svgp" / stage / "runs" / kernel / dataset / f"M{m}"
        / method / f"seed{seed}" / "params.npz"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--stationary-kernels", nargs="+", default=["rbf", "matern52"]
    )
    parser.add_argument(
        "--extended-kernels", nargs="+", default=["amplitude-rbf", "gibbs"]
    )
    parser.add_argument(
        "--transplant-kernels",
        nargs="+",
        default=["rbf", "matern52", "amplitude-rbf", "gibbs"],
    )
    parser.add_argument("--require-transplant-predictions", action="store_true")
    parser.add_argument("--prediction-gpu-contains", default="")
    parser.add_argument("--write-manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    expected_head_path = root / "EXPECTED_HEAD"
    if not expected_head_path.is_file():
        raise SystemExit(f"missing {expected_head_path}")
    expected_head = expected_head_path.read_text().strip()
    errors: list[str] = []
    artifacts: set[Path] = {expected_head_path}
    prediction_gpus: set[str] = set()
    prediction_policies: set[tuple[object, object]] = set()
    counts = {
        "stationary_training": 0,
        "extended_training": 0,
        "transplant_training": 0,
        "transplant_prediction": 0,
    }

    for stage, kernels in (
        ("stationary", args.stationary_kernels),
        ("extended", args.extended_kernels),
        ("transplant", args.transplant_kernels),
    ):
        for kernel, dataset, m, method, seed, run_dir in expected_paths(
            root, stage, kernels, args.datasets
        ):
            metrics_path = run_dir / "metrics.json"
            training_path = run_dir / "training.json"
            metrics = load_record(metrics_path)
            training = load_record(training_path)
            record = training or metrics
            prefix = f"{stage}/{kernel}/{dataset}/M{m}/{method}/seed{seed}"
            problems = record_errors(record, expected_head)
            for problem in problems:
                errors.append(f"{prefix}: {problem}")
            if problems:
                continue

            required = [run_dir / "params.npz", run_dir / "curve.npz"]
            if method in BLOCK_METHODS and stage != "transplant":
                required.append(run_dir / "final_blocks.npz")
            missing = [path for path in required if not valid_npz(path)]
            if missing:
                errors.extend(f"{prefix}: missing/invalid {path.name}" for path in missing)
                continue
            artifacts.update(required)
            artifacts.add(training_path if training else metrics_path)
            counts[f"{stage}_training"] += 1

            if stage == "extended" and kernel == "gibbs":
                config = record["config"]
                expected = {
                    "epochs": 300,
                    "init_from_kernel": "rbf",
                    "refresh_every": 40,
                    "refresh_until": 81,
                    "warp_gate_fraction": 0.0,
                }
                for key, value in expected.items():
                    if config.get(key) != value:
                        errors.append(
                            f"{prefix}: Gibbs protocol {key}={config.get(key)!r}, "
                            f"expected {value!r}"
                        )
                source_params = donor_params_path(
                    root, "rbf", dataset, m, method, seed
                )
                expected_hash = (
                    f"sha256:{sha256(source_params)}"
                    if source_params.is_file()
                    else ""
                )
                if not expected_hash:
                    errors.append(f"{prefix}: warm-start RBF parameters missing")
                elif config.get("init_params_sha256") != expected_hash:
                    errors.append(
                        f"{prefix}: warm-start RBF checkpoint hash mismatch"
                    )

            if stage == "transplant":
                config = record["config"]
                source_params = donor_params_path(
                    root, kernel, dataset, m, method, seed
                )
                expected_hash = (
                    f"sha256:{sha256(source_params)}"
                    if source_params.is_file()
                    else ""
                )
                if not expected_hash:
                    errors.append(f"{prefix}: donor parameters missing")
                elif config.get("donor_params_sha256") != expected_hash:
                    errors.append(f"{prefix}: donor checkpoint hash mismatch")

            if stage == "transplant" and metrics is not None:
                problems = record_errors(metrics, expected_head)
                if training and metrics.get("config_id") != training.get("config_id"):
                    problems.append("prediction/training config mismatch")
                if problems or "rmse" not in metrics or "nll" not in metrics:
                    errors.append(
                        f"{prefix}: invalid predictive metrics"
                        + (f" ({'; '.join(problems)})" if problems else "")
                    )
                else:
                    counts["transplant_prediction"] += 1
                    artifacts.add(metrics_path)
                    kind = gpu_kind(metrics)
                    prediction_gpus.add(kind)
                    execution = metrics.get("prediction_execution") or {}
                    prediction_policies.add(
                        (execution.get("test_chunk"), metrics.get("prediction_reps"))
                    )
            elif stage == "transplant" and args.require_transplant_predictions:
                errors.append(f"{prefix}: transplant prediction missing")

    if args.require_transplant_predictions:
        if len(prediction_gpus) != 1 or "" in prediction_gpus:
            errors.append(
                f"prediction hardware is not one identifiable GPU kind: "
                f"{sorted(prediction_gpus)}"
            )
        if args.prediction_gpu_contains and not all(
            args.prediction_gpu_contains.lower() in kind.lower()
            for kind in prediction_gpus
        ):
            errors.append(
                f"prediction GPU does not contain {args.prediction_gpu_contains!r}: "
                f"{sorted(prediction_gpus)}"
            )
        if len(prediction_policies) != 1:
            errors.append(
                f"prediction chunk/repetition policy is mixed: "
                f"{sorted(prediction_policies, key=str)}"
            )

    manifest = {
        "schema": "svgp-split-freeze-v1",
        "root": str(root),
        "numerical_reference_commit": expected_head,
        "datasets": args.datasets,
        "stationary_kernels": args.stationary_kernels,
        "extended_kernels_training_only": args.extended_kernels,
        "transplant_kernels": args.transplant_kernels,
        "counts": counts,
        "transplant_predictions_required": args.require_transplant_predictions,
        "prediction_gpu_kinds": sorted(prediction_gpus),
        "prediction_policies": sorted(
            [list(policy) for policy in prediction_policies], key=str
        ),
        "release_ready": not errors,
        "errors": errors[:200],
    }
    if not errors and args.write_manifest:
        manifest["artifacts"] = {
            str(path.relative_to(root)): f"sha256:{sha256(path)}"
            for path in sorted(artifacts)
        }
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({key: value for key, value in manifest.items() if key != "artifacts"}, indent=2))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
