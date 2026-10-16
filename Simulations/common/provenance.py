"""Small, dependency-light helpers for experiment provenance."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "matplotlib",
    "tensorflow",
    "tf-keras",
    "tensorflow-probability",
    "gpflow",
    "check-shapes",
    "jax",
    "jaxlib",
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    # The CUDA compiler ships as an unpinned extra of jax[cuda12]: two hosts
    # with identical jax versions can hold different ptxas builds and emit
    # different device code. Recording it closes that blind spot.
    "nvidia-cuda-nvcc-cu12",
    "optax",
    "scikit-learn",
    "pandas",
    "pymetis",
    "uci-datasets",
)

RELEASE_PACKAGE_VERSIONS = {
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "matplotlib": "3.9.4",
    "pandas": "2.3.3",
    "scikit-learn": "1.6.1",
    "jax": "0.4.30",
    "jaxlib": "0.4.30",
    "jax-cuda12-plugin": "0.4.30",
    "jax-cuda12-pjrt": "0.4.30",
    # Matches the CUDA 12.8 driver of the compute nodes; a newer ptxas makes
    # XLA fall back to serial compilation and changes the emitted code.
    "nvidia-cuda-nvcc-cu12": "12.8.93",
    "optax": "0.2.4",
    "tensorflow": "2.16.2",
    "tf-keras": "2.16.0",
    "tensorflow-probability": "0.24.0",
    "gpflow": "2.10.1",
    "check-shapes": "1.1.1",
    "pymetis": "2023.1.1",
    # Not on PyPI. requirements.txt pins the commit; importlib.metadata still
    # reports the distribution version, so that is what a runtime manifest
    # carries and what this profile compares against.
    "uci-datasets": "1.1.0",
}
RELEASE_PYTHON_MINOR = "3.12"

SVGP_RELEASE_PACKAGES = (
    "numpy", "scipy", "pandas", "scikit-learn", "jax", "jaxlib", "optax",
    "uci-datasets",
)
SGPR_RELEASE_PACKAGES = (*SVGP_RELEASE_PACKAGES, "pymetis")
# The Snelson figure is GPflow/TensorFlow, but since it started drawing its
# bands with the shared residual predictor it also needs JAX at run time. The
# profile has to say so, or a jax-less environment produces a manifest that
# passes validation for a figure it could not have drawn.
SNELSON_RELEASE_PACKAGES = (
    "numpy", "scipy", "matplotlib", "jax", "jaxlib", "tensorflow", "tf-keras",
    "tensorflow-probability", "gpflow", "check-shapes",
)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return a content-based signature suitable for experiment fingerprints."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def config_fingerprint(config: dict[str, object]) -> str:
    """Return the canonical 16-hex experiment-configuration fingerprint."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def release_runtime_matches(
    runtime: dict[str, object], required: tuple[str, ...]
) -> bool:
    """Return whether a runtime exactly matches the pinned release profile."""
    python_version = str(runtime.get("python", ""))
    if ".".join(python_version.split(".")[:2]) != RELEASE_PYTHON_MINOR:
        return False
    packages = runtime.get("packages") if isinstance(runtime, dict) else None
    if not isinstance(packages, dict):
        return False
    if not all(
        packages.get(name) == RELEASE_PACKAGE_VERSIONS[name]
        for name in required
    ):
        return False
    devices = runtime.get("jax_devices")
    uses_gpu = isinstance(devices, list) and any(
        isinstance(device, dict) and device.get("platform") == "gpu"
        for device in devices
    )
    if uses_gpu:
        # ptxas is checked here rather than in the release tuples because it is
        # a Linux/GPU-only wheel: requiring it unconditionally would make every
        # CPU host, including a laptop replaying an analysis, fail the profile.
        cuda_packages = (
            "jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cuda-nvcc-cu12",
        )
        if not all(
            packages.get(name) == RELEASE_PACKAGE_VERSIONS[name]
            for name in cuda_packages
        ):
            return False
        if not runtime.get("nvidia_smi"):
            return False
    return True


def _git_output(project_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@lru_cache(maxsize=4)
def code_state(project_root: str | Path) -> dict[str, object]:
    """Return the commit and experiment-source state recorded with every run.

    The repository containing this project may have unrelated siblings. Only
    ``Simulations/`` affects numerical artifacts, but both tracked changes and
    non-ignored untracked source files there must invalidate an official run.
    """
    root = Path(project_root).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    status = _git_output(
        root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        "Simulations",
    )
    dirty = bool(status and status != "unknown")
    return {
        "commit": commit,
        "source_dirty": dirty,
        "source_paths": ["Simulations"],
        "dirty_entries": [] if not dirty else status.splitlines(),
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


@lru_cache(maxsize=1)
def runtime_manifest() -> dict[str, object]:
    """Capture software, accelerator, and CUDA visibility for one result."""
    manifest: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import jax

        manifest["jax_devices"] = [
            {
                "platform": device.platform,
                "kind": getattr(device, "device_kind", "unknown"),
                "id": int(device.id),
            }
            for device in jax.devices()
        ]
    except Exception as exc:  # diagnostics must not make a completed run fail
        manifest["jax_devices_error"] = f"{type(exc).__name__}: {exc}"
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()
        manifest["nvidia_smi"] = query
    except (OSError, subprocess.CalledProcessError):
        manifest["nvidia_smi"] = []
    return json.loads(json.dumps(manifest, sort_keys=True))
