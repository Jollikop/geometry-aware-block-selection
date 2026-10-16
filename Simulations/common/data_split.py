#!/usr/bin/env python3
"""Deterministic train/test preprocessing shared by both stacks.

Dataset discovery, download, caching and provenance belong to
``common.datasets``. This module starts from already loaded raw arrays and owns
only:

1. deterministic train/test indices;
2. standardisation fitted on the training set;
3. conversion of the final arrays to JAX.

New experiments use NumPy's PCG64 generator on the CPU. The legacy JAX split is
retained solely so analysis scripts can reconstruct older frozen runs whose
configuration predates an explicit ``split_rng`` field.

The Snelson loader lives here too: it is the only dataset both the GPflow and
the JAX collapsed experiments read, and it used to carry its own
standardisation with a slightly different constant-column rule than
:func:`fit_standardizer`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


# Single shared dataset root for the whole project. Official worktrees point
# this at one prevalidated cache, preventing five Slurm workers from racing to
# download and replace the same archive.
DATASET_DIR = Path(os.environ.get(
    "SCALABLE_GP_DATASET_DIR",
    Path(__file__).resolve().parent.parent / "dataset",
)).expanduser().resolve()
SNELSON_CSV = DATASET_DIR / "snelson.csv"

DEFAULT_SPLIT_RNG = "numpy-pcg64-v1"
LEGACY_SPLIT_RNG = "jax-threefry-v1"
SUPPORTED_SPLIT_RNGS = (DEFAULT_SPLIT_RNG, LEGACY_SPLIT_RNG)
PREPROCESSING_PROTOCOL = "cpu-train-standardize-v1"
LEGACY_PREPROCESSING_PROTOCOL = "jax-train-standardize-v1"


def validate_raw_arrays(X, y):
    """Return finite float64 host arrays with a strict regression shape."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y have different row counts: {X.shape[0]} != {y.shape[0]}"
        )
    if X.shape[0] < 2:
        raise ValueError("at least two observations are required")
    if X.shape[1] < 1:
        raise ValueError("at least one input feature is required")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must contain only finite values")
    return X, y


def _validate_split_sizes(n, n_train, n_test):
    n, n_train, n_test = int(n), int(n_train), int(n_test)
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must both be positive")
    if n_train + n_test > n:
        raise ValueError(
            f"requested {n_train} train + {n_test} test rows from {n} rows"
        )
    return n, n_train, n_test


def split_indices(n, split_seed, n_train, n_test,
                  split_rng=DEFAULT_SPLIT_RNG):
    """Return deterministic, disjoint host indices for one declared split."""
    n, n_train, n_test = _validate_split_sizes(n, n_train, n_test)
    split_seed = int(split_seed)
    split_rng = str(split_rng)
    if split_rng == DEFAULT_SPLIT_RNG:
        permutation = np.random.default_rng(split_seed).permutation(n)
    elif split_rng == LEGACY_SPLIT_RNG:
        # Archival compatibility only. New campaigns must use DEFAULT_SPLIT_RNG.
        import jax

        permutation = np.asarray(
            jax.random.permutation(jax.random.PRNGKey(split_seed), n)
        )
    else:
        raise ValueError(
            f"unsupported split_rng {split_rng!r}; "
            f"choose from {SUPPORTED_SPLIT_RNGS}"
        )
    train = np.asarray(permutation[:n_train], dtype=np.int64)
    test = np.asarray(
        permutation[n_train:n_train + n_test], dtype=np.int64
    )
    return train, test


def _angle_indices(angle_cols, n_features):
    values = () if angle_cols is None else tuple(int(v) for v in angle_cols)
    if len(set(values)) != len(values):
        raise ValueError(f"angle_cols contains duplicates: {values}")
    if any(value < 0 or value >= n_features for value in values):
        raise ValueError(
            f"angle_cols {values} is invalid for {n_features} features"
        )
    return values


def fit_standardizer(x_train, y_train, angle_cols=None):
    """Fit input and target standardization using training observations only."""
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    angles = _angle_indices(angle_cols, x_train.shape[1])
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())

    # Constant columns remain finite and map to zero. Angular columns stay on
    # their native radian scale so a declared 2*pi period remains meaningful.
    x_std = np.where(x_std > 1e-12, x_std, 1.0)
    if angles:
        idx = np.asarray(angles, dtype=np.int64)
        x_mean[idx] = 0.0
        x_std[idx] = 1.0
    if y_std <= 1e-12:
        raise ValueError("the training target has zero variance")

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "angle_cols": angles,
    }


def apply_standardizer(X, y, standardizer):
    """Apply one fitted standardizer to host arrays."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_scaled = (X - standardizer["x_mean"]) / standardizer["x_std"]
    y_scaled = (y - standardizer["y_mean"]) / standardizer["y_std"]
    return x_scaled, y_scaled


def to_jax(*arrays, dtype=None):
    """Transfer preprocessed host arrays to JAX at the model boundary."""
    import jax.numpy as jnp

    target_dtype = jnp.float64 if dtype is None else dtype
    return tuple(jnp.asarray(array, dtype=target_dtype) for array in arrays)


def _index_sha256(indices):
    canonical = np.asarray(indices, dtype="<i8")
    return "sha256:" + hashlib.sha256(canonical.tobytes()).hexdigest()


def _legacy_jax_split_standardize(X, y, split_seed, n_train, n_test,
                                  dtype, angle_cols):
    """Reproduce the preprocessing used by configurations without split_rng."""
    import jax
    import jax.numpy as jnp

    target_dtype = jnp.float64 if dtype is None else dtype
    X = jnp.asarray(X, dtype=target_dtype)
    y = jnp.asarray(y, dtype=target_dtype).reshape(-1)
    permutation = jax.random.permutation(
        jax.random.PRNGKey(int(split_seed)), X.shape[0]
    )
    train_idx = permutation[:n_train]
    test_idx = permutation[n_train:n_train + n_test]
    x_train, y_train = X[train_idx], y[train_idx]
    x_test, y_test = X[test_idx], y[test_idx]
    x_mean = jnp.mean(x_train, axis=0)
    x_std = jnp.std(x_train, axis=0) + 1e-12
    angles = _angle_indices(angle_cols, X.shape[1])
    if angles:
        idx = jnp.asarray(angles, dtype=jnp.int32)
        x_mean = x_mean.at[idx].set(0.0)
        x_std = x_std.at[idx].set(1.0)
    y_mean = jnp.mean(y_train)
    y_std = jnp.std(y_train) + 1e-12
    x_train = (x_train - x_mean) / x_std
    x_test = (x_test - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std
    train_host = np.asarray(train_idx, dtype=np.int64)
    test_host = np.asarray(test_idx, dtype=np.int64)
    metadata = {
        "x_mean": np.asarray(x_mean).tolist(),
        "x_std": np.asarray(x_std).tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "d": int(X.shape[1]),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "angle_cols": list(angles),
        "split_seed": int(split_seed),
        "split_rng": LEGACY_SPLIT_RNG,
        "preprocessing_protocol": LEGACY_PREPROCESSING_PROTOCOL,
        "train_indices_sha256": _index_sha256(train_host),
        "test_indices_sha256": _index_sha256(test_host),
    }
    return x_train, y_train, x_test, y_test, metadata


def split_standardize(X, y, split_seed, n_train, n_test, dtype=None,
                      angle_cols=None, split_rng=DEFAULT_SPLIT_RNG):
    """Split and standardize on the CPU, then return JAX model arrays.

    The return value is ``(x_train, y_train, x_test, y_test, metadata)``.
    Metadata records the complete preprocessing contract and hashes of the
    exact split indices without duplicating large index arrays in every run.
    """
    X, y = validate_raw_arrays(X, y)
    _, n_train, n_test = _validate_split_sizes(
        X.shape[0], n_train, n_test
    )
    if split_rng == LEGACY_SPLIT_RNG:
        return _legacy_jax_split_standardize(
            X, y, split_seed, n_train, n_test, dtype, angle_cols
        )
    train_idx, test_idx = split_indices(
        X.shape[0], split_seed, n_train, n_test, split_rng=split_rng
    )
    x_train, y_train = X[train_idx], y[train_idx]
    x_test, y_test = X[test_idx], y[test_idx]
    standardizer = fit_standardizer(x_train, y_train, angle_cols=angle_cols)
    x_train, y_train = apply_standardizer(x_train, y_train, standardizer)
    x_test, y_test = apply_standardizer(x_test, y_test, standardizer)
    x_train, y_train, x_test, y_test = to_jax(
        x_train, y_train, x_test, y_test, dtype=dtype
    )

    metadata = {
        "x_mean": standardizer["x_mean"].tolist(),
        "x_std": standardizer["x_std"].tolist(),
        "y_mean": standardizer["y_mean"],
        "y_std": standardizer["y_std"],
        "d": int(X.shape[1]),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "angle_cols": list(standardizer["angle_cols"]),
        "split_seed": int(split_seed),
        "split_rng": str(split_rng),
        "preprocessing_protocol": PREPROCESSING_PROTOCOL,
        "train_indices_sha256": _index_sha256(train_idx),
        "test_indices_sha256": _index_sha256(test_idx),
    }
    return x_train, y_train, x_test, y_test, metadata


def read_npz(path):
    """Read raw ``X`` and ``y`` arrays from a project NPZ cache."""
    with np.load(path) as data:
        return validate_raw_arrays(data["X"], data["y"])


def load_npz(path, split_seed, n_train, n_test, dtype=None,
             angle_cols=None, split_rng=DEFAULT_SPLIT_RNG):
    X, y = read_npz(path)
    return split_standardize(
        X,
        y,
        split_seed,
        n_train,
        n_test,
        dtype=dtype,
        angle_cols=angle_cols,
        split_rng=split_rng,
    )


# --------------------------------------------------------------------------- #
# Snelson (1-D reference set, shared by the GPflow and JAX collapsed figures)
# --------------------------------------------------------------------------- #
def load_snelson():
    """Return the 1-D Snelson regression data as float64 ``(x, y)`` columns."""
    data = np.loadtxt(SNELSON_CSV, delimiter=",")
    x = data[:, 0].reshape(-1, 1).astype(np.float64)
    y = data[:, 1].reshape(-1, 1).astype(np.float64)
    return x, y


def standardize_pair(x_train, y_train, x_test, y_test):
    """Standardise a pre-split pair with training statistics only.

    Snelson is a fit diagnostic evaluated on its own training set, so it never
    goes through :func:`split_standardize`; this wrapper gives it the same
    constant-column and zero-variance-target rules as every benchmark.
    """
    standardizer = fit_standardizer(x_train, y_train)
    x_tr, y_tr = apply_standardizer(x_train, y_train, standardizer)
    x_te, y_te = apply_standardizer(x_test, y_test, standardizer)
    return (
        x_tr,
        y_tr.reshape(-1, 1),
        x_te,
        y_te.reshape(-1, 1),
    )
