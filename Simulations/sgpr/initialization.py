"""Deterministic, nested inducing-point initialisation for SGPR v22."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from common import kernels as kn


INITIALIZATION_STRATEGIES = ("greedy-variance", "kmeans++", "random")


def _check_budget(x: np.ndarray, m: int, nested_max_m: int | None) -> int:
    if x.ndim != 2:
        raise ValueError(f"x must be rank two, found {x.shape}")
    if not 1 <= int(m) <= len(x):
        raise ValueError(f"M must be in [1, {len(x)}], found {m}")
    budget = int(m if nested_max_m is None else max(m, nested_max_m))
    return min(budget, len(x))


def greedy_variance_indices(
    x,
    m: int,
    params,
    *,
    kernel_name: str,
    seed: int,
    nested_max_m: int | None = None,
    jitter: float = 1e-12,
) -> np.ndarray:
    """Pivoted-Cholesky conditional-variance selection.

    Running to ``nested_max_m`` and returning a prefix makes the M sweep
    nested without materialising the N x N covariance matrix.  The random
    seed only resolves the otherwise arbitrary first pivot/ties.
    """
    x_np = np.asarray(x, dtype=np.float64)
    budget = _check_budget(x_np, m, nested_max_m)
    x_jax = jnp.asarray(x_np)
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    warp = params.get("warp")
    diagonal = np.asarray(jax.device_get(kn.kernel_diag(
        x_jax, ell, sf2, kernel_name, warp
    )), dtype=np.float64)
    residual = np.maximum(diagonal, 0.0)
    factors = np.zeros((len(x_np), budget), dtype=np.float64)
    selected = np.empty(budget, dtype=np.int32)
    available = np.ones(len(x_np), dtype=bool)
    rng = np.random.default_rng(int(seed) + 1)
    tie_break = rng.random(len(x_np))

    for column in range(budget):
        if column == 0 and float(np.ptp(residual)) <= 1e-12:
            # Stationary kernels have a constant prior diagonal, so their
            # first conditional-variance pivot is mathematically arbitrary.
            pivot = int(rng.integers(len(x_np)))
        else:
            scale = max(1.0, float(np.max(residual[available])))
            score = np.where(
                available,
                residual + np.finfo(np.float64).eps * scale * tie_break,
                -np.inf,
            )
            pivot = int(np.argmax(score))
        selected[column] = pivot
        available[pivot] = False

        kernel_column = np.asarray(jax.device_get(kn.kernel(
            x_jax,
            x_jax[pivot:pivot + 1],
            ell,
            sf2,
            kernel_name,
            warp,
        )), dtype=np.float64).reshape(-1)
        if column:
            kernel_column = (
                kernel_column
                - factors[:, :column] @ factors[pivot, :column]
            )
        denominator = np.sqrt(max(float(residual[pivot]), float(jitter)))
        factors[:, column] = kernel_column / denominator
        residual = np.maximum(
            residual - factors[:, column] ** 2,
            0.0,
        )
        residual[~available] = 0.0

    return selected[:int(m)]


def kmeanspp_indices(
    x,
    m: int,
    *,
    seed: int,
    nested_max_m: int | None = None,
) -> np.ndarray:
    """Nested deterministic k-means++ seeding, restricted to data points."""
    x_np = np.asarray(x, dtype=np.float64)
    budget = _check_budget(x_np, m, nested_max_m)
    rng = np.random.default_rng(int(seed) + 1)
    selected = np.empty(budget, dtype=np.int32)
    available = np.ones(len(x_np), dtype=bool)
    selected[0] = int(rng.integers(len(x_np)))
    available[selected[0]] = False
    nearest_sq = np.sum((x_np - x_np[selected[0]]) ** 2, axis=1)

    for column in range(1, budget):
        weights = np.where(available, nearest_sq, 0.0)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            pivot = int(np.flatnonzero(available)[0])
        else:
            pivot = int(rng.choice(len(x_np), p=weights / total))
        selected[column] = pivot
        available[pivot] = False
        distance_sq = np.sum((x_np - x_np[pivot]) ** 2, axis=1)
        nearest_sq = np.minimum(nearest_sq, distance_sq)

    return selected[:int(m)]


def inducing_indices(
    x,
    m: int,
    params,
    *,
    kernel_name: str,
    seed: int,
    strategy: str,
    nested_max_m: int | None,
) -> np.ndarray:
    """Dispatch one fingerprinted inducing initialisation policy."""
    if strategy == "greedy-variance":
        return greedy_variance_indices(
            x,
            m,
            params,
            kernel_name=kernel_name,
            seed=seed,
            nested_max_m=nested_max_m,
        )
    if strategy == "kmeans++":
        return kmeanspp_indices(
            x,
            m,
            seed=seed,
            nested_max_m=nested_max_m,
        )
    if strategy == "random":
        rng = np.random.default_rng(int(seed) + 1)
        return rng.choice(len(x), size=int(m), replace=False).astype(np.int32)
    raise ValueError(
        f"unknown inducing initialisation {strategy!r}; choose one of "
        f"{INITIALIZATION_STRATEGIES}"
    )
