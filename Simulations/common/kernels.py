#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JAX kernels and the name registry shared by both stacks.

Every kernel supports leading batch dimensions, so the same function evaluates
a full ``[N, N]`` covariance and a stacked ``[T, S, S]`` block covariance.

``kernel(...)`` is the only dispatch point. It takes ``warp``, which the
non-stationary kernels -- gibbs and amplitude-rbf -- consume and every purely
stationary kernel ignores. Structure selection passes the fitted warp
explicitly, so a post-refresh partition uses the same kernel as training.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from common import jax_config
from common.kernels_nonstationary import (
    WARP_PARAMETERIZATION,
    gibbs,
    gibbs_lengthscales,
)

# Imported for its side effect: float64 must be on before any array exists.
_ = jax_config

__all__ = [
    "KERNEL_ORDER", "KERNEL_ALIASES", "WARP_PARAMETERIZATION",
    "canonical_kernel_name", "kernel", "kernel_diag",
    "rbf", "amplitude_rbf", "matern32", "matern52", "periodic_angular",
    "matern_periodic",
    "gibbs", "gibbs_lengthscales",
]

KERNEL_ORDER = [
    "rbf",
    "amplitude-rbf",
    "matern32",
    "matern52",
    "gibbs",
    "periodic-angular",
    "matern-periodic",
]

KERNEL_ALIASES = {
    "rbf": "rbf",
    "se": "rbf",
    "squared-exponential": "rbf",
    "squared_exponential": "rbf",
    "amplitude-rbf": "amplitude-rbf",
    "amplitude_rbf": "amplitude-rbf",
    "amplitude-modulated-rbf": "amplitude-rbf",
    "matern": "matern32",
    "matern-3/2": "matern32",
    "matern32": "matern32",
    "matern-5/2": "matern52",
    "matern52": "matern52",
    "gibbs": "gibbs",
    "periodic": "periodic-angular",
    "periodic-angular": "periodic-angular",
    "periodic_angular": "periodic-angular",
    "angular": "periodic-angular",
    "matern-periodic": "matern-periodic",
    "matern52-periodic": "matern-periodic",
    "matern-per": "matern-periodic",
    "mp": "matern-periodic",
    "composite": "matern-periodic",
}


def canonical_kernel_name(name):
    """Normalise a kernel alias; unknown names fail loudly rather than default."""
    key = str(name or "rbf").strip().lower()
    if key not in KERNEL_ALIASES:
        raise ValueError(f"unknown kernel {name!r}; choose one of {KERNEL_ORDER}")
    return KERNEL_ALIASES[key]


def rbf(x1, x2, ell, sf2):
    a, b = x1 / ell, x2 / ell
    d2 = (jnp.sum(a * a, -1)[..., :, None]
          + jnp.sum(b * b, -1)[..., None, :]
          - 2.0 * jnp.matmul(a, jnp.swapaxes(b, -1, -2)))
    return sf2 * jnp.exp(-0.5 * jnp.maximum(d2, 0.0))


def amplitude_rbf(x1, x2, ell, sf2, weights):
    """PSD amplitude-modulated RBF with ``a(x)=0.5+sigmoid(x @ weights)``."""
    if weights is None:
        raise ValueError("amplitude-rbf requires its amplitude weights")
    a1 = 0.5 + jax.nn.sigmoid(jnp.matmul(x1, weights))
    a2 = 0.5 + jax.nn.sigmoid(jnp.matmul(x2, weights))
    return a1[..., :, None] * rbf(x1, x2, ell, sf2) * a2[..., None, :]


def _scaled_distance(x1, x2, ell):
    a, b = x1 / ell, x2 / ell
    d2 = (jnp.sum(a * a, -1)[..., :, None]
          + jnp.sum(b * b, -1)[..., None, :]
          - 2.0 * jnp.matmul(a, jnp.swapaxes(b, -1, -2)))
    return jnp.sqrt(jnp.maximum(d2, 0.0) + 1e-30)


def matern32(x1, x2, ell, sf2):
    r = _scaled_distance(x1, x2, ell)
    c = jnp.sqrt(3.0) * r
    return sf2 * (1.0 + c) * jnp.exp(-c)


def matern52(x1, x2, ell, sf2):
    r = _scaled_distance(x1, x2, ell)
    c = jnp.sqrt(5.0) * r
    return sf2 * (1.0 + c + 5.0 * r * r / 3.0) * jnp.exp(-c)


def periodic_angular(x1, x2, ell, sf2):
    diff = x1[..., :, None, :] - x2[..., None, :, :]
    sin2 = jnp.sin(0.5 * diff) ** 2
    return sf2 * jnp.exp(-2.0 * jnp.sum(sin2 / (ell * ell), axis=-1))


def matern_periodic(x1, x2, ell, sf2):
    """Matern-5/2 x periodic-angular PRODUCT kernel with shared ARD lengthscales.

    Each factor is a correlation in [0, 1] at sf2 = 1, so the product is PSD and
    scaled by sf2. Suits a periodic cycle plus smooth non-periodic drift, e.g.
    traffic. NOTE: the two factors deliberately share ``ell``.
    """
    return sf2 * matern52(x1, x2, ell, 1.0) * periodic_angular(x1, x2, ell, 1.0)


def kernel(x1, x2, ell, sf2, kernel_name="rbf", warp=None):
    """Dispatch on the canonical kernel name. ``warp`` is used only by gibbs."""
    name = canonical_kernel_name(kernel_name)
    if name == "rbf":
        return rbf(x1, x2, ell, sf2)
    if name == "amplitude-rbf":
        return amplitude_rbf(x1, x2, ell, sf2, warp)
    if name == "matern-periodic":
        return matern_periodic(x1, x2, ell, sf2)
    if name == "matern32":
        return matern32(x1, x2, ell, sf2)
    if name == "matern52":
        return matern52(x1, x2, ell, sf2)
    if name == "gibbs":
        return gibbs(x1, x2, ell, sf2, warp)
    if name == "periodic-angular":
        return periodic_angular(x1, x2, ell, sf2)
    raise AssertionError(name)


def kernel_diag(x, ell, sf2, kernel_name="rbf", warp=None):
    """Return ``k(x_i, x_i)`` without materialising the covariance matrix."""
    name = canonical_kernel_name(kernel_name)
    if name == "amplitude-rbf":
        if warp is None:
            raise ValueError("amplitude-rbf requires its amplitude weights")
        amplitude = 0.5 + jax.nn.sigmoid(jnp.matmul(x, warp))
        return sf2 * amplitude * amplitude
    return sf2 * jnp.ones((x.shape[0],), dtype=x.dtype)
