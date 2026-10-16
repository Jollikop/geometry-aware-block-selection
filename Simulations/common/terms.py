#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The packed block representation shared by both stacks.

A residual block cover is passed to every loss as a fixed-shape triple::

    coeffs [T]      per-term weight; 0 marks a padding row
    idx    [T, S]   training-set positions, int32
    mask   [T, S]   1.0 on real entries, 0.0 on padding

The shapes are fixed so that re-selecting the structure only changes the
*contents* of these arrays. Nothing recompiles: XLA sees the same shapes and
TensorFlow sees the same Variables.

The residual penalty consumed downstream is

    P = sum_t coeffs[t] * logdet(I + D[idx[t], idx[t]] / sigma^2)

and the net coefficient of every training point must be exactly 1, which is
what makes each cover a certified lower bound (Hadamard-Fischer).

Four separate pack/unpack implementations existed before the two-stack split.
This module is the only one now; the TensorFlow path in
``sgpr.gpflow_bounds.StructuredSGPR`` builds its Variables from :func:`pack_terms`.
"""
from __future__ import annotations

import math

import numpy as np


def default_capacity(n_points, block_size):
    """Rows to allocate for a cover of ``n_points`` in blocks of ``block_size``.

    Two spare rows absorb a cover that ends with a short block after a refresh
    without forcing a recompile.
    """
    return int(math.ceil(int(n_points) / int(block_size))) + 2


def pack_terms(blocks, block_size, n_points, *, coeffs=None, capacity=None,
               as_jax=True):
    """Pack a list of index arrays into ``(coeffs, idx, mask)``.

    ``blocks``     sequence of 1-D integer index arrays
    ``coeffs``     per-block weights; defaults to 1.0 for every block
    ``capacity``   number of rows T; defaults to :func:`default_capacity`
    ``as_jax``     return jnp arrays (device) instead of NumPy (host)
    """
    block_size = max(1, int(block_size))
    t_cap = int(capacity) if capacity is not None else default_capacity(
        n_points, block_size
    )
    if len(blocks) > t_cap:
        raise ValueError(f"{len(blocks)} blocks exceed the padded capacity {t_cap}")

    out_coeffs = np.zeros((t_cap,), dtype=np.float64)
    out_idx = np.zeros((t_cap, block_size), dtype=np.int32)
    out_mask = np.zeros((t_cap, block_size), dtype=np.float64)
    weights = [1.0] * len(blocks) if coeffs is None else [float(c) for c in coeffs]
    if len(weights) != len(blocks):
        raise ValueError("coeffs and blocks have different lengths")

    for row, (weight, block) in enumerate(zip(weights, blocks)):
        block = np.asarray(block, dtype=np.int32).reshape(-1)
        if block.size > block_size:
            raise ValueError(f"block of size {block.size} exceeds width {block_size}")
        out_coeffs[row] = weight
        out_idx[row, : block.size] = block
        out_mask[row, : block.size] = 1.0

    if not as_jax:
        return out_coeffs, out_idx, out_mask
    import jax.numpy as jnp

    return (
        jnp.asarray(out_coeffs),
        jnp.asarray(out_idx, dtype=jnp.int32),
        jnp.asarray(out_mask),
    )


def pack_order(order, block_size, n_points, **kwargs):
    """Chunk one ordering of the training set into consecutive equal blocks."""
    order = np.asarray(order, dtype=np.int32).reshape(-1)
    block_size = max(1, int(block_size))
    blocks = [order[i:i + block_size] for i in range(0, int(n_points), block_size)]
    return pack_terms(blocks, block_size, n_points, **kwargs)


def unpack_terms(packed):
    """Return the list of index arrays carried by a packed triple.

    Padding rows (coefficient 0) and masked-out columns are dropped, so the
    result is the partition a method actually conditions on.
    """
    coeffs, idx, mask = (_to_host(part) for part in packed)
    blocks = []
    for row in range(idx.shape[0]):
        if coeffs[row] <= 0.0:
            continue
        active = idx[row, mask[row] > 0.5].astype(np.int64)
        if active.size:
            blocks.append(active)
    return blocks


def assert_partition(blocks, n_points):
    """Raise unless ``blocks`` cover every training index exactly once.

    Cheap (one sort) relative to any training step, and it turns a silent
    coverage bug into an immediate failure. Both benchmarks call it after a
    structure refresh.
    """
    if not blocks:
        raise ValueError("empty block cover")
    covered = np.sort(np.concatenate([np.asarray(b).reshape(-1) for b in blocks]))
    expected = np.arange(int(n_points))
    if covered.size != expected.size or not np.array_equal(covered, expected):
        raise ValueError(
            f"block cover holds {covered.size} entries for {n_points} training "
            "points and is not a partition"
        )


def net_coefficients(packed, n_points):
    """Total weight carried by each training point; must be 1 for a valid bound."""
    coeffs, idx, mask = (_to_host(part) for part in packed)
    net = np.zeros(int(n_points), dtype=np.float64)
    for row in range(idx.shape[0]):
        if coeffs[row] == 0.0:
            continue
        active = idx[row, mask[row] > 0.5]
        net[active] += coeffs[row]
    return net


def _to_host(array):
    """Copy a JAX or NumPy array to the host without importing JAX eagerly."""
    if hasattr(array, "device_buffer") or type(array).__module__.startswith("jax"):
        import jax

        return np.asarray(jax.device_get(array))
    return np.asarray(array)


# --------------------------------------------------------------------------- #
# Device-side packing (fixed shapes, used inside jitted structure builders)
# --------------------------------------------------------------------------- #
def pack_blocks_device(order, n_points, block_size):
    """Pad an ordering to ``B * block`` and reshape to ``[B, block]`` + mask."""
    import jax.numpy as jnp

    n, block = int(n_points), int(block_size)
    n_blocks = (n + block - 1) // block
    pad = n_blocks * block - n
    order_padded = jnp.concatenate([order, jnp.zeros(pad, order.dtype)])
    idx = order_padded.reshape(n_blocks, block)
    mask = jnp.concatenate([jnp.ones(n), jnp.zeros(pad)]).reshape(n_blocks, block)
    return idx, mask, n_blocks


def widen_to_terms(idx, mask, coeff, width, dtype=None):
    """Widen ``[T, s]`` block arrays to ``[T, width]`` term arrays."""
    import jax.numpy as jnp

    dtype = jnp.float64 if dtype is None else dtype
    t, s = idx.shape
    if s < width:
        idx = jnp.concatenate([idx, jnp.zeros((t, width - s), idx.dtype)], axis=1)
        mask = jnp.concatenate([mask, jnp.zeros((t, width - s), mask.dtype)], axis=1)
    return jnp.full((t,), coeff, dtype=dtype), idx, mask.astype(dtype)
