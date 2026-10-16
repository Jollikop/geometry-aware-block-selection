#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block covers: BT (random) and SBT (inducing-anchored).

Both stacks select residual blocks the same way, so the selectors live here and
neither stack owns them.

Two covers exist:

``BT``   disjoint random blocks. Structure-agnostic control: it isolates
         "does blocking help" from "does *this* blocking help".
``SBT``  inducing-anchored blocks. Order the inducing points along a greedy
         nearest-neighbour chain, assign each datum to its nearest anchor,
         order the points inside each anchor cell, concatenate the cells in
         chain order, and cut the result into equal-size blocks.

Both give every point a net coefficient of exactly 1, hence a certified lower
bound by the Hadamard-Fischer argument.

Cost of SBT (M inducing points, N data, d dimensions):
    anchor assignment      O(N M d)
    inducing chain         O(M^2 d)
    block materialisation  O(N)

HOST AND DEVICE PATHS ARE NOT INTERCHANGEABLE
---------------------------------------------
``*_host`` runs in NumPy and serves the GPflow experiments (Snelson, residual
illustration). ``*_device`` runs in JAX and serves both JAX benchmarks. They
implement the same construction but measure distance differently:

    host    squared Euclidean distance in the ARD metric,  ||x/l - z/l||^2
    device  kernel-induced distance,  k(x,x) + k(z,z) - 2 k(x,z), including
            the fitted Gibbs warp when present

For every kernel in the registry the second is a monotone function of the
first, so the parent assignment and the within-cell ``center`` ordering agree
exactly. The default ``boundary`` ordering sorts on the DIFFERENCE
``d_prev - d_next``, which a monotone reparameterisation does not preserve, so
the two paths can cut a cell at a different point. ``tests/test_partitions.py``
measures that disagreement instead of assuming it away.

Do not "fix" this by switching one path to the other metric without re-running
the campaign that depends on it: it changes published numbers.
"""
from __future__ import annotations



import numpy as np

from common import terms as tm
from common.kernels import canonical_kernel_name, kernel, kernel_diag

DEFAULT_SPLIT_MODE = "boundary"
SPLIT_MODES = ("boundary", "center")


# --------------------------------------------------------------------------- #
# BT: disjoint random blocks
# --------------------------------------------------------------------------- #
def bt_blocks_host(n_points, block_size, rng):
    """Random disjoint blocks from a host RNG (GPflow path)."""
    order = np.asarray(rng.permutation(int(n_points)), dtype=np.int32)
    block = max(1, int(block_size))
    return [np.sort(order[i:i + block]).astype(np.int32)
            for i in range(0, order.size, block)]


def bt_terms_host(n_points, block_size, rng):
    """``[(coeff, idx)]`` random cover for ``StructuredSGPR.set_terms``."""
    return [(1.0, blk) for blk in bt_blocks_host(n_points, block_size, rng)]


def bt_terms_device(n_points, block_size, seed, dtype=None):
    """Random disjoint blocks drawn on device, packed to fixed shapes.

    Reproducible from ``seed`` alone: it uses neither the kernel nor the
    inducing points, which is what lets the analysis scripts regenerate a run's
    BT partition without retraining it.
    """
    import jax
    import jax.numpy as jnp

    n, block = int(n_points), max(1, int(block_size))
    order = jax.random.permutation(
        jax.random.PRNGKey(int(seed)), jnp.arange(n, dtype=jnp.int32)
    )
    idx, mask, _ = tm.pack_blocks_device(order, n, block)
    return tm.widen_to_terms(idx, mask, 1.0, block,
                             jnp.float64 if dtype is None else dtype)


# --------------------------------------------------------------------------- #
# SBT, host path (NumPy; ARD Euclidean metric)
# --------------------------------------------------------------------------- #
def _ard_lengthscales(lengthscales, d):
    ell = np.asarray(lengthscales, dtype=float).reshape(-1)
    if ell.size != d:
        ell = np.full(d, float(np.mean(ell)))
    return np.maximum(ell, 1e-9)


def _greedy_anchor_chain(dist_aa):
    """Visit anchors nearest-first starting from anchor 0."""
    m = dist_aa.shape[0]
    if m <= 2:
        return np.arange(m, dtype=np.int32)
    dist = np.asarray(dist_aa, dtype=float).copy()
    np.fill_diagonal(dist, np.inf)
    visited = np.zeros(m, dtype=bool)
    cur = 0
    chain = [cur]
    visited[cur] = True
    for _ in range(m - 1):
        row = np.where(visited, np.inf, dist[cur])
        if not np.any(np.isfinite(row)):
            break
        cur = int(np.argmin(row))
        visited[cur] = True
        chain.append(cur)
    if len(chain) < m:
        chain.extend(np.where(~visited)[0].astype(np.int32).tolist())
    return np.asarray(chain, dtype=np.int32)


def anchor_order_host(x, z, ell, split_mode=DEFAULT_SPLIT_MODE):
    """Data order induced by a nearest-neighbour chain over the anchors.

    ``center``    inside a cell, sort by distance to the parent anchor.
    ``boundary``  sort by ``d_prev - d_next`` so the points a chunk cut moves
                  across are the ones closest to the neighbouring anchor.

    Returns ``(order, parent, chain, parent_dist)``.
    """
    n = len(x)
    if len(z) == 0:
        return (np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32),
                np.empty(0, dtype=np.int32), np.zeros(n, dtype=float))

    x_scaled, z_scaled = x / ell, z / ell
    dist_xz = np.maximum(
        np.sum(x_scaled * x_scaled, axis=1)[:, None]
        + np.sum(z_scaled * z_scaled, axis=1)[None, :]
        - 2.0 * x_scaled @ z_scaled.T,
        0.0,
    )
    parent = np.argmin(dist_xz, axis=1).astype(np.int32)
    parent_dist = dist_xz[np.arange(n), parent]

    zz = np.sum(z_scaled * z_scaled, axis=1)
    dist_zz = np.maximum(zz[:, None] + zz[None, :] - 2.0 * z_scaled @ z_scaled.T, 0.0)
    chain = _greedy_anchor_chain(dist_zz)

    mode = str(split_mode or DEFAULT_SPLIT_MODE).lower()
    order_parts = []
    for chain_pos, anchor in enumerate(chain):
        members = np.where(parent == anchor)[0]
        if members.size == 0:
            continue
        if mode == "boundary":
            prev_anchor = chain[chain_pos - 1] if chain_pos > 0 else None
            next_anchor = (chain[chain_pos + 1]
                           if chain_pos + 1 < len(chain) else None)
            if prev_anchor is not None and next_anchor is not None:
                score = dist_xz[members, prev_anchor] - dist_xz[members, next_anchor]
            elif next_anchor is not None:
                # First cell: points near the next anchor belong at the right
                # boundary, where a chunk cut can move them forward.
                score = -dist_xz[members, next_anchor]
            elif prev_anchor is not None:
                score = dist_xz[members, prev_anchor]
            else:
                score = parent_dist[members]
            local = members[np.lexsort((parent_dist[members], score))]
        else:
            local = members[np.argsort(dist_xz[members, anchor])]
        order_parts.append(local.astype(np.int32))

    order = (np.concatenate(order_parts).astype(np.int32)
             if order_parts else np.arange(n, dtype=np.int32))
    seen = np.zeros(n, dtype=bool)
    seen[order] = True
    if not np.all(seen):
        order = np.concatenate([order, np.where(~seen)[0].astype(np.int32)])
    return order, parent, chain, parent_dist


def sbt_blocks_host(x, z, lengthscales, block_size, *,
                    split_mode=DEFAULT_SPLIT_MODE, return_info=False):
    """Inducing-anchored equal-size block cover, host path."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    ell = _ard_lengthscales(lengthscales, x.shape[1])
    order, parent, _, parent_dist = anchor_order_host(x, z, ell, split_mode)
    block = max(1, int(block_size))
    blocks = [np.sort(order[i:i + block]).astype(np.int32)
              for i in range(0, order.size, block)]
    if not return_info:
        return blocks
    info = {
        "terms": len(blocks),
        "cost": float(sum(len(idx) ** 3 for idx in blocks)),
        "anchor_chunks": len(blocks),
        "block_size": block,
        "anchor_mode": "inducing-order",
        "split_mode": str(split_mode or DEFAULT_SPLIT_MODE).lower(),
        "inducing_anchor_count": int(len(z)),
        "inducing_anchor_occupancy": np.bincount(
            parent, minlength=max(len(z), 1)
        ).astype(int),
        "mean_anchor_dist": float(np.mean(parent_dist)) if len(x) else 0.0,
        "max_anchor_dist": float(np.max(parent_dist)) if len(x) else 0.0,
        "order": order,
        "blocks": blocks,
    }
    return blocks, info


def sbt_terms_host(x, z, lengthscales, block_size, *,
                   split_mode=DEFAULT_SPLIT_MODE, return_info=False):
    """``[(coeff, idx)]`` SBT cover for ``StructuredSGPR.set_terms``."""
    result = sbt_blocks_host(x, z, lengthscales, block_size,
                             split_mode=split_mode, return_info=return_info)
    if not return_info:
        return [(1.0, idx) for idx in result]
    blocks, info = result
    return [(1.0, idx) for idx in blocks], info


# --------------------------------------------------------------------------- #
# SBT, device path (JAX; kernel-induced metric)
# --------------------------------------------------------------------------- #
def _anchor_details_device(x, z, ell, kernel_name="rbf", warp=None):
    """Parent labels, chain ranks and data-anchor distances, all on device."""
    import jax
    import jax.numpy as jnp

    kxz = kernel(x, z, ell, 1.0, kernel_name, warp)
    d2xz = jnp.maximum(
        kernel_diag(x, ell, 1.0, kernel_name, warp)[:, None]
        + kernel_diag(z, ell, 1.0, kernel_name, warp)[None, :]
        - 2.0 * kxz,
        0.0,
    )
    parent = jnp.argmin(d2xz, axis=1).astype(jnp.int32)
    dist = jnp.min(d2xz, axis=1)

    m = z.shape[0]
    kzz = kernel(z, z, ell, 1.0, kernel_name, warp)
    kzz_diag = kernel_diag(z, ell, 1.0, kernel_name, warp)
    d2zz = jnp.maximum(kzz_diag[:, None] + kzz_diag[None, :] - 2.0 * kzz, 0.0)
    big = jnp.max(d2zz) + 1.0

    def step(carry, _):
        cur, used = carry
        masked = jnp.where(used, big * 10.0, d2zz[cur]).at[cur].set(big * 100.0)
        nxt = jnp.argmin(masked).astype(jnp.int32)
        nxt = jnp.where(jnp.all(used | (jnp.arange(m) == cur)), cur, nxt)
        return (nxt, used.at[cur].set(True)), cur

    # Start from anchor 0 and visit the nearest unused anchor, matching
    # ``_greedy_anchor_chain``.
    (_, _), chain = jax.lax.scan(
        step, (jnp.int32(0), jnp.zeros(m, dtype=bool)), None, length=m
    )
    chain = chain.astype(jnp.int32)
    rank = jnp.zeros(m, dtype=jnp.int32).at[chain].set(
        jnp.arange(m, dtype=jnp.int32))
    return parent, rank, dist, d2xz, chain


def anchor_order_device(x, z, ell, kernel_name="rbf",
                        split_mode=DEFAULT_SPLIT_MODE, warp=None):
    """JIT-compiled anchor ordering.

    Compiled once per ``(N, M, kernel, split_mode)`` shape and reused, so the
    greedy-chain scan is not re-charged to ``search_s`` at every refresh. The
    jit wrapper is built on first use so that the GPflow-only experiments can
    import this module without paying for a JAX import.
    """
    global _ANCHOR_ORDER_JIT
    if _ANCHOR_ORDER_JIT is None:
        import jax

        _ANCHOR_ORDER_JIT = jax.jit(
            _anchor_order_device,
            static_argnames=("kernel_name", "split_mode"),
        )
    return _ANCHOR_ORDER_JIT(
        x, z, ell, warp, kernel_name=kernel_name, split_mode=split_mode
    )


_ANCHOR_ORDER_JIT = None


def _anchor_order_device(x, z, ell, warp=None, kernel_name="rbf",
                         split_mode=DEFAULT_SPLIT_MODE):
    import jax.numpy as jnp

    parent, rank, parent_dist, d2xz, chain = _anchor_details_device(
        x, z, ell, kernel_name, warp
    )
    rank_key = rank[parent]
    mode = str(split_mode or DEFAULT_SPLIT_MODE).lower()
    if mode == "boundary" and z.shape[0] > 1:
        pos = rank_key
        prev_chain = jnp.concatenate([chain[:1], chain[:-1]])
        next_chain = jnp.concatenate([chain[1:], chain[-1:]])
        row = jnp.arange(x.shape[0])
        d_prev = d2xz[row, prev_chain[pos]]
        d_next = d2xz[row, next_chain[pos]]
        score = jnp.where(
            pos == 0,
            -d_next,
            jnp.where(pos == z.shape[0] - 1, d_prev, d_prev - d_next),
        )
    else:
        score = parent_dist
    return jnp.lexsort((parent_dist, score, rank_key)).astype(jnp.int32)


def sbt_terms_device(x, params, block_size, kernel_name,
                     split_mode=DEFAULT_SPLIT_MODE):
    """Packed SBT cover at the current hyperparameters.

    The ordering runs on device; the chunking and packing are host work, which
    is why the benchmarks account for this call under ``search_s`` rather than
    inside the compiled training scan.
    """
    import jax
    import jax.numpy as jnp

    x_np = np.asarray(x, dtype=np.float64)
    n = int(x_np.shape[0])
    block = max(1, int(block_size))
    order = np.asarray(
        jax.device_get(
            anchor_order_device(
                jnp.asarray(x_np),
                jnp.asarray(params["z"]),
                jnp.exp(params["log_ell"]),
                canonical_kernel_name(kernel_name),
                split_mode,
                params.get("warp"),
            )
        ),
        dtype=np.int32,
    )
    return tm.pack_order(order, block, n)


def blocks_from_order(order, block_size, n_points):
    """Consecutive equal-size chunks of an arbitrary ordering."""
    order = np.asarray(order, dtype=np.int32).reshape(-1)
    block = max(1, int(block_size))
    return [order[i:i + block] for i in range(0, int(n_points), block)]
