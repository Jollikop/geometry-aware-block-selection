#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Residual predictive corrections, shared by both stacks.

Every sparse GP here predicts with

    var(x*) = q-term(x*) + residual(x*) + sigma^2.

Writing the training residual covariance as ``D_ff = K_ff - Q_ff``, four
predictors exist and the methods differ only in which residual model they use:

``sparse``  residual = d_**
            the term Bui, Ashman & Turner drop. Used by the trace-bound
            methods, SGPR and SVGP, whose objective keeps no residual model.

``diagonal`` residual = d_** - sum_i d_*i^2 / (d_ii + sigma^2)

``block``   residual = d_** - sum_b d_*b (D_bb + sigma^2 I)^-1 d_b*

``full``    residual = d_** - d_*f (D_ff + sigma^2 I)^-1 d_f*

The diagonal, block and full variants therefore use the same residual
approximation as T, RBT/SBT and E-SGPR respectively.  The standard sparse
predictor is recorded beside every structured result for a predictor-held-fixed
comparison, but it is not silently substituted for the method's own predictor.

There is no top-k truncation in a reported predictor. Every diagonal or block
model uses its complete fitted cover. A separate sparse result allows fair
comparison with the standard Titsias predictor.

DENSE DIAGNOSTIC DECOMPOSITION
------------------------------
For diagnostics only, the dense correction can be split over a cover. Permute
the training set into the cover order and factor

    A = D_ff + sigma^2 I = L L^T   (Cholesky, in that order)

and let ``w = L^-1 d_f*``. Then

    d_*f A^-1 d_f* = ||w||^2 = sum_j c_j(x*),   c_j = sum_{n in B_j} w_n^2.

Three properties follow, and all three are used:

* every ``c_j >= 0``, so the correction only grows as blocks are added;
* the total is *exactly* Eq. (12), whatever the diagnostic cover;
* the partial sum over the first j blocks is exactly the correction obtained by
  conditioning on the union of those blocks, because a Cholesky factorises
  sequential conditioning.

This decomposition does not define the RBT/SBT predictor: their model uses
independent block factors and therefore depends on the fitted partition.

This module owns only the residual term. The caller supplies the projections it
has already computed and adds its own q-term and noise, which is what lets the
collapsed bound (q from the Sigma factorisation) and the variational bound
(q from L_q) share one implementation.
"""
from __future__ import annotations

import numpy as np

#: Predictive residual models shared by the collapsed and variational stacks.
PREDICTOR_KINDS = ("sparse", "diagonal", "block", "full")

#: Floor applied to any reported predictive variance.
#: One chunking policy for every predictor. A per-predictor default made the
#: reported prediction time compare batching strategies rather than algorithms:
#: the diagonal path split the test set into 256-point pieces while the block
#: and dense paths processed it whole. n_test is small in every reported table,
#: so the default is "no split" for all of them, and callers that need a memory
#: ceiling pass one value that all three honour identically.
DEFAULT_PREDICT_CHUNK = None


def _test_chunk(n_test, chunk):
    """Rows of the test set processed per device call, same rule everywhere."""
    return max(1, int(chunk) if chunk else int(n_test))


VARIANCE_FLOOR = 1e-10

# Which predictor each published method reports. The rule is the same in both
# stacks: a method reports the predictive covariance its own bound models, so an
# NLL column reports the covariance represented by each bound. A separate
# sparse NLL is always saved for a predictor-held-fixed comparison.
#
# This registry lives here, next to the predictors it selects, because both
# stacks and the Snelson figure need it and none of them should own it.
COLLAPSED_METHOD_PREDICTOR = {
    "Exact-GP":   "exact",
    "SGPR":       "sparse",
    "T-SGPR":     "diagonal",
    "E-SGPR":     "full",
    "RBT-SGPR":   "block",
    "BT-SGPR":    "block",  # compatibility with archived runs
    "SBT-SGPR":   "block",
    "METIS-SGPR": "block",
}
VARIATIONAL_METHOD_PREDICTOR = {
    "SVGP":     "sparse",
    "T-SVGP":   "diagonal",
    "RBT-SVGP": "block",
    "BT-SVGP":  "block",  # compatibility with archived runs
    "SBT-SVGP": "block",
}


def cover_order(blocks):
    """Flatten a cover into the training-point order it induces.

    Raises unless the cover is a partition: the decomposition below is only a
    decomposition if every point appears exactly once.
    """
    if not blocks:
        raise ValueError("residual predictor received an empty cover")
    sizes = []
    parts = []
    for block in blocks:
        values = np.asarray(block, dtype=np.int64).reshape(-1)
        if not values.size:
            raise ValueError("residual predictor received an empty block")
        parts.append(values)
        sizes.append(values.size)
    order = np.concatenate(parts)
    if np.unique(order).size != order.size:
        raise ValueError("cover repeats a training index; it is not a partition")
    return order, np.asarray(sizes, dtype=np.int64)


def full_residual_contributions(
    x_train, x_test, a_train, a_test, s2, kernel_fn, *, blocks=None,
    jitter=1e-8, chunk=None,
):
    """Per-block share of the Eq. (12) correction, ``[n_test, n_blocks]``.

    ``blocks=None`` returns the whole correction in a single column, which is
    the dense endpoint. Otherwise the cover supplies both the permutation the
    Cholesky is taken in and the grouping of ``w`` used to split the result.

    One Cholesky of the permuted ``D_ff + sigma^2 I`` and one triangular solve
    against all test points; the split is then free. ``chunk`` batches the test
    columns of the solve when the test set is large.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    n_train = int(np.asarray(a_train).shape[1])
    if blocks is None:
        order = np.arange(n_train, dtype=np.int64)
        sizes = np.asarray([n_train], dtype=np.int64)
    else:
        order, sizes = cover_order(blocks)
        if order.size != n_train:
            raise ValueError(
                f"cover holds {order.size} indices for {n_train} training points"
            )

    perm = jnp.asarray(order)
    x_perm = x_train[perm]
    a_perm = a_train[:, perm]
    dff = kernel_fn(x_perm, x_perm) - a_perm.T @ a_perm
    dff = 0.5 * (dff + dff.T)
    chol = jnp.linalg.cholesky(
        dff + (s2 + jitter) * jnp.eye(n_train, dtype=dff.dtype)
    )

    n_test = int(np.asarray(a_test).shape[1])
    step = _test_chunk(n_test, chunk)
    squares = np.empty((n_train, n_test), dtype=np.float64)
    for start in range(0, n_test, step):
        stop = min(start + step, n_test)
        d_tf = (
            kernel_fn(x_test[start:stop], x_perm) - a_test[:, start:stop].T @ a_perm
        )
        w = solve_triangular(chol, d_tf.T, lower=True)          # [n_train, chunk]
        squares[:, start:stop] = np.asarray(jax.device_get(w * w))

    bounds = np.concatenate([[0], np.cumsum(sizes)])
    contributions = np.empty((n_test, sizes.size), dtype=np.float64)
    for j in range(sizes.size):
        contributions[:, j] = squares[bounds[j]:bounds[j + 1], :].sum(axis=0)
    return contributions


def diagonal_residual_contributions(components, *, chunk=None):
    """Total singleton correction as one column, computed in test chunks."""
    import jax
    import jax.numpy as jnp

    denominator = components["d_train"] + components["s2"]
    n_test = int(np.asarray(components["a_test"]).shape[1])
    out = np.empty((n_test, 1), dtype=np.float64)
    step = _test_chunk(n_test, chunk)
    for start in range(0, n_test, step):
        stop = min(start + step, n_test)
        d_cross = (
            components["kernel_fn"](components["xs"][start:stop], components["x"])
            - components["a_test"][:, start:stop].T @ components["a_train"]
        )
        correction = np.asarray(
            jax.device_get(jnp.sum((d_cross * d_cross) / denominator[None, :], axis=1))
        )
        out[start:stop, 0] = correction
    return out


def block_residual_contributions(components, blocks, *, jitter=1e-8,
                                 chunk=None):
    """Independent block corrections, ``[n_test, n_blocks]``.

    Each column is ``d_*b (D_bb + sigma^2 I)^-1 d_b*``.  Unlike
    :func:`full_residual_contributions`, blocks are factorised independently;
    consequently the partition changes the predictor and must match the one
    used by the fitted block objective.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    order, _ = cover_order(blocks)
    n_train = int(np.asarray(components["a_train"]).shape[1])
    if order.size != n_train:
        raise ValueError(
            f"cover holds {order.size} indices for {n_train} training points"
        )

    n_test = int(np.asarray(components["a_test"]).shape[1])
    out = np.empty((n_test, len(blocks)), dtype=np.float64)
    step = _test_chunk(n_test, chunk)
    for block_id, block in enumerate(blocks):
        idx = jnp.asarray(np.asarray(block, dtype=np.int64))
        xb = components["x"][idx]
        ab = components["a_train"][:, idx]
        dbb = components["kernel_fn"](xb, xb) - ab.T @ ab
        dbb = 0.5 * (dbb + dbb.T)
        chol = jnp.linalg.cholesky(
            dbb + (components["s2"] + jitter) * jnp.eye(idx.size, dtype=dbb.dtype)
        )
        for start in range(0, n_test, step):
            stop = min(start + step, n_test)
            d_tb = (
                components["kernel_fn"](components["xs"][start:stop], xb)
                - components["a_test"][:, start:stop].T @ ab
            )
            solved = solve_triangular(chol, d_tb.T, lower=True)
            out[start:stop, block_id] = np.asarray(
                jax.device_get(jnp.sum(solved * solved, axis=0))
            )
    return out


def residual_variance(d_test, contributions, *, units=None):
    """Residual variance left after the first ``units`` block contributions.

    ``units=None`` uses the whole residual model.  For ``full`` contributions
    this is Eq. (12); for ``diagonal`` and ``block`` contributions it is the
    corresponding fitted approximation.  A smaller value draws one point of
    the appendix curve for that same model.
    """
    import jax

    d_test = np.asarray(jax.device_get(d_test)).reshape(-1)
    contributions = np.asarray(contributions)
    if contributions.size:
        taken = contributions if units is None else contributions[:, : int(units)]
        d_test = d_test - taken.sum(axis=1)
    return np.maximum(d_test, 0.0)


def residual_contributions(components, *, predictor, blocks=None, jitter=1e-8,
                           chunk=None):
    """Dispatch to the contribution matrix of one predictive residual model.

    ``components`` is the projection dict both stacks build: ``x``, ``xs``,
    ``a_train``, ``a_test``, ``d_train``, ``d_test``, ``s2``, ``mean``,
    ``qvar`` and a ``kernel_fn`` closure bound to the fitted hyperparameters.
    """
    if predictor not in PREDICTOR_KINDS:
        raise ValueError(
            f"unknown predictor {predictor!r}; choose one of {PREDICTOR_KINDS}"
        )
    if predictor == "sparse":
        n_test = int(np.asarray(components["d_test"]).shape[0])
        return np.zeros((n_test, 0), dtype=np.float64)
    if predictor == "diagonal":
        return diagonal_residual_contributions(components, chunk=chunk)
    if predictor == "block":
        if blocks is None:
            raise ValueError("the block predictor requires the fitted partition")
        return block_residual_contributions(
            components, blocks, jitter=jitter, chunk=chunk
        )
    return full_residual_contributions(
        components["x"], components["xs"],
        components["a_train"], components["a_test"],
        components["s2"], components["kernel_fn"],
        blocks=blocks, jitter=jitter, chunk=chunk,
    )


def predictive_variance(components, *, predictor, blocks=None, jitter=1e-8,
                        chunk=None, return_contributions=False):
    """Predictive mean and variance under one residual model.

    Returns ``(mean, var)``, or ``(mean, var, contributions)`` when asked. The
    mean is the sparse posterior mean in every case; only the variance depends
    on ``predictor``, so RMSE is invariant to this choice.
    """
    import jax

    contributions = residual_contributions(
        components, predictor=predictor, blocks=blocks, jitter=jitter,
        chunk=chunk,
    )
    variance = (
        np.asarray(jax.device_get(components["qvar"])).reshape(-1)
        + residual_variance(components["d_test"], contributions)
        + float(jax.device_get(components["s2"]))
    )
    mean = np.asarray(jax.device_get(components["mean"])).reshape(-1)
    variance = np.maximum(variance, VARIANCE_FLOOR)
    if return_contributions:
        return mean, variance, contributions
    return mean, variance


def predictive_metrics(components, y_test, *, predictor, blocks=None,
                       jitter=1e-8, chunk=None):
    """RMSE and mean Gaussian NLL under one residual model."""
    mean, variance = predictive_variance(
        components, predictor=predictor, blocks=blocks, jitter=jitter,
        chunk=chunk,
    )
    return gaussian_rmse_nll(mean, variance, y_test)


def blocks_included_curve(components, y_test, contributions, *, block_sizes=None):
    """NLL after adding the first j residual units, for j = 0 .. n_units.

    ``j = 0`` is the sparse predictor.  The final point is the predictor that
    produced ``contributions``: diagonal, independently factorised blocks, or
    the dense Eq. (12) correction.  Only the dense Cholesky decomposition has
    the sequential-conditioning interpretation; block-model partial sums are
    nested truncations of that fitted block predictor.

    ``block_sizes`` records how many training points each block holds, so a
    figure can put covers of different unit size on one axis: T-SGPR splits the
    correction over singletons and the block methods over blocks of B, and
    "eight blocks" and "eight points" are not the same amount of conditioning.
    Points, not blocks, is the axis that compares them.

    RMSE is not reported: the predictive mean is the same at every j.
    """
    import jax

    contributions = np.asarray(contributions)
    if contributions.ndim != 2:
        raise ValueError("contributions must be [n_test, n_blocks]")

    d_test = np.asarray(jax.device_get(components["d_test"])).reshape(-1)
    qvar = np.asarray(jax.device_get(components["qvar"])).reshape(-1)
    mean = np.asarray(jax.device_get(components["mean"])).reshape(-1)
    s2 = float(jax.device_get(components["s2"]))

    cumulative = np.concatenate(
        [np.zeros((contributions.shape[0], 1)), np.cumsum(contributions, axis=1)],
        axis=1,
    )
    nll = np.empty(cumulative.shape[1], dtype=np.float64)
    for j in range(cumulative.shape[1]):
        variance = np.maximum(
            qvar + np.maximum(d_test - cumulative[:, j], 0.0) + s2, VARIANCE_FLOOR
        )
        nll[j] = gaussian_rmse_nll(mean, variance, y_test)[1]
    n_blocks = contributions.shape[1]
    if block_sizes is None:
        sizes = np.ones(n_blocks, dtype=np.int64)
    else:
        sizes = np.asarray(block_sizes, dtype=np.int64).reshape(-1)
        if sizes.size != n_blocks:
            raise ValueError(
                f"{sizes.size} block sizes for {n_blocks} contribution columns"
            )
    return {
        "blocks_included": np.arange(n_blocks + 1, dtype=np.int64),
        "points_included": np.concatenate([[0], np.cumsum(sizes)]),
        "block_sizes": sizes,
        "nll": nll,
        "block_mean_contribution": contributions.mean(axis=0),
        "captured_fraction": _captured_fraction(contributions),
    }


def _captured_fraction(contributions):
    """Share of the supplied residual model held by the first j units."""
    totals = contributions.sum(axis=1)
    scale = float(np.mean(totals))
    cumulative = np.concatenate(
        [np.zeros((contributions.shape[0], 1)), np.cumsum(contributions, axis=1)],
        axis=1,
    ).mean(axis=0)
    if scale <= 0.0:
        return np.full(cumulative.size, np.nan)
    return cumulative / scale

def gaussian_rmse_nll(mu, var, y):
    """RMSE and mean Gaussian negative log predictive density, on the host."""
    import jax

    mu = np.asarray(jax.device_get(mu)).reshape(-1)
    var = np.asarray(jax.device_get(var)).reshape(-1)
    y = np.asarray(jax.device_get(y)).reshape(-1)
    residual = y - mu
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    nll = float(np.mean(0.5 * (np.log(2.0 * np.pi * var) + residual ** 2 / var)))
    return rmse, nll


def gaussian_rmse_nll_device(mu, var, y):
    """Device-side twin of :func:`gaussian_rmse_nll`, for jitted metric paths."""
    import jax.numpy as jnp

    mu = jnp.asarray(mu).reshape(-1)
    var = jnp.asarray(var).reshape(-1)
    y = jnp.asarray(y).reshape(-1)
    residual = y - mu
    rmse = jnp.sqrt(jnp.mean(residual ** 2))
    nll = jnp.mean(0.5 * (jnp.log(2.0 * jnp.pi * var) + residual ** 2 / var))
    return rmse, nll
