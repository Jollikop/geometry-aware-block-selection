#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Uncollapsed SVGP bounds with structured residual corrections.

The engine of the SVGP stack. q(u) stays explicit, so training is minibatched
and the two block methods use the exact decomposition

    F_block = F_T + 1/2 * sum_B gain(B),

where ``gain(B) = sum_i log(A_ii) - logdet(A_B)`` and ``A = I + D_ff/sigma^2``.
``gain(B)`` is non-negative by Hadamard-Fischer, which is what makes an
importance-sampled estimate of the sum well behaved.

A global likelihood/T-SVGP minibatch and an independent structural minibatch
therefore estimate two different sums; that separation is the "double batch" in
the implementation identifier.
"""
from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from common import kernels as kn
from common import predictors as pr
from common import terms as tm
from common.partitions import sbt_terms_device

rbf = kn.rbf
kernel = kn.kernel
kernel_diag = kn.kernel_diag
canonical_kernel_name = kn.canonical_kernel_name
KERNEL_ORDER = kn.KERNEL_ORDER
# Re-exported so the runner can record the warp parameterisation in a run's
# configuration without duplicating the string. A duplicated literal would
# drift the day the parameterisation changes, and the fingerprint would then
# claim a model the code no longer implements.
WARP_PARAMETERIZATION = kn.WARP_PARAMETERIZATION
# v12: each bound reports its own sparse, diagonal or fitted-block predictive
# covariance. The standard sparse predictive is retained as a separate metric.
IMPLEMENTATION = "svgp-residual-gain-double-batch-jax-v12"

# method -> (penalty_mode, selection, adapts)
#
# The four base uncollapsed SVGP variants differ only in the residual
# correction applied to q(f|u): SVGP keeps the prior conditional (trace bound),
# T-SVGP a diagonal scaling, RBT-SVGP a random block scaling, SBT-SVGP a
# selected-block scaling.
PAPER_METHODS = {
    "SVGP":              ("trace", "BT", False),
    "T-SVGP":            ("diag", "BT", False),
    "RBT-SVGP":          ("gain", "BT-gain", False),
    "SBT-SVGP":          ("gain", "SBT-gain", True),
}

# The executable registry intentionally exposes exactly the four paper methods.
METHODS = PAPER_METHODS.copy()
METHODS["BT-SVGP"] = PAPER_METHODS["RBT-SVGP"]  # archived-name compatibility

# Owned by ``common.predictors``; re-exported so callers find it beside METHODS.
METHOD_PREDICTOR = pr.VARIATIONAL_METHOD_PREDICTOR

# Which parameters each freezable group holds. A staged protocol names groups,
# not leaves, so a schedule stays readable and cannot half-freeze a model by
# forgetting that the noise lives with the kernel.
PARAMETER_GROUPS = {
    "theta": ("log_ell", "log_sf2", "log_s2", "warp"),
    "z": ("z",),
    "q": ("q_mu", "q_cov", "q_raw"),
}


def parse_phases(value, total_epochs):
    """Parse ``"10:theta,z 140:theta"`` into ``[(end_epoch, frozen_groups)]``.

    Each entry is ``<epochs>:<comma-separated groups>``; ``<epochs>:-`` freezes
    nothing. Lengths are consumed in order and must sum to the epoch budget, so
    a schedule that does not add up fails before any compute is spent rather
    than silently training a different protocol.
    """
    if not value:
        return [(int(total_epochs), frozenset())]
    tokens = str(value).replace(",,", ",").split()
    phases, consumed = [], 0
    for token in tokens:
        length, _, groups = token.partition(":")
        length = int(length)
        if length < 1:
            raise ValueError(f"phase {token!r} has a non-positive length")
        names = frozenset(
            g for g in groups.split(",") if g and g not in {"-", "none"}
        )
        unknown = names - set(PARAMETER_GROUPS)
        if unknown:
            raise ValueError(
                f"phase {token!r} freezes unknown group(s) {sorted(unknown)}; "
                f"choose from {sorted(PARAMETER_GROUPS)}"
            )
        if names == frozenset(PARAMETER_GROUPS):
            raise ValueError(f"phase {token!r} freezes every group: nothing trains")
        consumed += length
        phases.append((consumed, names))
    if consumed != int(total_epochs):
        raise ValueError(
            f"phases sum to {consumed} epochs but --epochs is {total_epochs}"
        )
    return phases

BASE_METHOD_ORDER = [
    "SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP",
]
METHOD_ORDER = BASE_METHOD_ORDER.copy()


# --------------------------------------------------------------------------- #
# Params
# --------------------------------------------------------------------------- #
def transform(params):
    """Unpack constrained hyperparameters.

    Returns ``(ell, sf2, s2, lq, warp)``. ``warp`` holds the optional Gibbs
    matrix or amplitude-RBF weight vector.
    """
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    s2 = jnp.exp(params["log_s2"])
    warp = params.get("warp")
    if "q_cov" in params:
        q_cov = 0.5 * (params["q_cov"] + params["q_cov"].T)
        lq = jnp.linalg.cholesky(
            q_cov + 1e-10 * jnp.eye(q_cov.shape[0], dtype=q_cov.dtype)
        )
    else:
        raw = params["q_raw"]
        lq = jnp.tril(raw, -1) + jnp.diag(jnp.exp(jnp.diag(raw)))
    return ell, sf2, s2, lq, warp


def init_params(x_tr, m, seed, kernel_name="rbf"):
    d = x_tr.shape[1]
    sel = jax.random.choice(jax.random.PRNGKey(seed + 1), x_tr.shape[0],
                            (m,), replace=False)
    params = {
        "log_ell": jnp.full((d,), math.log(0.7)),
        "log_sf2": jnp.asarray(math.log(1.0)),
        "log_s2": jnp.asarray(math.log(0.1)),
        "z": jnp.asarray(x_tr[sel]),
        "q_mu": jnp.zeros((m,)),
        "q_raw": jnp.zeros((m, m)),
    }
    if canonical_kernel_name(kernel_name) == "gibbs":
        # Zero warp == stationary RBF with lengthscale 1.5 * ell. The entry is
        # allocated from step 0 so the optimizer state keeps a fixed pytree
        # shape; the non-stationary phase is switched on by releasing the
        # gradient of this block, not by growing the parameter dict.
        params["warp"] = jnp.zeros((d, d))
    elif canonical_kernel_name(kernel_name) == "amplitude-rbf":
        params["warp"] = jnp.zeros((d,))
    return params


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def make_engine(x_tr, y_tr, x_te, y_te, *, jitter=1e-8,
                kernel_name="rbf"):
    kernel_name = canonical_kernel_name(kernel_name)
    x = jnp.asarray(x_tr)
    y = jnp.asarray(y_tr).ravel()
    xt = jnp.asarray(x_te)
    yt = jnp.asarray(y_te).ravel()

    def cache(params):
        ell, sf2, s2, lq, warp = transform(params)
        kuu = kernel(params["z"], params["z"], ell, sf2, kernel_name, warp) \
            + jitter * jnp.eye(params["z"].shape[0])
        luu = jnp.linalg.cholesky(kuu)
        return ell, sf2, s2, lq, luu, x, xt, warp

    def kl(params, lq):
        return 0.5 * (jnp.sum(params["q_mu"] ** 2) + jnp.sum(lq ** 2)
                      - params["q_mu"].shape[0]
                      - 2.0 * jnp.sum(jnp.log(jnp.diag(lq))))

    def block_data_from_features(params, idx, mask, c, a):
        """Expected data term from one already projected feature set."""
        _, _, s2, lq, _, _, _, _ = c
        mean = a.T @ params["q_mu"]
        qvar = jnp.sum((lq.T @ a) ** 2, axis=0)
        resid = y[idx] - mean
        return -0.5 * jnp.sum(mask * (
            jnp.log(2.0 * jnp.pi * s2) + (resid * resid + qvar) / s2))

    def block_v_from_features(params, idx, mask, c, a):
        """Residual V matrix from one already projected feature set."""
        ell, sf2, s2, _, _, xk, _, warp = c
        xb = xk[idx]
        d = kernel(xb, xb, ell, sf2, kernel_name, warp) - a.T @ a
        d = 0.5 * (d + d.T)
        pair = mask[:, None] * mask[None, :]
        # A_B = I + D_B/sigma^2 is already uniformly positive definite; do
        # not add a second jitter here, otherwise T + gain ceases to be an
        # exact algebraic decomposition of the direct block objective.
        return (jnp.eye(idx.shape[0]) + d / s2) * pair \
            + jnp.diag(1.0 - mask)

    def diag_logdet_from_features(params, idx, mask, c, a):
        """Singleton/T-SVGP residual logdet on an arbitrary point batch."""
        ell, sf2, s2, _, _, xk, _, _ = c
        dii = jnp.maximum(
            kernel_diag(xk[idx], ell, sf2, kernel_name, c[7])
            - jnp.sum(a * a, axis=0),
            0.0,
        )
        return jnp.sum(mask * jnp.log1p(dii / s2))

    def block_logdet_and_gain_from_features(params, idx, mask, c, a):
        """Return full block penalty and its exact improvement over T-SVGP."""
        v = block_v_from_features(params, idx, mask, c, a)
        logdet = jnp.linalg.slogdet(v)[1]
        diag_logdet = jnp.sum(
            mask * jnp.log(jnp.maximum(jnp.diag(v), 1e-12))
        )
        return logdet, diag_logdet - logdet

    def block_tau_from_features(params, idx, mask, c, a, mode):
        """Per-unit ELBO contribution under the trace or diagonal penalty."""
        ell, sf2, s2, lq, _, xk, _, _ = c
        xb = xk[idx]
        mean = a.T @ params["q_mu"]
        qvar = jnp.sum((lq.T @ a) ** 2, axis=0)
        resid = y[idx] - mean
        data = -0.5 * jnp.sum(mask * (
            jnp.log(2.0 * jnp.pi * s2) + (resid * resid + qvar) / s2))
        dii = jnp.maximum(
            kernel_diag(xb, ell, sf2, kernel_name, c[7]) - jnp.sum(a * a, axis=0),
            0.0,
        )
        if mode == "trace":
            pen = jnp.sum(mask * dii) / s2
        else:
            pen = jnp.sum(mask * jnp.log1p(dii / s2))
        return data - 0.5 * pen

    def _project(params, c, flat_idx):
        """One batched inducing projection  A = L_uu^{-1} K_uf  for a flat list
        of point indices. Computing this ONCE per minibatch and gathering, in
        place of one solve per unit, is the efficient layout (a single batched
        triangular solve instead of many small ones)."""
        ell, sf2, _, _, luu, xk, _, warp = c
        return solve_triangular(
            luu,
            kernel(params["z"], xk[flat_idx], ell, sf2, kernel_name, warp),
            lower=True)

    def obj_simple(params, idx, mask, scale, mode):
        c = cache(params)
        m_ind, mb, s = params["z"].shape[0], idx.shape[0], idx.shape[1]
        # one batched projection over all mb*S minibatch points, then gather.
        a_all = _project(params, c, idx.reshape(-1)).reshape(m_ind, mb, s)
        a_all = jnp.transpose(a_all, (1, 0, 2))               # [mb, M, S]
        taus = jax.vmap(
            block_tau_from_features, in_axes=(None, 0, 0, None, 0, None))(
            params, idx, mask, c, a_all, mode)
        return -kl(params, c[3]) + scale * jnp.sum(taus)

    def obj_split(params, d_idx, d_mask, d_scale, b_idx, b_mask,
                  b_weights, kind):
        """Independent data/structure estimator.

        ``kind='gain'`` computes the exact control-variate decomposition

            F_T(data batch) + 1/2 * gain(structural blocks),

        while ``kind='direct'`` retains the v6 estimator

            likelihood(data batch) - 1/2 * logdet(structural blocks).

        ``b_weights`` contains either R/K uniform weights or per-draw
        ``1/(K p_b)`` Horvitz-Thompson weights.
        """
        c = cache(params)
        m_ind, mb, s = params["z"].shape[0], b_idx.shape[0], b_idx.shape[1]
        # Fused projection: ONE K_uf evaluation + ONE triangular solve over the
        # concatenated data batch and penalty blocks, then split. Numerically
        # identical to two separate projections, half the kernel launches.
        n_d = d_idx.shape[0]
        flat_idx = jnp.concatenate([d_idx, b_idx.reshape(-1)])
        a_all = _project(params, c, flat_idx)                  # [M, |B1|+mb*S]
        a_d = a_all[:, :n_d]                                   # [M, |B1|]
        data = block_data_from_features(params, d_idx, d_mask, c, a_d)
        if kind == "gain":
            data = data - 0.5 * diag_logdet_from_features(
                params, d_idx, d_mask, c, a_d
            )
        ab = jnp.transpose(
            a_all[:, n_d:].reshape(m_ind, mb, s),
            (1, 0, 2))
        logdets, gains = jax.vmap(
            block_logdet_and_gain_from_features,
            in_axes=(None, 0, 0, None, 0),
        )(params, b_idx, b_mask, c, ab)
        correction = (
            0.5 * jnp.sum(b_weights * gains)
            if kind == "gain"
            else -0.5 * jnp.sum(b_weights * logdets)
        )
        return -kl(params, c[3]) + d_scale * data + correction

    def metrics(params):
        ell, sf2, s2, lq, luu, _, xtk, warp = cache(params)
        a = solve_triangular(
            luu, kernel(params["z"], xtk, ell, sf2, kernel_name, warp), lower=True)
        mean = a.T @ params["q_mu"]
        d = jnp.maximum(
            kernel_diag(xtk, ell, sf2, kernel_name, warp) - jnp.sum(a * a, axis=0),
            0.0,
        )
        var = jnp.maximum(d + jnp.sum((lq.T @ a) ** 2, axis=0) + s2, 1e-8)
        rmse = jnp.sqrt(jnp.mean((yt - mean) ** 2))
        nll = jnp.mean(0.5 * (jnp.log(2.0 * jnp.pi * var)
                              + (yt - mean) ** 2 / var))
        return rmse, nll

    def vg_mode(mode):
        return jax.jit(jax.value_and_grad(
            lambda p, i, m, s: -obj_simple(p, i, m, s, mode)))

    def single_block_logdet(params, b_idx, b_mask):
        """Residual log-determinant log|I + D_b/sigma^2| of one block."""
        c = cache(params)
        a = _project(params, c, b_idx)
        v = block_v_from_features(params, b_idx, b_mask, c, a)
        return jnp.linalg.slogdet(v)[1]

    def block_logdets(params, idx, mask):
        """Residual log-determinants for all anchor blocks."""
        return jax.vmap(single_block_logdet, in_axes=(None, 0, 0))(
            params, idx, mask)

    def single_block_gain(params, b_idx, b_mask):
        """Exact non-negative tightening from diagonal T to one block."""
        c = cache(params)
        a = _project(params, c, b_idx)
        _, gain = block_logdet_and_gain_from_features(
            params, b_idx, b_mask, c, a
        )
        return gain

    def block_gains(params, idx, mask):
        return jax.vmap(single_block_gain, in_axes=(None, 0, 0))(
            params, idx, mask
        )

    return {
        "trace": vg_mode("trace"), "diag": vg_mode("diag"),
        "gain_split": jax.jit(jax.value_and_grad(
            lambda p, di, dm, ds, bi, bm, bw:
            -obj_split(p, di, dm, ds, bi, bm, bw, "gain"))),
        "direct_split": jax.jit(jax.value_and_grad(
            lambda p, di, dm, ds, bi, bm, bw:
            -obj_split(p, di, dm, ds, bi, bm, bw, "direct"))),
        "block_logdets": jax.jit(block_logdets),
        "block_gains": jax.jit(block_gains),
        "metrics": jax.jit(metrics),
    }


# --------------------------------------------------------------------------- #
# SVGP bank selection: BT random blocks and SBT inducing-anchor blocks
# --------------------------------------------------------------------------- #
def _blocks(x, z, ell, block, selection, seed, kernel_name="rbf"):
    n = x.shape[0]
    if selection == "BT":
        order = jax.random.permutation(
            jax.random.PRNGKey(seed), jnp.arange(n, dtype=jnp.int32)
        )
        return tm.pack_blocks_device(order, n, block)
    raise ValueError(f"unsupported non-SBT bank selection {selection!r}")


def build_partition_bank(x, z, ell, block, selection, seed,
                         kernel_name="rbf"):
    idx, mask, _ = _blocks(x, z, ell, block, selection, seed, kernel_name)
    return {"base_idx": idx, "base_mask": mask}


def build_method_bank(method, x, params, *, block, seed, jitter,
                      kernel_name="rbf", common_ll_batches=True,
                      ll_batch_size=None):
    """Build the common likelihood bank and, for BT/SBT, a residual bank."""
    kernel_name = canonical_kernel_name(kernel_name)
    mode, selection, _ = METHODS[method]
    x_np = np.asarray(x, dtype=np.float64)
    x = jnp.asarray(x_np)
    ell = jnp.exp(params["log_ell"])
    ll_width = max(1, int(ll_batch_size or block))
    is_split = mode == "gain"
    if is_split:
        # The likelihood and residual corrections use independent partitions.
        if selection == "BT-gain":
            residual_bank = build_partition_bank(
                x, params["z"], ell, block, "BT", int(seed), kernel_name
            )
        elif selection == "SBT-gain":
            packed = sbt_terms_device(x_np, params, block, kernel_name)
            coeffs, sbt_idx, sbt_mask = packed
            active = np.asarray(jax.device_get(coeffs)) > 0.5
            residual_bank = {
                "base_idx": np.asarray(jax.device_get(sbt_idx))[active],
                "base_mask": np.asarray(jax.device_get(sbt_mask))[active],
            }
        else:
            raise ValueError(f"unsupported paper selection {selection!r}")
        method_offset = sum((i + 1) * ord(char) for i, char in enumerate(method))
        data_seed = int(seed) if common_ll_batches else int(seed) + method_offset
        likelihood_bank = build_partition_bank(
            x, params["z"], ell, ll_width, "BT", data_seed, kernel_name
        )
        return bank_to_device({
            "base_idx": residual_bank["base_idx"],
            "base_mask": residual_bank["base_mask"],
            "data_idx": likelihood_bank["base_idx"],
            "data_mask": likelihood_bank["base_mask"],
        })
    return bank_to_device(
        build_partition_bank(
            x, params["z"], ell, ll_width, selection, seed, kernel_name
        )
    )


def bank_to_device(bank):
    out = {}
    for k, v in bank.items():
        integer = (
            k.endswith("idx")
            or k.endswith("pos")
            or k.endswith("count")
            or k in {"nominal_batch_points"}
        )
        out[k] = jnp.asarray(v, jnp.int32 if integer else jnp.float64)
    return out


# --------------------------------------------------------------------------- #
# Train-test local residual prediction
# --------------------------------------------------------------------------- #
def _predictive_components(params, x_train, x_test, kernel_name, jitter):
    """Projections every SVGP predictor needs, computed once."""
    kernel_name = canonical_kernel_name(kernel_name)
    x_train_k = jnp.asarray(x_train, dtype=jnp.float64)
    x_test_k = jnp.asarray(x_test, dtype=jnp.float64)
    ell, sf2, s2, lq, warp = transform(params)
    z = params["z"]
    kuu = kernel(z, z, ell, sf2, kernel_name, warp) + jitter * jnp.eye(z.shape[0])
    luu = jnp.linalg.cholesky(kuu)
    a_train = solve_triangular(
        luu, kernel(z, x_train_k, ell, sf2, kernel_name, warp), lower=True
    )
    a_test = solve_triangular(
        luu, kernel(z, x_test_k, ell, sf2, kernel_name, warp), lower=True
    )
    d_train = jnp.maximum(
        kernel_diag(x_train_k, ell, sf2, kernel_name, warp)
        - jnp.sum(a_train * a_train, axis=0),
        0.0,
    )
    d_test = jnp.maximum(
        kernel_diag(x_test_k, ell, sf2, kernel_name, warp)
        - jnp.sum(a_test * a_test, axis=0),
        0.0,
    )
    return {
        "x": x_train_k,
        "xs": x_test_k,
        "s2": s2,
        "a_train": a_train,
        "a_test": a_test,
        "d_train": d_train,
        "d_test": d_test,
        # The variational q-term uses L_q, unlike the collapsed stack, which is
        # the only place the two predictors differ.
        "mean": a_test.T @ params["q_mu"],
        "qvar": jnp.sum((lq.T @ a_test) ** 2, axis=0),
        "kernel_fn": lambda xa, xb: kernel(xa, xb, ell, sf2, kernel_name, warp),
    }


def predict_svgp(params, x_train, x_test, *, predictor, blocks=None,
                 kernel_name="rbf", jitter=1e-8, return_contributions=False):
    """Bound-matched predictive mean and variance for one variational method.

    ``predictor`` selects the residual covariance represented by the method's
    bound. The fitted-block predictor requires the run's complete partition.

    The residual algebra is ``common.predictors``, the same object the collapsed
    stack calls, so the two cannot drift apart.
    """
    components = _predictive_components(
        params, x_train, x_test, canonical_kernel_name(kernel_name), jitter
    )
    return pr.predictive_variance(
        components, predictor=predictor, blocks=blocks, jitter=jitter,
        return_contributions=return_contributions,
    )


def metrics_svgp(params, x_train, x_test, y_test, *, predictor, blocks=None,
                 kernel_name="rbf", jitter=1e-8):
    """RMSE and NLL of the bound-matched variational predictor."""
    mu, var = predict_svgp(
        params, x_train, x_test, predictor=predictor, blocks=blocks,
        kernel_name=kernel_name, jitter=jitter,
    )
    return pr.gaussian_rmse_nll(mu, var, y_test)


def blocks_from_bank(block_idx, block_mask):
    """Recover a run's cover from its saved packed bank."""
    import jax

    idx = np.asarray(jax.device_get(block_idx))
    mask = np.asarray(jax.device_get(block_mask)) > 0.5
    return [row[m].astype(np.int64) for row, m in zip(idx, mask) if m.any()]


def time_grad(grad_call, params, bank, sel, scale, reps=50):
    # Jit the WHOLE call (bank gathers included) so the metric measures one
    # fused executable per step, like the compiled training scan does.
    # Without this, methods whose grad_call performs more eager gathers
    # (e.g. the SBT double batch) pay per-op dispatch overhead that the real
    # training loop never sees, inflating their per-gradient time.
    grad_call = jax.jit(grad_call)
    # Compile and warm up the accelerator/BLAS path before timing.
    for warmup in range(3):
        v, g = grad_call(params, bank, sel, scale)
        jax.block_until_ready((v, g))
        if warmup == 0:
            finite = jnp.isfinite(v) & jnp.all(jnp.stack([
                jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(g)
            ]))
            if not bool(jax.device_get(finite)):
                raise FloatingPointError(
                    "non-finite value/gradient in the initial JAX warm-up"
                )
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        v, g = grad_call(params, bank, sel, scale)
        jax.block_until_ready((v, g))
        ts.append(time.perf_counter() - t)
    # Trim scheduler outliers, which otherwise dominate short CPU/JAX timings.
    ordered = sorted(ts)
    trim = max(1, len(ordered) // 10)
    kept = ordered[trim:-trim]
    mean = sum(kept) / len(kept)
    variance = sum((value - mean) ** 2 for value in kept) / len(kept)
    return float(mean * 1e3), float(math.sqrt(variance) * 1e3)


# --------------------------------------------------------------------------- #
# Hybrid minibatch training: Gaussian NatGrad for q, Adam for model parameters
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Mid-training checkpoints
#
# Checkpoint continuation remains available for diagnostics, but the official
# Gibbs campaign starts cold. A continuation has to capture more than weights:
#
#   * the optimizer state carries optax's step counter, so restoring it makes
#     the selected schedule continue instead of restarting -- no offset
#     arithmetic, and nothing to get subtly wrong;
#   * the momentum of the shared parameters carries over, so the branch does
#     not suffer a transient the continuous run never sees.
#
# Measured on a synthetic problem: a same-kernel resume reproduces the
# continuous run bit for bit, for SVGP and for SBT-SVGP alike. A stationary ->
# gated-Gibbs fork agrees to 6e-17 at the fork point and stays there under
# Adam, but drifts to ~4e-6 in the test metrics under Lion -- because Lion
# updates by sign(), which turns a 1e-15 difference into a full-size step
# whenever a gradient component sits near zero. That is a property of the
# optimizer, not of the fork, and it is four orders of magnitude below the
# effects the campaign measures.
#
# Leaves are keyed by their tree path rather than by position, because the
# Gibbs parameter tree has one leaf the stationary tree does not (``warp``).
# Position-based restoration would silently misalign every leaf after it.
# --------------------------------------------------------------------------- #
def optimizer_state_leaves(state):
    """Map ``{tree path: array}`` for an optimizer state."""
    from jax.tree_util import keystr, tree_flatten_with_path

    leaves, _ = tree_flatten_with_path(state)
    return {keystr(path): np.asarray(jax.device_get(leaf))
            for path, leaf in leaves}


def merge_optimizer_state(fresh, saved):
    """Overwrite leaves of ``fresh`` with same-path, same-shape ones of ``saved``.

    Returns ``(state, restored, missing)``. Leaves with no counterpart keep the
    freshly initialised value, which for Lion means zero momentum -- correct
    for ``warp``, since it is a parameter the stationary branch never had.
    """
    from jax.tree_util import keystr, tree_flatten_with_path, tree_unflatten

    leaves, treedef = tree_flatten_with_path(fresh)
    values, restored, missing = [], 0, []
    for path, leaf in leaves:
        key = keystr(path)
        candidate = saved.get(key)
        if candidate is not None and np.shape(candidate) == jnp.shape(leaf):
            values.append(jnp.asarray(candidate, dtype=jnp.asarray(leaf).dtype))
            restored += 1
        else:
            values.append(leaf)
            missing.append(key)
    return tree_unflatten(treedef, values), restored, missing


def save_training_checkpoint(path, epoch, params, opt_state, meta=None):
    """Write params, optimizer state and the epoch reached, in one archive."""
    arrays = {}
    for key, value in params.items():
        if not isinstance(value, (jnp.ndarray, np.ndarray)):
            raise TypeError(
                f"checkpoint supports flat array parameters only; {key!r} is "
                f"{type(value).__name__}")
        arrays[f"param::{key}"] = np.asarray(jax.device_get(value))
    for key, value in optimizer_state_leaves(opt_state).items():
        arrays[f"opt::{key}"] = value
    arrays["meta::epoch"] = np.asarray(int(epoch))
    for key, value in (meta or {}).items():
        arrays[f"meta::{key}"] = np.asarray(value)
    np.savez_compressed(path, **arrays)


def load_training_checkpoint(path):
    """Return ``(params, optimizer leaves, meta)`` written by the above."""
    loaded = np.load(path, allow_pickle=False)
    params, opt, meta = {}, {}, {}
    for key in loaded.files:
        kind, _, name = key.partition("::")
        if kind == "param":
            params[name] = jnp.asarray(loaded[key])
        elif kind == "opt":
            opt[name] = loaded[key]
        elif kind == "meta":
            meta[name] = loaded[key].item() if loaded[key].ndim == 0 else loaded[key]
    if "epoch" not in meta:
        raise ValueError(f"{path} carries no meta::epoch; not a checkpoint")
    return params, opt, meta


def train_svgp(engine, method, params, make_bank, metrics_fn, n, args,
               struct_metrics_fn=None, checkpoint_fn=None,
               start_epoch=0, resume_optimizer_leaves=None):
    """Structured minibatch SVGP training.

    Canonical RBT/SBT methods use an independent split estimator: a random global
    T-SVGP minibatch and one or more structured gain blocks. Direct-logdet and
    coupled variants are retained solely as controlled ablations.

    ``struct_metrics_fn`` (optional): callable
    ``(params, current_bank) -> (rmse, nll)`` for a second predictor recorded
    alongside the reported one. When given and ``--record-history`` is on, it is
    evaluated at every history log point.
    """
    import optax
    # Paper protocol: every method uses the same Gaussian natural-gradient
    # update for q(u). Callers may explicitly opt back into LION-q for an
    # ablation, but absence of a flag must never silently select LION-q.
    natural_q = bool(getattr(args, "natural_q", True))
    natural_lr = float(getattr(args, "natural_lr", 1e-2))
    model_lr = float(args.lr)
    z_lr = float(getattr(args, "z_lr", model_lr))
    if model_lr <= 0.0:
        raise ValueError("lr must be > 0")
    if z_lr <= 0.0:
        raise ValueError("z_lr must be > 0")
    z_update_ratio = z_lr / model_lr
    if natural_q:
        if natural_lr <= 0.0 or natural_lr > 1.0:
            raise ValueError("natural_lr must be in (0, 1]")
        if "q_cov" not in params:
            raw = params["q_raw"]
            lq = (
                jnp.tril(raw, -1)
                + jnp.diag(jnp.exp(jnp.diag(raw)))
            )
            params = dict(params)
            params["q_cov"] = lq @ lq.T
            del params["q_raw"]
    normalize_objective = bool(getattr(args, "normalize_objective", False))
    objective_scale = jnp.asarray(
        1.0 / float(n) if normalize_objective else 1.0,
        dtype=jnp.float64,
    )
    clip_threshold = float(args.grad_clip)
    track_grad_stats = bool(getattr(args, "track_grad_stats", False))
    if args.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if args.blocks_per_step < 1:
        raise ValueError("blocks_per_step must be >= 1")
    if args.log_every < 1:
        raise ValueError("log_every must be >= 1")
    if args.refresh_every < 1:
        raise ValueError("refresh_every must be >= 1")
    refresh_until = getattr(args, "refresh_until", None)
    if refresh_until is not None and int(refresh_until) < 0:
        raise ValueError("refresh_until must be >= 0")
    importance_refresh_every = int(getattr(
        args, "importance_refresh_every", args.refresh_every
    ))
    if importance_refresh_every < 1:
        raise ValueError("importance_refresh_every must be >= 1")
    importance_epsilon = float(getattr(args, "importance_epsilon", 0.05))
    if not 0.0 <= importance_epsilon < 1.0:
        raise ValueError("importance_epsilon must be in [0, 1)")

    total_epochs = int(args.epochs)

    # ---- non-stationary gate and stochastic weight averaging --------------
    # Gibbs stores one local length scale per inducing point/dimension, while
    # amplitude-RBF stores a single projection vector. Only Gibbs needs the
    # stationary warm-up gate.
    has_warp = "warp" in params
    gates_warp = has_warp and len(params["warp"].shape) == 2
    warp_gate_fraction = float(getattr(args, "warp_gate_fraction", 0.75))
    if not 0.0 <= warp_gate_fraction < 1.0:
        raise ValueError("warp_gate_fraction must be in [0, 1)")
    warp_switch_epoch = (
        int(math.ceil(warp_gate_fraction * total_epochs)) if gates_warp else 0
    )
    swa_fraction = float(getattr(args, "swa_fraction", 0.0))
    if not 0.0 <= swa_fraction <= 1.0:
        raise ValueError("swa_fraction must be in [0, 1]")
    swa_enabled = swa_fraction > 0.0
    swa_start_epoch = (
        total_epochs - int(round(swa_fraction * total_epochs))
        if swa_enabled else total_epochs
    )
    swa_sum, swa_count = None, 0
    warp_active_flag = [not gates_warp]

    # ---- fork point -------------------------------------------------------
    # total_epochs stays the full declared budget on a resumed run, so the
    # optimizer schedule, the gate epoch and the averaging window are defined
    # against the same 0..total_epochs axis whether the run starts at 0 or
    # part way through. Only the loop bound moves.
    start_epoch = int(start_epoch)
    if not 0 <= start_epoch < total_epochs:
        raise ValueError(
            f"start_epoch must lie in [0, {total_epochs}), got {start_epoch}")
    checkpoint_fraction = float(getattr(args, "checkpoint_at_fraction", 0.0))
    if not 0.0 <= checkpoint_fraction < 1.0:
        raise ValueError("checkpoint_at_fraction must be in [0, 1)")
    checkpoint_epoch = (
        int(math.ceil(checkpoint_fraction * total_epochs))
        if (checkpoint_fn is not None and checkpoint_fraction > 0.0) else 0
    )
    if checkpoint_epoch and checkpoint_epoch <= start_epoch:
        raise ValueError(
            f"checkpoint epoch {checkpoint_epoch} is not ahead of the resume "
            f"point {start_epoch}")
    # Optional explicit checkpoints, including the final epoch. An empty grid
    # preserves the historical training path and checkpoint-fraction behavior.
    checkpoint_epochs = sorted({
        int(e) for e in (getattr(args, "checkpoint_at_epochs", None) or ())
        if start_epoch < int(e) <= total_epochs
    }) if checkpoint_fn is not None else []

    # The staged protocol: which parameter groups are frozen in which epoch
    # range. The warp gate is the same mechanism applied to one leaf, so both go
    # through one masking function.
    phases = parse_phases(getattr(args, "phases", None), total_epochs)
    frozen_flag = [phases[0][1]]
    staged = any(groups for _, groups in phases)

    def gate_warp_gradient(grad):
        """Zero the gradient of every frozen leaf, warp gate included.

        Freezing is a gradient mask rather than a zero learning rate: a zero
        rate still lets Adam accumulate moment estimates, so the first
        unfrozen step would take a jump built from gradients the protocol says
        were never applied.
        """
        frozen = frozen_flag[0]
        gate_warp = gates_warp and not warp_active_flag[0]
        if not frozen and not gate_warp:
            return grad
        grad = dict(grad)
        names = {n for group in frozen for n in PARAMETER_GROUPS[group]}
        if gate_warp:
            names.add("warp")
        for name in names & set(grad):
            grad[name] = jnp.zeros_like(grad[name])
        return grad

    mode, selection, adapts = METHODS[method]
    is_gain = mode == "gain"
    is_split = is_gain
    batch_coupling = str(getattr(args, "batch_coupling", "split"))
    penalty_sampling = str(getattr(args, "penalty_sampling", "gain"))
    coupled_batches = is_gain and batch_coupling == "coupled"
    is_importance = (
        selection == "SBT-gain" and penalty_sampling == "gain"
    )
    if coupled_batches and selection != "SBT-gain":
        raise ValueError("coupled-batch diagnostic is defined only for SBT-SVGP")
    if coupled_batches and is_importance:
        raise ValueError(
            "coupled batches require uniform penalty sampling so the shared "
            "local block remains an unbiased likelihood estimator"
        )
    common_ll_batches = bool(getattr(args, "common_ll_batches", True))
    is_bt_cycle = selection in {"BT", "BT-gain"}
    if is_importance and bool(getattr(args, "full_batch", False)):
        raise ValueError(
            f"{method} importance sampling is incompatible with --full-batch"
        )
    grad_fn = engine["gain_split"] if is_gain else engine[mode]

    def grad_call(params, bank, sel, scale):
        if is_split:
            data_sel, penalty_sel = sel
            data_scale, penalty_weights = scale
            return grad_fn(
                params,
                bank["data_idx"][data_sel].reshape(-1),
                bank["data_mask"][data_sel].reshape(-1),
                data_scale,
                bank["base_idx"][penalty_sel],
                bank["base_mask"][penalty_sel],
                penalty_weights,
            )
        return grad_fn(params, bank["base_idx"][sel],
                       bank["base_mask"][sel], scale)

    t = time.perf_counter()
    bank = make_bank(params)
    # Synchronise the complete bank so search_s includes structure selection.
    jax.block_until_ready(bank)
    structure_search_s = time.perf_counter() - t
    importance_refresh_s = 0.0

    base_block_count = int(bank["base_idx"].shape[0])
    data_block_count = (
        base_block_count
        if coupled_batches
        else int(bank["data_idx"].shape[0])
        if is_split
        else base_block_count
    )
    block_width = int(bank["base_idx"].shape[1])
    data_block_width = int(
        bank["base_idx"].shape[1]
        if coupled_batches
        else bank["data_idx"].shape[1]
        if is_split
        else bank["base_idx"].shape[1]
    )

    def _importance_state(params_now):
        values_fn = engine["block_gains"] if is_gain else engine["block_logdets"]
        w = values_fn(
            params_now, bank["base_idx"], bank["base_mask"])
        w = np.asarray(jax.device_get(w), dtype=np.float64)
        w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
        total = float(w.sum())
        if total <= 1e-12:
            p = np.full(base_block_count, 1.0 / base_block_count)
        else:
            target = w / total
            p = (
                (1.0 - importance_epsilon) * target
                + importance_epsilon / base_block_count
            )
            p = p / p.sum()
        stats = {
            "importance_value_sum": total,
            "importance_value_max": float(np.max(w)) if w.size else 0.0,
            "importance_ess": float(1.0 / np.sum(p * p)),
            "importance_min_p": float(p.min()),
            "importance_max_weight": float(1.0 / p.min()),
        }
        return p, stats

    importance_stats = {}
    if is_importance:
        t_p = time.perf_counter()
        p_np, importance_stats = _importance_state(params)
        p_dev = jnp.asarray(p_np)
        jax.block_until_ready(p_dev)
        importance_refresh_s += time.perf_counter() - t_p
    else:
        p_dev = None

    # An epoch is one complete likelihood pass for every method. Structured
    # methods draw K penalty blocks per likelihood step; choosing
    # ll_batch=K*block gives one expected structural pass as well.
    penalty_units_per_step = (
        min(int(args.blocks_per_step), base_block_count) if is_split else 1
    )
    data_units_per_step = (
        penalty_units_per_step if coupled_batches else 1
    )
    steps_per_epoch = int(math.ceil(
        data_block_count / data_units_per_step
    ))
    macro_count = base_block_count
    macro_mb = penalty_units_per_step
    units_per_step = penalty_units_per_step
    full_batch = bool(getattr(args, "full_batch", False))
    if full_batch:
        data_units_per_step = data_block_count
        penalty_units_per_step = base_block_count
        macro_mb = base_block_count
        units_per_step = base_block_count
        data_units_per_step = data_block_count
        steps_per_epoch = 1
    method_epochs = args.epochs
    method_log_every = args.log_every
    method_refresh_every = args.refresh_every
    # Warmup-only refresh: no re-anchoring at or after refresh_cap epochs.
    refresh_cap = (
        method_epochs if refresh_until is None
        else min(int(refresh_until), method_epochs)
    )
    total_steps = method_epochs * steps_per_epoch
    lr_schedule = str(getattr(args, "lr_schedule", "constant"))
    if lr_schedule == "constant":
        sched = model_lr
    elif lr_schedule == "cosine":
        sched = optax.cosine_decay_schedule(
            model_lr, max(1, total_steps), 0.05
        )
    else:
        raise ValueError(
            f"unknown lr_schedule {lr_schedule!r}; choose constant or cosine"
        )
    chain = ([optax.clip_by_global_norm(args.grad_clip)]
             if args.grad_clip > 0 else [])
    chain.append(
        optax.adam(sched)
        if str(getattr(args, "optimizer", "adam")).lower() == "adam"
        else optax.lion(sched))
    opt = optax.chain(*chain)
    opt_state = opt.init(params)
    resume_restored, resume_missing = 0, []
    if resume_optimizer_leaves:
        opt_state, resume_restored, resume_missing = merge_optimizer_state(
            opt_state, resume_optimizer_leaves
        )
        if resume_restored == 0:
            raise ValueError(
                "resume: no optimizer leaf matched by path; the checkpoint "
                "does not belong to this optimizer configuration"
            )
        # Everything except the warp slots must come from the checkpoint. A
        # silent partial restore would restart the schedule and lose momentum
        # while still looking like a continuation.
        unexpected = [k for k in resume_missing if "warp" not in k]
        if unexpected:
            raise ValueError(
                "resume: these optimizer leaves have no counterpart in the "
                f"checkpoint and are not warp slots: {unexpected}"
            )

    def scale_gradient_for_optimizers(grad):
        """Scale model gradients by objective_scale, keeping NG in ELBO units."""
        scaled = jax.tree.map(lambda leaf: leaf * objective_scale, grad)
        if natural_q:
            scaled = dict(scaled)
            scaled["q_mu"] = grad["q_mu"]
            scaled["q_cov"] = grad["q_cov"]
        return scaled

    def optimizer_step(p_now, state_now, grad):
        """Adam/LION model update plus optional exact Gaussian NG for q(u)."""
        if natural_q:
            model_grad = dict(grad)
            model_grad["q_mu"] = jnp.zeros_like(grad["q_mu"])
            model_grad["q_cov"] = jnp.zeros_like(grad["q_cov"])
        else:
            model_grad = grad

        updates, state_now = opt.update(model_grad, state_now, p_now)
        updates = dict(updates)
        updates["z"] = updates["z"] * z_update_ratio
        p_next = optax.apply_updates(p_now, updates)

        if natural_q:
            covariance = 0.5 * (
                p_now["q_cov"] + p_now["q_cov"].T
            )
            covariance_grad = 0.5 * (
                grad["q_cov"] + grad["q_cov"].T
            )
            eye = jnp.eye(covariance.shape[0], dtype=covariance.dtype)
            precision = jnp.linalg.solve(covariance, eye)
            precision = 0.5 * (precision + precision.T)
            eta1 = precision @ p_now["q_mu"]

            # For expectation coordinates (m, S+mm^T):
            # d loss/d E[u] = g_m - 2 G_S m, d loss/d E[uu^T] = G_S.
            expectation_grad_1 = (
                grad["q_mu"]
                - 2.0 * covariance_grad @ p_now["q_mu"]
            )
            eta1_next = eta1 - natural_lr * expectation_grad_1
            precision_next = (
                precision + 2.0 * natural_lr * covariance_grad
            )
            precision_next = 0.5 * (
                precision_next + precision_next.T
            )

            # For the Gaussian likelihood and rho in (0, 1], this is a convex
            # combination of positive precision matrices. No spectral
            # projection is needed; only a numerical Cholesky jitter is used.
            precision_chol = jnp.linalg.cholesky(
                precision_next + 1e-10 * eye
            )
            whitened_eye = solve_triangular(
                precision_chol, eye, lower=True
            )
            covariance_next = solve_triangular(
                precision_chol.T, whitened_eye, lower=False
            )
            whitened_eta = solve_triangular(
                precision_chol, eta1_next, lower=True
            )
            mean_next = solve_triangular(
                precision_chol.T, whitened_eta, lower=False
            )
            p_next = dict(p_next)
            p_next["q_mu"] = mean_next
            p_next["q_cov"] = 0.5 * (
                covariance_next + covariance_next.T
            )

        return p_next, state_now, model_grad

    if is_split:
        data_timing_sel = jnp.arange(
            data_units_per_step, dtype=jnp.int32
        )
        penalty_timing_sel = (
            data_timing_sel
            if coupled_batches
            else jnp.arange(penalty_units_per_step, dtype=jnp.int32)
        )
        timing_sel = (data_timing_sel, penalty_timing_sel)
        timing_weights = (
            1.0 /
            (penalty_units_per_step * p_dev[:penalty_units_per_step])
            if is_importance
            else jnp.full(
                (penalty_units_per_step,),
                float(base_block_count) / penalty_units_per_step,
            )
        )
        timing_scale = (
            jnp.asarray(float(data_block_count) / data_units_per_step),
            timing_weights,
        )
    else:
        timing_sel = jnp.arange(data_units_per_step, dtype=jnp.int32)
        timing_scale = jnp.asarray(
            float(data_block_count) / data_units_per_step
        )
    grad_ms, grad_ms_std = time_grad(
        grad_call, params, bank, timing_sel, timing_scale
    )

    live = getattr(args, "live", True)
    evaluate_during_training = bool(
        getattr(args, "evaluate_during_training", False)
    )
    record_history = bool(getattr(args, "record_history", False))
    method_offset = sum((i + 1) * ord(char) for i, char in enumerate(method))
    data_master_key = jax.random.PRNGKey(
        args.seed + 1701 + (0 if common_ll_batches else method_offset)
    )
    penalty_master_key = jax.random.PRNGKey(args.seed + 170003 + method_offset)

    def reshuffled_cycle_selections(
        key,
        n_epochs,
        *,
        count,
        per_step,
        steps,
        cycle_order=None,
        start_epoch=0,
    ):
        """Fresh without-replacement cycle for each epoch-like segment."""
        count = int(count)
        per_step = int(per_step)
        steps = int(steps)
        if count < 1 or per_step < 1 or steps < 1:
            raise ValueError("cycle count, per_step and steps must be >= 1")
        per_epoch = steps * per_step
        cycles_per_epoch = int(math.ceil(per_epoch / count))
        epoch_ids = jnp.arange(
            int(start_epoch), int(start_epoch) + int(n_epochs), dtype=jnp.int32
        )
        keys = jax.vmap(
            lambda epoch_id: jax.random.split(
                jax.random.fold_in(key, epoch_id), cycles_per_epoch
            )
        )(epoch_ids)
        order = (
            cycle_order[:count]
            if cycle_order is not None
            else jnp.arange(count, dtype=jnp.int32)
        )
        permutations = jax.vmap(jax.vmap(
            lambda epoch_key: jax.random.permutation(epoch_key, order)
        ))(keys)
        rows = permutations.reshape(n_epochs, -1)[:, :per_epoch]
        return rows.reshape(n_epochs, steps, per_step)

    def importance_penalty_selections(key, n_epochs, p_device, start_epoch):
        """Draw K penalty blocks and return per-draw 1/(K p_b) weights."""
        epoch_ids = jnp.arange(
            int(start_epoch), int(start_epoch) + int(n_epochs), dtype=jnp.int32
        )
        epoch_keys = jax.vmap(lambda epoch_id: jax.random.fold_in(
            key, epoch_id
        ))(epoch_ids)
        draws = jax.vmap(lambda epoch_key: jax.random.choice(
            epoch_key,
            base_block_count,
            shape=(steps_per_epoch, penalty_units_per_step),
            replace=True,
            p=p_device,
        ))(epoch_keys).astype(jnp.int32)
        weights = 1.0 / (
            penalty_units_per_step * jnp.take(p_device, draws)
        )
        return draws, weights

    @jax.jit
    def train_segment(
        params_now,
        opt_state_now,
        selections,
        base_idx,
        base_mask,
        data_idx,
        data_mask,
        data_scale,
        uniform_penalty_weights,
    ):
        """Fused SVGP/T/BT/SBT epochs: gradient + optimizer updates in JAX.

        BT/SBT use data_idx/data_mask for the global data/T term and
        base_idx/base_mask for the structured gain.
        """
        def one_epoch(carry, epoch_selections):
            def one_step(step_carry, sel):
                p_now, state_now = step_carry
                if is_split:
                    if is_importance:
                        data_sel, penalty_sel, penalty_weights_step = sel
                    else:
                        data_sel, penalty_sel = sel
                        penalty_weights_step = uniform_penalty_weights
                    val, grad = grad_fn(
                        p_now,
                        data_idx[data_sel].reshape(-1),
                        data_mask[data_sel].reshape(-1),
                        data_scale,
                        base_idx[penalty_sel],
                        base_mask[penalty_sel],
                        penalty_weights_step,
                    )
                else:
                    val, grad = grad_fn(
                        p_now,
                        base_idx[sel],
                        base_mask[sel],
                        data_scale,
                    )
                val = val * objective_scale
                grad = gate_warp_gradient(grad)
                grad = scale_gradient_for_optimizers(grad)
                p_next, state_now, model_grad = optimizer_step(
                    p_now, state_now, grad
                )
                if track_grad_stats:
                    grad_norm = optax.global_norm(model_grad)
                    was_clipped = jnp.asarray(
                        clip_threshold > 0, dtype=jnp.float64
                    ) * (grad_norm > clip_threshold).astype(jnp.float64)
                else:
                    grad_norm = jnp.asarray(0.0, dtype=jnp.float64)
                    was_clipped = jnp.asarray(0.0, dtype=jnp.float64)
                return (p_next, state_now), (
                    val, grad_norm, was_clipped
                )

            carry, step_stats = jax.lax.scan(
                one_step, carry, epoch_selections
            )
            values, grad_norms, was_clipped = step_stats
            epoch_stats = jnp.stack((
                jnp.mean(values),
                jnp.sum(grad_norms),
                jnp.sum(jnp.square(grad_norms)),
                jnp.max(grad_norms),
                jnp.sum(was_clipped),
                jnp.asarray(grad_norms.size, dtype=jnp.float64),
            ))
            return carry, epoch_stats

        return jax.lax.scan(
            one_epoch, (params_now, opt_state_now), selections
        )

    # An explicit metric grid, so a protocol can ask for 0, 10, 25, 50, 100,
    # 150 and get exactly those. Epoch 0 is the untrained model, recorded before
    # the loop because the loop only ever reports what it has just finished.
    metric_epochs = sorted({
        int(e) for e in (getattr(args, "metrics_at_epochs", None) or ())
        if 0 <= int(e) <= total_epochs
    })

    optimizer_s, compile_s, refreshes, structures_built, curve = 0.0, 0.0, 0, 1, []
    importance_refreshes = 0
    training_history = []
    history_struct_prediction_s = 0.0
    compiled_segments = {}
    grad_norm_sum = 0.0
    grad_norm_sq_sum = 0.0
    grad_norm_max = 0.0
    clipped_steps = 0.0
    diagnostic_steps = 0.0
    epoch = start_epoch
    checkpoint_written = None
    if record_history and 0 in metric_epochs and start_epoch == 0:
        r0, n0 = map(float, jax.device_get(metrics_fn(params)))
        row0 = {"epoch": 0,
                "stochastic_objective_estimate_per_datum": float("nan"),
                "rmse": r0, "nll": n0}
        if struct_metrics_fn is not None:
            rs0, ns0 = struct_metrics_fn(params, bank)
            row0["rmse_reported"], row0["nll_reported"] = float(rs0), float(ns0)
            # Archived readers used ``full`` for every structured predictor.
            row0["rmse_full"], row0["nll_full"] = float(rs0), float(ns0)
        training_history.append(row0)
    while epoch < total_epochs:
        next_log = min(
            total_epochs,
            ((epoch // method_log_every) + 1) * method_log_every,
        )
        next_refresh = (
            min(
                total_epochs,
                ((epoch // method_refresh_every) + 1)
                * method_refresh_every,
            )
            if adapts and epoch < refresh_cap else total_epochs
        )
        next_importance_refresh = (
            min(
                total_epochs,
                ((epoch // importance_refresh_every) + 1)
                * importance_refresh_every,
            )
            if is_importance else total_epochs
        )
        segment_end = min(
            next_log, next_refresh, next_importance_refresh, total_epochs
        )
        # The gate switch, a phase edge and the start of the averaging window
        # are hard segment boundaries; inside the averaging window each segment
        # is a single epoch so that every epoch contributes one SWA snapshot.
        if gates_warp and epoch < warp_switch_epoch:
            segment_end = min(segment_end, warp_switch_epoch)
        for phase_end, groups in phases:
            if epoch < phase_end:
                segment_end = min(segment_end, phase_end)
                frozen_flag[0] = groups
                break
        # Metrics land exactly on the requested epochs, not on the nearest
        # multiple of --log-every.
        for mark in metric_epochs:
            if epoch < mark:
                segment_end = min(segment_end, mark)
                break
        # The checkpoint must land exactly on its epoch, so it is a hard
        # segment boundary like the gate switch.
        if checkpoint_epoch and epoch < checkpoint_epoch:
            segment_end = min(segment_end, checkpoint_epoch)
        for mark in checkpoint_epochs:
            if epoch < mark:
                segment_end = min(segment_end, mark)
                break
        if swa_enabled:
            segment_end = (
                min(segment_end, swa_start_epoch) if epoch < swa_start_epoch
                else min(segment_end, epoch + 1)
            )
        warp_active_flag[0] = (not gates_warp) or (epoch >= warp_switch_epoch)
        segment_epochs = segment_end - epoch
        segment_start = epoch
        if is_split:
            selection_key = (
                penalty_master_key if coupled_batches else data_master_key
            )
            data_selections = reshuffled_cycle_selections(
                selection_key,
                segment_epochs,
                count=data_block_count,
                per_step=data_units_per_step,
                steps=steps_per_epoch,
                start_epoch=segment_start,
            )
            data_scale = jnp.asarray(
                float(data_block_count) / data_units_per_step
            )
            if is_importance:
                penalty_idx, penalty_weights = importance_penalty_selections(
                    penalty_master_key,
                    segment_epochs,
                    p_dev,
                    segment_start,
                )
                selections = (
                    data_selections, penalty_idx, penalty_weights
                )
                uniform_penalty_weights = jnp.ones(
                    (penalty_units_per_step,), dtype=jnp.float64
                )
            else:
                penalty_selections = (
                    data_selections
                    if coupled_batches
                    else reshuffled_cycle_selections(
                        penalty_master_key,
                        segment_epochs,
                        count=base_block_count,
                        per_step=penalty_units_per_step,
                        steps=steps_per_epoch,
                        start_epoch=segment_start,
                    )
                )
                selections = (data_selections, penalty_selections)
                uniform_penalty_weights = jnp.full(
                    (penalty_units_per_step,),
                    float(base_block_count) / penalty_units_per_step,
                )
        else:
            selections = reshuffled_cycle_selections(
                data_master_key,
                segment_epochs,
                count=data_block_count,
                per_step=data_units_per_step,
                steps=steps_per_epoch,
                start_epoch=segment_start,
            )
            data_scale = jnp.asarray(
                float(data_block_count) / data_units_per_step
            )
            uniform_penalty_weights = jnp.ones((1,), dtype=jnp.float64)

        data_idx = (
            bank["base_idx"]
            if coupled_batches
            else bank["data_idx"]
            if is_split
            else bank["base_idx"]
        )
        data_mask = (
            bank["base_mask"]
            if coupled_batches
            else bank["data_mask"]
            if is_split
            else bank["base_mask"]
        )
        segment_args = (
            params,
            opt_state,
            selections,
            bank["base_idx"],
            bank["base_mask"],
            data_idx,
            data_mask,
            data_scale,
            uniform_penalty_weights,
        )
        # The frozen set changes which leaves receive updates, so it is part of
        # the compiled program and must key the cache.
        compile_key = (
            int(segment_epochs), bool(warp_active_flag[0]),
            tuple(sorted(frozen_flag[0])),
        )
        if compile_key not in compiled_segments:
            t_compile = time.perf_counter()
            compiled_segments[compile_key] = (
                train_segment.lower(*segment_args).compile()
            )
            compile_s += time.perf_counter() - t_compile

        t = time.perf_counter()
        (params, opt_state), segment_stats = compiled_segments[compile_key](
            *segment_args
        )
        jax.block_until_ready((params, opt_state, segment_stats))
        finite = jnp.all(jnp.isfinite(segment_stats)) & jnp.all(jnp.stack([
            jnp.all(jnp.isfinite(leaf))
            for leaf in jax.tree.leaves(params)
        ]))
        if not bool(jax.device_get(finite)):
            raise FloatingPointError(
                f"{method} became non-finite by epoch {segment_end}"
            )
        optimizer_s += time.perf_counter() - t
        segment_diag = jax.device_get(jnp.stack((
            jnp.sum(segment_stats[:, 1]),
            jnp.sum(segment_stats[:, 2]),
            jnp.max(segment_stats[:, 3]),
            jnp.sum(segment_stats[:, 4]),
            jnp.sum(segment_stats[:, 5]),
        )))
        grad_norm_sum += float(segment_diag[0])
        grad_norm_sq_sum += float(segment_diag[1])
        grad_norm_max = max(grad_norm_max, float(segment_diag[2]))
        clipped_steps += float(segment_diag[3])
        diagnostic_steps += float(segment_diag[4])
        epoch = segment_end

        if swa_enabled and epoch > swa_start_epoch:
            # One snapshot per epoch inside the averaging window. q_cov is
            # averaged too: a convex combination of PSD matrices is PSD, so
            # the averaged variational posterior stays valid.
            swa_count += 1
            swa_sum = (
                jax.tree.map(jnp.asarray, params) if swa_sum is None
                else jax.tree.map(jnp.add, swa_sum, params)
            )

        if (epoch % method_log_every == 0 or epoch == total_epochs
                or epoch in metric_epochs):
            last = float(jax.device_get(segment_stats[-1, 0]))
            stoch_obj_n = -last if normalize_objective else -last / n
            curve.append((epoch, stoch_obj_n))
            if record_history:
                r_hist, nl_hist = map(
                    float, jax.device_get(metrics_fn(params))
                )
                row = {
                    "epoch": int(epoch),
                    "stochastic_objective_estimate_per_datum": float(stoch_obj_n),
                    "rmse": r_hist,
                    "nll": nl_hist,
                }
                if struct_metrics_fn is not None:
                    # structured (block) test metric at this epoch, so the SBT
                    # curve is the structured one everywhere.
                    t_struct = time.perf_counter()
                    rs_hist, ns_hist = struct_metrics_fn(params, bank)
                    history_struct_prediction_s += (
                        time.perf_counter() - t_struct
                    )
                    row["rmse_reported"] = float(rs_hist)
                    row["nll_reported"] = float(ns_hist)
                    row["rmse_full"] = float(rs_hist)
                    row["nll_full"] = float(ns_hist)
                training_history.append(row)

        def _refresh_importance():
            """Update stopped-gradient sampling probabilities on fixed blocks."""
            nonlocal p_dev, importance_stats, importance_refresh_s
            nonlocal importance_refreshes
            if not is_importance:
                return
            t_p = time.perf_counter()
            p_np, importance_stats = _importance_state(params)
            p_dev = jnp.asarray(p_np)
            jax.block_until_ready(p_dev)
            importance_refresh_s += time.perf_counter() - t_p
            importance_refreshes += 1

        def _refresh_bank():
            """Rebuild the discrete geometry, then refresh its sampler."""
            nonlocal bank, structure_search_s, refreshes, structures_built
            t_r = time.perf_counter()
            bank = make_bank(params)
            jax.block_until_ready(bank)
            structure_search_s += time.perf_counter() - t_r
            refreshes += 1
            structures_built += 1
            _refresh_importance()

        if (epoch % method_log_every == 0 or epoch == total_epochs) and live:
            message = (
                f"     epoch {epoch:4d}/{total_epochs}  "
                f"stoch_obj/N={curve[-1][1]:+.4f}  "
            )
            if evaluate_during_training:
                r, nl = map(
                    float, jax.device_get(metrics_fn(params))
                )
                message += f"rmse={r:.4f}  nll={nl:.4f}  "
            message += f"{1e3*optimizer_s/max(1, epoch):.2f} ms/epoch"
            print(message)
        geometry_due = (
            adapts
            and epoch % method_refresh_every == 0
            and epoch < total_epochs
            and epoch < refresh_cap
        )
        importance_due = (
            is_importance
            and epoch % importance_refresh_every == 0
            and epoch < total_epochs
        )
        if geometry_due:
            _refresh_bank()
        elif importance_due:
            _refresh_importance()
        if (checkpoint_epoch and epoch == checkpoint_epoch) or epoch in checkpoint_epochs:
            # The checkpoint format stores parameters and optimizer state, not
            # a potentially large block bank. Rebuild adaptive geometry here so
            # the uninterrupted run and a resume both derive the bank from the
            # same checkpoint parameters.
            if adapts and not geometry_due:
                _refresh_bank()
            checkpoint_fn(int(epoch), params, opt_state)
            checkpoint_written = int(epoch)

    epochs_run = epoch
    post_swa_structure_rebuilds = 0
    final_structure_rebuilds = 0
    if swa_enabled and swa_count > 0:
        # Replace the last iterate by the average over the tail window. Every
        # reported metric, the saved params and the predictive diagnostics use
        # the averaged parameters from here on.
        params = jax.tree.map(lambda s: s / float(swa_count), swa_sum)
        jax.block_until_ready(params)
        if adapts:
            # The final predictor and final_blocks.npz must describe the SWA
            # parameters actually reported, not the last pre-SWA structure.
            t_final_bank = time.perf_counter()
            bank = make_bank(params)
            jax.block_until_ready(bank)
            structure_search_s += time.perf_counter() - t_final_bank
            structures_built += 1
            post_swa_structure_rebuilds = 1
            final_structure_rebuilds = 1
    elif adapts:
        # Training uses the most recent scheduled partition (for example the
        # epoch-320 partition in a 350-epoch run). Prediction artefacts must be
        # defined at the parameters that are actually saved, so rebuild once at
        # the final iterate without performing another optimiser update.
        t_final_bank = time.perf_counter()
        bank = make_bank(params)
        jax.block_until_ready(bank)
        structure_search_s += time.perf_counter() - t_final_bank
        structures_built += 1
        final_structure_rebuilds = 1
    warp_norm = (
        float(jax.device_get(jnp.linalg.norm(params["warp"])))
        if has_warp else 0.0
    )
    final_unit_count = int(base_block_count)
    final_scale = float(base_block_count) / penalty_units_per_step
    final_data_scale = float(data_block_count) / data_units_per_step
    kuf_columns = (
        data_units_per_step * data_block_width
        + penalty_units_per_step * block_width
        if is_split
        else data_units_per_step * data_block_width
    )
    base_logdet_width = int(block_width) if mode in {"block", "gain"} else 0
    search_s = structure_search_s + importance_refresh_s
    if natural_q:
        # This is a final diagnostic only.  Running the symmetric eigensolver
        # through JAX/cuSolver can hang for M >= 256 on some cluster drivers.
        # Transfer the small M x M matrix once and use the host LAPACK path.
        q_cov_host = np.asarray(jax.device_get(params["q_cov"]))
        q_cov_eigs = np.linalg.eigvalsh(
            0.5 * (q_cov_host + q_cov_host.T)
        )
        q_cov_min_eig = float(q_cov_eigs[0])
        q_cov_max_eig = float(q_cov_eigs[-1])
    else:
        q_cov_min_eig = math.nan
        q_cov_max_eig = math.nan
    info = {"search_s": search_s, "optimizer_s": optimizer_s,
            "compile_s": compile_s, "grad_ms": grad_ms,
            "grad_ms_std": grad_ms_std, "refreshes": refreshes,
            "structures_built": int(structures_built),
            "post_swa_structure_rebuilds": int(
                post_swa_structure_rebuilds
            ),
            "final_structure_rebuilds": int(final_structure_rebuilds),
            "structure_search_s": structure_search_s,
            "importance_refresh_s": importance_refresh_s,
            "importance_refreshes": int(importance_refreshes),
            "n_blocks": int(base_block_count),
            "blocks_per_step": int(units_per_step),
            "unit_scale": final_scale,
            "data_blocks": int(data_block_count),
            "data_blocks_per_step": int(data_units_per_step),
            "likelihood_batch_size": int(
                data_units_per_step * data_block_width
            ),
            "structural_block_size": int(block_width),
            "structural_blocks_per_step": int(penalty_units_per_step),
            "data_unit_scale": final_data_scale,
            "unit_count": final_unit_count,
            "nominal_batch_points": int(kuf_columns),
            "kuf_columns_per_step": int(kuf_columns),
            "base_logdet_width": base_logdet_width,
            "macro_units": int(macro_count),
            "macro_units_per_step": int(macro_mb),
            "steps_per_epoch": int(steps_per_epoch),
            "epochs": int(epochs_run),
            "epochs_cap": int(method_epochs),
            "warp_trainable": bool(has_warp),
            "warp_gate_fraction": (
                float(warp_gate_fraction) if gates_warp else None
            ),
            "staged_training": bool(staged),
            "phases": [
                {"until_epoch": int(end), "frozen": sorted(groups)}
                for end, groups in phases
            ],
            "metrics_at_epochs": list(metric_epochs),
            "warp_switch_epoch": int(warp_switch_epoch) if gates_warp else None,
            "warp_frobenius_norm": warp_norm,
            "swa_fraction": float(swa_fraction),
            "swa_start_epoch": int(swa_start_epoch) if swa_enabled else None,
            "swa_snapshots": int(swa_count),
            "start_epoch": int(start_epoch),
            "checkpoint_epoch": checkpoint_written,
            "optimizer_leaves_restored": int(resume_restored),
            # Steps actually executed here. On a resumed branch this is not
            # epochs_run * steps_per_epoch, and using that would divide the
            # measured time by the whole budget instead of the executed part.
            "optimizer_steps": int(
                (epochs_run - start_epoch) * steps_per_epoch
            ),
            "train_step_ms": (
                1e3 * optimizer_s
                / max(1, (epochs_run - start_epoch) * steps_per_epoch)
            ),
            "log_every": int(method_log_every),
            "refresh_every": int(method_refresh_every),
            "refresh_until": (
                None if refresh_until is None else int(refresh_cap)
            ),
            "compiled_epoch_scan": True,
            "fused_optimizer_step": True,
            "objective_normalized": normalize_objective,
            "curve_kind": "last_epoch_stochastic_objective_estimate_per_datum",
            "evaluated_test_during_training": (
                evaluate_during_training or record_history
            ),
            "recorded_training_history": record_history,
            "training_history": training_history,
            "history_struct_prediction_s": history_struct_prediction_s,
            "q_optimizer": "natural_gradient" if natural_q else "lion",
            "natural_lr": natural_lr if natural_q else 0.0,
            "model_lr": model_lr,
            "z_lr": z_lr,
            "lr_schedule": lr_schedule,
            "q_cov_min_eig": q_cov_min_eig,
            "q_cov_max_eig": q_cov_max_eig,
            "importance_sampling": is_importance,
            "gain_decomposition": is_gain,
            "single_batch": coupled_batches,
            "batch_coupling": batch_coupling,
            "penalty_sampling": penalty_sampling,
            "common_likelihood_batches": common_ll_batches,
            "importance_kind": "current-block-gain" if is_importance else "",
            "importance_refresh_every": int(importance_refresh_every),
            "importance_epsilon": float(importance_epsilon),
            **importance_stats,
            "grad_norm_mean": (
                grad_norm_sum / diagnostic_steps if diagnostic_steps else 0.0
            ),
            "grad_norm_rms": (
                math.sqrt(grad_norm_sq_sum / diagnostic_steps)
                if diagnostic_steps else 0.0
            ),
            "grad_norm_max": grad_norm_max,
            "clip_fraction": (
                clipped_steps / diagnostic_steps
                if diagnostic_steps else 0.0
            ),
            "sampling": (
                "coupled_selected_T_and_uniform_gain"
                if coupled_batches
                else "independent_global_T_and_residual_gain_IM"
                if is_importance
                else "independent_global_T_and_uniform_gain"
                if is_gain
                else "epoch_reshuffled_random_partition"
                if is_bt_cycle
                else "random_reshuffling"
            )}
    return params, curve, info, bank
