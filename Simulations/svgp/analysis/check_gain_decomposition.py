#!/usr/bin/env python3
"""Numerical invariants for the T+gain SVGP decomposition."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parent

from common import predictors as pr
from svgp import engine as sv  # noqa: E402


def main():
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 3))
    y = np.sin(x[:, 0]) + 0.1 * rng.normal(size=24)
    xt = rng.normal(size=(8, 3))
    yt = np.sin(xt[:, 0])
    params = sv.init_params(x, 6, 11, kernel_name="rbf")
    engine = sv.make_engine(x, y, xt, yt, kernel_name="rbf")
    bank = sv.build_partition_bank(
        jnp.asarray(x), params["z"], jnp.exp(params["log_ell"]),
        6, "BT", 19, "rbf",
    )
    bank = sv.bank_to_device(bank)
    idx, mask = bank["base_idx"], bank["base_mask"]
    all_blocks = jnp.arange(idx.shape[0], dtype=jnp.int32)
    ones = jnp.ones((idx.shape[0],), dtype=jnp.float64)
    flat_idx, flat_mask = idx.reshape(-1), mask.reshape(-1)

    loss_direct, _ = engine["direct_split"](
        params, flat_idx, flat_mask, 1.0, idx, mask, ones
    )
    loss_gain, _ = engine["gain_split"](
        params, flat_idx, flat_mask, 1.0, idx, mask, ones
    )
    loss_t, _ = engine["diag"](
        params, idx[all_blocks], mask[all_blocks], 1.0
    )
    gains = engine["block_gains"](params, idx, mask)

    direct = float(jax.device_get(loss_direct))
    gain = float(jax.device_get(loss_gain))
    t_loss = float(jax.device_get(loss_t))
    gains_np = np.asarray(jax.device_get(gains), dtype=np.float64)
    np.testing.assert_allclose(gain, direct, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(
        gain, t_loss - 0.5 * gains_np.sum(), rtol=2e-9, atol=2e-9
    )
    if gains_np.min() < -1e-8:
        raise AssertionError(f"negative block gain: {gains_np.min()}")

    epsilon = 0.05
    target = gains_np / gains_np.sum()
    p = (1.0 - epsilon) * target + epsilon / gains_np.size
    expected_ht = np.sum(p * gains_np / p)
    np.testing.assert_allclose(expected_ht, gains_np.sum(), rtol=1e-12)

    common = None
    for method in (
        "SVGP",
        "T-SVGP",
        "RBT-SVGP",
        "SBT-SVGP",
    ):
        method_bank = sv.build_method_bank(
            method,
            x,
            params,
            block=6,
            ll_batch_size=8,
            seed=23,
            jitter=1e-8,
            kernel_name="rbf",
            common_ll_batches=True,
        )
        data_idx = np.asarray(jax.device_get(
            method_bank.get("data_idx", method_bank["base_idx"])
        ))
        if common is None:
            common = data_idx
        else:
            np.testing.assert_array_equal(data_idx, common)

    # One block holding the whole training set: the decomposition then has a
    # single term and must reproduce Eq. (12) exactly.
    full_cover = [np.arange(x.shape[0], dtype=np.int64)]
    mu_local, var_local = sv.predict_svgp(
        params, x, xt, predictor="full", blocks=full_cover, kernel_name="rbf"
    )
    ell, sf2, s2, lq, warp = sv.transform(params)
    z = params["z"]
    kuu = sv.kernel(z, z, ell, sf2, "rbf", warp) + 1e-8 * jnp.eye(z.shape[0])
    luu = jnp.linalg.cholesky(kuu)
    a_train = solve_triangular(
        luu, sv.kernel(z, jnp.asarray(x), ell, sf2, "rbf"), lower=True
    )
    a_test = solve_triangular(
        luu, sv.kernel(z, jnp.asarray(xt), ell, sf2, "rbf"), lower=True
    )
    dff = sv.kernel(x, x, ell, sf2, "rbf") - a_train.T @ a_train
    dtf = sv.kernel(xt, x, ell, sf2, "rbf") - a_test.T @ a_train
    dtt_diag = sv.kernel_diag(xt, ell, sf2, "rbf") - jnp.sum(
        a_test * a_test, axis=0
    )
    chol = jnp.linalg.cholesky(
        0.5 * (dff + dff.T) + (s2 + 1e-8) * jnp.eye(x.shape[0])
    )
    solved = solve_triangular(chol, dtf.T, lower=True)
    expected_mu = a_test.T @ params["q_mu"]
    expected_var = (
        jnp.sum((lq.T @ a_test) ** 2, axis=0)
        + dtt_diag
        - jnp.sum(solved * solved, axis=0)
        + s2
    )
    mu_full, var_full = sv.predict_svgp(
        params, x, xt, predictor="full", kernel_name="rbf"
    )
    np.testing.assert_allclose(mu_local, expected_mu, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(var_local, expected_var, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(mu_full, expected_mu, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(var_full, expected_var, rtol=2e-8, atol=2e-8)

    # The cover-invariance identity, checked against the single-block result
    # computed above.
    # A ragged multi-block cover, including a short final block. Splitting the
    # same correction over five blocks instead of one must not move it: the
    # cover changes how Eq. (12) is distributed, never its value.
    n_points = x.shape[0]
    local_cover = [
        np.asarray(part, dtype=np.int64)
        for part in np.array_split(np.arange(n_points), 5)
    ]
    mu_blocks, var_blocks = sv.predict_svgp(
        params, x, xt, predictor="full", blocks=local_cover, kernel_name="rbf"
    )
    qvar_np = np.asarray(jax.device_get(jnp.sum((lq.T @ a_test) ** 2, axis=0)))
    s2_np = float(jax.device_get(s2))
    sparse_var = qvar_np + np.asarray(jax.device_get(dtt_diag)) + s2_np

    # Splitting the correction over five blocks must land on the same numbers
    # as one block, which in turn matched the direct Eq. (12) above.
    np.testing.assert_allclose(mu_blocks, expected_mu, rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(var_blocks, var_local, rtol=0, atol=1e-10)
    np.testing.assert_allclose(var_blocks, expected_var, rtol=2e-8, atol=2e-8)

    # The per-block terms are the pieces that identity is made of: each one
    # non-negative, and their partial sums the nested conditionings.
    components = sv._predictive_components(params, x, xt, "rbf", 1e-8)
    contributions = pr.residual_contributions(
        components, predictor="full", blocks=local_cover
    )
    if contributions.shape[1] != len(local_cover):
        raise AssertionError("one contribution column per block")
    if contributions.min() < -1e-12:
        raise AssertionError("negative per-block contribution")
    np.testing.assert_allclose(
        contributions.sum(axis=1),
        np.asarray(jax.device_get(jnp.sum(solved * solved, axis=0))),
        rtol=2e-8, atol=2e-8,
    )
    if not np.all(np.isfinite(var_blocks)):
        raise AssertionError("non-finite predictive variance")
    if np.any(var_blocks > sparse_var + 1e-8):
        raise AssertionError("conditioning increased the residual variance")
    if np.any(var_blocks < qvar_np + s2_np - 1e-8):
        raise AssertionError("residual variance became negative")

    print(
        "gain decomposition OK: "
        f"loss={gain:.10f}, blocks={gains_np.size}, "
        f"gain_sum={gains_np.sum():.10f}, ESS={1.0/np.sum(p*p):.2f}; "
        f"Eq. (12) cover-invariant over {len(local_cover)} blocks, "
        f"max|1-block - 5-block| = "
        f"{float(np.max(np.abs(var_blocks - var_local))):.2e}"
    )


if __name__ == "__main__":
    main()
