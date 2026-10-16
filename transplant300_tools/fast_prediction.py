"""Prepared, memory-bounded SVGP inference; no training or artifact writes.

Unlike the historical end-to-end predictor, preparation is reusable. Timing
must report preparation/compilation separately from warm prediction. All fitted
blocks are included, including ragged blocks; no neighbourhood truncation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from common import kernels as kn
from common.predictors import VARIANCE_FLOOR, cover_order

IMPLEMENTATION = "svgp-prepared-batched-prediction-v1"


def _pad_rows(a, width):
    extra = (-a.shape[0]) % width
    return jnp.pad(a, [(0, extra)] + [(0, 0)] * (a.ndim - 1))


class PreparedSVGP:
    """One immutable model, repeatedly callable on test data.

    ``block_batch`` bounds simultaneous block factorizations/solves. Cached
    factors are capped by ``factor_cache_bytes``; larger covers stream one
    block batch at a time, reusing its factors across every test chunk. Training
    projections cost O(NM) storage, cached factors at most the given budget.
    Compilation workspace and input/output arrays are additional memory.
    """

    def __init__(self, params, x_train, *, predictor, blocks=None,
                 kernel_name="rbf", jitter=1e-8, test_chunk=256,
                 train_chunk=512, block_batch=4, factor_cache_bytes=128 << 20):
        if predictor not in {"sparse", "diagonal", "block"}:
            raise ValueError("prepared predictor supports sparse, diagonal, block")
        if min(test_chunk, train_chunk, block_batch) < 1 or factor_cache_bytes < 0:
            raise ValueError("chunk/batch sizes must be positive; cache nonnegative")
        from svgp.engine import transform

        self.predictor = predictor
        self.test_chunk = int(test_chunk)
        self.block_batch = int(block_batch)
        x = jnp.asarray(x_train, dtype=jnp.float64)
        if x.ndim != 2 or not x.shape[0]:
            raise ValueError("nonempty rank-2 training inputs required")
        self.input_dim = x.shape[1]
        ell, sf2, s2, lq, warp = transform(params)
        z, mu = params["z"], params["q_mu"]
        name = kn.canonical_kernel_name(kernel_name)

        def kernel(a, b):
            return kn.kernel(a, b, ell, sf2, name, warp)

        luu = jax.jit(lambda: jnp.linalg.cholesky(
            kernel(z, z) + jitter * jnp.eye(z.shape[0])))()

        @jax.jit
        def project(rows):
            a = solve_triangular(luu, kernel(z, rows), lower=True)
            d = jnp.maximum(kn.kernel_diag(rows, ell, sf2, name, warp)
                            - jnp.sum(a * a, axis=0), 0.0)
            return a.T, d

        def test_base(rows):
            a, d = project(rows)
            return a, a @ mu, jnp.sum((a @ lq) ** 2, axis=1), d

        self.factor_bytes = 0
        self.factors_cached = False
        self.n_blocks = 0
        self._factors = None
        # Sparse transplant inference never constructs an M x N projection.
        if predictor == "sparse":
            @jax.jit
            def evaluate(test_batches):
                def one(rows):
                    _, mean, qvar, d = test_base(rows)
                    return mean, jnp.maximum(qvar + d + s2, VARIANCE_FLOOR)
                return jax.lax.map(one, test_batches)
            self._evaluate = evaluate
            self._args = ()
            jax.block_until_ready(luu)
            return

        width = min(int(train_chunk), x.shape[0])
        padded = _pad_rows(x, width)
        train_a, train_d = jax.jit(lambda rows: jax.lax.map(project, rows))(
            padded.reshape(-1, width, x.shape[1]))
        train_a = train_a.reshape(-1, z.shape[0])[:x.shape[0]]
        train_d = train_d.reshape(-1)[:x.shape[0]]

        if predictor == "diagonal":
            xa = _pad_rows(x, width).reshape(-1, width, x.shape[1])
            aa = _pad_rows(train_a, width).reshape(-1, width, z.shape[0])
            dd = _pad_rows(train_d + s2, width).reshape(-1, width)
            mask = _pad_rows(jnp.ones(x.shape[0]), width).reshape(-1, width)

            @jax.jit
            def evaluate(test_batches, xa, aa, dd, mask):
                def one(rows):
                    at, mean, qvar, dt = test_base(rows)
                    def add(total, batch):
                        xb, ab, den, valid = batch
                        cross = kernel(rows, xb) - at @ ab.T
                        value = jnp.sum(cross ** 2 * valid[None, :]
                                        / jnp.where(valid > 0, den, 1)[None, :], axis=1)
                        return total + value, None
                    correction, _ = jax.lax.scan(
                        add, jnp.zeros(rows.shape[0]), (xa, aa, dd, mask))
                    return mean, jnp.maximum(
                        qvar + jnp.maximum(dt - correction, 0) + s2, VARIANCE_FLOOR)
                return jax.lax.map(one, test_batches)
            self._evaluate = evaluate
            self._args = (xa, aa, dd, mask)
        else:
            order, _ = cover_order(blocks if blocks is not None else [])
            if not np.array_equal(np.sort(order), np.arange(x.shape[0])):
                raise ValueError("blocks must partition exactly the training indices")
            self.n_blocks = len(blocks)
            bwidth = max(len(b) for b in blocks)
            count = ((len(blocks) + block_batch - 1) // block_batch) * block_batch
            idx = np.zeros((count, bwidth), dtype=np.int32)
            valid = np.zeros((count, bwidth), dtype=np.float64)
            for i, b in enumerate(blocks):
                idx[i, :len(b)] = b
                valid[i, :len(b)] = 1
            idx = jnp.asarray(idx.reshape(-1, block_batch, bwidth))
            valid = jnp.asarray(valid.reshape(-1, block_batch, bwidth))

            def factor_one(xb, ab, vb):
                d = kernel(xb, xb) - ab @ ab.T
                d = (d + d.T) * 0.5 * vb[:, None] * vb[None, :]
                # Padding is independent with unit variance, and has zero
                # test cross-covariance, so it contributes exactly zero.
                d = d + jnp.diag(jnp.where(vb > 0, s2 + jitter, 1.0))
                return jnp.linalg.cholesky(d)

            factor_batch = jax.vmap(factor_one)
            self.factor_bytes = count * bwidth * bwidth * x.dtype.itemsize
            self.factors_cached = self.factor_bytes <= factor_cache_bytes
            if self.factors_cached:
                self._factors = jax.jit(lambda x, a, ids, masks: jax.lax.map(
                    lambda pair: factor_batch(x[pair[0]], a[pair[0]], pair[1]),
                    (ids, masks)))(x, train_a, idx, valid)

            cached = self.factors_cached

            @jax.jit
            def evaluate(test_batches, x, train_a, idx, valid, factors):
                at, mean, qvar, dt = jax.lax.map(test_base, test_batches)

                def batch_contribution(ids, vb, chol):
                    xb, ab = x[ids], train_a[ids]
                    def one_test(pair):
                        rows, a = pair
                        def one_block(xb, ab, v, chol):
                            cross = (kernel(rows, xb) - a @ ab.T) * v[None, :]
                            solved = solve_triangular(chol, cross.T, lower=True)
                            return jnp.sum(solved ** 2, axis=0)
                        return jnp.sum(jax.vmap(one_block)(xb, ab, vb, chol), axis=0)
                    return jax.lax.map(one_test, (test_batches, at))

                def add(total, batch):
                    if cached:
                        ids, vb, chol = batch
                    else:
                        ids, vb = batch
                        chol = factor_batch(x[ids], train_a[ids], vb)
                    return total + batch_contribution(ids, vb, chol), None

                batches = (idx, valid, factors) if cached else (idx, valid)
                correction, _ = jax.lax.scan(add, jnp.zeros_like(dt), batches)
                return mean, jnp.maximum(qvar + jnp.maximum(dt - correction, 0)
                                         + s2, VARIANCE_FLOOR)

            self._evaluate = evaluate
            self._args = (x, train_a, idx, valid, self._factors)
        jax.block_until_ready((luu, self._args))

    def __call__(self, x_test):
        """Return device arrays; caller decides when to synchronize/transfer."""
        xs = jnp.asarray(x_test, dtype=jnp.float64)
        if xs.ndim != 2 or xs.shape[1] != self.input_dim:
            raise ValueError("test input feature dimension does not match training")
        n = xs.shape[0]
        if not n:
            return jnp.empty(0), jnp.empty(0)
        batches = _pad_rows(xs, self.test_chunk).reshape(
            -1, self.test_chunk, xs.shape[1])
        mean, variance = self._evaluate(batches, *self._args)
        return mean.reshape(-1)[:n], variance.reshape(-1)[:n]
