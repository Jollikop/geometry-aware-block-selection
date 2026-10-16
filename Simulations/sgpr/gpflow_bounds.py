#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collapsed GPflow bounds: the TensorFlow half of the SGPR stack.

One family, four bounds on the same collapsed marginal likelihood, differing
only in how they treat the residual log-determinant:

* ``SGPR``              the standard GPflow trace bound (re-exported);
* ``TighterSGPR``       diagonal correction                    (T-SGPR);
* ``FullResidualSGPR``  dense correction, the endpoint         (E-SGPR);
* ``StructuredSGPR``    padded block corrections           (BT/SBT-SGPR).

Only the Snelson figure and the residual illustration use this path; the
benchmarks use the JAX engine in ``sgpr.jax_collapsed``. Both consume the same
block covers from ``common.partitions``.
"""

import numpy as np
import tensorflow as tf
from check_shapes import check_shapes
from gpflow.config import default_float, default_jitter
from gpflow.kernels import SquaredExponential
from gpflow.models.sgpr import SGPR  # re-export
from gpflow.utilities import to_default_float

from common import terms as tm


class TighterSGPR(SGPR):
    """SGPR with the diagonal Jensen log-det correction."""

    @check_shapes("return: []")
    def logdet_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        sigma_sq = common.sigma_sq
        lb = common.LB
        a = common.A

        x, y = self.data
        outdim = to_default_float(tf.shape(y)[1])
        kdiag = self.kernel(x, full_cov=False)
        qdiag = tf.reduce_sum(tf.square(a), 0)
        correction = tf.reduce_sum(tf.math.log1p(kdiag / sigma_sq - qdiag))

        half_logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(lb)))
        log_sigma_sq = tf.reduce_sum(tf.math.log(sigma_sq))
        return -outdim * (
            half_logdet_b + 0.5 * log_sigma_sq + 0.5 * correction
        )


class FullResidualSGPR(SGPR):
    """SGPR with the dense residual log-det correction.

    This is the non-scalable dense endpoint of the residual-correction
    hierarchy:
    ``logdet(I + sigma^{-2} (Kff - Qff))``. It is intended for small-data
    diagnostics and method comparisons, not for large benchmark sweeps.
    """

    @check_shapes("return: []")
    def logdet_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        sigma_sq = common.sigma_sq
        lb = common.LB
        a = common.A

        x, y = self.data
        outdim = to_default_float(tf.shape(y)[1])
        n = tf.shape(x)[0]

        sigma_vec = tf.reshape(sigma_sq, [-1])
        sigma_vec = tf.cond(
            tf.equal(tf.size(sigma_vec), 1),
            lambda: tf.fill([n], sigma_vec[0]),
            lambda: sigma_vec,
        )
        scale = tf.sqrt(sigma_vec[:, None] * sigma_vec[None, :])

        kff = self.kernel(x)
        qff_scaled = tf.linalg.matmul(a, a, transpose_a=True)
        residual_scaled = kff / scale - qff_scaled
        residual_scaled = 0.5 * (
            residual_scaled + tf.transpose(residual_scaled)
        )

        h = tf.eye(n, dtype=default_float()) + residual_scaled
        h = 0.5 * (h + tf.transpose(h)) + default_jitter() * tf.eye(
            n, dtype=default_float()
        )
        chol = tf.linalg.cholesky(h)
        correction = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(chol))
        )

        half_logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(lb)))
        log_sigma_sq = tf.reduce_sum(tf.math.log(sigma_sq))
        return -outdim * (
            half_logdet_b + 0.5 * log_sigma_sq + 0.5 * correction
        )


class StructuredSGPR(SGPR):
    """Collapsed SGPR bound with signed residual log-det terms.

    ``terms`` is a list of ``(coeff, idx)``. In the active simulations all
    coefficients are positive block weights and the residual penalty is
    ``sum coeff * logdet(I + D_idx / sigma^2)``. The net coefficient of every
    training point must be exactly one, as in the BT/SBT block covers.
    """

    def __init__(self, *args, terms=None, structure_capacity=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.terms: list[tuple[float, np.ndarray]] = []
        self._capacity = None                 # (T_cap, S_cap)
        self._requested_capacity = structure_capacity
        self._coeffs_var = None               # [T_cap]
        self._idx_var = None                  # [T_cap, S_cap] int32
        self._mask_var = None                 # [T_cap, S_cap]
        if terms is not None:
            self.set_terms(terms)

    def _ensure_capacity(self, tmax: int, smax: int) -> None:
        """Allocate fixed-shape structure Variables once (padded form).

        The penalty is evaluated on padded tensors of static shape
        [T_cap, S_cap], so re-selecting the structure only assigns the
        Variables and never re-traces a compiled training loss. The capacity
        is fixed at the first ``set_terms`` (with headroom on T); exceeding
        it later raises instead of silently invalidating compiled functions.
        """
        if self._capacity is None:
            if self._requested_capacity is not None:
                cap_t, cap_s = self._requested_capacity
                cap_t = max(int(cap_t), tmax)
                cap_s = max(int(cap_s), smax)
            else:
                cap_t = max(2 * tmax, tmax + 8)
                cap_s = smax
            self._capacity = (cap_t, cap_s)
            self._coeffs_var = tf.Variable(
                tf.zeros([cap_t], dtype=default_float()), trainable=False)
            self._idx_var = tf.Variable(
                tf.zeros([cap_t, cap_s], dtype=tf.int32), trainable=False)
            self._mask_var = tf.Variable(
                tf.zeros([cap_t, cap_s], dtype=default_float()),
                trainable=False)
        elif tmax > self._capacity[0] or smax > self._capacity[1]:
            raise ValueError(
                f"structure ({tmax} terms, max size {smax}) exceeds padded "
                f"capacity {self._capacity}; build the model with "
                f"structure_capacity=(T, S) large enough")

    def set_terms(self, terms) -> None:
        """Validate a block cover and write it into the padded Variables.

        ``terms`` is ``[(coeff, idx)]``. The net coefficient of every training
        point must be exactly one: that is the condition that makes the
        correction a valid bound, and it is far cheaper to check here than to
        diagnose from a wrong number later.
        """
        x, _ = self.data
        num_data = int(x.shape[0])
        clean = []
        for coeff, idx in terms:
            idx = np.array(sorted(set(map(int, idx))), dtype=np.int32)
            coeff = float(coeff)
            if idx.size == 0 or coeff == 0.0:
                continue
            if np.any(idx < 0) or np.any(idx >= num_data):
                raise ValueError("term index out of range")
            clean.append((coeff, idx))
        if not clean:
            raise ValueError("structure must contain at least one term")

        blocks = [idx for _, idx in clean]
        self._ensure_capacity(len(clean), max(len(idx) for idx in blocks))
        cap_t, cap_s = self._capacity

        packed = tm.pack_terms(
            blocks, cap_s, num_data,
            coeffs=[coeff for coeff, _ in clean],
            capacity=cap_t, as_jax=False,
        )
        if not np.allclose(tm.net_coefficients(packed, num_data),
                           np.ones(num_data), atol=1e-5):
            raise ValueError("net coefficient of every point must be 1")

        self.terms = clean
        coeffs, idx_pad, mask = packed
        self._coeffs_var.assign(coeffs)
        self._idx_var.assign(idx_pad)
        self._mask_var.assign(mask)

    @check_shapes("return: []")
    def logdet_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        if not self.terms:
            raise ValueError("set a block structure before training")

        sigma_sq = common.sigma_sq
        lb = common.LB
        x, y = self.data
        outdim = to_default_float(tf.shape(y)[1])

        correction = self.block_correction(common, x, sigma_sq)

        half_logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(lb)))
        log_sigma_sq = tf.reduce_sum(tf.math.log(sigma_sq))
        return -outdim * (
            half_logdet_b + 0.5 * log_sigma_sq + 0.5 * correction
        )

    def block_correction(self, common, x, sigma_sq) -> tf.Tensor:
        """Padded batched residual block penalty.

        Works on the fixed-shape Variables set by ``set_terms``: one batched
        Cholesky over [T_cap, S_cap, S_cap]. Padded rows/columns are masked
        to an identity sub-block (exact zero log-det contribution) and padded
        terms have coefficient zero, so the value equals the per-term sum.
        """
        idx = self._idx_var
        mask = self._mask_var
        coeffs = self._coeffs_var

        x_blocks = tf.gather(x, idx, axis=0)              # [T, S, D]
        sigma_blocks = tf.gather(sigma_sq, idx, axis=0)   # [T, S]
        a_blocks = tf.transpose(
            tf.gather(common.A, idx, axis=1), perm=[1, 0, 2])  # [T, M, S]
        q_blocks = tf.linalg.matmul(a_blocks, a_blocks, transpose_a=True)

        if isinstance(self.kernel, SquaredExponential):
            variance = tf.cast(self.kernel.variance, default_float())
            lengthscales = tf.cast(self.kernel.lengthscales, default_float())
            x_scaled = x_blocks / lengthscales                       # [T, S, D]
            # Squared distances via the inner-product identity
            #     ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            # to avoid materialising the [T, S, S, D] pairwise-difference
            # tensor. That intermediate is the peak-memory term and scales with
            # the block width S: for cap_s=500, D=8 it is ~1.3 GB before
            # autodiff, which can overflow GPU memory. The
            # matmul form keeps everything at [T, S, S].
            sq = tf.reduce_sum(tf.square(x_scaled), axis=-1)         # [T, S]
            cross = tf.matmul(x_scaled, x_scaled, transpose_b=True)  # [T, S, S]
            d2 = sq[:, :, None] + sq[:, None, :] - 2.0 * cross
            d2 = tf.maximum(d2, 0.0)
            k_blocks = variance * tf.exp(-0.5 * d2)
        else:
            k_blocks = self.kernel(x_blocks)              # batched full cov

        scale = tf.sqrt(sigma_blocks[:, :, None] * sigma_blocks[:, None, :])
        residual_scaled = k_blocks / scale - q_blocks
        residual_scaled = 0.5 * (
            residual_scaled + tf.transpose(residual_scaled, perm=[0, 2, 1])
        )

        cap_t = tf.shape(idx)[0]
        cap_s = tf.shape(idx)[1]
        eye = tf.eye(cap_s, batch_shape=[cap_t], dtype=default_float())
        pair_mask = mask[:, :, None] * mask[:, None, :]
        block_matrix = (
            ((1.0 + default_jitter()) * eye + residual_scaled) * pair_mask
            + tf.linalg.diag(1.0 - mask)
        )
        # Optionally chunk the batched Cholesky over the term axis. A large
        # batch of *large* blocks can crash
        # some GPU cuSOLVER builds with an illegal memory access (segfault).
        # Chunking keeps each potrf small while leaving the result bit-for-bit
        # identical -- BUT it replaces one batched call with several, adding
        # kernel-launch overhead per step. So only chunk in the dangerous
        # large-block regime; small-block methods (BT/SBT with block <= ~100)
        # keep the single fast batched Cholesky.
        t_cap = self._capacity[0] if self._capacity is not None else None
        s_cap = self._capacity[1] if self._capacity is not None else None
        chunk = 16
        if (t_cap is not None and s_cap is not None
                and int(s_cap) > 256 and int(t_cap) > chunk):
            parts = []
            for start in range(0, int(t_cap), chunk):
                ch = tf.linalg.cholesky(block_matrix[start:start + chunk])
                parts.append(2.0 * tf.reduce_sum(
                    tf.math.log(tf.linalg.diag_part(ch)), axis=1))
            logdets = tf.concat(parts, axis=0)
        else:
            chol = tf.linalg.cholesky(block_matrix)
            logdets = 2.0 * tf.reduce_sum(
                tf.math.log(tf.linalg.diag_part(chol)), axis=1)
        return tf.reduce_sum(coeffs * logdets)
