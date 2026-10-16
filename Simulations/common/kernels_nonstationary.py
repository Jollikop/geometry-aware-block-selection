#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""common.kernels_nonstationary -- input-dependent lengthscales and amplitude.

Isolation contract
------------------
Every stationary campaign (RBF, Matern-3/2, Matern-5/2, periodic) imports
``common.kernels`` but never the symbols defined here: the only
reference is the dispatch branch taken for the non-stationary kernel names.

Consequently, edits confined to this file cannot change any stationary result,
and that claim is checkable rather than asserted -- see
``freeze_campaign.py --verify``, which reports the diff of a frozen campaign's
declared dependency set between two commits. Keep it that way: iterations on
the non-stationary model belong here, not in ``common/kernels.py``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

WARP_PARAMETERIZATION = "ell_d*(1+sigmoid(w_d.x))_v1"

# Multiplier applied to every lengthscale when the warp is zero: sigmoid(0)=1/2.
# A gated Gibbs kernel is therefore the stationary kernel of lengthscale
# GATED_LENGTHSCALE_FACTOR * ell, not of lengthscale ell.
GATED_LENGTHSCALE_FACTOR = 1.5


def stationary_to_gated_log_ell(log_ell):
    """Re-express a stationary log-lengthscale in the gated Gibbs parameters.

    Resuming a Gibbs run from a stationary checkpoint has to preserve the
    *effective* lengthscale, not the raw parameter: copying ``log_ell`` across
    verbatim would multiply it by GATED_LENGTHSCALE_FACTOR and restart the run
    somewhere the stationary branch never was, which is precisely what a
    checkpointed fork is meant to avoid.
    """
    import math

    return log_ell - math.log(GATED_LENGTHSCALE_FACTOR)


def gibbs_lengthscales(x, ell, warp=None):
    """Input-dependent ARD lengthscales ``ell_d(x)``.

    ``ell_d(x) = ell_d * (1 + sigmoid(w_d . x))`` where row ``d`` of ``warp``
    is ``w_d``. The multiplier lies in ``(1, 2)``, so local lengthscales are
    positive and bounded by construction, with no constraint on ``warp``.

    ``warp=None`` or an all-zero ``warp`` gives ``ell_d(x) = 1.5 * ell_d``
    everywhere, i.e. exactly the stationary RBF of lengthscale ``1.5 * ell``.
    This is the gated state used before the non-stationary phase opens.
    """
    if warp is None:
        return jnp.broadcast_to(1.5 * ell, x.shape)
    proj = jnp.matmul(x, jnp.swapaxes(warp, -1, -2))
    return ell * (1.0 + jax.nn.sigmoid(proj))


def gibbs(x1, x2, ell, sf2, warp=None):
    """ARD Gibbs kernel with trainable input-dependent lengthscales.

    k(x,x') = sf2 * prod_d [2 l_d(x) l_d(x') / (l_d(x)^2 + l_d(x')^2)]^{1/2}
              * exp( -sum_d (x_d - x'_d)^2 / (l_d(x)^2 + l_d(x')^2) )

    On the diagonal the prefactor is 1 and the exponent 0, so ``k(x,x) = sf2``
    exactly as for the stationary kernels: the marginal variance is unchanged
    and the non-stationarity acts only on the correlation.
    """
    l1 = gibbs_lengthscales(x1, ell, warp)
    l2 = gibbs_lengthscales(x2, ell, warp)
    l1e = l1[..., :, None, :]
    l2e = l2[..., None, :, :]
    num = 2.0 * l1e * l2e
    den = jnp.maximum(l1e * l1e + l2e * l2e, 1e-30)
    pref = jnp.sqrt(jnp.prod(num / den, axis=-1))
    diff = x1[..., :, None, :] - x2[..., None, :, :]
    expo = jnp.sum(diff * diff / den, axis=-1)
    return sf2 * pref * jnp.exp(-expo)
