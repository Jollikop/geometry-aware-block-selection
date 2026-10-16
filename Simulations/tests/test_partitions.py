#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The block covers are what the paper claims they are.

Two things were previously asserted only in prose:

1. the host (GPflow) and device (JAX) SBT selectors "implement the same
   geometry" -- nothing checked it, and the figure and the table use different
   ones;
2. every cover is an exact partition with unit net coefficient, which is what
   makes the bound valid.

What the tests below establish, measured rather than assumed:

* in ``center`` mode the two paths produce *identical* orderings, for every
  kernel and dimension tried;
* in ``boundary`` mode (the default) they are identical in 1-D, which is the
  only case the host path is actually used for -- Snelson and the residual
  illustration;
* in ``boundary`` mode above 1-D they order points differently (they sort on a
  difference of distances, and the two metrics are monotonically related but
  not affinely), while still agreeing on which anchor owns which point and
  placing 80-95% of points in the same block.

The last point is a latent hazard, not a current bug: no experiment runs the
host selector above one dimension. The threshold below exists so that if one
ever does, this fails loudly instead of quietly producing a figure that no
longer illustrates the table.
"""
from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from common import partitions as pt  # noqa: E402
from common import terms as tm  # noqa: E402

jax.config.update("jax_enable_x64", True)

CASES = [
    # (n, d, m, block, kernel)
    (200, 1, 5, 20, "rbf"),
    (200, 1, 8, 25, "rbf"),
    (500, 3, 16, 50, "rbf"),
    (500, 3, 16, 50, "matern32"),
    (1000, 2, 32, 100, "matern52"),
]


def _problem(n, d, m, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    z = x[rng.permutation(n)[:m]].copy()
    ell = np.full(d, 0.7)
    return x, z, ell


class BlockCoversAreValid(unittest.TestCase):
    def test_sbt_host_cover_is_an_exact_partition(self):
        for n, d, m, block, _ in CASES:
            with self.subTest(n=n, d=d, m=m):
                x, z, ell = _problem(n, d, m)
                blocks = pt.sbt_blocks_host(x, z, ell, block)
                tm.assert_partition(blocks, n)

    def test_sbt_device_cover_is_an_exact_partition(self):
        for n, d, m, block, kernel_name in CASES:
            with self.subTest(n=n, kernel=kernel_name):
                x, z, ell = _problem(n, d, m)
                params = {"z": jnp.asarray(z), "log_ell": jnp.log(jnp.asarray(ell))}
                packed = pt.sbt_terms_device(x, params, block, kernel_name)
                tm.assert_partition(tm.unpack_terms(packed), n)

    def test_bt_device_cover_is_an_exact_partition(self):
        for n, _, _, block, _ in CASES:
            with self.subTest(n=n):
                packed = pt.bt_terms_device(n, block, seed=3)
                tm.assert_partition(tm.unpack_terms(packed), n)

    def test_bt_device_is_reproducible_from_the_seed_alone(self):
        first = tm.unpack_terms(pt.bt_terms_device(500, 50, seed=7))
        second = tm.unpack_terms(pt.bt_terms_device(500, 50, seed=7))
        other = tm.unpack_terms(pt.bt_terms_device(500, 50, seed=8))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(first, other)))

    def test_net_coefficient_of_every_point_is_one(self):
        x, z, ell = _problem(500, 3, 16)
        blocks = pt.sbt_blocks_host(x, z, ell, 50)
        packed = tm.pack_terms(blocks, 50, 500, as_jax=False)
        np.testing.assert_allclose(tm.net_coefficients(packed, 500), np.ones(500))


class HostAndDeviceSelectorsAgree(unittest.TestCase):
    """Properties that must hold exactly, plus a measured ordering distance."""

    def _orders(self, n, d, m, kernel_name, split_mode):
        x, z, ell = _problem(n, d, m)
        host_order, host_parent, _, _ = pt.anchor_order_host(
            x, z, ell, split_mode
        )
        device_order = np.asarray(jax.device_get(pt.anchor_order_device(
            jnp.asarray(x), jnp.asarray(z), jnp.asarray(ell),
            kernel_name, split_mode,
        )))
        return x, z, ell, host_order, device_order, host_parent

    def test_center_mode_orderings_are_identical(self):
        # In ``center`` mode both paths sort by distance to the parent anchor,
        # and the kernel-induced distance is a monotone function of the ARD
        # one, so the orderings must match exactly.
        for n, d, m, _, kernel_name in CASES:
            with self.subTest(n=n, kernel=kernel_name):
                _, _, _, host, device, _ = self._orders(
                    n, d, m, kernel_name, "center"
                )
                np.testing.assert_array_equal(host, device)

    def test_both_orderings_are_permutations(self):
        for n, d, m, _, kernel_name in CASES:
            for mode in pt.SPLIT_MODES:
                with self.subTest(n=n, kernel=kernel_name, mode=mode):
                    _, _, _, host, device, _ = self._orders(
                        n, d, m, kernel_name, mode
                    )
                    np.testing.assert_array_equal(np.sort(host), np.arange(n))
                    np.testing.assert_array_equal(np.sort(device), np.arange(n))

    def test_boundary_mode_keeps_the_same_anchor_cells(self):
        # The cut point inside a cell may differ between the two metrics, but
        # the cells themselves -- which anchor owns which point, and in which
        # order the anchors are visited -- must not.
        for n, d, m, _, kernel_name in CASES:
            with self.subTest(n=n, kernel=kernel_name):
                _, _, _, host, device, parent = self._orders(
                    n, d, m, kernel_name, "boundary"
                )
                np.testing.assert_array_equal(parent[host], parent[device])

    def test_boundary_mode_is_exact_in_one_dimension(self):
        # 1-D is the only regime the host selector is used in (Snelson and the
        # residual illustration), so there the two paths must agree exactly or
        # the figure stops illustrating the table.
        for n, d, m, _, kernel_name in CASES:
            if d != 1:
                continue
            with self.subTest(n=n, m=m):
                _, _, _, host, device, _ = self._orders(
                    n, d, m, kernel_name, "boundary"
                )
                np.testing.assert_array_equal(host, device)

    def test_boundary_mode_blocks_mostly_coincide_above_one_dimension(self):
        # Above 1-D the cut inside a cell can differ (see the module docstring).
        # Most points must still land in the same block; a collapse here means
        # the two selectors have genuinely diverged.
        for n, d, m, block, kernel_name in CASES:
            if d == 1:
                continue
            with self.subTest(n=n, d=d, kernel=kernel_name):
                _, _, _, host, device, _ = self._orders(
                    n, d, m, kernel_name, "boundary"
                )
                host_id = np.empty(n, dtype=int)
                device_id = np.empty(n, dtype=int)
                host_id[host] = np.arange(n) // block
                device_id[device] = np.arange(n) // block
                overlap = float(np.mean(host_id == device_id))
                self.assertGreater(
                    overlap, 0.75,
                    msg=f"host and device blocks agree on {overlap:.1%} of points",
                )


class PackedRepresentationRoundTrips(unittest.TestCase):
    def test_pack_then_unpack_returns_the_blocks(self):
        blocks = [np.array([3, 1, 2]), np.array([0, 4]), np.array([5])]
        packed = tm.pack_terms(blocks, 3, 6, as_jax=False)
        recovered = tm.unpack_terms(packed)
        self.assertEqual(len(recovered), len(blocks))
        for original, back in zip(blocks, recovered):
            np.testing.assert_array_equal(np.sort(original), np.sort(back))

    def test_padding_rows_are_dropped(self):
        packed = tm.pack_terms([np.array([0, 1])], 2, 2, capacity=6, as_jax=False)
        self.assertEqual(packed[0].shape, (6,))
        self.assertEqual(len(tm.unpack_terms(packed)), 1)

    def test_an_oversized_cover_is_rejected(self):
        with self.assertRaises(ValueError):
            tm.pack_terms([np.array([0])] * 9, 1, 2, capacity=4, as_jax=False)

    def test_a_gap_in_the_cover_is_rejected(self):
        with self.assertRaises(ValueError):
            tm.assert_partition([np.array([0, 1]), np.array([3])], 4)


if __name__ == "__main__":
    unittest.main()
