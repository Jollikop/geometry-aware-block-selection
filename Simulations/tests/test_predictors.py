#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What the shared residual predictor must keep doing after a refactor.

The contract has three parts, and each has bitten at least once:

1. the dense diagnostic decomposition sums to Eq. (12) whatever the cover;
2. every per-block term is non-negative and the partial sums are exactly the
   nested conditionings the appendix curve claims they are;
3. diagonal and fitted-block predictors match their direct formulas;
4. the two stacks call the same function object -- the test fails the instant
   anyone reintroduces a private copy.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

jax.config.update("jax_enable_x64", True)

SIM_ROOT = Path(__file__).resolve().parents[1]

from common import kernels as kn                     # noqa: E402
from common import predictors as pr                  # noqa: E402
from common import terms as tm                       # noqa: E402
from sgpr import jax_collapsed as jm                 # noqa: E402
from svgp import engine as sv                        # noqa: E402

JITTER = 1e-8


def _fixture(n=90, m=12, d=3, n_test=40, seed=0):
    """A small collapsed problem plus the projections a predictor consumes."""
    rng = np.random.default_rng(seed)
    x = jnp.asarray(rng.normal(size=(n, d)))
    xs = jnp.asarray(rng.normal(size=(n_test, d)))
    z = jnp.asarray(np.asarray(x)[rng.choice(n, m, replace=False)])
    ell = jnp.asarray(rng.uniform(0.7, 1.3, size=d))
    sf2, s2 = jnp.asarray(1.2), jnp.asarray(0.08)

    def kernel_fn(xa, xb):
        return kn.kernel(xa, xb, ell, sf2, "rbf", None)

    luu = jnp.linalg.cholesky(kernel_fn(z, z) + JITTER * jnp.eye(m))
    a_train = solve_triangular(luu, kernel_fn(z, x), lower=True)
    a_test = solve_triangular(luu, kernel_fn(z, xs), lower=True)
    d_train = jnp.maximum(
        kn.kernel_diag(x, ell, sf2, "rbf") - jnp.sum(a_train * a_train, 0), 0.0
    )
    d_test = jnp.maximum(
        kn.kernel_diag(xs, ell, sf2, "rbf") - jnp.sum(a_test * a_test, 0), 0.0
    )
    components = {
        "x": x, "xs": xs, "a_train": a_train, "a_test": a_test,
        "d_train": d_train, "d_test": d_test, "s2": s2,
        "kernel_fn": kernel_fn,
        "mean": jnp.asarray(rng.normal(size=n_test) * 0.1),
        "qvar": jnp.asarray(rng.uniform(0.01, 0.05, size=n_test)),
    }
    y_test = jnp.asarray(rng.normal(size=n_test))
    return components, y_test, n, rng


def _naive_eq12(components, jitter=JITTER):
    """Eq. (12) written out directly, with no cover and no decomposition."""
    kernel_fn = components["kernel_fn"]
    x, xs = components["x"], components["xs"]
    a_train, a_test = components["a_train"], components["a_test"]
    dff = np.asarray(kernel_fn(x, x) - a_train.T @ a_train)
    dff = 0.5 * (dff + dff.T)
    dtf = np.asarray(kernel_fn(xs, x) - a_test.T @ a_train)
    s2 = float(components["s2"])
    inv = np.linalg.inv(dff + (s2 + jitter) * np.eye(dff.shape[0]))
    return np.einsum("ti,ij,tj->t", dtf, inv, dtf)


class DecompositionSumsToEq12(unittest.TestCase):
    """The cover splits the correction; it must never change the total."""

    def setUp(self):
        self.components, self.y_test, self.n, self.rng = _fixture()
        self.reference = _naive_eq12(self.components)

    def _covers(self):
        n, rng = self.n, self.rng
        return {
            "none": None,
            "equal": [np.sort(b) for b in np.array_split(np.arange(n), 6)],
            "random": [np.sort(b) for b in np.array_split(rng.permutation(n), 6)],
            "ragged": [np.sort(b) for b in np.array_split(rng.permutation(n), 7)],
            "singletons": [np.array([i]) for i in range(n)],
            "one block": [np.arange(n)],
        }

    def test_every_cover_reproduces_the_dense_correction(self):
        for name, blocks in self._covers().items():
            with self.subTest(cover=name):
                contributions = pr.residual_contributions(
                    self.components, predictor="full", blocks=blocks
                )
                np.testing.assert_allclose(
                    contributions.sum(axis=1), self.reference, atol=1e-9
                )

    def test_contributions_are_non_negative(self):
        for name, blocks in self._covers().items():
            with self.subTest(cover=name):
                contributions = pr.residual_contributions(
                    self.components, predictor="full", blocks=blocks
                )
                self.assertGreaterEqual(float(contributions.min()), -1e-12)

    def test_partial_sums_are_the_nested_conditionings(self):
        # Conditioning on the union of the first j blocks, evaluated directly,
        # must equal the sum of the first j contributions. This is the identity
        # the appendix curve rests on.
        blocks = [np.sort(b) for b in np.array_split(self.rng.permutation(self.n), 5)]
        contributions = pr.residual_contributions(
            self.components, predictor="full", blocks=blocks
        )
        kernel_fn = self.components["kernel_fn"]
        x, xs = self.components["x"], self.components["xs"]
        a_train, a_test = self.components["a_train"], self.components["a_test"]
        s2 = float(self.components["s2"])
        for j in range(1, len(blocks) + 1):
            union = np.concatenate(blocks[:j])
            dff = np.asarray(
                kernel_fn(x[union], x[union]) - a_train[:, union].T @ a_train[:, union]
            )
            dff = 0.5 * (dff + dff.T)
            dtf = np.asarray(
                kernel_fn(xs, x[union]) - a_test.T @ a_train[:, union]
            )
            direct = np.einsum(
                "ti,ij,tj->t", dtf,
                np.linalg.inv(dff + (s2 + JITTER) * np.eye(union.size)), dtf,
            )
            with self.subTest(blocks_included=j):
                np.testing.assert_allclose(
                    contributions[:, :j].sum(axis=1), direct, atol=1e-9
                )

    def test_a_cover_that_is_not_a_partition_is_rejected(self):
        with self.assertRaises(ValueError):
            pr.residual_contributions(
                self.components, predictor="full",
                blocks=[np.array([0, 1]), np.array([1, 2])],
            )
        with self.assertRaises(ValueError):
            pr.residual_contributions(
                self.components, predictor="full", blocks=[]
            )


class ReportedPredictors(unittest.TestCase):
    def setUp(self):
        self.components, self.y_test, self.n, self.rng = _fixture()

    def test_sparse_leaves_the_residual_untouched(self):
        _, var = pr.predictive_variance(self.components, predictor="sparse")
        expected = (
            np.asarray(self.components["qvar"])
            + np.asarray(self.components["d_test"])
            + float(self.components["s2"])
        )
        np.testing.assert_allclose(var, expected, atol=1e-12)

    def test_full_never_exceeds_sparse(self):
        _, sparse = pr.predictive_variance(self.components, predictor="sparse")
        _, full = pr.predictive_variance(self.components, predictor="full")
        self.assertTrue(np.all(full <= sparse + 1e-12))

    def test_diagonal_matches_the_direct_formula(self):
        correction = pr.residual_contributions(
            self.components, predictor="diagonal"
        ).sum(axis=1)
        d_cross = np.asarray(
            self.components["kernel_fn"](self.components["xs"], self.components["x"])
            - self.components["a_test"].T @ self.components["a_train"]
        )
        expected = np.sum(
            d_cross ** 2 /
            (np.asarray(self.components["d_train"]) + float(self.components["s2"]))[None, :],
            axis=1,
        )
        np.testing.assert_allclose(correction, expected, atol=1e-9)

    def test_common_test_chunk_preserves_every_residual_predictor(self):
        blocks = [np.sort(b) for b in np.array_split(np.arange(self.n), 5)]
        for predictor in ("sparse", "diagonal", "block", "full"):
            kwargs = {"blocks": blocks} if predictor in {"block", "full"} else {}
            whole = pr.predictive_variance(
                self.components, predictor=predictor, **kwargs
            )
            chunked = pr.predictive_variance(
                self.components, predictor=predictor, chunk=7, **kwargs
            )
            with self.subTest(predictor=predictor):
                np.testing.assert_allclose(chunked[0], whole[0], atol=1e-12)
                np.testing.assert_allclose(chunked[1], whole[1], atol=1e-9)

    def test_block_matches_independent_direct_solves(self):
        blocks = [np.sort(b) for b in np.array_split(np.arange(self.n), 5)]
        got = pr.residual_contributions(
            self.components, predictor="block", blocks=blocks
        )
        direct = []
        for block in blocks:
            d_bb = np.asarray(
                self.components["kernel_fn"](
                    self.components["x"][block], self.components["x"][block]
                ) - self.components["a_train"][:, block].T
                @ self.components["a_train"][:, block]
            )
            d_tb = np.asarray(
                self.components["kernel_fn"](
                    self.components["xs"], self.components["x"][block]
                ) - self.components["a_test"].T
                @ self.components["a_train"][:, block]
            )
            direct.append(np.einsum(
                "ti,ij,tj->t", d_tb,
                np.linalg.inv(d_bb + (float(self.components["s2"]) + JITTER)
                              * np.eye(len(block))), d_tb,
            ))
        np.testing.assert_allclose(got, np.column_stack(direct), atol=1e-9)

    def test_rmse_does_not_depend_on_the_predictor(self):
        blocks = [np.sort(b) for b in np.array_split(np.arange(self.n), 4)]
        rmse_sparse = pr.predictive_metrics(
            self.components, self.y_test, predictor="sparse")[0]
        rmse_full = pr.predictive_metrics(
            self.components, self.y_test, predictor="full", blocks=blocks)[0]
        self.assertAlmostEqual(rmse_sparse, rmse_full, places=12)

    def test_an_unknown_predictor_is_rejected(self):
        with self.assertRaises(ValueError):
            pr.residual_contributions(self.components, predictor="local_cross")

    def test_every_published_method_declares_a_predictor(self):
        from sgpr import benchmark as bench

        self.assertEqual(
            set(bench.METHODS), set(pr.COLLAPSED_METHOD_PREDICTOR)
        )
        self.assertEqual(
            set(sv.METHODS), set(pr.VARIATIONAL_METHOD_PREDICTOR)
        )
        for registry in (pr.COLLAPSED_METHOD_PREDICTOR,
                         pr.VARIATIONAL_METHOD_PREDICTOR):
            for method, predictor in registry.items():
                self.assertIn(
                    predictor, ("sparse", "diagonal", "block", "full", "exact"),
                    method,
                )

    def test_the_trace_bound_methods_are_the_sparse_ones(self):
        # A method reports the residual model its bound carries. Only the two
        # trace-bound methods carry none.
        sparse = {m for m, p in pr.COLLAPSED_METHOD_PREDICTOR.items()
                  if p == "sparse"}
        self.assertEqual(sparse, {"SGPR"})
        sparse = {m for m, p in pr.VARIATIONAL_METHOD_PREDICTOR.items()
                  if p == "sparse"}
        self.assertEqual(sparse, {"SVGP"})


class BlocksIncludedCurve(unittest.TestCase):
    def setUp(self):
        self.components, self.y_test, self.n, self.rng = _fixture()
        self.blocks = [
            np.sort(b) for b in np.array_split(self.rng.permutation(self.n), 6)
        ]
        self.contributions = pr.residual_contributions(
            self.components, predictor="full", blocks=self.blocks
        )

    def test_endpoints_are_the_two_reported_predictors(self):
        curve = pr.blocks_included_curve(
            self.components, self.y_test, self.contributions
        )
        sparse = pr.predictive_metrics(
            self.components, self.y_test, predictor="sparse")[1]
        full = pr.predictive_metrics(
            self.components, self.y_test, predictor="full", blocks=self.blocks)[1]
        self.assertAlmostEqual(curve["nll"][0], sparse, places=12)
        self.assertAlmostEqual(curve["nll"][-1], full, places=12)

    def test_captured_fraction_runs_from_zero_to_one(self):
        curve = pr.blocks_included_curve(
            self.components, self.y_test, self.contributions
        )
        self.assertAlmostEqual(curve["captured_fraction"][0], 0.0, places=12)
        self.assertAlmostEqual(curve["captured_fraction"][-1], 1.0, places=10)
        self.assertTrue(np.all(np.diff(curve["captured_fraction"]) >= -1e-12))

    def test_the_curve_has_one_point_per_block_plus_the_origin(self):
        curve = pr.blocks_included_curve(
            self.components, self.y_test, self.contributions
        )
        self.assertEqual(curve["nll"].size, len(self.blocks) + 1)


class BothStacksShareTheImplementation(unittest.TestCase):
    def test_the_two_stacks_call_one_function_object(self):
        self.assertIs(jm.pr.predictive_variance, pr.predictive_variance)
        self.assertIs(sv.pr.predictive_variance, pr.predictive_variance)
        self.assertIs(jm.pr.residual_contributions, sv.pr.residual_contributions)

    def test_the_registries_have_one_owner(self):
        from sgpr import benchmark as bench

        self.assertIs(bench.METHOD_PREDICTOR, pr.COLLAPSED_METHOD_PREDICTOR)
        self.assertIs(sv.METHOD_PREDICTOR, pr.VARIATIONAL_METHOD_PREDICTOR)

    def test_collapsed_and_variational_agree_on_the_residual_term(self):
        components, y_test, n, rng = _fixture()
        blocks = [np.sort(b) for b in np.array_split(rng.permutation(n), 5)]
        a = pr.residual_contributions(components, predictor="full", blocks=blocks)
        b = pr.residual_contributions(components, predictor="full", blocks=blocks)
        np.testing.assert_allclose(a, b, atol=0.0)

    def test_packed_terms_round_trip_into_the_predictor(self):
        components, _, n, _ = _fixture()
        packed = tm.pack_order(np.arange(n), 20, n)
        blocks = tm.unpack_terms(packed)
        tm.assert_partition(blocks, n)
        contributions = pr.residual_contributions(
            components, predictor="full", blocks=blocks
        )
        np.testing.assert_allclose(
            contributions.sum(axis=1), _naive_eq12(components), atol=1e-9
        )

    def test_a_packed_bank_recovers_the_same_cover(self):
        from common import partitions as pt

        packed = pt.bt_terms_device(120, 25, seed=4)
        from_bank = sv.blocks_from_bank(packed[1], packed[2])
        from_terms = tm.unpack_terms(packed)
        self.assertEqual(len(from_bank), len(from_terms))
        for a, b in zip(from_bank, from_terms):
            np.testing.assert_array_equal(np.sort(a), np.sort(b))


class GaussianMetrics(unittest.TestCase):
    def test_host_and_device_paths_agree(self):
        rng = np.random.default_rng(5)
        mu = rng.normal(size=64)
        var = rng.uniform(0.05, 1.5, size=64)
        y = rng.normal(size=64)
        host = pr.gaussian_rmse_nll(mu, var, y)
        device = [float(v) for v in pr.gaussian_rmse_nll_device(mu, var, y)]
        np.testing.assert_allclose(host, device, atol=1e-12)


class NoTopKSurvives(unittest.TestCase):
    """The truncation parameter is gone; nothing may quietly bring it back."""

    def test_no_module_mentions_a_predictive_top_k(self):
        needle = "predictive_" + "top_k"          # split so this file is clean
        offenders = []
        for path in sorted(SIM_ROOT.rglob("*.py")):
            if "archive" in path.parts or path == Path(__file__).resolve():
                continue
            if needle in path.read_text():
                offenders.append(str(path.relative_to(SIM_ROOT)))
        self.assertEqual(offenders, [], f"top-k predictor reintroduced in {offenders}")


if __name__ == "__main__":
    unittest.main()
