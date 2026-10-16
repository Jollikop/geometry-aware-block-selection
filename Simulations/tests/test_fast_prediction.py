"""Numerical equivalence of prepared inference to the frozen predictor."""
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from svgp import engine as sv
from svgp.fast_prediction import PreparedSVGP

jax.config.update("jax_enable_x64", True)


class PreparedPredictionTests(unittest.TestCase):
    def test_all_kernels_predictors_ragged_padding_and_streaming(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(19, 3))
        xs = rng.normal(size=(9, 3))
        blocks = [np.array(b) for b in np.array_split(rng.permutation(19), 5)]
        for kernel in ("rbf", "matern52", "amplitude-rbf", "gibbs"):
            p = sv.init_params(x, 5, 1, kernel_name=kernel)
            p["q_mu"] = jnp.asarray(rng.normal(size=5))
            p["q_cov"] = jnp.eye(5) * 0.8
            if "warp" in p:
                p["warp"] = jnp.asarray(rng.normal(size=p["warp"].shape) * 0.2)
            for kind in ("sparse", "diagonal", "block"):
                reference = sv.predict_svgp(p, x, xs, predictor=kind,
                                            blocks=blocks, kernel_name=kernel)
                for cache in ((0, 1 << 20) if kind == "block" else (0,)):
                    with self.subTest(kernel=kernel, kind=kind, cache=cache):
                        prepared = PreparedSVGP(
                            p, x, predictor=kind, blocks=blocks, kernel_name=kernel,
                            test_chunk=4, train_chunk=7, block_batch=2,
                            factor_cache_bytes=cache)
                        result = prepared(xs)
                        for got, want in zip(result, reference):
                            np.testing.assert_allclose(got, want, atol=2e-9, rtol=1e-9)
                        for got, want in zip(prepared(xs[:3]), reference):
                            np.testing.assert_allclose(got, want[:3], atol=2e-9, rtol=1e-9)
                        self.assertEqual(prepared(xs[:0])[0].size, 0)
                        self.assertEqual(prepared.factors_cached, kind == "block" and cache > 0)

    def test_partition_validation(self):
        x = np.ones((5, 2))
        p = sv.init_params(x, 2, 1)
        for blocks in (None, [], [[0, 1], [1, 2, 3, 4]], [[0, 1], [2, 3, 5]]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                PreparedSVGP(p, x, predictor="block", blocks=blocks)

    def test_singleton_and_one_block_raw_variational_parameters(self):
        rng = np.random.default_rng(7)
        x, xs = rng.normal(size=(11, 2)), rng.normal(size=(7, 2))
        p = sv.init_params(x, 4, 3)
        for blocks in ([np.arange(11)], [np.array([i]) for i in range(11)]):
            prepared = PreparedSVGP(p, x, predictor="block", blocks=blocks,
                                    block_batch=4, test_chunk=3)
            ref = sv.predict_svgp(p, x, xs, predictor="block", blocks=blocks)
            for got, want in zip(prepared(xs), ref):
                np.testing.assert_allclose(got, want, rtol=1e-9, atol=2e-9)


if __name__ == "__main__":
    unittest.main()
