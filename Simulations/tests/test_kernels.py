import unittest

import jax
import jax.numpy as jnp
import numpy as np

from common import kernels as kn
from svgp import engine as sv


jax.config.update("jax_enable_x64", True)


class AmplitudeRBF(unittest.TestCase):
    def setUp(self):
        self.x = jnp.asarray([
            [-1.0, 0.2],
            [-0.2, -0.5],
            [0.4, 0.8],
            [1.3, -0.1],
        ])
        self.ell = jnp.asarray([0.7, 1.2])
        self.sf2 = jnp.asarray(1.4)

    def test_zero_projection_is_exactly_the_rbf(self):
        weights = jnp.zeros((self.x.shape[1],))
        actual = kn.kernel(
            self.x, self.x, self.ell, self.sf2,
            "amplitude-rbf", weights,
        )
        expected = kn.kernel(self.x, self.x, self.ell, self.sf2, "rbf")
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_gram_matrix_is_symmetric_positive_semidefinite(self):
        weights = jnp.asarray([0.8, -0.35])
        gram = np.asarray(kn.kernel(
            self.x, self.x, self.ell, self.sf2,
            "amplitude-rbf", weights,
        ))
        np.testing.assert_allclose(gram, gram.T, rtol=0.0, atol=1e-13)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(gram).min()), -1e-12)

    def test_fast_diagonal_matches_the_materialized_kernel(self):
        weights = jnp.asarray([0.8, -0.35])
        fast = kn.kernel_diag(
            self.x, self.ell, self.sf2, "amplitude-rbf", weights
        )
        dense = jnp.diag(kn.kernel(
            self.x, self.x, self.ell, self.sf2,
            "amplitude-rbf", weights,
        ))
        np.testing.assert_allclose(fast, dense, rtol=1e-13, atol=1e-13)

    def test_svgp_initializes_one_amplitude_weight_per_dimension(self):
        params = sv.init_params(self.x, 2, 0, kernel_name="amplitude-rbf")
        self.assertEqual(params["warp"].shape, (self.x.shape[1],))

    def test_sgpr_initializes_one_amplitude_weight_per_dimension(self):
        from sgpr import benchmark

        params = benchmark.init_params(
            self.x, 2, 0, kernel_name="amplitude-rbf"
        )
        self.assertEqual(params["warp"].shape, (self.x.shape[1],))

    def test_metis_residual_builder_uses_the_kernel_module(self):
        from sgpr import benchmark

        params = benchmark.init_params(self.x, 2, 0, kernel_name="rbf")
        residual = benchmark._residual_matrix(self.x, params, "rbf")
        self.assertEqual(residual.shape, (self.x.shape[0], self.x.shape[0]))
        np.testing.assert_allclose(residual, residual.T, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
