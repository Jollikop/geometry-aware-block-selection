"""The sweep reuses D's eigenbasis and matches the original dense formula."""
from pathlib import Path
import sys
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import dense_cutoff_sweep as cutoff
import real_predictor_campaign as real


class DenseCutoffTests(unittest.TestCase):
    def components(self):
        eig = np.array([1e-12, 1e-8, .2, .3, .4, .6, .8, 1.])
        x = np.diag(np.sqrt(eig))
        xs = np.random.default_rng(23).normal(size=(5,8))*.1
        return dict(x=jnp.asarray(x), xs=jnp.asarray(xs),
            a_train=jnp.zeros((1,8)), a_test=jnp.zeros((1,5)),
            d_train=jnp.asarray(eig), d_test=jnp.asarray((xs*xs).sum(axis=1)),
            qvar=jnp.full(5,.02), mean=jnp.zeros(5), s2=jnp.asarray(.1),
            kernel_fn=lambda x,y: x@y.T)

    def test_all_scales_match_original_T_and_BT(self):
        components = self.components()
        blocks = [np.arange(4), np.arange(4,8)]
        for method in ("T-SVGP", "RBT-SVGP", "METIS-SGPR"):
            rows, diagnostics = cutoff.sweep(components, blocks, method, test_chunk=2)
            self.assertEqual(diagnostics["dense_eigendecompositions"], 1)
            for row in rows:
                _, variance, _ = real.dense_variational_prediction(
                    components, blocks, method, test_chunk=2, eig_cutoff_scale=row["scale"])
                np.testing.assert_allclose(row["variance"], variance, atol=1e-12, rtol=1e-12)
                self.assertTrue(row["valid"])
            counts = [r["discarded_eigenvalues"] for r in rows]
            self.assertEqual(counts, sorted(counts))
            self.assertGreater(counts[-1], counts[0])
            self.assertEqual(rows[1]["max_additional_weighted_cross_relative_to_dtest"], 0.)
            self.assertGreater(rows[-1]["max_additional_weighted_cross_relative_to_dtest"], 0.)


if __name__ == "__main__":
    unittest.main()
