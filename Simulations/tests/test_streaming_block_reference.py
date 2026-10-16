"""Streaming reference must preserve all fitted blocks and final clipping."""
import sys
from pathlib import Path
import unittest

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from svgp import engine
from common import predictors
from evaluate_svgp_frozen import streaming_block_reference

jax.config.update('jax_enable_x64', True)


class StreamingReferenceTests(unittest.TestCase):
    def test_ragged_blocks_match_full_historical_reference(self):
        rng = np.random.default_rng(172)
        x, xs = rng.normal(size=(37, 4)), rng.normal(size=(8, 4))
        blocks = list(np.array_split(rng.permutation(len(x)), 6))
        for kernel in ('rbf', 'matern52', 'amplitude-rbf', 'gibbs'):
            with self.subTest(kernel=kernel):
                params = engine.init_params(x, 8, 3, kernel_name=kernel)
                params['q_mu'] = rng.normal(size=8)
                if 'warp' in params:
                    params['warp'] = rng.normal(size=params['warp'].shape) * .3
                expected = engine.predict_svgp(params, x, xs, predictor='block',
                    blocks=blocks, kernel_name=kernel)
                actual = streaming_block_reference(engine, predictors, params,
                    x, xs, blocks, kernel, 1e-8)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, atol=2e-8, rtol=2e-7)


if __name__ == '__main__': unittest.main()
