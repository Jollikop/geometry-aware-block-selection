"""Prediction controls preserve the donor and the historical default geometry."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import hard_block_prediction_campaign as hard
import partition_controls as controls
from svgp.engine import _blocks


class PartitionControlTests(unittest.TestCase):
    def test_random_matches_training_implementation_with_recorded_seed(self):
        components = {"x": jnp.arange(46).reshape(23, 2)}
        row = dict(seed=3, kernel="rbf", method="SBT-SVGP", family="svgp")
        blocks, source, seed = hard.prediction_blocks(
            components, {}, row, {"block_size_for_T": 5}, None, Path("unused"), "random")
        idx, mask, _ = _blocks(components["x"], None, None, 5, "BT", 10010)
        expected = [np.asarray(i[m > .5]) for i, m in zip(np.asarray(idx), np.asarray(mask))]
        self.assertEqual(seed, 10010)
        self.assertEqual(source, "prediction-time-uniform-random")
        for actual, ref in zip(blocks, expected):
            np.testing.assert_array_equal(actual, ref)
        self.assertEqual([len(b) for b in blocks], [5, 5, 5, 5, 3])

    def test_default_BT_keeps_frozen_partition(self):
        blocks = [np.array([0, 2]), np.array([1, 3])]
        actual, source, seed = hard.prediction_blocks(
            {"x": np.zeros((4, 1))}, {}, dict(method="RBT-SGPR", family="sgpr"),
            {}, blocks, Path("unused"))
        self.assertEqual(source, "frozen-fitted-blocks")
        self.assertIsNone(seed)
        for a, b in zip(actual, blocks):
            np.testing.assert_array_equal(a, b)

    def test_anchor_override_uses_identical_T_selector(self):
        blocks = [np.array([0, 1]), np.array([2, 3])]
        params = {"z": jnp.array([[1.]])}
        original = copy.deepcopy(params)
        with patch.object(hard.anchor, "anchor_blocks", return_value=blocks) as function:
            actual, _, _ = hard.prediction_blocks({"x": np.zeros((4, 1))}, params,
                dict(method="RBT-SVGP", family="svgp", kernel="rbf"),
                {"block_size_for_T": 500}, None, Path("unused"), "anchor")
            self.assertEqual(function.call_args.args[2:], (500, "rbf"))
        np.testing.assert_array_equal(params["z"], original["z"])
        np.testing.assert_array_equal(actual[0], blocks[0])

    def test_control_scopes_have_complete_grids_and_leave_base_untouched(self):
        rows = [dict(family="svgp", kernel=k, dataset=d, M=m, method=method, seed=s,
                     relative=f"svgp/runs/{k}/{d}/M{m}/{method}/seed{s}",
                     output_relative=f"svgp/{k}/{d}/M{m}/{method}/seed{s}")
                for k in hard.anchor.KERNEL_STAGE for d in hard.anchor.DATASETS
                for m in hard.anchor.M_VALUES for method in ("T-SVGP", "RBT-SVGP", "SBT-SVGP")
                for s in range(10)]
        base = {"runs": rows}
        for experiment in ("E1", "E2"):
            self.assertEqual(len(controls.select_runs(base, experiment, "fig4")), 320)
        self.assertEqual(len(controls.select_runs(base, "E3", "all")), 5120)
        self.assertEqual(len(rows), 3840)


if __name__ == "__main__":
    unittest.main()
