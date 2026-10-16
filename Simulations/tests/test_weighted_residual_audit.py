"""Finite-dimensional checks for the prediction-only gated residual proxy."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from weighted_residual_audit import (cross_block_duplicate_groups,
                                     inverse_distance_weights)


class WeightedResidualAuditTests(unittest.TestCase):
    def test_weights_are_partition_of_unity_with_exact_site_limits(self):
        distances = np.array([[0.0, 2.0, 5.0], [3.0, 1.0, 4.0],
                              [1.0, 0.0, 2.0]])
        weights = inverse_distance_weights(distances, power=2.0)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        np.testing.assert_array_equal(weights[0], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(weights[2], [0.0, 1.0, 0.0])
        self.assertTrue((weights >= 0).all())
        self.assertTrue((weights[1] > 0).all())

    def test_joint_kernel_is_psd_and_train_matrix_is_block_diagonal(self):
        rng = np.random.default_rng(711)
        features = rng.normal(size=(8, 4))
        residual = features @ features.T
        weights = np.zeros((8, 3))
        weights[:2, 0] = 1.0
        weights[2:5, 1] = 1.0
        weights[5:6, 2] = 1.0
        weights[6:] = inverse_distance_weights(
            rng.uniform(0.2, 2.0, size=(2, 3)), power=2.0)
        gate = np.sqrt(weights) @ np.sqrt(weights).T
        joint = residual * gate
        self.assertGreaterEqual(np.linalg.eigvalsh(joint).min(), -1e-10)
        np.testing.assert_array_equal(joint[:2, 2:5], 0.0)
        np.testing.assert_array_equal(joint[2:5, 5:6], 0.0)
        np.testing.assert_allclose(np.diag(joint), np.diag(residual))

    def test_weighted_T_and_BT_corrections_do_not_exceed_test_residual(self):
        rng = np.random.default_rng(412)
        features = rng.normal(size=(7, 4))
        d = features @ features.T
        noise = 0.3
        test_variance = d[-1, -1]
        blocks = (np.arange(0, 3), np.arange(3, 6))
        contributions_t = []
        contributions_bt = []
        for block in blocks:
            local = d[np.ix_(block, block)]
            cross = d[block, -1]
            eig, vec = np.linalg.eigh(local)
            keep = eig > 1e-12
            inv_sqrt = (vec[:, keep] / np.sqrt(eig[keep])) @ vec[:, keep].T
            gain = np.diag(local) / (np.diag(local) + noise)
            white = inv_sqrt @ cross
            contributions_t.append(np.sum(gain * white * white))
            contributions_bt.append(cross @ np.linalg.solve(
                local + noise * np.eye(len(block)), cross))
        weights = inverse_distance_weights(np.array([[1.0, 2.0]]), 2.0)[0]
        self.assertLessEqual(weights @ contributions_t, test_variance + 1e-10)
        self.assertLessEqual(weights @ contributions_bt, test_variance + 1e-10)

    def test_unweighted_all_block_sum_can_be_negative(self):
        residual = np.ones((3, 3))
        noise = 0.1
        correction = sum(residual[i, 2] ** 2 / (residual[i, i] + noise)
                         for i in (0, 1))
        self.assertLess(residual[2, 2] - correction, 0.0)

    def test_duplicate_input_split_across_blocks_is_not_an_input_kernel(self):
        points = np.array([[0.0], [1.0], [0.0], [2.0]])
        self.assertEqual(cross_block_duplicate_groups(
            points, [np.array([0, 1]), np.array([2, 3])]), 1)
        self.assertEqual(cross_block_duplicate_groups(
            points, [np.array([0, 2]), np.array([1, 3])]), 0)


if __name__ == "__main__":
    unittest.main()
