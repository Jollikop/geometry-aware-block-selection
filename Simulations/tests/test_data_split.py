#!/usr/bin/env python3
"""Unit tests for host-side dataset preprocessing."""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


SIM_ROOT = Path(__file__).resolve().parents[1]

from common import data_split as jd  # noqa: E402


class SplitIndicesTest(unittest.TestCase):
    def test_numpy_split_is_deterministic_disjoint_and_exact(self):
        first = jd.split_indices(20, 7, 12, 5)
        second = jd.split_indices(20, 7, 12, 5)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(len(first[0]), 12)
        self.assertEqual(len(first[1]), 5)
        self.assertFalse(set(first[0]) & set(first[1]))

    def test_invalid_size_is_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "requested"):
            jd.split_indices(10, 0, 8, 3)
        with self.assertRaisesRegex(ValueError, "positive"):
            jd.split_indices(10, 0, 10, 0)


class StandardizerTest(unittest.TestCase):
    def test_training_statistics_and_angular_passthrough(self):
        x = np.asarray([[0.0, 1.0], [np.pi, 3.0], [2 * np.pi, 5.0]])
        y = np.asarray([1.0, 2.0, 3.0])
        stats = jd.fit_standardizer(x, y, angle_cols=[0])
        x_scaled, y_scaled = jd.apply_standardizer(x, y, stats)
        np.testing.assert_allclose(x_scaled[:, 0], x[:, 0])
        self.assertAlmostEqual(float(x_scaled[:, 1].mean()), 0.0)
        self.assertAlmostEqual(float(x_scaled[:, 1].std()), 1.0)
        self.assertAlmostEqual(float(y_scaled.mean()), 0.0)
        self.assertAlmostEqual(float(y_scaled.std()), 1.0)

    def test_constant_input_is_finite_but_constant_target_is_rejected(self):
        x = np.ones((4, 2))
        stats = jd.fit_standardizer(x, np.arange(4.0))
        x_scaled, _ = jd.apply_standardizer(x, np.arange(4.0), stats)
        self.assertTrue(np.isfinite(x_scaled).all())
        with self.assertRaisesRegex(ValueError, "zero variance"):
            jd.fit_standardizer(x, np.ones(4))

    def test_raw_shape_and_finite_checks(self):
        with self.assertRaisesRegex(ValueError, "row counts"):
            jd.validate_raw_arrays(np.ones((3, 2)), np.ones(2))
        with self.assertRaisesRegex(ValueError, "finite"):
            jd.validate_raw_arrays(np.asarray([[1.0], [np.nan]]), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
