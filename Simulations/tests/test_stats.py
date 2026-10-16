#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error bars mean the same thing in every figure.

Three plotting scripts used to carry their own ``uncertainty`` helper, with
different defaults and only one of them tested. These tests pin the single
definition, including the reduction over the seed axis of a curve stack.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.stats import t as student_t

from common import stats as st


class Uncertainty(unittest.TestCase):
    def setUp(self):
        self.values = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

    def test_std_is_the_sample_standard_deviation(self):
        # ddof=1, not the population value: with 10 seeds the difference is
        # about 5%, which is visible on an error bar.
        self.assertAlmostEqual(
            st.uncertainty(self.values, "std"), float(np.std(self.values, ddof=1))
        )
        self.assertNotAlmostEqual(
            st.uncertainty(self.values, "std"), float(np.std(self.values, ddof=0))
        )

    def test_sem_divides_by_sqrt_n(self):
        expected = float(np.std(self.values, ddof=1)) / math.sqrt(self.values.size)
        self.assertAlmostEqual(st.uncertainty(self.values, "sem"), expected)

    def test_ci95_is_the_two_sided_student_interval(self):
        n = self.values.size
        expected = (
            float(student_t.ppf(0.975, n - 1))
            * float(np.std(self.values, ddof=1)) / math.sqrt(n)
        )
        self.assertAlmostEqual(st.uncertainty(self.values, "ci95"), expected)

    def test_ci95_is_wider_than_the_sem(self):
        self.assertGreater(
            st.uncertainty(self.values, "ci95"), st.uncertainty(self.values, "sem")
        )

    def test_a_single_observation_carries_no_dispersion(self):
        self.assertEqual(st.uncertainty([3.0], "sem"), 0.0)
        self.assertEqual(st.uncertainty([3.0], "std"), 0.0)
        self.assertTrue(math.isnan(st.uncertainty([3.0], "ci95")))

    def test_an_empty_sample_is_nan(self):
        self.assertTrue(math.isnan(st.uncertainty([], "sem")))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            st.uncertainty(self.values, "stderr")

    def test_axis_reduction_matches_the_flat_case_column_by_column(self):
        stack = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
        per_column = st.uncertainty(stack, "sem", axis=0)
        for column in range(stack.shape[1]):
            self.assertAlmostEqual(
                float(per_column[column]),
                st.uncertainty(stack[:, column], "sem"),
            )

    def test_axis_reduction_ignores_nans(self):
        stack = np.array([[1.0, np.nan], [2.0, 20.0], [4.0, 40.0]])
        result = st.uncertainty(stack, "std", axis=0)
        self.assertTrue(np.all(np.isfinite(result)))


class MeanStd(unittest.TestCase):
    def test_single_observation_reports_no_spread(self):
        self.assertEqual(st.mean_std([2.0]), ("2.0000", ""))

    def test_uses_the_sample_standard_deviation(self):
        mean, std = st.mean_std([1.0, 3.0])
        self.assertEqual(mean, "2.0000")
        self.assertEqual(std, f"{np.std([1.0, 3.0], ddof=1):.4f}")


class GroupSeeds(unittest.TestCase):
    def test_grouping_preserves_the_mean_exactly(self):
        values = np.arange(12, dtype=float)
        grouped = st.group_seeds(values, 3)
        self.assertEqual(grouped.size, 4)
        self.assertAlmostEqual(float(grouped.mean()), float(values.mean()))

    def test_a_trailing_partial_group_is_dropped(self):
        # Unequal groups would weight seeds unequally and break the identity
        # above, so the remainder is discarded rather than averaged short.
        self.assertEqual(st.group_seeds(np.arange(10, dtype=float), 3).size, 3)

    def test_grouping_is_a_no_op_below_two_full_groups(self):
        values = np.arange(5, dtype=float)
        np.testing.assert_array_equal(st.group_seeds(values, 4), values)
        np.testing.assert_array_equal(st.group_seeds(values, 0), values)


if __name__ == "__main__":
    unittest.main()
