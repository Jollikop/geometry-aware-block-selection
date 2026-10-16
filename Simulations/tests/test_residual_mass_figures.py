import unittest
import numpy as np
from Simulations.analysis.make_residual_mass_figures import normalized_mass, partition_stats, area_mean, mass_palette
from Simulations.analysis.paper_labels import paper_label


class ResidualMassTests(unittest.TestCase):
    def test_normalized_display_is_entrywise_square_and_rho_is_sum(self):
        d = np.array([[3., -2., 1.], [-2., 4., .5], [1., .5, 2.]])
        w, moff = normalized_mass(d)
        squared = d ** 2
        np.fill_diagonal(squared, 0.)
        np.testing.assert_allclose(w * moff, squared)
        stats, _ = partition_stats(w, np.arange(3), 2)
        labels = np.array([0, 0, 1])
        self.assertAlmostEqual(stats["rho"], w[labels[:, None] != labels[None, :]].sum())

    def test_mass_leakage_and_random(self):
        d = np.array([[3., 1., 0., 0.], [1., 3., 0., 0.],
                      [0., 0., 3., 2.], [0., 0., 2., 3.]])
        original = d.copy()
        w, total = normalized_mass(d)
        self.assertEqual(total, 10.)
        self.assertAlmostEqual(w.sum(), 1.)
        np.testing.assert_array_equal(w.diagonal(), 0.)
        np.testing.assert_array_equal(d, original)
        selected, _ = partition_stats(w, np.arange(4), 2)
        random_mixed, _ = partition_stats(w, np.array([0, 2, 1, 3]), 2)
        self.assertAlmostEqual(selected["rho"], 0.)
        self.assertAlmostEqual(random_mixed["rho"], 1.)
        self.assertAlmostEqual(selected["retained_vs_random"], 3.)
        self.assertAlmostEqual(selected["random_expected_rho"], 2/3)
        w2, _ = normalized_mass(d * 100)
        np.testing.assert_allclose(w, w2)

    def test_degenerate_and_bad_order(self):
        with self.assertRaises(ValueError):
            normalized_mass(np.eye(4))
        with self.assertRaises(ValueError):
            partition_stats(np.ones((4, 4))/16, np.array([0, 0, 2, 3]), 2)

    def test_area_means_keep_edges(self):
        a = np.arange(25).reshape(5, 5)
        reduced, step = area_mean(a, target=3)
        self.assertEqual(step, 2)
        self.assertEqual(reduced[-1, -1], 24)
        self.assertEqual(reduced[0, 0], a[:2, :2].mean())

    def test_display_only_names(self):
        self.assertEqual(paper_label("RBT-SVGP"), "(R)BT-SVGP (2025)")
        self.assertEqual(paper_label("SBT-SVGP"), "(S)BT-SVGP (2026)")
        self.assertEqual(paper_label("SVGP"), "SVGP (2013)")
        self.assertEqual(paper_label("T-SVGP"), "T-SVGP (2025)")

    def test_blue_zero_is_white(self):
        np.testing.assert_array_equal(mass_palette("blue")(0.), (1, 1, 1, 1))
        with self.assertRaises(ValueError):
            mass_palette("unrecognized")
