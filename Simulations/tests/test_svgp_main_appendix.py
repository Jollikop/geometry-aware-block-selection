import copy
import unittest
from unittest.mock import patch

import numpy as np
from Simulations.analysis import make_svgp_main_appendix as plot


class MainAppendixTests(unittest.TestCase):
    def test_objective_negated_once_and_sem_unchanged(self):
        rows = {("stationary", "rbf", "pol", 64, "SVGP", s):
                dict(rmse=1 + s, nll=-2 + s, objective=-0.3 + s * 0.1)
                for s in range(10)}
        original = copy.deepcopy(rows)
        with patch.multiple(plot, KERNELS=("rbf",), DATASETS=("pol",), METHODS=("SVGP",), BUDGETS=(64,)):
            result = {r["metric"]: r for r in plot.aggregate(rows)}
            objective = np.array([r["objective"] for r in rows.values()])
            self.assertAlmostEqual(result["negative_objective"]["mean"], -objective.mean())
            self.assertAlmostEqual(result["negative_objective"]["sem"], objective.std(ddof=1)/np.sqrt(10))
            self.assertAlmostEqual(result["nll"]["mean"], np.mean([r["nll"] for r in rows.values()]))
            self.assertEqual(rows, original)
            del rows[next(iter(rows))]
            with self.assertRaisesRegex(ValueError, "Missing or invalid"):
                plot.aggregate(rows)

    def test_arrows_and_solid_equal_lines(self):
        lookup = {( "rbf", d, method, m, metric): dict(mean=0.2, sem=0.01)
                  for d in plot.DATASETS[:4] for method in plot.METHODS
                  for m in plot.BUDGETS for metric in plot.METRICS}
        fig = plot.draw(lookup, "rbf", plot.DATASETS[:4])
        try:
            for i, ax in enumerate(fig.axes):
                texts = [t.get_text() for t in ax.texts]
                self.assertEqual(texts.count("lower is better"), int(i % 4 == 0))
                for container in ax.containers:
                    line = container.lines[0]
                    self.assertEqual(line.get_linestyle(), "-")
                    self.assertEqual(line.get_linewidth(), 1.4)
            self.assertIn("Negative last-epoch", fig.axes[8].get_ylabel())
        finally:
            plot.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
