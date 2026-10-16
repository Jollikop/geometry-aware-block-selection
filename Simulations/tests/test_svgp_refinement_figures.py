import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Simulations.analysis import make_svgp_refinement_figures as plot
from Simulations.tests.test_svgp_complete_figures import CompleteFiguresTests


class RefinementTests(unittest.TestCase):
    def test_missing_sparse_baseline_is_not_replaced_by_full_predictor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "training"
            predictions = Path(tmp).resolve() / "predictions"
            rows = CompleteFiguresTests().fixture(root, predictions)
            self.assertEqual(plot.collect_pairs(rows, root), [])
            for seed in range(5):
                before = rows[("stationary", "rbf", "pol", 64, "T-SVGP", seed)]
                before["config_id"] = "test"
                record = {"config": before["config"], "config_id": "test",
                          "rmse_sparse": 2 + seed * 0.01, "nll_sparse": 3 + seed * 0.01}
                (root / before["run"] / "metrics.json").write_text(json.dumps(record))
            pairs = plot.collect_pairs(rows, root)
            self.assertEqual(len(pairs), 10)
            self.assertAlmostEqual(pairs[0]["delta"], -1.1)
            with patch.multiple(plot, KERNELS=("rbf",), DATASETS=("pol",), METHODS=("T-SVGP",), BUDGETS=(64,)):
                stats = plot.summarize(pairs)
                row = stats["rbf", "pol", "T-SVGP", 64, "rmse"]
                self.assertEqual(row["n"], 5)
                self.assertAlmostEqual(row["delta_sem"], 0)


if __name__ == "__main__":
    unittest.main()
