import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from Simulations.analysis import make_svgp_complete_figures as figures


class CompleteFiguresTests(unittest.TestCase):
    def fixture(self, root, predictions):
        rows = {}
        for seed in range(10):
            relative = f"svgp/stationary/runs/rbf/pol/M64/T-SVGP/seed{seed}"
            checkpoint = root / relative / "params.npz"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            config = dict(train_indices_sha256="train", test_indices_sha256="test",
                          preprocessing_protocol="same", data_signature="data")
            before = dict(config=config, run=relative, rmse=1 + seed * 0.01,
                          nll=2 + seed * 0.01, prediction_source=str(checkpoint.with_name("metrics.json")))
            after_relative = relative.replace("stationary", "transplant").replace("/T-SVGP/", "/T-SVGP-to-SVGP/")
            after = dict(config={**config, "donor_method": "T-SVGP",
                                 "donor_params_sha256": figures.digest(checkpoint)},
                         run=after_relative, rmse=before["rmse"] - 0.1,
                         nll=before["nll"] - 0.2,
                         prediction_source=str(predictions / after_relative / "prediction.json"))
            rows[("stationary", "rbf", "pol", 64, "T-SVGP", seed)] = before
            rows[("transplant", "rbf", "pol", 64, "T-SVGP", seed)] = after
        return rows

    def test_complete_pairs_and_delta_sem(self):
        with tempfile.TemporaryDirectory() as tmp, patch.multiple(
                figures, KERNELS=("rbf",), DATASETS=("pol",), BUDGETS=(64,), METHODS=("T-SVGP",)):
            root, predictions = Path(tmp).resolve() / "training", Path(tmp).resolve() / "predictions"
            rows = self.fixture(root, predictions)
            original = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            selected, paired, hashes = figures.validate_rows(rows, root, [predictions])
            self.assertEqual((len(selected), len(paired), len(hashes)), (20, 10, 10))
            result = figures.aggregate(rows, paired)
            delta = next(r for r in result if r["kind"] == "paired_delta" and r["metric"] == "rmse")
            self.assertAlmostEqual(delta["mean"], -0.1)
            self.assertAlmostEqual(delta["sem"], 0)
            self.assertEqual(original, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
            key = ("transplant", "rbf", "pol", 64, "T-SVGP", 0)
            rows[key]["config"]["donor_params_sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                figures.validate_rows(rows, root, [predictions])
            rows[key]["config"]["test_indices_sha256"] = "different"
            with self.assertRaisesRegex(ValueError, "split/preprocessing"):
                figures.validate_rows(rows, root, [predictions])
            del rows[key]
            with self.assertRaisesRegex(ValueError, "Missing/invalid"):
                figures.validate_rows(rows, root, [predictions])

    def test_combined_lines_share_color_and_differ_in_style(self):
        lookup = {}
        for method in figures.METHODS:
            for variant in ("complete", "transplant"):
                for m in figures.BUDGETS:
                    for metric in ("rmse", "nll"):
                        lookup[variant, "rbf", "pol", method, m, metric] = dict(mean=0.1, sem=0.01)
        fig = figures.draw_page(lookup, "combined", "rbf", ("pol",))
        try:
            curves = [c.lines[0] for c in fig.axes[0].containers]
            self.assertEqual(len(curves), 8)
            self.assertEqual({c.get_linewidth() for c in curves}, {figures.LINE_WIDTH})
            for i in range(0, 8, 2):
                self.assertEqual(curves[i].get_color(), curves[i+1].get_color())
                self.assertEqual(curves[i].get_linestyle(), "-")
                self.assertEqual(curves[i+1].get_linestyle(), "--")
        finally:
            figures.plt.close(fig)

    def test_paper_transplant_only_is_solid_and_has_direction_arrows(self):
        lookup = {("transplant", "rbf", d, method, m, metric): dict(mean=0.1, sem=0.01)
                  for d in ("pol", "protein") for method in figures.METHODS
                  for m in figures.BUDGETS for metric in ("rmse", "nll")}
        fig = figures.draw_page(lookup, "transplant", "rbf", ("pol", "protein"),
                                transplant_only_solid=True, direction_arrows=True)
        try:
            for i, ax in enumerate(fig.axes):
                self.assertEqual(sum(t.get_text() == "lower is better" for t in ax.texts), int(i % 2 == 0))
                for container in ax.containers:
                    self.assertEqual(container.lines[0].get_linestyle(), "-")
                    self.assertEqual(container.lines[0].get_linewidth(), figures.LINE_WIDTH)
            self.assertTrue(all(t.get_text().endswith(" donor") for t in fig.legends[0].get_texts()))
        finally:
            figures.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
