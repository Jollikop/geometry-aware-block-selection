import json
from pathlib import Path
import tempfile
import unittest

from Simulations.analysis import make_svgp_preview as preview


class PreviewTests(unittest.TestCase):
    def test_sidecars_are_deduplicated_checked_and_paired_without_input_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, predictions = Path(tmp).resolve() / "training", Path(tmp).resolve() / "prediction"
            root.mkdir()
            predictions.mkdir()
            (root / "EXPECTED_HEAD").write_text("frozen-head")
            configs = []
            for stage, method in (("stationary", "T-SVGP"), ("transplant", "T-SVGP-to-SVGP")):
                run = root / f"svgp/{stage}/runs/rbf/pol/M64/{method}/seed2"
                run.mkdir(parents=True)
                (run / "params.npz").write_bytes(b"test checkpoint")
                cfg = {"code_commit": "frozen-head", "kernel": "rbf", "dataset": "pol",
                       "M": 64, "method": method, "seed": 2, "train_indices_sha256": "train",
                       "test_indices_sha256": "test", "data_signature": "data", "preprocessing_protocol": "test"}
                configs.append(cfg)
                record = {"config": cfg, "config_id": method, "training_complete": True,
                          preview.OBJECTIVE: 0.3}
                (run / "training.json").write_text(json.dumps(record))
                if stage == "stationary":
                    (run / "metrics.json").write_text(json.dumps({**record, "rmse": 0.4, "nll": 0.2,
                                                                 "reported_predictor": "diagonal"}))
            relative = run.relative_to(root)
            dest = predictions / relative / "prediction.json"
            dest.parent.mkdir(parents=True)
            sidecar = {"identity": {"config_id": "T-SVGP-to-SVGP", "predictor": "sparse",
                                   "inputs": {str(relative / "params.npz"): preview.digest(run / "params.npz"),
                                              "EXPECTED_HEAD": preview.digest(root / "EXPECTED_HEAD")}},
                       "training_config": configs[-1], "rmse": 0.3, "nll": 0.1}
            dest.write_text(json.dumps(sidecar))
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            rows, issues, _ = preview.read_campaign(root, [predictions, predictions])
            self.assertFalse(issues)
            self.assertEqual(len(rows), 2)
            paired = preview.paired_deltas(rows)
            self.assertEqual(len(paired), 1)
            self.assertAlmostEqual(next(iter(paired.values()))["rmse"], -0.1)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
            # A stale checkpoint must not contribute even in a PREVIEW.
            (run / "params.npz").write_bytes(b"changed checkpoint")
            rows, issues, _ = preview.read_campaign(root, [predictions])
            self.assertTrue(any("stale prediction input" in r["reason"] for r in issues))
            self.assertEqual(len(preview.paired_deltas(rows)), 0)

    def test_transplant_without_predicted_donor_is_kept_but_not_used_for_deltas(self):
        key = ("transplant", "rbf", "pol", 64, "SVGP", 2)
        rows = {key: {"rmse": 0.2, "nll": 0.1, "config": {}}}
        self.assertEqual(preview.paired_deltas(rows), {})
        self.assertIn(key, rows)

    def test_invalid_scope_is_rejected(self):
        with self.assertRaises(ValueError):
            preview.key_from_relative(Path("svgp/transplant/runs/rbf/pol/M64/SVGP/seed2"))


if __name__ == "__main__":
    unittest.main()
