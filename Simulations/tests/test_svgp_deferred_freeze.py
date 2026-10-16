from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Simulations.analysis import validate_svgp_deferred_freeze as freeze


class DeferredFreezeTests(unittest.TestCase):
    def test_donor_path_uses_stationary_or_extended_stage(self) -> None:
        root = Path("/freeze")
        self.assertEqual(
            freeze.donor_params_path(root, "rbf", "pol", 128, "SBT-SVGP", 3),
            root / "svgp/stationary/runs/rbf/pol/M128/SBT-SVGP/seed3/params.npz",
        )
        self.assertEqual(
            freeze.donor_params_path(
                root, "gibbs", "pol", 128, "SBT-SVGP", 3
            ),
            root / "svgp/extended/runs/gibbs/pol/M128/SBT-SVGP/seed3/params.npz",
        )

    def test_record_requires_matching_fingerprint_and_reference(self) -> None:
        config = {"code_commit": "abc", "epochs": 300}
        record = {
            "implementation": "x",
            "config": config,
            "config_id": freeze.fingerprint(config),
            "training_complete": True,
        }
        self.assertEqual(freeze.record_errors(record, "abc"), [])
        self.assertIn(
            "numerical reference commit mismatch",
            freeze.record_errors(record, "def"),
        )
        record["config_id"] = "bad"
        self.assertIn(
            "config fingerprint mismatch",
            freeze.record_errors(record, "abc"),
        )

    def test_npz_and_json_checks_reject_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / "params.npz"
            with good.open("wb") as handle:
                np.savez_compressed(handle, value=np.arange(3))
            bad = root / "partial.npz"
            bad.write_bytes(good.read_bytes()[:20])
            self.assertTrue(freeze.valid_npz(good))
            self.assertFalse(freeze.valid_npz(bad))

            payload = root / "training.json"
            payload.write_text(json.dumps({"training_complete": True}))
            self.assertEqual(
                freeze.load_record(payload), {"training_complete": True}
            )


if __name__ == "__main__":
    unittest.main()
