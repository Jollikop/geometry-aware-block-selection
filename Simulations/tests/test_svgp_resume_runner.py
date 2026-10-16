from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SIM_ROOT = Path(__file__).resolve().parents[1]


class ResumeRunnerTests(unittest.TestCase):
    def test_chunk_and_provenance_wrap_only_new_historical_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            source_sim = tmp / "Simulations"
            common = source_sim / "common"
            runner = source_sim / "svgp" / "run_benchmark.py"
            common.mkdir(parents=True)
            runner.parent.mkdir(parents=True)
            (common / "__init__.py").write_text("")
            (common / "predictors.py").write_text(
                "def predictive_variance(*args, chunk=None, **kwargs):\n"
                "    return chunk\n"
            )
            runner.write_text(
                "import json, sys\n"
                "from pathlib import Path\n"
                "from common import predictors\n"
                "assert predictors.predictive_variance() == 37\n"
                "Path(sys.argv[1]).write_text(json.dumps({\n"
                "  'implementation': 'svgp-frozen-test',\n"
                "  'config_id': 'historical-fingerprint',\n"
                "  'config': {'code_commit': 'historical-commit'}\n"
                "}))\n"
            )
            output = tmp / "metrics.json"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SIM_ROOT / "svgp_resume_runner.py"),
                    "--source-runner", str(runner),
                    "--prediction-chunk", "37",
                    "--orchestrator-commit", "repair-commit",
                    "--", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(output.read_text())

        self.assertIn("common test chunk=37", completed.stdout)
        self.assertEqual(payload["config_id"], "historical-fingerprint")
        self.assertEqual(payload["config"]["code_commit"], "historical-commit")
        self.assertEqual(payload["prediction_execution"]["test_chunk"], 37)
        self.assertTrue(
            payload["prediction_execution"]["mathematical_predictor_unchanged"]
        )
        self.assertEqual(
            payload["prediction_execution"]["orchestrator_commit"],
            "repair-commit",
        )


if __name__ == "__main__":
    unittest.main()
