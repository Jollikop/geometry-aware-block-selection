import argparse
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from Simulations.analysis import evaluate_svgp_frozen as evaluation
from svgp import run_benchmark as runner


class FrozenEvaluationTests(unittest.TestCase):
    def test_selected_completed_seeds_are_disjoint_and_have_their_own_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "frozen", Path(tmp) / "predictions"
            selected = []
            for kernel in ("rbf", "matern52", "amplitude-rbf", "gibbs"):
                for m in (64, 128, 256, 512):
                    for method in evaluation.METHODS:
                        for seed in (2, 6):
                            relative = Path(f"svgp/transplant/runs/{kernel}/kin40k/M{m}/{method}-to-SVGP/seed{seed}")
                            path = root / relative
                            path.mkdir(parents=True)
                            for name in ("training.json", "params.npz", "curve.npz"):
                                (path / name).touch()
            def fake_child(command):
                relative = Path(command[-1])
                selected.append(relative)
                evaluation.write_json(output / relative / "prediction.json",
                                      {"identity": {"gpu_kinds": ["test-GPU"]}})
                return SimpleNamespace(returncode=0)
            argv = ["evaluate", "--root", str(root), "--output", str(output),
                    "--source-sim", str(Path(__file__).resolve().parents[1]),
                    "--stage", "transplant", "--datasets", "kin40k",
                    "--seeds", "6", "2", "--measurement-context", "concurrent-training"]
            with patch.object(sys, "argv", argv), patch.object(evaluation.subprocess, "run", side_effect=fake_child), \
                    patch("builtins.print"), self.assertRaises(SystemExit) as exit_context:
                evaluation.main()
            self.assertEqual(exit_context.exception.code, 0)
            self.assertEqual(len(selected), 128)
            self.assertEqual(len(set(selected)), 128)
            self.assertEqual({p.name for p in selected}, {"seed2", "seed6"})
            summary = json.loads((output / "transplant_seeds_2-6_evaluation_summary.json").read_text())
            self.assertTrue(summary["evaluation_complete"])
            self.assertEqual(summary["seeds"], [2, 6])
            self.assertFalse((output / "transplant_evaluation_summary.json").exists())

    def test_overlapping_output_is_rejected(self):
        root = Path("/tmp/frozen")
        for output in (root, root / "predictions", root.parent):
            with self.assertRaises(ValueError):
                evaluation.separate_output(root, output)
        evaluation.separate_output(root, Path("/tmp/separate_predictions"))

    def test_transplant_reads_saved_q_and_leaves_every_source_byte_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp).resolve() / "frozen", Path(tmp).resolve() / "predictions"
            relative = "svgp/transplant/runs/rbf/kin40k/M4/SBT-SVGP-to-SVGP/seed0"
            run = root / relative
            run.mkdir(parents=True)
            (root / "EXPECTED_HEAD").write_text("test-head")
            rng = np.random.default_rng(5)
            X, y = rng.normal(size=(27, 3)), rng.normal(size=27)
            x, _, _, _, meta = runner.jd.split_standardize(X, y, 0, 19, 8)
            params = runner.sv.init_params(x, 4, 0)
            # Non-optimal q: do not accidentally replace it by collapsed SGPR.
            params["q_mu"] = np.ones(4) * 0.2
            np.savez(run / "params.npz", **params)
            np.savez(run / "curve.npz", epoch=[1], stoch_obj_n=[0.1])
            donor = root / "svgp/stationary/runs/rbf/kin40k/M4/SBT-SVGP/seed0/params.npz"
            donor.parent.mkdir(parents=True)
            np.savez(donor, **params)
            raw = Path(tmp) / "raw.npz"
            np.savez(raw, X=X, y=y)
            cfg = {"code_commit": "test-head", "method": "SBT-SVGP-to-SVGP",
                   "training_method": "SVGP", "donor_method": "SBT-SVGP",
                   "donor_params_sha256": evaluation.sha(donor), "kernel": "rbf",
                   "dataset": "kin40k", "M": 4, "seed": 0, "split_seed": 0,
                   "n_train": 19, "n_test": 8, "data_signature": evaluation.sha(raw),
                   **{k: meta[k] for k in ("split_rng", "train_indices_sha256",
                       "test_indices_sha256", "preprocessing_protocol")}}
            (run / "training.json").write_text(json.dumps({
                "config": cfg, "config_id": runner.config_fingerprint(cfg),
                "training_complete": True}))
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            args = argparse.Namespace(root=root, output=output, run=relative,
                source_sim=Path(__file__).resolve().parents[1], stage="transplant",
                test_chunk=4, train_chunk=7, block_batch=2, factor_cache_mib=1,
                reps=1, reference_points=8)
            with patch.object(runner.dm, "raw_cache_path", return_value=raw), \
                 patch.object(runner.dm, "load_raw", return_value=(X, y)), \
                 patch.object(runner.sv, "train_svgp", side_effect=AssertionError("NO TRAINING")):
                evaluation.evaluate(args)
                evaluation.evaluate(args)  # exact identity reuses separate output
            after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            result = json.loads((output / relative / "prediction.json").read_text())
            self.assertEqual(result["identity"]["predictor"], "sparse")
            self.assertLess(result["reference_mean_max_abs"], 1e-8)


if __name__ == "__main__":
    unittest.main()
