import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from Simulations.analysis import run_svgp_missing_predictions as missing
from Simulations.analysis.validate_svgp_deferred_freeze import fingerprint


class MissingPredictionTests(unittest.TestCase):
    def test_selection_keeps_matching_metrics_rejects_stale_or_wrong_predictor(self):
        cfg = dict(code_commit='head', kernel='gibbs', dataset='pol', M=64,
                   method='SBT-SVGP', seed=0, training_method='SBT-SVGP')
        record = dict(config=cfg, config_id=fingerprint(cfg), training_complete=True)
        metrics = {**record, 'reported_predictor': 'block', 'rmse': 0.2, 'nll': -0.1}
        self.assertTrue(missing.predictive_metrics_valid(metrics, record, 'block', 'head'))
        for changes in ({'reported_predictor': 'sparse'}, {'nll': float('nan')},
                        {'config_id': 'old'}, {'rmse': None}):
            self.assertFalse(missing.predictive_metrics_valid({**metrics, **changes}, record, 'block', 'head'))

    def test_inspection_requires_block_partition_and_does_not_write_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            cfg = dict(code_commit='head', kernel='rbf', dataset='pol', M=64,
                       method='SBT-SVGP', seed=0, training_method='SBT-SVGP')
            record = dict(config=cfg, config_id=fingerprint(cfg), training_complete=True)
            (path / 'training.json').write_text(json.dumps(record))
            for name in ('params.npz', 'curve.npz', 'final_blocks.npz'):
                with zipfile.ZipFile(path / name, 'w') as archive:
                    archive.writestr('test.npy', b'test')
            before = {p: p.read_bytes() for p in path.iterdir()}
            cached, required, _ = missing.inspect_run(path, 'SBT-SVGP', 'head', {'method': 'SBT-SVGP'})
            self.assertFalse(cached)
            self.assertIn(path / 'final_blocks.npz', required)
            self.assertEqual(before, {p: p.read_bytes() for p in path.iterdir()})
            (path / 'final_blocks.npz').unlink()
            with self.assertRaises(ValueError):
                missing.inspect_run(path, 'SBT-SVGP', 'head', {'method': 'SBT-SVGP'})


if __name__ == '__main__':
    unittest.main()
