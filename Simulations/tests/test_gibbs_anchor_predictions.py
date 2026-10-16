"""Reject stale or scientifically mismatched results before certification."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import gibbs_anchor_predictions as campaign


class PredictionIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'training'
        self.out = Path(self.tmp.name) / 'prediction'
        self.rel = 'svgp/extended/runs/gibbs/pol/M64/SBT-SVGP/seed0'
        self.cfg = dict(n_test=12, sbt_geometry='anchor', init_from_kernel='rbf', epochs=300)
        self.task = dict(run=self.rel, config=self.cfg)
        inputs = {str(Path(self.rel) / n): 'hash-' + n for n in
                  ('training.json', 'params.npz', 'curve.npz', 'final_blocks.npz')}
        inputs['EXPECTED_HEAD'] = 'head'
        self.plan = dict(root=str(self.root), training_inputs=inputs,
                         source_files={'common/kernels.py': 'kernel', 'svgp/engine.py': 'engine',
                                       'svgp/run_benchmark.py': 'runner', 'svgp/fast_prediction.py': 'backend'},
                         tools={'tools/evaluate_svgp_frozen.py': 'evaluator'}, settings=campaign.SETTINGS)
        self.record = dict(training_config=self.cfg, reference_check_points=8,
            identity=dict(config_id='id', predictor='block', forced_predictor=None,
                inputs=inputs, source_modules={k:v for k,v in self.plan['source_files'].items()
                                               if k != 'svgp/fast_prediction.py'},
                backend_sha256='backend', evaluator_sha256='evaluator',
                reference_policy='frozen-full-training-reference-v1',
                gpu_kinds=['NVIDIA A100-PCIE-40GB'], **campaign.SETTINGS))
        for key in ('rmse', 'nll', 'rmse_orig', 'nll_orig', 'warm_prediction_ms', 'prepare_s',
                    'compile_and_first_prediction_s', 'reference_mean_max_abs', 'reference_variance_max_abs'):
            self.record[key] = 0.1
        campaign.write_json(self.root / self.rel / 'training.json', dict(config_id='id'))

    def check(self, record):
        campaign.write_json(self.out / self.rel / 'prediction.json', record)
        return campaign.check_prediction(self.out, self.plan, self.task)

    def test_matching_full_block_result_is_accepted(self):
        self.assertEqual(self.check(self.record)['rmse'], 0.1)

    def test_wrong_checkpoint_predictor_or_backend_is_rejected(self):
        for key, value in [('predictor', 'sparse'), ('backend_sha256', 'changed'),
                           ('inputs', {}), ('reference_points', 0), ('gpu_kinds', ['CPU'])]:
            with self.subTest(key=key):
                record = copy.deepcopy(self.record)
                record['identity'][key] = value
                with self.assertRaises(ValueError): self.check(record)

    def test_probe_config_cannot_be_certified_as_anchor(self):
        record = copy.deepcopy(self.record)
        record['training_config']['sbt_geometry'] = 'residual-probe'
        with self.assertRaises(ValueError): self.check(record)

    def test_missing_reference_checks_are_rejected(self):
        record = copy.deepcopy(self.record)
        record['reference_check_points'] = 0
        with self.assertRaises(ValueError): self.check(record)


if __name__ == '__main__': unittest.main()
