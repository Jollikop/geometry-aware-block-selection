import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import tempfile
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from analysis.continue_svgp_transplant import deterministic_elbo_fn, expected_runs, build_args, separate
from svgp import engine as sv
from tests.test_checkpoint_fork import make_args, X_TR, Y_TR
from analysis import continue_svgp_transplant as cont


class ContinuationTests(unittest.TestCase):
    def test_persistence_end_to_end_and_cache(self):
        from svgp import run_benchmark as runner
        with tempfile.TemporaryDirectory() as d:
            root, output, data = (Path(d).resolve() / name for name in ('parent', 'out', 'data'))
            root.mkdir(); output.mkdir(); data.mkdir()
            (root / 'EXPECTED_HEAD').write_text('test-parent\n')
            rng = np.random.default_rng(44)
            X = rng.normal(size=(40, 3)); y = np.sin(X[:, 0])
            runner.dm._write_cache('pol', data, X, y)
            x, yy, _, _, meta = runner.jd.split_standardize(X, y, 0, 32, 8, split_rng='numpy-pcg64-v1')
            cfg = dict(vars(make_args()), kernel='rbf', dataset='pol', M=8, seed=0,
                       method='SVGP-to-SVGP', training_method='SVGP', donor_method='SVGP',
                       split_seed=0, split_rng='numpy-pcg64-v1', n_train=32, n_test=8,
                       block=16, ll_batch_size=16, q_optimizer='natural',
                       optimizer='adam', lr_schedule='constant')
            for key in ('train_indices_sha256', 'test_indices_sha256', 'preprocessing_protocol'):
                cfg[key] = meta[key]
            rel = 'svgp/transplant/runs/rbf/pol/M8/SVGP-to-SVGP/seed0'
            run = root / rel; run.mkdir(parents=True)
            initial = sv.init_params(x, 8, 0)
            initial['q_cov'] = jnp.eye(8); del initial['q_raw']
            runner._atomic_npz(run / 'params.npz', **runner._parameter_arrays(initial))
            runner._atomic_npz(run / 'curve.npz', epoch=np.array([150]), stoch_obj_n=np.array([0.]))
            cont.write_json(run / 'training.json', dict(config=cfg, config_id='synthetic', epochs=150))
            inputs = {str(f.relative_to(root)): cont.sha(f) for f in run.iterdir()}
            raw = data / 'pol.npz'
            plan = dict(protocol=cont.PROTOCOL, root=str(root), source=cont.source_hashes(),
                        expected_head_sha256=cont.sha(root / 'EXPECTED_HEAD'), data_dir=str(data),
                        datasets={'pol': dict(path=str(raw), sha256=cont.sha(raw),
                                             size=raw.stat().st_size, mtime_ns=raw.stat().st_mtime_ns)},
                        runs=[dict(run=rel, config_id='synthetic', inputs=inputs)])
            cont.write_json(output / 'continuation_plan.json', plan)
            args = SimpleNamespace(output=output, index=0, cpu_test=True)
            cont.run_one(args)
            out = output / rel
            self.assertTrue((out / 'COMPLETE.json').is_file())
            for epoch in (150, 200, 250, 300):
                self.assertTrue(json.loads((out / f'epoch{epoch}.json').read_text())['theta_unchanged'])
            before = {str(f): cont.sha(f) for f in out.iterdir()}
            cont.run_one(args)
            self.assertEqual(before, {str(f): cont.sha(f) for f in out.iterdir()})
            cont.assert_inputs(root, plan['runs'][0])

    def test_full_grid_and_shards(self):
        rows = list(expected_runs())
        self.assertEqual(len(rows), 5120)
        self.assertEqual(len({r for r, _ in rows}), 5120)
        self.assertEqual([len(rows[i::5]) for i in range(5)], [1024] * 5)
        self.assertEqual(sum(ids['dataset'] == '3droad' for _, ids in rows), 640)

    def test_output_safety(self):
        for other in ('/a', '/a/b', '/'):
            with self.assertRaises(ValueError):
                separate(Path('/a'), Path(other))
        separate(Path('/a'), Path('/b'))

    def test_frozen_groups(self):
        a = build_args({})
        self.assertEqual(sv.parse_phases(a.phases, a.epochs)[-1], (300, frozenset({'theta'})))
        self.assertFalse(a.record_history)
        self.assertEqual(a.checkpoint_at_epochs, [200, 250, 300])

    def test_deterministic_elbo_matches_engine_all_kernels(self):
        x, y = X_TR[:37], Y_TR[:37]
        for k in ('rbf', 'matern52', 'amplitude-rbf', 'gibbs'):
            with self.subTest(kernel=k):
                params = sv.init_params(x, 8, 0, kernel_name=k)
                engine = sv.make_engine(x, y, x[:1], y[:1], kernel_name=k)
                idx = jnp.arange(len(x))[None, :]
                value, _ = engine['trace'](params, idx, jnp.ones_like(idx), 1.)
                full = deterministic_elbo_fn(x, y, k, sv, chunk=16)(params)
                np.testing.assert_allclose(float(full), -float(value)/len(x), rtol=1e-10, atol=1e-10)

    def test_final_checkpoint_resume_exact_and_theta_fixed(self):
        x, y = X_TR[:80], Y_TR[:80]
        args = make_args(epochs=10, phases='10:theta', optimizer='adam', log_every=5,
                         block=20, ll_batch_size=20, warp_gate_fraction=0.,
                         checkpoint_at_epochs=[5, 10], track_grad_stats=True)
        engine = sv.make_engine(x, y, x[:1], y[:1])
        initial = sv.init_params(x, 8, 0)
        def bank(p):
            return sv.build_method_bank('SVGP', jnp.asarray(x), p, block=20, seed=0, ll_batch_size=20, jitter=1e-8)
        with tempfile.TemporaryDirectory() as d:
            def save(epoch, p, state):
                sv.save_training_checkpoint(Path(d) / f'{epoch}.npz', epoch, p, state)
            final, _, _, _ = sv.train_svgp(engine, 'SVGP', initial, bank, engine['metrics'], len(x),
                                            args, checkpoint_fn=save)
            p, leaves, meta = sv.load_training_checkpoint(Path(d) / '5.npz')
            resumed, _, info, _ = sv.train_svgp(engine, 'SVGP', p, bank, engine['metrics'], len(x),
                                              args, start_epoch=5, resume_optimizer_leaves=leaves)
            saved_final, _, meta = sv.load_training_checkpoint(Path(d) / '10.npz')
            self.assertEqual(meta['epoch'], 10)
            self.assertGreater(info['optimizer_leaves_restored'], 0)
            for key in final:
                np.testing.assert_array_equal(final[key], resumed[key])
                np.testing.assert_array_equal(final[key], saved_final[key])
            for key in ('log_ell', 'log_s2', 'log_sf2'):
                np.testing.assert_array_equal(initial[key], final[key])


if __name__ == '__main__':
    unittest.main()
