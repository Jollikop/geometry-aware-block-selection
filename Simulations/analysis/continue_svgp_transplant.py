#!/usr/bin/env python3
"""Separate, resumable 150 -> 300 epoch extension of the 5,120 SVGP transplants.

The epoch-150 archive has no Adam state: restart Adam ONCE, preserve q and Z,
and continue the absolute-epoch minibatch stream. New checkpoints carry Adam.
No test prediction, threshold relaxation, original-file writes or auto-requeue.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
DATASETS = 'pol superconductivity kin40k protein elevators traffic electricity 3droad'.split()
KERNELS = 'rbf matern52 amplitude-rbf gibbs'.split()
METHODS = 'SVGP T-SVGP RBT-SVGP SBT-SVGP'.split()
PROTOCOL = 'svgp-transplant-extension-150-to-300-adam-reset-v1'
NUMERIC_PACKAGES = ('numpy', 'jax', 'jaxlib', 'optax', 'scipy')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return 'sha256:' + h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def separate(a, b):
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('output must be disjoint from input campaign and code')


def source_hashes():
    paths = list((SIM / 'common').glob('*.py'))
    paths += list((SIM / 'svgp').glob('*.py'))
    paths += [Path(__file__), SIM / 'svgp_transplant_continue.sbatch',
              SIM / 'submit_svgp_transplant_continue.sh']
    return {str(p.relative_to(SIM)): sha(p) for p in sorted(paths)}


def expected_runs():
    for k in KERNELS:
        for d in DATASETS:
            for m in (64, 128, 256, 512):
                for donor in METHODS:
                    for seed in range(10):
                        rel = f'svgp/transplant/runs/{k}/{d}/M{m}/{donor}-to-SVGP/seed{seed}'
                        yield rel, dict(kernel=k, dataset=d, M=m, donor_method=donor,
                                        seed=seed, method=donor + '-to-SVGP')


def check_parent(record, ids, head):
    from common.provenance import config_fingerprint
    cfg = record['config']
    if record.get('training_complete') is not True or record.get('epochs') != 150:
        raise ValueError('parent must be complete at epoch 150')
    if config_fingerprint(cfg) != record['config_id']:
        raise ValueError('parent config fingerprint mismatch')
    expected = dict(ids, code_commit=head, epochs=150, training_method='SVGP',
                    phases='10:theta,z 140:theta', optimizer='adam',
                    q_optimizer='natural', lr=0.001, z_lr=0.001, natural_lr=0.05,
                    lr_schedule='constant', refresh_until=0, swa_fraction=0.0)
    for k, v in expected.items():
        if cfg.get(k) != v:
            raise ValueError(f'unsupported parent {k}: {cfg.get(k)!r} != {v!r}')
    return cfg


def plan(a):
    import numpy as np
    from common import datasets as dm
    head = (a.root / 'EXPECTED_HEAD').read_text().strip()
    rows, datasets = [], {}
    versions = {name: importlib.metadata.version(name) for name in NUMERIC_PACKAGES}
    for rel, ids in expected_runs():
        run = a.root / rel
        record = json.loads((run / 'training.json').read_text())
        cfg = check_parent(record, ids, head)
        parent_versions = record['runtime']['packages']
        for name, version in versions.items():
            if parent_versions.get(name) != version:
                raise ValueError(f'{name} runtime mismatch: {version} != {parent_versions.get(name)}')
        with np.load(run / 'params.npz', allow_pickle=False) as z:
            if not {'z', 'q_mu', 'q_cov', 'log_ell', 'log_s2', 'log_sf2'} <= set(z.files):
                raise ValueError(f'missing saved posterior: {rel}')
            if not all(np.all(np.isfinite(z[k])) for k in z.files):
                raise ValueError(f'nonfinite parent: {rel}')
        with np.load(run / 'curve.npz', allow_pickle=False) as curve:
            if int(curve['epoch'][-1]) != 150:
                raise ValueError(f'parent curve not at 150: {rel}')
        stage = 'stationary' if ids['kernel'] in KERNELS[:2] else 'extended'
        donor = Path('svgp') / stage / 'runs' / ids['kernel'] / ids['dataset'] / f'M{ids["M"]}' / ids['donor_method'] / f'seed{ids["seed"]}' / 'params.npz'
        if sha(a.root / donor) != cfg['donor_params_sha256']:
            raise ValueError(f'donor lineage mismatch: {rel}')
        files = [run / name for name in ('training.json', 'params.npz', 'curve.npz')]
        files.append(a.root / donor)
        rows.append(dict(run=rel, config_id=record['config_id'],
                         inputs={str(p.relative_to(a.root)): sha(p) for p in files}))
        dataset = ids['dataset']
        if dataset not in datasets:
            raw = dm.raw_cache_path(dataset, a.data_dir)
            if sha(raw) != cfg['data_signature']:
                raise ValueError(f'dataset hash mismatch: {dataset}')
            datasets[dataset] = dict(path=str(raw), sha256=sha(raw),
                                     size=raw.stat().st_size, mtime_ns=raw.stat().st_mtime_ns)
        elif datasets[dataset]['sha256'] != cfg['data_signature']:
            raise ValueError('inconsistent dataset signatures')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'slurm').mkdir()
    (a.output / 'state').mkdir()
    (a.output / 'EXPECTED_HEAD').write_text(head + '\n')
    write_json(a.output / 'continuation_plan.json', dict(
        protocol=PROTOCOL, root=str(a.root), output=str(a.output),
        data_dir=str(a.data_dir), expected_head_sha256=sha(a.root / 'EXPECTED_HEAD'),
        workers=5, runs=rows, source=source_hashes(), datasets=datasets, numeric_packages=versions,
        start_epoch=150, total_epochs=300, checkpoint_epochs=[200, 250, 300],
        optimizer_state_at_150='not_saved; Adam reset once, posterior retained',
        selection_rule='all 5120 configurations, no selection using test performance',
        prediction='deferred', release_ready=False))
    print(f'PLANNED {len(rows)} continuations, 5 workers; output={a.output}', flush=True)


def read_plan(output):
    p = json.loads((output / 'continuation_plan.json').read_text())
    if p['protocol'] != PROTOCOL or p['source'] != source_hashes():
        raise ValueError('continuation source changed; do not mix implementations')
    if sha(Path(p['root']) / 'EXPECTED_HEAD') != p['expected_head_sha256']:
        raise ValueError('parent numerical reference changed')
    for name, version in p.get('numeric_packages', {}).items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'{name} version changed after planning')
    return p


def assert_inputs(root, row):
    for rel, digest in row['inputs'].items():
        if sha(root / rel) != digest:
            raise ValueError(f'parent input changed: {rel}')


def build_args(cfg):
    values = dict(cfg)
    values.update(epochs=300, phases='10:theta,z 290:theta',
                  natural_q=True, record_history=False, evaluate_during_training=False,
                  track_grad_stats=True, live=True, log_every=10,
                  checkpoint_at_fraction=0.0, checkpoint_at_epochs=[200, 250, 300],
                  metrics_at_epochs=[], warp_gate_fraction=0.0, swa_fraction=0.0,
                  refresh_until=0, refresh_every=301)
    return SimpleNamespace(**values)


def deterministic_elbo_fn(x, y, kernel_name, sv, chunk=512, jitter=1e-8):
    """Full-data, fixed-parameter sparse ELBO; O(chunk*M + M^2), never N^2.

    Matches engine trace objective, including residual clipping and q jitter.
    This diagnostic reads TRAINING targets only and never tunes on test data.
    """
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular
    n = len(x)
    pad = (-n) % chunk
    xx = jnp.pad(jnp.asarray(x), ((0, pad), (0, 0))).reshape(-1, chunk, x.shape[1])
    yy = jnp.pad(jnp.asarray(y).ravel(), (0, pad)).reshape(-1, chunk)
    mask = (jnp.arange(n + pad) < n).reshape(-1, chunk)

    @jax.jit
    def evaluate(params):
        ell, sf2, s2, lq, warp = sv.transform(params)
        z = params['z']
        luu = jnp.linalg.cholesky(sv.kernel(z, z, ell, sf2, kernel_name, warp)
                                   + jitter * jnp.eye(len(z)))
        kl = 0.5 * (jnp.sum(params['q_mu'] ** 2) + jnp.sum(lq ** 2)
                    - len(z) - 2 * jnp.sum(jnp.log(jnp.diag(lq))))

        def add(total, batch):
            xb, yb, valid = batch
            a = solve_triangular(luu, sv.kernel(z, xb, ell, sf2, kernel_name, warp), lower=True)
            mu = a.T @ params['q_mu']
            qvar = jnp.sum((lq.T @ a) ** 2, axis=0)
            residual = jnp.maximum(sv.kernel_diag(xb, ell, sf2, kernel_name, warp)
                                   - jnp.sum(a * a, axis=0), 0)
            term = -0.5 * (jnp.log(2 * jnp.pi * s2)
                            + ((yb - mu) ** 2 + qvar + residual) / s2)
            return total + jnp.sum(jnp.where(valid, term, 0)), None

        likelihood, _ = jax.lax.scan(add, jnp.asarray(0., dtype=jnp.float64), (xx, yy, mask))
        return (likelihood - kl) / n
    return evaluate


def finished(run, identity):
    marker = run / 'COMPLETE.json'
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    if record['identity'] != identity:
        raise ValueError('existing output belongs to another continuation')
    for name, digest in record['files'].items():
        if sha(run / name) != digest:
            raise ValueError(f'completed output changed: {run / name}')
    return True


def run_one(a):
    import numpy as np
    from svgp import run_benchmark as runner
    import jax
    p = read_plan(a.output)
    row = p['runs'][a.index]
    root, out = Path(p['root']), a.output / row['run']
    separate(root, a.output)
    if a.output not in out.resolve().parents:
        raise ValueError('output symlink escapes continuation root')
    os.environ['SCALABLE_GP_DATASET_DIR'] = p['data_dir']
    identity = dict(plan_sha256=sha(a.output / 'continuation_plan.json'),
                    parent_config_id=row['config_id'], protocol=PROTOCOL)
    assert_inputs(root, row)
    if finished(out, identity):
        print('CACHED ' + row['run'], flush=True)
        return
    devices = jax.devices()
    if not a.cpu_test and (len(devices) != 1 or devices[0].platform != 'gpu'):
        raise ValueError(f'exactly one assigned usable GPU required: {devices}')
    parent = root / row['run']
    record = json.loads((parent / 'training.json').read_text())
    cfg = record['config']
    raw = Path(p['datasets'][cfg['dataset']]['path'])
    stored = p['datasets'][cfg['dataset']]
    if raw.stat().st_size != stored['size'] or raw.stat().st_mtime_ns != stored['mtime_ns']:
        raise ValueError('dataset changed since planning')
    X, y = runner.dm.load_raw(cfg['dataset'], data_dir=Path(p['data_dir']))
    x, y, _, _, meta = runner.jd.split_standardize(
        X, y, cfg['split_seed'], cfg['n_train'], cfg['n_test'],
        angle_cols=runner.dm.angle_cols_for(cfg['dataset']), split_rng=cfg['split_rng'])
    for key in ('train_indices_sha256', 'test_indices_sha256', 'preprocessing_protocol'):
        if cfg[key] != meta[key]:
            raise ValueError(f'preprocessing mismatch: {key}')
    sv = runner.sv
    original = runner._load_parameter_arrays(parent / 'params.npz')
    params, leaves, start = original, None, 150
    out.mkdir(parents=True, exist_ok=True)
    diagnostic = deterministic_elbo_fn(x, y, cfg['kernel'], sv)
    theta = {k: np.asarray(v) for k, v in original.items() if k in sv.PARAMETER_GROUPS['theta']}

    def audit(epoch, current):
        for k, v in theta.items():
            if not np.array_equal(v, np.asarray(current[k])):
                raise ValueError(f'frozen theta changed: {k}')
        if not all(np.all(np.isfinite(np.asarray(v))) for v in current.values()):
            raise ValueError('nonfinite parameters')
        elbo = float(diagnostic(current))
        if not np.isfinite(elbo):
            raise ValueError('nonfinite deterministic ELBO')
        return dict(epoch=epoch, train_elbo_per_datum=elbo,
                    negative_train_elbo_per_datum=-elbo, theta_unchanged=True)

    baseline = out / 'epoch150.json'
    if not baseline.exists():
        write_json(baseline, dict(identity=identity, **audit(150, original)))
    elif json.loads(baseline.read_text())['identity'] != identity:
        raise ValueError('incompatible baseline')
    for epoch in (300, 250, 200):
        mark = out / f'epoch{epoch}.json'
        checkpoint = out / f'checkpoint_epoch{epoch}.npz'
        if mark.exists():
            saved = json.loads(mark.read_text())
            if saved['identity'] != identity or sha(checkpoint) != saved['checkpoint_sha256']:
                raise ValueError('checkpoint identity/hash mismatch')
            params, leaves, ckmeta = sv.load_training_checkpoint(checkpoint)
            if int(ckmeta['epoch']) != epoch:
                raise ValueError('checkpoint epoch mismatch')
            start = epoch
            audit(start, params)
            break
    args = build_args(cfg)
    engine = sv.make_engine(x, y, x[:1], y[:1], jitter=runner.JITTER, kernel_name=cfg['kernel'])

    def checkpoint(epoch, current, state):
        data = audit(epoch, current)
        target = out / f'checkpoint_epoch{epoch}.npz'
        temp = out / f'.checkpoint_epoch{epoch}.{os.getpid()}.npz'
        sv.save_training_checkpoint(temp, epoch, current, state,
                                    meta=dict(protocol=PROTOCOL, total_epochs=300))
        os.replace(temp, target)
        write_json(out / f'epoch{epoch}.json', dict(identity=identity, **data,
                   checkpoint_sha256=sha(target)))
        print(f'CHECKPOINT epoch={epoch}/300 train_ELBO/N={data["train_elbo_per_datum"]:+.8f}', flush=True)

    print(f'CONTINUE {row["run"]} epoch={start}->300 theta=fixed Adam=' +
          ('reset-at-150' if start == 150 else 'restored'), flush=True)
    info = {}
    if start < 300:
        params, curve, info, _ = sv.train_svgp(
            engine, 'SVGP', params, runner.make_bank_fn('SVGP', x, args, cfg['seed']),
            engine['metrics'], len(x), args, checkpoint_fn=checkpoint,
            start_epoch=start, resume_optimizer_leaves=leaves)
        # Timing/curve per attempt, so a resumed process does not erase earlier segments.
        runner._atomic_npz(out / f'curve_from_epoch{start}.npz',
                           epoch=np.asarray([e for e, _ in curve]),
                           stoch_obj_n=np.asarray([v for _, v in curve]))
        keys = ('optimizer_s', 'compile_s', 'search_s', 'optimizer_steps', 'steps_per_epoch',
                'start_epoch', 'epochs', 'optimizer_leaves_restored')
        write_json(out / f'attempt_from_epoch{start}.json', {k: info[k] for k in keys})
    runner._atomic_npz(out / 'params.npz', **runner._parameter_arrays(params))
    checkpoints = [json.loads((out / f'epoch{e}.json').read_text()) for e in (150, 200, 250, 300)]
    runner._atomic_npz(out / 'deterministic_curve.npz',
                       epoch=np.asarray([r['epoch'] for r in checkpoints]),
                       train_elbo_per_datum=np.asarray([r['train_elbo_per_datum'] for r in checkpoints]))
    newcfg = dict(cfg, epochs=300, phases=args.phases, training_protocol=PROTOCOL,
                  continuation_start_epoch=150, optimizer_reset_at_epoch=150,
                  parent_config_id=row['config_id'], parent_params_sha256=sha(parent / 'params.npz'))
    write_json(out / 'training.json', dict(
        implementation=sv.IMPLEMENTATION, training_protocol=PROTOCOL,
        config=newcfg, config_id=runner.config_fingerprint(newcfg),
        training_complete=True, epochs=300, prediction_deferred=True,
        continuation_identity=identity, parent_root=str(root),
        optimizer_restart_disclosed=True, convergence_certified=False,
        deterministic_training_diagnostics=checkpoints,
        runtime=runner.runtime_manifest()))
    assert_inputs(root, row)
    if source_hashes() != p['source']:
        raise ValueError('code changed while training')
    files = [f for f in out.iterdir() if not f.name.startswith('.') and f.is_file() and f.name != 'COMPLETE.json']
    write_json(out / 'COMPLETE.json', dict(identity=identity,
               files={f.name: sha(f) for f in files}, release_ready=False))
    print('TRAINED epoch=300/300 params=saved prediction=deferred ' + row['run'], flush=True)


def worker(a):
    p = read_plan(a.output)
    if not 0 <= a.worker < 5:
        raise ValueError('worker index must be 0..4')
    # Check raw inputs once per worker; per-run stat checks catch later modifications.
    for d in p['datasets'].values():
        if sha(d['path']) != d['sha256']:
            raise ValueError('dataset hash mismatch')
    state = a.output / 'state' / f'worker_{a.worker}.json'
    errors, complete, consecutive = [], 0, 0
    indices = list(range(a.worker, len(p['runs']), 5))
    def report(status):
        write_json(state, dict(status=status, worker=a.worker, completed=complete,
                              assigned=len(indices), failed=len(errors), errors=errors))
    with (a.output / 'state' / f'worker_{a.worker}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report('RUNNING')
        for count, index in enumerate(indices, 1):
            print(f'[{count}/{len(indices)}] {p["runs"][index]["run"]}', flush=True)
            result = subprocess.run([sys.executable, '-u', __file__, 'one',
                                     '--output', str(a.output), '--index', str(index)])
            if result.returncode:
                errors.append(dict(index=index, run=p['runs'][index]['run'], exit_code=result.returncode))
                consecutive += 1
            else:
                complete += 1
                consecutive = 0
            report('RUNNING')
            if consecutive >= 3:
                print('Stopped after three consecutive failures; no automatic requeue.', flush=True)
                break
        report('COMPLETED' if complete == len(indices) else 'FAILED')
    raise SystemExit(0 if complete == len(indices) else 1)


def status(a):
    p = json.loads((a.output / 'continuation_plan.json').read_text())
    counts = {e: 0 for e in (200, 250, 300)}
    complete = 0
    for row in p['runs']:
        run = a.output / row['run']
        for epoch in counts:
            counts[epoch] += (run / f'epoch{epoch}.json').is_file()
        complete += (run / 'COMPLETE.json').is_file()
    print('\n===== TRANSPLANT 150 -> 300 | OUTPUT SEPARATO =====')
    for epoch, n in counts.items():
        print(f'epoch {epoch}: [{"#" * (24*n//5120):.<24}] {n}/5120')
    print(f'completi: {complete}/5120; prediction differite; non certifica convergenza')
    for f in sorted((a.output / 'state').glob('worker_*.json')):
        s = json.loads(f.read_text())
        print(f'worker {s["worker"]}: {s["status"]} {s["completed"]}/{s["assigned"]} failed={s["failed"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['plan', 'worker', 'one', 'status'])
    ap.add_argument('--root', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path)
    ap.add_argument('--worker', type=int)
    ap.add_argument('--index', type=int)
    ap.add_argument('--cpu-test', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    a.output = a.output.resolve()
    separate(SIM, a.output)
    if a.mode == 'plan':
        if not a.root or not a.data_dir:
            ap.error('plan requires --root and --data-dir')
        a.root, a.data_dir = a.root.resolve(), a.data_dir.resolve()
        separate(a.root, a.output)
    {'plan': plan, 'worker': worker, 'one': run_one, 'status': status}[a.mode](a)


if __name__ == '__main__':
    main()
