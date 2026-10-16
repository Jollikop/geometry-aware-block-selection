#!/usr/bin/env python3
"""Plan and evaluate missing donor predictions from immutable saved training.

This coordinator never invokes a training entry point. Existing legacy donor
metrics are reused only with matching configuration and predictor. Each pending
run invokes the separately hashed prepared evaluator in a new process.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

try:
    from .validate_svgp_deferred_freeze import expected_paths, load_record, record_errors, sha256, valid_npz
except ImportError:
    from validate_svgp_deferred_freeze import expected_paths, load_record, record_errors, sha256, valid_npz

DATASETS = 'pol superconductivity kin40k protein elevators traffic electricity 3droad'.split()
PREDICTOR = {'SVGP': 'sparse', 'T-SVGP': 'diagonal', 'RBT-SVGP': 'block', 'SBT-SVGP': 'block'}


def finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def predictive_metrics_valid(metrics, record, predictor, head):
    return (isinstance(metrics, dict) and not record_errors(metrics, head)
            and metrics.get('config_id') == record['config_id']
            and metrics.get('reported_predictor') == predictor
            and all(finite(metrics.get(k)) for k in ('rmse', 'nll')))


def inspect_run(path, method, head, identifiers):
    record_path = path / 'training.json'
    if not record_path.is_file():
        record_path = path / 'metrics.json'
    record = load_record(record_path)
    errors = record_errors(record, head)
    if errors:
        raise ValueError(f'{path}: {errors}')
    cfg = record['config']
    for key, value in identifiers.items():
        if cfg.get(key) != value:
            raise ValueError(f'{path}: path/config mismatch {key}')
    if cfg.get('training_method', method) != method:
        raise ValueError(f'{path}: unexpected training method')
    required = [record_path, path / 'params.npz', path / 'curve.npz']
    if PREDICTOR[method] == 'block':
        required.append(path / 'final_blocks.npz')
    if not all(valid_npz(p) for p in required[1:]):
        raise ValueError(f'{path}: missing/invalid saved training arrays')
    metrics = load_record(path / 'metrics.json')
    cached = predictive_metrics_valid(metrics, record, PREDICTOR[method], head)
    if cached and path / 'metrics.json' not in required:
        required.append(path / 'metrics.json')
    return cached, required, record['config_id']


def atomic_json(path, payload):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def plan(a):
    pending, preserved, counts = [], [], collections.defaultdict(collections.Counter)
    head = (a.root / 'EXPECTED_HEAD').read_text().strip()
    for stage, kernels in (('stationary', ['rbf', 'matern52']), ('extended', ['amplitude-rbf', 'gibbs'])):
        for kernel, dataset, m, method, seed, path in expected_paths(a.root, stage, kernels, DATASETS):
            cached, files, config_id = inspect_run(path, method, head, {
                'kernel': kernel, 'dataset': dataset, 'M': m, 'method': method, 'seed': seed})
            row = dict(run=str(path.relative_to(a.root)), stage=stage, method=method,
                       config_id=config_id, inputs={str(p.relative_to(a.root)): sha256(p) for p in files})
            (preserved if cached else pending).append(row)
            counts[stage]['preserved' if cached else 'pending'] += 1
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'slurm').mkdir()
    (a.output / 'state').mkdir()
    record = dict(root=str(a.root), source_sim=str(a.source_sim), evaluator=str(a.evaluator),
                  evaluator_sha256=sha256(a.evaluator),
                  backend_sha256=sha256(a.evaluator.parents[1] / 'svgp/fast_prediction.py'),
                  expected_head_sha256=sha256(a.root / 'EXPECTED_HEAD'),
                  workers=a.workers, counts=counts, pending=pending, preserved=preserved,
                  scope='stationary + extended, all eight datasets including 3droad',
                  release_ready=False)
    atomic_json(a.output / 'prediction_plan.json', record)
    print(json.dumps(dict(output=str(a.output), workers=a.workers, counts=counts, pending=len(pending)), indent=2))


def worker(a):
    plan_path = a.output / 'prediction_plan.json'
    plan_hash = sha256(plan_path)
    data = json.loads(plan_path.read_text())
    if not 0 <= a.worker < data['workers']:
        raise SystemExit('invalid worker index')
    root, evaluator = Path(data['root']), Path(data['evaluator'])
    if sha256(evaluator) != data['evaluator_sha256'] or sha256(evaluator.parents[1] / 'svgp/fast_prediction.py') != data['backend_sha256']:
        raise SystemExit('Evaluator/backend changed after planning')
    if sha256(root / 'EXPECTED_HEAD') != data['expected_head_sha256']:
        raise SystemExit('Training numerical reference changed')
    assigned = data['pending'][a.worker::data['workers']]
    success, errors, consecutive = [], [], 0
    state_path = a.output / 'state' / f'worker_{a.worker}.json'
    def report(status):
        atomic_json(state_path, dict(status=status, worker=a.worker, assigned=len(assigned),
                                    completed=len(success), failed=len(errors), errors=errors,
                                    plan_sha256=plan_hash, release_ready=False))
    # Prevent two executions of the same shard from writing the same outputs.
    with (a.output / 'state' / f'worker_{a.worker}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report('RUNNING')
        for index, row in enumerate(assigned, 1):
            print(f"[{index}/{len(assigned)}] {row['run']} predictor={PREDICTOR[row['method']]}", flush=True)
            changed = [p for p, h in row['inputs'].items() if not (root / p).is_file() or sha256(root / p) != h]
            if changed:
                errors.append(dict(run=row['run'], reason='input changed since planning', files=changed))
                report('FAILED')
                raise SystemExit(2)
            result = subprocess.run([sys.executable, str(evaluator), '--root', str(root),
                '--output', str(a.output), '--source-sim', data['source_sim'],
                '--stage', row['stage'], '--require-single-gpu', '--test-chunk', '256',
                '--train-chunk', '512', '--block-batch', '4', '--factor-cache-mib', '128',
                '--reps', '10', '--reference-points', '8', '--run', row['run']])
            if result.returncode:
                errors.append(dict(run=row['run'], exit_code=result.returncode))
                consecutive += 1
            else:
                success.append(row['run'])
                consecutive = 0
            report('RUNNING')
            if consecutive >= 3:
                print('Stopped after three consecutive failures; completed predictions are preserved.', flush=True)
                break
        report('COMPLETED' if len(success) == len(assigned) else 'FAILED')
        raise SystemExit(0 if len(success) == len(assigned) else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['plan', 'worker'])
    p.add_argument('--root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-sim', type=Path)
    p.add_argument('--evaluator', type=Path)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--worker', type=int)
    a = p.parse_args()
    a.output = a.output.resolve()
    if a.mode == 'plan':
        if not all((a.root, a.source_sim, a.evaluator)) or not 1 <= a.workers <= 5:
            p.error('plan needs root/source-sim/evaluator and 1..5 workers')
        a.root, a.source_sim, a.evaluator = a.root.resolve(), a.source_sim.resolve(), a.evaluator.resolve()
        if any(a.output == path or path in a.output.parents or a.output in path.parents
               for path in (a.root, a.source_sim, a.evaluator.parent.parent)):
            p.error('output must be separate from training/source/evaluator')
        plan(a)
    else:
        if a.worker is None:
            p.error('worker requires --worker')
        worker(a)


if __name__ == '__main__':
    main()
