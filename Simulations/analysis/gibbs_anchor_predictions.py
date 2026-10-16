#!/usr/bin/env python3
"""Evaluate and certify the isolated 320-run Gibbs inducing-anchor campaign.

The historical training tree is read-only. A snapshot of the historical
prediction source plus the prepared backend is placed in the evaluation tree.
Certification covers provenance, complete coverage and numerical reference
checks; it makes no claim of optimizer convergence or controlled GPU timings.
"""
import argparse
import collections
import csv
import fcntl
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from evaluate_svgp_frozen import sha, write_json, separate_output

DATASETS = 'pol superconductivity kin40k protein elevators traffic electricity 3droad'.split()
SETTINGS = dict(test_chunk=256, train_chunk=512, block_batch=4,
                factor_cache_mib=128, reps=10, reference_points=8)


def read(path):
    return json.loads(Path(path).read_text())


def training_tool(root):
    spec = importlib.util.spec_from_file_location('anchor_training_audit', root / 'tools/gibbs_anchor_campaign.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit_training(root):
    tool = training_tool(root)
    plan = tool.load_plan(root)
    tool.verify_inputs(root, plan)
    expected = set(itertools.product(DATASETS, (64, 128, 256, 512), range(10)))
    actual = [(t['config']['dataset'], t['config']['M'], t['config']['seed']) for t in plan['tasks']]
    if len(actual) != 320 or set(actual) != expected:
        raise ValueError('training grid is not exactly the expected 320 runs')
    files = {'campaign.json': sha(root / 'campaign.json'), 'EXPECTED_HEAD': sha(root / 'EXPECTED_HEAD')}
    files.update(plan['inputs'])
    for task in plan['tasks']:
        cfg = task['config']
        if any(cfg.get(k) != v for k, v in dict(kernel='gibbs', method='SBT-SVGP',
                sbt_geometry='anchor', epochs=300, init_from_kernel='rbf').items()):
            raise ValueError('unexpected anchor/warm-start protocol')
        if sha(root / task['warm']) != cfg['init_params_sha256']:
            raise ValueError('RBF initializer hash mismatch')
        state = tool.validate_task(root, task, deep=True)
        if state != 'valid':
            raise ValueError(f"{task['run']}: {state}")
        for name in tool.ARTIFACTS:
            relative = str(Path(task['run']) / name)
            files[relative] = sha(root / relative)
    return plan, files


def prepare(a):
    root, output = a.root.resolve(), a.output.resolve()
    separate_output(root, output)
    if output.exists():
        raise ValueError('choose a new evaluation output directory')
    training, inputs = audit_training(root)
    output.mkdir(parents=True)
    for name in ('tools', 'slurm', 'state', 'dispatch'):
        (output / name).mkdir()
    source = output / 'source/Simulations'
    for module in ('common', 'svgp'):
        target = source / module
        target.mkdir(parents=True)
        for path in (Path(training['source']) / 'Simulations' / module).glob('*.py'):
            shutil.copyfile(path, target / path.name)
    # All historical modules must be byte-identical before adding the backend.
    for rel, digest in training['source_modules'].items():
        if sha(output / 'source' / rel) != digest:
            raise ValueError(f'historical source copy mismatch: {rel}')
    shutil.copyfile(a.backend, source / 'svgp/fast_prediction.py')
    here = Path(__file__).resolve().parent
    for name in ('gibbs_anchor_predictions.py', 'evaluate_svgp_frozen.py'):
        shutil.copyfile(here / name, output / 'tools' / name)
    source_files = {str(p.relative_to(source)): sha(p) for p in source.rglob('*.py')}
    tools = {str(p.relative_to(output)): sha(p) for p in (output / 'tools').glob('*.py')}
    plan = dict(root=str(root), source_sim=str(source), python=training['python'],
                dataset_dir=training['dataset_dir'], workers=5, expected=320,
                tasks=training['tasks'], training_inputs=inputs, source_files=source_files,
                tools=tools, settings=SETTINGS, training_audited=True,
                scope='Gibbs SBT-SVGP inducing-anchor full block predictor; RBF warm start + 300 epochs')
    if a.previous:
        previous = a.previous.resolve()
        separate_output(previous, output)
        old = load(previous)
        if old['training_inputs'] != inputs or old['root'] != str(root):
            raise ValueError('recovery must use exactly the same training artifacts')
        plan['previous_plan'] = old
        plan['previous_output'] = str(previous)
        plan['previous_plan_sha256'] = sha(previous / 'prediction_plan.json')
        plan['preserved'] = {}
        for task in plan['tasks']:
            path = previous / task['run'] / 'prediction.json'
            if path.is_file():
                check_prediction(previous, old, task)
                destination = output / task['run'] / 'prediction.json'
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                plan['preserved'][task['run']] = sha(path)
                if sha(destination) != sha(path):
                    raise ValueError('preserved prediction copy mismatch')
        plan['reference_policy'] = 'frozen-blockwise-reference-v2'
    write_json(output / 'prediction_plan.json', plan)
    (output / 'prediction_plan.sha256').write_text(sha(output / 'prediction_plan.json') + '\n')
    print(f'PREPARED: audited 320 training runs; output={output}', flush=True)


def load(output):
    if sha(output / 'prediction_plan.json') != (output / 'prediction_plan.sha256').read_text().strip():
        raise ValueError('evaluation plan changed')
    plan = read(output / 'prediction_plan.json')
    for name, digest in plan['tools'].items():
        if sha(output / name) != digest:
            raise ValueError(f'evaluation tool changed: {name}')
    for name, digest in plan['source_files'].items():
        if sha(Path(plan['source_sim']) / name) != digest:
            raise ValueError(f'prediction source changed: {name}')
    return plan


def check_prediction(output, plan, task):
    relative = task['run']
    if relative in plan.get('preserved', {}):
        if sha(output / relative / 'prediction.json') != plan['preserved'][relative]:
            raise ValueError('preserved prediction changed')
        plan = plan['previous_plan']
    record = read(output / relative / 'prediction.json')
    identity = record['identity']
    root = Path(plan['root'])
    training = read(root / relative / 'training.json')
    if record['training_config'] != task['config'] or identity['config_id'] != training['config_id']:
        raise ValueError('prediction/training config mismatch')
    expected_inputs = {str(Path(relative) / n): plan['training_inputs'][str(Path(relative) / n)]
                       for n in ('training.json', 'params.npz', 'curve.npz', 'final_blocks.npz')}
    expected_inputs['EXPECTED_HEAD'] = plan['training_inputs']['EXPECTED_HEAD']
    expected_sources = {k: v for k, v in plan['source_files'].items()
                        if k.startswith('common/') or k in ('svgp/engine.py', 'svgp/run_benchmark.py')}
    required = dict(predictor='block', forced_predictor=None, inputs=expected_inputs,
                    source_modules=expected_sources,
                    backend_sha256=plan['source_files']['svgp/fast_prediction.py'],
                    evaluator_sha256=plan['tools']['tools/evaluate_svgp_frozen.py'],
                    reference_policy=plan.get('reference_policy', 'frozen-full-training-reference-v1'),
                    **plan['settings'])
    if any(identity.get(k) != v for k, v in required.items()):
        raise ValueError('prediction identity/settings mismatch')
    if not identity['gpu_kinds'] or any('A100' not in kind for kind in identity['gpu_kinds']):
        raise ValueError('expected A100 evaluation')
    if record['reference_check_points'] != min(8, task['config']['n_test']):
        raise ValueError('missing frozen predictor reference checks')
    for key in ('rmse', 'nll', 'rmse_orig', 'nll_orig', 'warm_prediction_ms',
                'prepare_s', 'compile_and_first_prediction_s',
                'reference_mean_max_abs', 'reference_variance_max_abs'):
        if not math.isfinite(record[key]):
            raise ValueError(f'nonfinite {key}')
    if record['rmse'] < 0 or record['warm_prediction_ms'] < 0:
        raise ValueError('negative RMSE or timing')
    return record


def certify(output, only_if_ready=False):
    with (output / 'state/certification.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if only_if_ready and not all((output / f'state/worker{i}.json').is_file()
                and read(output / f'state/worker{i}.json').get('status') == 'COMPLETED' for i in range(5)):
            return
        plan = load(output)
        root = Path(plan['root'])
        _, current = audit_training(root)
        if current != plan['training_inputs']:
            raise ValueError('training artifacts changed since prediction planning')
        predictions, rows, errors = {}, [], []
        for task in plan['tasks']:
            try:
                record = check_prediction(output, plan, task)
                relative = str(Path(task['run']) / 'prediction.json')
                predictions[relative] = sha(output / relative)
                rows.append((task['config'], record))
            except (ValueError, KeyError, OSError, TypeError) as exc:
                errors.append(dict(run=task['run'], error=str(exc)))
        if errors:
            write_json(output / 'CERTIFICATION_FAILED.json', dict(release_certified=False, errors=errors))
            raise ValueError(f'certification failed: {len(errors)} predictions')
        groups = collections.defaultdict(list)
        for cfg, row in rows:
            for metric in ('rmse', 'nll'):
                groups[cfg['kernel'], cfg['dataset'], cfg['M'], cfg['method'], metric].append(row[metric])
        aggregate = output / 'gibbs_anchor_aggregate.csv'
        with aggregate.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['kernel', 'dataset', 'M', 'method', 'metric', 'n', 'mean', 'sem'])
            for key, values in sorted(groups.items()):
                if len(values) != 10:
                    raise ValueError('aggregate requires exactly ten seeds')
                writer.writerow([*key, 10, statistics.mean(values), statistics.stdev(values) / math.sqrt(10)])
        write_json(output / 'GIBBS_ANCHOR_PREDICTION_CERTIFICATE.json', dict(
            release_certified=True, scope=plan['scope'], complete_grid=True,
            training_params=320, predictions=320, training_unchanged=True,
            predictor='block', reference_check='eight test points per run; mean/variance allclose atol=2e-8 rtol=2e-7',
            prediction_plan_sha256=sha(output / 'prediction_plan.json'), training_inputs=current,
            preserved_predictions=plan.get('preserved', {}),
            previous_plan_sha256=plan.get('previous_plan_sha256'),
            prediction_files=predictions, outputs={aggregate.name: sha(aggregate)},
            gpu_kinds=sorted({k for _, r in rows for k in r['identity']['gpu_kinds']}),
            optimizer_convergence_certified=False, controlled_timing_certified=False,
            note='Certification of this 320-run anchor prediction campaign only; no transplant or full-paper certification.'))
        print('CERTIFIED: 320/320 Gibbs-anchor predictions; training unchanged', flush=True)


def worker(a):
    output = a.output.resolve()
    plan = load(output)
    if a.worker not in range(5):
        raise ValueError('worker index must be 0..4')
    assigned = plan['tasks'][a.worker::5]
    state = dict(status='RUNNING', assigned=len(assigned), completed=0, failed=0)
    def save():
        write_json(output / f'state/worker{a.worker}.json', state)
    with (output / f'state/worker{a.worker}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        try:
            probe = subprocess.run([plan['python'], '-c',
                'import jax,jax.numpy as j; d=jax.devices(); print(d,flush=True); '
                'assert len(d)==1 and d[0].platform=="gpu" and "A100" in d[0].device_kind; '
                'jax.block_until_ready(j.ones(8)+1)'])
            if probe.returncode:
                raise RuntimeError('assigned A100 failed CUDA preflight')
            consecutive = 0
            for i, task in enumerate(assigned, 1):
                print(f"[{i}/{len(assigned)}] {task['run']}", flush=True)
                state['current'] = task['run']; save()
                if task['run'] in plan.get('preserved', {}):
                    check_prediction(output, plan, task)
                    state['completed'] += 1; save()
                    print('PRESERVED ' + task['run'], flush=True)
                    continue
                for name in ('training.json', 'params.npz', 'curve.npz', 'final_blocks.npz'):
                    rel = str(Path(task['run']) / name)
                    if sha(Path(plan['root']) / rel) != plan['training_inputs'][rel]:
                        raise ValueError(f'training artifact changed: {rel}')
                cmd = [plan['python'], '-u', str(output / 'tools/evaluate_svgp_frozen.py'),
                       '--root', plan['root'], '--output', str(output), '--source-sim', plan['source_sim'],
                       '--stage', 'extended', '--require-single-gpu', '--run', task['run']]
                for k, v in plan['settings'].items():
                    cmd += ['--' + k.replace('_', '-'), str(v)]
                if plan.get('reference_policy') == 'frozen-blockwise-reference-v2':
                    cmd += ['--block-reference-streaming']
                result = subprocess.run(cmd)
                if result.returncode:
                    state['failed'] += 1; consecutive += 1
                else:
                    check_prediction(output, plan, task)
                    state['completed'] += 1; consecutive = 0
                save()
                if consecutive >= 3:
                    raise RuntimeError('three consecutive failures; no automatic requeue')
            if state['failed']:
                raise RuntimeError('some predictions failed')
            state['status'] = 'COMPLETED'; save()
            certify(output, only_if_ready=True)
        except Exception as exc:
            state.update(status='FAILED', error=str(exc)); save()
            raise


def status(output):
    plan = read(output / 'prediction_plan.json')
    count = sum((output / t['run'] / 'prediction.json').is_file() for t in plan['tasks'])
    print('GIBBS ANCHOR | RBF warm-start + 300 epochs | full block predictor')
    print(f'Training audited at launch: 320/320 | Prediction saved: {count}/320')
    if plan.get('preserved'):
        print(f"Preserved: {len(plan['preserved'])}/320 | New: {count-len(plan['preserved'])}/{320-len(plan['preserved'])}")
    cert = output / 'GIBBS_ANCHOR_PREDICTION_CERTIFICATE.json'
    print('Certification: ' + ('COMPLETE' if cert.is_file() and read(cert).get('release_certified') else 'pending'))
    for i in range(5):
        path = output / f'state/worker{i}.json'
        row = read(path) if path.exists() else {}
        print(f"worker {i}: {row.get('status', 'WAITING')} {row.get('completed', 0)}/64 failed={row.get('failed', 0)}")
        if row.get('error'): print('  ' + row['error'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'worker', 'certify', 'status'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root', type=Path)
    p.add_argument('--backend', type=Path)
    p.add_argument('--previous', type=Path, help='preserve validated predictions from a previous evaluation')
    p.add_argument('--worker', type=int)
    a = p.parse_args()
    if a.mode == 'prepare': prepare(a)
    elif a.mode == 'worker': worker(a)
    elif a.mode == 'certify': certify(a.output.resolve())
    else: status(a.output.resolve())


if __name__ == '__main__':
    main()
