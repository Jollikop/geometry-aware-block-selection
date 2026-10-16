#!/usr/bin/env python3
"""Orchestrate the unchanged SVGP evaluator; input campaign stays read-only."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

DATASETS = 'pol superconductivity kin40k protein elevators traffic electricity 3droad'.split()
METHODS = 'SVGP T-SVGP RBT-SVGP SBT-SVGP'.split()


def targets():
    return [Path(f'svgp/transplant/runs/{kernel}/{dataset}/M{m}/{method}-to-SVGP/seed{seed}')
            for kernel in ('rbf', 'matern52', 'amplitude-rbf', 'gibbs')
            for dataset in DATASETS for m in (64, 128, 256, 512)
            for method in METHODS for seed in range(10)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('root', 'output', 'source-sim', 'evaluator'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--evaluator-sha', required=True)
    p.add_argument('--mode', choices=('active', 'finish'), required=True)
    p.add_argument('--worker', type=int, required=True)
    p.add_argument('--workers', type=int, required=True)
    a = p.parse_args()
    if not 0 <= a.worker < a.workers:
        p.error('invalid shard')
    a.root, a.output = a.root.resolve(), a.output.resolve()
    if a.root == a.output or a.root in a.output.parents or a.output in a.root.parents:
        p.error('output must be disjoint from frozen input')
    if hashlib.sha256(a.evaluator.read_bytes()).hexdigest() != a.evaluator_sha:
        raise SystemExit('Evaluator changed: refusing to invalidate existing prediction identities')
    state = a.output / 'wave_state'
    state.mkdir(exist_ok=True)
    assigned = targets()[a.worker::a.workers]
    counts = dict(evaluated_or_verified=0, already_saved=0, not_yet_trained=0, failed=0)
    errors = []
    consecutive = 0
    for index, relative in enumerate(assigned, 1):
        run = a.root / relative
        result_path = a.output / relative / 'prediction.json'
        if a.mode == 'active' and result_path.is_file():
            # Finish jobs verify every cache entry through the original evaluator.
            counts['already_saved'] += 1
            continue
        try:
            record_path = run / 'training.json'
            if not record_path.is_file():
                record_path = run / 'metrics.json'
            record = json.loads(record_path.read_text())
            ready = (record.get('training_complete') is True or 'rmse' in record) \
                and (run / 'params.npz').is_file() and (run / 'curve.npz').is_file()
        except (OSError, ValueError):
            ready = False
        if not ready:
            counts['not_yet_trained'] += 1
            if a.mode == 'finish':
                errors.append(f'MISSING {relative}')
            continue
        print(f'[{index}/{len(assigned)}] mode={a.mode} worker={a.worker} {relative}', flush=True)
        lock_path = state / (hashlib.sha256(str(relative).encode()).hexdigest() + '.lock')
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                counts['failed'] += 1
                errors.append(f'LOCKED {relative}')
                continue
            result = subprocess.run([
                sys.executable, str(a.evaluator), '--root', str(a.root), '--output', str(a.output),
                '--source-sim', str(a.source_sim), '--stage', 'transplant',
                '--measurement-context', 'concurrent-training', '--require-single-gpu',
                '--reps', '10', '--reference-points', '8', '--run', str(relative)])
        if result.returncode:
            counts['failed'] += 1
            consecutive += 1
            errors.append(f'FAILED {relative}')
            if consecutive >= 3:
                errors.append('STOPPED after three consecutive failures; unfinished targets remain')
                break
        else:
            counts['evaluated_or_verified'] += 1
            consecutive = 0
    report = dict(mode=a.mode, worker=a.worker, workers=a.workers, assigned=len(assigned),
                  counts=counts, errors=errors, release_ready=False,
                  note='Orchestration report only; predictions stay outside the training freeze')
    (state / f'{a.mode}_{a.worker}_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
