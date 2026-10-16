#!/usr/bin/env bash
set -euo pipefail
: "${ABLATION_ROOT:?set the output root}"

printf '===== RBF–Pol M128 | block-diagonal predictor =====\n'
printf 'root: %s\n' "$ABLATION_ROOT"
printf 'snapshot manifests: '
find "$ABLATION_ROOT/runs" -name snapshot_manifest.json 2>/dev/null | wc -l
printf 'epochwise predictions: '
find "$ABLATION_ROOT/runs" -path '*/block_diag/epoch*.json' 2>/dev/null | wc -l
if [[ -f "$ABLATION_ROOT/BLOCK_DIAG_ABLATION_CERTIFICATE.json" ]]; then
  printf 'CERTIFIED=yes\n'
else
  printf 'CERTIFIED=no (target: 50 training histories, 900 predictions)\n'
fi
printf '\n----- WORKERS -----\n'
for index in 0 1 2 3 4; do
  state="$ABLATION_ROOT/state/worker${index}.json"
  if [[ -f "$state" ]]; then
    "$AIPY" -c 'import json,sys; p=json.load(open(sys.argv[1])); print("{}: {} {} seeds={}/10 failures={}".format(sys.argv[2],p["method"],p["coupling"],p["processed_seeds"],len(p["failures"])))' "$state" "$index"
  else
    printf '%s: not started\n' "$index"
  fi
done
if [[ -n "${ABLATION_JOB:-}" ]]; then
  printf '\n----- SLURM -----\n'
  squeue -j "$ABLATION_JOB" -o '%.18i %.18j %.9T %.10M %.20R' || true
  printf '\n----- ACCOUNTING -----\n'
  sacct -j "$ABLATION_JOB" -X -o JobID,State,Elapsed,ExitCode || true
fi
printf '\n----- LATEST ACTIVITY -----\n'
for index in 0 1 2 3 4; do
  latest=$(find "$ABLATION_ROOT/slurm" -name "pol_blockdiag_*_${index}.out" -print 2>/dev/null | sort | tail -n 1)
  if [[ -n "$latest" ]]; then
    printf '%s\n' "$(basename "$latest")"
    tail -n 3 "$latest"
  fi
done
