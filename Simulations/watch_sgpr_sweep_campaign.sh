#!/usr/bin/env bash

set -u

ENV_FILE="${SGPR_SWEEP_ENV_FILE:-$HOME/aistats2026_sgpr_sweep_v22_jobs.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE" >&2
  exit 2
fi
source "$ENV_FILE"

count() {
  local root=$1 pattern=$2
  find "$root" -type f -name "$pattern" 2>/dev/null | wc -l | tr -d ' '
}

bar() {
  local done=$1 total=$2 width=24 filled empty
  filled=$(( done * width / total ))
  empty=$(( width - filled ))
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '.'
  printf '] %4d/%-4d' "$done" "$total"
}

echo "===== $(basename "$SWEEP_ROOT") | $(date +%H:%M:%S) ====="
echo
echo "----- SGPR SWEEP RESULTS -----"
for M in 128 256 512; do
  main=$(find "$SWEEP_ROOT/runs" -path "*/M$M/*/metrics.json" 2>/dev/null | wc -l | tr -d ' ')
  params=$(find "$SWEEP_ROOT/runs" -path "*/M$M/*/params.npz" 2>/dev/null | wc -l | tr -d ' ')
  transplant=$(count "$SWEEP_ROOT/transplant/M$M" metrics.json)
  printf '  M=%-3s main       ' "$M"; bar "$main" 280; echo
  printf '        params     '; bar "$params" 280; echo
  printf '        transplant '; bar "$transplant" 40; echo
done

echo
echo "----- PRESERVED SVGP (UNCHANGED ROOT) -----"
printf '  stationary '; bar "$(count "$PRESERVED_CAMPAIGN_ROOT/svgp/stationary/runs" metrics.json)" 2560; echo
printf '  extended   '; bar "$(count "$PRESERVED_CAMPAIGN_ROOT/svgp/extended/runs" metrics.json)" 2560; echo
printf '  transplant '; bar "$(count "$PRESERVED_CAMPAIGN_ROOT/svgp/transplant/runs" metrics.json)" 5120; echo
printf '  convergence'; bar "$(count "$PRESERVED_CAMPAIGN_ROOT/svgp/convergence/runs" metrics.json)" 16; echo

echo
echo "----- SWEEP WORKERS -----"
sacct -j "$SWEEP_JOB" -X -n -o JobID,State,Elapsed,NodeList 2>/dev/null |
  sed '/^[[:space:]]*$/d'

echo
echo "----- QUEUE -----"
squeue -j "$SWEEP_JOB" -o "%.18i %.18j %.9T %.10M %.22R" 2>/dev/null

echo
echo "----- LATEST ACTIVITY -----"
find "$SWEEP_ROOT/slurm" -type f -name '*.out' -print0 2>/dev/null |
  xargs -0 ls -1t 2>/dev/null |
  head -n 5 |
  while read -r log; do
    printf '  %-34s ' "$(basename "$log")"
    tail -n 80 "$log" |
      grep -E 'COMPLETE|FAILED CONVERGENCE|CONVERGENCE|rmse=' |
      tail -n 1
  done

echo
echo "----- FAILURES -----"
failures=$(count "$SWEEP_ROOT/state/generation-$SWEEP_JOB" '*.failed')
echo "  worker failure markers: $failures"
grep -hE 'FAILED CONVERGENCE|Traceback|Out of memory|no usable GPU|giving up' \
  "$SWEEP_ROOT"/slurm/sgpr_sweep_"$SWEEP_JOB"_*.out 2>/dev/null | tail -n 12

echo
echo "----- VALIDATOR -----"
PYTHONPATH="$PROJECT_ROOT/Simulations" \
  "${AIPY:-python}" "$PROJECT_ROOT/Simulations/analysis/validate_final_suite.py" \
  --root "$SWEEP_ROOT" --scope sgpr-sweep --brief 2>&1 || true

echo
echo "To change concurrency without resubmitting (N must be 1..5):"
echo "  scontrol update JobId=$SWEEP_JOB ArrayTaskThrottle=N"
