#!/usr/bin/env bash
# Compact counters and activity for the five-worker full paper campaign.

set -uo pipefail

ENV_FILE="${FIVE_WORKER_ENV_FILE:-$HOME/aistats2026_five_v18_jobs.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "no campaign environment at $ENV_FILE; submit first." >&2
  exit 1
fi
source "$ENV_FILE"

SLURM_OUT="$FINAL_ROOT/slurm"
STATE="$FINAL_ROOT/state"

count() { find "$1" -name "$2" 2>/dev/null | wc -l | tr -d ' '; }
bar() {
  local done=$1 total=$2 width=${3:-24} filled
  filled=$(( total > 0 ? done * width / total : 0 ))
  printf '['
  (( filled > 0 )) && printf '%0.s#' $(seq 1 "$filled")
  (( width - filled > 0 )) && printf '%0.s.' $(seq 1 $((width - filled)))
  printf ']'
}
progress() {
  local label=$1 done=$2 total=$3
  printf '  %-18s %s %5s/%-5s\n' "$label" "$(bar "$done" "$total")" "$done" "$total"
}

echo "===== $(basename "$FINAL_ROOT") | $(date '+%H:%M:%S') ====="
echo
echo "----- RESULTS -----"
progress sgpr-main \
  "$(count "$FINAL_ROOT/sgpr/runs" metrics.json)" 280
progress sgpr-params \
  "$(count "$FINAL_ROOT/sgpr/runs" params.npz)" 280
progress sgpr-transplant \
  "$(count "$FINAL_ROOT/sgpr/transplant" metrics.json)" 40
progress svgp-stationary \
  "$(count "$FINAL_ROOT/svgp/stationary/runs" metrics.json)" 2560
progress svgp-extended \
  "$(count "$FINAL_ROOT/svgp/extended/runs" metrics.json)" 2560
progress svgp-transplant \
  "$(count "$FINAL_ROOT/svgp/transplant/runs" metrics.json)" 5120
progress svgp-convergence \
  "$(count "$FINAL_ROOT/svgp/convergence/runs" metrics.json)" 16

echo
echo "----- WORKERS -----"
sacct -j "$CAMPAIGN_JOB" -n -X -o JobID,State,Elapsed,NodeList 2>/dev/null \
  | sed '/^[[:space:]]*$/d' | head -n 8
printf '  completed workers: %s/5\n' \
  "$(find "$STATE" -maxdepth 1 -name 'worker.*.complete' 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "----- QUEUE -----"
squeue -j "$CAMPAIGN_JOB" \
  -o "%.18i %.18j %.9T %.10M %.24R" 2>/dev/null | head -n 8

echo
echo "----- LATEST ACTIVITY -----"
find "$SLURM_OUT" -type f -name '*.out' -print0 2>/dev/null \
  | xargs -0 ls -1t 2>/dev/null | head -n 5 \
  | while read -r file; do
      printf '  %-30s ' "$(basename "$file")"
      grep -E '===== worker=|^\[[0-9]+/[0-9]+\]|COMPLETE|Done\.' "$file" \
        | tail -n 1 || echo "started"
    done

echo
echo "----- FAILURES -----"
failures=$(find "$STATE" -path '*/worker.*.failed' -type f 2>/dev/null \
  | wc -l | tr -d ' ')
printf '  worker failure markers: %s\n' "$failures"
grep -hE 'FAILED CONVERGENCE|run\(s\) failed|Traceback|giving up|out of memory' \
  "$SLURM_OUT"/*.out 2>/dev/null | tail -n 8 | sed 's/^/  /' || true
