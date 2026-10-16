#!/usr/bin/env bash
# One screen for the collapsed SGPR campaign.
#
# Reads only finished artifacts and Slurm state, so it is safe to run under
# watch at any interval. Expected GPU retries are counted rather than printed
# as errors: a requeue is the preflight working, not a failure.
set -uo pipefail

ENV_FILE="${SBT_SGPR_ENV_FILE:-$HOME/aistats2026_sgpr_jobs.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "no campaign environment at $ENV_FILE; submit first." >&2
  exit 1
fi
source "$ENV_FILE"

PYTHON=${AIPY:-python}
SIM_ROOT="$PROJECT_ROOT/Simulations"
RUNS="$FINAL_ROOT/sgpr/runs"
SLURM_OUT="$FINAL_ROOT/slurm"
KERNELS=(rbf gibbs matern52 amplitude-rbf)

count() { find "$1" -name "$2" 2>/dev/null | wc -l | tr -d ' '; }
logs()  { grep -h "$1" "$SLURM_OUT"/*.out 2>/dev/null | wc -l | tr -d ' '; }

bar() {                      # bar <done> <total> <width>
  local done=$1 total=$2 width=${3:-24} filled
  filled=$(( total > 0 ? done * width / total : 0 ))
  printf '['
  (( filled > 0 )) && printf '%0.s#' $(seq 1 "$filled")
  (( width - filled > 0 )) && printf '%0.s.' $(seq 1 $(( width - filled )))
  printf ']'
}

states() {                   # states <jobid> <label> <total>
  local job=$1 label=$2 total=$3 line
  line=$(sacct -j "$job" -n -X -o State 2>/dev/null \
    | awk '{print $1}' | sort | uniq -c \
    | awk '{printf "%s=%s  ", $2, $1}')
  printf '  %-12s %s\n' "$label" "${line:-not started}"
}

echo "===== $(basename "$FINAL_ROOT")  |  $(date '+%H:%M:%S') ====="

echo
echo "----- PROGRESS -----"
main=$(count "$RUNS" metrics.json)
trans=$(count "$FINAL_ROOT/sgpr/transplant" metrics.json)
printf '  main      %s %3s/280\n' "$(bar "$main" 280)" "$main"
for k in "${KERNELS[@]}"; do
  n=$(count "$RUNS/$k" metrics.json)
  printf '    %-14s %s %3s/70\n' "$k" "$(bar "$n" 70 14)" "$n"
done
printf '  params    %s %3s/280\n' "$(bar "$(count "$RUNS" params.npz)" 280)" "$(count "$RUNS" params.npz)"
printf '  transplant %s %3s/40\n' "$(bar "$trans" 40)" "$trans"

echo
echo "----- TASK STATES -----"
states "$SGPR_JOB" main 40
states "$TRANSPLANT_JOB" transplant 40

echo
echo "----- GPU CONTENTION -----"
printf '  probe retries %s   requeues %s   exhausted %s\n' \
  "$(logs 'GPU not usable')" "$(logs 'requeueing')" "$(logs 'giving up')"

echo
echo "----- QUEUE -----"
squeue -j "$SGPR_JOB,$RESIDUAL_JOB,$SNELSON_JOB,$TRANSPLANT_JOB,$FINALIZE_JOB" \
  -o "%.16i %.16j %.9T %.10M %.24R" 2>/dev/null | head -12

echo
echo "----- LATEST ACTIVITY -----"
find "$SLURM_OUT" -type f -name '*.out' -print0 2>/dev/null \
  | xargs -0 ls -1t 2>/dev/null | head -n 3 \
  | while read -r file; do
      printf '  %s\n' "$(basename "$file")"
      grep -E '^  ->|^     step|^     TIMING|COMPLETE|kernel=' "$file" \
        | tail -n 2 | sed 's/^/    /'
    done

echo
echo "----- REAL ERRORS -----"
real=$(grep -hE '^[A-Za-z_][A-Za-z_.]*(Error|Exception|Exit)\b|out of memory|non-finite' \
  "$SLURM_OUT"/*.out 2>/dev/null \
  | grep -viE 'DEVICE_UNAVAILABLE|no supported devices|XlaRuntimeError' \
  | sort | uniq -c | sort -rn | head -n 6)
if [[ -z "$real" ]]; then
  echo "  none"
else
  printf '%s\n' "$real"
fi
echo "  (GPU contention is counted above, not repeated here)"

echo
echo "----- VALIDATOR -----"
PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$SIM_ROOT/analysis/validate_final_suite.py" \
  --root "$FINAL_ROOT" --scope sgpr --brief 2>/dev/null \
  | sed 's/^/  /' || echo "  not ready"
