#!/usr/bin/env bash
set -u

ENV_FILE="${SGPR_GUIDED_ENV:-$HOME/sgpr_sbt_guided_jobs.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  exit 1
fi
source "$ENV_FILE"

echo "===== SBT-GUIDED SGPR ====="
echo "seeds:  $(find "$SGPR_GUIDED_OUT" -path '*/seed*/metrics.json' 2>/dev/null | wc -l | tr -d ' ')/5"
echo "params: $(find "$SGPR_GUIDED_OUT" -path '*/seed*/params_*.npz' 2>/dev/null | wc -l | tr -d ' ')/25"

echo
echo "===== JOBS ====="
squeue -j "$RUN_JOB,$PLOT_JOB" \
  -o "%.18i %.20j %.9T %.10M %.24R" 2>/dev/null || true

echo
echo "===== CURRENT RUNS ====="
find "$SGPR_GUIDED_SLURM" -maxdepth 1 -type f -name 'run_*.out' -print0 \
  2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -n 5 |
while IFS= read -r file; do
  echo "--- $(basename "$file")"
  grep -E '^seed[0-9]+:|Traceback|FAILED|Error|wrote ' "$file" | tail -n 4
done

echo
echo "===== ARTIFACTS ====="
find "$SGPR_GUIDED_OUT" -maxdepth 1 -type f \
  \( -name 'summary.csv' -o -name '*.pdf' -o -name '*.png' \) \
  -printf '%f\n' 2>/dev/null | sort
