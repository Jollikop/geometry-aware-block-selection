#!/usr/bin/env bash
set -u
source "${SGPR_THETA_ENV:-$HOME/sgpr_fixed_theta_jobs.env}"

echo "===== FIXED-THETA BOUND EXPERIMENT ====="
echo "seeds:  $(find "$SGPR_THETA_OUT" -path '*/seed*/metrics.json' 2>/dev/null | wc -l | tr -d ' ')/5"
echo "params: $(find "$SGPR_THETA_OUT" -path '*/seed*/params_*.npz' 2>/dev/null | wc -l | tr -d ' ')/40"
echo
squeue -j "$RUN_JOB,$PLOT_JOB" -o "%.18i %.20j %.9T %.10M %.24R" 2>/dev/null || true
echo
find "$SGPR_THETA_SLURM" -maxdepth 1 -type f \
  \( -name 'run_*.out' -o -name 'cpu_*.out' \) -print0 \
  2>/dev/null | xargs -0 ls -1t 2>/dev/null | head -n 5 |
while IFS= read -r file; do
  echo "--- $(basename "$file")"
  grep -E '^seed[0-9]+:|^  ->|Traceback|FAILED|Error|wrote ' "$file" | tail -n 5
done
echo
echo "===== ARTIFACTS ====="
find "$SGPR_THETA_OUT" -maxdepth 1 -type f \
  \( -name 'summary.csv' -o -name '*.pdf' -o -name '*.png' \) \
  -printf '%f\n' 2>/dev/null | sort
