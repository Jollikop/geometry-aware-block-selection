#!/usr/bin/env bash
set -uo pipefail

source "$HOME/aistats2026_final_jobs.env"
PYTHON=${AIPY:-python}
SIM_ROOT="$PROJECT_ROOT/Simulations"

echo "===== FINAL SUITE ====="
PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$SIM_ROOT/analysis/validate_final_suite.py" \
  --root "$FINAL_ROOT" --brief 2>/dev/null || true

echo
echo "===== ARTIFACTS ====="
printf 'metrics: %s/10576\n' "$(find "$FINAL_ROOT" -name metrics.json 2>/dev/null | wc -l | tr -d ' ')"
printf 'params:  %s/10536\n' "$(find "$FINAL_ROOT" -name params.npz 2>/dev/null | wc -l | tr -d ' ')"
printf 'blocks:  %s/2688\n' "$(find "$FINAL_ROOT" -name final_blocks.npz 2>/dev/null | wc -l | tr -d ' ')"
printf 'history: %s/5136\n' "$(find "$FINAL_ROOT" -name history.csv 2>/dev/null | wc -l | tr -d ' ')"

echo
echo "===== JOBS ====="
IDS="$SGPR_JOB,$RESIDUAL_JOB,$SNELSON_JOB,$SGPR_TRANSPLANT_JOB,$STATIONARY_JOB,$EXTENDED_JOB,$TRANSPLANT_JOB,$CONVERGENCE_JOB,$FINALIZE_JOB"
squeue -j "$IDS" -o "%.18i %.20j %.9T %.10M %.28R" 2>/dev/null || true

echo
echo "===== LATEST RUNS ====="
find "$FINAL_ROOT/slurm" -type f -name '*.out' -print0 2>/dev/null \
  | xargs -0 ls -1t 2>/dev/null | head -n 5 \
  | while read -r file; do
      echo "--- $(basename "$file")"
      grep -E '^  ->|^\[[0-9]+/[0-9]+\]|COMPLETE|FAILED|Traceback|FINAL SUITE' \
        "$file" | tail -n 3
    done

echo
echo "===== NEW ERRORS ====="
grep -HniE 'FAILED|Traceback|CUDA_ERROR|DEVICE_UNAVAILABLE|out of memory|non-finite' \
  "$FINAL_ROOT"/slurm/*.out 2>/dev/null | tail -n 8 || true
