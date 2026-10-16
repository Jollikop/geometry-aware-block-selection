#!/usr/bin/env bash
set -u

source "${SGPR_THETA_GPU_ENV:-$HOME/sgpr_fixed_theta_gpu_jobs.env}"

count_metrics() {
  find "$1" -path '*/seed*/metrics.json' -type f 2>/dev/null | wc -l | tr -d ' '
}

count_params() {
  find "$1" -path '*/seed*/params_*.npz' -type f 2>/dev/null | wc -l | tr -d ' '
}

echo "===== FIXED-THETA SGPR ====="
printf 'RBF:   %s/5 seeds, %s/40 parameter files\n' \
  "$(count_metrics "$RBF_OUT")" "$(count_params "$RBF_OUT")"
printf 'Gibbs: %s/5 seeds, %s/40 parameter files\n' \
  "$(count_metrics "$GIBBS_OUT")" "$(count_params "$GIBBS_OUT")"

echo
echo "===== JOBS ====="
squeue -j "$RBF_JOB,$GIBBS_JOB,$RBF_PLOT_JOB,$GIBBS_PLOT_JOB" \
  -o "%.18i %.20j %.9T %.10M %.24R" 2>/dev/null || true

echo
echo "===== CURRENT RUNS ====="
find "$SLURM_LOGS" -maxdepth 1 -type f \
  \( -name 'rbf_*.out' -o -name 'gibbs_*.out' \) -print0 2>/dev/null |
  xargs -0 ls -1t 2>/dev/null | head -n 8 |
while IFS= read -r file; do
  echo "--- $(basename "$file")"
  grep -E '^seed[0-9]+:|^  ->|Traceback|FAILED|Error|wrote ' "$file" |
    tail -n 5
done

echo
echo "===== ERRORS ====="
grep -HniE 'Traceback|FAILED|CUDA_ERROR|DEVICE_UNAVAILABLE|out of memory|non-finite' \
  "$SLURM_LOGS"/rbf_*.out "$SLURM_LOGS"/gibbs_*.out 2>/dev/null |
  tail -n 12 || true

echo
echo "===== FIGURES ====="
find "$RESULT_ROOT" -maxdepth 2 -type f \
  \( -name 'summary.csv' -o -name '*.pdf' -o -name '*.png' \) \
  -printf '%P\n' 2>/dev/null | sort
