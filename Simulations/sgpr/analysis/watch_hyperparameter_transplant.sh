#!/usr/bin/env bash
set -u
source "${SGPR_TRANSPLANT_ENV:-$HOME/sgpr_hyperparameter_transplant_jobs.env}"

count() {
  find "$1" -path '*/seed*/metrics.json' -type f 2>/dev/null | wc -l | tr -d ' '
}

echo "===== HYPERPARAMETER -> SGPR TRANSPLANT ====="
echo "RBF:   $(count "$RBF_OUT")/5"
if [[ "${GIBBS_ENABLED:-1}" == 1 ]]; then
  echo "Gibbs: $(count "$GIBBS_OUT")/5"
fi

echo
echo "===== JOBS ====="
job_ids=("$RBF_JOB" "$RBF_PLOT_JOB")
[[ -n "${RBF_RETRY_JOB:-}" ]] && job_ids+=("$RBF_RETRY_JOB")
if [[ "${GIBBS_ENABLED:-1}" == 1 ]]; then
  job_ids+=("$GIBBS_JOB" "$GIBBS_PLOT_JOB")
fi
job_csv=$(IFS=,; echo "${job_ids[*]}")
squeue -j "$job_csv" \
  -o "%.18i %.22j %.9T %.10M %.24R" 2>/dev/null || true

echo
echo "===== CURRENT RUNS ====="
if [[ "${GIBBS_ENABLED:-1}" == 1 ]]; then
  log_names=(\( -name 'rbf_*.out' -o -name 'gibbs_*.out' \))
else
  log_names=(\( -name 'rbf_*.out' \))
fi
find "$SLURM_LOGS" -maxdepth 1 -type f "${log_names[@]}" -print0 2>/dev/null |
  xargs -0 ls -1t 2>/dev/null | head -n 8 |
while IFS= read -r file; do
  echo "--- $(basename "$file")"
  grep -E '^seed[0-9]+:|^  ->|Traceback|FAILED|Error|wrote ' "$file" |
    tail -n 5
done

echo
echo "===== ERRORS ====="
error_logs=("$SLURM_LOGS"/rbf_*.out)
if [[ "${GIBBS_ENABLED:-1}" == 1 ]]; then
  error_logs+=("$SLURM_LOGS"/gibbs_*.out)
fi
grep -HniE 'Traceback|FAILED|CUDA_ERROR|DEVICE_UNAVAILABLE|out of memory|non-finite' \
  "${error_logs[@]}" 2>/dev/null |
  tail -n 12 || true

echo
echo "===== TABLES / FIGURES ====="
find "$RESULT_ROOT" -maxdepth 2 -type f \
  \( -name 'summary.csv' -o -name '*.pdf' -o -name '*.png' \) \
  -printf '%P\n' 2>/dev/null | sort
