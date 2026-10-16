#!/usr/bin/env bash

set -u

ENV_FILE="${SVGP_RESUME_ENV_FILE:-$HOME/aistats2026_svgp_resume_jobs.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing $ENV_FILE" >&2
  exit 2
fi
source "$ENV_FILE"

count() { find "$1" -type f -name "$2" 2>/dev/null | wc -l | tr -d ' '; }
count_datasets() {
  local root=$1 total=0 dataset value
  for dataset in $RESUME_DATASETS; do
    value=$(find "$root" -type f -path "*/$dataset/*/metrics.json" 2>/dev/null | wc -l | tr -d ' ')
    total=$((total + value))
  done
  echo "$total"
}
count_filtered() {
  local root=$1 kernels=$2 artifact=${3:-metrics.json} total=0 dataset kernel value
  for dataset in $RESUME_DATASETS; do
    for kernel in $kernels; do
      value=$(find "$root/$kernel/$dataset" -type f -name "$artifact" 2>/dev/null | wc -l | tr -d ' ')
      total=$((total + value))
    done
  done
  echo "$total"
}
count_trained_filtered() {
  local root=$1 kernels=$2 total=0 dataset kernel value
  for dataset in $RESUME_DATASETS; do
    for kernel in $kernels; do
      value=$(
        find "$root/$kernel/$dataset" -type f \
          \( -name training.json -o -name metrics.json \) -print 2>/dev/null |
          sed 's#/[^/]*$##' | sort -u | wc -l | tr -d ' '
      )
      total=$((total + value))
    done
  done
  echo "$total"
}
word_count() {
  local words=$1
  local -a values
  read -ra values <<< "$words"
  echo "${#values[@]}"
}
stage_enabled() { [[ " ${SVGP_RESUME_STAGES:-stationary extended transplant convergence} " == *" $1 "* ]]; }
bar() {
  local done=$1 total=$2 width=24 filled empty
  filled=$(( total > 0 ? done * width / total : 0 ))
  empty=$((width - filled))
  printf '['
  printf '%*s' "$filled" '' | tr ' ' '#'
  printf '%*s' "$empty" '' | tr ' ' '.'
  printf '] %5d/%-5d' "$done" "$total"
}

case "$RESUME_SCOPE" in
  all)
    DONOR_TOTAL=2560
    TRANSPLANT_TOTAL=5120
    CONVERGENCE_TOTAL=16
    ;;
  3droad)
    DONOR_TOTAL=320
    TRANSPLANT_TOTAL=640
    CONVERGENCE_TOTAL=0
    ;;
  *)
    DONOR_TOTAL=2240
    TRANSPLANT_TOTAL=4480
    CONVERGENCE_TOTAL=16
    ;;
esac

echo "===== $(basename "$FINAL_ROOT") | SVGP $RESUME_SCOPE | $(date +%H:%M:%S) ====="
echo "  stages: ${SVGP_RESUME_STAGES:-stationary extended transplant convergence}"
echo "  prediction mode: ${SVGP_RESUME_PREDICTION_MODE:-full} reps=${SVGP_PREDICTION_REPS:-10}"
echo "  extended kernels: ${SVGP_RESUME_EXTENDED_KERNELS:-amplitude-rbf gibbs}"
echo "  transplant kernels: ${SVGP_RESUME_TRANSPLANT_KERNELS:-rbf matern52 amplitude-rbf gibbs}"

DATASET_COUNT=$(word_count "$RESUME_DATASETS")
if stage_enabled extended; then
  ACTIVE_EXTENDED_KERNELS="${SVGP_RESUME_EXTENDED_KERNELS:-amplitude-rbf gibbs}"
  ACTIVE_EXTENDED_KERNEL_COUNT=$(word_count "$ACTIVE_EXTENDED_KERNELS")
  ACTIVE_EXTENDED_TOTAL=$((DATASET_COUNT * ACTIVE_EXTENDED_KERNEL_COUNT * 160))
  echo
  echo "----- ACTIVE TARGETS -----"
  printf '  extended trained   '; bar "$(count_trained_filtered "$FINAL_ROOT/svgp/extended/runs" "$ACTIVE_EXTENDED_KERNELS")" "$ACTIVE_EXTENDED_TOTAL"; echo
  printf '  extended predicted '; bar "$(count_filtered "$FINAL_ROOT/svgp/extended/runs" "$ACTIVE_EXTENDED_KERNELS")" "$ACTIVE_EXTENDED_TOTAL"; echo
fi
if stage_enabled transplant; then
  ACTIVE_TRANSPLANT_KERNELS="${SVGP_RESUME_TRANSPLANT_KERNELS:-rbf matern52 amplitude-rbf gibbs}"
  ACTIVE_TRANSPLANT_KERNEL_COUNT=$(word_count "$ACTIVE_TRANSPLANT_KERNELS")
  ACTIVE_TRANSPLANT_TOTAL=$((DATASET_COUNT * ACTIVE_TRANSPLANT_KERNEL_COUNT * 160))
  if ! stage_enabled extended; then
    echo
    echo "----- ACTIVE TARGETS -----"
  fi
  printf '  transplant trained   '; bar "$(count_trained_filtered "$FINAL_ROOT/svgp/transplant/runs" "$ACTIVE_TRANSPLANT_KERNELS")" "$ACTIVE_TRANSPLANT_TOTAL"; echo
  printf '  transplant predicted '; bar "$(count_filtered "$FINAL_ROOT/svgp/transplant/runs" "$ACTIVE_TRANSPLANT_KERNELS")" "$ACTIVE_TRANSPLANT_TOTAL"; echo
fi
echo
echo "----- SCOPE RESULTS -----"
printf '  stationary  '; bar "$(count_datasets "$FINAL_ROOT/svgp/stationary/runs")" "$DONOR_TOTAL"; echo
printf '  extended    '; bar "$(count_datasets "$FINAL_ROOT/svgp/extended/runs")" "$DONOR_TOTAL"; echo
printf '  transplant  '; bar "$(count_datasets "$FINAL_ROOT/svgp/transplant/runs")" "$TRANSPLANT_TOTAL"; echo
if (( CONVERGENCE_TOTAL > 0 )); then
  printf '  convergence '; bar "$(count "$FINAL_ROOT/svgp/convergence/runs" metrics.json)" "$CONVERGENCE_TOTAL"; echo
fi

echo
echo "----- PRESERVED ROOT TOTALS -----"
printf '  stationary  '; bar "$(count "$FINAL_ROOT/svgp/stationary/runs" metrics.json)" 2560; echo
printf '  extended    '; bar "$(count "$FINAL_ROOT/svgp/extended/runs" metrics.json)" 2560; echo
printf '  transplant  '; bar "$(count "$FINAL_ROOT/svgp/transplant/runs" metrics.json)" 5120; echo
printf '  convergence '; bar "$(count "$FINAL_ROOT/svgp/convergence/runs" metrics.json)" 16; echo

echo
echo "----- WORKERS -----"
if [[ "${RESUME_EXECUTION_MODE:-array}" == bundle ]]; then
  sacct -j "$RESUME_JOB" -n -o JobID,State,Elapsed,NodeList 2>/dev/null |
    awk '$1 !~ /\.(batch|extern)$/ && NF'
else
  sacct -j "$RESUME_JOB" -X -n -o JobID,State,Elapsed,NodeList 2>/dev/null |
    sed '/^[[:space:]]*$/d'
fi
echo
echo "----- QUEUE -----"
squeue -j "$RESUME_JOB" -o "%.18i %.18j %.9T %.10M %.24R" 2>/dev/null

echo
echo "----- LATEST ACTIVITY -----"
find "$FINAL_ROOT/slurm" -type f -name "*_${RESUME_JOB}_*.out" -print0 2>/dev/null |
  xargs -0 ls -1t 2>/dev/null | head -n 5 |
  while read -r log; do
    printf '  %-34s ' "$(basename "$log")"
    grep -E '^\[[0-9]+/[0-9]+\]|===== .*COMPLETE|FAILED:' "$log" | tail -n 1 || echo started
  done

echo
echo "----- CURRENT-GENERATION FAILURES -----"
state="$FINAL_ROOT/state/svgp-resume-$RESUME_JOB"
printf '  worker failure markers: %s\n' "$(count "$state" '*.failed')"
grep -hE 'FAILED:|run\(s\) failed|Traceback|Out of memory|RESOURCE_EXHAUSTED|giving up' \
  "$FINAL_ROOT"/slurm/*_"$RESUME_JOB"_*.out 2>/dev/null | tail -n 12 || true
