#!/usr/bin/env bash
# One Slurm allocation owns all resume GPUs. Concurrent exclusive steps bind
# one worker to one GPU, avoiding an array element landing on an unusable GPU.
#SBATCH --job-name=svgp_resume
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --mem=192G
#SBATCH --time=6-23:59:00
#SBATCH --no-requeue

set -Eeuo pipefail

FINAL_ROOT="${FINAL_ROOT:?set FINAL_ROOT}"
ORCHESTRATOR_PROJECT_ROOT="${ORCHESTRATOR_PROJECT_ROOT:?set ORCHESTRATOR_PROJECT_ROOT}"
NSHARDS="${NSHARDS:-3}"
BUNDLE_ID="${SLURM_JOB_ID:?submit through Slurm}"
WORKER="$ORCHESTRATOR_PROJECT_ROOT/Simulations/svgp_resume_worker.sbatch"

if (( NSHARDS < 1 || NSHARDS > 5 )); then
  echo "NSHARDS must be in [1, 5], found $NSHARDS" >&2
  exit 2
fi

pids=()
for ((worker_id=0; worker_id<NSHARDS; worker_id++)); do
  log="$FINAL_ROOT/slurm/svgp_resume_${BUNDLE_ID}_${worker_id}.out"
  srun \
    --exclusive \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gres=gpu:1 \
    --output="$log" \
    env \
      RESUME_WORKER_ID="$worker_id" \
      RESUME_ARRAY_ID="$BUNDLE_ID" \
      bash "$WORKER" &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done
if (( failures > 0 )); then
  echo "$failures/$NSHARDS resume worker(s) failed" >&2
  exit 1
fi
echo "===== SVGP RESUME BUNDLE COMPLETE $(date -Is) ====="
