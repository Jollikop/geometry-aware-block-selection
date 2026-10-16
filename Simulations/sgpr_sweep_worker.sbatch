#!/usr/bin/env bash
# Three-budget collapsed-SGPR sweep in five resumable workers.
# The submission wrapper starts with one active worker so an already-running
# SVGP campaign keeps four GPU slots; the throttle can be raised in place.
#SBATCH --job-name=sgpr_sweep
#SBATCH --array=0-4%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6-23:59:00
#SBATCH --no-requeue

set -Eeuo pipefail

SWEEP_ROOT="${SWEEP_ROOT:?set SWEEP_ROOT}"
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
EXPECTED_HEAD="${EXPECTED_HEAD:?set EXPECTED_HEAD}"
WORKER_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-4}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?}}"
NSHARDS=5

SIM_ROOT="$PROJECT_ROOT/Simulations"
SGPR="$SIM_ROOT/sgpr"
STATE="$SWEEP_ROOT/state"
GENERATION="$STATE/generation-$ARRAY_ID"

if (( WORKER_ID < 0 || WORKER_ID >= NSHARDS )); then
  echo "worker id must be in [0, $((NSHARDS - 1))]: $WORKER_ID" >&2
  exit 2
fi
if [[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  echo "Expected commit $EXPECTED_HEAD, found $(git -C "$PROJECT_ROOT" rev-parse HEAD)." >&2
  exit 2
fi
if [[ -n "$(git -C "$SIM_ROOT" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "Experiment-source changes detected; refusing an official run." >&2
  exit 2
fi

mkdir -p "$GENERATION" "$STATE/completed"
# A bounded GPU-preflight requeue keeps the same array identity.  Clear the
# marker left by the previous allocation before trying the newly assigned GPU;
# a genuine final failure will recreate it through the EXIT trap.
rm -f "$GENERATION/worker.$WORKER_ID.failed"

on_exit() {
  local status=$?
  if (( status != 0 )); then
    printf 'worker=%s status=%s time=%s\n' \
      "$WORKER_ID" "$status" "$(date -Is)" \
      > "$GENERATION/worker.$WORKER_ID.failed"
  fi
}
trap on_exit EXIT

# A bad GRES mapping must not kill a multi-day shard immediately, but retries
# remain strictly bounded so a sick cluster cannot recreate the old infinite
# requeue loop.  The submission should offer more than one CUDA-probed node.
export GPU_MAX_REQUEUE="${GPU_MAX_REQUEUE:-2}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

source "$SIM_ROOT/gpu_preflight.sh"
require_gpu pymetis

# Produce the paper's M=256 row first, then add the two sweep budgets. Each
# task owns one
# (kernel, seed) identity in both the donor and transplant stages, so there is
# no cross-worker dependency and no barrier that can deadlock after a failure.
M_VALUES=(256 128 512)
for M in "${M_VALUES[@]}"; do
  for ((task=WORKER_ID; task<40; task+=NSHARDS)); do
    echo "===== worker=$WORKER_ID M=$M task=$task main ====="
    env \
      SLURM_ARRAY_TASK_ID="$task" \
      SGPR_M="$M" \
      SGPR_SAVE_BLOCK_CONTRIBUTIONS=0 \
      SGPR_BENCHMARK_DIR="$SGPR" \
      RUNS_DIR="$SWEEP_ROOT/runs" \
      LOGS_DIR="$SWEEP_ROOT/logs" \
      bash "$SGPR/paper_sgpr_main.sbatch"

    echo "===== worker=$WORKER_ID M=$M task=$task transplant ====="
    env \
      SLURM_ARRAY_TASK_ID="$task" \
      SGPR_M="$M" \
      SGPR_BENCHMARK_DIR="$SGPR" \
      SGPR_DONOR_RUNS="$SWEEP_ROOT/runs" \
      SGPR_TRANSPLANT_OUT="$SWEEP_ROOT/transplant/M$M" \
      bash "$SGPR/paper_sgpr_transplant.sbatch"

    printf 'commit=%s worker=%s time=%s\n' \
      "$EXPECTED_HEAD" "$WORKER_ID" "$(date -Is)" \
      > "$STATE/completed/M${M}.task${task}.done"
  done
done

touch "$STATE/worker.$WORKER_ID.complete"
echo "===== worker=$WORKER_ID COMPLETE $(date -Is) ====="
