#!/usr/bin/env bash
# Complete GPU campaign in exactly five long-lived Slurm array workers.
# Each worker owns one fifth of every sharded campaign and runs its stages
# sequentially. Re-submit the same FINAL_ROOT to resume from validated caches.
# Snelson remains a local CPU/Mac stage and is intentionally not run here.
#SBATCH --job-name=paper_worker
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6-23:59:00
#SBATCH --no-requeue

set -Eeuo pipefail

FINAL_ROOT="${FINAL_ROOT:?set FINAL_ROOT}"
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
EXPECTED_HEAD="${EXPECTED_HEAD:?set EXPECTED_HEAD}"
WORKER_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-4}"
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?}}"
NSHARDS=5

SIM_ROOT="$PROJECT_ROOT/Simulations"
SGPR="$SIM_ROOT/sgpr"
SVGP="$SIM_ROOT/svgp"
STATE="$FINAL_ROOT/state"
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

mkdir -p "$GENERATION"

on_exit() {
  local status=$?
  if (( status != 0 )); then
    printf 'worker=%s status=%s time=%s\n' \
      "$WORKER_ID" "$status" "$(date -Is)" \
      > "$GENERATION/worker.$WORKER_ID.failed"
  fi
}
trap on_exit EXIT

stage_done() {
  local stage=$1
  mkdir -p "$STATE/completed"
  printf 'worker=%s time=%s commit=%s\n' \
    "$WORKER_ID" "$(date -Is)" "$EXPECTED_HEAD" \
    > "$STATE/completed/$stage.$WORKER_ID.done"
}

echo "===== worker=$WORKER_ID/$NSHARDS array=$ARRAY_ID start=$(date -Is) ====="

# Never requeue a worker onto arbitrary devices. A bad allocation fails once;
# the five resumable workers can then be submitted again after choosing GPUs.
export GPU_MAX_REQUEUE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

source "$SIM_ROOT/gpu_preflight.sh"
require_gpu pymetis

echo "===== worker=$WORKER_ID stage=sgpr-main ====="
for ((task=WORKER_ID; task<40; task+=NSHARDS)); do
  env \
    SLURM_ARRAY_TASK_ID="$task" \
    SGPR_BENCHMARK_DIR="$SGPR" \
    RUNS_DIR="$FINAL_ROOT/sgpr/runs" \
    LOGS_DIR="$FINAL_ROOT/sgpr/logs" \
    bash "$SGPR/paper_sgpr_main.sbatch"
done
stage_done sgpr-main

echo "===== worker=$WORKER_ID stage=sgpr-transplant ====="
for ((task=WORKER_ID; task<40; task+=NSHARDS)); do
  env \
    SLURM_ARRAY_TASK_ID="$task" \
    SGPR_BENCHMARK_DIR="$SGPR" \
    SGPR_DONOR_RUNS="$FINAL_ROOT/sgpr/runs" \
    SGPR_TRANSPLANT_OUT="$FINAL_ROOT/sgpr/transplant" \
    bash "$SGPR/paper_sgpr_transplant.sbatch"
done
stage_done sgpr-transplant

# The residual figure needs the RBF/Gibbs seed-0 donors; both belong to worker 0.
if (( WORKER_ID == 0 )); then
  echo "===== worker=0 stage=sgpr-residual ====="
  env \
    RESIDUAL_DIR="$SGPR/residual_illustration" \
    SGPR_DONOR_RUNS="$FINAL_ROOT/sgpr/runs" \
    RESIDUAL_OUT="$FINAL_ROOT/sgpr/residual_illustration" \
    bash "$SGPR/residual_illustration/paper_residual_kin40k.sbatch"
  stage_done sgpr-residual
fi

echo "===== worker=$WORKER_ID stage=svgp-stationary ====="
env \
  SLURM_ARRAY_TASK_ID="$WORKER_ID" \
  NSHARDS="$NSHARDS" \
  SVGP_BENCHMARK_DIR="$SVGP" \
  RUN_TAG=aistats2026_svgp_stationary_v12 \
  RUNS_DIR="$FINAL_ROOT/svgp/stationary/runs" \
  LOGS_DIR="$FINAL_ROOT/svgp/stationary/logs" \
  bash "$SVGP/paper_grid_stationary.sbatch"
stage_done svgp-stationary

echo "===== worker=$WORKER_ID stage=svgp-extended ====="
env \
  SLURM_ARRAY_TASK_ID="$WORKER_ID" \
  NSHARDS="$NSHARDS" \
  SVGP_BENCHMARK_DIR="$SVGP" \
  RUN_TAG=aistats2026_svgp_extended_v12 \
  RUNS_DIR="$FINAL_ROOT/svgp/extended/runs" \
  LOGS_DIR="$FINAL_ROOT/svgp/extended/logs" \
  bash "$SVGP/paper_grid_gibbs.sbatch"
stage_done svgp-extended
touch "$STATE/svgp-donors.$WORKER_ID.ready"

# Transplant task identities are sharded in a different order from the donor
# grids. Do not read another worker's checkpoint until all donor shards exist.
echo "===== worker=$WORKER_ID waiting-for-all-svgp-donors ====="
deadline=$(( $(date +%s) + ${DONOR_BARRIER_TIMEOUT_SECONDS:-86400} ))
while :; do
  ready=$(find "$STATE" -maxdepth 1 -name 'svgp-donors.*.ready' 2>/dev/null \
    | wc -l | tr -d ' ')
  (( ready >= NSHARDS )) && break

  failed=$(find "$GENERATION" -maxdepth 1 -name 'worker.*.failed' 2>/dev/null \
    | wc -l | tr -d ' ')
  if (( failed > 0 )); then
    echo "A sibling worker failed before the donor barrier; stop and resume all five." >&2
    exit 1
  fi
  if (( $(date +%s) >= deadline )); then
    echo "Timed out waiting for all five SVGP donor shards ($ready/$NSHARDS ready)." >&2
    exit 1
  fi
  echo "donor barrier: $ready/$NSHARDS ready"
  sleep 60
done

echo "===== worker=$WORKER_ID stage=svgp-transplant ====="
env \
  SLURM_ARRAY_TASK_ID="$WORKER_ID" \
  NSHARDS="$NSHARDS" \
  SVGP_BENCHMARK_DIR="$SVGP" \
  RUN_TAG=aistats2026_svgp_sparse_transplant_v1 \
  RUNS_DIR="$FINAL_ROOT/svgp/transplant/runs" \
  LOGS_DIR="$FINAL_ROOT/svgp/transplant/logs" \
  SOURCE_STATIONARY_RUNS="$FINAL_ROOT/svgp/stationary/runs" \
  SOURCE_EXTENDED_RUNS="$FINAL_ROOT/svgp/extended/runs" \
  bash "$SVGP/paper_sparse_transplant.sbatch"
stage_done svgp-transplant

# Four kernels map naturally to workers 0--3; worker 4 has no convergence task.
if (( WORKER_ID < 4 )); then
  echo "===== worker=$WORKER_ID stage=svgp-convergence ====="
  env \
    SLURM_ARRAY_TASK_ID="$WORKER_ID" \
    SVGP_BENCHMARK_DIR="$SVGP" \
    RUN_TAG=aistats2026_svgp_convergence_v1 \
    RUNS_DIR="$FINAL_ROOT/svgp/convergence/runs" \
    LOGS_DIR="$FINAL_ROOT/svgp/convergence/logs" \
    bash "$SVGP/paper_convergence.sbatch"
  stage_done svgp-convergence
fi

touch "$STATE/worker.$WORKER_ID.complete"
echo "===== worker=$WORKER_ID COMPLETE $(date -Is) ====="
