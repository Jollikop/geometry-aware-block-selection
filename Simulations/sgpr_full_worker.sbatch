#!/usr/bin/env bash
# Paper SGPR v23: Kin40k, RBF/Gibbs, M={128,256,512}, five resumable workers.
# Each worker owns two seeds. RBF is completed before the corresponding Gibbs
# fit, so Gibbs can use the certified stationary solution as a nested warm start.
#SBATCH --job-name=sgpr_full
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
SIM_ROOT="$PROJECT_ROOT/Simulations"
SGPR_ROOT="$SIM_ROOT/sgpr"
STATE="$FINAL_ROOT/state"
GENERATION="$STATE/generation-$ARRAY_ID"

if [[ "$(git -C "$PROJECT_ROOT" rev-parse HEAD)" != "$EXPECTED_HEAD" ]]; then
  echo "Expected commit $EXPECTED_HEAD, found $(git -C "$PROJECT_ROOT" rev-parse HEAD)" >&2
  exit 2
fi
if [[ -n "$(git -C "$SIM_ROOT" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "Experiment source is dirty; refusing an official run" >&2
  exit 2
fi

mkdir -p "$FINAL_ROOT/runs" "$FINAL_ROOT/logs" "$FINAL_ROOT/transplant" \
  "$STATE/completed" "$GENERATION"
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

export GPU_MAX_REQUEUE="${GPU_MAX_REQUEUE:-2}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset JAX_PLATFORMS || true

source "${PROJECT_VENV_ACTIVATE:?set PROJECT_VENV_ACTIVATE}"
source "$SIM_ROOT/gpu_preflight.sh"
require_gpu pymetis

# E-SGPR is intentionally deferred: its dense residual endpoint dominates the
# wall clock and is not needed by the sparsification/transplant table.
METHODS=(Exact-GP METIS-SGPR SGPR T-SGPR RBT-SGPR SBT-SGPR)
M_VALUES=(256 128 512)
SEEDS=("$WORKER_ID" "$((WORKER_ID + 5))")
failed=0

run_main() {
  local kernel=$1 m=$2 seed=$3
  local extra=()
  if [[ "$kernel" == "gibbs" ]]; then
    extra+=(--warm-start-runs-dir "$FINAL_ROOT/runs")
  fi
  echo "===== worker=$WORKER_ID kernel=$kernel M=$m seed=$seed main ====="
  if python -u "$SGPR_ROOT/run_benchmark.py" \
      --datasets kin40k \
      --kernels "$kernel" \
      --methods "${METHODS[@]}" \
      --m "$m" \
      --seeds "$seed" \
      --n-train 5000 \
      --n-test 2000 \
      --block-size 500 \
      --optimizer lbfgs \
      --refresh-schedule 15,15,20,25,99925 \
      --require-convergence \
      --full-predictor-max-n 20000 \
      --logdet-max-n 5000 \
      --save-params \
      --runs-dir "$FINAL_ROOT/runs" \
      --logs-dir "$FINAL_ROOT/logs" \
      --no-live \
      --no-tables \
      "${extra[@]}"; then
    printf 'commit=%s kernel=%s M=%s seed=%s time=%s\n' \
      "$EXPECTED_HEAD" "$kernel" "$m" "$seed" "$(date -Is)" \
      > "$STATE/completed/main.${kernel}.M${m}.seed${seed}.done"
  else
    echo "MAIN INCOMPLETE kernel=$kernel M=$m seed=$seed; continuing shard" >&2
    failed=$((failed + 1))
  fi
}

run_transplant() {
  local kernel=$1 m=$2 seed=$3
  echo "===== worker=$WORKER_ID kernel=$kernel M=$m seed=$seed transplant ====="
  if python -u "$SGPR_ROOT/analysis/explore_hyperparameter_transplant.py" \
      --dataset kin40k \
      --kernel "$kernel" \
      --m "$m" \
      --seeds "$seed" \
      --n-train 5000 \
      --n-test 2000 \
      --block-size 500 \
      --transplant-steps 100000 \
      --donor-runs-dir "$FINAL_ROOT/runs" \
      --out-dir "$FINAL_ROOT/transplant/M$m/$kernel" \
      --no-summary; then
    printf 'commit=%s kernel=%s M=%s seed=%s time=%s\n' \
      "$EXPECTED_HEAD" "$kernel" "$m" "$seed" "$(date -Is)" \
      > "$STATE/completed/transplant.${kernel}.M${m}.seed${seed}.done"
  else
    echo "TRANSPLANT INCOMPLETE kernel=$kernel M=$m seed=$seed; continuing shard" >&2
    failed=$((failed + 1))
  fi
}

for m in "${M_VALUES[@]}"; do
  # The ordering is a dependency: every Gibbs method needs its matching RBF
  # donor in the same result tree.
  for seed in "${SEEDS[@]}"; do run_main rbf "$m" "$seed"; done
  for seed in "${SEEDS[@]}"; do run_main gibbs "$m" "$seed"; done
  for kernel in rbf gibbs; do
    for seed in "${SEEDS[@]}"; do run_transplant "$kernel" "$m" "$seed"; done
  done
done

printf 'worker=%s incomplete_phases=%s time=%s\n' \
  "$WORKER_ID" "$failed" "$(date -Is)" \
  > "$STATE/worker.$WORKER_ID.complete"
echo "===== worker=$WORKER_ID COMPLETE incomplete_phases=$failed $(date -Is) ====="
