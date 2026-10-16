#!/usr/bin/env bash
# Kin40k: reuse the four official donor fits, freeze theta/noise, and optimize
# only Z under the standard SGPR trace bound.
#SBATCH --job-name=sgpr_transplant
#SBATCH --array=0-39%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_sgpr_transplant_%A_%a.out

set -euo pipefail

HERE="${SGPR_BENCHMARK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/.." && pwd)}
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$HERE/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$PROJECT_ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
source "$VENV_ACTIVATE"

CURRENT_HEAD="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
if [[ -n "${EXPECTED_HEAD:-}" && "$CURRENT_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "Expected commit $EXPECTED_HEAD, found $CURRENT_HEAD." >&2
  exit 1
fi
if [[ -n "$(git -C "$SIM_ROOT" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "Experiment-source changes detected; refusing an official run." >&2
  exit 1
fi

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

# Keep the task identity identical to paper_sgpr_main.sbatch so a sweep worker
# can run one donor group and its transplant back-to-back without a barrier.
KERNELS=(rbf gibbs matern52 amplitude-rbf)
TASK="${SLURM_ARRAY_TASK_ID:?submit this file as an array}"
KERNEL="${KERNELS[$((TASK / 10))]}"
SEED="$((TASK % 10))"
M="${SGPR_M:-256}"
if [[ "$M" != "128" && "$M" != "256" && "$M" != "512" ]]; then
  echo "SGPR_M must be one of 128, 256, 512; found $M" >&2
  exit 2
fi

source "$SIM_ROOT/gpu_preflight.sh"
require_gpu
python -u "$HERE/analysis/explore_hyperparameter_transplant.py" \
  --dataset kin40k \
  --kernel "$KERNEL" \
  --m "$M" \
  --seeds "$SEED" \
  --n-train 5000 \
  --n-test 2000 \
  --block-size 500 \
  --transplant-steps 10000 \
  --donor-runs-dir "${SGPR_DONOR_RUNS:?set SGPR_DONOR_RUNS}" \
  --out-dir "${SGPR_TRANSPLANT_OUT:?set SGPR_TRANSPLANT_OUT}/$KERNEL" \
  --no-summary
