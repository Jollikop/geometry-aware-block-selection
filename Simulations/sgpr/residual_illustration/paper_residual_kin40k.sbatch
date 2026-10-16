#!/usr/bin/env bash
# kin40k residual-structure heatmaps, RBF and Gibbs, one GPU job.
#   sbatch --partition=PA10080q --nodelist=node04 paper_residual_kin40k.sbatch
#SBATCH --job-name=residual_kin40k
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_residual_kin40k_%j.out

set -euo pipefail

HERE="${RESIDUAL_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"      # .../Simulations/sgpr/residual_illustration
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/../.." && pwd)}       # .../Simulations
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ROOT="${PROJECT_ROOT:-$(cd "$SIM_ROOT/.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Virtual environment not found: $VENV_ACTIVATE" >&2
  exit 1
fi
source "$VENV_ACTIVATE"
cd "$SIM_ROOT"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

source "$SIM_ROOT/gpu_preflight.sh"
require_gpu

python -u -m sgpr.residual_illustration.run_residual_structure_kin40k \
  --kernels rbf gibbs \
  --n-train 5000 \
  --n-test 2000 \
  --m 256 \
  --block-size 500 \
  --seed 0 \
  --runs-dir "${SGPR_DONOR_RUNS:?set SGPR_DONOR_RUNS}" \
  --outdir "${RESIDUAL_OUT:?set RESIDUAL_OUT}" \
  --optimizer lbfgs \
  --refresh-schedule 15,15,20,25,29925 \
  --split-mode boundary \
  --save-npz
