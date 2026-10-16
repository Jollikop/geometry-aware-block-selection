#!/usr/bin/env bash
# Single-seed Kin40k convergence histories for the four official kernels.
#SBATCH --job-name=svgp_convergence
#SBATCH --array=0-3%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=slurm_svgp_convergence_%A_%a.out

set -euo pipefail

HERE="${SVGP_BENCHMARK_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/.." && pwd)}
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$HERE/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$PROJECT_ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
source "$VENV_ACTIVATE"
cd "$HERE"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

# Same protocol as the two grid campaigns, so the curve and the table are
# the same experiment: --warp-gate-fraction 0.0 in particular, or the warp
# gradient is held at zero for 375 of the Gibbs run's 500 epochs.
KERNELS=(rbf matern52 amplitude-rbf gibbs)
TASK="${SLURM_ARRAY_TASK_ID:?submit this file as an array}"
KERNEL="${KERNELS[$TASK]}"
if [[ "$KERNEL" == "rbf" || "$KERNEL" == "matern52" ]]; then
  EPOCHS=350
  REFRESH_UNTIL=200
else
  EPOCHS=500
  REFRESH_UNTIL=500
fi

python -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices())'
python -u run_benchmark.py \
  --plan custom \
  --run-tag "${RUN_TAG:-aistats2026_svgp_convergence_v1}" \
  --kernels "$KERNEL" \
  --datasets kin40k \
  --methods SVGP T-SVGP RBT-SVGP SBT-SVGP \
  --m 256 \
  --seeds 0 \
  --all-data \
  --epochs "$EPOCHS" \
  --block 500 \
  --ll-batch-size 1000 \
  --blocks-per-step 2 \
  --refresh-every 40 \
  --refresh-until "$REFRESH_UNTIL" \
  --common-ll-batches \
  --batch-coupling split \
  --penalty-sampling uniform \
  --optimizer adam \
  --lr 0.005 \
  --z-lr 0.005 \
  --lr-profile fixed \
  --lr-schedule constant \
  --q-optimizer natural \
  --natural-lr 0.05 \
  --sbt-geometry anchor \
  --warp-gate-fraction 0.0 \
  --swa-fraction 0.0 \
  --normalize-objective \
  --record-history \
  --save-params \
  --runs-dir "${RUNS_DIR:-$HERE/runs_v12/aistats2026_svgp_convergence_v1}" \
  --logs-dir "${LOGS_DIR:-$HERE/logs_v12/aistats2026_svgp_convergence_v1}" \
  --no-live
