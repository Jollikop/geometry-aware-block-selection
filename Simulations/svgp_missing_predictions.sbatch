#!/usr/bin/env bash
#SBATCH --job-name=svgp_pred_missing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6-23:59:00
#SBATCH --no-requeue
set -euo pipefail
: "${PRED_OUTPUT:?}"
: "${PRED_TOOLS:?}"
: "${AIPY:?}"
: "${SCALABLE_GP_DATASET_DIR:?}"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset XLA_PYTHON_CLIENT_ALLOCATOR XLA_FLAGS
echo "Prediction worker=${SLURM_ARRAY_TASK_ID} host=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
exec "$AIPY" "$PRED_TOOLS/analysis/run_svgp_missing_predictions.py" worker \
  --output "$PRED_OUTPUT" --worker "$SLURM_ARRAY_TASK_ID"
