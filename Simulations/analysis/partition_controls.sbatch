#!/usr/bin/env bash
# Prediction-only controls. Every checkpoint and historical source is read-only.
#SBATCH --job-name=partition_ctrl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
set -euo pipefail
: "${AIPY:?export the pinned Python interpreter}"
: "${CONTROL_ROOT:?set a NEW output root containing plan.json}"
: "${TOOLS_DIR:?set the evaluator snapshot directory}"
: "${SCALABLE_GP_DATASET_DIR:?set the frozen dataset cache}"
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
"$AIPY" -u "$TOOLS_DIR/partition_controls.py" worker --output "$CONTROL_ROOT" \
  --worker "$SLURM_ARRAY_TASK_ID" --workers "${CONTROL_WORKERS:-5}"
