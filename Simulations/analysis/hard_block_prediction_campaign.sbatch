#!/usr/bin/env bash
# Prediction-only official hard PIC/PITC-style campaign.
# Frozen checkpoints and training code are read-only; outputs use a separate root.
#SBATCH --job-name=hard_pic_predict
#SBATCH --array=0-4%5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
set -euo pipefail

: "${HARD_PREDICTION_ROOT:?set a separate prediction output root}"
: "${AIPY:?set the pinned Python interpreter}"
: "${SCALABLE_GP_DATASET_DIR:?set the frozen dataset cache directory}"

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

"$AIPY" -u "$HARD_PREDICTION_ROOT/tools/hard_block_prediction_campaign.py" \
  worker --output "$HARD_PREDICTION_ROOT" \
  --worker "$SLURM_ARRAY_TASK_ID" --workers 5 --test-chunk 512
