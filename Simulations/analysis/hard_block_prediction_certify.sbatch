#!/usr/bin/env bash
# Read-only verification of the completed prediction-only campaign.
#SBATCH --job-name=hard_pic_certify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail

: "${HARD_PREDICTION_ROOT:?set the prediction output root}"
: "${AIPY:?set the pinned Python interpreter}"

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1

"$AIPY" -u "$HARD_PREDICTION_ROOT/tools/hard_block_prediction_campaign.py" \
  certify --output "$HARD_PREDICTION_ROOT"
