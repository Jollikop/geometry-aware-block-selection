#!/usr/bin/env bash
# Bound-matched Snelson figure and manifest.
#SBATCH --job-name=sgpr_snelson
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_sgpr_snelson_%j.out

set -euo pipefail

HERE="${SNELSON_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:?set PROJECT_VENV_ACTIVATE}"
OUT="${SNELSON_OUT:?set SNELSON_OUT}"
source "$VENV_ACTIVATE"
mkdir -p "$OUT"

export AIPY="$(command -v python)"
export REPO_ROOT="$PROJECT_ROOT"
export SCALABLE_GP_DATASET_DIR="${SCALABLE_GP_DATASET_DIR:?set dataset cache}"
export SNELSON_OUT="$OUT"
bash "$HERE/make_snelson_figure.sh"
echo "SNELSON COMPLETE: $OUT"
