#!/usr/bin/env bash
#SBATCH --job-name=gibbs_anchor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=6-23:59:00
#SBATCH --no-requeue
set -euo pipefail
ROOT=${1:?campaign root}
AIPY=${2:?pinned Python executable}
exec "$AIPY" -u "$ROOT/tools/gibbs_anchor_campaign.py" worker \
  --root "$ROOT" --worker "${SLURM_ARRAY_TASK_ID:?array index}"
