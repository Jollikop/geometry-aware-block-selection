#!/usr/bin/env bash
#SBATCH --job-name=gibbs_anchor_cert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${AIPY:?}" "${TOOLS_ROOT:?}" "${CAMPAIGN_ROOT:?}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
exec "$AIPY" -u "$TOOLS_ROOT/analysis/certify_gibbs_anchor_transplant.py" \
  --campaign "$CAMPAIGN_ROOT"
