#!/usr/bin/env bash
#SBATCH --job-name=gibbs_anchor_plan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
set -euo pipefail

: "${STAGE:?}" "${AIPY:?}" "${TOOLS_ROOT:?}" "${ANCHOR_ROOT:?}"
: "${PARENT_ROOT:?}" "${CONT_ROOT:?}" "${SCALABLE_GP_DATASET_DIR:?}"
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TOOLS_ROOT"

case "$STAGE" in
  plan)
    exec "$AIPY" -u "$TOOLS_ROOT/analysis/gibbs_anchor_transplant_scope.py" \
      plan --continuation-tool "$TOOLS_ROOT/analysis/continue_svgp_transplant.py" \
      --parent "$PARENT_ROOT" --donor-root "$ANCHOR_ROOT" \
      --data-dir "$SCALABLE_GP_DATASET_DIR" --output "$CONT_ROOT"
    ;;
  prepare-prediction)
    exec "$AIPY" -u "$TOOLS_ROOT/analysis/gibbs_anchor_transplant_scope.py" \
      prepare-prediction \
      --continuation-tool "$TOOLS_ROOT/analysis/continue_svgp_transplant.py" \
      --output "$CONT_ROOT"
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
