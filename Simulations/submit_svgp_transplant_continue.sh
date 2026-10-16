#!/usr/bin/env bash
# Plan first; submit only this isolated extension. Never cancel older jobs.
set -euo pipefail
export CONT_TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AIPY="${AIPY:?set AIPY to the python of the release environment}"
export SCALABLE_GP_DATASET_DIR="${SCALABLE_GP_DATASET_DIR:-$CONT_TOOLS/dataset}"
PARENT_ROOT="${PARENT_ROOT:?set PARENT_ROOT to the finished v18 campaign root}"
PARTITION="${PARTITION:-NA100q}"
GPU_NODE="${GPU_NODE:-node01}"
export CONT_ROOT="${CONT_ROOT:-${PARENT_ROOT}_transplant300_$(date +%Y%m%d_%H%M%S)}"
export PYTHONDONTWRITEBYTECODE=1
[[ -x "$AIPY" && -f "$PARENT_ROOT/EXPECTED_HEAD" ]]
if [[ -e "$CONT_ROOT" ]]; then
  echo "Output already exists: $CONT_ROOT. Refusing a duplicate dispatch." >&2
  exit 2
fi
# CPU-only validation does not occupy a GPU or train any model.
JAX_PLATFORMS=cpu "$AIPY" "$CONT_TOOLS/analysis/continue_svgp_transplant.py" plan \
  --root "$PARENT_ROOT" --output "$CONT_ROOT" --data-dir "$SCALABLE_GP_DATASET_DIR"
mkdir "$CONT_ROOT/dispatch"
JOB=$(sbatch --parsable --array=0-4%5 --partition="$PARTITION" --nodelist="$GPU_NODE" \
  --export=ALL --output="$CONT_ROOT/slurm/continue_%A_%a.out" \
  "$CONT_TOOLS/svgp_transplant_continue.sbatch")
JOB=${JOB%%;*}
printf '%s\n' "$JOB" > "$CONT_ROOT/dispatch/job_id.txt"
ENV_FILE="$CONT_ROOT/jobs.env"
{
  printf 'export CONT_TOOLS=%q\n' "$CONT_TOOLS"
  printf 'export CONT_ROOT=%q\n' "$CONT_ROOT"
  printf 'export CONT_JOB=%q\n' "$JOB"
  printf 'export AIPY=%q\n' "$AIPY"
} > "$ENV_FILE"
# A pointer only, never overwrites an existing campaign environment.
cp "$ENV_FILE" "$HOME/aistats2026_transplant300_latest.env"
printf 'JOB=%s\nCONT_ROOT=%s\nENV_FILE=%s\n' "$JOB" "$CONT_ROOT" "$ENV_FILE"
echo '5 single-GPU workers requested. Pending tasks wait for Slurm resources.'
