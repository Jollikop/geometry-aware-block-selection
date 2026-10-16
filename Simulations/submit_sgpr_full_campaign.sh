#!/usr/bin/env bash
# Submit the isolated five-worker SGPR v23 paper campaign.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PARTITION NODELIST|- [CONCURRENCY]" >&2
  exit 2
fi
PARTITION=$1
NODELIST=$2
CONCURRENCY=${3:-5}
if [[ ! "$CONCURRENCY" =~ ^[1-5]$ ]]; then
  echo "CONCURRENCY must be in 1..5" >&2
  exit 2
fi

SIM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$PROJECT_ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv-aistats-20260801/bin/activate}"
AIPY="$(dirname "$VENV_ACTIVATE")/python"
DATASET_CACHE="${SCALABLE_GP_DATASET_DIR:-$SIM_ROOT/dataset}"
EXPECTED_HEAD=$(git -C "$PROJECT_ROOT" rev-parse HEAD)

[[ -f "$VENV_ACTIVATE" ]] || { echo "missing $VENV_ACTIVATE" >&2; exit 2; }
[[ -x "$AIPY" ]] || { echo "missing $AIPY" >&2; exit 2; }
[[ -d "$DATASET_CACHE" ]] || { echo "missing $DATASET_CACHE" >&2; exit 2; }
if [[ -n "$(git -C "$SIM_ROOT" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "official experiment source is dirty; commit it before submission" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
FINAL_ROOT="${SGPR_FULL_ROOT:-$HOME/experiment_results/aistats2026_sgpr_full_v23_${EXPECTED_HEAD:0:12}_$STAMP}"
mkdir -p "$FINAL_ROOT/slurm" "$FINAL_ROOT/state" "$FINAL_ROOT/runs" \
  "$FINAL_ROOT/logs" "$FINAL_ROOT/transplant"
printf '%s\n' "$EXPECTED_HEAD" > "$FINAL_ROOT/EXPECTED_HEAD"

ARGS=(
  --parsable
  --partition="$PARTITION"
  --array="0-4%$CONCURRENCY"
  --output="$FINAL_ROOT/slurm/sgpr_full_%A_%a.out"
  --export="ALL,FINAL_ROOT=$FINAL_ROOT,PROJECT_ROOT=$PROJECT_ROOT,PROJECT_VENV_ACTIVATE=$VENV_ACTIVATE,SCALABLE_GP_DATASET_DIR=$DATASET_CACHE,EXPECTED_HEAD=$EXPECTED_HEAD"
)
[[ "$NODELIST" == "-" ]] || ARGS+=(--nodelist="$NODELIST")
SGPR_JOB=$(sbatch "${ARGS[@]}" "$SIM_ROOT/sgpr_full_worker.sbatch")

ENV_FILE="${SGPR_FULL_ENV_FILE:-$HOME/aistats2026_sgpr_full_v23_jobs.env}"
{
  printf 'export FINAL_ROOT=%q\n' "$FINAL_ROOT"
  printf 'export PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
  printf 'export EXPECTED_HEAD=%q\n' "$EXPECTED_HEAD"
  printf 'export AIPY=%q\n' "$AIPY"
  printf 'export SGPR_JOB=%q\n' "$SGPR_JOB"
} > "$ENV_FILE"

printf 'FINAL_ROOT=%s\nSGPR_JOB=%s\nENV_FILE=%s\n' \
  "$FINAL_ROOT" "$SGPR_JOB" "$ENV_FILE"
printf 'watch -n 10 bash %q\n' "$SIM_ROOT/watch_sgpr_full_campaign.sh"
