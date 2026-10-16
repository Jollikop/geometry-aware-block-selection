#!/usr/bin/env bash
# Submit one array containing exactly five resumable full-campaign workers.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PARTITION NODELIST|- [DATASET_CACHE]" >&2
  echo "example: $0 RTXA6Kq node08,node11" >&2
  exit 2
fi

PARTITION=$1
NODELIST=$2
SIM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$PROJECT_ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
DATASET_CACHE="${3:-${SCALABLE_GP_DATASET_DIR:-$SIM_ROOT/dataset}}"
EXPECTED_HEAD=$(git -C "$PROJECT_ROOT" rev-parse HEAD)

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "missing environment: $VENV_ACTIVATE" >&2
  exit 2
fi
if [[ ! -d "$DATASET_CACHE" ]]; then
  echo "missing dataset cache: $DATASET_CACHE" >&2
  exit 2
fi
if [[ -n "$(git -C "$SIM_ROOT" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "official experiment source is dirty; commit it before submission" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
# Deliberately ignore a generic FINAL_ROOT inherited from an older campaign.
# Reuse is opt-in through this launcher-specific variable only.
FINAL_ROOT="${FIVE_WORKER_FINAL_ROOT:-$HOME/experiment_results/aistats2026_five_v18_${EXPECTED_HEAD:0:12}_$STAMP}"
COMMIT_FILE="$FINAL_ROOT/EXPECTED_HEAD"
if [[ -f "$COMMIT_FILE" && "$(<"$COMMIT_FILE")" != "$EXPECTED_HEAD" ]]; then
  echo "refusing to reuse $FINAL_ROOT with a different commit" >&2
  echo "stored: $(<"$COMMIT_FILE")" >&2
  echo "current: $EXPECTED_HEAD" >&2
  exit 2
fi
mkdir -p \
  "$FINAL_ROOT/slurm" "$FINAL_ROOT/state" \
  "$FINAL_ROOT/sgpr/runs" "$FINAL_ROOT/sgpr/logs" \
  "$FINAL_ROOT/sgpr/transplant" "$FINAL_ROOT/sgpr/snelson" \
  "$FINAL_ROOT/sgpr/residual_illustration" \
  "$FINAL_ROOT/svgp/stationary/runs" "$FINAL_ROOT/svgp/stationary/logs" \
  "$FINAL_ROOT/svgp/extended/runs" "$FINAL_ROOT/svgp/extended/logs" \
  "$FINAL_ROOT/svgp/transplant/runs" "$FINAL_ROOT/svgp/transplant/logs" \
  "$FINAL_ROOT/svgp/convergence/runs" "$FINAL_ROOT/svgp/convergence/logs"
printf '%s\n' "$EXPECTED_HEAD" > "$COMMIT_FILE"

COMMON_EXPORT="ALL,FINAL_ROOT=$FINAL_ROOT,PROJECT_ROOT=$PROJECT_ROOT,PROJECT_VENV_ACTIVATE=$VENV_ACTIVATE,SCALABLE_GP_DATASET_DIR=$DATASET_CACHE,EXPECTED_HEAD=$EXPECTED_HEAD"
SBATCH_ARGS=(
  --parsable
  --partition="$PARTITION"
  --array=0-4%5
  --output="$FINAL_ROOT/slurm/paper_worker_%A_%a.out"
  --export="$COMMON_EXPORT"
)
if [[ "$NODELIST" != "-" ]]; then
  SBATCH_ARGS+=(--nodelist="$NODELIST")
fi

CAMPAIGN_JOB=$(sbatch "${SBATCH_ARGS[@]}" "$SIM_ROOT/paper_five_worker.sbatch")

ENV_FILE="${FIVE_WORKER_ENV_FILE:-$HOME/aistats2026_five_v18_jobs.env}"
{
  printf 'export FINAL_ROOT=%q\n' "$FINAL_ROOT"
  printf 'export PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
  printf 'export EXPECTED_HEAD=%q\n' "$EXPECTED_HEAD"
  printf 'export CAMPAIGN_JOB=%q\n' "$CAMPAIGN_JOB"
} > "$ENV_FILE"

printf 'FINAL_ROOT=%s\nCAMPAIGN_JOB=%s (exactly five array workers)\n' \
  "$FINAL_ROOT" "$CAMPAIGN_JOB"
printf 'ENV_FILE=%s\n' "$ENV_FILE"
printf 'watch -n 10 bash %q\n' "$SIM_ROOT/watch_five_worker_campaign.sh"
