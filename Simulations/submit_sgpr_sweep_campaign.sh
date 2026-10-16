#!/usr/bin/env bash
# Submit an isolated SGPR M-sweep without modifying or cancelling the active
# five-worker paper/SVGP campaign.

set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 PRESERVED_CAMPAIGN_ROOT PARTITION NODELIST|- [CONCURRENCY]" >&2
  echo "example: $0 \$HOME/experiment_results/aistats2026_five_v18_x PA100q node02,node03 1" >&2
  exit 2
fi

PRESERVED_CAMPAIGN_ROOT=$1
PARTITION=$2
NODELIST=$3
CONCURRENCY=${4:-1}
if [[ ! "$CONCURRENCY" =~ ^[1-5]$ ]]; then
  echo "CONCURRENCY must be between 1 and 5" >&2
  exit 2
fi
if [[ ! -d "$PRESERVED_CAMPAIGN_ROOT/svgp" ]]; then
  echo "missing preserved SVGP tree: $PRESERVED_CAMPAIGN_ROOT/svgp" >&2
  exit 2
fi

SIM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$PROJECT_ROOT/../.." && pwd)}"
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
AIPY="$(dirname "$VENV_ACTIVATE")/python"
DATASET_CACHE="${SCALABLE_GP_DATASET_DIR:-$SIM_ROOT/dataset}"
EXPECTED_HEAD=$(git -C "$PROJECT_ROOT" rev-parse HEAD)

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "missing environment: $VENV_ACTIVATE" >&2
  exit 2
fi
if [[ ! -x "$AIPY" ]]; then
  echo "missing Python interpreter: $AIPY" >&2
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
SWEEP_ROOT="${SGPR_SWEEP_ROOT:-$HOME/experiment_results/aistats2026_sgpr_sweep_v22_${EXPECTED_HEAD:0:12}_$STAMP}"
COMMIT_FILE="$SWEEP_ROOT/EXPECTED_HEAD"
if [[ -f "$COMMIT_FILE" && "$(<"$COMMIT_FILE")" != "$EXPECTED_HEAD" ]]; then
  echo "refusing to reuse $SWEEP_ROOT with a different commit" >&2
  exit 2
fi

mkdir -p \
  "$SWEEP_ROOT/slurm" "$SWEEP_ROOT/state" \
  "$SWEEP_ROOT/runs" "$SWEEP_ROOT/logs" \
  "$SWEEP_ROOT/transplant"
printf '%s\n' "$EXPECTED_HEAD" > "$COMMIT_FILE"
printf '%s\n' "$PRESERVED_CAMPAIGN_ROOT" > "$SWEEP_ROOT/PRESERVED_SVGP_ROOT"

SBATCH_ARGS=(
  --parsable
  --partition="$PARTITION"
  --array="0-4%$CONCURRENCY"
  --output="$SWEEP_ROOT/slurm/sgpr_sweep_%A_%a.out"
  --export="ALL,SWEEP_ROOT=$SWEEP_ROOT,PROJECT_ROOT=$PROJECT_ROOT,PROJECT_VENV_ACTIVATE=$VENV_ACTIVATE,SCALABLE_GP_DATASET_DIR=$DATASET_CACHE,EXPECTED_HEAD=$EXPECTED_HEAD"
)
if [[ "$NODELIST" != "-" ]]; then
  SBATCH_ARGS+=(--nodelist="$NODELIST")
fi

SWEEP_JOB=$(sbatch "${SBATCH_ARGS[@]}" "$SIM_ROOT/sgpr_sweep_worker.sbatch")
ENV_FILE="${SGPR_SWEEP_ENV_FILE:-$HOME/aistats2026_sgpr_sweep_v22_jobs.env}"
{
  printf 'export SWEEP_ROOT=%q\n' "$SWEEP_ROOT"
  printf 'export PRESERVED_CAMPAIGN_ROOT=%q\n' "$PRESERVED_CAMPAIGN_ROOT"
  printf 'export PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
  printf 'export EXPECTED_HEAD=%q\n' "$EXPECTED_HEAD"
  printf 'export AIPY=%q\n' "$AIPY"
  printf 'export SWEEP_JOB=%q\n' "$SWEEP_JOB"
} > "$ENV_FILE"

printf 'SWEEP_ROOT=%s\n' "$SWEEP_ROOT"
printf 'PRESERVED_SVGP_ROOT=%s (read-only; no jobs cancelled)\n' \
  "$PRESERVED_CAMPAIGN_ROOT"
printf 'SWEEP_JOB=%s (five resumable workers, throttle=%s)\n' \
  "$SWEEP_JOB" "$CONCURRENCY"
printf 'ENV_FILE=%s\n' "$ENV_FILE"
printf 'watch -n 10 bash %q\n' "$SIM_ROOT/watch_sgpr_sweep_campaign.sh"
