#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 PARTITION NODE [OUTPUT_DIR]" >&2
  exit 2
fi

PARTITION=$1
NODE=$2
SGPR_DIR="${SGPR_BENCHMARK_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SIM_ROOT="${SIM_ROOT:-$(cd "$SGPR_DIR/.." && pwd)}"
ROOT="${PROJECT_ROOT:-$(cd "$SIM_ROOT/.." && pwd)}"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${3:-$HOME/experiment_results/sgpr_sbt_guided_${STAMP}}"
SLURM_DIR="$OUT_DIR/slurm"
mkdir -p "$SLURM_DIR"

if [[ -z "${PROJECT_VENV_ACTIVATE:-}" ]]; then
  for candidate in \
    "$WORKSPACE_ROOT/.venv/bin/activate" \
    "$WORKSPACE_ROOT/.venv-aistats/bin/activate"
  do
    if [[ -f "$candidate" ]]; then
      PROJECT_VENV_ACTIVATE=$candidate
      break
    fi
  done
fi
test -f "${PROJECT_VENV_ACTIVATE:-}" || {
  echo "No supported project virtual environment found" >&2
  exit 1
}

export PROJECT_ROOT="$ROOT"
export SIM_ROOT SGPR_BENCHMARK_DIR="$SGPR_DIR"
export SGPR_GUIDED_OUT="$OUT_DIR"
export EXPECTED_HEAD="${EXPECTED_HEAD:-$(git -C "$ROOT" rev-parse HEAD)}"
export PROJECT_VENV_ACTIVATE

RUN_JOB=$(sbatch --parsable \
  --partition="$PARTITION" \
  --nodelist="$NODE" \
  --array=0-4%5 \
  --output="$SLURM_DIR/run_%A_%a.out" \
  --export=ALL \
  "$SGPR_DIR/analysis/sbt_guided_sgpr_exploratory.sbatch")

PLOT_JOB=$(sbatch --parsable \
  --dependency="afterok:$RUN_JOB" \
  --partition="$PARTITION" \
  --nodelist="$NODE" \
  --output="$SLURM_DIR/plot_%j.out" \
  --export=ALL \
  "$SGPR_DIR/analysis/sbt_guided_sgpr_summary.sbatch")

ENV_FILE="$HOME/sgpr_sbt_guided_jobs.env"
printf 'export RUN_JOB=%q\nexport PLOT_JOB=%q\nexport SGPR_GUIDED_OUT=%q\nexport SGPR_GUIDED_SLURM=%q\n' \
  "$RUN_JOB" "$PLOT_JOB" "$OUT_DIR" "$SLURM_DIR" > "$ENV_FILE"

echo "RUN_JOB=$RUN_JOB"
echo "PLOT_JOB=$PLOT_JOB"
echo "SGPR_GUIDED_OUT=$OUT_DIR"
echo "Monitor: watch -n 10 bash \"$SGPR_DIR/analysis/watch_sbt_guided_sgpr.sh\""
