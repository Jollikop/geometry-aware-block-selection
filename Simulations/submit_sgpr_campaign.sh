#!/usr/bin/env bash
# Submit the collapsed-only paper campaign: main table, residual figure,
# Snelson, transplant table, finalize. Concurrency is ARRAY_THROTTLE
# (default 5); set it to the number of GPUs actually free on the node.

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PARTITION NODE [DATASET_CACHE]" >&2
  exit 2
fi
PARTITION=$1
NODE=$2

SIM_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SIM_ROOT/.." && pwd)
SGPR="$SIM_ROOT/sgpr"
# Same default as every sbatch in the chain: the workspace two levels
# above this checkout.
WORKSPACE_ROOT=$(cd "$PROJECT_ROOT/../.." && pwd)
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$WORKSPACE_ROOT/.venv/bin/activate}"
DATASET_CACHE="${3:-${SCALABLE_GP_DATASET_DIR:-$SIM_ROOT/dataset}}"
EXPECTED_HEAD=$(git -C "$PROJECT_ROOT" rev-parse HEAD)

[[ -f "$VENV_ACTIVATE" ]] || { echo "missing environment: $VENV_ACTIVATE" >&2; exit 2; }
[[ -d "$DATASET_CACHE" ]] || { echo "missing dataset cache: $DATASET_CACHE" >&2; exit 2; }
# One owner for "is the experiment source clean": the same tree, with the
# same flags, that common.provenance.code_state() records in every
# manifest. A launcher checking a narrower set can pass a run that the
# validator then rejects as dirty -- which cost one full campaign.
if [[ -n "$(git -C "$SIM_ROOT" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "collapsed experiment source is dirty; commit it before submission" >&2
  exit 2
fi

ARRAY_THROTTLE="${ARRAY_THROTTLE:-5}"
STAMP=$(date +%Y%m%d_%H%M%S)
FINAL_ROOT="${FINAL_ROOT:-$HOME/experiment_results/aistats2026_sgpr_${EXPECTED_HEAD:0:12}_$STAMP}"
mkdir -p "$FINAL_ROOT/slurm" "$FINAL_ROOT/sgpr/runs" \
  "$FINAL_ROOT/sgpr/logs" "$FINAL_ROOT/sgpr/transplant" \
  "$FINAL_ROOT/sgpr/snelson" "$FINAL_ROOT/sgpr/residual_illustration"

# Official jobs fail once on an unusable GPU. They never requeue themselves:
# a manual resume against the SAME FINAL_ROOT reuses only fully valid cached
# runs and makes the campaign state easy to audit.
COMMON="ALL,PROJECT_ROOT=$PROJECT_ROOT,PROJECT_VENV_ACTIVATE=$VENV_ACTIVATE,SCALABLE_GP_DATASET_DIR=$DATASET_CACHE,EXPECTED_HEAD=$EXPECTED_HEAD,GPU_MAX_REQUEUE=0"

SGPR_JOB=$(sbatch --parsable \
  --partition="$PARTITION" --nodelist="$NODE" --array="0-39%$ARRAY_THROTTLE" \
  --output="$FINAL_ROOT/slurm/sgpr_main_%A_%a.out" \
  --export="$COMMON,SGPR_BENCHMARK_DIR=$SGPR,RUNS_DIR=$FINAL_ROOT/sgpr/runs,LOGS_DIR=$FINAL_ROOT/sgpr/logs" \
  "$SGPR/paper_sgpr_main.sbatch")

RESIDUAL_JOB=$(sbatch --parsable --dependency="afterok:$SGPR_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --output="$FINAL_ROOT/slurm/sgpr_residual_%j.out" \
  --export="$COMMON,RESIDUAL_DIR=$SGPR/residual_illustration,SGPR_DONOR_RUNS=$FINAL_ROOT/sgpr/runs,RESIDUAL_OUT=$FINAL_ROOT/sgpr/residual_illustration" \
  "$SGPR/residual_illustration/paper_residual_kin40k.sbatch")

SNELSON_JOB=$(sbatch --parsable --dependency="afterok:$RESIDUAL_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --output="$FINAL_ROOT/slurm/sgpr_snelson_%j.out" \
  --export="$COMMON,SNELSON_DIR=$SGPR/snelson,SNELSON_OUT=$FINAL_ROOT/sgpr/snelson" \
  "$SGPR/snelson/paper_snelson.sbatch")

TRANSPLANT_JOB=$(sbatch --parsable --dependency="afterok:$SNELSON_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" --array="0-39%$ARRAY_THROTTLE" \
  --output="$FINAL_ROOT/slurm/sgpr_transplant_%A_%a.out" \
  --export="$COMMON,SGPR_BENCHMARK_DIR=$SGPR,SGPR_DONOR_RUNS=$FINAL_ROOT/sgpr/runs,SGPR_TRANSPLANT_OUT=$FINAL_ROOT/sgpr/transplant" \
  "$SGPR/paper_sgpr_transplant.sbatch")

FINALIZE_JOB=$(sbatch --parsable --dependency="afterok:$TRANSPLANT_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --output="$FINAL_ROOT/slurm/sgpr_finalize_%j.out" \
  --export="$COMMON,FINAL_ROOT=$FINAL_ROOT" \
  "$SGPR/paper_sgpr_finalize.sbatch")

ENV_FILE="$HOME/aistats2026_sgpr_jobs.env"
{
  printf 'export FINAL_ROOT=%q\n' "$FINAL_ROOT"
  printf 'export PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
  printf 'export SGPR_JOB=%q\n' "$SGPR_JOB"
  printf 'export RESIDUAL_JOB=%q\n' "$RESIDUAL_JOB"
  printf 'export SNELSON_JOB=%q\n' "$SNELSON_JOB"
  printf 'export TRANSPLANT_JOB=%q\n' "$TRANSPLANT_JOB"
  printf 'export FINALIZE_JOB=%q\n' "$FINALIZE_JOB"
} > "$ENV_FILE"

printf 'FINAL_ROOT=%s\n' "$FINAL_ROOT"
printf 'jobs: main=%s residual=%s snelson=%s transplant=%s finalize=%s\n' \
  "$SGPR_JOB" "$RESIDUAL_JOB" "$SNELSON_JOB" "$TRANSPLANT_JOB" "$FINALIZE_JOB"
printf 'watch -n 10 bash %q\n' "$SIM_ROOT/watch_sgpr_campaign.sh"
