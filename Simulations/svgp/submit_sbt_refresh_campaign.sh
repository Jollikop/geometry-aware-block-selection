#!/usr/bin/env bash
# Submit the uniform-cycle refresh ablation, complete Gibbs SBT rerun and
# adaptive top-eight prediction with at most five concurrent GPUs.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM_ROOT=${SIM_ROOT:-$(cd "$HERE/.." && pwd)}
export PYTHONPATH="$SIM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PARTITION=${PARTITION:-PA10080q}
NODE=${NODE:-node04}
EXPECTED_HEAD=${EXPECTED_HEAD:-$(git -C "$HERE" rev-parse HEAD)}
LAUNCHER="$HERE/paper_sbt_structure_refresh.sbatch"
ELECTRICITY_FILE="$SIM_ROOT/dataset/electricity/data_v2.csv.gz"

if [[ ! -f "$ELECTRICITY_FILE" ]]; then
  echo "Missing manual Electricity source: $ELECTRICITY_FILE" >&2
  echo "Copy the frozen data_v2.csv.gz before submission." >&2
  exit 1
fi

if [[ -n "$(git -C "$HERE" status --porcelain --untracked-files=all -- \
    "$HERE" "$SIM_ROOT/common")" ]]; then
  echo "Experiment-source changes detected; commit them before submission." >&2
  exit 1
fi

SMOKE_RUNS="$HERE/runs_structure_refresh/sbt_gibbs_uniform_refresh_smoke_v1"
ABLATION_RUNS="$HERE/runs_structure_refresh/sbt_rbf_uniform_refresh_ablation_v1"
GIBBS_REFRESH_RUNS="$HERE/runs_structure_refresh/sbt_gibbs_uniform_refresh_v1"

SMOKE_JOB=$(sbatch --parsable \
  --partition="$PARTITION" --nodelist="$NODE" \
  --array=0 --time=00:30:00 \
  --output="$HERE/slurm_sbt_refresh_smoke_%A_%a.out" \
  --export=ALL,MODE=gibbs-smoke,NSHARDS=1,RUN_TAG=sbt_gibbs_uniform_refresh_smoke_v1,RUNS_DIR="$SMOKE_RUNS",EXPECTED_HEAD="$EXPECTED_HEAD" \
  "$LAUNCHER")

ABLATION_JOB=$(sbatch --parsable \
  --dependency="afterok:$SMOKE_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --array=0-4%5 --time=12:00:00 \
  --output="$HERE/slurm_sbt_refresh_ablation_%A_%a.out" \
  --export=ALL,MODE=rbf-ablation,NSHARDS=5,RUN_TAG=sbt_rbf_uniform_refresh_ablation_v1,RUNS_DIR="$ABLATION_RUNS",EXPECTED_HEAD="$EXPECTED_HEAD" \
  "$LAUNCHER")

GIBBS_REFRESH_JOB=$(sbatch --parsable \
  --dependency="afterok:$ABLATION_JOB" \
  --partition="$PARTITION" --nodelist="$NODE" \
  --array=0-4%5 --time=3-00:00:00 \
  --output="$HERE/slurm_sbt_refresh_gibbs_%A_%a.out" \
  --export=ALL,MODE=gibbs,NSHARDS=5,RUN_TAG=sbt_gibbs_uniform_refresh_v1,RUNS_DIR="$GIBBS_REFRESH_RUNS",EXPECTED_HEAD="$EXPECTED_HEAD" \
  "$LAUNCHER")

ENV_FILE=${SBT_REFRESH_ENV_FILE:-$HOME/sbt_refresh_jobs.env}
{
  printf 'export SMOKE_JOB=%q\n' "$SMOKE_JOB"
  printf 'export ABLATION_JOB=%q\n' "$ABLATION_JOB"
  printf 'export GIBBS_REFRESH_JOB=%q\n' "$GIBBS_REFRESH_JOB"
  printf 'export SMOKE_RUNS=%q\n' "$SMOKE_RUNS"
  printf 'export ABLATION_RUNS=%q\n' "$ABLATION_RUNS"
  printf 'export GIBBS_REFRESH_RUNS=%q\n' "$GIBBS_REFRESH_RUNS"
} > "$ENV_FILE"

echo "SMOKE_JOB=$SMOKE_JOB"
echo "ABLATION_JOB=$ABLATION_JOB"
echo "GIBBS_REFRESH_JOB=$GIBBS_REFRESH_JOB"
echo "Saved job environment to $ENV_FILE"

squeue -j "$SMOKE_JOB,$ABLATION_JOB,$GIBBS_REFRESH_JOB" \
  -o "%.18i %.20j %.9T %.10M %.24R"
