#!/usr/bin/env bash
# Internal submitter shared by the normal resume and the isolated 3droad queue.

set -euo pipefail

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "usage: $0 all|non3droad|3droad SOURCE_PROJECT_ROOT FINAL_ROOT PARTITION NODELIST|- [CONCURRENCY] [array|bundle]" >&2
  exit 2
fi

SCOPE=$1
SOURCE_PROJECT_ROOT=$2
FINAL_ROOT=$3
PARTITION=$4
NODELIST=$5
CONCURRENCY=${6:-5}
EXECUTION_MODE=${7:-array}

case "$SCOPE" in
  all)
    RESUME_DATASETS="pol superconductivity kin40k protein elevators traffic electricity 3droad"
    RUN_CONVERGENCE=1
    JOB_NAME=svgp_all
    ENV_FILE_DEFAULT="$HOME/aistats2026_svgp_all_jobs.env"
    ;;
  non3droad)
    RESUME_DATASETS="pol superconductivity kin40k protein elevators traffic electricity"
    RUN_CONVERGENCE=1
    JOB_NAME=svgp_resume
    ENV_FILE_DEFAULT="$HOME/aistats2026_svgp_resume_jobs.env"
    ;;
  3droad)
    RESUME_DATASETS="3droad"
    RUN_CONVERGENCE=0
    JOB_NAME=svgp_3droad
    ENV_FILE_DEFAULT="$HOME/aistats2026_svgp_3droad_jobs.env"
    ;;
  *)
    echo "scope must be all, non3droad, or 3droad" >&2
    exit 2
    ;;
esac
if [[ ! "$CONCURRENCY" =~ ^[1-5]$ ]]; then
  echo "CONCURRENCY must be between 1 and 5" >&2
  exit 2
fi
if [[ "$EXECUTION_MODE" != array && "$EXECUTION_MODE" != bundle ]]; then
  echo "execution mode must be array or bundle" >&2
  exit 2
fi

ORCHESTRATOR_SIM=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ORCHESTRATOR_PROJECT_ROOT=$(cd "$ORCHESTRATOR_SIM/.." && pwd)
SOURCE_SIM="$SOURCE_PROJECT_ROOT/Simulations"
EXPECTED_SOURCE_HEAD=$(git -C "$SOURCE_PROJECT_ROOT" rev-parse HEAD)
EXPECTED_ORCHESTRATOR_HEAD=$(git -C "$ORCHESTRATOR_PROJECT_ROOT" rev-parse HEAD)
VENV_ACTIVATE="${PROJECT_VENV_ACTIVATE:-$(cd "$ORCHESTRATOR_PROJECT_ROOT/../.." && pwd)/.venv/bin/activate}"
DATASET_CACHE="${SCALABLE_GP_DATASET_DIR:-$SOURCE_SIM/dataset}"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "missing environment: $VENV_ACTIVATE" >&2
  exit 2
fi
if [[ ! -d "$DATASET_CACHE" ]]; then
  echo "missing dataset cache: $DATASET_CACHE" >&2
  exit 2
fi
if [[ ! -d "$FINAL_ROOT/svgp/stationary/runs" ]]; then
  echo "missing preserved SVGP campaign: $FINAL_ROOT" >&2
  exit 2
fi
if [[ ! -f "$FINAL_ROOT/EXPECTED_HEAD" ]]; then
  echo "missing campaign provenance: $FINAL_ROOT/EXPECTED_HEAD" >&2
  exit 2
fi
NUMERICAL_REFERENCE_HEAD=$(<"$FINAL_ROOT/EXPECTED_HEAD")
if [[ -n "${SVGP_NUMERICAL_REFERENCE_HEAD:-}" \
      && "$SVGP_NUMERICAL_REFERENCE_HEAD" != "$NUMERICAL_REFERENCE_HEAD" ]]; then
  echo "SVGP_NUMERICAL_REFERENCE_HEAD disagrees with the preserved root" >&2
  echo "results:   $NUMERICAL_REFERENCE_HEAD" >&2
  echo "requested: $SVGP_NUMERICAL_REFERENCE_HEAD" >&2
  exit 2
fi
if [[ "$EXPECTED_SOURCE_HEAD" != "$NUMERICAL_REFERENCE_HEAD" ]]; then
  mapfile -t NUMERICAL_CHANGES < <(
    git -C "$SOURCE_PROJECT_ROOT" diff --name-only \
      "$NUMERICAL_REFERENCE_HEAD..$EXPECTED_SOURCE_HEAD" -- \
      Simulations/common Simulations/svgp |
    grep -v '/Simulations/svgp/run_benchmark.py$' || true
  )
  if (( ${#NUMERICAL_CHANGES[@]} )); then
    echo "execution checkout changes numerical SVGP modules:" >&2
    printf '  %s\n' "${NUMERICAL_CHANGES[@]}" >&2
    exit 2
  fi
fi
for simulation_root in "$SOURCE_SIM" "$ORCHESTRATOR_SIM"; do
  if [[ -n "$(git -C "$simulation_root" status --porcelain --untracked-files=all -- .)" ]]; then
    echo "experiment source is dirty: $simulation_root" >&2
    exit 2
  fi
done

mkdir -p \
  "$FINAL_ROOT/slurm" "$FINAL_ROOT/state" \
  "$FINAL_ROOT/svgp/stationary/logs" \
  "$FINAL_ROOT/svgp/extended/runs" "$FINAL_ROOT/svgp/extended/logs" \
  "$FINAL_ROOT/svgp/transplant/runs" "$FINAL_ROOT/svgp/transplant/logs" \
  "$FINAL_ROOT/svgp/convergence/runs" "$FINAL_ROOT/svgp/convergence/logs"

PREDICTION_CHUNK=${SVGP_PREDICTION_CHUNK:-256}
if [[ ! "$PREDICTION_CHUNK" =~ ^[1-9][0-9]*$ ]]; then
  echo "SVGP_PREDICTION_CHUNK must be a positive integer" >&2
  exit 2
fi
PREDICTION_MODE=${SVGP_RESUME_PREDICTION_MODE:-full}
PREDICTION_REPS=${SVGP_PREDICTION_REPS:-10}
if [[ "$PREDICTION_MODE" != full && "$PREDICTION_MODE" != defer \
      && "$PREDICTION_MODE" != prediction-only ]]; then
  echo "SVGP_RESUME_PREDICTION_MODE must be full, defer, or prediction-only" >&2
  exit 2
fi
if [[ ! "$PREDICTION_REPS" =~ ^[0-9]+$ ]]; then
  echo "SVGP_PREDICTION_REPS must be a non-negative integer" >&2
  exit 2
fi
RESUME_STAGES="${SVGP_RESUME_STAGES:-stationary extended transplant convergence}"
STATIONARY_KERNELS="${SVGP_RESUME_STATIONARY_KERNELS:-rbf matern52}"
EXTENDED_KERNELS="${SVGP_RESUME_EXTENDED_KERNELS:-amplitude-rbf gibbs}"
TRANSPLANT_KERNELS="${SVGP_RESUME_TRANSPLANT_KERNELS:-rbf matern52 amplitude-rbf gibbs}"

validate_words() {
  local label=$1 allowed=$2 values=$3 value
  [[ -n "$values" ]] || { echo "$label must not be empty" >&2; exit 2; }
  for value in $values; do
    if [[ " $allowed " != *" $value "* ]]; then
      echo "$label contains unsupported value: $value" >&2
      exit 2
    fi
  done
}
validate_words SVGP_RESUME_STAGES \
  "stationary extended transplant convergence" "$RESUME_STAGES"
validate_words SVGP_RESUME_STATIONARY_KERNELS \
  "rbf matern52" "$STATIONARY_KERNELS"
validate_words SVGP_RESUME_EXTENDED_KERNELS \
  "amplitude-rbf gibbs" "$EXTENDED_KERNELS"
validate_words SVGP_RESUME_TRANSPLANT_KERNELS \
  "rbf matern52 amplitude-rbf gibbs" "$TRANSPLANT_KERNELS"

COMMON_EXPORT="ALL,FINAL_ROOT=$FINAL_ROOT,SOURCE_PROJECT_ROOT=$SOURCE_PROJECT_ROOT,EXPECTED_SOURCE_HEAD=$EXPECTED_SOURCE_HEAD,SVGP_NUMERICAL_REFERENCE_HEAD=$NUMERICAL_REFERENCE_HEAD,ORCHESTRATOR_PROJECT_ROOT=$ORCHESTRATOR_PROJECT_ROOT,EXPECTED_ORCHESTRATOR_HEAD=$EXPECTED_ORCHESTRATOR_HEAD,PROJECT_VENV_ACTIVATE=$VENV_ACTIVATE,SCALABLE_GP_DATASET_DIR=$DATASET_CACHE,RESUME_DATASETS=$RESUME_DATASETS,RUN_CONVERGENCE=$RUN_CONVERGENCE,NSHARDS=$CONCURRENCY,SVGP_PREDICTION_CHUNK=$PREDICTION_CHUNK,SVGP_RESUME_PREDICTION_MODE=$PREDICTION_MODE,SVGP_PREDICTION_REPS=$PREDICTION_REPS,SVGP_RESUME_STAGES=$RESUME_STAGES,SVGP_RESUME_STATIONARY_KERNELS=$STATIONARY_KERNELS,SVGP_RESUME_EXTENDED_KERNELS=$EXTENDED_KERNELS,SVGP_RESUME_TRANSPLANT_KERNELS=$TRANSPLANT_KERNELS"
SBATCH_ARGS=(
  --parsable
  --job-name="$JOB_NAME"
  --partition="$PARTITION"
  --export="$COMMON_EXPORT"
)
if [[ "$NODELIST" != "-" ]]; then
  SBATCH_ARGS+=(--nodelist="$NODELIST")
fi
if [[ -n "${SVGP_RESUME_DEPENDENCY:-}" ]]; then
  SBATCH_ARGS+=(--dependency="$SVGP_RESUME_DEPENDENCY")
fi

if [[ "$EXECUTION_MODE" == bundle ]]; then
  SBATCH_ARGS+=(
    --gres="gpu:$CONCURRENCY"
    --cpus-per-task="$((4 * CONCURRENCY))"
    --mem="$((64 * CONCURRENCY))G"
    --output="$FINAL_ROOT/slurm/${JOB_NAME}_%j_bundle.out"
  )
  RESUME_JOB=$(sbatch "${SBATCH_ARGS[@]}" "$ORCHESTRATOR_SIM/svgp_resume_bundle.sbatch")
else
  SBATCH_ARGS+=(
    --array="0-$((CONCURRENCY - 1))%$CONCURRENCY"
    --output="$FINAL_ROOT/slurm/${JOB_NAME}_%A_%a.out"
  )
  RESUME_JOB=$(sbatch "${SBATCH_ARGS[@]}" "$ORCHESTRATOR_SIM/svgp_resume_worker.sbatch")
fi
ENV_FILE="${SVGP_RESUME_ENV_FILE:-$ENV_FILE_DEFAULT}"
{
  printf 'export FINAL_ROOT=%q\n' "$FINAL_ROOT"
  printf 'export SOURCE_PROJECT_ROOT=%q\n' "$SOURCE_PROJECT_ROOT"
  printf 'export ORCHESTRATOR_PROJECT_ROOT=%q\n' "$ORCHESTRATOR_PROJECT_ROOT"
  printf 'export EXPECTED_SOURCE_HEAD=%q\n' "$EXPECTED_SOURCE_HEAD"
  printf 'export SVGP_NUMERICAL_REFERENCE_HEAD=%q\n' "$NUMERICAL_REFERENCE_HEAD"
  printf 'export EXPECTED_ORCHESTRATOR_HEAD=%q\n' "$EXPECTED_ORCHESTRATOR_HEAD"
  printf 'export RESUME_DATASETS=%q\n' "$RESUME_DATASETS"
  printf 'export RESUME_SCOPE=%q\n' "$SCOPE"
  printf 'export RESUME_JOB=%q\n' "$RESUME_JOB"
  printf 'export RESUME_EXECUTION_MODE=%q\n' "$EXECUTION_MODE"
  printf 'export SVGP_PREDICTION_CHUNK=%q\n' "$PREDICTION_CHUNK"
  printf 'export SVGP_RESUME_PREDICTION_MODE=%q\n' "$PREDICTION_MODE"
  printf 'export SVGP_PREDICTION_REPS=%q\n' "$PREDICTION_REPS"
  printf 'export SVGP_RESUME_DEPENDENCY=%q\n' "${SVGP_RESUME_DEPENDENCY:-}"
  printf 'export SVGP_RESUME_STAGES=%q\n' "$RESUME_STAGES"
  printf 'export SVGP_RESUME_STATIONARY_KERNELS=%q\n' "$STATIONARY_KERNELS"
  printf 'export SVGP_RESUME_EXTENDED_KERNELS=%q\n' "$EXTENDED_KERNELS"
  printf 'export SVGP_RESUME_TRANSPLANT_KERNELS=%q\n' "$TRANSPLANT_KERNELS"
} > "$ENV_FILE"

{
  printf 'scope=%s\n' "$SCOPE"
  printf 'datasets=%s\n' "$RESUME_DATASETS"
  printf 'execution_source=%s\n' "$SOURCE_PROJECT_ROOT"
  printf 'execution_commit=%s\n' "$EXPECTED_SOURCE_HEAD"
  printf 'numerical_reference_commit=%s\n' "$NUMERICAL_REFERENCE_HEAD"
  printf 'orchestrator_commit=%s\n' "$EXPECTED_ORCHESTRATOR_HEAD"
  printf 'execution_mode=%s\n' "$EXECUTION_MODE"
  printf 'prediction_chunk=%s\n' "$PREDICTION_CHUNK"
  printf 'prediction_mode=%s\n' "$PREDICTION_MODE"
  printf 'prediction_reps=%s\n' "$PREDICTION_REPS"
  printf 'dependency=%s\n' "${SVGP_RESUME_DEPENDENCY:-none}"
  printf 'stages=%s\n' "$RESUME_STAGES"
  printf 'stationary_kernels=%s\n' "$STATIONARY_KERNELS"
  printf 'extended_kernels=%s\n' "$EXTENDED_KERNELS"
  printf 'transplant_kernels=%s\n' "$TRANSPLANT_KERNELS"
} > "$FINAL_ROOT/state/svgp-resume-$RESUME_JOB.manifest"

printf 'FINAL_ROOT=%s (existing metrics preserved)\n' "$FINAL_ROOT"
printf 'SOURCE_HEAD=%s\n' "$EXPECTED_SOURCE_HEAD"
printf 'NUMERICAL_REFERENCE_HEAD=%s\n' "$NUMERICAL_REFERENCE_HEAD"
printf 'RESUME_JOB=%s scope=%s workers=%s mode=%s prediction=%s reps=%s chunk=%s\n' \
  "$RESUME_JOB" "$SCOPE" "$CONCURRENCY" "$EXECUTION_MODE" \
  "$PREDICTION_MODE" "$PREDICTION_REPS" "$PREDICTION_CHUNK"
printf 'ENV_FILE=%s\n' "$ENV_FILE"
printf 'watch -n 10 env SVGP_RESUME_ENV_FILE=%q bash %q\n' \
  "$ENV_FILE" "$ORCHESTRATOR_SIM/watch_svgp_resume_campaign.sh"
