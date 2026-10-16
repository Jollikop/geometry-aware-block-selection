#!/usr/bin/env bash
# Resume cached SVGP work without mixing experiment commits. Each Python
# process owns one kernel/dataset/method group so device allocations and JAX
# executables are released before the next group.
#SBATCH --job-name=svgp_resume
#SBATCH --array=0-4%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6-23:59:00
#SBATCH --no-requeue

set -Eeuo pipefail

FINAL_ROOT="${FINAL_ROOT:?set FINAL_ROOT}"
SOURCE_PROJECT_ROOT="${SOURCE_PROJECT_ROOT:?set SOURCE_PROJECT_ROOT}"
EXPECTED_SOURCE_HEAD="${EXPECTED_SOURCE_HEAD:?set EXPECTED_SOURCE_HEAD}"
ORCHESTRATOR_PROJECT_ROOT="${ORCHESTRATOR_PROJECT_ROOT:?set ORCHESTRATOR_PROJECT_ROOT}"
EXPECTED_ORCHESTRATOR_HEAD="${EXPECTED_ORCHESTRATOR_HEAD:?set EXPECTED_ORCHESTRATOR_HEAD}"
RESUME_DATASETS="${RESUME_DATASETS:?set RESUME_DATASETS}"
RUN_CONVERGENCE="${RUN_CONVERGENCE:-0}"
WORKER_ID="${RESUME_WORKER_ID:-${SLURM_ARRAY_TASK_ID:-}}"
ARRAY_ID="${RESUME_ARRAY_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?}}}"
NSHARDS="${NSHARDS:-5}"
RESUME_STAGES="${SVGP_RESUME_STAGES:-stationary extended transplant convergence}"

if [[ -z "$WORKER_ID" ]]; then
  echo "missing worker id: use an array or set RESUME_WORKER_ID" >&2
  exit 2
fi

SOURCE_SIM="$SOURCE_PROJECT_ROOT/Simulations"
ORCHESTRATOR_SIM="$ORCHESTRATOR_PROJECT_ROOT/Simulations"
STATE="$FINAL_ROOT/state/svgp-resume-$ARRAY_ID"

if (( NSHARDS < 1 || NSHARDS > 5 )); then
  echo "NSHARDS must be in [1, 5], found $NSHARDS" >&2
  exit 2
fi
if (( WORKER_ID < 0 || WORKER_ID >= NSHARDS )); then
  echo "worker id $WORKER_ID is outside [0, $((NSHARDS - 1))]" >&2
  exit 2
fi

if [[ "$(git -C "$SOURCE_PROJECT_ROOT" rev-parse HEAD)" != "$EXPECTED_SOURCE_HEAD" ]]; then
  echo "historical source commit changed" >&2
  exit 2
fi
if [[ "$(git -C "$ORCHESTRATOR_PROJECT_ROOT" rev-parse HEAD)" != "$EXPECTED_ORCHESTRATOR_HEAD" ]]; then
  echo "resume orchestrator commit changed" >&2
  exit 2
fi
for simulation_root in "$SOURCE_SIM" "$ORCHESTRATOR_SIM"; do
  if [[ -n "$(git -C "$simulation_root" status --porcelain --untracked-files=all -- .)" ]]; then
    echo "experiment source is dirty: $simulation_root" >&2
    exit 2
  fi
done

mkdir -p "$STATE"
on_exit() {
  local status=$?
  if (( status != 0 )); then
    printf 'worker=%s status=%s time=%s\n' "$WORKER_ID" "$status" "$(date -Is)" \
      > "$STATE/worker.$WORKER_ID.failed"
  fi
}
trap on_exit EXIT

source "${PROJECT_VENV_ACTIVATE:?set PROJECT_VENV_ACTIVATE}"
export GPU_MAX_REQUEUE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# The platform allocator returns buffers between groups. This changes memory
# management, not the numerical protocol or its cache fingerprint.
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1
unset JAX_PLATFORMS || true

source "$SOURCE_SIM/gpu_preflight.sh"
require_gpu pymetis

export WORKER_ID NSHARDS
stage_enabled() {
  [[ " $RESUME_STAGES " == *" $1 "* ]]
}

for stage in stationary extended; do
  if ! stage_enabled "$stage"; then
    echo "===== worker=$WORKER_ID stage=$stage SKIPPED by filter ====="
    continue
  fi
  echo "===== worker=$WORKER_ID stage=$stage datasets=$RESUME_DATASETS ====="
  bash "$ORCHESTRATOR_SIM/run_svgp_resume_stage.sh" "$stage"
  touch "$STATE/$stage.$WORKER_ID.done"
done
touch "$STATE/donors.$WORKER_ID.ready"

if stage_enabled transplant; then
  echo "===== worker=$WORKER_ID waiting-for-donors ====="
  deadline=$(( $(date +%s) + ${DONOR_BARRIER_TIMEOUT_SECONDS:-259200} ))
  while :; do
    ready=$(find "$STATE" -maxdepth 1 -name 'donors.*.ready' 2>/dev/null | wc -l | tr -d ' ')
    (( ready >= NSHARDS )) && break
    failed=$(find "$STATE" -maxdepth 1 -name 'worker.*.failed' 2>/dev/null | wc -l | tr -d ' ')
    if (( failed > 0 )); then
      echo "a donor worker failed; resume this scope after inspecting its log" >&2
      exit 1
    fi
    if (( $(date +%s) >= deadline )); then
      echo "timed out waiting for donor shards ($ready/$NSHARDS)" >&2
      exit 1
    fi
    echo "donor barrier: $ready/$NSHARDS ready"
    sleep 60
  done

  echo "===== worker=$WORKER_ID stage=transplant datasets=$RESUME_DATASETS ====="
  bash "$ORCHESTRATOR_SIM/run_svgp_resume_stage.sh" transplant
  touch "$STATE/transplant.$WORKER_ID.done"
else
  echo "===== worker=$WORKER_ID stage=transplant SKIPPED by filter ====="
fi

if [[ "$RUN_CONVERGENCE" == 1 ]] && stage_enabled convergence; then
  for ((task=WORKER_ID; task<4; task+=NSHARDS)); do
    echo "===== worker=$WORKER_ID stage=convergence task=$task ====="
    env \
      SLURM_ARRAY_TASK_ID="$task" \
      SVGP_BENCHMARK_DIR="$SOURCE_SIM/svgp" \
      PROJECT_ROOT="$SOURCE_PROJECT_ROOT" \
      RUN_TAG=aistats2026_svgp_convergence_v1 \
      RUNS_DIR="$FINAL_ROOT/svgp/convergence/runs" \
      LOGS_DIR="$FINAL_ROOT/svgp/convergence/logs" \
      bash "$SOURCE_SIM/svgp/paper_convergence.sbatch"
  done
  touch "$STATE/convergence.$WORKER_ID.done"
else
  echo "===== worker=$WORKER_ID stage=convergence SKIPPED by filter ====="
fi

touch "$STATE/worker.$WORKER_ID.complete"
echo "===== worker=$WORKER_ID COMPLETE $(date -Is) ====="
