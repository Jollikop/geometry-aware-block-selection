#!/usr/bin/env bash
# Resume one SVGP stage with the historical experiment source that owns the
# cached fingerprints. Filters change only task enumeration: every per-run
# configuration remains the original v12 protocol.

set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 stationary|extended|transplant" >&2
  exit 2
fi

STAGE=$1
SOURCE_PROJECT_ROOT="${SOURCE_PROJECT_ROOT:?set SOURCE_PROJECT_ROOT}"
FINAL_ROOT="${FINAL_ROOT:?set FINAL_ROOT}"
EXPECTED_SOURCE_HEAD="${EXPECTED_SOURCE_HEAD:?set EXPECTED_SOURCE_HEAD}"
WORKER_ID="${WORKER_ID:?set WORKER_ID}"
NSHARDS="${NSHARDS:-5}"
RESUME_DATASETS="${RESUME_DATASETS:?set RESUME_DATASETS}"
SOURCE_SIM="$SOURCE_PROJECT_ROOT/Simulations"
SOURCE_SVGP="$SOURCE_SIM/svgp"
RUNNER="$SOURCE_SVGP/run_benchmark.py"
RESUME_RUNNER="$ORCHESTRATOR_PROJECT_ROOT/Simulations/svgp_resume_runner.py"
PREDICTION_CHUNK="${SVGP_PREDICTION_CHUNK:-256}"
PREDICTION_MODE="${SVGP_RESUME_PREDICTION_MODE:-full}"
PREDICTION_REPS="${SVGP_PREDICTION_REPS:-10}"
STATIONARY_KERNEL_WORDS="${SVGP_RESUME_STATIONARY_KERNELS:-rbf matern52}"
EXTENDED_KERNEL_WORDS="${SVGP_RESUME_EXTENDED_KERNELS:-amplitude-rbf gibbs}"
TRANSPLANT_KERNEL_WORDS="${SVGP_RESUME_TRANSPLANT_KERNELS:-rbf matern52 amplitude-rbf gibbs}"

if [[ "$(git -C "$SOURCE_PROJECT_ROOT" rev-parse HEAD)" != "$EXPECTED_SOURCE_HEAD" ]]; then
  echo "historical source commit changed" >&2
  exit 2
fi
if [[ -n "$(git -C "$SOURCE_SIM" status --porcelain --untracked-files=all -- .)" ]]; then
  echo "historical experiment source is dirty" >&2
  exit 2
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "missing historical runner: $RUNNER" >&2
  exit 2
fi
if [[ ! -f "$RESUME_RUNNER" ]]; then
  echo "missing resume wrapper: $RESUME_RUNNER" >&2
  exit 2
fi
if [[ ! "$PREDICTION_CHUNK" =~ ^[1-9][0-9]*$ ]]; then
  echo "SVGP_PREDICTION_CHUNK must be a positive integer" >&2
  exit 2
fi
if [[ "$PREDICTION_MODE" != full && "$PREDICTION_MODE" != defer \
      && "$PREDICTION_MODE" != prediction-only ]]; then
  echo "SVGP_RESUME_PREDICTION_MODE must be full, defer, or prediction-only" >&2
  exit 2
fi
if [[ ! "$PREDICTION_REPS" =~ ^[0-9]+$ ]]; then
  echo "SVGP_PREDICTION_REPS must be a non-negative integer" >&2
  exit 2
fi

PREDICTION_ARGS=()
case "$PREDICTION_MODE" in
  defer) PREDICTION_ARGS+=(--defer-prediction) ;;
  prediction-only)
    PREDICTION_ARGS+=(--prediction-only --prediction-reps "$PREDICTION_REPS")
    ;;
esac

export PYTHONPATH="$SOURCE_SIM${PYTHONPATH:+:$PYTHONPATH}"

run_group() {
  python -u "$RESUME_RUNNER" \
    --source-runner "$RUNNER" \
    --prediction-chunk "$PREDICTION_CHUNK" \
    --orchestrator-commit "$EXPECTED_ORCHESTRATOR_HEAD" \
    -- "$@" \
    --num-shards 1 \
    --shard-id 0 \
    "${PREDICTION_ARGS[@]}" \
    --no-live
}

run_stationary_group() {
  local kernel=$1 dataset=$2 method=$3 seed=$4
  echo "===== stationary kernel=$kernel dataset=$dataset method=$method seed=$seed worker=$WORKER_ID/$NSHARDS ====="
  run_group \
    --plan grid \
    --run-tag aistats2026_svgp_stationary_v12 \
    --kernels "$kernel" \
    --datasets "$dataset" \
    --methods "$method" \
    --seeds "$seed" \
    --all-data \
    --epochs 350 \
    --block 500 \
    --ll-batch-size 1000 \
    --blocks-per-step 2 \
    --refresh-every 40 \
    --refresh-until 200 \
    --common-ll-batches \
    --batch-coupling split \
    --penalty-sampling uniform \
    --optimizer adam \
    --lr 0.005 \
    --z-lr 0.005 \
    --lr-profile fixed \
    --lr-schedule constant \
    --q-optimizer natural \
    --natural-lr 0.05 \
    --sbt-geometry anchor \
    --normalize-objective \
    --save-params \
    --full-predictor-max-n 50000 \
    --runs-dir "$FINAL_ROOT/svgp/stationary/runs" \
    --logs-dir "$FINAL_ROOT/svgp/stationary/logs"
}

run_extended_group() {
  local kernel=$1 dataset=$2 method=$3 seed=$4
  local -a kernel_args
  local run_tag
  if [[ "$kernel" == gibbs ]]; then
    run_tag=aistats2026_svgp_gibbs_rbf_warm300_v13
    kernel_args=(
      --epochs 300
      --refresh-every 40
      --refresh-until 81
      --init-from-kernel rbf
      --init-from-runs-dir "$FINAL_ROOT/svgp/stationary/runs"
      --warp-gate-fraction 0.0
    )
  else
    run_tag=aistats2026_svgp_extended_v12
    kernel_args=(
      --epochs 500
      --refresh-every 40
      --refresh-until 500
      --warp-gate-fraction 0.0
    )
  fi
  echo "===== extended kernel=$kernel dataset=$dataset method=$method seed=$seed worker=$WORKER_ID/$NSHARDS ====="
  run_group \
    --plan grid-exploratory \
    --run-tag "$run_tag" \
    --kernels "$kernel" \
    --datasets "$dataset" \
    --methods "$method" \
    --seeds "$seed" \
    --all-data \
    --block 500 \
    --ll-batch-size 1000 \
    --blocks-per-step 2 \
    --common-ll-batches \
    --batch-coupling split \
    --penalty-sampling uniform \
    --optimizer adam \
    --lr 0.005 \
    --z-lr 0.005 \
    --lr-profile fixed \
    --lr-schedule constant \
    --q-optimizer natural \
    --natural-lr 0.05 \
    --sbt-geometry anchor \
    --normalize-objective \
    "${kernel_args[@]}" \
    --swa-fraction 0.0 \
    --save-params \
    --full-predictor-max-n 50000 \
    --runs-dir "$FINAL_ROOT/svgp/extended/runs" \
    --logs-dir "$FINAL_ROOT/svgp/extended/logs"
}

run_transplant_group() {
  local kernel=$1 dataset=$2 donor=$3 source_runs=$4 seed=$5
  echo "===== transplant kernel=$kernel dataset=$dataset donor=$donor seed=$seed worker=$WORKER_ID/$NSHARDS ====="
  run_group \
    --plan custom \
    --run-tag aistats2026_svgp_sparse_transplant_v1 \
    --kernels "$kernel" \
    --datasets "$dataset" \
    --methods "$donor" \
    --m 64 128 256 512 \
    --seeds "$seed" \
    --all-data \
    --epochs 150 \
    --phases "10:theta,z 140:theta" \
    --metrics-at-epochs 0 10 25 50 100 150 \
    --block 500 \
    --ll-batch-size 1000 \
    --blocks-per-step 2 \
    --refresh-every 151 \
    --refresh-until 0 \
    --common-ll-batches \
    --batch-coupling split \
    --penalty-sampling uniform \
    --optimizer adam \
    --lr 0.001 \
    --z-lr 0.001 \
    --lr-profile fixed \
    --lr-schedule constant \
    --q-optimizer natural \
    --natural-lr 0.05 \
    --normalize-objective \
    --transplant-to-svgp \
    --init-from-runs-dir "$source_runs" \
    --save-params \
    --record-history \
    --full-predictor-max-n 50000 \
    --runs-dir "$FINAL_ROOT/svgp/transplant/runs" \
    --logs-dir "$FINAL_ROOT/svgp/transplant/logs"
}

METHODS=(SVGP T-SVGP RBT-SVGP SBT-SVGP)
read -ra DATASETS <<< "$RESUME_DATASETS"
read -ra STATIONARY_KERNELS <<< "$STATIONARY_KERNEL_WORDS"
read -ra EXTENDED_KERNELS <<< "$EXTENDED_KERNEL_WORDS"
read -ra TRANSPLANT_KERNELS <<< "$TRANSPLANT_KERNEL_WORDS"
failures=0

case "$STAGE" in
  stationary)
    for dataset in "${DATASETS[@]}"; do
      for kernel in "${STATIONARY_KERNELS[@]}"; do
        for method in "${METHODS[@]}"; do
          for ((seed=WORKER_ID; seed<10; seed+=NSHARDS)); do
            run_stationary_group "$kernel" "$dataset" "$method" "$seed" \
              || failures=$((failures + 1))
          done
        done
      done
    done
    ;;
  extended)
    for dataset in "${DATASETS[@]}"; do
      for kernel in "${EXTENDED_KERNELS[@]}"; do
        for method in "${METHODS[@]}"; do
          for ((seed=WORKER_ID; seed<10; seed+=NSHARDS)); do
            run_extended_group "$kernel" "$dataset" "$method" "$seed" \
              || failures=$((failures + 1))
          done
        done
      done
    done
    ;;
  transplant)
    for dataset in "${DATASETS[@]}"; do
      for kernel in "${TRANSPLANT_KERNELS[@]}"; do
        if [[ "$kernel" == rbf || "$kernel" == matern52 ]]; then
          source_runs="$FINAL_ROOT/svgp/stationary/runs"
        else
          source_runs="$FINAL_ROOT/svgp/extended/runs"
        fi
        for donor in "${METHODS[@]}"; do
          for ((seed=WORKER_ID; seed<10; seed+=NSHARDS)); do
            run_transplant_group \
              "$kernel" "$dataset" "$donor" "$source_runs" "$seed" \
              || failures=$((failures + 1))
          done
        done
      done
    done
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac

if (( failures > 0 )); then
  echo "$STAGE: $failures group(s) still contain failed runs" >&2
  exit 1
fi
echo "===== $STAGE shard=$WORKER_ID/$NSHARDS COMPLETE ====="
