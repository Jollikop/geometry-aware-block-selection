#!/usr/bin/env bash
#SBATCH --job-name=gibbs_anchor_tp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue
set -euo pipefail

: "${STAGE:?}" "${AIPY:?}" "${TOOLS_ROOT:?}" "${ANCHOR_ROOT:?}"
: "${PARENT_ROOT:?}" "${CONT_ROOT:?}" "${PRED_ROOT:?}"
: "${SCALABLE_GP_DATASET_DIR:?}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$TOOLS_ROOT"
export SVGP_NUMERICAL_REFERENCE_HEAD="$(<"$ANCHOR_ROOT/EXPECTED_HEAD")"
unset XLA_PYTHON_CLIENT_MEM_FRACTION

"$AIPY" -c 'import jax; d=jax.devices(); print("assigned devices:", d, flush=True); assert len(d)==1 and d[0].platform=="gpu" and "A100" in d[0].device_kind, d'

case "$STAGE" in
  parent150)
    exec "$AIPY" -u "$TOOLS_ROOT/svgp/run_benchmark.py" \
      --plan custom --run-tag gibbs_anchor_sparse_transplant_v1 \
      --kernels gibbs \
      --datasets pol superconductivity kin40k protein elevators traffic electricity 3droad \
      --methods SBT-SVGP --m 64 128 256 512 --seeds 0 1 2 3 4 5 6 7 8 9 \
      --all-data --epochs 150 --phases '10:theta,z 140:theta' \
      --metrics-at-epochs 0 10 25 50 100 150 \
      --block 500 --ll-batch-size 1000 --blocks-per-step 2 \
      --refresh-every 151 --refresh-until 0 \
      --common-ll-batches --batch-coupling split --penalty-sampling uniform \
      --optimizer adam --lr 0.001 --z-lr 0.001 \
      --lr-profile fixed --lr-schedule constant \
      --q-optimizer natural --natural-lr 0.05 --normalize-objective \
      --warp-gate-fraction 0.75 --sbt-geometry anchor \
      --transplant-to-svgp \
      --init-from-runs-dir "$ANCHOR_ROOT/svgp/extended/runs" \
      --save-params --defer-prediction \
      --full-predictor-max-n 50000 \
      --runs-dir "$PARENT_ROOT/svgp/transplant/runs" \
      --logs-dir "$PARENT_ROOT/svgp/transplant/logs" \
      --num-shards 5 --shard-id "$SLURM_ARRAY_TASK_ID" --no-live
    ;;
  continue300)
    exec "$AIPY" -u "$TOOLS_ROOT/analysis/gibbs_anchor_transplant_scope.py" \
      worker --continuation-tool "$TOOLS_ROOT/analysis/continue_svgp_transplant.py" \
      --output "$CONT_ROOT" --worker "$SLURM_ARRAY_TASK_ID"
    ;;
  predict)
    exec "$AIPY" -u "$TOOLS_ROOT/analysis/evaluate_svgp_frozen.py" \
      --root "$CONT_ROOT" --output "$PRED_ROOT" --source-sim "$TOOLS_ROOT" \
      --stage transplant --runs-file "$CONT_ROOT/runs.txt" \
      --worker-index "$SLURM_ARRAY_TASK_ID" --workers 5 \
      --require-single-gpu --test-chunk 256 --train-chunk 512 \
      --block-batch 4 --factor-cache-mib 128 --reps 10 --reference-points 8
    ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
