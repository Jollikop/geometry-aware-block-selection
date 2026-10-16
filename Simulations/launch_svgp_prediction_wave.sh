#!/usr/bin/env bash
# Two active steps inside the existing 4-GPU bundle; three finishing jobs
# wait for that allocation to terminate. Never changes the frozen evaluator.
set -euo pipefail
MODE=${1:?launch or active}
PARENT=${2:?parent job}
[[ "$PARENT" =~ ^[0-9]+$ ]] || exit 2
source "$HOME/aistats2026_svgp_all_training_jobs.env"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export AIPY="${AIPY:?set AIPY to the python of the release environment}"
export SCALABLE_GP_DATASET_DIR="${SCALABLE_GP_DATASET_DIR:-$HERE/dataset}"
OUTPUT="${FINAL_ROOT}_transplant_live_eval_$PARENT"
EVALUATOR="${EVALUATOR:-$HERE/analysis/evaluate_svgp_frozen.py}"
EVALUATOR_SHA=2a1a6e5ef17e64ca161da74b1bce431a2a45894771ff5ea61f07b7133cad47ed
[[ -d "$OUTPUT/slurm" && -x "$AIPY" && -f "$EVALUATOR" ]]
[[ "$(sha256sum "$EVALUATOR" | cut -d ' ' -f1)" == "$EVALUATOR_SHA" ]]
BASE=(env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
  PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  "SCALABLE_GP_DATASET_DIR=$SCALABLE_GP_DATASET_DIR"
  "$AIPY" "$HERE/analysis/run_svgp_prediction_wave.py"
  --root "$FINAL_ROOT" --output "$OUTPUT" --source-sim "$SOURCE_PROJECT_ROOT/Simulations"
  --evaluator "$EVALUATOR" --evaluator-sha "$EVALUATOR_SHA")
if [[ "$MODE" == active ]]; then
  pids=()
  for worker in 0 1; do
    srun --jobid="$PARENT" --immediate=20 --exclusive --exact \
      --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=64G \
      --output="$OUTPUT/slurm/remaining_${PARENT}_${worker}_%s.out" \
      "${BASE[@]}" --mode active --worker "$worker" --workers 2 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  echo "Active wave ended status=$status. Queued finishing jobs will verify and resume."
  exit "$status"
fi
[[ "$MODE" == launch && "$RESUME_JOB" == "$PARENT" ]]
[[ "$(squeue -j "$PARENT" -h -o '%T')" == RUNNING ]]
[[ "$(squeue -j "$PARENT" -h -o '%u')" == "$(id -un)" ]]
# mkdir is an atomic duplicate-dispatch guard. Never remove it automatically.
mkdir "$OUTPUT/wave_dispatch_$PARENT"
mkdir -p "$OUTPUT/wave_state"
for worker in 0 1 2; do
  printf -v command '%q ' "${BASE[@]}" --mode finish --worker "$worker" --workers 3
  job=$(sbatch --parsable --job-name="svgp_pred_finish_$worker" \
    --dependency="afterany:$PARENT" --kill-on-invalid-dep=yes \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=64G \
    --partition=NA100q --nodelist=node01 --time=2-00:00:00 --no-requeue \
    --output="$OUTPUT/slurm/finish_${worker}_%j.out" --wrap="exec $command")
  printf '%s\n' "$job" >> "$OUTPUT/wave_dispatch_$PARENT/queued_jobs.txt"
  printf 'QUEUED worker=%s job=%s dependency=afterany:%s\n' "$worker" "$job" "$PARENT"
done
nohup bash "$0" active "$PARENT" \
  >> "$OUTPUT/wave_dispatch_$PARENT/active_launcher.log" 2>&1 < /dev/null &
printf 'Active launcher PID=%s\nOutput=%s\n' "$!" "$OUTPUT"
