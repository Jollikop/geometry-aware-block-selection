# Separate prepared SVGP prediction

`evaluate_svgp_frozen.py` reads the finished campaign, with no optimizer call,
and writes `prediction.json` files into a disjoint output tree. It NEVER writes
`metrics.json`, `training.json`, `params.npz`, `curve.npz` or `final_blocks.npz`
inside the training campaign. The historical predictor remains unchanged.

Transplants use the saved variational `q_mu` and `q_cov`/`q_raw`, i.e. ordinary
SVGP inference, not an analytically recomputed SGPR posterior. Stationary and
extended donor evaluations use their own sparse, diagonal or full fitted-block
predictor. Every training block is included; no top-k approximation is made.

Prepared inference uses `jit`/`lax.map`/`lax.scan` and `vmap` over at most four
blocks by default. It bounds both train and test kernel batches. Projections
are cached once. Cholesky caching is capped at 128 MiB; larger covers stream
factor batches, using each factor across all test chunks. Input arrays, O(NM)
training projections, output arrays and XLA workspace are additional memory.
This is NOT a constant-memory predictor for arbitrary N, M or block widths.

Timings are explicitly separated:

- `prepare_s`: projection/factor preparation (including its compilation).
- `compile_and_first_prediction_s`: compilation and first full evaluation.
- `warm_prediction_ms`: median of ten synchronized calls on a prepared model
  with device inputs/outputs, excluding metrics and host transfers. In streaming
  mode this includes rebuilding each batch of factors.

These warm times must NOT be pooled with the historical `pred_ms`, which includes
model preparation, CPU transfers and metric computation. GPU identity, source
hashes, backend hashes, settings, training config and input artifact hashes are
saved with every result. Eight evenly spaced test points per model are checked
against the frozen predictor (mean and variance) before publishing results.
Synthetic unit tests additionally cover all four campaign kernels, ragged
blocks, non-divisible test chunks, cache/stream modes and both q representations.

An existing output is reused only when its identity matches, otherwise evaluation
refuses to overwrite it. On three consecutive failures the campaign stops; it
never automatically requeues. Rerunning the same command resumes cached outputs.
The final summary is an evaluation-completion report, not a replacement for
`SVGP_REVIEW_FREEZE.json`. Reconcile these sidecars into a new review manifest
only after successful completion; leave the original manifests intact.

## Copy the isolated evaluator from the Mac

```bash
cd $PROJECT_ROOT
ssh gpu-lab 'mkdir -p "$HOME/svgp-prediction-v1"'
rsync -avR \
  Simulations/svgp/fast_prediction.py \
  Simulations/analysis/evaluate_svgp_frozen.py \
  Simulations/submit_svgp_frozen_predictions.sh \
  gpu-lab:svgp-prediction-v1/
```

## On the cluster, AFTER training finishes

Choose a currently usable GPU with `freep --all`. This example requests ONE
A100 on node01; it does not imply that the node is currently available.

```bash
source "$HOME/aistats2026_svgp_all_training_jobs.env"
export AIPY="$PROJECT_ROOT/.venv/bin/python"
export SCALABLE_GP_DATASET_DIR="$PROJECT_ROOT/Simulations/dataset"

# Read-only validation. Do not ignore errors or launch before training ends.
if "$AIPY" "$ORCHESTRATOR_PROJECT_ROOT/Simulations/analysis/validate_svgp_deferred_freeze.py" \
  --root "$FINAL_ROOT" \
  --datasets pol superconductivity kin40k protein elevators traffic electricity 3droad
then
  PRED_ROOT="${FINAL_ROOT}_transplant_predictions_$(date +%Y%m%d_%H%M%S)"
  bash "$HOME/svgp-prediction-v1/Simulations/submit_svgp_frozen_predictions.sh" \
    "$SOURCE_PROJECT_ROOT" "$FINAL_ROOT" "$PRED_ROOT" NA100q node01 transplant
else
  echo "Validation failed: no prediction job submitted."
fi
```

The transplant scope is 5120 evaluations (including 3droad and Gibbs donors),
all using the sparse SVGP predictor. It does NOT evaluate direct donor methods,
train any model, regenerate intermediate-epoch predictions, or change the freeze.
For a deliberately requested donor evaluation use `stationary` or `extended`
as the final argument, each with its own new output directory.

## Opportunistic helpers inside a four-worker training bundle

When workers 2 and 3 have completion markers but workers 0 and 1 are still
training, `run_svgp_prediction_helpers.sh JOB SOURCE ROOT OUTPUT` starts two
exclusive one-GPU steps in the existing allocation. It evaluates only transplant
seeds 2,6 and 3,7 (1024 models each), into disjoint paths in a separate output
root. It does not steal training tasks or change static sharding. It refuses
missing completion markers and five-way schedules. There is no manual GPU index
selection and no `--overlap`; unavailable step resources cause a failure, not
resource sharing with the training steps.

Copy this script alongside the evaluator before use. Keep the wrapper in a
persistent terminal or launch with `nohup`, redirecting its output outside the
training root. Slurm can terminate these extra steps when the parent bundle
exits. Completed atomic prediction files survive. Each subset has its own
summary, which must not be interpreted as a complete 5120-model report.

These measurements carry `measurement_context=concurrent-training`; do not
use them as controlled final paper timings. To finish the same provisional
output after training ends, run the evaluator on a fresh single-GPU allocation
with the SAME options and `--measurement-context concurrent-training`, omitting
`--seeds` to include all ten. Exact matching cached predictions are skipped.
For controlled timings, choose a new output directory. Do not update evaluator
or backend files during evaluation: their hashes are part of result identity.
