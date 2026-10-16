# SVGP transplant: separate extension to 300 epochs

All 5,120 transplants are continued from the saved epoch-150 posterior, including
3droad and all four kernels. The original campaign and predictions are read-only
inputs. Do not label this new campaign release-ready until it has been audited.

## Protocol

- Epochs 151–300 update q and Z only. Kernel/noise/warp parameters (theta) stay
  fixed, checked bit-for-bit at each saved checkpoint.
- The old archive does NOT contain Adam momentum/counters. Adam is initialized
  once at epoch 150; q, Z and theta are retained exactly. This is a documented
  warm continuation, not an exact resumption of uninterrupted 300-epoch Adam.
- Same Adam rates 0.001 for model/Z, natural q rate 0.05, constant schedule and
  existing gradient clipping and minibatch settings. Absolute-epoch random
  batches continue at 150 rather than replaying epochs 1–150. No refresh.
- Parameters AND optimizer state are saved at epochs 200, 250 and 300. A manual
  retry uses the newest validated checkpoint, retaining its optimizer state.
  Work since the most recent 50-epoch checkpoint may need to be repeated.
- Fixed-parameter full-training sparse ELBO/N is evaluated at 150/200/250/300
  using bounded 512-row chunks. It is not the noisy last-epoch minibatch average.
  It does not evaluate test predictions or change optimization/stopping.
- No convergence guarantee, automatic threshold relaxation, test-based model
  selection, automatic job requeue, or original freeze overwrite.
- Prediction is deferred. The epoch-150 prediction files remain valid for the
  OLD posterior only; do not attach them to epoch-300 parameters.

The original engine's training math is unchanged. Its only extension is an
optional checkpoint epoch list, including the final epoch. The standalone code
bundle and numeric package versions are hashed/checked before and after runs.

## Run (GPU cluster, Bash)

First, on the Mac, copy the prepared archive (no git pull required):

```bash
scp $PROJECT_ROOT/output/launchers/svgp-transplant300-v1-20260907.tar.gz \
  gpu-lab:svgp-transplant300-v1-20260907.tar.gz
```

Then on the GPU login machine, extract into a NEW directory:

```bash
mkdir "$HOME/svgp-transplant300-v1" && \
tar -xzf "$HOME/svgp-transplant300-v1-20260907.tar.gz" -C "$HOME/svgp-transplant300-v1"
```

If that directory already exists, do not overwrite it: inspect the earlier
dispatch before launching again. With extraction successful, submit once:

```bash
export AIPY="$PROJECT_ROOT/.venv/bin/python"
export PARTITION=NA100q
export GPU_NODE=node01
bash "$HOME/svgp-transplant300-v1/Simulations/submit_svgp_transplant_continue.sh"
```

The launcher first validates the full parent grid, saved posteriors, lineage,
configuration fingerprints and dataset hashes, without training on the login
node. It then submits five single-GPU array workers, no requeues. Each worker
owns 1,024 disjoint configurations and trains one model at a time in an isolated
Python process. Slurm may leave workers pending until resources become available;
a successful historical GPU probe does not reserve that device for this launch.

Outputs go to a new timestamped `${PARENT_ROOT}_transplant300_*` directory.
`PARENT_ROOT` can explicitly override the original campaign root. The source
worktree and completed prediction roots are never changed.

```bash
watch -n 10 bash "$HOME/svgp-transplant300-v1/Simulations/watch_svgp_transplant_continue.sh"
```

Dispatch details are retained in the new campaign's `jobs.env`; a latest pointer
is `$HOME/aistats2026_transplant300_latest.env`. The watch includes counters for
all three checkpoints, five worker statuses, queue/accounting, and recent logs.
Counters are progress, not a final scientific certification.

## Saved artifacts per run

- `epoch150.json`: deterministic baseline and immutable parent identity.
- `checkpoint_epoch{200,250,300}.npz`: parameters and optimizer state.
- `epoch{200,250,300}.json`: deterministic training ELBO, theta check, checkpoint
  digest and continuation-plan identity.
- `params.npz`: final posterior; `deterministic_curve.npz`: full-training ELBO.
- `curve_from_epoch*.npz` / `attempt_from_epoch*.json`: stochastic curves and
  timing for completed invocation segments. Interrupted work before a checkpoint
  may lack its stochastic curve, but the validated checkpoint survives.
- `training.json`: new configuration, parent linkage, explicit optimizer reset,
  runtime and no claim of certified convergence.
- `COMPLETE.json`: written last, binds the above output files by SHA256. A retry
  verifies this marker before skipping an already completed run.

No figures or test predictions are produced by this launcher. Evaluation and
paper aggregation of the new posterior form a subsequent, separate operation.
