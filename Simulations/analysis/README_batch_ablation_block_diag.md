# RBF–Pol M128 batch-separation diagnostic

This is a separate, reproducible Figure 5/6 experiment. It does not alter the
donor/transplant freeze. It trains five arms, ten seeds per arm for 350 epochs,
snapshots parameters at 18 fixed epochs, and evaluates the paper's
block-diagonal predictor for each snapshot. The `certify` command rejects a
missing seed, epoch, parameter hash, nonfinite metric, or changed source file.

Use the clean numerical checkout at commit
`b4a2ebcad2dba7b2e712b212a749460c5d3e54bf`.

This checkout's `stable` learning-rate profile resolves the RBF model and
inducing-point rate to $5\times10^{-3}$. The older published ablation used
an earlier numerical implementation with a different effective rate, so
this is a new paired replication: replace the objective/RMSE curves together
with the certified NLL curve rather than appending its NLL to the old plot.

Set:

```bash
export AIPY=/path/to/pinned-venv/bin/python
export SOURCE_SIM=/path/to/clean-checkout/Simulations
export TOOLS_DIR=/path/to/this-repository/Simulations/analysis
export ABLATION_ROOT=/path/to/new-results-directory
export SCALABLE_GP_DATASET_DIR=/path/to/Simulations/dataset
mkdir -p "$ABLATION_ROOT/slurm" "$ABLATION_ROOT/state"
```

The source checkout must be clean under `Simulations`. The dataset loader
downloads/uses the same raw Pol cache and records its hash, split indices,
preprocessing protocol, and numerical commit. For a five-GPU Slurm node:

```bash
sbatch --array=0-4%5 \
  --output="$ABLATION_ROOT/slurm/pol_blockdiag_%A_%a.out" \
  --error="$ABLATION_ROOT/slurm/pol_blockdiag_%A_%a.err" \
  --export=ALL,AIPY,SOURCE_SIM,TOOLS_DIR,ABLATION_ROOT,SCALABLE_GP_DATASET_DIR \
  "$TOOLS_DIR/batch_ablation_block_diag.sbatch"
```

Specify a site-appropriate partition/nodelist on `sbatch`. Each array task
uses one GPU and ten seeds; failed seeds are visible in its state file, with
no automatic indefinite retry. To reproduce one run independently:

```bash
"$AIPY" "$TOOLS_DIR/batch_ablation_block_diag.py" train \
  --source-sim "$SOURCE_SIM" --output "$ABLATION_ROOT" \
  --method SBT-SVGP --coupling split --seed 0
"$AIPY" "$TOOLS_DIR/batch_ablation_block_diag.py" evaluate \
  --source-sim "$SOURCE_SIM" --output "$ABLATION_ROOT" \
  --tools "$TOOLS_DIR" --method SBT-SVGP --coupling split --seed 0
```

After all five workers have completed:

```bash
"$AIPY" "$TOOLS_DIR/batch_ablation_block_diag.py" certify \
  --source-sim "$SOURCE_SIM" --output "$ABLATION_ROOT" --tools "$TOOLS_DIR"
```

The certificate covers 50 training histories and 900 epochwise predictions.
Render the objective, RMSE and NLL PDFs from a verified local copy of the
certificate, aggregate and prediction JSON files with
`Report/notes/build_block_diag_ablation.py --input ...`.
