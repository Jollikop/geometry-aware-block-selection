# Frozen prediction-partition controls

These tools never train or edit a donor. Use NEW output roots, not the roots
containing checkpoints or earlier predictions. A plan pins historical source
hashes, checkpoint hashes and commits, baseline prediction/certificate hashes,
and the evaluator snapshot. Use one GPU per worker and the pinned numerical
environment. Do not update evaluator files after making a plan.

1. E1: RBT donors, inducing-anchor prediction blocks.
2. E2: SBT donors, uniform random prediction blocks, seed = donor seed + 10007.
3. E3: all four donor methods, direct standard sparse prediction without refinement.
4. E4: dense-reference inverse-root cutoff sensitivity, one dense eigenbasis per run.

E1/E2 `fig4` is exactly two kernels (RBF, Amplitude-RBF), four datasets
(`pol`, `superconductivity`, `kin40k`, `protein`), four M values and ten seeds:
320 predictions each. E3 `all` is 5120 donors and keeps the certified Gibbs
anchor SBT donors. E4 is the 280 non-sparse dense references, five cutoff scales.

```bash
bash REPRODUCE.sh partition-controls --baseline "$HARD_REFERENCE_ROOT" \
  --experiment E1 --scope fig4 --output "$E1_ROOT"
bash REPRODUCE.sh partition-controls --baseline "$HARD_REFERENCE_ROOT" \
  --experiment E2 --scope fig4 --output "$E2_ROOT"
bash REPRODUCE.sh partition-controls --baseline "$HARD_REFERENCE_ROOT" \
  --experiment E3 --scope all --output "$E3_ROOT"
bash REPRODUCE.sh partition-controls-run --output "$E1_ROOT" --worker 0 --workers 5
bash REPRODUCE.sh partition-controls-certify --output "$E1_ROOT"
bash REPRODUCE.sh dense-cutoff-sweep plan --baseline "$DENSE_REFERENCE_ROOT" --output "$CUTOFF_ROOT"
bash REPRODUCE.sh dense-cutoff-sweep worker --output "$CUTOFF_ROOT" --worker 0 --workers 5
bash REPRODUCE.sh dense-cutoff-sweep certify --output "$CUTOFF_ROOT"
```

The first-batch Slurm launcher executes E1(a) then E2(a) in each of five shards.
Gate E3 on the successful E1/E2 certificate job and E4 on the successful E3
certificate job, rather than merely a successful worker process. A completed
worker is not a certificate. No automatic requeue or deletion is performed.

Per-prediction JSON contains partition source/hash, random seed, variance
validity, minimum raw residual, jitter, and synchronized timing components.
`total_prediction_seconds` includes loading, compilation, geometry, transfers,
and prediction while one GPU is allocated; it is not CUDA-kernel-only time.
Worker state also records full subprocess wall time including startup.
Certificates list every prediction hash, reject invalid predictions explicitly,
and write seed means/SEMs and paired NLL differences in `summary.csv`.

After certification, collect the independent output roots with
`analysis/collect_partition_controls.py --roots <E1> <E2> <E3> <E4> --output <NEW collection root>`.
It writes the combined `partition_controls_summary.csv`, `paired_per_seed.csv`
and a collection certificate. An optional `--sparse-reference-root` supplies
the old deferred standard-SVGP predictions; their config IDs and checkpoint
input hashes must match. Otherwise the frozen standard-SVGP metrics are used
when present. Missing paired references are counted explicitly, never replaced
with another donor. Copy only the new combined CSV into `Report/Results/`.

E4 keeps rejected cutoff variants in its diagnostic certificate (never reports
their NLL as a valid prediction). Scale 10 must reproduce each stored NLL to
absolute tolerance 1e-10. The dense eigendecomposition is shared across all
five variants; block eigendecompositions retain the original BT formula.
