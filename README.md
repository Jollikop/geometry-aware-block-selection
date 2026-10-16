# Geometry-Aware Block Selection for Sparse Variational Gaussian Processes

Matteo Hoxha and Daniel Paulin, Nanyang Technological University, Singapore.

Code and compact results for the paper *Geometry-Aware Block Selection for
Sparse Variational Gaussian Processes* (preprint on arXiv; the link will be
added here). This repository holds the complete experiment code, the launcher of
every campaign in the paper, and the compact result files that the tables and
figures are built from. Raw run trees (tens of GB of saved parameters and
logs) and the datasets themselves are not included. The repository can rerender
the supplied aggregate plots and run CPU checks on its own; reproducing the
historical GPU runs additionally requires the exact source commits and dataset
caches, plus either running the predecessor stages or supplying their outputs
as described below. A fresh commit made from
this repository has different provenance and must not be described as the original
execution commit.

In the code, `RBT-*` and `SBT-*` are the paper's (R)BT and (S)BT: residual
blocks chosen at random, and blocks selected by Algorithm 1. The paper's
block-diagonal predictor (BLOCK DIAG. PREDICTOR in the figures) is called the
hard-block or one-block PIC predictor in file names (`hard_pic`, `hard_block`).

## One entry point

`REPRODUCE.sh` launches every experiment of the paper, one stage at a time and
in dependency order: `bash REPRODUCE.sh list` prints the stages,
`bash REPRODUCE.sh <stage>` runs one, and `DRY_RUN=1` prints the commands
without running them. Each stage calls the launcher listed under "Campaigns"
below; `bash REPRODUCE.sh check` runs the CPU tests and `bash REPRODUCE.sh
figures` rebuilds the tables and figures from the certified outputs.

## Layout

```
Simulations/                 experiment code (import root: common, sgpr, svgp, analysis)
  common/                    shared by both stacks: data, kernels, block selection,
                             residual terms, predictors, provenance
  sgpr/                      collapsed bounds (SGPR, T-, RBT-, SBT-, E-, METIS-SGPR),
                             Snelson reference, residual illustration
  svgp/                      minibatch variational bounds (SVGP, T-, RBT-, SBT-SVGP),
                             training engine, campaign plans
  svgp/analysis/             validators that recompute at saved parameters (no training)
  analysis/                  suite validation, prediction-only evaluations at saved
                             parameters (block-diagonal predictor, dense reference)
  tests/                     unit and contract tests (CPU only)
  submit_*.sh                Slurm dependency chains, one per campaign
  *_worker.sbatch, */paper_*.sbatch   array workers called by the submitters
  watch_*.sh                 progress monitors for a submitted campaign
  run_submission_checks.sh   everything that runs without a GPU (see below)
  dataset/snelson.csv        the 200-point Snelson data; other datasets are downloaded
transplant300_tools/         evaluation and certification of the 300-epoch sparse
                             SVGP refinement (run from a finished continuation)
Report/Results/              summaries (CSV) and release manifests (JSON)
Report/Results/aggregates/   per-cell aggregates behind the SVGP figures
Report/tables/               LaTeX tables as included in the paper
Report/notes/                figure and table builders for the paper pages
```

Where the paper's objects live in the code:

| Paper | Code |
|---|---|
| Algorithm 1 (inducing-anchor chain and blocks) | `common/partitions.py`: `anchor_order_device`, `sbt_terms_device`, `sbt_blocks_host` |
| Random blocks, (R)BT | `common/partitions.py`: `bt_terms_device`, `bt_blocks_host` |
| Block residual terms of the bound | `common/terms.py` (packing), `sgpr/jax_collapsed.py` (collapsed), `svgp/engine.py` (variational) |
| Sparse and exact predictors | `common/predictors.py` |
| Block-diagonal predictor of the paper (PIC-style residual correction with one assigned block per test point, for the T- and BT-models) | `analysis/hard_block_prediction_campaign.py` (prediction only, at saved parameters); Snelson: `analysis/hard_snelson_prediction.py` |
| Dense-residual reference for the block-diagonal predictor (appendix figure and table) | `analysis/real_predictor_campaign.py` |
| Kernels (RBF, Matérn-5/2, amplitude-RBF, Gibbs) | `common/kernels.py`, `common/kernels_nonstationary.py` |
| SGPR optimisation protocol (L-BFGS-B, refresh schedule, convergence gate) | `sgpr/protocol.py` |
| SVGP campaign grids | `svgp/plans.py` |

## Environment

Python 3.12 (`.python-version`). The release environment is pinned in
`Simulations/requirements.txt`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r Simulations/requirements.txt        # Linux + CUDA 12 (JAX 0.4.30)
# without building dataset caches:  pip install -r Simulations/requirements-nodata.txt
pip install -e Simulations                          # optional; the scripts also run in place
```

On macOS the same file installs the CPU build of JAX; on a Linux machine
without a GPU set `JAX_PLATFORMS=cpu`. The campaigns ran under Slurm on NVIDIA
GPUs, and every run records the GPU model it used.

## Quick check (CPU, a few minutes)

```bash
JAX_PLATFORMS=cpu bash Simulations/run_submission_checks.sh
```

It runs the test suite, compiles every module, parses every launcher, expands
the campaign plans, checks the numerical gain decomposition and that the
method/predictor registries agree between the two stacks.

To run the check from a clean clone of this repository:

```bash
git clone https://github.com/Jollikop/geometry-aware-block-selection.git
cd geometry-aware-block-selection
export PYTHONDONTWRITEBYTECODE=1
JAX_PLATFORMS=cpu bash Simulations/run_submission_checks.sh
```

This is a clean-checkout CPU test, **not** a full GPU replay:
the downloaded dataset caches, historical Git objects, Slurm allocation, and
predecessor checkpoint trees are not in this ZIP.

Small CPU reproductions (from `Simulations/`, seconds each on a laptop):

```bash
cd Simulations
python -m sgpr.snelson.run_snelson                          # Snelson fit, figure and table
python -m sgpr.residual_illustration.run_residual_structure # residual structure on Snelson
python analysis/make_recovery_clouds_preview.py --write-report  # geometric-recovery figure (main text)
```

All three figures are written to `Report/figures/` (created if absent), and
the Snelson summary and manifest are rewritten in place, so a fresh run can be
compared with the shipped files. The residual illustration also keeps its
figure, table and CSV next to its script; the recovery script keeps a preview
copy with `audit.json` and `examples.npz` in `output/pdf/recovery_clouds`
(`--out` to change it).

## Data

Datasets are fetched and cached by `common/datasets.py` (`python -m
common.datasets --help`); each cache is stored with a provenance sidecar
recording its source and sha256, and a run refuses a cache that does not
match. The Electricity series is prepared by `common/prepare_electricity.py`.

| Dataset | Source | Licence |
|---|---|---|
| Snelson (1-D, 200 points) | Snelson and Ghahramani (2006); `dataset/snelson.csv` | distributed with this repository for convenience |
| Kin40k, Elevators, Protein, 3D road | `uci_datasets` package (MIT licence), pinned commit in `requirements.txt` | Protein and 3D road: UCI repository, CC BY 4.0; Kin40k and Elevators: see the original sources listed by `uci_datasets` |
| Pol | OpenML dataset 201 | listed as public on OpenML |
| Superconductivity | UCI repository, id 464 | CC BY 4.0 |
| Electricity load diagrams 2011-2014 | UCI repository, id 321 | CC BY 4.0 |
| Traffic (Metro Interstate Traffic Volume) | UCI repository, id 492 | CC BY 4.0 |

## Campaigns

Every official run records its source commit, whether the tree was clean
(`common/provenance.py:code_state`), the runtime and GPU, and a content hash of
its data split. The launchers pin the commit they were started from and refuse
a dirty tree, so they must be run from a clean git checkout, such as a clone
of this repository. This permits a **new**
run of the packaged code; it does not recreate historical commit IDs or their
exact provenance. Result roots default
to `$HOME/experiment_results/…`; partition and node names in the launchers are
arguments or defaults for the cluster that was used.

| Results in the paper | Submit | Worker(s) | Historical execution identity |
|---|---|---|---|
| RBF Kin40k SGPR comparison, M = 128/256/512, and sparse refinement of the same donors | `submit_sgpr_full_campaign.sh` | `sgpr_full_worker.sbatch` | `2597598a42bfd857b6028b100c545a0a6d3a3270` (release manifest) |
| Original four-kernel donor campaign: Kin40k SGPR at M = 256 and the SVGP grid | `submit_five_worker_campaign.sh` | `paper_five_worker.sbatch` (stages in `sgpr/` and `svgp/`) | `518db753c8a18e5a3d6349da9289568df36cb13e` (root `EXPECTED_HEAD`) |
| Resumed training-only SVGP grid, including 3Droad | `submit_svgp_all_campaign.sh` (wrapper over `submit_svgp_scope_campaign.sh all`) | `svgp_resume_bundle.sbatch`, `svgp_resume_worker.sbatch` | `b4a2ebcad2dba7b2e712b212a749460c5d3e54bf` in the downloaded deferred training records; numerical reference `518db753c8a18e5a3d6349da9289568df36cb13e` |
| Gibbs (S)BT-SVGP with inducing anchors | `submit_gibbs_anchor_campaign.sh`, then `submit_gibbs_anchor_predictions.sh` | `gibbs_anchor_worker.sbatch`, `gibbs_anchor_prediction_worker.sbatch` | `b4a2ebcad2dba7b2e712b212a749460c5d3e54bf` required by `gibbs_anchor_campaign.py`; numerical reference `518db753c8a18e5a3d6349da9289568df36cb13e` |
| 300-epoch sparse SVGP refinement | `submit_svgp_transplant_continue.sh` | `svgp_transplant_continue.sbatch`; evaluation in `transplant300_tools/` | numerical reference `518db753c8a18e5a3d6349da9289568df36cb13e`; original continuation plan records source-file SHA256s, **not** an execution commit |
| Earlier full-test-set evaluation of the SVGP donors (training-time records; superseded in the paper by the block-diagonal predictor below, kept because the grid validators read it) | `submit_svgp_frozen_predictions.sh`, `submit_svgp_missing_predictions.sh` | `svgp_missing_predictions.sbatch` | separate prediction certificates and file hashes; the compact aggregate does not attest one execution commit |
| Block-diagonal predictions: Snelson, RBF Kin40k SGPR at M = 512, and T-, (R)BT- and (S)BT-SVGP over the full grid (3880 records), at the saved checkpoints of the rows above | `analysis/hard_block_prediction_campaign.py plan`, then `sbatch analysis/hard_block_prediction_campaign.sbatch` and `analysis/hard_block_prediction_certify.sbatch`; Snelson: `analysis/hard_snelson_prediction.py` | same | prediction only; plan, evaluator and input-checkpoint hashes in `Report/Results/hard_pic_full_v1/HARD_PIC_PREDICTION_CERTIFICATE.json` (redacted copy) |
| Dense-residual reference for the block-diagonal predictor (Pol SVGP, RBF Kin40k SGPR) | `analysis/real_predictor_campaign.py plan`, then `sbatch analysis/real_predictor_main.sbatch` | same | prediction only; pairing checked in `Report/Results/hard_pic_dense_diagnostic/HARD_PIC_DENSE_DIAGNOSTIC.json` |
| Split versus shared batch on RBF Pol, M = 128 (Figures 5 and 6): 5 curves x 10 seeds, 350 epochs, parameters saved every 20 epochs, block-diagonal RMSE/NLL at every snapshot (900 predictions) | `REPRODUCE.sh split-ablation`, then `REPRODUCE.sh split-certify` (steps in `analysis/README_batch_ablation_block_diag.md`) | `analysis/batch_ablation_block_diag.sbatch` | numerical checkout `b4a2ebcad2dba7b2e712b212a749460c5d3e54bf`; certificate and aggregate in `Report/Results/block_diag_ablation/` |
| Paired Gibbs anchor donor to sparse SVGP, 150 then 300 epochs | `submit_gibbs_anchor_transplant300.sh` | `gibbs_anchor_transplant_{gpu,cpu,certify}.sbatch` | donor numerical reference `518db753c8a18e5a3d6349da9289568df36cb13e`; its compact certificate does not attest a transplant execution commit |

Most submitters name their `watch_*.sh` monitor. The run notes
`GIBBS_ANCHOR_CAMPAIGN.md`, `analysis/TRANSPLANT300.md` and
`analysis/FROZEN_PREDICTION.md` describe those three campaigns step by step.
Validation of a finished tree: `analysis/validate_final_suite.py`,
`svgp/analysis/paper_status.py`, `svgp/analysis/freeze_campaign.py`.

### Historical reproduction and prerequisites

The execution order matters. Build and verify the dataset caches first. Run
the four-kernel donor campaign at its recorded source commit, then its resumed
SVGP training and predictions. These stages generate the RBF donor checkpoints
and reference Gibbs training records used by the Gibbs inducing-anchor
preparation; its preparation program requires the historical execution commit
recorded in `gibbs_anchor_campaign.py` and verifies the initializer hashes.
The Gibbs anchor training then generates the donor checkpoints consumed by the
separate paired transplant campaign, which writes its own 150- and 300-epoch
outputs. Thus an end-to-end run generates the checkpoints it later uses. The
300-epoch continuation of the original donor grid is a different branch. SGPR
v23 is a separate campaign. The repository contains compact aggregates, not the
checkpoint trees needed to restart directly at a downstream stage.

For an **exact historical-source rerun**, use the full Git repository
containing the three commits above; this repository is a current source
snapshot and does not contain that Git history. Before allocating GPUs, check the refs in
that repository:

```bash
: "${HISTORY_REPO:?set HISTORY_REPO to the full Git repository}"
for ref in \
  518db753c8a18e5a3d6349da9289568df36cb13e \
  b4a2ebcad2dba7b2e712b212a749460c5d3e54bf \
  2597598a42bfd857b6028b100c545a0a6d3a3270; do
  git -C "$HISTORY_REPO" cat-file -e "$ref^{commit}" || exit 1
done
```

The dependency order is: dataset caches and the original donor root at
`518db753...` -> SVGP resume/fill from `b4a2ebcad...` into that preserved
root -> separate donor prediction and original-grid 300-epoch transplant
branches. Gibbs-anchor preparation also checks the baseline Gibbs training
records and matching RBF initializer checkpoints in the preserved donor root;
its 320 donor runs use `b4a2ebcad...`. The paired Gibbs-anchor transplant then
starts from those new donors, not from the old residual-probe donor. Both
transplant branches have their own output roots and predictions. Run the RBF
Kin40k SGPR v23 branch independently at `2597598a42bf...`.

The original-grid continuation has source-file hashes in its archived plan;
the paired Gibbs transplant has a plan hash and a certified aggregate. Neither
compact record attests a single execution commit. Do not describe them as
exact commit-level replays from this ZIP. For a historical-source rerun,
compare the archived plans and input checkpoints; otherwise run the packaged
code as a **new** experiment with a new commit and provenance.

On a Slurm cluster, in a clone of this repository and after preparing
the pinned Python environment and dataset cache, the independent SGPR entry
point has this interface (replace the resource values with currently verified
ones):

```bash
export PROJECT_ROOT="$PWD"
export PROJECT_VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
export SCALABLE_GP_DATASET_DIR="$PROJECT_ROOT/Simulations/dataset"
bash Simulations/submit_sgpr_full_campaign.sh <PARTITION> <NODE_OR_DASH> 5
```

The original SVGP lineage starts with
`bash Simulations/submit_five_worker_campaign.sh <PARTITION> <NODE_OR_DASH>`
from a clean checkout of `518db753c8a18e5a3d6349da9289568df36cb13e`,
using the matching cache. Resume into that root using a separate clean
`b4a2ebcad2dba7b2e712b212a749460c5d3e54bf` checkout and
`submit_svgp_all_campaign.sh`; that launcher requires both the source checkout
and the preserved result root. The subsequent launchers are listed in the
table above. For Gibbs, follow
`Simulations/GIBBS_ANCHOR_CAMPAIGN.md`: it supplies the `prepare`, `submit`,
`watch`, and audit commands and explicitly requires the RBF baseline root.
The paired Gibbs transplant launcher requires `AIPY`, `TOOLS_ROOT`,
`ANCHOR_ROOT`, `SCALABLE_GP_DATASET_DIR`, and a new `CAMPAIGN_ROOT`. Its
partition and node choices are specific to the original Slurm cluster; review
them before running on another cluster. The downstream launchers are not
standalone: run their predecessors first or supply the outputs those stages
would generate. Exact GPU-bitwise agreement across
hardware is not promised.

`submit_capped.sh`, `submit_final_campaign.sh`, `submit_sgpr_campaign.sh`,
`submit_sgpr_sweep_campaign.sh`, `submit_svgp_3droad_campaign.sh`,
`submit_svgp_resume_campaign.sh` are retained
historical or auxiliary entry points, **not** the direct route to the final
reported release. The table above identifies the release entry points.

### Freeze source check

`python -m svgp.analysis.freeze_campaign --verify --frozen-commit <COMMIT>
--plan grid` compares two **commits** (`<COMMIT>` and `HEAD`). It does not see
uncommitted edits. Run it from `Simulations/`, repeat for `grid-exploratory`,
and separately inspect `git diff --name-only HEAD -- Simulations/svgp
Simulations/common` and `git status --short -- Simulations/svgp
Simulations/common`. A changed current source does not mutate archived results;
it does mean that current source cannot be claimed identical to the historical
execution source. Use each result's recorded commit for that claim.

### Gibbs paired certificate

`Report/Results/aggregates/GIBBS_ANCHOR_TRANSPLANT_CERTIFICATE.json` is a
redacted copy of the original cluster certificate. Its original SHA256 is
`271f5c760eaff786c45089f6980cbd8c8abd362403547d59b29a409930481bce`;
redacting the donor-root path changes the distributed file's SHA256. The
`aggregate_sha256` field remains unchanged and is checked against the shipped
`gibbs_anchor_transplant300_aggregate.csv` by the archive builder. This
certificate attests to that 320-run transplant aggregate; it is not a substitute
for the unavailable raw run tree.

## Tables and figures

Prediction-only controls (Table 3, Appendix D.3, and the cutoff sweep of
Appendix D.4) are documented in `Simulations/analysis/README_partition_controls.md`.
Use the `partition-controls` and `dense-cutoff-sweep` stages of `REPRODUCE.sh`
with new output roots. E1 predicts with the fitted (R)BT-SVGP donors on blocks
from Algorithm 1 at their saved parameters, and E2 with the fitted (S)BT-SVGP
donors on uniformly random blocks of the same size; with the block-diagonal
records above, this gives both models the same kind of prediction partition
(320 + 320 predictions on the configurations of the main SVGP figure). E4
repeats the 280 dense evaluations with five inverse-root cutoffs on one dense
eigenbasis per reference. The E3 option of that evaluator (direct sparse
prediction of the donors) was not run to completion and is not used in the
paper. No training or existing result is changed. Each control writes an
input/evaluator-bound certificate and a seed summary CSV, shipped in
`Report/Results/partition_controls/`; GPU timings include compilation,
geometry and transfers rather than only accelerator kernels. Table 3 is then
built by `Report/notes/build_partition_controls_table.py`, which reads the
block-diagonal records and the E1/E2 records from `output/results/`.

| In the paper | Built by | Inputs in `Report/Results/` |
|---|---|---|
| RBF Kin40k SGPR table of the main text (block-diagonal predictor) | `Report/notes/build_hard_pic_paper_assets.py` | `hard_pic_full_v1/HARD_PIC_PAPER_ASSETS.json`, `kin40k_sgpr_main_summary.csv` (search times) |
| Sparse SGPR refinement table (appendix) | `analysis/build_rbf_v23_paper_tables.py` (M = 512 rows) | `sgpr_rbf_v23_manifest.json`, `kin40k_sgpr_transplant_rbf_summary.csv` |
| Snelson figures and table (panels arranged for the page from this output) | `sgpr/snelson/make_snelson_figure.sh` → `sgpr/snelson/run_snelson.py` | `snelson_sgpr_t_rbt_sbt_manifest.json`, `snelson_summary_sgpr_t_rbt_sbt.csv` |
| Snelson prediction panel with the block-diagonal predictor | `analysis/hard_snelson_prediction.py`, laid out by `Report/notes/reflow_snelson_prediction.py` (DejaVu fonts) | `hard_pic_full_v1/snelson_hard_prediction.json` |
| Final Snelson training panel layout | `Report/notes/reflow_snelson_training.py` after generating its source PDF; requires a DejaVu font directory | Snelson manifest and summary |
| Residual structure on Snelson | `sgpr/residual_illustration/run_residual_structure.py` | — |
| Geometric recovery (main text) | `analysis/make_recovery_clouds_preview.py` | `geometric_recovery_main.json` |
| SVGP RBF and amplitude-RBF page (main text) and the eight four-kernel appendix pages | `Report/notes/build_hard_pic_paper_assets.py`, which reuses the plotting code of `Report/notes/build_focused_svgp_page.py` and `Report/notes/build_svgp_frozen_appendix.py`, labelled by `Report/notes/render_block_diag_paper_figures.py`; block-diagonal curves from the block-diagonal predictions, sparse curves and objectives from the frozen aggregates | `hard_pic_full_v1/svgp_hard_pic_summary.csv`, `hard_pic_full_v1/HARD_PIC_PAPER_ASSETS.json`, `svgp_frozen_campaign_summary.csv`, `aggregates/*.csv` |
| Block-diagonal versus dense NLL (appendix figure and table) | `Report/notes/build_hard_pic_dense_diagnostic.py` (validation helpers in `Report/notes/build_real_predictor_nll.py`) | `hard_pic_dense_diagnostic/HARD_PIC_DENSE_DIAGNOSTIC.json`; table in `Report/tables/kin40k_sgpr_hard_vs_dense_nll.tex` |
| Split versus shared batch, RBF Pol M = 128 (objective in the main text, test RMSE and NLL in the appendix) | `Report/notes/build_block_diag_ablation.py --input <ablation root>` | `block_diag_ablation/BLOCK_DIAG_ABLATION_CERTIFICATE.json`, `block_diag_ablation/epochs_rbf_pol_M128_block_diag_summary.csv` |
| Final legend wording of the figures (no publication years, (R)/(S)BT names) | `Report/notes/relabel_figures.py`, run after the builders; text-only, curves untouched | — |
| Prediction-partition control (Table 3, Appendix D.3) | `analysis/partition_controls.py` (E1, E2), then `Report/notes/build_partition_controls_table.py` | `partition_controls_summary.csv` (values of Table 3), `partition_controls/E1/`, `partition_controls/E2/` (certificates and seed summaries); table in `Report/tables/partition_controls.tex` |
| Dense cutoff sweep (Appendix D.4) | `analysis/dense_cutoff_sweep.py` (E4) | `partition_controls/E4/DENSE_CUTOFF_CERTIFICATE.json`, `partition_controls/E4/summary.csv` |

The aggregate CSVs are unchanged copies; their sha256 values are those
recorded in `svgp_frozen_campaign_manifest.json`. Absolute paths inside the
manifests and certificates were replaced by `<root>/`.

Every value plotted or tabulated in the paper is in the shipped summaries:
`hard_pic_full_v1/svgp_hard_pic_summary.csv` holds the ten-seed mean and
standard error of each block-diagonal and sparse SVGP metric, and
`HARD_PIC_PAPER_ASSETS.json` records the hashes of the figures and tables
built from it. The builders in `Report/notes/` re-derive these summaries from
the per-run prediction records (3880 block-diagonal records and the paired dense
records), which are not included: their plans embed absolute paths of the
original machines, and redacting them would invalidate the certified plan
hashes. Rerunning the prediction stage with the launchers above recreates
them from the saved checkpoints.

The two Snelson reflow scripts verify the included source PDF against its
manifest before changing only the page layout. They need `pypdf`, `pdfplumber`,
and `reportlab` in addition to the pinned experiment requirements. PDF hashes
can differ because of metadata; compare rendered pages if a recorded artifact
SHA256 does not match.

## Citation

```bibtex
@misc{hoxha2026geometry,
  title  = {Geometry-Aware Block Selection for Sparse Variational Gaussian Processes},
  author = {Hoxha, Matteo and Paulin, Daniel},
  year   = {2026},
  note   = {Preprint}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The datasets keep
their own licences, listed in the table above.
