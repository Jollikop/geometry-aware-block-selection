# svgp/analysis

Validators and post-hoc ablations. **Nothing here retrains a model**: every
script reads a finished runs tree and recomputes at the saved parameters, so it
all runs on a laptop against a downloaded campaign.

Shared plumbing is in `_common.py`: `split_for` rebuilds the exact split a run
used, `engine_for` rebuilds its engine, `blocks_for` returns the partition it
conditions on, `iter_runs` walks a tree. Those four used to be copied into
three scripts with three caches.

| Script | Question |
|---|---|
| `paper_status.py` | is the campaign complete, valid and release-ready? |
| `freeze_campaign.py` | which later commits could have changed a frozen campaign? |
| `check_gain_decomposition.py` | does `F_T + 1/2 sum gain(B)` equal the direct block objective, and are the gains non-negative? |
| `check_convergence.py` | has each inducing-point budget actually converged? |
| `screen_leakage.py` | is a dataset's target a function of its own features? |
| `compare_sbt_structure_refresh.py` | does refreshing the geometry during training matter? |
| `sweep_block_size.py` | how much of the available gain does a block of size B capture? |

## `freeze_campaign.py` and the dependency set

`--verify` answers "could this commit have changed that campaign" by comparing
against a declared set of source files. The set is in
`STATIONARY_DEPENDENCIES`, and it must list **every** file whose content can
move a number — including `common/partitions.py`, since every SBT partition
comes from there. Omitting it once made `--verify` report UNAFFECTED for a
commit that changed every SBT result.

The non-stationary kernel is deliberately *not* in the stationary set: it lives
alone in `common/kernels_nonstationary.py` and is reachable only through the
`gibbs` branch of the dispatch, so edits confined to it provably cannot change
a frozen stationary result.
