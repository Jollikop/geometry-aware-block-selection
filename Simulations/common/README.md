# common

Everything that does not depend on which bound is being optimised.

The rule that decides what belongs here: **if the collapsed and the variational
stack would otherwise each need a copy, it goes in `common`.** That is not a
style preference. Before the split, the block selector existed twice, the
packing format four times and the residual predictor five times, with three
different defaults between them — and nothing checked that the copies agreed.

## Modules

| Module | Contents | Depends on |
|---|---|---|
| `jax_config.py` | enables float64, once, before any array exists | jax |
| `provenance.py` | SHA-256 file hashes, config fingerprints, Git state, package and GPU manifests | — |
| `stats.py` | error bars (`std`/`var`/`sem`/`ci95`), `mean_std`, seed grouping | numpy, scipy |
| `plot_style.py` | method colours, markers, line weights, Type-42 fonts | matplotlib |
| `datasets_uci.py` | raw UCI archive download, parsing, caching | numpy, pandas |
| `datasets.py` | the declared-source registry, caches, provenance sidecars | `datasets_uci`, `provenance` |
| `prepare_electricity.py` | builds the manual UCI-321 electricity cache | numpy, pandas |
| `data_split.py` | deterministic split, train-only standardisation, Snelson loader | numpy |
| `kernels_nonstationary.py` | the Gibbs kernel and its warp parameterisation | jax |
| `kernels.py` | kernel registry, name canonicalisation, dispatch | `kernels_nonstationary` |
| `terms.py` | the packed `(coeffs, idx, mask)` block cover; pack, unpack, validate | numpy |
| `partitions.py` | BT and SBT block selection, host and device | `kernels`, `terms` |
| `predictors.py` | local-cross and full-cross residual corrections | numpy, jax |
| `results_io.py` | per-run JSON, parallel-safe result CSVs, collection | numpy |

Dependencies point downwards only. Nothing in `common` imports `sgpr` or
`svgp`; the analysis helpers in `svgp/analysis/_common.py` are the one place
that imports a stack, and they are not part of `common`.

## The packed representation (`terms.py`)

A block cover reaches every loss as three fixed-shape arrays:

```
coeffs [T]      per-term weight; 0 marks a padding row
idx    [T, S]   training-set positions, int32
mask   [T, S]   1.0 on real entries, 0.0 on padding
```

Fixed shapes are the point: re-selecting the structure changes the *contents*,
never the shapes, so XLA does not recompile and TensorFlow does not re-trace.
The residual penalty is

```
P = sum_t coeffs[t] * logdet(I + D[idx[t], idx[t]] / sigma^2)
```

and the net coefficient of every training point must be exactly 1. That is the
condition making each cover a certified lower bound (Hadamard–Fischer), and
`net_coefficients` checks it rather than assuming it.

## Block selection (`partitions.py`)

`bt_*` draws disjoint random blocks. It is the structure-agnostic control: it
separates "does blocking help" from "does *this* blocking help".

`sbt_*` orders the inducing points along a greedy nearest-neighbour chain,
assigns each datum to its nearest anchor, orders the points inside each anchor
cell, concatenates the cells in chain order, and cuts the result into
equal-size blocks. If a cut splits a cell, the points that move across it are
the ones closest to the neighbouring anchor.

There are a host (NumPy) and a device (JAX) path, because the GPflow figures
and the JAX benchmarks need different things. They implement the same
construction but measure distance differently — ARD Euclidean versus
kernel-induced. For the kernels in the registry the second is a monotone
function of the first, so:

* the parent assignment and the `center` ordering are **identical**;
* the default `boundary` ordering sorts on a *difference* of distances, which a
  monotone reparameterisation does not preserve, so above one dimension the two
  paths can cut a cell at a different point.

In practice the host path is only ever used on 1-D data (Snelson, residual
illustration) where the two agree exactly. `tests/test_partitions.py` pins
both facts, so if that ever stops being true it fails loudly instead of
producing a figure that no longer illustrates the table.

## Residual predictors (`predictors.py`)

Every method here predicts with `q-term + residual + sigma^2` and differs only
in `residual`:

```
sparse     d_**
diagonal   d_** - sum_n d_*n (d_nn + sigma^2)^-1 d_n*       (singleton cover)
block      d_** - sum_j c_j(x*) over the method's fitted cover
full       d_** - d_*f (D_ff + sigma^2 I)^-1 d_f*           (dense, O(N^3))

`block` and `full` are the same number computed two ways: permute the training
set into the cover's order, factor A = D_ff + sigma^2 I = L L^T, and the
correction is ||L^-1 d_f*||^2, whose terms group by block. The cover changes
how that number is distributed, never its value.
```

This module owns only the residual term. The caller passes the projections it
has already computed and adds its own q-term and noise, which is exactly what
lets the collapsed bound (q from the Σ factorisation) and the variational bound
(q from L_q) share one implementation instead of two that drift.

## Adding a dataset

1. Add a registry entry in `datasets.py` with an explicit `spec`/`loader`, a
   `source`, and a `note` saying what it is and why it is there.
2. If it needs custom feature engineering, write a `_load_<key>` function and
   register it in `_CUSTOM_LOADERS`; angular features go in `angle_cols` so
   `data_split` leaves them on their native radian scale.
3. Run `python -m common.datasets --check-data --force --datasets <key>` and
   commit neither the cache nor the sidecar — both are generated.
4. Add the pair to a plan in `svgp/plans.py` only after that passes.

Retiring a dataset means adding a `retired` marker with the reason, not
deleting the entry: archived campaigns must stay loadable.
`assert_not_retired` then fails at plan-construction time rather than after the
compute is spent.
