#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared plumbing for the post-hoc analysis scripts.

Every script here reads a finished runs tree and recomputes something at the
trained parameters. They all need the same three things -- the run's split, its
residual partition, and sometimes its engine -- and each used to carry its own
copy, with its own cache and its own subtly different defaults.

Nothing in this module retrains anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import data_split as jd
from common import datasets as dm

_SPLIT_CACHE: dict = {}
_ENGINE_CACHE: dict = {}


def _split_key(cfg):
    return (
        cfg["dataset"],
        int(cfg.get("split_seed", cfg["seed"])),
        str(cfg.get("split_rng", jd.LEGACY_SPLIT_RNG)),
        int(cfg["requested_n_train"]),
        int(cfg["requested_n_test"]),
        bool(cfg.get("all_data", False)),
    )


def split_for(cfg):
    """Rebuild the exact train/test split a finished run used.

    ``split_rng`` defaults to the legacy JAX generator because runs predating
    the explicit field were produced with it; a modern run always records its
    own value and never reaches that default.
    """
    from svgp import run_benchmark as runner

    key = _split_key(cfg)
    if key not in _SPLIT_CACHE:
        X, y = dm.load_raw(cfg["dataset"], data_dir=runner.DATASET_DIR)
        n_train, n_test = runner.effective_split_sizes(
            len(y), cfg["requested_n_train"], cfg["requested_n_test"],
            all_data=bool(cfg.get("all_data", False)),
        )
        _SPLIT_CACHE[key] = jd.split_standardize(
            X, y,
            int(cfg.get("split_seed", cfg["seed"])),
            n_train, n_test,
            angle_cols=dm.angle_cols_for(cfg["dataset"]),
            split_rng=str(cfg.get("split_rng", jd.LEGACY_SPLIT_RNG)),
        )
    return _SPLIT_CACHE[key]


def engine_for(cfg):
    """Build (once per configuration) the engine a run was trained with."""
    from svgp import engine as sv
    from svgp import run_benchmark as runner

    key = (*_split_key(cfg), cfg["kernel"])
    if key not in _ENGINE_CACHE:
        x_tr, y_tr, x_te, y_te, _ = split_for(cfg)
        _ENGINE_CACHE[key] = sv.make_engine(
            x_tr, y_tr, x_te, y_te,
            jitter=runner.JITTER, kernel_name=cfg["kernel"],
        )
    return _ENGINE_CACHE[key]

def blocks_for(run_dir, cfg, params, x_train):
    """Return ``(idx, mask, origin)`` of the partition a run conditions on.

    SBT has an anchored partition, saved at the end of training because it
    depends on the final hyperparameters. Every other method has no residual
    partition of its own, so it borrows the random one BT trains with: a
    permutation of ``PRNGKey(seed)`` cut into fixed-width blocks. That branch
    uses neither the kernel nor the inducing points, so it is reproducible from
    the seed alone -- and it is the honest counterpart, because it asks whether
    the correction needs *anchored* blocks or merely *some* blocks.
    """
    import jax
    import jax.numpy as jnp

    from common import partitions as pt
    from common import terms as tm

    run_dir = Path(run_dir)
    if str(cfg["method"]).startswith("SBT"):
        stored = np.load(run_dir / "final_blocks.npz")
        return stored["idx"], stored["mask"] > 0.5, "anchored"

    packed = pt.bt_terms_device(x_train.shape[0], int(cfg["block"]),
                                int(cfg["seed"]))
    idx = np.asarray(jax.device_get(packed[1]))
    mask = np.asarray(jax.device_get(packed[2])) > 0.5
    tm.assert_partition(tm.unpack_terms(packed), x_train.shape[0])
    del jnp
    return idx, mask, f"random(seed={int(cfg['seed'])})"


def iter_runs(runs_dir, *, method_prefix=None, run_tag=None):
    """Yield ``(path, metrics, config)`` for every run under ``runs_dir``."""
    for path in sorted(Path(runs_dir).rglob("metrics.json")):
        try:
            metrics = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        cfg = metrics.get("config", {})
        if method_prefix and not str(cfg.get("method", "")).startswith(method_prefix):
            continue
        if run_tag and str(cfg.get("run_tag", "")) != run_tag:
            continue
        yield path.parent, metrics, cfg
