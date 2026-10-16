#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official driver of the variational benchmark.

A thin layer over ``svgp.engine``: it resolves a campaign plan, reads data,
fingerprints the configuration, runs, and persists. All numerics -- objectives,
block selection, natural gradient, minibatch optimisation, prediction and
per-gradient timing -- live in the engine and in ``common``.

Methods: SVGP, T-SVGP, RBT-SVGP, SBT-SVGP.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from common import data_split as jd            # noqa: E402
from common import datasets as dm              # noqa: E402
from common import results_io as rio           # noqa: E402
from common.provenance import (                # noqa: E402
    SVGP_RELEASE_PACKAGES,
    code_state,
    config_fingerprint,
    release_runtime_matches,
    runtime_manifest,
    sha256_file,
)
from svgp import engine as sv                  # noqa: E402
from svgp import plans                         # noqa: E402

DATASET_DIR = jd.DATASET_DIR                   # shared Simulations/dataset/
RUNS_DIR = HERE / "runs_v12"
LOGS_DIR = HERE / "logs_v12"
TABLES_DIR = HERE / "tables_v12"
JITTER = 1e-8
# Eq. (12) factorises an N x N residual, so it is O(N^2) memory. 40k points is
# 12.8 GB before the factor; the cap makes that a refusal rather than an OOM
# three hours into a run.
FULL_PREDICTOR_MAX_N = 50000
IMPLEMENTATION = sv.IMPLEMENTATION
TRAINING_PROTOCOL = "svgp-gain-double-batch-v3"


def apply_lr_profile(args, kernel_name):
    """Resolve conservative kernel-specific rates before hashing the config."""
    if str(getattr(args, "lr_profile", "fixed")) != "stable":
        return
    kernel_name = sv.canonical_kernel_name(kernel_name)
    args.lr, args.z_lr, args.natural_lr = plans.stable_learning_rates(
        kernel_name
    )


def dataset_signature(dataset):
    path = dm.raw_cache_path(dataset, DATASET_DIR)
    return sha256_file(path)


def effective_split_sizes(total, requested_train, requested_test,
                          all_data=False):
    """Resolve either the paper 90/10 split or exact requested counts."""
    total = int(total)
    requested_train = int(requested_train)
    requested_test = int(requested_test)
    if all_data:
        n_test = max(1, int(round(0.10 * total)))
        return total - n_test, n_test
    required = requested_train + requested_test
    if total < required:
        raise ValueError(
            f"dataset has {total} rows but the fixed split requires {required}; "
            "use --all-data for the deterministic 90/10 paper split"
        )
    return requested_train, requested_test


def make_run_config(method, m, dataset, seed, args, signature,
                    n_train, n_test, split_meta):
    training_method = str(getattr(args, "training_method", method))
    donor_method = str(getattr(args, "donor_method", "") or "")
    state = code_state(HERE.parent.parent)
    config = {
        "implementation": IMPLEMENTATION,
        "training_protocol": TRAINING_PROTOCOL,
        "kernel": str(args.kernel),
        "method": method,
        "training_method": training_method,
        "donor_method": donor_method,
        "M": int(m),
        "dataset": dataset,
        "seed": int(seed),
        "model_seed": int(seed),
        "split_seed": int(split_meta["split_seed"]),
        "split_rng": str(split_meta["split_rng"]),
        "preprocessing_protocol": str(split_meta["preprocessing_protocol"]),
        "train_indices_sha256": str(split_meta["train_indices_sha256"]),
        "test_indices_sha256": str(split_meta["test_indices_sha256"]),
        "run_tag": str(getattr(args, "run_tag", "")),
        "plan": str(getattr(args, "plan", "report")),
        "data_cache": dm.raw_cache_path(dataset, DATASET_DIR).name,
        "data_signature": signature,
        "data_source": dm.dataset_source(dataset),
        # Execution-only changes (deferred prediction and atomic persistence)
        # do not alter the training map.  A resumed frozen campaign may pin the
        # numerical source commit while recording the actual execution commit
        # outside the fingerprint in training.json / metrics.json.
        "code_commit": str(
            os.environ.get("SVGP_NUMERICAL_REFERENCE_HEAD", state["commit"])
        ),
        "code_source_dirty": state["source_dirty"],
        "requested_n_train": int(args.n_train),
        "requested_n_test": int(args.n_test),
        "all_data": bool(getattr(args, "all_data", False)),
        "split_protocol": (
            "all-rows-deterministic-90-10"
            if bool(getattr(args, "all_data", False))
            else "fixed-counts"
        ),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "block": int(args.block),
        "blocks_per_step": int(args.blocks_per_step),
        "ll_batch_size": int(args.ll_batch_size),
        "full_batch": bool(getattr(args, "full_batch", False)),
        "common_ll_batches": bool(getattr(args, "common_ll_batches", True)),
        "batch_coupling": str(args.batch_coupling),
        "penalty_sampling": str(args.penalty_sampling),
        "epochs": int(args.epochs),
        "refresh_every": int(args.refresh_every),
        "importance_refresh_every": int(args.importance_refresh_every),
        "importance_epsilon": float(args.importance_epsilon),
        "log_every": int(args.log_every),
        "refresh_until": (
            None if args.refresh_until is None else int(args.refresh_until)
        ),
        "lr": float(args.lr),
        "z_lr": float(args.z_lr),
        "lr_profile": str(args.lr_profile),
        "lr_schedule": str(args.lr_schedule),
        "optimizer": str(args.optimizer),
        "weight_decay": 0.0,
        "grad_clip": float(args.grad_clip),
        "normalize_objective": bool(args.normalize_objective),
        "q_optimizer": str(args.q_optimizer),
        "natural_lr": float(args.natural_lr),
        "evaluate_during_training": False,
        "record_history": bool(getattr(args, "record_history", False)),
        "save_params": bool(getattr(args, "save_params", False)),
        "reported_predictor": sv.METHOD_PREDICTOR[training_method],
        "sbt_geometry": "anchor",
        "swa_fraction": float(getattr(args, "swa_fraction", 0.0)),
        "phases": str(getattr(args, "phases", "") or ""),
        "metrics_at_epochs": sorted(
            int(e) for e in (getattr(args, "metrics_at_epochs", None) or ())
        ),
        "init_from_kernel": str(getattr(args, "init_from_kernel", "")),
        **(
            {
                "init_params_sha256": str(
                    getattr(args, "init_params_sha256", "")
                )
            }
            if str(getattr(args, "init_params_sha256", "") or "")
            else {}
        ),
        "transplant_to_svgp": bool(getattr(args, "transplant_to_svgp", False)),
        "donor_params_sha256": str(
            getattr(args, "donor_params_sha256", "") or ""
        ),
        "resume_from_kernel": str(getattr(args, "resume_from_kernel", "")),
        "full_predictor_max_n": int(getattr(
            args, "full_predictor_max_n", FULL_PREDICTOR_MAX_N
        )),
        "base_logdet_width": (
            int(args.block)
            if sv.METHODS[training_method][0] in {"block", "gain"} else 0
        ),
        # Recorded only where they are meaningful, so the fingerprints of the
        # stationary kernels stay comparable with the previous protocol.
        **(
            {
                "warp_parameterization": sv.WARP_PARAMETERIZATION,
                "warp_gate_fraction": float(
                    getattr(args, "warp_gate_fraction", 0.75)
                ),
            }
            if sv.canonical_kernel_name(args.kernel) == "gibbs" else {}
        ),
    }
    return config, config_fingerprint(config)


def resume_from_checkpoint(params, args, method, m, dataset, seed):
    """Continue a run from the mid-training checkpoint of another kernel.

    This is the fork that makes the Gibbs comparison budget-neutral: the
    stationary run writes its state at the gate epoch, and the Gibbs run picks
    it up and spends the remaining epochs with the warp released. Both branches
    therefore see the same total number of epochs and, up to the fork, the same
    trajectory -- so the difference between them is the warp and nothing else.

    Returns ``(params, start_epoch, optimizer leaves, source description)``.
    """
    source_kernel = str(getattr(args, "resume_from_kernel", "") or "")
    if not source_kernel:
        return params, 0, None, None
    source_kernel = sv.canonical_kernel_name(source_kernel)
    if source_kernel == args.kernel:
        raise SystemExit("--resume-from-kernel must differ from --kernels")
    src = (
        Path(getattr(args, "resume_from_runs_dir", "") or args.runs_dir)
        / source_kernel / dataset / f"M{m}" / method / f"seed{seed}"
        / "checkpoint.npz"
    )
    if not src.exists():
        raise SystemExit(
            f"resume source not found: {src}\n"
            "run the source kernel first with --checkpoint-at-fraction, or "
            "drop --resume-from-kernel."
        )
    loaded, optimizer_leaves, meta = sv.load_training_checkpoint(src)
    start_epoch = int(meta["epoch"])
    source_total = int(meta.get("total_epochs", args.epochs))
    if source_total != int(args.epochs):
        raise SystemExit(
            f"{src} was written by a run of {source_total} epochs but this run "
            f"declares {args.epochs}. Optimizer state and any schedule are "
            "defined against the total budget, so the branch is incompatible."
        )

    warm = dict(params)
    copied = []
    for key, value in loaded.items():
        if key not in warm or jnp.shape(value) != jnp.shape(warm[key]):
            continue
        warm[key] = jnp.asarray(value)
        copied.append(key)
    if "q_cov" in loaded and "q_cov" not in copied:
        warm = adopt_variational_covariance(warm, loaded["q_cov"], args)
        copied.append("q_cov")
    if "log_ell" not in copied:
        raise SystemExit(f"{src} shares no lengthscale with this run")
    # A fork that dropped part of the state is not a continuation. Everything
    # the source carries, except the warp the source never had, must arrive.
    dropped = sorted(set(loaded) - set(copied) - {"warp", "q_raw"})
    if dropped:
        raise SystemExit(
            f"{src} holds parameters this run did not adopt: {dropped}. "
            "The branch would not continue the source run."
        )

    if "warp" in warm and "warp" not in loaded:
        # Gated Gibbs is the stationary kernel of lengthscale 1.5*ell, so the
        # raw parameter has to be re-expressed or the branch would restart at
        # an effective lengthscale the stationary run never had. Imported here
        # rather than at module scope so the non-stationary module stays off
        # the import path of a purely stationary run.
        from common import kernels_nonstationary as jns

        warm["log_ell"] = jns.stationary_to_gated_log_ell(warm["log_ell"])

    description = f"{source_kernel}@epoch{start_epoch}"
    print(f"     resume from {description}: {', '.join(sorted(copied))}"
          + ("  (log_ell re-expressed for the gated kernel)"
             if "warp" in warm and "warp" not in loaded else ""))
    return warm, start_epoch, optimizer_leaves, description


def make_bank_fn(method, x, args, seed):
    def make_bank(params):
        return sv.build_method_bank(
            method,
            x,
            params,
            block=args.block,
            seed=seed,
            jitter=JITTER,
            kernel_name=args.kernel,
            common_ll_batches=bool(getattr(args, "common_ll_batches", True)),
            ll_batch_size=int(args.ll_batch_size),
        )
    return make_bank


def predictive_diagnostics(params, x_train, x_eval, y_eval, kernel_name, *,
                           predictor="sparse", blocks=None):
    """Final predictive calibration diagnostics on the standardized scale.

    Computed under the predictor this row REPORTS. Using the sparse variance
    for every method would make ``nll_log_term + nll_quad_term`` disagree with
    ``nll`` on anything that models a residual, and put two estimators in one
    row -- the thing the predictor contract exists to prevent.
    """
    kernel_name = sv.canonical_kernel_name(kernel_name)
    x_eval = jnp.asarray(x_eval)
    ell, sf2, s2, lq, warp = sv.transform(params)
    mean, var = sv.predict_svgp(
        params, x_train, x_eval, predictor=predictor, blocks=blocks,
        kernel_name=kernel_name, jitter=JITTER,
    )
    mean = jnp.ravel(mean)
    residual = jnp.ravel(y_eval) - mean
    var = jnp.maximum(var, 1e-8)
    log_term = 0.5 * jnp.mean(jnp.log(2.0 * jnp.pi * var))
    quad_term = 0.5 * jnp.mean(residual ** 2 / var)
    values = jax.device_get({
        "log_s2": params["log_s2"],
        "noise_var": s2,
        "noise_sd": jnp.sqrt(s2),
        "log_sf2": params["log_sf2"],
        "signal_var": sf2,
        "ell_min": jnp.min(ell),
        "ell_mean": jnp.mean(ell),
        "ell_max": jnp.max(ell),
        "pred_var_min": jnp.min(var),
        "pred_var_mean": jnp.mean(var),
        "pred_var_max": jnp.max(var),
        "pred_sd_mean": jnp.mean(jnp.sqrt(var)),
        "mean_abs_residual": jnp.mean(jnp.abs(residual)),
        "std_resid_sq_mean": jnp.mean(residual ** 2 / var),
        "nll_log_term": log_term,
        "nll_quad_term": quad_term,
    })
    return {key: float(value) for key, value in values.items()}


WARM_START_KEYS = ("log_ell", "log_sf2", "log_s2", "warp", "z", "q_mu",
                   "q_cov", "q_raw")


def adopt_variational_covariance(warm, value, args):
    """Carry a source ``q_cov`` into a target that still holds ``q_raw``.

    ``init_params`` creates the raw factor ``q_raw``; ``train_svgp`` replaces it
    by the covariance ``q_cov`` only once, at the start of training, and only
    under the natural-gradient optimizer. So a saved run stores ``q_cov`` while
    a freshly initialised target still has ``q_raw``, and a key-by-key copy
    skips the variational covariance without saying so -- the largest trained
    object of the model at M=512. Swapping the representation here makes
    ``train_svgp`` keep the loaded one, since its conversion is guarded by
    ``"q_cov" not in params``.
    """
    if str(getattr(args, "q_optimizer", "natural")) != "natural":
        raise SystemExit(
            "the source stores q_cov (a natural-gradient posterior) but this "
            "run parameterises q(u) by q_raw (--q-optimizer lion); the "
            "variational posterior cannot be carried across."
        )
    warm = dict(warm)
    warm.pop("q_raw", None)
    warm["q_cov"] = jnp.asarray(value)
    return warm


def warm_start_params(params, args, method, m, dataset, seed):
    """Initialize from the trained parameters of another kernel.

    ``--init-from-kernel rbf`` loads ``params.npz`` of the run with the same
    run tag, dataset, M, method and seed but a different kernel, and copies
    the shared hyperparameter blocks. Only entries whose shape matches are
    copied. A same-kernel transplant also carries the non-stationary ``warp``
    block; a cross-kernel warm start cannot do so because the source lacks a
    shape-compatible leaf.

    Requires the source run to have been produced with ``--save-params``.
    """
    source_kernel = str(getattr(args, "init_from_kernel", "") or "")
    if not source_kernel:
        return params
    source_kernel = sv.canonical_kernel_name(source_kernel)
    # Same-kernel warm starts are the staged protocol: a finished run supplies
    # theta and Z, and the staged run refines q(u) and then Z with theta held.
    # Only a same-kernel, same-runs-dir start would be a no-op, and that is the
    # one case worth refusing.
    if (source_kernel == args.kernel
            and not str(getattr(args, "init_from_runs_dir", "") or "")):
        raise SystemExit(
            "--init-from-kernel matches --kernels with no --init-from-runs-dir: "
            "the source and the target are the same run. Point "
            "--init-from-runs-dir at the finished tree."
        )
    src = (
        Path(getattr(args, "init_from_runs_dir", "") or args.runs_dir)
        / source_kernel / dataset / f"M{m}" / method / f"seed{seed}"
        / "params.npz"
    )
    if not src.exists():
        raise SystemExit(
            f"warm start source not found: {src}\n"
            "run the source kernel first with --save-params, or drop "
            "--init-from-kernel."
        )
    loaded = np.load(src)
    warm = dict(params)
    copied = []
    for key in WARM_START_KEYS:
        if key not in loaded.files or key not in warm:
            continue
        value = jnp.asarray(loaded[key])
        if value.shape != jnp.shape(warm[key]):
            continue
        warm[key] = value
        copied.append(key)
    if "q_cov" in loaded.files and "q_cov" not in copied:
        warm = adopt_variational_covariance(warm, loaded["q_cov"], args)
        copied.append("q_cov")
    if not copied:
        raise SystemExit(f"warm start source {src} shares no usable block")
    if "warp" in warm and "warp" not in loaded.files and "log_ell" in copied:
        # Same correction as the checkpoint fork, and for the same reason: a
        # gated Gibbs kernel is the stationary kernel of lengthscale 1.5*ell,
        # so copying log_ell verbatim starts the run at an effective
        # lengthscale the source never had. Measured on a synthetic problem,
        # the un-shifted kernel differs from its source by 0.49 in absolute
        # value; shifted, by 3e-15.
        from common import kernels_nonstationary as jns

        warm["log_ell"] = jns.stationary_to_gated_log_ell(warm["log_ell"])
        copied.append("log_ell (re-expressed for the gated kernel)")
    print(f"     warm start from {source_kernel}: {', '.join(copied)}")
    return warm


def _atomic_json(path, payload):
    """Publish JSON only after the complete payload reached stable storage."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _atomic_npz(path, **arrays):
    """Write an NPZ beside its target and atomically rename it into place."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _parameter_arrays(params):
    arrays = {}
    for key, value in params.items():
        if key == "nn":
            for layer_id, leaf in enumerate(value):
                arrays[f"nn_{layer_id}"] = np.asarray(jax.device_get(leaf))
        else:
            arrays[key] = np.asarray(jax.device_get(value))
    return arrays


def _load_parameter_arrays(path):
    loaded = np.load(path)
    params = {
        key: jnp.asarray(loaded[key])
        for key in loaded.files
        if not key.startswith("nn_")
    }
    nn_keys = sorted(
        (key for key in loaded.files if key.startswith("nn_")),
        key=lambda key: int(key.split("_", 1)[1]),
    )
    if nn_keys:
        params["nn"] = tuple(jnp.asarray(loaded[key]) for key in nn_keys)
    return params


def _record_matches(record, config_id):
    try:
        config = record.get("config") or {}
        return (
            record.get("implementation") == IMPLEMENTATION
            and record.get("config_id") == config_id
            and config_fingerprint(config) == config_id
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _training_artifacts_complete(out, record, config_id, is_block_method):
    return bool(
        isinstance(record, dict)
        and record.get("training_complete") is True
        and _record_matches(record, config_id)
        and (out / "params.npz").is_file()
        and (out / "curve.npz").is_file()
        and (not is_block_method or (out / "final_blocks.npz").is_file())
    )


def _save_training_artifacts(out, params, curve, final_bank, is_block_method,
                             training_record):
    """Persist the expensive training result before any test prediction."""
    out.mkdir(parents=True, exist_ok=True)
    _atomic_npz(out / "params.npz", **_parameter_arrays(params))
    if curve:
        values = np.asarray(curve)
        _atomic_npz(
            out / "curve.npz",
            epoch=values[:, 0],
            stoch_obj_n=values[:, 1],
        )
    else:
        _atomic_npz(
            out / "curve.npz",
            epoch=np.empty((0,), dtype=np.float64),
            stoch_obj_n=np.empty((0,), dtype=np.float64),
        )
    if is_block_method:
        _atomic_npz(
            out / "final_blocks.npz",
            idx=np.asarray(jax.device_get(final_bank["base_idx"])),
            mask=np.asarray(jax.device_get(final_bank["base_mask"])),
        )
    _atomic_json(out / "training.json", training_record)


def _load_saved_blocks(out):
    path = out / "final_blocks.npz"
    if not path.is_file():
        return None
    loaded = np.load(path)
    return sv.blocks_from_bank(loaded["idx"], loaded["mask"])


def _make_training_record(*, args, config, config_id, result_method,
                          training_method, donor_method, dataset, m, seed,
                          n, n_test, d, info, curve):
    stochastic_objective = float(curve[-1][1]) if curve else float("nan")
    return {
        "implementation": IMPLEMENTATION,
        "training_protocol": TRAINING_PROTOCOL,
        "training_artifact_version": "svgp-training-artifact-v1",
        "training_complete": True,
        "prediction_deferred": True,
        "execution_commit": code_state(HERE.parent.parent)["commit"],
        "run_tag": str(getattr(args, "run_tag", "")),
        "plan": str(getattr(args, "plan", "report")),
        "config": config,
        "config_id": config_id,
        "kernel": args.kernel,
        "method": result_method,
        "training_method": training_method,
        "donor_method": donor_method,
        "dataset": dataset,
        "M": m,
        "seed": seed,
        "n_train": n,
        "n_test": int(n_test),
        "d": d,
        "block": args.block,
        "ll_batch_size": info["likelihood_batch_size"],
        "base_logdet_width": info["base_logdet_width"],
        "n_blocks": info["n_blocks"],
        "blocks_per_step": info["blocks_per_step"],
        "data_blocks": info["data_blocks"],
        "data_blocks_per_step": info["data_blocks_per_step"],
        "data_unit_scale": info["data_unit_scale"],
        "last_epoch_stochastic_objective_estimate_per_datum":
            stochastic_objective,
        "runtime": runtime_manifest(),
        "schedule_epochs": args.epochs,
        "epochs": info["epochs"],
        "steps_per_epoch": info["steps_per_epoch"],
        "optimizer_steps": info["optimizer_steps"],
        "sampling": info["sampling"],
        "importance_sampling": info["importance_sampling"],
        "importance_kind": info["importance_kind"],
        "gain_decomposition": info["gain_decomposition"],
        "importance_refresh_every": info["importance_refresh_every"],
        "importance_refreshes": info["importance_refreshes"],
        "importance_epsilon": info["importance_epsilon"],
        "importance_ess": info.get("importance_ess"),
        "importance_min_p": info.get("importance_min_p"),
        "importance_max_weight": info.get("importance_max_weight"),
        "common_likelihood_batches": info["common_likelihood_batches"],
        "unit_scale": info["unit_scale"],
        "unit_count": info["unit_count"],
        "nominal_batch_points": info["nominal_batch_points"],
        "kuf_columns_per_step": info["kuf_columns_per_step"],
        "grad_ms": info["grad_ms"],
        "grad_ms_std": info["grad_ms_std"],
        "train_step_ms": info["train_step_ms"],
        "search_s": info["search_s"],
        "compile_s": info["compile_s"],
        "structure_search_s": info["structure_search_s"],
        "importance_refresh_s": info["importance_refresh_s"],
        "history_struct_prediction_s": info["history_struct_prediction_s"],
        "optimizer_s": info["optimizer_s"],
        "refreshes": info["refreshes"],
        "structures_built": info["structures_built"],
        "lr": args.lr,
        "lr_schedule": info["lr_schedule"],
        "post_swa_structure_rebuilds": info.get(
            "post_swa_structure_rebuilds", 0
        ),
        "final_structure_rebuilds": info.get("final_structure_rebuilds", 0),
        "z_lr": info["z_lr"],
        "sbt_geometry": "anchor",
        "optimizer": str(args.optimizer),
        "normalize_objective": info["objective_normalized"],
        "curve_kind": info["curve_kind"],
        "evaluated_test_during_training": info["evaluated_test_during_training"],
        "q_optimizer": info["q_optimizer"],
        "natural_lr": info["natural_lr"],
        "q_cov_min_eig": info["q_cov_min_eig"],
        "q_cov_max_eig": info["q_cov_max_eig"],
        "warp_trainable": info.get("warp_trainable"),
        "warp_switch_epoch": info.get("warp_switch_epoch"),
        "warp_frobenius_norm": info.get("warp_frobenius_norm"),
        "swa_start_epoch": info.get("swa_start_epoch"),
        "swa_snapshots": info.get("swa_snapshots"),
        "start_epoch": info.get("start_epoch", 0),
        "resumed_from": info.get("resumed_from"),
        "optimizer_leaves_restored": info.get("optimizer_leaves_restored"),
    }


def run_single(kernel_name, method, m, dataset, seed, args):
    args = copy.copy(args)
    if (getattr(args, "defer_prediction", False)
            or getattr(args, "prediction_only", False)):
        # History RMSE/NLL calls are predictions too. Keep the objective curve
        # and its requested epoch boundaries, but do not hide test inference
        # inside an ostensibly training-only run.
        args.record_history = False
        args.plot_history = False
    args.kernel = sv.canonical_kernel_name(kernel_name)
    donor_method = method if bool(getattr(args, "transplant_to_svgp", False)) else ""
    training_method = "SVGP" if donor_method else method
    result_method = f"{donor_method}-to-SVGP" if donor_method else method
    args.training_method = training_method
    args.donor_method = donor_method
    if donor_method:
        if not str(getattr(args, "init_from_runs_dir", "") or ""):
            raise SystemExit(
                "--transplant-to-svgp requires --init-from-runs-dir pointing "
                "at the completed donor campaign"
            )
        args.init_from_kernel = args.kernel
        donor_path = (
            Path(args.init_from_runs_dir) / args.kernel / dataset / f"M{m}"
            / donor_method / f"seed{seed}" / "params.npz"
        )
        if not donor_path.exists():
            raise SystemExit(f"transplant donor parameters not found: {donor_path}")
        args.donor_params_sha256 = sha256_file(donor_path)
    elif str(getattr(args, "init_from_kernel", "") or ""):
        source_kernel = sv.canonical_kernel_name(args.init_from_kernel)
        source_params = (
            Path(getattr(args, "init_from_runs_dir", "") or args.runs_dir)
            / source_kernel / dataset / f"M{m}" / method / f"seed{seed}"
            / "params.npz"
        )
        if not source_params.is_file():
            raise SystemExit(f"warm-start parameters not found: {source_params}")
        args.init_params_sha256 = sha256_file(source_params)
    if getattr(args, "refresh_until", None) == -1:
        args.refresh_until = 350 if args.kernel == "gibbs" else 200
    apply_lr_profile(args, args.kernel)
    out = (
        Path(args.runs_dir)
        / args.kernel
        / dataset
        / f"M{m}"
        / result_method
        / f"seed{seed}"
    )
    cache_dirs = [out]

    X, y = dm.load_raw(dataset, data_dir=DATASET_DIR)        # host file IO
    angle_cols = dm.angle_cols_for(dataset)   # periodic (radian) feature cols
    n_train, n_test = effective_split_sizes(
        len(y), args.n_train, args.n_test,
        all_data=bool(getattr(args, "all_data", False)),
    )
    split_seed = seed if args.split_seed is None else int(args.split_seed)
    x_tr, y_tr, x_te, y_te, meta = jd.split_standardize(
        X,
        y,
        split_seed,
        n_train,
        n_test,
        angle_cols=angle_cols,
        split_rng=args.split_rng,
    )
    x_eval, y_eval = x_te, y_te

    n = int(x_tr.shape[0])
    config, config_id = make_run_config(
        result_method, m, dataset, seed, args, dataset_signature(dataset),
        n, n_test, meta,
    )
    predictor = sv.METHOD_PREDICTOR[training_method]
    is_sbt = sv.METHODS[training_method][1].startswith("SBT")
    is_block_method = predictor == "block"
    complete_record = None
    for cache_dir in cache_dirs:
        cached = _load_json(cache_dir / "metrics.json")
        if not isinstance(cached, dict) or not _record_matches(cached, config_id):
            continue
        cached_config = cached.get("config") or {}
        history_path = cache_dir / "history.csv"
        history_has_struct = False
        if history_path.exists() and is_sbt:
            with open(history_path, newline="") as history_file:
                history_columns = next(csv.reader(history_file), [])
                history_has_struct = (
                    "nll_reported" in history_columns
                    or "nll_full" in history_columns
                )
        missing_history = (
            getattr(args, "record_history", False)
            and not getattr(args, "defer_prediction", False)
            and (
                not history_path.exists()
                or (is_sbt and not history_has_struct)
            )
        )
        missing_params = (
            getattr(args, "save_params", False)
            and not (cache_dir / "params.npz").exists()
        )
        missing_checkpoint = (
            float(getattr(args, "checkpoint_at_fraction", 0.0)) > 0.0
            and not (cache_dir / "checkpoint.npz").exists()
        )
        missing_extra = (
            cached_config.get("reported_predictor") != predictor
            or cached.get("reported_predictor") != predictor
            or "nll_sparse" not in cached
            or (predictor == "full" and "nll_full" not in cached)
        )
        if (
            "rmse" in cached
            and not missing_history
            and not missing_params
            and not missing_checkpoint
            and not missing_extra
        ):
            complete_record = cached
            if not args.force:
                return cached, True
            break

    training_record = _load_json(out / "training.json")
    training_cached = _training_artifacts_complete(
        out, training_record, config_id, is_block_method
    )
    if getattr(args, "defer_prediction", False) and training_cached and not args.force:
        return training_record, True
    if getattr(args, "defer_prediction", False) and args.force:
        training_cached = False
    if getattr(args, "prediction_only", False):
        if not training_cached:
            # Fully evaluated archived rows predate training.json but already
            # carry the exact training manifest and saved parameters. They are
            # valid inputs to a forced, uniform re-timing pass.
            if complete_record is not None and (out / "params.npz").is_file():
                training_record = dict(complete_record)
                training_record["training_complete"] = True
                training_record["prediction_deferred"] = False
                training_cached = True
            else:
                raise SystemExit(
                    f"prediction-only requested but training artifacts are "
                    f"missing or incompatible: {out}"
                )
    run_args = copy.copy(args)
    run_args.seed = seed
    run_args.natural_q = run_args.q_optimizer == "natural"
    run_args.track_grad_stats = True
    run_args.evaluate_during_training = False
    # Big run is always clean: no Python per-epoch printing, only the per-run
    # summary below. History is optional and uses fixed log points.
    run_args.live = False
    engine = sv.make_engine(x_tr, y_tr, x_eval, y_eval, jitter=JITTER,
                            kernel_name=args.kernel)
    make_bank = make_bank_fn(training_method, x_tr, run_args, seed)

    def history_bound_metrics(params_now, bank_now):
        """Method-coherent metrics at a requested history epoch."""
        blocks = (
            sv.blocks_from_bank(bank_now["base_idx"], bank_now["base_mask"])
            if predictor == "block" else None
        )
        return sv.metrics_svgp(
            params_now, x_tr, x_eval, y_eval, predictor=predictor, blocks=blocks,
            kernel_name=args.kernel, jitter=JITTER,
        )

    if training_cached:
        params = _load_parameter_arrays(out / "params.npz")
        curve_file = np.load(out / "curve.npz")
        curve = list(zip(curve_file["epoch"], curve_file["stoch_obj_n"]))
        final_bank = None
        print(
            f"  -> {args.kernel} {result_method} {dataset} M{m} seed{seed}: "
            "TRAINING CACHED; loading params for prediction only"
        )
    else:
        params = sv.init_params(x_tr, m, seed, kernel_name=args.kernel)
        params = warm_start_params(
            params, args, donor_method or method, m, dataset, seed
        )
        params, start_epoch, resume_leaves, resume_source = resume_from_checkpoint(
            params, args, training_method, m, dataset, seed
        )
        penalty_mode = sv.METHODS[training_method][0]
        printed_base_logdet = (
            run_args.block if penalty_mode in {"block", "gain"} else 0
        )
        print(f"  -> {args.kernel} {result_method} {dataset} M{m} seed{seed}: "
              f"implementation={IMPLEMENTATION} "
              f"protocol={TRAINING_PROTOCOL} "
              f"epochs={run_args.epochs} "
              f"N=train/test={n}/{n_test} B={run_args.block} "
              f"Bll={run_args.ll_batch_size} "
              f"logdet={printed_base_logdet} "
              f"blocks/step={run_args.blocks_per_step} "
              f"optimizer={run_args.optimizer} "
              f"q-opt={run_args.q_optimizer} "
              f"lr={run_args.lr:.1e}/{run_args.z_lr:.1e}/"
              f"{run_args.natural_lr:.1e}")

        def write_checkpoint(epoch, params_now, opt_state_now):
            out.mkdir(parents=True, exist_ok=True)
            sv.save_training_checkpoint(
                out / "checkpoint.npz", epoch, params_now, opt_state_now,
                meta={"kernel": args.kernel, "method": result_method,
                      "training_method": training_method,
                      "donor_method": donor_method, "M": int(m),
                      "seed": int(seed), "total_epochs": int(run_args.epochs)},
            )
            print(
                f"     checkpoint at epoch {epoch} -> {out / 'checkpoint.npz'}"
            )

        params, curve, info, final_bank = sv.train_svgp(
            engine, training_method, params, make_bank,
            engine["metrics"], n, run_args,
            struct_metrics_fn=(
                history_bound_metrics if predictor != "sparse" else None
            ),
            checkpoint_fn=write_checkpoint,
            start_epoch=start_epoch,
            resume_optimizer_leaves=resume_leaves,
        )
        info["resumed_from"] = resume_source
        print(f"     TIMING {result_method}: search={info['search_s']:.2f}s "
              f"compile={info['compile_s']:.2f}s "
              f"{run_args.optimizer}={info['optimizer_s']:.2f}s "
              f"per-grad={info['grad_ms']:.2f}ms "
              f"train-step={info['train_step_ms']:.2f}ms "
              f"(refreshes={info['refreshes']}, "
              f"structures={info['structures_built']})")
        print(f"     PROTOCOL {result_method}: sampling={info['sampling']} "
              f"N={n}/{int(x_te.shape[0])} "
              f"epochs={info['epochs']} "
              f"steps={info['optimizer_steps']} "
              f"steps/epoch={info['steps_per_epoch']} "
              f"units={info['unit_count']} U/K={info['unit_scale']:.4g}")
        training_record = _make_training_record(
            args=run_args,
            config=config,
            config_id=config_id,
            result_method=result_method,
            training_method=training_method,
            donor_method=donor_method,
            dataset=dataset,
            m=m,
            seed=seed,
            n=n,
            n_test=x_te.shape[0],
            d=meta["d"],
            info=info,
            curve=curve,
        )
        if getattr(args, "save_params", False):
            _save_training_artifacts(
                out, params, curve, final_bank, is_block_method, training_record
            )
            print(f"     TRAINING SAVED -> {out / 'training.json'}")

    if getattr(args, "defer_prediction", False):
        if not getattr(args, "save_params", False):
            raise SystemExit("--defer-prediction requires --save-params")
        return training_record, False

    # ---------------------------------------------------------------- #
    # Prediction. One estimator per method, matched to its bound: SVGP drops
    # the residual and reports the sparse predictive, every method that models
    # the residual reports Eq. (12) at its own fitted parameters. The other
    # estimator is recorded beside it so the choice can be priced without a
    # rerun.
    # ---------------------------------------------------------------- #
    def _time_ms(fn, reps):
        """Median wall-clock after the correctness call warmed the path."""
        if reps == 0:
            return float("nan")
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = fn()
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)
        return 1e3 * sorted(times)[len(times) // 2]

    # Both rows are timed through sv.metrics_svgp, the one entry point that
    # serves every predictor. engine["metrics"] is a jitted, fused program that
    # builds only the test-side projection; the structured predictors go
    # through the shared un-jitted path and build the train side as well.
    # Timing the baseline on the fused program and the competitors on the
    # shared one compared compilation strategies, not predictors.
    sparse_fn = lambda: sv.metrics_svgp(
        params, x_tr, x_eval, y_eval, predictor="sparse", blocks=None,
        kernel_name=args.kernel, jitter=JITTER,
    )
    prediction_reps = int(getattr(args, "prediction_reps", 10))
    rmse_sparse, nll_sparse = (float(v) for v in sparse_fn())
    pred_sparse_ms = _time_ms(sparse_fn, prediction_reps)
    rmse, nll, pred_ms = rmse_sparse, nll_sparse, pred_sparse_ms
    reported_blocks = None
    extra = {
        "reported_predictor": predictor,
        "rmse_sparse": rmse_sparse,
        "nll_sparse": nll_sparse,
        "pred_sparse_ms": pred_sparse_ms,
    }

    if predictor != "sparse":
        full_cap = int(getattr(args, "full_predictor_max_n", FULL_PREDICTOR_MAX_N))
        if predictor == "full" and n > full_cap:
            raise SystemExit(
                f"{result_method} reports Eq. (12), which factorises an {n}x{n} "
                f"residual ({n * n * 8 / 2**30:.1f} GB), but "
                f"--full-predictor-max-n is {full_cap}.\n"
                "Either lower --n-train / drop --all-data so the residual "
                "fits, or raise the cap if the device really has the memory. "
                "Falling back to the sparse predictor here would put two "
                "estimators in one NLL column, which is the thing this "
                "protocol exists to prevent."
            )
        blocks = None
        if predictor == "block":
            blocks = (
                sv.blocks_from_bank(
                    final_bank["base_idx"], final_bank["base_mask"]
                )
                if final_bank is not None
                else _load_saved_blocks(out)
            )
            if blocks is None:
                raise SystemExit(
                    f"block prediction requested without final_blocks.npz: {out}"
                )
        reported_blocks = blocks
        bound_fn = lambda: sv.metrics_svgp(
            params, x_tr, x_eval, y_eval, predictor=predictor, blocks=blocks,
            kernel_name=args.kernel, jitter=JITTER,
        )
        rmse_bound, nll_bound = bound_fn()
        # Same repetitions and the same median as the sparse row: a single
        # unfiltered sample is the one most exposed to a scheduler hiccup.
        pred_bound_ms = _time_ms(bound_fn, prediction_reps)
        rmse, nll, pred_ms = float(rmse_bound), float(nll_bound), pred_bound_ms
        extra.update({
            "rmse_bound": rmse,
            "nll_bound": nll,
            "pred_bound_ms": pred_bound_ms,
            f"rmse_{predictor}": rmse,
            f"nll_{predictor}": nll,
            f"pred_{predictor}_ms": pred_bound_ms,
            "prediction_reps": prediction_reps,
            "full_predictor_max_n": full_cap,
        })
        print(f"     {predictor.upper()} {result_method}: nll={nll:+.4f} "
              f"(sparse {nll_sparse:+.4f})  rmse={rmse:.4f}  "
              f"pred={pred_bound_ms:.1f}ms")
    else:
        print(f"     SPARSE {result_method}: nll={nll:+.4f}  rmse={rmse:.4f}"
              f"  pred={pred_ms:.1f}ms")
        extra["prediction_reps"] = prediction_reps
    extra["pred_ms"] = pred_ms
    diag = predictive_diagnostics(
        params, x_tr, x_eval, y_eval, args.kernel,
        predictor=predictor, blocks=reported_blocks,
    )
    rmse_orig = rmse * float(meta["y_std"])
    nll_orig = nll + math.log(float(meta["y_std"]))
    extra["rmse_sparse_orig"] = rmse_sparse * float(meta["y_std"])
    extra["nll_sparse_orig"] = nll_sparse + math.log(float(meta["y_std"]))

    res = dict(training_record)
    res.update({
        "training_complete": True,
        "prediction_deferred": False,
        "prediction_runtime": runtime_manifest(),
        "prediction_reps": prediction_reps,
        "prediction_execution": {
            "test_chunk": int(os.environ.get("SVGP_PREDICTION_CHUNK", "0")),
            "policy": "same-test-chunk-for-every-residual-predictor",
            "mathematical_predictor_unchanged": True,
            "orchestrator_commit": os.environ.get(
                "SVGP_ORCHESTRATOR_COMMIT", ""
            ),
        },
        "rmse": rmse,
        "nll": nll,
        "rmse_orig": rmse_orig,
        "nll_orig": nll_orig,
        **diag,
        **extra,
    })
    _atomic_json(out / "metrics.json", res)
    history = [] if training_cached else info.get("training_history", [])
    if history:
        history_path = out / "history.csv"
        with open(history_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "stochastic_objective_estimate_per_datum",
                    "rmse",
                    "nll",
                    "rmse_full",
                    "nll_full",
                ],
                extrasaction="ignore",
            )
            w.writeheader()
            for row in history:
                w.writerow(row)
        if getattr(args, "plot_history", False):
            try:
                import matplotlib.pyplot as plt

                h_epoch = [row["epoch"] for row in history]
                fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
                axes[0].plot(
                    h_epoch,
                    [
                        row["stochastic_objective_estimate_per_datum"]
                        for row in history
                    ],
                )
                axes[0].set_ylabel("Stochastic objective / N")
                axes[1].plot(h_epoch, [row["rmse"] for row in history])
                axes[1].set_ylabel("RMSE")
                axes[2].plot(h_epoch, [row["nll"] for row in history])
                axes[2].set_ylabel("NLL")
                axes[2].set_xlabel("epoch")
                fig.suptitle(
                    f"{args.kernel} {dataset} {result_method} M={m} seed={seed}"
                )
                fig.tight_layout()
                fig.savefig(out / "history.png", dpi=160)
                plt.close(fig)
            except Exception as exc:  # noqa: BLE001
                print(f"     history plot skipped: {exc}")
    return res, False


def append_row(row, logs_dir, *, replace=False):
    """Record one result in the shared CSV, extending the header if needed.

    Optional predictors add columns only for some methods, so the schema grows
    over a campaign; ``common.results_io`` rewrites the file under the lock
    when that happens and deduplicates by ``config_id``.
    """
    return rio.upsert_row_evolving_schema(
        logs_dir, row, IMPLEMENTATION, replace=replace
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=dm.DATASET_KEYS)
    ap.add_argument(
        "--methods", nargs="*", default=sv.BASE_METHOD_ORDER
    )
    ap.add_argument(
        "--kernel",
        default="rbf",
        help="single kernel to run",
    )
    ap.add_argument(
        "--kernels",
        nargs="*",
        default=None,
        help="kernels in the custom Cartesian-product plan.",
    )
    ap.add_argument(
        "--plan",
        choices=("custom", "report", "grid", "grid-exploratory"),
        default="report",
        help="experiment design. 'custom': the Cartesian product of "
             "--kernels and --datasets. 'report' (default): the fixed "
             "eight-pair paper campaign. 'grid': RBF and Matern-3/2 "
             "on all eight datasets. 'grid-exploratory': Gibbs on all eight "
             "datasets. "
             "Named plans can be filtered by --kernels/--datasets.",
    )
    ap.add_argument(
        "--m",
        nargs="*",
        type=int,
        default=[64, 128, 256, 512],
        help="inducing-point grid for custom/general plans; the report plan "
             "uses its centrally defined per-pair grids",
    )
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    ap.add_argument(
        "--runs-dir",
        default=str(RUNS_DIR),
        help="root directory for per-run metrics and curves.",
    )
    ap.add_argument(
        "--logs-dir",
        default=str(LOGS_DIR),
        help="directory for the shared result CSV.",
    )
    ap.add_argument(
        "--run-tag",
        default="",
        help="stable suite label saved in the config and every metrics row",
    )
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-test", type=int, default=5000)
    ap.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="train/test split seed; default: use each run's model seed",
    )
    ap.add_argument(
        "--split-rng",
        choices=jd.SUPPORTED_SPLIT_RNGS,
        default=jd.DEFAULT_SPLIT_RNG,
        help="deterministic split implementation; legacy JAX is archival only",
    )
    ap.add_argument(
        "--all-data",
        action="store_true",
        help="ignore --n-train/--n-test and use the full dataset with a 90/10 "
             "train/test split.",
    )
    ap.add_argument("--block", type=int, default=500, help="block size B")
    ap.add_argument(
        "--ll-batch-size",
        type=int,
        default=1000,
        help="number of points in the independent global likelihood/T batch",
    )
    ap.add_argument(
        "--blocks-per-step",
        type=int,
        default=2,
        help="number K of structural gain/logdet blocks per optimizer step",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=700,
        help="fixed number of training epochs.",
    )
    ap.add_argument("--refresh-every", type=int, default=40)
    ap.add_argument(
        "--refresh-until",
        type=int,
        default=None,
        help="stop SBT re-anchoring at this epoch (warmup-only refresh); "
             "default: refresh for the whole run; -1 selects epoch 350 for "
             "Gibbs and epoch 200 for stationary kernels.",
    )
    ap.add_argument(
        "--importance-refresh-every",
        type=int,
        default=10,
        help="recompute stopped-gradient IM probabilities on fixed blocks",
    )
    ap.add_argument(
        "--importance-epsilon",
        type=float,
        default=0.05,
        help="uniform mass in defensive gain importance sampling",
    )
    ap.add_argument("--full-batch", action="store_true",
                    help="one deterministic step per epoch over "
                         "ALL blocks (no minibatch sampling); "
                         "diagnostic: removes estimator noise.")
    ll_group = ap.add_mutually_exclusive_group()
    ll_group.add_argument(
        "--common-ll-batches",
        dest="common_ll_batches",
        action="store_true",
        help="share the exact likelihood partition and epoch stream across methods",
    )
    ll_group.add_argument(
        "--independent-ll-batches",
        dest="common_ll_batches",
        action="store_false",
        help="use method-specific likelihood streams (diagnostic only)",
    )
    ap.set_defaults(common_ll_batches=True)
    ap.add_argument(
        "--batch-coupling",
        choices=("split", "coupled"),
        default="split",
        help="split is the paper protocol; coupled is an SBT-only diagnostic "
             "that reuses the same selected blocks for likelihood and gain",
    )
    ap.add_argument(
        "--penalty-sampling",
        choices=("gain", "uniform"),
        default="gain",
        help="gain is the paper SBT importance sampler; uniform is used for "
             "the controlled split-vs-coupled batch ablation",
    )
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument(
        "--optimizer",
        choices=("adam", "lion"),
        default="adam",
        help="optimizer for kernel/noise and inducing locations (default: adam)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=5e-3,
        help="Adam/LION learning rate for kernel and noise",
    )
    ap.add_argument(
        "--z-lr",
        type=float,
        default=5e-3,
        help="Adam/LION learning rate for inducing locations Z",
    )
    ap.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="learning-rate schedule for model and inducing parameters",
    )
    ap.add_argument(
        "--lr-profile",
        choices=("fixed", "stable"),
        default="fixed",
        help="stable applies conservative kernel-specific model/Z/natural rates",
    )
    ap.add_argument("--grad-clip", type=float, default=10.0)
    objective_group = ap.add_mutually_exclusive_group()
    objective_group.add_argument(
        "--normalize-objective",
        dest="normalize_objective",
        action="store_true",
        help="divide the objective by N before clipping/model updates (default)",
    )
    objective_group.add_argument(
        "--raw-objective",
        dest="normalize_objective",
        action="store_false",
        help="use the full-scale objective for clipping/model updates",
    )
    ap.set_defaults(normalize_objective=True)
    ap.add_argument(
        "--q-optimizer",
        choices=("natural", "lion"),
        default="natural",
        help="optimizer for q(u); natural is the shared paper protocol",
    )
    ap.add_argument(
        "--natural-lr",
        type=float,
        default=5e-2,
        help="natural-gradient step size when --q-optimizer=natural",
    )
    ap.add_argument(
        "--sbt-geometry",
        choices=("anchor",),
        default="anchor",
        help="SBT training-block geometry: inducing-anchor distance",
    )
    ap.add_argument(
        "--phases",
        default=None,
        help="staged training schedule, e.g. \"10:theta,z 140:theta\": "
             "<epochs>:<frozen groups> per phase, lengths summing to --epochs. "
             "Groups are theta (kernel, noise, warp), z and q. Freezing is a "
             "gradient mask, so a frozen group accumulates no optimiser state "
             "either. Omit for a single unfrozen phase.",
    )
    ap.add_argument(
        "--metrics-at-epochs",
        nargs="*",
        type=int,
        default=None,
        help="record history at exactly these epochs, 0 included, instead of "
             "every --log-every. Needs --record-history.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-live", action="store_true",
                    help="disable live per-epoch print (keeps timing summary).")
    ap.add_argument(
        "--record-history",
        action="store_true",
        help="save per-log-point stochastic-objective/RMSE/NLL diagnostics.",
    )
    ap.add_argument(
        "--plot-history",
        action="store_true",
        help="also save a history.png curve when --record-history is active.",
    )
    ap.add_argument(
        "--save-params",
        action="store_true",
        help="save final trained parameters (params.npz) next to metrics.json.",
    )
    prediction_mode = ap.add_mutually_exclusive_group()
    prediction_mode.add_argument(
        "--defer-prediction",
        action="store_true",
        help="finish training and atomically save params/curve/final blocks plus "
             "training.json, but perform no test prediction. A later "
             "--prediction-only pass consumes those artifacts.",
    )
    prediction_mode.add_argument(
        "--prediction-only",
        action="store_true",
        help="load a completed training.json and params.npz, perform prediction "
             "without any optimizer step, and write metrics.json.",
    )
    ap.add_argument(
        "--prediction-reps",
        type=int,
        default=10,
        help="timed repetitions in the prediction pass after one warm-up; use a "
             "small value for expensive residual predictors.",
    )
    ap.add_argument(
        "--checkpoint-at-fraction",
        type=float,
        default=0.0,
        help="write checkpoint.npz (parameters AND optimizer state) at this "
             "fraction of the epoch budget, so another kernel can fork the "
             "run from there. 0 disables. Does not change any result.",
    )
    ap.add_argument(
        "--resume-from-kernel",
        type=str,
        default="",
        help="continue from the checkpoint.npz of the same dataset/M/method/"
             "seed trained with this kernel, instead of starting at epoch 0. "
             "The epoch budget must match the source run.",
    )
    ap.add_argument(
        "--resume-from-runs-dir",
        type=str,
        default="",
        help="runs tree holding the resume source (default: --runs-dir).",
    )
    ap.add_argument(
        "--warp-gate-fraction",
        type=float,
        default=0.75,
        help="Gibbs kernel only: fraction of the epoch budget trained with "
             "the warp gradient held at zero. Until this point the kernel is "
             "exactly stationary RBF with lengthscale 1.5*ell; afterwards the "
             "input-dependent lengthscales w_d are trained. Ignored by every "
             "stationary kernel.",
    )
    ap.add_argument(
        "--swa-fraction",
        type=float,
        default=0.0,
        help="tail fraction of epochs averaged with stochastic weight "
             "averaging; 0 disables it. All reported metrics, saved params "
             "and predictive diagnostics use the averaged parameters.",
    )
    ap.add_argument(
        "--init-from-kernel",
        type=str,
        default="",
        help="warm start from the params.npz of the same dataset/M/method/"
             "seed trained with this kernel (requires --save-params on the "
             "source run). The warp block is never inherited.",
    )
    ap.add_argument(
        "--init-from-runs-dir",
        type=str,
        default="",
        help="runs directory holding the warm-start source; defaults to "
             "--runs-dir.",
    )
    ap.add_argument(
        "--transplant-to-svgp",
        action="store_true",
        help="treat each --methods entry as a donor checkpoint, load its "
             "same-kernel parameters from --init-from-runs-dir, then train "
             "the standard sparse SVGP objective. Output methods are named "
             "<donor>-to-SVGP and record both roles in their manifest.",
    )
    ap.add_argument(
        "--full-predictor-max-n",
        type=int,
        default=FULL_PREDICTOR_MAX_N,
        help="largest training size for the Eq. (12) diagnostic; above it the "
             "N x N residual factorisation does not fit.",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="number of parallel GPU workers.")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="zero-based worker index in [0, num-shards).")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the configuration and print the sharded task count only.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    runtime = runtime_manifest()
    if not release_runtime_matches(runtime, SVGP_RELEASE_PACKAGES):
        raise SystemExit(
            "runtime package versions do not match Simulations/requirements.txt"
        )
    args.live = not args.no_live
    requested_datasets = list(args.datasets or [])
    requested_kernels = list(args.kernels or [])
    kernel_inputs = args.kernels if args.kernels is not None else [args.kernel]
    try:
        args.kernels = [sv.canonical_kernel_name(k) for k in kernel_inputs]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.kernel = args.kernels[0]
    if args.smoke:
        args.plan = "custom"
        args.datasets = ["kin40k"]
        args.m, args.seeds = [32], [0]
        args.n_train, args.n_test = 1500, 500
        args.block, args.epochs = 128, 30
        args.ll_batch_size, args.blocks_per_step = 128, 1
        args.refresh_every, args.log_every = 10, 10
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.prediction_reps < 0:
        raise SystemExit("--prediction-reps must be >= 0")
    if args.defer_prediction and not args.save_params:
        raise SystemExit("--defer-prediction requires --save-params")
    if args.block < 1 or args.ll_batch_size < 1:
        raise SystemExit("--block and --ll-batch-size must be >= 1")
    if args.blocks_per_step < 1:
        raise SystemExit("--blocks-per-step must be >= 1")
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("--shard-id must satisfy 0 <= shard-id < num-shards")

    bad = [x for x in args.methods if x not in sv.METHODS]
    if bad:
        raise SystemExit(
            f"unknown methods {bad}; "
            f"benchmark defaults: {sv.BASE_METHOD_ORDER}"
        )

    # (kernel, dataset) pairs. 'custom' = full product of --kernels x
    # --datasets. The named plans build the CROSS design
    # (fixed kernel over all datasets + full kernel sweep over a small
    # kernel-geometry subset) and override --kernels/--datasets.
    if args.plan == "custom":
        kd_pairs = [(k, d) for k in args.kernels for d in args.datasets]
    else:
        kd_pairs = plans.build_plan(args.plan)
        if requested_kernels:
            keep = set(args.kernels)
            kd_pairs = [(k, d) for k, d in kd_pairs if k in keep]
        if requested_datasets:
            keep = set(requested_datasets)
            kd_pairs = [(k, d) for k, d in kd_pairs if d in keep]
        if not kd_pairs:
            raise SystemExit(
                f"plan {args.plan!r} has no pairs after "
                f"kernel filter {requested_kernels or '*'} and "
                f"dataset filter {requested_datasets or '*'}"
            )
        args.kernels = sorted({k for k, _ in kd_pairs})
        args.datasets = sorted({d for _, d in kd_pairs})

    all_tasks = [
        (kernel_name, dataset, m, method, seed)
        for (kernel_name, dataset) in kd_pairs
        for m in plans.m_values_for_pair(
            args.plan, kernel_name, dataset, args.m
        )
        for method in args.methods
        for seed in args.seeds
    ]
    tasks = all_tasks[args.shard_id::args.num_shards]
    n_runs = len(tasks)
    if args.dry_run:
        print(
            f"Dry run OK: plan={args.plan} "
            f"kd_pairs={len(kd_pairs)} kernels={args.kernels} "
            f"methods={args.methods} datasets={len(args.datasets)} "
            f"M={'per-pair' if args.plan != 'custom' else args.m} "
            f"seeds={len(args.seeds)} global_runs={len(all_tasks)} "
            f"shard={args.shard_id}/{args.num_shards} shard_runs={n_runs}"
        )
        return
    print(f"Device: {jax.devices()[0]} (float64)  "
          f"shard={args.shard_id}/{args.num_shards} planned runs: {n_runs} "
          f"(global={len(all_tasks)})")
    done, failures = 0, 0
    for kernel_name, dataset, m, method, seed in tasks:
        done += 1
        tag = (
            f"[{done}/{n_runs}] {kernel_name} {dataset} "
            f"M{m} {method} seed{seed}"
        )
        # Everything per task is guarded: a failure in training, metric saving
        # or logging must not stop the remaining runs in a campaign shard.
        try:
            res, skipped = run_single(kernel_name, method, m, dataset, seed,
                                      args)
            if skipped:
                if res.get("prediction_deferred") is True:
                    print(f"{tag}  (training cached; prediction deferred)",
                          flush=True)
                else:
                    # Repairs a missing CSV row if a previous worker stopped
                    # after metrics.json but before the shared log.
                    append_row(res, args.logs_dir, replace=False)
                    print(f"{tag}  (cached)", flush=True)
                continue
            if res.get("prediction_deferred") is True:
                objective = res[
                    "last_epoch_stochastic_objective_estimate_per_datum"
                ]
                print(
                    f"{tag}  TRAINED stochastic-obj/N={objective:+.4f} "
                    f"params=saved search={res['search_s']:.1f}s "
                    f"{args.optimizer}={res['optimizer_s']:.1f}s "
                    f"grad={res['grad_ms']:.2f}ms prediction=deferred",
                    flush=True,
                )
                continue
            append_row(res, args.logs_dir, replace=True)
            objective = res[
                "last_epoch_stochastic_objective_estimate_per_datum"
            ]
            print(f"{tag}  stochastic-obj/N={objective:+.4f} "
                  f"rmse={res['rmse']:.4f} nll={res['nll']:.4f} "
                  f"search={res['search_s']:.1f}s "
                  f"{args.optimizer}={res['optimizer_s']:.1f}s "
                  f"grad={res['grad_ms']:.2f}ms", flush=True)
        except KeyboardInterrupt:
            raise
        except BaseException as e:  # noqa: BLE001
            failures += 1
            print(f"{tag}  FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
    print(f"Done. {done - failures}/{done} ok, {failures} failed.", flush=True)
    if failures:
        # Non-zero exit for visibility, but only AFTER every task was attempted.
        raise SystemExit(f"{failures} run(s) failed")


if __name__ == "__main__":
    main()
