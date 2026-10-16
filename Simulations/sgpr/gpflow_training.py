#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""libraries.training -- shared GPflow/TF training utilities.

Single source of truth for the L-BFGS-B training loop used by all
GPflow experiments (sgpr/snelson, sgpr/residual_illustration).

Public API
----------
split_iterations(total_iter, n_phases) -> list[int]
    Distribute total_iter across n_phases as evenly as possible.

make_compiled_value_and_grad(model) -> (fun, get_x, set_x)
    Build a single tf.function (loss + gradient) that is never re-traced
    when the structure Variables inside StructuredSGPR are refreshed.
    Returns three callables:
        fun(x)  -> (float loss, np.ndarray grad)
        get_x() -> np.ndarray of current trainable parameters (flat)
        set_x(x) -> assign flat parameter vector back to model Variables

fit_model_static(model, maxiter, compile=True) -> (losses, time, stats)
    L-BFGS-B with no structure refresh, via gpflow.optimizers.Scipy.
    Use for the non-refresh baselines and fixed RBT-SGPR blocks.

fit_model_limited_refreshes(model, total_iter, n_refreshes, refresh_fn,
                            initial_info=None, name="model")
    -> (losses, time, last_info, stats)
    ROUNDS phases, each: refresh structure → L-BFGS-B for phase_iter steps.
    Uses make_compiled_value_and_grad so the tf.function is traced once.
    Use for SBT-SGPR block refreshes.
"""

from __future__ import annotations

import time

import gpflow
import numpy as np
import scipy.optimize
import tensorflow as tf


# ---------------------------------------------------------------------------
# Iteration splitting
# ---------------------------------------------------------------------------

def split_iterations(total_iter: int, n_phases: int) -> list[int]:
    """Distribute ``total_iter`` L-BFGS steps across ``n_phases`` evenly."""
    total_iter = int(total_iter)
    n_phases   = max(1, int(n_phases))
    base, rem  = divmod(total_iter, n_phases)
    return [base + (1 if i < rem else 0) for i in range(n_phases)]


# ---------------------------------------------------------------------------
# Compiled value-and-gradient (used by fit_model_limited_refreshes)
# ---------------------------------------------------------------------------

def make_compiled_value_and_grad(model):
    """Single tf.function for loss + gradient; never re-traced on refresh.

    Structure refreshes inside StructuredSGPR only assign the padded
    Variables, so this compiled function stays valid for the whole training
    run (a single trace instead of one per phase).

    Returns
    -------
    fun(x)   : (float, np.ndarray)  –  (loss, flat gradient)
    get_x()  : np.ndarray           –  current flat parameter vector
    set_x(x) : None                 –  assign flat vector back to Variables
    """
    variables = model.trainable_variables
    sizes = [int(np.prod(v.shape)) for v in variables]

    @tf.function
    def _eval(x_flat):
        parts = tf.split(x_flat, sizes)
        for v, p in zip(variables, parts):
            v.assign(tf.reshape(p, v.shape))
        with tf.GradientTape() as tape:
            loss = model.training_loss()
        grads = tape.gradient(loss, variables)
        grads = [
            tf.zeros_like(v) if g is None else g
            for g, v in zip(grads, variables)
        ]
        flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return loss, flat

    def fun(x):
        loss, grad = _eval(tf.convert_to_tensor(x, dtype=tf.float64))
        return float(loss.numpy()), np.asarray(grad.numpy(), dtype=np.float64)

    def get_x():
        return np.concatenate(
            [np.reshape(v.numpy(), [-1]) for v in variables]
        ).astype(np.float64)

    def set_x(xv):
        offset = 0
        for v, size in zip(variables, sizes):
            v.assign(np.reshape(xv[offset:offset + size], v.shape))
            offset += size

    return fun, get_x, set_x


# ---------------------------------------------------------------------------
# Static training (no structure refresh)
# ---------------------------------------------------------------------------

def fit_model_static(model, maxiter: int, compile: bool = True, metric_fn=None,
                     record_curve: bool = True, tol: float = 1e-11):
    """L-BFGS-B via gpflow.optimizers.Scipy, no structure refresh.

    Parameters
    ----------
    model   : any gpflow model with a ``training_loss`` method
    maxiter : maximum L-BFGS-B iterations
    compile : whether to tf.function-compile the loss (default True)

    Returns
    -------
    losses : np.ndarray   – loss recorded at each optimizer step
    dt     : float        – wall time in seconds
    stats  : dict         – {iterations, callback_count, status, message}
    """
    t0 = time.perf_counter()
    losses: list[float] = []
    metrics: list[float] = []
    callback_count = 0

    def _rec_metric():
        if metric_fn is not None:
            metrics.append(float(metric_fn()))

    def step_callback(*_):
        nonlocal callback_count
        callback_count += 1
        if record_curve:                      # extra forward + host sync/iter
            losses.append(float(model.training_loss().numpy()))
            _rec_metric()

    losses.append(float(model.training_loss().numpy()))
    _rec_metric()

    opt = gpflow.optimizers.Scipy()
    result = opt.minimize(
        model.training_loss,
        model.trainable_variables,
        method="L-BFGS-B",
        options={"maxiter": int(maxiter)},
        tol=tol,
        step_callback=step_callback,
        compile=compile,
    )

    final_loss = float(model.training_loss().numpy())
    if not np.isclose(losses[-1], final_loss):
        losses.append(final_loss)
        _rec_metric()

    stats = {
        "iterations":    int(getattr(result, "nit",     callback_count)),
        "callback_count": int(callback_count),
        "status":        int(getattr(result, "status",  0)),
        "message":       str(getattr(result, "message", "")),
        "metric_history": metrics,
    }
    return np.asarray(losses, dtype=float), time.perf_counter() - t0, stats


# ---------------------------------------------------------------------------
# Training with structure refreshes
# ---------------------------------------------------------------------------

def fit_model_limited_refreshes(
    model,
    total_iter:   int,
    n_refreshes:  int,
    refresh_fn,
    initial_info=None,
    name:         str = "model",
    phase_iters=None,
    metric_fn=None,
    record_curve: bool = True,
    tol: float = 1e-11,
):
    """Train with ``n_refreshes`` structure refreshes.

    The initial structure in ``initial_info`` is used for the first phase;
    later phases call ``refresh_fn()`` before optimisation. A single
    tf.function is compiled once and reused across all phases.

    Parameters
    ----------
    model        : StructuredSGPR (or any model with trainable_variables)
    total_iter   : total L-BFGS-B iterations, split evenly across phases
    n_refreshes  : number of phases (= number of structure refreshes)
    refresh_fn   : callable() → info dict; called at the start of each phase
    initial_info : info dict from the model's initial structure (optional)
    name         : string label for progress printing

    Returns
    -------
    losses    : np.ndarray  – concatenated per-phase loss traces
    dt        : float       – total wall time in seconds
    last_info : dict        – info returned by the last refresh_fn() call
    stats     : dict        – {iterations, callback_count, refreshes, phase_iters}
    """
    initial_search_s = (
        float(initial_info.get("_initial_search_s", 0.0))
        if isinstance(initial_info, dict) else 0.0
    )
    t0 = time.perf_counter()
    if phase_iters is None:
        phase_iters = split_iterations(total_iter, n_refreshes)
    else:
        phase_iters = [int(p) for p in phase_iters]
        n_refreshes = len(phase_iters)
    fun, get_x, set_x = make_compiled_value_and_grad(model)

    all_losses:     list[float] = []
    all_metrics:    list[float] = []
    last_info = initial_info
    total_cb_count  = 0
    total_opt_iters = 0

    def _metric():
        return float(metric_fn()) if metric_fn is not None else None

    for phase, phase_steps in enumerate(phase_iters, start=1):
        if phase == 1 and initial_info is not None:
            last_info = initial_info
        else:
            last_info = refresh_fn()

        phase_losses = [fun(get_x())[0]]
        phase_metrics = [_metric()] if metric_fn is not None else []
        cb_count = 0

        def step_callback(xk, _count=None, _pl=phase_losses, _pm=phase_metrics):
            nonlocal cb_count
            cb_count += 1
            if record_curve:                  # extra value+grad eval per iter
                _pl.append(fun(xk)[0])
                if metric_fn is not None:
                    _pm.append(_metric())

        result = scipy.optimize.minimize(
            fun,
            get_x(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": int(phase_steps)},
            tol=tol,
            callback=step_callback,
        )
        set_x(result.x)
        final_phase = fun(get_x())[0]
        if not np.isclose(phase_losses[-1], final_phase):
            phase_losses.append(final_phase)
            if metric_fn is not None:
                phase_metrics.append(_metric())

        all_losses.extend(phase_losses if phase == 1 else phase_losses[1:])
        if metric_fn is not None:
            all_metrics.extend(phase_metrics if phase == 1 else phase_metrics[1:])
        total_cb_count  += int(cb_count)
        total_opt_iters += int(getattr(result, "nit", cb_count))

        # Pretty-print refresh info.
        if isinstance(last_info, dict) and "anchor_mode" in last_info:
            info_text = (
                f", anchor_mode={last_info.get('anchor_mode', '')},"
                f" split_mode={last_info.get('split_mode', '')},"
                f" cost={last_info.get('cost', 0.0):.2e},"
                f" chunks={last_info.get('anchor_chunks', '')}"
            )
        else:
            info_text = ""

        print(
            f"{name:16s} refresh={phase:02d}/{n_refreshes:02d}"
            f" phase_iter={phase_steps:02d}"
            f" loss={float(model.training_loss().numpy()):.6f}"
            f"{info_text}"
        )

    final_loss = float(model.training_loss().numpy())
    if not all_losses or not np.isclose(all_losses[-1], final_loss):
        all_losses.append(final_loss)
        if metric_fn is not None:
            all_metrics.append(_metric())

    stats = {
        "iterations":    int(total_opt_iters),
        "callback_count": int(total_cb_count),
        "refreshes":     int(n_refreshes),
        "phase_iters":   phase_iters,
        "metric_history": all_metrics,
    }
    elapsed = initial_search_s + time.perf_counter() - t0
    return np.asarray(all_losses, dtype=float), elapsed, last_info, stats
