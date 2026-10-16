#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collapsed structured bounds in JAX.

The collapsed engine of the SGPR stack. Everything differentiable is JAX; the
only host code is the per-gradient timing loop and the segment boundaries of
the training loop (logging, structure refresh, timing).

  * ``make_loss_exact(x, y)``        exact GP negative log marginal
  * ``make_loss(x, y, dense_mode)``  collapsed structured bound. The packed
        structure is a runtime argument with fixed shapes, so re-selecting it
        never recompiles. ``dense_mode``: None (block terms), "trace" (SGPR),
        "diag" (T-SGPR), "full" (dense residual endpoint).
  * ``train_collapsed(...)``         Adam/LION in a jitted ``lax.scan`` or
        SciPy L-BFGS-B with compiled JAX gradients, with structure re-selection
        ("search") timed separately from optimisation.
  * ``predict_jax`` / ``metrics_jax``  standard sparse predictor.
  * ``predict_collapsed`` / ``metrics_collapsed``  bound-matched predictor:
        the sparse one for the trace bound, Eq. (12) for every method that
        models the residual. The residual algebra lives in
        ``common.predictors`` and is shared with the variational stack.

float64 throughout.
"""
from __future__ import annotations

import math
import time
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from common import kernels as kn
from common import predictors as pr
from common import terms as tm
from sgpr import protocol

jax.config.update("jax_enable_x64", True)


def _pytree_abstract_signature(tree):
    """Hashable shape/dtype signature for an AOT-compiled pytree argument."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    leaves_signature = tuple(
        (
            tuple(getattr(leaf, "shape", ())),
            str(getattr(leaf, "dtype", type(leaf).__qualname__)),
            bool(getattr(leaf, "weak_type", False)),
        )
        for leaf in leaves
    )
    return treedef, leaves_signature


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def init_params(x, z, kernel_name="rbf"):
    """Collapsed hyperparameters at the shared initial point, for a given Z.

    ``benchmark.init_params`` samples Z from the training set and calls this;
    the residual illustration supplies its own Z. One definition, so the two
    cannot start from different points.
    """
    kernel_name = kn.canonical_kernel_name(kernel_name)
    params = {
        "log_ell": jnp.full((x.shape[1],), math.log(0.7)),
        "log_sf2": jnp.asarray(math.log(1.0)),
        "log_s2": jnp.asarray(math.log(0.1)),
        "z": jnp.asarray(z),
    }
    if kernel_name == "gibbs":
        # Cold start: zero warp is the stationary member of the Gibbs family.
        # L-BFGS optimises this full d x d matrix jointly with theta and Z.
        params["warp"] = jnp.zeros((x.shape[1], x.shape[1]), dtype=x.dtype)
    elif kernel_name == "amplitude-rbf":
        params["warp"] = jnp.zeros((x.shape[1],), dtype=x.dtype)
    return params


def make_loss_exact(x, y, kernel_name="rbf"):
    x, y = jnp.array(x), jnp.array(y).ravel()
    n = x.shape[0]
    kernel_name = kn.canonical_kernel_name(kernel_name)

    def loss(params, coeffs, idx, mask):
        ell = jnp.exp(params["log_ell"])
        sf2 = jnp.exp(params["log_sf2"])
        s2 = jnp.exp(params["log_s2"])
        kyy = kn.kernel(
            x, x, ell, sf2, kernel_name, params.get("warp")
        ) \
            + (s2 + 1e-8) * jnp.eye(n)
        ch = jnp.linalg.cholesky(kyy)
        al = jax.scipy.linalg.solve_triangular(ch, y, lower=True)
        return 0.5 * (al @ al) + jnp.sum(jnp.log(jnp.diag(ch))) \
            + 0.5 * n * jnp.log(2 * jnp.pi)

    return jax.jit(loss)


def make_loss(x, y, dense_mode=None, kernel_name="rbf"):
    """Collapsed structured bound F(theta) = logN(y;0,Qff+s2 I) - P/2."""
    x, y = jnp.array(x), jnp.array(y).ravel()
    n = x.shape[0]
    jit_eps = 1e-8
    kernel_name = kn.canonical_kernel_name(kernel_name)

    def loss(params, coeffs, idx, mask):
        ell = jnp.exp(params["log_ell"])
        sf2 = jnp.exp(params["log_sf2"])
        s2 = jnp.exp(params["log_s2"])
        z = params["z"]
        warp = params.get("warp")
        kuu = kn.kernel(z, z, ell, sf2, kernel_name, warp) \
            + jit_eps * jnp.eye(z.shape[0])
        kuf = kn.kernel(z, x, ell, sf2, kernel_name, warp)
        lm = jnp.linalg.cholesky(kuu)
        a = jax.scipy.linalg.solve_triangular(lm, kuf, lower=True)   # [M,N]
        big = jnp.eye(z.shape[0]) + (a @ a.T) / s2
        lb = jnp.linalg.cholesky(big)
        c = jax.scipy.linalg.solve_triangular(lb, a @ y, lower=True) / s2
        quad = (y @ y) / s2 - c @ c
        logdet = n * jnp.log(s2) + 2 * jnp.sum(jnp.log(jnp.diag(lb)))
        logn = -0.5 * (quad + logdet + n * jnp.log(2 * jnp.pi))
        if dense_mode == "trace":            # SGPR: trace penalty, O(NM)
            ddiag = kn.kernel_diag(x, ell, sf2, kernel_name, warp) - jnp.sum(a * a, 0)
            pen = jnp.sum(ddiag) / s2
        elif dense_mode == "diag":           # T-SGPR: diagonal log-det, O(NM)
            ddiag = kn.kernel_diag(x, ell, sf2, kernel_name, warp) - jnp.sum(a * a, 0)
            pen = jnp.sum(jnp.log(1.0 + ddiag / s2))
        elif dense_mode == "full":           # dense residual endpoint, O(N^3)
            dff = kn.kernel(x, x, ell, sf2, kernel_name, warp) - a.T @ a
            hfull = jnp.eye(n) + dff / s2
            hfull = 0.5 * (hfull + hfull.T) + 1e-8 * jnp.eye(n)
            _, pen = jnp.linalg.slogdet(hfull)
        else:                                # BT/SBT block terms
            # Residual blocks computed directly on the gathered inputs:
            # O(T S^2 (M + d + S)) per evaluation, never the dense N x N.
            xb = x[idx]                                          # [T,S,D]
            kb = kn.kernel(xb, xb, ell, sf2, kernel_name, warp)
            ab = jnp.transpose(a[:, idx], (1, 0, 2))             # [T,M,S]
            db = kb - jnp.einsum("tms,tmr->tsr", ab, ab)
            mm = mask[:, :, None] * mask[:, None, :]
            h = (jnp.eye(idx.shape[1]) + db / s2) * mm \
                + jnp.vectorize(jnp.diag, signature="(s)->(s,s)")(1.0 - mask)
            h = 0.5 * (h + jnp.swapaxes(h, 1, 2)) \
                + 1e-9 * jnp.eye(idx.shape[1])
            ch = jnp.linalg.cholesky(h)
            ld = 2 * jnp.sum(jnp.log(
                jnp.diagonal(ch, axis1=1, axis2=2)), axis=1)
            pen = jnp.sum(coeffs * ld)
        return -(logn - 0.5 * pen)

    return jax.jit(loss)


# --------------------------------------------------------------------------- #
# Per-gradient JAX timing
# --------------------------------------------------------------------------- #
def time_value_and_grad(loss, params, terms, reps=20):
    """Mean/std wall time of ONE value_and_grad call (the collapsed bound).

    Warm-up compiles first (excluded); each timed call is forced to completion
    with ``block_until_ready`` so async dispatch does not flatter the number.
    Returns (ms_mean, ms_std).
    """
    vg = jax.jit(jax.value_and_grad(lambda p, c, i, m: loss(p, c, i, m)))
    v, g = vg(params, *terms)
    jax.block_until_ready((v, g))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        v, g = vg(params, *terms)
        jax.block_until_ready((v, g))
        ts.append(time.perf_counter() - t)
    ts = np.asarray(ts)
    return float(ts.mean() * 1e3), float(ts.std() * 1e3)


# --------------------------------------------------------------------------- #
# Training: SciPy L-BFGS-B or first-order jitted scan; search timed separately
# --------------------------------------------------------------------------- #
def _train_scipy_lbfgs(
    loss,
    params0,
    *,
    terms0,
    term_fn,
    steps,
    refresh_every,
    train_z,
    log_every,
    log_cb,
    refresh_schedule,
    convergence_abs_tol,
    convergence_window_steps,
    convergence_patience,
    convergence_grad_max_tol,
    objective_scale,
):
    """Mature host-driven L-BFGS-B (``scipy-l-bfgs-b``) with restarts.

    SciPy owns line search and termination; JAX supplies one compiled float64
    value-and-gradient evaluation on the selected device.  Every structure
    refresh starts a new SciPy solve, because carrying inverse-Hessian history
    across a changed objective is invalid.  Once the final structure is fixed,
    one uninterrupted solve retains its curvature history.  Convergence is
    assessed retrospectively on consecutive windows of that same trajectory.
    """
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize

    host_refresh = bool(getattr(term_fn, "_host_refresh", False))
    refresh = (
        term_fn if host_refresh
        else jax.jit(term_fn) if term_fn is not None
        else None
    )
    if refresh_schedule is not None:
        phase_lengths = [int(value) for value in refresh_schedule]
        steps = int(sum(phase_lengths))
    elif refresh is not None:
        phase_lengths = []
        left = int(steps)
        width = max(1, int(refresh_every))
        while left > 0:
            length = min(width, left)
            phase_lengths.append(length)
            left -= length
    else:
        phase_lengths = [int(steps)]

    params, terms = params0, terms0
    value_and_grad = jax.jit(jax.value_and_grad(
        lambda fitted, live_terms: loss(fitted, *live_terms)
    ))

    # Compile once and account for it separately.  All official covers use the
    # same padded shapes, so refreshed values reuse this executable.
    compile_start = time.perf_counter()
    compiled_vg = value_and_grad.lower(params, terms).compile()
    compile_s = time.perf_counter() - compile_start

    curve = []
    convergence_checks = []
    scipy_attempts = []
    phase_actual_iters = []
    stable_windows = 0
    convergence_reached = False
    optimizer_terminal_convergence = False
    search_s = 0.0
    optimizer_s = 0.0
    refreshes = 0
    actual_steps = 0

    def tree_value_grad(fitted, live_terms):
        value, gradient = compiled_vg(fitted, live_terms)
        if not train_z and "z" in gradient:
            gradient = dict(gradient)
            gradient["z"] = jnp.zeros_like(gradient["z"])
        jax.block_until_ready((value, gradient))
        return value, gradient

    def run_attempt(
        fitted, live_terms, maxiter, *, monitor_convergence=False,
        phase_start_step=0,
    ):
        nonlocal optimizer_s, stable_windows, convergence_reached
        flat0, unravel = ravel_pytree(fitted)
        accepted_values = []

        scale = float(objective_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("objective_scale must be finite and positive")

        def evaluate(flat):
            """Return the per-scale objective to SciPy.

            Curves and reviewer-facing checks below remain in the original
            objective units.  For official SGPR fits ``scale=N``, so line
            search sees loss/N and is not made artificially ill-scaled by the
            number of observations.
            """
            candidate = unravel(jnp.asarray(flat))
            value, gradient = tree_value_grad(candidate, live_terms)
            flat_gradient, _ = ravel_pytree(gradient)
            return (
                float(value) / scale,
                np.asarray(flat_gradient, dtype=np.float64) / scale,
            )

        initial_scaled, _ = evaluate(np.asarray(flat0, dtype=np.float64))
        initial_value = initial_scaled * scale

        # SciPy 1.13 passes an OptimizeResult when the callback parameter has
        # this exact name, avoiding a duplicate value evaluation per accepted
        # iterate.
        def callback(intermediate_result):
            nonlocal stable_windows, convergence_reached
            accepted_values.append(float(intermediate_result.fun) * scale)
            if not monitor_convergence or convergence_abs_tol is None:
                return

            width = int(convergence_window_steps)
            iteration = len(accepted_values)
            prefix = int(maxiter) % width
            if (
                iteration < prefix + width
                or (iteration - prefix) % width != 0
            ):
                return

            window = np.asarray(
                [initial_value, *accepted_values], dtype=float
            )[-(width + 1):]
            delta = float(window[0] - window[-1])
            value_range = float(np.max(window) - np.min(window))
            objective_stable = (
                abs(delta) <= float(convergence_abs_tol)
                and value_range <= float(convergence_abs_tol)
            )
            _, flat_gradient_scaled = evaluate(
                np.asarray(intermediate_result.x, dtype=np.float64)
            )
            flat_gradient = flat_gradient_scaled * scale
            gradient_l2 = float(np.linalg.norm(flat_gradient))
            gradient_max = float(np.max(np.abs(flat_gradient)))
            gradient_stable = (
                convergence_grad_max_tol is None
                or gradient_max <= float(convergence_grad_max_tol)
            )
            stable = bool(objective_stable and gradient_stable)
            convergence_checks.append({
                "start_step": phase_start_step + iteration - width,
                "end_step": phase_start_step + iteration,
                "budget": width,
                "nit": width,
                "delta": delta,
                "range": value_range,
                "grad_l2": gradient_l2,
                "grad_max": gradient_max,
                "objective_stable": bool(objective_stable),
                "grad_stable": bool(gradient_stable),
                "scipy_success": False,
                "scipy_status": 99 if stable else 1,
                "stable": stable,
            })
            stable_windows = stable_windows + 1 if stable else 0
            if stable_windows >= int(convergence_patience):
                convergence_reached = True
                # SciPy documents StopIteration from the callback as its clean
                # early-stop mechanism; the current iterate is returned.
                raise StopIteration

        started = time.perf_counter()
        result = minimize(
            evaluate,
            np.asarray(flat0, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            callback=callback,
            options={
                "maxiter": int(maxiter),
                "maxfun": protocol.SCIPY_LBFGSB_MAXFUN,
                "maxcor": protocol.SCIPY_LBFGSB_MAXCOR,
                "maxls": protocol.SCIPY_LBFGSB_MAXLS,
                # External per-datum objective and full-gradient checks own the
                # release decision.  These values only prevent a numerically
                # pointless line search from continuing forever.
                "ftol": protocol.SCIPY_LBFGSB_FTOL,
                "gtol": protocol.SCIPY_LBFGSB_GTOL,
            },
        )
        optimizer_s += time.perf_counter() - started
        fitted = unravel(jnp.asarray(result.x))
        final_value, final_gradient = tree_value_grad(fitted, live_terms)
        final_value = float(final_value)
        gradient_leaves = jax.tree_util.tree_leaves(final_gradient)
        gradient_sq = sum(
            float(jnp.vdot(leaf, leaf)) for leaf in gradient_leaves
        )
        gradient_max = max(
            (float(jnp.max(jnp.abs(leaf))) for leaf in gradient_leaves),
            default=0.0,
        )
        # ``accepted_values`` has one entry per SciPy iteration.  Keep a
        # separate trajectory without the duplicate returned endpoint so a
        # 100-step convergence window contains exactly 101 objective values.
        trajectory = np.asarray(
            [initial_value, *accepted_values], dtype=float
        )
        values = np.asarray(
            [initial_value, *accepted_values, final_value], dtype=float
        )
        return (
            fitted,
            result,
            values,
            trajectory,
            math.sqrt(gradient_sq),
            gradient_max,
        )

    for phase, phase_budget in enumerate(phase_lengths):
        if refresh is not None and phase > 0:
            started = time.perf_counter()
            terms = refresh(params)
            jax.block_until_ready(terms[0])
            search_s += time.perf_counter() - started
            refreshes += 1

        in_final_phase = phase == len(phase_lengths) - 1
        start_step = actual_steps
        (
            params,
            result,
            values,
            trajectory,
            gradient_l2,
            gradient_max,
        ) = run_attempt(
            params,
            terms,
            int(phase_budget),
            monitor_convergence=(
                in_final_phase and convergence_abs_tol is not None
            ),
            phase_start_step=start_step,
        )
        nit = int(result.nit)
        actual_steps += nit
        phase_actual_iters.append(nit)

        # Record accepted iterates on a monotone global iteration axis.  The
        # returned endpoint is retained even if SciPy did not invoke a callback
        # for it (for example after a line-search termination).
        for offset, value in enumerate(values[1:-1], start=1):
            curve.append((start_step + offset, float(value)))
        curve.append((actual_steps, float(values[-1])))

        scipy_attempts.append({
            "phase": int(phase),
            "maxiter": int(phase_budget),
            "nit": nit,
            "nfev": int(result.nfev),
            "status": int(result.status),
            "success": bool(result.success),
            "message": str(result.message),
        })

        if log_cb is not None and log_every > 0:
            log_cb(
                actual_steps,
                params,
                float(values[-1]),
                1e3 * optimizer_s / max(1, actual_steps),
                terms,
            )

        if convergence_abs_tol is not None and in_final_phase:
            # A smooth fit can terminate before two complete windows either at
            # SciPy's stationary-point test or because line-search resolution
            # is exhausted near a stationary point.  Accept that terminal
            # certificate only when the available final-phase trajectory is
            # itself flat and the independently recomputed gradient passes.
            terminal_width = min(
                int(convergence_window_steps), max(0, nit)
            )
            terminal_window = trajectory[-(terminal_width + 1):]
            terminal_delta = float(
                terminal_window[0] - terminal_window[-1]
            )
            terminal_range = float(
                np.max(terminal_window) - np.min(terminal_window)
            )
            terminal_objective_stable = bool(
                abs(terminal_delta) <= float(convergence_abs_tol)
                and terminal_range <= float(convergence_abs_tol)
            )
            optimizer_terminal_convergence = bool(
                int(result.status) in (0, 2)
                and terminal_objective_stable
                and (
                    convergence_grad_max_tol is None
                    or gradient_max <= float(convergence_grad_max_tol)
                )
            )
            if optimizer_terminal_convergence and not convergence_reached:
                convergence_checks.append({
                    "start_step": actual_steps - terminal_width,
                    "end_step": actual_steps,
                    "budget": terminal_width,
                    "nit": terminal_width,
                    "delta": terminal_delta,
                    "range": terminal_range,
                    "grad_l2": float(gradient_l2),
                    "grad_max": float(gradient_max),
                    "objective_stable": terminal_objective_stable,
                    "grad_stable": True,
                    "scipy_success": bool(result.success),
                    "scipy_status": int(result.status),
                    "solver_terminal": True,
                    "stable": True,
                })
            convergence_reached = bool(
                convergence_reached or optimizer_terminal_convergence
            )

    return params, np.asarray(curve, dtype=float), {
        "search_s": search_s,
        "optimizer_s": optimizer_s,
        "compile_s": compile_s,
        "refreshes": refreshes,
        "phase_iters": phase_lengths,
        "phase_actual_iters": phase_actual_iters,
        "actual_steps": actual_steps,
        "convergence_reached": convergence_reached,
        "optimizer_terminal_convergence": optimizer_terminal_convergence,
        "convergence_checks": convergence_checks,
        "convergence_grad_max_tol": convergence_grad_max_tol,
        "optimizer_backend": "scipy-l-bfgs-b",
        "scipy_attempts": scipy_attempts,
        "final_terms": terms,
    }


def train_collapsed(loss, params0, *, terms0, term_fn=None, steps=3000,
                    refresh_every=600, optimizer="lion", lr=1e-2,
                    noise_lr=None, grad_clip=10.0, train_z=True, log_every=0,
                    log_cb=None, refresh_schedule=None,
                    convergence_abs_tol=None, convergence_window_steps=100,
                    convergence_patience=2, convergence_grad_max_tol=None,
                    objective_scale=1.0):
    """Collapsed-bound training in JAX, shared by every collapsed method.

    ``optimizer="adam"``  : robust first-order Adam (optax.adam).
    ``optimizer="lion"``  : first-order momentum (optax.lion).
    ``optimizer="lbfgs"`` : SciPy L-BFGS-B with JAX-compiled value/gradient.
        A fresh solve starts after every structure refresh, so curvature history
        is never carried across a changed objective.

    Adam and Lion run inside a jitted ``lax.scan``; SciPy calls the compiled
    JAX value/gradient from the host. Structure re-selection ('search') is
    normally a jitted JAX call timed in isolation. Host-side combinatorial
    selectors can mark ``term_fn._host_refresh = True`` and will be called
    outside XLA at refresh boundaries. ``noise_lr`` optionally slows only the
    Gaussian noise parameter ``log_s2`` for Adam/Lion. ``log_cb`` is called as
    ``log_cb(step, params, value, ms_per_step, terms)``; the trailing argument
    is the cover in force, which the block predictor needs. ``refresh_schedule`` uses
    an explicit Snelson-style phase schedule: the initial structure is used for
    phase one, then later phases refresh before optimisation. ``term_fn=None``
    => dense method, no refresh. Returns (params, curve, info) with
        ``optimizer_s`` as the optimiser wall time.
    """
    import optax

    opt_name = str(optimizer).lower()
    is_lbfgs = opt_name == "lbfgs"
    if is_lbfgs:
        return _train_scipy_lbfgs(
            loss,
            params0,
            terms0=terms0,
            term_fn=term_fn,
            steps=steps,
            refresh_every=refresh_every,
            train_z=train_z,
            log_every=log_every,
            log_cb=log_cb,
            refresh_schedule=refresh_schedule,
            convergence_abs_tol=convergence_abs_tol,
            convergence_window_steps=convergence_window_steps,
            convergence_patience=convergence_patience,
            convergence_grad_max_tol=convergence_grad_max_tol,
            objective_scale=objective_scale,
        )
    noise_lr = lr if noise_lr is None else float(noise_lr)
    noise_update_ratio = (
        1.0 if float(lr) == 0.0 else float(noise_lr) / float(lr)
    )
    chain = []
    if grad_clip and grad_clip > 0:
        chain.append(optax.clip_by_global_norm(grad_clip))
    if opt_name == "adam":
        chain.append(optax.adam(learning_rate=lr))
    elif opt_name == "lion":
        chain.append(optax.lion(learning_rate=lr))
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")
    opt = optax.chain(*chain)

    @partial(jax.jit, static_argnums=(3,))
    def segment(params, opt_state, terms, length):
        def mask_grad(g):
            if train_z or "z" not in g:
                return g
            return {k: (jnp.zeros_like(v) if k == "z" else v)
                    for k, v in g.items()}

        def body(carry, _):
            params, opt_state = carry
            val, g = jax.value_and_grad(
                lambda p: loss(p, *terms))(params)
            g = mask_grad(g)
            updates, opt_state = opt.update(g, opt_state, params)
            if noise_update_ratio != 1.0 and "log_s2" in updates:
                updates = dict(updates)
                updates["log_s2"] = (
                    updates["log_s2"] * noise_update_ratio
                )
            params = optax.apply_updates(params, updates)
            return (params, opt_state), val
        (params, opt_state), vals = jax.lax.scan(
            body, (params, opt_state), None, length=length)
        return params, opt_state, vals

    host_refresh = bool(getattr(term_fn, "_host_refresh", False))
    refresh = (
        term_fn if host_refresh
        else jax.jit(term_fn) if term_fn is not None
        else None
    )

    params, terms = params0, terms0
    opt_state = opt.init(params)
    seg_len = int(log_every) if log_every > 0 else int(refresh_every)
    if convergence_abs_tol is not None:
        # Optimisation is chunked into complete diagnostic windows. This keeps
        # the convergence decision independent of logging/refresh chunk sizes.
        seg_len = max(1, int(convergence_window_steps))
    curve, search_s, opt_s, compile_s, n_ref = [], 0.0, 0.0, 0.0, 0
    done = 0
    convergence_checks = []
    stable_windows = 0
    convergence_reached = False

    # Separate XLA compilation from the optimizer wall time (matches the SVGP
    # engine's compile_s / optimizer_s split). ``segment`` is jitted with the
    # scan length as a static arg, so cache by the complete dynamic-argument
    # signature as well as length. The timed ``opt_s`` below then measures pure
    # execution, never compilation.
    _compiled = {}

    def run_segment(params, opt_state, terms, n):
        nonlocal compile_s
        compile_key = (
            int(n),
            _pytree_abstract_signature(params),
            _pytree_abstract_signature(opt_state),
            _pytree_abstract_signature(terms),
        )
        exe = _compiled.get(compile_key)
        if exe is None:
            tc = time.perf_counter()
            exe = segment.lower(params, opt_state, terms, n).compile()
            compile_s += time.perf_counter() - tc      # compile() is synchronous
            _compiled[compile_key] = exe
        # ``n`` is baked into the AOT executable (static arg); do not re-pass it.
        return exe(params, opt_state, terms)
    if refresh_schedule is not None:
        phase_lengths = [int(p) for p in refresh_schedule]
        steps = int(sum(phase_lengths))
    elif refresh is not None:
        phase_lengths = []
        left = int(steps)
        width = max(1, int(refresh_every))
        while left > 0:
            n_phase = min(width, left)
            phase_lengths.append(n_phase)
            left -= n_phase
    else:
        phase_lengths = [int(steps)]

    for phase, phase_steps in enumerate(phase_lengths):
        if refresh is not None and phase > 0:
            t = time.perf_counter()
            terms = refresh(params)
            jax.block_until_ready(terms[0])
            search_s += time.perf_counter() - t
            n_ref += 1

        left = int(phase_steps)
        while left > 0:
            n = int(min(max(1, seg_len), left))
            if (
                convergence_abs_tol is not None
                and phase == len(phase_lengths) - 1
                and left == int(phase_steps)
            ):
                # Align complete windows with the MAXIMUM step. For the paper
                # schedule, 925 = 25 + 9*100: the 25-step prefix follows the
                # last refresh, and the last 100-step window ends at the cap.
                prefix = int(phase_steps) % int(convergence_window_steps)
                if prefix:
                    n = prefix
            t = time.perf_counter()
            params, opt_state, vals = run_segment(params, opt_state, terms, n)
            jax.block_until_ready(params["z"])
            opt_s += time.perf_counter() - t
            vals = np.asarray(jax.device_get(vals))
            # Always record the final PRE-update evaluation in a segment. A
            # bare stride would skip it whenever n is not a multiple of n // 5.
            # Callers needing the objective at returned params must re-evaluate
            # it after training (the optimiser curve is intentionally a trace
            # of values seen before each update).
            marks = list(range(0, n, max(1, n // 5)))
            if marks[-1] != n - 1:
                marks.append(n - 1)
            for k in marks:
                curve.append((done + k, float(vals[k])))
            done += n
            left -= n
            if log_cb is not None and log_every > 0:
                # ``terms`` is the cover in force for this segment. The block
                # predictor needs it, and it is not reconstructible from
                # ``params`` alone once a refresh has happened.
                log_cb(
                    done, params, float(vals[-1]),
                    1e3 * opt_s / max(1, done), terms,
                )
            # Refreshes happen only at phase boundaries. Convergence is tested
            # exclusively inside the LAST phase, where the fitted partition is
            # frozen. Two consecutive stable windows are required so one lucky
            # small line-search step cannot terminate an official run.
            if convergence_abs_tol is not None and phase == len(phase_lengths) - 1:
                # A complete segment is one convergence window. ``vals`` are
                # the pre-update evaluations within it; append a fresh value at
                # the returned parameters so the last update is also covered.
                # Requiring the RANGE (not only the two endpoints) rules out a
                # small net change caused by oscillation or line-search noise.
                if n == int(convergence_window_steps):
                    value, gradient = jax.value_and_grad(
                        lambda fitted: loss(fitted, *terms)
                    )(params)
                    jax.block_until_ready((value, gradient))
                    window_values = np.concatenate((vals, [float(value)]))
                    window_delta = float(vals[0]) - float(value)
                    window_range = float(np.max(window_values) - np.min(window_values))
                    gradient_sq = sum(
                        float(jnp.vdot(leaf, leaf))
                        for leaf in jax.tree_util.tree_leaves(gradient)
                    )
                    gradient_l2 = math.sqrt(gradient_sq)
                    gradient_max = max(
                        (
                            float(jnp.max(jnp.abs(leaf)))
                            for leaf in jax.tree_util.tree_leaves(gradient)
                        ),
                        default=0.0,
                    )
                    gradient_stable = (
                        convergence_grad_max_tol is None
                        or gradient_max <= float(convergence_grad_max_tol)
                    )
                    stable = (
                        abs(window_delta) <= float(convergence_abs_tol)
                        and window_range <= float(convergence_abs_tol)
                        and gradient_stable
                    )
                    convergence_checks.append({
                        "start_step": done - n,
                        "end_step": done,
                        "delta": window_delta,
                        "range": window_range,
                        "grad_l2": gradient_l2,
                        "grad_max": gradient_max,
                        "grad_stable": bool(gradient_stable),
                        "stable": bool(stable),
                    })
                    stable_windows = stable_windows + 1 if stable else 0
                    if stable_windows >= int(convergence_patience):
                        convergence_reached = True
                        left = 0
                        break
    info = {
        "search_s": search_s,
        "optimizer_s": opt_s,          # pure execution, compile excluded
        "compile_s": compile_s,        # one-time XLA compilation, tracked apart
        "refreshes": n_ref,
        "phase_iters": phase_lengths,
        "actual_steps": done,
        "convergence_reached": convergence_reached,
        "convergence_checks": convergence_checks,
        "convergence_grad_max_tol": convergence_grad_max_tol,
        # Internal hand-off to prediction. Callers must remove this tuple
        # before serialising ``info``.
        "final_terms": terms,
    }
    return params, curve, info


# --------------------------------------------------------------------------- #
# Prediction in JAX (RMSE / NLL on device)
# --------------------------------------------------------------------------- #
def predict_jax(params, x, y, xs, exact=False, jitter=1e-8,
                kernel_name="rbf"):
    """Predictive mean/variance in pure JAX. Returns (mu[Ns], var[Ns])."""
    x, y, xs = jnp.asarray(x), jnp.asarray(y).ravel(), jnp.asarray(xs)
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    s2 = jnp.exp(params["log_s2"])
    z = params["z"]
    warp = params.get("warp")
    n = x.shape[0]
    kernel_name = kn.canonical_kernel_name(kernel_name)
    if exact:
        kyy = kn.kernel(x, x, ell, sf2, kernel_name, warp) \
            + (s2 + jitter) * jnp.eye(n)
        L = jnp.linalg.cholesky(kyy)
        alpha = jax.scipy.linalg.cho_solve((L, True), y)
        ksx = kn.kernel(xs, x, ell, sf2, kernel_name, warp)
        mu = ksx @ alpha
        w = jax.scipy.linalg.solve_triangular(L, ksx.T, lower=True)
        var = kn.kernel_diag(xs, ell, sf2, kernel_name, warp) \
            - jnp.sum(w * w, axis=0) + s2
        return mu, jnp.maximum(var, 1e-10)

    m = z.shape[0]
    kuu = kn.kernel(z, z, ell, sf2, kernel_name, warp) + jitter * jnp.eye(m)
    kuf = kn.kernel(z, x, ell, sf2, kernel_name, warp)
    ksu = kn.kernel(xs, z, ell, sf2, kernel_name, warp)
    # Whitened collapsed posterior. Sigma = Kuu + Kuf Kfu / s2 is NEVER formed:
    # cond(Sigma) ~ cond(Kuu) * cond(B), which at M=256 with a long lengthscale
    # runs past 1e18 and breaks float64 even when Kuu alone factorises cleanly.
    # Factor Sigma = Lk B Lk^T instead -- Lk is the very factor the training
    # loss already takes (so a finite objective implies a finite predictor) and
    # B = I + A A^T / s2 has eigenvalues >= 1. Algebraically identical to the
    # dense form, ~7 decimal digits more accurate against extended precision.
    lk = jnp.linalg.cholesky(kuu)
    a_train = jax.scipy.linalg.solve_triangular(lk, kuf, lower=True)   # [M,N]
    a_test = jax.scipy.linalg.solve_triangular(lk, ksu.T, lower=True)  # [M,Ns]
    b = jnp.eye(m) + (a_train @ a_train.T) / s2
    lb = jnp.linalg.cholesky(0.5 * (b + b.T))
    mu = a_test.T @ jax.scipy.linalg.cho_solve((lb, True), a_train @ y) / s2
    a_sig = jax.scipy.linalg.solve_triangular(lb, a_test, lower=True)
    var = (kn.kernel_diag(xs, ell, sf2, kernel_name, warp)
           - jnp.sum(a_test * a_test, axis=0)
           + jnp.sum(a_sig * a_sig, axis=0) + s2)
    return mu, jnp.maximum(var, 1e-10)


def metrics_jax(params, x, y, xs, ys, exact=False, kernel_name="rbf"):
    """RMSE and Gaussian NLL on the test set, in JAX."""
    mu, var = predict_jax(params, x, y, xs, exact=exact,
                          kernel_name=kernel_name)
    return pr.gaussian_rmse_nll_device(mu, var, ys)



def _collapsed_predictive_components(params, x, y, xs, kernel_name, jitter):
    """Shared collapsed posterior and residual projections."""
    x = jnp.asarray(x)
    y = jnp.asarray(y).ravel()
    xs = jnp.asarray(xs)
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    s2 = jnp.exp(params["log_s2"])
    z = params["z"]
    warp = params.get("warp")
    kernel_name = kn.canonical_kernel_name(kernel_name)
    m = z.shape[0]
    kuu = kn.kernel(z, z, ell, sf2, kernel_name, warp) + jitter * jnp.eye(m)
    kuf = kn.kernel(z, x, ell, sf2, kernel_name, warp)
    kus = kn.kernel(z, xs, ell, sf2, kernel_name, warp)
    # Same whitening as predict_jax: Sigma = lk B lk^T, never formed densely,
    # so E-SGPR / BT / SBT / METIS share the conditioning fix with SGPR.
    lk = jnp.linalg.cholesky(kuu)
    a_train = jax.scipy.linalg.solve_triangular(lk, kuf, lower=True)
    a_test = jax.scipy.linalg.solve_triangular(lk, kus, lower=True)
    b = jnp.eye(m) + (a_train @ a_train.T) / s2
    lb = jnp.linalg.cholesky(0.5 * (b + b.T))

    mean = a_test.T @ jax.scipy.linalg.cho_solve((lb, True), a_train @ y) / s2
    a_sig = jax.scipy.linalg.solve_triangular(lb, a_test, lower=True)
    qvar = jnp.sum(a_sig * a_sig, axis=0)
    d_train = jnp.maximum(
        kn.kernel_diag(x, ell, sf2, kernel_name, warp)
        - jnp.sum(a_train * a_train, axis=0),
        0.0,
    )
    d_test = jnp.maximum(
        kn.kernel_diag(xs, ell, sf2, kernel_name, warp)
        - jnp.sum(a_test * a_test, axis=0),
        0.0,
    )
    return {
        "x": x,
        "xs": xs,
        "ell": ell,
        "sf2": sf2,
        "s2": s2,
        "kernel_name": kernel_name,
        # Bound once so the shared residual predictors never need to know which
        # kernel or hyperparameters produced these projections.
        "kernel_fn": lambda xa, xb: kn.kernel(
            xa, xb, ell, sf2, kernel_name, warp
        ),
        "mean": mean,
        "qvar": qvar,
        "a_train": a_train,
        "a_test": a_test,
        "d_train": d_train,
        "d_test": d_test,
    }


def collapsed_predictive_components(params, x, y, xs, *, kernel_name="rbf",
                                    jitter=1e-8):
    """Public accessor for the projections every collapsed predictor needs.

    Both the benchmark and the figures build their predictors from this dict
    plus ``common.predictors``, so the predictive covariance a run reports and
    the one a figure draws are produced by one function.
    """
    return _collapsed_predictive_components(
        params, x, y, xs, kn.canonical_kernel_name(kernel_name), jitter
    )


def predict_collapsed(params, x, y, xs, *, predictor, blocks=None,
                      kernel_name="rbf", jitter=1e-8,
                      return_contributions=False):
    """Bound-matched predictive mean and variance for one collapsed method.

    ``predictor`` selects the sparse, diagonal, fitted-block or dense residual
    covariance. ``blocks`` is required by the fitted-block predictor and is
    optional for decomposing the dense diagnostic.
    """
    components = collapsed_predictive_components(
        params, x, y, xs, kernel_name=kernel_name, jitter=jitter
    )
    return pr.predictive_variance(
        components, predictor=predictor, blocks=blocks, jitter=jitter,
        return_contributions=return_contributions,
    )


def metrics_collapsed(params, x, y, xs, ys, *, predictor, blocks=None,
                      kernel_name="rbf", jitter=1e-8):
    """RMSE and NLL of the bound-matched collapsed predictor."""
    mu, var = predict_collapsed(
        params, x, y, xs, predictor=predictor, blocks=blocks,
        kernel_name=kernel_name, jitter=jitter,
    )
    return pr.gaussian_rmse_nll(mu, var, ys)


def predictive_diagnostics_from_prediction_jax(params, mu, var, ys):
    """Calibration diagnostics for the exact predictor used by a result row."""
    mu = jnp.asarray(mu).ravel()
    var = jnp.asarray(var).ravel()
    ys = jnp.asarray(ys).ravel()
    residual = ys - mu
    log_term = 0.5 * jnp.mean(jnp.log(2.0 * jnp.pi * var))
    quad_term = 0.5 * jnp.mean(residual ** 2 / var)
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    s2 = jnp.exp(params["log_s2"])
    pred_sd = jnp.sqrt(var)
    rmse = jnp.sqrt(jnp.mean(residual ** 2))
    return {
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
        "pred_sd_mean": jnp.mean(pred_sd),
        "mean_abs_residual": jnp.mean(jnp.abs(residual)),
        "std_resid_sq_mean": jnp.mean(residual ** 2 / var),
        "rmse_over_pred_sd": rmse / jnp.maximum(jnp.mean(pred_sd), 1e-12),
        "nll_log_term": log_term,
        "nll_quad_term": quad_term,
    }


# --------------------------------------------------------------------------- #
# Log-determinant captured-mass diagnostic (theory measures, one-shot at final
# theta). Quantifies how much of the residual log-determinant / off-diagonal
# residual energy each structured penalty captures, relative to the loose SGPR
# trace bound and the exact dense residual endpoint (E-SGPR).
# --------------------------------------------------------------------------- #
def _residual_dff(params, x, kernel_name="rbf", jitter=1e-8):
    """Dense residual Dff = Kff - Qff and its diagonal at fixed theta."""
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    z = params["z"]
    warp = params.get("warp")
    m = z.shape[0]
    kernel_name = kn.canonical_kernel_name(kernel_name)
    kuu = kn.kernel(z, z, ell, sf2, kernel_name, warp) + jitter * jnp.eye(m)
    kuf = kn.kernel(z, x, ell, sf2, kernel_name, warp)
    lm = jnp.linalg.cholesky(kuu)
    a = jax.scipy.linalg.solve_triangular(lm, kuf, lower=True)     # [M,N]
    kff = kn.kernel(x, x, ell, sf2, kernel_name, warp)
    dff = kff - a.T @ a
    dff = 0.5 * (dff + dff.T)
    ddiag = kn.kernel_diag(x, ell, sf2, kernel_name, warp) - jnp.sum(a * a, 0)
    return dff, jnp.maximum(ddiag, 0.0)


def logdet_capture_jax(params, x, *, kernel_name="rbf", blocks=None,
                       jitter=1e-8):
    """Residual log-determinant capture and off-block energy at fixed theta.

    All penalties are the residual-log-determinant corrections that appear in
    the collapsed bound P/2 (see paper, Eq. for the SGPR/T-SGPR/E-SGPR/block
    objectives), evaluated at the SAME theta:

      * ``pen_trace``  = trace(Dff)/s2                      (SGPR upper bound)
      * ``pen_diag``   = sum log(1 + Dff_ii/s2)             (T-SGPR)
      * ``pen_dense``  = logdet(I + Dff/s2)                 (E-SGPR endpoint)
      * ``pen_block``  = sum_b logdet(I + Dff[b,b]/s2)      (BT/SBT partition)

    Ordering (theory): pen_dense <= pen_block <= pen_diag <= pen_trace. The
    fraction of the trace->dense log-det gap that a penalty closes is

        captured = (pen_trace - pen_penalty) / (pen_trace - pen_dense) in [0,1]

    (0 for SGPR/trace, 1 for the dense endpoint). ``logdet_gap`` = pen_block -
    pen_dense is the absolute distance to the endpoint. ``energy_frac`` is the
    share of total off-diagonal squared residual mass sum_{i!=j} Dff_ij^2 that
    lies INSIDE the blocks; its random-partition baseline is (b-1)/(N-1).

    ``blocks``: list of 1-D index arrays (a partition) for BT/SBT; None for the
    dense/trace/diag-only summary.
    """
    x = jnp.asarray(x)
    s2 = jnp.exp(params["log_s2"])
    n = int(x.shape[0])
    dff, ddiag = _residual_dff(params, x, kernel_name, jitter)

    pen_trace = jnp.sum(ddiag) / s2
    pen_diag = jnp.sum(jnp.log1p(ddiag / s2))
    hfull = jnp.eye(n) + dff / s2
    hfull = 0.5 * (hfull + hfull.T) + jitter * jnp.eye(n)
    _, pen_dense = jnp.linalg.slogdet(hfull)

    denom = pen_trace - pen_dense
    off_sq_total = jnp.sum(dff * dff) - jnp.sum(ddiag * ddiag)

    out = {
        "logdet_pen_trace": pen_trace,
        "logdet_pen_diag": pen_diag,
        "logdet_pen_dense": pen_dense,
        "logdet_off_sq_total": off_sq_total,
        "logdet_diag_captured": (pen_trace - pen_diag)
        / jnp.where(denom == 0, jnp.nan, denom),
    }

    if blocks is None:
        return out

    pen_block = 0.0
    energy_in = 0.0
    sizes = []
    for b in blocks:
        b = jnp.asarray(b, dtype=jnp.int32)
        if b.size == 0:
            continue
        sizes.append(int(b.size))
        dbb = dff[jnp.ix_(b, b)]
        dbb = 0.5 * (dbb + dbb.T)
        hb = jnp.eye(b.size) + dbb / s2 + jitter * jnp.eye(b.size)
        _, ld = jnp.linalg.slogdet(hb)
        pen_block = pen_block + ld
        diag_b = jnp.diagonal(dbb)
        energy_in = energy_in + (jnp.sum(dbb * dbb) - jnp.sum(diag_b * diag_b))

    captured = (pen_trace - pen_block) / jnp.where(denom == 0, jnp.nan, denom)
    b_mean = float(np.mean(sizes)) if sizes else 0.0
    random_baseline = (b_mean - 1.0) / (n - 1.0) if n > 1 else float("nan")
    out.update({
        "logdet_pen_block": pen_block,
        "logdet_captured": captured,          # fraction of trace->dense gap
        "logdet_gap": pen_block - pen_dense,   # absolute distance to endpoint
        "energy_frac": energy_in / jnp.where(off_sq_total == 0, jnp.nan,
                                             off_sq_total),
        "energy_random_baseline": random_baseline,   # (b-1)/(N-1)
        "block_mean_size": b_mean,
        "n_blocks_diag": len(sizes),
    })
    return out
