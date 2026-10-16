#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark core of the collapsed stack.

Single source of truth for the collapsed-SGPR family. The runners are thin
drivers: they read data from disk and call ``train_and_eval`` / ``save_run`` /
``build_tables`` here.

Default methods are SGPR, T-SGPR, RBT-SGPR and SBT-SGPR. Optional dense
diagnostics are Exact-GP (true GP), E-SGPR (dense residual endpoint) and
METIS-SGPR (a dense residual-graph oracle, deliberately not scalable).

Everything differentiable is JAX. The SBT selector is a cheap host refresh from
``common.partitions``; the same builder feeds the variational stack, so the two
benchmarks cannot select different geometries.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from sgpr import jax_collapsed as jm
from sgpr import initialization as initz
from sgpr import protocol
from common import data_split as jd
from common import kernels as kn
from common import partitions as pt
from common import predictors as pr
from common import results_io as rio
from common import stats as st
from common import terms as tm
from common.kernels import canonical_kernel_name
from common.provenance import code_state, config_fingerprint, runtime_manifest

# v23: squared-residual METIS weights, per-datum L-BFGS scaling, nested
# RBF-to-Gibbs warm starts, and a 100k resumable convergence ceiling.
IMPLEMENTATION = "sgpr-collapsed-sbt-jax-v23"

KERNEL_ORDER = list(kn.KERNEL_ORDER)


def parse_refresh_schedule(value):
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        phases = [int(v) for v in value]
    else:
        phases = [
            int(token.strip())
            for token in str(value).replace(";", ",").split(",")
            if token.strip()
        ]
    if not phases:
        return None
    if any(p <= 0 for p in phases):
        raise ValueError("refresh schedule phases must be positive integers")
    return phases


def configure_refresh_protocol(args):
    """Resolve refresh flags in one place for every SGPR runner.

    ``--refresh-schedule`` has priority and reproduces the Snelson protocol:
    the initial structure is used for phase one and later phases refresh before
    optimisation. Therefore a schedule of length K means K structures and K-1
    post-initial refreshes.
    """
    schedule = parse_refresh_schedule(getattr(args, "refresh_schedule", None))
    args.refresh_schedule = schedule
    if schedule is not None:
        args.steps = int(sum(schedule))
        args.refresh_every = int(args.steps + 1)
        return args
    if getattr(args, "refreshes", None) is not None:
        if args.refreshes < 0:
            raise ValueError("refreshes must be >= 0")
        args.refresh_every = (
            int(args.steps) + 1 if args.refreshes == 0
            else max(1, -(-int(args.steps) // (int(args.refreshes) + 1)))
        )
    return args


def objective_convergence_summary(
    curve,
    *,
    final_loss: float,
    n_points: int,
    total_steps: int,
    window_steps: int = protocol.CONVERGENCE_WINDOW_STEPS,
    tolerance: float = protocol.CONVERGENCE_OBJ_N_TOL,
):
    """Measure objective stability over a fixed final optimisation window.

    ``curve`` stores pre-update objective evaluations.  ``final_loss`` is
    recomputed at the returned parameters, avoiding the one-step stale value
    that v14 wrote as ``obj_n``.  The start point is the last recorded value no
    later than ``total_steps - window_steps``; the final fixed-partition phase
    in the release schedule is long enough to contain the whole window.
    """
    values = np.asarray(curve, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        return {
            "convergence_passed": False,
            "convergence_tail_start_step": -1,
            "convergence_tail_span_steps": 0,
            "convergence_tail_delta_obj_n": float("nan"),
            "convergence_tail_range_obj_n": float("nan"),
        }
    target = int(total_steps) - int(window_steps)
    candidates = np.flatnonzero(values[:, 0] <= target)
    if candidates.size == 0:
        start_index = 0
    else:
        start_index = int(candidates[-1])
    start_step = int(values[start_index, 0])
    tail_losses = np.concatenate((values[start_index:, 1], [float(final_loss)]))
    scale = float(n_points)
    delta = (float(values[start_index, 1]) - float(final_loss)) / scale
    tail_range = (float(np.max(tail_losses)) - float(np.min(tail_losses))) / scale
    span = max(0, int(total_steps) - start_step)
    enough_span = span >= int(window_steps)
    finite = math.isfinite(delta) and math.isfinite(tail_range)
    passed = bool(
        enough_span
        and finite
        and abs(delta) <= float(tolerance)
        and tail_range <= float(tolerance)
    )
    return {
        "convergence_passed": passed,
        "convergence_tail_start_step": start_step,
        "convergence_tail_span_steps": span,
        "convergence_tail_delta_obj_n": delta,
        "convergence_tail_range_obj_n": tail_range,
    }


def resolve_block_size(n_points, args):
    """Resolve block controls without changing old defaults.

    ``--block-size`` has priority when present; otherwise keep the historical
    ``B = ceil(N / n_blocks)`` convention.
    """
    explicit = getattr(args, "block_size", None)
    if explicit is not None:
        block = max(1, int(explicit))
        n_blocks = int(math.ceil(int(n_points) / block))
        return block, n_blocks
    n_blocks = max(1, int(args.n_blocks))
    block = int(math.ceil(int(n_points) / n_blocks))
    return block, n_blocks

# method -> loss mode. Values are consumed by train_and_eval:
#   "exact" => exact GP negative log marginal (Exact-GP);
#   "full"  => dense SGPR residual log-det endpoint (E-SGPR);
#   "terms" => structured block terms.
METHODS = {
    "Exact-GP": "exact",
    "SGPR":    "trace",
    "T-SGPR":  "diag",
    "E-SGPR":  "full",
    "RBT-SGPR": "terms",
    "BT-SGPR": "terms",  # archived-name compatibility
    "SBT-SGPR": "terms",
    # Dense residual-graph oracle: same block objective, METIS partition.
    "METIS-SGPR": "terms",
}
METHOD_ORDER = ["Exact-GP", "SGPR", "T-SGPR", "E-SGPR", "RBT-SGPR", "SBT-SGPR",
                "METIS-SGPR"]

# The registry is owned by ``common.predictors`` so the benchmark, the Snelson
# figure and the variational stack cannot disagree about which estimator a
# method reports. Re-exported here because this is where callers look for it.
METHOD_PREDICTOR = pr.COLLAPSED_METHOD_PREDICTOR
DEFAULT_METHOD_ORDER = ["SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR"]

# Eq. (12) factorises an N x N residual: O(N^2) memory, O(N^3) time. At the
# paper's N = 8000 that is 0.5 GB and a few seconds on a GPU. The cap is a
# guard rail, not a fallback -- exceeding it stops the run rather than quietly
# reporting a different estimator.
DEFAULT_FULL_PREDICTOR_MAX_N = 50000

# One official runner invocation evaluates every method for the same split.
# The primary selector depends only on that split, theta0 and M, so cache its
# indices instead of repeating the O(N M^2) greedy calculation per method.
# Warm-up reselections pass a fitted ``theta`` and deliberately bypass this
# cache because their kernels differ by method and cycle.
_INITIAL_INDEX_CACHE = {}


def timed_eval(fn, warmup=1, reps=3):
    """Return (result, wall_seconds) for a JAX-returning callable.

    The clock is honest for JAX: one warmup call is discarded so XLA
    compilation / tracing is NOT charged to the timing, and every timed call is
    forced to completion with ``block_until_ready`` so asynchronous dispatch
    cannot understate the wall time. Reports the MEDIAN over ``reps`` calls.
    """
    out = fn()
    jax.block_until_ready(out)                       # warmup: compile excluded
    for _ in range(max(0, warmup - 1)):
        out = fn(); jax.block_until_ready(out)
    ts = []
    for _ in range(max(1, reps)):
        t = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t)
    return out, float(np.median(ts))


def init_params(
    x_tr,
    m,
    seed,
    kernel_name="rbf",
    *,
    strategy="random",
    nested_max_m=None,
    theta=None,
):
    """Initialise theta and select Z under one explicit policy.

    Exploratory callers retain the historical random default.  The release
    runner fingerprints and passes greedy-variance (with k-means++ fallback).
    ``theta`` lets the variational-EM warm-up reselect Z under the currently
    fitted kernel without resetting theta/noise.
    """
    base = (
        jm.init_params(x_tr, x_tr[:int(m)], kernel_name=kernel_name)
        if theta is None
        else {key: jnp.asarray(value) for key, value in theta.items()}
    )
    cache_key = None
    if theta is None and strategy != "random":
        cache_key = (
            id(x_tr), tuple(x_tr.shape), str(x_tr.dtype), int(m), int(seed),
            str(kernel_name), str(strategy), nested_max_m,
        )
    indices = _INITIAL_INDEX_CACHE.get(cache_key)
    if indices is None:
        indices = initz.inducing_indices(
            x_tr,
            int(m),
            base,
            kernel_name=kernel_name,
            seed=int(seed),
            strategy=strategy,
            nested_max_m=nested_max_m,
        )
        if cache_key is not None:
            _INITIAL_INDEX_CACHE[cache_key] = np.asarray(indices).copy()
    fitted = dict(base)
    fitted["z"] = jnp.asarray(x_tr)[jnp.asarray(indices)]
    return fitted


# --------------------------------------------------------------------------- #
# Structure term callbacks
# --------------------------------------------------------------------------- #
# Dense residual-graph oracle. The residual build and METIS partition are host
# work included in ``search_s``; this is intentionally not a scalable method.

def _residual_matrix(x, params, kernel_name, jitter=1e-8):
    """Dense residual D_ff = K_ff - Q_ff at the current theta (host numpy).

    O(N^2) memory / O(N^2 M) time -- feasible only at medium N (this is the
    'oracle' regime the review calls for); guard with the run's N.
    """
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    z = params["z"]
    m = z.shape[0]
    xk = jnp.asarray(x)
    kernel_id = canonical_kernel_name(kernel_name)
    warp = params.get("warp")
    kuu = kn.kernel(z, z, ell, sf2, kernel_id, warp) + jitter * jnp.eye(m)
    kuf = kn.kernel(z, xk, ell, sf2, kernel_id, warp)
    lm = jnp.linalg.cholesky(kuu)
    a = jax.scipy.linalg.solve_triangular(lm, kuf, lower=True)          # [M,N]
    kff = kn.kernel(xk, xk, ell, sf2, kernel_id, warp)
    dff = kff - a.T @ a
    return np.asarray(jax.device_get(0.5 * (dff + dff.T)))


def _knn_residual_graph(dff, k=15):
    """Symmetric kNN graph on D_ff^2 (squared-residual weights). Returns
    (indptr, indices, weights) CSR-style arrays for the undirected graph."""
    n = dff.shape[0]
    a = np.square(np.asarray(dff, dtype=np.float64))
    np.fill_diagonal(a, 0.0)
    k = int(min(max(1, k), n - 1))
    nbr = np.argpartition(-a, kth=k - 1, axis=1)[:, :k]     # top-k per row
    rows = np.repeat(np.arange(n), k)
    cols = nbr.reshape(-1)
    w = a[rows, cols]
    # symmetrise (union of directed kNN)
    r2 = np.concatenate([rows, cols])
    c2 = np.concatenate([cols, rows])
    w2 = np.concatenate([w, w])
    keep = r2 != c2
    return r2[keep], c2[keep], w2[keep], n


def _metis_weighted_csr(edge_maps):
    """Return the flat weighted CSR representation required by PyMetis."""
    xadj = [0]
    adjncy = []
    eweights = []
    for neighbors in edge_maps:
        for vertex in sorted(neighbors):
            adjncy.append(int(vertex))
            eweights.append(int(neighbors[vertex]))
        xadj.append(len(adjncy))
    return xadj, adjncy, eweights


def _order_metis(x, params, block, kernel_name):
    """Partition the thresholded residual graph, order points by partition.

    The expensive residual build and METIS partition are the search cost being
    measured. Import failure is fatal so a METIS-SGPR run cannot silently become
    a different spectral baseline.
    """
    try:
        import pymetis
    except ImportError as exc:
        raise RuntimeError(
            "METIS-SGPR requires pymetis; install Simulations/requirements.txt"
        ) from exc

    kernel_name = canonical_kernel_name(kernel_name)
    dff = _residual_matrix(x, params, kernel_name)
    n = dff.shape[0]
    nparts = max(1, (n + int(block) - 1) // int(block))
    r, c, w, n = _knn_residual_graph(dff, k=15)
    edge_maps = [dict() for _ in range(n)]
    scale = 1e6 / (w.max() + 1e-30)
    for i, j, wij in zip(r, c, w):
        i, j = int(i), int(j)
        weight = int(max(1, wij * scale))
        edge_maps[i][j] = max(edge_maps[i].get(j, 0), weight)
    xadj, adjncy, eweights = _metis_weighted_csr(edge_maps)
    _, membership = pymetis.part_graph(
        nparts, xadj=xadj, adjncy=adjncy, eweights=eweights
    )
    labels = np.asarray(membership, dtype=np.int64)
    return np.argsort(labels, kind="stable").astype(np.int32)


def method_blocks(method, x, params, block, args, seed):
    """Final-theta block partition a structured method actually uses.

    Returns a list of index arrays for BT/SBT, ``"diag"`` for the singleton
    (T-SGPR) partition, ``"dense"`` for the full endpoint (Exact-GP/E-SGPR), or
    ``None`` for the trace bound (SGPR).
    """
    n = int(x.shape[0])
    if method in ("RBT-SGPR", "BT-SGPR"):
        return tm.unpack_terms(pt.bt_terms_device(n, block, seed))
    if method in ("SBT-SGPR", "METIS-SGPR"):
        # reuse the method's own host builder so captured-mass uses the real
        # partition each baseline trains on.
        refresh = make_term_fn(method, x, block, args, jitter=1e-8,
                               structure_seed=seed)
        return tm.unpack_terms(refresh(params))
    if method == "T-SGPR":
        return "diag"
    if method in ("Exact-GP", "E-SGPR"):
        return "dense"
    return None                                   # SGPR (trace)


def logdet_capture_for(method, x, params, block, args, seed, kernel_name,
                       jitter=1e-8):
    """Theory captured-mass measures at final theta for one method.

    Always fills the dense/trace/diag summary; adds block/captured/energy for
    the structured methods. Captured is 0 for SGPR (trace) and 1 for the dense
    endpoint by definition.
    """
    part = method_blocks(method, x, params, block, args, seed)
    blocks = part if isinstance(part, list) else None
    caps = jm.logdet_capture_jax(params, x, kernel_name=kernel_name,
                                 blocks=blocks, jitter=jitter)
    caps = {k: float(v) for k, v in jax.device_get(caps).items()}
    if part == "dense":
        caps["logdet_captured"] = 1.0
        caps["logdet_gap"] = 0.0
    elif part is None:                            # SGPR trace bound
        caps["logdet_captured"] = 0.0
    elif part == "diag":                          # T-SGPR diagonal
        caps["logdet_captured"] = caps.get("logdet_diag_captured", float("nan"))
    if isinstance(part, list):
        caps["logdet_partition"] = f"blocks({len(part)})"
    elif isinstance(part, str):
        caps["logdet_partition"] = part
    else:
        caps["logdet_partition"] = "trace"        # SGPR (no partition)
    return caps


def make_term_fn(method, x, block, args, jitter, structure_seed=None, m=None):
    """Return the structure refresh a method uses, or None for dense methods.

    Host-side selectors carry ``_host_refresh`` so the training loop calls them
    outside XLA at refresh boundaries instead of trying to trace them.
    """
    kernel_name = canonical_kernel_name(getattr(args, "kernel", "rbf"))

    if method in ("RBT-SGPR", "BT-SGPR"):
        seed = args.seed if structure_seed is None else int(structure_seed)
        terms = pt.bt_terms_device(x.shape[0], block, seed)  # param-free
        return lambda params: terms
    seed = args.seed if structure_seed is None else int(structure_seed)
    host_builders = {
        "SBT-SGPR": lambda params: pt.sbt_terms_device(
            x, params, block, kernel_name),
        "METIS-SGPR": lambda params: tm.pack_order(
            _order_metis(x, params, block, kernel_name), block, x.shape[0]),
    }
    if method in host_builders:
        refresh = host_builders[method]
        refresh._host_refresh = True
        return refresh
    return None


def make_run_config(method, m, seed, dataset, args, data_signature):
    kernel_name = canonical_kernel_name(getattr(args, "kernel", "rbf"))
    state = code_state(Path(__file__).resolve().parents[2])
    config = {
        "implementation": IMPLEMENTATION,
        "method": method,
        "kernel": kernel_name,
        "M": int(m),
        "seed": int(seed),
        "model_seed": int(getattr(args, "model_seed", seed)),
        "split_seed": int(getattr(args, "split_seed", seed)),
        "split_rng": str(getattr(args, "split_rng", jd.DEFAULT_SPLIT_RNG)),
        "preprocessing_protocol": jd.PREPROCESSING_PROTOCOL,
        "dataset": dataset,
        "data_signature": str(data_signature),
        "data_source": str(getattr(args, "data_source", "unspecified")),
        "code_commit": state["commit"],
        "code_source_dirty": state["source_dirty"],
        "n_train": int(args.n_train),
        "n_test": int(args.n_test),
        "n_blocks": int(args.n_blocks),
        "block_size": (
            None if getattr(args, "block_size", None) is None
            else int(getattr(args, "block_size"))
        ),
        "steps": int(args.steps),
        "refresh_every": int(args.refresh_every),
        "refresh_schedule": (
            None if getattr(args, "refresh_schedule", None) is None
            else [int(v) for v in getattr(args, "refresh_schedule")]
        ),
        "target_refreshes": (
            None if getattr(args, "refreshes", None) is None
            else int(args.refreshes)
        ),
        "log_every": int(args.log_every),
        "optimizer": str(getattr(args, "optimizer", "lion")),
        "optimizer_backend": (
            "scipy-l-bfgs-b"
            if str(getattr(args, "optimizer", "lion")) == "lbfgs"
            else "optax"
        ),
        "scipy_lbfgsb_ftol": protocol.SCIPY_LBFGSB_FTOL,
        "scipy_lbfgsb_gtol": protocol.SCIPY_LBFGSB_GTOL,
        "scipy_lbfgsb_maxcor": protocol.SCIPY_LBFGSB_MAXCOR,
        "scipy_lbfgsb_maxfun": protocol.SCIPY_LBFGSB_MAXFUN,
        "scipy_lbfgsb_maxls": protocol.SCIPY_LBFGSB_MAXLS,
        "lr": float(args.lr),
        "noise_lr": (
            None if getattr(args, "noise_lr", None) is None
            else float(getattr(args, "noise_lr"))
        ),
        "grad_clip": float(args.grad_clip),
        "train_inducing": not bool(getattr(args, "no_train_inducing", False)),
        "initialization_policy": list(protocol.INITIALIZATION_POLICY),
        "initialization_nested_max_m": protocol.INITIALIZATION_NESTED_MAX_M,
        "warmup_max_cycles": protocol.WARMUP_MAX_CYCLES,
        "warmup_theta_steps": protocol.WARMUP_THETA_STEPS,
        "warmup_reselect_obj_n_tol": protocol.WARMUP_RESELECT_OBJ_N_TOL,
        "optimizer_objective_scale": "per-datum",
        "gibbs_warm_start": getattr(args, "warm_start_donor", None),
        "gibbs_warm_start_params_sha256": getattr(
            args, "warm_start_params_sha256", None
        ),
        "require_convergence": bool(getattr(args, "require_convergence", False)),
        "convergence_window_steps": int(protocol.CONVERGENCE_WINDOW_STEPS),
        "convergence_obj_n_tol": float(protocol.CONVERGENCE_OBJ_N_TOL),
        "convergence_grad_max_n_tol": float(
            protocol.CONVERGENCE_GRAD_MAX_N_TOL
        ),
        "convergence_patience": int(protocol.CONVERGENCE_PATIENCE),
        "full_predictor_max_n": int(getattr(
            args, "full_predictor_max_n", DEFAULT_FULL_PREDICTOR_MAX_N
        )),
        "save_block_contributions": bool(getattr(
            args, "save_block_contributions", False
        )),
        # Declared so a row states which estimator produced its NLL without
        # anyone having to look the method up in a table.
        "reported_predictor": METHOD_PREDICTOR[method],
    }
    return config, config_fingerprint(config)


def _dummy_terms():
    return (
        jnp.zeros(1),
        jnp.zeros((1, 1), jnp.int32),
        jnp.zeros((1, 1)),
    )


def _build_terms(params, dense_mode, raw_fn):
    if dense_mode != "terms":
        return _dummy_terms()
    terms = raw_fn(params)
    jax.block_until_ready(terms[0])
    return terms


def variational_em_warmup(
    method,
    params0,
    *,
    loss,
    dense_mode,
    raw_fn,
    x_train,
    m,
    seed,
    kernel_name,
    strategy,
):
    """Precondition theta/Z before the single release-critical joint solve.

    The warm-up is deliberately finite and fingerprinted.  It never certifies
    a result: certification belongs exclusively to the subsequent joint solve
    on the final frozen residual structure.
    """
    if method == "Exact-GP":
        return params0, [], {
            "steps": 0,
            "optimizer_s": 0.0,
            "search_s": 0.0,
            "compile_s": 0.0,
        }

    n = int(x_train.shape[0])
    params = params0
    records = []
    totals = {
        "steps": 0,
        "optimizer_s": 0.0,
        "search_s": 0.0,
        "compile_s": 0.0,
    }

    for cycle in range(protocol.WARMUP_MAX_CYCLES):
        search_started = time.perf_counter()
        terms = _build_terms(params, dense_mode, raw_fn)
        totals["search_s"] += time.perf_counter() - search_started

        fitted, _, info = jm.train_collapsed(
            loss,
            params,
            terms0=terms,
            term_fn=None,
            steps=protocol.WARMUP_THETA_STEPS,
            refresh_every=protocol.WARMUP_THETA_STEPS + 1,
            optimizer="lbfgs",
            lr=0.0,
            noise_lr=None,
            grad_clip=0.0,
            train_z=False,
            log_every=0,
            log_cb=None,
            refresh_schedule=[protocol.WARMUP_THETA_STEPS],
            convergence_abs_tol=protocol.CONVERGENCE_OBJ_N_TOL * n,
            convergence_window_steps=protocol.CONVERGENCE_WINDOW_STEPS,
            convergence_patience=1,
            convergence_grad_max_tol=(
                protocol.CONVERGENCE_GRAD_MAX_N_TOL * n
            ),
            objective_scale=n,
        )
        info.pop("final_terms", None)
        totals["steps"] += int(info.get("actual_steps", 0))
        totals["optimizer_s"] += float(info.get("optimizer_s", 0.0))
        totals["compile_s"] += float(info.get("compile_s", 0.0))

        search_started = time.perf_counter()
        baseline_terms = _build_terms(fitted, dense_mode, raw_fn)
        baseline_value = float(loss(fitted, *baseline_terms))
        candidate = init_params(
            x_train,
            m,
            seed,
            kernel_name=kernel_name,
            strategy=strategy,
            # Theta is already M-specific after the conditional maximisation;
            # there is no cross-M nesting claim at this stage.
            nested_max_m=None,
            theta=fitted,
        )
        candidate_terms = _build_terms(candidate, dense_mode, raw_fn)
        candidate_value = float(loss(candidate, *candidate_terms))
        totals["search_s"] += time.perf_counter() - search_started

        improvement_n = (baseline_value - candidate_value) / n
        accepted = bool(np.isfinite(candidate_value) and improvement_n > 0.0)
        params = candidate if accepted else fitted
        records.append({
            "cycle": int(cycle + 1),
            "theta_steps": int(info.get("actual_steps", 0)),
            "theta_convergence_reached": bool(
                info.get("convergence_reached", False)
            ),
            "reselection_improvement_obj_n": float(improvement_n),
            "reselection_accepted": accepted,
        })
        if improvement_n < protocol.WARMUP_RESELECT_OBJ_N_TOL:
            break

    return params, records, totals


# --------------------------------------------------------------------------- #
# One training run (no IO) -> (result dict, curve array)
# --------------------------------------------------------------------------- #
def train_and_eval(method, m, seed, dataset, data, args, jitter=1e-8):
    x_tr, y_tr, x_te, y_te, meta = data
    dense_mode = METHODS[method]
    kernel_name = canonical_kernel_name(getattr(args, "kernel", "rbf"))
    n = x_tr.shape[0]
    block, resolved_n_blocks = resolve_block_size(n, args)
    strategy = str(getattr(
        args, "initialization_strategy", protocol.INITIALIZATION_POLICY[0]
    ))
    initialization_started = time.perf_counter()
    supplied = getattr(args, "_initial_params", None)
    if supplied is None:
        params0 = init_params(
            x_tr,
            m,
            seed,
            kernel_name=kernel_name,
            strategy=("random" if method == "Exact-GP" else strategy),
            nested_max_m=(
                None if method == "Exact-GP"
                else protocol.INITIALIZATION_NESTED_MAX_M
            ),
        )
    else:
        params0 = {key: jnp.asarray(value) for key, value in supplied.items()}
        expected_z = (int(m), int(x_tr.shape[1]))
        if tuple(params0["z"].shape) != expected_z:
            raise ValueError(
                f"warm-start Z has shape {params0['z'].shape}, expected "
                f"{expected_z}"
            )
    initialization_s = time.perf_counter() - initialization_started
    if dense_mode == "exact":
        loss = jm.make_loss_exact(x_tr, y_tr, kernel_name=kernel_name)
    else:
        loss = jm.make_loss(x_tr, y_tr,
                            dense_mode=None if dense_mode == "terms"
                            else dense_mode,
                            kernel_name=kernel_name)

    raw_fn = None
    if dense_mode == "terms":
        raw_fn = make_term_fn(
            method, x_tr, block, args, jitter, structure_seed=seed, m=m
        )

    params0, warmup_records, warmup = variational_em_warmup(
        method,
        params0,
        loss=loss,
        dense_mode=dense_mode,
        raw_fn=raw_fn,
        x_train=x_tr,
        m=m,
        seed=seed,
        kernel_name=kernel_name,
        strategy=strategy,
    )

    if dense_mode != "terms":
        terms0, loop_fn = _dummy_terms(), None
        initial_search_s = initialization_s + warmup["search_s"]
    else:
        host_terms = bool(getattr(raw_fn, "_host_refresh", False))
        compiled_raw_fn = raw_fn if host_terms else jax.jit(raw_fn)
        t_search = time.perf_counter()
        terms0 = compiled_raw_fn(params0)
        jax.block_until_ready(terms0[0])
        initial_search_s = (
            initialization_s
            + warmup["search_s"]
            + time.perf_counter() - t_search
        )
        loop_fn = None if method in ("RBT-SGPR", "BT-SGPR") else raw_fn

    refresh_schedule = getattr(args, "refresh_schedule", None)
    steps = args.steps                                 # SAME for all methods
    if refresh_schedule is not None:
        n_ref = (max(0, len(refresh_schedule) - 1)
                 if loop_fn is not None else 0)
        phase_text = f"phases={list(refresh_schedule)}"
    else:
        n_ref = (max(0, (steps - 1) // args.refresh_every)
                 if loop_fn is not None else 0)
        phase_text = f"every {args.refresh_every}"
    print(f"  -> {method} {dataset} M{m} seed{seed}: steps={steps} "
          f"refreshes={n_ref} ({phase_text}) "
          f"n_blocks={resolved_n_blocks} B={block}")

    grad_ms, grad_ms_std = jm.time_value_and_grad(loss, params0, terms0)
    print(f"     per-gradient: {grad_ms:.2f}+/-{grad_ms_std:.2f} ms/grad (JAX)")

    live = getattr(args, "live", True)
    predictor = METHOD_PREDICTOR[method]
    exact_gp_predictive = predictor == "exact"
    full_cap = int(getattr(args, "full_predictor_max_n", DEFAULT_FULL_PREDICTOR_MAX_N))
    if predictor == "full" and n > full_cap:
        raise SystemExit(
            f"{method} reports Eq. (12), which needs an {n}x{n} residual "
            f"factorisation, but --full-predictor-max-n is {full_cap}. Raise the "
            "cap or reduce --n-train; silently falling back to another "
            "predictor would put two estimators in one column."
        )

    def evaluation_metrics(p_now, terms_now=None):
        """Per-log-point metrics, under the predictor this method reports.

        The block predictor conditions on the cover, which changes at every
        structure refresh, so the live cover is passed in rather than rebuilt.
        """
        if exact_gp_predictive:
            return jm.metrics_jax(
                p_now, x_tr, y_tr, x_te, y_te, exact=True,
                kernel_name=kernel_name,
            )
        live_blocks = None
        if predictor == "block" and terms_now is not None:
            live_blocks = tm.unpack_terms(terms_now)
        return jm.metrics_collapsed(
            p_now, x_tr, y_tr, x_te, y_te, predictor=predictor,
            blocks=live_blocks, kernel_name=kernel_name, jitter=jitter,
        )

    def log_cb(step, p_now, train_val, ms, terms_now=None):
        r, nl = evaluation_metrics(p_now, terms_now)
        opt_name = getattr(args, "optimizer", "lion")
        obj_n = train_val / n
        print(f"     step {step:5d}/{steps}  obj/N={obj_n:+.4f}  "
              f"elbo/N={-obj_n:+.4f}  "
              f"rmse={float(r):.4f}  nll={float(nl):.4f}  "
              f"{ms:.2f} ms/{opt_name}-step")

    t0 = time.perf_counter()
    params, curve, info = jm.train_collapsed(
        loss, params0, terms0=terms0, term_fn=loop_fn, steps=steps,
        refresh_every=args.refresh_every,
        optimizer=getattr(args, "optimizer", "lion"),
        lr=args.lr,
        noise_lr=getattr(args, "noise_lr", None),
        grad_clip=args.grad_clip,
        train_z=not bool(getattr(args, "no_train_inducing", False)),
        log_every=args.log_every,
        log_cb=(log_cb if live else None),
        refresh_schedule=refresh_schedule,
        convergence_abs_tol=(
            protocol.CONVERGENCE_OBJ_N_TOL * n
            if bool(getattr(args, "require_convergence", False))
            else None
        ),
        convergence_window_steps=protocol.CONVERGENCE_WINDOW_STEPS,
        convergence_patience=protocol.CONVERGENCE_PATIENCE,
        convergence_grad_max_tol=(
            protocol.CONVERGENCE_GRAD_MAX_N_TOL * n
            if bool(getattr(args, "require_convergence", False))
            else None
        ),
        objective_scale=n)
    final_terms = info.pop("final_terms")
    total_s = (
        initial_search_s
        + warmup["optimizer_s"]
        + warmup["compile_s"]
        + time.perf_counter() - t0
    )
    search_s = initial_search_s + info["search_s"]
    optimizer_s = warmup["optimizer_s"] + info["optimizer_s"]
    compile_s = warmup["compile_s"] + info.get("compile_s", 0.0)
    opt_name = getattr(args, "optimizer", "lion").upper()
    print(f"     TIMING {method}: search/refresh={search_s:.2f}s  "
          f"{opt_name}-opt={optimizer_s:.2f}s  compile={compile_s:.2f}s  "
          f"per-grad={grad_ms:.2f}ms  (refreshes={info['refreshes']})")

    # Re-evaluate at the RETURNED parameters. Values emitted inside the
    # optimiser scan are pre-update evaluations, so curve[-1] is one L-BFGS
    # update stale and cannot certify either the final objective or convergence.
    final_value, final_grad = jax.value_and_grad(
        lambda fitted: loss(fitted, *final_terms)
    )(params)
    jax.block_until_ready(final_value)
    final_loss = float(final_value)
    gradient_leaves = jax.tree_util.tree_leaves(final_grad)
    grad_sq = sum(float(jnp.vdot(leaf, leaf)) for leaf in gradient_leaves)
    grad_max = max(
        (float(jnp.max(jnp.abs(leaf))) for leaf in gradient_leaves),
        default=float("nan"),
    )
    actual_steps = int(info.get("actual_steps", steps))
    if info.get("optimizer_backend") == "scipy-l-bfgs-b" and \
            info.get("convergence_checks"):
        last_check = info["convergence_checks"][-1]
        convergence = {
            "convergence_passed": bool(info.get("convergence_reached", False)),
            "convergence_tail_start_step": int(last_check["start_step"]),
            "convergence_tail_span_steps": int(last_check["nit"]),
            "convergence_tail_delta_obj_n": (
                float(last_check["delta"]) / n
            ),
            "convergence_tail_range_obj_n": (
                float(last_check["range"]) / n
            ),
        }
    else:
        convergence = objective_convergence_summary(
            curve,
            final_loss=final_loss,
            n_points=n,
            total_steps=actual_steps,
        )
    if bool(getattr(args, "require_convergence", False)):
        convergence["convergence_passed"] = bool(
            convergence["convergence_passed"]
            and info.get("convergence_reached", False)
            and grad_max / n <= protocol.CONVERGENCE_GRAD_MAX_N_TOL
        )
    final_grad_l2_n = math.sqrt(grad_sq) / n
    final_grad_max_n = grad_max / n
    failure_reasons = []
    if not bool(info.get("convergence_reached", False)):
        checks = info.get("convergence_checks") or []
        if checks and not bool(checks[-1].get("objective_stable", True)):
            failure_reasons.append("objective")
        if checks and not bool(checks[-1].get("grad_stable", True)):
            failure_reasons.append("gradient")
        if not failure_reasons:
            failure_reasons.append("patience")
    if final_grad_max_n > protocol.CONVERGENCE_GRAD_MAX_N_TOL \
            and "gradient" not in failure_reasons:
        failure_reasons.append("gradient")
    convergence.update({
        "convergence_window_steps": protocol.CONVERGENCE_WINDOW_STEPS,
        "convergence_obj_n_tol": protocol.CONVERGENCE_OBJ_N_TOL,
        "convergence_grad_max_n_tol": protocol.CONVERGENCE_GRAD_MAX_N_TOL,
        "convergence_patience": protocol.CONVERGENCE_PATIENCE,
        "optimizer_convergence_reached": bool(
            info.get("convergence_reached", False)
        ),
        "optimizer_terminal_convergence": bool(
            info.get("optimizer_terminal_convergence", False)
        ),
        "max_steps": steps,
        "final_grad_l2_per_datum": final_grad_l2_n,
        "final_grad_max_per_datum": final_grad_max_n,
        "convergence_failure_reasons": failure_reasons,
        "optimizer_backend": info.get("optimizer_backend", "optax"),
        "phase_actual_iters": info.get("phase_actual_iters", []),
        "convergence_checks": info.get("convergence_checks", []),
        "scipy_attempts": info.get("scipy_attempts", []),
    })
    print(
        "     CONVERGENCE "
        f"tail={convergence['convergence_tail_range_obj_n']:.3e}/datum "
        f"over {convergence['convergence_tail_span_steps']} steps "
        f"grad_max/N={convergence['final_grad_max_per_datum']:.3e} "
        f"optimizer={convergence['optimizer_backend']} "
        f"passed={convergence['convergence_passed']} "
        f"reasons={','.join(failure_reasons) if failure_reasons else 'none'}"
    )

    # ---------------------------------------------------------------- #
    # Prediction. One estimator per method (METHOD_PREDICTOR), plus the other
    # one recorded beside it so a reader can price the choice without rerunning
    # anything. Components are built once and reused by every predictor here,
    # which is also what keeps the two timings comparable.
    # ---------------------------------------------------------------- #
    # The block predictor receives exactly the partition used by the objective.
    blocks = None
    if dense_mode == "terms":
        blocks = tm.unpack_terms(final_terms)
        tm.assert_partition(blocks, n)
    # The diagonal predictor streams the sum of all singleton corrections as
    # one column. Materialising [N_test, N_train] merely to draw a prefix curve
    # would destroy its linear-memory advantage, so it deliberately has no
    # explicit singleton cover here.

    if exact_gp_predictive:
        (rmse_j, nll_j), t_exact = timed_eval(
            lambda: jm.metrics_jax(
                params, x_tr, y_tr, x_te, y_te, exact=True,
                kernel_name=kernel_name,
            )
        )
        rmse, nll = float(rmse_j), float(nll_j)
        extra_metrics = {
            "reported_predictor": "exact",
            # The table reads pred_s_sparse for sparse and exact rows alike.
            "pred_s_sparse": t_exact,
        }
    else:
        # Both timings enter through predict_collapsed, from the same raw
        # parameters. Building the shared projection outside the sparse lambda
        # and inside the bound one charged ~1.3 GFLOP of Kuu/Kuf/Cholesky to
        # every structured method and to none of the baseline, which made the
        # reported prediction time a comparison of call sites.
        (mu_sparse, var_sparse, _), t_sparse = timed_eval(
            lambda: jm.predict_collapsed(
                params, x_tr, y_tr, x_te, predictor="sparse",
                kernel_name=kernel_name, jitter=jitter,
                return_contributions=True,
            )
        )
        rmse_sparse, nll_sparse = pr.gaussian_rmse_nll(
            mu_sparse, var_sparse, y_te
        )
        components = jm.collapsed_predictive_components(
            params, x_tr, y_tr, x_te, kernel_name=kernel_name, jitter=jitter
        )
        extra_metrics = {
            "reported_predictor": predictor,
            "rmse_sparse": float(rmse_sparse),
            "nll_sparse": float(nll_sparse),
            "pred_s_sparse": t_sparse,
        }
        rmse, nll = float(rmse_sparse), float(nll_sparse)

        if predictor != "sparse":
            (mu_bound, var_bound, contributions), pred_s_bound = timed_eval(
                lambda: jm.predict_collapsed(
                    params, x_tr, y_tr, x_te, predictor=predictor,
                    blocks=blocks, kernel_name=kernel_name, jitter=jitter,
                    return_contributions=True,
                )
            )
            rmse_bound, nll_bound = pr.gaussian_rmse_nll(
                mu_bound, var_bound, y_te
            )
            rmse, nll = float(rmse_bound), float(nll_bound)
            extra_metrics.update({
                "rmse_bound": rmse,
                "nll_bound": nll,
                "pred_s_bound": pred_s_bound,
                f"rmse_{predictor}": rmse,
                f"nll_{predictor}": nll,
                f"pred_s_{predictor}": pred_s_bound,
                "n_predictor_blocks": (
                    int(n) if predictor == "diagonal"
                    else int(contributions.shape[1])
                ),
            })
            curve_blocks = pr.blocks_included_curve(
                components, y_te, contributions,
                block_sizes=(
                    [len(b) for b in blocks] if blocks else [n]
                ),
            )
            block_arrays = {
                "blocks_included": curve_blocks["blocks_included"],
                "points_included": curve_blocks["points_included"],
                "block_sizes": curve_blocks["block_sizes"],
                "nll_vs_blocks": curve_blocks["nll"],
                "block_mean_contribution":
                    curve_blocks["block_mean_contribution"],
                "captured_fraction": curve_blocks["captured_fraction"],
            }
            if bool(getattr(args, "save_block_contributions", False)):
                block_arrays["contributions"] = contributions
            extra_metrics["_block_arrays"] = block_arrays
            # Half the curve's information in two scalars, so a table can carry
            # it without opening the npz.
            captured = curve_blocks["captured_fraction"]
            half = int(np.searchsorted(captured, 0.5))
            points = curve_blocks["points_included"]
            extra_metrics.update({
                "blocks_for_half_correction": half,
                "points_for_half_correction": int(points[min(half, points.size - 1)]),
                "captured_first_block": float(captured[1])
                if captured.size > 1 else float("nan"),
            })
            print(
                f"     {method} NLL: sparse={nll_sparse:+.4f}"
                f"({t_sparse * 1e3:.1f}ms)  {predictor}={nll:+.4f}"
                f"({pred_s_bound * 1e3:.1f}ms)  "
                f"blocks={contributions.shape[1]} "
                f"half-correction@{half}"
            )
        else:
            print(
                f"     {method} NLL: sparse={nll_sparse:+.4f} "
                f"({t_sparse * 1e3:.1f}ms)  [reported]"
            )
    y_std = meta["y_std"]
    # Calibration diagnostics describe the prediction this row REPORTS, so they
    # are read off the same predictor rather than rebuilt from a different one.
    if exact_gp_predictive:
        diag_mu, diag_var = jm.predict_jax(
            params, x_tr, y_tr, x_te, exact=True, kernel_name=kernel_name
        )
    elif predictor != "sparse":
        diag_mu, diag_var = mu_bound, var_bound
    else:
        diag_mu, diag_var = jm.predict_collapsed(
            params, x_tr, y_tr, x_te, predictor="sparse",
            kernel_name=kernel_name, jitter=jitter,
        )
    diag = jax.device_get(jm.predictive_diagnostics_from_prediction_jax(
        params, diag_mu, diag_var, y_te
    ))
    diag = {key: float(value) for key, value in diag.items()}
    y_var = float(y_std) ** 2
    diag.update({
        "noise_var_orig": diag["noise_var"] * y_var,
        "noise_sd_orig": diag["noise_sd"] * float(y_std),
        "signal_var_orig": diag["signal_var"] * y_var,
        "pred_var_min_orig": diag["pred_var_min"] * y_var,
        "pred_var_mean_orig": diag["pred_var_mean"] * y_var,
        "pred_var_max_orig": diag["pred_var_max"] * y_var,
        "pred_sd_mean_orig": diag["pred_sd_mean"] * float(y_std),
        "mean_abs_residual_orig": diag["mean_abs_residual"] * float(y_std),
        "nll_log_term_orig": diag["nll_log_term"] + math.log(float(y_std)),
        "nll_quad_term_orig": diag["nll_quad_term"],
    })

    # Theory measures: residual log-determinant captured mass at final theta.
    # BT/SBT expose their captured fraction and the gap to the E-SGPR dense
    # endpoint, plus the off-block residual-energy fraction vs the (b-1)/(N-1)
    # random baseline. Dense (O(N^3)/O(N^2) memory) is gated by --logdet-max-n.
    logdet_caps = {}
    want_caps = bool(getattr(args, "logdet_diagnostics", True))
    cap_n = int(getattr(args, "logdet_max_n", 6000) or 6000)
    if want_caps and n <= cap_n:
        try:
            logdet_caps = logdet_capture_for(
                method, x_tr, params, block, args, seed, kernel_name,
                jitter=jitter)
        except Exception as exc:  # noqa: BLE001  (diagnostic must never abort a run)
            print(f"     [logdet-capture skipped: {type(exc).__name__}: {exc}]")
    elif want_caps:
        print(f"     [logdet-capture skipped: n={n} > logdet-max-n={cap_n}]")

    # CLEAN structure-build (search) cost, isolated: one warm-up call (XLA
    # compile excluded), then the median of a few timed builds, each forced to
    # completion with block_until_ready. This is the honest per-refresh cost, in
    # ms -- unlike search_s, which also includes the one-time compile and every
    # refresh over the run. BT (precomputed blocks) is ~0; SBT and the dense
    # METIS diagnostic report their real partition-build costs.
    search_build_ms = float("nan")
    if dense_mode == "terms":
        try:
            # one warm-up (compile excluded), then time up to 5 builds but stop
            # early after ~0.3 s total, so the O(N^2) METIS host build is not
            # repeated many times merely to estimate its timing.
            out = raw_fn(params)
            jax.block_until_ready(out[0])
            ts, t_budget = [], 0.3
            t_start = time.perf_counter()
            while len(ts) < 5 and (time.perf_counter() - t_start) < t_budget:
                t = time.perf_counter()
                out = raw_fn(params)
                jax.block_until_ready(out[0])
                ts.append(time.perf_counter() - t)
            search_build_ms = float(np.median(ts)) * 1e3
            print(f"     search-build: {search_build_ms:.2f} ms/build "
                  f"(clean, warm-up excluded, {len(ts)} rep)")
        except Exception as exc:  # noqa: BLE001
            print(f"     [search-build timing skipped: {exc}]")

    obj_n = final_loss / n
    result = {
        "method": method, "dataset": dataset, "kernel": kernel_name,
        "M": m, "seed": seed,
        "n_train": n, "n_test": int(x_te.shape[0]), "d": meta["d"],
        "n_blocks": int(resolved_n_blocks), "block": block,
        "rmse": rmse, "nll": nll,
        "rmse_orig": rmse * y_std, "nll_orig": nll + math.log(y_std),
        "obj_n": obj_n, "elbo_n": -obj_n,
        "steps": actual_steps, "max_steps": steps,
        "refreshes": info["refreshes"], "lr": args.lr,
        "noise_lr": (
            None if getattr(args, "noise_lr", None) is None
            else float(getattr(args, "noise_lr"))
        ),
        "phase_iters": info.get("phase_iters", []),
        "phase_actual_iters": info.get("phase_actual_iters", []),
        "initialization_strategy": strategy,
        "initialization_s": initialization_s,
        "warmup_cycles": len(warmup_records),
        "warmup_steps": int(warmup["steps"]),
        "warmup_records": warmup_records,
        "grad_ms": grad_ms, "grad_ms_std": grad_ms_std,
        "search_s": search_s, "optimizer_s": optimizer_s,
        "compile_s": compile_s, "search_build_ms": search_build_ms,
        "total_s": total_s,
        "implementation": IMPLEMENTATION,
        "config": getattr(args, "_run_config", {}),
        "config_id": getattr(args, "_config_id", ""),
        "runtime": runtime_manifest(),
    }
    result.update(convergence)
    result.update(diag)
    result.update(extra_metrics)
    result.update(logdet_caps)
    artifacts = {
        "params.npz": {
            key: np.asarray(jax.device_get(value))
            for key, value in params.items()
        }
    }
    if dense_mode == "terms":
        coeffs, idx, mask = final_terms
        artifacts["final_blocks.npz"] = {
            "coeffs": np.asarray(jax.device_get(coeffs)),
            "idx": np.asarray(jax.device_get(idx)),
            "mask": np.asarray(jax.device_get(mask)),
        }
    return result, (
        np.asarray(curve)
        if curve is not None and np.asarray(curve).size
        else None
    ), artifacts


# --------------------------------------------------------------------------- #
# IO: per-run save, results CSV, LaTeX tables  (host)
# --------------------------------------------------------------------------- #
RESULT_FIELDS = ["method", "dataset", "kernel", "M", "seed", "n_train",
                 "n_test", "d", "n_blocks", "block", "rmse", "nll",
                 "rmse_orig", "nll_orig",
                 "reported_predictor",
                 "rmse_sparse", "nll_sparse", "pred_s_sparse",
                 # One pair of bound columns whose meaning is fixed by
                 # reported_predictor, plus the per-kind names the row also
                 # carries. Declaring only "*_full" left every diagonal and
                 # block row blank and dropped pred_s_bound, which is the
                 # number the table prints as Pred. (ms).
                 "rmse_bound", "nll_bound", "pred_s_bound",
                 "rmse_diagonal", "nll_diagonal", "pred_s_diagonal",
                 "rmse_block", "nll_block", "pred_s_block",
                 "rmse_full", "nll_full", "pred_s_full",
                 "n_predictor_blocks", "blocks_for_half_correction",
                 "points_for_half_correction", "captured_first_block",
                 "obj_n", "elbo_n", "steps",
                 "convergence_passed", "convergence_window_steps",
                 "convergence_obj_n_tol", "convergence_grad_max_n_tol",
                 "convergence_patience",
                 "optimizer_convergence_reached", "max_steps",
                 "convergence_tail_start_step",
                 "convergence_tail_span_steps",
                 "convergence_tail_delta_obj_n",
                 "convergence_tail_range_obj_n",
                 "final_grad_l2_per_datum", "final_grad_max_per_datum",
                 "convergence_failure_reasons", "optimizer_backend",
                 "convergence_checks",
                 "refreshes", "phase_iters", "phase_actual_iters",
                 "initialization_strategy", "initialization_s",
                 "warmup_cycles", "warmup_steps", "warmup_records",
                 "restart_count", "optimization_attempts", "lr", "noise_lr",
                 "grad_ms", "grad_ms_std",
                 "search_s", "search_build_ms", "optimizer_s", "compile_s",
                 "total_s",
                 "log_s2", "noise_var", "noise_sd", "noise_var_orig",
                 "noise_sd_orig", "log_sf2", "signal_var",
                 "signal_var_orig", "ell_min", "ell_mean", "ell_max",
                 "pred_var_min", "pred_var_mean", "pred_var_max",
                 "pred_sd_mean", "pred_var_min_orig",
                 "pred_var_mean_orig", "pred_var_max_orig",
                 "pred_sd_mean_orig", "mean_abs_residual",
                 "mean_abs_residual_orig", "std_resid_sq_mean",
                 "rmse_over_pred_sd", "nll_log_term", "nll_quad_term",
                 "nll_log_term_orig", "nll_quad_term_orig",
                 # theory: residual log-determinant captured-mass diagnostic
                 "logdet_partition", "logdet_pen_trace", "logdet_pen_diag",
                 "logdet_pen_dense", "logdet_pen_block", "logdet_captured",
                 "logdet_gap", "logdet_diag_captured", "logdet_off_sq_total",
                 "energy_frac", "energy_random_baseline", "block_mean_size",
                 "n_blocks_diag",
                 "implementation", "config_id"]


def save_run(runs_dir, result, curve, n, *, artifacts=None):
    """Persist one run: metrics, loss curve, block decomposition, thumbnail.

    ``_block_arrays`` is lifted out of the result before it is serialised: it
    holds the Eq. (12) decomposition over this method's cover, which belongs in
    an npz beside the row rather than inside a JSON field.
    """
    block_arrays = result.pop("_block_arrays", None)
    extra_arrays = dict(artifacts or {})
    if block_arrays:
        extra_arrays["block_decomposition.npz"] = block_arrays
    out = rio.save_run(
        runs_dir, result, curve,
        extra_arrays=extra_arrays,
    )
    if curve is not None:
        _save_png(out / "curve.png", curve[:, 0], curve[:, 1] / n,
                  f"{result['method']} {result['dataset']} M{result['M']}")


def append_row(logs_dir, row):
    """Append one result to the shared CSV (fixed schema, parallel-safe)."""
    rio.append_row_fixed_schema(logs_dir, row, IMPLEMENTATION, RESULT_FIELDS)


def _save_png(path, steps, obj, title):
    """Best-effort convergence thumbnail; never fails a completed run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(steps, obj, lw=1.4)
        ax.set_xlabel("step")
        ax.set_ylabel("neg-bound / N")
        ax.set_title(title, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"     [curve png skipped: {type(exc).__name__}: {exc}]")


def collect(runs_dir, config_ids=None):
    """Load every result of this implementation, reporting what it skipped."""
    rows, skipped = rio.collect_metrics(
        runs_dir, implementation=IMPLEMENTATION, config_ids=config_ids,
        methods=set(METHODS),
    )
    for path, reason in skipped:
        print(f"  [skipped {path}: {reason}]")
    return rows


def build_tables(runs_dir, tables_dir, datasets, ms, config_ids=None):
    rows = collect(runs_dir, config_ids=config_ids)
    if not rows:
        print("No results to tabulate.")
        return
    Path(tables_dir).mkdir(parents=True, exist_ok=True)

    def agg(metric):
        out = {}
        for r in rows:
            key = (
                canonical_kernel_name(r.get("kernel", "rbf")),
                r["dataset"],
                r["M"],
                r["method"],
            )
            out.setdefault(key, []).append(r[metric])
        return out

    a_rmse, a_nll = agg("rmse"), agg("nll")
    a_obj, a_elbo = agg("obj_n"), agg("elbo_n")
    present = {canonical_kernel_name(r.get("kernel", "rbf")) for r in rows}
    kernels = [k for k in KERNEL_ORDER if k in present]
    kernels.extend(sorted(present.difference(kernels)))

    with open(Path(tables_dir) / "aggregate_mean_std.csv", "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "dataset", "M", "method", "n_seeds",
                    "obj_n_mean", "obj_n_std", "elbo_n_mean", "elbo_n_std",
                    "rmse_mean", "rmse_std", "nll_mean", "nll_std"])
        for kernel_name in kernels:
            for ds in datasets:
                for m in ms:
                    for meth in METHOD_ORDER:
                        key = (kernel_name, ds, m, meth)
                        if key not in a_rmse:
                            continue
                        rr = np.asarray(a_rmse[key])
                        nn = np.asarray(a_nll[key])
                        oo = np.asarray(a_obj[key])
                        ee = np.asarray(a_elbo[key])
                        oo_mean, oo_std = st.mean_std(oo)
                        ee_mean, ee_std = st.mean_std(ee)
                        rr_mean, rr_std = st.mean_std(rr)
                        nn_mean, nn_std = st.mean_std(nn)
                        w.writerow([kernel_name, ds, m, meth, len(rr),
                                    oo_mean, oo_std, ee_mean, ee_std,
                                    rr_mean, rr_std, nn_mean, nn_std])
    print(f"Tables written to {tables_dir}")
