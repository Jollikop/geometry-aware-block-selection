#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snelson benchmark: SGPR/T/RBT/SBT with trainable inducing points.

All structure builders live in ``common.partitions`` (single source of
truth); this script only defines the protocol, training loop and outputs.

Protocol:
- z0 random training points.
- Z trainable for all sparse/structured methods.
- SBT orders inducing anchors, assigns data to the nearest anchor, applies the
  boundary-aware within-cell ordering, and chunks the resulting order directly.
- MAXITER L-BFGS iterations (split across the refresh phases).
- SBT uses the fixed structural refresh schedule below.
- Method names are kept fixed:
    SGPR
    T-SGPR
    RBT-SGPR
    SBT-SGPR
"""

import os
import time
import argparse
import json
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR") or "/tmp", "mpl_cache"))

import gpflow
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.models.sgpr import SGPR
from gpflow.utilities import set_trainable

HERE = Path(__file__).resolve().parent
# sgpr/snelson/ sits two levels below Simulations/ and three below the
# repository root. Counting parents by hand got both of these one level too
# shallow: the figure was written to Simulations/Report/figures instead of
# Report/figures, and the manifest looked for the dataset under sgpr/dataset.
SIM_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
REPORT_FIGURES = REPO_ROOT / "Report" / "figures"

from common import predictors as pr
from common.predictors import COLLAPSED_METHOD_PREDICTOR as METHOD_PREDICTOR
from common.data_split import DATASET_DIR, load_snelson
from common.partitions import bt_terms_host, sbt_terms_host
from common.plot_style import (
    UNCERTAINTY_DASHES,
    method_color,
    method_linewidth,
)
from common.provenance import (
    SNELSON_RELEASE_PACKAGES,
    code_state,
    release_runtime_matches,
    runtime_manifest,
    sha256_file,
)
from sgpr.gpflow_bounds import FullResidualSGPR, StructuredSGPR, TighterSGPR
from sgpr.gpflow_training import (
    fit_model_limited_refreshes,
    fit_model_static,
)


# ============================================================
# Hard-coded parameters
# ============================================================

SEED = 72
M = 5
N_BLOCKS = 10
SNELSON_BAND_LINEWIDTH = 1.0

# Hard cap on total L-BFGS iterations: EVERY method (static and structured) is
# optimised for the same number of steps, so the figure compares them at equal
# optimisation. Override with --max-iters.
MAX_ITERS = 60

# Front-loaded structure-refresh schedule. Each entry is the number of L-BFGS
# iterations run BEFORE the next structure refresh: short phases at the start
# search the structure more often (when the hyperparameters move fast). The
# schedule is clamped so its total never exceeds MAX_ITERS. Override with
# --refresh-schedule "2,3,5,20,30". The first 10 iterations are split into
# three short phases, adding two early refreshes while preserving the later
# 20/30-iteration phases. Default sums to MAX_ITERS=60.
REFRESH_SCHEDULE = [2, 3, 5, 20, 30]
STRUCTURE_REFRESHES = len(REFRESH_SCHEDULE)   # number of structure refreshes
MAXITER = sum(REFRESH_SCHEDULE)               # static baselines: same total iters

X_PLOT_MIN = -2.0
X_PLOT_MAX = 8.0
N_PLOT = 250

# ============================================================
# Generic helpers
# ============================================================

def make_kernel():
    return gpflow.kernels.SquaredExponential()


def random_inducing(x_train, m, seed=43):
    np.random.seed(seed)
    ind = np.random.permutation(x_train.shape[0])[:m]
    return x_train[ind].copy()


def ensure_trainable_z(model):
    if hasattr(model, "inducing_variable") and hasattr(model.inducing_variable, "Z"):
        set_trainable(model.inducing_variable.Z, True)
    return model



def current_structure_inputs(model):
    x, _ = model.data
    z = np.asarray(model.inducing_variable.Z.numpy(), dtype=float)
    ell = np.atleast_1d(np.asarray(model.kernel.lengthscales.numpy(), dtype=float))
    return np.asarray(x), z, ell


def block_size_from_n_blocks(n_points, n_blocks):
    return max(1, int(np.ceil(int(n_points) / int(n_blocks))))




def scalar_lengthscale(model):
    return float(np.ravel(np.asarray(model.kernel.lengthscales.numpy()))[0])


# ============================================================
# Training  (imported from libraries.training)
# ============================================================
# split_iterations, make_compiled_value_and_grad,
# fit_model_static, fit_model_limited_refreshes
# are all imported at the top of this file.


# ============================================================
# Model builders / refresh functions
# ============================================================

def build_base_models(x_train, y_train, z0):
    models = {}

    models["SGPR"] = ensure_trainable_z(SGPR(
        (x_train, y_train),
        kernel=make_kernel(),
        inducing_variable=z0.copy(),
    ))

    models["T-SGPR"] = ensure_trainable_z(TighterSGPR(
        (x_train, y_train),
        kernel=make_kernel(),
        inducing_variable=z0.copy(),
    ))

    return models


def sbt_structure_capacity(n, block_size):
    """Tight padded capacity for SBT blocks of size at most b."""
    return (2 * int(np.ceil(n / block_size)) + 4, int(block_size))


def build_bt_model(x_train, y_train, z0, block_size, seed):
    n_blocks = int(np.ceil(len(x_train) / block_size))
    bt = ensure_trainable_z(StructuredSGPR(
        (x_train, y_train),
        kernel=make_kernel(),
        inducing_variable=z0.copy(),
        structure_capacity=(n_blocks + 2, int(block_size)),
    ))
    bt.set_terms(
        bt_terms_host(len(x_train), block_size, np.random.default_rng(seed))
    )
    return "RBT-SGPR", bt


def refresh_sbt(model, block_size):
    x, z, ell = current_structure_inputs(model)
    terms, info = sbt_terms_host(
        x,
        z,
        ell,
        block_size,
        return_info=True,
        split_mode="boundary",
    )
    model.set_terms(terms)
    info["terms"] = len(model.terms)
    return info


def build_sbt_model(x_train, y_train, z0, block_size):
    model = ensure_trainable_z(StructuredSGPR(
        (x_train, y_train),
        kernel=make_kernel(),
        inducing_variable=z0.copy(),
        structure_capacity=sbt_structure_capacity(len(x_train), block_size),
    ))
    t0 = time.perf_counter()
    info = refresh_sbt(model, block_size)
    info["_initial_search_s"] = time.perf_counter() - t0
    return model, info


# ============================================================
# Output helpers
# ============================================================

def _predictive_residual_components(model, x_plot, jitter=1e-8):
    """Return standard mean, inducing variance, residual D_**, and noise.

    GPflow's ``predict_y`` gives qvar + residual + noise. Structured plotting
    keeps the same posterior mean/q(u) uncertainty and replaces only the test
    residual term by the bound-matched diagonal/block residual conditional.
    """
    mean, var_y = model.predict_y(x_plot)
    mean = np.asarray(mean).reshape(-1)
    var_y = np.asarray(var_y).reshape(-1)
    s2 = float(model.likelihood.variance.numpy())

    z = np.asarray(model.inducing_variable.Z.numpy(), dtype=np.float64)
    kuu = np.asarray(model.kernel(z), dtype=np.float64)
    kuu = 0.5 * (kuu + kuu.T) + jitter * np.eye(len(z))
    kus = np.asarray(model.kernel(z, x_plot), dtype=np.float64)
    kss_diag = np.asarray(model.kernel(x_plot, full_cov=False), dtype=np.float64).reshape(-1)
    l_uu = np.linalg.cholesky(kuu)
    a_s = np.linalg.solve(l_uu, kus)
    d_diag = np.maximum(kss_diag - np.sum(a_s * a_s, axis=0), 0.0)
    qvar = np.maximum(var_y - d_diag - s2, 0.0)
    return mean, qvar, d_diag, a_s, s2


def gpflow_predictive_components(model, x_train, x_plot, jitter=1e-8):
    """Projections in the layout ``common.predictors`` expects, from GPflow.

    This is the bridge that lets the figure call the same predictor the tables
    call. Everything downstream of it is shared code, so a band drawn here and
    a row in the Kin40k table cannot be two different estimators.
    """
    import jax.numpy as jnp

    mean, qvar, d_test, a_test, s2 = _predictive_residual_components(
        model, x_plot, jitter=jitter
    )
    _, _, d_train, a_train, train_s2 = _predictive_residual_components(
        model, x_train, jitter=jitter
    )
    if not np.isclose(s2, train_s2):
        raise ValueError("inconsistent predictive noise variance")

    def kernel_fn(xa, xb):
        return jnp.asarray(
            np.asarray(model.kernel(np.asarray(xa), np.asarray(xb)),
                       dtype=np.float64)
        )

    return {
        "x": jnp.asarray(np.asarray(x_train, dtype=np.float64)),
        "xs": jnp.asarray(np.asarray(x_plot, dtype=np.float64)),
        "a_train": jnp.asarray(a_train),
        "a_test": jnp.asarray(a_test),
        "d_train": jnp.asarray(d_train),
        "d_test": jnp.asarray(d_test),
        "s2": jnp.asarray(s2),
        "kernel_fn": kernel_fn,
        "mean": jnp.asarray(mean),
        "qvar": jnp.asarray(qvar),
    }


def model_blocks(model):
    """The cover a structured GPflow model conditions on, or None."""
    terms = getattr(model, "terms", None)
    if not terms:
        return None
    blocks = [
        np.asarray(idx, dtype=np.int64)
        for coeff, idx in terms
        if float(coeff) > 0.0 and len(idx)
    ]
    return blocks or None


def prediction_for_plot(model, x_train, x_plot, name, jitter=1e-8):
    """Predictive mean and variance of ``name``, under the predictor it reports.

    SGPR keeps the sparse predictive because its bound models no residual.
    T-SGPR uses the fitted diagonal residual, RBT/SBT use every block in their
    fitted partition, and E-SGPR uses the dense residual endpoint. Thus each
    band is the covariance represented by that method's training bound.
    """
    predictor = METHOD_PREDICTOR[name]
    if predictor == "sparse":
        mean, var = model.predict_y(x_plot)
        return np.asarray(mean).reshape(-1), np.asarray(var).reshape(-1)
    components = gpflow_predictive_components(
        model, x_train, x_plot, jitter=jitter
    )
    return pr.predictive_variance(
        components, predictor=predictor, blocks=model_blocks(model),
        jitter=jitter,
    )


def prediction_metrics(model, x_train, y_train, name, jitter=1e-8):
    """RMSE and NLL of ``name`` on the Snelson training inputs.

    Snelson has no held-out split -- it is a fit diagnostic, and the table says
    so. The point is that the number comes from the same predictor the band is
    drawn with.
    """
    mu, var = prediction_for_plot(model, x_train, x_train, name, jitter=jitter)
    return pr.gaussian_rmse_nll(mu, var, np.asarray(y_train).reshape(-1))


def plot_model(ax, model, x_train, y_train, x_plot, name, color):
    mean, var = prediction_for_plot(model, x_train, x_plot, name)
    sigma = np.sqrt(np.maximum(var, 1e-12))
    line_width = method_linewidth(name)
    ax.plot(
        x_plot[:, 0], mean, color=color, linewidth=line_width, label=name
    )
    for bound in (mean - 2.0 * sigma, mean + 2.0 * sigma):
        ax.plot(
            x_plot[:, 0],
            bound,
            color=color,
            linewidth=SNELSON_BAND_LINEWIDTH,
            linestyle=UNCERTAINTY_DASHES,
            alpha=0.78,
        )

    z = model.inducing_variable.Z.numpy()
    ax.plot(z[:, 0], np.full(len(z), 1.78), "o", color=color, markersize=5)


def add_exact_sgpr_boundary(ax, exact_sgpr_loss, n):
    y = exact_sgpr_loss / n
    ax.axhline(y, color="black", linestyle="--", linewidth=1.0)
    ax.text(
        0.98, y * 1.03, r"E-SGPR",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=7, fontweight="bold", color="black",
    )


def parameter_values(model):
    variance = float(model.kernel.variance.numpy())
    lengthscales = np.atleast_1d(np.asarray(model.kernel.lengthscales.numpy()))
    noise = float(model.likelihood.variance.numpy())
    if lengthscales.size == 1:
        lengthscale_text = f"{float(lengthscales[0]):.3f}"
    else:
        lengthscale_text = "[" + ", ".join(f"{float(v):.2f}" for v in lengthscales) + "]"
    return (
        fr"$\sigma^2_f$={variance:.3f}, "
        fr"$l_f$={lengthscale_text}, $\sigma^2$={noise:.3f}"
    )


def add_parameter_text(ax, models, exact_model=None):
    rows = [(name, model, method_color(name)) for name, model in models.items()]
    if exact_model is not None:
        rows.append(("True GP", exact_model, method_color("True GP")))

    fontsize = 6.5 if len(rows) > 5 else 8
    step = 0.043 if len(rows) > 5 else 0.05
    y_bottom = 0.005
    y_top = y_bottom + step * (len(rows) - 1)

    for idx, (name, model, color) in enumerate(rows):
        y = y_top - step * idx
        ax.text(
            0.405, y, f"{name}:",
            transform=ax.transAxes, fontsize=fontsize,
            horizontalalignment="left", verticalalignment="bottom", color=color,
        )
        ax.text(
            0.535, y, parameter_values(model),
            transform=ax.transAxes, fontsize=fontsize,
            horizontalalignment="left", verticalalignment="bottom", color=color,
        )


def plot_exact_gp(ax, model, x_plot, color=method_color("True GP"),
                  linewidth=2.8):
    """Reference exact (full) GP: thick black mean and dashed +-2 std."""
    mean, var = model.predict_y(x_plot)
    mean = np.asarray(mean).reshape(-1)
    sigma = np.sqrt(np.maximum(np.asarray(var).reshape(-1), 1e-12))
    xs = x_plot[:, 0]

    ax.plot(xs, mean, color=color, linewidth=linewidth,
            label="True GP", zorder=1.5)
    # Keep the reference interval distinct without a dense short-dash texture.
    for s in (mean - 2.0 * sigma, mean + 2.0 * sigma):
        ax.plot(
            xs,
            s,
            color=color,
            linewidth=SNELSON_BAND_LINEWIDTH,
            linestyle=UNCERTAINTY_DASHES,
            zorder=1.5,
        )


def save_figure(x_train, y_train, x_plot, models, losses,
                exact_sgpr_loss=None, report_pdf=None, exact_model=None):
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 4]}
    )

    ax = axes[0]
    if exact_sgpr_loss is not None:
        add_exact_sgpr_boundary(ax, exact_sgpr_loss, len(x_train))

    for name, values in losses.items():
        ax.plot(
            np.arange(len(values)),
            np.maximum(values / len(x_train), 1e-12),
            color=method_color(name),
            label=name,
            linewidth=1.5,
            linestyle="-",
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("negative lower bound / N")
    ax.set_yscale("log")
    ax.legend(fontsize=8.0, framealpha=0.8, loc="upper right")

    ax = axes[1]
    ax.plot(x_train[:, 0], y_train[:, 0], "xk", markersize=4, alpha=0.75)

    if exact_model is not None:
        plot_exact_gp(ax, exact_model, x_plot)

    for name, model in models.items():
        plot_model(ax, model, x_train, y_train, x_plot, name,
                   method_color(name))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(float(x_plot[0, 0]), float(x_plot[-1, 0]))
    ax.set_ylim(-3.0, 2.0)
    add_parameter_text(ax, models, exact_model=exact_model)

    fig.tight_layout()
    if report_pdf is not None:
        report_pdf = Path(report_pdf)
        report_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(report_pdf, bbox_inches="tight", dpi=300, pad_inches=0.05)
    plt.close(fig)


def write_summary(models, losses, times, iterations, out_csv,
                  exact_sgpr_loss=None, exact_sgpr_time=None, exact_sgpr_iter=None,
                  extra_info=None, metrics=None):
    metrics = metrics or {}
    lines = [
        "method,reported_predictor,rmse,nll,"
        "final_negative_bound,negative_bound_per_n,iterations,time_seconds,"
        "kernel_variance,lengthscale,noise,terms,cost,anchor_chunks,"
        "refreshes,anchor_mode,split_mode,anchor_count,"
        "max_residual_diag,mean_residual_diag"
    ]

    if exact_sgpr_loss is not None:
        lines.append(",".join([
            "E-SGPR", METHOD_PREDICTOR.get("E-SGPR", ""),
            *(f"{value:.6g}" for value in metrics.get("E-SGPR", ("", ""))
              if value != ""),
            *(["", ""] if "E-SGPR" not in metrics else []),
            f"{exact_sgpr_loss:.10g}",
            f"{exact_sgpr_loss / len(next(iter(models.values())).data[0]):.10g}",
            str(exact_sgpr_iter) if exact_sgpr_iter is not None else "",
            f"{exact_sgpr_time:.10g}" if exact_sgpr_time is not None else "",
            "", "", "", "", "", "", "", "", "", "", "", "",
        ]))

    extra_info = extra_info or {}

    for name, model in models.items():
        info = extra_info.get(name, {}) or {}
        rmse, nll = metrics.get(name, ("", ""))
        lines.append(",".join([
            name,
            METHOD_PREDICTOR.get(name, ""),
            f"{rmse:.6g}" if rmse != "" else "",
            f"{nll:.6g}" if nll != "" else "",
            f"{losses[name][-1]:.10g}",
            f"{losses[name][-1] / len(model.data[0]):.10g}",
            str(iterations.get(name, len(losses[name]) - 1)),
            f"{times[name]:.10g}",
            f"{float(model.kernel.variance.numpy()):.10g}",
            f"{scalar_lengthscale(model):.10g}",
            f"{float(model.likelihood.variance.numpy()):.10g}",
            str(info.get("terms", len(getattr(model, "terms", [])))),
            f"{float(info.get('cost', info.get('total_cost', np.nan))):.10g}"
            if ("cost" in info or "total_cost" in info) else "",
            str(info.get("anchor_chunks", "")),
            str(info.get("refreshes", "")),
            str(info.get("anchor_mode", "")),
            str(info.get("split_mode", "")),
            str(info.get("inducing_anchor_count", "")),
            f"{float(info.get('max_residual_diag', np.nan)):.10g}"
            if "max_residual_diag" in info else "",
            f"{float(info.get('mean_residual_diag', np.nan)):.10g}"
            if "mean_residual_diag" in info else "",
        ]))

    out_csv.write_text("\n".join(lines) + "\n")


def write_latex_table(models, losses, times, iterations, out_tex, n,
                      exact_sgpr_loss=None, exact_sgpr_time=None,
                      exact_sgpr_iter=None, metrics=None):
    metrics = metrics or {}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            r"\caption{Snelson collapsed benchmark with trainable random inducing "
            r"points. Every method is optimised for the same number of L-BFGS "
            r"iterations and reports the predictor its own bound implies: the "
            r"sparse predictive for SGPR, which models no residual, and "
            r"the fitted diagonal, random-block, selected-block, or dense "
            r"residual covariance for T-SGPR, RBT-SGPR, SBT-SGPR and E-SGPR. "
            r"RMSE and NLL "
            r"are measured on the training inputs -- Snelson has no held-out "
            r"split, so they are a fit diagnostic, not a generalisation "
            r"estimate. Lower is better throughout.}"
        ),
        r"\label{tab:snelson-sgpr-t-rbt-sbt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Method & Predictor & RMSE & NLL & Neg. bound & Neg. bound$/N$ "
        r"& Iter. & Time (s) \\",
        r"\midrule",
    ]

    if exact_sgpr_loss is not None:
        e_rmse, e_nll = metrics.get("E-SGPR", (None, None))
        lines.append(
            "E-SGPR final & "
            f"{METHOD_PREDICTOR.get('E-SGPR', '--')} & "
            f"{e_rmse:.4f} & " if e_rmse is not None else "E-SGPR final & -- & -- & "
        )
        lines[-1] += (
            (f"{e_nll:+.4f} & " if e_nll is not None else "-- & ")
            + f"{exact_sgpr_loss:.3f} & {exact_sgpr_loss / n:.4f} & "
            f"{exact_sgpr_iter if exact_sgpr_iter is not None else '--'} & "
            f"{exact_sgpr_time:.2f} " + r"\\"
        )

    for name, values in losses.items():
        rmse, nll = metrics.get(name, (None, None))
        lines.append(
            f"{name} & {METHOD_PREDICTOR.get(name, '--')} & "
            + (f"{rmse:.4f} & " if rmse is not None else "-- & ")
            + (f"{nll:+.4f} & " if nll is not None else "-- & ")
            + f"{values[-1]:.3f} & {values[-1] / n:.4f} & "
            f"{iterations.get(name, len(values) - 1)} & {times[name]:.2f} "
            + r"\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    out_tex.write_text("\n".join(lines))


def write_release_manifest(path, x_train, models, losses, true_gp_loss,
                           exact_sgpr_loss, block_size, phase_iters,
                           metrics, artifacts):
    """Bind the plotted Snelson numbers to data, code, and runtime state."""
    # The dataset directory has one owner, common.data_split, and the manifest
    # must hash the file the run actually read.
    data_path = DATASET_DIR / "snelson.csv"
    payload = {
        "schema": 1,
        "implementation": "snelson-bound-matched-sgpr-v6",
        "data_source": "repository:snelson.csv",
        "data_signature": sha256_file(data_path),
        "code": code_state(REPO_ROOT),
        "runtime": runtime_manifest(),
        "protocol": {
            "seed": SEED,
            "n_train": int(len(x_train)),
            "M": M,
            "block_size": int(block_size),
            "reported_predictor": dict(METHOD_PREDICTOR),
            "phase_iterations": [int(value) for value in phase_iters],
            "max_iterations": MAXITER,
        },
        "training_fit_metrics": {
            name: {"rmse": float(rmse), "nll": float(nll)}
            for name, (rmse, nll) in metrics.items()
        },
        "final_negative_bound_per_datum": {
            "True GP": float(true_gp_loss) / len(x_train),
            "E-SGPR": float(exact_sgpr_loss) / len(x_train),
            **{
                name: float(values[-1]) / len(x_train)
                for name, values in losses.items()
            },
        },
        "artifacts": {
            artifact.name: sha256_file(artifact) for artifact in artifacts
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ============================================================
# Main
# ============================================================

def _clamp_schedule(schedule, max_iters):
    """Truncate the phase schedule so its total never exceeds ``max_iters``."""
    out, tot = [], 0
    for p in schedule:
        p = int(p)
        if p <= 0:
            continue
        if tot + p >= max_iters:
            rem = max_iters - tot
            if rem > 0:
                out.append(rem)
            tot = max_iters
            break
        out.append(p)
        tot += p
    return out or [int(max_iters)]


def _parse_args():
    ap = argparse.ArgumentParser(description="Snelson SGPR/T/RBT/SBT run.")
    ap.add_argument("--n-blocks", type=int, default=N_BLOCKS,
                    help=f"RBT/SBT block count -> B = ceil(N/n_blocks) "
                         f"(default {N_BLOCKS}).")
    ap.add_argument("--block-size", type=int, default=None,
                    help="explicit RBT/SBT block size B; overrides --n-blocks.")
    ap.add_argument("--refresh-schedule", default=None,
                    help='comma-separated L-BFGS iters per phase before each '
                         'refresh, e.g. "2,3,5,20,30" (front-loaded).')
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS,
                    help=f"hard cap on total L-BFGS iters for ALL methods "
                         f"(default {MAX_ITERS}; keeps the figure fair).")
    ap.add_argument(
        "--out-dir", type=Path,
        help="write every release artifact here instead of into the checkout",
    )
    return ap.parse_args()


def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    global REFRESH_SCHEDULE, STRUCTURE_REFRESHES, MAXITER, MAX_ITERS
    args = _parse_args()
    state = code_state(REPO_ROOT)
    if state["source_dirty"]:
        raise SystemExit(
            "experiment source changes detected; commit them before generating "
            "the release figure and manifest"
        )
    runtime = runtime_manifest()
    if not release_runtime_matches(runtime, SNELSON_RELEASE_PACKAGES):
        raise SystemExit(
            "runtime package versions do not match Simulations/requirements.txt"
        )
    MAX_ITERS = int(args.max_iters)
    sched = ([int(s) for s in args.refresh_schedule.split(",") if s.strip()]
             if args.refresh_schedule else REFRESH_SCHEDULE)
    REFRESH_SCHEDULE = _clamp_schedule(sched, MAX_ITERS)
    STRUCTURE_REFRESHES = len(REFRESH_SCHEDULE)
    MAXITER = sum(REFRESH_SCHEDULE)              # == total iters, all methods equal

    x_train, y_train = load_snelson()
    x_plot = np.linspace(X_PLOT_MIN, X_PLOT_MAX, N_PLOT, dtype=np.float64)[:, None]

    if args.block_size is not None:
        block_size = max(1, int(args.block_size))
        n_blocks = int(np.ceil(len(x_train) / block_size))
    else:
        n_blocks = max(1, int(args.n_blocks))
        block_size = block_size_from_n_blocks(len(x_train), n_blocks)
    phase_iters = REFRESH_SCHEDULE

    print(
        f"setup: seed={SEED}, M={M}, n_blocks={n_blocks}, "
        f"block_size={block_size}, "
        f"maxiter={MAXITER}, refreshes={STRUCTURE_REFRESHES}, "
        f"phase_iters={phase_iters}, inducing_anchors={M}"
    )
    print("z0: random training points")
    print("Z: trainable for all sparse/structured methods")
    print("kernel/noise init: GPflow defaults")
    print("method names: SGPR, T-SGPR, RBT-SGPR, SBT-SGPR")
    print(
        "plot prediction: bound-matched. SGPR draws the sparse predictive; "
        "T-SGPR, RBT-SGPR, SBT-SGPR and E-SGPR draw their fitted diagonal, "
        "block, block and dense residual predictors, respectively."
    )

    z0 = random_inducing(x_train, M, seed=SEED)

    true_gp = gpflow.models.GPR((x_train, y_train), kernel=make_kernel())
    true_gp_losses, true_gp_time, true_gp_stats = fit_model_static(
        true_gp, MAXITER
    )
    print(
        f"{'True GP':10s} loss={true_gp_losses[-1]:.6f} "
        f"iters={true_gp_stats['iterations']} time={true_gp_time:.2f}s"
    )

    exact_sgpr = ensure_trainable_z(FullResidualSGPR(
        (x_train, y_train),
        kernel=make_kernel(),
        inducing_variable=z0.copy(),
    ))
    exact_sgpr_losses, exact_sgpr_time, exact_sgpr_stats = fit_model_static(
        exact_sgpr, MAXITER
    )
    exact_sgpr_loss = float(exact_sgpr_losses[-1])
    exact_sgpr_iter = exact_sgpr_stats["iterations"]

    print(
        f"{'E-SGPR final':10s} loss={exact_sgpr_loss:.6f} "
        f"iters={exact_sgpr_iter} time={exact_sgpr_time:.2f}s"
    )

    models = build_base_models(x_train, y_train, z0)
    losses = {}
    times = {}
    iterations = {}
    extra_info = {}

    for name, model in models.items():
        losses[name], times[name], stats = fit_model_static(model, MAXITER)
        iterations[name] = stats["iterations"]
        print(
            f"{name:10s} loss={losses[name][-1]:.6f} "
            f"iters={iterations[name]} time={times[name]:.2f}s"
        )

    # RBT-SGPR
    bt_name, bt_model = build_bt_model(x_train, y_train, z0, block_size, SEED)
    models[bt_name] = bt_model
    losses[bt_name], times[bt_name], stats = fit_model_static(bt_model, MAXITER)
    iterations[bt_name] = stats["iterations"]
    extra_info[bt_name] = {"terms": len(bt_model.terms)}
    print(f"{bt_name:10s} terms={len(bt_model.terms)}")
    print(
        f"{bt_name:10s} loss={losses[bt_name][-1]:.6f} "
        f"iters={iterations[bt_name]} time={times[bt_name]:.2f}s"
    )

    # SBT-SGPR
    sbt_name = "SBT-SGPR"
    sbt_model, sbt_info = build_sbt_model(x_train, y_train, z0, block_size)
    models[sbt_name] = sbt_model

    losses[sbt_name], times[sbt_name], sbt_info, stats = fit_model_limited_refreshes(
        sbt_model,
        total_iter=MAXITER,
        n_refreshes=STRUCTURE_REFRESHES,
        refresh_fn=lambda: refresh_sbt(sbt_model, block_size),
        initial_info=sbt_info,
        name=sbt_name,
        phase_iters=REFRESH_SCHEDULE,
    )
    iterations[sbt_name] = stats["iterations"]
    sbt_info["refreshes"] = stats.get("refreshes", "")
    extra_info[sbt_name] = sbt_info

    print(
        f"{sbt_name:10s} terms={len(sbt_model.terms)}, "
        f"anchor_mode={sbt_info.get('anchor_mode', '')}, "
        f"split_mode={sbt_info.get('split_mode', '')}, "
        f"cost={sbt_info.get('cost', 0.0):.2e}, "
        f"refreshes={STRUCTURE_REFRESHES}"
    )
    print(
        f"{sbt_name:10s} loss={losses[sbt_name][-1]:.6f} "
        f"iters={iterations[sbt_name]} time={times[sbt_name]:.2f}s"
    )

    output_root = args.out_dir.expanduser().resolve() if args.out_dir else None
    if output_root:
        output_root.mkdir(parents=True, exist_ok=True)
    out_csv = (output_root or HERE) / "snelson_summary_sgpr_t_rbt_sbt.csv"
    out_tex = (output_root or HERE) / "snelson_table_sgpr_t_rbt_sbt.tex"
    out_report_pdf = (
        output_root or REPORT_FIGURES
    ) / "snelson_sgpr_t_rbt_sbt.pdf"
    out_manifest = (
        output_root
        if output_root else REPO_ROOT / "Report" / "Results"
    ) / "snelson_sgpr_t_rbt_sbt_manifest.json"

    save_figure(
        x_train, y_train, x_plot, models, losses,
        exact_sgpr_loss=exact_sgpr_loss,
        report_pdf=out_report_pdf,
        exact_model=true_gp,
    )

    # One predictor per method, evaluated once and reused by the CSV, the LaTeX
    # table and the manifest, so the three cannot disagree.
    metrics = {
        name: prediction_metrics(model, x_train, y_train, name)
        for name, model in models.items()
    }
    for name, (rmse, nll) in metrics.items():
        print(f"     {name:10s} predictor={METHOD_PREDICTOR[name]:6s} "
              f"rmse={rmse:.4f}  nll={nll:+.4f}")

    write_summary(
        models, losses, times, iterations, out_csv,
        exact_sgpr_loss=exact_sgpr_loss, exact_sgpr_time=exact_sgpr_time, exact_sgpr_iter=exact_sgpr_iter,
        extra_info=extra_info, metrics=metrics,
    )

    write_latex_table(
        models, losses, times, iterations, out_tex, len(x_train),
        exact_sgpr_loss=exact_sgpr_loss, exact_sgpr_time=exact_sgpr_time, exact_sgpr_iter=exact_sgpr_iter,
        metrics=metrics,
    )

    write_release_manifest(
        out_manifest,
        x_train,
        models,
        losses,
        true_gp_losses[-1],
        exact_sgpr_loss,
        block_size,
        phase_iters,
        metrics,
        (out_report_pdf, out_csv, out_tex),
    )

    print(f"report figure -> {out_report_pdf}")
    print(f"summary -> {out_csv}")
    print(f"table -> {out_tex}")
    print(f"manifest -> {out_manifest}")


if __name__ == "__main__":
    main()
