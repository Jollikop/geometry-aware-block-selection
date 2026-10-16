#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snelson residual-structure illustration.

The report figure trains SBT-SGPR, freezes those parameters, and then evaluates
the dense Exact residual correction, SBT anchor blocks, and a random BT
partition at
the same parameters. This isolates the block-selection effect: the same model
and residual matrix, different diagonal block covers.
"""
from __future__ import annotations

import csv
import argparse
import math
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "mpl_cache"),
)

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"mathtext.fontset": "cm"})

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parents[1]
REPORT_FIGURES = SIM_ROOT.parent / "Report" / "figures"

from sgpr import jax_collapsed as jm  # noqa: E402
from common import kernels as kn  # noqa: E402
from common.data_split import load_snelson, standardize_pair  # noqa: E402
from common.plot_style import METHOD_COLORS  # noqa: E402
from common import partitions as pt  # noqa: E402
from common import terms as tm  # noqa: E402
from common.partitions import sbt_blocks_host  # noqa: E402

jax.config.update("jax_enable_x64", True)

SEED = int(os.environ.get("RESIDUAL_SEED", "7"))
DATASET_LABEL = "Snelson"
FIGURE_STEM = "residual_structure_snelson"
TABLE_STEM = "snelson_residual"
M = int(os.environ.get("RESIDUAL_M", "5"))
BLOCK_ENV = os.environ.get("RESIDUAL_BLOCK")
N_BLOCKS_ENV = os.environ.get("RESIDUAL_N_BLOCKS")
BLOCK = int(BLOCK_ENV) if BLOCK_ENV is not None else 40
SPLIT_MODE = os.environ.get("RESIDUAL_SPLIT_MODE", "boundary").lower()
OPTIMIZER = os.environ.get("RESIDUAL_OPTIMIZER", "lbfgs").lower()
if OPTIMIZER not in {"adam", "lion", "lbfgs"}:
    raise ValueError(f"unknown RESIDUAL_OPTIMIZER={OPTIMIZER!r}")
DEFAULT_STEPS = "120" if OPTIMIZER == "lbfgs" else "500"
TRAIN_STEPS = int(os.environ.get("RESIDUAL_STEPS", DEFAULT_STEPS))
TRAIN_LR = float(os.environ.get("RESIDUAL_LR", "0.02"))
LOG_EVERY = int(os.environ.get("RESIDUAL_LOG_EVERY", "10"))
JITTER = 1e-8
OUTPUT_TAG = os.environ.get("RESIDUAL_OUTPUT_TAG", "")
WRITE_REPORT = os.environ.get("RESIDUAL_WRITE_REPORT", "1") != "0"
WRITE_TABLE = os.environ.get("RESIDUAL_WRITE_TABLE", "1") != "0"

METHODS = ["SBT-SGPR"]
COLORS = {
    # This figure keeps its own RBT hue: it shows blocks next to the anchored
    # cover, where the shared pink sits too close to the red.
    "RBT-SGPR": "#E69F00",
    "SBT-SGPR": METHOD_COLORS["SBT-SGPR"],
    "FULL": METHOD_COLORS["Exact-GP"],
}
BLOCK_COLORS = (
    "#ff2d55",
    "#00d9ff",
    "#ff8c00",
    "#d100ff",
    "#64ff4a",
    "#ffea00",
    "#00ffb3",
    "#ff5c00",
    "#b000ff",
    "#00b7ff",
)
PANEL_TITLE_FONTSIZE = 8.4
AXIS_LABEL_FONTSIZE = 7.6
ANNOTATION_FONTSIZE = 6.4
TICK_FONTSIZE = 6.2


def make_sbt_refresh_schedule(total_steps):
    total_steps = max(1, int(total_steps))
    if total_steps < 6:
        return [total_steps]
    weights = np.asarray([0.10, 0.10, 0.15, 0.20, 0.20, 0.25])
    phases = np.maximum(1, np.floor(weights * total_steps).astype(int))
    phases[-1] += total_steps - int(phases.sum())
    return [int(v) for v in phases]


SBT_REFRESH_SCHEDULE = make_sbt_refresh_schedule(TRAIN_STEPS)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Snelson residual-block illustration."
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="explicit BT/SBT block size B; overrides --n-blocks and env.",
    )
    ap.add_argument(
        "--n-blocks",
        type=int,
        default=None,
        help="BT/SBT block count -> B = ceil(N/n_blocks).",
    )
    ap.add_argument(
        "--split-mode",
        choices=("boundary", "center"),
        default=None,
        help="SBT ordering inside anchor cells. boundary moves cut points "
             "towards adjacent anchors; center is the old distance-to-anchor "
             "ordering. Overrides RESIDUAL_SPLIT_MODE.",
    )
    return ap.parse_args()


def resolve_residual_block_size(n_points, *, block_size=None, n_blocks=None):
    if block_size is not None:
        block = max(1, int(block_size))
        return block, int(math.ceil(int(n_points) / block))
    if n_blocks is not None:
        n_block_value = max(1, int(n_blocks))
        return max(1, int(math.ceil(int(n_points) / n_block_value))), n_block_value
    if BLOCK_ENV is not None:
        block = max(1, int(BLOCK_ENV))
        return block, int(math.ceil(int(n_points) / block))
    if N_BLOCKS_ENV is not None:
        n_blocks = max(1, int(N_BLOCKS_ENV))
        return max(1, int(math.ceil(int(n_points) / n_blocks))), n_blocks
    block = max(1, int(BLOCK))
    return block, int(math.ceil(int(n_points) / block))


def theta_numpy(params):
    return (
        np.asarray(jax.device_get(jnp.exp(params["log_ell"]))),
        float(jax.device_get(jnp.exp(params["log_sf2"]))),
        float(jax.device_get(jnp.exp(params["log_s2"]))),
        np.asarray(jax.device_get(params["z"])),
    )


def sbt_terms_for_params(x_train, params, *, return_info=False):
    """Packed SBT cover at the current hyperparameters (host selector)."""
    ell, _, _, z = theta_numpy(params)
    blocks, info = sbt_blocks_host(
        np.asarray(x_train), z, ell, BLOCK,
        split_mode=SPLIT_MODE, return_info=True,
    )
    packed = tm.pack_terms(
        blocks, BLOCK, len(x_train),
        capacity=2 * int(math.ceil(len(x_train) / BLOCK)) + 4,
    )
    return (packed, info) if return_info else packed


def make_sbt_refresh(x_train):
    def refresh(params):
        return sbt_terms_for_params(x_train, params)

    refresh._host_refresh = True
    return refresh


def load_residual_dataset():
    x, y = load_snelson()
    x_train, y_train, x_test, y_test = standardize_pair(x, y, x, y)
    return (
        np.asarray(x_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.float64).reshape(-1),
        np.asarray(x_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.float64).reshape(-1),
    )


def random_inducing(x_train, m, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x_train.shape[0])[: int(m)]
    return np.asarray(x_train[idx], dtype=np.float64).copy()


def run_method(method, x_train, y_train, x_test, y_test, z0):
    params0 = jm.init_params(x_train, z0)
    y_train = np.asarray(y_train).reshape(-1)
    dense_mode = {
        "SGPR": "trace",
        "T-SGPR": "diag",
        "RBT-SGPR": None,
        "SBT-SGPR": None,
    }[method]
    loss = jm.make_loss(x_train, y_train, dense_mode=dense_mode, kernel_name="rbf")

    dummy = (jnp.zeros(1), jnp.zeros((1, 1), jnp.int32), jnp.zeros((1, 1)))
    initial_info = {}
    if method == "RBT-SGPR":
        terms0 = pt.bt_terms_device(len(x_train), BLOCK, SEED)
        refresh = None
    elif method == "SBT-SGPR":
        terms0, initial_info = sbt_terms_for_params(
            x_train, params0, return_info=True
        )
        refresh = make_sbt_refresh(x_train)
    else:
        terms0 = dummy
        refresh = None

    noise_curve = [(0, float(jax.device_get(jnp.exp(params0["log_s2"]))))]

    def log_noise(step, params, _value, _ms_per_step, _terms=None):
        noise_curve.append(
            (int(step), float(jax.device_get(jnp.exp(params["log_s2"]))))
        )

    t0 = time.perf_counter()
    params, curve, info = jm.train_collapsed(
        loss,
        params0,
        terms0=terms0,
        term_fn=refresh,
        steps=TRAIN_STEPS,
        refresh_every=TRAIN_STEPS,
        optimizer=OPTIMIZER,
        lr=TRAIN_LR,
        grad_clip=10.0,
        train_z=True,
        log_every=LOG_EVERY,
        log_cb=log_noise,
        refresh_schedule=SBT_REFRESH_SCHEDULE if method == "SBT-SGPR" else None,
    )
    final_terms = info.pop("final_terms")
    total_s = time.perf_counter() - t0
    rmse, nll = jm.metrics_jax(
        params, x_train, y_train, x_test, y_test, exact=False, kernel_name="rbf"
    )
    return {
        "method": method,
        "params": params,
        "terms": final_terms,
        "curve": np.asarray(curve, dtype=float),
        "noise_curve": np.asarray(noise_curve, dtype=float),
        "rmse": float(rmse),
        "nll": float(nll),
        "time": total_s,
        "search_s": float(info.get("search_s", 0.0)),
        "optimizer_s": float(info.get("optimizer_s", 0.0)),
        "refreshes": int(info.get("refreshes", 0)),
        "phase_iters": list(info.get("phase_iters", [TRAIN_STEPS])),
        "optimizer": OPTIMIZER,
        "steps": TRAIN_STEPS,
        "initial_info": initial_info,
    }


def choose_colors(n):
    """Restore the original high-contrast block palette."""
    if n <= len(BLOCK_COLORS):
        return list(BLOCK_COLORS[:n])
    extra = [
        matplotlib.colors.to_hex(color)
        for color in plt.cm.hsv(
            np.linspace(0, 1, n - len(BLOCK_COLORS) + 1)
        )[:-1]
    ]
    return list(BLOCK_COLORS) + extra


def points_panel(ax, x_train, y_train, blocks, z, title, colors):
    y_train = np.asarray(y_train).reshape(-1)
    if x_train.shape[1] == 1:
        order = np.argsort(x_train[:, 0])
        for color, block in zip(colors, blocks):
            ax.scatter(
                x_train[block, 0],
                y_train[block],
                s=12,
                color=color,
                alpha=0.82,
                edgecolors="#202020",
                linewidths=0.16,
                zorder=3,
            )
        z_y = np.interp(z[:, 0], x_train[order, 0], y_train[order])
        ax.scatter(
            z[:, 0],
            z_y,
            marker="o",
            s=115,
            color="white",
            edgecolor="#171717",
            linewidth=0.65,
            zorder=6,
        )
        ax.scatter(
            z[:, 0],
            z_y,
            marker="*",
            s=96,
            color="#111111",
            edgecolor="white",
            linewidth=0.65,
            zorder=7,
        )
        ax.set_aspect("auto")
    else:
        for color, block in zip(colors, blocks):
            ax.scatter(
                x_train[block, 0],
                x_train[block, 1],
                s=10,
                color=color,
                alpha=0.78,
                edgecolors="#202020",
                linewidths=0.14,
            )
        ax.scatter(
            z[:, 0],
            z[:, 1],
            marker="o",
            s=115,
            color="white",
            edgecolor="#171717",
            linewidth=0.65,
            zorder=6,
        )
        ax.scatter(
            z[:, 0],
            z[:, 1],
            marker="*",
            s=96,
            color="#111111",
            edgecolor="white",
            linewidth=0.65,
            zorder=7,
        )
        ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])


def matrix_panel(ax, mat, blocks, ordering, title, colors, vmax):
    from matplotlib.patches import Rectangle

    invp = np.empty(len(ordering), dtype=int)
    invp[ordering] = np.arange(len(ordering))
    shown = np.abs(mat)[ordering][:, ordering]
    image = ax.imshow(
        shown,
        cmap="viridis",
        vmin=0.0,
        vmax=max(vmax, 1e-12),
        interpolation="nearest",
    )
    overlay = np.zeros((len(ordering), len(ordering), 4))
    for color, block in zip(colors, blocks):
        p = np.sort(invp[np.asarray(block, dtype=int)])
        overlay[np.ix_(p, p)] = matplotlib.colors.to_rgba(color, alpha=0.26)
        ax.add_patch(
            Rectangle(
                (p[0] - 0.5, p[0] - 0.5),
                p[-1] - p[0] + 1,
                p[-1] - p[0] + 1,
                fill=False,
                edgecolor=color,
                linewidth=0.9,
            )
        )
    ax.imshow(overlay, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


#: Internal short keys -> the names the paper uses.
PANEL_LABELS = {"Exact": "Exact", "SBT": "Selected", "BT": "Random",
                "Anchor": "Anchor", "Residual": "Residual"}


def panel_label(name: str) -> str:
    return PANEL_LABELS.get(name, name)


def logdet_scale_panel(ax, summary):
    vals = {
        "Exact": summary["penalties"]["FULL"],
        "SBT": summary["penalties"]["SBT"],
        "BT": summary["penalties"]["BT"],
    }
    colors = {
        "Exact": COLORS["FULL"],
        "SBT": COLORS["SBT-SGPR"],
        "BT": COLORS["RBT-SGPR"],
    }
    y = np.asarray(list(vals.values()), dtype=float)
    pad = max(0.16 * (float(y.max()) - float(y.min())), 1e-3)
    ax.set_ylim(float(y.min()) - pad, float(y.max()) + 1.15 * pad)
    ax.set_xlim(0.0, 1.0)
    axis_x = 0.22
    ax.axvline(axis_x, color="#3f3f3f", linewidth=0.9, zorder=1)
    for name, value in vals.items():
        ax.hlines(
            value,
            axis_x - 0.10,
            axis_x + 0.10,
            color=colors[name],
            linewidth=0.8,
            alpha=0.75,
            zorder=2,
        )
        ax.scatter(
            [axis_x],
            [value],
            s=22,
            color=colors[name],
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        ax.text(
            axis_x + 0.22,
            value,
            f"{panel_label(name)} {value:.1f}",
            color=colors[name],
            fontsize=ANNOTATION_FONTSIZE,
            ha="left",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=0.25),
            clip_on=False,
        )
    ax.text(
        0.50,
        1.08,
        "log det",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=PANEL_TITLE_FONTSIZE,
        clip_on=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def comparison_logdet_scale_panel(ax, values):
    """Compact log-determinant scale for the three-cover comparison."""
    colors = {
        "Exact": COLORS["FULL"],
        "Anchor": COLORS["SBT-SGPR"],
        "Residual": "#7A0019",
        "BT": COLORS["RBT-SGPR"],
    }
    y = np.asarray(list(values.values()), dtype=float)
    pad = max(0.14 * (float(y.max()) - float(y.min())), 1e-3)
    ax.set_ylim(float(y.min()) - pad, float(y.max()) + 1.2 * pad)
    ax.set_xlim(0.0, 1.0)
    axis_x = 0.18
    ax.axvline(axis_x, color="#3f3f3f", linewidth=0.9, zorder=1)
    for name, value in values.items():
        ax.scatter(
            [axis_x], [value], s=20, color=colors[name], edgecolor="white",
            linewidth=0.4, zorder=3,
        )
        ax.text(
            axis_x + 0.18, value, f"{panel_label(name)} {value:.1f}",
            color=colors[name], fontsize=ANNOTATION_FONTSIZE,
            ha="left", va="center", clip_on=False,
        )
    ax.set_title("log det", fontsize=PANEL_TITLE_FONTSIZE, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def order_from_blocks(blocks, reference_order=None):
    """Build a plotting permutation that lays every accepted block contiguously."""
    if reference_order is None:
        return np.concatenate([np.asarray(b, dtype=np.int32) for b in blocks])
    inv = np.empty(len(reference_order), dtype=np.int32)
    inv[np.asarray(reference_order, dtype=np.int32)] = np.arange(len(reference_order))
    keyed = []
    for block in blocks:
        block = np.asarray(block, dtype=np.int32)
        local = block[np.argsort(inv[block])]
        keyed.append((float(np.mean(inv[local])), local))
    keyed.sort(key=lambda item: item[0])
    return np.concatenate([block for _key, block in keyed])


def collapsed_components(params, x_train, y_train):
    """Return log N(y;0,Q+s2I), residual D, and noise variance."""
    xj = jnp.asarray(x_train)
    yj = jnp.asarray(y_train).reshape(-1)
    n = xj.shape[0]
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    s2 = jnp.exp(params["log_s2"])
    z = params["z"]
    kuu = kn.kernel(z, z, ell, sf2, "rbf") + JITTER * jnp.eye(z.shape[0])
    kuf = kn.kernel(z, xj, ell, sf2, "rbf")
    luu = jnp.linalg.cholesky(kuu)
    a = jax.scipy.linalg.solve_triangular(luu, kuf, lower=True)

    big = jnp.eye(z.shape[0]) + (a @ a.T) / s2
    lb = jnp.linalg.cholesky(big)
    c = jax.scipy.linalg.solve_triangular(lb, a @ yj, lower=True) / s2
    quad = (yj @ yj) / s2 - c @ c
    logdet_q = n * jnp.log(s2) + 2 * jnp.sum(jnp.log(jnp.diag(lb)))
    logn = -0.5 * (quad + logdet_q + n * jnp.log(2 * jnp.pi))

    kff = kn.kernel(xj, xj, ell, sf2, "rbf")
    dff = kff - a.T @ a
    dff = 0.5 * (dff + dff.T)
    return (
        float(jax.device_get(logn)),
        np.asarray(jax.device_get(dff), dtype=np.float64),
        float(jax.device_get(s2)),
    )


def stable_logdet(h):
    h = 0.5 * (h + h.T)
    sign, ld = np.linalg.slogdet(h + 1e-9 * np.eye(h.shape[0]))
    if sign > 0:
        return float(ld)
    chol = np.linalg.cholesky(h + 1e-6 * np.eye(h.shape[0]))
    return float(2.0 * np.sum(np.log(np.diag(chol))))


def full_residual_penalty(dff, s2):
    return stable_logdet(np.eye(dff.shape[0]) + dff / s2)


def block_residual_penalty(dff, blocks, s2):
    total = 0.0
    for block in blocks:
        block = np.asarray(block, dtype=np.int32)
        if block.size == 0:
            continue
        db = dff[np.ix_(block, block)]
        total += stable_logdet(np.eye(block.size) + db / s2)
    return total


def objective_summary(params, x_train, y_train, bt_blocks, sbt_blocks):
    logn, dff, s2 = collapsed_components(params, x_train, y_train)
    full_pen = full_residual_penalty(dff, s2)
    bt_pen = block_residual_penalty(dff, bt_blocks, s2)
    sbt_pen = block_residual_penalty(dff, sbt_blocks, s2)
    penalties = {"FULL": full_pen, "BT": bt_pen, "SBT": sbt_pen}
    elbos = {name: logn - 0.5 * pen for name, pen in penalties.items()}

    denom = max(abs(full_pen), 1e-12)
    bt_excess = max(bt_pen - full_pen, 0.0)
    sbt_excess = max(sbt_pen - full_pen, 0.0)
    recovered = 100.0 * max(bt_pen - sbt_pen, 0.0) / max(bt_excess, 1e-12)
    return {
        "logn": logn,
        "dff": dff,
        "s2": s2,
        "penalties": penalties,
        "elbos": elbos,
        "excess_pct": {
            "BT": 100.0 * bt_excess / denom,
            "SBT": 100.0 * sbt_excess / denom,
        },
        "gain_pct_full": 100.0 * max(bt_pen - sbt_pen, 0.0) / denom,
        "recovered_pct": recovered,
    }


def write_outputs(rows):
    csv_path = HERE / f"{TABLE_STEM}_results{OUTPUT_TAG}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows)

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Snelson same-parameter residual-block comparison. SBT-SGPR is trained once with the configured optimizer; Exact, SBT, and BT corrections are evaluated at the learned parameters.}",
        r"\label{tab:snelson-residual-sbt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Correction & ELBO$/N$ & Residual logdet & Excess (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['correction']} & {row['elbo_n']:.4f} & "
            f"{row['residual_logdet']:.3f} & {row['excess_pct']:.2f} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    table_path = HERE / f"{TABLE_STEM}_table{OUTPUT_TAG}.tex"
    table_path.write_text("\n".join(tex_lines))
    print(f"csv    -> {csv_path}")
    print(f"table  -> {table_path}")


def main():
    global BLOCK, SPLIT_MODE
    args = parse_args()
    if args.split_mode is not None:
        SPLIT_MODE = args.split_mode
    if SPLIT_MODE not in {"boundary", "center"}:
        raise ValueError(f"unknown split mode {SPLIT_MODE!r}")
    np.random.seed(SEED)
    x_train, y_train, x_test, y_test = load_residual_dataset()
    n_train = x_train.shape[0]
    BLOCK, n_blocks = resolve_residual_block_size(
        n_train,
        block_size=args.block_size,
        n_blocks=args.n_blocks,
    )
    z0 = random_inducing(x_train, M, seed=SEED)

    print("\n=== SBT-SGPR ===", flush=True)
    print(
        f"dataset={DATASET_LABEL} n={n_train} M={M} "
        f"n_blocks={n_blocks} block={BLOCK} "
        f"z0=random-training-points seed={SEED} "
        f"split_mode={SPLIT_MODE} "
        f"optimizer={OPTIMIZER} steps={TRAIN_STEPS} "
        f"refresh_phases={SBT_REFRESH_SCHEDULE}",
        flush=True,
    )
    sbt_result = run_method("SBT-SGPR", x_train, y_train, x_test, y_test, z0)
    print(
        f"SBT-SGPR: rmse={sbt_result['rmse']:.4f} nll={sbt_result['nll']:.4f} "
        f"time={sbt_result['time']:.2f}s refreshes={sbt_result['refreshes']}",
        flush=True,
    )

    z_final = theta_numpy(sbt_result["params"])[3]
    bt_terms = pt.bt_terms_device(len(x_train), BLOCK, SEED)
    bt_blocks = tm.unpack_terms(bt_terms)
    _, sbt_info = sbt_terms_for_params(
        x_train, sbt_result["params"], return_info=True
    )
    sbt_blocks = [np.asarray(b, dtype=int) for b in sbt_info["blocks"]]
    sbt_order = order_from_blocks(
        sbt_blocks, np.asarray(sbt_info["order"], dtype=int)
    )
    bt_order = np.concatenate(bt_blocks)
    summary = objective_summary(
        sbt_result["params"], x_train, y_train, bt_blocks, sbt_blocks
    )
    dff = summary["dff"]
    bt_colors = choose_colors(len(bt_blocks))
    sbt_colors = choose_colors(len(sbt_blocks))

    print(
        "same-parameter residual logdet: "
        f"Exact={summary['penalties']['FULL']:.3f}, "
        f"SBT={summary['penalties']['SBT']:.3f}, "
        f"BT={summary['penalties']['BT']:.3f}",
        flush=True,
    )
    print(
        "excess over Exact: "
        f"SBT={summary['excess_pct']['SBT']:.2f}%, "
        f"BT={summary['excess_pct']['BT']:.2f}%; "
        f"SBT recovers {summary['recovered_pct']:.2f}% of BT excess",
        flush=True,
    )

    fig = plt.figure(figsize=(3.55, 4.10))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.075],
        hspace=0.35,
        wspace=0.10,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    logdet_ax = fig.add_subplot(gs[0, 2])
    cbar_ax = fig.add_subplot(gs[1, 2])

    points_panel(
        axes[0], x_train, y_train, bt_blocks, z_final,
        "Random blocks", bt_colors
    )
    points_panel(
        axes[1], x_train, y_train, sbt_blocks, z_final,
        "Selected blocks", sbt_colors
    )
    heat_vmax = float(np.percentile(np.abs(dff), 99.5))
    matrix_panel(
        axes[2], dff, bt_blocks, bt_order,
        "", bt_colors, heat_vmax
    )
    image = matrix_panel(
        axes[3], dff, sbt_blocks, sbt_order,
        "", sbt_colors, heat_vmax
    )
    logdet_scale_panel(logdet_ax, summary)
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label(
        r"$|(\mathbf{D}_{\mathbf{ff}})_{ij}|$",
        fontsize=AXIS_LABEL_FONTSIZE,
        labelpad=1,
    )
    cbar.ax.tick_params(
        labelsize=TICK_FONTSIZE,
        length=2,
        pad=1,
    )
    from matplotlib.lines import Line2D

    fig.legend(
        handles=[Line2D([], [], linestyle="none", marker="*",
                        markersize=9, markerfacecolor="#111111",
                        markeredgecolor="#111111", label="Inducing points")],
        loc="upper center", bbox_to_anchor=(0.45, 1.015),
        frameon=False, fontsize=ANNOTATION_FONTSIZE + 0.6,
        handletextpad=0.35, borderpad=0.1,
    )
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.02, top=0.90)
    left_pos = axes[2].get_position()
    right_pos = axes[3].get_position()
    fig.text(
        0.5 * (left_pos.x0 + right_pos.x1),
        left_pos.y1 + 0.014,
        r"Residual $|\mathbf{D}_{\mathbf{ff}}|$ under each partition",
        ha="center",
        va="bottom",
        fontsize=PANEL_TITLE_FONTSIZE,
    )
    HERE.mkdir(parents=True, exist_ok=True)
    out_png = HERE / f"{FIGURE_STEM}{OUTPUT_TAG}.png"
    report_png = REPORT_FIGURES / f"{FIGURE_STEM}.png"
    report_pdf = REPORT_FIGURES / f"{FIGURE_STEM}.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    if WRITE_REPORT:
        REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(report_png, dpi=180, bbox_inches="tight")
        fig.savefig(report_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    full_pen = summary["penalties"]["FULL"]
    correction_labels = {"FULL": "Exact", "SBT": "SBT", "BT": "RBT"}
    for correction in ["FULL", "SBT", "BT"]:
        pen = summary["penalties"][correction]
        excess_pct = 0.0 if correction == "FULL" else (
            100.0 * max(pen - full_pen, 0.0) / max(abs(full_pen), 1e-12)
        )
        rows.append({
            "correction": correction_labels[correction],
            "elbo_n": summary["elbos"][correction] / n_train,
            "residual_logdet": pen,
            "excess_pct": excess_pct,
            "sbt_gain_pct_full": summary["gain_pct_full"],
            "sbt_recovered_pct": summary["recovered_pct"],
            "anchor_chunks": int(sbt_info.get("anchor_chunks", 0)),
            "accepted_blocks": int(sbt_info.get("terms", 0)),
            "n_blocks": int(n_blocks),
            "block": int(BLOCK),
            "split_mode": SPLIT_MODE,
            "train_time_s": sbt_result["time"],
            "refreshes": sbt_result["refreshes"],
            "optimizer": sbt_result["optimizer"],
            "steps": sbt_result["steps"],
        })
    if WRITE_TABLE:
        write_outputs(rows)
    print(f"figure -> {out_png}")
    if WRITE_REPORT:
        print(f"report -> {report_pdf}")


if __name__ == "__main__":
    main()
