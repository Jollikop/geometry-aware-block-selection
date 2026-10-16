#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kin40k residual-structure heatmaps for RBF and Gibbs.

Same idea as ``run_residual_structure`` (Snelson), lifted to a real benchmark
dataset and to the non-stationary kernel:

  * load the official fitted SBT-SGPR parameters (or train them for a standalone
    exploratory invocation);
  * freeze the fitted parameters;
  * build the RBT cover (random equal-size blocks), the SBT cover
    (inducing-anchored blocks) and the dense-residual METIS oracle at those
    SAME parameters;
  * show |D_ff| = |K_ff - Q_ff| reordered by each cover, so the two covers are
    compared on one residual matrix -- block selection only.

Rows of the figure are kernels; columns are covers (RBT, SBT, METIS).

GPU: JAX picks the visible CUDA device automatically; nothing here is CPU-bound
except the dense N x N residual, which is streamed in row chunks.

Examples
--------
    python -m sgpr.residual_illustration.run_residual_structure_kin40k
    python -m sgpr.residual_illustration.run_residual_structure_kin40k \
        --kernels rbf gibbs --n-train 2000 --m 128 --block-size 200 --seed 0
"""
from __future__ import annotations

import argparse
import csv
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
from matplotlib.patches import Rectangle

matplotlib.rcParams.update({"mathtext.fontset": "cm"})

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).resolve().parent
SIM_ROOT = HERE.parent.parent
REPORT_FIGURES = SIM_ROOT.parent / "Report" / "figures"

from sgpr import jax_collapsed as jm            # noqa: E402
from sgpr import benchmark as bench              # noqa: E402
from sgpr import protocol                        # noqa: E402
from common import kernels as kn                # noqa: E402
from common import data_split as jd             # noqa: E402
from common import datasets as dm               # noqa: E402
from common import partitions as pt             # noqa: E402
from common import terms as tm                  # noqa: E402

KIN40K_NPZ = jd.DATASET_DIR / "kin40k.npz"
JITTER = 1e-8

KERNEL_TITLES = {"rbf": "RBF (stationary)", "gibbs": "Gibbs (non-stationary)"}
COVER_TITLES = {
    "RBT": "RBT blocks (random)",
    "SBT": "SBT blocks (anchored)",
    "METIS": "METIS blocks (dense oracle)",
}
PANEL_TITLE_FONTSIZE = 8.4
AXIS_LABEL_FONTSIZE = 7.6
ANNOTATION_FONTSIZE = 6.6
TICK_FONTSIZE = 6.2


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description="Kin40k residual-structure heatmaps (RBF vs Gibbs)."
    )
    ap.add_argument("--kernels", nargs="+", default=["rbf", "gibbs"],
                    help="kernels to compare, one figure row each")
    ap.add_argument("--n-train", type=int, default=2000,
                    help="training points; the dense residual is N x N")
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--m", type=int, default=128, help="inducing points")
    ap.add_argument("--block-size", type=int, default=None,
                    help="block size B; overrides --n-blocks")
    ap.add_argument("--n-blocks", type=int, default=10,
                    help="block count -> B = ceil(N / n_blocks)")
    ap.add_argument("--seed", type=int, default=0,
                    help="split seed, inducing-point seed and BT seed")
    ap.add_argument("--optimizer", default="lbfgs",
                    choices=("lbfgs", "adam", "lion"))
    ap.add_argument("--steps", type=int, default=protocol.LBFGS_STEPS)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument(
        "--refresh-schedule",
        default=",".join(map(str, protocol.LBFGS_REFRESH_SCHEDULE)),
        help="phase schedule; phases after the first refresh first",
    )
    ap.add_argument("--split-mode", default="boundary",
                    choices=("boundary", "center"))
    ap.add_argument("--chunk", type=int, default=512,
                    help="row chunk used to stream the dense residual")
    ap.add_argument("--outdir", default=str(HERE / "kin40k"))
    ap.add_argument(
        "--runs-dir", type=Path,
        help="official SGPR runs tree. When set, load SBT params and its fitted "
             "final blocks instead of retraining.",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-npz", action="store_true",
                    help="also dump D_ff, blocks and orders per kernel")
    ap.add_argument("--write-report", action="store_true",
                    help="also copy the figure into Report/figures")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Data / model
# --------------------------------------------------------------------------- #
def load_kin40k(seed, n_train, n_test):
    dm.load_raw("kin40k", data_dir=jd.DATASET_DIR)
    x_tr, y_tr, x_te, y_te, meta = jd.load_npz(
        KIN40K_NPZ, seed, n_train, n_test
    )
    return (
        np.asarray(jax.device_get(x_tr), dtype=np.float64),
        np.asarray(jax.device_get(y_tr), dtype=np.float64).reshape(-1),
        np.asarray(jax.device_get(x_te), dtype=np.float64),
        np.asarray(jax.device_get(y_te), dtype=np.float64).reshape(-1),
        meta,
    )


def random_inducing(x_train, m, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x_train.shape[0])[: int(m)]
    return np.asarray(x_train[idx], dtype=np.float64).copy()


def make_sbt_refresh(x_train, block, kernel_name, split_mode):
    """Host-side structure refresh: SBT cover at the current parameters."""
    def refresh(params):
        return pt.sbt_terms_device(
            x_train, params, block, kernel_name, split_mode=split_mode
        )

    refresh._host_refresh = True
    return refresh


def train_sbt(x_train, y_train, x_test, y_test, z0, *, kernel_name, block,
              split_mode, optimizer, steps, lr, refresh_schedule):
    params0 = jm.init_params(x_train, z0, kernel_name=kernel_name)
    loss = jm.make_loss(x_train, y_train, dense_mode=None,
                        kernel_name=kernel_name)
    terms0 = pt.sbt_terms_device(
        x_train, params0, block, kernel_name, split_mode=split_mode
    )
    refresh = make_sbt_refresh(x_train, block, kernel_name, split_mode)

    t0 = time.perf_counter()
    params, curve, info = jm.train_collapsed(
        loss,
        params0,
        terms0=terms0,
        term_fn=refresh,
        steps=steps,
        refresh_every=steps,
        optimizer=optimizer,
        lr=lr,
        grad_clip=10.0,
        train_z=True,
        log_every=max(1, steps // 10),
        refresh_schedule=refresh_schedule,
    )
    info.pop("final_terms", None)
    wall = time.perf_counter() - t0
    rmse, nll = jm.metrics_jax(
        params, x_train, y_train, x_test, y_test,
        exact=False, kernel_name=kernel_name,
    )
    return params, {
        "rmse": float(rmse),
        "nll": float(nll),
        "time_s": wall,
        "refreshes": int(info.get("refreshes", 0)),
        "search_s": float(info.get("search_s", 0.0)),
        "optimizer_s": float(info.get("optimizer_s", 0.0)),
        "final_loss": float(np.asarray(curve, dtype=float)[-1]),
    }


# --------------------------------------------------------------------------- #
# Residual algebra at frozen parameters
# --------------------------------------------------------------------------- #
def collapsed_components(params, x_train, y_train, kernel_name, chunk):
    """log N(y; 0, Qff + s2 I), the dense residual D_ff, and s2.

    D_ff is streamed in row chunks so the peak device memory is O(chunk * N * d)
    rather than O(N^2 * d): the Gibbs kernel materialises a [., ., d] difference
    tensor and that is what dominates on a real GPU.
    """
    xj = jnp.asarray(x_train)
    yj = jnp.asarray(y_train).reshape(-1)
    n = int(xj.shape[0])
    ell = jnp.exp(params["log_ell"])
    sf2 = jnp.exp(params["log_sf2"])
    s2 = jnp.exp(params["log_s2"])
    z = params["z"]
    warp = params.get("warp")

    kuu = kn.kernel(z, z, ell, sf2, kernel_name, warp) \
        + JITTER * jnp.eye(z.shape[0])
    kuf = kn.kernel(z, xj, ell, sf2, kernel_name, warp)
    luu = jnp.linalg.cholesky(kuu)
    a = jax.scipy.linalg.solve_triangular(luu, kuf, lower=True)      # [M, N]

    big = jnp.eye(z.shape[0]) + (a @ a.T) / s2
    lb = jnp.linalg.cholesky(big)
    c = jax.scipy.linalg.solve_triangular(lb, a @ yj, lower=True) / s2
    quad = (yj @ yj) / s2 - c @ c
    logdet_q = n * jnp.log(s2) + 2 * jnp.sum(jnp.log(jnp.diag(lb)))
    logn = -0.5 * (quad + logdet_q + n * jnp.log(2 * jnp.pi))

    @jax.jit
    def residual_rows(x_rows, a_rows):
        kfr = kn.kernel(x_rows, xj, ell, sf2, kernel_name, warp)
        return kfr - a_rows.T @ a

    chunk = max(1, int(chunk))
    out = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        rows = residual_rows(xj[start:stop], a[:, start:stop])
        out[start:stop] = np.asarray(jax.device_get(rows), dtype=np.float64)
    out = 0.5 * (out + out.T)
    return float(jax.device_get(logn)), out, float(jax.device_get(s2))


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
        block = np.asarray(block, dtype=np.int64)
        if block.size == 0:
            continue
        db = dff[np.ix_(block, block)]
        total += stable_logdet(np.eye(block.size) + db / s2)
    return total


def sbt_cover(x_train, params, block, kernel_name, split_mode):
    """Anchored ordering and its equal-size blocks at the fitted parameters."""
    order = np.asarray(
        jax.device_get(
            pt.anchor_order_device(
                jnp.asarray(x_train),
                jnp.asarray(params["z"]),
                jnp.exp(params["log_ell"]),
                kn.canonical_kernel_name(kernel_name),
                split_mode,
                params.get("warp"),
            )
        ),
        dtype=np.int64,
    )
    blocks = [np.sort(b) for b in
              pt.blocks_from_order(order, block, len(x_train))]
    tm.assert_partition(blocks, len(x_train))
    return blocks, order


def metis_cover(x_train, params, block, kernel_name):
    """Dense residual-graph oracle at the same frozen parameters."""
    order = np.asarray(
        bench._order_metis(x_train, params, block, kernel_name),
        dtype=np.int64,
    )
    blocks = [np.sort(b) for b in
              pt.blocks_from_order(order, block, len(x_train))]
    tm.assert_partition(blocks, len(x_train))
    return blocks, order


def load_official_sbt(runs_dir, kernel_name, m, seed):
    run_dir = (
        Path(runs_dir) / kernel_name / "kin40k" / f"M{m}" / "SBT-SGPR"
        / f"seed{seed}"
    )
    metrics_path = run_dir / "metrics.json"
    params_path = run_dir / "params.npz"
    blocks_path = run_dir / "final_blocks.npz"
    if not all(path.is_file() for path in (metrics_path, params_path, blocks_path)):
        raise FileNotFoundError(f"incomplete official SBT run: {run_dir}")
    metrics = __import__("json").loads(metrics_path.read_text())
    identity = (kernel_name, "kin40k", int(m), "SBT-SGPR", int(seed))
    observed = tuple(metrics.get(key) for key in
                     ("kernel", "dataset", "M", "method", "seed"))
    if observed != identity:
        raise ValueError(f"official SBT identity mismatch: {observed} != {identity}")
    with np.load(params_path) as archive:
        params = {key: jnp.asarray(archive[key]) for key in archive.files}
    with np.load(blocks_path) as archive:
        terms = tuple(jnp.asarray(archive[key]) for key in ("coeffs", "idx", "mask"))
    blocks = [np.sort(np.asarray(block, dtype=np.int64))
              for block in tm.unpack_terms(terms)]
    tm.assert_partition(blocks, int(metrics["n_train"]))
    fit = {
        "rmse": float(metrics["rmse"]), "nll": float(metrics["nll"]),
        "time_s": float(metrics.get("total_s", 0.0)),
        "refreshes": int(metrics.get("refreshes", 0)),
        "search_s": float(metrics.get("search_s", 0.0)),
        "optimizer_s": float(metrics.get("optimizer_s", 0.0)),
        "final_loss": float(metrics.get("obj_n", float("nan")))
                        * int(metrics["n_train"]),
    }
    return params, fit, blocks


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def block_colors(n):
    """Use one red outline for every block in the paper heatmap."""
    return ["#E31A1C"] * int(n)


def matrix_panel(ax, dff, blocks, ordering, title, colors, vmax):
    """|D_ff| reordered by ``ordering`` with the block cover outlined."""
    ordering = np.asarray(ordering, dtype=np.int64)
    invp = np.empty(ordering.size, dtype=np.int64)
    invp[ordering] = np.arange(ordering.size)
    shown = np.abs(dff)[np.ix_(ordering, ordering)]
    image = ax.imshow(
        shown, cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-12),
        interpolation="nearest", rasterized=True,
    )
    for color, block in zip(colors, blocks):
        p = np.sort(invp[np.asarray(block, dtype=np.int64)])
        ax.add_patch(Rectangle(
            (p[0] - 0.5, p[0] - 0.5),
            p[-1] - p[0] + 1, p[-1] - p[0] + 1,
            fill=False, edgecolor=color, linewidth=0.8,
        ))
    if title:
        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def make_figure(results, out_stem, outdir, write_report, meta_line):
    n_rows = len(results)
    fig = plt.figure(figsize=(8.0, 2.85 * n_rows))
    gs = fig.add_gridspec(
        n_rows, 4, width_ratios=[1.0, 1.0, 1.0, 0.055],
        hspace=0.24, wspace=0.08,
    )
    for row, res in enumerate(results):
        dff = res["dff"]
        vmax = float(np.percentile(np.abs(dff), 99.5))
        axes = [fig.add_subplot(gs[row, col]) for col in range(3)]
        cbar_ax = fig.add_subplot(gs[row, 3])
        image = None
        for col, cover in enumerate(("RBT", "SBT", "METIS")):
            pen = res["penalties"][cover]
            excess = res["excess_pct"][cover]
            title = (f"{COVER_TITLES[cover]}\n"
                     f"log det = {pen:.1f}  (+{excess:.1f}% vs Exact)")
            image = matrix_panel(
                axes[col], dff, res["blocks"][cover], res["orders"][cover],
                title, res["colors"][cover], vmax,
            )
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label(r"$|(\mathbf{D}_{\mathbf{ff}})_{ij}|$",
                       fontsize=AXIS_LABEL_FONTSIZE, labelpad=1)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE, length=2, pad=1)
        pos = axes[0].get_position()
        fig.text(
            0.012, 0.5 * (pos.y0 + pos.y1),
            f"{KERNEL_TITLES.get(res['kernel'], res['kernel'])}\n"
            f"Exact log det = {res['penalties']['Exact']:.1f}",
            rotation=90, ha="left", va="center",
            fontsize=PANEL_TITLE_FONTSIZE,
        )
    fig.suptitle(
        r"kin40k: heatmap of $|\mathbf{D}_{\mathbf{ff}}|$ under each partition",
        fontsize=PANEL_TITLE_FONTSIZE + 1.2, y=0.995,
    )
    fig.text(0.5, 0.002, meta_line, ha="center", va="bottom",
             fontsize=ANNOTATION_FONTSIZE, color="#444444")
    fig.subplots_adjust(left=0.075, right=0.93, bottom=0.035, top=0.93)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{out_stem}.png"
    pdf = outdir / f"{out_stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    if write_report:
        REPORT_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(REPORT_FIGURES / f"{out_stem}.png", dpi=200,
                    bbox_inches="tight")
        fig.savefig(REPORT_FIGURES / f"{out_stem}.pdf", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_csv(results, out_stem, outdir, config):
    path = Path(outdir) / f"{out_stem}_results.csv"
    fields = [
        "kernel", "correction", "elbo_n", "residual_logdet", "excess_pct",
        "sbt_recovered_pct", "metis_recovered_pct", "rmse", "nll",
        "train_time_s", "refreshes",
        "n_train", "M", "block", "n_blocks", "seed", "optimizer", "steps",
        "split_mode",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for res in results:
            for correction in ("Exact", "METIS", "SBT", "RBT"):
                pen = res["penalties"][correction]
                writer.writerow({
                    "kernel": res["kernel"],
                    "correction": correction,
                    "elbo_n": (res["logn"] - 0.5 * pen) / config["n_train"],
                    "residual_logdet": pen,
                    "excess_pct": res["excess_pct"].get(correction, 0.0),
                    "sbt_recovered_pct": res["recovered_pct"],
                    "metis_recovered_pct": res["metis_recovered_pct"],
                    "rmse": res["fit"]["rmse"],
                    "nll": res["fit"]["nll"],
                    "train_time_s": res["fit"]["time_s"],
                    "refreshes": res["fit"]["refreshes"],
                    **{k: config[k] for k in (
                        "n_train", "M", "block", "n_blocks", "seed",
                        "optimizer", "steps", "split_mode")},
                })
    return path


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    kernels = [kn.canonical_kernel_name(k) for k in args.kernels]

    print("JAX devices:", jax.devices(), flush=True)

    x_train, y_train, x_test, y_test, meta = load_kin40k(
        args.seed, args.n_train, args.n_test
    )
    n_train = x_train.shape[0]
    if args.block_size is not None:
        block = max(1, int(args.block_size))
        n_blocks = int(math.ceil(n_train / block))
    else:
        n_blocks = max(1, int(args.n_blocks))
        block = max(1, int(math.ceil(n_train / n_blocks)))
    refresh_schedule = [int(v) for v in args.refresh_schedule.split(",") if v]
    steps = int(args.steps)
    if refresh_schedule and sum(refresh_schedule) != steps:
        steps = sum(refresh_schedule)

    z0 = random_inducing(x_train, args.m, seed=args.seed)
    config = {
        "n_train": n_train, "M": int(args.m), "block": block,
        "n_blocks": n_blocks, "seed": int(args.seed),
        "optimizer": args.optimizer, "steps": steps,
        "split_mode": args.split_mode,
    }
    print(
        f"dataset=kin40k d={x_train.shape[1]} n_train={n_train} "
        f"n_test={x_test.shape[0]} M={args.m} block={block} "
        f"n_blocks={n_blocks} seed={args.seed} "
        f"optimizer={args.optimizer} steps={steps} "
        f"refresh_phases={refresh_schedule}",
        flush=True,
    )

    rbt_blocks = [np.sort(np.asarray(b, dtype=np.int64))
                 for b in tm.unpack_terms(
                     pt.bt_terms_device(n_train, block, args.seed))]
    tm.assert_partition(rbt_blocks, n_train)
    rbt_order = np.concatenate(rbt_blocks)

    results = []
    for kernel_name in kernels:
        print(f"\n=== SBT-SGPR | kernel={kernel_name} ===", flush=True)
        if args.runs_dir:
            params, fit, sbt_blocks = load_official_sbt(
                args.runs_dir, kernel_name, args.m, args.seed
            )
            if int(n_train) != sum(len(value) for value in sbt_blocks):
                raise ValueError(
                    "--n-train must match the official run when --runs-dir is set"
                )
            sbt_order = np.concatenate(sbt_blocks)
        else:
            params, fit = train_sbt(
                x_train, y_train, x_test, y_test, z0,
                kernel_name=kernel_name, block=block,
                split_mode=args.split_mode, optimizer=args.optimizer,
                steps=steps, lr=args.lr, refresh_schedule=refresh_schedule,
            )
            sbt_blocks, sbt_order = sbt_cover(
                x_train, params, block, kernel_name, args.split_mode
            )
        print(
            f"{kernel_name}: rmse={fit['rmse']:.4f} nll={fit['nll']:.4f} "
            f"time={fit['time_s']:.1f}s refreshes={fit['refreshes']}",
            flush=True,
        )

        metis_blocks, metis_order = metis_cover(
            x_train, params, block, kernel_name
        )
        logn, dff, s2 = collapsed_components(
            params, x_train, y_train, kernel_name, args.chunk
        )
        penalties = {
            "Exact": full_residual_penalty(dff, s2),
            "METIS": block_residual_penalty(dff, metis_blocks, s2),
            "SBT": block_residual_penalty(dff, sbt_blocks, s2),
            "RBT": block_residual_penalty(dff, rbt_blocks, s2),
        }
        denom = max(abs(penalties["Exact"]), 1e-12)
        rbt_excess = max(penalties["RBT"] - penalties["Exact"], 0.0)
        excess_pct = {
            "Exact": 0.0,
            "METIS": 100.0 * max(
                penalties["METIS"] - penalties["Exact"], 0.0
            ) / denom,
            "SBT": 100.0 * max(penalties["SBT"] - penalties["Exact"], 0.0) / denom,
            "RBT": 100.0 * rbt_excess / denom,
        }
        recovered = (100.0 * max(penalties["RBT"] - penalties["SBT"], 0.0)
                     / max(rbt_excess, 1e-12))
        metis_recovered = (
            100.0 * max(penalties["RBT"] - penalties["METIS"], 0.0)
            / max(rbt_excess, 1e-12)
        )
        print(
            "same-parameter residual logdet: "
            f"Exact={penalties['Exact']:.3f}, METIS={penalties['METIS']:.3f}, "
            f"SBT={penalties['SBT']:.3f}, RBT={penalties['RBT']:.3f} | "
            f"SBT recovers {recovered:.2f}% and METIS "
            f"{metis_recovered:.2f}% of the RBT excess",
            flush=True,
        )

        results.append({
            "kernel": kernel_name,
            "params": params,
            "fit": fit,
            "logn": logn,
            "s2": s2,
            "dff": dff,
            "penalties": penalties,
            "excess_pct": excess_pct,
            "recovered_pct": recovered,
            "metis_recovered_pct": metis_recovered,
            "blocks": {
                "RBT": rbt_blocks, "SBT": sbt_blocks,
                "METIS": metis_blocks,
            },
            "orders": {
                "RBT": rbt_order, "SBT": sbt_order,
                "METIS": metis_order,
            },
            "colors": {
                "RBT": block_colors(len(rbt_blocks)),
                "SBT": block_colors(len(sbt_blocks)),
                "METIS": block_colors(len(metis_blocks)),
            },
        })

        if args.save_npz:
            outdir = Path(args.outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                outdir / f"kin40k_{kernel_name}_residual{args.tag}.npz",
                dff=dff, s2=s2, logn=logn,
                sbt_order=sbt_order, rbt_order=rbt_order,
                metis_order=metis_order,
                z=np.asarray(jax.device_get(params["z"])),
                log_ell=np.asarray(jax.device_get(params["log_ell"])),
            )

    stem = f"residual_structure_kin40k_{'_'.join(kernels)}{args.tag}"
    meta_line = (
        f"N={n_train}, M={args.m}, B={block} ({n_blocks} blocks), "
        f"seed={args.seed}, {args.optimizer} {steps} steps, "
        f"split={args.split_mode}"
    )
    png, pdf = make_figure(results, stem, args.outdir, args.write_report,
                           meta_line)
    csv_path = write_csv(results, stem, args.outdir, config)
    print(f"\nfigure -> {png}")
    print(f"figure -> {pdf}")
    print(f"csv    -> {csv_path}")


if __name__ == "__main__":
    main()
