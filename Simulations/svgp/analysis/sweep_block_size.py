#!/usr/bin/env python3
"""Block-size ablation: how much of the available gain does a block of size B capture?

The two block methods rest on the exact decomposition

    F_block = F_T + 1/2 * sum_B gain(B),    gain(B) = sum_i log A_ii - logdet A_B,

with A = I + D_ff / sigma^2. ``gain(B)`` is non-negative (Hadamard-Fischer) and
is the entire difference between a block bound and the diagonal T-SVGP bound,
so summing it over a partition measures exactly what the block structure buys.

This script sweeps B on a *nested* chain (each value doubles the previous one)
using the anchor ordering of the trained run. Because the blocks are consecutive
chunks of one fixed ordering, a partition at 2B is a coarsening of the partition
at B, and Fischer's inequality then makes the total gain provably non-decreasing
in B. The curve is therefore a genuine saturation curve, not a noisy sequence.

Nothing is retrained. The kernel hyperparameters, the inducing points and hence
the anchor ordering are read from the saved run, so the question answered is
"at the trained theta, how much gain would a different B have captured" -- not
"what theta would training at that B have produced". That is the right first
experiment because it is free, and because the quantity of interest (the decay
of residual correlation away from an inducing point) is a property of the
geometry, not of the block size used while training.

The reference scale is N/M: with M inducing points the anchor ordering induces
M cells of about N/M points each, so B ~ N/M is the size at which a block stops
matching a cell and starts merging cells. The plots mark it.

Input is a runs tree written by ``svgp/run_benchmark.py --save-params``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def load_backend():
    """Import JAX and the engine.

    Deferred so that ``--from-json`` can redraw the figures on a laptop with no
    JAX installed.
    """
    global jax, jnp, sbt_terms_device, split_for, engine_for
    import jax
    import jax.numpy as jnp
    from common.partitions import sbt_terms_device
    from svgp.analysis._common import engine_for, split_for


KERNEL_LABELS = {"rbf": "RBF", "matern32": "Matern-3/2",
                 "matern52": "Matern-5/2", "gibbs": "Gibbs"}
M_COLORS = {16: "#937860", 32: "#DA8BC3", 64: "#4C72B0", 128: "#55A868",
            256: "#C44E52", 512: "#8172B2", 1024: "#64B5CD"}

# A chunk of blocks is materialised as a [c, B, B] stack, and the gain needs a
# few temporaries of that size. Roughly a hundred megabytes of working set.
MEMORY_BUDGET = 1.2e7

def gain_at_block(engine, params, x_np, block, kernel_name):
    """Total sum_B gain(B) of the anchor partition at block size ``block``."""
    coeffs, idx, mask = sbt_terms_device(x_np, params, block, kernel_name)
    active = np.asarray(jax.device_get(coeffs)) > 0.5
    idx = jnp.asarray(np.asarray(jax.device_get(idx))[active])
    mask = jnp.asarray(np.asarray(jax.device_get(mask))[active])
    n_blocks = int(idx.shape[0])
    chunk = max(1, int(MEMORY_BUDGET // (block * block)))
    total = 0.0
    for start in range(0, n_blocks, chunk):
        stop = min(start + chunk, n_blocks)
        gains = engine["block_gains"](params, idx[start:stop], mask[start:stop])
        total += float(jnp.sum(gains))
    return total, n_blocks


def sweep(runs_dir: Path, blocks, datasets, kernels, m_values, seeds):
    load_backend()
    rows = []
    for path in sorted(runs_dir.rglob("metrics.json")):
        meta = json.loads(path.read_text())
        cfg = meta.get("config", {})
        if not str(cfg.get("method", "")).startswith("SBT"):
            continue
        if not (path.parent / "params.npz").exists():
            continue
        if datasets and cfg.get("dataset") not in datasets:
            continue
        if kernels and cfg.get("kernel") not in kernels:
            continue
        if m_values and int(cfg["M"]) not in m_values:
            continue
        if seeds and int(cfg["seed"]) not in seeds:
            continue

        params = {k: jnp.asarray(v)
                  for k, v in np.load(path.parent / "params.npz").items()}
        x_tr, *_ = split_for(cfg)
        x_np = np.asarray(x_tr, dtype=np.float64)
        engine = engine_for(cfg)
        n_train = int(x_np.shape[0])

        curve, counts, seconds = {}, {}, {}
        for block in blocks:
            if block > n_train:
                continue
            t0 = time.perf_counter()
            total, n_blocks = gain_at_block(
                engine, params, x_np, int(block), cfg["kernel"])
            seconds[block] = time.perf_counter() - t0
            curve[block] = total
            counts[block] = n_blocks
            print(f"    B={block:5d}  blocks={n_blocks:5d}  "
                  f"gain={total:12.3f}  gain/N={total / n_train:8.5f}  "
                  f"({seconds[block]:.1f}s)")

        used = sorted(curve)
        # Monotonicity is a theorem for a nested chain; a violation therefore
        # means the chain was not nested (custom --blocks) or a numerical
        # problem, and must be reported rather than smoothed away.
        drops = [(a, b) for a, b in zip(used, used[1:])
                 if curve[b] < curve[a] - 1e-6 * max(1.0, abs(curve[a]))]

        rows.append({
            "kernel": cfg["kernel"],
            "dataset": cfg["dataset"],
            "M": int(cfg["M"]),
            "seed": int(cfg["seed"]),
            "n_train": n_train,
            "block_trained": int(cfg.get("block", 0)),
            "n_over_m": n_train / float(cfg["M"]),
            "blocks": used,
            "gain": [curve[b] for b in used],
            "gain_per_point": [curve[b] / n_train for b in used],
            "n_blocks": [counts[b] for b in used],
            "seconds": [seconds[b] for b in used],
            "monotonicity_violations": drops,
        })
        print(f"  done {cfg['kernel']}/{cfg['dataset']}/M{cfg['M']}"
              f"/seed{cfg['seed']}   N={n_train}  N/M={n_train / cfg['M']:.0f}"
              + ("   VIOLAZIONI DI MONOTONIA: " + str(drops) if drops else ""))
    return rows


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def draw(rows, out_base: Path, formats, mode: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pairs = sorted({(r["kernel"], r["dataset"]) for r in rows})
    kernels = sorted({k for k, _ in pairs})
    datasets = sorted({d for _, d in pairs})
    fig, axes = plt.subplots(
        len(kernels), len(datasets), squeeze=False,
        figsize=(3.9 * len(datasets), 3.2 * len(kernels)))

    ms_seen = []
    for row_i, kernel in enumerate(kernels):
        for col_i, dataset in enumerate(datasets):
            ax = axes[row_i][col_i]
            entries = [r for r in rows
                       if r["kernel"] == kernel and r["dataset"] == dataset]
            if not entries:
                ax.set_visible(False)
                continue
            entries.sort(key=lambda r: r["M"])
            n_train = entries[0]["n_train"]
            for r in entries:
                b = np.asarray(r["blocks"], dtype=float)
                g = np.asarray(r["gain"], dtype=float)
                colour = M_COLORS.get(r["M"], "0.4")
                if mode == "share":
                    x_plot, y = b, (g / g[-1] if g[-1] > 0 else g * 0.0)
                elif mode == "absolute":
                    x_plot, y = b, g / r["n_train"]
                else:
                    # Relative gain bought by doubling B. Cost per structural
                    # pass is (N/B) * B^3 = N*B^2, so a doubling costs 4x; the
                    # point at which this curve falls well under that is where
                    # a larger block stops paying. Deliberately NOT gain/B^2,
                    # which is always maximised at B=1 and merely recovers
                    # T-SVGP instead of answering the question.
                    if len(g) < 2:
                        continue
                    x_plot = b[:-1]
                    y = np.where(g[:-1] > 0, (g[1:] - g[:-1]) / g[:-1], np.nan)
                ax.plot(x_plot, y, marker="o", markersize=4.0, linewidth=1.6,
                        color=colour, label=f"M={r['M']}")
                ax.axvline(r["n_over_m"], color=colour, linestyle=":",
                           linewidth=1.1, alpha=0.85)
                ms_seen.append(r["M"])
            ax.axvline(entries[0]["block_trained"], color="0.35",
                       linestyle="--", linewidth=1.1)
            ax.set_xscale("log", base=2)
            if mode == "share":
                ax.axhline(1.0, color="0.75", linewidth=0.8)
                ax.set_ylim(-0.02, 1.05)
            elif mode == "marginal":
                ax.set_yscale("log")
            ax.set_title(f"{dataset} - {KERNEL_LABELS.get(kernel, kernel)}\n"
                         f"N = {n_train}", fontsize=10.5)
            ax.grid(True, alpha=0.22, linewidth=0.7)
            ax.tick_params(labelsize=8.5)
            if row_i == len(kernels) - 1:
                ax.set_xlabel("block size B", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(
                    {"share": "share of gain at largest B",
                     "absolute": "gain per training point (nats)",
                     "marginal": "relative gain from doubling B"}[mode],
                    fontsize=9.5)

    handles = [Line2D([0], [0], color=M_COLORS.get(m, "0.4"), marker="o",
                      markersize=4.0, linewidth=1.6, label=f"M={m}")
               for m in sorted(set(ms_seen))]
    handles += [
        Line2D([0], [0], color="0.4", linestyle=":", linewidth=1.1,
               label="B = N/M (anchor cell size)"),
        Line2D([0], [0], color="0.35", linestyle="--", linewidth=1.1,
               label="B used for training"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(6, len(handles)), frameon=True, fontsize=9.5)
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 1.0))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        target = out_base.with_suffix(f".{extension}")
        fig.savefig(target, dpi=190, bbox_inches="tight", pad_inches=0.04)
        print(f"wrote {target}")
    plt.close(fig)


def write_csv(rows, path: Path):
    lines = ["kernel,dataset,M,seed,n_train,n_over_m,block_trained,B,"
             "n_blocks,gain,gain_per_point,share_of_max,seconds"]
    for r in rows:
        g = np.asarray(r["gain"], dtype=float)
        top = g[-1] if g[-1] > 0 else 1.0
        for i, b in enumerate(r["blocks"]):
            lines.append(
                f"{r['kernel']},{r['dataset']},{r['M']},{r['seed']},"
                f"{r['n_train']},{r['n_over_m']:.2f},{r['block_trained']},"
                f"{b},{r['n_blocks'][i]},{g[i]:.6f},"
                f"{r['gain_per_point'][i]:.8f},{g[i] / top:.6f},"
                f"{r['seconds'][i]:.3f}")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def summarise(rows):
    print("\nknee of the saturation curve "
          "(smallest B reaching 90% / 95% of the gain at the largest B):")
    for r in sorted(rows, key=lambda r: (r["kernel"], r["dataset"], r["M"])):
        g = np.asarray(r["gain"], dtype=float)
        if g[-1] <= 0:
            continue
        share = g / g[-1]

        def first(threshold):
            hit = [b for b, s in zip(r["blocks"], share) if s >= threshold]
            return hit[0] if hit else None

        b90, b95 = first(0.90), first(0.95)
        ratio = f"{b90 / r['n_over_m']:.2f}" if b90 else "-"
        print(f"  {r['kernel']:9} {r['dataset']:18} M{r['M']:<5} "
              f"N/M={r['n_over_m']:7.0f}   B90={str(b90):>6}  "
              f"B95={str(b95):>6}   B90/(N/M)={ratio:>6}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="block_size_ablation")
    ap.add_argument("--blocks", type=int, nargs="*",
                    default=[32, 64, 128, 256, 512, 1024, 2048, 4096],
                    help="nested chain of block sizes; each should double the "
                         "previous one, otherwise monotonicity is not implied")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--kernels", nargs="*", default=None)
    ap.add_argument("--m-values", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--formats", nargs="*", default=["png", "pdf"])
    ap.add_argument("--from-json", default=None,
                    help="replot a previous sweep instead of recomputing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.prefix}.json"

    if args.from_json:
        rows = json.loads(Path(args.from_json).read_text())
    else:
        rows = sweep(Path(args.runs_dir), sorted(args.blocks),
                     set(args.datasets or []), set(args.kernels or []),
                     set(args.m_values or []), set(args.seeds or []))
        if not rows:
            raise SystemExit(
                "no SBT run with params.npz found: this needs a campaign run "
                "with --save-params")
        json_path.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {json_path}")
        write_csv(rows, out_dir / f"{args.prefix}.csv")

    draw(rows, out_dir / f"{args.prefix}_share", args.formats, "share")
    draw(rows, out_dir / f"{args.prefix}_absolute", args.formats, "absolute")
    draw(rows, out_dir / f"{args.prefix}_marginal", args.formats, "marginal")
    summarise(rows)

    violations = [r for r in rows if r["monotonicity_violations"]]
    if violations:
        print("\nWARNING: monotonicity is violated in "
              f"{len(violations)} runs. On a nested chain Fischer's inequality "
              "makes that impossible, so check --blocks (each value must "
              "double the previous one) before trusting the figures.")


if __name__ == "__main__":
    main()
