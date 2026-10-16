"""Replot frozen Kin40k residuals; no training, source edits or new partitions.

Images show entrywise D_ij**2 (not a matrix square), with the diagonal omitted.
Leakage statistics use W_ij = D_ij**2 / sum_{i != j} D_ij**2, W_ii = 0.
All statistics use full-resolution W; images use area means, without clipping.
Saved orders are split at the archived block size and their log-determinants
are checked against the archived CSV before accepting the reconstruction.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, PowerNorm, LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.linalg import cholesky

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
COVERS = ("RBT", "SBT", "METIS")
TITLES = {"RBT": "Random blocks", "SBT": "Selected blocks", "METIS": "METIS blocks"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_mass(d):
    d = np.asarray(d, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 2:
        raise ValueError("Residual must be square with N >= 2")
    if not np.isfinite(d).all() or not np.allclose(d, d.T, rtol=1e-9, atol=1e-10):
        raise ValueError("Residual must be finite and symmetric")
    w = np.square(d)
    np.fill_diagonal(w, 0.0)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("M_off = 0 or nonfinite: normalized leakage is undefined")
    w /= total
    return w, total


def partition_stats(w, order, block_size):
    n = len(w)
    order = np.asarray(order)
    if block_size < 1 or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("Invalid partition order or block size")
    blocks = [order[i:i + block_size] for i in range(0, n, block_size)]
    within = sum(float(w[np.ix_(b, b)].sum()) for b in blocks)
    p = sum(len(b) * (len(b) - 1) for b in blocks) / (n * (n - 1))
    rho = float(np.clip(1 - within, 0, 1))
    return dict(retained_mass=within, rho=rho, random_expected_retained=p,
                random_expected_rho=1-p,
                retained_vs_random=within/p if p > 0 else None), blocks


def area_mean(a, target=650):
    step = max(1, int(np.ceil(len(a)/target)))
    starts = np.arange(0, len(a), step)
    sizes = np.diff(np.r_[starts, len(a)])
    out = np.add.reduceat(np.add.reduceat(a, starts, axis=0), starts, axis=1)
    return out / (sizes[:, None] * sizes[None, :]), step


def load_row(source, kernel, csv_rows, quantity="squared"):
    table = {r["correction"]: r for r in csv_rows if r["kernel"] == kernel}
    cfg = table["Exact"]
    n, block = int(cfg["n_train"]), int(cfg["block"])
    if any(any(r[k] != cfg[k] for k in ("n_train", "M", "block", "seed")) for r in table.values()):
        raise ValueError("Inconsistent archived configuration")
    with np.load(source / f"kin40k_{kernel}_residual.npz", allow_pickle=False) as archive:
        d, s2 = archive["dff"], float(archive["s2"])
        if d.shape != (n, n) or not s2 > 0:
            raise ValueError("Residual/configuration mismatch")
        w, moff = normalized_mass(d)
        panels = []
        for cover in COVERS:
            order = archive[f"{cover.lower()}_order"]
            stats, blocks = partition_stats(w, order, block)
            penalty = 0.0
            for b in blocks:
                db = d[np.ix_(b, b)] / s2
                h = (db + db.T)/2 + (1 + 1e-9)*np.eye(len(b))
                chol = cholesky(h, lower=True, check_finite=False)
                penalty += float(2*np.log(np.diag(chol)).sum())
            expected = float(table[cover]["residual_logdet"])
            if not np.isclose(penalty, expected, rtol=1e-7, atol=1e-5):
                raise ValueError(f"Archived partition/logdet mismatch: {kernel}/{cover}")
            shown, step = area_mean(w[np.ix_(order, order)])
            if quantity == "squared":
                shown = shown * moff
            elif quantity != "normalized":
                raise ValueError("quantity must be squared or normalized")
            stats.update(kernel=kernel, cover=cover, M_off=moff, n_train=n,
                         M=int(cfg["M"]), block_size=block, seed=int(cfg["seed"]),
                         residual_logdet=expected,
                         logdet_vs_dense=expected/float(cfg["residual_logdet"]))
            panels.append(dict(image=shown, step=step, stats=stats))
    return dict(kernel=kernel, n=n, block=block, config=cfg, panels=panels)


def mass_palette(name):
    if name == "gray":
        return "gray_r"
    if name == "blue":
        # Preserve exact white at zero, unlike the off-white endpoint of Blues.
        colors = matplotlib.colormaps["Blues"](np.linspace(0, 1, 256))
        colors[0] = (1, 1, 1, 1)
        return LinearSegmentedColormap.from_list("white_to_blue", colors)
    raise ValueError(f"Unknown palette: {name}")


def draw(rows, gamma=1.0, palette="gray", quantity="squared"):
    fig = plt.figure(figsize=(11.8, 4.25 * len(rows)))
    gs = fig.add_gridspec(len(rows), 4, width_ratios=(1, 1, 1, .045),
                          left=.06, right=.89, bottom=.10, top=.85,
                          hspace=.38, wspace=.16)
    for r, row in enumerate(rows):
        vmax = max(float(p["image"].max()) for p in row["panels"])
        norm = Normalize(vmin=0, vmax=vmax) if gamma == 1 else PowerNorm(gamma, vmin=0, vmax=vmax)
        for c, panel in enumerate(row["panels"]):
            ax = fig.add_subplot(gs[r, c])
            stat = panel["stats"]
            image = ax.imshow(panel["image"], cmap=mass_palette(palette), norm=norm,
                              interpolation="nearest", rasterized=True,
                              extent=(-.5, row["n"]-.5, row["n"]-.5, -.5))
            for start in range(0, row["n"], row["block"]):
                size = min(row["block"], row["n"]-start)
                ax.add_patch(Rectangle((start-.5, start-.5), size, size,
                                      fill=False, edgecolor="0.40", linewidth=.65))
            ax.set_title(f"{TITLES[stat['cover']]}\n"
                         f"log-det: {stat['logdet_vs_dense']:.2f} " + r"$\times$ dense",
                         fontsize=12, pad=11)
            ax.set_xlabel(r"Off-block mass $\rho(\mathcal{P}) = $" + f"{stat['rho']:.3f}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel("RBF" if row["kernel"] == "rbf" else "Gibbs", fontsize=14, labelpad=12)
        cb = fig.colorbar(image, cax=fig.add_subplot(gs[r, 3]))
        # Space ticks in display coordinates; linear-data ticks crowd the dark
        # end when a strong gamma is used. Tick values still show actual mass.
        cb.set_ticks(vmax * np.linspace(0, 1, 4)**(1/gamma))
        cb.formatter = FuncFormatter(lambda x, pos: "0" if x == 0 else f"{x:.1e}")
        cb.update_ticks()
        cb.ax.tick_params(labelsize=10, length=3)
        cb.set_label(r"$[(D_{ff})_{ij}]^{2}$" if quantity == "squared" else
                     r"$W_{ij}=[(D_{ff})_{ij}]^{2}/M_{\mathrm{off}}$", fontsize=14, labelpad=10)
    cfg = rows[0]["config"]
    fig.suptitle(r"Squared residual covariance $[(D_{ff})_{ij}]^2$" if quantity == "squared" else
                 r"Normalized squared residual mass $W_{ij}$", fontsize=17, y=.995)
    fig.text(.5, .935, f"Kin40k | N={cfg['n_train']}, M={cfg['M']}, b={cfg['block']}, seed={cfg['seed']}"
             " | Same fitted parameters within each kernel", ha="center", fontsize=11)
    fig.text(.5, .046, "Diagonal omitted. "
             f"White = 0; {'black' if palette == 'gray' else 'dark blue'} = row maximum. " +
             ("Linear color scale" if gamma == 1 else f"Color gamma={gamma:g}") +
             "; no clipping; shared scale within row.",
             ha="center", fontsize=10)
    fig.text(.5, .020, "Images show area means; leakage uses all original entries. "
             "Dense denotes the residual log-determinant endpoint, not exact GP evidence.",
             ha="center", fontsize=9.5)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gamma", type=float, default=1., help="Display contrast only; 1 = linear, 0.4 reveals small entries")
    ap.add_argument("--palette", choices=("gray", "blue"), default="gray")
    ap.add_argument("--quantity", choices=("squared", "normalized"), default="squared",
                    help="Entrywise squared covariance, or off-diagonal normalized mass")
    a = ap.parse_args()
    if not np.isfinite(a.gamma) or a.gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    source, out = a.source.resolve(), a.out.resolve()
    if out.exists() or source in out.parents or out in source.parents or out == source:
        raise ValueError("Choose a new output directory disjoint from frozen input")
    paths = [source / f"kin40k_{k}_residual.npz" for k in ("rbf", "gibbs")]
    paths.append(source / "residual_structure_kin40k_rbf_gibbs_results.csv")
    before = {str(p): digest(p) for p in paths}
    with paths[-1].open() as f:
        csv_rows = list(csv.DictReader(f))
    rows = [load_row(source, k, csv_rows, a.quantity) for k in ("rbf", "gibbs")]
    out.mkdir(parents=True)
    fig = draw(rows, a.gamma, a.palette, a.quantity)
    fig.savefig(out / "residual_mass_kin40k.pdf", dpi=300)
    fig.savefig(out / "residual_mass_kin40k.png", dpi=180)
    plt.close(fig)
    stats = [p["stats"] for r in rows for p in r["panels"]]
    with (out / "residual_mass_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    if before != {str(p): digest(p) for p in paths}:
        raise ValueError("Frozen input changed")
    manifest = dict(source_hashes=before, source_unchanged=True, metrics=stats,
                    figure_only=True, freeze_certification=False,
                    diagonal="zero; excluded from numerator and denominator",
                    scale="area means; no clipping; shared within kernel row",
                    color_gamma=a.gamma,
                    color_scale="linear" if a.gamma == 1 else "power",
                    color_palette=a.palette,
                    displayed_quantity=a.quantity,
                    entrywise_square=True,
                    logdet_check="reconstructed blocks match archived penalties",
                    script_sha256=digest(__file__))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
