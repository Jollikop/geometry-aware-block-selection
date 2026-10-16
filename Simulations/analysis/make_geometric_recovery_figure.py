"""Reproducible RBF illustration of Theorem 1, using the production selector."""
from pathlib import Path
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sgpr-mpl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import jax.numpy as jnp

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from common.partitions import anchor_order_device


def rbf(x, y):
    return np.exp(-0.5 * np.sum((x[:, None] - y[None, :]) ** 2, axis=-1))


def main():
    root = SIM.parent / "Report"
    z = np.array([[0., 0.], [6., 0.], [10.8, 3.6], [16.8, 3.6]])
    b = 6
    offsets = np.array([[-.28, -.09], [-.17, .13], [-.07, -.15],
                        [.07, .15], [.17, -.13], [.28, .09]])
    offsets *= 1.6
    x = np.concatenate([(z[m] + z[m+1]) / 2 + offsets for m in range(3)])
    labels = np.repeat(np.arange(3), b)
    kxz, kzz, kxx = rbf(x, z), rbf(z, z), rbf(x, x)
    dist = 2 - 2*kxz
    parent = dist.argmin(axis=1)
    order = np.asarray(anchor_order_device(jnp.asarray(x), jnp.asarray(z),
                       jnp.ones(2), kernel_name="rbf", split_mode="boundary"))
    assert all(set(order[m*b:(m+1)*b]) == set(np.where(labels == m)[0]) for m in range(3))
    margins = []
    for m in range(3):
        local_dist = dist[labels == m]
        outside = [j for j in range(4) if j not in (m, m+1)]
        margin = local_dist[:, outside].min(axis=1) - local_dist[:, [m, m+1]].max(axis=1)
        margins.append(float(margin.min()))
    assert min(margins) > 0
    for m in range(2):
        assert all(kzz[m, m+1] > kzz[m, j] for j in range(m+2, 4))
    assert all(set(parent[labels == m]) == {m, m+1} for m in range(3))
    off = labels[:, None] != labels[None, :]
    h2 = float((2 - 2*kxx)[~off].max())
    delta = float(max(kxx[off].max(), kxz.max(), (kzz-np.eye(4)).max()))
    zeta = 4*delta**2/(1-3*delta)
    noise = 1.0
    u, v = 1-h2/2-zeta, delta+zeta
    certificate_margin = u-(1+b/noise)*v
    assert certificate_margin > 0
    residual = kxx-kxz @ np.linalg.solve(kzz, kxz.T)
    within_off = ~off & ~np.eye(len(x), dtype=bool)
    actual_u = float(residual[within_off].min())
    actual_v = float(np.abs(residual[off]).max())
    kappa = 1 + max(np.linalg.eigvalsh(residual[m*b:(m+1)*b, m*b:(m+1)*b])[-1] for m in range(3))/noise

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.0,
                         "mathtext.fontset": "cm", "axes.linewidth": .65,
                         "pdf.fonttype": 42})
    colors = ["#ff2d55", "#00d9ff", "#ff8c00"]
    fig = plt.figure(figsize=(3.3, 1.45))
    ax = fig.add_axes([.025, .34, .95, .53])
    ax.plot(z[:, 0], z[:, 1], color="#a0a0a0", lw=.7, zorder=0)
    for m, color in enumerate(colors):
        midpoint = (z[m]+z[m+1])/2
        direction = (z[m+1]-z[m])/np.linalg.norm(z[m+1]-z[m])
        normal = np.array([-direction[1], direction[0]])
        ends = midpoint + np.array([[-.9], [.9]])*normal
        ax.plot(ends[:, 0], ends[:, 1], color="#808080", ls=(0, (2, 2)), lw=.65)
        ax.scatter(x[labels == m, 0], x[labels == m, 1], s=16, color=color,
                   edgecolors="#202020", linewidths=.22, zorder=3)
        ax.text(midpoint[0], midpoint[1]+1.12, rf"$C_{m+1}$", ha="center", fontsize=8)
    ax.scatter(z[:, 0], z[:, 1], marker="*", s=48, color="black",
               edgecolors="white", linewidths=.3, zorder=4)
    for m, point in enumerate(z):
        ax.text(point[0], point[1]-.67, rf"$z_{m+1}$", ha="center", fontsize=7.5)
    ax.set(xlim=(-.9,17.7), ylim=(-1.4,5.4), xticks=[], yticks=[])
    ax.set_title("Clusters across nearest-anchor cells", fontsize=8.4, pad=3)
    row = fig.add_axes([.12, .055, .86, .18])
    for m, color in enumerate(colors):
        row.add_patch(Rectangle((m*b-.42, -.33), b-.16, .66, fill=False,
                               edgecolor=color, linewidth=.85))
    row.scatter(np.arange(len(x)), np.zeros(len(x)),
                c=[colors[label] for label in labels[order]], s=13,
                edgecolors="#202020", linewidths=.2)
    row.text(-1.0, 0, r"$\pi$", ha="right", va="center", fontsize=9)
    row.set(xlim=(-1,18), ylim=(-.5,.5))
    row.axis("off")
    stem = root / "figures/geometric_recovery_main"
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    plt.close(fig)
    report = {"kernel": "rbf", "lengthscale": 1., "signal_variance": 1.,
              "noise_variance": noise, "block_size": b, "inducing_inputs": z.tolist(),
              "observations": x.tolist(), "reference_groups": labels.tolist(),
              "selector": "common.partitions.anchor_order_device", "order": order.tolist(),
              "nearest_anchor": parent.tolist(), "recovery_passed": True,
              "nearest_pair_strict_margins": margins, "h_squared": h2,
              "delta": delta, "zeta_bound": zeta, "sufficient_margin": certificate_margin,
              "actual_u": actual_u, "actual_v": actual_v, "actual_kappa": float(kappa),
              "actual_separation_margin": float(actual_u-kappa*actual_v)}
    (root / "Results/geometric_recovery_main.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ["recovery_passed", "h_squared", "delta", "sufficient_margin", "actual_separation_margin"]}, indent=2))


if __name__ == "__main__":
    main()
