"""Local Snelson illustration and fixed-state audit; no frozen outputs changed."""
from pathlib import Path
import os
import sys
import json
import hashlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sgpr-mpl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("RESIDUAL_WRITE_REPORT", "0")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import jax.numpy as jnp

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))
from sgpr.residual_illustration import run_residual_structure as illustration
from common.partitions import anchor_order_device


def main():
    out = SIM.parent / "output/pdf/snelson_geometry"
    out.mkdir(parents=True, exist_ok=True)
    x, y, xt, yt = illustration.load_residual_dataset()
    cache = out / "local_fit_params.npz"
    if cache.exists():
        with np.load(cache) as saved:
            params = {key: jnp.asarray(saved[key]) for key in saved.files}
    else:
        z0 = illustration.random_inducing(x, illustration.M, illustration.SEED)
        fit = illustration.run_method("SBT-SGPR", x, y, xt, yt, z0)
        params = fit["params"]
        np.savez_compressed(cache, **{key: np.asarray(value) for key, value in params.items()})
    ell, sf2, noise, z = illustration.theta_numpy(params)
    order = np.asarray(anchor_order_device(jnp.asarray(x), jnp.asarray(z),
                                          jnp.asarray(ell), kernel_name="rbf"))
    n, b = len(x), illustration.BLOCK
    groups = [order[i:i+b] for i in range(0, n, b)]
    labels = np.empty(n, dtype=int)
    for i, group in enumerate(groups):
        labels[group] = i
    # The theorem uses the exact Schur complement; audit without jitter.
    kernel = lambda a, c: sf2*np.exp(-.5*np.sum(((a[:,None]-c[None,:])/ell)**2, axis=-1))
    kxz, kzz, kxx = kernel(x,z), kernel(z,z), kernel(x,x)
    d = kxx-kxz @ np.linalg.solve(kzz,kxz.T)
    d = (d+d.T)/2
    mass = d*d
    np.fill_diagonal(mass,0)
    retained = float(mass[labels[:,None] == labels[None,:]].sum()/mass.sum())
    penalty = lambda blocks: sum(np.linalg.slogdet(np.eye(len(g))+d[np.ix_(g,g)]/noise)[1] for g in blocks)
    selected_penalty = penalty(groups)
    rng = np.random.default_rng(23)
    random_penalties, certificates = [], []
    kappa = 1+max(np.linalg.eigvalsh(d[np.ix_(g,g)])[-1] for g in groups)/noise
    inside = labels[:,None] == labels[None,:]
    for _ in range(100):
        shuffle = rng.permutation(n)
        random_groups = [shuffle[i:i+b] for i in range(0,n,b)]
        old = np.empty(n, dtype=int)
        for j, group in enumerate(random_groups):
            old[group] = j
        old_inside = old[:,None] == old[None,:]
        gain = mass[inside & ~old_inside].sum()
        loss = mass[~inside & old_inside].sum()
        certificates.append(float((gain/kappa**2-loss)/(4*noise**2)))
        random_penalties.append(penalty(random_groups))
    parent = kxz.argmax(axis=1)
    anchor_sort = np.argsort(z[:,0])
    rank = np.empty(len(z),dtype=int)
    rank[anchor_sort] = np.arange(len(z))
    closest_pairs = np.sort(rank[np.argsort(-kxz,axis=1)[:,:2]],axis=1)
    pair_counts = [int(np.all(closest_pairs == [m,m+1],axis=1).sum()) for m in range(len(z)-1)]
    offdiag = ~np.eye(n,dtype=bool)
    actual_u = float(np.abs(d)[inside & offdiag].min())
    actual_v = float(np.abs(d)[~inside].max())
    elbo_gains = .5*(np.array(random_penalties)-selected_penalty)
    report = {"status": "local illustrative rerun, not a frozen campaign result",
              "n": n, "m": len(z), "b": b, "seed": illustration.SEED,
              "lbfgs_max_steps": illustration.TRAIN_STEPS, "refresh_phases": illustration.SBT_REFRESH_SCHEDULE,
              "source_data_sha256": hashlib.sha256((SIM/"dataset/snelson.csv").read_bytes()).hexdigest(),
              "selector": "common.partitions.anchor_order_device", "kernel": "rbf",
              "lengthscale": ell.tolist(), "signal_variance": sf2, "noise_variance": noise,
              "anchor_cell_sizes": np.bincount(rank[parent],minlength=len(z)).tolist(),
              "nearest_pair_group_sizes": pair_counts, "theorem1_cardinality_holds": n == (len(z)-1)*b,
              "within_residual_min_abs": actual_u, "cross_residual_max_abs": actual_v,
              "kappa": float(kappa), "uniform_separation_holds": bool(actual_u > kappa*actual_v),
              "retained_mass_selected": retained, "retained_mass_random_expectation": (b-1)/(n-1),
              "selected_logdet": selected_penalty, "random_logdet_mean": float(np.mean(random_penalties)),
              "random_comparisons":100, "positive_elbo_gains":int(np.sum(elbo_gains>0)),
              "elbo_gain_per_datum_mean":float(elbo_gains.mean()/n),
              "elbo_gain_per_datum_min":float(elbo_gains.min()/n),
              "exchange_condition_certified":int(np.sum(np.array(certificates)>0))}
    (out/"audit.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(out/"figure_data.npz", x=x,y=y,z=z,order=order,labels=labels,parent=parent,residual=d)

    plt.rcParams.update({"font.family":"sans-serif", "font.size":7,
                         "mathtext.fontset":"cm", "axes.linewidth":.65,"pdf.fonttype":42})
    palette = np.array(["#ff2d55","#00d9ff","#ff8c00","#d100ff","#64ff4a"])
    fig, axs = plt.subplots(1,2,figsize=(3.55,1.70))
    ix = np.argsort(x[:,0])
    zy = np.interp(z[:,0],x[ix,0],y[ix])
    for ax, point_labels, title in zip(axs,[rank[parent],labels],["Nearest-anchor cells","Selected blocks"]):
        ax.scatter(x[:,0],y,s=6,c=palette[point_labels],edgecolors="#202020",linewidths=.13,alpha=.9)
        ax.scatter(z[:,0],zy,marker="*",s=36,c="black",edgecolors="white",linewidths=.3,zorder=5)
        ax.set_title(title,fontsize=8.4,pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Input $x$",fontsize=7.6,labelpad=3)
    axs[0].set_ylabel("Response $y$",fontsize=7.6,labelpad=3)
    fig.subplots_adjust(left=.07,right=.995,bottom=.22,top=.85,wspace=.10)
    fig.savefig(out/"snelson_anchor_partition.pdf")
    fig.savefig(out/"snelson_anchor_partition.png",dpi=260)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
