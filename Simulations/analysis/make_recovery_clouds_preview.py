"""Higher-dimensional, non-collinear example satisfying geometric recovery."""
from pathlib import Path
import argparse
import json
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sgpr-mpl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SIM))
from common.partitions import anchor_order_device


def certify(x,z,b):
    kernel = lambda a,c: np.exp(-.5*np.sum((a[:,None]-c[None,:])**2,axis=-1))
    xx,xz,zz = kernel(x,x),kernel(x,z),kernel(z,z)
    labels = np.repeat(np.arange(len(z)-1),b)
    # Correlations preserve strict distance comparisons without saturation at 2.
    pair_margin = min(float((xz[labels==m][:,[m,m+1]].min(axis=1)-
                     xz[labels==m][:,[j for j in range(len(z)) if j not in (m,m+1)]].max(axis=1)).min())
                      for m in range(len(z)-1))
    chain_margin = min(float(zz[m,m+1]-zz[m,j]) for m in range(len(z)-2) for j in range(m+2,len(z)))
    off = labels[:,None] != labels[None,:]
    h2 = float((2-2*xx)[~off].max())
    delta = float(max(xx[off].max(),xz.max(),(zz-np.eye(len(z))).max()))
    zeta = len(z)*delta**2/(1-(len(z)-1)*delta)
    u,v = 1-h2/2-zeta,delta+zeta
    margin = u-(1+b)*v
    order = np.asarray(anchor_order_device(jnp.asarray(x),jnp.asarray(z),jnp.ones(x.shape[1]),kernel_name="rbf"))
    recovered = all(set(order[m*b:(m+1)*b])==set(np.where(labels==m)[0]) for m in range(len(z)-1))
    d = xx-xz@np.linalg.solve(zz,xz.T)
    actual_u = float(d[(~off)&(~np.eye(len(x),dtype=bool))].min())
    actual_v = float(np.abs(d[off]).max())
    kappa = 1+max(np.linalg.eigvalsh(d[m*b:(m+1)*b,m*b:(m+1)*b])[-1] for m in range(len(z)-1))
    assert pair_margin>0 and chain_margin>0 and margin>0 and recovered
    return {"dimension":x.shape[1],"n":len(x),"m":len(z),"b":b,
            "kernel":"rbf","lengthscales":[1.]*x.shape[1],"signal_variance":1.,"noise_variance":1.,
            "pair_correlation_margin":pair_margin,"chain_correlation_margin":chain_margin,
            "h_squared":h2,"delta":delta,"zeta_bound":zeta,"sufficient_margin":margin,
            "actual_u":actual_u,"actual_v":actual_v,"actual_kappa":float(kappa),
            "actual_residual_margin":float(actual_u-kappa*actual_v),
            "recovery_passed":bool(recovered),"order":order.tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--write-report",action="store_true")
    parser.add_argument("--out",type=Path,default=SIM.parent/"output/pdf/recovery_clouds",
                        help="directory for the preview figure, audit.json and examples.npz")
    args=parser.parse_args()
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(41)
    z = np.array([[0.,0.],[6.5,0.],[11.7,3.9],[18.2,3.9],[23.4,0.]])
    b = 12
    centers = (z[:-1]+z[1:])/2
    clouds=[]
    for center in centers:
        angles = rng.uniform(0,2*np.pi,b)
        radii = .55*np.sqrt(rng.uniform(0,1,b))
        clouds.append(center+np.column_stack([radii*np.cos(angles),radii*np.sin(angles)]))
    x=np.concatenate(clouds)
    report=certify(x,z,b)
    parents=np.argmax(np.exp(-.5*np.sum((x[:,None]-z[None,:])**2,axis=-1)),axis=1)
    assert all(set(parents[m*b:(m+1)*b])=={m,m+1} for m in range(len(z)-1))
    # A nondegenerate 5D perturbation, checked in the full 5D kernel geometry.
    z5=np.column_stack([z,rng.normal(0,.06,(len(z),3))])
    x5=np.column_stack([x,np.repeat((z5[:-1,2:]+z5[1:,2:])/2,b,axis=0)+rng.normal(0,.025,(len(x),3))])
    report5=certify(x5,z5,b)
    (out/"audit.json").write_text(json.dumps({"2d":report,"5d":report5,
        "scope":"Constructed fixed-state examples; no training or campaign claims."},indent=2)+"\n")
    np.savez_compressed(out/"examples.npz",x=x,z=z,x5=x5,z5=z5)
    plt.rcParams.update({"font.family":"sans-serif","font.size":8,"mathtext.fontset":"cm",
                         "axes.linewidth":.65,"pdf.fonttype":42})
    colors=["#ff2d55","#00d9ff","#ff8c00","#d100ff"]
    fig=plt.figure(figsize=(4.4,1.9))
    ax=fig.add_axes([.025,.34,.95,.53])
    ax.plot(z[:,0],z[:,1],c="#969696",lw=.75)
    for m,(center,color,cloud) in enumerate(zip(centers,colors,clouds)):
        direction=z[m+1]-z[m]
        normal=np.array([-direction[1],direction[0]])/np.linalg.norm(direction)
        line=center+np.array([[-1.], [1.]])*normal
        ax.plot(line[:,0],line[:,1],color="#777777",ls=(0,(2,2)),lw=.65)
        ax.add_patch(Ellipse(center,1.55,1.55,facecolor=color,edgecolor="none",alpha=.07))
        ax.scatter(cloud[:,0],cloud[:,1],s=13,c=color,edgecolors="#202020",linewidths=.2,zorder=3)
        ax.text(center[0],center[1]+1.02,rf"$C_{m+1}$",ha="center",fontsize=9)
    ax.scatter(z[:,0],z[:,1],marker="*",s=55,c="black",edgecolors="white",linewidths=.35,zorder=5)
    for m,point in enumerate(z):
        ax.text(point[0],point[1]-.65,rf"$z_{m+1}$",ha="center",fontsize=8)
    ax.set(xlim=(-1,24.4),ylim=(-1.35,6.0),xticks=[],yticks=[])
    ax.set_title("Geometric recovery in two dimensions",fontsize=9,pad=3)
    row=fig.add_axes([.075,.065,.90,.17])
    for m,color in enumerate(colors):
        row.add_patch(Rectangle((m*b-.45,-.35),b-.1,.7,fill=False,edgecolor=color,lw=.9))
        row.scatter(np.arange(m*b,(m+1)*b),np.zeros(b),s=9,c=color,edgecolors="#202020",linewidths=.15)
    row.text(-1.5,0,r"$\pi$",ha="right",va="center",fontsize=10)
    row.set(xlim=(-2,len(x)),ylim=(-.5,.5))
    row.axis("off")
    fig.savefig(out/"geometric_recovery_clouds.pdf")
    fig.savefig(out/"geometric_recovery_clouds.png",dpi=260)
    if args.write_report:
        root=SIM.parent/"Report"
        fig.savefig(root/"figures/geometric_recovery_main.pdf")
        fig.savefig(root/"figures/geometric_recovery_main.png",dpi=260)
        (root/"Results/geometric_recovery_main.json").write_text(json.dumps(
            {"2d":report,"5d":report5,"figure_dimension":2,"generator":str(Path(__file__).relative_to(SIM.parent))},indent=2)+"\n")
    plt.close(fig)
    print(json.dumps({"2d":report,"5d":report5},indent=2))


if __name__=="__main__":
    main()
