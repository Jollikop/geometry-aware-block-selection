#!/usr/bin/env python3
"""Local, training-only SGPR pilot with fixed-state partition certificates.

Uses the project loss, selector and SciPy training engine unchanged. Outputs
are exploratory diagnostics, never a replacement for a frozen campaign.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/sgpr-refresh-audit-mpl")
SIM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM))

import jax
import jax.numpy as jnp
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from common import data_split, kernels, partitions, terms
from common.provenance import sha256_file
from sgpr import initialization, jax_collapsed, protocol

KUU_JITTER = 1e-8
BLOCK_JITTER = 1e-9


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def labels_from_blocks(blocks, n):
    terms.assert_partition(blocks, n)
    labels = np.empty(n, dtype=int)
    for k, block in enumerate(blocks):
        labels[block] = k
    return labels


def partition_quantities(d, blocks, s2):
    """Unjittered theoretical penalties; implementation jitter also recorded."""
    n = len(d)
    labels = labels_from_blocks(blocks, n)
    logdet = logdet_engine = inside = 0.0
    eigenmax = 0.0
    eigenmin = float("inf")
    within_min = float("inf")
    for block in blocks:
        db = d[np.ix_(block, block)]
        eig = linalg.eigvalsh(db, check_finite=False)
        eigenmax = max(eigenmax, float(eig[-1]))
        eigenmin = min(eigenmin, float(eig[0]))
        if eig[0] < -1e-8 * max(1.0, float(eig[-1])):
            raise ValueError("Residual block is materially non-PSD")
        # No eigenvalue clipping: significant inconsistencies must be visible.
        logdet += float(np.log1p(eig / s2).sum())
        logdet_engine += float(np.log1p(eig / s2 + BLOCK_JITTER).sum())
        mask = ~np.eye(len(block), dtype=bool)
        vals = db[mask]
        inside += float(vals @ vals)
        if len(vals):
            within_min = min(within_min, float(np.min(np.abs(vals))))
    cross_max = 0.0
    off_mass = 0.0
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        rows = d[start:stop].copy()
        rows[np.arange(stop-start), np.arange(start, stop)] = 0.0
        off_mass += float(np.einsum("ij,ij->", rows, rows))
        cross = labels[start:stop, None] != labels[None, :]
        if np.any(cross):
            cross_max = max(cross_max, float(np.max(np.abs(rows[cross]))))
    return {
        "J": logdet, "J_engine": logdet_engine,
        "inside_mass": inside, "off_diagonal_mass": off_mass,
        "retained_fraction": inside / off_mass if off_mass > 0 else 0.0,
        "kappa": 1.0 + eigenmax / s2, "min_block_eigenvalue": eigenmin,
        "u": within_min if np.isfinite(within_min) else 0.0,
        "v": cross_max,
    }


def exchange_quantities(d, p, q, s2, p_stats=None, q_stats=None):
    n = len(d)
    lp, lq = labels_from_blocks(p, n), labels_from_blocks(q, n)
    g_plus = g_minus = 0.0
    n_plus = n_minus = 0
    # Every interaction appears twice, as required by the ordered-pair bound.
    for block in q:
        db = d[np.ix_(block, block)]
        selected = lp[block, None] != lp[None, block]
        g_plus += float(np.sum(db[selected] ** 2))
        n_plus += int(selected.sum())
    for block in p:
        db = d[np.ix_(block, block)]
        selected = lq[block, None] != lq[None, block]
        g_minus += float(np.sum(db[selected] ** 2))
        n_minus += int(selected.sum())
    ps = partition_quantities(d, p, s2) if p_stats is None else p_stats
    qs = partition_quantities(d, q, s2) if q_stats is None else q_stats
    delta = 0.5 * (ps["J"] - qs["J"])
    delta_engine = 0.5 * (ps["J_engine"] - qs["J_engine"])
    lower = (g_plus / qs["kappa"]**2 - g_minus) / (4 * s2**2)
    upper = (g_plus - g_minus / ps["kappa"]**2) / (4 * s2**2)
    tol = 1e-8 * max(1.0, abs(delta), abs(lower), abs(upper))
    margin = g_plus - qs["kappa"]**2 * g_minus
    return {
        "G_plus": g_plus, "G_minus": g_minus,
        "gained_ordered_pairs": n_plus, "lost_ordered_pairs": n_minus,
        "kappa_Q": qs["kappa"], "kappa_P": ps["kappa"],
        "exchange_margin": margin,
        "exchange_ratio": g_plus / (qs["kappa"]**2 * g_minus) if g_minus > 0 else None,
        "lower_bound_per_datum": lower / n,
        "upper_bound_per_datum": upper / n,
        "delta_elbo_per_datum": delta / n,
        "delta_engine_penalty_per_datum": delta_engine / n,
        "bound_holds_numerically": bool(lower-tol <= delta <= upper+tol),
        "sufficient_condition_passed": bool(margin >= 0),
        "strict_improvement_observed": bool(delta / n > 1e-10),
        "same_partition": bool(n_plus == 0 and n_minus == 0),
        "retained_P": ps["retained_fraction"], "retained_Q": qs["retained_fraction"],
    }


@jax.jit
def residual_rows(x_rows, x, params, a_rows, a):
    k = kernels.kernel(x_rows, x, jnp.exp(params["log_ell"]),
                       jnp.exp(params["log_sf2"]), "rbf")
    return k - a_rows.T @ a


def residual_matrix(x, params):
    ell, sf2 = jnp.exp(params["log_ell"]), jnp.exp(params["log_sf2"])
    z = params["z"]
    kuu = np.asarray(kernels.kernel(z, z, ell, sf2, "rbf"))
    kuf = np.asarray(kernels.kernel(z, x, ell, sf2, "rbf"))
    chol = linalg.cholesky(kuu + KUU_JITTER*np.eye(len(z)), lower=True)
    a = jnp.asarray(linalg.solve_triangular(chol, kuf, lower=True))
    d = np.empty((len(x), len(x)), dtype=np.float64)
    for start in range(0, len(x), 256):
        stop = min(start + 256, len(x))
        d[start:stop] = np.asarray(residual_rows(x[start:stop], x, params, a[:, start:stop], a))
    return 0.5 * (d + d.T)


def packed_host(packed):
    return {key: np.asarray(value) for key, value in zip(("coeffs", "idx", "mask"), packed)}


class RefreshAudit:
    def __init__(self, x, loss, out, block_size, random_count, seed):
        self.x, self.loss, self.out, self.b = x, loss, out, block_size
        self.step, self.records, self.events, self.audit_s = 0, [], [], 0.0
        self.random = []
        for k in range(random_count):
            rng = np.random.default_rng(100000 + seed*1000 + k)
            order = rng.permutation(len(x))
            self.random.append([order[i:i+block_size] for i in range(0,len(x),block_size)])
        self.current = None

    def select(self, params):
        started = time.perf_counter()
        result = partitions.sbt_terms_device(self.x, params, self.b, "rbf", split_mode="boundary")
        jax.block_until_ready(result)
        return result, time.perf_counter()-started

    def audit(self, params, p_terms, q_terms, event, selection_s):
        started = time.perf_counter()
        print(f"AUDIT START event={event} step={self.step}", flush=True)
        destination = self.out / f"state_{self.step:04d}_{event}"
        destination.mkdir()
        np.savez_compressed(destination/"params.npz", **{k:np.asarray(v) for k,v in params.items()})
        np.savez_compressed(destination/"incumbent_blocks.npz", **packed_host(p_terms))
        np.savez_compressed(destination/"candidate_blocks.npz", **packed_host(q_terms))
        p, q = terms.unpack_terms(p_terms), terms.unpack_terms(q_terms)
        d = residual_matrix(self.x, params)
        s2 = float(jnp.exp(params["log_s2"]))
        qs = partition_quantities(d, q, s2)
        ps = partition_quantities(d, p, s2)
        parent, _, _, _, chain = partitions._anchor_details_device(
            self.x, params["z"], jnp.exp(params["log_ell"]), "rbf")
        parent = np.asarray(parent)
        counts = np.bincount(parent, minlength=len(params["z"]))
        qlabels = labels_from_blocks(q, len(d))
        cell_blocks = [np.flatnonzero(parent == a) for a in np.flatnonzero(counts)]
        cell_sizes_pass = bool(np.all(counts[counts > 0] == self.b))
        cells_recovered = bool(cell_sizes_pass and all(len(np.unique(qlabels[c])) == 1 for c in cell_blocks))
        sep_margin = qs["u"] - qs["kappa"]*qs["v"]
        separation_pass = bool(qs["u"] > 0 and sep_margin >= 0)
        state = {
            "event": event, "step": self.step, "sigma2": s2,
            "u": qs["u"], "v": qs["v"], "kappa_Q": qs["kappa"],
            "separation_margin": sep_margin,
            "separation_ratio": qs["u"]/(qs["kappa"]*qs["v"]) if qs["v"]>0 else None,
            "uniform_separation_passed": separation_pass,
            "nonempty_cells": int(np.count_nonzero(counts)),
            "cell_size_min": int(counts[counts>0].min()), "cell_size_max": int(counts.max()),
            "all_nonempty_cells_size_b": cell_sizes_pass,
            "candidate_recovers_cells": cells_recovered,
            "theorem1_hypotheses_passed": bool(cells_recovered and separation_pass),
            "min_candidate_block_eigenvalue": qs["min_block_eigenvalue"],
            "min_residual_diagonal": float(np.min(np.diag(d))),
            "selection_s": selection_s,
            "grouped_anchor_extension": "not tested",
        }
        np.savez_compressed(destination/"anchor_geometry.npz", parent=parent, chain=np.asarray(chain), cell_sizes=counts)
        observations = []
        for name, blocks, stats in [("incumbent",p,ps)]+[(f"random_{i}",b,None) for i,b in enumerate(self.random)]:
            row = {"event": event, "step": self.step, "comparison": name,
                   **exchange_quantities(d,blocks,q,s2,p_stats=stats,q_stats=qs)}
            if not row["bound_holds_numerically"]:
                raise AssertionError(f"Partition-exchange inequality violated: {row}")
            observations.append(row)
        engine_delta = float(self.loss(params,*p_terms)-self.loss(params,*q_terms))/len(d)
        error = abs(engine_delta-observations[0]["delta_engine_penalty_per_datum"])
        if error > 1e-7:
            raise AssertionError(f"Reconstructed penalty disagrees with project objective: {error}")
        state.update(engine_delta_elbo_per_datum=engine_delta, reconstruction_error_per_datum=error,
                     audit_s=time.perf_counter()-started)
        self.audit_s += state["audit_s"]
        self.records.extend(observations)
        self.events.append(state)
        write_json(destination/"audit.json", {"state":state,"comparisons":observations})
        write_json(self.out/"events.json", self.events)
        with (self.out/"comparisons.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(self.records[0]))
            writer.writeheader(); writer.writerows(self.records)
        rand = observations[1:]
        print(f"AUDIT DONE step={self.step} refresh_delta/N={engine_delta:+.6g} "
              f"lower/N={observations[0]['lower_bound_per_datum']:+.6g} "
              f"certificate={observations[0]['sufficient_condition_passed']} "
              f"random_wins={sum(r['strict_improvement_observed'] for r in rand)}/{len(rand)} "
              f"u={qs['u']:.3e} v={qs['v']:.3e} kappa={qs['kappa']:.3e} "
              f"audit_s={state['audit_s']:.1f}", flush=True)

    def refresh(self, params):
        candidate, selection_s = self.select(params)
        self.audit(params,self.current,candidate,"refresh",selection_s)
        self.current = candidate
        return candidate

    def phase_done(self, step, params, value, ms, live_terms):
        self.step = int(step)
        self.current = live_terms
        np.savez_compressed(self.out/f"phase_end_{step:04d}.npz", **{k:np.asarray(v) for k,v in params.items()})
        print(f"TRAINING step={step} loss/N={value/len(self.x):+.8f} average_ms_per_step={ms:.1f}",flush=True)


def make_report(out, config, events, rows, info):
    refreshes = [r for r in rows if r['event']=='refresh' and r['comparison']=='incumbent']
    lines = ["# Local SGPR refresh audit", "",
             f"RBF Kin40k; N={config['n_train']}, M={config['m']}, b={config['block_size']}, seed={config['seed']}.",
             f"New greedy-variance initialization; {info['actual_steps']} accepted L-BFGS iterations; no EM warm-up.",
             "Exploratory training diagnostic, not a converged or frozen paper result.", "",
             "| Step | Actual refresh ELBO gain / N | Proposition 2 lower bound / N | Certificate | Random beaten | u | v | kappa |", 
             "|---:|---:|---:|:---:|:---:|---:|---:|---:|"]
    for event in events:
        records=[r for r in rows if r['event']==event['event'] and r['step']==event['step']]
        r=records[0]; rand=records[1:]
        lines.append(f"| {r['step']} ({r['event']}) | {r['delta_elbo_per_datum']:+.6g} | {r['lower_bound_per_datum']:+.6g} | {r['sufficient_condition_passed']} | {sum(t['strict_improvement_observed'] for t in rand)}/{len(rand)} | {event['u']:.3e} | {event['v']:.3e} | {event['kappa_Q']:.3e} |")
    lines += ["", "Only `refresh` rows are applied training refreshes; initial and final states are diagnostic comparisons.",
              "Theorem 1 requires both balanced anchor-cell recovery and uniform residual separation. Full pair extrema are evaluated; no sampling/percentile substitution.",
              "The grouped-anchor recovery extension is not tested.",
              "D uses the same Kuu + 1e-8 I as training. Proposition 2 applies to this PSD residual; exact anchor-zero claims are only approximate with jitter.",
              "The theoretical comparison omits block jitter; the separate engine comparison includes the project's 1e-9 I and is cross-checked against the actual loss.",
              "Random partitions use fixed independent seeds shared across audit states; they do not affect training or acceptance.",
              "Every proposed refresh is applied regardless of the measured ELBO or certificate.", "",
              f"Improving actual refreshes: {sum(r['strict_improvement_observed'] for r in refreshes)}/{len(refreshes)}.",
              f"Nonnegative sufficient certificates at actual refreshes: {sum(r['sufficient_condition_passed'] for r in refreshes)}/{len(refreshes)}.",
              f"All computed Proposition 2 inequalities respected within numerical tolerance: {all(r['bound_holds_numerically'] for r in rows)}."]
    (out/"report.md").write_text("\n".join(lines)+"\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--n-train",type=int,default=5000)
    ap.add_argument("--n-test",type=int,default=2000)
    ap.add_argument("--m",type=int,default=512)
    ap.add_argument("--block-size",type=int,default=500)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--random-partitions",type=int,default=5)
    ap.add_argument("--phases",default="15,15,20,25,125")
    ap.add_argument("--blas-threads",type=int,default=4)
    args=ap.parse_args()
    phases=[int(v) for v in args.phases.split(',')]
    if any(v<=0 for v in phases) or args.n_train%args.block_size:
        raise ValueError("Positive phase lengths and equal-size blocks are required")
    args.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    dataset=SIM/"dataset/kin40k.npz"
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(phases=phases, steps=sum(phases), kernel="rbf", device=str(jax.devices()),
                  jax_version=jax.__version__,numpy_version=np.__version__,
                  initialization="greedy-variance", warmup_cycles=0,
                  dataset_sha256=sha256_file(dataset), source_sha256={})
    for p in [Path(__file__), SIM/"sgpr/jax_collapsed.py",SIM/"sgpr/protocol.py",SIM/"sgpr/initialization.py",
              SIM/"common/partitions.py",SIM/"common/kernels.py",SIM/"common/data_split.py",SIM/"common/terms.py"]:
        config["source_sha256"][str(p.relative_to(SIM))]=sha256_file(p)
    write_json(args.out/"config.json",config)
    print(f"CONFIG {json.dumps(config)}",flush=True)
    with threadpool_limits(limits=args.blas_threads):
        x,y,_,_,metadata=data_split.load_npz(dataset,args.seed,args.n_train,args.n_test)
        write_json(args.out/"data_metadata.json",metadata)
        np.savez_compressed(args.out/"training_data.npz",x=np.asarray(x),y=np.asarray(y))
        params=jax_collapsed.init_params(x,x[:args.m],"rbf")
        inds=initialization.inducing_indices(x,args.m,params,kernel_name="rbf",seed=args.seed,
                                            strategy="greedy-variance",nested_max_m=args.m)
        params["z"]=x[inds]
        loss=jax_collapsed.make_loss(x,y,kernel_name="rbf")
        auditor=RefreshAudit(x,loss,args.out,args.block_size,args.random_partitions,args.seed)
        initial, selection_s=auditor.select(params)
        auditor.current=initial
        auditor.audit(params,initial,initial,"initial",selection_s)
        def refresh(p):
            return auditor.refresh(p)
        refresh._host_refresh=True
        print("TRAINING compile/start",flush=True)
        params,curve,info=jax_collapsed.train_collapsed(
            loss,params,terms0=initial,term_fn=refresh,optimizer="lbfgs",steps=sum(phases),
            refresh_schedule=phases,train_z=True,log_every=1,log_cb=auditor.phase_done,
            convergence_abs_tol=None,objective_scale=args.n_train)
        final_terms=info.pop("final_terms")
        np.savez_compressed(args.out/"final_params.npz",**{k:np.asarray(v) for k,v in params.items()})
        np.savez_compressed(args.out/"final_blocks.npz",**packed_host(final_terms))
        np.savez_compressed(args.out/"curve.npz",step=curve[:,0],loss=curve[:,1])
        candidate,selection_s=auditor.select(params)
        auditor.audit(params,final_terms,candidate,"final_prospective",selection_s)
        info.update(total_wall_s=time.perf_counter()-started,audit_s=auditor.audit_s,
                    release_ready=False,convergence_claimed=False,
                    timing_note="engine search_s includes audit callbacks; use per-event selection_s for selector timings")
        write_json(args.out/"training_summary.json",info)
        make_report(args.out,config,auditor.events,auditor.records,info)
        print(f"COMPLETE {args.out} elapsed_s={info['total_wall_s']:.1f}",flush=True)


if __name__=="__main__":
    main()
