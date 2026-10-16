#!/usr/bin/env python3
"""Audit a downloaded RBF v23 subset and regenerate the existing paper tables.

Only reads the source artifacts. Does not certify numerical reproducibility or
global optimality. Reports the SGPR self-transplant checks independently.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.provenance import config_fingerprint, release_runtime_matches, SGPR_RELEASE_PACKAGES

COMMIT = "2597598a42bfd857b6028b100c545a0a6d3a3270"
METHODS = ("Exact-GP", "METIS-SGPR", "SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR")
DONORS = METHODS[2:]
PRED = dict(zip(METHODS, ("exact", "block", "sparse", "diagonal", "block", "block")))
FIELDS = ("donor_complete_rmse", "donor_complete_nll", "donor_sparse_nll",
          "transplant_sgpr_rmse", "transplant_sgpr_nll", "delta_sgpr_elbo_n")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    check(len(values) == 10 and all(math.isfinite(x) for x in values), "ten finite values required")
    return st.mean(values), st.stdev(values)


def cell(values, sem=False, digits=3):
    mean, sd = stats(values)
    return f"{mean:.{digits}f} ({sd / math.sqrt(10) if sem else sd:.{digits}f})"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--report", type=Path, default=Path("Report"))
    args = ap.parse_args()
    root, report = args.root.resolve(), args.report.resolve()
    check((root / "EXPECTED_HEAD").read_text().strip() == COMMIT, "wrong source commit")
    main_rows, transplant_rows, null_failures = [], [], []
    by_key, signatures, device_kinds = {}, set(), set()
    check(len(list((root / "runs").rglob("metrics.json"))) == 180, "main count")
    check(len(list((root / "transplant").rglob("metrics.json"))) == 30, "transplant count")
    for m in (128, 256, 512):
        for method in METHODS:
            for seed in range(10):
                p = root / "runs/rbf/kin40k" / f"M{m}" / method / f"seed{seed}/metrics.json"
                d = json.loads(p.read_text())
                c = d["config"]
                for k, v in dict(method=method, M=m, seed=seed, kernel="rbf", dataset="kin40k",
                                 n_train=5000, n_test=2000).items():
                    check(d.get(k) == v and c.get(k) == v, f"{p}: {k}")
                check(d["implementation"] == c["implementation"] == "sgpr-collapsed-sbt-jax-v23", str(p))
                check(c["code_commit"] == COMMIT and c["code_source_dirty"] is False, str(p))
                check(d["config_id"] == config_fingerprint(c), f"{p}: config fingerprint")
                check(d["convergence_passed"] and d["optimizer_convergence_reached"], str(p))
                check(0 <= d["final_grad_max_per_datum"] <= 1e-3, f"{p}: gradient")
                check(0 <= d["convergence_tail_range_obj_n"] <= 1e-4, f"{p}: objective")
                check(1 <= d["steps"] <= d["max_steps"] == 100000, f"{p}: steps")
                check(c["refresh_schedule"] == [15, 15, 20, 25, 99925] and d["block"] == 500, str(p))
                check(release_runtime_matches(d["runtime"], SGPR_RELEASE_PACKAGES), f"{p}: runtime")
                check(d["reported_predictor"] == c["reported_predictor"] == PRED[method], str(p))
                check(c["split_seed"] == seed and c["model_seed"] == seed, f"{p}: seed pairing")
                check(c["data_signature"].startswith("sha256:"), str(p))
                signatures.add(c["data_signature"])
                device_kinds.update(x["kind"] for x in d["runtime"]["jax_devices"])
                with np.load(p.parent / "params.npz") as params:
                    check(all(np.isfinite(params[k]).all() for k in params), f"{p}: parameters")
                by_key[m, method, seed] = d
        for seed in range(10):
            p = root / "transplant" / f"M{m}/rbf/seed{seed}/metrics.json"
            t = json.loads(p.read_text())
            check(t["implementation"] == "sgpr-hyperparameter-transplant-v7", str(p))
            check(t["code"]["commit"] == COMMIT and not t["code"]["source_dirty"], str(p))
            check(release_runtime_matches(t["runtime"], SGPR_RELEASE_PACKAGES), str(p))
            check(len(t["rows"]) == 4 and {x["donor_method"] for x in t["rows"]} == set(DONORS), str(p))
            for d in t["rows"]:
                method = d["donor_method"]
                check(d["M"] == m and d["seed"] == seed and d["kernel"] == "rbf", str(p))
                check(d["transplant_convergence_reached"], str(p))
                check(0 <= d["transplant_final_grad_max_per_datum"] <= 1e-3, str(p))
                check(1 <= d["transplant_steps"] <= d["transplant_max_steps"] == 100000, str(p))
                donor = by_key[m, method, seed]
                check(d["donor_config_id"] == donor["config_id"], f"{p}: donor identity")
                for a, b in (("donor_complete_rmse", "rmse"), ("donor_complete_nll", "nll")):
                    check(abs(d[a] - donor[b]) < 1e-8, f"{p}: donor prediction")
                src = root / "runs/rbf/kin40k" / f"M{m}/{method}/seed{seed}/params.npz"
                with np.load(src) as old, np.load(p.parent / f"params_transplant_{method}.npz") as new:
                    check(set(old.files) == set(new.files), f"{p}: parameter keys")
                    for k in old:
                        check(np.isfinite(new[k]).all(), f"{p}: non-finite parameters")
                        if k != "z":
                            check(np.array_equal(old[k], new[k]), f"{p}: changed frozen {k}")
                if method == "SGPR" and (abs(d["delta_sgpr_elbo_n"]) > 5e-4 or abs(d["delta_sparse_nll"]) > 1e-3):
                    null_failures.append({k: d[k] for k in ("M", "seed", "delta_sgpr_elbo_n", "delta_sparse_nll")})
                transplant_rows.append(d)
    check(len(signatures) == 1, "dataset signature mismatch")
    check(device_kinds == {"NVIDIA A100-SXM4-40GB"}, "heterogeneous prediction hardware")
    main_tex = [r"% Generated from the audited RBF v23 subset.", r"\begin{table*}[t]",
        r"\caption{Kin40k collapsed SGPR with an RBF kernel, \(5000/2000\) train/test points and block size \(b=500\). Entries are mean (standard deviation) over ten seeds. All fits pass the v23 convergence checks. Each row uses its own sparse, diagonal, or fitted-block predictor; Exact-GP uses the exact posterior. \emph{Pred.} is the mean wall time for the full test set on A100-SXM4-40GB GPUs.}",
        r"\label{tab:kin40k-sgpr-details}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Method & Pred. cov. & Obj./N & RMSE & NLL & Pred. (ms) \\"]
    tp_tex = [r"% Generated from the audited RBF v23 subset.", r"\begin{table*}[t]",
        r"\caption{Kin40k RBF hyperparameter transplant, \(5000/2000\) train/test points. Donor kernel and noise parameters are fixed; only inducing locations are refined under the SGPR trace bound, with at most 100000 L-BFGS-B steps. Entries are mean (standard error) over ten seeds. All fits pass the optimization gate. Four of 30 SGPR self-transplant controls exceed the separate invariance tolerances (see text). Lower RMSE/NLL and larger \(\Delta\mathrm{ELBO}/N\) are better.}",
        r"\label{tab:kin40k-sgpr-transplant}", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.94}", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"& \multicolumn{2}{c}{Donor, own predictor} & \multicolumn{1}{c}{Donor as sparse} & \multicolumn{2}{c}{After SGPR transplant} & \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-4}\cmidrule(lr){5-6}",
        r"Donor bound & RMSE & NLL & NLL & RMSE & NLL & \(\Delta\mathrm{ELBO}/N\) \\"]
    tp_summary = []
    for m in (128, 256, 512):
        main_tex += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{RBF, \(M={m}\)}}}} \\"]
        for method in METHODS:
            if method in ("Exact-GP", "SGPR"):
                group = "Dense covariance or dense search" if method == "Exact-GP" else "No dense residual matrix"
                main_tex.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{group}}}}} \\")
            ds = [by_key[m, method, s] for s in range(10)]
            timing = "pred_s_sparse" if method in ("Exact-GP", "SGPR") else "pred_s_bound"
            pred_ms = 1000 * stats([d[timing] for d in ds])[0]
            label = {"exact": "exact", "sparse": "sparse", "diagonal": "diagonal residual", "block": "all fitted blocks"}[PRED[method]]
            cells = [method, label, *[cell([d[k] for d in ds]) for k in ("obj_n", "rmse", "nll")], f"{pred_ms:.1f}"]
            if method == "SBT-SGPR":
                cells = [rf"\textbf{{{v}}}" for v in cells]
            main_tex.append(" & ".join(cells) + r" \\")
            out = dict(kernel="rbf", M=m, method=method, n_seeds=10, pred_ms_mean=pred_ms)
            for k in ("obj_n", "rmse", "nll", "search_build_ms"):
                vals = [d[k] for d in ds]
                mu, sd = (st.mean(vals), st.stdev(vals)) if all(math.isfinite(x) for x in vals) else (None, None)
                out.update({k+"_mean": mu, k+"_std": sd})
            main_rows.append(out)
        tp_tex += [r"\midrule", rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{RBF, \(M={m}\)}}}} \\"]
        for method in DONORS:
            ds = [d for d in transplant_rows if d["M"] == m and d["donor_method"] == method]
            cells = [method, *[cell([d[k] for d in ds], sem=True, digits=4 if k == FIELDS[-1] else 3) for k in FIELDS]]
            if method == "SBT-SGPR":
                cells = [rf"\textbf{{{v}}}" for v in cells]
            tp_tex.append(" & ".join(cells) + r" \\")
            out = dict(kernel="rbf", M=m, donor_method=method, n_seeds=10)
            for k in FIELDS:
                mu, sd = stats([d[k] for d in ds]); out.update({k+"_mean": mu, k+"_sem": sd/math.sqrt(10)})
            tp_summary.append(out)
    build = []
    for m in (128, 256, 512):
        rows = {d["method"]: d for d in main_rows if d["M"] == m}
        build.append(rf"\(M={m}\): METIS {rows['METIS-SGPR']['search_build_ms_mean']:.0f} ms, SBT {rows['SBT-SGPR']['search_build_ms_mean']:.2f} ms")
    main_tex += [r"\midrule", r"\multicolumn{6}{@{}p{0.98\textwidth}@{}}{\textbf{Partition construction:}\enspace " + r";\enspace ".join(build) + r"} \\"]
    for lines, name in ((main_tex, "kin40k_sgpr_main"), (tp_tex, "kin40k_sgpr_transplant")):
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
        (report / "tables" / (name+".tex")).write_text("\n".join(lines))
    write_csv(report / "Results/kin40k_sgpr_main_summary.csv", main_rows)
    write_csv(report / "Results/kin40k_sgpr_transplant_rbf_summary.csv", tp_summary)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.rglob("*")) if p.is_file()}
    manifest = dict(source_commit=COMMIT, local_root=str(root), kernel="rbf", inducing_budgets=[128,256,512],
                    main_runs=180, transplant_experiments=30, transplant_fits=120,
                    artifact_and_recorded_convergence_checks_passed=True,
                    independent_numerical_replay_performed=False, global_optimality_certified=False,
                    self_transplant_checks_passed=not null_failures, self_transplant_failures=null_failures,
                    data_signature=next(iter(signatures)), gpu_kinds=sorted(device_kinds), sha256=hashes)
    (report / "Results/sgpr_rbf_v23_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({k:v for k,v in manifest.items() if k!="sha256"}, indent=2))


if __name__ == "__main__":
    main()
