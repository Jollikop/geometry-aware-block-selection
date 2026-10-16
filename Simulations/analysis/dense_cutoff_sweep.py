#!/usr/bin/env python3
"""Dense inverse-root cutoff sensitivity, with one eigendecomposition per donor.

The formula, observation noise and rejection thresholds match the historical
dense evaluator. All output is separate; checkpoint and reference hashes are
checked before and after evaluation. No optimizer or training code is called.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import real_predictor_campaign as real
import weighted_residual_campaign as campaign

SCALES = (1.0, 10.0, 100.0, 1e4, 1e6)


def sweep(components, blocks, method, *, test_chunk=32, scales=SCALES):
    import jax
    import jax.numpy as jnp

    start = time.perf_counter()
    x, a = components["x"], components["a_train"]
    residual = components["kernel_fn"](x, x) - a.T @ a
    residual = .5 * (residual + residual.T)
    eigenvalues, vec = jnp.linalg.eigh(residual)
    spectrum = np.asarray(jax.device_get(eigenvalues))
    n, n_test = len(x), len(components["xs"])
    s2 = float(jax.device_get(components["s2"]))
    base_cutoff = np.finfo(np.float64).eps * n * max(float(spectrum[-1]), 0.0)
    indices = weights = None
    if not method.startswith("T-"):
        if not blocks:
            raise ValueError("fitted blocks missing for a block method")
        width = max(map(len, blocks))
        padded_idx, padded_weight = [], []
        for block in blocks:
            idx = np.asarray(block, dtype=np.int32).reshape(-1)
            padded_idx.append(np.pad(idx, (0, width-len(idx))))
            db = residual[jnp.ix_(jnp.asarray(idx), jnp.asarray(idx))]
            lam, u = jnp.linalg.eigh(db)
            lam = jnp.maximum(lam, 0.0)
            wb = (u * (lam/(lam+s2))[None, :]) @ u.T
            padded_weight.append(jnp.pad(wb, ((0,width-len(idx)), (0,width-len(idx)))))
        indices, weights = jnp.asarray(np.stack(padded_idx)), jnp.stack(padded_weight)
        jax.block_until_ready(weights)
    preparation_seconds = time.perf_counter() - start
    state = {}
    for scale in scales:
        cutoff = base_cutoff * scale
        retained = eigenvalues > cutoff
        state[scale] = dict(cutoff=cutoff, retained=retained,
            invsqrt=jnp.where(retained, 1.0/jnp.sqrt(jnp.maximum(eigenvalues, cutoff)), 0.0),
            min_latent=float("inf"), discarded_fraction=0.0,
            weighted_extra_relative=0.0, variance_parts=[])
    for first in range(0, n_test, test_chunk):
        last = min(first+test_chunk, n_test)
        cross = (components["kernel_fn"](components["xs"][first:last], x)
                 - components["a_test"][:, first:last].T @ a)
        coeff = vec.T @ cross.T
        coeff2 = coeff*coeff
        total = jnp.sum(coeff2, axis=0)
        dtest = components["d_test"][first:last]
        for scale, s in state.items():
            white = vec @ (s["invsqrt"][:, None] * coeff)
            if method.startswith("T-"):
                gain = components["d_train"]/(components["d_train"]+s2)
                correction = jnp.sum(gain[:,None]*white**2, axis=0)
            else:
                part = white[indices]
                correction = jnp.einsum("bik,bij,bjk->k", part, weights, part)
            latent = components["qvar"][first:last] + dtest - correction
            s["min_latent"] = min(s["min_latent"], float(jax.device_get(jnp.min(latent))))
            s["variance_parts"].append(np.asarray(jax.device_get(latent+s2)).reshape(-1))
            discarded = jnp.sum(jnp.where(s["retained"][:,None], 0., coeff2), axis=0)
            s["discarded_fraction"] = max(s["discarded_fraction"], float(jax.device_get(
                jnp.max(discarded/jnp.maximum(total, 1e-30)))))
            extra = (eigenvalues > 10.0*base_cutoff) & (eigenvalues <= s["cutoff"])
            weighted = jnp.sum(jnp.where(extra[:,None], coeff2/jnp.maximum(
                eigenvalues[:,None], np.finfo(float).tiny), 0.0), axis=0)
            s["weighted_extra_relative"] = max(s["weighted_extra_relative"], float(jax.device_get(
                jnp.max(weighted/jnp.maximum(dtest, 1e-30)))))
    results = []
    for scale, s in state.items():
        variance = np.concatenate(s["variance_parts"])
        reasons = []
        if spectrum[0] < -s["cutoff"]:
            reasons.append("materially negative residual eigenvalue at this cutoff")
        if not np.isfinite(variance).all() or np.min(variance) <= 0:
            reasons.append("nonfinite/nonpositive observation variance")
        if s["min_latent"] < -1e-7:
            reasons.append("materially negative latent variance")
        results.append(dict(scale=scale, cutoff=s["cutoff"], valid=not reasons,
            rejection_reason="; ".join(reasons) or None,
            discarded_eigenvalues=int(np.count_nonzero(spectrum <= s["cutoff"])),
            max_discarded_unweighted_fraction=s["discarded_fraction"],
            max_additional_weighted_cross_relative_to_dtest=s["weighted_extra_relative"],
            min_latent_variance=s["min_latent"], variance=variance))
    return results, dict(preparation_seconds=preparation_seconds,
        total_prediction_seconds=time.perf_counter()-start,
        min_eigenvalue=float(spectrum[0]), max_eigenvalue=float(spectrum[-1]),
        dense_eigendecompositions=1, n_train=n, n_test=n_test)


def prepare(args):
    baseline = args.baseline.resolve()
    output = args.output.resolve()
    old = json.loads((baseline/"plan.json").read_text())
    if old["evaluator_sha256"] != real.sha256(Path(real.__file__)):
        raise ValueError("dense reference loader differs from historical evaluator")
    rows, sources = [], {}
    for r in old["runs"]:
        if r["method"] in ("SGPR", "SVGP"):
            continue
        row = copy.deepcopy(r)
        root, sim = Path(row["root"]), Path(row["source_sim"])
        run = root/row["relative"]
        reference_path = baseline/row["output_relative"]/"prediction.json"
        ref = json.loads(reference_path.read_text())
        for name, digest in ref["identity"]["inputs"].items():
            if real.sha256(root/name) != digest:
                raise ValueError("dense reference checkpoint has changed")
        row.update(reference=str(reference_path), reference_sha256=real.sha256(reference_path),
                   frozen_inputs=ref["identity"]["inputs"])
        key = row["source_sim"]
        if key not in sources:
            head = campaign.checked_checkout(sim.parent, root)
            sources[key] = dict(commit=head, files=campaign.source_hashes(sim, family=row["family"]))
        rows.append(row)
    if len(rows) != 280:
        raise ValueError(f"expected 280 nontrivial dense references, got {len(rows)}")
    real.guard_separate(output, [baseline] + [Path(r["root"]) for r in rows]
                        + [Path(r["source_sim"]) for r in rows])
    if (output/"plan.json").exists():
        raise ValueError("select a NEW output root")
    real.atomic_json(output/"plan.json", dict(implementation="dense-cutoff-sweep-v1",
        baseline_plan=str(baseline/"plan.json"), baseline_plan_sha256=real.sha256(baseline/"plan.json"),
        evaluator_sha256=real.sha256(Path(__file__)), loader_sha256=real.sha256(Path(real.__file__)),
        sources=sources, scales=SCALES, runs=rows))
    print(f"E4 planned {len(rows)} references, {len(SCALES)} cutoffs each")


def load_plan(output):
    plan = json.loads((output/"plan.json").read_text())
    if plan["evaluator_sha256"] != real.sha256(Path(__file__)) or (
            plan["loader_sha256"] != real.sha256(Path(real.__file__))):
        raise ValueError("planned evaluator/loader changed")
    if real.sha256(Path(plan["baseline_plan"])) != plan["baseline_plan_sha256"]:
        raise ValueError("baseline plan changed")
    return plan


def verify_row(plan, row):
    root, sim = Path(row["root"]), Path(row["source_sim"])
    for name, digest in row["frozen_inputs"].items():
        if real.sha256(root/name) != digest:
            raise ValueError("frozen input changed")
    for name, digest in plan["sources"][str(sim)]["files"].items():
        if real.sha256(sim/name) != digest:
            raise ValueError("historical source changed")
    if real.sha256(Path(row["reference"])) != row["reference_sha256"]:
        raise ValueError("reference prediction changed")


def evaluate(args):
    import jax
    plan = load_plan(args.output)
    row = plan["runs"][args.index]
    verify_row(plan, row)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("one assigned GPU required")
    root, sim = Path(row["root"]), Path(row["source_sim"])
    real.guard_separate(args.output, [root, sim.parent, Path(row["reference"]).parent])
    run = root/row["relative"]
    record = json.loads(campaign.record_path(run).read_text())
    if record["config"]["code_commit"] != plan["sources"][str(sim)]["commit"]:
        raise ValueError("checkpoint source commit mismatch")
    destination = args.output/row["output_relative"]/"cutoff_sweep.json"
    if destination.exists():
        p = json.loads(destination.read_text())
        if p["plan_sha256"] != real.sha256(args.output/"plan.json"):
            raise ValueError("existing result names another plan")
        print(f"CACHED {row['output_relative']}", flush=True)
        return
    start = time.perf_counter()
    components, blocks, ys, _ = real.load_components(row, run, record, sim)
    from common import predictors
    results, diagnostics = sweep(components, blocks, row["method"],
                                 test_chunk=args.test_chunk, scales=plan["scales"])
    mean = np.asarray(jax.device_get(components["mean"])).reshape(-1)
    base = next(p for p in results if p["scale"] == 10.)
    base_var = base["variance"].copy()
    base_rmse, base_nll = predictors.gaussian_rmse_nll(mean, base_var, ys)
    stored = json.loads(Path(row["reference"]).read_text())
    error = abs(float(base_nll)-stored["nll"])
    if not base["valid"] or error > 1e-10:
        raise ValueError(f"scale 10 failed reference reproduction: error={error:.17g}")
    for result in results:
        variance = result.pop("variance")
        if result["valid"]:
            rmse, nll = predictors.gaussian_rmse_nll(mean, variance, ys)
            result.update(rmse=float(rmse), nll=float(nll), nll_difference_from_10=float(nll-base_nll))
        else:
            result.update(rmse=None, nll=None, nll_difference_from_10=None)
        result["max_relative_observation_variance_change_from_10"] = float(np.max(
            np.abs(variance-base_var)/base_var))
    verify_row(plan, row)
    real.atomic_json(destination, dict(plan_sha256=real.sha256(args.output/"plan.json"),
        evaluator_sha256=plan["evaluator_sha256"], source_commit=record["config"]["code_commit"],
        config_id=record["config_id"], frozen_inputs=row["frozen_inputs"],
        reference_sha256=row["reference_sha256"], scale10_reference_nll_error=error,
        results=results, diagnostics=diagnostics, allocated_gpu_wall_seconds=time.perf_counter()-start,
        device_kind=jax.devices()[0].device_kind))
    print(f"E4 {row['output_relative']} scale10_error={error:.3g} seconds={diagnostics['total_prediction_seconds']:.2f}", flush=True)


def worker(args):
    plan = load_plan(args.output)
    indices = list(range(args.worker, len(plan["runs"]), args.workers))
    failures = []
    for position, index in enumerate(indices, 1):
        result = subprocess.run([sys.executable, "-u", str(Path(__file__)), "evaluate",
            "--output", str(args.output), "--index", str(index),
            "--test-chunk", str(args.test_chunk)], check=False)
        if result.returncode:
            failures.append(dict(index=index, exit_code=result.returncode))
        real.atomic_json(args.output/"state"/f"worker{args.worker}.json",
            dict(assigned=len(indices), processed=position, failures=failures,
                 complete=position==len(indices)))
    return 1 if failures else 0


def certify(args):
    plan = load_plan(args.output)
    digests, missing, rejected, rows = {}, [], [], []
    for row in plan["runs"]:
        verify_row(plan, row)
        path = args.output/row["output_relative"]/"cutoff_sweep.json"
        if not path.is_file():
            missing.append(row["output_relative"])
            continue
        p = json.loads(path.read_text())
        if p["plan_sha256"] != real.sha256(args.output/"plan.json") or (
                p["evaluator_sha256"] != plan["evaluator_sha256"]):
            raise ValueError("result provenance mismatch")
        if p["scale10_reference_nll_error"] > 1e-10 or p["frozen_inputs"] != row["frozen_inputs"]:
            raise ValueError("result reference/checkpoint mismatch")
        if [r["scale"] for r in p["results"]] != plan["scales"]:
            raise ValueError("incomplete cutoff grid")
        digests[row["output_relative"]] = real.sha256(path)
        for r in p["results"]:
            rows.append(dict(run=row["output_relative"], family=row["family"], kernel=row["kernel"],
                dataset=row.get("dataset", "kin40k" if row["family"]=="sgpr" else "pol"),
                M=row["M"], method=row["method"], seed=row["seed"], **r))
            if not r["valid"]:
                rejected.append(dict(run=row["output_relative"], scale=r["scale"], reason=r["rejection_reason"]))
    if rows:
        with (args.output/"summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    certificate = dict(plan_sha256=real.sha256(args.output/"plan.json"), count=len(digests),
        expected=len(plan["runs"]), predictions=digests, missing=missing,
        valid_count=len(rows)-len(rejected), rejected_count=len(rejected), rejected=rejected,
        scale10_all_reproduced=not missing, complete=not missing,
        interpretation="sensitivity diagnostic; rejected cutoff variants are not valid predictions")
    real.atomic_json(args.output/"DENSE_CUTOFF_CERTIFICATE.json", certificate)
    print(f"E4 references={len(digests)}/280 variants={len(rows)} rejected={len(rejected)} complete={not missing}")
    return 0 if not missing else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--baseline", type=Path, required=True)
    for name in ("evaluate", "worker", "certify"):
        p = sub.add_parser(name)
        if name == "evaluate":
            p.add_argument("--index", type=int, required=True)
        if name == "worker":
            p.add_argument("--worker", type=int, required=True)
            p.add_argument("--workers", type=int, default=5)
        if name in ("evaluate", "worker"):
            p.add_argument("--test-chunk", type=int, default=32)
    for p in sub.choices.values():
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    return {"plan": prepare, "evaluate": evaluate, "worker": worker, "certify": certify}[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
