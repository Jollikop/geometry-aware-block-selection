#!/usr/bin/env python3
"""Compare residual-cross approximations at fixed trained SBT-SGPR models."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from common import stats as st  # noqa: E402
from common.plot_style import METHOD_COLORS  # noqa: E402
from common.provenance import (  # noqa: E402
    SGPR_RELEASE_PACKAGES,
    config_fingerprint,
    release_runtime_matches,
    sha256_file,
)


PROJECT_ROOT = HERE.parents[1]
from sgpr.benchmark import IMPLEMENTATION  # noqa: E402  (owner of the id)
from sgpr import protocol  # noqa: E402
EXPECTED_CONFIG = {
    "implementation": IMPLEMENTATION,
    "method": "SBT-SGPR",
    "kernel": "rbf",
    "M": 128,
    "dataset": "kin40k",
    "data_source": "uci_datasets-1.1.0:kin40k",
    "n_train": 1500,
    "n_test": 500,
    "n_blocks": 20,
    "block_size": 150,
    "steps": protocol.LBFGS_STEPS,
    "refresh_schedule": list(protocol.LBFGS_REFRESH_SCHEDULE),
    "refresh_every": protocol.LBFGS_STEPS + 1,
    "target_refreshes": None,
    "log_every": 100,
    "optimizer": "lbfgs",
    "optimizer_backend": "scipy-l-bfgs-b",
    "scipy_lbfgsb_ftol": protocol.SCIPY_LBFGSB_FTOL,
    "scipy_lbfgsb_gtol": protocol.SCIPY_LBFGSB_GTOL,
    "scipy_lbfgsb_maxcor": protocol.SCIPY_LBFGSB_MAXCOR,
    "scipy_lbfgsb_maxfun": protocol.SCIPY_LBFGSB_MAXFUN,
    "scipy_lbfgsb_maxls": protocol.SCIPY_LBFGSB_MAXLS,
    "lr": 0.02,
    "noise_lr": None,
    "grad_clip": 10.0,
    "train_inducing": True,
    "initialization_policy": list(protocol.INITIALIZATION_POLICY),
    "initialization_nested_max_m": protocol.INITIALIZATION_NESTED_MAX_M,
    "warmup_max_cycles": protocol.WARMUP_MAX_CYCLES,
    "warmup_theta_steps": protocol.WARMUP_THETA_STEPS,
    "warmup_reselect_obj_n_tol": protocol.WARMUP_RESELECT_OBJ_N_TOL,
    "require_convergence": True,
    "convergence_window_steps": protocol.CONVERGENCE_WINDOW_STEPS,
    "convergence_obj_n_tol": protocol.CONVERGENCE_OBJ_N_TOL,
    "convergence_grad_max_n_tol": protocol.CONVERGENCE_GRAD_MAX_N_TOL,
    "convergence_patience": protocol.CONVERGENCE_PATIENCE,
    "predictive_full_cross_max_n": 1500,
}


VARIANTS = (
    ("nll_sparse", "Sparse predictive (no residual)"),
    ("nll_full", "Dense Eq. (12)"),
)


def validate_row(row, path, seed):
    if row.get("implementation") != IMPLEMENTATION:
        raise SystemExit(f"{path}: incompatible implementation")
    if int(row.get("seed", -1)) != seed:
        raise SystemExit(f"{path}: expected seed {seed}")
    if row.get("convergence_passed") is not True:
        raise SystemExit(f"{path}: optimisation did not pass convergence")
    if row.get("optimizer_convergence_reached") is not True:
        raise SystemExit(f"{path}: optimiser did not reach stable windows")
    if not (1 <= int(row.get("steps", -1)) <= protocol.LBFGS_STEPS):
        raise SystemExit(f"{path}: invalid actual optimisation step count")
    if int(row.get("max_steps", -1)) != protocol.LBFGS_STEPS:
        raise SystemExit(f"{path}: wrong optimisation step cap")
    config = row.get("config")
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: missing run configuration")
    if row.get("config_id") != config_fingerprint(config):
        raise SystemExit(f"{path}: configuration fingerprint mismatch")
    mismatches = {
        key: (config.get(key), value)
        for key, value in EXPECTED_CONFIG.items()
        if config.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"{path}: protocol mismatch {mismatches}")
    if not str(config.get("data_signature", "")).startswith("sha256:"):
        raise SystemExit(f"{path}: missing content-based dataset signature")
    commit = str(config.get("code_commit", ""))
    if len(commit) != 40 or config.get("code_source_dirty") is not False:
        raise SystemExit(
            f"{path}: result is not tied to a clean experiment source tree"
        )
    runtime = row.get("runtime")
    devices = runtime.get("jax_devices") if isinstance(runtime, dict) else None
    if not release_runtime_matches(runtime, SGPR_RELEASE_PACKAGES):
        raise SystemExit(f"{path}: runtime package versions do not match release")
    if not isinstance(devices, list) or not any(
        isinstance(device, dict) and device.get("platform") == "gpu"
        for device in devices
    ):
        raise SystemExit(f"{path}: no GPU JAX device recorded")
    if row.get("reported_predictor") != "full":
        raise SystemExit(f"{path}: wrong headline predictor")
    missing = [key for key, _ in VARIANTS if key not in row]
    if missing:
        raise SystemExit(f"{path}: missing predictive metrics {missing}")
    if not math.isclose(
        float(row["nll"]),
        float(row["nll_full"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise SystemExit(f"{path}: headline NLL is not the local-cross NLL")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(HERE / "runs_v12"))
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--dataset", default="kin40k")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--error", choices=("sem", "ci95"), default="sem")
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "Report/figures/predictive_nll_ablation.pdf"),
    )
    parser.add_argument(
        "--summary-out",
        default=str(
            PROJECT_ROOT / "Report/Results/predictive_nll_ablation_summary.csv"
        ),
    )
    parser.add_argument(
        "--manifest-out",
        default=str(
            PROJECT_ROOT / "Report/Results/predictive_nll_ablation_manifest.json"
        ),
    )
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        path = (
            Path(args.runs_dir) / args.kernel / args.dataset / f"M{args.m}"
            / "SBT-SGPR" / f"seed{seed}" / "metrics.json"
        )
        if not path.exists():
            raise SystemExit(f"missing required run: {path}")
        rows.append(validate_row(json.loads(path.read_text()), path, seed))

    signatures = {row["config"]["data_signature"] for row in rows}
    commits = {row["config"]["code_commit"] for row in rows}
    config_ids = {row.get("config_id") for row in rows}
    if len(signatures) != 1 or len(commits) != 1 or len(config_ids) != len(rows):
        raise SystemExit(
            "the release requires one dataset SHA, one code commit, and "
            "one unique run configuration per seed"
        )

    labels, means, errors, summary = [], [], [], []
    for key, label in VARIANTS:
        values = np.asarray([float(row[key]) for row in rows])
        mean = float(np.mean(values))
        half_width = st.uncertainty(values, args.error)
        labels.append(label)
        means.append(mean)
        errors.append(half_width)
        summary.append({
            "variant": key,
            "label": label,
            "n_seeds": len(values),
            "mean_nll": mean,
            "error_mode": args.error,
            "half_width": half_width,
        })

    colors = (METHOD_COLORS["SGPR"], METHOD_COLORS["SBT-SGPR"],
              METHOD_COLORS["Exact-GP"])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    for xpos, mean, error, color in zip(x, means, errors, colors):
        ax.errorbar(
            [xpos],
            [mean],
            yerr=[error],
            fmt="none",
            ecolor=color,
            elinewidth=1.5,
            capsize=4,
            zorder=2,
        )
    ax.scatter(x, means, s=58, c=colors, edgecolor="white", linewidth=0.7, zorder=3)
    for row in rows:
        ax.plot(
            x,
            [float(row[key]) for key, _ in VARIANTS],
            color="#8A8A8A",
            alpha=0.16,
            linewidth=0.7,
            zorder=1,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Test NLL")
    ax.set_title(f"{args.dataset} / {args.kernel} / M={args.m}")
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "variant",
                "label",
                "n_seeds",
                "mean_nll",
                "error_mode",
                "half_width",
            ),
        )
        writer.writeheader()
        writer.writerows(summary)
    manifest = {
        "schema": 1,
        "experiment": "predictive-cross-term-ablation",
        "implementation": IMPLEMENTATION,
        "protocol": EXPECTED_CONFIG,
        "seeds": list(args.seeds),
        "error_bars": args.error,
        "sample_standard_deviation_ddof": 1,
        "data_signature": next(iter(signatures)),
        "code_commit": next(iter(commits)),
        "config_ids": sorted(config_ids),
        "artifacts": {
            out.name: sha256_file(out),
            summary_out.name: sha256_file(summary_out),
        },
    }
    manifest_out = Path(args.manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(f"wrote {summary_out}")
    print(f"wrote {manifest_out}")


if __name__ == "__main__":
    main()
