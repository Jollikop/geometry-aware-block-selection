#!/usr/bin/env python3
"""Plot the RBF-pol-M128 epoch curves and SBT batch-separation ablation."""
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
from common.plot_style import (  # noqa: E402
    method_color,
    method_marker,
)
from common.provenance import (  # noqa: E402
    SVGP_RELEASE_PACKAGES,
    config_fingerprint,
    release_runtime_matches,
    sha256_file,
)
from common import datasets as dm


from common.predictors import (  # noqa: E402  (owner of the registry)
    VARIATIONAL_METHOD_PREDICTOR,
)
from svgp.engine import IMPLEMENTATION  # noqa: E402  (owner of the id)


LINE_WIDTH = 1.7
MARKER_SIZE = 6.5
SAME_BATCH_RED = "#9E2F3A"
PANEL_TITLE_FONTSIZE = 14.5
AXIS_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 13
CURVES = (
    {
        "key": "SVGP",
        "method": "SVGP",
        "variant": "split",
        "label": "SVGP",
        "color": method_color("SVGP"),
        "linestyle": "-",
        "marker": method_marker("SVGP"),
    },
    {
        "key": "T-SVGP",
        "method": "T-SVGP",
        "variant": "split",
        "label": "T-SVGP",
        "color": method_color("T-SVGP"),
        "linestyle": "-",
        "marker": method_marker("T-SVGP"),
    },
    {
        "key": "RBT-SVGP",
        "method": "RBT-SVGP",
        "variant": "split",
        "label": "RBT-SVGP",
        "color": method_color("RBT-SVGP"),
        "linestyle": "-",
        "marker": method_marker("RBT-SVGP"),
    },
    {
        "key": "SBT-SVGP",
        "method": "SBT-SVGP",
        "variant": "split",
        "label": "SBT-SVGP",
        "color": method_color("SBT-SVGP"),
        "linestyle": "-",
        "marker": "D",
    },
    {
        "key": "SBT-SVGP-same-batch",
        "method": "SBT-SVGP",
        "variant": "coupled",
        "label": "SBT-SVGP (same batch)",
        "color": SAME_BATCH_RED,
        "linestyle": "--",
        "marker": "P",
    },
)
PANELS = (
    ("objective", "Stochastic objective / N"),
    ("rmse", "Test RMSE"),
    ("nll", "Test NLL"),
)
ZOOM_YLIMS = {
    "objective": (-0.25, 0.43),
    "rmse": (0.125, 0.23),
    "nll": (-0.53, 0.18),
}


def _float(row, key):
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return np.nan


def read_seed(path: Path, structured: bool):
    rows = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            epoch = int(float(row["epoch"]))
            # SBT arms record the local-cross predictor alongside the plain
            # sparse one; use it when the curve is a structured method.
            rmse_local = _float(row, "rmse_full")
            nll_local = _float(row, "nll_full")
            rows[epoch] = {
                "objective": _float(
                    row, "stochastic_objective_estimate_per_datum"
                ),
                "rmse": (
                    rmse_local
                    if structured and math.isfinite(rmse_local)
                    else _float(row, "rmse")
                ),
                "nll": (
                    nll_local
                    if structured and math.isfinite(nll_local)
                    else _float(row, "nll")
                ),
            }
    return rows


def validate_result(path: Path, curve, kernel, dataset, m, seed) -> dict:
    if not path.exists():
        raise SystemExit(f"missing required metrics: {path}")
    result = json.loads(path.read_text())
    config = result.get("config") or {}
    expected = {
        "implementation": IMPLEMENTATION,
        "method": curve["method"],
        "kernel": kernel,
        "dataset": dataset,
        "M": int(m),
        "seed": int(seed),
        "run_tag": f"pol_m128_epochs_{curve['variant']}_v1",
        "plan": "custom",
        "training_protocol": "svgp-gain-double-batch-v2",
        "requested_n_train": 40000,
        "requested_n_test": 5000,
        "all_data": True,
        "split_protocol": "all-rows-deterministic-90-10",
        "epochs": 350,
        "block": 500,
        "ll_batch_size": 1000,
        "blocks_per_step": 2,
        "full_batch": False,
        "common_ll_batches": True,
        "batch_coupling": curve["variant"],
        "penalty_sampling": "uniform",
        "refresh_every": 40,
        "refresh_until": 200,
        "importance_refresh_every": 10,
        "importance_epsilon": 0.05,
        "log_every": 20,
        "lr": 0.003,
        "z_lr": 0.003,
        "lr_profile": "stable",
        "optimizer": "lion",
        "grad_clip": 10.0,
        "normalize_objective": True,
        "q_optimizer": "natural",
        "natural_lr": 0.05,
        "evaluate_during_training": False,
        "record_history": True,
        "save_params": False,

        "data_source": dm.dataset_source(dataset),
    }
    expected["base_logdet_width"] = (
        500 if curve["method"] in {"RBT-SVGP", "SBT-SVGP"} else 0
    )
    if result.get("implementation") != IMPLEMENTATION:
        raise SystemExit(f"incompatible result: {path}")
    if result.get("config_id") != config_fingerprint(config):
        raise SystemExit(f"configuration fingerprint mismatch: {path}")
    if any(config.get(key) != value for key, value in expected.items()):
        raise SystemExit(f"protocol mismatch: {path}")
    if not str(config.get("data_signature", "")).startswith("sha256:"):
        raise SystemExit(f"missing data SHA-256: {path}")
    if len(str(config.get("code_commit", ""))) != 40:
        raise SystemExit(f"missing full Git commit: {path}")
    if config.get("code_source_dirty") is not False:
        raise SystemExit(f"dirty-code result rejected: {path}")
    # Every method reports the estimator its own bound models; the registry in
    # common.predictors is the only place that decides which. Pinning a literal
    # here is what made this validator reject every run after the rewrite.
    expected_predictor = VARIATIONAL_METHOD_PREDICTOR.get(curve["method"])
    if expected_predictor is not None and (
        config.get("reported_predictor") != expected_predictor
        or result.get("reported_predictor") != expected_predictor
    ):
        raise SystemExit(
            f"{curve['method']} should report {expected_predictor}: {path}"
        )
    runtime = result.get("runtime") or {}
    devices = runtime.get("jax_devices") or []
    if not release_runtime_matches(runtime, SVGP_RELEASE_PACKAGES):
        raise SystemExit(f"runtime package versions do not match release: {path}")
    if not any(
        item.get("platform") == "gpu"
        for item in devices if isinstance(item, dict)
    ):
        raise SystemExit(f"missing GPU runtime manifest: {path}")
    return result


def stack_curve(root: Path, curve, kernel, dataset, m, seeds):
    histories = []
    results = []
    method = curve["method"]
    for seed in seeds:
        run_dir = (
            root / curve["variant"] / kernel / dataset / f"M{m}" / method
            / f"seed{seed}"
        )
        path = run_dir / "history.csv"
        if not path.exists():
            raise SystemExit(f"missing required history: {path}")
        results.append(validate_result(
            run_dir / "metrics.json", curve, kernel, dataset, m, seed
        ))
        histories.append(read_seed(path, structured=method == "SBT-SVGP"))

    epochs = sorted(set.intersection(*(set(history) for history in histories)))
    if not epochs:
        raise SystemExit(f"no common epochs for {curve['label']}")
    data = {
        key: np.asarray(
            [[history[epoch][key] for epoch in epochs] for history in histories],
            dtype=float,
        )
        for key, _ in PANELS
    }
    return np.asarray(epochs, dtype=int), data, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(HERE / "runs_batch_ablation"))
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--dataset", default="pol")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--error", choices=("sem", "std"), default="sem")
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="show the complete warm-up range instead of the paper zoom",
    )
    parser.add_argument(
        "--out",
        default=str(
            HERE / "figures_batch_ablation" / "epochs_rbf_pol_M128.pdf"
        ),
    )
    args = parser.parse_args()

    root = Path(args.runs_dir)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.6, 3.55),
        sharex=True,
        constrained_layout=False,
    )
    summary = []
    validated_results = []
    for curve in CURVES:
        epochs, data, results = stack_curve(
            root, curve, args.kernel, args.dataset, args.m, args.seeds
        )
        validated_results.extend(results)
        marker_idx = np.arange(0, len(epochs), max(1, args.every))
        for ax, (key, title) in zip(axes, PANELS):
            values = data[key]
            mean = np.nanmean(values, axis=0)
            band = st.uncertainty(values, args.error)
            ax.plot(
                epochs,
                mean,
                label=curve["label"],
                color=curve["color"],
                linestyle=curve["linestyle"],
                linewidth=LINE_WIDTH,
            )
            ax.fill_between(
                epochs,
                mean - band,
                mean + band,
                color=curve["color"],
                alpha=0.13,
                linewidth=0,
            )
            ax.plot(
                epochs[marker_idx],
                mean[marker_idx],
                linestyle="",
                marker=curve["marker"],
                markersize=MARKER_SIZE,
                color=curve["color"],
                markeredgecolor="white",
                markeredgewidth=0.65,
            )
            ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE, pad=7)
            for epoch, mu, spread in zip(epochs, mean, band):
                summary.append({
                    "curve": curve["key"],
                    "metric": key,
                    "epoch": int(epoch),
                    "n_seeds": len(args.seeds),
                    "mean": float(mu),
                    args.error: float(spread),
                })

    commits = {
        str(result["config"]["code_commit"])
        for result in validated_results
    }
    if len(commits) != 1:
        plt.close(fig)
        raise SystemExit(
            f"a release figure requires one code commit, found {sorted(commits)}"
        )

    for ax, (key, _) in zip(axes, PANELS):
        ax.set_xlabel("Epoch", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_xlim(0, 350)
        if not args.full_range:
            ax.set_ylim(*ZOOM_YLIMS[key])
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.tick_params(labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.1,
        columnspacing=1.25,
    )
    fig.tight_layout(rect=(0.0, 0.20, 1.0, 1.0), w_pad=1.25)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        out.with_suffix(".png"),
        dpi=190,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)

    summary_path = out.with_name(f"{out.stem}_summary.csv")
    fields = ("curve", "metric", "epoch", "n_seeds", "mean", args.error)
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    manifest = {
        "implementation": IMPLEMENTATION,
        "pair": f"{args.kernel}:{args.dataset}",
        "M": int(args.m),
        "seeds": [int(seed) for seed in args.seeds],
        "curves": [curve["key"] for curve in CURVES],
        "error_bar": "sample SEM" if args.error == "sem" else "sample SD",
        "sample_standard_deviation_ddof": 1,
        "result_count": len(validated_results),
        "config_ids": sorted({
            str(result["config_id"]) for result in validated_results
        }),
        "data_signatures": sorted({
            str(result["config"]["data_signature"])
            for result in validated_results
        }),
        "code_commit": next(iter(commits)),
        "artifact": {out.name: sha256_file(out)},
    }
    manifest_path = out.with_name(f"{out.stem}_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    print(f"wrote {summary_path}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
