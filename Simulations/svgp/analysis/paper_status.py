#!/usr/bin/env python3
"""Report progress for the strict four-method SVGP paper campaign."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


from svgp import plans  # noqa: E402
from common import datasets as dm
from common.predictors import VARIATIONAL_METHOD_PREDICTOR as METHOD_PREDICTOR
from common.provenance import (  # noqa: E402
    SVGP_RELEASE_PACKAGES,
    config_fingerprint,
    release_runtime_matches,
)
from svgp.run_benchmark import FULL_PREDICTOR_MAX_N


METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")
SEEDS = tuple(range(10))
# Per-plan protocol knobs. The single-commit rule is enforced per campaign:
# results are filtered by run tag and read from that campaign's own runs
# directory, so a frozen campaign is unaffected by later commits belonging to
# a different run tag.
# The runs root must track ``svgp.run_benchmark``: when the implementation id
# moved these once still pointed at an older tree, so the validator scanned an
# empty directory and reported complete=0 for a finished campaign.
# ``tests/test_job_scripts.py::test_no_job_writes_to_a_retired_runs_tree``
# now fails if any launcher names a tree the code no longer knows.
PLAN_PROFILE = {
    "report": {
        "runs_root": "runs_v12",
        "seeds": lambda kernel: tuple(range(10)),
        "extra_required": {
            "record_history": False, "save_params": False,
            "epochs": 350, "refresh_until": 200,
        },
    },
    "grid": {
        "runs_root": "runs_v12",
        "seeds": lambda kernel: plans.seeds_for_kernel(kernel),
        "extra_required": {
            "record_history": False, "save_params": True,
            "epochs": 350, "refresh_until": 200,
        },
    },
    "grid-exploratory": {
        "runs_root": "runs_v12",
        "seeds": lambda kernel: plans.seeds_for_kernel(kernel),
        "extra_required": {
            "record_history": False, "save_params": True,
            "epochs": 500, "refresh_until": 500,
        },
    },
}
from svgp.engine import IMPLEMENTATION  # noqa: E402  (owner of the id)
TRAINING_PROTOCOL = "svgp-gain-double-batch-v3"


def campaign_pairs(plan: str = "report") -> list[tuple[str, str]]:
    """Return the ordered pairs used by the launcher of ``plan``."""
    return plans.build_plan(plan)


def expected_keys(pairs: list[tuple[str, str]], plan: str = "report") -> set[tuple]:
    seeds_for = PLAN_PROFILE[plan]["seeds"]
    return {
        (kernel, dataset, m, method, seed)
        for kernel, dataset in pairs
        for m in plans.m_values_for_pair(plan, kernel, dataset)
        for method in METHODS
        for seed in seeds_for(kernel)
    }


def matches_protocol(result: dict, run_tag: str, plan: str = "report") -> bool:
    """Check fields that define the public paper protocol."""
    config = dict(result.get("config") or {})
    method = config.get("method")
    kernel = str(config.get("kernel", result.get("kernel", "")))
    required = {
        "run_tag": run_tag,
        "plan": plan,
        "training_protocol": TRAINING_PROTOCOL,
        "requested_n_train": 40000,
        "requested_n_test": 5000,
        "all_data": True,
        "split_protocol": "all-rows-deterministic-90-10",
        "block": 500,
        "blocks_per_step": 2,
        "ll_batch_size": 1000,
        "full_batch": False,
        "common_ll_batches": True,
        "batch_coupling": "split",
        "penalty_sampling": "gain",
        "refresh_every": 40,
        "importance_refresh_every": 10,
        "log_every": 20,
        "lr_profile": "fixed",
        "optimizer": "adam",
        "grad_clip": 10.0,
        "q_optimizer": "natural",
        "normalize_objective": True,
        "evaluate_during_training": False,
        "full_predictor_max_n": FULL_PREDICTOR_MAX_N,
        **PLAN_PROFILE[plan]["extra_required"],
    }
    if method not in METHODS:
        return False
    identity = ("method", "kernel", "dataset", "M", "seed")
    if any(config.get(key) != result.get(key) for key in identity):
        return False
    if result.get("implementation") != IMPLEMENTATION:
        return False
    if result.get("training_protocol") != TRAINING_PROTOCOL:
        return False
    if not str(config.get("data_signature", "")).startswith("sha256:"):
        return False
    if not config.get("data_source"):
        return False
    commit = str(config.get("code_commit", ""))
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        return False
    if config.get("code_source_dirty") is not False:
        return False
    if result.get("config_id") != config_fingerprint(config):
        return False
    runtime = result.get("runtime")
    if not isinstance(runtime, dict):
        return False
    devices = runtime.get("jax_devices")
    if not release_runtime_matches(runtime, SVGP_RELEASE_PACKAGES):
        return False
    if not isinstance(devices, list) or not any(
        device.get("platform") == "gpu"
        for device in devices if isinstance(device, dict)
    ):
        return False
    if any(config.get(key) != value for key, value in required.items()):
        return False
    try:
        expected_source = dm.dataset_source(str(config.get("dataset")))
    except KeyError:
        return False
    if config.get("data_source") != expected_source:
        return False
    # One estimator per method, declared in common.predictors, and the headline
    # NLL must be the one that estimator produced. This is the check that keeps
    # an NLL column comparable across rows.
    expected_predictor = METHOD_PREDICTOR[method]
    if config.get("reported_predictor") != expected_predictor:
        return False
    if result.get("reported_predictor") != expected_predictor:
        return False
    headline_key = (
        "nll_sparse" if expected_predictor == "sparse" else "nll_bound"
    )
    try:
        if not math.isclose(
            float(result["nll"]), float(result[headline_key]),
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    expected_width = 500 if method in {"RBT-SVGP", "SBT-SVGP"} else 0
    if config.get("base_logdet_width") != expected_width:
        return False
    if expected_predictor == "full":
        if int(config.get("n_train", -1)) > int(config["full_predictor_max_n"]):
            return False
    model_lr, z_lr, natural_lr = plans.stable_learning_rates(kernel)
    numeric = {
        "importance_epsilon": 0.05,
        "lr": model_lr,
        "z_lr": z_lr,
        "natural_lr": natural_lr,
    }
    return all(
        math.isclose(float(config.get(key, math.nan)), value)
        for key, value in numeric.items()
    )


def result_key(result: dict) -> tuple | None:
    method = result.get("method")
    try:
        return (
            str(result["kernel"]),
            str(result["dataset"]),
            int(result["M"]),
            str(method),
            int(result["seed"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def is_valid_result(result: dict, run_tag: str, expected: set[tuple],
                    plan: str = "report") -> bool:
    """Return whether one metrics row is a finite official campaign result."""
    key = result_key(result)
    try:
        finite_metrics = all(
            math.isfinite(float(result[name]))
            for name in (
                "rmse",
                "nll",
                "last_epoch_stochastic_objective_estimate_per_datum",
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        key in expected
        and finite_metrics
        and matches_protocol(result, run_tag, plan)
    )


def scan_results(root: Path, run_tag: str, expected: set[tuple],
                 plan: str = "report"):
    complete: set[tuple] = set()
    commits: set[str] = set()
    invalid = 0
    duplicates = 0
    for metrics_path in root.rglob("metrics.json"):
        try:
            result = json.loads(metrics_path.read_text())
            key = result_key(result)
            valid = is_valid_result(result, run_tag, expected, plan)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            valid = False
            key = None
        if not valid:
            invalid += 1
        elif key in complete:
            duplicates += 1
        else:
            complete.add(key)
            commits.add(str(result["config"]["code_commit"]))
    return complete, invalid, duplicates, commits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", default="paper_official_v1")
    parser.add_argument(
        "--plan",
        choices=tuple(PLAN_PROFILE),
        default="report",
        help="campaign whose expected result set is checked. Each plan is "
             "validated independently: one commit is required within a "
             "campaign, not across campaigns.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="default: <plan runs root>/<run-tag> next to this script",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="print only the global completion line",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    runs_root = PLAN_PROFILE[args.plan]["runs_root"]
    root = args.runs_dir or here / runs_root / args.run_tag
    pairs = campaign_pairs(args.plan)
    expected = expected_keys(pairs, args.plan)
    complete, invalid, duplicates, commits = scan_results(
        root, args.run_tag, expected, args.plan
    )
    release_ready = (
        len(complete) == len(expected)
        and invalid == 0
        and duplicates == 0
        and len(commits) == 1
    )

    print(
        f"complete={len(complete)}/{len(expected)} "
        f"remaining={len(expected) - len(complete)} "
        f"invalid_or_other={invalid} duplicates={duplicates} "
        f"commits={len(commits)} "
        f"release_ready={'yes' if release_ready else 'no'}"
    )
    if args.brief:
        return

    print(f"runs_dir={root}")
    print(f"plan={args.plan} pairs={len(pairs)} methods={len(METHODS)}")
    if len(commits) == 1:
        print(f"campaign commit: {next(iter(commits))}")
    if len(commits) > 1:
        print("mixed code commits:")
        for commit in sorted(commits):
            print(f"  {commit}")
    seeds_for = PLAN_PROFILE[args.plan]["seeds"]
    for kernel, dataset in pairs:
        m_values = plans.m_values_for_pair(args.plan, kernel, dataset)
        per_pair = len(m_values) * len(METHODS) * len(seeds_for(kernel))
        done = sum(
            key[0] == kernel and key[1] == dataset for key in complete
        )
        if done != per_pair:
            m_label = ",".join(str(value) for value in m_values)
            print(
                f"{kernel:18} {dataset:20} {done:3}/{per_pair} "
                f"M={m_label}"
            )


if __name__ == "__main__":
    main()
