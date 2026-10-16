#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Campaign plans for the variational benchmark.

Which (kernel, dataset) pairs run, with how many inducing points and how many
seeds, is a property of a campaign rather than of the data, so it lives here
and not in ``common.datasets``.

``report``            the eight declared kernel-dataset pairs of the paper;
``grid``              RBF and Matern-5/2 on all eight datasets;
``grid-exploratory``  amplitude-RBF and Gibbs on all eight datasets.

Every plan is validated by ``common.datasets.assert_not_retired`` at
construction time, so a retired dataset fails before any compute is spent
rather than after.

Print a plan with::

    python -m svgp.plans --show-plan report
"""
from __future__ import annotations

import argparse
import sys

from common.datasets import assert_not_retired

# The paper uses one declared kernel per dataset rather than a full product.
REPORT_PAIRS = [
    # Main-paper inducing curves.
    ("rbf", "kin40k"),
    ("matern52", "protein"),
    ("gibbs", "3droad"),
    ("matern-periodic", "traffic"),
    # Appendix inducing curves.
    ("rbf", "electricity"),
    ("matern52", "superconductivity"),
    ("matern32", "elevators"),
    ("rbf", "pol"),
]

REPORT_DEFAULT_M_VALUES = (64, 128, 256, 512)
# Reduced budgets of the retired co2 row. Kept so that the archived campaign
# can still be validated and re-analysed; no live plan reaches these entries.
# Reduced budgets of the retired co2 row, kept so the archived campaign can
# still be validated. Keyed by dataset only: the pair-keyed twin held the same
# fact and could only ever drift from it.
REPORT_M_VALUES_BY_DATASET = {
    "co2": (16, 32, 64, 128),
}


# ---------------------------------------------------------------------------
# Grid campaign: every declared grid kernel is run on every dataset.
# ---------------------------------------------------------------------------
# The main figure shows four datasets, the appendix the remaining four; the
# order below fixes the panel order in both.
#
# v10: bike is out (target leakage) and co2 is out (~800 rows give a single
# block at B = 500, so the block methods degenerate to the exact bound and the
# row cannot separate them). elevators replaces bike at matched size and input
# dimension. Eight datasets, hence a 4x2 appendix grid instead of 5.
GRID_DATASETS = [
    # Main figure, ordered by increasing number of observations.
    "pol", "superconductivity", "kin40k", "protein",
    # Appendix, also ordered by size; the last two are capped at 200k.
    "elevators", "traffic", "electricity", "3droad",
]
GRID_KERNELS = ["rbf", "matern52"]
GRID_SEEDS = tuple(range(10))
# Gibbs uses the same ten-seed budget as the stationary kernels.
GRID_EXPLORATORY_KERNELS = ["amplitude-rbf", "gibbs"]
# Gibbs is reported with the same ten-seed budget and is trained independently
# from a cold start; no stationary checkpoint is a hidden dependency.
GRID_EXPLORATORY_SEEDS = tuple(range(10))



def grid_pairs(kernels=None, datasets=None):
    """Ordered (kernel, dataset) pairs of the full-grid campaign."""
    kernels = list(kernels or GRID_KERNELS)
    datasets = list(datasets or GRID_DATASETS)
    assert_not_retired(datasets)
    return [(k, d) for k in kernels for d in datasets]


def seeds_for_kernel(kernel: str) -> tuple[int, ...]:
    """Seed budget declared for one kernel of the grid campaign."""
    if str(kernel) in GRID_EXPLORATORY_KERNELS:
        return GRID_EXPLORATORY_SEEDS
    return GRID_SEEDS


def m_values_for_pair(plan: str, kernel: str, dataset: str,
                      requested=None) -> list[int]:
    """Return the inducing-point grid for one experiment-plan pair."""
    if plan in {"report", "grid", "grid-exploratory"}:
        values = REPORT_M_VALUES_BY_DATASET.get(
            str(dataset), REPORT_DEFAULT_M_VALUES
        )
    else:
        values = requested or REPORT_DEFAULT_M_VALUES
    return [int(value) for value in values]


# The official (model, inducing, natural) learning rates, shared by the runner,
# the status checker and the paper plotter so a valid run is never rejected as
# stale. Every kernel currently uses one profile; the per-kernel override
# stays because the launchers already pass these three numbers explicitly and
# a future kernel would otherwise have to change all of them at once.
STABLE_LR_PROFILE: dict[str, tuple[float, float, float]] = {}
DEFAULT_STABLE_LRS = (5e-3, 5e-3, 5e-2)


def stable_learning_rates(kernel: str) -> tuple[float, float, float]:
    """Return the official (model, inducing, natural) rates for a kernel."""
    return STABLE_LR_PROFILE.get(str(kernel), DEFAULT_STABLE_LRS)


def build_plan(plan: str):
    """Return the ordered pairs of a named campaign.

    ``report``            eight declared kernel-dataset pairs;
    ``grid``              RBF and Matern-5/2 on all eight datasets, 10 seeds;
    ``grid-exploratory``  amplitude-RBF and Gibbs, 10 seeds, 500 epochs.
    """
    if plan == "report":
        assert_not_retired([dataset for _, dataset in REPORT_PAIRS])
        return list(REPORT_PAIRS)
    if plan == "grid":
        return grid_pairs(GRID_KERNELS, GRID_DATASETS)
    if plan == "grid-exploratory":
        return grid_pairs(GRID_EXPLORATORY_KERNELS, GRID_DATASETS)
    raise ValueError(
        f"unknown named plan {plan!r}; choose 'report', 'grid' or "
        "'grid-exploratory'"
    )




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-plan",
                        choices=("report", "grid", "grid-exploratory"),
                        default="report",
                        help="print the (kernel, dataset) pairs and exit.")
    args = parser.parse_args()
    pairs = build_plan(args.show_plan)
    print(f"plan={args.show_plan}: {len(pairs)} (kernel, dataset) pairs")
    for kernel_name, dataset in pairs:
        print(f"  {kernel_name:16s} {dataset}")
    sys.exit(0)


if __name__ == "__main__":
    main()
