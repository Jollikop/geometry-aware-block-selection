#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample statistics shared by every figure and table.

Before the two-stack split each plotting script carried its own ``uncertainty``
helper. They disagreed on the default (SEM vs standard deviation) and only one
of them was covered by a test, so two figures in the same paper could report
different error bars for the same quantity. This module is the single
definition; ``tests/test_stats.py`` pins it.

Every estimator uses the SAMPLE standard deviation (``ddof=1``).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import t as student_t


#: Error-bar modes accepted by :func:`uncertainty`.
UNCERTAINTY_MODES = ("std", "var", "sem", "ci95")


def uncertainty(values, mode="sem", *, axis=None):
    """Return the requested dispersion measure of ``values``.

    ``std``   sample standard deviation (ddof=1)
    ``var``   sample variance (ddof=1)
    ``sem``   standard error of the mean, ``std / sqrt(n)``
    ``ci95``  half-width of the two-sided 95% Student-t interval

    ``axis=None`` treats ``values`` as one flat sample and returns a float.
    Passing ``axis=0`` reduces over the seed axis of a ``[seeds, points]``
    curve stack and returns one value per point, with NaNs ignored.
    """
    if mode not in UNCERTAINTY_MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {UNCERTAINTY_MODES}")
    values = np.asarray(values, dtype=np.float64)

    if axis is None:
        flat = values.reshape(-1)
        n = flat.size
        if n == 0:
            return float("nan")
        if n < 2:
            # One seed carries no dispersion information. std/var/sem are
            # legitimately zero; a t interval is undefined.
            return float("nan") if mode == "ci95" else 0.0
        std = float(np.std(flat, ddof=1))
        if mode == "std":
            return std
        if mode == "var":
            return float(np.var(flat, ddof=1))
        sem = std / math.sqrt(n)
        if mode == "sem":
            return sem
        return float(student_t.ppf(0.975, n - 1)) * sem

    n = values.shape[axis]
    std = np.nanstd(values, axis=axis, ddof=1)
    if mode == "std":
        return std
    if mode == "var":
        return np.nanvar(values, axis=axis, ddof=1)
    sem = std / math.sqrt(max(1, n))
    if mode == "sem":
        return sem
    return float(student_t.ppf(0.975, max(1, n - 1))) * sem


def mean_std(values, digits=4):
    """Return ``(mean, std)`` preformatted for a LaTeX/CSV cell.

    ``std`` is the empty string for a single observation, so a table cell reads
    ``0.1234`` rather than ``0.1234 +/- 0.0000``, which would claim a precision
    one seed cannot support.
    """
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return "", ""
    mean = f"{array.mean():.{digits}f}"
    if array.size < 2:
        return mean, ""
    return mean, f"{array.std(ddof=1):.{digits}f}"


def group_seeds(values, size=0):
    """Replace a per-seed sample by the means of equal-sized groups.

    With equal groups the mean of the group means equals the mean of the seeds
    exactly, so curves do not move; only the uncertainty does. The grouped
    standard error estimates the same quantity from ``G-1`` degrees of freedom
    instead of ``N-1``, which makes it noisier and its t interval wider.

    A trailing partial group is dropped: unequal groups would weight seeds
    unequally and break the identity above.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    size = int(size or 0)
    if size < 2 or values.size < 2 * size:
        return values
    usable = (values.size // size) * size
    return values[:usable].reshape(-1, size).mean(axis=1)
