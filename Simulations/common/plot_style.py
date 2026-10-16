"""Shared public method names and plotting palette."""
from __future__ import annotations

import matplotlib
from .paper_palette import METHOD_COLORS, ROLE_COLORS, SAME_BATCH_COLOR


# AISTATS asks authors to avoid Type-3 fonts in camera-ready figures.  Set this
# once in the shared plotting module so every PDF/PS generator embeds TrueType
# outlines, including scripts that imported pyplot before this module.
matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# One colour per residual model across the SGPR and SVGP stacks.
# The canonical palette is the original Snelson comparison palette.

METHOD_MARKERS = {
    "Exact-GP": "o",
    "E-SGPR": "D",
    "SGPR": "o",
    "SVGP": "o",
    "T-SGPR": "s",
    "T-SVGP": "s",
    "RBT-SGPR": "^",
    "RBT-SVGP": "^",
    "SBT-SGPR": "*",
    "SBT-SVGP": "*",
}

MAIN_SVGP_METHODS = ("SVGP", "T-SVGP", "RBT-SVGP", "SBT-SVGP")
MAIN_SGPR_METHODS = ("SGPR", "T-SGPR", "RBT-SGPR", "SBT-SGPR")
UNCERTAINTY_DASHES = (0, (5, 3))


def canonical_method(name: str) -> str:
    """Normalize a method label to the current public spelling."""
    return {
        "BT-SGPR": "RBT-SGPR",
        "BT-SVGP": "RBT-SVGP",
    }.get(str(name), str(name))


def method_color(name: str, fallback: str = "#6B7280") -> str:
    return METHOD_COLORS.get(canonical_method(name), fallback)


def method_marker(name: str, fallback: str = "o") -> str:
    return METHOD_MARKERS.get(canonical_method(name), fallback)


def method_linewidth(name: str) -> float:
    """Keep baselines quiet and emphasize the proposed method."""
    return 2.6 if canonical_method(name).startswith("SBT-") else 1.2


def method_markersize(name: str) -> float:
    return 6.2 if canonical_method(name).startswith("SBT-") else 4.6


def method_band_linewidth(name: str) -> float:
    return 1.3 if canonical_method(name).startswith("SBT-") else 0.85


def main_plot_methods(methods) -> list[str]:
    """Return the four paper methods in the caller's requested order."""
    normalized = []
    for method in methods:
        public = canonical_method(method)
        if public not in MAIN_SVGP_METHODS and public not in MAIN_SGPR_METHODS:
            continue
        if public not in normalized:
            normalized.append(public)
    return normalized
