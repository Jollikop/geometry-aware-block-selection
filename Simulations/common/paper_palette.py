"""Official paper method colours, shared by raster and vector renderers.

Restored from the original Snelson comparison palette. No numerical
dependencies: the PDF-only Snelson reflow also imports this palette.
"""

ROLE_COLORS = {
    "sparse": "#8EC5E8",
    "diagonal": "#8FD694",
    "blocks": "#E78AC3",
    "anchored": "#E31A1C",
}
SAME_BATCH_COLOR = "#8d1822"

METHOD_COLORS = {
    "Exact-GP": "#242424", "True GP": "#000000", "E-SGPR": "#6B7280",
    **{name: ROLE_COLORS[role]
       for stack in ("SGPR", "SVGP")
       for prefix, role in (("", "sparse"), ("T-", "diagonal"),
                            ("RBT-", "blocks"), ("SBT-", "anchored"))
       for name in (prefix + stack,)},
}
