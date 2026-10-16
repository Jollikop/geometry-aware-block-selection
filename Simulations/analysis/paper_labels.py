"""Display-only paper labels. Never change serialized method/config identities."""

SVGP_LABELS = {
    "SVGP": "SVGP (2013)",
    "T-SVGP": "T-SVGP (2025)",
    "RBT-SVGP": "(R)BT-SVGP (2025)",
    "SBT-SVGP": "(S)BT-SVGP (2026)",
}


def paper_label(method):
    return SVGP_LABELS.get(method, method)
