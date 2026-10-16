#!/usr/bin/env python3
"""Final legend wording of the paper figures, applied without replotting.

Each listed figure is copied to <name>_lbl.pdf with only the listed text runs
replaced or removed (same size, colour and baseline); curves, axes and data
are untouched. Run after the figure builders, from any directory.
"""
import os
from pathlib import Path

import matplotlib
import pymupdf

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FONTFILE = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans.ttf")
FONT = pymupdf.Font(fontfile=FONTFILE)

# Method names without publication years in the legends.
YEARS = [("SVGP (2013)", "SVGP"), ("T-SVGP (2025)", "T-SVGP"),
         ("(R)BT-SVGP (2025)", "(R)BT-SVGP"), ("(S)BT-SVGP (2026)", "(S)BT-SVGP")]
SGPR_NAMES = [("RBT-SGPR", "(R)BT-SGPR"), ("SBT-SGPR", "(S)BT-SGPR")]
GIBBS_NOTES = [("(S)BT: inducing-anchor donor", None),
               ("(S)BT: matched 300-epoch anchor transplant", None)]

SPECS = {
    "snelson_training_wide.pdf": SGPR_NAMES,
    # The longer reference label is set, with the other four, at a size
    # that keeps it clear of the parameter table on its row.
    "snelson_prediction_hard_pic_readable.pdf":
        SGPR_NAMES + [("SGPR", "SGPR"), ("T-SGPR", "T-SGPR"),
                      ("True GP", "Exact GP reference")],
    "epochs_rbf_pol_M128_objective.pdf": YEARS,
    "hard_pic_full_v1/svgp_rbf_amplitude_final_page_block_diag.pdf": YEARS,
    "hard_pic_dense_diagnostic/pol_hard_vs_dense_nll.pdf":
        YEARS + [("Full residual", "Dense"), ("D", None), ("ff", None)],
}
for kernel in ("rbf", "matern52", "amplitude-rbf", "gibbs"):
    for page in (1, 2):
        SPECS[f"hard_pic_full_v1/{kernel}_{page}_block_diag.pdf"] = (
            YEARS + (GIBBS_NOTES if kernel == "gibbs" else []))


def rgb(c):
    return ((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255


# Font size (pt) of the replaced labels, where it differs from the original.
LABEL_SIZE = {"snelson_prediction_hard_pic_readable.pdf": 11.0}


def relabel(src, spec, size=None):
    doc = pymupdf.open(FIGURES / src)
    page = doc[0]
    todo = dict(spec)
    found = [(s["text"].strip(), s)
             for b in page.get_text("dict")["blocks"]
             for line in b.get("lines", []) for s in line["spans"]
             if s["text"].strip() in todo]
    missing = sorted(set(todo) - {t for t, _ in found})
    for _, s in found:
        r = pymupdf.Rect(s["bbox"])
        r.y0 += 0.6
        r.y1 -= 0.6
        page.add_redact_annot(r, fill=False, cross_out=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    for text, s in found:
        new = todo[text]
        if new is not None:
            page.insert_text(s["origin"], new, fontsize=size or s["size"],
                             fontname="DejaVuSans", fontfile=FONTFILE,
                             color=rgb(s["color"]))
    out = FIGURES / (src[:-4] + "_lbl.pdf")
    doc.save(out, garbage=3, deflate=True)
    return out, len(found), missing


if __name__ == "__main__":
    for src, spec in SPECS.items():
        if not (FIGURES / src).is_file():
            print(f"skip (not built yet): {src}")
            continue
        out, n, missing = relabel(src, spec, LABEL_SIZE.get(src))
        note = f"; not present: {missing}" if missing else ""
        print(f"{out.relative_to(FIGURES.parent)}: {n} labels{note}")
