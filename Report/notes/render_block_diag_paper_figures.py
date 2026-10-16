#!/usr/bin/env python3
"""Relabel certified hard-PIC aggregate figures without rerunning numerics."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPORT = PROJECT / "Report"
RESULTS = REPORT / "Results" / "hard_pic_full_v1"
SUMMARY = RESULTS / "svgp_hard_pic_summary.csv"
MANIFEST = RESULTS / "HARD_PIC_PAPER_ASSETS.json"
FIGURES = REPORT / "figures" / "hard_pic_full_v1"
sys.path.insert(0, str(HERE))

import build_svgp_frozen_appendix as appendix  # noqa: E402
import build_focused_svgp_page as focused  # noqa: E402


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_table() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    expected = manifest["outputs"][str(SUMMARY.relative_to(PROJECT))]
    if digest(SUMMARY) != expected:
        raise ValueError("certified hard-PIC aggregate changed")
    table = {}
    with SUMMARY.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["variant"], row["kernel"], row["dataset"],
                   int(row["M"]), row["method"], row["metric"])
            if key in table or int(row["n"]) != 10:
                raise ValueError(f"duplicate or non-ten-seed aggregate: {key}")
            values = float(row["mean"]), float(row["sem"])
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"nonfinite aggregate: {key}")
            table[key] = values
    if len(table) != 2560:
        raise ValueError(f"expected 2560 aggregate cells, got {len(table)}")
    return table


def main() -> None:
    table = load_table()
    FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = []
    for kernel in appendix.KERNELS:
        for group in range(2):
            datasets = appendix.DATASETS[4 * group:4 * (group + 1)]
            target = FIGURES / f"{kernel}_{group + 1}_block_diag.pdf"
            appendix.draw(table, kernel, datasets, target)
            outputs.append(target)
    target = FIGURES / "svgp_rbf_amplitude_final_page_block_diag.pdf"
    focused.campaign(table_override=table, output=target, wide=False)
    outputs.append(target)
    record = {
        "source_summary": str(SUMMARY.relative_to(PROJECT)),
        "source_summary_sha256": digest(SUMMARY),
        "render_only": True,
        "numerical_results_unchanged": True,
        "outputs": {str(path.relative_to(PROJECT)): digest(path)
                    for path in outputs},
    }
    (RESULTS / "BLOCK_DIAG_FIGURES.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"RENDERED={len(outputs)} SOURCE_CELLS={len(table)}")


if __name__ == "__main__":
    main()
