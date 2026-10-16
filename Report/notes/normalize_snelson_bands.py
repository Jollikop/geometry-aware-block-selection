"""Give all Snelson uncertainty curves the same dash and stroke width.

This edits PDF drawing styles only; the frozen prediction paths are unchanged.
The two readable layouts can be passed as input PDFs without rerunning models.
"""

import argparse
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, ContentStream, FloatObject


PATH_OPERATORS = {b"m", b"l", b"c", b"re", b"S", b"f", b"B"}
BAND_PATTERNS = {(5.0, 3.0), (4.25, 2.55), (6.5, 3.9)}


def normalize(source: Path, destination: Path) -> None:
    reader = PdfReader(source)
    if len(reader.pages) != 1:
        raise ValueError(f"expected one Snelson page: {source}")
    writer = PdfWriter(clone_from=reader)
    page = writer.pages[0]
    stream = ContentStream(page.get_contents(), writer)
    paths_before = [(args, op) for args, op in stream.operations if op in PATH_OPERATORS]
    band_count = 0
    changed_count = 0
    for index, (args, operator) in enumerate(stream.operations):
        if operator != b"d" or len(args[0]) != 2:
            continue
        pattern = tuple(round(float(value), 2) for value in args[0])
        if pattern not in BAND_PATTERNS:
            continue
        band_count += 1
        if pattern == (5.0, 3.0):
            continue  # Exact GP already uses the intended style.

        # The two layouts place the width either immediately before the dash
        # or after the colour command. Restrict the search to this local style.
        adjacent = list(range(index - 3, index)) + list(range(index + 1, index + 4))
        widths = [pos for pos in adjacent if stream.operations[pos][1] == b"w"]
        if len(widths) != 1:
            raise ValueError(f"cannot identify band stroke at operation {index}")
        width_index = widths[0]
        width = float(stream.operations[width_index][0][0])
        if width not in (0.85, 1.3):
            raise ValueError(f"unexpected band width {width} at operation {index}")
        stream.operations[index] = (
            [ArrayObject([FloatObject(5), FloatObject(3)]), FloatObject(0)],
            b"d",
        )
        stream.operations[width_index] = ([FloatObject(1)], b"w")
        changed_count += 1

    if band_count != 10 or changed_count not in (0, 8):
        raise ValueError(f"expected ten bands and zero or eight edits; got {band_count}, {changed_count}")
    paths_after = [(args, op) for args, op in stream.operations if op in PATH_OPERATORS]
    if paths_after != paths_before:
        raise AssertionError("prediction geometry changed")

    page.replace_contents(stream)
    temporary = destination.with_suffix(".tmp.pdf")
    with temporary.open("wb") as output:
        writer.write(output)
    temporary.replace(destination)
    print(f"{destination}: {band_count} bands at width 1 and dash (5, 3); paths unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="+", type=Path)
    options = parser.parse_args()
    for pdf in options.pdf:
        normalize(pdf, pdf)
