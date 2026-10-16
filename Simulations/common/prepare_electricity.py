#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare UCI ElectricityLoadDiagrams for the temporal RBF benchmark.

The benchmark loader expects:

    Simulations/dataset/electricity/data_v2.csv.gz

with eight headerless columns:

    sin(hour), cos(hour), sin(week), cos(week),
    sin(year), cos(year), linear_trend_years, load

By default this script regresses the aggregate load over all 370 clients on
cyclic calendar features and a non-periodic trend. Pass ``--client N`` to use
one client column instead.
"""
from __future__ import annotations

import argparse
import gzip
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "dataset"
ELEC_DIR = DATA_DIR / "electricity"
ARCHIVE = ELEC_DIR / "electricityloaddiagrams20112014.zip"
OUT = ELEC_DIR / "data_v2.csv.gz"
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/321/"
    "electricityloaddiagrams20112014.zip"
)


def download(force: bool = False) -> None:
    ELEC_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists() and not force:
        print(f"[ok] archive exists: {ARCHIVE}")
        return
    print(f"[download] {UCI_URL}")
    with urllib.request.urlopen(UCI_URL) as response:
        with ARCHIVE.open("wb") as fh:
            shutil.copyfileobj(response, fh)
    print(f"[ok] saved {ARCHIVE} ({ARCHIVE.stat().st_size / 1e6:.1f} MB)")


def raw_member() -> str:
    with zipfile.ZipFile(ARCHIVE) as zf:
        names = [
            name for name in zf.namelist()
            if name.endswith("LD2011_2014.txt")
            and not name.startswith("__MACOSX/")
        ]
    if not names:
        raise FileNotFoundError("LD2011_2014.txt not found inside archive")
    return names[0]


def load_chunks(member: str, chunksize: int):
    with zipfile.ZipFile(ARCHIVE) as zf:
        with zf.open(member) as fh:
            yield from pd.read_csv(
                fh,
                sep=";",
                decimal=",",
                chunksize=chunksize,
                low_memory=False,
            )


def chunk_to_array(chunk: pd.DataFrame, client: Optional[int]) -> np.ndarray:
    time_col = chunk.columns[0]
    value_cols = [
        col for col in chunk.columns[1:]
        if str(col).strip() and not str(col).startswith("Unnamed")
    ]
    if client is None:
        y = (
            chunk[value_cols]
            .apply(pd.to_numeric, errors="coerce")
            .sum(axis=1)
            .to_numpy(np.float64)
        )
    else:
        if client < 0 or client >= len(value_cols):
            raise IndexError(
                f"--client must be in [0,{len(value_cols) - 1}], got {client}"
            )
        y = pd.to_numeric(chunk[value_cols[client]], errors="coerce").to_numpy(
            np.float64
        )

    dt = pd.to_datetime(chunk[time_col], errors="coerce")
    hour = dt.dt.hour.to_numpy(np.float64) + dt.dt.minute.to_numpy(np.float64) / 60.0
    dow = dt.dt.dayofweek.to_numpy(np.float64)
    day_of_year = dt.dt.dayofyear.to_numpy(np.float64) - 1.0
    year = dt.dt.year.to_numpy(np.float64)
    hour_phase = 2.0 * math.pi * hour / 24.0
    week_phase = 2.0 * math.pi * (dow + hour / 24.0) / 7.0
    year_phase = 2.0 * math.pi * (
        day_of_year + hour / 24.0
    ) / 365.2425
    trend_years = (
        year + (day_of_year + hour / 24.0) / 365.2425
    )
    arr = np.column_stack([
        np.sin(hour_phase),
        np.cos(hour_phase),
        np.sin(week_phase),
        np.cos(week_phase),
        np.sin(year_phase),
        np.cos(year_phase),
        trend_years,
        y,
    ])
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]


def prepare(client: Optional[int], chunksize: int, force: bool) -> None:
    if OUT.exists() and not force:
        print(f"[ok] prepared file exists: {OUT}")
        return
    download(force=False)
    member = raw_member()
    print(f"[prepare] member={member} target={'aggregate' if client is None else client}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(OUT, "wt") as fh:
        for chunk in load_chunks(member, chunksize):
            arr = chunk_to_array(chunk, client)
            if arr.size:
                np.savetxt(fh, arr, delimiter=",", fmt="%.12g")
                rows += arr.shape[0]
    print(f"[ok] saved {OUT} rows={rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client",
        type=int,
        default=None,
        help="0-based client column. Omit for aggregate load over all clients.",
    )
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.client, args.chunksize, args.force)


if __name__ == "__main__":
    main()
