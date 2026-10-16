#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declared data sources, caches and provenance for every experiment.

Both stacks read their raw arrays through this registry, so a dataset has one
declared source and one content hash across the whole project. Sources are
explicit -- OpenML, UCI, NOAA, a manually prepared UCI Electricity file -- and
there is no provider fallback: a failed download raises rather than silently
substituting another copy of "the same" data.

Every generated NPZ carries a sidecar with its source identifier and SHA-256
digest. A cache without a matching sidecar is rejected as a legacy cache.

Which (kernel, dataset) pairs a campaign runs is a property of the campaign,
not of the data, and lives in ``svgp.plans``.

Validate and rebuild before a campaign with::

    python -m common.datasets --check-data

Each registry entry is::

    key: {
        "spec":    <int data_id>  or  ("name", <str>, <version int|None>),
        "target":  <str column name>  or  None  (None = OpenML default target),
        "max_n":   <int>  or  None  (subsample cap for very large sets),
        "note":    <str>,
    }
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from common.provenance import sha256_file

HERE = Path(__file__).resolve().parent
# Shared project dataset folder: Simulations/dataset/
DATA_DIR = HERE.parent / "dataset"
_ACTIVE_DATA_DIR = DATA_DIR

# Default split / preprocessing configuration.
TEST_FRACTION = 0.10
GLOBAL_MAX_N = 200_000        # hard cap on (train+test) before splitting
CACHE_VERSIONS = {
    "electricity": "v2",
}


def raw_cache_path(key: str, data_dir: Path = DATA_DIR) -> Path:
    """Return the cache path for the current preprocessing version."""
    cache_version = CACHE_VERSIONS.get(key)
    cache_stem = f"{key}_{cache_version}" if cache_version else key
    return Path(data_dir) / f"{cache_stem}.npz"


def provenance_path(key: str, data_dir: Path = DATA_DIR) -> Path:
    """Return the sidecar that binds a cache to its declared source."""
    cache = raw_cache_path(key, data_dir)
    return cache.with_suffix(cache.suffix + ".provenance.json")


def _write_cache(key: str, data_dir: Path, X, y) -> None:
    cache = raw_cache_path(key, data_dir)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    np.savez_compressed(cache, X=X, y=y)
    metadata = {
        "schema": 1,
        "dataset": key,
        "source": dataset_source(key),
        "cache_sha256": sha256_file(cache),
        "shape_X": list(X.shape),
        "shape_y": list(y.shape),
    }
    provenance_path(key, data_dir).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def _load_verified_cache(key: str, data_dir: Path):
    cache = raw_cache_path(key, data_dir)
    sidecar = provenance_path(key, data_dir)
    if not sidecar.exists():
        raise RuntimeError(
            f"unverified legacy cache {cache}; rerun with --force to rebuild "
            "it from the declared source"
        )
    metadata = json.loads(sidecar.read_text())
    expected = {
        "dataset": key,
        "source": dataset_source(key),
        "cache_sha256": sha256_file(cache),
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"cache provenance mismatch for {cache}")
    with np.load(cache) as data:
        return data["X"].astype(np.float64), data["y"].astype(np.float64)


# ---------------------------------------------------------------------------
# Registry  (edit ids/targets here; validate with --check-data)
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "pol":            {"spec": 201, "source": "openml", "target": None,
                       "max_n": None,
                       "note": "OpenML 'pol' v1 (regression, ~15k x 48)."},
    # RETIRED from every campaign plan: the target is log(casual + registered)
    # and both columns are features (Spearman(x15+x16, y) = 0.999997,
    # Pearson(log1p(x15+x16), y) = 1.000000), so the regression is a
    # deterministic monotone function of two of its own inputs. sigma^2 -> 0
    # and no finite optimum exists, which is what the diverging ELBO showed.
    # The entry is kept, and only the entry, so archived campaigns containing
    # bike runs stay loadable and re-analysable.
    "bike":           {"spec": None, "source": "uci_datasets", "target": None,
                       "max_n": None, "retired": "target leakage",
                       "note": "uci-datasets Bike sharing demand (~17k). "
                               "RETIRED: target leakage, see comment above."},
    "elevators":      {"spec": None, "source": "uci_datasets", "target": None,
                       "max_n": None,
                       "note": "uci-datasets elevators (~16.6k x 18). "
                               "Replaces bike at matched size and input "
                               "dimension, so the compute profile of the grid "
                               "is unchanged."},
    "kin40k":         {"spec": None, "source": "uci_datasets", "target": None,
                       "max_n": None,
                       "note": "uci-datasets Kin40k (40k x 8)."},
    "protein":        {"spec": None, "source": "uci_datasets",
                       "target": None, "max_n": None,
                       "note": "uci-datasets CASP protein (~45.7k x 9)."},
    "3droad":         {"spec": None, "source": "uci_datasets", "target": None,
                       "max_n": GLOBAL_MAX_N, "family": "spatial",
                       "note": "uci-datasets 3D road network (~434k x 3); capped by max_n. "
                               "Low-d, filamentary/anisotropic input density -- "
                               "locality-aware (Gibbs) showcase."},

    # --- Periodic / phase-locality datasets ------------------------------- #
    # Inputs are time-only phase features mapped to RADIANS so the
    # periodic-angular kernel (period 2*pi) can group same-phase points that are
    # far apart in calendar time -- something no Euclidean clustering can do.
    # ``angle_cols`` marks the columns kept on their native radian scale (never
    # standardised); ``loader`` builds (X, y) directly, bypassing the generic
    # OpenML numeric extraction.
    "traffic":        {"spec": ("name", "Metro_Interstate_Traffic_Volume", None),
                       "target": "traffic_volume", "max_n": None,
                       "family": "periodic", "loader": "traffic",
                       "angle_cols": [0, 1, 2],
                       "note": "Metro Interstate Traffic Volume (~48k). Inputs = "
                               "[hour, day-of-week, month] as radians; target = "
                               "traffic_volume. Flagship periodic-angular case."},
    # RETIRED from every campaign plan. Not for leakage -- the high correlation
    # with the trend column is the legitimate signal of a time series -- but
    # because ~800 rows give a single partition block at B = 500, so SBT, BT
    # and the exact log-determinant coincide and the dataset cannot separate
    # the methods it is supposed to compare. Kept as a registry entry so
    # archived campaigns containing co2 runs stay loadable.
    "co2":            {"spec": None, "target": None, "max_n": None,
                       "family": "periodic", "loader": "co2",
                       "angle_cols": [0], "retired": "too small for blocks",
                       "note": "Mauna Loa monthly CO2 (NOAA GML). Inputs = "
                               "[annual phase (rad), decimal-year trend]. Small "
                               "(~800 pts): canonical periodic-kernel figure, "
                               "not a large-scale benchmark row. RETIRED."},
    "electricity":    {"spec": None, "target": None, "max_n": GLOBAL_MAX_N,
                       "family": "periodic", "loader": "electricity",
                       "note": "UCI ElectricityLoadDiagrams2011-2014. Inputs "
                               "= sine/cosine embeddings of hour, week and "
                               "year plus a linear trend. Prepare "
                               "dataset/electricity/data_v2.csv.gz with "
                               "prepare_electricity.py."},

    "superconductivity": {
                       "spec": None, "target": None, "max_n": None,
                       "family": "failure", "loader": "superconductivity",
                       "note": "UCI 464 superconductivity, high-dimensional "
                               "honest failure/control case. Reuses the SGPR "
                               "UCI downloader/parser."},
}

DATASET_KEYS = list(DATASETS.keys())
# Datasets kept in the registry so archived runs stay loadable, but excluded
# from every campaign plan. Never add one of these to a plan without also
# deleting its ``retired`` marker and saying why in the commit message.
RETIRED_DATASETS = tuple(
    key for key, entry in DATASETS.items() if entry.get("retired")
)

def assert_not_retired(datasets) -> None:
    """Refuse to build a plan containing a retired dataset.

    bike cost a full campaign column before anyone noticed its target was a
    function of its own features. This turns "remember not to use it" into a
    failure at plan-construction time, which is the only place it is cheap.
    """
    used = [d for d in datasets if d in RETIRED_DATASETS]
    if used:
        detail = ", ".join(
            f"{d} ({DATASETS[d]['retired']})" for d in used
        )
        raise ValueError(
            f"retired dataset(s) in an experiment plan: {detail}. "
            "Remove them, or drop the 'retired' marker in DATASETS and record "
            "the reason in the commit message."
        )


def angle_cols_for(key: str):
    """Angular (radian, never-standardised) feature indices for ``key`` or []."""
    return list(DATASETS.get(key, {}).get("angle_cols", []) or [])


def dataset_source(key: str) -> str:
    """Return the immutable source identifier recorded in run metadata."""
    entry = DATASETS[key]
    loader = entry.get("loader")
    custom = {
        "traffic": "UCI:492",
        "co2": "NOAA-GML:MLO-monthly",
        "electricity": "UCI:321:prepared-v2",
        "superconductivity": "UCI:464",
    }
    if loader in custom:
        return custom[loader]
    if loader:
        return f"manual:{loader}"
    source = entry.get("source", "uci_datasets")
    if source == "openml":
        return f"OpenML:{entry['spec']}"
    return f"uci_datasets-1.1.0:{key}"


# ---------------------------------------------------------------------------
# Custom loaders for periodic datasets (return raw (X, y); X angle cols in rad)
# ---------------------------------------------------------------------------

def _load_traffic(*, force: bool = False):
    """Metro Interstate Traffic Volume -> time-only angular inputs.

    X = [hour, day-of-week, month] mapped to radians (period 2*pi); the weather
    covariates are intentionally dropped so the input geometry is pure phase and
    the periodic-angular kernel can wrap same-phase points across the calendar.
    """
    import pandas as pd

    url = (
        "https://archive.ics.uci.edu/static/public/492/"
        "metro+interstate+traffic+volume.zip"
    )
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [
            name for name in archive.namelist()
            if name.lower().endswith((".csv", ".csv.gz"))
            and "macosx" not in name.lower()
        ]
        if len(members) != 1:
            raise RuntimeError(f"unexpected UCI traffic archive members: {members}")
        member = members[0]
        compression = "gzip" if member.lower().endswith(".gz") else None
        with archive.open(member) as stream:
            fr = pd.read_csv(stream, compression=compression)
    if "date_time" not in fr.columns or "traffic_volume" not in fr.columns:
        raise KeyError(f"unexpected columns {list(fr.columns)}")
    dt = pd.to_datetime(fr["date_time"], errors="coerce")
    two_pi = 2.0 * np.pi
    hour = dt.dt.hour.to_numpy(dtype=np.float64)
    dow = dt.dt.dayofweek.to_numpy(dtype=np.float64)
    month = dt.dt.month.to_numpy(dtype=np.float64) - 1.0
    X = np.stack([two_pi * hour / 24.0,
                  two_pi * dow / 7.0,
                  two_pi * month / 12.0], axis=1)
    y = pd.to_numeric(fr["traffic_volume"], errors="coerce").to_numpy(np.float64)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


def _load_co2(*, force: bool = False):
    """Mauna Loa monthly mean CO2 (NOAA GML). Canonical periodic-kernel data.

    X = [annual phase (radians), decimal-year trend]; y = CO2 ppm. Reads a
    local cache ``dataset/co2/co2_mm_mlo.csv`` if present, else downloads the
    published NOAA file once. Only column 0 (phase) is angular.
    """
    import pandas as pd

    local = _dataset_dir() / "co2" / "co2_mm_mlo.csv"
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
    src = str(local) if local.exists() else url
    # NOAA file has ~50 comment lines beginning with '#'.
    df = pd.read_csv(src, comment="#")
    df.columns = [c.strip() for c in df.columns]
    dec = pd.to_numeric(df.get("decimal date", df.iloc[:, 2]), errors="coerce")
    co2 = pd.to_numeric(df.get("average", df.iloc[:, 3]), errors="coerce")
    dec = dec.to_numpy(np.float64)
    co2 = co2.to_numpy(np.float64)
    phase = 2.0 * np.pi * (dec % 1.0)          # annual phase in radians
    X = np.stack([phase, dec], axis=1)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(co2) & (co2 > 0)
    return X[mask], co2[mask]


def _load_electricity(*, force: bool = False):
    """UCI ElectricityLoadDiagrams (single client or aggregate vs time).

    Source cache: ``dataset/electricity/data_v2.csv.gz`` with sine/cosine
    embeddings of hour, week and year, a linear trend, and the load target.
    The helper ``prepare_electricity.py`` builds it from the UCI archive,
    either as aggregate load or a chosen single client.
    """
    path = _dataset_dir() / "electricity" / "data_v2.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"electricity is a manual dataset: create {path} with columns "
            "[sin_hour, cos_hour, sin_week, cos_week, sin_year, cos_year, "
            "trend_years, load] (see DATASETS['electricity']['note']).")
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"{path} must contain exactly 8 columns, got {arr.shape}")
    return arr[:, :-1].astype(np.float64), arr[:, -1].astype(np.float64).ravel()


def _dataset_dir() -> Path:
    return Path(_ACTIVE_DATA_DIR)


def _finite_xy(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError(f"invalid manual dataset shapes X={X.shape}, y={y.shape}")
    if X.shape[1] == 0:
        raise ValueError("manual dataset has no input columns")
    if float(np.std(y)) <= 0.0:
        raise ValueError("manual dataset target is constant")
    return X, y


def _load_superconductivity(*, force: bool = False):
    """Reuse the UCI zip parser for the superconductivity failure case."""
    from common import datasets_uci

    X, y, _ = datasets_uci.load_raw("superconductivity", force=force)
    return _finite_xy(X, y)


_CUSTOM_LOADERS = {
    "traffic": _load_traffic,
    "co2": _load_co2,
    "electricity": _load_electricity,
    "superconductivity": _load_superconductivity,
}


# ---------------------------------------------------------------------------
# Raw download + cache
# ---------------------------------------------------------------------------

def _fetch_openml_raw(spec, target):
    """Return (X float64 [N,D], y float64 [N]) from OpenML for ``spec``."""
    from sklearn.datasets import fetch_openml
    import pandas as pd

    if isinstance(spec, int):
        ds = fetch_openml(data_id=spec, as_frame=True)
    else:
        _, name, version = spec
        kw = {"name": name, "as_frame": True}
        if version is not None:
            kw["version"] = version
        ds = fetch_openml(**kw)

    frame = ds.frame.copy()
    tgt = target or ds.target_names[0] if getattr(ds, "target_names", None) else target
    if tgt is None:
        # fall back to last column of the frame
        tgt = frame.columns[-1]
    if tgt not in frame.columns:
        raise KeyError(f"target '{tgt}' not in columns {list(frame.columns)[:8]}...")

    y = pd.to_numeric(frame[tgt], errors="coerce").to_numpy(dtype=np.float64)
    X_df = frame.drop(columns=[tgt])
    # keep numeric features only (drop ids / datetimes / categoricals)
    X_num = X_df.apply(pd.to_numeric, errors="coerce")
    X_num = X_num.dropna(axis=1, how="all")
    X = X_num.to_numpy(dtype=np.float64)

    # drop rows with any NaN in X or y
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("no usable numeric data after cleaning")
    return X, y


def load_raw(key: str, data_dir: Path = DATA_DIR, force: bool = False):
    """Load raw (X, y) for ``key``, downloading + caching on first use."""
    if key not in DATASETS:
        raise KeyError(f"unknown dataset '{key}'. Known: {DATASET_KEYS}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = raw_cache_path(key, data_dir)

    if cache.exists() and not force:
        return _load_verified_cache(key, data_dir)

    # Custom loaders (periodic datasets with engineered angular features) take
    # priority over the generic declared-source loader.
    loader_key = DATASETS[key].get("loader")
    if loader_key:
        global _ACTIVE_DATA_DIR
        old_data_dir = _ACTIVE_DATA_DIR
        _ACTIVE_DATA_DIR = data_dir
        try:
            X, y = _CUSTOM_LOADERS[loader_key](force=force)
        finally:
            _ACTIVE_DATA_DIR = old_data_dir
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64).ravel()
        _write_cache(key, data_dir, X, y)
        return X, y

    entry = DATASETS[key]
    source = entry.get("source", "uci_datasets")
    if source == "uci_datasets":
        try:
            from uci_datasets import Dataset

            ds = Dataset(key)
            X = np.asarray(ds.x, dtype=np.float64)
            y = np.asarray(ds.y, dtype=np.float64).ravel()
            finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X, y = X[finite], y[finite]
            if not X.shape[0] or not X.shape[1]:
                raise ValueError("source returned no finite numeric observations")
        except Exception as exc:
            raise RuntimeError(
                f"failed to load {key!r} from its declared source "
                "uci_datasets==1.1.0; no automatic fallback is permitted"
            ) from exc
    elif source == "openml":
        X, y = _fetch_openml_raw(entry["spec"], entry["target"])
    else:
        raise ValueError(f"unsupported declared source {source!r} for {key!r}")
    _write_cache(key, data_dir, X, y)
    return X, y


# ---------------------------------------------------------------------------
# Validation CLI
# ---------------------------------------------------------------------------

def check_data(keys=None, data_dir: Path = DATA_DIR, force: bool = False):
    keys = keys or DATASET_KEYS
    print(f"Checking {len(keys)} datasets  (cache: {data_dir})\n")
    ok, bad = [], []
    for key in keys:
        try:
            ac = angle_cols_for(key)
            X, y = load_raw(key, data_dir, force=force)
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64).reshape(-1)
            if X.ndim != 2 or X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"invalid regression arrays X={X.shape}, y={y.shape}"
                )
            if not np.isfinite(X).all() or not np.isfinite(y).all():
                raise ValueError("raw cache contains non-finite values")
            n_test = max(1, int(round(TEST_FRACTION * X.shape[0])))
            n_train = X.shape[0] - n_test
            fam = DATASETS[key].get("family", "tabular")
            tag = f" [{fam}, angle_cols={ac}]" if ac else f" [{fam}]"
            print(f"[ OK ] {key:15s} raw={X.shape}  "
                  f"declared_split={n_train}/{n_test}  "
                  f"y[min,max]=[{y.min():.3g},{y.max():.3g}]{tag}")
            ok.append(key)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {key:15s} {type(e).__name__}: {e}")
            # An unknown key has no registry entry, so the hint itself used to
            # raise a second KeyError and hide the real failure.
            entry = DATASETS.get(key)
            print(f"        -> fix DATASETS['{key}'] (note: {entry['note']})"
                  if entry else
                  f"        -> '{key}' is not in DATASETS; add a registry entry")
            bad.append(key)
    print(f"\n{len(ok)} ok, {len(bad)} failed.")
    if bad:
        print("Failed:", ", ".join(bad))
    return ok, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-data", action="store_true",
                    help="download + validate all datasets, print shapes")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of dataset keys to check")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild caches and provenance sidecars from declared sources",
    )
    args = ap.parse_args()

    if args.check_data:
        ok, bad = check_data(args.datasets, Path(args.data_dir), args.force)
        sys.exit(1 if bad else 0)
    print("Nothing to do. Use --check-data to validate the datasets.")
    print("Known datasets:", ", ".join(DATASET_KEYS))


if __name__ == "__main__":
    main()
