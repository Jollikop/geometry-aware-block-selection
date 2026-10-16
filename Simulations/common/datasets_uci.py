#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download and load the UCI datasets used by the SGPR benchmark.

The raw UCI archives are retained in ``Simulations/dataset/sgpr/raw`` and the
cleaned numeric arrays are cached as ``Simulations/dataset/sgpr/<key>.npz``.
There is deliberately no synthetic fallback: download, parsing, or validation
failures raise an exception.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
# Shared project dataset folder, in a sgpr/ sub-namespace so its caches (which
# carry extra metadata) never collide with the Modern OpenML caches that reuse
# the same dataset names (protein, bike, ...).
DATA_DIR = HERE.parent / "dataset" / "sgpr"
RAW_DIR = DATA_DIR / "raw"

DATASETS = {
    "protein": {
        "label": "Protein tertiary structure",
        "url": (
            "https://archive.ics.uci.edu/static/public/265/"
            "physicochemical+properties+of+protein+tertiary+structure.zip"
        ),
        "parser": "protein",
        "uci_id": 265,
    },
    "superconductivity": {
        "label": "Superconductivity critical temperature",
        "url": (
            "https://archive.ics.uci.edu/static/public/464/"
            "superconductivty+data.zip"
        ),
        "parser": "superconductivity",
        "uci_id": 464,
    },
    "appliances": {
        "label": "Appliances energy",
        "url": (
            "https://archive.ics.uci.edu/static/public/374/"
            "appliances+energy+prediction.zip"
        ),
        "parser": "appliances",
        "uci_id": 374,
    },
    "bike": {
        "label": "Bike sharing hourly demand",
        "url": (
            "https://archive.ics.uci.edu/static/public/275/"
            "bike+sharing+dataset.zip"
        ),
        "parser": "bike",
        "uci_id": 275,
    },
    "gas_turbine": {
        "label": "Gas turbine CO emissions",
        "url": (
            "https://archive.ics.uci.edu/static/public/551/"
            "gas+turbine+co+and+nox+emission+data+set.zip"
        ),
        "parser": "gas_turbine",
        "uci_id": 551,
    },
}

DATASET_KEYS = tuple(DATASETS)


def cache_path(key: str, data_dir=DATA_DIR) -> Path:
    """Path to the cache produced from the declared UCI source."""
    return Path(data_dir) / f"{key}.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _member(zf: zipfile.ZipFile, basename: str) -> str:
    matches = [
        name for name in zf.namelist()
        if Path(name).name.lower() == basename.lower()
    ]
    if not matches:
        raise FileNotFoundError(
            f"{basename!r} not found in archive; members={zf.namelist()[:12]}"
        )
    return matches[0]


def _read_csv(zf: zipfile.ZipFile, member: str, **kwargs) -> pd.DataFrame:
    with zf.open(member) as stream:
        return pd.read_csv(stream, **kwargs)


def _xy_numeric(frame: pd.DataFrame, target: str, drop=()):
    columns = {str(c).strip().lower(): c for c in frame.columns}
    target_key = target.strip().lower()
    if target_key not in columns:
        raise KeyError(
            f"target {target!r} absent; columns={list(frame.columns)[:12]}"
        )
    target_col = columns[target_key]
    drop_cols = [columns[d.lower()] for d in drop if d.lower() in columns]
    y = pd.to_numeric(frame[target_col], errors="coerce")
    xdf = frame.drop(columns=[target_col, *drop_cols], errors="ignore")
    xdf = xdf.apply(pd.to_numeric, errors="coerce")
    xdf = xdf.dropna(axis=1, how="all")
    mask = np.isfinite(xdf.to_numpy(dtype=np.float64)).all(axis=1)
    mask &= np.isfinite(y.to_numpy(dtype=np.float64))
    xdf = xdf.loc[mask]
    y = y.loc[mask]

    # Constant columns carry no information and make ARD parameters pointless.
    varying = xdf.nunique(dropna=False) > 1
    xdf = xdf.loc[:, varying]
    return (
        xdf.to_numpy(dtype=np.float64),
        y.to_numpy(dtype=np.float64),
        [str(c) for c in xdf.columns],
    )


def _parse_protein(zf):
    frame = _read_csv(zf, _member(zf, "CASP.csv"))
    return _xy_numeric(frame, "RMSD")


def _parse_superconductivity(zf):
    frame = _read_csv(zf, _member(zf, "train.csv"))
    return _xy_numeric(frame, "critical_temp")


def _parse_appliances(zf):
    frame = _read_csv(zf, _member(zf, "energydata_complete.csv"))
    # date is non-numeric; rv1/rv2 are intentionally random control variables.
    return _xy_numeric(frame, "Appliances", drop=("date", "rv1", "rv2"))


def _parse_bike(zf):
    frame = _read_csv(zf, _member(zf, "hour.csv"))
    # casual + registered sum exactly to cnt and would leak the target.
    return _xy_numeric(
        frame, "cnt",
        drop=("instant", "dteday", "casual", "registered"),
    )


def _parse_gas_turbine(zf):
    members = sorted(
        name for name in zf.namelist()
        if re.fullmatch(r"gt_20\d\d\.csv", Path(name).name.lower())
    )
    if not members:
        raise FileNotFoundError("no gt_20XX.csv files found in archive")
    frames = []
    for member in members:
        frame = _read_csv(zf, member)
        year_match = re.search(r"(20\d\d)", Path(member).name)
        if year_match:
            frame.insert(0, "year", int(year_match.group(1)))
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    # NOX is the second emission target, not a sensor input for the CO task.
    return _xy_numeric(frame, "CO", drop=("NOX",))


PARSERS = {
    "protein": _parse_protein,
    "superconductivity": _parse_superconductivity,
    "appliances": _parse_appliances,
    "bike": _parse_bike,
    "gas_turbine": _parse_gas_turbine,
}


def _download(url: str, destination: Path, attempts: int = 5) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "Correlation-Block-SGPR/real-data-benchmark"}
    )
    last_error = None
    for attempt in range(1, int(attempts) + 1):
        print(f"Downloading {url}  attempt {attempt}/{attempts}")
        try:
            if tmp.exists():
                tmp.unlink()
            with urllib.request.urlopen(request, timeout=180) as response:
                with tmp.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
            with zipfile.ZipFile(tmp) as zf:
                bad_member = zf.testzip()
                if bad_member is not None:
                    raise zipfile.BadZipFile(
                        f"CRC failed for archive member {bad_member!r}"
                    )
            tmp.replace(destination)
            return
        except (
            http.client.IncompleteRead,
            TimeoutError,
            urllib.error.URLError,
            zipfile.BadZipFile,
        ) as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            if attempt >= int(attempts):
                break
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(
        f"failed to download a valid archive from {url} after {attempts} attempts"
    ) from last_error


def _validate(key, X, y, feature_names):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"{key}: invalid shapes X={X.shape}, y={y.shape}")
    if len(X) < 7_000:
        raise ValueError(f"{key}: only {len(X)} rows; need at least 7000")
    if X.shape[1] == 0:
        raise ValueError(f"{key}: no usable features")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{key}: non-finite values remain after cleaning")
    if float(np.std(y)) <= 0.0:
        raise ValueError(f"{key}: constant target")
    if len(feature_names) != X.shape[1]:
        raise ValueError(f"{key}: feature-name count does not match X")
    return X, y


def load_raw(key: str, data_dir=DATA_DIR, force=False):
    """Return cleaned real data ``(X, y, metadata)`` for one dataset."""
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; choose from {DATASET_KEYS}")
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    cache = cache_path(key, data_dir)
    archive = raw_dir / f"{key}.zip"
    entry = DATASETS[key]

    if cache.exists() and not force:
        d = np.load(cache, allow_pickle=False)
        meta = json.loads(str(d["metadata"]))
        if (
            meta.get("uci_id") != entry["uci_id"]
            or meta.get("source_url") != entry["url"]
            or not str(meta.get("raw_archive_sha256", "")).startswith("sha256:")
        ):
            raise RuntimeError(
                f"{cache} lacks current source provenance; rebuild with --force"
            )
        X, y = _validate(key, d["X"], d["y"], list(d["feature_names"]))
        return X, y, meta

    if force or not archive.exists():
        _download(entry["url"], archive)

    try:
        with zipfile.ZipFile(archive) as zf:
            X, y, feature_names = PARSERS[entry["parser"]](zf)
    except zipfile.BadZipFile as exc:
        if force:
            raise RuntimeError(
                f"{archive} is not a valid UCI zip archive"
            ) from exc
        archive.unlink(missing_ok=True)
        _download(entry["url"], archive)
        with zipfile.ZipFile(archive) as zf:
            X, y, feature_names = PARSERS[entry["parser"]](zf)

    X, y = _validate(key, X, y, feature_names)
    meta = {
        "key": key,
        "label": entry["label"],
        "uci_id": entry["uci_id"],
        "source_url": entry["url"],
        "raw_archive_sha256": _sha256(archive),
        "n": int(len(X)),
        "d": int(X.shape[1]),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        X=X,
        y=y,
        feature_names=np.asarray(feature_names, dtype=str),
        metadata=np.asarray(json.dumps(meta)),
    )
    print(f"Cached {key}: X={X.shape}, y={y.shape} -> {cache}")
    return X, y, meta


def prepare_dataset(key: str, seed: int, n_train=5_000, n_test=2_000,
                    data_dir=DATA_DIR):
    """Random split, train standardization, PC1 ordering."""
    X, y, meta = load_raw(key, data_dir=data_dir)
    need = int(n_train) + int(n_test)
    if len(X) < need:
        raise ValueError(f"{key}: {len(X)} rows available, {need} required")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:need]
    x_tr = X[idx[:n_train]].astype(np.float64, copy=True)
    y_tr = y[idx[:n_train]].astype(np.float64, copy=True)
    x_te = X[idx[n_train:]].astype(np.float64, copy=True)
    y_te = y[idx[n_train:]].astype(np.float64, copy=True)

    xm, xs = x_tr.mean(0), x_tr.std(0) + 1e-12
    ym, ys = y_tr.mean(), y_tr.std() + 1e-12
    x_tr = (x_tr - xm) / xs
    x_te = (x_te - xm) / xs
    y_tr = (y_tr - ym) / ys
    y_te = (y_te - ym) / ys

    from sklearn.decomposition import PCA
    order = np.argsort(PCA(n_components=1).fit_transform(x_tr)[:, 0])
    x_tr, y_tr = x_tr[order], y_tr[order]

    split_meta = dict(
        meta,
        n_train=int(n_train),
        n_test=int(n_test),
        seed=int(seed),
        y_train_mean=float(ym),
        y_train_std=float(ys),
    )
    return (
        x_tr,
        y_tr.reshape(-1, 1),
        x_te,
        y_te.reshape(-1, 1),
        split_meta,
    )


def download_all(keys=DATASET_KEYS, data_dir=DATA_DIR, force=False):
    rows = []
    for key in keys:
        X, y, meta = load_raw(key, data_dir=data_dir, force=force)
        rows.append(meta)
        print(
            f"[OK] {key:18s} N={len(X):6d} D={X.shape[1]:3d} "
            f"target std={np.std(y):.6g}"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_KEYS))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-splits", action="store_true")
    args = parser.parse_args()

    unknown = sorted(set(args.datasets) - set(DATASET_KEYS))
    if unknown:
        parser.error(f"unknown datasets: {unknown}")
    download_all(args.datasets, args.data_dir, force=args.force)
    if args.check_splits:
        for key in args.datasets:
            xtr, ytr, xte, yte, _ = prepare_dataset(
                key, seed=42, data_dir=args.data_dir
            )
            print(
                f"[SPLIT] {key:18s} train={xtr.shape}/{ytr.shape} "
                f"test={xte.shape}/{yte.shape}"
            )


if __name__ == "__main__":
    main()
