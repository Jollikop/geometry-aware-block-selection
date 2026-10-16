#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reading and writing per-run results, safe for parallel Slurm workers.

Both stacks write one ``metrics.json`` per run and append one row to a shared
CSV. The two stacks previously had a function called ``append_row`` each, with
different argument orders and opposite semantics; they are now
:func:`append_row_fixed_schema` and :func:`upsert_row_evolving_schema`, and
which one a stack wants is a property of its result schema:

* the collapsed benchmark declares every column up front, so it appends;
* the variational benchmark adds columns for optional predictors that only some
  methods produce, so it rewrites the file under the lock when the header grows
  and deduplicates by ``config_id``.

Every writer holds an exclusive ``flock`` for the whole read-modify-write, and
decides whether a header is needed *under* that lock, so concurrent workers
cannot interleave a header.
"""
from __future__ import annotations

import csv
import fcntl
import json
import os
from pathlib import Path


def results_csv_path(logs_dir, implementation):
    return Path(logs_dir) / f"results_per_run_{implementation}.csv"


def append_row_fixed_schema(logs_dir, row, implementation, fieldnames):
    """Append one row against a fixed, fully declared column set.

    Unknown keys in ``row`` are dropped rather than silently widening the file,
    which is what keeps a parallel append safe without rewriting.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = results_csv_path(logs_dir, implementation)
    with open(path, "a", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            need_header = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                    extrasaction="ignore")
            if need_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def upsert_row_evolving_schema(logs_dir, row, implementation, *, replace=False):
    """Insert or update one row, extending the header when a new key appears.

    Returns True when the file changed. An append-only writer that matched each
    row against the *first* row's header could silently drop columns or produce
    a ragged file once an optional predictor started emitting extra fields.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = results_csv_path(logs_dir, implementation)
    with open(path, "a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            reader = csv.DictReader(handle)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader) if existing_fields else []
            fields = existing_fields + [
                key for key in row if key not in existing_fields
            ]

            config_id = str(row.get("config_id", ""))
            match = next(
                (i for i, old in enumerate(existing_rows)
                 if config_id and old.get("config_id") == config_id),
                None,
            )
            schema_changed = fields != existing_fields
            if match is not None and not replace and not schema_changed:
                return False

            if not existing_fields or schema_changed or match is not None:
                if match is not None:
                    existing_rows[match] = row
                else:
                    existing_rows.append(row)
                handle.seek(0)
                handle.truncate()
                writer = csv.DictWriter(handle, fieldnames=fields,
                                        extrasaction="ignore")
                writer.writeheader()
                writer.writerows(existing_rows)
            else:
                handle.seek(0, os.SEEK_END)
                writer = csv.DictWriter(handle, fieldnames=existing_fields,
                                        extrasaction="ignore")
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            return True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def collect_metrics(runs_dir, *, implementation=None, config_ids=None,
                    methods=None, require=("rmse",)):
    """Load every ``metrics.json`` under ``runs_dir`` that passes the filters.

    Returns ``(rows, skipped)``. ``skipped`` lists ``(path, reason)`` for files
    that could not be read; the callers print it rather than dropping runs
    silently, which is how a corrupted result used to disappear from a table
    without anyone noticing.
    """
    config_ids = set(config_ids) if config_ids is not None else None
    rows, skipped = [], []
    for path in sorted(Path(runs_dir).glob("**/metrics.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            skipped.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        if any(key not in row for key in require):
            skipped.append((path, f"missing {require}"))
            continue
        if implementation is not None and row.get("implementation") != implementation:
            continue
        if config_ids is not None and row.get("config_id") not in config_ids:
            continue
        if methods is not None and row.get("method") not in methods:
            continue
        rows.append(row)
    return rows, skipped


def save_run(runs_dir, result, curve=None, *, extra_arrays=None):
    """Write ``metrics.json`` (and an optional loss curve) for one run."""
    import numpy as np

    out = (
        Path(runs_dir)
        / result["kernel"]
        / result["dataset"]
        / f"M{result['M']}"
        / result["method"]
        / f"seed{result['seed']}"
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    if curve is not None:
        np.savez_compressed(out / "curve.npz", step=curve[:, 0], loss=curve[:, 1])
    for name, arrays in (extra_arrays or {}).items():
        np.savez_compressed(out / name, **arrays)
    return out
