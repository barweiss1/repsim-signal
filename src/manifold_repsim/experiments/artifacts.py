"""Atomic artifact helpers shared by the MDS experiments.

Every write goes through a temporary file in the destination directory and an
``os.replace``, so an interrupted run leaves either the previous artifact or
the new one, never a truncated file. Both experiment packages read and write
their checkpoints through here; a neutral module under ``experiments/``
(alongside ``mds.py``) keeps them from importing each other.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


def save_npz(path: Path, **arrays: Any) -> None:
    """Atomically write one compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def save_json(path: Path, value: Any) -> None:
    """Atomically write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def save_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    """Atomically write dictionaries to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            suffix=".csv",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Load detached arrays from one NumPy archive."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def load_json(path: Path) -> Any:
    """Load one JSON document written by :func:`save_json`."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
