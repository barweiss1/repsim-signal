"""Manifest status inspection.

Reads generated result files only. This module imports no ReSi code, so a run
can be inspected from any machine that can see the result tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .campaign import measures_present, read_manifest, result_metric_column

STATUS_COLUMNS = (
    "index",
    "kind",
    "state",
    "domain",
    "benchmark",
    "dataset",
    "architectures",
    "missing_measures",
    "error",
    "nan_measures",
    "result_path",
    "full_csv_path",
)


def manifest_status(manifest: str | Path) -> pd.DataFrame:
    """Classify every task in a manifest as missing, incomplete, or complete."""
    rows: list[dict[str, Any]] = []
    for entry in read_manifest(manifest):
        result_path = Path(entry.result_path)
        full_path = Path(entry.full_csv_path)
        available_measures: set[str] = set()
        nan_measures: set[str] = set()
        error = ""
        if result_path.is_file():
            try:
                frame = pd.read_parquet(result_path, columns=None)
                available_measures = measures_present(frame)
                if entry.kind == "compute" and "metric_value" in frame:
                    metric_column = result_metric_column(frame)
                    expected_rows = frame[
                        frame[metric_column].astype(str).isin(entry.measures)
                    ]
                    nan_measures = {
                        str(value)
                        for value in expected_rows.loc[
                            expected_rows["metric_value"].isna(), metric_column
                        ]
                        .dropna()
                        .unique()
                    }
            except Exception as exc:  # A corrupt or incompatible file is incomplete.
                error = str(exc)
        missing_measures = sorted(set(entry.measures) - available_measures)
        if not result_path.is_file():
            state = "missing"
        elif error or missing_measures or nan_measures or not full_path.is_file():
            state = "incomplete"
        else:
            state = "complete"
        rows.append(
            {
                "index": entry.index,
                "kind": entry.kind,
                "state": state,
                "domain": entry.domain,
                "benchmark": entry.benchmark,
                "dataset": entry.dataset,
                "architectures": ",".join(entry.architectures),
                "missing_measures": ",".join(missing_measures),
                "error": error,
                "nan_measures": ",".join(sorted(nan_measures)),
                "result_path": entry.result_path,
                "full_csv_path": entry.full_csv_path,
            }
        )
    return pd.DataFrame(rows, columns=list(STATUS_COLUMNS))


__all__ = ["STATUS_COLUMNS", "manifest_status"]
