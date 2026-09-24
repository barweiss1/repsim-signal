"""Reading ReSi result files into one normalized long table.

Loading is manifest-driven: only the files an entry names are read. Result
directories are never scanned, so an unrelated leftover parquet cannot enter the
analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from manifold_repsim.resi.campaign import Campaign, KnownMissingModel, ManifestEntry

from .config import quality_direction
from ..identifiers import parse_comparison_id

NORMALIZED_COLUMNS = [
    "domain",
    "benchmark",
    "dataset",
    "architecture",
    "observation_id",
    "metric",
    "quality_measure",
    "value",
    "direction",
    "source",
    "identifier",
    "representation_dataset",
    "model",
    "functional_similarity_measure",
    "source_file",
]

# Together these identify one comparable observation of one measure.
OBSERVATION_COLUMNS = [
    "identifier",
    "representation_dataset",
    "model",
    "functional_similarity_measure",
]


def select_measures(
    frame: pd.DataFrame,
    measures: Iterable[str] | None,
    *,
    require_all: bool = True,
) -> pd.DataFrame:
    """Restrict a frame with a ``metric`` column to an explicit measure list.

    Ranks are only meaningful between the measures actually being compared, so
    a run whose measure coverage differs across architectures has to state
    which set it is reporting. Passing ``None`` keeps every measure present.

    With ``require_all``, a requested measure that appears nowhere is an error
    rather than an empty selection: quietly ranking eight measures when nine
    were asked for is how an incomplete run gets published as a complete one.
    Signal frames pass ``require_all=False``, since only the AUC measures ever
    produce signals and the rest are legitimately absent there.
    """
    if measures is None:
        return frame
    requested = list(dict.fromkeys(str(measure) for measure in measures))
    if not requested:
        raise ValueError("Measure selection is empty")
    if frame.empty:
        return frame
    metrics = frame["metric"].astype(str)
    missing = [measure for measure in requested if measure not in set(metrics)]
    if missing and require_all:
        raise ValueError(
            f"Requested measure(s) absent from the loaded results: {', '.join(missing)}"
        )
    return frame[metrics.isin(set(requested))].copy()


def _read_full_csv(entry: ManifestEntry) -> pd.DataFrame:
    path = Path(entry.full_csv_path)
    frame = pd.read_csv(path)
    return frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed:")]


def normalize_entry(
    entry: ManifestEntry, quality_measures: Iterable[str]
) -> pd.DataFrame:
    """Read one task's full CSV into the shared normalized column layout.

    Output-correlation experiments write their value in a ``corr`` column rather
    than ``value``, so either is accepted.
    """
    frame = _read_full_csv(entry)
    required = {"similarity_measure", "quality_measure", "architecture"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {entry.full_csv_path}: {sorted(missing)}")
    value_column = (
        "value"
        if "value" in frame.columns
        else "corr" if "corr" in frame.columns else None
    )
    if value_column is None:
        raise ValueError(f"No value or corr column in {entry.full_csv_path}")
    frame = frame[
        frame["quality_measure"].astype(str).isin(set(quality_measures))
    ].copy()
    frame["value"] = pd.to_numeric(frame[value_column], errors="coerce")
    frame["metric"] = frame["similarity_measure"].astype(str)
    frame["domain"] = entry.domain
    frame["benchmark"] = entry.benchmark
    frame["dataset"] = entry.dataset
    frame["source"] = "baseline" if entry.kind == "baseline" else "manifold"
    frame["source_file"] = entry.full_csv_path
    for column in OBSERVATION_COLUMNS:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    frame["architecture"] = frame["architecture"].fillna("").astype(str)
    frame["direction"] = frame["quality_measure"].map(quality_direction)
    frame["observation_id"] = (
        frame[OBSERVATION_COLUMNS].astype(str).agg("|".join, axis=1)
    )
    return frame[NORMALIZED_COLUMNS]


def normalize_manifests(
    entries: Iterable[ManifestEntry],
    campaign: Campaign,
    *,
    allow_incomplete: bool,
    quality_measures: Iterable[str] | None = None,
    exclude_functional_measures: Iterable[str] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize every manifest entry and report per-entry input coverage.

    ``quality_measures`` narrows the campaign's own list, so an analysis can
    report on one quality measure without redefining the campaign.

    ``exclude_functional_measures`` drops correlation results computed against
    named functional similarity measures. It is resolved against what the
    results contain rather than against a declared list, so naming one a domain
    never had is harmless. This exists because a functional measure the ReSi
    archive never scored for its own measures, but that our recreated baselines
    do score, produces cases only the new measures can cover -- which reads as
    the established measures failing a coverage threshold rather than as a gap
    in the archive.
    """
    selected_quality = tuple(
        quality_measures if quality_measures is not None else campaign.quality_measures
    )
    unknown = sorted(set(selected_quality) - set(campaign.quality_measures))
    if unknown:
        raise ValueError(
            f"quality_measures not defined by the campaign: {', '.join(unknown)}"
        )
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for entry in entries:
        full_path = Path(entry.full_csv_path)
        result_path = Path(entry.result_path)
        if not result_path.is_file():
            state = "missing"
        elif not full_path.is_file():
            state = "incomplete"
        else:
            state = "complete"
        error = ""
        normalized = pd.DataFrame(columns=NORMALIZED_COLUMNS)
        if state == "complete":
            try:
                normalized = normalize_entry(entry, selected_quality)
                frames.append(normalized)
            except Exception as exc:
                state = "incomplete"
                error = str(exc)
        if state != "complete" and not allow_incomplete:
            raise RuntimeError(
                f"{state.capitalize()} analysis input for manifest index "
                f"{entry.index}: {entry.full_csv_path}"
                f"{': ' + error if error else ''}"
            )
        coverage_rows.append(
            {
                "kind": entry.kind,
                "index": entry.index,
                "state": state,
                "domain": entry.domain,
                "benchmark": entry.benchmark,
                "dataset": entry.dataset,
                "architectures": ",".join(entry.architectures),
                "rows": len(normalized),
                "finite_values": (
                    int(np.isfinite(normalized["value"]).sum())
                    if not normalized.empty
                    else 0
                ),
                "nan_values": (
                    int(normalized["value"].isna().sum()) if not normalized.empty else 0
                ),
                "error": error,
                "result_path": entry.result_path,
                "full_csv_path": entry.full_csv_path,
            }
        )
    values = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=NORMALIZED_COLUMNS)
    )
    dropped_functional = tuple(exclude_functional_measures)
    if dropped_functional and not values.empty:
        values = values[
            ~values["functional_similarity_measure"]
            .astype(str)
            .isin(set(dropped_functional))
        ].reset_index(drop=True)
    if not values.empty:
        values = values.drop_duplicates(
            subset=[
                "domain",
                "benchmark",
                "dataset",
                "architecture",
                "observation_id",
                "metric",
                "quality_measure",
            ],
            keep="last",
        ).sort_values(
            [
                "domain",
                "benchmark",
                "dataset",
                "architecture",
                "quality_measure",
                "metric",
                "observation_id",
            ]
        )
    return values, pd.DataFrame(coverage_rows)


def _matches_known_missing_model(
    parsed: dict[str, Any],
    prefix: str,
    missing: KnownMissingModel,
) -> bool:
    seed = parsed.get(f"{prefix}_seed", np.nan)
    return (
        parsed.get(f"{prefix}_setting") == missing.setting
        and parsed.get(f"{prefix}_architecture") == missing.architecture
        and parsed.get(f"{prefix}_train_dataset") == missing.train_dataset
        and not pd.isna(seed)
        and int(seed) == missing.seed
    )


def _entry_matches_known_missing(
    entry: ManifestEntry, missing: KnownMissingModel
) -> bool:
    return (
        entry.benchmark == missing.benchmark
        and entry.dataset == missing.dataset
        and missing.architecture in entry.architectures
    )


def known_missing_coverage(
    entries: Iterable[ManifestEntry],
    campaign: Campaign,
) -> pd.DataFrame:
    """Report each declared missing model as observed or merely declared.

    A model declared missing but never actually absent is reported too, so a
    stale declaration is visible instead of silently suppressing nothing.
    """
    rows: list[dict[str, Any]] = []
    for missing in campaign.known_missing_models:
        model_id = (
            f"{missing.setting}__{missing.architecture}__"
            f"{missing.train_dataset}__{missing.seed}"
        )
        for entry in entries:
            if not _entry_matches_known_missing(entry, missing):
                continue
            path = Path(entry.result_path)
            affected = pd.DataFrame()
            if path.is_file():
                frame = pd.read_parquet(path)
                if {"id", "metric_value"}.issubset(frame.columns):
                    parsed = frame["id"].astype(str).map(parse_comparison_id)
                    model_mask = parsed.map(
                        lambda value: _matches_known_missing_model(
                            value, "source", missing
                        )
                        or _matches_known_missing_model(value, "target", missing)
                    )
                    affected = frame.loc[
                        model_mask
                        & pd.to_numeric(frame["metric_value"], errors="coerce").isna()
                    ]
            metrics = (
                ",".join(sorted(affected["metric"].dropna().astype(str).unique()))
                if "metric" in affected
                else ""
            )
            rows.append(
                {
                    "record_type": "known_missing_model",
                    "kind": "baseline" if entry.kind == "baseline" else "manifold",
                    "index": entry.index,
                    "state": (
                        "excluded_known_missing"
                        if len(affected)
                        else "declared_not_observed"
                    ),
                    "domain": entry.domain,
                    "benchmark": entry.benchmark,
                    "dataset": entry.dataset,
                    "architectures": missing.architecture,
                    "metric": metrics,
                    "quality_measure": "",
                    "rows": len(affected),
                    "finite_values": 0,
                    "nan_values": len(affected),
                    "excluded_model": model_id,
                    "exclusion_reason": missing.reason,
                    "error": "",
                    "result_path": entry.result_path,
                    "full_csv_path": entry.full_csv_path,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "NORMALIZED_COLUMNS",
    "OBSERVATION_COLUMNS",
    "known_missing_coverage",
    "normalize_entry",
    "normalize_manifests",
    "select_measures",
]
