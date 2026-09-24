"""Similarity-signal parsing, canonicalization, grouping, and coverage.

A signal is the full score curve behind one AUC value. Curves arrive from three
places: embedded JSON in the result parquet, a per-task JSONL sidecar, and a
campaign-level JSONL file. All three are normalized to one layout here.

The central invariant is that condition and seed pairs stay distinct from
setting and seed pairs. Two comparisons can share a setting while differing in
training dataset; collapsing them would average unrelated conditions together.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from manifold_repsim.resi.campaign import Campaign, ManifestEntry

from ..identifiers import canonical_comparison_id, parse_comparison_id

SIGNAL_COLUMNS = [
    "domain",
    "benchmark",
    "dataset",
    "source",
    "signal_source",
    "source_file",
    "comparison_id",
    "canonical_comparison_id",
    "metric",
    "base_metric",
    "param_name",
    "param_index",
    "param_value",
    "score",
    "auc_value",
    "integration_method",
    "logscale",
    "setting_pair",
    "condition_pair",
    "seed_pair",
    "condition_seed_pair",
    "architecture_pair",
    "source_setting",
    "source_architecture",
    "source_train_dataset",
    "source_seed",
    "source_representation_dataset",
    "source_layer_id",
    "target_setting",
    "target_architecture",
    "target_train_dataset",
    "target_seed",
    "target_representation_dataset",
    "target_layer_id",
]

# Later sources win when the same curve appears more than once.
SIGNAL_SOURCE_PRIORITY = {"campaign_jsonl": 0, "manifest_jsonl": 1, "embedded": 2}

SIGNAL_DEDUPE_KEYS = [
    "domain",
    "benchmark",
    "dataset",
    "metric",
    "canonical_comparison_id",
    "param_name",
    "param_value",
]

SEED_GROUP_KEYS = [
    "domain",
    "benchmark",
    "dataset",
    "metric",
    "architecture_pair",
    "setting_pair",
    "condition_pair",
    "seed_pair",
    "condition_seed_pair",
    "param_name",
    "param_value",
    "logscale",
]

# Training-dataset suffixes carry no information in a condition label.
CONDITION_DATASET_SUFFIXES = (
    "_ImageNet100DataModule",
    "_IN100_DataModule",
    "_DataModule",
)


def _canonical_pair(source: object, target: object) -> str:
    return "-vs-".join(sorted([str(source), str(target)]))


def _seed_label(value: object) -> str:
    if pd.isna(value):
        return "s?"
    number = float(value)
    return f"s{int(number)}" if number.is_integer() else f"s{number:g}"


def _seed_pair(
    source_setting: object,
    target_setting: object,
    source_seed: object,
    target_seed: object,
) -> str:
    def key(setting: object, seed: object):
        numeric = float(seed) if not pd.isna(seed) else float("inf")
        return (str(setting), numeric, _seed_label(seed))

    first, second = sorted(
        [key(source_setting, source_seed), key(target_setting, target_seed)]
    )
    return f"{first[2]}-vs-{second[2]}"


def _condition_label(setting: object, train_dataset: object) -> str:
    setting_text = "" if pd.isna(setting) else str(setting)
    dataset_text = "" if pd.isna(train_dataset) else str(train_dataset)
    for suffix in CONDITION_DATASET_SUFFIXES:
        if dataset_text.endswith(suffix):
            dataset_text = dataset_text.removesuffix(suffix)
            break
    if setting_text and dataset_text:
        return f"{setting_text}:{dataset_text}"
    return setting_text or dataset_text


def _condition_seed_pair(
    source_condition: str,
    target_condition: str,
    source_seed: object,
    target_seed: object,
) -> str:
    def key(condition: str, seed: object):
        numeric = float(seed) if not pd.isna(seed) else float("inf")
        return (condition, numeric, _seed_label(seed))

    first, second = sorted(
        [
            key(source_condition, source_seed),
            key(target_condition, target_seed),
        ]
    )
    return f"{first[0]}@{first[2]}-vs-{second[0]}@{second[2]}"


def _signal_rows(
    record: dict[str, Any],
    comparison_id: str,
    metric: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expand one stored curve into one row per swept parameter value.

    A record whose parameter and score lists disagree in length is dropped: it
    cannot be interpreted, and guessing an alignment would fabricate data.
    """
    params = record.get("param_values") or []
    scores = record.get("scores") or []
    if len(params) != len(scores):
        return []
    parsed = parse_comparison_id(comparison_id)
    source_setting = parsed.get("source_setting", "")
    target_setting = parsed.get("target_setting", "")
    source_seed = parsed.get("source_seed", np.nan)
    target_seed = parsed.get("target_seed", np.nan)
    setting_pair = _canonical_pair(source_setting, target_setting)
    source_condition = _condition_label(
        source_setting, parsed.get("source_train_dataset", "")
    )
    target_condition = _condition_label(
        target_setting, parsed.get("target_train_dataset", "")
    )
    condition_pair = _canonical_pair(source_condition, target_condition)
    seed_pair = _seed_pair(source_setting, target_setting, source_seed, target_seed)
    condition_seed_pair = _condition_seed_pair(
        source_condition, target_condition, source_seed, target_seed
    )
    architecture_pair = _canonical_pair(
        parsed.get("source_architecture", ""), parsed.get("target_architecture", "")
    )
    rows = []
    canonical_id = canonical_comparison_id(comparison_id, metric)
    for index, (param, score) in enumerate(zip(params, scores)):
        rows.append(
            {
                **metadata,
                "comparison_id": comparison_id,
                "canonical_comparison_id": canonical_id,
                "metric": metric,
                "base_metric": record.get("base_metric", ""),
                "param_name": record.get("param_name", ""),
                "param_index": index,
                "param_value": pd.to_numeric(param, errors="coerce"),
                "score": pd.to_numeric(score, errors="coerce"),
                "auc_value": record.get("auc_value"),
                "integration_method": record.get("integration_method", ""),
                "logscale": record.get("logscale", False),
                "setting_pair": setting_pair,
                "condition_pair": condition_pair,
                "seed_pair": seed_pair,
                "condition_seed_pair": condition_seed_pair,
                "architecture_pair": architecture_pair,
                **parsed,
            }
        )
    return rows


def load_embedded_signals(entries: Iterable[ManifestEntry]) -> pd.DataFrame:
    """Read curves embedded in each task's result parquet."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        path = Path(entry.result_path)
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        if (
            "similarity_signal" not in frame
            or "id" not in frame
            or "metric" not in frame
        ):
            continue
        metadata = {
            "domain": entry.domain,
            "benchmark": entry.benchmark,
            "dataset": entry.dataset,
            "source": "baseline" if entry.kind == "baseline" else "manifold",
            "signal_source": "embedded",
            "source_file": str(path),
        }
        for _, row in frame[frame["similarity_signal"].notna()].iterrows():
            try:
                record = json.loads(row["similarity_signal"])
            except (json.JSONDecodeError, TypeError):
                continue
            rows.extend(
                _signal_rows(record, str(row["id"]), str(row["metric"]), metadata)
            )
    return pd.DataFrame(rows).reindex(columns=SIGNAL_COLUMNS)


def _load_signal_jsonl(path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            comparison_id = str(record.get("comparison_id", ""))
            metric = str(record.get("similarity_measure") or record.get("metric") or "")
            rows.extend(_signal_rows(record, comparison_id, metric, metadata))
    return rows


def load_manifest_legacy_signals(entries: Iterable[ManifestEntry]) -> pd.DataFrame:
    """Read curves from the JSONL sidecars each manifest entry declares."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        metadata = {
            "domain": entry.domain,
            "benchmark": entry.benchmark,
            "dataset": entry.dataset,
            "source": "baseline" if entry.kind == "baseline" else "manifold",
            "signal_source": "manifest_jsonl",
        }
        for value in entry.legacy_signal_paths:
            path = Path(value).expanduser()
            if not path.is_file():
                continue
            rows.extend(
                _load_signal_jsonl(path, {**metadata, "source_file": str(path)})
            )
    return pd.DataFrame(rows).reindex(columns=SIGNAL_COLUMNS)


def load_legacy_signals(campaign: Campaign) -> pd.DataFrame:
    """Read curves from campaign-level JSONL files predating embedded signals."""
    rows: list[dict[str, Any]] = []
    for value in campaign.legacy_signal_files:
        path = Path(os.path.expandvars(value)).expanduser()
        if not path.is_absolute():
            path = campaign.path.parent / path
        if not path.is_file():
            continue
        metadata = {
            "domain": campaign.domain,
            "benchmark": "legacy",
            "dataset": "legacy",
            "source": "manifold",
            "signal_source": "campaign_jsonl",
            "source_file": str(path),
        }
        rows.extend(_load_signal_jsonl(path, metadata))
    return pd.DataFrame(rows).reindex(columns=SIGNAL_COLUMNS)


def combine_signals(*frames: pd.DataFrame) -> pd.DataFrame:
    """Merge signal sources, keeping the highest-priority copy of each curve."""
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)
    combined = pd.concat(nonempty, ignore_index=True).reindex(columns=SIGNAL_COLUMNS)
    combined["_source_priority"] = (
        combined["signal_source"].map(SIGNAL_SOURCE_PRIORITY).fillna(-1)
    )
    combined = combined.sort_values("_source_priority")
    combined = combined.drop_duplicates(SIGNAL_DEDUPE_KEYS, keep="last").drop(
        columns="_source_priority"
    )
    return combined.sort_values(
        [
            "domain",
            "benchmark",
            "dataset",
            "metric",
            "canonical_comparison_id",
            "param_index",
        ]
    ).reset_index(drop=True)


def exclude_known_missing_signals(
    signals: pd.DataFrame, campaign: Campaign
) -> pd.DataFrame:
    """Drop curves involving a model the campaign declares as missing."""
    if signals.empty or not campaign.known_missing_models:
        return signals
    excluded = pd.Series(False, index=signals.index)
    for missing in campaign.known_missing_models:
        entry_mask = signals["benchmark"].eq(missing.benchmark) & signals["dataset"].eq(
            missing.dataset
        )
        source_mask = (
            signals["source_setting"].eq(missing.setting)
            & signals["source_architecture"].eq(missing.architecture)
            & signals["source_train_dataset"].eq(missing.train_dataset)
            & pd.to_numeric(signals["source_seed"], errors="coerce").eq(missing.seed)
        )
        target_mask = (
            signals["target_setting"].eq(missing.setting)
            & signals["target_architecture"].eq(missing.architecture)
            & signals["target_train_dataset"].eq(missing.train_dataset)
            & pd.to_numeric(signals["target_seed"], errors="coerce").eq(missing.seed)
        )
        excluded |= entry_mask & (source_mask | target_mask)
    return signals.loc[~excluded].copy()


def signal_input_coverage(entries: Iterable[ManifestEntry]) -> pd.DataFrame:
    """Report where each compute task's curves came from, or that none exist."""
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.kind != "compute":
            continue
        embedded_records = 0
        result_path = Path(entry.result_path)
        if result_path.is_file():
            frame = pd.read_parquet(result_path)
            if "similarity_signal" in frame:
                embedded_records = int(frame["similarity_signal"].notna().sum())
        found_paths = [
            Path(value) for value in entry.legacy_signal_paths if Path(value).is_file()
        ]
        jsonl_records = 0
        for path in found_paths:
            with path.open("r", encoding="utf-8") as handle:
                jsonl_records += sum(1 for line in handle if line.strip())
        if embedded_records:
            state = "available_embedded"
            source = "embedded"
        elif jsonl_records:
            state = "available_legacy"
            source = "manifest_jsonl"
        else:
            state = "signals_missing"
            source = ""
        rows.append(
            {
                "record_type": "signal_input",
                "kind": "manifold",
                "index": entry.index,
                "state": state,
                "domain": entry.domain,
                "benchmark": entry.benchmark,
                "dataset": entry.dataset,
                "architectures": ",".join(entry.architectures),
                "metric": "",
                "quality_measure": "",
                "rows": embedded_records + jsonl_records,
                "finite_values": embedded_records + jsonl_records,
                "nan_values": 0,
                "signal_source": source,
                "signal_files_declared": len(entry.legacy_signal_paths),
                "signal_files_found": len(found_paths),
                "excluded_model": "",
                "exclusion_reason": "",
                "error": "",
                "result_path": entry.result_path,
                "full_csv_path": entry.full_csv_path,
            }
        )
    return pd.DataFrame(rows)


def summarize_signals(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average curves within a seed pair, then across seed pairs.

    The two stages matter: the first collapses layers inside one seed pair, so
    the second's ``n_seeds`` counts distinct seed pairs rather than rows. The
    seed-bearing columns are the only ones dropped between stages, which is what
    keeps distinct conditions from being averaged together.
    """
    if signals.empty:
        grouped_columns = [
            column
            for column in SEED_GROUP_KEYS
            if column not in {"seed_pair", "condition_seed_pair"}
        ]
        grouped_columns += ["mean_score", "std_score", "n_seeds"]
        return (
            pd.DataFrame(columns=[*SEED_GROUP_KEYS, "score"]),
            pd.DataFrame(columns=grouped_columns),
        )
    signals = signals.drop_duplicates(SIGNAL_DEDUPE_KEYS, keep="last")
    by_seed = signals.groupby(SEED_GROUP_KEYS, dropna=False, as_index=False)[
        "score"
    ].mean()
    group_keys = [
        key
        for key in SEED_GROUP_KEYS
        if key not in {"seed_pair", "condition_seed_pair"}
    ]
    grouped = (
        by_seed.groupby(group_keys, dropna=False)["score"]
        .agg(mean_score="mean", std_score="std", n_seeds="count")
        .reset_index()
    )
    return by_seed.sort_values(SEED_GROUP_KEYS), grouped.sort_values(group_keys)


GROUPED_COMPARISON_COLUMNS = [
    "domain",
    "benchmark",
    "dataset",
    "metric",
    "architecture",
    "param_name",
    "condition_pair",
    "param_value",
    "logscale",
    "mean_score",
    "std_score",
    "n_seeds",
]

_GROUPED_COMPARISON_REQUIRED = {
    "domain",
    "benchmark",
    "dataset",
    "metric",
    "source_architecture",
    "target_architecture",
    "param_name",
    "param_value",
    "score",
    "logscale",
}


def _coerce_logscale(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value) if not pd.isna(value) else False


def summarize_grouped_signal_comparisons(signals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate same-architecture curves per condition pair for plotting.

    Restricted to comparisons whose two models share an architecture, because a
    cross-architecture curve answers a different question and would distort the
    per-architecture view this summary feeds.
    """
    if signals.empty or not _GROUPED_COMPARISON_REQUIRED.issubset(signals.columns):
        return pd.DataFrame(columns=GROUPED_COMPARISON_COLUMNS)

    work = signals.copy()
    for derived, fallback in (
        ("condition_pair", "setting_pair"),
        ("condition_seed_pair", "seed_pair"),
    ):
        if derived not in work.columns:
            raise ValueError(
                f"Signal frame is missing '{derived}'. Falling back to "
                f"'{fallback}' would silently merge distinct conditions."
            )
    work = work[
        work["source_architecture"].notna()
        & work["target_architecture"].notna()
        & work["source_architecture"].ne("")
        & work["source_architecture"].eq(work["target_architecture"])
    ].copy()
    work["param_value"] = pd.to_numeric(work["param_value"], errors="coerce")
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work[np.isfinite(work["param_value"]) & np.isfinite(work["score"])]
    if work.empty:
        return pd.DataFrame(columns=GROUPED_COMPARISON_COLUMNS)
    work["architecture"] = work["source_architecture"].astype(str)
    work["logscale"] = work["logscale"].map(_coerce_logscale)

    group_keys = [
        "domain",
        "benchmark",
        "dataset",
        "metric",
        "architecture",
        "param_name",
        "condition_pair",
        "param_value",
        "logscale",
    ]
    seed_means = work.groupby(
        [*group_keys, "condition_seed_pair"], dropna=False, as_index=False
    )["score"].mean()
    return (
        seed_means.groupby(group_keys, dropna=False)["score"]
        .agg(mean_score="mean", std_score="std", n_seeds="count")
        .reset_index()
        .sort_values(group_keys)
        .reindex(columns=GROUPED_COMPARISON_COLUMNS)
    )


__all__ = [
    "GROUPED_COMPARISON_COLUMNS",
    "SEED_GROUP_KEYS",
    "SIGNAL_COLUMNS",
    "SIGNAL_DEDUPE_KEYS",
    "SIGNAL_SOURCE_PRIORITY",
    "combine_signals",
    "exclude_known_missing_signals",
    "load_embedded_signals",
    "load_legacy_signals",
    "load_manifest_legacy_signals",
    "signal_input_coverage",
    "summarize_grouped_signal_comparisons",
    "summarize_signals",
]
