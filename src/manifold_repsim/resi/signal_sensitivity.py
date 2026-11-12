"""Re-reduce stored AUC signals under different sample counts and bandwidth limits.

Every ``CKArbfAUC`` comparison row stores its full similarity curve (the
``param_values`` grid and the ``scores`` measured on it) and its AUC is the plain
mean of that curve. A coarser grid or a clipped bandwidth range therefore needs
no new measurement: it is the same mean taken over a subset of the stored points.
This module holds only that arithmetic. It imports neither Torch nor ReSi, so it
is usable wherever the result parquets can be read.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from manifold_repsim.config import load_yaml_config
from manifold_repsim.resi.campaign import ManifestEntry

BASE_METRIC = "CKArbfAUC"
ID_SEPARATOR = "___"
SCRATCH_ROOT = "manifold-sensitivity"


@dataclass(frozen=True)
class SignalVariant:
    """One way of reducing a stored curve.

    ``samples`` keeps that many points spread evenly over the stored grid, both
    endpoints included. ``lower`` / ``upper`` keep the stored points whose
    parameter lies inside that interval. Both can be combined.
    """

    key: str
    samples: int | None = None
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("A signal variant needs a key")
        if self.samples is not None and self.samples < 2:
            raise ValueError(f"Variant {self.key}: samples must be at least 2")
        if (
            self.lower is not None
            and self.upper is not None
            and not self.lower < self.upper
        ):
            raise ValueError(f"Variant {self.key}: lower must be below upper")

    @property
    def metric_name(self) -> str:
        """Name under which the variant is stored and evaluated."""
        return f"{BASE_METRIC}-{self.key}"


def select_points(param_values: Sequence[float], variant: SignalVariant) -> np.ndarray:
    """Indices of the stored grid points the variant keeps, in grid order."""
    grid = np.asarray(param_values, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("A stored signal needs a one-dimensional grid of >= 2 points")
    indices = np.arange(grid.size)
    if variant.samples is not None:
        if variant.samples > grid.size:
            raise ValueError(
                f"Variant {variant.key} asks for {variant.samples} samples but the "
                f"stored grid has {grid.size}"
            )
        indices = np.unique(
            np.round(np.linspace(0, grid.size - 1, variant.samples)).astype(int)
        )
    keep = np.ones(indices.size, dtype=bool)
    if variant.lower is not None:
        keep &= grid[indices] >= variant.lower
    if variant.upper is not None:
        keep &= grid[indices] <= variant.upper
    selected = indices[keep]
    if selected.size == 0:
        raise ValueError(f"Variant {variant.key} keeps no stored grid points")
    return selected


def reduce_signal(
    param_values: Sequence[float], scores: Sequence[float], variant: SignalVariant
) -> float:
    """Plain mean of the stored scores at the points the variant keeps."""
    grid = np.asarray(param_values, dtype=float)
    values = np.asarray(scores, dtype=float)
    if grid.shape != values.shape:
        raise ValueError("Stored signal grid and scores differ in length")
    kept = values[select_points(grid, variant)]
    return float(np.mean(kept)) if np.all(np.isfinite(kept)) else math.nan


def _parse_signal(raw: Any) -> dict[str, Any] | None:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode()
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        return None
    if "param_values" not in parsed or "scores" not in parsed:
        return None
    return parsed


def _pair_key(comparison_id: str, metric: str) -> frozenset[str]:
    """Unordered pair of representation ids behind a comparison id."""
    suffix = f"{ID_SEPARATOR}{metric}"
    if not comparison_id.endswith(suffix):
        raise ValueError(f"Comparison id {comparison_id!r} does not end in {suffix!r}")
    parts = comparison_id[: -len(suffix)].split(ID_SEPARATOR)
    if len(parts) != 2:
        raise ValueError(
            f"Cannot split comparison id {comparison_id!r} into two representations"
        )
    return frozenset(parts)


def derive_variant_rows(
    store: pd.DataFrame,
    variants: Iterable[SignalVariant],
    *,
    base_metric: str = BASE_METRIC,
    mask_metric: str | None = None,
) -> pd.DataFrame:
    """Result-store rows for each variant, built from the base measure's signals.

    ``mask_metric`` names the reported measure the variants stand in for. A
    comparison that measure could not score (NaN, as on a collapsed checkpoint)
    stays NaN in every variant, so a variant is judged on the same comparisons as
    the measure it replaces rather than on extra ones its source copy happened to
    score.

    The store keeps both directions of a symmetric comparison but attaches the
    signal to only one of them, so the mirror row takes its value from its
    partner. A base row that is NaN stays NaN: a failed comparison is not turned
    into a number by a different reduction.
    """
    base = store[store["metric"] == base_metric]
    if base.empty:
        raise ValueError(f"The store has no rows for measure {base_metric}")
    signals: dict[frozenset[str], dict[str, Any]] = {}
    for comparison_id, raw in base["similarity_signal"].items():
        signal = _parse_signal(raw)
        if signal is not None:
            signals[_pair_key(str(comparison_id), base_metric)] = signal
    masked: set[frozenset[str]] = set()
    if mask_metric is not None:
        reported = store[store["metric"] == mask_metric]
        for comparison_id, value in reported["metric_value"].items():
            if not np.isfinite(value):
                masked.add(_pair_key(str(comparison_id), mask_metric))
    frames = []
    for variant in variants:
        cache: dict[frozenset[str], float] = {}
        values = []
        for comparison_id, original in base["metric_value"].items():
            if not np.isfinite(original):
                values.append(math.nan)
                continue
            key = _pair_key(str(comparison_id), base_metric)
            if key in masked:
                values.append(math.nan)
                continue
            if key not in cache:
                signal = signals.get(key)
                if signal is None:
                    raise ValueError(
                        f"Comparison {comparison_id} has a value but no stored signal"
                    )
                cache[key] = reduce_signal(
                    signal["param_values"], signal["scores"], variant
                )
            values.append(cache[key])
        rows = base.copy()
        rows["metric"] = variant.metric_name
        rows["metric_value"] = values
        rows["similarity_signal"] = None
        new_ids = [
            str(old)[: -len(base_metric)] + variant.metric_name for old in rows.index
        ]
        rows["id"] = new_ids
        rows.index = new_ids
        frames.append(rows)
    return pd.concat(frames)


def _scratch_relative(raw_relative: str, scratch_name: str) -> str:
    """Move a ``manifold/<run>/...`` result path under the scratch tree."""
    parts = Path(raw_relative).parts
    if parts[0] != "manifold":
        raise ValueError(f"Expected a manifold/... result path, got {raw_relative!r}")
    return str(Path(SCRATCH_ROOT, scratch_name, *parts[1:]))


def scratch_paths(
    entry: ManifestEntry, config: Mapping[str, Any], scratch_name: str
) -> tuple[Path, Path, Path]:
    """Scratch result store, variant full CSV, and results root for a task."""
    relative = Path(config["raw_results_filename"])
    result_path = Path(entry.result_path)
    if result_path.parts[-len(relative.parts) :] != relative.parts:
        raise ValueError(
            f"Manifest result path {result_path} does not end in the config's "
            f"raw_results_filename {relative}"
        )
    results_root = Path(*result_path.parts[: -len(relative.parts)])
    store = results_root / _scratch_relative(str(relative), scratch_name)
    full_relative = config["table_creation"]["full_df_filename"]
    csv = results_root / _scratch_relative(full_relative, scratch_name)
    return store, csv, results_root


@dataclass(frozen=True)
class AnalysisSpec:
    """One question: an ordered set of variants plotted along one axis."""

    name: str
    title: str
    x_label: str
    points: tuple[tuple[str, str], ...]  # (variant key, tick label)


@dataclass(frozen=True)
class BandwidthSpec:
    """Single-bandwidth measures to set against the swept ``reference`` measure."""

    reference: str
    points: tuple[tuple[str, float], ...]  # (measure, bandwidth multiplier)


@dataclass(frozen=True)
class SensitivityConfig:
    """``field_measure`` is the reported measure each variant stands in for."""

    base_metric: str
    scratch_name: str
    variants: tuple[SignalVariant, ...]
    analyses: tuple[AnalysisSpec, ...]
    field_measure: str = BASE_METRIC
    bandwidths: BandwidthSpec | None = None

    def variant(self, key: str) -> SignalVariant:
        for variant in self.variants:
            if variant.key == key:
                return variant
        raise KeyError(f"No variant named {key!r}")


def parse_sensitivity_config(raw: Mapping[str, Any]) -> SensitivityConfig:
    """Validate a sensitivity configuration mapping.

    Every analysis point must name a declared variant, so a typo fails here
    rather than as a missing column after hours of replay.
    """
    variants = tuple(
        SignalVariant(
            key=str(item["key"]),
            samples=item.get("samples"),
            lower=item.get("lower"),
            upper=item.get("upper"),
        )
        for item in raw.get("variants", ())
    )
    keys = [variant.key for variant in variants]
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Sensitivity variants must be non-empty with unique keys")
    analyses = []
    for name, item in dict(raw.get("analyses", {})).items():
        points = tuple(
            (str(point["variant"]), str(point["label"])) for point in item["points"]
        )
        unknown = [key for key, _label in points if key not in keys]
        if unknown:
            raise ValueError(f"Analysis {name} names unknown variants: {unknown}")
        analyses.append(
            AnalysisSpec(
                name=str(name),
                title=str(item.get("title", name)),
                x_label=str(item.get("x_label", "")),
                points=points,
            )
        )
    if not analyses:
        raise ValueError("Sensitivity config declares no analyses")
    bandwidths = None
    if "bandwidths" in raw:
        spec = raw["bandwidths"]
        points = tuple(
            (str(item["measure"]), float(item["sigma"])) for item in spec["measures"]
        )
        if not points:
            raise ValueError("bandwidths lists no measures")
        if len({measure for measure, _sigma in points}) != len(points):
            raise ValueError("bandwidths lists a measure twice")
        bandwidths = BandwidthSpec(
            reference=str(spec.get("reference", raw.get("field_measure", BASE_METRIC))),
            points=tuple(sorted(points, key=lambda item: item[1])),
        )
    return SensitivityConfig(
        base_metric=str(raw.get("base_metric", BASE_METRIC)),
        scratch_name=str(raw.get("scratch_name", "signal-sensitivity")),
        variants=variants,
        analyses=tuple(analyses),
        field_measure=str(raw.get("field_measure", BASE_METRIC)),
        bandwidths=bandwidths,
    )


def load_sensitivity_config(path: str | Path) -> SensitivityConfig:
    return parse_sensitivity_config(load_yaml_config(path))


__all__ = [
    "AnalysisSpec",
    "BASE_METRIC",
    "BandwidthSpec",
    "SensitivityConfig",
    "SCRATCH_ROOT",
    "SignalVariant",
    "derive_variant_rows",
    "load_sensitivity_config",
    "parse_sensitivity_config",
    "reduce_signal",
    "scratch_paths",
    "select_points",
]
