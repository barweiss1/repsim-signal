"""Analysis-time reporting settings: which measures to report, and how strictly.

Deliberately a separate file from the campaign definition. A campaign says what
gets computed and is read by `prepare`, which snapshots it beside the generated
manifests; these settings say what to report from results that already exist.
Keeping them apart means the reporting choices can be edited repeatedly while
looking at output without touching the file that defines the run.

This module performs no I/O beyond reading the settings file itself and imports
neither Torch, ReSi, nor a plotting backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import (
    DEFAULT_BOX_STYLE,
    DEFAULT_RANK_SCALE,
    DEFAULT_RANK_SORT,
    RANKED_FUNCTIONAL_MEASURES,
    RANKED_QUALITY_MEASURES,
    validate_box_style,
    validate_rank_scale,
    validate_rank_sort,
)

SETTINGS_KEYS = frozenset(
    {
        "version",
        "measures",
        "exclude_measures",
        "quality_measures",
        "exclude_functional_measures",
        "allow_incomplete",
        "rank_scale",
        "rank_sort",
        "box_style",
        "max_nan_fraction",
        "min_case_coverage",
    }
)

# A measure is dropped from a case once more than this fraction of that
# case's observations are non-finite. Half is the point past which the
# measure describes the failures more than the data.
DEFAULT_MAX_NAN_FRACTION = 0.5

# A measure must survive at least this fraction of a scope's cases to be
# pooled into that scope's summary. Below it, its mean is drawn from a
# materially easier subset than its competitors were scored on.
#
# 0.90 rather than 0.95: at 0.95 the threshold was excluding language's
# SecondOrderCosineSimilarity for missing two cases out of 38 (0.947), which
# is a rounding-level gap rather than the materially-easier-field problem this
# guard exists for. The measure it is actually aimed at is vision's
# LinearRegression at 0.795 -- wholly NaN for whole architecture groups. 0.90
# separates those two cleanly and leaves LinearRegression the only measure any
# domain drops on coverage.
DEFAULT_MIN_CASE_COVERAGE = 0.90


def _fraction(value: Any, field: str, default: float) -> float:
    """Parse an optional fraction in ``[0, 1]``, rejecting anything else."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number between 0 and 1")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1, got {number}")
    return number


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    """Parse a non-empty list of distinct names, rejecting duplicates."""
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list):
        values = tuple(str(item) for item in value)
    else:
        raise ValueError(f"{field} must be a string or list of strings")
    if not values:
        raise ValueError(f"{field} cannot be empty")
    if any(not item.strip() for item in values):
        raise ValueError(f"{field} contains an empty value")
    names = tuple(item.strip() for item in values)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{field} contains duplicate(s): {', '.join(duplicates)}")
    return names


@dataclass(frozen=True)
class AnalysisSettings:
    """Resolved reporting choices for one analysis invocation."""

    path: Path | None = None
    measures: tuple[str, ...] | None = None
    exclude_measures: tuple[str, ...] = ()
    quality_measures: tuple[str, ...] | None = None
    exclude_functional_measures: tuple[str, ...] = ()
    allow_incomplete: bool = False
    rank_scale: str = DEFAULT_RANK_SCALE
    rank_sort: str = DEFAULT_RANK_SORT
    box_style: str = DEFAULT_BOX_STYLE
    max_nan_fraction: float = DEFAULT_MAX_NAN_FRACTION
    min_case_coverage: float = DEFAULT_MIN_CASE_COVERAGE

    def resolve_measures(self, present: Iterable[str]) -> tuple[str, ...] | None:
        """Return the measures to report, or None to report every one present.

        An `exclude_measures` list is resolved against what the results actually
        contain, so naming a measure this domain never had is harmless: PWCCA is
        absent from the graph archive entirely, and one shared exclusion list
        should not fail there. An explicit `measures` list is returned unchanged
        so the caller's own strict check still reports anything missing.
        """
        if self.measures is not None:
            return self.measures
        if not self.exclude_measures:
            return None
        excluded = set(self.exclude_measures)
        kept = tuple(sorted({str(name) for name in present} - excluded))
        if not kept:
            raise ValueError(
                "exclude_measures removed every measure present in the results"
            )
        return kept


def parse_analysis_settings(
    data: Mapping[str, Any], *, path: Path | None = None
) -> AnalysisSettings:
    """Validate a settings mapping, rejecting unknown or contradictory keys."""
    if not isinstance(data, Mapping):
        raise ValueError("Analysis settings must be a mapping")
    unknown = sorted(set(data) - SETTINGS_KEYS)
    if unknown:
        raise ValueError(f"Unknown analysis settings key(s): {', '.join(unknown)}")
    measures = (
        _string_list(data["measures"], "measures") if "measures" in data else None
    )
    exclude = (
        _string_list(data["exclude_measures"], "exclude_measures")
        if "exclude_measures" in data
        else ()
    )
    # Mirrors ReSi's own treatment of included_measures/excluded_measures: one
    # or the other, never both, so the reported set is unambiguous.
    if measures is not None and exclude:
        raise ValueError("measures and exclude_measures are mutually exclusive")
    exclude_functional = (
        _string_list(data["exclude_functional_measures"], "exclude_functional_measures")
        if "exclude_functional_measures" in data
        else ()
    )
    quality_measures = (
        _string_list(data["quality_measures"], "quality_measures")
        if "quality_measures" in data
        else None
    )
    allow_incomplete = data.get("allow_incomplete", False)
    if not isinstance(allow_incomplete, bool):
        raise ValueError("allow_incomplete must be a boolean")
    rank_scale = validate_rank_scale(data.get("rank_scale"))
    rank_sort = validate_rank_sort(data.get("rank_sort"))
    box_style = validate_box_style(data.get("box_style"))
    max_nan_fraction = _fraction(
        data.get("max_nan_fraction"), "max_nan_fraction", DEFAULT_MAX_NAN_FRACTION
    )
    min_case_coverage = _fraction(
        data.get("min_case_coverage"), "min_case_coverage", DEFAULT_MIN_CASE_COVERAGE
    )
    return AnalysisSettings(
        path=path,
        measures=measures,
        exclude_measures=exclude,
        quality_measures=quality_measures,
        exclude_functional_measures=exclude_functional,
        allow_incomplete=allow_incomplete,
        rank_scale=rank_scale,
        rank_sort=rank_sort,
        box_style=box_style,
        max_nan_fraction=max_nan_fraction,
        min_case_coverage=min_case_coverage,
    )


def load_analysis_settings(path: str | Path) -> AnalysisSettings:
    """Load and validate an analysis settings YAML file."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"Analysis settings file not found: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return parse_analysis_settings(data, path=resolved)


def settings_snapshot(
    settings: AnalysisSettings,
    selected: Iterable[str] | None,
    *,
    rank_scale: str | None = None,
    rank_sort: str | None = None,
    box_style: str | None = None,
) -> dict[str, Any]:
    """Describe the resolved reporting choices for the saved `config.yaml`.

    ``rank_scale``, ``rank_sort`` and ``box_style`` record what the run actually
    ranked, sorted and drew, which may be a command line override rather than
    the settings file's own value.
    """
    return {
        "settings_path": str(settings.path) if settings.path else None,
        "measures": list(settings.measures) if settings.measures else None,
        "exclude_measures": list(settings.exclude_measures),
        "quality_measures": (
            list(settings.quality_measures) if settings.quality_measures else None
        ),
        # Which of those got ranked, and which functional similarity measures
        # did. Neither is a setting: both are ReSi's fixed choices, recorded
        # here so a run says so rather than leaving a reader to discover that
        # its conformity-rate and Disagreement columns never entered a rank.
        "ranked_quality_measures": list(RANKED_QUALITY_MEASURES),
        "ranked_functional_measures": list(RANKED_FUNCTIONAL_MEASURES),
        "exclude_functional_measures": list(settings.exclude_functional_measures),
        "allow_incomplete": settings.allow_incomplete,
        "rank_scale": (
            settings.rank_scale
            if rank_scale is None
            else validate_rank_scale(rank_scale)
        ),
        "rank_sort": (
            settings.rank_sort if rank_sort is None else validate_rank_sort(rank_sort)
        ),
        "box_style": (
            settings.box_style if box_style is None else validate_box_style(box_style)
        ),
        "max_nan_fraction": settings.max_nan_fraction,
        "min_case_coverage": settings.min_case_coverage,
        "reported_measures": list(selected) if selected else None,
    }


__all__ = [
    "AnalysisSettings",
    "DEFAULT_MAX_NAN_FRACTION",
    "DEFAULT_MIN_CASE_COVERAGE",
    "SETTINGS_KEYS",
    "load_analysis_settings",
    "parse_analysis_settings",
    "settings_snapshot",
]
