"""Manifest-driven analysis of combined native and manifold ReSi results.

Responsibilities are split by module: `loaders` reads and normalizes result
values, `ranking` compares measures within a case, `signals` owns score-curve
parsing and grouping, `plotting` produces figures from completed frames, and
`workflows` orchestrates and writes the public artifacts. ReSi's comparison-id
grammar lives one level up in `manifold_repsim.resi.identifiers`, since campaign
preparation reads ids too; it is re-exported here for convenience.

Plotting and workflow names resolve lazily from an explicit catalogue, following
the same pattern as the permuted-Gaussian MDS package, so importing the data
modules does not pull in a plotting backend.
"""

from __future__ import annotations

from typing import Any

from .. import identifiers
from . import config, loaders, ranking, settings, signals
from .config import (
    DEFAULT_RANK_SCALE,
    MEASURE_CATEGORY_COLORS,
    MEASURE_CATEGORY_ORDER,
    RANKED_FUNCTIONAL_MEASURES,
    RANKED_QUALITY_MEASURES,
    RANK_SCALES,
    rank_scale_case_column,
    validate_rank_scale,
    QUALITY_DIRECTIONS,
    measure_category,
    quality_direction,
)
from ..identifiers import canonical_comparison_id, parse_comparison_id
from .loaders import (
    NORMALIZED_COLUMNS,
    known_missing_coverage,
    normalize_entry,
    normalize_manifests,
    select_measures,
)
from .settings import (
    SETTINGS_KEYS,
    AnalysisSettings,
    load_analysis_settings,
    parse_analysis_settings,
)
from .ranking import (
    normalize_case_rank,
    rank_case_observations,
    select_ranked_functional,
    select_ranked_quality,
    summarize_domains,
)
from .signals import (
    SIGNAL_COLUMNS,
    combine_signals,
    exclude_known_missing_signals,
    load_embedded_signals,
    load_legacy_signals,
    load_manifest_legacy_signals,
    signal_input_coverage,
    summarize_grouped_signal_comparisons,
    summarize_signals,
)

# Names that require a plotting backend or the full workflow. The mapping is a
# fixed catalogue, not a dynamic lookup: every name is listed in __all__ below.
_LAZY_EXPORTS = {
    "plot_auprc_vs_violation_rate": "plotting",
    "plot_grouped_signal_comparisons": "plotting",
    "plot_quality_heatmaps": "plotting",
    "plot_rank_boxplot": "plotting",
    "plot_signal_groups": "plotting",
    "plotting": None,
    "workflows": None,
    "analyze_campaign": "workflows",
}


def __getattr__(name: str) -> Any:
    """Resolve the plotting and workflow catalogue on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module_name = _LAZY_EXPORTS[name] or name
    module = import_module(f".{module_name}", __name__)
    value = module if _LAZY_EXPORTS[name] is None else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


# Submodule names are listed deliberately: the historical star-import wrapper
# re-exported them as a side effect of a dynamically built __all__, and dropping
# them would break callers that reach them through that wrapper.
__all__ = [
    "DEFAULT_RANK_SCALE",
    "MEASURE_CATEGORY_COLORS",
    "MEASURE_CATEGORY_ORDER",
    "NORMALIZED_COLUMNS",
    "QUALITY_DIRECTIONS",
    "RANKED_FUNCTIONAL_MEASURES",
    "RANKED_QUALITY_MEASURES",
    "RANK_SCALES",
    "SETTINGS_KEYS",
    "SIGNAL_COLUMNS",
    "AnalysisSettings",
    "analyze_campaign",
    "canonical_comparison_id",
    "combine_signals",
    "config",
    "exclude_known_missing_signals",
    "identifiers",
    "known_missing_coverage",
    "load_analysis_settings",
    "load_embedded_signals",
    "load_legacy_signals",
    "load_manifest_legacy_signals",
    "loaders",
    "measure_category",
    "normalize_case_rank",
    "normalize_entry",
    "normalize_manifests",
    "parse_analysis_settings",
    "parse_comparison_id",
    "plot_auprc_vs_violation_rate",
    "plot_grouped_signal_comparisons",
    "plot_quality_heatmaps",
    "plot_rank_boxplot",
    "plot_signal_groups",
    "plotting",
    "quality_direction",
    "rank_case_observations",
    "rank_scale_case_column",
    "ranking",
    "select_measures",
    "select_ranked_functional",
    "select_ranked_quality",
    "settings",
    "signal_input_coverage",
    "signals",
    "summarize_domains",
    "validate_rank_scale",
    "summarize_grouped_signal_comparisons",
    "summarize_signals",
    "workflows",
]
