"""Recreated ReSi result tables and rank figures.

ReSi's published tables and aggregate rank plots exist only as notebook cells:
they cannot be imported, they duplicate most of their code across three
notebooks, and they cannot represent this repository's manifold measures. This
package recreates that analysis as maintained modules.

Only the presentation layer is recreated. Input comes from this repository's
manifest-normalized results, so there is no second result-loading path that
recovers metadata from result filenames.

`figures` imports a plotting backend and is resolved lazily.
"""

from __future__ import annotations

from typing import Any

from . import beautify, latex, pivots, ranking, report, significance, taxonomy
from .beautify import build_display_frame
from .latex import render_latex_table, write_latex_table
from .pivots import build_correlation_table, build_overview_table, build_value_table
from .ranking import rank_measures, summarize_ranks
from .report import write_appendix_report
from .significance import floatify, pval_str, separate_significance_indicator

_LAZY_EXPORTS = {
    "plot_rank_distributions": "figures",
    "plot_task_value_distributions": "figures",
    "figures": None,
}


def __getattr__(name: str) -> Any:
    """Resolve the figure catalogue on first access."""
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


__all__ = [
    "beautify",
    "build_correlation_table",
    "build_display_frame",
    "build_overview_table",
    "build_value_table",
    "figures",
    "floatify",
    "latex",
    "pivots",
    "plot_rank_distributions",
    "plot_task_value_distributions",
    "pval_str",
    "rank_measures",
    "ranking",
    "report",
    "render_latex_table",
    "separate_significance_indicator",
    "significance",
    "summarize_ranks",
    "taxonomy",
    "write_appendix_report",
    "write_latex_table",
]
