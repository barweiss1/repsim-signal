"""ReSi benchmark integration: measure catalogue, adapters, workflow, analysis.

Only the measure catalogue is imported eagerly. It is deliberately free of Torch
and ReSi imports so campaign preparation and result analysis can use it in
environments where neither is installed.
"""

from __future__ import annotations

from .measures import (
    MANIFOLD_RESI_MEASURE_CLASSES,
    RESI_MEASURE_SPECS,
    ResiMeasureSpec,
    get_measure_spec,
    manifold_measure_categories,
)

__all__ = [
    "MANIFOLD_RESI_MEASURE_CLASSES",
    "RESI_MEASURE_SPECS",
    "ResiMeasureSpec",
    "get_measure_spec",
    "manifold_measure_categories",
]
