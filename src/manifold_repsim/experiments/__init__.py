"""Reusable experiment computation APIs.

Plotting is intentionally not imported here, so experiment execution does not
load or require a plotting backend.
"""

from .aggregation import aggregate_metric_curve
from .execution import run_metric_comparison
from .execution import run_parameter_sweep
from .results import MetricComparisonResult
from .specifications import MetricAggregation
from .specifications import MetricComparisonSpec

__all__ = [
    "MetricAggregation",
    "MetricComparisonResult",
    "MetricComparisonSpec",
    "aggregate_metric_curve",
    "run_metric_comparison",
    "run_parameter_sweep",
]
