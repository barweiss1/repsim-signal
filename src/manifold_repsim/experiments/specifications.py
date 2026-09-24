"""Explicit metric specifications for synthetic experiment comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from manifold_repsim.metrics.registry import get_metric_spec
from manifold_repsim.sweeps import SweepGridConfig


class MetricAggregation(str, Enum):
    """Supported reductions of a metric-parameter signal."""

    FIXED = "fixed"
    AUC = "auc"
    CUTOFF_AUC = "cutoff_auc"
    CONVEX_AUC = "convex_auc"
    MIN_TO_MAX_AUC = "min_to_max_auc"
    VARIANCE_WEIGHTED_AUC = "variance_weighted_auc"
    MINIMUM_ENVELOPE = "minimum_envelope"
    MINIMUM_POINT = "minimum_point"
    MINIMUM = "minimum"
    MIN_TO_CUTOFF = "min_to_cutoff"


@dataclass(frozen=True)
class MetricComparisonSpec:
    """One explicitly named score in a metric-comparison experiment.

    ``metric_name`` always names a registered base metric. ``aggregation``
    states directly whether that metric is evaluated once or reduced across a
    parameter signal; callers do not encode behavior in ``name`` suffixes.
    """

    name: str
    metric_name: str
    aggregation: MetricAggregation | str = MetricAggregation.FIXED
    metric_grid: SweepGridConfig | Mapping[str, Any] | None = None
    metric_kwargs: Mapping[str, Any] | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("A metric comparison requires a non-empty name.")
        get_metric_spec(self.metric_name)
        try:
            aggregation = MetricAggregation(self.aggregation)
        except ValueError:
            raise ValueError(
                f"Unsupported metric aggregation: {self.aggregation!r}."
            ) from None
        object.__setattr__(self, "aggregation", aggregation)

        metric_grid = self.metric_grid
        if isinstance(metric_grid, Mapping):
            metric_grid = SweepGridConfig.from_mapping(metric_grid)
        elif metric_grid is not None and not isinstance(metric_grid, SweepGridConfig):
            raise TypeError("metric_grid must be a SweepGridConfig or mapping.")
        if aggregation is MetricAggregation.FIXED and metric_grid is not None:
            raise ValueError("A fixed metric comparison cannot define metric_grid.")
        object.__setattr__(self, "metric_grid", metric_grid)

        if self.metric_kwargs is None:
            metric_kwargs = {}
        elif isinstance(self.metric_kwargs, Mapping):
            metric_kwargs = dict(self.metric_kwargs)
        else:
            raise TypeError("metric_kwargs must be a mapping.")
        object.__setattr__(self, "metric_kwargs", MappingProxyType(metric_kwargs))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MetricComparisonSpec":
        """Parse the YAML representation of one metric comparison."""
        if not isinstance(value, Mapping):
            raise TypeError("A metric comparison specification must be a mapping.")
        allowed = {
            "name",
            "metric",
            "aggregation",
            "metric_grid",
            "metric_kwargs",
            "label",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown metric comparison fields: {sorted(unknown)}")
        metric_name = value.get("metric")
        name = value.get("name", metric_name)
        return cls(
            name=name,
            metric_name=metric_name,
            aggregation=value.get("aggregation", MetricAggregation.FIXED.value),
            metric_grid=value.get("metric_grid"),
            metric_kwargs=value.get("metric_kwargs"),
            label=value.get("label"),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical YAML-compatible specification."""
        value = {
            "name": self.name,
            "metric": self.metric_name,
            "aggregation": self.aggregation.value,
            "metric_kwargs": dict(self.metric_kwargs),
        }
        if self.metric_grid is not None:
            value["metric_grid"] = self.metric_grid.to_mapping()
        if self.label is not None:
            value["label"] = self.label
        return value
