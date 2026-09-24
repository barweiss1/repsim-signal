"""Named results produced by experiment computations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .specifications import MetricComparisonSpec


@dataclass
class MetricComparisonResult:
    """Scores for explicit metrics across one data-parameter sweep."""

    data_parameter_name: str
    data_parameter_values: np.ndarray
    specifications: tuple[MetricComparisonSpec, ...]
    scores: Mapping[str, np.ndarray]
    labels: Mapping[str, str] = field(default_factory=dict)
    base_parameter_value: int | float | None = None
    effective_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_parameter_values = np.asarray(self.data_parameter_values)
        if self.data_parameter_values.ndim != 1:
            raise ValueError("data_parameter_values must be one-dimensional.")
        names = [spec.name for spec in self.specifications]
        if len(names) != len(set(names)):
            raise ValueError("Metric comparison names must be unique.")
        if set(self.scores) != set(names):
            raise ValueError("scores must contain exactly the specified metric names.")

        normalized_scores = {}
        for name in names:
            values = np.asarray(self.scores[name], dtype=float)
            if values.shape != self.data_parameter_values.shape:
                raise ValueError(
                    f"Scores for {name!r} must match data_parameter_values."
                )
            if not np.isfinite(values).all():
                raise ValueError(f"Scores for {name!r} must be finite.")
            normalized_scores[name] = values
        self.scores = MappingProxyType(normalized_scores)

        unknown_labels = set(self.labels) - set(names)
        if unknown_labels:
            raise ValueError(f"Labels reference unknown scores: {unknown_labels}")
        self.labels = MappingProxyType(
            {name: str(self.labels.get(name, "")) for name in names}
        )

    def score(self, name: str) -> np.ndarray:
        """Return one named score vector."""
        return self.scores[name]
