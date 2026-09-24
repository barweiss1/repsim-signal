"""Named results for two-dimensional metric parameter sweeps."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibration import PermutationCalibrationResult


@dataclass
class ParameterSweepResult:
    """Result of sweeping one data parameter and one metric parameter.

    ``score_grid`` has shape ``(n_data_values, n_metric_values)``.
    ``metric_parameter_values`` is a shared one-dimensional grid with shape
    ``(n_metric_values,)``. ``infinity_scores`` has shape
    ``(n_data_values,)``. Calibration arrays, when present, are described by
    :class:`PermutationCalibrationResult`.

    Iteration temporarily exposes the historical five- or six-item tuple so
    notebooks can continue to unpack this object while callers migrate to the
    named fields.
    """

    metric_name: str
    metric_parameter_name: str
    data_parameter_name: str
    data_parameter_values: np.ndarray
    metric_parameter_values: np.ndarray
    score_grid: np.ndarray
    infinity_scores: np.ndarray
    calibration: PermutationCalibrationResult | None = None
    effective_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data_parameter_values = np.asarray(self.data_parameter_values)
        self.metric_parameter_values = np.asarray(self.metric_parameter_values)
        self.score_grid = np.asarray(self.score_grid, dtype=float)
        self.infinity_scores = np.asarray(self.infinity_scores, dtype=float)

        if self.data_parameter_values.ndim != 1:
            raise ValueError("data_parameter_values must be one-dimensional.")
        if self.score_grid.ndim != 2:
            raise ValueError("score_grid must be two-dimensional.")
        n_data_values, n_metric_values = self.score_grid.shape
        if self.data_parameter_values.shape != (n_data_values,):
            raise ValueError(
                "data_parameter_values length must match the score_grid rows."
            )
        if self.metric_parameter_values.shape != (n_metric_values,):
            raise ValueError(
                "metric_parameter_values must be a shared 1D grid matching "
                "the score_grid columns."
            )
        if self.infinity_scores.shape != (n_data_values,):
            raise ValueError("infinity_scores length must match the score_grid rows.")

        if self.calibration is not None:
            calibrated_scores = np.asarray(self.calibration.calibrated_scores)
            if calibrated_scores.shape != self.score_grid.shape:
                raise ValueError(
                    "calibration calibrated_scores must have the score_grid shape."
                )

    def as_legacy_tuple(self) -> tuple[Any, ...]:
        """Return the deprecated positional result representation."""
        values: tuple[Any, ...] = (
            self.metric_parameter_name,
            self.data_parameter_values,
            self.metric_parameter_values,
            self.score_grid,
            self.infinity_scores,
        )
        if self.calibration is not None:
            return (*values, self.calibration)
        return values

    def __iter__(self) -> Iterator[Any]:
        return iter(self.as_legacy_tuple())

    def __len__(self) -> int:
        return len(self.as_legacy_tuple())

    def __getitem__(self, index: int | slice) -> Any:
        return self.as_legacy_tuple()[index]
