"""Aggregate already-computed sweep signals into scalar summaries."""

from __future__ import annotations

import numpy as np


def integrate_metric_over_param(
    param_vec,
    scores,
    integration_method: str = "trapezoidal",
) -> float:
    """Integrate one score curve, normalized over its parameter interval."""
    param_values = np.asarray(param_vec, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if param_values.ndim != 1 or scores.ndim != 1:
        raise ValueError("param_values and scores must be one-dimensional.")
    if len(param_values) != len(scores):
        raise ValueError("param_values and scores must have the same length.")
    if len(scores) == 0:
        raise ValueError("Cannot integrate an empty score curve.")
    if len(scores) == 1:
        return float(scores[0])
    if integration_method == "trapezoidal":
        interval = param_values[-1] - param_values[0]
        if interval <= 0:
            raise ValueError("param_values must be strictly increasing.")
        return float(np.trapezoid(scores, param_values) / interval)
    if integration_method == "average":
        return float(np.mean(scores))
    raise ValueError(f"Unsupported integration method: {integration_method}")


def integrate_metric_over_param_weighted(
    param_vec,
    scores,
    weights,
    integration_method: str = "trapezoidal",
    eps: float = 1e-12,
) -> float:
    """Integrate one score curve using non-negative point weights."""
    param_values = np.asarray(param_vec, dtype=float)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if param_values.ndim != 1 or scores.ndim != 1 or weights.ndim != 1:
        raise ValueError("param_values, scores, and weights must be one-dimensional.")
    if len(param_values) != len(scores) or len(scores) != len(weights):
        raise ValueError("param_values, scores, and weights must have the same length.")
    if len(scores) == 0:
        raise ValueError("Cannot integrate an empty score curve.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")

    total_weight = np.sum(weights)
    if total_weight <= eps:
        return integrate_metric_over_param(param_values, scores, integration_method)
    if len(scores) == 1:
        return float(scores[0])
    if integration_method == "trapezoidal":
        weight_area = np.trapezoid(weights, param_values)
        if weight_area <= eps:
            return integrate_metric_over_param(param_values, scores, integration_method)
        return float(np.trapezoid(scores * weights, param_values) / weight_area)
    if integration_method == "average":
        return float(np.sum(scores * weights) / total_weight)
    raise ValueError(f"Unsupported integration method: {integration_method}")


def _normalize_grid_inputs(param_values, signal_grid):
    signal_grid = np.asarray(signal_grid, dtype=float)
    if signal_grid.ndim != 2:
        raise ValueError("signal_grid must be a two-dimensional array.")
    param_values = np.asarray(param_values, dtype=float)
    if param_values.ndim != 1 or param_values.shape != (signal_grid.shape[1],):
        raise ValueError("param_values must have one value per signal_grid column.")
    return param_values, signal_grid


def aggregate_signal_auc(
    param_values,
    signal_grid,
    integration_method: str = "trapezoidal",
) -> np.ndarray:
    """Integrate every row of an already-computed parameter sweep."""
    param_values, signal_grid = _normalize_grid_inputs(param_values, signal_grid)
    return np.asarray(
        [
            integrate_metric_over_param(
                param_values,
                scores,
                integration_method,
            )
            for row, scores in enumerate(signal_grid)
        ]
    )


def aggregate_signal_variance_weighted_auc(
    param_values,
    signal_grid,
    integration_method: str = "trapezoidal",
    weights=None,
) -> np.ndarray:
    """Integrate each row using variance across data conditions as weights."""
    param_values, signal_grid = _normalize_grid_inputs(param_values, signal_grid)
    if weights is None:
        weights = np.var(signal_grid, axis=0)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (signal_grid.shape[1],):
        raise ValueError("weights must have one value per signal_grid column.")
    return np.asarray(
        [
            integrate_metric_over_param_weighted(
                param_values,
                scores,
                weights,
                integration_method,
            )
            for row, scores in enumerate(signal_grid)
        ]
    )


def calc_variance_weighted_auc_from_sweep(
    param_values,
    score_grid,
    integration_method: str = "trapezoidal",
    ddof: int = 0,
) -> np.ndarray:
    """Calculate row AUCs weighted by each sweep column's score variance."""
    score_grid = np.asarray(score_grid, dtype=float)
    if score_grid.ndim != 2:
        raise ValueError("score_grid must be a two-dimensional array.")
    return aggregate_signal_variance_weighted_auc(
        param_values,
        score_grid,
        integration_method=integration_method,
        weights=np.var(score_grid, axis=0, ddof=ddof),
    )
