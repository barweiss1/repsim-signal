"""Experimental reductions applied to completed metric signals."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from manifold_repsim.sweeps import calc_local_minimas
from manifold_repsim.sweeps import get_convex_regions
from manifold_repsim.sweeps import get_cut_idx
from manifold_repsim.sweeps import integrate_metric_over_param

from .specifications import MetricAggregation

_CUTOFF_METRICS = frozenset({"cka_rbf", "cka_rbf_quantile"})


def _validated_curve(parameter_values, scores):
    parameter_values = np.asarray(parameter_values, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if parameter_values.ndim != 1 or scores.ndim != 1:
        raise ValueError("Metric parameter values and scores must be one-dimensional.")
    if parameter_values.shape != scores.shape or len(scores) == 0:
        raise ValueError("Metric parameter values and scores must have equal length.")
    if not np.isfinite(parameter_values).all() or not np.isfinite(scores).all():
        raise ValueError("Metric parameter values and scores must be finite.")
    return parameter_values, scores


def _cutoff_curve(
    metric_name,
    parameter_values,
    scores,
    infinity_score,
    cutoff_threshold,
):
    if metric_name not in _CUTOFF_METRICS:
        return parameter_values, scores, len(scores) - 1
    if infinity_score is None or not np.isfinite(infinity_score):
        raise ValueError(f"{metric_name} cutoff aggregation requires infinity_score.")
    cut_index = get_cut_idx(scores, infinity_score, threshold=cutoff_threshold)
    return parameter_values[: cut_index + 1], scores[: cut_index + 1], cut_index


def aggregate_metric_curve(
    aggregation: MetricAggregation | str,
    parameter_values,
    scores,
    *,
    metric_name: str,
    infinity_score: float | None = None,
    integration_method: str = "trapezoidal",
    cutoff_threshold: float = 0.05,
    logscale: bool = False,
) -> float:
    """Reduce one already-computed metric signal according to an explicit kind."""
    aggregation = MetricAggregation(aggregation)
    if aggregation in {
        MetricAggregation.FIXED,
        MetricAggregation.VARIANCE_WEIGHTED_AUC,
    }:
        raise ValueError(f"{aggregation.value} is not a single-curve aggregation.")
    parameter_values, scores = _validated_curve(parameter_values, scores)

    if aggregation is MetricAggregation.AUC:
        return float(
            integrate_metric_over_param(
                parameter_values,
                scores,
                integration_method,
            )
        )

    if aggregation is MetricAggregation.CUTOFF_AUC:
        cut_params, cut_scores, _ = _cutoff_curve(
            metric_name,
            parameter_values,
            scores,
            infinity_score,
            cutoff_threshold,
        )
        return float(
            integrate_metric_over_param(cut_params, cut_scores, integration_method)
        )

    minima_params, minima_scores = calc_local_minimas(
        parameter_values,
        scores,
        robust=True,
    )

    if aggregation is MetricAggregation.MINIMUM_POINT:
        return float(minima_params[0] if len(minima_scores) else parameter_values[-1])

    if aggregation is MetricAggregation.MINIMUM:
        return float(np.mean(minima_scores) if len(minima_scores) else np.min(scores))

    if aggregation is MetricAggregation.MINIMUM_ENVELOPE:
        if len(minima_scores) == 0:
            return float(
                integrate_metric_over_param(
                    parameter_values,
                    scores,
                    integration_method,
                )
            )
        values = []
        weights = []
        window_size = 3
        # Preserve the historical window indexing until experimental aggregate
        # definitions receive a separate correctness review.
        for minimum_number, _minimum_parameter in enumerate(minima_params):
            start = max(0, minimum_number - window_size)
            end = min(len(parameter_values) - 1, minimum_number + window_size)
            window_params = parameter_values[start : end + 1]
            window_scores = scores[start : end + 1]
            values.append(
                integrate_metric_over_param(
                    window_params,
                    window_scores,
                    integration_method,
                )
            )
            if integration_method == "trapezoidal":
                weights.append(window_params[-1] - window_params[0])
            elif integration_method == "average":
                weights.append(len(window_params))
            else:
                raise ValueError(
                    f"Unsupported integration method: {integration_method}"
                )
        return float(np.average(values, weights=weights))

    if aggregation is MetricAggregation.MIN_TO_MAX_AUC:
        if len(scores) < 2:
            return float(
                integrate_metric_over_param(
                    parameter_values,
                    scores,
                    integration_method,
                )
            )
        maxima, _ = find_peaks(scores, prominence=0.01, distance=4)
        minima, _ = find_peaks(-scores, prominence=0.01, distance=4)
        minimum_index = int(minima[0]) if len(minima) else int(np.argmin(scores))
        maximum_index = int(maxima[0]) if len(maxima) else int(np.argmax(scores))
        start = min(minimum_index, maximum_index)
        end = max(minimum_index, maximum_index) + 1
        return float(
            integrate_metric_over_param(
                parameter_values[start:end],
                scores[start:end],
                integration_method,
            )
        )

    if aggregation is MetricAggregation.MIN_TO_CUTOFF:
        cut_params, cut_scores, _ = _cutoff_curve(
            metric_name,
            parameter_values,
            scores,
            infinity_score,
            cutoff_threshold,
        )
        if len(minima_scores):
            minimum_index = int(np.searchsorted(cut_params, minima_params[0]))
            if minimum_index < len(cut_params):
                cut_params = cut_params[minimum_index:]
                cut_scores = cut_scores[minimum_index:]
        return float(
            integrate_metric_over_param(cut_params, cut_scores, integration_method)
        )

    if aggregation is MetricAggregation.CONVEX_AUC:
        if len(scores) < 3:
            return float(
                integrate_metric_over_param(
                    parameter_values,
                    scores,
                    integration_method,
                )
            )
        regions = get_convex_regions(
            parameter_values,
            scores,
            side_lobe=0.5,
            curvature_threshold=0.0,
            min_region_len=4,
            logscale=logscale,
        )
        cut_params, cut_scores, cut_index = _cutoff_curve(
            metric_name,
            parameter_values,
            scores,
            infinity_score,
            cutoff_threshold,
        )
        regions = [region for region in regions if cut_index not in region]
        if not regions:
            return float(
                integrate_metric_over_param(cut_params, cut_scores, integration_method)
            )
        selected = set()
        total_area = 0.0
        for region in regions:
            start = max(region[0], 0)
            end = min(region[-1] + 2, len(cut_scores) - 1)
            if end < start:
                continue
            if integration_method == "trapezoidal":
                total_area += np.trapezoid(
                    cut_scores[start : end + 1],
                    cut_params[start : end + 1],
                )
            elif integration_method == "average":
                selected.update(range(start, end + 1))
            else:
                raise ValueError(
                    f"Unsupported integration method: {integration_method}"
                )
        if integration_method == "average":
            if not selected:
                return float(np.mean(cut_scores))
            return float(np.mean(cut_scores[sorted(selected)]))
        width = cut_params[-1] - cut_params[0]
        if width == 0:
            return float(cut_scores[0])
        return float(total_area / width)

    raise ValueError(f"Unsupported metric aggregation: {aggregation.value}")
