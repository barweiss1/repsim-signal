"""Experiment computations with no plotting-backend dependency."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from manifold_repsim.config import save_effective_config
from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.metrics import prepare_metric_curve
from manifold_repsim.metrics import score_prepared_curve
from manifold_repsim.metrics.registry import get_metric_spec
from manifold_repsim.metrics.registry import get_sweep_parameter_default
from manifold_repsim.sweeps import ParameterSweepResult
from manifold_repsim.sweeps import SweepGridConfig
from manifold_repsim.sweeps import aggregate_signal_variance_weighted_auc
from manifold_repsim.sweeps import compute_permutation_calibration
from manifold_repsim.sweeps import make_permutation_indices
from manifold_repsim.sweeps import map_param_name_to_kwargs
from manifold_repsim.sweeps import measure_with_permutation_null
from manifold_repsim.sweeps import resolve_metric_sweep_grid

from .aggregation import aggregate_metric_curve
from .results import MetricComparisonResult
from .specifications import MetricAggregation
from .specifications import MetricComparisonSpec

_INFINITY_METRICS = frozenset({"cka_rbf", "cknna", "cka_rbf_quantile"})
_CUTOFF_AGGREGATIONS = frozenset(
    {
        MetricAggregation.CUTOFF_AUC,
        MetricAggregation.CONVEX_AUC,
        MetricAggregation.MIN_TO_CUTOFF,
    }
)


def _validate_data_values(data_parameter_values) -> np.ndarray:
    values = np.asarray(data_parameter_values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("data_parameter_values must be a non-empty 1D sequence.")
    return values


def _resolve_legacy_grid(
    *,
    metric_sweep_values,
    metric_sweep_range,
    metric_sweep_grid,
    metric_sweep_len,
    logscale,
):
    supplied = sum(
        value is not None
        for value in (metric_sweep_values, metric_sweep_range, metric_sweep_grid)
    )
    if supplied > 1:
        raise ValueError(
            "Provide only one of metric_sweep_values, metric_sweep_range, "
            "or metric_sweep_grid."
        )
    scale = "log" if logscale else "linear"
    if metric_sweep_values is not None:
        return SweepGridConfig(values=tuple(metric_sweep_values))
    if metric_sweep_range is not None:
        if set(metric_sweep_range) != {"min", "max"}:
            raise ValueError("metric_sweep_range must define exactly 'min' and 'max'.")
        return SweepGridConfig(
            minimum=metric_sweep_range["min"],
            maximum=metric_sweep_range["max"],
            num=metric_sweep_len,
            scale=scale,
        )
    return metric_sweep_grid


def _representations_for_data_sweep(
    dataset,
    base_transform_config,
    transform_config,
    data_parameter_name,
    data_parameter_values,
):
    """Materialize representation pairs once for all metric consumers."""
    pairs = []
    current_transform_config = transform_config.copy()
    dataset_config = dataset.config.copy()
    for data_parameter_value in data_parameter_values:
        if data_parameter_name == "dim":
            dataset_config.dim = int(data_parameter_value)
            current_dataset = dataset.__class__(config=dataset_config)
            current_dataset.transform_base(base_transform_config)
            current_dataset.transform_current(transform_config)
        else:
            current_transform_config.set_param(
                data_parameter_name,
                data_parameter_value,
            )
            dataset.transform_current(current_transform_config)
            current_dataset = dataset
        pairs.append(
            (
                current_dataset.get_base_torch(),
                current_dataset.get_current_torch(),
            )
        )
    return pairs


def run_parameter_sweep(
    metric_name: str,
    dataset,
    base_transform_config,
    transform_config,
    data_param_name: str,
    data_param_values,
    metric_sweep_len=25,
    metric_sweep_values=None,
    logscale=False,
    permutation_test=False,
    n_permutations=50,
    calibration_alpha=0.05,
    permutation_seed=None,
    metric_kwargs=None,
    metric_sweep_range=None,
    metric_sweep_grid=None,
    output_dir=None,
    run_config=None,
) -> ParameterSweepResult:
    """Run a fixed-grid data/metric parameter sweep.

    Exact or generated experiment grids override registry defaults locally.
    When ``output_dir`` is provided, the resolved configuration is written
    before metric computation begins.
    """
    data_param_values = _validate_data_values(data_param_values)
    grid_request = _resolve_legacy_grid(
        metric_sweep_values=metric_sweep_values,
        metric_sweep_range=metric_sweep_range,
        metric_sweep_grid=metric_sweep_grid,
        metric_sweep_len=metric_sweep_len,
        logscale=logscale,
    )
    scale = "log" if logscale else "linear"
    resolution = resolve_metric_sweep_grid(
        metric_name,
        grid_request,
        default_num=metric_sweep_len,
        default_scale=scale,
    )
    metric_param_name = resolution.parameter_name
    metric_parameter_values = resolution.values
    fixed_metric_kwargs = dict(metric_kwargs or {})

    effective_config = dict(run_config or {})
    effective_config["parameter_sweep"] = {
        "metric": metric_name,
        "metric_parameter": metric_param_name,
        "metric_grid": resolution.effective_config(),
        "metric_kwargs": fixed_metric_kwargs,
        "data_parameter": data_param_name,
        "data_values": data_param_values.tolist(),
        "permutation_test": {
            "enabled": permutation_test,
            "n_permutations": n_permutations,
            "alpha": calibration_alpha,
            "seed": permutation_seed,
        },
    }
    if output_dir is not None:
        save_effective_config(output_dir, effective_config)

    representation_pairs = _representations_for_data_sweep(
        dataset,
        base_transform_config,
        transform_config,
        data_param_name,
        data_param_values,
    )
    score_grid = np.zeros(
        (len(data_param_values), len(metric_parameter_values)),
        dtype=float,
    )
    null_score_grid = (
        np.zeros((*score_grid.shape, n_permutations), dtype=float)
        if permutation_test
        else None
    )
    infinity_scores = np.full(len(data_param_values), np.nan, dtype=float)

    for row_index, (base_data, transformed_data) in enumerate(representation_pairs):
        if permutation_test:
            if base_data.shape[0] != transformed_data.shape[0]:
                raise ValueError(
                    "Permutation calibration requires equal representation rows."
                )
            row_seed = (
                None if permutation_seed is None else int(permutation_seed) + row_index
            )
            permutation_indices = make_permutation_indices(
                transformed_data.shape[0],
                n_permutations,
                seed=row_seed,
                device=transformed_data.device,
            )
            for column_index, parameter_value in enumerate(metric_parameter_values):
                kwargs = {
                    **fixed_metric_kwargs,
                    **map_param_name_to_kwargs(metric_param_name, parameter_value),
                }
                score, null_scores = measure_with_permutation_null(
                    metric_name,
                    base_data,
                    transformed_data,
                    kwargs,
                    permutation_indices,
                )
                score_grid[row_index, column_index] = score
                null_score_grid[row_index, column_index] = null_scores
        else:
            prepared = prepare_metric_curve(
                metric_name,
                base_data,
                transformed_data,
                metric_param_name,
                metric_parameter_values,
                **fixed_metric_kwargs,
            )
            score_grid[row_index] = score_prepared_curve(prepared)

        if metric_name in _INFINITY_METRICS:
            infinity_scores[row_index] = AlignmentMetrics.measure(
                feats_A=base_data,
                feats_B=transformed_data,
                metric="cka",
            )

    calibration = None
    if permutation_test:
        calibration = compute_permutation_calibration(
            score_grid,
            null_score_grid,
            alpha=calibration_alpha,
        )
        calibration.seed = permutation_seed

    return ParameterSweepResult(
        metric_name=metric_name,
        metric_parameter_name=metric_param_name,
        data_parameter_name=data_param_name,
        data_parameter_values=data_param_values,
        metric_parameter_values=metric_parameter_values,
        score_grid=score_grid,
        infinity_scores=infinity_scores,
        calibration=calibration,
        effective_config=effective_config,
    )


def _fixed_metric_kwargs(specification: MetricComparisonSpec) -> dict[str, Any]:
    metric_spec = get_metric_spec(specification.metric_name)
    kwargs = {}
    if metric_spec.sweep is not None:
        kwargs.update(
            map_param_name_to_kwargs(
                metric_spec.sweep.name,
                get_sweep_parameter_default(metric_spec.sweep.name),
            )
        )
    kwargs.update(specification.metric_kwargs)
    return kwargs


def _comparison_label(
    specification: MetricComparisonSpec,
    *,
    metric_parameter_name: str | None = None,
    metric_kwargs: Mapping[str, Any] | None = None,
) -> str:
    if specification.label is not None:
        return specification.label
    if specification.aggregation is MetricAggregation.VARIANCE_WEIGHTED_AUC:
        return f" (var-weighted {metric_parameter_name})"
    kwargs = dict(metric_kwargs or {})
    if "quantile" in kwargs:
        return rf" (q={kwargs['quantile']})"
    if "rbf_sigma" in kwargs:
        return rf" ($\sigma$={kwargs['rbf_sigma']})"
    if "topk" in kwargs:
        return rf" (k={kwargs['topk']})"
    return ""


def _cache_value(value):
    """Convert configuration values into a stable in-process cache key."""
    if isinstance(value, Mapping):
        return tuple(sorted((key, _cache_value(item)) for key, item in value.items()))
    if isinstance(value, np.ndarray):
        return (str(value.dtype), tuple(value.shape), tuple(value.ravel().tolist()))
    if isinstance(value, (list, tuple)):
        return tuple(_cache_value(item) for item in value)
    return value


def run_metric_comparison(
    *,
    dataset,
    base_transform_config,
    transform_config,
    data_parameter_name: str,
    data_parameter_values,
    specifications: Sequence[MetricComparisonSpec | Mapping[str, Any]],
    metric_sweep_len: int = 25,
    default_grid_scale: str = "linear",
    integration_method: str = "trapezoidal",
    cutoff_threshold: float = 0.05,
    base_parameter_value: int | float | None = None,
    output_dir=None,
    run_config: Mapping[str, Any] | None = None,
) -> MetricComparisonResult:
    """Compute explicit metric specifications across one data sweep."""
    data_parameter_values = _validate_data_values(data_parameter_values)
    parsed_specs = tuple(
        (
            value
            if isinstance(value, MetricComparisonSpec)
            else MetricComparisonSpec.from_mapping(value)
        )
        for value in specifications
    )
    if not parsed_specs:
        raise ValueError("At least one metric comparison specification is required.")
    names = [spec.name for spec in parsed_specs]
    if len(names) != len(set(names)):
        raise ValueError("Metric comparison names must be unique.")
    if default_grid_scale not in {"linear", "log"}:
        raise ValueError("default_grid_scale must be 'linear' or 'log'.")

    resolved = {}
    effective_specs = []
    labels = {}
    for spec in parsed_specs:
        if spec.aggregation is MetricAggregation.FIXED:
            kwargs = _fixed_metric_kwargs(spec)
            resolved[spec.name] = (None, kwargs)
            effective = spec.to_mapping()
            effective["metric_kwargs"] = kwargs
            labels[spec.name] = _comparison_label(spec, metric_kwargs=kwargs)
        else:
            grid = resolve_metric_sweep_grid(
                spec.metric_name,
                spec.metric_grid,
                default_num=metric_sweep_len,
                default_scale=default_grid_scale,
            )
            resolved[spec.name] = (grid, dict(spec.metric_kwargs))
            effective = spec.to_mapping()
            effective["metric_grid"] = grid.effective_config()
            labels[spec.name] = _comparison_label(
                spec,
                metric_parameter_name=grid.parameter_name,
            )
        effective_specs.append(effective)

    effective_config = dict(run_config or {})
    effective_config["metric_comparison"] = {
        "data_parameter": data_parameter_name,
        "data_values": data_parameter_values.tolist(),
        "base_parameter_value": base_parameter_value,
        "integration_method": integration_method,
        "cutoff_threshold": cutoff_threshold,
        "metrics": effective_specs,
    }
    if output_dir is not None:
        save_effective_config(output_dir, effective_config)

    representation_pairs = _representations_for_data_sweep(
        dataset,
        base_transform_config,
        transform_config,
        data_parameter_name,
        data_parameter_values,
    )
    scores = {}
    fixed_score_cache = {}
    signal_cache = {}
    infinity_score_cache = None
    for spec in parsed_specs:
        grid, metric_kwargs = resolved[spec.name]
        if spec.aggregation is MetricAggregation.FIXED:
            cache_key = (spec.metric_name, _cache_value(metric_kwargs))
            if cache_key not in fixed_score_cache:
                fixed_score_cache[cache_key] = np.asarray(
                    [
                        AlignmentMetrics.measure(
                            feats_A=base_data,
                            feats_B=transformed_data,
                            metric=spec.metric_name,
                            **metric_kwargs,
                        )
                        for base_data, transformed_data in representation_pairs
                    ],
                    dtype=float,
                )
            scores[spec.name] = fixed_score_cache[cache_key].copy()
            continue

        cache_key = (
            spec.metric_name,
            grid.parameter_name,
            _cache_value(grid.values),
            _cache_value(metric_kwargs),
        )
        if cache_key not in signal_cache:
            signal_grid = np.zeros(
                (len(representation_pairs), len(grid.values)),
                dtype=float,
            )
            for row_index, (base_data, transformed_data) in enumerate(
                representation_pairs
            ):
                prepared = prepare_metric_curve(
                    spec.metric_name,
                    base_data,
                    transformed_data,
                    grid.parameter_name,
                    grid.values,
                    **metric_kwargs,
                )
                signal_grid[row_index] = score_prepared_curve(prepared)
            signal_cache[cache_key] = signal_grid
        signal_grid = signal_cache[cache_key]

        infinity_scores = np.full(len(representation_pairs), np.nan, dtype=float)
        if (
            spec.aggregation in _CUTOFF_AGGREGATIONS
            and spec.metric_name in _INFINITY_METRICS
        ):
            if infinity_score_cache is None:
                infinity_score_cache = np.asarray(
                    [
                        AlignmentMetrics.measure(
                            feats_A=base_data,
                            feats_B=transformed_data,
                            metric="cka",
                        )
                        for base_data, transformed_data in representation_pairs
                    ],
                    dtype=float,
                )
            infinity_scores = infinity_score_cache

        if spec.aggregation is MetricAggregation.VARIANCE_WEIGHTED_AUC:
            scores[spec.name] = aggregate_signal_variance_weighted_auc(
                grid.values,
                signal_grid,
                integration_method=integration_method,
            )
        else:
            scores[spec.name] = np.asarray(
                [
                    aggregate_metric_curve(
                        spec.aggregation,
                        grid.values,
                        row,
                        metric_name=spec.metric_name,
                        infinity_score=infinity_scores[row_index],
                        integration_method=integration_method,
                        cutoff_threshold=cutoff_threshold,
                        logscale=(grid.requested.scale or default_grid_scale) == "log",
                    )
                    for row_index, row in enumerate(signal_grid)
                ],
                dtype=float,
            )

    return MetricComparisonResult(
        data_parameter_name=data_parameter_name,
        data_parameter_values=data_parameter_values,
        specifications=parsed_specs,
        scores=scores,
        labels=labels,
        base_parameter_value=base_parameter_value,
        effective_config=effective_config,
    )
