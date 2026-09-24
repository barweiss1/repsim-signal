"""Notebook-facing compatibility adapters for the experiments package.

These functions translate historical notebook call conventions (positional
score arrays, flat ``sim_params`` dictionaries, suffix-named aggregate
metrics) onto the modern result-based API in :mod:`manifold_repsim.experiments`.
They exist because supported notebooks still call them by these names and
shapes; the root-level ``experiment_utils.py`` module that used to hold them
was a temporary compatibility path and has been removed. New code should call
:func:`manifold_repsim.experiments.run_parameter_sweep`,
:func:`manifold_repsim.experiments.run_metric_comparison`, and
:mod:`manifold_repsim.experiments.plotting` directly instead of adding new
callers here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from manifold_repsim.metrics.registry import get_metric_spec
from manifold_repsim.metrics.registry import get_sweep_parameter_default
from manifold_repsim.sweeps import ParameterSweepResult
from manifold_repsim.sweeps import SweepGridConfig
from manifold_repsim.sweeps import map_param_name_to_kwargs
from synthetic_datasets import get_data_param_sweep_values as _build_data_param_sweep
from synthetic_datasets import get_transform_config

from .execution import run_metric_comparison
from .execution import run_parameter_sweep
from .specifications import MetricAggregation
from .specifications import MetricComparisonSpec

# Pure rename: the historical wrapper only ever forwarded its arguments.
run_param_sweep = run_parameter_sweep


def get_sigma_inf_x(param_values, logscale=False):
    """Compatibility name for the parameter-sweep plotting helper."""
    from .plotting import get_sigma_infinity_x

    return get_sigma_infinity_x(param_values, logscale=logscale)


def set_infty_tick(ax, finite_values, sigma_inf_x):
    """Compatibility name for the parameter-sweep plotting helper."""
    from .plotting import set_infinity_tick

    return set_infinity_tick(ax, finite_values, sigma_inf_x)


def get_metric_sweep_config(metric_name, metric_sweep_range=None):
    """Return the legacy sweep mapping without mutating registry defaults."""
    metric_spec = get_metric_spec(metric_name)
    sweep_config = metric_spec.compatibility_sweep_dict()
    if metric_sweep_range is None:
        return sweep_config
    if set(metric_sweep_range) != {"min", "max"}:
        raise ValueError("metric_sweep_range must define both 'min' and 'max'.")
    validated = SweepGridConfig(
        minimum=metric_sweep_range["min"],
        maximum=metric_sweep_range["max"],
        num=2,
        scale="linear",
    )
    sweep_config.update(min=validated.minimum, max=validated.maximum)
    return sweep_config


def plot_param_sweep(
    metric_name: str,
    plot_kind: str,
    metric_param_name: str,
    metric_param_values,
    data_param_name: str,
    data_param_values,
    score_grid,
    inf_scores,
    sim_params,
    save_path=None,
    logscale=False,
    calibration_result=None,
    plot_calibrated=False,
):
    """Adapt legacy positional arrays to the result-based plotting API."""
    del sim_params
    metric_param_values = np.asarray(metric_param_values)
    if metric_param_values.ndim != 1:
        raise ValueError("Plots require one shared metric-parameter grid.")
    data_param_values = np.asarray(data_param_values)
    infinity_scores = (
        np.full(len(data_param_values), np.nan, dtype=float)
        if inf_scores is None
        else np.asarray(inf_scores, dtype=float)
    )
    result = ParameterSweepResult(
        metric_name=metric_name,
        metric_parameter_name=metric_param_name,
        data_parameter_name=data_param_name,
        data_parameter_values=data_param_values,
        metric_parameter_values=metric_param_values,
        score_grid=score_grid,
        infinity_scores=infinity_scores,
        calibration=calibration_result,
    )
    output_path = None
    if save_path is not None:
        suffix = "_calibrated" if plot_calibrated else ""
        output_path = Path(save_path) / (
            f"{data_param_name}_{metric_name}_{metric_param_name}_sweep{suffix}.png"
        )
    from .plotting import plot_parameter_sweep

    return plot_parameter_sweep(
        result,
        plot_kind=plot_kind,
        logscale=logscale,
        plot_calibrated=plot_calibrated,
        output_path=output_path,
    )


def get_data_param_sweep_values(dataset, data_param_name: str, sweep_len=20):
    """Return a dataset-family sweep through the legacy experiment API."""
    return _build_data_param_sweep(dataset, data_param_name, sweep_len)


def get_variance_weighted_auc_parent_metric(metric_name: str):
    """Return the parent metric encoded by a legacy aggregate output name."""
    if metric_name.endswith("_variance_weighted_auc"):
        return metric_name[:-22]
    if metric_name.endswith("_var_auc"):
        return metric_name[:-8]
    return None


def _legacy_grid(metric_name: str, sim_params: dict):
    configured_grids = sim_params.get(
        "metric_grids",
        sim_params.get("metric_sweep_grids", {}),
    )
    if metric_name in configured_grids:
        return configured_grids[metric_name]
    exact_values = sim_params.get("metric_param_sweep_values")
    if exact_values is not None:
        return {"values": exact_values}
    sweep_range = sim_params.get("metric_sweep_ranges", {}).get(metric_name)
    if sweep_range is None:
        return None
    return {
        "min": sweep_range["min"],
        "max": sweep_range["max"],
        "num": sim_params["metric_param_sweep_len"],
        "scale": "log" if sim_params.get("logscale", False) else "linear",
    }


def _fixed_legacy_spec(name: str, metric_name: str, *, local: bool = False):
    metric_spec = get_metric_spec(metric_name)
    kwargs = {}
    if metric_spec.sweep is not None:
        kwargs = map_param_name_to_kwargs(
            metric_spec.sweep.name,
            get_sweep_parameter_default(metric_spec.sweep.name),
            local=local,
        )
    return MetricComparisonSpec(
        name=name,
        metric_name=metric_name,
        aggregation=MetricAggregation.FIXED,
        metric_kwargs=kwargs,
    )


def _legacy_metric_spec(name: str, sim_params: dict) -> MetricComparisonSpec:
    """Translate one historical suffix once at the compatibility boundary."""
    variance_parent = get_variance_weighted_auc_parent_metric(name)
    suffixes = (
        ("_cutoff_auc", MetricAggregation.CUTOFF_AUC),
        ("_convex_auc", MetricAggregation.CONVEX_AUC),
        ("_min_max_auc", MetricAggregation.MIN_TO_MAX_AUC),
        ("_auc", MetricAggregation.AUC),
        ("_min_env", MetricAggregation.MINIMUM_ENVELOPE),
        ("_min_point", MetricAggregation.MINIMUM_POINT),
        ("_min_to_cutoff", MetricAggregation.MIN_TO_CUTOFF),
        ("_min", MetricAggregation.MINIMUM),
    )
    if variance_parent is not None:
        parent = variance_parent
        aggregation = MetricAggregation.VARIANCE_WEIGHTED_AUC
    else:
        parent = None
        aggregation = None
        for suffix, candidate in suffixes:
            if name.endswith(suffix):
                parent = name[: -len(suffix)]
                aggregation = candidate
                break
    if parent is not None:
        return MetricComparisonSpec(
            name=name,
            metric_name=parent,
            aggregation=aggregation,
            metric_grid=_legacy_grid(parent, sim_params),
        )
    if name.endswith("_local"):
        return _fixed_legacy_spec(name, name[:-6], local=True)
    return _fixed_legacy_spec(name, name)


def _comparison_specs(metric_names, sim_params):
    explicit = sim_params.get("metric_comparisons")
    if explicit is not None:
        return tuple(MetricComparisonSpec.from_mapping(value) for value in explicit)
    return tuple(_legacy_metric_spec(name, sim_params) for name in metric_names)


def run_metric_comparison_across_data_param(
    dataset,
    current_transform_params: dict,
    data_param_name: str,
    metric_names: list,
    sim_params: dict,
    figures_path: str,
    metric_color_map: dict,
):
    """Preserve the notebook workflow while using separated compute and plot APIs."""
    data_param_values = get_data_param_sweep_values(
        dataset,
        data_param_name,
        sim_params["data_param_sweep_len"],
    )
    base_transform_config = get_transform_config(
        sim_params["base_transform_params"],
        dataset,
    )
    transform_config = get_transform_config(current_transform_params, dataset)
    base_parameter_value = sim_params["base_transform_params"].get(data_param_name)
    result = run_metric_comparison(
        dataset=dataset,
        base_transform_config=base_transform_config,
        transform_config=transform_config,
        data_parameter_name=data_param_name,
        data_parameter_values=data_param_values,
        specifications=_comparison_specs(metric_names, sim_params),
        metric_sweep_len=sim_params["metric_param_sweep_len"],
        default_grid_scale="log" if sim_params.get("logscale", False) else "linear",
        integration_method=sim_params["auc_integration_method"],
        cutoff_threshold=sim_params["cutoff_threshold"],
        base_parameter_value=base_parameter_value,
        output_dir=figures_path,
        run_config=sim_params,
    )
    from .plotting import plot_metric_comparison

    plot_metric_comparison(
        result,
        metric_color_map=metric_color_map,
        output_path=Path(figures_path) / f"alignment_scores_{data_param_name}.png",
    )
    return result


__all__ = [
    "get_data_param_sweep_values",
    "get_metric_sweep_config",
    "get_sigma_inf_x",
    "get_variance_weighted_auc_parent_metric",
    "plot_param_sweep",
    "run_metric_comparison_across_data_param",
    "run_param_sweep",
    "set_infty_tick",
]
