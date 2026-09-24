"""Plots that consume completed experiment result objects."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from manifold_repsim.sweeps import ParameterSweepResult

from .results import MetricComparisonResult

_INFINITY_METRICS = frozenset({"cka_rbf", "cknna", "cka_rbf_quantile"})


def get_sigma_infinity_x(parameter_values, *, logscale: bool = False) -> float:
    """Choose a visible x position for a metric's limiting linear score."""
    values = np.asarray(parameter_values, dtype=float)
    maximum = float(np.max(values))
    minimum = float(np.min(values))
    if logscale:
        return maximum * 3.0
    return maximum + max((maximum - minimum) * 0.12, 1e-3)


def set_infinity_tick(ax, finite_values, infinity_x) -> None:
    """Append the limiting linear-kernel tick to an existing axis."""
    finite_values = np.asarray(finite_values, dtype=float)
    finite_min = float(np.min(finite_values))
    finite_max = float(np.max(finite_values))
    base_ticks = [tick for tick in ax.get_xticks() if finite_min <= tick <= finite_max]
    ax.set_xticks(base_ticks + [infinity_x])
    ax.set_xticklabels([f"{tick:g}" for tick in base_ticks] + ["CKA-lin"])
    ticklabels = ax.get_xticklabels()
    if ticklabels:
        ticklabels[-1].set_ha("right")


def _data_value_label(parameter_name: str, value) -> str:
    if parameter_name == "noise_scale":
        return rf"$\sigma_{{noise}}/\sigma_{{signal}}$={value:.2f}"
    if parameter_name == "t":
        return f"t={value:.2f}"
    if parameter_name == "dim":
        return f"dim={int(value)}"
    if parameter_name == "cluster_mixing_probability":
        return rf"$p_{{mix}}={value:.2f}$"
    return f"{parameter_name}={value}"


def plot_parameter_sweep(
    result: ParameterSweepResult,
    *,
    plot_kind: str = "line",
    logscale: bool = False,
    plot_calibrated: bool = False,
    output_path: str | Path | None = None,
):
    """Plot a completed parameter sweep without executing any metrics."""
    if not isinstance(result, ParameterSweepResult):
        raise TypeError("result must be a ParameterSweepResult.")
    if plot_kind not in {"line", "heatmap"}:
        raise ValueError("plot_kind must be 'line' or 'heatmap'.")
    if plot_calibrated:
        if result.calibration is None:
            raise ValueError("plot_calibrated=True requires calibration results.")
        score_grid = result.calibration.calibrated_scores
    else:
        score_grid = result.score_grid

    if plot_kind == "line":
        fig, ax = plt.subplots(figsize=(11, 6))
        add_infinity = (
            result.metric_name in _INFINITY_METRICS
            and np.isfinite(result.infinity_scores).any()
            and not plot_calibrated
        )
        infinity_x = (
            get_sigma_infinity_x(
                result.metric_parameter_values,
                logscale=logscale,
            )
            if add_infinity
            else None
        )
        colors = plt.cm.magma(
            np.linspace(0.08, 0.95, len(result.data_parameter_values))
        )
        for row_index, (data_value, color) in enumerate(
            zip(result.data_parameter_values, colors)
        ):
            (line,) = ax.plot(
                result.metric_parameter_values,
                score_grid[row_index],
                marker="o",
                linewidth=2,
                color=color,
                label=_data_value_label(result.data_parameter_name, data_value),
            )
            if result.calibration is not None and not plot_calibrated:
                null_mean = result.calibration.null_mean[row_index]
                null_std = result.calibration.null_std[row_index]
                ax.plot(
                    result.metric_parameter_values,
                    null_mean,
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.8,
                )
                ax.fill_between(
                    result.metric_parameter_values,
                    null_mean - null_std,
                    null_mean + null_std,
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                )
            if add_infinity:
                infinity_score = result.infinity_scores[row_index]
                ax.plot(
                    [result.metric_parameter_values[-1], infinity_x],
                    [result.score_grid[row_index, -1], infinity_score],
                    linestyle="--",
                    linewidth=1.5,
                    color=line.get_color(),
                    alpha=0.9,
                    zorder=3,
                )
                ax.scatter(
                    [infinity_x],
                    [infinity_score],
                    color=line.get_color(),
                    marker="X",
                    s=100,
                    zorder=4,
                )
        if logscale:
            ax.set_xscale("log")
        if add_infinity:
            set_infinity_tick(ax, result.metric_parameter_values, infinity_x)
        ylabel = (
            f"calibrated {result.metric_name}"
            if plot_calibrated
            else result.metric_name
        )
        ax.set_xlabel(result.metric_parameter_name, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(
            f"{ylabel} vs. {result.metric_parameter_name} for different "
            f"{result.data_parameter_name} values",
            fontsize=14,
        )
        ax.legend(loc="best", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        mesh = ax.pcolormesh(
            result.metric_parameter_values,
            result.data_parameter_values,
            score_grid,
            shading="auto",
            cmap="viridis",
        )
        ax.set_xlabel(result.metric_parameter_name, fontsize=12)
        ax.set_ylabel(result.data_parameter_name, fontsize=12)
        ax.set_title(
            f"{result.metric_name} over {result.metric_parameter_name} and "
            f"{result.data_parameter_name}",
            fontsize=14,
        )
        colorbar_label = (
            f"calibrated {result.metric_name}"
            if plot_calibrated
            else result.metric_name
        )
        fig.colorbar(mesh, ax=ax, label=colorbar_label)
        ax.grid(False)

    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
    return fig, ax


def _comparison_xlabel(parameter_name: str) -> str:
    if parameter_name == "noise_scale":
        return r"$\sigma_{noise}/\sigma_{signal}$"
    if parameter_name == "t":
        return "Interpolation Parameter t"
    return parameter_name


def plot_metric_comparison(
    result: MetricComparisonResult,
    *,
    metric_color_map=None,
    output_path: str | Path | None = None,
):
    """Plot completed metric-comparison vectors without executing metrics."""
    if not isinstance(result, MetricComparisonResult):
        raise TypeError("result must be a MetricComparisonResult.")
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [spec.name for spec in result.specifications]
    if metric_color_map is None:
        base_colors = plt.get_cmap("tab20").colors
        colors = {
            name: base_colors[index % len(base_colors)]
            for index, name in enumerate(names)
        }
    else:
        colors = {name: metric_color_map.get(name, "black") for name in names}

    handles = []
    for name in names:
        ax.plot(
            result.data_parameter_values,
            result.scores[name],
            marker="o",
            linestyle="-",
            color=colors[name],
            linewidth=2,
        )
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=colors[name],
                linestyle="-",
                linewidth=2,
                marker="o",
                label=f"{name}{result.labels[name]}",
            )
        )
    if result.base_parameter_value is not None:
        ax.axvline(
            x=result.base_parameter_value,
            color="gray",
            linestyle="--",
            linewidth=1,
        )
    xlabel = _comparison_xlabel(result.data_parameter_name)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Alignment Score", fontsize=12)
    ax.set_title(f"Alignment Scores vs. {xlabel}", fontsize=14)
    ax.legend(handles=handles, loc="best", fontsize=10, handlelength=2.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    return fig, ax
