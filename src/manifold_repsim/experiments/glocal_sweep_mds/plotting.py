"""Signal, MDS, and PCA plots for the glocal sweep experiment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from tqdm import tqdm

from ..artifacts import load_npz
from .config import output_paths
from .discovery import DatasetGroup, GroupRef
from .scoring import lambda_split_result_path

MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")
TITLE_FONT_SIZE = 18
OVERVIEW_TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 18


def choose_mds_3d_view(
    embedding: np.ndarray, view: str | dict[str, float]
) -> tuple[float, float]:
    """Choose a camera that exposes the two largest-variance directions.

    In automatic mode the camera looks along the embedding's least-varying
    principal direction. An explicit YAML camera is returned unchanged.
    """
    if view != "auto":
        return float(view["elevation"]), float(view["azimuth"])
    embedding = np.asarray(embedding, dtype=float)
    if embedding.ndim != 2 or embedding.shape[1] != 3:
        raise ValueError("A 3D MDS view requires an n-by-3 embedding")
    if not np.isfinite(embedding).all():
        raise ValueError("A 3D MDS embedding must be finite")
    _, _, principal_directions = np.linalg.svd(
        embedding - embedding.mean(axis=0), full_matrices=True
    )
    view_direction = principal_directions[-1].copy()
    pivot = int(np.argmax(np.abs(view_direction)))
    if view_direction[pivot] < 0:
        view_direction *= -1
    horizontal = float(np.hypot(view_direction[0], view_direction[1]))
    elevation = float(np.degrees(np.arctan2(view_direction[2], horizontal)))
    azimuth = float(np.degrees(np.arctan2(view_direction[1], view_direction[0])))
    return elevation, azimuth


def _apply_mds_3d_view(
    axis: plt.Axes,
    embedding: np.ndarray,
    view: str | dict[str, float],
) -> None:
    """Apply a separation-oriented orthographic camera and equal data scale."""
    elevation, azimuth = choose_mds_3d_view(embedding, view)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_proj_type("ortho")
    ranges = np.ptp(embedding, axis=0)
    largest = float(ranges.max())
    if largest == 0:
        ranges = np.ones(3)
    else:
        ranges = np.maximum(ranges, largest * 0.05)
    axis.set_box_aspect(ranges)


def _encodings(
    result: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[float, str], Normalize]:
    transformed = ~result["is_none"]
    lambda_levels = np.unique(result["lambda_values"][transformed])
    tau_levels = np.unique(result["tau_values"][transformed])
    if len(lambda_levels) > len(MARKERS):
        raise ValueError("Too many lambda levels for distinct MDS markers")
    markers = {
        float(value): MARKERS[index] for index, value in enumerate(lambda_levels)
    }
    low, high = float(tau_levels.min()), float(tau_levels.max())
    if low == high:
        low -= 0.5
        high += 0.5
    return transformed, markers, Normalize(low, high)


def _point_sizes(result: dict[str, np.ndarray]) -> np.ndarray:
    transformed = ~result["is_none"]
    levels = np.unique(result["alpha_values"][transformed])
    if len(levels) == 1:
        mapped = {float(levels[0]): 80.0}
    else:
        mapped = {
            float(value): float(size)
            for value, size in zip(levels, np.linspace(45.0, 150.0, len(levels)))
        }
    return np.asarray(
        [
            95.0 if is_none else mapped[float(alpha)]
            for is_none, alpha in zip(result["is_none"], result["alpha_values"])
        ]
    )


def _scatter_mds(
    axis: plt.Axes,
    result: dict[str, np.ndarray],
    embedding: np.ndarray,
    *,
    is_3d: bool,
    view: str | dict[str, float] = "auto",
) -> None:
    transformed, markers, norm = _encodings(result)
    sizes = _point_sizes(result)
    coordinates = tuple(embedding.T)
    for lambda_value, marker in markers.items():
        mask = transformed & np.isclose(result["lambda_values"], lambda_value)
        kwargs = {"depthshade": False} if is_3d else {}
        axis.scatter(
            *(coordinate[mask] for coordinate in coordinates),
            c=result["tau_values"][mask],
            s=sizes[mask],
            cmap="viridis",
            norm=norm,
            marker=marker,
            alpha=0.82,
            edgecolors="none",
            **kwargs,
        )
    base = result["is_none"]
    kwargs = {"depthshade": False} if is_3d else {}
    axis.scatter(
        *(coordinate[base] for coordinate in coordinates),
        s=sizes[base],
        marker="*",
        c="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
        **kwargs,
    )
    axis.set_xlabel("MDS 1", fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel("MDS 2", fontsize=AXIS_LABEL_FONT_SIZE)
    if is_3d:
        axis.set_zlabel("MDS 3", fontsize=AXIS_LABEL_FONT_SIZE)
        _apply_mds_3d_view(axis, embedding, view)
    else:
        axis.set_aspect("equal", adjustable="datalim")
    axis.tick_params(labelsize=TICK_FONT_SIZE)
    axis.grid(alpha=0.2)


def _scatter_pca(
    axis: plt.Axes,
    result: dict[str, np.ndarray],
) -> None:
    """Plot a two-dimensional PCA embedding with the shared encodings."""
    embedding = result["pca_embedding_2d"]
    transformed, markers, norm = _encodings(result)
    sizes = _point_sizes(result)
    for lambda_value, marker in markers.items():
        mask = transformed & np.isclose(result["lambda_values"], lambda_value)
        axis.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=result["tau_values"][mask],
            s=sizes[mask],
            cmap="viridis",
            norm=norm,
            marker=marker,
            alpha=0.82,
            edgecolors="none",
        )
    base = result["is_none"]
    axis.scatter(
        embedding[base, 0],
        embedding[base, 1],
        s=sizes[base],
        marker="*",
        c="black",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    explained = 100.0 * result["pca_explained_variance_ratio_2d"]
    axis.set_xlabel(f"PC 1 ({explained[0]:.1f}%)", fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel(f"PC 2 ({explained[1]:.1f}%)", fontsize=AXIS_LABEL_FONT_SIZE)
    axis.set_aspect("equal", adjustable="datalim")
    axis.tick_params(labelsize=TICK_FONT_SIZE)
    axis.grid(alpha=0.2)


def _guides(
    figure: plt.Figure, axes: list[plt.Axes], result: dict[str, np.ndarray]
) -> None:
    transformed, markers, norm = _encodings(result)
    tau_levels = np.unique(result["tau_values"][transformed])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes, shrink=0.72, pad=0.03
    )
    colorbar.set_label(r"$\tau$", fontsize=AXIS_LABEL_FONT_SIZE)
    colorbar.set_ticks(tau_levels)
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)
    lambda_handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            color="0.35",
            label=rf"$\lambda={value:g}$",
        )
        for value, marker in markers.items()
    ]
    alpha_levels = np.unique(result["alpha_values"][transformed])
    sizes = _point_sizes(result)[transformed]
    alpha_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color="0.55",
            markersize=np.sqrt(
                sizes[np.isclose(result["alpha_values"][transformed], value)][0]
            ),
            label=rf"$\alpha={value:g}$",
        )
        for value in alpha_levels
    ]
    none_handle = Line2D(
        [], [], marker="*", linestyle="none", color="black", markersize=10, label="none"
    )
    figure.legend(
        handles=lambda_handles + alpha_handles + [none_handle],
        loc="lower center",
        ncol=min(7, len(lambda_handles) + len(alpha_handles) + 1),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )


def _plot_metric_mds(
    result: dict[str, np.ndarray],
    figure_dir: Path,
    dpi: int,
    view: str | dict[str, float],
) -> None:
    metric_id = result["metric_id"].item()
    metric_dir = figure_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    _scatter_mds(axis, result, result["embedding_2d"], is_3d=False)
    axis.set_title(
        f"{metric_id} | normalized stress={result['embedding_2d_normalized_stress'].item():.4f}",
        fontsize=TITLE_FONT_SIZE,
    )
    _guides(figure, [axis], result)
    figure.savefig(metric_dir / f"{metric_id}_2d.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)

    figure = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis_3d = figure.add_subplot(111, projection="3d")
    _scatter_mds(axis_3d, result, result["embedding_3d"], is_3d=True, view=view)
    axis_3d.set_title(
        f"{metric_id} | normalized stress={result['embedding_3d_normalized_stress'].item():.4f}",
        fontsize=TITLE_FONT_SIZE,
    )
    _guides(figure, [axis_3d], result)
    figure.savefig(metric_dir / f"{metric_id}_3d.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_metric_pca(result: dict[str, np.ndarray], figure_dir: Path, dpi: int) -> None:
    """Plot PCA of one metric's concatenated condition signals."""
    metric_id = result["metric_id"].item()
    metric_dir = figure_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    _scatter_pca(axis, result)
    total = 100.0 * result["pca_explained_variance_ratio_2d"].sum()
    axis.set_title(
        f"{metric_id} | variance explained={total:.1f}%",
        fontsize=TITLE_FONT_SIZE,
    )
    _guides(figure, [axis], result)
    figure.savefig(metric_dir / f"{metric_id}_pca_2d.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_overview(
    results: list[dict[str, np.ndarray]],
    figure_dir: Path,
    title: str,
    dpi: int,
) -> None:
    """Plot one two-dimensional overview for compatible MDS results."""
    columns = min(3, len(results))
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 5 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, result in zip(axes.flat, results):
        _scatter_mds(axis, result, result["embedding_2d"], is_3d=False)
        axis.set_title(
            f"{result['metric_id'].item()} | "
            f"stress={result['embedding_2d_normalized_stress'].item():.4f}",
            fontsize=TITLE_FONT_SIZE,
        )
    for axis in axes.flat[len(results) :]:
        axis.set_visible(False)
    figure.suptitle(title, fontsize=OVERVIEW_TITLE_FONT_SIZE)
    _guides(figure, list(axes.flat[: len(results)]), results[0])
    figure.savefig(figure_dir / "mds_2d_overview.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_3d_overview(
    results: list[dict[str, np.ndarray]],
    figure_dir: Path,
    title: str,
    dpi: int,
    view: str | dict[str, float],
) -> None:
    """Plot one three-dimensional overview for compatible MDS results."""
    columns = min(3, len(results))
    rows = math.ceil(len(results) / columns)
    figure = plt.figure(
        figsize=(6 * columns, 5 * rows),
        constrained_layout=True,
    )
    axes = []
    for index, result in enumerate(results):
        axis = figure.add_subplot(rows, columns, index + 1, projection="3d")
        axes.append(axis)
        _scatter_mds(axis, result, result["embedding_3d"], is_3d=True, view=view)
        axis.set_title(
            f"{result['metric_id'].item()} | "
            f"stress={result['embedding_3d_normalized_stress'].item():.4f}",
            fontsize=TITLE_FONT_SIZE,
        )
    figure.suptitle(title, fontsize=OVERVIEW_TITLE_FONT_SIZE)
    _guides(figure, axes, results[0])
    figure.savefig(figure_dir / "mds_3d_overview.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_pca_overview(
    results: list[dict[str, np.ndarray]],
    figure_dir: Path,
    title: str,
    dpi: int,
) -> None:
    """Plot one two-dimensional PCA overview for compatible results."""
    columns = min(3, len(results))
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6 * columns, 5 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    for axis, result in zip(axes.flat, results):
        _scatter_pca(axis, result)
        explained = 100.0 * result["pca_explained_variance_ratio_2d"].sum()
        axis.set_title(
            f"{result['metric_id'].item()} | variance explained={explained:.1f}%",
            fontsize=TITLE_FONT_SIZE,
        )
    for axis in axes.flat[len(results) :]:
        axis.set_visible(False)
    figure.suptitle(f"{title} | signal PCA", fontsize=OVERVIEW_TITLE_FONT_SIZE)
    _guides(figure, list(axes.flat[: len(results)]), results[0])
    figure.savefig(figure_dir / "pca_2d_overview.png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _likely_log(values: np.ndarray) -> bool:
    if len(values) < 3 or np.any(values <= 0):
        return False
    linear_error = np.std(np.diff(values)) / max(abs(np.mean(np.diff(values))), 1e-12)
    log_differences = np.diff(np.log(values))
    log_error = np.std(log_differences) / max(abs(np.mean(log_differences)), 1e-12)
    return log_error < linear_error


def _plot_signal_slices(
    result: dict[str, np.ndarray],
    figure_dir: Path,
    dpi: int,
    progress: Any | None = None,
) -> None:
    transformed = ~result["is_none"]
    lambdas = np.unique(result["lambda_values"][transformed])
    alphas = np.unique(result["alpha_values"][transformed])
    taus = np.unique(result["tau_values"][transformed])
    metric_id = result["metric_id"].item()
    signal_dir = figure_dir / "signals" / metric_id
    signal_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(taus)))
    for lambda_value in lambdas:
        for alpha_value in alphas:
            figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            if result["metric_mode"].item() == "signal":
                x_values = result["parameter_values"]
                for tau_value, color in zip(taus, colors):
                    mask = (
                        transformed
                        & np.isclose(result["lambda_values"], lambda_value)
                        & np.isclose(result["alpha_values"], alpha_value)
                        & np.isclose(result["tau_values"], tau_value)
                    )
                    index = np.flatnonzero(mask)
                    if len(index) != 1:
                        raise ValueError(
                            "Signal slice does not identify exactly one condition"
                        )
                    batch_scores = result["scores"][index[0]]
                    mean = batch_scores.mean(axis=0)
                    std = batch_scores.std(axis=0)
                    axis.plot(
                        x_values,
                        mean,
                        color=color,
                        linewidth=2,
                        label=rf"$\tau={tau_value:g}$",
                    )
                    axis.fill_between(
                        x_values,
                        mean - std,
                        mean + std,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                    )
                if _likely_log(x_values):
                    axis.set_xscale("log")
                axis.set_xlabel(
                    result["parameter_name"].item(),
                    fontsize=AXIS_LABEL_FONT_SIZE,
                )
            else:
                means, deviations = [], []
                for tau_value in taus:
                    mask = (
                        transformed
                        & np.isclose(result["lambda_values"], lambda_value)
                        & np.isclose(result["alpha_values"], alpha_value)
                        & np.isclose(result["tau_values"], tau_value)
                    )
                    index = np.flatnonzero(mask)
                    if len(index) != 1:
                        raise ValueError(
                            "Fixed-score slice does not identify exactly one condition"
                        )
                    values = result["scores"][index[0], :, 0]
                    means.append(values.mean())
                    deviations.append(values.std())
                means = np.asarray(means)
                deviations = np.asarray(deviations)
                axis.plot(taus, means, marker="o", linewidth=2, color="tab:blue")
                axis.fill_between(
                    taus,
                    means - deviations,
                    means + deviations,
                    color="tab:blue",
                    alpha=0.18,
                    linewidth=0,
                )
                axis.set_xlabel(r"$\tau$", fontsize=AXIS_LABEL_FONT_SIZE)
            axis.set_ylabel("similarity", fontsize=AXIS_LABEL_FONT_SIZE)
            axis.set_title(
                rf"{metric_id}: $\lambda={lambda_value:g}$, $\alpha={alpha_value:g}$",
                fontsize=TITLE_FONT_SIZE,
            )
            axis.tick_params(labelsize=TICK_FONT_SIZE)
            axis.grid(alpha=0.3)
            if result["metric_mode"].item() == "signal":
                axis.legend(fontsize=LEGEND_FONT_SIZE)
            figure.savefig(
                signal_dir / f"lambda_{lambda_value:g}_alpha_{alpha_value:g}.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(figure)
            if progress is not None:
                progress.update(1)


def _plot_alpha_signal_slices(
    result: dict[str, np.ndarray],
    figure_dir: Path,
    dpi: int,
    progress: Any | None = None,
) -> None:
    """Plot alpha behavior for every fixed lambda and tau pair."""
    transformed = ~result["is_none"]
    lambdas = np.unique(result["lambda_values"][transformed])
    alphas = np.unique(result["alpha_values"][transformed])
    taus = np.unique(result["tau_values"][transformed])
    metric_id = result["metric_id"].item()
    signal_dir = figure_dir / "signals_by_alpha" / metric_id
    signal_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(alphas)))
    for lambda_value in lambdas:
        for tau_value in taus:
            figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
            if result["metric_mode"].item() == "signal":
                x_values = result["parameter_values"]
                for alpha_value, color in zip(alphas, colors):
                    mask = (
                        transformed
                        & np.isclose(result["lambda_values"], lambda_value)
                        & np.isclose(result["alpha_values"], alpha_value)
                        & np.isclose(result["tau_values"], tau_value)
                    )
                    index = np.flatnonzero(mask)
                    if len(index) != 1:
                        raise ValueError(
                            "Alpha signal slice does not identify exactly one condition"
                        )
                    batch_scores = result["scores"][index[0]]
                    mean = batch_scores.mean(axis=0)
                    std = batch_scores.std(axis=0)
                    axis.plot(
                        x_values,
                        mean,
                        color=color,
                        linewidth=2,
                        label=rf"$\alpha={alpha_value:g}$",
                    )
                    axis.fill_between(
                        x_values,
                        mean - std,
                        mean + std,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                    )
                if _likely_log(x_values):
                    axis.set_xscale("log")
                axis.set_xlabel(
                    result["parameter_name"].item(),
                    fontsize=AXIS_LABEL_FONT_SIZE,
                )
            else:
                means, deviations = [], []
                for alpha_value in alphas:
                    mask = (
                        transformed
                        & np.isclose(result["lambda_values"], lambda_value)
                        & np.isclose(result["alpha_values"], alpha_value)
                        & np.isclose(result["tau_values"], tau_value)
                    )
                    index = np.flatnonzero(mask)
                    if len(index) != 1:
                        raise ValueError(
                            "Fixed alpha slice does not identify exactly one condition"
                        )
                    values = result["scores"][index[0], :, 0]
                    means.append(values.mean())
                    deviations.append(values.std())
                means = np.asarray(means)
                deviations = np.asarray(deviations)
                axis.plot(alphas, means, marker="o", linewidth=2, color="tab:purple")
                axis.fill_between(
                    alphas,
                    means - deviations,
                    means + deviations,
                    color="tab:purple",
                    alpha=0.18,
                    linewidth=0,
                )
                axis.set_xlabel(r"$\alpha$", fontsize=AXIS_LABEL_FONT_SIZE)
            axis.set_ylabel("similarity", fontsize=AXIS_LABEL_FONT_SIZE)
            axis.set_title(
                rf"{metric_id}: $\lambda={lambda_value:g}$, $\tau={tau_value:g}$",
                fontsize=TITLE_FONT_SIZE,
            )
            axis.tick_params(labelsize=TICK_FONT_SIZE)
            axis.grid(alpha=0.3)
            if result["metric_mode"].item() == "signal":
                axis.legend(fontsize=LEGEND_FONT_SIZE)
            figure.savefig(
                signal_dir / f"lambda_{lambda_value:g}_tau_{tau_value:g}.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(figure)
            if progress is not None:
                progress.update(1)


def plot_group(
    config: dict[str, Any], group: DatasetGroup | GroupRef, result_paths: list[Path]
) -> None:
    """Create full-grid, lambda-split, and metric-signal figures."""
    data_root, figures_root = output_paths(config)
    figure_dir = figures_root / group.model / group.dataset
    figure_dir.mkdir(parents=True, exist_ok=True)
    results = [load_npz(path) for path in result_paths]
    dpi = config["visualization"]["dpi"]
    mds_3d_view = config["visualization"]["mds_3d_view"]
    first = results[0]
    transformed = ~first["is_none"]
    lambda_count = len(np.unique(first["lambda_values"][transformed]))
    alpha_count = len(np.unique(first["alpha_values"][transformed]))
    tau_count = len(np.unique(first["tau_values"][transformed]))
    tau_slice_count = lambda_count * alpha_count
    alpha_slice_count = lambda_count * tau_count
    with tqdm(
        total=(
            len(results) * (tau_slice_count + alpha_slice_count + 3)
            + 3
            + len(config["lambda_splits"]) * (len(results) * 3 + 3)
        ),
        desc=f"Plots {group.model}/{group.dataset}",
        unit="figure",
        leave=False,
        disable=None,
    ) as progress:
        for result in results:
            progress.set_postfix_str(result["metric_id"].item(), refresh=False)
            _plot_metric_mds(result, figure_dir, dpi, mds_3d_view)
            progress.update(2)
            _plot_metric_pca(result, figure_dir, dpi)
            progress.update(1)
            _plot_signal_slices(result, figure_dir, dpi, progress)
            _plot_alpha_signal_slices(result, figure_dir, dpi, progress)

        _plot_overview(results, figure_dir, f"{group.model} | {group.dataset}", dpi)
        _plot_3d_overview(
            results,
            figure_dir,
            f"{group.model} | {group.dataset}",
            dpi,
            mds_3d_view,
        )
        _plot_pca_overview(results, figure_dir, f"{group.model} | {group.dataset}", dpi)
        progress.update(3)

        for lambda_value in config["lambda_splits"]:
            split_results = [
                load_npz(
                    lambda_split_result_path(
                        data_root,
                        group,
                        result["metric_id"].item(),
                        lambda_value,
                    )
                )
                for result in results
            ]
            split_dir = figure_dir / "lambda_splits" / f"lambda_{lambda_value:g}"
            for result in split_results:
                progress.set_postfix_str(
                    f"lambda={lambda_value:g}/{result['metric_id'].item()}",
                    refresh=False,
                )
                _plot_metric_mds(result, split_dir, dpi, mds_3d_view)
                progress.update(2)
                _plot_metric_pca(result, split_dir, dpi)
                progress.update(1)
            _plot_overview(
                split_results,
                split_dir,
                f"{group.model} | {group.dataset} | lambda={lambda_value:g}",
                dpi,
            )
            _plot_3d_overview(
                split_results,
                split_dir,
                f"{group.model} | {group.dataset} | lambda={lambda_value:g}",
                dpi,
                mds_3d_view,
            )
            _plot_pca_overview(
                split_results,
                split_dir,
                f"{group.model} | {group.dataset} | lambda={lambda_value:g}",
                dpi,
            )
            progress.update(3)
