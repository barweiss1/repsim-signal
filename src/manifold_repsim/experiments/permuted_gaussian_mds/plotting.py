"""Dataset and MDS plotting for the permuted-Gaussian experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

from ..paper_style import highlight_reference_point
from .config import OPACITY_RANGE
from .config import PARAMETERS
from .config import PARAMETER_LABELS
from .config import POINT_SIZE_RANGE
from .config import SIGNAL_METRICS
from .config import output_paths
from .dataset import metadata
from .dataset import parameter_levels
from .mds import fit_metric_mds


def _project(data: np.ndarray) -> np.ndarray:
    return data if data.shape[1] == 2 else PCA(n_components=2).fit_transform(data)


def _color_norm(config: dict[str, Any]) -> Normalize:
    color_parameter = config["visualization"]["encodings"]["color"]
    if color_parameter is None:
        return Normalize(0.0, 1.0)
    levels = parameter_levels(color_parameter, config)
    return Normalize(levels.min(), levels.max())


def _scaled_value(
    value: float,
    levels: np.ndarray,
    target: tuple[float, float],
) -> float:
    if levels.max() == levels.min():
        return target[0]
    fraction = (value - levels.min()) / (levels.max() - levels.min())
    return target[0] + fraction * np.ptp(target)


def parameter_value_label(name: str, value: float) -> str:
    """Format one configured data parameter for plot legends."""
    return f"{PARAMETER_LABELS[name]} = {value:g}"


def _sample_levels(levels: np.ndarray, count: int = 5) -> np.ndarray:
    """Return up to `count` evenly spaced unique levels, ends included."""
    # indices = np.unique(
    #     np.round(np.linspace(0, len(levels) - 1, min(len(levels), count))).astype(int)
    # )
    indices = np.arange(len(levels))
    new_levels = np.linspace(0, levels.max(), min(len(levels), count))
    # return np.linspace(0, len(levels) - 1, min(len(levels), count))
    return levels[indices]


def _scatter_encoded(
    ax: plt.Axes,
    coordinates: tuple[np.ndarray, ...],
    encodings: dict[str, np.ndarray],
    base_mask: np.ndarray,
    config: dict[str, Any],
    *,
    is_3d: bool = False,
    cmap: Any = "inferno",
    norm: Any = None,
) -> None:
    color_parameter = config["visualization"]["encodings"]["color"]
    common = {"marker": "o"}
    if color_parameter is None:
        common["color"] = "tab:blue"
    else:
        common.update(cmap=cmap, norm=norm if norm is not None else _color_norm(config))
    if is_3d:
        common["depthshade"] = False
    for opacity in np.unique(encodings["opacity"]):
        mask = np.isclose(encodings["opacity"], opacity) & ~base_mask
        if mask.any():
            color = {} if color_parameter is None else {"c": encodings["color"][mask]}
            ax.scatter(
                *(axis[mask] for axis in coordinates),
                s=encodings["size"][mask],
                alpha=float(opacity),
                edgecolors="none",
                **color,
                **common,
            )
        base = np.isclose(encodings["opacity"], opacity) & base_mask
        if base.any():
            color = {} if color_parameter is None else {"c": encodings["color"][base]}
            ax.scatter(
                *(axis[base] for axis in coordinates),
                s=encodings["size"][base],
                alpha=float(opacity),
                edgecolors="none",
                zorder=5,
                **color,
                **common,
            )
    if base_mask.any():
        highlight_reference_point(
            ax,
            tuple(axis[base_mask] for axis in coordinates),
            encodings["size"][base_mask],
            is_3d=is_3d,
        )


COLORBAR_SHRINK = 0.6
COLORBAR_TITLE_FONTSIZE = 12


def _shrunk_span(
    bottom: float,
    top: float,
    shrink: float = COLORBAR_SHRINK,
) -> tuple[float, float]:
    """Center a `shrink`-fraction sub-span inside [bottom, top]."""
    height = (top - bottom) * shrink
    new_bottom = bottom + ((top - bottom) - height) / 2
    return new_bottom, new_bottom + height


def _channel_panel_axes(
    figure: plt.Figure,
    rect: tuple[float, float, float, float],
    levels: np.ndarray,
    target: tuple[float, float],
    *,
    is_size: bool,
    dense: bool,
    title: str,
) -> plt.Axes:
    """Draw one vertical legend panel for the size or opacity channel.

    Point size always shows a few discrete representative circle sizes
    rather than a continuous ramp, since overlapping circles of similar
    size otherwise read as one solid shape. Opacity draws a continuous
    colorbar-style gradient when `dense=True`, or the same sparse
    representative-marker style as size when `dense=False` (the "markers"
    style). Both channels always share the same `rect`, so the panel is the
    same size either way and matches the color colorbar's shrunk height.
    """
    axis = figure.add_axes(rect)
    if is_size:
        samples = _sample_levels(levels, count=8)
        axis.scatter(
            np.zeros_like(samples),
            samples,
            s=[_scaled_value(value, levels, target) for value in samples],
            color="0.5",
            edgecolors="none",
            clip_on=False,
        )
        tick_levels = samples
    elif dense:
        samples = np.linspace(levels.min(), levels.max(), 256)
        image = np.zeros((256, 1, 4))
        image[:, 0, :3] = 0.15
        image[:, 0, 3] = [_scaled_value(value, levels, target) for value in samples]
        axis.imshow(
            image,
            aspect="auto",
            origin="lower",
            extent=(-1, 1, levels.min(), levels.max()),
        )
        tick_levels = _sample_levels(levels)
    else:
        samples = _sample_levels(levels)
        axis.scatter(
            np.zeros_like(samples),
            samples,
            s=80,
            color=[
                (0.15, 0.15, 0.15, _scaled_value(value, levels, target))
                for value in samples
            ],
            edgecolors="none",
            clip_on=False,
        )
        tick_levels = samples
    axis.set_xlim(-1, 1)
    axis.set_ylim(levels.min(), levels.max())
    axis.set_xticks([])
    axis.yaxis.tick_right()
    axis.set_yticks(tick_levels)
    axis.set_yticklabels([f"{value:.2f}" for value in tick_levels])
    for spine in ("top", "left", "bottom"):
        axis.spines[spine].set_visible(False)
    axis.yaxis.set_label_position("right")
    axis.set_ylabel(title, fontsize=COLORBAR_TITLE_FONTSIZE, rotation=0, labelpad=28)
    return axis


def _side_panel_bounds(
    figure: plt.Figure,
    axes_list: list[plt.Axes],
    colorbar: Any,
) -> tuple[float, float, float]:
    """Return (left, bottom, top) figure-fraction bounds for side legends.

    Uses each artist's rendered tight bounding box (not its nominal
    ``get_position()`` slot) since 3D axes draw tick/axis labels outside
    their reserved rectangle, and a wrong estimate here means the legend
    panels overlap real plot content.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inverted = figure.transFigure.inverted()

    def tight_bbox(artist: Any) -> Any:
        return artist.get_tightbbox(renderer).transformed(inverted)

    boxes = [tight_bbox(axis) for axis in axes_list]
    top = max(box.y1 for box in boxes)
    bottom = min(box.y0 for box in boxes)
    if colorbar is not None:
        left = tight_bbox(colorbar.ax).x1 + 0.03
    else:
        left = max(box.x1 for box in boxes) + 0.02
    return left, bottom, top


def _encoding_guides(
    figure: plt.Figure,
    axes: Any,
    config: dict[str, Any],
    encodings: dict[str, np.ndarray],
) -> None:
    mapping = config["visualization"]["encodings"]
    dense = config["visualization"]["legend_style"] == "colorbar"
    axes_list = list(np.asarray(axes, dtype=object).ravel())
    colorbar = None
    if mapping["color"] is not None:
        colorbar = figure.colorbar(
            plt.cm.ScalarMappable(norm=_color_norm(config), cmap="viridis"),
            ax=axes_list,
            shrink=COLORBAR_SHRINK,
            pad=0.02,
        )
        colorbar.set_label(
            PARAMETER_LABELS[mapping["color"]],
            rotation=0,
            labelpad=12,
            fontsize=COLORBAR_TITLE_FONTSIZE,
        )
        colorbar.set_ticks(parameter_levels(mapping["color"], config))

    left, bottom, top = _side_panel_bounds(figure, axes_list, colorbar)
    panel_bottom, panel_top = _shrunk_span(bottom, top)
    channels = []
    if mapping["size"] is not None:
        channels.append(
            ("Size", mapping["size"], encodings["size_levels"], POINT_SIZE_RANGE, True)
        )
    if mapping["opacity"] is not None:
        channels.append(
            (
                "Opacity",
                mapping["opacity"],
                encodings["opacity_levels"],
                OPACITY_RANGE,
                False,
            )
        )

    # Sized in physical inches (not figure-fraction) so panels stay a
    # consistent width regardless of figure width. Each panel's rotated
    # label extent isn't known ahead of time, so the next panel's left edge
    # is placed from the previous one's measured rendered bbox rather than
    # a guessed gap - otherwise labels overlap on narrower figures.
    fig_width_in = figure.get_size_inches()[0]
    panel_width = 0.35 / fig_width_in
    inter_gap = 0.15 / fig_width_in
    panel_left = left
    for label, name, levels, target, is_size in channels:
        # title = f"{label} \n {PARAMETER_LABELS[name]}"
        title = PARAMETER_LABELS[name]
        axis = _channel_panel_axes(
            figure,
            (panel_left, panel_bottom, panel_width, panel_top - panel_bottom),
            levels,
            target,
            is_size=is_size,
            dense=dense,
            title=title,
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        right_edge = (
            axis.get_tightbbox(renderer).transformed(figure.transFigure.inverted()).x1
        )
        panel_left = right_edge + inter_gap


def plot_dataset_examples(
    base: np.ndarray,
    transformed: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray,
    figure_dir: Path,
    config: dict[str, Any],
) -> None:
    """Plot representative datasets and the configured parameter encodings."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("parameter_grid_rgb.png", "rgb_key.png"):
        (figure_dir / legacy_name).unlink(missing_ok=True)
    moderate = np.array(
        [
            levels[(len(levels) - 1) // 2]
            for levels in (np.unique(grid[:, index]) for index in range(3))
        ]
    )
    targets = (
        ("Nominal base", None),
        ("Zero grid corner", (0, 0, 0)),
        (parameter_value_label(PARAMETERS[0], moderate[0]), (moderate[0], 0, 0)),
        (parameter_value_label(PARAMETERS[1], moderate[1]), (0, moderate[1], 0)),
        (parameter_value_label(PARAMETERS[2], moderate[2]), (0, 0, moderate[2])),
        ("Moderate combined", tuple(moderate)),
    )
    examples = [
        (
            title,
            (
                base
                if target is None
                else transformed[np.square(grid - target).sum(axis=1).argmin()]
            ),
        )
        for title, target in targets
    ]
    projected = [(title, _project(data)) for title, data in examples]
    combined = np.vstack([data for _, data in projected])
    padding = np.maximum(np.ptp(combined, axis=0) * 0.05, 0.05)
    limits = tuple(zip(combined.min(axis=0) - padding, combined.max(axis=0) + padding))
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    cmap = plt.get_cmap("tab20", config["dataset"]["n_clusters"])
    for axis, (title, data) in zip(axes.flat, projected):
        axis.scatter(
            data[:, 0],
            data[:, 1],
            c=labels,
            cmap=cmap,
            s=10,
            alpha=0.65,
            linewidths=0,
        )
        axis.set(title=title, xlim=limits[0], ylim=limits[1])
        axis.set_aspect("equal")
    figure.savefig(
        figure_dir / "dataset_examples.png",
        dpi=config["visualization"]["dpi"],
    )
    plt.close(figure)

    encodings, _ = metadata(config, grid)
    figure = plt.figure(figsize=(8, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    parameters = np.vstack(
        ([config["base_transform"][name] for name in PARAMETERS], grid)
    )
    _scatter_encoded(
        axis,
        tuple(parameters.T),
        encodings,
        np.arange(len(parameters)) == 0,
        config,
        is_3d=True,
    )
    axis.set(xlabel=PARAMETERS[0], ylabel=PARAMETERS[1], zlabel=PARAMETERS[2])
    _encoding_guides(figure, [axis], config, encodings)
    figure.savefig(
        figure_dir / "parameter_grid_encodings.png",
        dpi=config["visualization"]["dpi"],
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_embedding(
    axis: plt.Axes,
    embedding: np.ndarray,
    encodings: dict[str, np.ndarray],
    base_mask: np.ndarray,
    config: dict[str, Any],
    title: str,
    stress: float,
    *,
    is_3d: bool = False,
    cmap: Any = "inferno",
    norm: Any = None,
    aspect: str = "equal",
) -> None:
    _scatter_encoded(
        axis,
        tuple(embedding.T),
        encodings,
        base_mask,
        config,
        is_3d=is_3d,
        cmap=cmap,
        norm=norm,
    )
    labels = {
        # "title": f"{title}\nstress={stress:.4f}",
        "title": f"{title}",
        "xlabel": "MDS 1",
        "ylabel": "MDS 2",
    }
    if is_3d:
        labels["zlabel"] = "MDS 3"
        center = (embedding.min(axis=0) + embedding.max(axis=0)) / 2
        radius = max(np.ptp(embedding, axis=0).max() / 2, 1e-8)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_box_aspect((1, 1, 1))
    elif aspect == "equal":
        axis.set_aspect("equal", adjustable="datalim")
    else:
        # The paper figures pass "auto": an MDS/PCA panel whose two axes have
        # very different variance gets padded out under equal units until its
        # real structure is illegible. The exploratory default stays "equal".
        axis.set_aspect(aspect)
    axis.set(**labels)
    axis.grid(alpha=0.2)


def plot_metric_results(
    config: dict[str, Any],
    paths: list[Path],
    encodings: dict[str, np.ndarray],
    base_mask: np.ndarray,
) -> None:
    """Plot per-metric and overview 2D/3D MDS embeddings."""
    figure_dir = output_paths(config)[1]
    metric_dir = figure_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        with np.load(path) as saved:
            results.append({key: saved[key].copy() for key in saved.files})
    for result in results:
        for prefix in ("base_relative", "direct"):
            result[f"{prefix}_3d"] = fit_metric_mds(
                result[f"{prefix}_distance"],
                config,
                n_components=3,
            )
    dpi = config["visualization"]["dpi"]
    for result in results:
        metric_id = result["metric_id"].item()
        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for axis, prefix, title in zip(
            axes,
            ("base_relative", "direct"),
            ("base-relative", "direct pairwise"),
        ):
            _plot_embedding(
                axis,
                result[f"{prefix}_embedding"],
                encodings,
                base_mask,
                config,
                f"{metric_id}: {title}",
                result[f"{prefix}_normalized_stress"].item(),
            )
        _encoding_guides(figure, axes, config, encodings)
        figure.savefig(metric_dir / f"{metric_id}.png", dpi=dpi, bbox_inches="tight")
        plt.close(figure)

        metric_3d_dir = figure_dir / "metrics_3d"
        metric_3d_dir.mkdir(parents=True, exist_ok=True)
        figure = plt.figure(figsize=(13, 6), constrained_layout=True)
        axes_3d = [figure.add_subplot(1, 2, index, projection="3d") for index in (1, 2)]
        for axis, prefix, title in zip(
            axes_3d,
            ("base_relative", "direct"),
            ("base-relative", "direct pairwise"),
        ):
            embedding, _, stress, _ = result[f"{prefix}_3d"]
            _plot_embedding(
                axis,
                embedding,
                encodings,
                base_mask,
                config,
                f"{metric_id}: {title}",
                stress,
                is_3d=True,
            )
        _encoding_guides(figure, axes_3d, config, encodings)
        figure.savefig(
            metric_3d_dir / f"{metric_id}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)

    for prefix, title, filename in (
        ("base_relative", "Base-relative MDS", "mds_base_relative_overview.png"),
        ("direct", "Direct-pairwise MDS", "mds_pairwise_overview.png"),
    ):
        figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
        for axis, result in zip(axes.flat, results):
            _plot_embedding(
                axis,
                result[f"{prefix}_embedding"],
                encodings,
                base_mask,
                config,
                result["metric_id"].item(),
                result[f"{prefix}_normalized_stress"].item(),
            )
        for axis in axes.flat[len(results) :]:
            axis.set_visible(False)
        figure.suptitle(title)
        _encoding_guides(figure, axes.flat[: len(results)], config, encodings)
        figure.savefig(figure_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(figure)

    for prefix, title, filename in (
        ("base_relative", "Base-relative 3D MDS", "mds_3d_base_relative_overview.png"),
        ("direct", "Direct-pairwise 3D MDS", "mds_3d_pairwise_overview.png"),
    ):
        figure = plt.figure(figsize=(18, 10), constrained_layout=True)
        axes_3d = [
            figure.add_subplot(2, 4, index + 1, projection="3d")
            for index in range(len(results))
        ]
        for axis, result in zip(axes_3d, results):
            embedding, _, stress, _ = result[f"{prefix}_3d"]
            _plot_embedding(
                axis,
                embedding,
                encodings,
                base_mask,
                config,
                result["metric_id"].item(),
                stress,
                is_3d=True,
            )
        figure.suptitle(title)
        _encoding_guides(figure, axes_3d, config, encodings)
        figure.savefig(figure_dir / filename, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def plot_signal_sweeps(
    config: dict[str, Any],
    curves: dict[str, dict[str, Any]],
) -> None:
    """Plot signal-vs-rbf_sigma curves for isolated single-parameter sweeps."""
    figure_dir = output_paths(config)[1] / "signal_sweeps"
    figure_dir.mkdir(parents=True, exist_ok=True)
    dpi = config["visualization"]["dpi"]
    logscale = config["metrics"]["signals"]["scale"] == "log"
    for parameter, curve in curves.items():
        sigma = curve["sigma"]
        values = curve["values"]
        colors = plt.cm.magma(np.linspace(0.08, 0.85, len(values)))
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        for axis, metric_name in zip(axes.flat, SIGNAL_METRICS):
            for value, color, row in zip(values, colors, curve[metric_name]):
                axis.plot(
                    sigma,
                    row,
                    marker="o",
                    markersize=3,
                    linewidth=1.6,
                    color=color,
                    label=parameter_value_label(parameter, value),
                )
            axis.set(title=metric_name, xlabel="rbf_sigma", ylabel="similarity to base")
            if logscale:
                axis.set_xscale("log")
            axis.grid(alpha=0.25)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside right upper",
            title=PARAMETER_LABELS[parameter],
            fontsize=COLORBAR_TITLE_FONTSIZE,
            title_fontsize=9,
            frameon=False,
        )
        figure.suptitle(f"Signal sweep: {PARAMETER_LABELS[parameter]}")
        figure.savefig(
            figure_dir / f"{parameter}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(figure)
