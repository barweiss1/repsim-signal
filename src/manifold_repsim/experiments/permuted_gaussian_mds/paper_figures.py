"""Publication-ready figures for the synthetic permuted-Gaussian MDS experiment.

This module builds a small, curated set of paper figures from an already
generated and analyzed experiment run. It reads saved dataset and metric
artifacts directly (via `load_dataset_artifacts`/`load_npz`) and performs no
metric computation, MDS fitting, or dataset transforms of its own; it is a
thin composition of the existing plotting primitives in `plotting.py`, styled
for print rather than exploration.

Everything that should look identical to the glocal experiment's paper
figures -- the two appear side by side in the same paper -- lives in
`manifold_repsim.experiments.paper_style` and is imported rather than
reimplemented: rcParams, the magma ramp, legend-column geometry, the
rank-banded colorbar and size-legend panels, and the signal-row styling
tail. The two experiment packages still never import each other.

Both the color and opacity legend columns are rank-banded colorbars, not
smooth gradients: `_color_bands`/`_opacity_bands` give one solid band per
configured level -- color from the per-level magma samples the signal-sweep
curves also use, opacity from the shared `paper_style.OPACITY_RANGE` -- and
`_paper_encodings` re-expresses the scatter points' color as a band index
(`paper_style.rank_index`) over the same `levels` array, so a point's color
band and its colorbar label can never disagree. This overrides the exploratory pipeline's continuous "inferno"
default for the paper MDS grid only. Size has no colorbar equivalent and
stays a representative-marker column (`paper_style.size_legend_panel`).

MDS grid panels drop their tick numbers/marks (the embedding's numeric scale
is arbitrary) but keep the exploratory `aspect="equal"`. Independent per-axis
scaling ("auto") is unsafe here: several base-relative embeddings are
effectively one-dimensional (the second MDS axis carries ~0.1% of the
variance), so "auto" magnified that residual by an order of magnitude and
turned a straight line into an apparent trend -- and whether it did so at all
depended on the arbitrary orientation of the MDS solution, not on the data.

The data-description and MDS grid figures are laid out as a single
horizontal row (1x4) rather than a 2x2 grid, matching the signal-sweep
figures, which were already horizontal (1x3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap

from ..artifacts import load_npz
from ..paper_style import EMBEDDING_GRID_FIGSIZE
from ..paper_style import EMBEDDING_GRID_RECT
from ..paper_style import OPACITY_RANGE
from ..paper_style import POINT_SIZE_RANGE
from ..paper_style import SIGNAL_LEGEND_FONTSIZE
from ..paper_style import SIGNAL_ROW_FIGSIZE
from ..paper_style import SIGNAL_SHAPE_FIGSIZE
from ..paper_style import apply_paper_style
from ..paper_style import discrete_colorbar_panel
from ..paper_style import legend_panel_metrics
from ..paper_style import magma_ramp
from ..paper_style import opacity_swatches
from ..paper_style import place_panel
from ..paper_style import rank_index
from ..paper_style import rescale_channel
from ..paper_style import save_figure
from ..paper_style import shrunk_span
from ..paper_style import side_panel_bounds
from ..paper_style import size_legend_panel
from ..paper_style import style_signal_row
from .analysis import result_path
from .config import PARAMETER_LABELS
from .config import PARAMETERS
from .config import REPO_ROOT
from .config import SIGNAL_METRICS
from .config import output_paths
from .dataset import load_dataset_artifacts
from .dataset import metadata
from .dataset import parameter_levels
from .plotting import _plot_embedding
from .plotting import parameter_value_label

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "figures" / "paper_ready" / "synthetic"

# (metric_id, paper title) pairs for the MDS grid, in display order.
MDS_GRID_METRICS: tuple[tuple[str, str], ...] = (
    ("cka_lin", r"$CKA_{\mathrm{lin}}$"),
    ("cka_rbf_sigma_02", r"$CKA_{\mathrm{RBF}}(\sigma=0.2)$"),
    ("mutual_knn_dist_k10", r"Mutual $10$-NN"),
    ("cka_rbf", r"$s_{CKA}(\sigma)$"),
)

# metric_id -> y-axis label for `plot_signal_sweep_rows`, matching the
# glocal paper module's own SIGNAL_SWEEP_METRICS convention (a per-metric
# symbol, e.g. "$s_{CKA}(\sigma)$", rather than a generic "similarity to
# base" shared across every metric's figure); "cka_rbf"/"rbf_degree_crwka"
# use the identical label glocal uses for the same metric.
SIGNAL_SWEEP_METRIC_LABELS: dict[str, str] = {
    "cka_rbf": r"$s_{CKA}(\sigma)$",
    "rbf_uka": r"$s_{UKA}(\sigma)$",
    "rbf_rwka_symmetric": r"$s_{RWKA}(\sigma)$",
    "rbf_degree_crwka": r"$s_{dRWKA}(\sigma)$",
}


def _channel_levels(config: dict[str, Any], channel: str) -> np.ndarray | None:
    """Configured levels for one visualization channel, or `None` if unmapped."""
    parameter = config["visualization"]["encodings"][channel]
    if parameter is None:
        return None
    return parameter_levels(parameter, config)


def _color_bands(levels: np.ndarray) -> list[tuple[float, ...]]:
    """One magma color per color-channel level.

    Sampled by rank across the shared magma range, exactly as
    `plot_signal_sweep_rows` samples its curve colors, so a given parameter
    value reads as the same color in the MDS grid and the signal sweeps.
    """
    return [tuple(color) for color in magma_ramp(len(levels))]


def _opacity_bands(levels: np.ndarray) -> list[tuple[float, ...]]:
    """One gray-at-opacity swatch per opacity-channel level, using the same
    `OPACITY_RANGE` mapping the scatter points themselves use."""
    return opacity_swatches(rescale_channel(levels, levels, OPACITY_RANGE))


def _paper_encodings(
    config: dict[str, Any], encodings: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Re-express `dataset.metadata`'s encodings for the paper figures.

    Two differences from the exploratory encodings, which stay untouched on
    disk and in `plotting.py`'s own figures:

    - `color` becomes a band index (`rank_index`) rather than a raw parameter
      value, so points land on exactly the bands their rank-banded colorbar
      labels.
    - `size`/`opacity` are rescaled onto the shared `paper_style` ranges
      rather than this experiment's own `config` ranges, so points read at the
      same scale as the glocal figures' equivalents.
    """
    paper = dict(encodings)
    color_levels = _channel_levels(config, "color")
    if color_levels is not None:
        paper["color"] = (
            rank_index(encodings["color"], color_levels).astype(float) + 0.5
        )
    paper["size"] = rescale_channel(
        encodings["size_value"], encodings["size_levels"], POINT_SIZE_RANGE
    )
    if config["visualization"]["encodings"]["opacity"] is not None:
        paper["opacity"] = rescale_channel(
            encodings["opacity_value"], encodings["opacity_levels"], OPACITY_RANGE
        )
    return paper


def _nearest_condition(
    transformed: np.ndarray,
    grid: np.ndarray,
    target: tuple[float, float, float],
) -> np.ndarray:
    """Return the transformed dataset at an exact grid point.

    Raises if the nearest grid point is not an exact match, since a silent
    nearest-neighbor snap would mislabel the resulting figure panel.
    """
    target_array = np.asarray(target, dtype=float)
    index = int(np.square(grid - target_array).sum(axis=1).argmin())
    if not np.allclose(grid[index], target_array):
        raise ValueError(
            f"No exact grid point at {target}; nearest is {grid[index].tolist()}"
        )
    return transformed[index]


def plot_data_description(config: dict[str, Any], output_dir: Path) -> None:
    """Data description, 1x4 horizontal row: base data plus one panel per
    swept parameter."""
    base, transformed, labels, grid = load_dataset_artifacts(config)
    # A moderate (middle) level of each parameter's *actual* configured grid,
    # exactly like `plotting.plot_dataset_examples` - not a hardcoded value,
    # so this keeps working whatever the config's grid resolution is.
    moderate = np.array(
        [
            # levels[(len(levels) - 1) // 2]
            levels[-2]
            for levels in (np.unique(grid[:, index]) for index in range(3))
        ]
    )

    def single_parameter_target(
        parameter: str,
    ) -> tuple[str, tuple[float, float, float]]:
        """Label and grid point where only `parameter` leaves its base value.

        `moderate` and the grid columns are indexed by `PARAMETERS`' canonical
        order, so the coordinate is built by canonical position while the
        panels themselves follow `PAPER_PANEL_ORDER`.
        """
        position = PARAMETERS.index(parameter)
        coordinates = tuple(
            moderate[index] if index == position else 0.0 for index in range(3)
        )
        return parameter_value_label(parameter, moderate[position]), coordinates

    targets: tuple[tuple[str, tuple[float, float, float] | None], ...] = (
        ("Base data", None),
        *(single_parameter_target(parameter) for parameter in PAPER_PANEL_ORDER),
    )
    examples = [
        (
            title,
            base if target is None else _nearest_condition(transformed, grid, target),
        )
        for title, target in targets
    ]
    combined = np.vstack([data for _, data in examples])
    padding = np.maximum(np.ptp(combined, axis=0) * 0.05, 0.05)
    limits = tuple(zip(combined.min(axis=0) - padding, combined.max(axis=0) + padding))

    figure, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("tab20", config["dataset"]["n_clusters"])
    for axis, (title, data) in zip(axes.flat, examples):
        axis.scatter(
            data[:, 0], data[:, 1], c=labels, cmap=cmap, s=32, alpha=0.75, linewidths=0
        )
        axis.set(title=title, xlim=limits[0], ylim=limits[1])
        axis.set_aspect("equal")
    save_figure(figure, output_dir / "data_description.pdf")


# Left-to-right panel order for the paper's parameter-panel figures
# (convention: sigma, mix, perm) -- independent of `PARAMETERS`' canonical
# dataset/grid order, which other figures and the dataset column layout
# still rely on unchanged. Shared by `plot_data_description` and
# `plot_signal_sweep_rows` so the two read in the same order.
PAPER_PANEL_ORDER: tuple[str, ...] = (
    "noise_scale",
    "cluster_mixing_probability",
    "n_permute",
)


def plot_signal_sweep_rows(config: dict[str, Any], output_dir: Path) -> None:
    """One 1x3 figure per signal metric: all parameter sweeps, side by side."""
    data_dir, _ = output_paths(config)
    parts_dir = data_dir / ".scratch" / "signal_sweep_parts"
    curves = {
        parameter: load_npz(parts_dir / f"{parameter}.npz") for parameter in PARAMETERS
    }
    logscale = config["metrics"]["signals"]["scale"] == "log"

    for metric_name in SIGNAL_METRICS:
        figure, axes = plt.subplots(
            1, 3, figsize=SIGNAL_ROW_FIGSIZE, constrained_layout=True
        )
        for axis, parameter in zip(axes, PAPER_PANEL_ORDER):
            curve = curves[parameter]
            sigma = curve["sigma"]
            values = curve["values"]
            for value, color, row in zip(
                values, magma_ramp(len(values)), curve[metric_name]
            ):
                label = f"{int(value)}" if parameter == "n_permute" else f"{value:g}"
                axis.plot(sigma, row, marker="o", color=color, label=label)
        style_signal_row(
            axes,
            logscale=logscale,
            legend_titles=[
                PARAMETER_LABELS[parameter] for parameter in PAPER_PANEL_ORDER
            ],
            y_label=SIGNAL_SWEEP_METRIC_LABELS[metric_name],
        )
        save_figure(figure, output_dir / f"signal_sweep_{metric_name}.pdf")


# (filename stem, metric id pairs) for `plot_signal_shapes`. Each figure
# overlays two signal metrics measured on one single comparison, so the
# reader sees the shape of s(sigma) itself rather than how a sweep moves it.
# Paired as the paper discusses them: the two kernel-alignment signals, then
# the two random-walk ones.
SIGNAL_SHAPE_FIGURES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("cka_uka", ("cka_rbf", "rbf_uka")),
    ("rwka_drwka", ("rbf_rwka_symmetric", "rbf_degree_crwka")),
)

# Colour and marker follow the signal's sigma -> inf limit rather than its
# position in the pair, so the two signals sharing a limit read alike across
# both figures: s_CKA with s_dRWKA, and s_UKA with s_RWKA. Keyed off
# `SIGNAL_INFINITE_LIMIT` below so the pairing cannot drift from the limits
# actually drawn. These panels are the one place a plain series contrast is
# wanted rather than a position on the shared magma ramp, which encodes
# parameter level everywhere else, so the Matplotlib defaults are used and
# carry no level meaning.
SIGNAL_SHAPE_STYLE: dict[str, tuple[str, str]] = {
    "cka_lin": ("tab:blue", "o"),
    "one": ("tab:orange", "s"),
}

# Analytic limits of each signal. As sigma -> 0 every signal approaches 1.
# As sigma -> inf the RBF kernel degenerates toward the linear one, so the
# plain alignment signals approach linear CKA while the normalized ones
# approach 1. "cka_lin" is read from that metric's own result artifact at the
# same comparison rather than hardcoded.
SIGNAL_ZERO_LIMIT: float = 1.0
SIGNAL_INFINITE_LIMIT: dict[str, str] = {
    "cka_rbf": "cka_lin",
    "rbf_degree_crwka": "cka_lin",
    "rbf_uka": "one",
    "rbf_rwka_symmetric": "one",
}


def _signal_shape_fixed(config: dict[str, Any], metric_id: str) -> float:
    """One fixed metric measured at the shape comparison.

    `analysis.analyze_signal_shapes` measures the configured fixed metrics at
    its own condition and stores them beside the curves, because that
    condition need not lie on the swept grid and so has no row in the
    per-condition result artifacts.
    """
    data_dir, _ = output_paths(config)
    shapes = load_npz(data_dir / ".scratch" / "signal_shapes.npz")
    return float(shapes[f"fixed_{metric_id}"])


def _signal_shape_curves(
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Signal curves for the one comparison the shape figures draw.

    Written by `analysis.analyze_signal_shapes`, which computes its own
    transform rather than reading the swept grid, so this condition need not
    be one the grid visits.
    """
    data_dir, _ = output_paths(config)
    shapes = load_npz(data_dir / ".scratch" / "signal_shapes.npz")
    sigma = np.asarray(shapes["sigma"], dtype=float)
    curves = {
        metric: np.asarray(shapes[metric], dtype=float) for metric in SIGNAL_METRICS
    }
    return sigma, curves


def plot_signal_shapes(config: dict[str, Any], output_dir: Path) -> None:
    """One figure per metric pair: both signals of a single comparison.

    The sweep figures show how a signal moves as a parameter changes; these
    show the shape of the signal itself at one fixed, representative
    comparison, which is what the multiscale argument is actually about.
    Each signal's two analytic limits are drawn as reference lines, so the
    plotted range reads against where the curves are heading rather than
    only against itself.
    """
    sigma, curves = _signal_shape_curves(config)
    logscale = config["metrics"]["signals"]["scale"] == "log"
    linear_cka = _signal_shape_fixed(config, "cka_lin")

    for stem, metrics in SIGNAL_SHAPE_FIGURES:
        figure, axis = plt.subplots(
            figsize=SIGNAL_SHAPE_FIGSIZE, constrained_layout=True
        )
        # One line per distinct limit the panel's two metrics actually reach:
        # `SIGNAL_ZERO_LIMIT` is shared by both, and whichever metric tends to
        # linear CKA adds that second line.
        limits: dict[float, str] = {SIGNAL_ZERO_LIMIT: "$1$"}
        for metric in metrics:
            if SIGNAL_INFINITE_LIMIT[metric] == "cka_lin":
                limits[linear_cka] = r"$CKA_{\mathrm{lin}}$"
        for value, label in limits.items():
            axis.axhline(value, color="0.45", linestyle="--", linewidth=1.6, zorder=1)
            axis.annotate(
                label,
                xy=(0.015, value),
                xycoords=axis.get_yaxis_transform(),
                xytext=(0, -18 if value == SIGNAL_ZERO_LIMIT else 6),
                textcoords="offset points",
                fontsize=SIGNAL_LEGEND_FONTSIZE,
                color="0.35",
            )

        for metric in metrics:
            color, marker = SIGNAL_SHAPE_STYLE[SIGNAL_INFINITE_LIMIT[metric]]
            axis.plot(
                sigma,
                curves[metric],
                marker=marker,
                markersize=7,
                color=color,
                linewidth=2.5,
                zorder=2,
                label=SIGNAL_SWEEP_METRIC_LABELS[metric],
            )
        # The shared tail does the whole styling job here, legend included:
        # it already places a lower-left legend and widens the left x-limit
        # to clear room for it. These panels only differ in carrying no swept
        # parameter, so the legend goes untitled and names the two signals.
        style_signal_row(
            [axis],
            logscale=logscale,
            legend_titles=[""],
            y_label="similarity to base",
        )
        # Headroom above 1 so the unit limit reads as its own line rather
        # than merging into the top spine. Whichever curves actually meet it
        # -- the signals tending to 1, and any curve reaching the sigma -> 0
        # limit when the configured range extends far enough down -- have to
        # stay distinguishable from it.
        axis.set_ylim(0.0, 1.05)
        save_figure(figure, output_dir / f"signal_shapes_{stem}.pdf")


# Panel order for the outlier figure: the same four signals, in the order
# the sweep-row figures are written.
OUTLIER_PANEL_METRICS: tuple[str, ...] = (
    "cka_rbf",
    "rbf_uka",
    "rbf_rwka_symmetric",
    "rbf_degree_crwka",
)
# Not "d": the data dimension is already d throughout the paper.
OUTLIER_LEGEND_TITLE = r"$\Vert \delta \Vert$"

# Where the sigma -> inf limit is drawn, as a fraction of the swept range
# past its right end: far enough to read as a separate position, close
# enough that the dotted continuation still reads as the same curve.
OUTLIER_LIMIT_GAP: float = 0.07


def _outlier_limit_position(sigma: np.ndarray, *, logscale: bool) -> float:
    """The x position standing in for sigma -> inf, just past the grid."""
    low, high = float(sigma.min()), float(sigma.max())
    if logscale:
        return high * (high / low) ** OUTLIER_LIMIT_GAP
    return high + (high - low) * OUTLIER_LIMIT_GAP


def _swept_bandwidth_ticks(
    bounds: tuple[float, float],
    *,
    logscale: bool,
) -> tuple[list[float], list[str]]:
    """Ticks and labels for the swept part of a bandwidth axis.

    Derived from the bounds rather than read back from the axis: before a
    draw, a log axis reports the strided ticks a locator picks for an
    unknown tick space, which here would silently drop every other decade.
    """
    left, right = bounds
    if not logscale:
        locator = ticker.MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10])
        ticks = [
            tick for tick in locator.tick_values(left, right) if left <= tick <= right
        ]
        return ticks, [f"{tick:g}" for tick in ticks]
    exponents = range(int(np.floor(np.log10(left))), int(np.ceil(np.log10(right))) + 1)
    ticks = [
        10.0**exponent for exponent in exponents if left <= 10.0**exponent <= right
    ]
    return ticks, [rf"$10^{{{int(round(np.log10(tick)))}}}$" for tick in ticks]


def _outlier_limit_ticks(
    axis: plt.Axes,
    sigma: np.ndarray,
    position: float,
    *,
    logscale: bool,
) -> None:
    """Mark the limit position on the x axis as infinity.

    The infinity mark is carried by the minor locator, so it cannot collide
    with a swept tick. The majors are pinned to the swept range, because the
    room made for the limit position would otherwise earn a tick of its own
    and label it as a bandwidth. Clearing the intervening minor ticks keeps
    the axis from reading as continuous into a position that stands for a
    limit rather than a value.
    """
    left, _ = axis.get_xlim()
    ticks, labels = _swept_bandwidth_ticks(
        (left, float(sigma.max())), logscale=logscale
    )
    axis.set_xticks(ticks, labels=labels)
    # Half the gap again past the limit position, so its marker stands clear
    # of the right spine instead of being bisected by it.
    high = float(sigma.max())
    axis.set_xlim(
        left,
        (
            position * (position / high) ** 0.5
            if logscale
            else position + (position - high) * 0.5
        ),
    )
    axis.xaxis.set_minor_locator(ticker.FixedLocator([position]))
    axis.xaxis.set_minor_formatter(ticker.FixedFormatter([r"$\infty$"]))
    axis.tick_params(
        axis="x",
        which="minor",
        labelsize=plt.rcParams["xtick.labelsize"],
        length=plt.rcParams["xtick.major.size"],
    )


def plot_outlier_sweep(config: dict[str, Any], output_dir: Path) -> None:
    """Robustness to a single far-away point, one panel per signal metric.

    An additional figure rather than a fourth panel on the sweep rows: the
    swept quantity is not one of the dataset's three parameters but a
    perturbation of the base data against itself, so it reads on its own.
    Styled through the same tail as those rows, so the two sit together.

    The bandwidth axis is the outlier sweep's own grid, which reaches far
    past the shared one: the kernel cannot react to the translated point
    until the bandwidth is comparable to how far it moved.
    """
    data_dir, _ = output_paths(config)
    sweep = load_npz(data_dir / ".scratch" / "outlier_sweep.npz")
    sigma = np.asarray(sweep["sigma"], dtype=float)
    distances = np.asarray(sweep["values"], dtype=float)
    logscale = config["metrics"]["outlier_sweep"]["signals"]["scale"] == "log"

    figure, axes = plt.subplots(
        1,
        len(OUTLIER_PANEL_METRICS),
        figsize=SIGNAL_ROW_FIGSIZE,
        constrained_layout=True,
        sharey=True,
    )
    linear_cka = np.asarray(sweep["fixed_cka_lin"], dtype=float)
    colors = magma_ramp(len(distances))
    limit_position = _outlier_limit_position(sigma, logscale=logscale)

    for axis, metric in zip(axes, OUTLIER_PANEL_METRICS):
        # Each curve is continued, dotted, to its own sigma -> inf limit at a
        # position standing for infinity: linear CKA for the plain alignment
        # signals, 1 for the normalized ones. A continuation rather than a
        # horizontal rule, so a limit cannot be read against the wrong curve.
        limits = (
            linear_cka
            if SIGNAL_INFINITE_LIMIT[metric] == "cka_lin"
            else np.full(len(distances), SIGNAL_ZERO_LIMIT)
        )
        for distance, color, row, limit in zip(
            distances, colors, np.asarray(sweep[metric], float), limits
        ):
            axis.plot(
                sigma, row, marker="o", color=color, zorder=2, label=f"{distance:g}"
            )
            axis.plot(
                [sigma[-1], limit_position],
                [row[-1], limit],
                linestyle=":",
                color=color,
                zorder=2,
            )
            # Hollow, so the limit is not mistaken for a swept measurement.
            axis.plot(
                [limit_position],
                [limit],
                marker="o",
                color=color,
                markerfacecolor="white",
                zorder=2,
            )
        axis.set_title(SIGNAL_SWEEP_METRIC_LABELS[metric])
    style_signal_row(
        axes,
        logscale=logscale,
        legend_titles=[OUTLIER_LEGEND_TITLE] * len(OUTLIER_PANEL_METRICS),
        y_label="similarity to base",
    )
    for axis in axes:
        _outlier_limit_ticks(axis, sigma, limit_position, logscale=logscale)
    save_figure(figure, output_dir / "signal_outlier_sweep.pdf")


def _paper_encoding_guides(
    figure: plt.Figure,
    axes: Any,
    config: dict[str, Any],
    encodings: dict[str, np.ndarray],
) -> None:
    """The paper legend block: side-by-side columns on the right, sharing one
    vertical span -- color and opacity as rank-banded colorbars, size as a
    representative-marker column -- all from the shared `paper_style`
    primitives, so this block and the glocal module's are the same object.

    Deliberately separate from the exploratory `plotting._encoding_guides`,
    which titles its panels to the side and draws a continuous opacity
    gradient; that path is untouched by this one.

    Every panel shares one explicitly computed vertical span rather than each
    asking matplotlib to shrink to a nominally equal fraction --
    `figure.colorbar(shrink=...)` and `shrunk_span` do not produce matching
    pixel heights, which reads as visibly unequal.
    """
    mapping = config["visualization"]["encodings"]
    axes_list = list(np.asarray(axes, dtype=object).ravel())

    # `colorbar=None` means "measure the plot axes only" - this is the shared
    # vertical span every legend element uses, computed once so they end up
    # pixel-equal, not just nominally equal.
    left, bottom, top = side_panel_bounds(figure, axes_list, None)
    panel_bottom, panel_top = shrunk_span(bottom, top)
    panel_height = panel_top - panel_bottom
    panel_width, inter_gap = legend_panel_metrics(figure)

    def add_panel(
        builder: Any,
        values: Any,
        labels: list[str],
        name: str,
        panel_left: float,
        min_left: float,
    ) -> float:
        """Draw one legend column at `panel_left` and return the next
        column's left anchor."""
        rect = (panel_left, panel_bottom, panel_width, panel_height)
        axis = builder(figure, rect, values, labels, PARAMETER_LABELS[name])
        return place_panel(figure, axis, rect, min_left) + inter_gap

    panel_left = left
    if mapping["color"] is not None:
        levels = _channel_levels(config, "color")
        panel_left = add_panel(
            discrete_colorbar_panel,
            _color_bands(levels),
            [f"{value:g}" for value in levels],
            mapping["color"],
            panel_left,
            left,
        )

    if mapping["opacity"] is not None:
        levels = _channel_levels(config, "opacity")
        panel_left = add_panel(
            discrete_colorbar_panel,
            _opacity_bands(levels),
            [f"{value:g}" for value in levels],
            mapping["opacity"],
            panel_left,
            panel_left,
        )

    if mapping["size"] is not None:
        # Size has no colorbar equivalent, so it stays a representative
        # -marker column.
        levels = encodings["size_levels"]
        panel_left = add_panel(
            size_legend_panel,
            rescale_channel(levels, levels, POINT_SIZE_RANGE),
            [f"{value:g}" for value in levels],
            mapping["size"],
            panel_left,
            panel_left,
        )


def plot_mds_grid(
    config: dict[str, Any], output_dir: Path, embedding_type: str
) -> None:
    """1x4 horizontal MDS row (one panel per `MDS_GRID_METRICS` entry) with a
    shared discrete colorbar / size / opacity legend."""
    _, _, _, grid = load_dataset_artifacts(config)
    encodings, rows = metadata(config, grid)
    base_mask = np.array([bool(row["is_base"]) for row in rows])

    encodings = _paper_encodings(config, encodings)
    color_levels = _channel_levels(config, "color")
    if color_levels is None:
        cmap: Any = "inferno"
        norm: Any = None
    else:
        # The colorbar bands by rank, so the points must be expressed in the
        # same band indices -- both are derived from this one `color_levels`
        # array (see `_paper_encodings`, which rank-indexes `color` from it),
        # so a point's band and its colorbar label cannot drift apart.
        cmap = ListedColormap(_color_bands(color_levels))
        norm = BoundaryNorm(np.arange(len(color_levels) + 1), len(color_levels))
    figure, axes = plt.subplots(
        1, 4, figsize=EMBEDDING_GRID_FIGSIZE, constrained_layout=True
    )
    # Reserves a fixed right-hand margin for the legend columns (see
    # `_paper_encoding_guides`) -- with four panels already close to filling
    # the canvas width, relying on leftover slack is not reliable.
    figure.get_layout_engine().set(rect=EMBEDDING_GRID_RECT)
    for axis, (metric_id, title) in zip(axes.flat, MDS_GRID_METRICS):
        result = load_npz(result_path(config, metric_id))
        _plot_embedding(
            axis,
            result[f"{embedding_type}_embedding"],
            encodings,
            base_mask,
            config,
            title,
            result[f"{embedding_type}_normalized_stress"].item(),
            cmap=cmap,
            norm=norm,
        )
        # "MDS 1"/"MDS 2" are identical boilerplate on all four panels and
        # the axis units are arbitrary anyway; dropping the labels *and* the
        # tick numbers/marks keeps each panel to just its point cloud, for a
        # less crowded, more compact paper figure.
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xticks([])
        axis.set_yticks([])
    _paper_encoding_guides(figure, axes, config, encodings)
    save_figure(figure, output_dir / f"mds_grid_{embedding_type}.pdf")


def build_all(config: dict[str, Any], output_dir: Path | None = None) -> None:
    """Build every paper-ready synthetic-experiment figure."""
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT
    )
    apply_paper_style()
    plot_data_description(config, resolved_output_dir)
    plot_signal_sweep_rows(config, resolved_output_dir)
    plot_signal_shapes(config, resolved_output_dir)
    plot_outlier_sweep(config, resolved_output_dir)
    for embedding_type in ("base_relative", "direct"):
        plot_mds_grid(config, resolved_output_dir, embedding_type)
