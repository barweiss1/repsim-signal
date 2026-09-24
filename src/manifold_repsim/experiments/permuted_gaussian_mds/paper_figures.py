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
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap

from ..artifacts import load_npz
from ..paper_style import EMBEDDING_GRID_FIGSIZE
from ..paper_style import EMBEDDING_GRID_RECT
from ..paper_style import OPACITY_RANGE
from ..paper_style import POINT_SIZE_RANGE
from ..paper_style import SIGNAL_ROW_FIGSIZE
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
    targets: tuple[tuple[str, tuple[float, float, float] | None], ...] = (
        ("Base data", None),
        (parameter_value_label(PARAMETERS[0], moderate[0]), (moderate[0], 0.0, 0.0)),
        (parameter_value_label(PARAMETERS[1], moderate[1]), (0.0, moderate[1], 0.0)),
        (parameter_value_label(PARAMETERS[2], moderate[2]), (0.0, 0.0, moderate[2])),
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


# Left-to-right panel order for the signal-sweep row figure (paper
# convention: sigma, mix, perm) -- independent of `PARAMETERS`' canonical
# dataset/grid order, which other figures and the dataset column layout
# still rely on unchanged.
SIGNAL_SWEEP_PARAMETER_ORDER: tuple[str, ...] = (
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
        for axis, parameter in zip(axes, SIGNAL_SWEEP_PARAMETER_ORDER):
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
                PARAMETER_LABELS[parameter]
                for parameter in SIGNAL_SWEEP_PARAMETER_ORDER
            ],
            y_label=SIGNAL_SWEEP_METRIC_LABELS[metric_name],
        )
        save_figure(figure, output_dir / f"signal_sweep_{metric_name}.pdf")


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
    for embedding_type in ("base_relative", "direct"):
        plot_mds_grid(config, resolved_output_dir, embedding_type)
