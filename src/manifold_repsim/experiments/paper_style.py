"""Shared publication-figure style, geometry, and legend primitives.

Both MDS experiments' `paper_figures.py` modules build the same kind of
figure -- one horizontal row of embedding panels with a legend column of
discrete channel guides on the right, plus one horizontal row of signal
sweeps -- and their figures appear side by side in the same paper. This
module owns everything that should therefore look identical in both:
rcParams, the magma ramp, the legend column's geometry, the discrete
colorbar and size-legend panels, and the signal-row styling tail.

It deliberately holds no experiment knowledge: every function takes plain
colors, labels, levels, and rectangles, so neither experiment package has to
import the other (a neutral module under `experiments/`, the same way
`mds.py` is shared). What stays experiment-local is anything that encodes
*meaning* rather than format -- most importantly glocal's log-scale lambda
sampling and `10^{-3}` label formatting, which is why the ramp is exposed as
both `magma_ramp` (evenly spaced by rank) and `magma_at` (sampled at caller-
supplied positions) rather than one fixed policy.

Discrete channels are banded by *rank*, not by value: every band is the same
height and carries one level's label. For an evenly spaced grid this is
identical to banding at value midpoints, but for a log-spaced grid (glocal's
lambda) value-banding would make the largest level's band swamp the rest.
`rank_index` maps data values onto those same bands, so scatter points and
their colorbar always agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap

# Magma sub-range every paper figure samples colors from. Avoids the ramp's
# near-black and near-white ends, which read poorly as point colors.
MAGMA_RANGE: tuple[float, float] = (0.08, 0.85)

# Legend column geometry. Panel width and gap are in inches (divided by the
# figure width at call time) so columns stay a consistent physical size
# regardless of how wide the figure is.
LEGEND_HEIGHT_FRACTION = 0.85
LEGEND_PANEL_WIDTH_IN = 0.5
LEGEND_PANEL_GAP_IN = 0.25
# Points of separation between a legend panel's title and the panel itself.
# Scales with the (large) paper title fontsize rather than the matplotlib
# default, which is tuned for ~10pt text and lets the title overlap.
LEGEND_TITLE_PAD = 26
LEGEND_TICK_FONTSIZE = 16

# Channel ranges shared by both experiments' paper figures.
OPACITY_RANGE: tuple[float, float] = (0.3, 1.0)
POINT_SIZE_RANGE: tuple[float, float] = (45.0, 150.0)
# Gray level for the opacity colorbar's swatches: the bands vary only in
# alpha, so the underlying color is fixed.
OPACITY_SWATCH_GRAY = 0.15

# Shared figure sizing. The embedding grid reserves a right-hand margin that
# constrained_layout does not know about, so the legend column has somewhere
# to go; without it the panels expand across the full width and the legend
# lands on top of them.
EMBEDDING_GRID_FIGSIZE: tuple[float, float] = (24.0, 4.5)
EMBEDDING_GRID_RECT: tuple[float, float, float, float] = (0.0, 0.0, 0.8, 1.0)
SIGNAL_ROW_FIGSIZE: tuple[float, float] = (24.0, 4.5)

# Signal-sweep panels carry no per-panel title (the legend title names the
# swept parameter instead), so their axis labels and legend read larger than
# the module-wide defaults to keep the eye anchored.
SIGNAL_AXIS_LABEL_FONTSIZE = 26
SIGNAL_LEGEND_FONTSIZE = 20
SIGNAL_LEGEND_TITLE_FONTSIZE = 22
# How far past the smallest x value to extend a signal panel's left limit so
# its lower-left legend has room (a divisor on log axes, a fraction of the
# visible span on linear ones). Sized for the largest legend either
# experiment produces at the fontsizes above -- the synthetic experiment's
# sigma_noise panel is the tight case, since its curves descend at the very
# left edge rather than staying near 1 there.
SIGNAL_LEGEND_LOG_WIDENING = 3.2
SIGNAL_LEGEND_LINEAR_WIDENING = 0.16

PAPER_RCPARAMS: dict[str, Any] = {
    "font.size": 20,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "legend.title_fontsize": 18,
    "lines.linewidth": 2.6,
    "lines.markersize": 8,
    "savefig.bbox": "tight",
}


def apply_paper_style() -> None:
    """Set the shared publication-oriented matplotlib rcParams.

    These figures are composited at a fraction of their native size inside
    the paper (multi-panel layouts, half- or full-column width), so fonts are
    much larger than standalone viewing would need.
    """
    plt.rcParams.update(PAPER_RCPARAMS)


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Write one figure to `path`, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    print(f"Saved {path}")


def legend_title_fontsize() -> float:
    """Legend-panel titles match the subplot titles' current fontsize."""
    return plt.rcParams["axes.titlesize"]


def magma_ramp(count: int) -> np.ndarray:
    """`count` RGBA colors evenly spaced across `MAGMA_RANGE`.

    The default sampling policy: level ordering alone decides the color, so
    consecutive levels are always equally far apart in hue.
    """
    return plt.cm.magma(np.linspace(*MAGMA_RANGE, count))


def magma_at(positions: Sequence[float] | np.ndarray) -> np.ndarray:
    """RGBA colors sampled at explicit `[0, 1]` positions within `MAGMA_RANGE`.

    For callers whose levels are not evenly spaced in the space the reader
    perceives -- glocal's log-spaced lambda grid samples at each level's own
    log position -- so the ramp still reads correctly. `magma_at([0, 1])`
    reproduces `magma_ramp(2)`'s endpoints.
    """
    positions = np.asarray(positions, dtype=float)
    low, high = MAGMA_RANGE
    return plt.cm.magma(low + positions * (high - low))


def scaled_value(
    value: float,
    levels: np.ndarray,
    target: tuple[float, float],
) -> float:
    """Map one level linearly from `levels`' span onto `target`'s span."""
    levels = np.asarray(levels, dtype=float)
    if levels.max() == levels.min():
        return target[0]
    fraction = (value - levels.min()) / (levels.max() - levels.min())
    return target[0] + fraction * np.ptp(target)


def rescale_channel(
    values: np.ndarray,
    levels: np.ndarray,
    target: tuple[float, float],
) -> np.ndarray:
    """Vectorized `scaled_value` over an array of raw parameter values."""
    return np.asarray(
        [scaled_value(float(value), levels, target) for value in np.ravel(values)],
        dtype=float,
    ).reshape(np.shape(values))


def rank_index(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Map raw parameter values onto their `levels` band indices.

    The discrete colorbars band by rank, so anything colored to agree with
    one -- scatter points above all -- must be expressed in the same band
    indices rather than raw values. Matching is by `np.isclose` since the
    values reaching here have usually round-tripped through float artifacts.
    Raises when a value matches no level, rather than silently binning it
    into a neighbouring band and mislabeling the point.
    """
    levels = np.asarray(levels, dtype=float)
    flat = np.asarray(values, dtype=float).ravel()
    indices = np.empty(len(flat), dtype=int)
    for position, value in enumerate(flat):
        matches = np.flatnonzero(np.isclose(levels, value))
        if len(matches) != 1:
            raise ValueError(
                f"Value {value!r} matches {len(matches)} of the levels "
                f"{levels.tolist()}; expected exactly one"
            )
        indices[position] = matches[0]
    return indices.reshape(np.shape(values))


def opacity_swatches(opacities: Sequence[float]) -> list[tuple[float, ...]]:
    """Fixed-gray RGBA swatches varying only in alpha, for opacity colorbars."""
    gray = OPACITY_SWATCH_GRAY
    return [(gray, gray, gray, float(opacity)) for opacity in opacities]


def shrunk_span(
    bottom: float,
    top: float,
    shrink: float = LEGEND_HEIGHT_FRACTION,
) -> tuple[float, float]:
    """Return a `shrink`-fraction sub-span anchored at `bottom`.

    Bottom-anchored rather than centered: callers pass the measured span of
    the plot axes themselves, so the legend columns start exactly level with
    the bottom of the plots they annotate. All the leftover slack goes to the
    top, where the measured span already includes the panel titles.
    """
    return bottom, bottom + (top - bottom) * shrink


def side_panel_bounds(
    figure: plt.Figure,
    axes_list: list[plt.Axes],
    colorbar: Any = None,
) -> tuple[float, float, float]:
    """Return (left, bottom, top) figure-fraction bounds for the legend block.

    Uses each artist's rendered tight bounding box (not its nominal
    ``get_position()`` slot) since 3D axes draw tick/axis labels outside their
    reserved rectangle, and a wrong estimate here means the legend panels
    overlap real plot content. `colorbar` is for callers that already drew an
    attached colorbar the panels must clear; leave it `None` to measure the
    plotted axes alone.
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


def legend_panel_metrics(figure: plt.Figure) -> tuple[float, float]:
    """Return (panel_width, inter_gap) in figure fractions for `figure`."""
    fig_width_in = figure.get_size_inches()[0]
    return LEGEND_PANEL_WIDTH_IN / fig_width_in, LEGEND_PANEL_GAP_IN / fig_width_in


def place_panel(
    figure: plt.Figure,
    axis: plt.Axes,
    rect: tuple[float, float, float, float],
    min_left: float | None = None,
) -> float:
    """Measure a drawn legend panel and return its right edge.

    Each panel's title is centered and usually wider than the panel itself,
    so its rendered extent is not known until it is drawn. When `min_left` is
    given and the panel's rendered box starts left of it, the panel is shifted
    right by the difference and re-measured -- otherwise a wide title silently
    overlaps the previous column. The returned right edge is the next column's
    left anchor.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_figure = figure.transFigure.inverted()
    bbox = axis.get_tightbbox(renderer).transformed(to_figure)
    if min_left is not None:
        shift = max(0.0, min_left - bbox.x0)
        if shift > 0:
            left, bottom, width, height = rect
            axis.set_position((left + shift, bottom, width, height))
            figure.canvas.draw()
            bbox = axis.get_tightbbox(renderer).transformed(to_figure)
    return bbox.x1


def discrete_colorbar_panel(
    figure: plt.Figure,
    rect: tuple[float, float, float, float],
    colors: Sequence[tuple[float, ...]],
    labels: Sequence[str],
    title: str,
) -> plt.Axes:
    """Render one rank-banded colorbar occupying `rect`.

    One solid band per entry in `colors`, ascending bottom-to-top, equal in
    height, labeled at each band's center. Used for every discrete channel
    that has a color-like presentation: hue (a color channel) and alpha (an
    opacity channel, via `opacity_swatches`).

    Bands are equal-height rather than spaced by the channel's own numeric
    values because these channels are presented everywhere else in the figure
    as a small ordered set of labeled levels, not a continuous numeric axis --
    and because a log-spaced grid banded by value is unreadable. See
    `rank_index` for keeping the plotted points on the same bands.
    """
    axis = figure.add_axes(rect)
    count = len(colors)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(
            norm=BoundaryNorm(np.arange(count + 1), count),
            cmap=ListedColormap(list(colors)),
        ),
        cax=axis,
    )
    colorbar.set_ticks(np.arange(count) + 0.5)
    colorbar.set_ticklabels(list(labels))
    colorbar.ax.tick_params(labelsize=LEGEND_TICK_FONTSIZE)
    colorbar.ax.set_title(title, fontsize=legend_title_fontsize(), pad=LEGEND_TITLE_PAD)
    return axis


def size_legend_panel(
    figure: plt.Figure,
    rect: tuple[float, float, float, float],
    sizes: Sequence[float],
    labels: Sequence[str],
    title: str,
) -> plt.Axes:
    """Render one representative-marker column for a point-size channel.

    Size has no colorbar equivalent, so it stays a small column of points --
    one per level, at scatter `s=sizes[i]`, labeled on the right. Markers sit
    at rank positions so the column lines up band-for-band with the discrete
    colorbars beside it.
    """
    axis = figure.add_axes(rect)
    count = len(sizes)
    positions = np.arange(count)
    axis.scatter(
        np.zeros(count),
        positions,
        s=list(sizes),
        color="0.5",
        edgecolors="none",
        clip_on=False,
    )
    axis.set_xlim(-1, 1)
    axis.set_ylim(-0.6, count - 0.4)
    axis.set_xticks([])
    axis.yaxis.tick_right()
    axis.set_yticks(positions)
    axis.set_yticklabels(list(labels))
    axis.tick_params(labelsize=LEGEND_TICK_FONTSIZE)
    for spine in ("top", "left", "bottom"):
        axis.spines[spine].set_visible(False)
    axis.set_title(title, fontsize=legend_title_fontsize(), pad=LEGEND_TITLE_PAD)
    return axis


def highlight_reference_point(
    axis: plt.Axes,
    coordinates: tuple[np.ndarray, ...],
    sizes: np.ndarray,
    *,
    is_3d: bool = False,
) -> None:
    """Ring the run's reference condition so it reads at a glance.

    An unfilled black ring drawn over whatever marker the point already has,
    so the reference stays distinguishable whether it carries the usual
    channel encodings (the synthetic experiment's base transform) or none at
    all (glocal's `none` condition).
    """
    ring_kwargs = {"depthshade": False} if is_3d else {}
    axis.scatter(
        *coordinates,
        s=np.asarray(sizes, dtype=float) + 75,
        marker="o",
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        zorder=6,
        **ring_kwargs,
    )


def widen_for_corner_legend(axis: plt.Axes, *, logscale: bool) -> None:
    """Extend an axis's left x-limit to clear room for a lower-left legend.

    `loc="lower left"` positions in axes fractions, so the only way to keep
    the legend off the data is to move the data. Signal curves start at their
    highest similarity near the smallest sigma and fall away, leaving the
    bottom-left corner clear -- widening past the smallest sigma gives the
    legend that corner instead of crowding the first data points.
    """
    left, right = axis.get_xlim()
    widened = (
        left / SIGNAL_LEGEND_LOG_WIDENING
        if logscale
        else left - (right - left) * SIGNAL_LEGEND_LINEAR_WIDENING
    )
    axis.set_xlim(widened, right)


def style_signal_row(
    axes: Sequence[plt.Axes],
    *,
    logscale: bool,
    legend_titles: Sequence[str],
    y_label: str,
    x_label: str = r"$\sigma$",
) -> None:
    """Apply the shared styling tail to a 1xN signal-sweep row.

    No per-panel title: each panel's legend is titled with its swept
    parameter symbol instead, so panel identity is still conveyed without a
    separate caption line. Only the leftmost panel is y-labeled, since every
    panel shares the same metric and scale.
    """
    for axis, legend_title in zip(axes, legend_titles):
        if logscale:
            axis.set_xscale("log")
        widen_for_corner_legend(axis, logscale=logscale)
        axis.set_xlabel(x_label, fontsize=SIGNAL_AXIS_LABEL_FONTSIZE)
        axis.grid(alpha=0.25)
        axis.legend(
            title=legend_title,
            frameon=False,
            loc="lower left",
            fontsize=SIGNAL_LEGEND_FONTSIZE,
            title_fontsize=SIGNAL_LEGEND_TITLE_FONTSIZE,
        )
    axes[0].set_ylabel(y_label, fontsize=SIGNAL_AXIS_LABEL_FONTSIZE)
