"""One implementation of every measure boxplot in this repository.

Before this module there were six places that called `ax.boxplot` and styled the
result, two copies of the Matplotlib orientation-argument shim, four legend
builders, and four median-styling loops -- one of which was missing, so those
medians rendered in Matplotlib's default orange while every other figure drew
them grey. Divergence like that is invisible until two figures sit side by side
in a paper.

Everything here is presentation. Ordering and colour lookups live in
`measure_style`, which is deliberately importable without a backend; this module
is a figure module and imports Matplotlib.
"""

from __future__ import annotations

import inspect
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle

from .config import (
    BOXEN_BANDS,
    BOXEN_MEDIAN,
    BOXEN_WHISKERS,
    ROBUSTNESS_QUANTILE,
    boxen_band_label,
    unfavourable_quantile,
    validate_box_style,
    validate_rank_sort,
)
from .measure_style import (
    categories_present,
    measure_color,
    measure_label,
    order_measures,
)

# Figure geometry. One width and one per-row height everywhere, so a measure's
# row is the same physical size in every figure and the figures stack in a
# paper without one looking stretched next to another.
PANEL_WIDTH = 6.2
ROW_HEIGHT = 0.51
MIN_HEIGHT = 5.0
FIGURE_DPI = 180

# Percentile whiskers, not Tukey. Several benchmarks saturate -- vision's
# shortcut AUPRC is exactly 1.0 in more than half the cases for most measures --
# which drives the IQR to zero. Tukey whiskers reach 1.5 x IQR, so they collapse
# onto the box and the whole distribution renders as a hairline that reads as
# missing data. Percentiles keep showing the spread that is there.
WHISKER_PERCENTILES = (5, 95)

# One type scale for every figure in this package. Sizes had been chosen per
# call site -- 9, 10, 11, 12 and 13 all appear in one module -- and left at
# Matplotlib's 10pt default everywhere else, so the same axis label rendered at
# a different size depending on which figure it landed in.
TICK_FONTSIZE = 11
LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 13
SUPTITLE_FONTSIZE = 15
LEGEND_FONTSIZE = 11

# The robustness marker. A diamond on the box's centre line rather than a
# second vertical rule: the 90th percentile of a rank usually falls outside the
# box, between its edge and the whisker, where a rule would read as a second
# median or as a stray cap.
QUANTILE_MARKER = "D"
QUANTILE_MARKER_SIZE = 3.6

# The mean marker. An open circle rather than a filled one: the median is
# already a solid black line and a solid marker in the same colour would read
# as a second median instead of a distinct statistic. White keeps it legible
# over every category colour and every boxen band tint it can land on.
MEAN_MARKER = "o"
MEAN_MARKER_SIZE = 4.2
MEAN_MARKER_EDGE_WIDTH = 0.9
MEAN_FACE_COLOR = "#ffffff"

# Boxen band geometry. The innermost band is drawn at full box width and each
# band further out is narrower and paler, so the taper itself reads as "this is
# the tail" without needing the percentile numbers off the legend. Widths run
# outermost-first to match `BOXEN_BANDS`.
BOXEN_BAND_WIDTHS = (0.60, 1.0)

# How far each band's fill is mixed toward white, outermost first. Opaque
# tints rather than alpha, following seaborn's `boxenplot`: translucent bands
# composite with whatever is behind them, so a band's colour depended on the
# gridline or band it happened to overlap and the same percentile rendered
# differently in different rows. A tint is one colour, and it survives being
# printed or screenshotted onto a non-white ground.
BOXEN_BAND_TINTS = (0.45, 0.0)

# Bands sit inside one another, so a full-weight outline on each one reads as
# separate stacked boxes rather than one shape. Thinner than a plain box's edge
# for that reason.
BOXEN_EDGE_WIDTH = 0.6

# Cap length at the whisker ends, as a fraction of the full box width. Matched
# by eye to the caps `ax.boxplot` draws, so the two styles do not read as
# coming from different figures when they sit on facing pages.
BOXEN_CAP_WIDTH = 0.32

MEDIAN_COLOR = "#000000"
EDGE_COLOR = "#4c4c4c"
WHISKER_COLOR = "#666666"
BOX_WIDTH = 0.62
UNKNOWN_COLOR = "#cccccc"


def panel_size(n_rows: int, *, panels: int = 1) -> tuple[float, float]:
    """Figure size for `n_rows` measures across `panels` side-by-side panels.

    Height grows with the number of measures and width does not, so every
    figure keeps the same portrait proportions rather than one stretching wide
    because it happened to be drawn at a fixed size.
    """
    return (PANEL_WIDTH * max(panels, 1), max(MIN_HEIGHT, ROW_HEIGHT * max(n_rows, 1)))


def _orientation(ax) -> dict[str, object]:
    """Matplotlib renamed the horizontal-boxplot argument; support both names."""
    if "orientation" in inspect.signature(ax.boxplot).parameters:
        return {"orientation": "horizontal"}
    return {"vert": False}


def draw_boxes(
    ax,
    data: Sequence[np.ndarray],
    *,
    positions: Sequence[float],
    colors: Sequence[str],
    widths: float = BOX_WIDTH,
) -> dict:
    """Draw and style one set of horizontal boxes. The only `ax.boxplot` call.

    Medians are black everywhere: they are the number a reader takes off the
    figure, and a grey or default-coloured median competes with the box fill it
    sits on.
    """
    artists = ax.boxplot(
        list(data),
        positions=list(positions),
        widths=widths,
        showfliers=False,
        whis=WHISKER_PERCENTILES,
        patch_artist=True,
        manage_ticks=False,
        **_orientation(ax),
    )
    for box, color in zip(artists["boxes"], colors):
        box.set_facecolor(color)
        box.set_edgecolor(EDGE_COLOR)
        box.set_linewidth(0.8)
    for median in artists["medians"]:
        median.set_color(MEDIAN_COLOR)
        median.set_linewidth(1.3)
    for line in [*artists["whiskers"], *artists["caps"]]:
        line.set_color(WHISKER_COLOR)
        line.set_linewidth(0.9)
    return artists


def tint(color: str, amount: float) -> tuple[float, float, float]:
    """Mix a colour toward white: 0 leaves it alone, 1 returns white.

    Kept as a plain lightening rather than a saturation or luminance move in a
    perceptual space, because the bands only need to be ordered by lightness
    and a simple mix keeps every category's ramp behaving the same way.
    """
    return tuple(
        channel + (1.0 - channel) * amount for channel in to_rgb(color)
    )  # type: ignore[return-value]


def _draw_whiskers(ax, points, *, position: float, widths: float) -> None:
    """Whiskers from the outermost band's edges out to the extremes, with caps."""
    inner_low, inner_high = np.quantile(points, BOXEN_BANDS[0])
    low, high = np.quantile(points, BOXEN_WHISKERS)
    cap = widths * BOXEN_CAP_WIDTH
    for start, end in ((inner_low, low), (inner_high, high)):
        ax.plot(
            [float(start), float(end)],
            [position, position],
            color=WHISKER_COLOR,
            linewidth=0.9,
            zorder=1,
        )
    for value in (low, high):
        ax.plot(
            [float(value), float(value)],
            [position - cap / 2, position + cap / 2],
            color=WHISKER_COLOR,
            linewidth=0.9,
            zorder=1,
        )


def draw_boxen(
    ax,
    data: Sequence[np.ndarray],
    *,
    positions: Sequence[float],
    colors: Sequence[str],
    widths: float = BOX_WIDTH,
) -> None:
    """Draw one letter-value plot per series: nested percentile bands.

    A box says where the middle half sits and where two whisker percentiles
    fall, and nothing about the shape in between. The bands say where the
    distribution thickens: two measures with the same median and the same 90th
    percentile can still differ in whether the mass sits near the median or
    spreads evenly to the tail, and only this drawing shows which.

    Bands are drawn outermost first so the narrower inner ones land on top, and
    the median is drawn last so it is never covered. Whiskers carry on from the
    outermost band to the observed minimum and maximum, so a boxen hides
    nothing: a plain box's whisker percentiles cut the extremes off, and the
    reader cannot tell from the figure whether anything was cut. They are
    whiskers rather than a third band because two single observations should
    not carry the same visual weight as the 10-90 interval.
    """
    for points, position, color in zip(data, positions, colors):
        if not len(points):
            continue
        _draw_whiskers(ax, points, position=position, widths=widths)
        for (low, high), scale, amount in zip(
            BOXEN_BANDS, BOXEN_BAND_WIDTHS, BOXEN_BAND_TINTS
        ):
            left, right = np.quantile(points, [low, high])
            height = widths * scale
            ax.add_patch(
                Rectangle(
                    (float(left), position - height / 2),
                    float(right - left),
                    height,
                    facecolor=tint(color, amount),
                    edgecolor=EDGE_COLOR,
                    linewidth=BOXEN_EDGE_WIDTH,
                    zorder=2,
                )
            )
        median = float(np.quantile(points, BOXEN_MEDIAN))
        ax.plot(
            [median, median],
            [position - widths / 2, position + widths / 2],
            color=MEDIAN_COLOR,
            linewidth=1.3,
            solid_capstyle="butt",
            zorder=4,
        )


def draw_series(
    ax,
    data: Sequence[np.ndarray],
    *,
    positions: Sequence[float],
    colors: Sequence[str],
    widths: float = BOX_WIDTH,
    box_style: str | None = None,
) -> None:
    """Draw one set of distributions in whichever style the run selected.

    Every panel goes through here rather than choosing a drawing function
    itself, so a new style reaches all of them at once and none can be left
    behind on the old one.
    """
    if validate_box_style(box_style) == "boxen":
        draw_boxen(ax, data, positions=positions, colors=colors, widths=widths)
        return
    draw_boxes(ax, data, positions=positions, colors=colors, widths=widths)


def boxen_handles() -> list[Patch | Line2D]:
    """Legend entries naming the bands, outermost last so it reads inward-out.

    Drawn in neutral grey, tinted by the same ramp the bands use: they take the
    measure's category colour on the figure, so a coloured swatch here would
    claim to name a category.
    """
    handles = []
    for (low, high), amount in zip(reversed(BOXEN_BANDS), reversed(BOXEN_BAND_TINTS)):
        handles.append(
            Patch(
                facecolor=tint(WHISKER_COLOR, amount),
                edgecolor=EDGE_COLOR,
                linewidth=BOXEN_EDGE_WIDTH,
                label=boxen_band_label(low, high),
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            color=WHISKER_COLOR,
            linewidth=0.9,
            marker="|",
            markersize=6,
            markeredgewidth=0.9,
            label="min-max",
        )
    )
    return handles


def _shows_quantile_marker(rank_sort: str | None, box_style: str | None) -> bool:
    """Whether the robustness quantile needs its own marker.

    Only under `quantile90`, and only when the drawing does not already show
    it. A boxen's 10-90 band lands on the same number, so marking it again
    would put two symbols on one statistic. `mean` needs no marker of its own
    here: the mean is always drawn regardless of sort (see `mark_mean`), so
    the sorted-on statistic is already visible without this one.
    """
    return (
        validate_rank_sort(rank_sort) == "quantile90"
        and validate_box_style(box_style) != "boxen"
    )


def mark_quantile(
    ax,
    data: Sequence[np.ndarray],
    *,
    positions: Sequence[float],
    higher_is_better: bool,
) -> None:
    """Draw each box's robustness quantile as a diamond on its centre line.

    The quantile is the unfavourable tail -- the 90th percentile of a rank, the
    10th of a quality value -- so the marker always sits on the bad side of the
    distribution and the leftmost marker is the most robust measure.
    """
    quantile = unfavourable_quantile(higher_is_better)
    for points, position in zip(data, positions):
        if not len(points):
            continue
        ax.plot(
            [float(np.quantile(points, quantile))],
            [position],
            marker=QUANTILE_MARKER,
            markersize=QUANTILE_MARKER_SIZE,
            markerfacecolor=MEDIAN_COLOR,
            markeredgecolor=MEDIAN_COLOR,
            linestyle="none",
            zorder=5,
        )


def quantile_handle(higher_is_better: bool) -> Line2D:
    """Legend entry naming the quantile the markers show."""
    percent = int(round(unfavourable_quantile(higher_is_better) * 100))
    return Line2D(
        [],
        [],
        marker=QUANTILE_MARKER,
        markersize=QUANTILE_MARKER_SIZE,
        markerfacecolor=MEDIAN_COLOR,
        markeredgecolor=MEDIAN_COLOR,
        linestyle="none",
        label=f"{percent}th pct.",
    )


def mark_mean(
    ax,
    data: Sequence[np.ndarray],
    *,
    positions: Sequence[float],
) -> None:
    """Draw each box's mean as a white circle with a black outline.

    Drawn unconditionally, under both drawing styles: a box's median line and
    a boxen's nested percentile bands are both order statistics, and neither
    shows the mean directly -- a skewed distribution's mean can sit visibly
    away from either. The open marker keeps it distinct from the solid median
    line and from the robustness diamond.
    """
    for points, position in zip(data, positions):
        if not len(points):
            continue
        ax.plot(
            [float(np.mean(points))],
            [position],
            marker=MEAN_MARKER,
            markersize=MEAN_MARKER_SIZE,
            markerfacecolor=MEAN_FACE_COLOR,
            markeredgecolor=MEDIAN_COLOR,
            markeredgewidth=MEAN_MARKER_EDGE_WIDTH,
            linestyle="none",
            zorder=6,
        )


def mean_handle() -> Line2D:
    """Legend entry naming the mean marker."""
    return Line2D(
        [],
        [],
        marker=MEAN_MARKER,
        markersize=MEAN_MARKER_SIZE,
        markerfacecolor=MEAN_FACE_COLOR,
        markeredgecolor=MEDIAN_COLOR,
        markeredgewidth=MEAN_MARKER_EDGE_WIDTH,
        linestyle="none",
        label="mean",
    )


def _series(frame: pd.DataFrame, metric_column: str, metric: str, column: str):
    return frame.loc[frame[metric_column].eq(metric), column].dropna().to_numpy()


def measure_panel(
    ax,
    frame: pd.DataFrame,
    column: str,
    *,
    higher_is_better: bool,
    metric_column: str = "metric",
    order: Sequence[str] | None = None,
    rank_sort: str | None = None,
    box_style: str | None = None,
) -> list[str]:
    """One box per measure, category-coloured, best on top. Returns what it drew.

    Position 0 is the bottom of the axis, so the best measure has to land at the
    end of the order to be drawn on top. Which end that is depends on the
    column: a normalized rank is best at 0, a quality value is best at its
    maximum. `order_by_median` sorts ascending exactly when higher is better,
    which puts the best last.

    Pass `order` to share one ordering across panels; omit it and each panel
    sorts by its own statistic, which is what makes a measure's position carry
    its standing in that panel.

    `rank_sort` picks that statistic. Under `quantile90` the rows are sorted by
    the unfavourable tail instead of the centre, and the tail is drawn on each
    box, so the ordering the reader is asked to trust is visible rather than
    asserted.

    `box_style` picks the drawing only, never the ordering or the data. Under
    `boxen` the same distribution is drawn as nested percentile bands, and the
    quantile marker is dropped: the 10-90 band already has an edge exactly
    there, so the diamond would restate it and read as a separate statistic.
    The invariant that the sorted-on number is visible on the figure still
    holds -- it is a band edge instead of a marker.
    """
    if order is None:
        order = order_measures(
            frame,
            column,
            higher_is_better=higher_is_better,
            rank_sort=rank_sort,
            metric_column=metric_column,
        )
    drawn, data = [], []
    for metric in order:
        points = _series(frame, metric_column, metric, column)
        if len(points):
            drawn.append(metric)
            data.append(points)
    if not drawn:
        return []
    positions = [order.index(metric) for metric in drawn]
    draw_series(
        ax,
        data,
        positions=positions,
        colors=[measure_color(metric) for metric in drawn],
        box_style=box_style,
    )
    if _shows_quantile_marker(rank_sort, box_style):
        mark_quantile(ax, data, positions=positions, higher_is_better=higher_is_better)
    mark_mean(ax, data, positions=positions)
    _label_axis(ax, order)
    return drawn


def grouped_panel(
    ax,
    frame: pd.DataFrame,
    column: str,
    *,
    order: Sequence[str],
    group_column: str,
    group_order: Sequence[str],
    colors: Mapping[str, str],
    metric_column: str = "metric",
    span: float = 0.8,
    box_style: str | None = None,
) -> list[str]:
    """One box per (measure, group), grouped by colour within each measure row.

    Used where the group -- currently the domain -- is the information the panel
    carries, so it takes the colour channel and the measure category does not.
    """
    groups = [name for name in group_order if name in set(frame[group_column])]
    if not groups:
        return []
    step = span / len(groups)
    drawn: list[str] = []
    for offset, group in enumerate(groups):
        cell = frame[frame[group_column].eq(group)]
        data, positions = [], []
        for index, metric in enumerate(order):
            points = _series(cell, metric_column, metric, column)
            if not len(points):
                continue
            data.append(points)
            positions.append(index + (offset - (len(groups) - 1) / 2) * step)
        if not data:
            continue
        draw_series(
            ax,
            data,
            positions=positions,
            colors=[colors[group]] * len(data),
            widths=step * 0.85,
            box_style=box_style,
        )
        mark_mean(ax, data, positions=positions)
        drawn.append(group)
    _label_axis(ax, order, pad=0.6)
    return drawn


def _label_axis(ax, order: Sequence[str], *, pad: float = 0.7) -> None:
    """Abbreviated measure names on the y axis, with the grid behind the boxes."""
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([measure_label(metric) for metric in order])
    ax.set_ylim(-pad, len(order) - (1 - pad))
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)


def add_legend(
    fig,
    labels: Sequence[str],
    colors: Sequence[str],
    *,
    extra: Sequence[Line2D | Patch] = (),
) -> None:
    """The one legend style: a swatch column to the right, no frame, no title.

    Anchored outside the axes at x > 1 rather than inside them, so it does not
    take width away from the plotting area the way an `ax.legend` does. The
    saved figure widens to hold it instead, because `finish` saves with
    `bbox_inches="tight"`.
    """
    if not labels and not extra:
        return
    handles = [
        Patch(facecolor=color, edgecolor=EDGE_COLOR, linewidth=0.8, label=label)
        for label, color in zip(labels, colors)
    ]
    handles.extend(extra)
    fig.legend(
        handles=handles,
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
        bbox_to_anchor=(1.0, 0.5),
    )


def category_legend(
    fig,
    measures: Iterable[str],
    *,
    rank_sort: str | None = None,
    higher_is_better: bool = False,
    box_style: str | None = None,
) -> None:
    """Legend for the measure categories actually drawn, in the shared order.

    Gains a marker entry when the panels sort on the robustness quantile and
    draw a diamond for it, so the marker is named rather than left to be
    guessed at, and band entries under `boxen`, where the nested shapes are not
    self-explaining the way a box and two whiskers are. The mean marker is
    always drawn, so its entry is always listed, first among the extras so it
    does not disturb the ordering of the other two, which readers may be
    scanning for by position.
    """
    categories = categories_present(measures)
    extra: list = [mean_handle()]
    if validate_box_style(box_style) == "boxen":
        extra.extend(boxen_handles())
    if _shows_quantile_marker(rank_sort, box_style):
        extra.append(quantile_handle(higher_is_better))
    add_legend(
        fig,
        categories,
        [measure_color_for_category(c) for c in categories],
        extra=extra,
    )


def measure_color_for_category(category: str) -> str:
    """Colour of a category, for legends that are built from categories."""
    from .config import MEASURE_CATEGORY_COLORS

    return MEASURE_CATEGORY_COLORS.get(category, UNKNOWN_COLOR)


def scale_text(fig) -> None:
    """Apply the shared type scale to a figure's titles, labels, and ticks.

    Applied to the built artists rather than through rcParams, which are global
    and shared with the synthetic and glocal experiments' own paper styling.

    Legends are deliberately untouched: `add_legend` sets the one boxplot legend
    size, and the signal plots carry a smaller explicit size chosen for how many
    entries they list.
    """
    for text in fig.texts:
        text.set_fontsize(SUPTITLE_FONTSIZE)
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.title.set_fontsize(TITLE_FONTSIZE)
        ax.xaxis.label.set_fontsize(LABEL_FONTSIZE)
        ax.yaxis.label.set_fontsize(LABEL_FONTSIZE)


def finish(fig, out_path, *, rect: tuple[float, float, float, float] = (0, 0, 1, 1)):
    """Lay out, save, and close. Every boxplot figure ends the same way.

    `rect` reserves room for a suptitle where one is drawn. Nothing is reserved
    for the legend: it sits outside the axes and `bbox_inches="tight"` grows the
    saved image around it.
    """
    from pathlib import Path

    scale_text(fig)
    fig.tight_layout(rect=rect)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = [
    "BOXEN_BAND_TINTS",
    "BOXEN_CAP_WIDTH",
    "BOXEN_BAND_WIDTHS",
    "BOXEN_EDGE_WIDTH",
    "BOX_WIDTH",
    "EDGE_COLOR",
    "FIGURE_DPI",
    "LABEL_FONTSIZE",
    "LEGEND_FONTSIZE",
    "MEAN_MARKER",
    "MEAN_MARKER_SIZE",
    "MEAN_MARKER_EDGE_WIDTH",
    "MEAN_FACE_COLOR",
    "MEDIAN_COLOR",
    "MIN_HEIGHT",
    "PANEL_WIDTH",
    "QUANTILE_MARKER",
    "ROW_HEIGHT",
    "SUPTITLE_FONTSIZE",
    "TICK_FONTSIZE",
    "TITLE_FONTSIZE",
    "WHISKER_PERCENTILES",
    "add_legend",
    "category_legend",
    "boxen_handles",
    "draw_boxen",
    "draw_boxes",
    "draw_series",
    "finish",
    "grouped_panel",
    "mark_mean",
    "mark_quantile",
    "mean_handle",
    "measure_panel",
    "panel_size",
    "quantile_handle",
    "scale_text",
    "tint",
]
