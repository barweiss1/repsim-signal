"""Rank and value distribution figures for the recreated result tables."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..boxplots import category_legend, finish, measure_panel, panel_size
from ..measure_style import order_measures

DEFAULT_MODALITIES = ("Graph", "Text", "Vision")

MEASURE_COLUMN = "Sim Meas."

# These panels and the tables beside them now group by the same taxonomy:
# `measure_category`, resolved from the `Sim Meas.` abbreviation these frames
# carry. The source tables collapsed every local measure into one "Manifold"
# block, which asserted a family that does not exist -- the manifold measures
# land in three different categories and finish at opposite ends of the field.


def _faceted_figure(
    frame: pd.DataFrame,
    out_path: str | Path,
    *,
    facet_column: str,
    facets: Sequence[str],
    column: str,
    higher_is_better: bool,
    title: Callable[[str], str],
    xlabel: Callable[[str], str],
    shared_order: bool = False,
    box_style: str | None = None,
) -> Path:
    """One measure boxplot per facet, sharing the repository's boxplot style.

    Both figures in this module are this function: one facets the ranks by
    modality, the other facets the values by evaluation measure. They differ in
    which column they draw and which direction is good, and nothing else.

    ``shared_order`` sorts every panel by the same statistic over the whole
    frame, so a measure keeps its row across panels and the difference between
    them reads as a horizontal shift. Without it each panel sorts itself, which
    is what makes a measure's position carry its standing in that panel.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = frame[MEASURE_COLUMN].nunique() if MEASURE_COLUMN in frame else 1
    fig, axes = plt.subplots(
        1,
        len(facets),
        sharey=False,
        figsize=panel_size(rows, panels=len(facets)),
        squeeze=False,
    )
    order = (
        order_measures(
            frame,
            column,
            higher_is_better=higher_is_better,
            metric_column=MEASURE_COLUMN,
        )
        if shared_order and MEASURE_COLUMN in frame
        else None
    )
    drawn: set[str] = set()
    for index, facet in enumerate(facets):
        ax = axes[0][index]
        panel = (
            frame[frame[facet_column].astype(str).eq(facet)]
            if facet_column in frame.columns
            else frame.iloc[0:0]
        )
        ax.set_title(title(facet))
        ax.set_xlabel(xlabel(facet))
        if panel.empty:
            ax.set_yticks([])
            continue
        drawn.update(
            measure_panel(
                ax,
                panel,
                column,
                higher_is_better=higher_is_better,
                metric_column=MEASURE_COLUMN,
                order=order,
                box_style=box_style,
            )
        )
    category_legend(fig, drawn, box_style=box_style)
    return finish(fig, path)


def plot_rank_distributions(
    ranked: pd.DataFrame,
    out_path: str | Path,
    *,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    box_style: str | None = None,
) -> Path:
    """Draw one rank boxplot panel per modality into a single figure.

    The whole figure is written once, after every panel is drawn. The source
    notebook saved inside its modality loop, which wrote the same partially
    drawn figure to each file.
    """
    return _faceted_figure(
        ranked,
        out_path,
        facet_column="Domain",
        facets=list(modalities),
        column="rank",
        higher_is_better=False,
        title=lambda facet: facet,
        xlabel=lambda _facet: "rank",
        box_style=box_style,
    )


def plot_task_value_distributions(
    display: pd.DataFrame,
    out_path: str | Path,
    *,
    test: str | None = None,
    box_style: str | None = None,
) -> Path | None:
    """Draw one boxplot panel per evaluation measure of the raw measure values.

    This is the value-space companion to `plot_rank_distributions`. Ranks say
    which measure won a cell; these say by how much and how consistently, which
    a rank cannot show. Conformity rate has already been inverted by
    `build_display_frame`, so every panel reads "higher is better".

    Each box pools one point per (dataset, architecture) cell within the test.
    Returns None when there is nothing to draw, so callers can skip writing an
    empty figure.
    """
    frame = display if test is None else display[display["Test"].eq(test)]
    frame = frame[frame["value"].notna()]
    if frame.empty:
        return None
    evaluations = sorted(name for name in frame["Eval."].astype(str).unique() if name)
    if not evaluations:
        return None
    return _faceted_figure(
        frame,
        out_path,
        facet_column="Eval.",
        facets=evaluations,
        column="value",
        higher_is_better=True,
        title=lambda facet: facet if test is None else f"{test} - {facet}",
        xlabel=lambda facet: facet,
        box_style=box_style,
    )


__all__ = [
    "DEFAULT_MODALITIES",
    "plot_rank_distributions",
    "plot_task_value_distributions",
]
