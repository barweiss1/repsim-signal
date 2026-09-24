"""Cross-domain value figures: one per benchmark family.

Each figure holds one panel per quality measure that family reports, with the
measures on a shared vertical axis and one box per domain. Values, not ranks,
because within a single quality measure the three domains are on one scale.

Boxes use percentile whiskers because several of these benchmarks saturate;
see WHISKER_PERCENTILES for why the usual Tukey whiskers misrepresent them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .boxplots import (
    add_legend,
    category_legend,
    finish,
    grouped_panel,
    mean_handle,
    measure_panel,
    panel_size,
)
from .config import validate_rank_sort
from .measure_style import order_measures
from .cross_domain import (
    ALL_TASKS,
    DOMAIN_LABELS,
    QUALITY_LABELS,
    QUALITY_ORDER,
    TEST_GROUP_ORDER,
    UNCALIBRATED_QUALITY,
    _ordered,
    aggregate_across_domains,
    rank_axis_label,
    rank_column,
)


# A normalized rank spans a known [0, 1], so pinning the axis to it keeps every
# panel on one scale and shows where a measure sits within the possible range.
# A raw rank's range is the field size, which differs by domain, so there is no
# one right limit and the data sets it.
NORMALIZED_RANK_XLIM = (-0.05, 1.05)


def _apply_rank_axis(ax, rank_scale: str | None) -> None:
    """Label and bound a rank axis for the scale actually being drawn."""
    ax.set_xlabel(rank_axis_label(rank_scale))
    if rank_column(rank_scale) == "mean_normalized_rank":
        ax.set_xlim(*NORMALIZED_RANK_XLIM)


def _unit_count(frame: pd.DataFrame) -> str:
    """Describe a panel's sample as subtasks, with the cases behind them.

    Both numbers are shown because they answer different questions: the subtask
    count is how many points the box was built from, the case count is how much
    evaluation stands behind them.
    """
    if "subtask_id" not in frame.columns:
        return f"n={frame['case_id'].nunique()} cases"
    subtasks = frame["subtask_id"].nunique()
    # The most complete measure's total is the panel's case count. Summing
    # every measure would multiply by the field size, and a mean over measures
    # would be dragged down by one with partial coverage.
    cases = int(frame.groupby("metric", observed=True)["n_cases"].sum().max())
    return f"n={subtasks} subtasks, {cases} cases"


# Distinct hues rather than a ramp: domain is categorical, and the reader has
# to tell three boxes apart inside one measure's row.
DOMAIN_COLORS = {
    "graphs": "#7bb0e0",
    "vision": "#f0a868",
    "language": "#9fd4a3",
}
DOMAIN_ORDER = ["graphs", "vision", "language"]

FIGURE_INDEX_COLUMNS = ["test_group", "panels", "measures", "domains", "path", "note"]


def plot_test_group(
    values: pd.DataFrame,
    test_group: str,
    out_path: str | Path,
) -> Path | None:
    """Draw one benchmark family's panels, or return None when it has no data."""
    panel_all = values[values["test_group"].astype(str).eq(test_group)]
    if panel_all.empty:
        return None
    qualities = _ordered(
        {str(name) for name in panel_all["quality_measure"].dropna().unique()},
        QUALITY_ORDER,
    )
    if not qualities:
        return None

    # One ordering shared by every panel, so a measure keeps its row and a
    # domain that disagrees with the others shows as a horizontal shift.
    order = order_measures(panel_all, "value", higher_is_better=True)
    fig, axes = plt.subplots(
        1,
        len(qualities),
        figsize=panel_size(len(order), panels=len(qualities)),
        sharey=True,
        squeeze=False,
    )
    drawn: list[str] = []
    for column, quality in enumerate(qualities):
        ax = axes[0][column]
        cell = panel_all[panel_all["quality_measure"].astype(str).eq(quality)]
        title = QUALITY_LABELS.get(quality, quality)
        if quality in UNCALIBRATED_QUALITY:
            # Same scale, different chance level per domain: say so on the
            # panel itself rather than trusting a caption to travel with it.
            title = f"{title}\n(chance level differs by domain)"
        ax.set_title(title)
        ax.set_xlabel("value (higher is better)")
        for domain in grouped_panel(
            ax,
            cell,
            "value",
            order=order,
            group_column="domain",
            group_order=DOMAIN_ORDER,
            colors=DOMAIN_COLORS,
        ):
            if domain not in drawn:
                drawn.append(domain)

    add_legend(
        fig,
        [DOMAIN_LABELS.get(domain, domain) for domain in drawn],
        [DOMAIN_COLORS[domain] for domain in drawn],
        extra=[mean_handle()],
    )
    fig.suptitle(test_group)
    return finish(fig, out_path, rect=(0, 0, 1, 0.97))


def write_cross_domain_figures(
    values: pd.DataFrame, out_dir: str | Path
) -> pd.DataFrame:
    """Write one figure per benchmark family and an index of what was drawn."""
    directory = Path(out_dir)
    if values.empty:
        return pd.DataFrame(columns=FIGURE_INDEX_COLUMNS)
    directory.mkdir(parents=True, exist_ok=True)
    present = {str(name) for name in values["test_group"].dropna().unique()}
    rows = []
    for group in _ordered(present, TEST_GROUP_ORDER):
        panel = values[values["test_group"].astype(str).eq(group)]
        slug = group.lower().replace(" ", "-").replace(".", "")
        path = plot_test_group(values, group, directory / f"{slug}.png")
        domains = sorted(set(panel["domain"]))
        rows.append(
            {
                "test_group": group,
                "panels": panel["quality_measure"].nunique(),
                "measures": panel["metric"].nunique(),
                "domains": ",".join(domains),
                "path": str(path) if path else "",
                "note": (
                    ""
                    if len(domains) == 3
                    else f"only {len(domains)} of 3 domains present"
                ),
            }
        )
    index = pd.DataFrame(rows, columns=FIGURE_INDEX_COLUMNS)
    index.to_csv(directory / "cross_domain_index.csv", index=False)
    return index


__all__ = [
    "DOMAIN_COLORS",
    "plot_cross_domain_ranks",
    "plot_task_ranks_by_type",
    "plot_task_values_by_type",
    "write_by_type_figures",
    "FIGURE_INDEX_COLUMNS",
    "plot_test_group",
    "write_cross_domain_figures",
]


RANK_PANEL_WIDTH = 5.0
RANK_PANEL_COLUMNS = 3


def plot_cross_domain_ranks(
    ranks: pd.DataFrame,
    out_path: str | Path,
    *,
    rank_sort: str | None = None,
    rank_scale: str | None = None,
    box_style: str | None = None,
) -> Path | None:
    """Aggregate ranks over all domains: one panel overall, one per family.

    Every panel shares one measure ordering so a measure sits on the same row
    throughout and a family where it does unusually well or badly is visible as
    a horizontal shift rather than having to be found by name in a reshuffled
    axis.

    Under `median` and `mean` that ordering is the overall balanced mean, which
    weights the three domains equally and is the headline number in
    `cross_domain_ranks.csv`. A mean of per-domain means is well defined, so the
    domain-balanced correction is available there. Under `quantile90` it is the
    90th percentile of the pooled subtask ranks, because a quantile has no
    balanced-mean analogue -- so that ordering is *not* domain-balanced, and the
    domain with the most subtasks carries more of the tail. Stated because the
    orderings answer different questions and only some correct for the
    unbalanced design.
    """
    if ranks.empty:
        return None
    summary = aggregate_across_domains(ranks, rank_scale=rank_scale)
    overall = summary[summary["test_group"].eq(ALL_TASKS)]
    if overall.empty:
        return None
    # Worst first: index 0 sits at the bottom, so the best measure ends on top.
    if validate_rank_sort(rank_sort) in ("median", "mean"):
        order = overall.sort_values("balanced_mean", ascending=False)["metric"].tolist()
    else:
        order = order_measures(
            ranks, rank_column(rank_scale), higher_is_better=False, rank_sort=rank_sort
        )

    groups = [ALL_TASKS] + _ordered(
        {str(name) for name in ranks["test_group"].unique()}, TEST_GROUP_ORDER
    )
    rows = (len(groups) + RANK_PANEL_COLUMNS - 1) // RANK_PANEL_COLUMNS
    width, height = panel_size(len(order), panels=RANK_PANEL_COLUMNS)
    fig, axes = plt.subplots(
        rows,
        RANK_PANEL_COLUMNS,
        figsize=(width, height * rows),
        sharey=True,
        squeeze=False,
    )
    for position, group in enumerate(groups):
        ax = axes[position // RANK_PANEL_COLUMNS][position % RANK_PANEL_COLUMNS]
        panel = ranks if group == ALL_TASKS else ranks[ranks["test_group"].eq(group)]
        ax.set_title(f"{group}  ({_unit_count(panel)})")
        _apply_rank_axis(ax, rank_scale)
        # The shared ordering is passed in rather than recomputed, which is what
        # keeps a measure on the same row across all six panels.
        measure_panel(
            ax,
            panel,
            rank_column(rank_scale),
            higher_is_better=False,
            order=order,
            rank_sort=rank_sort,
            box_style=box_style,
        )

    category_legend(fig, order, rank_sort=rank_sort, box_style=box_style)

    for spare in range(len(groups), rows * RANK_PANEL_COLUMNS):
        axes[spare // RANK_PANEL_COLUMNS][spare % RANK_PANEL_COLUMNS].axis("off")

    fig.suptitle(
        "Measure rank aggregated across domains "
        "(one point per subtask: dataset x task x evaluation, "
        "averaged over architectures)",
    )
    return finish(fig, out_path, rect=(0, 0, 1, 0.97))


def plot_task_ranks_by_type(
    ranks: pd.DataFrame,
    test_group: str,
    out_path: str | Path,
    *,
    rank_sort: str | None = None,
    rank_scale: str | None = None,
    box_style: str | None = None,
) -> Path | None:
    """One family's aggregated rank boxplot, type-coloured and self-sorted."""
    panel = (
        ranks if test_group == ALL_TASKS else ranks[ranks["test_group"].eq(test_group)]
    )
    if panel.empty:
        return None
    fig, ax = plt.subplots(figsize=panel_size(panel["metric"].nunique()))
    drawn = measure_panel(
        ax,
        panel,
        rank_column(rank_scale),
        higher_is_better=False,
        rank_sort=rank_sort,
        box_style=box_style,
    )
    if not drawn:
        plt.close(fig)
        return None
    _apply_rank_axis(ax, rank_scale)
    ax.set_title(f"{test_group} -- rank across domains ({_unit_count(panel)})")
    category_legend(fig, drawn, rank_sort=rank_sort, box_style=box_style)
    return finish(fig, out_path)


def plot_task_values_by_type(
    values: pd.DataFrame, test_group: str, out_path: str | Path
) -> Path | None:
    """One family's actual values, one panel per quality measure.

    Panels do not share a y axis: each sorts by its own median, so the same
    measure can sit at different heights across panels. That is the point --
    it shows a measure that is strong on one quality measure and weak on
    another, which a shared ordering would flatten.
    """
    panel_all = values[values["test_group"].astype(str).eq(test_group)]
    if panel_all.empty:
        return None
    qualities = _ordered(
        {str(name) for name in panel_all["quality_measure"].dropna().unique()},
        QUALITY_ORDER,
    )
    if not qualities:
        return None
    fig, axes = plt.subplots(
        1,
        len(qualities),
        figsize=panel_size(panel_all["metric"].nunique(), panels=len(qualities)),
        sharey=False,
        squeeze=False,
    )
    drawn: set[str] = set()
    for column, quality in enumerate(qualities):
        ax = axes[0][column]
        cell = panel_all[panel_all["quality_measure"].astype(str).eq(quality)]
        drawn.update(measure_panel(ax, cell, "value", higher_is_better=True))
        title = QUALITY_LABELS.get(quality, quality)
        if quality in UNCALIBRATED_QUALITY:
            title = f"{title}\n(chance level differs by domain)"
        ax.set_title(title)
        ax.set_xlabel("value (higher is better)")
    if not drawn:
        plt.close(fig)
        return None
    fig.suptitle(f"{test_group} -- values pooled over domains")
    category_legend(fig, drawn)
    return finish(fig, out_path, rect=(0, 0, 1, 0.96))


def write_by_type_figures(
    values: pd.DataFrame,
    ranks: pd.DataFrame,
    out_dir: str | Path,
    *,
    rank_sort: str | None = None,
    rank_scale: str | None = None,
    box_style: str | None = None,
) -> pd.DataFrame:
    """Write the type-coloured rank and value figures for every family."""
    directory = Path(out_dir)
    rows = []
    groups = _ordered(
        {str(n) for n in ranks["test_group"].unique()} if not ranks.empty else set(),
        TEST_GROUP_ORDER,
    )
    for group in [ALL_TASKS, *groups]:
        slug = group.lower().replace(" ", "-").replace(".", "")
        rank_path = plot_task_ranks_by_type(
            ranks,
            group,
            directory / f"{slug}_ranks_by_type.png",
            rank_sort=rank_sort,
            rank_scale=rank_scale,
            box_style=box_style,
        )
        value_path = (
            None
            if group == ALL_TASKS
            else plot_task_values_by_type(
                values, group, directory / f"{slug}_values_by_type.png"
            )
        )
        rows.append(
            {
                "test_group": group,
                "rank_figure": str(rank_path) if rank_path else "",
                "value_figure": str(value_path) if value_path else "",
            }
        )
    index = pd.DataFrame(rows, columns=["test_group", "rank_figure", "value_figure"])
    if not index.empty:
        directory.mkdir(parents=True, exist_ok=True)
        index.to_csv(directory / "by_type_index.csv", index=False)
    return index
