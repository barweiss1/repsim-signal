"""One definition of how a measure is named and coloured in every figure.

Before this module three schemes were in use at once: `plotting.py` coloured by
measure category but labelled with full class names, `tables/figures.py` used a
separate `MEASURE_GROUPS` map keyed on abbreviations, and the cross-domain
figures used flat fills in some places and categories in others. The
`tables/figures.py` map carried only the historical manifold names, so every
current manifold measure fell through to a grey default while its legend
advertised a "Manifold" swatch nothing used.

Deliberately free of Matplotlib: `tests/test_resi_analysis_schemas.py` asserts
the data modules import with the backend blocked, and label/colour lookups are
data. The legend-patch builder that needs `Patch` lives in `plotting.py`.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .config import (
    DEFAULT_RANK_SORT,
    MEASURE_CATEGORY_COLORS,
    MEASURE_CATEGORY_ORDER,
    measure_category,
    measure_label,
    unfavourable_quantile,
    validate_rank_sort,
)

UNKNOWN_CATEGORY_COLOR = MEASURE_CATEGORY_COLORS["Other"]


def measure_color(measure: str) -> str:
    """The category colour for a measure, by class name or display label.

    ``measure_category`` resolves both spellings, so a display frame carrying
    ``Sim Meas.`` labels and a raw frame carrying class names produce the same
    colour for the same measure. That matters more now that the labels are
    mathematics rather than shortened class names.
    """
    return MEASURE_CATEGORY_COLORS.get(
        measure_category(measure), UNKNOWN_CATEGORY_COLOR
    )


def categories_present(measures: Iterable[str]) -> list[str]:
    """Categories drawn by these measures, in the shared display order."""
    present = {measure_category(measure) for measure in measures}
    known = [category for category in MEASURE_CATEGORY_ORDER if category in present]
    return known + sorted(present - set(known))


def measure_statistic(
    frame,
    column: str,
    *,
    higher_is_better: bool,
    rank_sort: str | None = None,
    metric_column: str = "metric",
):
    """One number per measure: the thing an ordering is built from.

    ``median`` is a measure's typical standing. ``mean`` is its average
    standing, which a handful of extreme cases can pull away from the median.
    ``quantile90`` is its unfavourable tail -- the 90th percentile of a rank,
    the 10th of a quality value -- which says how badly it does when it does
    badly. A measure can win on one and lose on another, which is the point of
    offering all three.
    """
    grouped = frame.groupby(metric_column)[column]
    sort = validate_rank_sort(rank_sort)
    if sort == "median":
        return grouped.median().dropna()
    if sort == "mean":
        return grouped.mean().dropna()
    return grouped.quantile(unfavourable_quantile(higher_is_better)).dropna()


def order_measures(
    frame,
    column: str,
    *,
    higher_is_better: bool,
    rank_sort: str | None = None,
    metric_column: str = "metric",
) -> list[str]:
    """Measures ordered worst-first, so index 0 lands at the bottom of an axis.

    Boxplot positions count upward from the bottom, so "best on top" means the
    best measure has to come last. ``higher_is_better`` says which end of the
    column is the good one: normalized ranks want the smallest statistic on
    top, raw quality values the largest. That direction is the same whichever
    statistic is sorted on, because `measure_statistic` already picks the tail
    that matches it.
    """
    statistic = measure_statistic(
        frame,
        column,
        higher_is_better=higher_is_better,
        rank_sort=rank_sort,
        metric_column=metric_column,
    )
    return statistic.sort_values(
        ascending=higher_is_better, kind="stable"
    ).index.tolist()


def label_sequence(measures: Sequence[str]) -> list[str]:
    """Label a whole axis at once."""
    return [measure_label(measure) for measure in measures]


__all__ = [
    "UNKNOWN_CATEGORY_COLOR",
    "categories_present",
    "label_sequence",
    "measure_color",
    "measure_label",
    "measure_statistic",
    "order_measures",
]
