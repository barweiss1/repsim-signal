"""Per-task summary tables for the cross-domain comparison.

One table per benchmark family: measures down the side, grouped by the paper's
measure taxonomy, and one column per (domain, quality measure). Cells hold the
median value over that family's cases, which is the same statistic the matching
figure's box centre shows -- the table exists so the numbers can be read and
cited exactly rather than estimated off an axis.

Best-in-column is bolded in the LaTeX rendering. "Best" is always the largest
value because `combine_domain_values` has already reported the violation rate
as its complement, so every column reads higher-is-better.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cross_domain import (
    DOMAIN_LABELS,
    QUALITY_LABELS,
    QUALITY_ORDER,
    TEST_GROUP_ORDER,
    UNCALIBRATED_QUALITY,
    _ordered,
)
from .config import MEASURE_CATEGORY_ORDER, measure_category, measure_label
from .tables.latex import latex_safe_labels
from .tables.taxonomy import measure_group_label

DOMAIN_ORDER = ["graphs", "vision", "language"]

TABLE_INDEX_COLUMNS = ["test_group", "rows", "columns", "csv_path", "latex_path"]


def _row_order(metrics: set[str]) -> list[str]:
    """Measures ordered by the shared taxonomy, alphabetically inside each group.

    The same `measure_category` the figures colour by, so a measure's row block
    here and its box colour there cannot disagree.
    """
    ordered: list[str] = []
    seen = {measure_category(metric) for metric in metrics}
    known = list(MEASURE_CATEGORY_ORDER)
    for group in [*known, *sorted(seen - set(known))]:
        ordered.extend(sorted(m for m in metrics if measure_category(m) == group))
    return ordered


def build_task_table(values: pd.DataFrame, test_group: str) -> pd.DataFrame:
    """Median value per measure, one column per (domain, quality measure)."""
    panel = values[values["test_group"].astype(str).eq(test_group)]
    if panel.empty:
        return pd.DataFrame()
    medians = (
        panel.groupby(["metric", "domain", "quality_measure"], observed=True)["value"]
        .median()
        .reset_index()
    )
    table = medians.pivot(
        index="metric", columns=["domain", "quality_measure"], values="value"
    )
    domains = [name for name in DOMAIN_ORDER if name in table.columns.levels[0]]
    qualities = _ordered(set(table.columns.levels[1]), QUALITY_ORDER)
    wanted = [
        (domain, quality)
        for domain in domains
        for quality in qualities
        if (domain, quality) in table.columns
    ]
    table = table[wanted]
    table.columns = pd.MultiIndex.from_tuples(
        [
            (
                DOMAIN_LABELS.get(domain, domain),
                QUALITY_LABELS.get(quality, quality)
                + ("*" if quality in UNCALIBRATED_QUALITY else ""),
            )
            for domain, quality in wanted
        ],
        names=["Domain", "Eval."],
    )
    order = [m for m in _row_order(set(table.index)) if m in table.index]
    table = table.loc[order]
    # Ordered and grouped by the raw name, printed by the display label: these
    # rows are the only ones in the repository that used to print class names,
    # so the same measure was `CKArbfAUC` here and mathematics in the boxplot
    # beside it.
    table.index = pd.MultiIndex.from_tuples(
        [
            (measure_group_label(measure_category(metric)), measure_label(metric))
            for metric in table.index
        ],
        names=["Group", "Sim Meas."],
    )
    return table


def render_latex(table: pd.DataFrame, test_group: str) -> str:
    """Two-decimal LaTeX with the best value in each column bolded."""
    if table.empty:
        raise ValueError("Cannot render an empty table.")
    # Sanitised before `best` is taken, so the bolding lookup and the loop
    # below agree on the column labels.
    table = latex_safe_labels(table)
    best = table.max(axis=0)
    formatted = table.copy().astype(object)
    for column in table.columns:
        for row in table.index:
            value = table.at[row, column]
            if pd.isna(value):
                formatted.at[row, column] = "--"
                continue
            text = f"{value:.2f}"
            # Compared on the rounded text so two cells printing the same
            # number are both bolded, rather than one winning on a digit the
            # reader cannot see.
            if f"{best[column]:.2f}" == text:
                text = rf"\textbf{{{text}}}"
            formatted.at[row, column] = text
    body = formatted.to_latex(
        escape=False,
        multicolumn=True,
        multicolumn_format="c",
        multirow=True,
        caption=(
            f"Cross-domain summary for {test_group}. "
            "Cells are median values; higher is better in every column. "
            "Columns marked * have a chance level that differs by domain."
        ),
        label=f"tab:cross-{test_group.lower().replace(' ', '-').replace('.', '')}",
    )
    return body


def write_cross_domain_tables(
    values: pd.DataFrame, out_dir: str | Path
) -> pd.DataFrame:
    """Write one summary table per benchmark family, as CSV and LaTeX."""
    directory = Path(out_dir)
    if values.empty:
        return pd.DataFrame(columns=TABLE_INDEX_COLUMNS)
    directory.mkdir(parents=True, exist_ok=True)
    present = {str(name) for name in values["test_group"].dropna().unique()}
    rows = []
    for group in _ordered(present, TEST_GROUP_ORDER):
        table = build_task_table(values, group)
        if table.empty:
            continue
        slug = group.lower().replace(" ", "-").replace(".", "")
        csv_path = directory / f"{slug}_summary.csv"
        table.to_csv(csv_path)
        latex_path = directory / f"{slug}_summary.tex"
        latex_path.write_text(render_latex(table, group), encoding="utf-8")
        rows.append(
            {
                "test_group": group,
                "rows": int(table.shape[0]),
                "columns": int(table.shape[1]),
                "csv_path": str(csv_path),
                "latex_path": str(latex_path),
            }
        )
    index = pd.DataFrame(rows, columns=TABLE_INDEX_COLUMNS)
    index.to_csv(directory / "cross_domain_tables_index.csv", index=False)
    return index


__all__ = [
    "TABLE_INDEX_COLUMNS",
    "build_task_table",
    "render_latex",
    "write_cross_domain_tables",
]
