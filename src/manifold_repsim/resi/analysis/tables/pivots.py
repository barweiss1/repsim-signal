"""Build the emphasized string table that the LaTeX writer renders.

This is the block the source notebooks repeat six times: pivot to a measure-by-
condition table, format every value to two decimals, mark the best value in each
column, and splice in correlation values carrying significance markers.

How deep the marking goes is the caller's choice. The appendix tables mark the
best value only, as the source notebooks do. The two scannable paper tables ask
for three, because their question is "which measures are near the top here",
and a lone bold cell answers "which one is top" instead -- with sixteen rows a
reader cannot see who came close without re-reading every number.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .significance import (
    floatify,
    pval_str,
    separate_significance_indicator,
)
from .taxonomy import (
    GROUNDING_BY_PREDICTION,
    MEASURE_GROUP_LABEL_ORDER,
    TEST_ORDER,
    TYPE_ORDER,
)

ROW_INDEX = ["Measure Type", "Sim Meas."]
DEFAULT_COLUMN_LEVELS = ["Type", "Test", "Eval.", "Domain", "Dataset", "Arch."]
VALUE_PRECISION = 2

# How the best, second-best and third-best value in a column are marked, in
# that order. Bold is the source tables' own marker and stays first, so a
# reader who knows those tables reads the best value the same way here and
# only has to learn what the two additions mean.
EMPHASIS_COMMANDS = (r"\textbf", r"\underline", r"\textit")

# Mark the best value only -- what the source notebooks do, and what every
# table here did before the deeper marking existed.
DEFAULT_EMPHASIS_DEPTH = 1


def _format_value(value: float) -> str:
    return f"{round(value, ndigits=VALUE_PRECISION):.{VALUE_PRECISION}f}"


def _emphasize_column_top(
    column: pd.Series, depth: int = DEFAULT_EMPHASIS_DEPTH
) -> pd.Series:
    """Mark the best `depth` distinct values in one column, ties included.

    Ranking is on the printed two-decimal values rather than the underlying
    floats, so cells printing the same number always carry the same marker. The
    alternative reads as a typo: 0.54 in bold beside a plain 0.54, differing
    only in a third decimal the table does not show.

    Places are counted over distinct printed values, so a tie shares a place
    rather than consuming the ones below it: two measures printing 0.98 are
    both bold, and the next value down is still the underlined one. A column
    thus shows all three levels whenever it holds three distinct values, which
    is what makes the marking mean the same thing in every column.
    """
    numeric = pd.to_numeric(column.map(floatify), errors="coerce")
    if numeric.notna().sum() == 0:
        return column
    values = column.map(floatify)
    markers = column.map(separate_significance_indicator)
    markers = markers.where(markers.notna(), "")
    levels = sorted(numeric.dropna().unique(), reverse=True)
    emphasized = column
    for command, level in zip(EMPHASIS_COMMANDS[:depth], levels):
        at_level = numeric.eq(level)
        emphasized = emphasized.where(~at_level, command + "{" + values + "}" + markers)
    return emphasized


def build_value_table(
    display: pd.DataFrame,
    *,
    column_levels: Sequence[str] = tuple(DEFAULT_COLUMN_LEVELS),
    emphasize_best: bool = True,
    emphasis_depth: int = DEFAULT_EMPHASIS_DEPTH,
) -> pd.DataFrame:
    """Pivot display rows into a measure-by-condition table of formatted strings.

    Rows are ordered by the shared measure taxonomy then measure name; columns
    follow the established type, test, and dataset orderings. Values become fixed
    two-decimal strings so LaTeX columns align, and the best `emphasis_depth`
    values in each column are marked when requested.
    """
    if display.empty:
        return pd.DataFrame()
    levels = list(column_levels)
    numeric = pd.pivot_table(
        display,
        index=ROW_INDEX,
        columns=levels,
        values="value",
        observed=False,
    )
    numeric = numeric.reindex(MEASURE_GROUP_LABEL_ORDER, axis="index", level=0)
    if "Test" in levels:
        numeric = numeric.reindex(TEST_ORDER, axis="columns", level="Test")
    if "Type" in levels:
        numeric = numeric.reindex(TYPE_ORDER, axis="columns", level="Type")
    numeric = numeric.dropna(axis="index", how="all").dropna(axis="columns", how="all")
    if numeric.empty:
        return numeric

    table = numeric.map(lambda value: "" if pd.isna(value) else _format_value(value))
    if emphasize_best:
        for column in table.columns:
            table[column] = _emphasize_column_top(table[column], emphasis_depth)
    return table


def build_correlation_table(
    display: pd.DataFrame,
    *,
    column_levels: Sequence[str] = tuple(DEFAULT_COLUMN_LEVELS),
    emphasize_best: bool = True,
    emphasis_depth: int = DEFAULT_EMPHASIS_DEPTH,
) -> pd.DataFrame:
    """Build the correlation table, appending a significance marker per value.

    When the display frame carries no ``pval`` column every value is marked as
    not significant, which keeps column widths identical to a run that does have
    p-values.
    """
    if display.empty:
        return pd.DataFrame()
    frame = display.copy()
    pvalues = (
        frame["pval"].astype(float)
        if "pval" in frame.columns
        else pd.Series([None] * len(frame), index=frame.index)
    )
    frame["_combined"] = frame["value"].map(_format_value) + pvalues.map(pval_str)
    table = frame.pivot_table(
        index=ROW_INDEX,
        columns=list(column_levels),
        values="_combined",
        aggfunc="first",
        observed=False,
    )
    table = table.reindex(MEASURE_GROUP_LABEL_ORDER, axis="index", level=0)
    table = table.dropna(axis="index", how="all").dropna(axis="columns", how="all")
    table = table.fillna("")
    if emphasize_best:
        for column in table.columns:
            table[column] = _emphasize_column_top(table[column], emphasis_depth)
    return table


def build_overview_table(
    display: pd.DataFrame,
    *,
    column_levels: Sequence[str] = tuple(DEFAULT_COLUMN_LEVELS),
    emphasis_depth: int = DEFAULT_EMPHASIS_DEPTH,
) -> pd.DataFrame:
    """Assemble the overview table, using correlation cells where they apply.

    Grounding-by-prediction columns carry significance markers and the remaining
    columns do not, which is why the two are built separately and then joined
    rather than formatted in one pass.
    """
    table = build_value_table(
        display, column_levels=column_levels, emphasis_depth=emphasis_depth
    )
    if table.empty or "Type" not in column_levels:
        return table
    prediction = display[display["Type"].eq(GROUNDING_BY_PREDICTION)]
    if prediction.empty:
        return table
    correlation = build_correlation_table(
        prediction, column_levels=column_levels, emphasis_depth=emphasis_depth
    )
    if correlation.empty:
        return table
    for column in correlation.columns:
        if column in table.columns:
            table[column] = correlation[column].reindex(table.index).fillna("")
    return table


__all__ = [
    "DEFAULT_COLUMN_LEVELS",
    "DEFAULT_EMPHASIS_DEPTH",
    "EMPHASIS_COMMANDS",
    "ROW_INDEX",
    "VALUE_PRECISION",
    "build_correlation_table",
    "build_overview_table",
    "build_value_table",
]
