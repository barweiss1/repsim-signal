"""Render an emphasized table to the paper's LaTeX layout.

Pandas emits a plain tabular; the paper layout needs a resized, row-coloured
table whose measure-group labels sit at the bottom of each block. That is done
by rewriting the emitted lines. The header/body boundary is located by
searching for its rule rather than by fixed index, so an added or removed
header level does not silently corrupt the output the way index-based edits
would; the group blocks below it come from the table's own index.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from pandas.io.formats.style import Styler

FIRST_HEADER_LINE = 7
# The trailing \bottomrule, \end{tabular}, \end{table} and final blank line.
TRAILING_LINES = 4


def _find_line_index(lines: list[str], search_items: list[str]) -> int:
    for index, line in enumerate(lines):
        if any(item in line for item in search_items):
            return index
    raise ValueError(f"Could not find line with any of {search_items}")


def _safe_label(value: object) -> str:
    """One label, underscores removed unless the label is already mathematics.

    A label wrapped in `$...$` is math the caller wrote on purpose -- the
    measure names are `$\\mathrm{CKA}_\\mathrm{lin}$` and the like -- and its
    underscores are subscripts, not characters to sanitise. Stripping them
    turned `CKA_lin` into `CKA lin`, which compiles and is wrong.
    """
    text = str(value)
    if text.startswith("$") and text.endswith("$") and len(text) > 1:
        return text
    return text.replace("_", " ")


def latex_safe_labels(table: pd.DataFrame) -> pd.DataFrame:
    """Replace underscores in index and column labels with spaces.

    These tables render with `escape=False`, because the cells carry `\\textbf`
    and significance markers that must reach LaTeX intact. An underscore in a
    label therefore reaches LaTeX unescaped, where it opens math mode and fails
    to compile. Architecture names are where this bites -- `ViT_B32`, `ViT_L32`
    -- and any unmapped value passes through the label tables unchanged, so
    sanitising here covers names no label map knows about.

    Replaced rather than escaped: a header is prose, and `ViT B32` reads as the
    name where `ViT\\_B32` reads as markup. Only the rendered table is touched;
    the CSV beside it keeps the raw names.
    """
    return table.rename(index=_safe_label, columns=_safe_label)


def _measure_group_blocks(table: pd.DataFrame) -> list[tuple[int, int, str]]:
    """Each measure-group block as (row offset, span, label), in row order.

    Read off the table's own index rather than recovered from the emitted
    LaTeX. Pandas writes a `\\multirow` only for a group holding more than one
    measure and puts a single-measure group's label inline, so scanning for
    multirows silently loses those blocks -- which costs them their separating
    rule, and misplaces the header/body boundary when the first group is the
    single-measure one. The finer shared taxonomy produces such groups
    routinely; the collapsed one this replaced almost never did.
    """
    if table.index.nlevels < 2:
        raise ValueError("Rendered table has no measure-group rows to lay out.")
    labels = [str(name) for name in table.index.get_level_values(0)]
    blocks: list[tuple[int, int, str]] = []
    for offset, label in enumerate(labels):
        if blocks and blocks[-1][2] == label:
            start, span, name = blocks[-1]
            blocks[-1] = (start, span + 1, name)
        else:
            blocks.append((offset, 1, label))
    return blocks


def _parse_table_layout(line: str) -> str:
    """Derive the column format from a header row's multicolumn spans."""
    row = line.replace("\\rowcolor{white}", "").strip()
    columns = [column.strip() for column in row.split("&")]
    layout = ["ll|"]
    for column in columns[2:]:
        if column.startswith("\\multicolumn"):
            span = int(column.split("{")[1].split("}")[0])
            layout.append("r" * span)
        else:
            layout.append("r")
    return "|".join(layout)


def render_latex_table(
    table: pd.DataFrame,
    caption: str,
    label: str,
    *,
    resizebox_width: float = 1.0,
    drop_header_rows: tuple[str, ...] = ("Sim Meas.", "Modality"),
) -> str:
    """Render one emphasized table to the paper's LaTeX layout.

    ``drop_header_rows`` names header rows whose labels are redundant in the
    printed table; each is located by content and removed. Measure-group labels
    are moved to the bottom of their block with a negative multirow span, and a
    rule is added between groups.
    """
    if table.empty:
        raise ValueError("Cannot render an empty table.")

    table = latex_safe_labels(table)
    styled = Styler(table, precision=2)
    latex_str = styled.to_latex(
        hrules=True,
        position="h",
        label=label,
        caption=caption,
        column_format="",
    )

    lines = latex_str.split("\n")
    lines = (
        lines[:3]
        + [r"\centering"]
        + [r"\resizebox{" + str(resizebox_width) + r"\linewidth}{!}{"]
        + [r"\rowcolors{2}{white}{Gray}"]
        + lines[3:]
    )
    lines = [
        re.sub(r"\{r\}", r"{c}", line) if "multicolumn" in line else line
        for line in lines
    ]
    for header in drop_header_rows:
        try:
            lines.pop(_find_line_index(lines, [header]))
        except ValueError:
            # A table without that header level simply has nothing to remove.
            continue

    blocks = _measure_group_blocks(table)
    first_data_line = _find_line_index(lines, [r"\midrule"]) + 1
    lines = (
        lines[:FIRST_HEADER_LINE]
        + [
            r"\rowcolor{white}" + line
            for line in lines[FIRST_HEADER_LINE:first_data_line]
        ]
        + lines[first_data_line:]
    )

    for position, (offset, span, name) in enumerate(blocks):
        first_index = first_data_line + offset
        last_index = first_index + span - 1
        # A one-measure group already carries its label on its only row.
        if span > 1:
            lines[first_index] = lines[first_index].replace(
                r"\multirow[c]{" + str(span) + "}{*}{" + name + "}", ""
            )
            lines[last_index] = (
                r"\multirow[c]{" + str(-span) + "}{*}{" + name + "}" + lines[last_index]
            )
        if position != len(blocks) - 1:
            lines[last_index] += r"\midrule"

    lines = (
        lines[:first_data_line]
        + [
            r"\cellcolor{white}" + line
            for line in lines[first_data_line:-TRAILING_LINES]
        ]
        + lines[-TRAILING_LINES:]
    )

    # The column layout comes from the second-deepest header row, whose
    # multicolumn spans are what group the leaf columns and place the vertical
    # rules between groups. The deepest row is all single columns, so deriving
    # from it would rule every column separately.
    #
    # Located by position rather than by name: this used to match the literal
    # "& Dataset", which silently emitted an empty `\begin{tabular}{}` -- an
    # uncompilable table -- for any table whose levels end differently, such as
    # the exemplary table's (Type, Test, Eval., Domain). For the appendix and
    # default level sets the second-deepest row is still `Dataset`, so this
    # derives exactly the layout it did before.
    header_rows = [
        line
        for line in lines[FIRST_HEADER_LINE:first_data_line]
        if line.startswith(r"\rowcolor{white} & ")
    ]
    if header_rows:
        grouping_row = header_rows[-2] if len(header_rows) >= 2 else header_rows[-1]
        lines[6] = r"\begin{tabular}{" + _parse_table_layout(grouping_row) + "}"

    lines = lines[:-2] + [r"}"] + lines[-2:]
    return "\n".join(lines)


def write_latex_table(
    table: pd.DataFrame,
    out_path: str | Path,
    caption: str,
    label: str,
    *,
    resizebox_width: float = 1.0,
) -> Path:
    """Render a table and write it, creating the output directory if needed."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_latex_table(table, caption, label, resizebox_width=resizebox_width),
        encoding="utf-8",
    )
    return path


__all__ = ["latex_safe_labels", "render_latex_table", "write_latex_table"]
