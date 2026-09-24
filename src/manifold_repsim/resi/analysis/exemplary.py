"""ReSi's Table 3 recreated, and the same table averaged over every model.

The per-domain appendix tables report every architecture and dataset, which is
the right level of detail for an appendix and the wrong one for a table a
reader is meant to scan. ReSi's Table 3 solves that by fixing one exemplary
(dataset, architecture) cell per domain and putting every test across the top,
so one page answers "does any measure do well everywhere". This module builds
the same table from this repository's results.

It also builds the averaged counterpart: the same rows, the same columns, but
each cell averaged over every architecture and dataset the domain ran instead
of read off one chosen cell. The exemplary table can be swung by one unlucky
model, which is the price of naming a cell; the averaged one cannot, at the
price of hiding how much a measure varies between models. They answer
different questions and are written side by side rather than one replacing the
other. `average_display_cells` documents the order the means are taken in,
which is what stops an architecture with more datasets or more seeds from
weighing more than its neighbours.

Two deliberate differences from the source table, both recorded rather than
papered over:

* It carries a `Disagreement` column. ReSi's main table keeps only JSD
  correlations and discards `Disagreement` entirely; we report it (see
  `docs/resi_analysis_methods.md` 7.8), so dropping it here to match would hide
  a result we have.
* It has no significance markers. Those need per-correlation p-values, which
  this repository does not propagate into the normalized frame. The formatter
  emits a `\\phantom` spacer for a missing p-value, so the columns keep the
  source table's alignment without claiming anything about significance.

The token dimension is also absent: ReSi's caption specifies BERT's CLS token,
and this repository's normalized results do not model tokens, so the language
column is whatever token convention the campaign ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .tables.beautify import build_display_frame
from .tables.pivots import build_overview_table
from .tables.taxonomy import DOMAIN_LABELS

# Both tables here mark the best three values in each column rather than the
# best one. These are the two tables meant to be scanned, and with sixteen
# measures a single bold cell says which measure won without saying who was
# close -- which is the question a reader comparing measures actually has. The
# appendix tables keep the source notebooks' bold-only marking.
EMPHASIS_DEPTH = 3
EMPHASIS_LEGEND = (
    "Best value in each column in bold, second best underlined, third best in "
    "italics."
)

# The exact cells ReSi's Table 3 caption names, so the two can be read side by
# side. Keys are raw domain names; values are raw (dataset, architecture) as
# they appear in `normalized_values.csv`, not their display labels.
EXEMPLARY_CELLS: dict[str, tuple[str, str]] = {
    "graphs": ("flickr", "GraphSAGE"),
    "language": ("sst2", "BERT-L"),
    "vision": ("ImageNet100", "ResNet18"),
}

# Which evaluation measure each task is reported on. ReSi's Table 3 reports
# Spearman for the two prediction-grounded tests, AUPRC for the three
# design-grounded ones, and the averaged Spearman for layer monotonicity. Our
# correlation tasks name the functional measure in `Eval.` rather than
# repeating "Spearman", so those entries name the functional measures instead.
TASK_EVALUATIONS: dict[str, tuple[str, ...]] = {
    "Acc Corr.": ("Acc Diff",),
    "Output Corr.": ("JSD", "Disagreement"),
    "Random Labels": ("AUPRC",),
    "Shortcuts": ("AUPRC",),
    "Augmentation": ("AUPRC",),
    "Layer Mono.": ("Spearman",),
}

# `Test` and `Eval.` both stay as levels rather than being merged into one
# synthetic label: `build_value_table` orders columns by reindexing on the
# shared `TEST_ORDER`, so an invented label would be dropped instead of
# ordered. Keeping them separate also names the quality measure in the header
# that the source table relegates to its caption.
EXEMPLARY_COLUMN_LEVELS = ("Type", "Test", "Eval.", "Domain")


def select_exemplary_rows(values: pd.DataFrame, cell: tuple[str, str]) -> pd.DataFrame:
    """Keep only the rows for one (dataset, architecture) cell.

    Returns an empty frame when the cell is absent, so a domain that never ran
    the exemplary model drops out of the table instead of raising.
    """
    if values.empty:
        return values
    dataset, architecture = cell
    keep = values["dataset"].astype(str).eq(dataset) & values["architecture"].astype(
        str
    ).eq(architecture)
    return values[keep]


def _keep_reported_evaluations(display: pd.DataFrame) -> pd.DataFrame:
    """Drop the evaluation measures this table does not report per task.

    Every task computes more than it shows -- the design tests carry both AUPRC
    and conformity rate, layer monotonicity carries both Spearman and
    conformity rate. Showing all of them would double the table's width and
    stop it being scannable, which is the one thing this table is for.
    """
    if display.empty:
        return display
    tests = display["Test"].astype(str)
    evaluations = display["Eval."].astype(str)
    keep = pd.Series(False, index=display.index)
    for test, allowed in TASK_EVALUATIONS.items():
        keep |= tests.eq(test) & evaluations.isin(allowed)
    return display[keep]


def build_exemplary_display(
    frames: Mapping[str, pd.DataFrame],
    *,
    cells: Mapping[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Display rows for every domain's exemplary cell, stacked into one frame.

    `frames` maps a domain name to that domain's `normalized_values.csv`
    contents, the same input the cross-domain layer takes. A domain absent from
    `cells` contributes nothing rather than contributing all of its rows.
    """
    chosen = dict(EXEMPLARY_CELLS if cells is None else cells)
    collected: list[pd.DataFrame] = []
    for domain, frame in frames.items():
        cell = chosen.get(domain)
        if cell is None or frame.empty:
            continue
        rows = select_exemplary_rows(frame, cell)
        if rows.empty:
            continue
        collected.append(build_display_frame(rows))
    if not collected:
        return pd.DataFrame()
    return _keep_reported_evaluations(pd.concat(collected, ignore_index=True))


def build_exemplary_table(
    frames: Mapping[str, pd.DataFrame],
    *,
    cells: Mapping[str, tuple[str, str]] | None = None,
    column_levels: Sequence[str] = EXEMPLARY_COLUMN_LEVELS,
) -> pd.DataFrame:
    """Measures down the side, every test across the top, one cell per domain."""
    display = build_exemplary_display(frames, cells=cells)
    if display.empty:
        return pd.DataFrame()
    return build_overview_table(
        display, column_levels=column_levels, emphasis_depth=EMPHASIS_DEPTH
    )


def exemplary_caption(
    cells: Mapping[str, tuple[str, str]] | None = None,
) -> str:
    """Name the cells the table was built from, in the source table's style."""
    chosen = dict(EXEMPLARY_CELLS if cells is None else cells)
    parts = [
        f"{architecture} on {dataset} for {domain}"
        for domain, (dataset, architecture) in chosen.items()
    ]
    return (
        "Exemplary results for selected datasets and models. We show results of "
        + ", ".join(parts)
        + ". Higher values indicate that a similarity measure better reflects "
        "the ground truths from our tests. " + EMPHASIS_LEGEND
    )


INDEX_COLUMNS = ["table", "rows", "columns", "csv_path", "latex_path", "note"]


def _write_table(
    table: pd.DataFrame,
    out_dir: str | Path,
    *,
    name: str,
    caption: str,
) -> pd.DataFrame:
    """Write one built table as CSV and LaTeX, returning a one-row index.

    Shared by both tables in this module so the empty-input contract and the
    unrenderable-layout note cannot drift apart between them.
    """
    if table.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{name}.csv"
    table.to_csv(csv_path)

    latex_path: Path | None = None
    note = ""
    from .tables.latex import write_latex_table

    try:
        latex_path = write_latex_table(
            table,
            directory / f"{name}.tex",
            caption=caption,
            label=f"tab:{name}",
        )
    except ValueError as exc:
        # Same contract as the appendix report: a table too small for the
        # paper's row-block layout is a real table with an unrenderable
        # layout, not a failed analysis.
        note = f"latex layout skipped: {exc}"

    return pd.DataFrame(
        [
            {
                "table": name,
                "rows": int(table.shape[0]),
                "columns": int(table.shape[1]),
                "csv_path": str(csv_path),
                "latex_path": "" if latex_path is None else str(latex_path),
                "note": note,
            }
        ],
        columns=INDEX_COLUMNS,
    )


def write_exemplary_table(
    frames: Mapping[str, pd.DataFrame],
    out_dir: str | Path,
    *,
    cells: Mapping[str, tuple[str, str]] | None = None,
    name: str = "exemplary",
) -> pd.DataFrame:
    """Write the exemplary table as CSV and LaTeX, returning a one-row index.

    Returns an empty frame when no domain supplied its exemplary cell, so a
    caller can report the gap rather than write an empty table.
    """
    return _write_table(
        build_exemplary_table(frames, cells=cells),
        out_dir,
        name=name,
        caption=exemplary_caption(cells),
    )


# The order the means are taken in, outermost group last. Each step collapses
# one dimension, and the dimension it collapses is what the step above then
# weights equally: seeds within a (dataset, architecture) cell, then that
# architecture's datasets, then the domain's architectures. Averaging in one
# pass instead would weight an architecture by how many datasets and seeds it
# happens to have run, which is exactly the imbalance the per-domain results
# carry -- vision runs seven architectures against language's two, and layer
# monotonicity carries ten observations per cell where every other test
# carries one.
AVERAGING_STEPS: tuple[tuple[str, ...], ...] = (
    ("Arch.", "Dataset"),
    ("Arch.",),
    (),
)

# What identifies one cell of the averaged table: a measure, a column, and a
# domain. Everything else is averaged away by `AVERAGING_STEPS`.
AVERAGED_CELL_KEYS = ("Measure Type", "Sim Meas.", "Type", "Test", "Eval.", "Domain")


def average_display_cells(display: pd.DataFrame) -> pd.DataFrame:
    """Average display rows down to one value per measure, column, and domain.

    Takes the means in `AVERAGING_STEPS` order rather than in one pass, so an
    architecture's weight does not depend on how many datasets or seeds it ran.
    Whatever is present is averaged: a measure missing from a domain, or an
    architecture that never ran a dataset, contributes nothing to that mean
    instead of emptying it.

    A `pval` column is dropped rather than carried. Averaging p-values is not a
    p-value, and the correlation formatter reads the column if it is there.
    """
    if display.empty:
        return display
    frame = display.drop(
        columns=[name for name in ("pval",) if name in display.columns]
    )
    keys = [name for name in AVERAGED_CELL_KEYS if name in frame.columns]
    for step in AVERAGING_STEPS:
        levels = keys + [name for name in step if name in frame.columns]
        frame = (
            frame.groupby(levels, dropna=False, observed=True)["value"]
            .mean()
            .reset_index()
        )
    return frame


def build_averaged_display(
    frames: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Averaged display rows for every domain, stacked into one frame.

    Takes the same `frames` mapping the exemplary table does and reports every
    (dataset, architecture) cell each domain ran, rather than one named cell.
    """
    collected: list[pd.DataFrame] = []
    for frame in frames.values():
        if frame.empty:
            continue
        collected.append(build_display_frame(frame))
    if not collected:
        return pd.DataFrame()
    display = _keep_reported_evaluations(pd.concat(collected, ignore_index=True))
    return average_display_cells(display)


def build_averaged_table(
    frames: Mapping[str, pd.DataFrame],
    *,
    column_levels: Sequence[str] = EXEMPLARY_COLUMN_LEVELS,
) -> pd.DataFrame:
    """The exemplary table's columns, averaged over every model and dataset."""
    display = build_averaged_display(frames)
    if display.empty:
        return pd.DataFrame()
    return build_overview_table(
        display, column_levels=column_levels, emphasis_depth=EMPHASIS_DEPTH
    )


def averaged_caption(frames: Mapping[str, pd.DataFrame] | None = None) -> str:
    """State the averaging order, and how many models each domain averages.

    The counts come from the frames rather than from a constant: a table that
    claims seven vision architectures when the run shipped six would be worse
    than one that claims nothing.
    """
    parts = []
    for domain, frame in sorted((frames or {}).items()):
        if frame.empty or "architecture" not in frame.columns:
            continue
        label = DOMAIN_LABELS.get(domain, domain)
        architectures = frame["architecture"].astype(str).nunique()
        datasets = frame["dataset"].astype(str).nunique()
        parts.append(f"{label}: {architectures} models on {datasets} datasets")
    scope = f" ({'; '.join(parts)})" if parts else ""
    return (
        "Averaged results over every model and dataset per domain"
        + scope
        + ". Each cell averages seeds first, then that model's datasets, then "
        "the domain's models, so every model counts once regardless of how "
        "many datasets or seeds it ran. Higher values indicate that a "
        "similarity measure better reflects the ground truths from our tests. "
        + EMPHASIS_LEGEND
    )


def write_averaged_table(
    frames: Mapping[str, pd.DataFrame],
    out_dir: str | Path,
    *,
    name: str = "averaged",
) -> pd.DataFrame:
    """Write the averaged table as CSV and LaTeX, returning a one-row index."""
    return _write_table(
        build_averaged_table(frames),
        out_dir,
        name=name,
        caption=averaged_caption(frames),
    )


__all__ = [
    "AVERAGED_CELL_KEYS",
    "AVERAGING_STEPS",
    "EMPHASIS_DEPTH",
    "EMPHASIS_LEGEND",
    "EXEMPLARY_CELLS",
    "EXEMPLARY_COLUMN_LEVELS",
    "INDEX_COLUMNS",
    "TASK_EVALUATIONS",
    "average_display_cells",
    "averaged_caption",
    "build_averaged_display",
    "build_averaged_table",
    "build_exemplary_display",
    "build_exemplary_table",
    "exemplary_caption",
    "select_exemplary_rows",
    "write_averaged_table",
    "write_exemplary_table",
]
