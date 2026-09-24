"""Write the recreated appendix tables and their figures for one analysis run.

The pieces this orchestrates already existed but were reachable only from
tests: nothing in the analysis workflow produced a table or a published-mode
rank figure. This module is the missing entry point.

Three outputs, matching the paper's appendix and the two rankings this
repository maintains:

* One table per test, laid out as the paper's appendix tables are -- measures
  down the side, evaluation/dataset/architecture across the top, best value in
  each column bolded, correlation cells carrying significance markers. The
  prediction-correlation benchmark contributes two of these tests, not one:
  `Output Corr.` (JSD and Disagreement, two ways of asking how far two models'
  outputs agree) and `Acc Corr.` (AbsoluteAccDiff, which asks how far apart
  their accuracies are). Within `Output Corr.` the two functional measures are
  named `Eval.` blocks, so they sit side by side without pooling silently.
* One value-space boxplot per test, so a measure's spread within a test is
  visible rather than only its rank. Every test's figure is panelled by its
  evaluation measures, which for `Output Corr.` means one panel each for JSD
  and Disagreement.
* The published-mode rank boxplot: ReSi's own ranking, reproduced step for
  step, including its filter to the three quality measures it ranks on. The
  repository's own figure is written by the analysis workflow itself, so a run
  carries both side by side and the remaining differences between them --
  tie-breaking and coverage filtering, listed in `ranking.py` -- can be read
  off rather than argued about.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from manifold_repsim.resi.paths import slug

from ..ranking import select_ranked_quality
from .beautify import build_display_frame
from .pivots import build_overview_table
from .ranking import rank_measures, summarize_ranks
from .taxonomy import CORRELATION_TESTS, FUNCTIONAL_TEST_LABELS, TEST_ORDER

TABLES_DIRNAME = "tables"

# Appendix tables fix the test, so the remaining levels are what varies inside
# one. `Type` stays because the overview builder uses it to decide which cells
# carry significance markers.
#
# Every task uses these levels, correlation tasks included. `Output Corr.`
# needs no extra `Test` level to keep JSD and Disagreement apart, because
# `Eval.` already names them -- the same level that separates AUPRC from
# conformity rate elsewhere. The defect this replaced was pooling functional
# measures under one *unnamed* heading; naming them in `Eval.` fixes that
# without a table structure unique to one family.
APPENDIX_COLUMN_LEVELS = ("Type", "Eval.", "Dataset", "Arch.")

REPORT_INDEX_COLUMNS = [
    "test",
    "rows",
    "columns",
    "latex_path",
    "csv_path",
    "figure_path",
    "note",
]


def _tests_in_order(display: pd.DataFrame) -> list[str]:
    """Tests present in the frame, in the paper's order, unknown ones last."""
    present = {str(name) for name in display["Test"].dropna().unique()}
    ordered = [name for name in TEST_ORDER if name in present]
    return ordered + sorted(present - set(ordered))


def _table_groups(
    display: pd.DataFrame,
) -> list[tuple[str, list[str], tuple[str, ...]]]:
    """The tables to write: (name, tests it covers, its column levels).

    Every task is its own table, correlation tasks included. `Output Corr.` and
    `Acc Corr.` are two different questions -- how far two models' outputs
    agree, and how far their accuracies do -- so they get separate tables for
    the same reason `Shortcuts` and `Augmentation` do. Within `Output Corr.`
    the two functional measures stay side by side as `Eval.` blocks, which is
    where a reader compares them.
    """
    return [(name, [name], APPENDIX_COLUMN_LEVELS) for name in _tests_in_order(display)]


def _missing_correlation_splits(display: pd.DataFrame) -> list[dict[str, object]]:
    """Index rows for correlation tasks this domain reports nothing for.

    A domain that scores no `AbsoluteAccDiff` correlations simply writes no
    accuracy-correlation table, and a missing file is indistinguishable from a
    file nobody generated. Recording the gap keeps it a reported absence: the
    graph archive carries no AbsoluteAccDiff values for ReSi's own measures and
    the analysis config drops the setting outright, while language has no
    SmolLM2 accuracy to difference.

    Reported per task rather than per functional measure, so a domain missing
    `Output Corr.` is charged one absent table naming both measures it covers
    rather than two rows for one file. Only reported when the domain runs a
    prediction-correlation benchmark at all.
    """
    present = {str(name) for name in display["Test"].dropna().unique()}
    if not present & set(CORRELATION_TESTS):
        return []
    covered: dict[str, list[str]] = {}
    for functional, label in FUNCTIONAL_TEST_LABELS.items():
        covered.setdefault(label, []).append(functional)
    return [
        {
            "test": label,
            "rows": 0,
            "columns": 0,
            "latex_path": "",
            "csv_path": "",
            "figure_path": "",
            "note": (
                f"not reported: no {' or '.join(functionals)} correlations in "
                "this domain's results"
            ),
        }
        for label, functionals in covered.items()
        if label not in present
    ]


def write_appendix_report(
    values: pd.DataFrame,
    out_dir: str | Path,
    *,
    token_aware: bool = False,
    box_style: str | None = None,
) -> pd.DataFrame:
    """Write every appendix table and figure for one domain's results.

    `values` is this repository's normalized analysis frame, the same one the
    domain summary is built from, so the tables cannot disagree with the
    ranking about what was measured.

    `box_style` is the run's chosen boxplot drawing, passed through so these
    figures match the rest of the domain's. It reaches only the drawing: the
    published ranking convention these tables reproduce is untouched.

    `token_aware` adds the language token level to the published ranking. It
    applies only when the display frame actually carries a `Token` column; this
    repository's normalized results do not model the token dimension yet, so
    asking for it is a no-op rather than an error.

    Returns an index of what was written, one row per test, empty when there is
    nothing to report. Prediction-correlation settings the domain reports no
    values for appear in that index with zero rows and a note, so a missing
    table reads as a recorded gap rather than as an output nobody generated.
    """
    directory = Path(out_dir) / TABLES_DIRNAME
    if values.empty:
        return pd.DataFrame(columns=REPORT_INDEX_COLUMNS)

    display = build_display_frame(values)
    if display.empty:
        return pd.DataFrame(columns=REPORT_INDEX_COLUMNS)

    directory.mkdir(parents=True, exist_ok=True)
    # Imported here so building tables does not require a plotting backend.
    from .figures import plot_rank_distributions, plot_task_value_distributions

    rows: list[dict[str, object]] = []
    for label, tests, levels in _table_groups(display):
        panel = display[display["Test"].astype(str).isin(tests)]
        if panel.empty:
            continue
        name = slug(label)
        table = build_overview_table(panel, column_levels=levels)
        latex_path = csv_path = None
        note = ""
        if not table.empty:
            from .latex import write_latex_table

            csv_path = directory / f"{name}.csv"
            table.to_csv(csv_path)
            # The paper's layout is built by rewriting pandas' emitted rows and
            # needs a measure group spanning more than one row to place its
            # label against. A table too small for that is a real table with an
            # unrenderable layout, not a failed analysis: keep the CSV, record
            # why the .tex is missing, and never fail the run over a figure.
            try:
                latex_path = write_latex_table(
                    table,
                    directory / f"{name}.tex",
                    # Test names already end in a period ("Layer Mono."),
                    # which made every caption read "Layer Mono..".
                    caption=f"Results of {label.rstrip('.')}.",
                    label=f"tab:{name}",
                )
            except ValueError as exc:
                note = f"latex layout skipped: {exc}"
        # One panel per evaluation measure, for every task alike: `Output Corr.`
        # panels into JSD and Disagreement because those are now its `Eval.`
        # values, which is the comparison a reader of that figure wants.
        figure_path = plot_task_value_distributions(
            display,
            directory / f"{name}_values.png",
            test=label,
            box_style=box_style,
        )
        rows.append(
            {
                "test": label,
                "rows": int(table.shape[0]) if not table.empty else 0,
                "columns": int(table.shape[1]) if not table.empty else 0,
                "latex_path": str(latex_path) if latex_path else "",
                "csv_path": str(csv_path) if csv_path else "",
                "figure_path": str(figure_path) if figure_path else "",
                "note": note,
            }
        )

    # The published ranking is built from its own display frame, narrowed to
    # the quality measures ReSi ranks on. The tables above deliberately keep
    # the conformity-rate columns this drops -- they are a reading of the same
    # tests that the source tables do not print, and the place to show them is
    # a table, not a ranking that would then count each design test twice.
    published = build_display_frame(select_ranked_quality(values))
    ranked = rank_measures(
        published, token_aware=token_aware and "Token" in published.columns
    )
    ranked.to_csv(directory / "published_case_ranks.csv", index=False)
    summarize_ranks(ranked).to_csv(directory / "published_ranks.csv", index=False)
    modalities = [str(name) for name in display["Domain"].dropna().unique()]
    if modalities:
        plot_rank_distributions(
            ranked,
            directory / "rank_boxplot_published.png",
            modalities=modalities,
            box_style=box_style,
        )

    rows.extend(_missing_correlation_splits(display))
    index = pd.DataFrame(rows, columns=REPORT_INDEX_COLUMNS)
    index.to_csv(directory / "tables_index.csv", index=False)
    return index


__all__ = [
    "APPENDIX_COLUMN_LEVELS",
    "REPORT_INDEX_COLUMNS",
    "TABLES_DIRNAME",
    "write_appendix_report",
]
