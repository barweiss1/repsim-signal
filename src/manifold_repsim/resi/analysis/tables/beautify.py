"""Turn the normalized analysis frame into the display frame tables are built from.

The recreated tables are fed from this repository's manifest-normalized results
rather than from result filenames, so this module is where the two vocabularies
meet: normalized column names in, source table display names out.
"""

from __future__ import annotations

import pandas as pd

from ..config import measure_category
from .taxonomy import (
    ARCHITECTURE_LABELS,
    CORRELATION_TESTS,
    DATASET_LABELS,
    DESIGN_TESTS,
    DOMAIN_LABELS,
    FUNCTIONAL_EVAL_LABELS,
    FUNCTIONAL_TEST_LABELS,
    GROUNDING_BY_DESIGN,
    GROUNDING_BY_PREDICTION,
    INVERTED_QUALITY_LABEL,
    MEASURE_LABELS,
    MEASURE_GROUP_LABEL_ORDER,
    PREDICTION_FAMILY,
    measure_group_label,
    QUALITY_LABELS,
    TEST_LABELS,
    TEST_ORDER,
)

# Display columns produced from the normalized frame.
DISPLAY_COLUMNS = [
    "Measure Type",
    "Sim Meas.",
    "Type",
    "Test",
    "Eval.",
    "Domain",
    "Dataset",
    "Arch.",
    "value",
]

# Carried through when present so correlation tables can mark significance.
OPTIONAL_COLUMNS = ["pval", "model", "Functional Similarity Measure"]

REQUIRED_NORMALIZED_COLUMNS = {
    "domain",
    "benchmark",
    "dataset",
    "architecture",
    "metric",
    "quality_measure",
    "value",
}


def _mapped(series: pd.Series, labels: dict[str, str]) -> pd.Series:
    """Map through a label table, leaving unmapped values as their own label."""
    return series.astype(str).map(lambda value: labels.get(value, value))


def _split_correlation_tests(tests: pd.Series, functional: pd.Series) -> pd.Series:
    """Name each prediction-correlation row after the task it belongs to.

    One correlation benchmark produces one row per functional similarity
    measure, so the benchmark id cannot name the test: graphs' benchmark is
    called `output_correlation_test` and vision's `accoutput`, but both carry
    JSD, Disagreement, and AbsoluteAccDiff alike. Splitting here is what sorts
    the output measures into `Output Corr.` and the accuracy one into its own
    `Acc Corr.` task.

    Rows outside the correlation family, and correlation rows whose functional
    measure is blank or unrecognised, are returned unchanged.
    """
    split = functional.map(FUNCTIONAL_TEST_LABELS)
    return tests.where(tests.ne(PREDICTION_FAMILY) | split.isna(), split)


def build_display_frame(values: pd.DataFrame) -> pd.DataFrame:
    """Rename, group, and reorient normalized results for table construction.

    Applies the source tables' display vocabulary: measure abbreviations and
    groups, architecture and dataset labels, benchmark-to-test names, and the
    quality-measure names. Conformity rate is reported as one minus the
    violation rate so every column reads "higher is better".

    An optional ``pval`` column is carried through untouched for correlation
    tables; without it, significance markers render as not significant.
    """
    missing = REQUIRED_NORMALIZED_COLUMNS - set(values.columns)
    if missing:
        raise ValueError(
            f"Normalized results are missing required columns: {sorted(missing)}"
        )
    if values.empty:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)

    frame = values.copy()
    frame["Sim Meas."] = _mapped(frame["metric"], MEASURE_LABELS)
    frame["Measure Type"] = pd.Categorical(
        frame["Sim Meas."].map(
            lambda name: measure_group_label(measure_category(name))
        ),
        categories=MEASURE_GROUP_LABEL_ORDER,
        ordered=True,
    )
    frame["Arch."] = _mapped(frame["architecture"], ARCHITECTURE_LABELS)
    frame["Domain"] = _mapped(frame["domain"], DOMAIN_LABELS)
    frame["Dataset"] = _mapped(frame["dataset"], DATASET_LABELS)
    functional = (
        frame["functional_similarity_measure"].fillna("").astype(str)
        if "functional_similarity_measure" in frame.columns
        else pd.Series("", index=frame.index)
    )
    frame["Test"] = pd.Categorical(
        _split_correlation_tests(_mapped(frame["benchmark"], TEST_LABELS), functional),
        categories=TEST_ORDER,
        ordered=True,
    )
    frame["Eval."] = _mapped(frame["quality_measure"], QUALITY_LABELS)

    conformity = frame["Eval."].eq(INVERTED_QUALITY_LABEL)
    frame.loc[conformity, "value"] = 1 - frame.loc[conformity, "value"]

    # Inside a correlation task the varying quantity is which functional
    # similarity measure was correlated against, not which statistic was used
    # -- that is Spearman throughout. Naming the functional measure here is
    # what makes `Output Corr.` read like every other test, with its two
    # metrics side by side under one task. Applied after the conformity
    # inversion so that rule still sees the original quality label.
    functional_eval = functional.map(FUNCTIONAL_EVAL_LABELS)
    frame["Eval."] = frame["Eval."].where(functional_eval.isna(), functional_eval)

    frame["Type"] = pd.NA
    frame.loc[frame["Test"].isin(CORRELATION_TESTS), "Type"] = GROUNDING_BY_PREDICTION
    frame.loc[frame["Test"].isin(DESIGN_TESTS), "Type"] = GROUNDING_BY_DESIGN

    if "functional_similarity_measure" in frame.columns:
        frame["Functional Similarity Measure"] = functional

    keep = DISPLAY_COLUMNS + [
        column for column in OPTIONAL_COLUMNS if column in frame.columns
    ]
    return frame[keep]


__all__ = ["DISPLAY_COLUMNS", "OPTIONAL_COLUMNS", "build_display_frame"]
