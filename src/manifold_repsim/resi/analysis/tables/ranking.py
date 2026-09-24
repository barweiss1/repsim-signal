"""Measure ranking exactly as the source notebooks compute it.

This is deliberately not the same ranking as
``manifold_repsim.resi.analysis.ranking``, and the difference is now narrow
enough to name precisely. Both keep unmatched rows and rank a measure only
against the measures finite alongside it. This one additionally:

* breaks ties with ``method="min"`` rather than ``"average"``, so a five-way
  tie for first gives all five rank 1 rather than rank 3;
* applies no coverage filtering, so a measure ranked in a small and
  self-selected subset of the cases is pooled with the rest anyway;
* ranks every row, including one whose field of comparable measures has
  collapsed to a single survivor.

"Evaluation cell" means one quality measure, not one benchmark. The source
notebook reaches that by filtering to ``AUPRC``, ``spearmanr``, and
``correlation`` and folding the functional similarity measure into the setting
name, which leaves exactly one quality measure per group; `select_ranked_quality`
applies the same filter upstream of this module, and `Eval.` and
`Functional Similarity Measure` in the grouping keys reproduce the fold. Layer
monotonicity reports one value per layer model, so its ranks are averaged
within a cell and then deduplicated -- what gives that experiment the same
weight as one whose cell is a single aggregate row.
"""

from __future__ import annotations

import pandas as pd

RANK_KEYS = ["Domain", "Test", "Eval.", "Dataset", "Arch.", "model"]
LAYER_AVERAGE_KEYS = ["Domain", "Test", "Eval.", "Dataset", "Arch."]
AGGREGATE_MODEL = "agg"

# A correlation test reports the same quality measure once per functional
# similarity measure. Those are separate evaluations, so they rank separately
# and survive deduplication separately; pooling them ranks a measure against
# itself and keeping one at random discards the rest.
FUNCTIONAL_COLUMN = "Functional Similarity Measure"

DEDUPE_COLUMNS = (
    "Domain",
    "Test",
    "Dataset",
    "Arch.",
    "Sim Meas.",
    "Eval.",
    FUNCTIONAL_COLUMN,
)


def rank_measures(
    display: pd.DataFrame,
    *,
    token_aware: bool = False,
) -> pd.DataFrame:
    """Rank measures within each evaluation cell, best first.

    ``token_aware`` adds the language token level to the grouping. The source
    notebooks branch on domain to decide this, because only the language results
    have a token dimension.

    Layer-monotonicity rows report one value per layer model. Their ranks are
    averaged within a cell and then deduplicated, so a benchmark with many layer
    models does not outweigh one with a single aggregate row.
    """
    if display.empty:
        return display.assign(rank=pd.Series(dtype=float))

    frame = display.copy()
    if "model" not in frame.columns:
        frame["model"] = AGGREGATE_MODEL
    frame["model"] = frame["model"].fillna(AGGREGATE_MODEL).replace("", AGGREGATE_MODEL)

    rank_keys = list(RANK_KEYS)
    average_keys = list(LAYER_AVERAGE_KEYS)
    if FUNCTIONAL_COLUMN in frame.columns:
        rank_keys.append(FUNCTIONAL_COLUMN)
        average_keys.append(FUNCTIONAL_COLUMN)
    if token_aware:
        if "Token" not in frame.columns:
            raise ValueError("token_aware ranking requires a 'Token' column.")
        frame["Token"] = frame["Token"].fillna("NA")
        rank_keys.append("Token")
        average_keys.append("Token")

    frame["rank"] = frame.groupby(rank_keys, dropna=False, observed=False)[
        "value"
    ].rank(ascending=False, method="min", na_option="keep")

    layer_rows = frame["model"].ne(AGGREGATE_MODEL) & frame["rank"].notna()
    if layer_rows.any():
        frame.loc[layer_rows, "rank"] = (
            frame[layer_rows]
            .groupby(average_keys, dropna=False, observed=False)["rank"]
            .transform("mean")
        )

    dedupe = [column for column in DEDUPE_COLUMNS if column in frame.columns]
    return frame.drop_duplicates(subset=dedupe)


def summarize_ranks(ranked: pd.DataFrame) -> pd.DataFrame:
    """Average and median each measure's rank per domain, best first."""
    columns = ["Domain", "Sim Meas.", "avg_rank", "med_rank"]
    if ranked.empty or "rank" not in ranked.columns:
        return pd.DataFrame(columns=columns)
    summary = (
        ranked.groupby(["Domain", "Sim Meas."], dropna=False, observed=False)["rank"]
        .agg(["mean", "median"])
        .reset_index()
        .rename(columns={"mean": "avg_rank", "median": "med_rank"})
    )
    return summary.sort_values(["med_rank", "avg_rank"]).reset_index(drop=True)


__all__ = [
    "AGGREGATE_MODEL",
    "DEDUPE_COLUMNS",
    "FUNCTIONAL_COLUMN",
    "LAYER_AVERAGE_KEYS",
    "RANK_KEYS",
    "rank_measures",
    "summarize_ranks",
]
