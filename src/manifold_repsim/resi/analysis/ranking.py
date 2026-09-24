"""Within-case ranking and domain-level rank summaries.

Measures are compared only within a case, and within a case only against the
measures that actually produced a finite value for the same observation. A
measure that failed there is skipped for that observation rather than costing
every other measure the observation too -- ReSi's ``na_option="keep"``
convention, which this module now follows.

Ranks are pooled across cases as raw within-case ranks by default: inside one
domain every case ranks the same field, so "3rd of 16" means the same thing in
each of them. ``normalize_case_rank`` rescales to ``[0, 1]`` for the cases where
that does not hold -- chiefly pooling across domains, which report different
field sizes. The scale is selectable via ``rank_scale``; both sets of statistics
are always written, so only ordering and the reported headline change.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# The rank-scale vocabulary lives in `config` so that `settings` can validate a
# YAML value without importing pandas through this module. It is re-exported
# here because this is where the ranking behavior it selects is implemented.
from .config import (
    DEFAULT_RANK_SCALE,
    RANKED_FUNCTIONAL_MEASURES,
    RANKED_QUALITY_MEASURES,
    RANK_SCALES,
    quality_direction,
    rank_scale_case_column,
    validate_rank_scale,
)

# One case is a single comparable evaluation context. Ranks are only meaningful
# between measures inside one of these.
CASE_COLUMNS = [
    "domain",
    "benchmark",
    "dataset",
    "architecture",
    "quality_measure",
    "identifier",
    "representation_dataset",
    "functional_similarity_measure",
]

DOMAIN_SUMMARY_COLUMNS = [
    "domain",
    "metric",
    "mean_normalized_rank",
    "median_normalized_rank",
    "std_normalized_rank",
    "mean_rank",
    "median_rank",
    "std_rank",
    "n_cases",
]


def normalize_case_rank(rank: Any, n_compared_metrics: Any) -> Any:
    """Rescale a within-case rank to ``[0, 1]``, 0 best and 1 worst.

    Formula: ``(rank - 1) / (n_compared_metrics - 1)``.

    A raw rank is only interpretable against the size of the field it was drawn
    from: 5th of 34 is the top 15%, while 5th of 19 is the top 26%. Pooling raw
    ranks across fields of different sizes therefore mixes two scales, and lets
    field size alone reorder the summary -- mid-pack is 10.0 in a 19-measure
    field and 17.5 in a 34-measure one, so a mediocre measure present only in
    small fields outranks an equally mediocre one present only in large ones.
    Rescaling first makes the pooled mean comparable.

    Within one domain the reported field is fixed, so this changes nothing but
    the axis; it is what makes the cross-domain layer able to pool at all.

    This equalizes scale, not difficulty: a normalized rank still reflects the
    particular competitors present in that observation. Comparing measures whose
    case coverage differs remains a judgement call, which is why the reported
    measure set is an explicit analysis input (see ``analysis/settings.py``).

    ``rank_case_observations`` skips observations that ranked fewer than two
    measures, so the denominator is at least one wherever this is applied.
    """
    return (rank - 1.0) / (n_compared_metrics - 1.0)


def select_ranked_quality(
    values: pd.DataFrame, quality_measures: Iterable[str] = RANKED_QUALITY_MEASURES
) -> pd.DataFrame:
    """Keep only the quality measures that get ranked.

    Ranking pools one case per (task, dataset, architecture, functional
    measure), so a task scored under two quality measures contributes two cases
    and counts twice. ``RANKED_QUALITY_MEASURES`` is ReSi's own answer to that:
    one quality measure per test, chosen the same way in every domain.

    Applied at ranking time rather than at load, so everything that is not a
    rank -- the value tables, the quality heatmaps, the AUPRC-versus-violation
    scatter -- still sees every quality measure the analysis config reported.
    """
    if values.empty or "quality_measure" not in values.columns:
        return values
    keep = values["quality_measure"].astype(str).isin(set(quality_measures))
    return values[keep].reset_index(drop=True)


def select_ranked_functional(
    values: pd.DataFrame,
    functional_measures: Iterable[str] = RANKED_FUNCTIONAL_MEASURES,
) -> pd.DataFrame:
    """Keep the one functional similarity measure that names each task.

    The correlation benchmark scores one set of representations against several
    functional similarity measures at once, and a case is keyed on which. Two of
    them, JSD and Disagreement, are two readings of one question -- how far two
    models' outputs agree -- so ranking both gives `Output Corr.` two cases per
    cell where every other test has one. ``RANKED_FUNCTIONAL_MEASURES`` reports
    that task on JSD, as ReSi's own overview table does, and `Acc Corr.` on
    AbsoluteAccDiff, which is a different task rather than a third reading.

    This does not merge or drop a task. Six remain, and `Acc Corr.` is reported
    over whatever domains and architectures have AbsoluteAccDiff values -- vision
    throughout, language's BERT-L but not SmolLM2, graphs not at all -- rather
    than being padded or harmonized away.

    Rows with no functional similarity measure are every test outside the
    correlation family and are always kept. Like ``select_ranked_quality`` this
    is applied at ranking time, so the appendix tables still show Disagreement
    beside JSD.
    """
    if values.empty or "functional_similarity_measure" not in values.columns:
        return values
    functional = values["functional_similarity_measure"].fillna("").astype(str)
    keep = functional.eq("") | functional.isin(set(functional_measures))
    return values[keep].reset_index(drop=True)


MEASURE_EXCLUSION_COLUMNS = [
    "reason",
    "metric",
    "scope",
    "case_id",
    "quality_measure",
    "nan_fraction",
    "cases_present",
    "cases_in_scope",
    "case_coverage",
    "rows_dropped",
]

CASE_NAN_REASON = "nan_fraction_exceeded"
SCOPE_COVERAGE_REASON = "case_coverage_below_threshold"


def filter_measures_by_coverage(
    values: pd.DataFrame,
    *,
    max_nan_fraction: float,
    min_case_coverage: float,
    scope_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop measures too incomplete to report, first per case then per scope.

    Two different failures need two thresholds, and both are about a measure
    being scored on an easier field than its competitors faced -- which is the
    one thing ranking cannot see, since it only ever compares what is present.

    Within a case, a measure whose NaN share exceeds ``max_nan_fraction`` is
    removed from that case outright. Ranking skips a measure on the observations
    where it failed, so a mostly-failed measure is ranked on the handful of
    observations it happened to survive, and those are not a random sample of
    the case: a measure that degrades on hard inputs is scored only on the easy
    ones. Its remaining ranks are a flattering subset, not a partial result.

    Across a scope, a measure that survived fewer than ``min_case_coverage`` of
    the scope's cases is removed from the scope entirely. Ranking is only within
    a case, so a sparse measure is never scored badly for being absent -- its
    pooled mean is simply drawn from the cases it happened to survive, which is
    a materially easier field than its competitors faced.

    ``scope_columns`` is the grouping of the output being produced, so the same
    rule serves the domain summary (``["domain"]``), the per-task outputs
    (``["benchmark", "dataset"]``), and per-architecture tables.

    Returns the surviving values and a frame recording every exclusion. The
    exclusion frame's columns do not depend on ``scope_columns``; the scope is
    recorded as a joined string so one reader handles every scope.
    """
    empty_exclusions = pd.DataFrame(columns=MEASURE_EXCLUSION_COLUMNS)
    if values.empty:
        return values, empty_exclusions
    scope_names = list(scope_columns)
    frame = values.copy()
    frame["_case_id"] = frame[CASE_COLUMNS].astype(str).agg("|".join, axis=1)
    frame["_scope"] = frame[scope_names].astype(str).agg("|".join, axis=1)

    group_keys = ["_scope", "_case_id", "quality_measure", "metric"]
    per_case = (
        frame.assign(_nan=~np.isfinite(frame["value"]))
        .groupby(group_keys, dropna=False, sort=True)
        .agg(rows=("value", "size"), nan_rows=("_nan", "sum"))
    )
    per_case["nan_fraction"] = per_case["nan_rows"] / per_case["rows"]
    over_threshold = per_case[per_case["nan_fraction"] > max_nan_fraction]
    dropped = pd.MultiIndex.from_frame(frame[group_keys]).isin(over_threshold.index)
    kept = frame.loc[~dropped]

    exclusions = [
        pd.DataFrame(
            {
                "reason": CASE_NAN_REASON,
                "metric": over_threshold.index.get_level_values("metric"),
                "scope": over_threshold.index.get_level_values("_scope"),
                "case_id": over_threshold.index.get_level_values("_case_id"),
                "quality_measure": over_threshold.index.get_level_values(
                    "quality_measure"
                ),
                "nan_fraction": over_threshold["nan_fraction"].to_numpy(),
                "rows_dropped": over_threshold["rows"].to_numpy(),
            }
        )
    ]

    # Cases counted over the surviving frame, not the unfiltered one. A case
    # the per-case filter emptied for every measure carries no ranking
    # information for anyone, so counting it would penalize each measure for a
    # case none of them could have reported -- which excludes the whole field
    # at once. A case that stayed alive for some measure and not others is
    # still counted, since that is the real sparseness this threshold is for.
    cases_in_scope = kept.groupby("_scope")["_case_id"].nunique()
    # Index over every (scope, measure) the scope started with, so a measure the
    # case filter wiped out entirely is reported at coverage 0 rather than
    # vanishing from the coverage table along with its rows.
    attempted = pd.MultiIndex.from_frame(
        frame[["_scope", "metric"]].drop_duplicates()
    ).sort_values()
    if kept.empty:
        present = pd.Series(0, index=attempted, dtype=int)
    else:
        present = (
            kept.groupby(["_scope", "metric"])["_case_id"]
            .nunique()
            .reindex(attempted, fill_value=0)
        )
    coverage = pd.DataFrame(
        {
            "cases_present": present,
            "cases_in_scope": [
                int(cases_in_scope.get(scope, 0)) for scope, _ in present.index
            ],
        }
    )
    coverage["case_coverage"] = coverage["cases_present"] / coverage[
        "cases_in_scope"
    ].replace(0, np.nan)
    sparse = coverage[coverage["case_coverage"] < min_case_coverage]
    if not sparse.empty:
        sparse_pairs = pd.MultiIndex.from_frame(kept[["_scope", "metric"]])
        removed = kept.loc[sparse_pairs.isin(sparse.index)]
        kept = kept.loc[~sparse_pairs.isin(sparse.index)]
        exclusions.append(
            pd.DataFrame(
                {
                    "reason": SCOPE_COVERAGE_REASON,
                    "metric": sparse.index.get_level_values("metric"),
                    "scope": sparse.index.get_level_values("_scope"),
                    "cases_present": sparse["cases_present"].to_numpy(),
                    "cases_in_scope": sparse["cases_in_scope"].to_numpy(),
                    "case_coverage": sparse["case_coverage"].to_numpy(),
                    "rows_dropped": [
                        int(
                            (
                                removed["_scope"].eq(scope) & removed["metric"].eq(name)
                            ).sum()
                        )
                        for scope, name in sparse.index
                    ],
                }
            )
        )

    report = pd.concat(exclusions, ignore_index=True, sort=False)
    report = report.reindex(columns=MEASURE_EXCLUSION_COLUMNS)
    return (
        kept.drop(columns=["_case_id", "_scope"]).reset_index(drop=True),
        report.reset_index(drop=True),
    )


def rank_case_observations(
    values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank measures within each case, returning observation and case ranks.

    Each observation is ranked over the measures that produced a finite value
    for it. A measure missing from an observation is skipped there and keeps its
    ranks from the observations it did produce -- ReSi's ``na_option="keep"``
    convention. The alternative, dropping the whole observation row unless every
    compared measure is present, made one measure's failure cost every other
    measure that observation as well.

    An observation ranking fewer than two measures is skipped: a field of one
    awards rank 1 to whichever measure happened to survive, which reads as a win
    and is not one. Cases with fewer than two measures contribute nothing.

    Averaging observation ranks into one number per case is what gives each
    case -- one task, dataset, architecture, and functional similarity measure
    -- a single vote in everything downstream, regardless of how many seeds or
    layer models it happens to contain.

    Both frames carry the raw rank and its ``normalize_case_rank`` rescaling.
    Within one domain the reported field is fixed and the raw rank is directly
    readable; the normalized one is what survives pooling across domains that
    report different field sizes.
    """
    rows: list[pd.DataFrame] = []
    case_rows: list[dict[str, Any]] = []
    # Drop individual measurements that failed outright (NaN/Inf). This removes
    # only that one (observation, metric) cell; the rest of the observation is
    # ranked over whatever remains.
    finite = values[np.isfinite(values["value"])].copy()
    for case_key, group in finite.groupby(CASE_COLUMNS, dropna=False, sort=True):
        pivot = group.pivot_table(
            index="observation_id", columns="metric", values="value", aggfunc="mean"
        )
        metric_count = len(pivot.columns)
        # `rank` keeps NaN as NaN by default, so a missing cell neither scores
        # nor displaces anything -- the remaining measures rank 1..k over
        # themselves alone.
        ranked_per_observation = pivot.notna().sum(axis=1)
        pivot = pivot[ranked_per_observation >= 2]
        if pivot.empty or metric_count < 2:
            continue
        direction = quality_direction(str(case_key[4]))
        ranks = pivot.rank(axis=1, method="average", ascending=direction != "higher")
        # The denominator is per observation, not per case: an observation that
        # ranked 14 of the case's 16 measures spans 1..14, and rescaling it
        # against 16 would place its last-placed measure short of 1.0.
        available = ranks.notna().sum(axis=1)
        normalized = normalize_case_rank(ranks, available.to_numpy()[:, None])

        long = pd.concat(
            [
                ranks.stack().rename("rank"),
                normalized.stack().rename("normalized_rank"),
            ],
            axis=1,
        ).reset_index()
        long["n_ranked_metrics"] = long["observation_id"].map(available)
        for column, value in zip(CASE_COLUMNS, case_key):
            long[column] = value
        long["n_compared_metrics"] = metric_count
        long["n_case_observations"] = len(pivot)
        long["case_id"] = long[CASE_COLUMNS].astype(str).agg("|".join, axis=1)
        rows.append(long)

        mean_rank = ranks.mean(axis=0)
        mean_normalized = normalized.mean(axis=0)
        ranked_observations = ranks.notna().sum(axis=0)
        for metric in ranks.columns:
            # A measure whose only observations were the skipped ones -- those
            # that ranked a field of one -- was never ranked in this case, so it
            # has no mean to report and must not enter as NaN.
            if not ranked_observations[metric]:
                continue
            case_rows.append(
                {
                    **dict(zip(CASE_COLUMNS, case_key)),
                    "case_id": "|".join(map(str, case_key)),
                    "metric": metric,
                    "mean_rank": float(mean_rank[metric]),
                    "mean_normalized_rank": float(mean_normalized[metric]),
                    # How many of the case's observations this measure was
                    # actually ranked in, against how many the case has. Equal
                    # is full coverage; below it says the mean came from a
                    # subset, which `filter_measures_by_coverage` bounds.
                    "n_ranked_observations": int(ranked_observations[metric]),
                    "n_case_observations": len(pivot),
                    "n_compared_metrics": metric_count,
                }
            )
    per_observation = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return per_observation, pd.DataFrame(case_rows)


def summarize_domains(
    case_ranks: pd.DataFrame, *, rank_scale: str | None = None
) -> pd.DataFrame:
    """Aggregate case ranks per domain and measure, best first.

    Both the normalized and raw statistics are always computed, so the frame's
    columns do not depend on ``rank_scale``; only the sort order does. The
    default raw scale is directly readable within a domain, where the reported
    field is fixed. ``normalized`` is the one to use if the pooled cases did not
    all rank the same field, which is the situation the cross-domain layer is
    always in.
    """
    sort_column = rank_scale_case_column(rank_scale)
    if case_ranks.empty:
        return pd.DataFrame(columns=DOMAIN_SUMMARY_COLUMNS)
    summary = case_ranks.groupby(["domain", "metric"], as_index=False).agg(
        mean_normalized_rank=("mean_normalized_rank", "mean"),
        median_normalized_rank=("mean_normalized_rank", "median"),
        std_normalized_rank=("mean_normalized_rank", "std"),
        mean_rank=("mean_rank", "mean"),
        median_rank=("mean_rank", "median"),
        std_rank=("mean_rank", "std"),
        n_cases=("mean_rank", "count"),
    )
    return summary[DOMAIN_SUMMARY_COLUMNS].sort_values(
        ["domain", sort_column, "metric"]
    )


__all__ = [
    "CASE_COLUMNS",
    "CASE_NAN_REASON",
    "MEASURE_EXCLUSION_COLUMNS",
    "SCOPE_COVERAGE_REASON",
    "DEFAULT_RANK_SCALE",
    "DOMAIN_SUMMARY_COLUMNS",
    "RANKED_FUNCTIONAL_MEASURES",
    "RANKED_QUALITY_MEASURES",
    "RANK_SCALES",
    "filter_measures_by_coverage",
    "normalize_case_rank",
    "rank_scale_case_column",
    "select_ranked_functional",
    "select_ranked_quality",
    "validate_rank_scale",
    "rank_case_observations",
    "summarize_domains",
]
