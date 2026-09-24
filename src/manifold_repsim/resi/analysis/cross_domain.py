"""Compare measures across domains within one benchmark family.

The per-domain outputs answer "which measure won this domain". They cannot
answer "is this measure good at detecting shortcuts, wherever shortcuts are
tested", because each domain names the same experiment differently and writes
it to its own directory.

Two facts make the cross-domain view possible, and both are load-bearing:

* The three domains run the same five experiments under different benchmark
  ids. `TEST_GROUPS` below is that equivalence, written out explicitly. Those
  five become six families, because the prediction-correlation experiment
  answers two different questions and `test_group_for` splits it accordingly.
* Within one raw quality measure the numbers mean the same thing everywhere.
  A Pearson correlation is a Pearson correlation in graphs, vision, and text,
  so these panels plot values rather than ranks -- no rank indirection, and no
  need to pool measures onto a common ordinal scale.

Faceting is on the raw `quality_measure`, never on the display label. The
display layer maps both `spearmanr` and `correlation` to "Spearman", but they
are different quantities: `correlation` scores layer monotonicity while
`spearmanr` scores prediction correlation. Grouping by label would pool them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import rank_scale_case_column

# The same experiment, named once per domain. Graph ids carry a `_test`
# suffix; vision and language use bare names and disagree with each other on
# the label and prediction-correlation experiments.
#
# The prediction-correlation benchmark is the one id that does not name a
# single family: one run scores JSD, Disagreement, and AbsoluteAccDiff, and the
# first two ask how far two models' outputs agree while the third asks how far
# apart their accuracies are. `test_group_for` splits it on the functional
# measure; the value here is the family the output measures belong to.
TEST_GROUPS = {
    "accoutput": "Output Corr.",
    "correlation": "Output Corr.",
    "output_correlation_test": "Output Corr.",
    "augmentation": "Augmentation",
    "augmentation_test": "Augmentation",
    "monotonicity": "Layer Mono.",
    "layer_test": "Layer Mono.",
    "randomlabel": "Random Labels",
    "memorization": "Random Labels",
    "label_test": "Random Labels",
    "shortcut": "Shortcuts",
    "shortcut_test": "Shortcuts",
}

# The functional measure that belongs to its own family rather than to the
# benchmark's default one.
ACCURACY_FUNCTIONAL = "AbsoluteAccDiff"
ACCURACY_TEST_GROUP = "Acc Corr."

TEST_GROUP_ORDER = [
    "Output Corr.",
    "Acc Corr.",
    "Random Labels",
    "Shortcuts",
    "Augmentation",
    "Layer Mono.",
]


def test_group_for(
    benchmark: pd.Series, functional: pd.Series | None = None
) -> pd.Series:
    """Family for each row, splitting prediction correlation into two.

    The benchmark id alone cannot name the family for a correlation run, since
    one run produces rows for every functional similarity measure. Rows scored
    against `AbsoluteAccDiff` become their own family; everything else keeps
    the benchmark's mapping. Benchmarks outside `TEST_GROUPS` stay NaN so the
    caller drops them rather than guessing.
    """
    groups = benchmark.astype(str).map(TEST_GROUPS)
    if functional is None:
        return groups
    accuracy = functional.fillna("").astype(str).eq(ACCURACY_FUNCTIONAL)
    return groups.where(~(accuracy & groups.notna()), ACCURACY_TEST_GROUP)


# Higher is better in every panel, so the violation rate is reported as its
# complement, matching the recreated appendix tables.
INVERTED_QUALITY = "violation_rate"

QUALITY_LABELS = {
    "AUPRC": "AUPRC",
    "violation_rate": "Conformity Rate",
    "correlation": "Spearman (monotonicity)",
    "spearmanr": "Spearman",
    "pearsonr": "Pearson",
    "kendalltau": "Kendall tau",
}

QUALITY_ORDER = [
    "AUPRC",
    "violation_rate",
    "correlation",
    "pearsonr",
    "spearmanr",
    "kendalltau",
]

DOMAIN_LABELS = {"graphs": "Graph", "vision": "Vision", "language": "Text"}

# ReSi computes the prediction-correlation experiment three ways but reports
# only Spearman: both published notebooks (`tables_and_plots.ipynb` cell 7 and
# `appendix_tables.ipynb` cell 7) drop `pearsonr` and `kendalltau` before
# ranking anything. Keeping them would not merely add two panels. A case is a
# ranking cell keyed on the quality measure, so three correlations means three
# cases where the paper has one, and prediction correlation would carry three
# times its published weight in every pooled rank.
UNPUBLISHED_QUALITY = frozenset({"pearsonr", "kendalltau"})


def drop_unpublished_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the quality measures ReSi computes but does not report.

    The per-domain analysis configs narrow `quality_measures` to the same set,
    so on a freshly analyzed run this removes nothing. It is applied here too
    because this layer reads whatever per-domain artifacts are on disk, and
    those predate the narrowing.
    """
    if frame.empty or "quality_measure" not in frame.columns:
        return frame
    keep = ~frame["quality_measure"].astype(str).isin(UNPUBLISHED_QUALITY)
    return frame[keep].reset_index(drop=True)


# AUPRC is on a [0, 1] scale everywhere but its chance level depends on how
# many groups a benchmark separates, and that differs by domain. The value is
# comparable in scale, not in achievement, so anything reporting it must say so
# rather than let a reader infer that 0.6 means the same thing in each panel.
UNCALIBRATED_QUALITY = {"AUPRC"}

COMBINED_COLUMNS = [
    "domain",
    "test_group",
    "benchmark",
    "quality_measure",
    "metric",
    "value",
]


def functional_measure_coverage(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Report which functional similarity measures each domain reports.

    Correlation tests score every functional similarity measure separately, and
    the domains do not all report the same ones -- graphs excludes
    `AbsoluteAccDiff` because ReSi's graph archive never scored it for its own
    measures. The family split keeps that from silently averaging a two-measure
    domain against a three-measure one, but which domains hold which measure is
    still worth reporting: it is what tells a reader that `Acc Corr.` is a
    two-domain family rather than a three-domain one.
    """
    rows: list[dict[str, object]] = []
    for domain, frame in frames.items():
        if frame.empty or "functional_similarity_measure" not in frame.columns:
            continue
        names = frame["functional_similarity_measure"].fillna("").astype(str)
        for name in sorted(set(names[names.ne("")])):
            rows.append({"domain": domain, "functional_similarity_measure": name})
    return pd.DataFrame(rows, columns=["domain", "functional_similarity_measure"])


def common_functional_measures(frames: dict[str, pd.DataFrame]) -> set[str]:
    """Functional similarity measures every reporting domain has in common."""
    coverage = functional_measure_coverage(frames)
    if coverage.empty:
        return set()
    per_domain = [
        set(group["functional_similarity_measure"])
        for _, group in coverage.groupby("domain")
    ]
    return set.intersection(*per_domain) if per_domain else set()


def combine_domain_values(
    frames: dict[str, pd.DataFrame],
    *,
    harmonize_functional: bool = False,
    published_quality_only: bool = True,
) -> pd.DataFrame:
    """Stack per-domain normalized results into one cross-domain frame.

    `frames` maps a domain name to that domain's `normalized_values.csv`
    contents. Benchmarks outside `TEST_GROUPS` are dropped rather than guessed
    at, so a new experiment shows up as absent instead of silently joining the
    wrong family.

    `harmonize_functional` restricts correlation rows to the functional
    similarity measures every domain reports. It defaults to False because the
    family split above already does the job it was introduced for: a panel can
    no longer pool one domain's two output measures against another's two
    output measures *plus* an accuracy one, since the accuracy rows are their
    own family. What remains is a genuine coverage difference between domains
    within a family, which is reported rather than silently equalized -- graphs
    contributes nothing to `Acc Corr.`, and that is a fact about the archive,
    not a reason to delete the other domains' data. Set it True to restore the
    old intersection.

    `published_quality_only` drops `pearsonr` and `kendalltau`, which ReSi
    computes but does not report; see `drop_unpublished_quality`.
    """
    keep_functional = (
        common_functional_measures(frames) if harmonize_functional else None
    )
    collected: list[pd.DataFrame] = []
    for domain, frame in frames.items():
        if frame.empty:
            continue
        subset = frame.copy()
        subset["domain"] = domain
        subset["test_group"] = test_group_for(
            subset["benchmark"],
            (
                subset["functional_similarity_measure"]
                if "functional_similarity_measure" in subset
                else None
            ),
        )
        subset = subset[subset["test_group"].notna()]
        if published_quality_only:
            subset = drop_unpublished_quality(subset)
        if keep_functional is not None and "functional_similarity_measure" in subset:
            functional = subset["functional_similarity_measure"].fillna("").astype(str)
            subset = subset[functional.eq("") | functional.isin(keep_functional)]
        if subset.empty:
            continue
        inverted = subset["quality_measure"].astype(str).eq(INVERTED_QUALITY)
        subset.loc[inverted, "value"] = 1 - subset.loc[inverted, "value"]
        collected.append(subset[COMBINED_COLUMNS])
    if not collected:
        return pd.DataFrame(columns=COMBINED_COLUMNS)
    return pd.concat(collected, ignore_index=True)


def _ordered(present: set[str], order: list[str]) -> list[str]:
    """Known names in their declared order, then anything unexpected, sorted."""
    known = [name for name in order if name in present]
    return known + sorted(present - set(known))


def summarize_cross_domain(values: pd.DataFrame) -> pd.DataFrame:
    """Median and spread per measure, within each test group and quality measure.

    One row per (test group, quality measure, domain, measure), so a reader can
    check a panel's numbers without reading them off the figure.
    """
    columns = [
        "test_group",
        "quality_measure",
        "domain",
        "metric",
        "n",
        "median",
        "mean",
        "std",
    ]
    if values.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        values.groupby(
            ["test_group", "quality_measure", "domain", "metric"], observed=True
        )["value"]
        .agg(["count", "median", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    summary["test_group"] = pd.Categorical(
        summary["test_group"], categories=TEST_GROUP_ORDER, ordered=True
    )
    return summary.sort_values(
        ["test_group", "quality_measure", "domain", "median"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)[columns]


# Both per-case rank statistics are carried through, exactly as the per-domain
# layer writes both, so the choice of scale never changes what these outputs
# contain -- only which column the figures draw and the aggregates rank on.
#
# `mean_rank` is the reported default, matching the per-domain figures and
# ReSi. The one caveat it carries is real and belongs in any caption: vision
# reports 15 measures where graphs and language report 16, so a raw rank is
# drawn from a field one measure smaller there. Mid-pack is 8.0 in vision
# against 8.5 elsewhere, which flatters a measure present in vision by about
# half a rank. `mean_normalized_rank` rescales each case to `(rank - 1)/(n - 1)`
# and removes exactly that, at the cost of an axis no longer readable as a
# placing. Switch with `rank_scale`.
RANK_COLUMNS = ("mean_rank", "mean_normalized_rank")
RANK_COLUMN = "mean_rank"

RANK_AXIS_LABELS = {
    "mean_rank": "mean rank per subtask (1 = best)",
    "mean_normalized_rank": "mean normalized rank per subtask (0 = best)",
}


def rank_column(rank_scale: str | None = None) -> str:
    """The per-case column this layer pools, from the selected rank scale."""
    return rank_scale_case_column(rank_scale)


def rank_axis_label(rank_scale: str | None = None) -> str:
    """The axis label naming what a cross-domain rank figure draws."""
    return RANK_AXIS_LABELS[rank_column(rank_scale)]


ALL_TASKS = "All tasks"

CASE_RANK_COLUMNS = [
    "domain",
    "test_group",
    "benchmark",
    "dataset",
    "quality_measure",
    "functional_similarity_measure",
    "case_id",
    "metric",
]

# One subtask is one comparable slice of one experiment, with the architecture
# dimension collapsed. Ranks are averaged within it before anything pools them,
# because a case count tracks how many architectures a domain happens to ship
# rather than how much evidence it carries: vision runs 7 architectures against
# graphs' 3-4 and language's 1-2, which is most of why vision held 180 of 266
# cases. Averaging first gives every subtask one vote.
#
# Architecture, identifier, and representation_dataset are averaged over.
# Quality measure is not: an AUPRC rank and a conformity-rate rank answer
# different questions, and their mean is not a quantity. The functional
# similarity measure is not either, for the same reason.
SUBTASK_KEYS = [
    "domain",
    "dataset",
    "test_group",
    "quality_measure",
    "functional_similarity_measure",
]

SUBTASK_COLUMNS = SUBTASK_KEYS + [
    "benchmark",
    "subtask_id",
    "metric",
    *RANK_COLUMNS,
    "n_cases",
]


def combine_case_ranks(
    frames: dict[str, pd.DataFrame],
    *,
    harmonize_functional: bool = False,
    published_quality_only: bool = True,
) -> pd.DataFrame:
    """Stack per-domain `per_case_ranks.csv` frames, grouped into families.

    `harmonize_functional` drops cases scored against a functional similarity
    measure some domain does not report. It defaults to False now that
    `AbsoluteAccDiff` forms its own family: the reweighting it guarded against
    was vision's 42 `AbsoluteAccDiff` cases carrying vision from 46% to 56% of
    a *single* prediction-correlation aggregate that graphs could only
    contribute output measures to. Those cases now land in `Acc Corr.`
    instead, where they reweight nothing, because the family is exactly the
    domains that report it. `Output Corr.` is left with the two measures every
    domain has, which is the balance the intersection was buying. Set it True
    to restore the old behaviour.

    `published_quality_only` drops the `pearsonr` and `kendalltau` cases,
    which does still matter here: they are two thirds of every
    prediction-correlation case in every domain.
    """
    keep_functional = (
        common_functional_measures(frames) if harmonize_functional else None
    )
    collected: list[pd.DataFrame] = []
    for domain, frame in frames.items():
        if frame.empty:
            continue
        subset = frame.copy()
        subset["domain"] = domain
        subset["test_group"] = test_group_for(
            subset["benchmark"],
            (
                subset["functional_similarity_measure"]
                if "functional_similarity_measure" in subset
                else None
            ),
        )
        subset = subset[subset["test_group"].notna()]
        if published_quality_only:
            subset = drop_unpublished_quality(subset)
        if keep_functional is not None and "functional_similarity_measure" in subset:
            functional = subset["functional_similarity_measure"].fillna("").astype(str)
            subset = subset[functional.eq("") | functional.isin(keep_functional)]
        if subset.empty:
            continue
        # `per_case_ranks.csv` always carries these; a caller assembling a
        # frame by hand may not, and an absent dimension is one that does not
        # split anything rather than an error.
        for column in CASE_RANK_COLUMNS:
            if column not in subset.columns:
                subset[column] = ""
        # Both scales are carried, so a caller can switch `rank_scale`
        # without this layer having to be re-run.
        for column in RANK_COLUMNS:
            if column not in subset.columns:
                subset[column] = float("nan")
        collected.append(subset[CASE_RANK_COLUMNS + list(RANK_COLUMNS)])
    if not collected:
        return pd.DataFrame(columns=CASE_RANK_COLUMNS + list(RANK_COLUMNS))
    return pd.concat(collected, ignore_index=True)


def average_subtask_ranks(ranks: pd.DataFrame) -> pd.DataFrame:
    """Collapse each subtask's cases into one mean normalized rank per measure.

    `n_cases` records how many cases went into each mean. It is not a filter:
    coverage filtering already ran per domain, and a measure that survives it
    but is still absent from part of a subtask should be visible as a smaller
    `n_cases` rather than silently averaged over an easier subset.

    `benchmark` is carried through unaggregated because it is determined by
    (domain, test_group) -- each domain names a family exactly once -- so it
    adds a readable label without splitting anything.
    """
    if ranks.empty:
        return pd.DataFrame(columns=SUBTASK_COLUMNS)
    frame = ranks.copy()
    for column in SUBTASK_KEYS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    grouped = (
        frame.groupby(SUBTASK_KEYS + ["metric"], dropna=False, observed=True)
        .agg(
            **{
                **{column: (column, "mean") for column in RANK_COLUMNS},
                "n_cases": ("case_id", "nunique"),
                "benchmark": ("benchmark", "first"),
            }
        )
        .reset_index()
    )
    grouped["subtask_id"] = grouped[SUBTASK_KEYS].astype(str).agg("|".join, axis=1)
    return grouped[SUBTASK_COLUMNS]


def subtask_balance(subtasks: pd.DataFrame) -> pd.DataFrame:
    """Units per (family, domain) -- the table that says whether this worked.

    A cross-domain figure is only as balanced as this table is, so it is
    reported rather than assumed.
    """
    if subtasks.empty:
        return pd.DataFrame()
    units = subtasks.drop_duplicates("subtask_id")
    table = (
        units.groupby(["test_group", "domain"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    table["TOTAL"] = table.sum(axis=1)
    return table


def aggregate_across_domains(
    ranks: pd.DataFrame, *, rank_scale: str | None = None
) -> pd.DataFrame:
    """Aggregate ranks over domains, two ways, per family and overall.

    `rank_scale` selects which per-case statistic is pooled; both are carried
    on the input either way, so switching it never changes what was measured.

    Both weightings are reported because they answer different questions and
    can disagree:

    * `pooled_mean` weights every subtask equally, so a domain running more
      datasets carries proportionally more weight -- graphs contributes 30 of
      70 subtasks because it has three datasets where the others have two.
    * `balanced_mean` averages the per-domain means, giving each domain one
      vote regardless of how many subtasks it ran.

    Both operate on `average_subtask_ranks` output, not on raw cases. `n_cases`
    is retained as the number of underlying cases behind a measure's subtasks,
    so a reader can trace a point back to the evaluation cells it came from.

    `n_domains` is reported alongside because a measure missing from a domain
    is averaged over fewer of them -- LinearRegression is absent from vision,
    so its balanced mean is a two-domain average against everyone else's three.
    """
    columns = [
        "test_group",
        "metric",
        "n_subtasks",
        "n_cases",
        "n_domains",
        "pooled_mean",
        "pooled_median",
        "balanced_mean",
    ]
    if ranks.empty:
        return pd.DataFrame(columns=columns)
    column = rank_column(rank_scale)

    def _summarize(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        per_domain = frame.groupby(["metric", "domain"], observed=True)[column].mean()
        balanced = per_domain.groupby("metric").mean()
        domains = per_domain.groupby("metric").size()
        pooled = frame.groupby("metric", observed=True)[column].agg(
            ["count", "mean", "median"]
        )
        cases = (
            frame.groupby("metric", observed=True)["n_cases"].sum()
            if "n_cases" in frame.columns
            else pooled["count"]
        )
        out = pd.DataFrame(
            {
                "test_group": label,
                "metric": pooled.index,
                "n_subtasks": pooled["count"].to_numpy(),
                "n_cases": cases.reindex(pooled.index).to_numpy(),
                "n_domains": domains.reindex(pooled.index).to_numpy(),
                "pooled_mean": pooled["mean"].to_numpy(),
                "pooled_median": pooled["median"].to_numpy(),
                "balanced_mean": balanced.reindex(pooled.index).to_numpy(),
            }
        )
        return out.sort_values("balanced_mean").reset_index(drop=True)

    parts = [_summarize(ranks, ALL_TASKS)]
    for group in _ordered(
        {str(n) for n in ranks["test_group"].unique()}, TEST_GROUP_ORDER
    ):
        parts.append(_summarize(ranks[ranks["test_group"].eq(group)], group))
    return pd.concat(parts, ignore_index=True)[columns]


__all__ = [
    "ALL_TASKS",
    "CASE_RANK_COLUMNS",
    "COMBINED_COLUMNS",
    "DOMAIN_LABELS",
    "INVERTED_QUALITY",
    "QUALITY_LABELS",
    "QUALITY_ORDER",
    "ACCURACY_TEST_GROUP",
    "TEST_GROUPS",
    "test_group_for",
    "TEST_GROUP_ORDER",
    "UNCALIBRATED_QUALITY",
    "UNPUBLISHED_QUALITY",
    "RANK_COLUMN",
    "RANK_COLUMNS",
    "rank_axis_label",
    "rank_column",
    "SUBTASK_COLUMNS",
    "SUBTASK_KEYS",
    "aggregate_across_domains",
    "average_subtask_ranks",
    "combine_case_ranks",
    "combine_domain_values",
    "common_functional_measures",
    "drop_unpublished_quality",
    "functional_measure_coverage",
    "subtask_balance",
    "summarize_cross_domain",
]
