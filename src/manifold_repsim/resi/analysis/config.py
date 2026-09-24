from __future__ import annotations

from manifold_repsim.resi.measures import (
    RESI_MEASURE_SPECS,
    manifold_measure_categories,
)

# Which per-case statistic drives rank ordering and the plotted axis. Both are
# always computed and written, so switching scales never changes an output's
# columns. Kept in this module, rather than beside the ranking code that uses
# it, so `settings` can validate a YAML value without importing pandas.
#
# `raw` is the default and the reported scale: within one domain every case
# ranks the same field, because the reported measure set is fixed by the
# analysis config and coverage filtering removes anything that is not present
# throughout. A raw rank is then directly readable ("3rd of 16") and matches
# how ReSi's own published figures and tables rank.
#
# `normalized` rescales to `(rank - 1) / (n - 1)`. It is what pooling across
# domains needs, since the domains report different field sizes, which is why
# the cross-domain layer uses it regardless of this setting. Within a domain it
# only matters if the pooled cases genuinely ranked different fields.
RANK_SCALES = ("normalized", "raw")
DEFAULT_RANK_SCALE = "raw"
RANK_SCALE_CASE_COLUMNS = {
    "normalized": "mean_normalized_rank",
    "raw": "mean_rank",
}

# The quality measures ReSi ranks on. Its rankplot notebook keeps exactly these
# three -- `data["Quality Measure"].isin(["AUPRC", "spearmanr", "correlation"])`
# -- and drops everything else before ranking anything, and its overview table
# reaches the same set benchmark by benchmark.
#
# Two things are excluded by it. `pearsonr` and `kendalltau` are computed by
# the prediction-correlation experiment and never reported. `violation_rate` is
# reported, but as a second reading of a test the AUPRC already scores: leaving
# it in makes every design-grounded test contribute two cases where correlation
# contributes one, so the design tests carry twice their published weight in
# any aggregate that pools cases. The values are untouched -- conformity rate
# still appears in the tables, the heatmaps, and the AUPRC scatter -- this only
# decides what gets ranked.
RANKED_QUALITY_MEASURES = ("AUPRC", "spearmanr", "correlation")

# One functional similarity measure per correlation task, matching ReSi's
# reported tables: `Output Corr.` is reported on JSD, `Acc Corr.` on
# AbsoluteAccDiff.
#
# `Disagreement` is the second of two readings of the same question -- how far
# two models' outputs agree -- so ranking it alongside JSD gives output
# correlation two cases per cell where every other test has one. That is the
# same double-counting `RANKED_QUALITY_MEASURES` removes for `violation_rate`,
# arriving by a different route. ReSi's overview table applies exactly this
# filter; its own rank figure does not, and the published-mode figure follows
# ReSi there rather than following us.
#
# These two name two of the six tasks, not two readings of one: `Output Corr.`
# asks how far two models' outputs agree, `Acc Corr.` how far their accuracies
# are apart. Neither substitutes for the other, so there is no fallback between
# them. Where a domain or architecture has no AbsoluteAccDiff values at all --
# ReSi's graph archive, and SmolLM2 in language -- `Acc Corr.` is reported over
# whatever does have them rather than dropped or padded (docs 2.1a, 7.2).
#
# A blank functional measure -- every design-grounded test and layer
# monotonicity -- is always kept; this only narrows the correlation family.
RANKED_FUNCTIONAL_MEASURES = ("JSD", "AbsoluteAccDiff")

# Which per-task output directory each functional similarity measure belongs
# to. The prediction-correlation benchmark runs two of the six tasks at once,
# so keying its per-task outputs on the benchmark id pooled them into one
# ranking and one boxplot labelled with the benchmark -- which is the thing the
# task split exists to prevent.
#
# Membership, not the ranked choice: `Disagreement` lands in `outcorr` beside
# `JSD` because it is a reading of that task, and shows up in its values, its
# heatmaps and its appendix table. Only the ranking inside that directory
# narrows to `RANKED_FUNCTIONAL_MEASURES`.
#
# The two names are shared across every domain rather than derived from the
# benchmark, because each domain names that benchmark differently -- vision
# `accoutput`, language `correlation`, graphs `output_correlation_test` -- and
# the same task should be at the same path in all three. Every other benchmark
# keeps its own id, so only these paths change.
#
# `tests/test_resi_analysis.py` pins this against `tables.taxonomy`'s
# `FUNCTIONAL_TEST_LABELS`, which groups the same measures for the tables;
# `taxonomy` imports from this module, so the mapping cannot live there.
CORRELATION_TASK_DIRS = {
    "JSD": "outcorr",
    "Disagreement": "outcorr",
    "AbsoluteAccDiff": "acccorr",
}


def validate_rank_scale(rank_scale: str | None) -> str:
    """Return a supported rank scale name, defaulting when None is given."""
    if rank_scale is None:
        return DEFAULT_RANK_SCALE
    name = str(rank_scale).strip()
    if name not in RANK_SCALES:
        raise ValueError(
            f"rank_scale must be one of {', '.join(RANK_SCALES)}, got {rank_scale!r}"
        )
    return name


def rank_scale_case_column(rank_scale: str | None) -> str:
    """Return the per-case column a rank scale ranks and plots on."""
    return RANK_SCALE_CASE_COLUMNS[validate_rank_scale(rank_scale)]


# Which per-measure number the boxplots sort on. `median` orders by typical
# standing. `mean` orders by the average standing instead, which a handful of
# extreme cases can pull away from the median -- the two agree on a symmetric
# distribution and disagree exactly when that is worth seeing. `quantile90`
# orders by the unfavourable tail instead, which is a robustness statement: a
# measure whose 90th-percentile rank is low never places badly, while one with
# a good median and a long tail is only usually good. All three are always
# computable from the same frame, so switching never changes what was measured
# -- only which number the rows are sorted on. The mean is also always drawn
# on every box regardless of sort (see `boxplots.mark_mean`), so `mean` needs
# no marker of its own the way `quantile90` does.
RANK_SORTS = ("median", "quantile90", "mean")
DEFAULT_RANK_SORT = "median"
ROBUSTNESS_QUANTILE = 0.9


def validate_rank_sort(rank_sort: str | None) -> str:
    """Return a supported boxplot sort statistic, defaulting when None."""
    if rank_sort is None:
        return DEFAULT_RANK_SORT
    name = str(rank_sort).strip()
    if name not in RANK_SORTS:
        raise ValueError(
            f"rank_sort must be one of {', '.join(RANK_SORTS)}, got {rank_sort!r}"
        )
    return name


# How a measure's distribution is drawn, independent of what it is sorted on.
# `box` is a quartile box with percentile whiskers. `boxen` is a letter-value
# plot: nested bands at the 10th, 30th, 50th, 70th and 90th percentiles, drawn
# narrower as they reach further into the tails, with whiskers carrying on to
# the observed minimum and maximum. The box shows one interval and two whisker
# ends; the boxen shows where the tail actually thickens, which is the
# difference between two measures that share a median and disagree about how
# often they fail. The whiskers reach the true extremes rather than a
# percentile, so nothing is clipped without the reader being able to see it.
BOX_STYLES = ("box", "boxen")
DEFAULT_BOX_STYLE = "box"

# The letter values drawn as filled bands, innermost pair last so a drawing
# loop can widen as it goes. The median is not in this list; it is a line, not
# a band.
BOXEN_BANDS = ((0.10, 0.90), (0.30, 0.70))
BOXEN_MEDIAN = 0.50

# Where the whiskers stop: the observed minimum and maximum. Two single
# observations rather than a statistic, which is why they get a whisker and a
# `min-max` label instead of a band and a percentile one -- a filled band gives
# them the same visual weight as the 10-90 interval, and they have not earned
# it.
BOXEN_WHISKERS = (0.0, 1.0)


def validate_box_style(box_style: str | None) -> str:
    """Return a supported boxplot rendering style, defaulting when None."""
    if box_style is None:
        return DEFAULT_BOX_STYLE
    name = str(box_style).strip()
    if name not in BOX_STYLES:
        raise ValueError(
            f"box_style must be one of {', '.join(BOX_STYLES)}, got {box_style!r}"
        )
    return name


def boxen_percentiles() -> tuple[int, ...]:
    """Every percentile a boxen shows, ascending, including the whisker ends."""
    edges = {
        BOXEN_MEDIAN,
        *BOXEN_WHISKERS,
        *(q for band in BOXEN_BANDS for q in band),
    }
    return tuple(int(round(q * 100)) for q in sorted(edges))


def boxen_band_label(low: float, high: float) -> str:
    """Name one filled band by the percentiles it spans."""
    return f"{int(round(low * 100))}-{int(round(high * 100))} pct."


def unfavourable_quantile(higher_is_better: bool) -> float:
    """The tail of a distribution that a robustness statistic must look at.

    A normalized rank is best at 0, so its bad tail is the upper one and the
    statistic is the 90th percentile. A quality value is best at its maximum,
    so its bad tail is the lower one and the same statement is the 10th. Using
    one fixed quantile for both would rank the value figures by their *best*
    cases and call it robustness.
    """
    return 1.0 - ROBUSTNESS_QUANTILE if higher_is_better else ROBUSTNESS_QUANTILE


QUALITY_DIRECTIONS = {
    "AUPRC": "higher",
    "correlation": "higher",
    "violation_rate": "lower",
}


def quality_direction(quality_measure: str) -> str:
    """Return whether a larger value of this quality measure is better.

    Violation rate is the only measure where lower is better, and unknown
    measures default to higher so a new quality measure ranks in the intuitive
    direction rather than silently inverting.
    """
    return "lower" if str(quality_measure) == "violation_rate" else "higher"


# The first six colors are Seaborn's colorblind palette in the order used by
# the ReSi paper. Signal extends that taxonomy for the signal-producing AUC
# measures.
MEASURE_CATEGORY_ORDER = (
    "Neighbors",
    "RSM",
    "Alignment",
    "Topology",
    "CCA",
    "Statistic",
    "Signal",
    "Other",
)
MEASURE_CATEGORY_COLORS = {
    "Neighbors": "#0173b2",
    "RSM": "#de8f05",
    "Alignment": "#029e73",
    "Topology": "#d55e00",
    "CCA": "#cc78bc",
    "Statistic": "#ca9161",
    "Signal": "#9B5DE0",
    "Other": "#949494",
}

_MEASURE_CATEGORIES = {
    # CCA
    "PWCCA": "CCA",
    "SVCCA": "CCA",
    # Alignment
    "AlignCos": "Alignment",
    "AlignedCosineSimilarity": "Alignment",
    "AngShape": "Alignment",
    "OrthogonalAngularShapeMetricCentered": "Alignment",
    "HardCorr": "Alignment",
    "HardCorrelationMatch": "Alignment",
    "LinReg": "Alignment",
    "LinearRegression": "Alignment",
    "OrthProc": "Alignment",
    "OrthogonalProcrustesCenteredAndNormalized": "Alignment",
    "PermProc": "Alignment",
    "PermutationProcrustes": "Alignment",
    "ProcDist": "Alignment",
    "ProcrustesSizeAndShapeDistance": "Alignment",
    "SoftCorr": "Alignment",
    "SoftCorrelationMatch": "Alignment",
    # Representational similarity matrix measures
    "CKA": "RSM",
    "DistCorr": "RSM",
    "DistanceCorrelation": "RSM",
    "EOS": "RSM",
    "EigenspaceOverlapScore": "RSM",
    "GULP": "RSM",
    "Gulp": "RSM",
    "RSA": "RSM",
    "RSMDiff": "RSM",
    "RSMNormDifference": "RSM",
    # Nearest-neighbor measures
    "2nd-Cos": "Neighbors",
    "SecondOrderCosineSimilarity": "Neighbors",
    "Jaccard": "Neighbors",
    "JaccardSimilarity": "Neighbors",
    "RankSim": "Neighbors",
    "RankSimilarity": "Neighbors",
    # Topological measures
    "GS": "Topology",
    "GeometryScore": "Topology",
    "IMD": "Topology",
    "IMDScore": "Topology",
    "RTD": "Topology",
    # Representation statistics
    "ConcDiff": "Statistic",
    "ConcentricityDifference": "Statistic",
    "MagDiff": "Statistic",
    "MagnitudeDifference": "Statistic",
    "UnifDiff": "Statistic",
    "UniformityDifference": "Statistic",
    # Historical names for the local manifold measures
    "CKA_RBF": "RSM",
    "CKA_RBF_AUC": "Signal",
    "RWKA": "RSM",
    "RWKA_AUC": "Signal",
}

# Registered manifold measures declare their own category in the shared
# catalogue, so this mapping cannot drift from what actually gets registered.
_MANIFOLD_CATEGORIES = manifold_measure_categories()

# How each measure is named in every figure and table. ReSi's own measures keep
# the abbreviations its notebooks print, so a recreated table can be read
# against the published one.
#
# Our measures are named as the mathematics they are, not as the class that
# implements them: a reader meets `CKArbfAUC` nowhere in the paper's text, and
# the class name hides that these are all the same construction over different
# kernels. Written in `$...$` so one string serves both renderers -- Matplotlib
# mathtext in the figures and real LaTeX in the tables.
#
# `\boldsymbol{s}` is braced deliberately. LaTeX accepts the bare
# `\boldsymbol s`, Matplotlib's mathtext does not, and the braced spelling is
# identical in both.
MEASURE_LABELS = {
    "AlignedCosineSimilarity": "AlignCos",
    "CKA": r"$\mathrm{CKA}_\mathrm{lin}$",
    "ConcentricityDifference": "ConcDiff",
    "DistanceCorrelation": "DistCorr",
    "EigenspaceOverlapScore": "EOS",
    "GeometryScore": "GS",
    "Gulp": "GULP",
    "HardCorrelationMatch": "HardCorr",
    "IMDScore": "IMD",
    "JaccardSimilarity": "Jaccard",
    "LinearRegression": "LinReg",
    "MagnitudeDifference": "MagDiff",
    "OrthogonalAngularShapeMetricCentered": "AngShape",
    "OrthogonalProcrustesCenteredAndNormalized": "OrthProc",
    "PWCCA": "PWCCA",
    "PermutationProcrustes": "PermProc",
    "ProcrustesSizeAndShapeDistance": "ProcDist",
    "RSA": "RSA",
    "RSMNormDifference": "RSMDiff",
    "RankSimilarity": "RankSim",
    "SVCCA": "SVCCA",
    "SecondOrderCosineSimilarity": "2nd-Cos",
    "SoftCorrelationMatch": "SoftCorr",
    "UniformityDifference": "UnifDiff",
    "RTD": "RTD",
    # Historical names for the local manifold measures.
    "CKA_RBF": "CKA_RBF",
    "CKA_RBF_AUC": "CKA_RBF_AUC",
    "RWKA": "RWKA",
    "RWKA_AUC": "RWKA_AUC",
    # Fixed manifold measures: the kernel, then what was held fixed.
    #
    # The random-walk kernel alignments keep their trailing A -- they are
    # alignments, like CKA and UKA beside them. The symmetric variant is the
    # one this work reports, so it carries the plain name and the asymmetric
    # one it is not reported against is marked `a`, in the same prefix
    # convention `d` (degree-centered) and `mc` (mean-centered) already use.
    "MutualKNNTop10": r"$\mathrm{MNN}_{10}$",
    "CKNNATop10": r"$\mathrm{CKNNA}_{10}$",
    "CKArbfSigma02": r"$\mathrm{CKA}_{\mathrm{RBF},0.2}$",
    "CKArbfSigma05": r"$\mathrm{CKA}_{\mathrm{RBF},0.5}$",
    "RWKArbfSigma05": r"$\mathrm{aRWKA}_{\mathrm{RBF},0.5}$",
    "sRWKArbfSigma05": r"$\mathrm{RWKA}_{\mathrm{RBF},0.5}$",
    "dRWKArbfSigma05": r"$\mathrm{dRWKA}_{\mathrm{RBF},0.5}$",
    "SoftmaxRWKATemp05": r"$\mathrm{RWKA}_{\mathrm{softmax},0.5}$",
    # Signal measures: the area under the sweep of the same kernel's signal.
    "CKArbfAUC": r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{CKA})$",
    "UKArbfAUC": r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{UKA})$",
    "sRWKArbfAUC": r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{RWKA})$",
    "dRWKArbfAUC": r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{dRWKA})$",
    "mcRWKArbfAUC": r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{mcRWKA})$",
    "RWKArbfAUC": r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{aRWKA})$",
    "MutualKNNAUC": r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{MNN})$",
    "CKNNAAUC": r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{CKNNA})$",
    "RWKAsoftmaxAUC": r"$\mathrm{AUC}(\boldsymbol{s}_{\mathrm{RWKA},\mathrm{softmax}})$",
}

# Every registered measure must be named, or a run including one would draw an
# axis mixing mathematics with a class name. Checked here rather than in a test
# so adding a measure without a label fails at import, where the omission is.
_DUPLICATE_LABELS = sorted(
    {
        label
        for label in MEASURE_LABELS.values()
        if list(MEASURE_LABELS.values()).count(label) > 1
    }
)
if _DUPLICATE_LABELS:
    raise RuntimeError(f"Measures sharing a display label: {_DUPLICATE_LABELS}")

_UNLABELLED = [
    spec.class_name
    for spec in RESI_MEASURE_SPECS
    if spec.class_name not in MEASURE_LABELS
]
if _UNLABELLED:
    raise RuntimeError(f"Manifold measures without a display label: {_UNLABELLED}")

# Labels resolve back to the measure they name, so anything keyed on a measure
# -- its category, and through that its colour and its table row block -- works
# whether it is handed a class name or the label a figure prints. Without this
# a display frame carrying `$\mathrm{CKA}_\mathrm{lin}$` would fall through to
# "Other", turning the box grey and the table block wrong.
_MEASURE_NAMES_BY_LABEL = {
    label: name for name, label in MEASURE_LABELS.items() if label != name
}


def measure_label(measure: str) -> str:
    """The display name for a measure, or its own name if it has none."""
    return MEASURE_LABELS.get(str(measure), str(measure))


def measure_category(measure: str) -> str:
    """Return the display category for a ReSi or manifold measure name."""
    name = str(measure)
    name = _MEASURE_NAMES_BY_LABEL.get(name, name)
    if name in _MANIFOLD_CATEGORIES:
        return _MANIFOLD_CATEGORIES[name]
    # Unregistered AUC names can still appear in historical result files.
    if name.startswith("Manifold") and name.endswith("AUC"):
        return "Signal"
    return _MEASURE_CATEGORIES.get(name, "Other")


__all__ = [
    "DEFAULT_RANK_SCALE",
    "MEASURE_CATEGORY_COLORS",
    "MEASURE_CATEGORY_ORDER",
    "MEASURE_LABELS",
    "BOXEN_BANDS",
    "BOXEN_MEDIAN",
    "BOXEN_WHISKERS",
    "BOX_STYLES",
    "DEFAULT_BOX_STYLE",
    "DEFAULT_RANK_SORT",
    "CORRELATION_TASK_DIRS",
    "QUALITY_DIRECTIONS",
    "RANKED_FUNCTIONAL_MEASURES",
    "RANKED_QUALITY_MEASURES",
    "RANK_SCALES",
    "RANK_SORTS",
    "ROBUSTNESS_QUANTILE",
    "RANK_SCALE_CASE_COLUMNS",
    "measure_category",
    "measure_label",
    "quality_direction",
    "rank_scale_case_column",
    "unfavourable_quantile",
    "boxen_band_label",
    "boxen_percentiles",
    "validate_box_style",
    "validate_rank_scale",
    "validate_rank_sort",
]
