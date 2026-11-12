"""Single source of truth for the manifold measures registered with ReSi.

Three consumers need this information: the adapter builds its measure classes
from it, campaign validation checks configured measure names against it, and
result analysis categorizes measures with it. Keeping one catalogue makes a
mismatch an import-time error instead of silent drift.

This module must not import Torch, ReSi, or any metric implementation. Campaign
preparation runs in environments without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ResiMeasureSpec:
    """One measure class registered into ReSi's in-memory measure registry."""

    class_name: str
    metric_name: str
    kind: str
    category: str
    metric_kwargs: tuple[tuple[str, Any], ...] = ()
    clamp_topk: bool = False
    # AUC measures whose whole point is the sampling grid: without a campaign
    # measure_sweeps entry they would silently fall back to the registry grid
    # and duplicate another measure, so campaign validation insists on one.
    requires_sweep_grid: bool = False

    def kwargs(self) -> dict[str, Any]:
        """Return a fresh mutable copy of the fixed metric arguments."""
        return dict(self.metric_kwargs)

    @property
    def is_auc(self) -> bool:
        """Return whether this measure sweeps a parameter and records a signal."""
        return self.kind == "auc"


# The ordinal indices count every pair of distance comparisons, so the exact
# QSI over all 10^4 rows of a large task needs tens of gigabytes. They score a
# seeded shared subset of at most this many rows instead -- exactly, not
# approximately -- and smaller comparisons are untouched. Changing it changes
# those measures' numbers, so it needs a recompute of them.
ORDINAL_MAX_POINTS = 2000

RESI_MEASURE_SPECS = (
    ResiMeasureSpec(
        class_name="MutualKNNTop10",
        metric_name="mutual_knn",
        kind="fixed",
        category="Neighbors",
        metric_kwargs=(("use_distance", False),),
        clamp_topk=True,
    ),
    ResiMeasureSpec(
        class_name="CKNNATop10",
        metric_name="cknna",
        kind="fixed",
        category="Neighbors",
        clamp_topk=True,
    ),
    ResiMeasureSpec(
        class_name="TSIEuclid",
        metric_name="tsi",
        kind="fixed",
        category="Neighbors",
        metric_kwargs=(("max_points", ORDINAL_MAX_POINTS),),
    ),
    ResiMeasureSpec(
        class_name="QSIEuclid",
        metric_name="qsi",
        kind="fixed",
        category="Neighbors",
        metric_kwargs=(("max_points", ORDINAL_MAX_POINTS),),
    ),
    ResiMeasureSpec(
        class_name="RWKArbfSigma05",
        metric_name="rbf_rwka",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("rbf_sigma", 0.5),),
    ),
    ResiMeasureSpec(
        class_name="SoftmaxRWKATemp05",
        metric_name="softmax_rwka",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("temperature", 0.5),),
    ),
    ResiMeasureSpec(
        class_name="sRWKArbfSigma05",
        metric_name="rbf_rwka_symmetric",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("rbf_sigma", 0.5),),
    ),
    ResiMeasureSpec(
        class_name="dRWKArbfSigma05",
        metric_name="rbf_degree_crwka",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("rbf_sigma", 0.5),),
    ),
    ResiMeasureSpec(
        class_name="CKArbfSigma05",
        metric_name="cka_rbf",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("rbf_sigma", 0.5),),
    ),
    # The bandwidth multiplier scales a data-derived median distance, so it is
    # scale-invariant either way; 0.5 and 0.2 are two points on that multiplier,
    # neither of them derived from a rule. Reporting a second one is what makes
    # the first one's standing checkable.
    ResiMeasureSpec(
        class_name="CKArbfSigma02",
        metric_name="cka_rbf",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("rbf_sigma", 0.2),),
    ),
    ResiMeasureSpec(
        class_name="CKArbfSigma04",
        metric_name="cka_rbf",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("rbf_sigma", 0.4),),
    ),
    ResiMeasureSpec(
        class_name="CKArbfSigma06",
        metric_name="cka_rbf",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("rbf_sigma", 0.6),),
    ),
    ResiMeasureSpec(
        class_name="CKArbfSigma08",
        metric_name="cka_rbf",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("rbf_sigma", 0.8),),
    ),
    ResiMeasureSpec(
        class_name="CKArbfSigma10",
        metric_name="cka_rbf",
        kind="fixed",
        category="RSM",
        metric_kwargs=(("rbf_sigma", 1.0),),
    ),
    ResiMeasureSpec(
        class_name="MutualKNNAUC",
        metric_name="mutual_knn",
        kind="auc",
        category="Signal",
        metric_kwargs=(("use_distance", False),),
    ),
    ResiMeasureSpec(
        class_name="CKNNAAUC",
        metric_name="cknna",
        kind="auc",
        category="Signal",
    ),
    ResiMeasureSpec(
        class_name="RWKArbfAUC",
        metric_name="rbf_rwka",
        kind="auc",
        category="Signal",
    ),
    ResiMeasureSpec(
        class_name="RWKAsoftmaxAUC",
        metric_name="softmax_rwka",
        kind="auc",
        category="Signal",
    ),
    ResiMeasureSpec(
        class_name="CKArbfAUC",
        metric_name="cka_rbf",
        kind="auc",
        category="Signal",
    ),
    # CKArbfAUC computed a second time, under a new name so ReSi does not skip
    # it. The first run stored the wrong similarity curve on most cross-group
    # comparisons (see the signal-hook note in resi/runtime.py); this one stores
    # every pair's own curve, which is what the signal-sensitivity analysis
    # re-reduces. Its scalar AUC must equal CKArbfAUC exactly.
    ResiMeasureSpec(
        class_name="CKArbfAUCRerun",
        metric_name="cka_rbf",
        kind="auc",
        category="Signal",
    ),
    # The same CKA sweep as CKArbfAUC (equally spaced in log sigma), sampled
    # on grids equally spaced in sqrt(sigma) and in sigma itself. The grids
    # come from the campaign's measure_sweeps; the sampling is the weighting.
    ResiMeasureSpec(
        class_name="CKArbfAUCSqrt",
        metric_name="cka_rbf",
        kind="auc",
        category="Signal",
        requires_sweep_grid=True,
    ),
    ResiMeasureSpec(
        class_name="CKArbfAUCLin",
        metric_name="cka_rbf",
        kind="auc",
        category="Signal",
        requires_sweep_grid=True,
    ),
    ResiMeasureSpec(
        class_name="sRWKArbfAUC",
        metric_name="rbf_rwka_symmetric",
        kind="auc",
        category="Signal",
    ),
    ResiMeasureSpec(
        class_name="UKArbfAUC",
        metric_name="rbf_uka",
        kind="auc",
        category="Signal",
    ),
    ResiMeasureSpec(
        class_name="dRWKArbfAUC",
        metric_name="rbf_degree_crwka",
        kind="auc",
        category="Signal",
    ),
    # Mean centering annihilates the constant and both row/column norm terms of
    # the large-bandwidth kernel expansion, so this variant is expected to tend
    # to linear CKA as rbf_sigma grows. Degree centering removes a different
    # mode and carries no such guarantee; the pair exists to test that.
    ResiMeasureSpec(
        class_name="mcRWKArbfAUC",
        metric_name="rbf_crwka",
        kind="auc",
        category="Signal",
    ),
)


MEASURE_SPECS_BY_NAME: Mapping[str, ResiMeasureSpec] = {
    spec.class_name: spec for spec in RESI_MEASURE_SPECS
}

# Registration order is preserved; ReSi keys its registry by class name.
MANIFOLD_RESI_MEASURE_CLASSES = [spec.class_name for spec in RESI_MEASURE_SPECS]


def get_measure_spec(class_name: str) -> ResiMeasureSpec:
    """Look up one registered measure specification by its ReSi class name."""
    try:
        return MEASURE_SPECS_BY_NAME[class_name]
    except KeyError:
        raise KeyError(f"Unknown ReSi manifold measure: {class_name}") from None


def manifold_measure_categories() -> dict[str, str]:
    """Return the analysis category for every registered manifold measure."""
    return {spec.class_name: spec.category for spec in RESI_MEASURE_SPECS}


__all__ = [
    "MANIFOLD_RESI_MEASURE_CLASSES",
    "MEASURE_SPECS_BY_NAME",
    "RESI_MEASURE_SPECS",
    "ResiMeasureSpec",
    "get_measure_spec",
    "manifold_measure_categories",
]
