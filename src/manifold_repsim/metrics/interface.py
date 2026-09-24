"""Registry-driven compatibility interface and reusable-computation routing.

The metric functions are public and can be called directly. This class remains
temporarily because repository scripts and notebooks still depend on three
class-based services: registry-name dispatch through ``measure``, legacy
catalogues such as ``SUPPORTED_METRICS`` and ``SWEEP_PARAMS``, and the prepared
kernel/cache API used by sweeps and permutation calibration. Keeping those
services here avoids duplicating routing logic or forcing the later caller
migration early. The interface contains no metric formulas or scoring algorithms
and can be removed after callers migrate to the package API in Step 16.
"""

from __future__ import annotations

import math
from typing import Any

from .kernels import cka
from .kernels import rbf_uka
from .kernels import score_cka_kernels
from .kernels import svcca
from .kernels import unbiased_cka
from .neighbors import cknna
from .neighbors import cycle_knn
from .neighbors import mutual_knn
from .prepared import PreparedMetricPair
from .prepared import prepare_metric_pair
from .prepared import score_prepared_metric
from .random_walk import rbf_crwka
from .random_walk import rbf_degree_crwka
from .random_walk import rbf_rwka
from .random_walk import rbf_rwka_diffusion_time
from .random_walk import rbf_rwka_symmetric
from .random_walk import score_rwka_kernels
from .random_walk import softmax_rwka
from .registry import METRIC_REGISTRY
from .registry import get_metric_spec
from .utils import to_torch_tensor


class AlignmentMetrics:
    """Temporary class interface preserving the repository's legacy metric API."""

    SUPPORTED_METRICS = list(METRIC_REGISTRY)
    SWEEP_PARAMS = {
        name: spec.compatibility_sweep_dict() for name, spec in METRIC_REGISTRY.items()
    }
    KERNEL_REUSABLE_METRICS = {
        name for name, spec in METRIC_REGISTRY.items() if spec.reuses_kernel
    }

    _to_torch_tensor = staticmethod(to_torch_tensor)
    cycle_knn = staticmethod(cycle_knn)
    mutual_knn = staticmethod(mutual_knn)
    cka = staticmethod(cka)
    rbf_uka = staticmethod(rbf_uka)
    unbiased_cka = staticmethod(unbiased_cka)
    cknna = staticmethod(cknna)
    svcca = staticmethod(svcca)
    softmax_rwka = staticmethod(softmax_rwka)
    rbf_rwka = staticmethod(rbf_rwka)
    rbf_rwka_symmetric = staticmethod(rbf_rwka_symmetric)
    rbf_crwka = staticmethod(rbf_crwka)
    rbf_degree_crwka = staticmethod(rbf_degree_crwka)
    rbf_rwka_diffusion_time = staticmethod(rbf_rwka_diffusion_time)
    score_cka_kernels = staticmethod(score_cka_kernels)
    score_rwka_kernels = staticmethod(score_rwka_kernels)

    @staticmethod
    def measure(
        metric: str,
        feats_A: Any,
        feats_B: Any,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """Dispatch a registered metric and return a finite Python float."""
        spec = get_metric_spec(metric)
        implementation = getattr(AlignmentMetrics, spec.implementation)
        result = float(
            implementation(
                feats_A,
                feats_B,
                *args,
                **spec.resolved_kwargs(kwargs),
            )
        )
        if not math.isfinite(result):
            raise ValueError(f"Metric {metric} returned a non-finite score: {result}.")
        return result

    @staticmethod
    def supports_kernel_reuse(metric: str) -> bool:
        """Return whether the metric exposes prepared-kernel scoring."""
        return metric in AlignmentMetrics.KERNEL_REUSABLE_METRICS

    @staticmethod
    def compute_kernel_pair(
        metric: str,
        feats_A: Any,
        feats_B: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Prepare the reusable kernel or diffusion-cache pair for a metric."""
        prepared = prepare_metric_pair(metric, feats_A, feats_B, **kwargs)
        return prepared.source, prepared.target

    @staticmethod
    def score_kernel_metric(
        metric: str,
        K: Any,
        L: Any,
        **kwargs: Any,
    ) -> float:
        """Score a pair returned by :meth:`compute_kernel_pair`."""
        prepared = PreparedMetricPair.create(metric, K, L, kwargs)
        return score_prepared_metric(prepared)
