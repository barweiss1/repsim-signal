"""Representation-similarity metrics, kernels, and registry metadata."""

from .interface import AlignmentMetrics
from .kernels import cka
from .kernels import center_kernel
from .kernels import compute_rbf_distance_cache
from .kernels import compute_rbf_kernel
from .kernels import compute_softmax_gram_cache
from .kernels import compute_softmax_kernel
from .kernels import hsic_biased
from .kernels import hsic_unbiased
from .kernels import rbf_uka
from .kernels import RBFDistanceCache
from .kernels import rbf_kernel_from_distance_cache
from .kernels import resolve_distance_tolerance
from .kernels import resolve_rbf_base_bandwidth
from .kernels import score_cka_kernels
from .kernels import SoftmaxGramCache
from .kernels import softmax_kernel_from_gram_cache
from .kernels import svcca
from .kernels import unbiased_cka
from .neighbors import cknna
from .neighbors import compute_knn_accuracy
from .neighbors import compute_nearest_neighbors
from .neighbors import cycle_knn
from .neighbors import mutual_knn
from .prepared import PreparedMetricCurve, PreparedMetricPair
from .prepared import metric_feature_matrix
from .prepared import prepare_metric_curve
from .prepared import prepare_metric_pair
from .prepared import prepare_rbf_kernel_for_metric
from .prepared import score_prepared_curve
from .prepared import score_prepared_metric
from .prepared import supports_prepared_curve
from .prepared import supports_prepared_metric
from .random_walk import RWKADiffusionCache
from .random_walk import compute_rbf_rwka_diffusion_cache
from .random_walk import degree_center_random_walk_kernel
from .random_walk import normalize_random_walk_kernel
from .random_walk import power_rbf_rwka_diffusion_cache
from .random_walk import rbf_crwka
from .random_walk import rbf_degree_crwka
from .random_walk import rbf_rwka
from .random_walk import rbf_rwka_diffusion_time
from .random_walk import rbf_rwka_symmetric
from .random_walk import score_rwka_kernels
from .random_walk import softmax_rwka

from .registry import (
    METRIC_REGISTRY,
    METRIC_SPECS,
    MetricSpec,
    SweepParameterSpec,
    get_metric_spec,
    get_sweep_parameter_default,
)
from .utils import DISTANCE_RELATIVE_EPS
from .utils import NORMALIZATION_EPS
from .utils import check_division_by_zero_warning
from .utils import remove_outliers

__all__ = [
    "METRIC_REGISTRY",
    "METRIC_SPECS",
    "MetricSpec",
    "SweepParameterSpec",
    "get_metric_spec",
    "get_sweep_parameter_default",
    "AlignmentMetrics",
    "DISTANCE_RELATIVE_EPS",
    "NORMALIZATION_EPS",
    "PreparedMetricPair",
    "PreparedMetricCurve",
    "RBFDistanceCache",
    "RWKADiffusionCache",
    "SoftmaxGramCache",
    "center_kernel",
    "check_division_by_zero_warning",
    "cka",
    "cknna",
    "compute_knn_accuracy",
    "compute_nearest_neighbors",
    "compute_rbf_distance_cache",
    "compute_rbf_kernel",
    "compute_rbf_rwka_diffusion_cache",
    "compute_softmax_kernel",
    "compute_softmax_gram_cache",
    "cycle_knn",
    "degree_center_random_walk_kernel",
    "hsic_biased",
    "hsic_unbiased",
    "mutual_knn",
    "metric_feature_matrix",
    "normalize_random_walk_kernel",
    "power_rbf_rwka_diffusion_cache",
    "prepare_metric_curve",
    "prepare_metric_pair",
    "prepare_rbf_kernel_for_metric",
    "rbf_crwka",
    "rbf_degree_crwka",
    "rbf_rwka",
    "rbf_rwka_diffusion_time",
    "rbf_rwka_symmetric",
    "rbf_uka",
    "rbf_kernel_from_distance_cache",
    "resolve_distance_tolerance",
    "resolve_rbf_base_bandwidth",
    "remove_outliers",
    "score_cka_kernels",
    "score_prepared_curve",
    "score_prepared_metric",
    "score_rwka_kernels",
    "softmax_rwka",
    "softmax_kernel_from_gram_cache",
    "svcca",
    "supports_prepared_curve",
    "supports_prepared_metric",
    "unbiased_cka",
]
