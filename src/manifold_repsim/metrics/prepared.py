"""Exact reusable preparation and scoring for metric comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch

from .kernels import center_kernel
from .kernels import compute_rbf_kernel
from .kernels import compute_rbf_distance_cache
from .kernels import compute_softmax_gram_cache
from .kernels import compute_softmax_kernel
from .kernels import rbf_kernel_from_distance_cache
from .kernels import resolve_rbf_base_bandwidth
from .kernels import score_cka_kernels
from .kernels import softmax_kernel_from_gram_cache
from .neighbors import compute_nearest_neighbors
from .neighbors import score_cknna_rankings
from .neighbors import score_cycle_neighbor_rankings
from .neighbors import score_mutual_neighbor_rankings
from .random_walk import compute_rbf_rwka_diffusion_cache
from .random_walk import degree_center_random_walk_kernel
from .random_walk import normalize_random_walk_kernel
from .random_walk import power_rbf_rwka_diffusion_cache
from .random_walk import score_rwka_kernels
from .registry import get_metric_spec
from .utils import validate_representation_pair
from .utils import validate_topk


@dataclass(frozen=True)
class PreparedMetricPair:
    """Reusable state for one registered metric comparison.

    ``source`` and ``target`` are raw kernels, normalized kernels, or diffusion
    caches according to ``metric_name``. ``metric_kwargs`` records the settings
    that affect scoring after preparation.
    """

    metric_name: str
    source: Any
    target: Any
    metric_kwargs: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        metric_name: str,
        source: Any,
        target: Any,
        metric_kwargs: Mapping[str, Any] | None = None,
    ) -> "PreparedMetricPair":
        """Create immutable prepared state without copying tensor data."""
        return cls(
            metric_name=metric_name,
            source=source,
            target=target,
            metric_kwargs=MappingProxyType(dict(metric_kwargs or {})),
        )


@dataclass(frozen=True)
class PreparedMetricCurve:
    """Reusable state for an exact sweep of one registered metric parameter."""

    metric_name: str
    parameter_name: str
    parameter_values: tuple[int | float, ...]
    source: Any
    target: Any
    metric_kwargs: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        metric_name: str,
        parameter_name: str,
        parameter_values: tuple[int | float, ...],
        source: Any,
        target: Any,
        metric_kwargs: Mapping[str, Any] | None = None,
    ) -> "PreparedMetricCurve":
        """Create immutable curve state without copying tensor data."""
        return cls(
            metric_name=metric_name,
            parameter_name=parameter_name,
            parameter_values=parameter_values,
            source=source,
            target=target,
            metric_kwargs=MappingProxyType(dict(metric_kwargs or {})),
        )


def supports_prepared_metric(metric_name: str) -> bool:
    """Return whether a registered metric has an exact prepared path."""
    return get_metric_spec(metric_name).prepared_family is not None


def supports_prepared_curve(metric_name: str) -> bool:
    """Return whether a registered metric has an exact prepared sweep path."""
    spec = get_metric_spec(metric_name)
    return spec.sweep is not None and spec.prepared_family is not None


def prepare_rbf_kernel_for_metric(
    metric_name: str,
    raw_kernel: torch.Tensor,
) -> torch.Tensor:
    """Apply the metric-specific normalization to one raw RBF kernel."""
    if metric_name in {"cka_rbf", "cka_rbf_quantile", "rbf_uka"}:
        return raw_kernel
    if metric_name == "rbf_degree_crwka":
        return degree_center_random_walk_kernel(raw_kernel)
    if metric_name in {
        "rbf_rwka",
        "rbf_rwka_symmetric",
        "rbf_crwka",
        "rbf_rwka_quantile",
    }:
        return normalize_random_walk_kernel(
            raw_kernel,
            symmetric=metric_name in {"rbf_rwka_symmetric", "rbf_crwka"},
        )
    raise ValueError(f"Metric {metric_name} is not based on a reusable RBF kernel.")


def metric_feature_matrix(
    metric_name: str,
    prepared_kernel: torch.Tensor,
) -> torch.Tensor:
    """Return a matrix whose flattened cosine equals its biased prepared score."""
    if metric_name in {"cka", "cka_rbf", "cka_rbf_quantile", "rbf_crwka"}:
        return center_kernel(prepared_kernel)
    if metric_name in {
        "rbf_uka",
        "softmax_rwka",
        "rbf_rwka",
        "rbf_rwka_symmetric",
        "rbf_degree_crwka",
        "rbf_rwka_quantile",
    }:
        return prepared_kernel
    raise ValueError(f"Metric {metric_name} has no reusable feature matrix.")


def prepare_metric_pair(
    metric_name: str,
    feats_A: Any,
    feats_B: Any,
    **kwargs: Any,
) -> PreparedMetricPair:
    """Prepare reusable state for an exact metric comparison."""
    spec = get_metric_spec(metric_name)
    if spec.prepared_family is None:
        raise ValueError(f"Metric {metric_name} does not support prepared scoring.")
    resolved_kwargs = spec.resolved_kwargs(kwargs)

    if metric_name == "cknna":
        min_samples = 4 if resolved_kwargs.get("unbiased", True) else 3
    else:
        min_samples = 2
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name=metric_name,
        min_samples=min_samples,
    )
    if metric_name in {"cycle_knn", "mutual_knn", "mutual_knn_dist"}:
        topk = validate_topk(
            resolved_kwargs["topk"],
            feats_A.shape[0],
            metric_name,
        )
        use_distance = resolved_kwargs.get("use_distance", False)
        source = compute_nearest_neighbors(
            feats_A,
            topk,
            use_distance=use_distance,
        )
        target = compute_nearest_neighbors(
            feats_B,
            topk,
            use_distance=use_distance,
        )
    elif metric_name == "cknna":
        topk = resolved_kwargs.get("topk")
        if topk is None:
            topk = feats_A.shape[0] - 1
        topk = validate_topk(topk, feats_A.shape[0], metric_name, minimum=2)
        unbiased = resolved_kwargs.get("unbiased", True)
        gram_a = feats_A @ feats_A.T
        gram_b = feats_B @ feats_B.T
        if unbiased:
            ranking_a = gram_a.clone().fill_diagonal_(float("-inf"))
            ranking_b = gram_b.clone().fill_diagonal_(float("-inf"))
        else:
            ranking_a, ranking_b = gram_a, gram_b
        source = (gram_a, torch.topk(ranking_a, topk, dim=1).indices)
        target = (gram_b, torch.topk(ranking_b, topk, dim=1).indices)
        resolved_kwargs["topk"] = topk
    elif metric_name == "cka":
        source = torch.mm(feats_A, feats_A.T)
        target = torch.mm(feats_B, feats_B.T)
    else:
        rbf_sigma = resolved_kwargs.get("rbf_sigma", 1.0)
        median = resolved_kwargs.get("median", True)
        quantile = resolved_kwargs.get("quantile")
        if metric_name in {
            "cka_rbf",
            "cka_rbf_quantile",
            "rbf_uka",
            "rbf_rwka",
            "rbf_rwka_symmetric",
            "rbf_crwka",
            "rbf_degree_crwka",
            "rbf_rwka_quantile",
        }:
            source = prepare_rbf_kernel_for_metric(
                metric_name,
                compute_rbf_kernel(
                    feats_A,
                    rbf_sigma=rbf_sigma,
                    median=median,
                    quantile=quantile,
                ),
            )
            target = prepare_rbf_kernel_for_metric(
                metric_name,
                compute_rbf_kernel(
                    feats_B,
                    rbf_sigma=rbf_sigma,
                    median=median,
                    quantile=quantile,
                ),
            )
        elif metric_name == "rbf_rwka_diffusion_time":
            diffusion_sigma = resolved_kwargs.get("rbf_sigma", 0.2)
            source = compute_rbf_rwka_diffusion_cache(
                feats_A,
                rbf_sigma=diffusion_sigma,
                median=median,
                quantile=quantile,
            )
            target = compute_rbf_rwka_diffusion_cache(
                feats_B,
                rbf_sigma=diffusion_sigma,
                median=median,
                quantile=quantile,
            )
        elif metric_name == "softmax_rwka":
            temperature = resolved_kwargs.get("temperature", 0.5)
            range_based = resolved_kwargs.get("range_based", True)
            source = compute_softmax_kernel(
                feats_A,
                temperature=temperature,
                range_based=range_based,
            )
            target = compute_softmax_kernel(
                feats_B,
                temperature=temperature,
                range_based=range_based,
            )
        else:  # pragma: no cover - guarded by registry capability metadata.
            raise ValueError(f"Metric {metric_name} has no prepared implementation.")
    return PreparedMetricPair.create(metric_name, source, target, resolved_kwargs)


def _permutation_indices(
    permutation: Any,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.as_tensor(
        permutation,
        dtype=torch.long,
        device=device,
    )
    if indices.ndim != 1 or indices.numel() != num_samples:
        raise ValueError(
            "A prepared target permutation must contain one index per sample."
        )
    expected = torch.arange(num_samples, device=device)
    if not torch.equal(torch.sort(indices).values, expected):
        raise ValueError("Prepared target indices must define a valid permutation.")
    return indices


def _permute_square(matrix: torch.Tensor, permutation: Any) -> torch.Tensor:
    indices = _permutation_indices(
        permutation,
        matrix.shape[0],
        matrix.device,
    )
    return matrix.index_select(0, indices).index_select(1, indices)


def _permute_neighbor_rankings(
    neighbors: torch.Tensor,
    permutation: Any,
) -> torch.Tensor:
    indices = _permutation_indices(
        permutation,
        neighbors.shape[0],
        neighbors.device,
    )
    inverse = torch.empty_like(indices)
    inverse[indices] = torch.arange(neighbors.shape[0], device=neighbors.device)
    return inverse[neighbors.index_select(0, indices)]


def score_prepared_metric(
    prepared: PreparedMetricPair,
    *,
    target_permutation: Any | None = None,
    **kwargs: Any,
) -> float:
    """Score prepared state, optionally permuting the target exactly once."""
    metric_name = prepared.metric_name
    resolved_kwargs = dict(prepared.metric_kwargs)
    resolved_kwargs.update(kwargs)
    source, target = prepared.source, prepared.target

    if metric_name in {"cycle_knn", "mutual_knn", "mutual_knn_dist"}:
        if target_permutation is not None:
            target = _permute_neighbor_rankings(target, target_permutation)
        topk = resolved_kwargs["topk"]
        if metric_name == "cycle_knn":
            return score_cycle_neighbor_rankings(source, target, topk)
        return score_mutual_neighbor_rankings(source, target, topk)
    if metric_name == "cknna":
        gram_a, neighbors_a = source
        gram_b, neighbors_b = target
        if target_permutation is not None:
            gram_b = _permute_square(gram_b, target_permutation)
            neighbors_b = _permute_neighbor_rankings(
                neighbors_b,
                target_permutation,
            )
        return score_cknna_rankings(
            gram_a,
            gram_b,
            neighbors_a,
            neighbors_b,
            resolved_kwargs["topk"],
            distance_agnostic=resolved_kwargs.get("distance_agnostic", False),
            unbiased=resolved_kwargs.get("unbiased", True),
        )

    if metric_name == "rbf_rwka_diffusion_time":
        diffusion_time = resolved_kwargs.get("diffusion_time", 1.0)
        source = power_rbf_rwka_diffusion_cache(source, diffusion_time)
        target = power_rbf_rwka_diffusion_cache(target, diffusion_time)
    if target_permutation is not None:
        target = _permute_square(target, target_permutation)

    if metric_name in {"cka", "cka_rbf", "cka_rbf_quantile"}:
        return score_cka_kernels(
            source,
            target,
            unbiased=resolved_kwargs.get("unbiased", False),
        )
    if metric_name == "rbf_uka":
        return score_rwka_kernels(source, target, metric_name="RBF UKA")
    if metric_name == "rbf_crwka":
        return score_cka_kernels(source, target, unbiased=False)
    if metric_name == "rbf_degree_crwka":
        return score_rwka_kernels(
            source,
            target,
            metric_name="Degree-centered RBF RWKA",
        )
    if metric_name in {
        "rbf_rwka",
        "rbf_rwka_symmetric",
        "rbf_rwka_quantile",
    }:
        label = (
            "Symmetric RBF RWKA" if metric_name == "rbf_rwka_symmetric" else "RBF RWKA"
        )
        return score_rwka_kernels(source, target, metric_name=label)
    if metric_name == "rbf_rwka_diffusion_time":
        return score_rwka_kernels(
            source,
            target,
            metric_name="RBF RWKA diffusion-time",
        )
    if metric_name == "softmax_rwka":
        return score_rwka_kernels(source, target, metric_name="Softmax RWKA")
    raise ValueError(f"Metric {metric_name} has no prepared scorer.")


def prepare_metric_curve(
    metric_name: str,
    feats_A: Any,
    feats_B: Any,
    parameter_name: str,
    parameter_values: Any,
    **kwargs: Any,
) -> PreparedMetricCurve:
    """Prepare invariant structures for an exact metric score curve."""
    spec = get_metric_spec(metric_name)
    if (
        not supports_prepared_curve(metric_name)
        or spec.sweep is None
        or spec.sweep.name != parameter_name
    ):
        raise ValueError(
            f"Metric {metric_name} does not define a {parameter_name} sweep."
        )
    values = tuple(parameter_values)
    if not values:
        raise ValueError("A prepared metric curve requires at least one value.")
    resolved_kwargs = spec.resolved_kwargs(kwargs)

    if metric_name == "cknna":
        min_samples = 4 if resolved_kwargs.get("unbiased", True) else 3
    else:
        min_samples = 2
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name=metric_name,
        min_samples=min_samples,
    )
    if parameter_name == "topk":
        minimum = 2 if metric_name == "cknna" else 1
        topks = tuple(
            validate_topk(value, feats_A.shape[0], metric_name, minimum=minimum)
            for value in values
        )
        max_topk = max(topks)
        if metric_name == "cknna":
            gram_a = feats_A @ feats_A.T
            gram_b = feats_B @ feats_B.T
            if resolved_kwargs.get("unbiased", True):
                ranking_a = gram_a.clone().fill_diagonal_(float("-inf"))
                ranking_b = gram_b.clone().fill_diagonal_(float("-inf"))
            else:
                ranking_a, ranking_b = gram_a, gram_b
            source = (
                gram_a,
                torch.topk(ranking_a, max_topk, dim=1).indices,
            )
            target = (
                gram_b,
                torch.topk(ranking_b, max_topk, dim=1).indices,
            )
        else:
            use_distance = resolved_kwargs.get("use_distance", False)
            source = compute_nearest_neighbors(
                feats_A,
                max_topk,
                use_distance=use_distance,
            )
            target = compute_nearest_neighbors(
                feats_B,
                max_topk,
                use_distance=use_distance,
            )
        values = topks
    elif parameter_name in {"rbf_sigma", "quantile"}:
        source = compute_rbf_distance_cache(feats_A)
        target = compute_rbf_distance_cache(feats_B)
    elif parameter_name == "temperature":
        source = compute_softmax_gram_cache(feats_A)
        target = compute_softmax_gram_cache(feats_B)
    elif parameter_name == "diffusion_time":
        pair = prepare_metric_pair(metric_name, feats_A, feats_B, **resolved_kwargs)
        source, target = pair.source, pair.target
    else:  # pragma: no cover - every registered sweep parameter is handled.
        raise ValueError(f"Unsupported prepared sweep parameter: {parameter_name}")

    return PreparedMetricCurve.create(
        metric_name,
        parameter_name,
        values,
        source,
        target,
        resolved_kwargs,
    )


def _score_rbf_curve_value(
    prepared: PreparedMetricCurve,
    value: int | float,
    source_bandwidth: float | None = None,
    target_bandwidth: float | None = None,
) -> float:
    kwargs = dict(prepared.metric_kwargs)
    kwargs[prepared.parameter_name] = value
    rbf_sigma = kwargs.get("rbf_sigma", 1.0)
    median = kwargs.get("median", True)
    quantile = kwargs.get("quantile")
    source = rbf_kernel_from_distance_cache(
        prepared.source,
        rbf_sigma=rbf_sigma,
        median=median,
        quantile=quantile,
        base_bandwidth=source_bandwidth,
    )
    target = rbf_kernel_from_distance_cache(
        prepared.target,
        rbf_sigma=rbf_sigma,
        median=median,
        quantile=quantile,
        base_bandwidth=target_bandwidth,
    )
    metric_name = prepared.metric_name
    source = prepare_rbf_kernel_for_metric(metric_name, source)
    target = prepare_rbf_kernel_for_metric(metric_name, target)
    pair = PreparedMetricPair.create(metric_name, source, target, kwargs)
    return score_prepared_metric(pair)


def score_prepared_curve(prepared: PreparedMetricCurve) -> list[float]:
    """Score every parameter value from prepared invariant structures."""
    source_bandwidth: float | None = None
    target_bandwidth: float | None = None
    if prepared.parameter_name == "rbf_sigma":
        # The data-derived bandwidth depends only on the distance cache, so it
        # is resolved once for the whole sweep rather than at every point. A
        # quantile sweep varies the quantile itself and must keep resolving.
        median = prepared.metric_kwargs.get("median", True)
        quantile = prepared.metric_kwargs.get("quantile")
        source_bandwidth = resolve_rbf_base_bandwidth(
            prepared.source,
            median=median,
            quantile=quantile,
        )
        target_bandwidth = resolve_rbf_base_bandwidth(
            prepared.target,
            median=median,
            quantile=quantile,
        )
    scores: list[float] = []
    for value in prepared.parameter_values:
        if prepared.parameter_name == "topk":
            topk = int(value)
            if prepared.metric_name == "cycle_knn":
                score = score_cycle_neighbor_rankings(
                    prepared.source,
                    prepared.target,
                    topk,
                )
            elif prepared.metric_name in {"mutual_knn", "mutual_knn_dist"}:
                score = score_mutual_neighbor_rankings(
                    prepared.source,
                    prepared.target,
                    topk,
                )
            elif prepared.metric_name == "cknna":
                gram_a, neighbors_a = prepared.source
                gram_b, neighbors_b = prepared.target
                score = score_cknna_rankings(
                    gram_a,
                    gram_b,
                    neighbors_a,
                    neighbors_b,
                    topk,
                    distance_agnostic=prepared.metric_kwargs.get(
                        "distance_agnostic", False
                    ),
                    unbiased=prepared.metric_kwargs.get("unbiased", True),
                )
            else:  # pragma: no cover - registry sweep metadata prevents this.
                raise ValueError(
                    f"Metric {prepared.metric_name} has no neighbor curve scorer."
                )
        elif prepared.parameter_name in {"rbf_sigma", "quantile"}:
            score = _score_rbf_curve_value(
                prepared,
                value,
                source_bandwidth,
                target_bandwidth,
            )
        elif prepared.parameter_name == "temperature":
            kwargs = dict(prepared.metric_kwargs)
            kwargs["temperature"] = value
            source = softmax_kernel_from_gram_cache(
                prepared.source,
                temperature=value,
                range_based=kwargs.get("range_based", True),
            )
            target = softmax_kernel_from_gram_cache(
                prepared.target,
                temperature=value,
                range_based=kwargs.get("range_based", True),
            )
            score = score_prepared_metric(
                PreparedMetricPair.create(
                    prepared.metric_name,
                    source,
                    target,
                    kwargs,
                )
            )
        elif prepared.parameter_name == "diffusion_time":
            score = score_prepared_metric(
                PreparedMetricPair.create(
                    prepared.metric_name,
                    prepared.source,
                    prepared.target,
                    prepared.metric_kwargs,
                ),
                diffusion_time=value,
            )
        else:  # pragma: no cover - preparation handles every parameter family.
            raise ValueError(
                f"Unsupported prepared sweep parameter: {prepared.parameter_name}"
            )
        scores.append(float(score))
    return scores
