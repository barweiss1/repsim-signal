"""Kernel construction, centering, and HSIC estimators."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.cross_decomposition import CCA

from .utils import DISTANCE_RELATIVE_EPS
from .utils import NORMALIZATION_EPS
from .utils import normalized_similarity_or_zero
from .utils import validate_integer_in_range
from .utils import validate_kernel_pair
from .utils import validate_positive_float
from .utils import validate_representation
from .utils import validate_representation_pair


@dataclass(frozen=True)
class RBFDistanceCache:
    """Prepared distance structures reused across one RBF sweep.

    ``squared_distances`` feeds the kernel exponent directly, so the square is
    paid once instead of at every sweep point. ``lower_distances`` stays raw
    because the bandwidth heuristic is a median over unsquared distances.
    ``distance_tolerance`` is the separation below which a distance carries no
    geometry, derived from the representation rather than from the distances so
    that it reflects the precision the rows were actually stored at.
    """

    squared_distances: torch.Tensor
    lower_distances: torch.Tensor
    distance_tolerance: float = 0.0


@dataclass(frozen=True)
class SoftmaxGramCache:
    """High-precision Gram matrix reused across softmax temperatures."""

    gram: torch.Tensor


def center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Double-center a square kernel without materializing the centering matrix."""
    row_mean = K.mean(dim=1, keepdim=True)
    column_mean = K.mean(dim=0, keepdim=True)
    return K - row_mean - column_mean + K.mean()


def hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Compute biased HSIC as the centered-kernel Frobenius inner product."""
    centered_a = center_kernel(K)
    centered_b = center_kernel(L)
    return torch.sum(centered_a * centered_b)


def hsic_unbiased(K: Any, L: Any) -> torch.Tensor:
    """Compute the finite-sample unbiased HSIC estimator from Song et al."""
    K, L = validate_kernel_pair(
        K,
        L,
        metric_name="unbiased HSIC",
        min_samples=4,
    )
    num_samples = K.shape[0]
    K = K.clone().fill_diagonal_(0)
    L = L.clone().fill_diagonal_(0)
    product_sum = torch.dot(K.sum(dim=0), L.sum(dim=1))
    value = (
        torch.sum(K * L.T)
        + torch.sum(K) * torch.sum(L) / ((num_samples - 1) * (num_samples - 2))
        - 2 * product_sum / (num_samples - 2)
    )
    return value / (num_samples * (num_samples - 3))


def compute_softmax_kernel(
    feats: Any,
    temperature: float = 0.1,
    range_based: bool = True,
) -> torch.Tensor:
    """Construct a row-stochastic softmax kernel in float64 precision."""
    cache = compute_softmax_gram_cache(feats)
    return softmax_kernel_from_gram_cache(
        cache,
        temperature=temperature,
        range_based=range_based,
    )


def compute_softmax_gram_cache(feats: Any) -> SoftmaxGramCache:
    """Prepare a high-precision Gram matrix for a temperature sweep."""
    feats = validate_representation(feats, name="softmax-kernel representation")
    return SoftmaxGramCache(gram=torch.mm(feats.double(), feats.double().T))


def softmax_kernel_from_gram_cache(
    cache: SoftmaxGramCache,
    temperature: float = 0.1,
    range_based: bool = True,
) -> torch.Tensor:
    """Construct a softmax kernel from a prepared Gram matrix."""
    temperature = validate_positive_float(temperature, "temperature")
    kernel = cache.gram

    if range_based:
        scale = kernel.max() - kernel.min()
    else:
        shifted = kernel - kernel.min()
        scale = shifted.median()
        if scale <= NORMALIZATION_EPS:
            positive = shifted[shifted > NORMALIZATION_EPS]
            scale = positive.median() if positive.numel() else kernel.new_tensor(1.0)
    if not torch.isfinite(scale) or scale <= NORMALIZATION_EPS:
        scale = kernel.new_tensor(1.0)
    return torch.softmax(kernel / (scale * temperature), dim=1)


def compute_rbf_kernel(
    feats: Any,
    rbf_sigma: float = 1.0,
    median: bool = True,
    quantile: float | None = None,
) -> torch.Tensor:
    """Construct a raw RBF kernel in float64 precision.

    Formula: ``K_ij = exp(-||x_i-x_j||^2 / (2 sigma^2))`` with
    ``sigma = rbf_sigma * base``, where ``base`` is the data-derived bandwidth
    from ``resolve_rbf_base_bandwidth``. A ``base`` at the numerical distance
    floor falls back to the median distance above that floor, or to one when no
    distance clears it.
    """
    cache = compute_rbf_distance_cache(feats)
    return rbf_kernel_from_distance_cache(
        cache,
        rbf_sigma=rbf_sigma,
        median=median,
        quantile=quantile,
    )


def resolve_distance_tolerance(feats: torch.Tensor) -> float:
    """Return the separation below which two rows are numerically indistinct.

    Formula: ``tol = max_i ||x_i|| * max(eps_dtype, sqrt(eps_float64))``.

    Two floors apply and the larger one governs. Rows stored in a given dtype
    cannot differ by less than one quantization step, which scales with
    ``eps_dtype``. Independently, ``torch.cdist`` uses the
    ``|x|^2 + |y|^2 - 2<x,y>`` expansion at benchmark sizes, whose cancellation
    for near-identical rows leaves about ``||x|| * sqrt(eps_float64)``.
    """
    if feats.shape[0] == 0:
        return 0.0
    row_norm = float(feats.double().norm(dim=1).max())
    dtype_eps = float(torch.finfo(feats.dtype).eps)
    return row_norm * max(dtype_eps, DISTANCE_RELATIVE_EPS)


def compute_rbf_distance_cache(feats: Any) -> RBFDistanceCache:
    """Prepare pairwise distances for an RBF bandwidth sweep."""
    feats = validate_representation(feats, name="RBF-kernel representation")
    distances = torch.cdist(feats.double(), feats.double())
    indices = torch.tril_indices(
        distances.shape[0],
        distances.shape[1],
        offset=-1,
        device=distances.device,
    )
    return RBFDistanceCache(
        distance_tolerance=resolve_distance_tolerance(feats),
        # Squaring the cdist result keeps the exponent bit-identical to the
        # former per-point ``distances**2``. The fused expansion used for
        # neighbor distances would not.
        squared_distances=distances**2,
        lower_distances=distances[indices[0], indices[1]],
    )


def _validate_optional_quantile(quantile: float | None) -> float | None:
    """Return a validated distance quantile, or ``None`` when unset."""
    if quantile is None:
        return None
    quantile = float(quantile)
    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError(
            f"quantile must be finite and between 0 and 1, got {quantile!r}."
        )
    return quantile


def resolve_rbf_base_bandwidth(
    cache: RBFDistanceCache,
    median: bool = True,
    quantile: float | None = None,
) -> float | None:
    """Resolve the data-derived bandwidth shared by every point of an RBF sweep.

    Formula: ``base = median_{i>j} d_ij``, or ``quantile_q`` when ``quantile`` is
    given. With ``tol = cache.distance_tolerance``, a ``base`` at or below that
    floor is numerical residue rather than geometry, so it is replaced by
    ``median{d_ij : d_ij > tol}``, and by ``1`` when no distance clears it.

    The result excludes the per-point ``rbf_sigma`` scaling, so one call serves a
    whole ``rbf_sigma`` sweep instead of one median per point. ``None`` means no
    data-derived bandwidth applies and ``rbf_sigma`` is the bandwidth itself.

    The floor scales with the representation because the alternative cannot
    work: rows duplicated to within storage precision put the median at the
    quantization step, and whether that lands above or below any absolute
    constant depends only on how the representation happens to be scaled.
    """
    quantile = _validate_optional_quantile(quantile)
    if quantile is None and not median:
        return None
    lower_distances = cache.lower_distances
    if lower_distances.numel() == 0:
        return 1.0
    tolerance = cache.distance_tolerance
    if not math.isfinite(tolerance) or tolerance < 0.0:
        tolerance = 0.0
    if quantile is not None:
        bandwidth = torch.quantile(lower_distances, q=quantile).item()
    else:
        bandwidth = torch.median(lower_distances).item()
    if not math.isfinite(bandwidth) or bandwidth <= tolerance:
        positive = lower_distances[lower_distances > tolerance]
        bandwidth = torch.median(positive).item() if positive.numel() else 1.0
    return bandwidth


def rbf_kernel_from_distance_cache(
    cache: RBFDistanceCache,
    rbf_sigma: float = 1.0,
    median: bool = True,
    quantile: float | None = None,
    base_bandwidth: float | None = None,
) -> torch.Tensor:
    """Construct an RBF kernel from prepared pairwise distances.

    ``base_bandwidth`` accepts an already-resolved value from
    ``resolve_rbf_base_bandwidth``. That value is invariant across an
    ``rbf_sigma`` sweep, so supplying it skips one median per sweep point.
    Leaving it unset resolves the bandwidth exactly as before.
    """
    rbf_sigma = validate_positive_float(rbf_sigma, "rbf_sigma")
    quantile = _validate_optional_quantile(quantile)

    if base_bandwidth is None:
        base_bandwidth = resolve_rbf_base_bandwidth(
            cache,
            median=median,
            quantile=quantile,
        )
    bandwidth = rbf_sigma if base_bandwidth is None else base_bandwidth
    if median and quantile is None:
        bandwidth *= rbf_sigma
    # The exponent divides by this, so it must stay representable. Rejecting a
    # merely small bandwidth would instead reject legitimately small-scale data:
    # the median heuristic is scale-invariant, and the degenerate case it needs
    # protection from is handled relative to the data in
    # ``resolve_rbf_base_bandwidth``.
    denominator = 2 * bandwidth**2
    if not math.isfinite(bandwidth) or bandwidth <= 0.0 or denominator <= 0.0:
        raise ValueError(
            "RBF bandwidth must resolve to a finite positive value with a "
            f"representable square, got {bandwidth!r}."
        )
    return torch.exp(-cache.squared_distances / denominator)


def score_cka_kernels(K: Any, L: Any, unbiased: bool = False) -> float:
    """Score ``CKA(K, L) = HSIC(K,L) / sqrt(HSIC(K,K) HSIC(L,L))``."""
    K, L = validate_kernel_pair(
        K,
        L,
        metric_name="CKA",
        min_samples=4 if unbiased else 1,
    )
    hsic = hsic_unbiased if unbiased else hsic_biased
    return normalized_similarity_or_zero(
        "CKA",
        hsic(K, L),
        hsic(K, K),
        hsic(L, L),
    )


def cka(
    feats_A: Any,
    feats_B: Any,
    kernel_metric: str = "ip",
    rbf_sigma: float = 1.0,
    unbiased: bool = False,
    median: bool = True,
    quantile: float | None = None,
) -> float:
    """Compute centered kernel alignment between two representations.

    Formula: ``CKA(K,L) = HSIC(K,L) / sqrt(HSIC(K,K) HSIC(L,L))``.
    The linear kernels are ``K = X X^T`` and ``L = Y Y^T``. For RBF CKA,
    ``K_ij = exp(-||x_i-x_j||^2 / (2 sigma_X^2))`` and likewise for
    ``L``. ``HSIC`` is the centered Frobenius inner product by default and
    the finite-sample unbiased estimator when ``unbiased=True``.
    A zero or non-positive self-alignment returns ``0.0`` with a
    ``RuntimeWarning``.
    """
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="CKA",
        min_samples=4 if unbiased else 2,
    )
    if kernel_metric == "ip":
        K = torch.mm(feats_A, feats_A.T)
        L = torch.mm(feats_B, feats_B.T)
    elif kernel_metric == "rbf":
        K = compute_rbf_kernel(
            feats_A,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        )
        L = compute_rbf_kernel(
            feats_B,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        )
    else:
        raise ValueError(f"Invalid kernel metric {kernel_metric}")
    return score_cka_kernels(K, L, unbiased=unbiased)


def unbiased_cka(*args: Any, **kwargs: Any) -> float:
    """Compute unbiased centered kernel alignment.

    Formula: ``HSIC_u(K,L) / sqrt(HSIC_u(K,K) HSIC_u(L,L))``.
    """
    kwargs["unbiased"] = True
    return cka(*args, **kwargs)


def rbf_uka(
    feats_A: Any,
    feats_B: Any,
    rbf_sigma: float,
    median: bool = True,
    quantile: float | None = None,
) -> float:
    """Compute uncentered alignment between raw RBF kernels.

    Formula: ``score = <K,L>_F / (||K||_F ||L||_F)``, where
    ``K_ij = exp(-||x_i-x_j||^2 / (2 sigma_X^2))`` and likewise for ``L``.
    """
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="rbf_uka",
    )
    K = compute_rbf_kernel(
        feats_A,
        rbf_sigma=rbf_sigma,
        median=median,
        quantile=quantile,
    )
    L = compute_rbf_kernel(
        feats_B,
        rbf_sigma=rbf_sigma,
        median=median,
        quantile=quantile,
    )
    return normalized_similarity_or_zero(
        "RBF UKA",
        torch.sum(K * L),
        torch.sum(K * K),
        torch.sum(L * L),
    )


def svcca(feats_A: Any, feats_B: Any, cca_dim: int = 10) -> float:
    """Compute mean canonical correlation after singular-vector reduction.

    Formula: standardize feature columns, retain ``q=cca_dim`` leading left
    singular vectors ``U_X`` and ``U_Y``, apply CCA to obtain canonical
    variates ``(a_r, b_r)``, then return
    ``score = (1/q) sum_r corr(a_r, b_r)``.
    """
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="SVCCA",
    )
    cca_dim = validate_integer_in_range(
        cca_dim,
        name="SVCCA cca_dim",
        minimum=1,
        maximum=min(
            feats_A.shape[0] - 1,
            feats_A.shape[1],
            feats_B.shape[1],
        ),
    )

    def preprocess(feats: torch.Tensor) -> torch.Tensor:
        centered = feats - torch.mean(feats, axis=0)
        return centered / (torch.std(centered, axis=0) + 1e-8)

    singular_a = torch.linalg.svd(preprocess(feats_A), full_matrices=False).U[
        :, :cca_dim
    ]
    singular_b = torch.linalg.svd(preprocess(feats_B), full_matrices=False).U[
        :, :cca_dim
    ]
    singular_a = singular_a.cpu().detach().numpy()
    singular_b = singular_b.cpu().detach().numpy()
    cca = CCA(n_components=cca_dim)
    cca.fit(singular_a, singular_b)
    canonical_a, canonical_b = cca.transform(singular_a, singular_b)

    correlations: list[float] = []
    for index in range(cca_dim):
        source = canonical_a[:, index]
        target = canonical_b[:, index]
        if np.std(source) <= NORMALIZATION_EPS or np.std(target) <= NORMALIZATION_EPS:
            warnings.warn(
                "SVCCA produced a constant canonical variate; using zero "
                "correlation for that component.",
                RuntimeWarning,
                stacklevel=2,
            )
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(source, target)[0, 1]))
    score = float(np.mean(correlations))
    if not math.isfinite(score):
        raise ValueError("SVCCA produced a non-finite score.")
    return score
