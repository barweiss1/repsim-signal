"""Random-walk kernel construction, preparation, and alignment metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .kernels import compute_rbf_kernel
from .kernels import compute_softmax_kernel
from .kernels import score_cka_kernels
from .utils import normalized_similarity_or_zero
from .utils import validate_kernel_pair
from .utils import validate_representation_pair


@dataclass
class RWKADiffusionCache:
    """Eigendecomposition and degree factors for powering one RBF random walk."""

    eigvals: torch.Tensor
    eigvecs: torch.Tensor
    degree_sqrt: torch.Tensor
    degree_inv_sqrt: torch.Tensor


def score_rwka_kernels(
    K: Any,
    L: Any,
    metric_name: str,
) -> float:
    """Score ``<K,L>_F / (||K||_F ||L||_F)`` for prepared kernels."""
    K, L = validate_kernel_pair(
        K,
        L,
        metric_name=metric_name,
    )
    return normalized_similarity_or_zero(
        metric_name,
        torch.sum(K * L),
        torch.sum(K * K),
        torch.sum(L * L),
    )


def normalize_random_walk_kernel(
    K: torch.Tensor,
    symmetric: bool = False,
) -> torch.Tensor:
    """Row-normalize a kernel, or apply symmetric degree normalization."""
    if symmetric:
        degree = torch.sum(K, dim=0)
        degree = torch.where(degree == 0, torch.ones_like(degree), degree)
        inv_sqrt_degree = degree.rsqrt()
        return inv_sqrt_degree[:, None] * K * inv_sqrt_degree[None, :]
    row_sum = torch.sum(K, dim=1, keepdim=True)
    row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
    return K / row_sum


def degree_center_random_walk_kernel(K: torch.Tensor) -> torch.Tensor:
    """Symmetrically normalize a kernel and project out its degree-sqrt mode."""
    degree = torch.sum(K, dim=1)
    degree = torch.where(degree == 0, torch.ones_like(degree), degree)
    degree_sqrt = torch.sqrt(degree)
    mode = degree_sqrt / torch.linalg.vector_norm(degree_sqrt)
    normalized = degree.rsqrt()[:, None] * K * degree.rsqrt()[None, :]
    normalized_mode = normalized @ mode
    mode_normalized = mode @ normalized
    mode_energy = mode @ normalized_mode
    return (
        normalized
        - mode[:, None] * mode_normalized[None, :]
        - normalized_mode[:, None] * mode[None, :]
        + mode_energy * mode[:, None] * mode[None, :]
    )


def compute_rbf_rwka_diffusion_cache(
    feats: Any,
    rbf_sigma: float = 0.2,
    median: bool = True,
    quantile: float | None = None,
) -> RWKADiffusionCache:
    """Prepare a symmetric eigendecomposition for RBF random-walk powers."""
    kernel = compute_rbf_kernel(
        feats,
        rbf_sigma=rbf_sigma,
        median=median,
        quantile=quantile,
    )
    degree = torch.sum(kernel, dim=1)
    degree = torch.where(degree == 0, torch.ones_like(degree), degree)
    degree_sqrt = torch.sqrt(degree)
    degree_inv_sqrt = degree.rsqrt()
    symmetric_kernel = degree_inv_sqrt[:, None] * kernel * degree_inv_sqrt[None, :]
    eigvals, eigvecs = torch.linalg.eigh(symmetric_kernel)
    return RWKADiffusionCache(
        eigvals=torch.clamp(eigvals, min=0.0),
        eigvecs=eigvecs,
        degree_sqrt=degree_sqrt,
        degree_inv_sqrt=degree_inv_sqrt,
    )


def power_rbf_rwka_diffusion_cache(
    cache: RWKADiffusionCache,
    diffusion_time: float,
) -> torch.Tensor:
    """Construct a possibly fractional power of a cached RBF random walk."""
    diffusion_time = float(diffusion_time)
    if diffusion_time <= 0:
        raise ValueError("diffusion_time must be positive.")
    eigvals_power = cache.eigvals.pow(diffusion_time)
    right = cache.degree_sqrt[:, None] * cache.eigvecs
    return (cache.degree_inv_sqrt[:, None] * cache.eigvecs) @ (
        eigvals_power[:, None] * right.T
    )


def softmax_rwka(
    feats_A: Any,
    feats_B: Any,
    temperature: float,
    unbiased: bool = False,
    range_based: bool = True,
) -> float:
    """Compute RWKA from row-stochastic softmax kernels.

    Formula: ``P_X[i,j] = exp(<x_i,x_j>/tau_X) /
    sum_l exp(<x_i,x_l>/tau_X)`` and
    ``score = <P_X,P_Y>_F / (||P_X||_F ||P_Y||_F)``. With range scaling,
    ``tau_X = temperature * (max(X X^T) - min(X X^T))``; otherwise the
    shifted inner-product median supplies the scale. ``unbiased`` is
    retained for compatibility and is not used. A zero data-derived scale
    falls back to a finite unit scale, yielding a uniform kernel for
    constant inputs.
    """
    del unbiased
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="softmax_rwka",
    )
    kernel_a = compute_softmax_kernel(
        feats_A,
        temperature,
        range_based=range_based,
    )
    kernel_b = compute_softmax_kernel(
        feats_B,
        temperature,
        range_based=range_based,
    )
    return score_rwka_kernels(kernel_a, kernel_b, metric_name="Softmax RWKA")


def rbf_rwka(
    feats_A: Any,
    feats_B: Any,
    rbf_sigma: float,
    unbiased: bool = False,
    median: bool = True,
    quantile: float | None = None,
) -> float:
    """Compute RWKA from row-stochastic RBF kernels.

    Formula: ``K_ij = exp(-||x_i-x_j||^2 / (2 sigma_X^2))``,
    ``P_X = D_K^(-1) K``, and
    ``score = <P_X,P_Y>_F / (||P_X||_F ||P_Y||_F)``. With the median
    heuristic, ``sigma_X = rbf_sigma * median_{i>j} ||x_i-x_j||``; a
    supplied quantile takes precedence. ``unbiased`` is retained for
    compatibility and is not used.
    """
    del unbiased
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="rbf_rwka",
    )
    kernel_a = normalize_random_walk_kernel(
        compute_rbf_kernel(
            feats_A,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        )
    )
    kernel_b = normalize_random_walk_kernel(
        compute_rbf_kernel(
            feats_B,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        )
    )
    return score_rwka_kernels(kernel_a, kernel_b, metric_name="RBF RWKA")


def rbf_rwka_symmetric(
    feats_A: Any,
    feats_B: Any,
    rbf_sigma: float,
    unbiased: bool = False,
    median: bool = True,
    quantile: float | None = None,
) -> float:
    """Compute alignment of symmetrically normalized RBF kernels.

    Formula: with RBF kernel ``K`` and degree matrix ``D_K``, let
    ``S_K = D_K^(-1/2) K D_K^(-1/2)``. Then
    ``score = <S_K,S_L>_F / (||S_K||_F ||S_L||_F)``. ``unbiased`` is
    retained for compatibility and is not used.
    """
    del unbiased
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="rbf_rwka_symmetric",
    )
    kernel_a = normalize_random_walk_kernel(
        compute_rbf_kernel(
            feats_A,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        ),
        symmetric=True,
    )
    kernel_b = normalize_random_walk_kernel(
        compute_rbf_kernel(
            feats_B,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        ),
        symmetric=True,
    )
    return score_rwka_kernels(
        kernel_a,
        kernel_b,
        metric_name="Symmetric RBF RWKA",
    )


def rbf_crwka(
    feats_A: Any,
    feats_B: Any,
    rbf_sigma: float,
    median: bool = True,
    quantile: float | None = None,
) -> float:
    """Compute centered alignment of symmetric RBF random-walk kernels.

    Formula: with RBF kernel ``K``, degree matrix ``D_K``, and
    ``S_K = D_K^(-1/2) K D_K^(-1/2)``, the score is
    ``<H S_K H, H S_L H>_F / (||H S_K H||_F ||H S_L H||_F)``, where
    ``H = I - 11^T/n``.
    """
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="rbf_crwka",
    )
    kernel_a = normalize_random_walk_kernel(
        compute_rbf_kernel(
            feats_A,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        ),
        symmetric=True,
    )
    kernel_b = normalize_random_walk_kernel(
        compute_rbf_kernel(
            feats_B,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        ),
        symmetric=True,
    )
    return score_cka_kernels(kernel_a, kernel_b, unbiased=False)


def rbf_degree_crwka(
    feats_A: Any,
    feats_B: Any,
    rbf_sigma: float,
    median: bool = True,
    quantile: float | None = None,
) -> float:
    """Align symmetric RBF kernels after removing their degree modes.

    Formula: let ``S_K = D_K^(-1/2) K D_K^(-1/2)``,
    ``v_K = sqrt(d_K) / ||sqrt(d_K)||_2``, ``Pi_K = I - v_K v_K^T``, and
    ``C_K = Pi_K S_K Pi_K``. The score is
    ``<C_K,C_L>_F / (||C_K||_F ||C_L||_F)``.
    """
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="rbf_degree_crwka",
    )
    kernel_a = degree_center_random_walk_kernel(
        compute_rbf_kernel(
            feats_A,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        )
    )
    kernel_b = degree_center_random_walk_kernel(
        compute_rbf_kernel(
            feats_B,
            rbf_sigma=rbf_sigma,
            median=median,
            quantile=quantile,
        )
    )
    return score_rwka_kernels(
        kernel_a,
        kernel_b,
        metric_name="Degree-centered RBF RWKA",
    )


def rbf_rwka_diffusion_time(
    feats_A: Any,
    feats_B: Any,
    diffusion_time: float = 1.0,
    rbf_sigma: float = 0.2,
    unbiased: bool = False,
    median: bool = True,
    quantile: float | None = None,
) -> float:
    """Compute RWKA between diffusion powers of RBF random walks.

    Formula: for ``P_X = D_K^(-1) K`` and diffusion time ``t``,
    ``score = <P_X^t,P_Y^t>_F / (||P_X^t||_F ||P_Y^t||_F)``. Fractional
    powers use ``S_K = D_K^(-1/2) K D_K^(-1/2) = U Lambda U^T`` and
    ``P_X^t = D_K^(-1/2) U Lambda^t U^T D_K^(1/2)``. Negative numerical
    eigenvalues are clamped to zero. ``unbiased`` is retained for
    compatibility and is not used.
    """
    del unbiased
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="rbf_rwka_diffusion_time",
    )
    cache_a = compute_rbf_rwka_diffusion_cache(
        feats_A,
        rbf_sigma=rbf_sigma,
        median=median,
        quantile=quantile,
    )
    cache_b = compute_rbf_rwka_diffusion_cache(
        feats_B,
        rbf_sigma=rbf_sigma,
        median=median,
        quantile=quantile,
    )
    return score_rwka_kernels(
        power_rbf_rwka_diffusion_cache(cache_a, diffusion_time),
        power_rbf_rwka_diffusion_cache(cache_b, diffusion_time),
        metric_name="RBF RWKA diffusion-time",
    )
