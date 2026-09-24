"""Nearest-neighbor construction and neighborhood alignment metrics."""

from __future__ import annotations

from typing import Any

import torch

from .kernels import hsic_biased
from .kernels import hsic_unbiased
from .utils import normalized_similarity_or_zero
from .utils import validate_representation
from .utils import validate_representation_pair
from .utils import validate_topk


def compute_nearest_neighbors(
    feats: Any,
    topk: int = 1,
    use_distance: bool = False,
) -> torch.Tensor:
    """Return nonself neighbor indices by inner product or squared distance."""
    feats = validate_representation(feats, name="nearest-neighbor representation")
    topk = validate_topk(
        topk,
        feats.shape[0],
        metric_name="nearest-neighbor search",
    )
    if use_distance:
        norms = (feats**2).sum(dim=1, keepdim=True)
        distances = norms + norms.T - 2 * (feats @ feats.T)
        return torch.topk(
            distances.fill_diagonal_(1e8),
            k=topk,
            dim=1,
            largest=False,
        ).indices
    similarities = (feats @ feats.T).fill_diagonal_(-1e8)
    return torch.topk(similarities, k=topk, dim=1, largest=True).indices


def compute_knn_accuracy(knn: torch.Tensor) -> torch.Tensor:
    """Return the fraction of rows whose composed neighbors contain that row."""
    num_samples = knn.shape[0]
    matches = knn == torch.arange(num_samples, device=knn.device).view(-1, 1, 1)
    return matches.float().view(num_samples, -1).max(dim=1).values.mean()


def score_cycle_neighbor_rankings(
    neighbors_a: torch.Tensor,
    neighbors_b: torch.Tensor,
    topk: int,
) -> float:
    """Score cycle consistency from reusable full neighbor rankings."""
    return compute_knn_accuracy(neighbors_a[:, :topk][neighbors_b[:, :topk]]).item()


def _dense_neighbor_mask(
    neighbors: torch.Tensor,
    topk: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    mask = torch.zeros(
        neighbors.shape[0],
        neighbors.shape[0],
        dtype=dtype,
        device=neighbors.device,
    )
    return mask.scatter_(1, neighbors[:, :topk], 1)


def score_mutual_neighbor_rankings(
    neighbors_a: torch.Tensor,
    neighbors_b: torch.Tensor,
    topk: int,
) -> float:
    """Score mutual-neighbor overlap from reusable full rankings."""
    mask_a = _dense_neighbor_mask(neighbors_a, topk)
    mask_b = _dense_neighbor_mask(neighbors_b, topk)
    return ((mask_a * mask_b).sum(dim=1) / topk).mean().item()


def cycle_knn(feats_A: Any, feats_B: Any, topk: int) -> float:
    """Compute directed nearest-neighbor cycle consistency.

    Formula: for the ``topk`` neighbor sets ``N_A(i)`` and ``N_B(i)``,
    ``score = (1/n) sum_i 1[i in union_{j in N_B(i)} N_A(j)]``.
    Neighbors are selected by inner-product similarity with self-neighbors
    excluded.
    """
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="cycle_knn",
    )
    topk = validate_topk(topk, feats_A.shape[0], "cycle_knn")
    neighbors_a = compute_nearest_neighbors(feats_A, topk)
    neighbors_b = compute_nearest_neighbors(feats_B, topk)
    return score_cycle_neighbor_rankings(neighbors_a, neighbors_b, topk)


def mutual_knn(
    feats_A: Any,
    feats_B: Any,
    topk: int,
    use_distance: bool = False,
) -> float:
    """Compute mean overlap between corresponding nearest-neighbor sets.

    Formula: ``score = (1/n) sum_i |N_A(i) intersect N_B(i)| / k``.
    ``N_A`` and ``N_B`` use inner products by default and squared Euclidean
    distance when ``use_distance=True``; self-neighbors are excluded.
    """
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="mutual_knn",
    )
    topk = validate_topk(topk, feats_A.shape[0], "mutual_knn")
    neighbors_a = compute_nearest_neighbors(
        feats_A,
        topk,
        use_distance=use_distance,
    )
    neighbors_b = compute_nearest_neighbors(
        feats_B,
        topk,
        use_distance=use_distance,
    )
    return score_mutual_neighbor_rankings(neighbors_a, neighbors_b, topk)


def score_cknna_rankings(
    gram_a: torch.Tensor,
    gram_b: torch.Tensor,
    neighbors_a: torch.Tensor,
    neighbors_b: torch.Tensor,
    topk: int,
    distance_agnostic: bool = False,
    unbiased: bool = True,
) -> float:
    """Score CKNNA from reusable Gram matrices and neighbor rankings."""

    def similarity(
        left: torch.Tensor,
        right: torch.Tensor,
        left_neighbors: torch.Tensor,
        right_neighbors: torch.Tensor,
    ) -> torch.Tensor:
        mask = _dense_neighbor_mask(
            left_neighbors,
            topk,
            dtype=left.dtype,
        ) * _dense_neighbor_mask(
            right_neighbors,
            topk,
            dtype=right.dtype,
        )
        if distance_agnostic:
            return mask.sum()
        if unbiased:
            return hsic_unbiased(mask * left, mask * right)
        return hsic_biased(mask * left, mask * right)

    sim_ab = similarity(gram_a, gram_b, neighbors_a, neighbors_b)
    sim_aa = similarity(gram_a, gram_a, neighbors_a, neighbors_a)
    sim_bb = similarity(gram_b, gram_b, neighbors_b, neighbors_b)
    return normalized_similarity_or_zero("CKNNA", sim_ab, sim_aa, sim_bb)


def cknna(
    feats_A: Any,
    feats_B: Any,
    topk: int | None = None,
    distance_agnostic: bool = False,
    unbiased: bool = True,
) -> float:
    """Compute CKA after masking Gram matrices to shared neighborhoods.

    Formula: let ``G_X = X X^T`` and ``M_X`` mark each row's ``k`` largest
    entries (excluding the diagonal when unbiased). For the cross term use
    ``M_XY = M_X elementwise_product M_Y`` and
    ``h_XY = HSIC(M_XY elementwise_product G_X,
    M_XY elementwise_product G_Y)``. Self terms use their corresponding
    single-representation masks, and
    ``score = h_XY / sqrt(h_XX h_YY)``.
    With ``distance_agnostic=True``, each ``h`` is instead the number of
    selected mask entries, reducing the score to normalized neighborhood
    overlap.
    """
    min_samples = 4 if unbiased else 3
    feats_A, feats_B = validate_representation_pair(
        feats_A,
        feats_B,
        metric_name="CKNNA",
        min_samples=min_samples,
    )
    num_samples = feats_A.shape[0]
    if topk is None:
        topk = num_samples - 1
    topk = validate_topk(topk, num_samples, "CKNNA", minimum=2)
    gram_a = feats_A @ feats_A.T
    gram_b = feats_B @ feats_B.T
    if unbiased:
        ranking_a = gram_a.clone().fill_diagonal_(float("-inf"))
        ranking_b = gram_b.clone().fill_diagonal_(float("-inf"))
    else:
        ranking_a, ranking_b = gram_a, gram_b
    neighbors_a = torch.topk(ranking_a, topk, dim=1).indices
    neighbors_b = torch.topk(ranking_b, topk, dim=1).indices
    return score_cknna_rankings(
        gram_a,
        gram_b,
        neighbors_a,
        neighbors_b,
        topk,
        distance_agnostic=distance_agnostic,
        unbiased=unbiased,
    )
