"""Exact ordinal-agreement counting for the triplet and quadruplet indices.

Both indices reduce to one question asked of two paired value sequences: what
share of their pairs is ordered the same way, where a tie in both sequences
counts as agreeing. Everything here is exact integer arithmetic on dense ranks;
floating-point equality is decided once, in ``tolerant_ranks``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .utils import validate_representation

# A comparison value is a sum over the feature dimension, so its float64 error
# is bounded by dimension * eps * magnitude. Two values closer than twice that
# bound cannot be told apart, and the factor leaves the same margin again.
TIE_ERROR_MARGIN = 4.0

# Row-chunk budget for the per-anchor counting in TSI. It bounds the temporary
# (rows x n) integer matrices and changes no result.
CHUNK_ENTRIES = 2_000_000


def ordinal_dissimilarity(
    feats: Any,
    use_distance: bool = True,
) -> tuple[torch.Tensor, float]:
    """Return the pairwise comparison matrix and its tie tolerance.

    With ``use_distance`` the matrix holds Euclidean distances, computed from
    row differences. The ``|x|^2 + |y|^2 - 2<x,y>`` expansion is avoided
    because its cancellation turns the zero distance between duplicate rows
    into noise, which would dissolve exactly the ties this index counts.
    Otherwise the matrix holds inner products, which order points the opposite
    way round; both spaces must use the same convention, and then agreement is
    unaffected by that reversal.

    The tolerance is the float64 accumulation error of one entry, so it names
    the separation below which two entries are one tie. It does not depend on
    the input dtype: float32 rows convert to float64 exactly.
    """
    feats = validate_representation(feats, name="ordinal representation").double()
    if use_distance:
        matrix = torch.cdist(feats, feats, compute_mode="donot_use_mm_for_euclid_dist")
    else:
        matrix = feats @ feats.T
    epsilon = float(np.finfo(np.float64).eps)
    scale = float(matrix.abs().max())
    return matrix, TIE_ERROR_MARGIN * feats.shape[1] * epsilon * scale


def tolerant_ranks(values: torch.Tensor, tolerance: float) -> torch.Tensor:
    """Dense ranks from 1 along the last axis, merging values within ``tolerance``.

    Consecutive sorted values closer than ``tolerance`` share a rank, so a tie
    survives accumulation-order noise. Chains of near values merge into one
    tie, which is harmless only because the tolerance is as small as float64
    error: it must never be sized to the data's spread or to its input dtype.
    """
    ordered, order = torch.sort(values, dim=-1)
    new_group = torch.ones_like(ordered, dtype=torch.long)
    new_group[..., 1:] = (ordered[..., 1:] - ordered[..., :-1] > tolerance).long()
    ranks = torch.empty_like(new_group)
    ranks.scatter_(-1, order, new_group.cumsum(dim=-1))
    return ranks


def _tied_pairs(sorted_values: torch.Tensor) -> torch.Tensor:
    """Per-row count of equal pairs in a row-sorted integer matrix."""
    rows, size = sorted_values.shape
    position = torch.arange(size, device=sorted_values.device).expand(rows, size)
    run_start = torch.ones_like(sorted_values, dtype=torch.bool)
    run_start[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    start = torch.where(run_start, position, torch.zeros_like(position))
    return (position - start.cummax(dim=1).values).sum(dim=1)


def _strict_inversions(values: torch.Tensor) -> torch.Tensor:
    """Per-row count of pairs ``p < q`` with ``values[p] > values[q]``.

    Radix inversion counting: at each bit, among elements sharing all higher
    bits, an inversion is a one-bit element before a zero-bit one. Elements are
    then stably partitioned by that bit, keeping each group contiguous.
    """
    rows, size = values.shape
    total = torch.zeros(rows, dtype=torch.long, device=values.device)
    current = values
    for level in range(int(size).bit_length() - 1, -1, -1):
        prefix = current >> (level + 1)
        bit = (current >> level) & 1
        first = torch.ones_like(prefix, dtype=torch.bool)
        first[:, 1:] = prefix[:, 1:] != prefix[:, :-1]
        ones_before = bit.cumsum(dim=1) - bit
        group_base = torch.where(first, ones_before, torch.zeros_like(bit))
        group_base = group_base.cummax(dim=1).values
        total += ((ones_before - group_base) * (1 - bit)).sum(dim=1)
        if level:
            order = torch.sort(current >> level, dim=1, stable=True).indices
            current = torch.gather(current, 1, order)
    return total


def agreeing_pairs(
    ranks_a: torch.Tensor,
    ranks_b: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Count, per row, unordered pairs whose order agrees, joint ties included.

    Returns ``(C + T_AB, M (M - 1) / 2)``: strictly concordant plus jointly tied
    pairs, and all pairs. A tie in only one sequence is a disagreement, and no
    pair leaves the total. Ranks must be integers in ``[1, M]``, which is what
    ``tolerant_ranks`` produces.
    """
    size = ranks_a.shape[1]
    total = size * (size - 1) // 2
    key = ranks_a * (size + 1) + ranks_b
    sorted_key, order = torch.sort(key, dim=1)
    ties_joint = _tied_pairs(sorted_key)
    ties_a = _tied_pairs(sorted_key // (size + 1))
    ties_b = _tied_pairs(torch.sort(ranks_b, dim=1).values)
    discordant = _strict_inversions(torch.gather(ranks_b, 1, order))
    concordant = total - discordant - ties_a - ties_b + ties_joint
    return concordant + ties_joint, total


def _without_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n = matrix.shape[0]
    keep = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return matrix[keep].reshape(n, n - 1)


def score_triplet_matrices(
    matrix_a: torch.Tensor,
    tolerance_a: float,
    matrix_b: torch.Tensor,
    tolerance_b: float,
) -> float:
    """Score TSI from two comparison matrices, averaging over every anchor."""
    n = matrix_a.shape[0]
    rows_a = _without_diagonal(matrix_a)
    rows_b = _without_diagonal(matrix_b)
    chunk = max(1, CHUNK_ENTRIES // max(n, 1))
    agreement = 0
    for start in range(0, n, chunk):
        agreeing, total = agreeing_pairs(
            tolerant_ranks(rows_a[start : start + chunk], tolerance_a),
            tolerant_ranks(rows_b[start : start + chunk], tolerance_b),
        )
        agreement += int(agreeing.sum())
    return agreement / (n * total)


def score_quadruplet_matrices(
    matrix_a: torch.Tensor,
    tolerance_a: float,
    matrix_b: torch.Tensor,
    tolerance_b: float,
) -> float:
    """Score QSI from two symmetric comparison matrices.

    The all-pairs agreement mixes triplet and quadruplet comparisons in fixed
    proportions, so TSI is needed to isolate the quadruplet part.
    """
    n = matrix_a.shape[0]
    triplet = score_triplet_matrices(matrix_a, tolerance_a, matrix_b, tolerance_b)
    upper = torch.triu_indices(n, n, offset=1, device=matrix_a.device)
    values_a = matrix_a[upper[0], upper[1]].unsqueeze(0)
    values_b = matrix_b[upper[0], upper[1]].unsqueeze(0)
    agreeing, total = agreeing_pairs(
        tolerant_ranks(values_a, tolerance_a),
        tolerant_ranks(values_b, tolerance_b),
    )
    return ((n + 1) * (int(agreeing[0]) / total) - 4 * triplet) / (n - 3)


__all__ = [
    "agreeing_pairs",
    "ordinal_dissimilarity",
    "score_quadruplet_matrices",
    "score_triplet_matrices",
    "tolerant_ranks",
]
