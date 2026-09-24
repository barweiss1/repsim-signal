"""Permutation-null calibration for pointwise and aggregate sweep scores."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from manifold_repsim.metrics import prepare_metric_pair
from manifold_repsim.metrics import score_prepared_metric
from manifold_repsim.metrics import supports_prepared_metric
from manifold_repsim.metrics.interface import AlignmentMetrics

from .signal_aggregation import aggregate_signal_auc
from .signal_aggregation import aggregate_signal_variance_weighted_auc


@dataclass
class PermutationCalibrationResult:
    """Observed-score calibration and the finite permutation null arrays."""

    null_scores: np.ndarray
    null_mean: np.ndarray
    null_std: np.ndarray
    p_values: np.ndarray
    tau_alpha: np.ndarray
    calibrated_scores: np.ndarray
    alpha: float
    n_permutations: int
    seed: int | None = None


def make_permutation_indices(
    n_points: int,
    n_permutations: int,
    seed: int | None = None,
    device=None,
) -> torch.Tensor:
    """Create reproducible row permutations on the requested device."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    device = torch.device("cpu" if device is None else device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
    return torch.stack(
        [
            torch.randperm(n_points, generator=generator, device=device)
            for _ in range(n_permutations)
        ]
    )


def compute_permutation_calibration(
    observed_scores,
    null_scores,
    alpha: float = 0.05,
    eps: float = 1e-12,
) -> PermutationCalibrationResult:
    """Calibrate scores with a nearest-rank right-tail permutation null."""
    observed_scores = np.asarray(observed_scores, dtype=float)
    null_scores = np.asarray(null_scores, dtype=float)
    if null_scores.ndim != observed_scores.ndim + 1:
        raise ValueError(
            "null_scores must have one more dimension than observed_scores."
        )
    if null_scores.shape[:-1] != observed_scores.shape:
        raise ValueError("null_scores leading dimensions must match observed_scores.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    n_permutations = null_scores.shape[-1]
    order_index = int(np.ceil((1.0 - alpha) * n_permutations)) - 1
    order_index = int(np.clip(order_index, 0, n_permutations - 1))
    tau_alpha = np.sort(null_scores, axis=-1)[..., order_index]
    p_values = (1.0 + np.sum(null_scores >= observed_scores[..., None], axis=-1)) / (
        n_permutations + 1.0
    )
    calibrated_scores = np.maximum(
        0.0,
        (observed_scores - tau_alpha) / np.maximum(1.0 - tau_alpha, eps),
    )
    return PermutationCalibrationResult(
        null_scores=null_scores,
        null_mean=np.mean(null_scores, axis=-1),
        null_std=np.std(null_scores, axis=-1),
        p_values=p_values,
        tau_alpha=tau_alpha,
        calibrated_scores=np.clip(calibrated_scores, 0.0, 1.0),
        alpha=alpha,
        n_permutations=n_permutations,
    )


def measure_with_permutation_null(
    metric_name,
    feats_A,
    feats_B,
    kwargs,
    permutation_indices,
):
    """Score a metric once observed and against each target permutation."""
    null_scores = np.zeros(permutation_indices.shape[0], dtype=float)
    if supports_prepared_metric(metric_name):
        prepared = prepare_metric_pair(metric_name, feats_A, feats_B, **kwargs)
        observed_score = score_prepared_metric(prepared)
        for index, permutation in enumerate(permutation_indices):
            null_scores[index] = score_prepared_metric(
                prepared, target_permutation=permutation
            )
        return observed_score, null_scores

    observed_score = AlignmentMetrics.measure(
        feats_A=feats_A, feats_B=feats_B, metric=metric_name, **kwargs
    )
    for index, permutation in enumerate(permutation_indices):
        null_scores[index] = AlignmentMetrics.measure(
            feats_A=feats_A,
            feats_B=feats_B[permutation],
            metric=metric_name,
            **kwargs,
        )
    return observed_score, null_scores


def compute_aggregate_permutation_calibration(
    param_values,
    observed_grid,
    null_score_grid,
    aggregate_kind,
    integration_method: str = "trapezoidal",
    alpha: float = 0.05,
) -> PermutationCalibrationResult:
    """Calibrate AUCs after aggregating each observed and null curve."""
    observed_grid = np.asarray(observed_grid, dtype=float)
    null_score_grid = np.asarray(null_score_grid, dtype=float)
    if observed_grid.ndim != 2:
        raise ValueError("observed_grid must be a two-dimensional array.")
    if null_score_grid.ndim != 3:
        raise ValueError("null_score_grid must be a three-dimensional array.")
    if null_score_grid.shape[:2] != observed_grid.shape:
        raise ValueError("null_score_grid leading dimensions must match observed_grid.")
    if aggregate_kind not in {"auc", "variance_weighted_auc", "var_auc"}:
        raise ValueError(f"Unsupported aggregate_kind: {aggregate_kind}")

    if aggregate_kind == "auc":
        aggregate = aggregate_signal_auc
        weights = None
        observed = aggregate(
            param_values, observed_grid, integration_method=integration_method
        )
    else:
        aggregate = aggregate_signal_variance_weighted_auc
        weights = np.var(observed_grid, axis=0)
        observed = aggregate(
            param_values,
            observed_grid,
            integration_method=integration_method,
            weights=weights,
        )

    null_aggregates = np.zeros(
        (observed_grid.shape[0], null_score_grid.shape[-1]), dtype=float
    )
    for index in range(null_score_grid.shape[-1]):
        kwargs = {"weights": weights} if weights is not None else {}
        null_aggregates[:, index] = aggregate(
            param_values,
            null_score_grid[:, :, index],
            integration_method=integration_method,
            **kwargs,
        )
    return compute_permutation_calibration(observed, null_aggregates, alpha=alpha)
