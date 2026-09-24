"""Shared validation and fitting for metric multidimensional scaling."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS


def validate_dissimilarity_matrix(values: np.ndarray) -> np.ndarray:
    """Validate and symmetrize a finite nonnegative dissimilarity matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Dissimilarity matrix must be square")
    if not np.isfinite(values).all():
        raise ValueError("Dissimilarity contains non-finite values")
    values = (values + values.T) / 2
    if values.min(initial=0.0) < -1e-7:
        raise ValueError("Dissimilarity contains negative values")
    values = np.maximum(values, 0)
    np.fill_diagonal(values, 0)
    return values


def fit_metric_mds(
    values: np.ndarray,
    settings: Mapping[str, Any],
    n_components: int = 2,
) -> tuple[np.ndarray, float, float, int]:
    """Fit reproducible metric MDS and return embedding and stress metadata.

    ``settings`` may be the complete experiment configuration, containing an
    ``mds`` mapping, or the MDS keyword mapping itself.
    """
    values = validate_dissimilarity_matrix(values)
    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise TypeError("n_components must be an integer")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    if not values.any():
        return np.zeros((len(values), n_components)), 0.0, 0.0, 0

    mds_settings = settings.get("mds", settings)
    model = MDS(
        n_components=n_components,
        metric=True,
        dissimilarity="precomputed",
        **dict(mds_settings),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
            module=r"sklearn\.utils\.extmath",
        )
        embedding = model.fit_transform(values)
    if not (
        np.isfinite(embedding).all()
        and np.isfinite(model.stress_)
        and np.isfinite(model.n_iter_)
    ):
        raise ValueError("MDS returned non-finite results")
    upper = np.triu_indices_from(values, 1)
    denominator = np.square(values[upper]).sum()
    normalized_stress = np.sqrt(
        np.square(values[upper] - squareform(pdist(embedding))[upper]).sum()
        / denominator
    )
    return (
        embedding,
        float(model.stress_),
        float(normalized_stress),
        int(model.n_iter_),
    )
