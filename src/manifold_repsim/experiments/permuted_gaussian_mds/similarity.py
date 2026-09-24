"""Shared-core similarity computation for the MDS experiment."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

from manifold_repsim.metrics import compute_nearest_neighbors
from manifold_repsim.metrics import compute_rbf_kernel
from manifold_repsim.metrics import metric_feature_matrix
from manifold_repsim.metrics import prepare_rbf_kernel_for_metric

from manifold_repsim.experiments.mds import validate_dissimilarity_matrix

from .config import require


def _validate_similarity(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    require(
        values.ndim == 2 and values.shape[0] == values.shape[1],
        "Similarity matrix must be square",
    )
    require(np.isfinite(values).all(), "Similarity matrix contains non-finite values")
    values = (values + values.T) / 2
    require(values.max() <= 1 + 1e-7, "Similarity exceeds 1")
    return np.minimum(values, 1)


def _similarities(features: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Compute normalized feature-vector similarities in bounded blocks."""
    norms = np.sqrt(np.einsum("ij,ij->i", features, features))
    denominator = np.outer(norms, norms) + 1e-10
    result = np.empty((len(features), len(features)))
    for start in range(0, len(features), block_size):
        stop = min(start + block_size, len(features))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result[start:stop] = (features[start:stop] @ features.T) / denominator[
                start:stop
            ]
    return _validate_similarity(result)


@contextmanager
def _feature_maps(
    datasets: np.ndarray,
    names: tuple[str, ...],
    scratch: Path,
) -> Iterator[dict[str, np.memmap]]:
    scratch.mkdir(parents=True, exist_ok=True)
    shape = (len(datasets), datasets.shape[1] ** 2)
    paths = {name: scratch / f"{name}.mmap" for name in names}
    maps = {
        name: np.memmap(path, dtype=np.float64, mode="w+", shape=shape)
        for name, path in paths.items()
    }
    try:
        yield maps
    finally:
        maps.clear()
        for path in paths.values():
            path.unlink(missing_ok=True)


def compute_fixed_similarity_matrix(
    datasets: np.ndarray,
    specification: dict[str, Any],
    scratch: Path,
) -> np.ndarray:
    """Compute one configured fixed metric using reusable metric features."""
    if specification["name"] == "mutual_knn_dist":
        n_datasets, n_points, _ = datasets.shape
        topk = specification["kwargs"]["topk"]
        rows, columns = [], []
        for index, data in enumerate(datasets):
            neighbors = compute_nearest_neighbors(
                torch.from_numpy(data),
                topk,
                use_distance=True,
            ).numpy()
            rows.extend(np.full(neighbors.size, index))
            columns.extend(
                (
                    np.repeat(np.arange(n_points), topk) * n_points + neighbors.ravel()
                ).tolist()
            )
        masks = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, columns)),
            shape=(n_datasets, n_points**2),
        )
        return _validate_similarity((masks @ masks.T).toarray() / (n_points * topk))

    feature_kind = "cka_lin" if specification["name"] == "cka" else "cka_rbf"
    with _feature_maps(datasets, (feature_kind,), scratch) as maps:
        for index, data in enumerate(datasets):
            tensor = torch.from_numpy(data)
            metric_name = "cka"
            kernel = tensor @ tensor.T
            if feature_kind == "cka_rbf":
                metric_name = "cka_rbf"
                kernel = prepare_rbf_kernel_for_metric(
                    metric_name,
                    compute_rbf_kernel(
                        tensor,
                        rbf_sigma=specification["kwargs"]["rbf_sigma"],
                    ),
                )
            kernel = metric_feature_matrix(metric_name, kernel)
            maps[feature_kind][index] = kernel.numpy().ravel()
        maps[feature_kind].flush()
        return _similarities(maps[feature_kind])


def compute_signal_similarity_matrices(
    datasets: np.ndarray,
    metric_names: list[str],
    rbf_sigma: float,
    scratch: Path,
) -> dict[str, np.ndarray]:
    """Compute all signal metrics from one shared raw RBF kernel per dataset."""
    names = tuple(metric_names)
    with _feature_maps(datasets, names, scratch) as maps:
        for index, data in enumerate(datasets):
            raw_kernel = compute_rbf_kernel(
                torch.from_numpy(data),
                rbf_sigma=rbf_sigma,
            )
            for name in names:
                prepared_kernel = prepare_rbf_kernel_for_metric(name, raw_kernel)
                maps[name][index] = (
                    metric_feature_matrix(name, prepared_kernel).numpy().ravel()
                )
        for feature_map in maps.values():
            feature_map.flush()
        return {name: _similarities(feature_map) for name, feature_map in maps.items()}


def scalar_base_relative_distance(values: np.ndarray) -> np.ndarray:
    """Pairwise absolute distance between scalar base-relative scores."""
    values = np.asarray(values)
    return np.abs(values[:, None] - values[None, :])


def signal_base_relative_distance(signals: np.ndarray) -> np.ndarray:
    """Root-mean-square distance between base-relative metric signals."""
    require(
        signals.ndim == 2 and signals.shape[1] > 0,
        "base_signals must be a non-empty matrix",
    )
    return squareform(pdist(signals)) / np.sqrt(signals.shape[1])


def direct_similarity_distance(similarities: np.ndarray) -> np.ndarray:
    """Convert a normalized similarity matrix into dissimilarities."""
    return validate_dissimilarity_matrix(1 - _validate_similarity(similarities))
