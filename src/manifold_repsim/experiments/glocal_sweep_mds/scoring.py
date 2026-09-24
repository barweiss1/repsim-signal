"""Exact metric signals and batch-major MDS/PCA vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from manifold_repsim.experiments.mds import fit_metric_mds
from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.metrics import prepare_metric_curve, prepare_metric_pair
from manifold_repsim.metrics import score_prepared_curve, score_prepared_metric
from manifold_repsim.metrics import supports_prepared_metric

from ..artifacts import load_npz, save_npz
from .config import fingerprint, input_root
from .discovery import DatasetGroup, load_aligned_pair


def concatenate_batch_scores(scores: np.ndarray) -> np.ndarray:
    """Flatten condition-by-batch-by-parameter scores in batch-major order."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 3 or not np.isfinite(scores).all():
        raise ValueError(
            "scores must be a finite condition-by-batch-by-parameter array"
        )
    return scores.reshape(scores.shape[0], -1)


def score_distances(vectors: np.ndarray) -> np.ndarray:
    """Return direct Euclidean distances between concatenated score vectors."""
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] < 2 or not np.isfinite(vectors).all():
        raise ValueError("vectors must be a finite 2D array with at least two rows")
    return squareform(pdist(vectors, metric="euclidean"))


def fit_signal_pca(
    vectors: np.ndarray, n_components: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit deterministic PCA to raw concatenated signals without scaling.

    Returns the projected coordinates, principal axes, explained-variance
    ratios, and feature mean. Outputs are zero-padded when the selected signal
    matrix has fewer dimensions than ``n_components``.
    """
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] < 2 or not np.isfinite(vectors).all():
        raise ValueError("vectors must be a finite 2D array with at least two rows")
    if not isinstance(n_components, int) or isinstance(n_components, bool):
        raise ValueError("n_components must be a positive integer")
    if n_components < 1:
        raise ValueError("n_components must be a positive integer")

    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    fitted_components = min(n_components, len(singular_values))
    components = right_vectors[:fitted_components].copy()
    embedding = centered @ components.T

    # Resolve the arbitrary SVD sign so regenerated artifacts are stable.
    for index, component in enumerate(components):
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            components[index] *= -1
            embedding[:, index] *= -1

    total_variance = float(np.square(singular_values).sum())
    explained = np.zeros(fitted_components, dtype=float)
    if total_variance > 0:
        explained = np.square(singular_values[:fitted_components]) / total_variance

    if fitted_components < n_components:
        embedding = np.pad(embedding, ((0, 0), (0, n_components - fitted_components)))
        components = np.pad(components, ((0, n_components - fitted_components), (0, 0)))
        explained = np.pad(explained, (0, n_components - fitted_components))
    return embedding, components, explained, mean


def _score_pair(
    metric: dict[str, Any],
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    source_tensor = torch.from_numpy(source)
    target_tensor = torch.from_numpy(target)
    if metric["mode"] == "signal":
        prepared = prepare_metric_curve(
            metric["name"],
            source_tensor,
            target_tensor,
            metric["parameter_name"],
            metric["parameter_values"],
            **metric["kwargs"],
        )
        return np.asarray(score_prepared_curve(prepared), dtype=float)
    if supports_prepared_metric(metric["name"]):
        prepared = prepare_metric_pair(
            metric["name"], source_tensor, target_tensor, **metric["kwargs"]
        )
        value = score_prepared_metric(prepared)
    else:
        value = AlignmentMetrics.measure(
            metric["name"], source_tensor, target_tensor, **metric["kwargs"]
        )
    return np.asarray([value], dtype=float)


def _checkpoint_fingerprint(
    config: dict[str, Any], group: DatasetGroup, metric: dict[str, Any]
) -> str:
    return fingerprint(
        {
            "version": 1,
            "source": group.source_fingerprint,
            "conditions": [condition.record() for condition in group.conditions],
            "batches": group.batch_indices,
            "metric": metric,
        }
    )


def _condition_scores(
    config: dict[str, Any],
    group: DatasetGroup,
    metric: dict[str, Any],
    condition_index: int,
    checkpoint_dir: Path,
    checkpoint_fingerprint: str,
    progress: Any | None = None,
) -> np.ndarray:
    checkpoint = checkpoint_dir / f"{condition_index:04d}.npz"
    condition = group.conditions[condition_index]
    expected_shape = (len(group.batch_indices), len(metric["parameter_values"]))
    if checkpoint.exists() and not config["execution"]["recompute_scores"]:
        saved = load_npz(checkpoint)
        if (
            saved.get("fingerprint", np.array("")).item() == checkpoint_fingerprint
            and saved.get("condition_name", np.array("")).item() == condition.name
            and saved.get("scores", np.empty(0)).shape == expected_shape
            and np.isfinite(saved["scores"]).all()
        ):
            if progress is not None:
                progress.update(len(group.batch_indices))
            return saved["scores"]

    rows = []
    for batch_index in group.batch_indices:
        source, target = load_aligned_pair(
            input_root(config), group, condition, batch_index
        )
        rows.append(_score_pair(metric, source, target))
        if progress is not None:
            progress.update(1)
    scores = np.stack(rows)
    if scores.shape != expected_shape or not np.isfinite(scores).all():
        raise ValueError(
            f"Unexpected score shape for {group.model}/{group.dataset}/{metric['id']}"
        )
    save_npz(
        checkpoint,
        fingerprint=np.array(checkpoint_fingerprint),
        condition_name=np.array(condition.name),
        scores=scores,
    )
    return scores


def metric_result_path(data_dir: Path, group: DatasetGroup, metric_id: str) -> Path:
    """Return one model/dataset/metric result path."""
    return data_dir / group.model / group.dataset / "metrics" / f"{metric_id}.npz"


def lambda_split_result_path(
    data_dir: Path,
    group: DatasetGroup,
    metric_id: str,
    lambda_value: float,
) -> Path:
    """Return one independently fitted lambda-split result path."""
    return (
        data_dir
        / group.model
        / group.dataset
        / "lambda_splits"
        / f"lambda_{lambda_value:g}"
        / "metrics"
        / f"{metric_id}.npz"
    )


def analyze_metric(
    config: dict[str, Any],
    group: DatasetGroup,
    metric: dict[str, Any],
    data_dir: Path,
) -> Path:
    """Compute or reuse one complete score-distance-MDS artifact."""
    result_path = metric_result_path(data_dir, group, metric["id"])
    score_fingerprint = _checkpoint_fingerprint(config, group, metric)
    analysis_fingerprint = fingerprint(
        {
            "version": 2,
            "scores": score_fingerprint,
            "mds": config["mds"],
            "mds_components": [2, 3],
            "pca_components": 2,
        }
    )
    if (
        result_path.exists()
        and not config["execution"]["recompute_scores"]
        and not config["execution"]["recompute_mds"]
    ):
        saved = load_npz(result_path)
        if (
            saved.get("analysis_fingerprint", np.array("")).item()
            == analysis_fingerprint
        ):
            return result_path

    checkpoint_dir = (
        data_dir / group.model / group.dataset / "checkpoints" / metric["id"]
    )
    with tqdm(
        total=len(group.conditions) * len(group.batch_indices),
        desc=f"Scores {group.model}/{group.dataset}/{metric['id']}",
        unit="pair",
        leave=False,
        disable=None,
    ) as progress:
        scores = np.stack(
            [
                _condition_scores(
                    config,
                    group,
                    metric,
                    condition_index,
                    checkpoint_dir,
                    score_fingerprint,
                    progress,
                )
                for condition_index in range(len(group.conditions))
            ]
        )
    vectors = concatenate_batch_scores(scores)
    distances = score_distances(vectors)
    embedding_2d, raw_2d, normalized_2d, iterations_2d = fit_metric_mds(
        distances, config, n_components=2
    )
    embedding_3d, raw_3d, normalized_3d, iterations_3d = fit_metric_mds(
        distances, config, n_components=3
    )
    pca_embedding_2d, pca_components_2d, pca_explained_2d, pca_mean = fit_signal_pca(
        vectors, n_components=2
    )
    conditions = group.conditions
    save_npz(
        result_path,
        version=np.array(1),
        model=np.array(group.model),
        dataset=np.array(group.dataset),
        metric_id=np.array(metric["id"]),
        metric_name=np.array(metric["name"]),
        metric_mode=np.array(metric["mode"]),
        parameter_name=np.array(metric["parameter_name"] or ""),
        parameter_values=np.asarray(metric["parameter_values"]),
        source_fingerprint=np.array(group.source_fingerprint),
        score_fingerprint=np.array(score_fingerprint),
        analysis_fingerprint=np.array(analysis_fingerprint),
        condition_names=np.asarray([condition.name for condition in conditions]),
        is_none=np.asarray([condition.is_none for condition in conditions]),
        lambda_values=np.asarray(
            [
                np.nan if condition.lambda_value is None else condition.lambda_value
                for condition in conditions
            ]
        ),
        alpha_values=np.asarray(
            [
                np.nan if condition.alpha_value is None else condition.alpha_value
                for condition in conditions
            ]
        ),
        tau_values=np.asarray(
            [
                np.nan if condition.tau_value is None else condition.tau_value
                for condition in conditions
            ]
        ),
        batch_indices=np.asarray(group.batch_indices),
        scores=scores,
        concatenated_vectors=vectors,
        distances=distances,
        embedding_2d=embedding_2d,
        embedding_2d_raw_stress=np.array(raw_2d),
        embedding_2d_normalized_stress=np.array(normalized_2d),
        embedding_2d_n_iter=np.array(iterations_2d),
        embedding_3d=embedding_3d,
        embedding_3d_raw_stress=np.array(raw_3d),
        embedding_3d_normalized_stress=np.array(normalized_3d),
        embedding_3d_n_iter=np.array(iterations_3d),
        pca_embedding_2d=pca_embedding_2d,
        pca_components_2d=pca_components_2d,
        pca_explained_variance_ratio_2d=pca_explained_2d,
        pca_mean=pca_mean,
    )
    return result_path


def analyze_lambda_splits(
    config: dict[str, Any],
    group: DatasetGroup,
    metric: dict[str, Any],
    data_dir: Path,
    result_path: Path,
) -> list[Path]:
    """Fit and save MDS results for none plus each configured lambda subset."""
    result = load_npz(result_path)
    paths = []
    for lambda_value in config["lambda_splits"]:
        selected = result["is_none"] | np.isclose(result["lambda_values"], lambda_value)
        indices = np.flatnonzero(selected)
        if len(indices) < 2 or selected.sum() == result["is_none"].sum():
            raise ValueError(
                f"Lambda split {lambda_value:g} has no transformed conditions for "
                f"{group.model}/{group.dataset}"
            )
        split_fingerprint = fingerprint(
            {
                "analysis": result["analysis_fingerprint"].item(),
                "lambda": lambda_value,
            }
        )
        path = lambda_split_result_path(data_dir, group, metric["id"], lambda_value)
        if (
            path.exists()
            and not config["execution"]["recompute_scores"]
            and not config["execution"]["recompute_mds"]
        ):
            saved = load_npz(path)
            if (
                saved.get("analysis_fingerprint", np.array("")).item()
                == split_fingerprint
            ):
                paths.append(path)
                continue

        distances = result["distances"][np.ix_(indices, indices)]
        embedding_2d, raw_2d, normalized_2d, iterations_2d = fit_metric_mds(
            distances, config, n_components=2
        )
        embedding_3d, raw_3d, normalized_3d, iterations_3d = fit_metric_mds(
            distances, config, n_components=3
        )
        vectors = result["concatenated_vectors"][indices]
        pca_embedding_2d, pca_components_2d, pca_explained_2d, pca_mean = (
            fit_signal_pca(vectors, n_components=2)
        )
        save_npz(
            path,
            version=np.array(1),
            model=result["model"],
            dataset=result["dataset"],
            metric_id=result["metric_id"],
            metric_name=result["metric_name"],
            metric_mode=result["metric_mode"],
            parameter_name=result["parameter_name"],
            parameter_values=result["parameter_values"],
            split_lambda=np.array(lambda_value),
            analysis_fingerprint=np.array(split_fingerprint),
            condition_names=result["condition_names"][indices],
            is_none=result["is_none"][indices],
            lambda_values=result["lambda_values"][indices],
            alpha_values=result["alpha_values"][indices],
            tau_values=result["tau_values"][indices],
            batch_indices=result["batch_indices"],
            scores=result["scores"][indices],
            concatenated_vectors=vectors,
            distances=distances,
            embedding_2d=embedding_2d,
            embedding_2d_raw_stress=np.array(raw_2d),
            embedding_2d_normalized_stress=np.array(normalized_2d),
            embedding_2d_n_iter=np.array(iterations_2d),
            embedding_3d=embedding_3d,
            embedding_3d_raw_stress=np.array(raw_3d),
            embedding_3d_normalized_stress=np.array(normalized_3d),
            embedding_3d_n_iter=np.array(iterations_3d),
            pca_embedding_2d=pca_embedding_2d,
            pca_components_2d=pca_components_2d,
            pca_explained_variance_ratio_2d=pca_explained_2d,
            pca_mean=pca_mean,
        )
        paths.append(path)
    return paths
