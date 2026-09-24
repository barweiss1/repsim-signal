"""Dataset grid construction and lifecycle for the MDS experiment."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from synthetic_datasets.permuted_gaussian_clusters import (
    PermutedGaussianClustersConfig,
)
from synthetic_datasets.permuted_gaussian_clusters import (
    PermutedGaussianClustersDataset,
)
from synthetic_datasets.permuted_gaussian_clusters import (
    PermutedGaussianClustersTransformConfig,
)

from ..artifacts import load_npz
from ..artifacts import save_csv
from ..artifacts import save_json
from ..artifacts import save_npz
from .config import OPACITY_RANGE
from .config import PARAMETERS
from .config import POINT_SIZE_RANGE
from .config import compatible_dataset_fingerprints
from .config import dataset_fingerprint
from .config import effective_config
from .config import output_paths
from .config import require
from .config import save_run_config


def grid_parameter_values(config: dict[str, Any]) -> np.ndarray:
    """Return the ordered Cartesian product of the configured data grid."""
    grid = config["grid"]

    def sweep(name: str) -> np.ndarray:
        specification = grid[name]
        return np.linspace(
            specification["min"],
            specification["max"],
            specification["num"],
        )

    axes = (
        sweep("cluster_mixing_probability"),
        sweep("noise_scale"),
        grid["n_permute"]["values"],
    )
    return np.asarray(list(itertools.product(*axes)), dtype=float)


def parameter_levels(name: str, config: dict[str, Any]) -> np.ndarray:
    """Return all configured levels, including the base transform value."""
    if name == "n_permute":
        levels = config["grid"][name]["values"]
    else:
        sweep = config["grid"][name]
        levels = np.linspace(sweep["min"], sweep["max"], sweep["num"])
    return np.unique([config["base_transform"][name], *levels])


def visual_encodings(
    values: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Map configured parameters to plot color, size, and opacity values."""
    values = np.asarray(values, dtype=float)
    require(
        values.ndim == 2 and values.shape[1] == 3,
        "parameter values must have three columns",
    )
    columns = {name: values[:, index] for index, name in enumerate(PARAMETERS)}
    mapping = config["visualization"]["encodings"]

    size_levels = (
        parameter_levels(mapping["size"], config)
        if mapping["size"] is not None
        else np.asarray([0.0])
    )
    size_min, size_max = size_levels.min(), size_levels.max()
    size_fraction = (
        (columns[mapping["size"]] - size_min) / (size_max - size_min)
        if mapping["size"] is not None and size_max > size_min
        else np.zeros(len(values))
    )
    opacity_levels = (
        parameter_levels(mapping["opacity"], config)
        if mapping["opacity"] is not None
        else np.asarray([0.0])
    )
    opacity_min, opacity_max = opacity_levels.min(), opacity_levels.max()
    opacity_fraction = (
        (columns[mapping["opacity"]] - opacity_min) / (opacity_max - opacity_min)
        if mapping["opacity"] is not None and opacity_max > opacity_min
        else np.zeros(len(values))
    )
    return {
        "color": (
            columns[mapping["color"]]
            if mapping["color"] is not None
            else np.zeros(len(values))
        ),
        "size_value": (
            columns[mapping["size"]]
            if mapping["size"] is not None
            else np.zeros(len(values))
        ),
        "size": POINT_SIZE_RANGE[0]
        + np.clip(size_fraction, 0, 1) * np.ptp(POINT_SIZE_RANGE),
        "opacity_value": (
            columns[mapping["opacity"]]
            if mapping["opacity"] is not None
            else np.zeros(len(values))
        ),
        "opacity": (
            OPACITY_RANGE[0] + np.clip(opacity_fraction, 0, 1) * np.ptp(OPACITY_RANGE)
            if mapping["opacity"] is not None
            else np.ones(len(values))
        ),
        "size_levels": size_levels,
        "opacity_levels": opacity_levels,
    }


# points.csv column order, matching the key order `metadata` builds below.
# The shared `save_csv` takes an explicit field list rather than inferring it
# from the first row, so the written column order stays pinned here instead of
# depending on dict insertion order at the call site.
POINTS_CSV_FIELDS: tuple[str, ...] = (
    "point_index",
    "is_base",
    *PARAMETERS,
    "color_value",
    "size_value",
    "marker_size",
    "opacity_value",
    "opacity",
)


def metadata(
    config: dict[str, Any],
    grid: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Build plot encodings and the stable points.csv rows."""
    base = config["base_transform"]
    parameters = np.vstack(
        (
            [
                base["cluster_mixing_probability"],
                base["noise_scale"],
                base["n_permute"],
            ],
            grid,
        )
    )
    encodings = visual_encodings(parameters, config)
    rows = [
        {
            "point_index": index,
            "is_base": int(index == 0),
            **{
                name: int(value) if name == "n_permute" else float(value)
                for name, value in zip(PARAMETERS, params)
            },
            "color_value": float(encodings["color"][index]),
            "size_value": float(encodings["size_value"][index]),
            "marker_size": float(encodings["size"][index]),
            "opacity_value": float(encodings["opacity_value"][index]),
            "opacity": float(encodings["opacity"][index]),
        }
        for index, params in enumerate(parameters)
    ]
    return encodings, rows


def transform_config(
    config: dict[str, Any],
    *,
    transform_mode: str | None = None,
    seed: int | None = None,
    **parameters: float,
) -> PermutedGaussianClustersTransformConfig:
    """Build one deterministic transform from a parameter-grid row.

    Defaults to ``transform.seed``; callers building the base transform pass
    ``transform.base_seed`` explicitly so the base and swept transforms draw
    from independent random streams even when their parameters coincide.
    """
    return PermutedGaussianClustersTransformConfig(
        n_permute=int(parameters["n_permute"]),
        noise_scale=float(parameters["noise_scale"]),
        seed=config["transform"]["seed"] if seed is None else seed,
        cluster_mixing_probability=float(parameters["cluster_mixing_probability"]),
        transform_mode=transform_mode or config["transform"]["mode"],
    )


def build_base_dataset(
    config: dict[str, Any],
) -> tuple[PermutedGaussianClustersDataset, np.ndarray]:
    """Construct the live dataset instance and its transformed base array."""
    dataset = PermutedGaussianClustersDataset(
        PermutedGaussianClustersConfig(
            **config["dataset"],
            noise_scale=config["base_transform"]["noise_scale"],
        )
    )
    base = dataset.transform_base(
        transform_config(
            config,
            transform_mode="resample",
            seed=config["transform"]["base_seed"],
            **config["base_transform"],
        )
    ).copy()
    return dataset, base


def _plot_dataset(
    config: dict[str, Any],
    base: np.ndarray,
    transformed: np.ndarray,
    labels: np.ndarray,
    grid: np.ndarray,
) -> None:
    from .plotting import plot_dataset_examples

    plot_dataset_examples(
        base,
        transformed,
        labels,
        grid,
        output_paths(config)[1],
        config,
    )


def generate_dataset(config: dict[str, Any]) -> Path:
    """Generate or reuse the configured dataset and its stable artifacts."""
    save_run_config(config)
    data_dir, _ = output_paths(config)
    dataset_path = data_dir / "dataset.npz"
    manifest_path = data_dir / "dataset_manifest.json"
    fingerprint = dataset_fingerprint(config)
    if (
        dataset_path.exists()
        and manifest_path.exists()
        and not config["execution"]["overwrite_dataset"]
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(
            manifest.get("dataset_fingerprint")
            in compatible_dataset_fingerprints(config),
            f"Existing dataset configuration does not match {dataset_path}; "
            "enable overwrite_dataset",
        )
        print(f"Reusing matching dataset: {dataset_path}")
        base, transformed, labels, grid = load_dataset_artifacts(config)
        _plot_dataset(config, base, transformed, labels, grid)
        return dataset_path
    require(
        config["execution"]["overwrite_dataset"]
        or not (dataset_path.exists() or manifest_path.exists()),
        f"Incomplete dataset artifacts in {data_dir}; enable overwrite_dataset",
    )

    dataset, base = build_base_dataset(config)
    grid = grid_parameter_values(config)
    transformed = np.stack(
        [
            dataset.transform_current(
                transform_config(
                    config,
                    cluster_mixing_probability=mixing,
                    noise_scale=noise,
                    n_permute=permutations,
                )
            )
            for mixing, noise, permutations in grid
        ]
    )
    encodings, rows = metadata(config, grid)
    save_npz(
        dataset_path,
        base_data=base,
        transformed_data=transformed,
        cluster_labels=dataset.plot_labels,
        grid_parameter_values=grid,
        all_color_values=encodings["color"],
        all_marker_sizes=encodings["size"],
        all_opacities=encodings["opacity"],
    )
    save_csv(data_dir / "points.csv", rows, list(POINTS_CSV_FIELDS))
    save_json(data_dir / "normalized_config.json", effective_config(config))
    save_json(
        manifest_path,
        {
            "version": 1,
            "dataset_fingerprint": fingerprint,
            "n_grid_datasets": len(grid),
            "n_total_with_base": len(grid) + 1,
            "parameter_names": list(PARAMETERS),
        },
    )
    _plot_dataset(config, base, transformed, dataset.plot_labels, grid)
    print(f"Saved dataset: {dataset_path}")
    return dataset_path


def load_dataset_artifacts(config: dict[str, Any]) -> tuple[np.ndarray, ...]:
    """Load, validate, and migrate compatible generated dataset artifacts."""
    data_dir, _ = output_paths(config)
    dataset_path = data_dir / "dataset.npz"
    manifest_path = data_dir / "dataset_manifest.json"
    require(
        dataset_path.exists() and manifest_path.exists(),
        "Dataset artifacts are missing; run generate first",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved_fingerprint = manifest.get("dataset_fingerprint")
    require(
        saved_fingerprint in compatible_dataset_fingerprints(config),
        "Dataset fingerprint mismatch",
    )
    arrays = load_npz(dataset_path)
    keys = (
        "base_data",
        "transformed_data",
        "cluster_labels",
        "grid_parameter_values",
    )
    artifacts = tuple(arrays[key] for key in keys)
    if saved_fingerprint != dataset_fingerprint(config):
        print("Migrating legacy visualization metadata; numerical data is unchanged")
        manifest.update(
            dataset_fingerprint=dataset_fingerprint(config),
            legacy_dataset_fingerprint=saved_fingerprint,
            fingerprint_scope="generation-v2",
        )
        save_json(manifest_path, manifest)

    encodings, rows = metadata(config, artifacts[3])
    current_metadata = {
        "all_color_values": encodings["color"],
        "all_marker_sizes": encodings["size"],
        "all_opacities": encodings["opacity"],
    }
    stale = (
        "all_rgb",
        "all_shape_indices",
        "all_point_sizes",
        "all_line_angles",
        "all_line_lengths",
    )
    if any(
        key not in arrays or not np.array_equal(arrays[key], value)
        for key, value in current_metadata.items()
    ) or any(key in arrays for key in stale):
        for key in stale:
            arrays.pop(key, None)
        arrays.update(current_metadata)
        save_npz(dataset_path, **arrays)
        save_csv(data_dir / "points.csv", rows, list(POINTS_CSV_FIELDS))
    save_json(data_dir / "normalized_config.json", effective_config(config))
    return artifacts
