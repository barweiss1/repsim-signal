from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

from synthetic_datasets.base_dataset import BaseDataset, DatasetConfig, TransformConfig
from synthetic_datasets.registry import DatasetSpec
from synthetic_datasets.registry import build_dataclass_config
from synthetic_datasets.registry import build_transform_config
from synthetic_datasets.registry import linear_sweep
from synthetic_datasets.registry import register_dataset


@dataclass
class SClustersConfig(DatasetConfig):
    n_clusters: int = 10
    sigma: float = 0.05
    line_x_min: float = 0.0
    line_x_max: float = 1.0


@dataclass
class SClustersTransformConfig(TransformConfig):
    t: float = 1.0
    noise_scale: float = 0.1
    seed: int = 42


class SClustersDataset(BaseDataset):
    def __init__(self, config: SClustersConfig):
        owned_config = config.copy()
        owned_config.dim = 2
        super().__init__(owned_config)

    def generate(self):
        rng = np.random.RandomState(self.config.seed)

        x_positions = np.linspace(
            self.config.line_x_min, self.config.line_x_max, self.config.n_clusters
        )
        self.cluster_centers_line = np.column_stack(
            [x_positions, np.zeros_like(x_positions)]
        )
        self.cluster_centers_curve = np.column_stack(
            [
                x_positions,
                (self.config.line_x_max / 2) * np.sin(2 * np.pi * x_positions),
            ]
        )

        base_assignments = np.arange(self.config.n_clusters).repeat(
            self.config.n_points // self.config.n_clusters
        )
        remainder = self.config.n_points - len(base_assignments)
        if remainder > 0:
            extra = rng.choice(self.config.n_clusters, size=remainder, replace=True)
            base_assignments = np.concatenate([base_assignments, extra])
        rng.shuffle(base_assignments)

        self.cluster_assignments = base_assignments
        noise = rng.randn(self.config.n_points, 2) * self.config.sigma
        self.init_data = self.cluster_centers_line[self.cluster_assignments] + noise
        self.base_data = self.init_data.copy()
        self.current_data = None
        self.plot_labels = self.cluster_assignments

        self._offsets = (
            self.init_data - self.cluster_centers_line[self.cluster_assignments]
        )

    def transform(self, transform_config: SClustersTransformConfig) -> np.ndarray:
        t = transform_config.t
        if not (0 <= t <= 1):
            raise ValueError("t must be in [0, 1].")

        centers = (1 - t) * self.cluster_centers_line + t * self.cluster_centers_curve
        data = centers[self.cluster_assignments] + self._offsets
        rng = np.random.RandomState(transform_config.seed)
        noise = rng.randn(*data.shape) * (
            transform_config.noise_scale * self.config.sigma
        )

        return data + noise


def _build_s_clusters_config(values):
    overrides = {}
    base_transform = values.get("base_transform_params")
    if values.get("n_clusters") is None and isinstance(base_transform, Mapping):
        if base_transform.get("n_clusters") is not None:
            overrides["n_clusters"] = base_transform["n_clusters"]
    return build_dataclass_config(SClustersConfig, values, overrides=overrides)


def _build_s_clusters_transform(values, dataset):
    return build_transform_config(
        SClustersTransformConfig,
        values,
        dataset,
        inherit_from_dataset={"seed": "seed"},
    )


DATASET_SPEC = register_dataset(
    DatasetSpec(
        name="s_clusters",
        dataset_type=SClustersDataset,
        config_type=SClustersConfig,
        transform_config_type=SClustersTransformConfig,
        config_builder=_build_s_clusters_config,
        transform_builder=_build_s_clusters_transform,
        sweep_builders={
            "t": linear_sweep(0.0, 1.0),
            "noise_scale": linear_sweep(0.1, 1.0),
        },
    )
)
