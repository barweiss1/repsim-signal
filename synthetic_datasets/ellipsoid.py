from dataclasses import dataclass
from typing import Optional

import numpy as np

from synthetic_datasets.base_dataset import BaseDataset, DatasetConfig, TransformConfig
from synthetic_datasets.registry import DatasetSpec
from synthetic_datasets.registry import build_dataclass_config
from synthetic_datasets.registry import build_transform_config
from synthetic_datasets.registry import integer_sweep
from synthetic_datasets.registry import linear_sweep
from synthetic_datasets.registry import register_dataset


@dataclass
class EllipsoidConfig(DatasetConfig):
    axis_scale: float = 0.5


@dataclass
class EllipsoidTransformConfig(TransformConfig):
    t: float = 1.0
    axis_scale: Optional[float] = None


class EllipsoidDataset(BaseDataset):
    def __init__(self, config: EllipsoidConfig):
        if config.dim < 2:
            raise ValueError(
                "EllipsoidDataset requires dim >= 2 to build orthogonal directions."
            )
        super().__init__(config)

    def _sample_unit_vector(self, rng):
        vec = rng.randn(self.config.dim)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return self._sample_unit_vector(rng)
        return vec / norm

    def _sample_orthogonal_vector(self, base_vec, rng):
        for _ in range(100):
            vec = self._sample_unit_vector(rng)
            vec = vec - np.dot(vec, base_vec) * base_vec
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                return vec / norm
        raise ValueError("Failed to sample an orthogonal direction.")

    def _apply_ellipsoid(self, data, direction, axis_scale):
        if axis_scale <= 0:
            raise ValueError("axis_scale must be positive.")
        direction = direction / np.linalg.norm(direction)
        projection = data @ direction
        return data + (axis_scale - 1.0) * projection[:, None] * direction[None, :]

    def generate(self):
        rng = np.random.RandomState(self.config.seed)
        self.init_data = rng.randn(self.config.n_points, self.config.dim)

        self.base_direction = self._sample_unit_vector(rng)
        self.target_direction = self._sample_orthogonal_vector(self.base_direction, rng)

        self.base_data = self._apply_ellipsoid(
            self.init_data,
            self.base_direction,
            self.config.axis_scale,
        )
        self.current_data = None
        self.plot_labels = self.init_data @ self.base_direction

    def transform(self, transform_config: EllipsoidTransformConfig) -> np.ndarray:
        t = transform_config.t
        if not (0 <= t <= 1):
            raise ValueError("t must be in [0, 1].")

        axis_scale = (
            transform_config.axis_scale
            if transform_config.axis_scale is not None
            else self.config.axis_scale
        )
        direction = (1 - t) * self.base_direction + t * self.target_direction
        if np.linalg.norm(direction) == 0:
            direction = self.base_direction

        data = self._apply_ellipsoid(self.init_data, direction, axis_scale)
        return data


def _build_ellipsoid_config(values):
    return build_dataclass_config(EllipsoidConfig, values)


def _build_ellipsoid_transform(values, dataset):
    return build_transform_config(EllipsoidTransformConfig, values, dataset)


DATASET_SPEC = register_dataset(
    DatasetSpec(
        name="ellipsoid",
        dataset_type=EllipsoidDataset,
        config_type=EllipsoidConfig,
        transform_config_type=EllipsoidTransformConfig,
        config_builder=_build_ellipsoid_config,
        transform_builder=_build_ellipsoid_transform,
        sweep_builders={
            "t": linear_sweep(0.0, 1.0),
            "dim": integer_sweep(2, lambda dataset: dataset.config.dim),
        },
    )
)
