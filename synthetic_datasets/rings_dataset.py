from dataclasses import dataclass
import numpy as np
from scipy.stats import chi

from synthetic_datasets.base_dataset import BaseDataset, DatasetConfig, TransformConfig
from synthetic_datasets.registry import DatasetSpec
from synthetic_datasets.registry import build_dataclass_config
from synthetic_datasets.registry import build_transform_config
from synthetic_datasets.registry import integer_sweep
from synthetic_datasets.registry import linear_sweep
from synthetic_datasets.registry import register_dataset


@dataclass
class RingsConfig(DatasetConfig):
    sigma: float = 1.0


@dataclass
class RingsTransformConfig(TransformConfig):
    n_rings: int = 2
    t: float = 1.0
    clustering_type: str = "closest"  # "closest" or "random"
    seed: int = 42


class RingsDataset(BaseDataset):
    def __init__(self, config: RingsConfig):
        super().__init__(config)

        # Apply the initial sigma scaling onto the normalized data correctly
        self.init_data = self.init_data * self.config.sigma
        self.current_data = None
        self.base_data = self.init_data.copy()
        self.plot_labels = np.linalg.norm(
            self.init_data, axis=1
        )  # Use norms for coloring in visualization

    def transform(self, transform_config: RingsTransformConfig) -> np.ndarray:
        t = transform_config.t
        n_rings = transform_config.n_rings

        if not (0 <= t <= 1):
            raise ValueError("t must be in [0, 1].")

        # Original norms and directions
        norms = np.linalg.norm(self.init_data, axis=1, keepdims=True)
        directions = self.init_data / np.maximum(norms, 1e-8)

        # Compute ring radii based on chi quantiles (midpoints of equiprobable intervals)
        p_mid = (np.arange(n_rings) + 0.5) / n_rings
        radii = self.config.sigma * chi.ppf(p_mid, df=self.config.dim)

        if transform_config.clustering_type == "closest":
            # Assign based on the original norm's probability mass to divide points equally
            p_values = chi.cdf(norms.flatten() / self.config.sigma, df=self.config.dim)
            ring_indices = np.clip((p_values * n_rings).astype(int), 0, n_rings - 1)
        elif transform_config.clustering_type == "random":
            rng = np.random.RandomState(transform_config.seed)
            ring_indices = rng.choice(n_rings, size=self.config.n_points)
        else:
            raise ValueError("clustering_type must be 'closest' or 'random'.")

        # Set the target norms according to the assigned ring
        target_norms = radii[ring_indices, None]
        target_positions = directions * target_norms

        # Interpolate between original position and target position on the ring
        return (1 - t) * self.init_data + t * target_positions


def _build_rings_config(values):
    return build_dataclass_config(RingsConfig, values)


def _build_rings_transform(values, dataset):
    return build_transform_config(
        RingsTransformConfig,
        values,
        dataset,
        inherit_from_dataset={"seed": "seed"},
    )


DATASET_SPEC = register_dataset(
    DatasetSpec(
        name="rings",
        dataset_type=RingsDataset,
        config_type=RingsConfig,
        transform_config_type=RingsTransformConfig,
        config_builder=_build_rings_config,
        transform_builder=_build_rings_transform,
        sweep_builders={
            "t": linear_sweep(0.0, 1.0),
            "dim": integer_sweep(2, lambda dataset: dataset.config.dim),
            "n_rings": integer_sweep(1, 10),
        },
    )
)
