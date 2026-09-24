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
class SCurveConfig(DatasetConfig):
    sigma: float = 1.0


@dataclass
class SCurveTransformConfig(TransformConfig):
    t: float = 1.0
    noise_scale: float = 0.1
    seed: int = 42


class SCurveDataset(BaseDataset):
    def __init__(self, config: SCurveConfig):
        owned_config = config.copy()
        owned_config.dim = 2
        super().__init__(owned_config)

    def generate(self):
        self.plot_labels, self.init_data = swiss_roll(self.config.n_points)
        rng = np.random.RandomState(self.config.seed)
        self.base_data = self.init_data + rng.randn(*self.init_data.shape) * (
            self.config.sigma * 0.5
        )
        self.current_data = None

    def transform_base(self, transform_config: SCurveTransformConfig) -> np.ndarray:
        """For S-curve, the base data is a swiss roll."""
        return self.base_data.copy()

    def transform(self, transform_config: SCurveTransformConfig) -> np.ndarray:
        t = transform_config.t
        noise_scale = transform_config.noise_scale
        rng = np.random.RandomState(transform_config.seed)

        if not (0 <= t <= 1):
            raise ValueError("t must be in [0, 1].")

        z = 3 * np.pi * (self.plot_labels - t)
        x = np.sin(z)
        y = 0.5 * np.sign(z) * (np.cos(z) - 1)

        data = np.stack([x, y], axis=1)

        # Add small Gaussian noise
        if noise_scale > 0:
            noise = rng.randn(*data.shape) * (noise_scale * self.config.sigma)
            data += noise

        # Interpolate between original position and target position on the ring
        return data


def swiss_roll(n_points: int) -> np.ndarray:
    """Generates a Swiss roll dataset with n_points."""
    t = np.linspace(0, 1, n_points)
    z = (3 * np.pi / 2) * (1 + 2 * t)
    x = z * np.cos(z)
    y = z * np.sin(z)
    return t, np.stack([x, y], axis=1)


def _build_s_curve_config(values):
    return build_dataclass_config(SCurveConfig, values)


def _build_s_curve_transform(values, dataset):
    return build_transform_config(
        SCurveTransformConfig,
        values,
        dataset,
        inherit_from_dataset={"seed": "seed"},
    )


DATASET_SPEC = register_dataset(
    DatasetSpec(
        name="s_curve",
        dataset_type=SCurveDataset,
        config_type=SCurveConfig,
        transform_config_type=SCurveTransformConfig,
        config_builder=_build_s_curve_config,
        transform_builder=_build_s_curve_transform,
        sweep_builders={
            "t": linear_sweep(0.0, 1.0),
            "noise_scale": linear_sweep(0.1, 1.0),
            "dim": integer_sweep(2, 2),
        },
    )
)
