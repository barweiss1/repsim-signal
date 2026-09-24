from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class DatasetConfig:
    n_points: int
    dim: int
    seed: int = 42

    def copy(self):
        """Return an independent copy, including mutable subclass fields."""
        return deepcopy(self)


@dataclass
class TransformConfig(ABC):
    def set_param(self, name: str, value):
        """Set one declared transform parameter in place."""
        if not hasattr(self, name):
            raise ValueError(
                f"{self.__class__.__name__} has no parameter named '{name}'."
            )
        setattr(self, name, value)

    def get_param_set(self):
        """Return the set of parameters represented by this configuration."""
        return set(self.__dict__)

    def copy(self):
        """Return an independent copy, including mutable subclass fields."""
        return deepcopy(self)


class BaseDataset(ABC):
    """Own the initial, base, and current states of one synthetic dataset.

    ``init_data`` is generated once from the owned dataset configuration and is
    the stable source used by resampling transforms. ``base_data`` is the
    reference representation established by :meth:`transform_base`.
    ``current_data`` is the latest comparison representation established by
    :meth:`transform_current`. Concrete :meth:`transform` implementations
    compute and return data but must not assign either lifecycle state.

    Configurations and arrays are copied at ownership boundaries so callers
    cannot mutate a dataset accidentally. Transform randomness must be derived
    from the transform configuration, never from invocation history.
    """

    def __init__(self, config: DatasetConfig):
        if not isinstance(config, DatasetConfig):
            raise TypeError("config must be a DatasetConfig instance.")
        self.config: DatasetConfig = config.copy()
        self.init_data: np.ndarray | None = None
        self.current_data: np.ndarray | None = None
        self.base_data: np.ndarray | None = None
        self.plot_labels: np.ndarray | None = None
        self.generate()

    def __len__(self):
        if self.current_data is not None:
            return len(self.current_data)
        return 0

    def __getitem__(self, idx: int):
        if self.current_data is not None:
            value = self.current_data[idx]
            return value.copy() if isinstance(value, np.ndarray) else value
        raise IndexError("Dataset has not been initialized or transformed.")

    def generate(self):
        """Generate the stable initial state using only the dataset seed."""
        rng = np.random.RandomState(self.config.seed)
        self.init_data = rng.randn(self.config.n_points, self.config.dim)
        self.current_data = None
        self.base_data = self.init_data.copy()
        self.plot_labels = np.arctan2(self.init_data[:, 1], self.init_data[:, 0])

    @abstractmethod
    def transform(self, transform_config: TransformConfig) -> np.ndarray:
        """Compute transformed data without assigning base or current state."""

    def _owned_transform_result(self, value: Any) -> np.ndarray:
        result = np.asarray(value)
        if result.shape != self.init_data.shape:
            raise ValueError(
                "A dataset transform must preserve the initial data shape: "
                f"expected {self.init_data.shape}, received {result.shape}."
            )
        if not np.isfinite(result).all():
            raise ValueError("A dataset transform must return only finite values.")
        return result.copy()

    def transform_base(self, transform_config: TransformConfig) -> np.ndarray:
        """Replace the owned reference state and return an independent copy."""
        self.base_data = self._owned_transform_result(self.transform(transform_config))
        return self.base_data.copy()

    def transform_current(self, transform_config: TransformConfig) -> np.ndarray:
        """Replace the owned comparison state and return an independent copy."""
        self.current_data = self._owned_transform_result(
            self.transform(transform_config)
        )
        return self.current_data.copy()

    def get_current_numpy(self) -> np.ndarray:
        """Return an independent copy of the current comparison state."""
        if self.current_data is None:
            raise RuntimeError("Current data has not been transformed yet.")
        return self.current_data.copy()

    def get_current_torch(self) -> torch.Tensor:
        """Return the current comparison state as an independent float tensor."""
        return torch.from_numpy(self.get_current_numpy()).float()

    def get_base_numpy(self) -> np.ndarray:
        """Return an independent copy of the reference state."""
        if self.base_data is None:
            raise RuntimeError("Base data has not been initialized.")
        return self.base_data.copy()

    def get_base_torch(self) -> torch.Tensor:
        """Return the reference state as an independent float tensor."""
        return torch.from_numpy(self.get_base_numpy()).float()

    def copy(self):
        """Return an independent snapshot of configuration, data, and metadata."""
        return deepcopy(self)
