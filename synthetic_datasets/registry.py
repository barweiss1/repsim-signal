"""Dataset-family registrations for construction, transforms, and data sweeps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, fields
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np

from .base_dataset import BaseDataset, DatasetConfig, TransformConfig

DatasetConfigBuilder = Callable[[Mapping[str, Any]], DatasetConfig]
TransformConfigBuilder = Callable[[Mapping[str, Any], BaseDataset], TransformConfig]
SweepBuilder = Callable[[BaseDataset, int], np.ndarray]


@dataclass(frozen=True)
class DatasetSpec:
    """Construction and sweep behavior owned by one dataset family."""

    name: str
    dataset_type: type[BaseDataset]
    config_type: type[DatasetConfig]
    transform_config_type: type[TransformConfig]
    config_builder: DatasetConfigBuilder
    transform_builder: TransformConfigBuilder
    sweep_builders: Mapping[str, SweepBuilder]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A dataset registration requires a non-empty name.")
        if not issubclass(self.dataset_type, BaseDataset):
            raise TypeError("dataset_type must inherit BaseDataset.")
        if not issubclass(self.config_type, DatasetConfig):
            raise TypeError("config_type must inherit DatasetConfig.")
        if not issubclass(self.transform_config_type, TransformConfig):
            raise TypeError("transform_config_type must inherit TransformConfig.")
        object.__setattr__(
            self,
            "sweep_builders",
            MappingProxyType(dict(self.sweep_builders)),
        )

    def make_dataset(self, values: Mapping[str, Any]) -> BaseDataset:
        """Build this family's owned dataset from experiment configuration."""
        config = self.config_builder(values)
        if not isinstance(config, self.config_type):
            raise TypeError(
                f"{self.name} config_builder must return {self.config_type.__name__}."
            )
        return self.dataset_type(config)

    def make_transform(
        self,
        values: Mapping[str, Any],
        dataset: BaseDataset,
    ) -> TransformConfig:
        """Build this family's transform configuration."""
        transform = self.transform_builder(values, dataset)
        if not isinstance(transform, self.transform_config_type):
            raise TypeError(
                f"{self.name} transform_builder must return "
                f"{self.transform_config_type.__name__}."
            )
        return transform

    def make_sweep(
        self,
        dataset: BaseDataset,
        parameter_name: str,
        sweep_len: int,
    ) -> np.ndarray:
        """Build one registered data-parameter sweep."""
        if isinstance(sweep_len, bool) or not isinstance(sweep_len, Integral):
            raise TypeError("sweep_len must be an integer.")
        sweep_len = int(sweep_len)
        if sweep_len < 1:
            raise ValueError("sweep_len must be at least 1.")
        try:
            builder = self.sweep_builders[parameter_name]
        except KeyError:
            raise ValueError(
                f"Dataset {self.name!r} does not support a {parameter_name!r} sweep. "
                f"Supported parameters: {sorted(self.sweep_builders)}"
            ) from None
        values = np.asarray(builder(dataset, sweep_len))
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(
                "A registered data sweep must return a non-empty 1D array."
            )
        return values


_DATASET_SPECS: dict[str, DatasetSpec] = {}
_DATASET_TYPES: dict[type[BaseDataset], DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    """Register one dataset family exactly once."""
    if spec.name in _DATASET_SPECS:
        raise ValueError(f"Dataset name {spec.name!r} is already registered.")
    if spec.dataset_type in _DATASET_TYPES:
        raise ValueError(
            f"Dataset type {spec.dataset_type.__name__} is already registered."
        )
    _DATASET_SPECS[spec.name] = spec
    _DATASET_TYPES[spec.dataset_type] = spec
    return spec


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return a registered dataset specification by public name."""
    try:
        return _DATASET_SPECS[name]
    except KeyError:
        raise ValueError(
            f"Unsupported dataset type: {name!r}. "
            f"Registered types: {sorted(_DATASET_SPECS)}"
        ) from None


def get_dataset_spec_for_instance(dataset: BaseDataset) -> DatasetSpec:
    """Return the exact registration associated with a dataset instance."""
    for dataset_type in type(dataset).__mro__:
        spec = _DATASET_TYPES.get(dataset_type)
        if spec is not None:
            return spec
    raise ValueError(f"Unregistered dataset type: {type(dataset).__name__}")


def get_dataset_specs() -> Mapping[str, DatasetSpec]:
    """Return a read-only view of every registered dataset family."""
    return MappingProxyType(_DATASET_SPECS)


def build_dataclass_config(
    config_type,
    values: Mapping[str, Any],
    *,
    overrides: Mapping[str, Any] | None = None,
):
    """Build a config from matching non-None keys and dataclass defaults."""
    if not isinstance(values, Mapping):
        raise TypeError("Configuration values must be a mapping.")
    overrides = dict(overrides or {})
    kwargs = {}
    missing = []
    for field in fields(config_type):
        if field.name in overrides:
            value = overrides[field.name]
        else:
            value = values.get(field.name)
        if value is not None:
            kwargs[field.name] = value
        elif field.default is MISSING and field.default_factory is MISSING:
            missing.append(field.name)
    if missing:
        raise ValueError(
            f"{config_type.__name__} requires configuration fields: {missing}"
        )
    return config_type(**kwargs)


def build_transform_config(
    config_type,
    values: Mapping[str, Any],
    dataset: BaseDataset,
    *,
    inherit_from_dataset: Mapping[str, str] | None = None,
):
    """Build a transform config using declared dataset-field inheritance."""
    inherited = {}
    for transform_field, dataset_field in (inherit_from_dataset or {}).items():
        if values.get(transform_field) is None:
            inherited[transform_field] = getattr(dataset.config, dataset_field)
    return build_dataclass_config(config_type, values, overrides=inherited)


def linear_sweep(start: float, stop: float) -> SweepBuilder:
    """Declare an inclusive linear sweep owned by a dataset registration."""

    def build(dataset: BaseDataset, sweep_len: int) -> np.ndarray:
        del dataset
        return np.linspace(start, stop, sweep_len)

    return build


def integer_sweep(start: int, stop) -> SweepBuilder:
    """Declare an inclusive, duplicate-free integer sweep."""

    def build(dataset: BaseDataset, sweep_len: int) -> np.ndarray:
        resolved_stop = stop(dataset) if callable(stop) else stop
        if resolved_stop < start:
            return np.asarray([resolved_stop], dtype=int)
        return np.unique(
            np.rint(np.linspace(start, resolved_stop, sweep_len)).astype(int)
        )

    return build
