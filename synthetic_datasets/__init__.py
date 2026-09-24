"""Registered synthetic dataset families and compatibility factory functions."""

from collections.abc import Mapping

from .base_dataset import BaseDataset
from .base_dataset import DatasetConfig
from .base_dataset import TransformConfig
from .ellipsoid import EllipsoidConfig
from .ellipsoid import EllipsoidDataset
from .ellipsoid import EllipsoidTransformConfig
from .gaussian_cluster import GaussianClusterConfig
from .gaussian_cluster import GaussianClusterDataset
from .gaussian_cluster import GaussianClusterTransformConfig
from .permuted_gaussian_clusters import PermutedGaussianClustersConfig
from .permuted_gaussian_clusters import PermutedGaussianClustersDataset
from .permuted_gaussian_clusters import PermutedGaussianClustersTransformConfig
from .registry import DatasetSpec
from .registry import get_dataset_spec
from .registry import get_dataset_spec_for_instance
from .registry import get_dataset_specs
from .rings_dataset import RingsConfig
from .rings_dataset import RingsDataset
from .rings_dataset import RingsTransformConfig
from .s_clusters import SClustersConfig
from .s_clusters import SClustersDataset
from .s_clusters import SClustersTransformConfig
from .s_curve import SCurveConfig
from .s_curve import SCurveDataset
from .s_curve import SCurveTransformConfig


def init_dataset(sim_params: Mapping) -> BaseDataset:
    """Construct a registered dataset from an experiment configuration."""
    if not isinstance(sim_params, Mapping):
        raise TypeError("sim_params must be a mapping.")
    dataset_type = sim_params.get("dataset_type")
    if not isinstance(dataset_type, str) or not dataset_type:
        raise ValueError("sim_params must define a non-empty dataset_type.")
    return get_dataset_spec(dataset_type).make_dataset(sim_params)


def get_transform_config(
    transform_params: Mapping,
    dataset: BaseDataset,
) -> TransformConfig:
    """Construct the registered transform config for a dataset instance."""
    if not isinstance(transform_params, Mapping):
        raise TypeError("transform_params must be a mapping.")
    return get_dataset_spec_for_instance(dataset).make_transform(
        transform_params,
        dataset,
    )


def get_data_param_sweep_values(
    dataset: BaseDataset,
    data_param_name: str,
    sweep_len: int = 20,
):
    """Construct a data-parameter sweep declared by the dataset family."""
    return get_dataset_spec_for_instance(dataset).make_sweep(
        dataset,
        data_param_name,
        sweep_len,
    )


__all__ = [
    "BaseDataset",
    "DatasetConfig",
    "DatasetSpec",
    "EllipsoidConfig",
    "EllipsoidDataset",
    "EllipsoidTransformConfig",
    "GaussianClusterConfig",
    "GaussianClusterDataset",
    "GaussianClusterTransformConfig",
    "PermutedGaussianClustersConfig",
    "PermutedGaussianClustersDataset",
    "PermutedGaussianClustersTransformConfig",
    "RingsConfig",
    "RingsDataset",
    "RingsTransformConfig",
    "SClustersConfig",
    "SClustersDataset",
    "SClustersTransformConfig",
    "SCurveConfig",
    "SCurveDataset",
    "SCurveTransformConfig",
    "TransformConfig",
    "get_data_param_sweep_values",
    "get_dataset_spec",
    "get_dataset_spec_for_instance",
    "get_dataset_specs",
    "get_transform_config",
    "init_dataset",
]
