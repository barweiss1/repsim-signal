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
class GaussianClusterConfig(DatasetConfig):
    sigma: float = 1.0
    radius_type: str = "median"  # "median" or "mean"
    cluster_centers: np.ndarray = (
        None  # Optional precomputed cluster centers for efficiency
    )


@dataclass
class GaussianClusterTransformConfig(TransformConfig):
    t: float = 0.0
    noise_scale: float = 0.1
    n_clusters: int = 10
    # Adding a seed specifically for the noise so changing t doesn't change the noise pattern
    seed: int = 42
    clustering_type: str = "closest"  # "closest" or "random"


class GaussianClusterDataset(BaseDataset):
    def __init__(self, config: GaussianClusterConfig):
        super().__init__(config)

        # Apply the initial sigma scaling onto the normalized data correctly
        self.init_data = self.init_data * self.config.sigma
        self.current_data = None
        self.base_data = self.init_data.copy()
        # save angles of initial data points for later use in visualization
        self.plot_labels = np.arctan2(self.init_data[:, 1], self.init_data[:, 0])

    def _set_cluster_centers(self, n_clusters: int, rng):
        """
        compute cluster centers on the hypersphere based on the radius determined by the config.
        For dim=2, this will be points evenly spaced on a circle. For higher dimensions, we use randomly distributed points
        on the hypersphere, which should be approximately evenly spaced for large n_clusters.
        """
        # Median/Mean of ||X|| where X ~ N(0, sigma^2 I_dim)
        if self.config.radius_type == "median":
            radius = self.config.sigma * chi.ppf(0.5, df=self.config.dim)
        elif self.config.radius_type == "mean":
            radius = self.config.sigma * chi.mean(df=self.config.dim)
        else:
            raise ValueError("radius_type must be either 'median' or 'mean'.")

        # comupte cluster centers on the hypersphere
        self.cluster_centers = _hypersphere_centers(
            n_clusters=n_clusters,
            dim=self.config.dim,
            radius=radius,
            rng=rng,
        )

    def transform(self, transform_config: GaussianClusterTransformConfig) -> np.ndarray:
        t = transform_config.t
        noise_scale = transform_config.noise_scale
        rng = np.random.RandomState(transform_config.seed)

        # Precompute target positions since they depend on the base distribution
        self._set_cluster_centers(n_clusters=transform_config.n_clusters, rng=rng)

        if not (0 <= t <= 1):
            raise ValueError("t must be in [0, 1].")

        # Assign each point to closest center
        distances = np.linalg.norm(
            self.init_data[:, None, :] - self.cluster_centers[None, :, :],
            axis=2,
        )

        if transform_config.clustering_type == "closest":
            closest_clusters = np.argmin(distances, axis=1)
        elif transform_config.clustering_type == "random":
            closest_clusters = rng.choice(
                transform_config.n_clusters, size=self.config.n_points
            )
        else:
            raise ValueError("clustering_type must be either 'closest' or 'random'.")

        target_positions = self.cluster_centers[closest_clusters]

        # Interpolate
        data = (1 - t) * self.init_data + t * target_positions
        # data = self.init_data + t * 10 * target_positions

        # Add small Gaussian noise
        if noise_scale > 0:
            noise = rng.randn(*data.shape) * (noise_scale * self.config.sigma)
            data += noise

        return data


def _regular_simplex_centers(n_clusters, dim, radius):
    """
    Construct n_clusters equidistant points on a hypersphere in R^dim.
    Possible when n_clusters <= dim + 1.
    """
    # Simplex in R^{n_clusters}
    X = np.eye(n_clusters) - np.ones((n_clusters, n_clusters)) / n_clusters

    # Reduce to rank n_clusters - 1 coordinates
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    centers = U[:, : n_clusters - 1] * S[: n_clusters - 1]

    # Pad to requested dimension
    if dim > n_clusters - 1:
        centers = np.pad(
            centers, ((0, 0), (0, dim - (n_clusters - 1))), mode="constant"
        )

    # Normalize to desired radius
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    centers *= radius

    return centers


def _repulsive_hypersphere_centers(
    n_clusters: int,
    dim: int,
    radius: float,
    rng,
    n_iter=1000,
    step_size=0.05,
    eps=1e-8,
):
    """
    Approximate evenly spaced points on a hypersphere using repulsion.
    """

    centers = rng.normal(size=(n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    for it in range(n_iter):
        diff = centers[:, None, :] - centers[None, :, :]  # (K, K, dim)
        dist_sq = np.sum(diff**2, axis=2) + eps

        np.fill_diagonal(dist_sq, np.inf)

        # Repulsive force: stronger for nearby points
        forces = np.sum(diff / (dist_sq[..., None] ** 1.5), axis=1)

        # Project force onto tangent space of the sphere
        radial_component = np.sum(forces * centers, axis=1, keepdims=True) * centers
        tangent_forces = forces - radial_component

        # Decaying step size improves stability
        eta = step_size / np.sqrt(it + 1)
        centers += eta * tangent_forces

        # Project back to unit sphere
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    return radius * centers


def _hypersphere_centers(n_clusters, dim, radius, rng):
    """
    Generate cluster centers approximately evenly spaced on a dim-dimensional hypersphere.
    """
    if dim < 1:
        raise ValueError("dim must be at least 1.")

    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1.")

    if n_clusters == 1:
        center = np.zeros((1, dim))
        center[0, 0] = radius
        return center

    if dim == 1:
        if n_clusters > 2:
            raise ValueError(
                "In dim=1, the hypersphere consists only of two points: {-radius, +radius}. "
                "Use n_clusters <= 2 or dim >= 2."
            )
        return np.array([[-radius], [radius]])[:n_clusters]

    if dim == 2:
        angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
        return radius * np.column_stack([np.cos(angles), np.sin(angles)])

    if n_clusters <= dim + 1:
        return _regular_simplex_centers(n_clusters, dim, radius)

    return _repulsive_hypersphere_centers(
        n_clusters=n_clusters,
        dim=dim,
        radius=radius,
        rng=rng,
    )


def _build_gaussian_config(values):
    return build_dataclass_config(GaussianClusterConfig, values)


def _build_gaussian_transform(values, dataset):
    return build_transform_config(
        GaussianClusterTransformConfig,
        values,
        dataset,
        inherit_from_dataset={"seed": "seed"},
    )


DATASET_SPEC = register_dataset(
    DatasetSpec(
        name="gaussian_cluster",
        dataset_type=GaussianClusterDataset,
        config_type=GaussianClusterConfig,
        transform_config_type=GaussianClusterTransformConfig,
        config_builder=_build_gaussian_config,
        transform_builder=_build_gaussian_transform,
        sweep_builders={
            "t": linear_sweep(0.0, 1.0),
            "noise_scale": linear_sweep(0.1, 1.0),
            "dim": integer_sweep(2, lambda dataset: dataset.config.dim),
            "n_clusters": integer_sweep(
                1,
                GaussianClusterTransformConfig().n_clusters,
            ),
        },
    )
)
