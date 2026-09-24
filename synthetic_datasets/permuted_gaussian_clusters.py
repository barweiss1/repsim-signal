import heapq
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from synthetic_datasets.base_dataset import BaseDataset, DatasetConfig, TransformConfig
from synthetic_datasets.gaussian_cluster import _hypersphere_centers
from synthetic_datasets.registry import DatasetSpec
from synthetic_datasets.registry import build_dataclass_config
from synthetic_datasets.registry import build_transform_config
from synthetic_datasets.registry import integer_sweep
from synthetic_datasets.registry import linear_sweep
from synthetic_datasets.registry import register_dataset

TRANSFORM_MODES = frozenset({"resample", "modify_base"})


@dataclass
class PermutedGaussianClustersConfig(DatasetConfig):
    n_clusters: int = 10
    noise_scale: float = 0.1
    balanced: bool = False
    center_sampling: str = "hypersphere"


@dataclass
class PermutedGaussianClustersTransformConfig(TransformConfig):
    n_permute: int = 0
    noise_scale: float = 0.1
    seed: int = 42
    center_sampling: Optional[str] = None
    noise_distribution: str = "gaussian"
    student_t_df: float = 3.0
    cluster_noise_scales: Optional[Sequence[float]] = None
    cluster_mixing_probability: float = 0.0
    transform_mode: str = "resample"

    def __post_init__(self):
        if self.transform_mode not in TRANSFORM_MODES:
            raise ValueError(
                f"transform_mode must be one of {sorted(TRANSFORM_MODES)}."
            )


class PermutedGaussianClustersDataset(BaseDataset):
    VALID_CENTER_SAMPLING = {"hypersphere", "uniform_grid", "nonuniform_grid"}
    VALID_NOISE_DISTRIBUTIONS = {"gaussian", "uniform", "student_t"}
    VALID_TRANSFORM_MODES = TRANSFORM_MODES

    def __init__(self, config: PermutedGaussianClustersConfig):
        super().__init__(config)

    @staticmethod
    def _rng(seed: int, offset: int = 0):
        return np.random.RandomState((int(seed) + offset) % (2**32))

    def _validate_config(self):
        if self.config.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if not np.isfinite(self.config.noise_scale) or self.config.noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative.")
        self._validate_center_sampling(self.config.center_sampling)

        minimum_unbalanced_points = (
            self.config.n_clusters * (self.config.n_clusters + 1) // 2
        )
        if (
            not self.config.balanced
            and self.config.n_points < minimum_unbalanced_points
        ):
            raise ValueError(
                "Unbalanced clusters require n_points >= "
                "n_clusters * (n_clusters + 1) / 2 so every cluster size is distinct and positive."
            )

    def _validate_center_sampling(self, center_sampling):
        if center_sampling not in self.VALID_CENTER_SAMPLING:
            raise ValueError(
                f"center_sampling must be one of {sorted(self.VALID_CENTER_SAMPLING)}."
            )
        if center_sampling != "hypersphere" and self.config.dim != 2:
            raise ValueError("Cartesian center sampling currently requires dim=2.")

    @staticmethod
    def _grid_axis_coordinates(size, sampling, rng):
        if size == 1:
            return np.array([0.0])
        if sampling == "uniform_grid" or size == 2:
            return np.linspace(-1.0, 1.0, size)

        interior = np.sort(rng.uniform(-1.0, 1.0, size=size - 2))
        return np.concatenate((np.array([-1.0]), interior, np.array([1.0])))

    def _make_grid_centers(self, center_sampling):
        rows = int(np.floor(np.sqrt(self.config.n_clusters)))
        columns = int(np.ceil(self.config.n_clusters / rows))
        grid_shape = (rows, columns)

        spacing_rng = self._rng(self.config.seed, offset=4)
        row_coordinates = self._grid_axis_coordinates(
            rows,
            center_sampling,
            spacing_rng,
        )
        column_coordinates = self._grid_axis_coordinates(
            columns,
            center_sampling,
            spacing_rng,
        )
        column_grid, row_grid = np.meshgrid(column_coordinates, row_coordinates)
        grid_sites = np.column_stack((column_grid.ravel(), row_grid.ravel()))

        site_rng = self._rng(self.config.seed, offset=3)
        if len(grid_sites) == self.config.n_clusters:
            selected_indices = np.arange(len(grid_sites))
        else:
            selected_indices = np.sort(
                site_rng.choice(
                    len(grid_sites),
                    size=self.config.n_clusters,
                    replace=False,
                )
            )
        return grid_sites[selected_indices], grid_shape, selected_indices

    def _make_cluster_centers(self, center_sampling):
        self._validate_center_sampling(center_sampling)
        if center_sampling == "hypersphere":
            centers = _hypersphere_centers(
                n_clusters=self.config.n_clusters,
                dim=self.config.dim,
                radius=1.0,
                rng=self._rng(self.config.seed),
            )
            return centers, None, None
        return self._make_grid_centers(center_sampling)

    def _make_assignments(self):
        rng = self._rng(self.config.seed, offset=1)
        if self.config.balanced:
            assignments = np.arange(self.config.n_points) % self.config.n_clusters
            rng.shuffle(assignments)
            self.cluster_sizes = np.bincount(
                assignments,
                minlength=self.config.n_clusters,
            )
            return assignments

        weights = np.arange(1, self.config.n_clusters + 1)
        cluster_sizes = np.floor(self.config.n_points * weights / weights.sum()).astype(
            int
        )
        remainder = self.config.n_points - cluster_sizes.sum()
        if remainder > 0:
            cluster_sizes[-remainder:] += 1

        rng.shuffle(cluster_sizes)
        assignments = np.repeat(np.arange(self.config.n_clusters), cluster_sizes)
        rng.shuffle(assignments)
        self.cluster_sizes = cluster_sizes
        return assignments

    def _resolve_noise_scales(self, noise_scale, cluster_noise_scales):
        try:
            scalar_noise_scale = float(noise_scale)
        except (TypeError, ValueError):
            raise ValueError("noise_scale must be finite and nonnegative.") from None
        if not np.isfinite(scalar_noise_scale) or scalar_noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative.")
        if cluster_noise_scales is None:
            return np.full(self.config.n_clusters, scalar_noise_scale)

        scales = np.asarray(cluster_noise_scales, dtype=float)
        if scales.ndim != 1 or len(scales) != self.config.n_clusters:
            raise ValueError(
                "cluster_noise_scales must contain exactly one scale per cluster."
            )
        if not np.all(np.isfinite(scales)) or np.any(scales < 0):
            raise ValueError(
                "cluster_noise_scales must contain finite, nonnegative values."
            )
        return scales.copy()

    def _sample_unit_noise(self, noise_rng, noise_distribution, student_t_df):
        shape = (self.config.n_points, self.config.dim)
        if noise_distribution not in self.VALID_NOISE_DISTRIBUTIONS:
            raise ValueError(
                "noise_distribution must be one of "
                f"{sorted(self.VALID_NOISE_DISTRIBUTIONS)}."
            )
        if noise_distribution == "gaussian":
            return noise_rng.randn(*shape)
        if noise_distribution == "uniform":
            bound = np.sqrt(3.0)
            return noise_rng.uniform(-bound, bound, size=shape)

        try:
            degrees_of_freedom = float(student_t_df)
        except (TypeError, ValueError):
            raise ValueError(
                "student_t_df must be finite and greater than 2."
            ) from None
        if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2:
            raise ValueError("student_t_df must be finite and greater than 2.")
        standardization = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
        return (
            noise_rng.standard_t(
                degrees_of_freedom,
                size=shape,
            )
            * standardization
        )

    def _sample(
        self,
        cluster_centers,
        center_mapping,
        noise_scale,
        seed,
        noise_distribution="gaussian",
        student_t_df=3.0,
        cluster_noise_scales=None,
        target_cluster_assignments=None,
    ):
        noise_rng = self._rng(seed, offset=1)
        if target_cluster_assignments is None:
            target_cluster_assignments = center_mapping[self.cluster_assignments]
        means = cluster_centers[target_cluster_assignments]
        effective_scales = self._resolve_noise_scales(
            noise_scale,
            cluster_noise_scales,
        )
        unit_noise = self._sample_unit_noise(
            noise_rng,
            noise_distribution,
            student_t_df,
        )
        row_scales = effective_scales[self.cluster_assignments, None]
        noise = unit_noise * row_scales
        return means + noise

    def generate(self):
        self._validate_config()
        centers, grid_shape, selected_grid_indices = self._make_cluster_centers(
            self.config.center_sampling
        )
        self.cluster_centers = centers
        self.base_cluster_centers = centers.copy()
        self.current_cluster_centers = None
        self.base_center_sampling = self.config.center_sampling
        self.current_center_sampling = None
        self.grid_shape = grid_shape
        self.selected_grid_indices = selected_grid_indices
        self.base_grid_shape = grid_shape
        self.base_selected_grid_indices = (
            selected_grid_indices.copy() if selected_grid_indices is not None else None
        )
        self.current_grid_shape = None
        self.current_selected_grid_indices = None
        self.cluster_assignments = self._make_assignments()
        self.plot_labels = self.cluster_assignments.copy()

        identity_mapping = np.arange(self.config.n_clusters)
        self.last_center_mapping = identity_mapping.copy()
        self.last_permuted_clusters = np.array([], dtype=int)
        self.last_cluster_centers = centers.copy()
        self.last_center_sampling = self.config.center_sampling
        self.last_grid_shape = grid_shape
        self.last_selected_grid_indices = (
            selected_grid_indices.copy() if selected_grid_indices is not None else None
        )
        initial_noise_scales = np.full(
            self.config.n_clusters,
            float(self.config.noise_scale),
        )
        self.last_noise_distribution = "gaussian"
        self.last_student_t_df = 3.0
        self.last_cluster_noise_scales = initial_noise_scales.copy()
        self.base_noise_distribution = "gaussian"
        self.base_student_t_df = 3.0
        self.base_cluster_noise_scales = initial_noise_scales.copy()
        self.current_noise_distribution = None
        self.current_student_t_df = None
        self.current_cluster_noise_scales = None
        self.current_cluster_partner_mapping = None
        self.current_mixed_point_mask = None
        self.current_target_cluster_assignments = None
        self.current_source_cluster_assignments = None
        self.current_swap_partner_indices = None
        self._last_cluster_partner_mapping = None
        self._last_mixed_point_mask = None
        self._last_target_cluster_assignments = self.cluster_assignments.copy()
        self._last_source_cluster_assignments = self.cluster_assignments.copy()
        self._last_swap_partner_indices = np.arange(self.config.n_points)
        self.init_data = self._sample(
            centers,
            identity_mapping,
            noise_scale=self.config.noise_scale,
            seed=self.config.seed + 1,
        )
        self.base_data = self.init_data.copy()
        self.base_source_cluster_assignments = self.cluster_assignments.copy()
        self.current_data = None

    def _make_center_mapping(self, n_permute, seed):
        if n_permute < 0:
            raise ValueError("n_permute must be nonnegative.")
        if n_permute == 1:
            raise ValueError(
                "n_permute=1 cannot move exactly one cluster; use 0 or at least 2."
            )
        if n_permute > self.config.n_clusters:
            raise ValueError("n_permute cannot exceed n_clusters.")

        mapping = np.arange(self.config.n_clusters)
        if n_permute == 0:
            return mapping, np.array([], dtype=int)

        permutation_rng = self._rng(seed)
        selected = permutation_rng.choice(
            self.config.n_clusters,
            size=n_permute,
            replace=False,
        )
        destinations = selected.copy()
        while np.any(destinations == selected):
            destinations = selected[permutation_rng.permutation(n_permute)]
        mapping[selected] = destinations
        return mapping, selected

    def _make_mixed_target_assignments(
        self,
        center_mapping,
        cluster_mixing_probability,
        seed,
    ):
        probability = self._validate_mixing_probability(cluster_mixing_probability)
        target_assignments = center_mapping[self.cluster_assignments].copy()
        partner_mapping = np.arange(self.config.n_clusters)
        if self.config.n_clusters >= 2:
            partner_rng = self._rng(seed, offset=2)
            while np.any(partner_mapping == np.arange(self.config.n_clusters)):
                partner_mapping = partner_rng.permutation(self.config.n_clusters)

        selection_rng = self._rng(seed, offset=3)
        mixed_mask = selection_rng.rand(self.config.n_points) < probability
        target_assignments[mixed_mask] = partner_mapping[target_assignments[mixed_mask]]
        return partner_mapping, mixed_mask, target_assignments

    def _validate_mixing_probability(self, cluster_mixing_probability):
        try:
            probability = float(cluster_mixing_probability)
        except (TypeError, ValueError):
            raise ValueError(
                "cluster_mixing_probability must be finite and between 0 and 1."
            ) from None
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(
                "cluster_mixing_probability must be finite and between 0 and 1."
            )
        if probability > 0 and self.config.n_clusters < 2:
            raise ValueError("Positive cluster mixing requires at least two clusters.")
        return probability

    def _transform_resample(
        self, transform_config: PermutedGaussianClustersTransformConfig
    ) -> np.ndarray:
        center_sampling = (
            self.config.center_sampling
            if transform_config.center_sampling is None
            else transform_config.center_sampling
        )
        centers, grid_shape, selected_grid_indices = self._make_cluster_centers(
            center_sampling
        )

        mapping, selected = self._make_center_mapping(
            n_permute=int(transform_config.n_permute),
            seed=transform_config.seed,
        )
        partner_mapping, mixed_mask, target_assignments = (
            self._make_mixed_target_assignments(
                mapping,
                transform_config.cluster_mixing_probability,
                transform_config.seed,
            )
        )
        data = self._sample(
            centers,
            mapping,
            noise_scale=transform_config.noise_scale,
            seed=transform_config.seed,
            noise_distribution=transform_config.noise_distribution,
            student_t_df=transform_config.student_t_df,
            cluster_noise_scales=transform_config.cluster_noise_scales,
            target_cluster_assignments=target_assignments,
        )
        self.last_center_mapping = mapping
        self.last_permuted_clusters = selected
        self.last_cluster_centers = centers
        self.last_center_sampling = center_sampling
        self.last_grid_shape = grid_shape
        self.last_selected_grid_indices = selected_grid_indices
        self.last_noise_distribution = transform_config.noise_distribution
        self.last_student_t_df = transform_config.student_t_df
        self.last_cluster_noise_scales = self._resolve_noise_scales(
            transform_config.noise_scale,
            transform_config.cluster_noise_scales,
        )
        self._last_cluster_partner_mapping = partner_mapping
        self._last_mixed_point_mask = mixed_mask
        self._last_target_cluster_assignments = target_assignments
        self._last_source_cluster_assignments = self.cluster_assignments.copy()
        self._last_swap_partner_indices = np.arange(self.config.n_points)
        return data

    def _cross_cluster_swap_pairs(
        self,
        source_assignments: np.ndarray,
        probability: float,
        seed: int,
    ) -> list[tuple[int, int]]:
        selected = self._rng(seed, offset=3).rand(self.config.n_points) < probability
        pairing_rng = self._rng(seed, offset=2)
        buckets = {}
        for cluster in range(self.config.n_clusters):
            indices = np.flatnonzero(selected & (source_assignments == cluster))
            pairing_rng.shuffle(indices)
            if len(indices):
                buckets[cluster] = list(indices)

        tie_order = pairing_rng.permutation(self.config.n_clusters)
        priority = np.empty(self.config.n_clusters, dtype=int)
        priority[tie_order] = np.arange(self.config.n_clusters)
        heap = [
            (-len(indices), int(priority[cluster]), cluster)
            for cluster, indices in buckets.items()
        ]
        heapq.heapify(heap)
        pairs = []
        while len(heap) >= 2:
            first_count, first_priority, first_cluster = heapq.heappop(heap)
            second_count, second_priority, second_cluster = heapq.heappop(heap)
            pairs.append((buckets[first_cluster].pop(), buckets[second_cluster].pop()))
            first_count += 1
            second_count += 1
            if first_count:
                heapq.heappush(heap, (first_count, first_priority, first_cluster))
            if second_count:
                heapq.heappush(heap, (second_count, second_priority, second_cluster))
        return pairs

    def _transform_modify_base(
        self, transform_config: PermutedGaussianClustersTransformConfig
    ) -> np.ndarray:
        if (
            transform_config.center_sampling is not None
            and transform_config.center_sampling != self.base_center_sampling
        ):
            raise ValueError(
                "modify_base must reuse the established base center_sampling."
            )
        probability = self._validate_mixing_probability(
            transform_config.cluster_mixing_probability
        )
        mapping, selected = self._make_center_mapping(
            n_permute=int(transform_config.n_permute),
            seed=transform_config.seed,
        )
        effective_scales = self._resolve_noise_scales(
            transform_config.noise_scale,
            transform_config.cluster_noise_scales,
        )
        source_assignments = self.base_source_cluster_assignments.copy()
        unit_noise = self._sample_unit_noise(
            self._rng(transform_config.seed, offset=1),
            transform_config.noise_distribution,
            transform_config.student_t_df,
        )
        data = self.base_data.copy()
        data += unit_noise * effective_scales[source_assignments, None]

        swap_partners = np.arange(self.config.n_points)
        mixed_mask = np.zeros(self.config.n_points, dtype=bool)
        for first, second in self._cross_cluster_swap_pairs(
            source_assignments,
            probability,
            transform_config.seed,
        ):
            data[[first, second]] = data[[second, first]]
            source_assignments[[first, second]] = source_assignments[[second, first]]
            swap_partners[[first, second]] = [second, first]
            mixed_mask[[first, second]] = True

        target_assignments = mapping[source_assignments]
        data += (
            self.base_cluster_centers[target_assignments]
            - self.base_cluster_centers[source_assignments]
        )

        self.last_center_mapping = mapping
        self.last_permuted_clusters = selected
        self.last_cluster_centers = self.base_cluster_centers.copy()
        self.last_center_sampling = self.base_center_sampling
        self.last_grid_shape = self.base_grid_shape
        self.last_selected_grid_indices = (
            self.base_selected_grid_indices.copy()
            if self.base_selected_grid_indices is not None
            else None
        )
        self.last_noise_distribution = transform_config.noise_distribution
        self.last_student_t_df = transform_config.student_t_df
        self.last_cluster_noise_scales = effective_scales
        self._last_cluster_partner_mapping = None
        self._last_mixed_point_mask = mixed_mask
        self._last_target_cluster_assignments = target_assignments
        self._last_source_cluster_assignments = source_assignments
        self._last_swap_partner_indices = swap_partners
        return data

    def transform(
        self, transform_config: PermutedGaussianClustersTransformConfig
    ) -> np.ndarray:
        if transform_config.transform_mode not in self.VALID_TRANSFORM_MODES:
            raise ValueError(
                "transform_mode must be one of "
                f"{sorted(self.VALID_TRANSFORM_MODES)}."
            )
        if int(transform_config.n_permute) != transform_config.n_permute:
            raise ValueError("n_permute must be an integer.")
        if transform_config.transform_mode == "modify_base":
            return self._transform_modify_base(transform_config)
        return self._transform_resample(transform_config)

    def transform_base(
        self, transform_config: PermutedGaussianClustersTransformConfig
    ) -> np.ndarray:
        if transform_config.transform_mode == "modify_base":
            raise ValueError(
                "transform_base does not accept transform_mode='modify_base'."
            )
        self.base_data = self._owned_transform_result(self.transform(transform_config))
        self.base_cluster_centers = self.last_cluster_centers.copy()
        self.base_center_sampling = self.last_center_sampling
        self.base_grid_shape = self.last_grid_shape
        self.base_selected_grid_indices = (
            self.last_selected_grid_indices.copy()
            if self.last_selected_grid_indices is not None
            else None
        )
        self.cluster_centers = self.base_cluster_centers
        self.grid_shape = self.base_grid_shape
        self.selected_grid_indices = self.base_selected_grid_indices
        self.base_noise_distribution = self.last_noise_distribution
        self.base_student_t_df = self.last_student_t_df
        self.base_cluster_noise_scales = self.last_cluster_noise_scales.copy()
        self.base_source_cluster_assignments = (
            self._last_target_cluster_assignments.copy()
        )
        return self.base_data.copy()

    def transform_current(
        self, transform_config: PermutedGaussianClustersTransformConfig
    ) -> np.ndarray:
        self.current_data = self._owned_transform_result(
            self.transform(transform_config)
        )
        self.current_cluster_centers = self.last_cluster_centers.copy()
        self.current_center_sampling = self.last_center_sampling
        self.current_grid_shape = self.last_grid_shape
        self.current_selected_grid_indices = (
            self.last_selected_grid_indices.copy()
            if self.last_selected_grid_indices is not None
            else None
        )
        self.current_noise_distribution = self.last_noise_distribution
        self.current_student_t_df = self.last_student_t_df
        self.current_cluster_noise_scales = self.last_cluster_noise_scales.copy()
        self.current_cluster_partner_mapping = (
            self._last_cluster_partner_mapping.copy()
            if self._last_cluster_partner_mapping is not None
            else None
        )
        self.current_mixed_point_mask = self._last_mixed_point_mask.copy()
        self.current_target_cluster_assignments = (
            self._last_target_cluster_assignments.copy()
        )
        self.current_source_cluster_assignments = (
            self._last_source_cluster_assignments.copy()
        )
        self.current_swap_partner_indices = self._last_swap_partner_indices.copy()
        return self.current_data.copy()


def _build_permuted_config(values):
    overrides = {}
    base_transform = values.get("base_transform_params")
    if isinstance(base_transform, Mapping):
        for field_name in ("n_clusters", "noise_scale"):
            if (
                values.get(field_name) is None
                and base_transform.get(field_name) is not None
            ):
                overrides[field_name] = base_transform[field_name]
    return build_dataclass_config(
        PermutedGaussianClustersConfig,
        values,
        overrides=overrides,
    )


def _build_permuted_transform(values, dataset):
    return build_transform_config(
        PermutedGaussianClustersTransformConfig,
        values,
        dataset,
        inherit_from_dataset={"seed": "seed", "noise_scale": "noise_scale"},
    )


def _permutation_sweep(dataset, sweep_len):
    del sweep_len
    if dataset.config.n_clusters < 2:
        return np.asarray([0], dtype=int)
    return np.concatenate(
        (np.asarray([0], dtype=int), np.arange(2, dataset.config.n_clusters + 1))
    )


DATASET_SPEC = register_dataset(
    DatasetSpec(
        name="permuted_gaussian_clusters",
        dataset_type=PermutedGaussianClustersDataset,
        config_type=PermutedGaussianClustersConfig,
        transform_config_type=PermutedGaussianClustersTransformConfig,
        config_builder=_build_permuted_config,
        transform_builder=_build_permuted_transform,
        sweep_builders={
            "noise_scale": linear_sweep(0.1, 1.0),
            "dim": integer_sweep(2, lambda dataset: dataset.config.dim),
            "n_permute": _permutation_sweep,
            "cluster_mixing_probability": linear_sweep(0.0, 0.5),
        },
    )
)
