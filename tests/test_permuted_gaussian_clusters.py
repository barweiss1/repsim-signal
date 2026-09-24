import unittest

import numpy as np

from manifold_repsim.experiments.legacy import (
    get_data_param_sweep_values,
    run_param_sweep,
)
from synthetic_datasets import get_transform_config, init_dataset
from synthetic_datasets.permuted_gaussian_clusters import (
    PermutedGaussianClustersConfig,
    PermutedGaussianClustersDataset,
    PermutedGaussianClustersTransformConfig,
)


class PermutedGaussianClustersTests(unittest.TestCase):
    def make_dataset(self, **overrides):
        params = {
            "n_points": 120,
            "dim": 2,
            "seed": 7,
            "n_clusters": 5,
            "noise_scale": 0.1,
            "balanced": False,
        }
        params.update(overrides)
        return PermutedGaussianClustersDataset(PermutedGaussianClustersConfig(**params))

    def test_centers_assignments_and_generation_are_deterministic(self):
        first = self.make_dataset()
        second = self.make_dataset()

        np.testing.assert_allclose(np.linalg.norm(first.cluster_centers, axis=1), 1.0)
        np.testing.assert_array_equal(first.plot_labels, first.cluster_assignments)
        np.testing.assert_allclose(first.cluster_centers, second.cluster_centers)
        np.testing.assert_array_equal(first.cluster_sizes, second.cluster_sizes)
        np.testing.assert_array_equal(
            first.cluster_assignments, second.cluster_assignments
        )
        np.testing.assert_allclose(first.init_data, second.init_data)
        self.assertEqual(first.config.center_sampling, "hypersphere")
        self.assertIsNone(first.grid_shape)
        self.assertIsNone(first.selected_grid_indices)

    def test_uniform_grid_shape_spacing_and_site_selection(self):
        dataset = self.make_dataset(center_sampling="uniform_grid")
        expected_indices = np.sort(
            np.random.RandomState(10).choice(6, size=5, replace=False)
        )

        self.assertEqual(dataset.grid_shape, (2, 3))
        np.testing.assert_array_equal(dataset.selected_grid_indices, expected_indices)
        x_coordinates = np.unique(dataset.cluster_centers[:, 0])
        y_coordinates = np.unique(dataset.cluster_centers[:, 1])
        np.testing.assert_allclose(np.diff(x_coordinates), np.diff(x_coordinates)[0])
        np.testing.assert_allclose(np.diff(y_coordinates), np.diff(y_coordinates)[0])

    def test_nonuniform_grid_is_reproducible_and_irregular(self):
        first = self.make_dataset(center_sampling="nonuniform_grid")
        second = self.make_dataset(center_sampling="nonuniform_grid")
        x_coordinates = np.unique(first.cluster_centers[:, 0])

        np.testing.assert_allclose(first.cluster_centers, second.cluster_centers)
        self.assertFalse(np.allclose(np.diff(x_coordinates), np.diff(x_coordinates)[0]))

    def test_grid_spacing_preserves_assignments_and_permutation(self):
        uniform = self.make_dataset(center_sampling="uniform_grid")
        nonuniform = self.make_dataset(center_sampling="nonuniform_grid")
        transform_config = PermutedGaussianClustersTransformConfig(
            n_permute=4,
            noise_scale=0.1,
            seed=19,
        )
        uniform_data = uniform.transform(transform_config)
        nonuniform_data = nonuniform.transform(transform_config)

        np.testing.assert_array_equal(
            uniform.selected_grid_indices,
            nonuniform.selected_grid_indices,
        )
        np.testing.assert_array_equal(uniform.cluster_sizes, nonuniform.cluster_sizes)
        np.testing.assert_array_equal(
            uniform.cluster_assignments,
            nonuniform.cluster_assignments,
        )
        np.testing.assert_array_equal(
            uniform.last_permuted_clusters,
            nonuniform.last_permuted_clusters,
        )
        np.testing.assert_array_equal(
            uniform.last_center_mapping,
            nonuniform.last_center_mapping,
        )
        self.assertFalse(
            np.allclose(uniform.cluster_centers, nonuniform.cluster_centers)
        )
        self.assertFalse(np.allclose(uniform_data, nonuniform_data))

    def test_base_and_current_transforms_can_use_different_center_sampling(self):
        dataset = self.make_dataset(center_sampling="hypersphere", noise_scale=0.0)
        assignments_before = dataset.cluster_assignments.copy()
        sizes_before = dataset.cluster_sizes.copy()

        base_data = dataset.transform_base(
            PermutedGaussianClustersTransformConfig(
                n_permute=0,
                noise_scale=0.0,
                seed=17,
                center_sampling="uniform_grid",
            )
        )
        current_data = dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=0,
                noise_scale=0.0,
                seed=18,
                center_sampling="nonuniform_grid",
            )
        )

        self.assertEqual(dataset.base_center_sampling, "uniform_grid")
        self.assertEqual(dataset.current_center_sampling, "nonuniform_grid")
        np.testing.assert_allclose(
            base_data,
            dataset.base_cluster_centers[assignments_before],
        )
        np.testing.assert_allclose(
            current_data,
            dataset.current_cluster_centers[assignments_before],
        )
        np.testing.assert_array_equal(dataset.cluster_assignments, assignments_before)
        np.testing.assert_array_equal(dataset.cluster_sizes, sizes_before)
        np.testing.assert_array_equal(dataset.plot_labels, assignments_before)
        np.testing.assert_array_equal(
            dataset.base_selected_grid_indices,
            dataset.current_selected_grid_indices,
        )
        self.assertFalse(
            np.allclose(dataset.base_cluster_centers, dataset.current_cluster_centers)
        )

    def test_all_base_and_transform_center_sampling_pairs_preserve_labels(self):
        center_sampling_values = (
            "hypersphere",
            "uniform_grid",
            "nonuniform_grid",
        )
        for base_sampling in center_sampling_values:
            for current_sampling in center_sampling_values:
                with self.subTest(
                    base_sampling=base_sampling,
                    current_sampling=current_sampling,
                ):
                    dataset = self.make_dataset(
                        center_sampling=base_sampling,
                        noise_scale=0.0,
                    )
                    dataset.transform_base(
                        PermutedGaussianClustersTransformConfig(
                            n_permute=0,
                            noise_scale=0.0,
                            seed=17,
                            center_sampling=base_sampling,
                        )
                    )
                    dataset.transform_current(
                        PermutedGaussianClustersTransformConfig(
                            n_permute=0,
                            noise_scale=0.0,
                            seed=18,
                            center_sampling=current_sampling,
                        )
                    )

                    np.testing.assert_allclose(
                        dataset.base_data,
                        dataset.base_cluster_centers[dataset.cluster_assignments],
                    )
                    np.testing.assert_allclose(
                        dataset.current_data,
                        dataset.current_cluster_centers[dataset.cluster_assignments],
                    )

    def test_permutation_maps_labels_to_transform_layout_centers(self):
        dataset = self.make_dataset(center_sampling="hypersphere", noise_scale=0.0)
        dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=4,
                noise_scale=0.0,
                seed=19,
                center_sampling="uniform_grid",
            )
        )

        expected = dataset.current_cluster_centers[
            dataset.last_center_mapping[dataset.cluster_assignments]
        ]
        np.testing.assert_allclose(dataset.current_data, expected)
        moved = np.flatnonzero(dataset.last_center_mapping != np.arange(5))
        np.testing.assert_array_equal(
            np.sort(moved),
            np.sort(dataset.last_permuted_clusters),
        )

    def test_transform_layout_inherits_default_and_uses_dataset_seed(self):
        dataset = self.make_dataset(center_sampling="nonuniform_grid")
        inherited = PermutedGaussianClustersTransformConfig(
            n_permute=0,
            noise_scale=0.0,
            seed=100,
        )
        dataset.transform_current(inherited)
        first_centers = dataset.current_cluster_centers.copy()

        dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=0,
                noise_scale=0.0,
                seed=101,
                center_sampling="nonuniform_grid",
            )
        )

        self.assertEqual(dataset.current_center_sampling, "nonuniform_grid")
        np.testing.assert_allclose(first_centers, dataset.cluster_centers)
        np.testing.assert_allclose(first_centers, dataset.current_cluster_centers)

    def test_unbalanced_assignments_have_distinct_linear_weight_counts(self):
        dataset = self.make_dataset(n_points=121)
        observed_counts = np.bincount(dataset.cluster_assignments, minlength=5)

        np.testing.assert_array_equal(observed_counts, dataset.cluster_sizes)
        np.testing.assert_array_equal(
            np.sort(dataset.cluster_sizes),
            np.array([8, 16, 24, 32, 41]),
        )
        self.assertEqual(dataset.cluster_sizes.sum(), 121)
        self.assertTrue(np.all(dataset.cluster_sizes > 0))
        self.assertEqual(len(np.unique(dataset.cluster_sizes)), 5)

    def test_balanced_assignments_have_nearly_equal_counts(self):
        dataset = self.make_dataset(n_points=23, balanced=True)
        counts = np.bincount(dataset.cluster_assignments, minlength=5)

        self.assertLessEqual(counts.max() - counts.min(), 1)
        np.testing.assert_array_equal(counts, dataset.cluster_sizes)
        np.testing.assert_array_equal(dataset.plot_labels, dataset.cluster_assignments)

    def test_transform_moves_exactly_selected_clusters(self):
        dataset = self.make_dataset(noise_scale=0.0)
        labels_before = dataset.plot_labels.copy()
        sizes_before = dataset.cluster_sizes.copy()
        transformed = dataset.transform_base(
            PermutedGaussianClustersTransformConfig(
                n_permute=4,
                noise_scale=0.0,
                seed=19,
            )
        )

        moved = np.flatnonzero(dataset.last_center_mapping != np.arange(5))
        np.testing.assert_array_equal(
            np.sort(moved), np.sort(dataset.last_permuted_clusters)
        )
        self.assertEqual(len(moved), 4)
        np.testing.assert_array_equal(dataset.plot_labels, labels_before)
        np.testing.assert_array_equal(dataset.cluster_sizes, sizes_before)
        np.testing.assert_array_equal(
            np.bincount(dataset.cluster_assignments, minlength=5),
            sizes_before,
        )
        expected = dataset.cluster_centers[
            dataset.last_center_mapping[dataset.cluster_assignments]
        ]
        np.testing.assert_allclose(transformed, expected)

    def test_all_clusters_are_resampled_and_seeded(self):
        dataset = self.make_dataset()
        config = PermutedGaussianClustersTransformConfig(
            n_permute=0,
            noise_scale=0.2,
            seed=100,
        )
        first = dataset.transform(config)
        repeated = dataset.transform(config)
        second = dataset.transform(
            PermutedGaussianClustersTransformConfig(
                n_permute=0,
                noise_scale=0.2,
                seed=101,
            )
        )

        np.testing.assert_allclose(first, repeated)
        self.assertTrue(np.all(np.linalg.norm(first - second, axis=1) > 0))

    def test_gaussian_noise_preserves_existing_seeded_behavior(self):
        dataset = self.make_dataset()
        seed = 100
        noise_scale = 0.2
        transformed = dataset.transform_base(
            PermutedGaussianClustersTransformConfig(
                n_permute=0,
                noise_scale=noise_scale,
                seed=seed,
            )
        )
        means = dataset.cluster_centers[dataset.cluster_assignments]
        expected_noise = (
            np.random.RandomState(seed + 1).randn(dataset.config.n_points, 2)
            * noise_scale
        )

        np.testing.assert_allclose(transformed, means + expected_noise)
        self.assertEqual(dataset.last_noise_distribution, "gaussian")
        self.assertEqual(dataset.base_noise_distribution, "gaussian")
        np.testing.assert_allclose(
            dataset.last_cluster_noise_scales,
            np.full(dataset.config.n_clusters, noise_scale),
        )
        np.testing.assert_allclose(
            dataset.base_cluster_noise_scales,
            np.full(dataset.config.n_clusters, noise_scale),
        )

    def test_uniform_noise_is_unit_variance_before_scaling(self):
        dataset = self.make_dataset()
        seed = 100
        noise_scale = 0.2
        transformed = dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=0,
                noise_scale=noise_scale,
                seed=seed,
                noise_distribution="uniform",
            )
        )
        means = dataset.current_cluster_centers[dataset.cluster_assignments]
        bound = np.sqrt(3.0)
        expected_unit_noise = np.random.RandomState(seed + 1).uniform(
            -bound,
            bound,
            size=(dataset.config.n_points, dataset.config.dim),
        )

        np.testing.assert_allclose(
            transformed,
            means + expected_unit_noise * noise_scale,
        )
        self.assertEqual(dataset.current_noise_distribution, "uniform")

    def test_student_t_noise_is_unit_variance_before_scaling(self):
        dataset = self.make_dataset()
        seed = 100
        noise_scale = 0.2
        degrees_of_freedom = 5.0
        transformed = dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=0,
                noise_scale=noise_scale,
                seed=seed,
                noise_distribution="student_t",
                student_t_df=degrees_of_freedom,
            )
        )
        means = dataset.current_cluster_centers[dataset.cluster_assignments]
        expected_unit_noise = np.random.RandomState(seed + 1).standard_t(
            degrees_of_freedom,
            size=(dataset.config.n_points, dataset.config.dim),
        ) * np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)

        np.testing.assert_allclose(
            transformed,
            means + expected_unit_noise * noise_scale,
        )
        self.assertEqual(dataset.current_noise_distribution, "student_t")
        self.assertEqual(dataset.current_student_t_df, degrees_of_freedom)

    def test_per_cluster_noise_scales_follow_base_labels_through_permutation(self):
        dataset = self.make_dataset()
        seed = 19
        cluster_noise_scales = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
        transformed = dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=4,
                noise_scale=99.0,
                seed=seed,
                cluster_noise_scales=cluster_noise_scales,
                cluster_mixing_probability=0.5,
            )
        )
        means = dataset.current_cluster_centers[
            dataset.current_target_cluster_assignments
        ]
        unit_noise = np.random.RandomState(seed + 1).randn(
            dataset.config.n_points,
            dataset.config.dim,
        )
        row_scales = cluster_noise_scales[dataset.cluster_assignments, None]

        np.testing.assert_allclose(transformed, means + unit_noise * row_scales)
        np.testing.assert_allclose(
            dataset.current_cluster_noise_scales,
            cluster_noise_scales,
        )

    def test_zero_cluster_mixing_preserves_existing_target_assignments(self):
        default_dataset = self.make_dataset(noise_scale=0.0)
        explicit_dataset = self.make_dataset(noise_scale=0.0)
        default_data = default_dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=4,
                noise_scale=0.0,
                seed=19,
            )
        )
        explicit_data = explicit_dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=4,
                noise_scale=0.0,
                seed=19,
                cluster_mixing_probability=0.0,
            )
        )
        expected_targets = explicit_dataset.last_center_mapping[
            explicit_dataset.cluster_assignments
        ]

        np.testing.assert_array_equal(explicit_data, default_data)
        np.testing.assert_array_equal(
            explicit_dataset.current_target_cluster_assignments,
            expected_targets,
        )
        self.assertFalse(explicit_dataset.current_mixed_point_mask.any())
        np.testing.assert_allclose(
            explicit_dataset.current_data,
            explicit_dataset.current_cluster_centers[expected_targets],
        )
        self.assertFalse(hasattr(explicit_dataset, "last_mixed_point_mask"))
        self.assertFalse(hasattr(explicit_dataset, "base_mixed_point_mask"))

    def test_full_cluster_mixing_moves_post_permutation_targets_to_partners(self):
        dataset = self.make_dataset(noise_scale=0.0)
        assignments_before = dataset.cluster_assignments.copy()
        sizes_before = dataset.cluster_sizes.copy()
        dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=4,
                noise_scale=0.0,
                seed=19,
                cluster_mixing_probability=1.0,
            )
        )
        partner_mapping = dataset.current_cluster_partner_mapping
        initial_targets = dataset.last_center_mapping[dataset.cluster_assignments]
        expected_targets = partner_mapping[initial_targets]

        np.testing.assert_array_equal(
            np.sort(partner_mapping),
            np.arange(dataset.config.n_clusters),
        )
        self.assertTrue(np.all(partner_mapping != np.arange(dataset.config.n_clusters)))
        self.assertTrue(dataset.current_mixed_point_mask.all())
        np.testing.assert_array_equal(
            dataset.current_target_cluster_assignments,
            expected_targets,
        )
        np.testing.assert_allclose(
            dataset.current_data,
            dataset.current_cluster_centers[expected_targets],
        )
        np.testing.assert_array_equal(dataset.cluster_assignments, assignments_before)
        np.testing.assert_array_equal(dataset.plot_labels, assignments_before)
        np.testing.assert_array_equal(dataset.cluster_sizes, sizes_before)

    def test_cluster_mixing_masks_are_seeded_and_nested(self):
        dataset = self.make_dataset()
        seed = 23
        dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                seed=seed,
                cluster_mixing_probability=0.25,
            )
        )
        low_probability_mask = dataset.current_mixed_point_mask.copy()
        partner_mapping = dataset.current_cluster_partner_mapping.copy()
        expected_draws = np.random.RandomState(seed + 3).rand(dataset.config.n_points)

        np.testing.assert_array_equal(
            low_probability_mask,
            expected_draws < 0.25,
        )
        dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                seed=seed,
                cluster_mixing_probability=0.75,
            )
        )

        np.testing.assert_array_equal(
            dataset.current_mixed_point_mask,
            expected_draws < 0.75,
        )
        np.testing.assert_array_equal(
            dataset.current_cluster_partner_mapping,
            partner_mapping,
        )
        self.assertTrue(np.all(dataset.current_mixed_point_mask[low_probability_mask]))

    def test_modify_base_zero_transform_is_exact_and_nonaccumulating(self):
        dataset = self.make_dataset(noise_scale=0.0)
        dataset.transform_base(
            PermutedGaussianClustersTransformConfig(
                noise_scale=0.15,
                seed=17,
            )
        )
        base = dataset.base_data.copy()
        config = PermutedGaussianClustersTransformConfig(
            noise_scale=0.0,
            seed=23,
            transform_mode="modify_base",
        )

        first = dataset.transform_current(config).copy()
        second = dataset.transform_current(config).copy()

        np.testing.assert_array_equal(first, base)
        np.testing.assert_array_equal(second, base)
        np.testing.assert_array_equal(dataset.base_data, base)
        np.testing.assert_array_equal(
            dataset.current_swap_partner_indices,
            np.arange(dataset.config.n_points),
        )
        self.assertFalse(dataset.current_mixed_point_mask.any())

    def test_modify_base_adds_seeded_noise_to_existing_base(self):
        dataset = self.make_dataset(noise_scale=0.0)
        base = dataset.transform_base(
            PermutedGaussianClustersTransformConfig(
                noise_scale=0.0,
                seed=17,
            )
        ).copy()
        seed = 31
        scale = 0.2
        cases = {
            "gaussian": np.random.RandomState(seed + 1).randn(
                dataset.config.n_points,
                dataset.config.dim,
            ),
            "uniform": np.random.RandomState(seed + 1).uniform(
                -np.sqrt(3.0),
                np.sqrt(3.0),
                size=(dataset.config.n_points, dataset.config.dim),
            ),
            "student_t": np.random.RandomState(seed + 1).standard_t(
                5.0,
                size=(dataset.config.n_points, dataset.config.dim),
            )
            * np.sqrt(3.0 / 5.0),
        }
        for distribution, unit_noise in cases.items():
            with self.subTest(distribution=distribution):
                observed = dataset.transform_current(
                    PermutedGaussianClustersTransformConfig(
                        noise_scale=scale,
                        seed=seed,
                        noise_distribution=distribution,
                        student_t_df=5.0,
                        transform_mode="modify_base",
                    )
                )
                np.testing.assert_allclose(observed, base + scale * unit_noise)

        cluster_scales = np.linspace(0.0, 0.4, dataset.config.n_clusters)
        observed = dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                noise_scale=99.0,
                seed=seed,
                cluster_noise_scales=cluster_scales,
                transform_mode="modify_base",
            )
        )
        gaussian = np.random.RandomState(seed + 1).randn(
            dataset.config.n_points,
            dataset.config.dim,
        )
        expected = (
            base
            + gaussian
            * cluster_scales[
                dataset.base_source_cluster_assignments,
                None,
            ]
        )
        np.testing.assert_allclose(observed, expected)

    def test_modify_base_swaps_disjoint_cross_cluster_rows(self):
        dataset = self.make_dataset(
            n_points=20,
            n_clusters=4,
            balanced=True,
            noise_scale=0.0,
        )
        base = dataset.transform_base(
            PermutedGaussianClustersTransformConfig(noise_scale=0.0, seed=17)
        ).copy()
        observed = dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                noise_scale=0.0,
                seed=29,
                cluster_mixing_probability=1.0,
                transform_mode="modify_base",
            )
        )
        partners = dataset.current_swap_partner_indices
        mask = dataset.current_mixed_point_mask

        self.assertTrue(mask.all())
        np.testing.assert_array_equal(partners[partners], np.arange(len(partners)))
        self.assertTrue(
            np.all(
                dataset.base_source_cluster_assignments[mask]
                != dataset.current_source_cluster_assignments[mask]
            )
        )
        np.testing.assert_array_equal(observed, base[partners])

        repeated = dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                noise_scale=0.0,
                seed=29,
                cluster_mixing_probability=1.0,
                transform_mode="modify_base",
            )
        )
        np.testing.assert_array_equal(repeated, observed)

    def test_modify_base_leaves_unpairable_selected_rows_unchanged(self):
        dataset = self.make_dataset(
            n_points=30,
            n_clusters=2,
            balanced=False,
            noise_scale=0.0,
        )
        dataset.transform_base(
            PermutedGaussianClustersTransformConfig(noise_scale=0.0, seed=17)
        )
        observed = dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                noise_scale=0.0,
                seed=29,
                cluster_mixing_probability=1.0,
                transform_mode="modify_base",
            )
        )

        expected_swapped = 2 * dataset.cluster_sizes.min()
        self.assertEqual(dataset.current_mixed_point_mask.sum(), expected_swapped)
        unchanged = ~dataset.current_mixed_point_mask
        np.testing.assert_array_equal(
            observed[unchanged],
            dataset.base_data[unchanged],
        )

    def test_modify_base_combines_noise_swaps_and_true_center_translation(self):
        dataset = self.make_dataset(noise_scale=0.0)
        base = dataset.transform_base(
            PermutedGaussianClustersTransformConfig(noise_scale=0.0, seed=17)
        ).copy()
        seed = 19
        scale = 0.2
        observed = dataset.transform_current(
            PermutedGaussianClustersTransformConfig(
                n_permute=4,
                noise_scale=scale,
                seed=seed,
                cluster_mixing_probability=0.5,
                transform_mode="modify_base",
            )
        )

        noisy = base + np.random.RandomState(seed + 1).randn(*base.shape) * scale
        partners = dataset.current_swap_partner_indices
        sources = dataset.current_source_cluster_assignments
        targets = dataset.current_target_cluster_assignments
        expected = noisy[partners]
        expected += (
            dataset.base_cluster_centers[targets]
            - dataset.base_cluster_centers[sources]
        )
        np.testing.assert_allclose(observed, expected)
        np.testing.assert_array_equal(
            targets,
            dataset.last_center_mapping[sources],
        )

    def test_modify_base_mode_validation_and_base_restrictions(self):
        dataset = self.make_dataset(noise_scale=0.0)
        with self.assertRaisesRegex(ValueError, "does not accept"):
            dataset.transform_base(
                PermutedGaussianClustersTransformConfig(transform_mode="modify_base")
            )
        with self.assertRaisesRegex(ValueError, "transform_mode"):
            dataset.transform(
                PermutedGaussianClustersTransformConfig(transform_mode="unknown")
            )
        with self.assertRaisesRegex(ValueError, "center_sampling"):
            dataset.transform_current(
                PermutedGaussianClustersTransformConfig(
                    center_sampling="uniform_grid",
                    transform_mode="modify_base",
                )
            )

    def test_invalid_transform_parameters_raise(self):
        dataset = self.make_dataset()
        invalid_configs = [
            PermutedGaussianClustersTransformConfig(n_permute=-1),
            PermutedGaussianClustersTransformConfig(n_permute=1),
            PermutedGaussianClustersTransformConfig(n_permute=6),
            PermutedGaussianClustersTransformConfig(n_permute=2.5),
            PermutedGaussianClustersTransformConfig(n_permute=2, noise_scale=-0.1),
            PermutedGaussianClustersTransformConfig(center_sampling="unknown"),
            PermutedGaussianClustersTransformConfig(center_sampling=""),
            PermutedGaussianClustersTransformConfig(noise_scale=np.nan),
            PermutedGaussianClustersTransformConfig(noise_scale=None),
            PermutedGaussianClustersTransformConfig(noise_distribution="laplace"),
            PermutedGaussianClustersTransformConfig(
                noise_distribution="student_t",
                student_t_df=2.0,
            ),
            PermutedGaussianClustersTransformConfig(
                noise_distribution="student_t",
                student_t_df=None,
            ),
            PermutedGaussianClustersTransformConfig(
                cluster_noise_scales=[0.1, 0.2],
            ),
            PermutedGaussianClustersTransformConfig(
                cluster_noise_scales=[0.1, 0.2, 0.3, 0.4, -0.5],
            ),
            PermutedGaussianClustersTransformConfig(
                cluster_noise_scales=[0.1, 0.2, 0.3, 0.4, np.inf],
            ),
            PermutedGaussianClustersTransformConfig(
                cluster_mixing_probability=-0.1,
            ),
            PermutedGaussianClustersTransformConfig(
                cluster_mixing_probability=1.1,
            ),
            PermutedGaussianClustersTransformConfig(
                cluster_mixing_probability=np.nan,
            ),
        ]

        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                dataset.transform(config)

    def test_invalid_dataset_parameters_raise(self):
        with self.assertRaises(ValueError):
            self.make_dataset(n_clusters=0)
        with self.assertRaises(ValueError):
            self.make_dataset(noise_scale=-0.1)
        with self.assertRaises(ValueError):
            self.make_dataset(noise_scale=np.nan)
        with self.assertRaises(ValueError):
            self.make_dataset(n_points=14, n_clusters=5, balanced=False)
        with self.assertRaises(ValueError):
            self.make_dataset(center_sampling="unknown")
        with self.assertRaises(ValueError):
            self.make_dataset(dim=3, center_sampling="uniform_grid")
        with self.assertRaises(ValueError):
            self.make_dataset(dim=3, center_sampling="nonuniform_grid")

        three_dimensional = self.make_dataset(
            dim=3,
            center_sampling="hypersphere",
        )
        with self.assertRaises(ValueError):
            three_dimensional.transform(
                PermutedGaussianClustersTransformConfig(center_sampling="uniform_grid")
            )

        single_cluster = self.make_dataset(
            n_points=5,
            n_clusters=1,
            balanced=True,
        )
        with self.assertRaises(ValueError):
            single_cluster.transform_current(
                PermutedGaussianClustersTransformConfig(
                    cluster_mixing_probability=0.1,
                )
            )

    def test_factory_transform_config_and_sweep_values(self):
        sim_params = {
            "dataset_type": "permuted_gaussian_clusters",
            "n_points": 30,
            "dim": 2,
            "seed": 3,
            "n_clusters": 4,
            "balanced": True,
            "center_sampling": "uniform_grid",
            "base_transform_params": {"n_clusters": 9, "noise_scale": 0.15},
        }
        dataset = init_dataset(sim_params)
        transform = get_transform_config(
            {
                "n_permute": 3,
                "noise_scale": 0.2,
                "seed": 11,
                "center_sampling": "nonuniform_grid",
                "noise_distribution": "student_t",
                "student_t_df": 5.0,
                "cluster_noise_scales": [0.1, 0.2, 0.3, 0.4],
                "cluster_mixing_probability": 0.25,
                "transform_mode": "modify_base",
            },
            dataset,
        )

        self.assertIsInstance(dataset, PermutedGaussianClustersDataset)
        self.assertEqual(dataset.config.n_clusters, 4)
        self.assertTrue(dataset.config.balanced)
        self.assertEqual(dataset.config.center_sampling, "uniform_grid")
        self.assertIsInstance(transform, PermutedGaussianClustersTransformConfig)
        self.assertEqual(transform.center_sampling, "nonuniform_grid")
        self.assertEqual(transform.noise_distribution, "student_t")
        self.assertEqual(transform.student_t_df, 5.0)
        self.assertEqual(transform.cluster_noise_scales, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(transform.cluster_mixing_probability, 0.25)
        self.assertEqual(transform.transform_mode, "modify_base")
        self.assertFalse(hasattr(transform, "n_clusters"))
        with self.assertRaisesRegex(ValueError, "transform_mode"):
            get_transform_config({"transform_mode": "unknown"}, dataset)
        np.testing.assert_array_equal(
            get_data_param_sweep_values(dataset, "n_permute"),
            np.array([0, 2, 3, 4]),
        )
        np.testing.assert_allclose(
            get_data_param_sweep_values(
                dataset,
                "cluster_mixing_probability",
                sweep_len=5,
            ),
            np.linspace(0.0, 0.5, 5),
        )

    def test_cluster_mixing_runs_through_calibrated_metric_sweep(self):
        dataset = self.make_dataset(n_points=30, n_clusters=3)
        base_config = PermutedGaussianClustersTransformConfig(
            n_permute=0,
            noise_scale=0.1,
            seed=20,
        )
        dataset.transform_base(base_config)

        result = run_param_sweep(
            metric_name="rbf_rwka_symmetric",
            dataset=dataset,
            base_transform_config=base_config,
            transform_config=PermutedGaussianClustersTransformConfig(
                n_permute=2,
                noise_scale=0.1,
                seed=21,
            ),
            data_param_name="cluster_mixing_probability",
            data_param_values=np.array([0.0, 0.5, 1.0]),
            metric_sweep_len=2,
            metric_sweep_values=[0.2, 0.5],
            permutation_test=True,
            n_permutations=2,
            permutation_seed=5,
        )

        _, data_values, metric_values, scores, _, calibration = result
        np.testing.assert_allclose(data_values, np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(metric_values, np.array([0.2, 0.5]))
        self.assertEqual(scores.shape, (3, 2))
        self.assertEqual(calibration.null_scores.shape, (3, 2, 2))
        self.assertTrue(np.isfinite(scores).all())

    def test_runs_through_calibrated_metric_sweep(self):
        dataset = self.make_dataset(n_points=30, n_clusters=3)
        base_config = PermutedGaussianClustersTransformConfig(
            n_permute=0,
            noise_scale=0.1,
            seed=20,
        )
        dataset.transform_base(base_config)

        result = run_param_sweep(
            metric_name="rbf_rwka_symmetric",
            dataset=dataset,
            base_transform_config=base_config,
            transform_config=PermutedGaussianClustersTransformConfig(
                n_permute=2,
                noise_scale=0.1,
                seed=21,
            ),
            data_param_name="n_permute",
            data_param_values=np.array([0, 2, 3]),
            metric_sweep_len=2,
            metric_sweep_values=[0.2, 0.5],
            permutation_test=True,
            n_permutations=2,
            permutation_seed=5,
        )

        _, data_values, metric_values, scores, _, calibration = result
        np.testing.assert_array_equal(data_values, np.array([0, 2, 3]))
        np.testing.assert_allclose(metric_values, np.array([0.2, 0.5]))
        self.assertEqual(scores.shape, (3, 2))
        self.assertEqual(calibration.null_scores.shape, (3, 2, 2))
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
