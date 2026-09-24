import unittest

import numpy as np

from synthetic_datasets import get_transform_config
from synthetic_datasets.ellipsoid import EllipsoidConfig
from synthetic_datasets.ellipsoid import EllipsoidDataset
from synthetic_datasets.ellipsoid import EllipsoidTransformConfig
from synthetic_datasets.gaussian_cluster import GaussianClusterConfig
from synthetic_datasets.gaussian_cluster import GaussianClusterDataset
from synthetic_datasets.gaussian_cluster import GaussianClusterTransformConfig
from synthetic_datasets.permuted_gaussian_clusters import (
    PermutedGaussianClustersConfig,
)
from synthetic_datasets.permuted_gaussian_clusters import (
    PermutedGaussianClustersDataset,
)
from synthetic_datasets.permuted_gaussian_clusters import (
    PermutedGaussianClustersTransformConfig,
)
from synthetic_datasets.rings_dataset import RingsConfig
from synthetic_datasets.rings_dataset import RingsDataset
from synthetic_datasets.rings_dataset import RingsTransformConfig
from synthetic_datasets.s_clusters import SClustersConfig
from synthetic_datasets.s_clusters import SClustersDataset
from synthetic_datasets.s_clusters import SClustersTransformConfig
from synthetic_datasets.s_curve import SCurveConfig
from synthetic_datasets.s_curve import SCurveDataset
from synthetic_datasets.s_curve import SCurveTransformConfig


class DatasetLifecycleTests(unittest.TestCase):
    def test_dataset_and_transform_configs_are_owned_copies(self):
        centers = np.arange(8, dtype=float).reshape(4, 2)
        config = GaussianClusterConfig(
            n_points=20,
            dim=2,
            seed=3,
            cluster_centers=centers,
        )
        dataset = GaussianClusterDataset(config)
        config.seed = 99
        centers[:] = -1

        self.assertEqual(dataset.config.seed, 3)
        np.testing.assert_array_equal(
            dataset.config.cluster_centers,
            np.arange(8, dtype=float).reshape(4, 2),
        )

        transform = PermutedGaussianClustersTransformConfig(
            cluster_noise_scales=[0.1, 0.2, 0.3]
        )
        transform_copy = transform.copy()
        transform_copy.cluster_noise_scales[0] = 9.0
        self.assertEqual(transform.cluster_noise_scales[0], 0.1)

    def test_forced_dimensions_do_not_mutate_caller_configs(self):
        s_clusters_config = SClustersConfig(
            n_points=20,
            dim=7,
            seed=3,
            n_clusters=4,
        )
        s_curve_config = SCurveConfig(n_points=20, dim=7, seed=3)

        s_clusters = SClustersDataset(s_clusters_config)
        s_curve = SCurveDataset(s_curve_config)

        self.assertEqual(s_clusters_config.dim, 7)
        self.assertEqual(s_curve_config.dim, 7)
        self.assertEqual(s_clusters.config.dim, 2)
        self.assertEqual(s_curve.config.dim, 2)

    def test_transforms_are_independent_of_invocation_order(self):
        cases = [
            (
                GaussianClusterDataset(
                    GaussianClusterConfig(n_points=24, dim=2, seed=3)
                ),
                GaussianClusterTransformConfig(
                    t=0.4,
                    noise_scale=0.2,
                    n_clusters=4,
                    seed=11,
                    clustering_type="random",
                ),
                GaussianClusterTransformConfig(
                    t=0.8,
                    noise_scale=0.1,
                    n_clusters=5,
                    seed=12,
                    clustering_type="random",
                ),
            ),
            (
                RingsDataset(RingsConfig(n_points=24, dim=3, seed=3)),
                RingsTransformConfig(
                    n_rings=3, t=0.4, clustering_type="random", seed=11
                ),
                RingsTransformConfig(
                    n_rings=4, t=0.8, clustering_type="random", seed=12
                ),
            ),
            (
                SClustersDataset(
                    SClustersConfig(n_points=24, dim=2, seed=3, n_clusters=4)
                ),
                SClustersTransformConfig(t=0.4, noise_scale=0.2, seed=11),
                SClustersTransformConfig(t=0.8, noise_scale=0.1, seed=12),
            ),
            (
                SCurveDataset(SCurveConfig(n_points=24, dim=2, seed=3)),
                SCurveTransformConfig(t=0.4, noise_scale=0.2, seed=11),
                SCurveTransformConfig(t=0.8, noise_scale=0.1, seed=12),
            ),
            (
                EllipsoidDataset(EllipsoidConfig(n_points=24, dim=3, seed=3)),
                EllipsoidTransformConfig(t=0.4, axis_scale=0.5),
                EllipsoidTransformConfig(t=0.8, axis_scale=0.7),
            ),
            (
                PermutedGaussianClustersDataset(
                    PermutedGaussianClustersConfig(
                        n_points=24,
                        dim=2,
                        seed=3,
                        n_clusters=4,
                        balanced=True,
                    )
                ),
                PermutedGaussianClustersTransformConfig(
                    n_permute=2, noise_scale=0.2, seed=11
                ),
                PermutedGaussianClustersTransformConfig(
                    n_permute=4, noise_scale=0.1, seed=12
                ),
            ),
        ]

        for dataset, primary, intervening in cases:
            with self.subTest(dataset=type(dataset).__name__):
                expected = dataset.transform_current(primary)
                dataset.transform_base(intervening)
                dataset.transform_current(intervening)
                observed = dataset.transform_current(primary)
                np.testing.assert_allclose(observed, expected)

    def test_declared_transform_seeds_control_randomness(self):
        cases = [
            (
                GaussianClusterDataset(
                    GaussianClusterConfig(n_points=30, dim=2, seed=3)
                ),
                GaussianClusterTransformConfig(
                    t=0.5,
                    noise_scale=0.2,
                    n_clusters=4,
                    clustering_type="random",
                    seed=11,
                ),
                GaussianClusterTransformConfig(
                    t=0.5,
                    noise_scale=0.2,
                    n_clusters=4,
                    clustering_type="random",
                    seed=12,
                ),
            ),
            (
                RingsDataset(RingsConfig(n_points=30, dim=3, seed=3)),
                RingsTransformConfig(
                    n_rings=4, t=0.5, clustering_type="random", seed=11
                ),
                RingsTransformConfig(
                    n_rings=4, t=0.5, clustering_type="random", seed=12
                ),
            ),
            (
                SClustersDataset(
                    SClustersConfig(n_points=30, dim=2, seed=3, n_clusters=5)
                ),
                SClustersTransformConfig(t=0.5, noise_scale=0.2, seed=11),
                SClustersTransformConfig(t=0.5, noise_scale=0.2, seed=12),
            ),
            (
                SCurveDataset(SCurveConfig(n_points=30, dim=2, seed=3)),
                SCurveTransformConfig(t=0.5, noise_scale=0.2, seed=11),
                SCurveTransformConfig(t=0.5, noise_scale=0.2, seed=12),
            ),
        ]

        for dataset, first_config, second_config in cases:
            with self.subTest(dataset=type(dataset).__name__):
                first = dataset.transform_current(first_config)
                repeated = dataset.transform_current(first_config)
                second = dataset.transform_current(second_config)
                np.testing.assert_allclose(first, repeated)
                self.assertFalse(np.allclose(first, second))

    def test_base_current_and_returned_arrays_have_separate_ownership(self):
        dataset = SClustersDataset(
            SClustersConfig(n_points=24, dim=2, seed=3, n_clusters=4)
        )
        current_config = SClustersTransformConfig(t=0.4, noise_scale=0.2, seed=11)
        base_config = SClustersTransformConfig(t=0.8, noise_scale=0.1, seed=12)

        returned_current = dataset.transform_current(current_config)
        current_snapshot = dataset.get_current_numpy()
        returned_current[:] = 99
        np.testing.assert_allclose(dataset.get_current_numpy(), current_snapshot)

        returned_base = dataset.transform_base(base_config)
        np.testing.assert_allclose(dataset.get_current_numpy(), current_snapshot)
        base_snapshot = dataset.get_base_numpy()
        returned_base[:] = 99
        np.testing.assert_allclose(dataset.get_base_numpy(), base_snapshot)

        exposed = dataset.get_base_numpy()
        exposed[:] = -99
        np.testing.assert_allclose(dataset.get_base_numpy(), base_snapshot)

    def test_dataset_copy_is_independent(self):
        dataset = RingsDataset(RingsConfig(n_points=24, dim=3, seed=3))
        transform = RingsTransformConfig(
            n_rings=4,
            t=0.5,
            clustering_type="random",
            seed=11,
        )
        dataset.transform_current(transform)
        original_current = dataset.get_current_numpy()
        copied = dataset.copy()

        copied.config.seed = 99
        copied.init_data[:] = 0
        copied.current_data[:] = 0
        copied.transform_current(
            RingsTransformConfig(
                n_rings=3,
                t=0.8,
                clustering_type="random",
                seed=12,
            )
        )

        self.assertEqual(dataset.config.seed, 3)
        self.assertFalse(np.allclose(dataset.init_data, 0))
        np.testing.assert_allclose(dataset.get_current_numpy(), original_current)

    def test_transform_factory_inherits_dataset_seed(self):
        datasets_and_params = [
            (
                GaussianClusterDataset(
                    GaussianClusterConfig(n_points=20, dim=2, seed=17)
                ),
                {"t": 0.5, "noise_scale": 0.1, "n_clusters": 4},
            ),
            (
                RingsDataset(RingsConfig(n_points=20, dim=2, seed=17)),
                {"t": 0.5, "n_rings": 3},
            ),
            (
                SClustersDataset(
                    SClustersConfig(n_points=20, dim=2, seed=17, n_clusters=4)
                ),
                {"t": 0.5, "noise_scale": 0.1},
            ),
            (
                SCurveDataset(SCurveConfig(n_points=20, dim=2, seed=17)),
                {"t": 0.5, "noise_scale": 0.1},
            ),
        ]

        for dataset, params in datasets_and_params:
            with self.subTest(dataset=type(dataset).__name__):
                transform = get_transform_config(params, dataset)
                self.assertEqual(transform.seed, 17)


if __name__ == "__main__":
    unittest.main()
