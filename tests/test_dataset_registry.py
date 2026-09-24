import unittest

import numpy as np

from synthetic_datasets import get_data_param_sweep_values
from synthetic_datasets import get_dataset_spec
from synthetic_datasets import get_dataset_spec_for_instance
from synthetic_datasets import get_dataset_specs
from synthetic_datasets import get_transform_config
from synthetic_datasets import init_dataset
from synthetic_datasets.ellipsoid import EllipsoidConfig
from synthetic_datasets.ellipsoid import EllipsoidDataset
from synthetic_datasets.ellipsoid import EllipsoidTransformConfig
from synthetic_datasets.gaussian_cluster import GaussianClusterDataset
from synthetic_datasets.permuted_gaussian_clusters import (
    PermutedGaussianClustersDataset,
)
from synthetic_datasets.rings_dataset import RingsDataset
from synthetic_datasets.s_clusters import SClustersDataset
from synthetic_datasets.s_curve import SCurveDataset


class DatasetRegistryTests(unittest.TestCase):
    DATASET_CASES = {
        "gaussian_cluster": (
            GaussianClusterDataset,
            {"n_points": 24, "dim": 3, "seed": 13},
            {"t": 0.5, "noise_scale": 0.1, "n_clusters": 4},
        ),
        "permuted_gaussian_clusters": (
            PermutedGaussianClustersDataset,
            {
                "n_points": 24,
                "dim": 2,
                "seed": 13,
                "n_clusters": 4,
                "balanced": True,
                "center_sampling": "uniform_grid",
            },
            {"n_permute": 2},
        ),
        "rings": (
            RingsDataset,
            {"n_points": 24, "dim": 3, "seed": 13},
            {"n_rings": 3, "t": 0.5},
        ),
        "s_curve": (
            SCurveDataset,
            {"n_points": 24, "dim": 3, "seed": 13},
            {"t": 0.5, "noise_scale": 0.1},
        ),
        "s_clusters": (
            SClustersDataset,
            {"n_points": 24, "dim": 3, "seed": 13, "n_clusters": 4},
            {"t": 0.5, "noise_scale": 0.1},
        ),
        "ellipsoid": (
            EllipsoidDataset,
            {"n_points": 24, "dim": 3, "seed": 13},
            {"t": 0.5},
        ),
    }

    def test_all_dataset_families_are_registered_once(self):
        specs = get_dataset_specs()

        self.assertEqual(set(specs), set(self.DATASET_CASES))
        for name, (dataset_type, _, _) in self.DATASET_CASES.items():
            with self.subTest(dataset=name):
                spec = get_dataset_spec(name)
                self.assertIs(spec.dataset_type, dataset_type)
                self.assertEqual(spec.name, name)

        with self.assertRaises(TypeError):
            specs["another_dataset"] = specs["ellipsoid"]

    def test_all_dataset_families_construct_and_transform_through_registry(self):
        for name, (
            dataset_type,
            config_values,
            transform_values,
        ) in self.DATASET_CASES.items():
            with self.subTest(dataset=name):
                dataset = init_dataset({"dataset_type": name, **config_values})
                transform = get_transform_config(transform_values, dataset)
                result = dataset.transform_current(transform)

                self.assertIsInstance(dataset, dataset_type)
                self.assertIs(
                    get_dataset_spec_for_instance(dataset), get_dataset_spec(name)
                )
                self.assertIsInstance(
                    transform, get_dataset_spec(name).transform_config_type
                )
                self.assertEqual(result.shape, dataset.init_data.shape)
                self.assertTrue(np.isfinite(result).all())

    def test_config_and_transform_builders_use_dataclass_defaults(self):
        dataset = init_dataset(
            {
                "dataset_type": "s_clusters",
                "n_points": 24,
                "dim": 2,
            }
        )
        transform = get_transform_config({}, dataset)

        self.assertEqual(dataset.config.n_clusters, 10)
        self.assertEqual(dataset.config.sigma, 0.05)
        self.assertEqual(transform.t, 1.0)
        self.assertEqual(transform.noise_scale, 0.1)
        self.assertEqual(transform.seed, dataset.config.seed)

        ellipsoid = init_dataset(
            {
                "dataset_type": "ellipsoid",
                "n_points": 12,
                "dim": 3,
                "axis_scale": 0.4,
            }
        )
        ellipsoid_transform = get_transform_config({}, ellipsoid)
        self.assertIsNone(ellipsoid_transform.axis_scale)
        np.testing.assert_allclose(
            ellipsoid.transform_current(ellipsoid_transform),
            ellipsoid.transform(EllipsoidTransformConfig()),
        )

    def test_sweeps_are_owned_by_dataset_family(self):
        gaussian = init_dataset(
            {
                "dataset_type": "gaussian_cluster",
                "n_points": 20,
                "dim": 4,
            }
        )
        permuted = init_dataset(
            {
                "dataset_type": "permuted_gaussian_clusters",
                "n_points": 20,
                "dim": 2,
                "n_clusters": 4,
                "balanced": True,
            }
        )

        np.testing.assert_array_equal(
            get_data_param_sweep_values(gaussian, "dim", 5),
            np.asarray([2, 3, 4]),
        )
        np.testing.assert_allclose(
            get_data_param_sweep_values(
                permuted,
                "cluster_mixing_probability",
                5,
            ),
            np.linspace(0.0, 0.5, 5),
        )
        np.testing.assert_array_equal(
            get_data_param_sweep_values(permuted, "n_permute", 2),
            np.asarray([0, 2, 3, 4]),
        )
        with self.assertRaisesRegex(ValueError, "does not support"):
            get_data_param_sweep_values(gaussian, "n_rings")

    def test_invalid_factory_inputs_are_rejected_at_registry_boundary(self):
        with self.assertRaisesRegex(ValueError, "dataset_type"):
            init_dataset({"n_points": 10, "dim": 2})
        with self.assertRaisesRegex(ValueError, "Unsupported dataset"):
            init_dataset({"dataset_type": "missing", "n_points": 10, "dim": 2})
        with self.assertRaisesRegex(ValueError, "requires configuration fields"):
            init_dataset({"dataset_type": "rings", "n_points": 10})


if __name__ == "__main__":
    unittest.main()
