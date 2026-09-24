import tempfile
import unittest
from pathlib import Path

import numpy as np

from manifold_repsim.experiments.legacy import run_param_sweep
from manifold_repsim.config import load_yaml_config
from manifold_repsim.metrics.registry import get_metric_spec
from manifold_repsim.sweeps import ParameterSweepResult
from manifold_repsim.sweeps import SweepGridConfig
from manifold_repsim.sweeps import aggregate_signal_auc
from manifold_repsim.sweeps import compute_permutation_calibration
from manifold_repsim.sweeps import get_convex_regions
from manifold_repsim.sweeps import calc_local_minimas
from manifold_repsim.sweeps import resolve_metric_sweep_grid
from tests.test_sweep_utilities import DummyDataset
from tests.test_sweep_utilities import DummyTransformConfig
from synthetic_datasets.base_dataset import DatasetConfig


class SweepGridTests(unittest.TestCase):
    def test_exact_and_generated_grid_fields_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            SweepGridConfig(
                values=(0.1, 0.2),
                minimum=0.1,
                maximum=1.0,
                num=3,
                scale="linear",
            )

    def test_generated_grid_requires_complete_valid_configuration(self):
        with self.assertRaisesRegex(ValueError, "requires min, max, num, and scale"):
            SweepGridConfig(minimum=0.1, maximum=1.0)
        with self.assertRaisesRegex(ValueError, "positive min"):
            SweepGridConfig(
                minimum=0.0,
                maximum=1.0,
                num=3,
                scale="log",
            )

    def test_exact_and_generated_experiment_grids_are_resolved(self):
        exact = resolve_metric_sweep_grid("cka_rbf", {"values": [0.2, 0.5, 1.5]})
        generated = resolve_metric_sweep_grid(
            "cka_rbf",
            {"min": 0.1, "max": 10.0, "num": 3, "scale": "log"},
        )
        np.testing.assert_allclose(exact.values, [0.2, 0.5, 1.5])
        np.testing.assert_allclose(generated.values, [0.1, 1.0, 10.0])
        self.assertEqual(exact.source, "experiment")
        self.assertEqual(generated.source, "experiment")

    def test_generated_integer_grids_round_instead_of_failing(self):
        """A generated grid over topk lands on fractional points by construction."""
        generated = resolve_metric_sweep_grid(
            "mutual_knn",
            {"min": 3, "max": 500, "num": 30, "scale": "log"},
        )
        self.assertEqual(generated.parameter_name, "topk")
        self.assertEqual(generated.values.dtype.kind, "i")
        self.assertEqual(generated.values[0], 3)
        self.assertEqual(generated.values[-1], 500)
        # Rounding can collide, so the grid may be shorter than the request.
        self.assertLessEqual(len(generated.values), 30)
        self.assertEqual(len(set(generated.values.tolist())), len(generated.values))
        self.assertTrue(np.all(np.diff(generated.values) > 0))

    def test_registry_default_grids_resolve_for_integer_parameters(self):
        for metric_name in ("mutual_knn", "cycle_knn", "cknna"):
            with self.subTest(metric=metric_name):
                resolved = resolve_metric_sweep_grid(metric_name)
                self.assertEqual(resolved.parameter_name, "topk")
                self.assertEqual(resolved.source, "registry")
                self.assertTrue(np.all(resolved.values >= 1))

    def test_exact_integer_grids_still_reject_fractional_values(self):
        with self.assertRaisesRegex(ValueError, "must be integers"):
            resolve_metric_sweep_grid("mutual_knn", {"values": [3, 4.5, 6]})

    def test_two_local_grids_do_not_mutate_registry_defaults(self):
        before = get_metric_spec("cka_rbf").sweep
        first = resolve_metric_sweep_grid("cka_rbf", {"values": [0.1, 0.2]})
        second = resolve_metric_sweep_grid("cka_rbf", {"values": [2.0, 4.0]})
        after = get_metric_spec("cka_rbf").sweep
        np.testing.assert_allclose(first.values, [0.1, 0.2])
        np.testing.assert_allclose(second.values, [2.0, 4.0])
        self.assertIs(before, after)


class ParameterSweepResultTests(unittest.TestCase):
    def test_named_shapes_and_legacy_tuple_compatibility(self):
        result = ParameterSweepResult(
            metric_name="cka_rbf",
            metric_parameter_name="rbf_sigma",
            data_parameter_name="t",
            data_parameter_values=np.array([0.0, 1.0]),
            metric_parameter_values=np.array([0.1, 1.0, 10.0]),
            score_grid=np.zeros((2, 3)),
            infinity_scores=np.ones(2),
        )
        self.assertEqual(result.score_grid.shape, (2, 3))
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], "rbf_sigma")

    def test_invalid_metric_grid_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shared 1D grid"):
            ParameterSweepResult(
                metric_name="cka_rbf",
                metric_parameter_name="rbf_sigma",
                data_parameter_name="t",
                data_parameter_values=np.array([0.0, 1.0]),
                metric_parameter_values=np.zeros((3, 2)),
                score_grid=np.zeros((2, 3)),
                infinity_scores=np.ones(2),
            )


class SweepExecutionTests(unittest.TestCase):
    @staticmethod
    def make_dataset():
        dataset = DummyDataset(DatasetConfig(n_points=24, dim=2, seed=0))
        base_transform = DummyTransformConfig(t=0.0)
        dataset.transform_base(base_transform)
        return dataset, base_transform

    def test_exact_grid_returns_named_result_and_saves_effective_config(self):
        dataset, base_transform = self.make_dataset()
        with tempfile.TemporaryDirectory() as directory:
            result = run_param_sweep(
                metric_name="cka_rbf",
                dataset=dataset,
                base_transform_config=base_transform,
                transform_config=DummyTransformConfig(t=0.0),
                data_param_name="t",
                data_param_values=[0.2, 0.7],
                metric_sweep_grid={"values": [0.2, 0.6, 1.4]},
                output_dir=directory,
                run_config={"experiment": "grid-test"},
            )

            self.assertIsInstance(result, ParameterSweepResult)
            np.testing.assert_allclose(result.metric_parameter_values, [0.2, 0.6, 1.4])
            self.assertEqual(result.score_grid.shape, (2, 3))
            saved = load_yaml_config(Path(directory) / "config.yaml")
            self.assertEqual(saved["experiment"], "grid-test")
            self.assertEqual(
                saved["parameter_sweep"]["metric_grid"]["values"],
                [0.2, 0.6, 1.4],
            )

    def test_calibration_and_signal_aggregation_work_without_plot_helpers(self):
        observed = np.array([[0.4, 0.6], [0.2, 0.8]])
        null = np.stack([observed - 0.2, observed - 0.1], axis=-1)
        calibration = compute_permutation_calibration(observed, null, alpha=0.5)
        auc = aggregate_signal_auc([0.1, 1.0], observed)
        self.assertEqual(calibration.calibrated_scores.shape, observed.shape)
        np.testing.assert_allclose(auc, [0.5, 0.5])
        self.assertIsInstance(
            get_convex_regions(
                param_vec=np.array([0.1, 0.2, 0.3]),
                scores=np.array([0.5, 0.2, 0.5]),
            ),
            list,
        )
        minima_params, _ = calc_local_minimas(
            param_vec=np.arange(10),
            scores=np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1]),
        )
        np.testing.assert_array_equal(minima_params, [4])


if __name__ == "__main__":
    unittest.main()
