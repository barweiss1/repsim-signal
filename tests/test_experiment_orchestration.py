import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manifold_repsim.config import load_yaml_config
from manifold_repsim.experiments import MetricAggregation
from manifold_repsim.experiments import MetricComparisonSpec
from manifold_repsim.experiments import aggregate_metric_curve
from manifold_repsim.experiments import execution as experiment_execution
from manifold_repsim.experiments import run_metric_comparison
from manifold_repsim.experiments.legacy import run_metric_comparison_across_data_param
from manifold_repsim.experiments.plotting import plot_metric_comparison
from manifold_repsim.experiments.plotting import plot_parameter_sweep
from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.sweeps import ParameterSweepResult
from synthetic_datasets import init_dataset
from synthetic_datasets.gaussian_cluster import GaussianClusterTransformConfig
from tests.test_sweep_utilities import DummyDataset
from tests.test_sweep_utilities import DummyTransformConfig
from synthetic_datasets.base_dataset import DatasetConfig


class ExperimentOrchestrationTests(unittest.TestCase):
    @staticmethod
    def make_dataset():
        dataset = DummyDataset(DatasetConfig(n_points=24, dim=2, seed=5))
        base_transform = DummyTransformConfig(t=0.0)
        dataset.transform_base(base_transform)
        return dataset, base_transform

    def test_computation_package_does_not_import_matplotlib(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import manifold_repsim.experiments; "
                "assert 'matplotlib' not in sys.modules",
            ],
            cwd=Path(__file__).parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_explicit_specs_allow_independent_experiment_grids(self):
        dataset, base_transform = self.make_dataset()
        specifications = (
            MetricComparisonSpec(
                name="narrow_signal",
                metric_name="cka_rbf",
                aggregation=MetricAggregation.AUC,
                metric_grid={"values": [0.1, 0.2, 0.4]},
            ),
            MetricComparisonSpec(
                name="wide_signal",
                metric_name="cka_rbf",
                aggregation=MetricAggregation.AUC,
                metric_grid={"values": [1.0, 2.0, 4.0]},
            ),
            MetricComparisonSpec(
                name="name_does_not_dispatch_auc",
                metric_name="cka",
                aggregation=MetricAggregation.FIXED,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_metric_comparison(
                dataset=dataset,
                base_transform_config=base_transform,
                transform_config=DummyTransformConfig(t=0.0),
                data_parameter_name="t",
                data_parameter_values=[0.2, 0.8],
                specifications=specifications,
                output_dir=directory,
                run_config={"experiment": "explicit-comparison"},
            )
            saved = load_yaml_config(Path(directory) / "config.yaml")

        self.assertEqual(set(result.scores), {spec.name for spec in specifications})
        self.assertEqual(result.score("narrow_signal").shape, (2,))
        metric_configs = saved["metric_comparison"]["metrics"]
        self.assertEqual(
            metric_configs[0]["metric_grid"]["values"],
            [0.1, 0.2, 0.4],
        )
        self.assertEqual(
            metric_configs[1]["metric_grid"]["values"],
            [1.0, 2.0, 4.0],
        )
        self.assertEqual(metric_configs[2]["aggregation"], "fixed")

    def test_aggregate_specs_reuse_matching_computed_signals(self):
        dataset, base_transform = self.make_dataset()
        grid = {"values": [0.1, 0.5, 1.0]}
        specifications = [
            MetricComparisonSpec(
                name="plain_auc",
                metric_name="cka_rbf",
                aggregation=MetricAggregation.AUC,
                metric_grid=grid,
            ),
            MetricComparisonSpec(
                name="weighted_auc",
                metric_name="cka_rbf",
                aggregation=MetricAggregation.VARIANCE_WEIGHTED_AUC,
                metric_grid=grid,
            ),
        ]
        with mock.patch.object(
            experiment_execution,
            "prepare_metric_curve",
            wraps=experiment_execution.prepare_metric_curve,
        ) as prepare:
            run_metric_comparison(
                dataset=dataset,
                base_transform_config=base_transform,
                transform_config=DummyTransformConfig(t=0.0),
                data_parameter_name="t",
                data_parameter_values=[0.2, 0.8],
                specifications=specifications,
            )
        self.assertEqual(prepare.call_count, 2)

    def test_explicit_aggregations_match_fixed_regression_scores(self):
        # These fixed values were the pre-Step-9 `metrics_utils.calc_*_metric`
        # aggregate heuristics' output for this exact dataset/grid, captured
        # before those functions were deleted in Step 17. They are ground
        # truth for the aggregation dispatch below, not a live parity check.
        expected = {
            "auc": 0.9799023178917607,
            "cutoff_auc": 0.9799023178917607,
            "convex_auc": 0.7318282501279472,
            "min_to_max_auc": 0.9799023178917607,
            "minimum_envelope": 0.9799023178917607,
            "minimum_point": 2.0,
            "minimum": 0.9691658388827067,
            "min_to_cutoff": 0.9799023178917607,
        }
        dataset, base_transform = self.make_dataset()
        transform = DummyTransformConfig(t=0.6)
        dataset.transform_current(transform)
        aggregations = {
            "auc": MetricAggregation.AUC,
            "cutoff_auc": MetricAggregation.CUTOFF_AUC,
            "convex_auc": MetricAggregation.CONVEX_AUC,
            "min_to_max_auc": MetricAggregation.MIN_TO_MAX_AUC,
            "minimum_envelope": MetricAggregation.MINIMUM_ENVELOPE,
            "minimum_point": MetricAggregation.MINIMUM_POINT,
            "minimum": MetricAggregation.MINIMUM,
            "min_to_cutoff": MetricAggregation.MIN_TO_CUTOFF,
        }
        specifications = [
            MetricComparisonSpec(
                name=name,
                metric_name="cka_rbf",
                aggregation=aggregation,
                metric_grid={
                    "min": 0.1,
                    "max": 2.0,
                    "num": 9,
                    "scale": "linear",
                },
            )
            for name, aggregation in aggregations.items()
        ]
        result = run_metric_comparison(
            dataset=dataset,
            base_transform_config=base_transform,
            transform_config=DummyTransformConfig(t=0.0),
            data_parameter_name="t",
            data_parameter_values=[0.6],
            specifications=specifications,
            metric_sweep_len=9,
        )
        for name, expected_score in expected.items():
            with self.subTest(aggregation=name):
                self.assertAlmostEqual(result.scores[name][0], expected_score)

    def test_effective_config_is_written_before_computation_failure(self):
        dataset, base_transform = self.make_dataset()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no parameter"):
                run_metric_comparison(
                    dataset=dataset,
                    base_transform_config=base_transform,
                    transform_config=DummyTransformConfig(t=0.0),
                    data_parameter_name="missing",
                    data_parameter_values=[0.2],
                    specifications=[
                        MetricComparisonSpec(name="linear", metric_name="cka")
                    ],
                    output_dir=directory,
                )
            saved = load_yaml_config(Path(directory) / "config.yaml")
        self.assertEqual(saved["metric_comparison"]["data_parameter"], "missing")

    def test_aggregate_behavior_is_selected_by_spec_not_output_name(self):
        parameter_values = np.linspace(0.1, 1.0, 9)
        scores = np.asarray([0.9, 0.8, 0.7, 0.3, 0.7, 0.8, 0.9, 0.8, 0.7])
        auc = aggregate_metric_curve(
            MetricAggregation.AUC,
            parameter_values,
            scores,
            metric_name="rbf_rwka",
            integration_method="average",
        )
        minimum = aggregate_metric_curve(
            MetricAggregation.MINIMUM,
            parameter_values,
            scores,
            metric_name="rbf_rwka",
        )
        minimum_point = aggregate_metric_curve(
            MetricAggregation.MINIMUM_POINT,
            parameter_values,
            scores,
            metric_name="rbf_rwka",
        )
        self.assertAlmostEqual(auc, np.mean(scores))
        self.assertAlmostEqual(minimum, 0.3)
        self.assertAlmostEqual(minimum_point, parameter_values[3])

    def test_plots_consume_results_without_running_metrics(self):
        parameter_result = ParameterSweepResult(
            metric_name="cka_rbf",
            metric_parameter_name="rbf_sigma",
            data_parameter_name="t",
            data_parameter_values=np.asarray([0.2, 0.8]),
            metric_parameter_values=np.asarray([0.1, 1.0]),
            score_grid=np.asarray([[0.5, 0.6], [0.4, 0.7]]),
            infinity_scores=np.asarray([0.8, 0.9]),
        )
        dataset, base_transform = self.make_dataset()
        comparison_result = run_metric_comparison(
            dataset=dataset,
            base_transform_config=base_transform,
            transform_config=DummyTransformConfig(t=0.0),
            data_parameter_name="t",
            data_parameter_values=[0.2],
            specifications=[MetricComparisonSpec(name="linear", metric_name="cka")],
        )
        with mock.patch.object(
            AlignmentMetrics,
            "measure",
            side_effect=AssertionError("plot attempted metric execution"),
        ):
            parameter_figure, _ = plot_parameter_sweep(parameter_result)
            comparison_figure, _ = plot_metric_comparison(comparison_result)
        plt.close(parameter_figure)
        plt.close(comparison_figure)

    def test_legacy_notebook_comparison_delegates_and_saves_outputs(self):
        dataset = init_dataset(
            {
                "dataset_type": "gaussian_cluster",
                "n_points": 24,
                "dim": 2,
                "seed": 3,
            }
        )
        base_transform = GaussianClusterTransformConfig(
            t=0.0,
            noise_scale=0.1,
            n_clusters=4,
            seed=3,
        )
        dataset.transform_base(base_transform)
        base_params = {
            "t": 0.0,
            "noise_scale": 0.1,
            "n_clusters": 4,
            "seed": 3,
        }
        simulation = {
            "data_param_sweep_len": 2,
            "metric_param_sweep_len": 3,
            "base_transform_params": base_params,
            "auc_integration_method": "average",
            "cutoff_threshold": 0.05,
            "logscale": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_metric_comparison_across_data_param(
                dataset=dataset,
                current_transform_params=base_params,
                data_param_name="t",
                metric_names=["cka", "cka_rbf_auc"],
                sim_params=simulation,
                figures_path=directory,
                metric_color_map=None,
            )
            self.assertTrue((Path(directory) / "config.yaml").is_file())
            self.assertTrue((Path(directory) / "alignment_scores_t.png").is_file())
        self.assertEqual(set(result.scores), {"cka", "cka_rbf_auc"})

    def test_specification_validation_is_strict(self):
        with self.assertRaisesRegex(ValueError, "fixed metric"):
            MetricComparisonSpec(
                name="invalid",
                metric_name="cka",
                metric_grid={"values": [0.1, 0.2]},
            )
        with self.assertRaisesRegex(ValueError, "Unknown metric comparison fields"):
            MetricComparisonSpec.from_mapping({"metric": "cka", "unexpected": True})


if __name__ == "__main__":
    unittest.main()
