import unittest

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.sweeps import (
    aggregate_signal_auc,
    aggregate_signal_variance_weighted_auc,
    compute_aggregate_permutation_calibration,
    compute_permutation_calibration,
    calc_variance_weighted_auc_from_sweep,
    get_param_sweep_for_metric,
    integrate_metric_over_param,
    integrate_metric_over_param_weighted,
    make_permutation_indices,
    map_param_name_to_kwargs,
    sweep_metric_over_param,
)
from manifold_repsim.experiments.legacy import (
    get_metric_sweep_config,
    get_variance_weighted_auc_parent_metric,
    plot_param_sweep,
    run_param_sweep,
)
from synthetic_datasets.base_dataset import BaseDataset, DatasetConfig, TransformConfig


class DummyTransformConfig(TransformConfig):
    def __init__(self, t=0.0):
        self.t = t


class DummyDataset(BaseDataset):
    def transform(self, transform_config: TransformConfig) -> np.ndarray:
        t = transform_config.t
        scale = np.array([1.0 + t, 1.0 + 2.0 * t])
        return self.init_data * scale


class SweepUtilityTests(unittest.TestCase):
    def test_compute_permutation_calibration_right_tail(self):
        observed = np.array([[0.8, 0.2]])
        null = np.array([[[0.1, 0.3, 0.7, 0.9], [0.1, 0.2, 0.3, 0.4]]])

        result = compute_permutation_calibration(observed, null, alpha=0.2)

        # The finite permutation threshold uses the nearest-rank empirical
        # quantile, so ceil((1 - 0.2) * 4) selects the fourth order statistic.
        np.testing.assert_allclose(result.tau_alpha, np.array([[0.9, 0.4]]))
        np.testing.assert_allclose(result.p_values, np.array([[2.0 / 5.0, 4.0 / 5.0]]))
        np.testing.assert_allclose(result.calibrated_scores, np.array([[0.0, 0.0]]))

    def test_compute_permutation_calibration_positive_excess(self):
        observed = np.array([0.9])
        null = np.array([[0.1, 0.2, 0.3, 0.4]])

        result = compute_permutation_calibration(observed, null, alpha=0.4)

        self.assertAlmostEqual(result.tau_alpha[0], 0.3)
        self.assertAlmostEqual(result.calibrated_scores[0], (0.9 - 0.3) / (1.0 - 0.3))

    def test_aggregate_signal_auc_from_existing_grid(self):
        params = np.array([0.0, 1.0, 2.0])
        signal_grid = np.array(
            [
                [0.2, 0.4, 0.8],
                [0.3, 0.3, 0.3],
            ]
        )

        auc_scores = aggregate_signal_auc(
            params,
            signal_grid,
            integration_method="average",
        )

        np.testing.assert_allclose(auc_scores, np.array([np.mean(signal_grid[0]), 0.3]))

    def test_aggregate_signal_variance_weighted_auc_from_existing_grid(self):
        params = np.array([0.0, 1.0, 2.0])
        signal_grid = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 2.0, 4.0],
            ]
        )
        weights = np.var(signal_grid, axis=0)

        scores = aggregate_signal_variance_weighted_auc(
            params,
            signal_grid,
            integration_method="average",
        )

        expected = np.array(
            [np.sum(row * weights) / np.sum(weights) for row in signal_grid]
        )
        np.testing.assert_allclose(scores, expected)

    def test_compute_aggregate_permutation_calibration_auc(self):
        params = np.array([0.0, 1.0, 2.0])
        observed_grid = np.array(
            [
                [0.2, 0.4, 0.8],
                [0.3, 0.3, 0.3],
            ]
        )
        null_grid = np.array(
            [
                [[0.1, 0.5], [0.2, 0.5], [0.3, 0.5]],
                [[0.2, 0.4], [0.2, 0.4], [0.2, 0.4]],
            ]
        )

        result = compute_aggregate_permutation_calibration(
            params,
            observed_grid,
            null_grid,
            aggregate_kind="auc",
            integration_method="average",
            alpha=0.4,
        )

        np.testing.assert_allclose(
            result.null_scores, np.array([[0.2, 0.5], [0.2, 0.4]])
        )
        self.assertEqual(result.calibrated_scores.shape, (2,))
        np.testing.assert_allclose(result.p_values, np.array([2.0 / 3.0, 2.0 / 3.0]))

    def test_compute_aggregate_permutation_calibration_var_auc(self):
        params = np.array([0.0, 1.0, 2.0])
        observed_grid = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 2.0, 4.0],
            ]
        )
        null_grid = np.array(
            [
                [[0.0, 0.0], [100.0, 1.0], [1.0, 2.0]],
                [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]],
            ]
        )
        observed_weights = np.var(observed_grid, axis=0)

        result = compute_aggregate_permutation_calibration(
            params,
            observed_grid,
            null_grid,
            aggregate_kind="variance_weighted_auc",
            integration_method="average",
            alpha=0.4,
        )

        self.assertEqual(result.null_scores.shape, (2, 2))
        expected_perm0 = np.array(
            [
                np.sum(null_grid[0, :, 0] * observed_weights)
                / np.sum(observed_weights),
                np.sum(null_grid[1, :, 0] * observed_weights)
                / np.sum(observed_weights),
            ]
        )
        np.testing.assert_allclose(result.null_scores[:, 0], expected_perm0)
        np.testing.assert_allclose(result.null_scores[:, 1], np.array([1.8, 3.6]))

    def test_aggregate_calibration_differs_from_pointwise_then_aggregate(self):
        params = np.array([0.0, 1.0])
        observed_grid = np.array([[0.9, 0.1]])
        null_grid = np.array([[[0.1, 0.8], [0.1, 0.0]]])

        pointwise = compute_permutation_calibration(
            observed_grid,
            null_grid,
            alpha=0.34,
        )
        pointwise_then_aggregate = aggregate_signal_auc(
            params,
            pointwise.calibrated_scores,
            integration_method="average",
        )
        aggregate = compute_aggregate_permutation_calibration(
            params,
            observed_grid,
            null_grid,
            aggregate_kind="auc",
            integration_method="average",
            alpha=0.34,
        )

        self.assertFalse(
            np.allclose(pointwise_then_aggregate, aggregate.calibrated_scores)
        )

    def test_make_permutation_indices_reproducible(self):
        first = make_permutation_indices(8, 3, seed=123)
        second = make_permutation_indices(8, 3, seed=123)

        self.assertTrue(torch.equal(first, second))

    def test_integrate_metric_over_param_weighted_average(self):
        params = np.array([1.0, 2.0, 3.0])
        scores = np.array([0.2, 0.5, 0.8])
        weights = np.array([0.0, 1.0, 3.0])

        weighted = integrate_metric_over_param_weighted(
            params,
            scores,
            weights,
            integration_method="average",
        )

        self.assertAlmostEqual(weighted, (0.5 + 3.0 * 0.8) / 4.0)

    def test_variance_weighted_auc_from_sweep_uses_column_variance(self):
        params = np.array([0.0, 1.0, 2.0])
        score_grid = np.array(
            [
                [0.2, 0.4, 0.6],
                [0.2, 0.8, 1.0],
                [0.2, 1.2, 1.4],
            ]
        )
        weights = np.var(score_grid, axis=0)

        scores = calc_variance_weighted_auc_from_sweep(
            params,
            score_grid,
            integration_method="average",
        )

        expected = np.array(
            [np.sum(row * weights) / np.sum(weights) for row in score_grid]
        )
        np.testing.assert_allclose(scores, expected)

    def test_variance_weighted_auc_falls_back_when_variance_is_zero(self):
        params = np.array([0.0, 1.0, 2.0])
        score_grid = np.array(
            [
                [0.2, 0.4, 0.6],
                [0.2, 0.4, 0.6],
            ]
        )

        scores = calc_variance_weighted_auc_from_sweep(
            params,
            score_grid,
            integration_method="trapezoidal",
        )

        expected = np.array(
            [
                integrate_metric_over_param(params, row, "trapezoidal")
                for row in score_grid
            ]
        )
        np.testing.assert_allclose(scores, expected)

    def test_diffusion_time_maps_to_kwargs(self):
        self.assertEqual(
            map_param_name_to_kwargs("diffusion_time", 0.25),
            {"diffusion_time": 0.25},
        )

    def test_diffusion_time_sweep_values(self):
        sweep_config = AlignmentMetrics.SWEEP_PARAMS["rbf_rwka_diffusion_time"]
        sweep = get_param_sweep_for_metric(
            metric_name="rbf_rwka_diffusion_time",
            sweep_len=4,
            logscale=True,
        )

        self.assertAlmostEqual(sweep[0], sweep_config["min"])
        self.assertAlmostEqual(sweep[-1], sweep_config["max"])
        self.assertTrue(np.all(np.diff(sweep) > 0))

    def test_diffusion_time_variance_weighted_parent_metric(self):
        self.assertEqual(
            get_variance_weighted_auc_parent_metric("rbf_rwka_diffusion_time_var_auc"),
            "rbf_rwka_diffusion_time",
        )

    def test_experiment_metric_sweep_range_does_not_change_global_default(self):
        global_config = dict(AlignmentMetrics.SWEEP_PARAMS["rbf_rwka_symmetric"])
        local_config = get_metric_sweep_config(
            "rbf_rwka_symmetric",
            {"min": 0.05, "max": 2.0},
        )
        sweep = get_param_sweep_for_metric(
            metric_name="rbf_rwka_symmetric",
            sweep_len=4,
            logscale=True,
            sweep_config=local_config,
        )

        self.assertAlmostEqual(sweep[0], 0.05)
        self.assertAlmostEqual(sweep[-1], 2.0)
        self.assertEqual(
            AlignmentMetrics.SWEEP_PARAMS["rbf_rwka_symmetric"], global_config
        )

    def test_new_rbf_metrics_expose_sigma_sweeps(self):
        for metric_name in ("rbf_uka", "rbf_crwka", "rbf_degree_crwka"):
            with self.subTest(metric=metric_name):
                config = AlignmentMetrics.SWEEP_PARAMS[metric_name]
                sweep = get_param_sweep_for_metric(
                    metric_name=metric_name,
                    sweep_len=4,
                    logscale=True,
                )

                self.assertEqual(config["param"], "rbf_sigma")
                self.assertAlmostEqual(sweep[0], config["min"])
                self.assertAlmostEqual(sweep[-1], config["max"])

    def test_auc_sweep_uses_experiment_metric_range(self):
        feats_A = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64
        )
        feats_B = torch.tensor(
            [[0.0, 0.0], [1.2, 0.1], [0.1, 0.8]], dtype=torch.float64
        )
        sweep_config = get_metric_sweep_config(
            "rbf_rwka_symmetric",
            {"min": 0.05, "max": 2.0},
        )

        params, scores = sweep_metric_over_param(
            feats_A,
            feats_B,
            metric_name="rbf_rwka_symmetric",
            sweep_config=sweep_config,
            sweep_len=4,
            logscale=True,
        )

        self.assertAlmostEqual(params[0], 0.05)
        self.assertAlmostEqual(params[-1], 2.0)
        self.assertTrue(np.isfinite(scores).all())

    def test_run_param_sweep_without_calibration_keeps_five_tuple(self):
        config = DatasetConfig(n_points=20, dim=2, seed=0)
        dataset = DummyDataset(config)
        base_transform_config = DummyTransformConfig(t=0.0)
        dataset.transform_base(base_transform_config)
        transform_config = DummyTransformConfig(t=0.0)

        result = run_param_sweep(
            metric_name="cka_rbf",
            dataset=dataset,
            base_transform_config=base_transform_config,
            transform_config=transform_config,
            data_param_name="t",
            data_param_values=np.array([0.2]),
            metric_sweep_len=3,
            permutation_test=False,
        )

        self.assertEqual(len(result), 5)

    def test_run_param_sweep_metric_kwargs_override_fixed_sigma(self):
        config = DatasetConfig(n_points=20, dim=2, seed=0)
        dataset = DummyDataset(config)
        base_transform_config = DummyTransformConfig(t=0.0)
        dataset.transform_base(base_transform_config)
        transform_config = DummyTransformConfig(t=0.0)

        _, _, _, low_sigma_scores, _ = run_param_sweep(
            metric_name="rbf_rwka_diffusion_time",
            dataset=dataset,
            base_transform_config=base_transform_config,
            transform_config=transform_config,
            data_param_name="t",
            data_param_values=np.array([0.5]),
            metric_sweep_len=3,
            metric_sweep_values=[0.5, 1.0, 2.0],
            metric_kwargs={"rbf_sigma": 0.05},
        )
        _, _, _, high_sigma_scores, _ = run_param_sweep(
            metric_name="rbf_rwka_diffusion_time",
            dataset=dataset,
            base_transform_config=base_transform_config,
            transform_config=transform_config,
            data_param_name="t",
            data_param_values=np.array([0.5]),
            metric_sweep_len=3,
            metric_sweep_values=[0.5, 1.0, 2.0],
            metric_kwargs={"rbf_sigma": 2.0},
        )

        self.assertFalse(np.allclose(low_sigma_scores, high_sigma_scores))

    def test_run_param_sweep_calibration_shapes_and_seed(self):
        config = DatasetConfig(n_points=20, dim=2, seed=0)
        dataset = DummyDataset(config)
        base_transform_config = DummyTransformConfig(t=0.0)
        dataset.transform_base(base_transform_config)
        transform_config = DummyTransformConfig(t=0.0)

        first = run_param_sweep(
            metric_name="cka_rbf",
            dataset=dataset,
            base_transform_config=base_transform_config,
            transform_config=transform_config,
            data_param_name="t",
            data_param_values=np.array([0.2, 0.6]),
            metric_sweep_len=3,
            permutation_test=True,
            n_permutations=4,
            permutation_seed=123,
        )
        second = run_param_sweep(
            metric_name="cka_rbf",
            dataset=dataset,
            base_transform_config=base_transform_config,
            transform_config=transform_config,
            data_param_name="t",
            data_param_values=np.array([0.2, 0.6]),
            metric_sweep_len=3,
            permutation_test=True,
            n_permutations=4,
            permutation_seed=123,
        )

        self.assertEqual(len(first), 6)
        calibration_result = first[-1]
        self.assertEqual(calibration_result.null_scores.shape, (2, 3, 4))
        self.assertEqual(calibration_result.null_mean.shape, (2, 3))
        self.assertEqual(calibration_result.calibrated_scores.shape, (2, 3))
        np.testing.assert_allclose(
            calibration_result.null_scores, second[-1].null_scores
        )

    def test_symmetric_rbf_rwka_runs_through_calibrated_sweep(self):
        config = DatasetConfig(n_points=20, dim=2, seed=0)
        dataset = DummyDataset(config)
        base_transform_config = DummyTransformConfig(t=0.0)
        dataset.transform_base(base_transform_config)

        result = run_param_sweep(
            metric_name="rbf_rwka_symmetric",
            dataset=dataset,
            base_transform_config=base_transform_config,
            transform_config=DummyTransformConfig(t=0.0),
            data_param_name="t",
            data_param_values=np.array([0.2, 0.6]),
            metric_sweep_len=3,
            metric_sweep_range={"min": 0.05, "max": 2.0},
            logscale=True,
            permutation_test=True,
            n_permutations=4,
            permutation_seed=123,
        )

        metric_param_name, _, metric_values, scores, _, calibration = result
        self.assertEqual(metric_param_name, "rbf_sigma")
        self.assertAlmostEqual(metric_values[0], 0.05)
        self.assertAlmostEqual(metric_values[-1], 2.0)
        self.assertEqual(scores.shape, (2, 3))
        self.assertEqual(calibration.null_scores.shape, (2, 3, 4))
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(np.isfinite(calibration.null_scores).all())

    def test_new_rbf_metrics_run_through_calibrated_sweep(self):
        config = DatasetConfig(n_points=20, dim=2, seed=0)
        dataset = DummyDataset(config)
        base_transform_config = DummyTransformConfig(t=0.0)
        dataset.transform_base(base_transform_config)

        for metric_name in ("rbf_uka", "rbf_crwka", "rbf_degree_crwka"):
            with self.subTest(metric=metric_name):
                result = run_param_sweep(
                    metric_name=metric_name,
                    dataset=dataset,
                    base_transform_config=base_transform_config,
                    transform_config=DummyTransformConfig(t=0.0),
                    data_param_name="t",
                    data_param_values=np.array([0.2, 0.6]),
                    metric_sweep_len=3,
                    metric_sweep_range={"min": 0.05, "max": 2.0},
                    logscale=True,
                    permutation_test=True,
                    n_permutations=3,
                    permutation_seed=123,
                )

                metric_param_name, _, metric_values, scores, _, calibration = result
                self.assertEqual(metric_param_name, "rbf_sigma")
                self.assertAlmostEqual(metric_values[0], 0.05)
                self.assertAlmostEqual(metric_values[-1], 2.0)
                self.assertEqual(scores.shape, (2, 3))
                self.assertEqual(calibration.null_scores.shape, (2, 3, 3))
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(np.isfinite(calibration.null_scores).all())

    def test_plot_param_sweep_heatmap_rejects_per_row_grids(self):
        metric_param_values = np.array([[0.1, 0.2], [0.2, 0.3]])
        score_grid = np.array([[0.2, 0.3], [0.4, 0.5]])
        sim_params = {"cutoff_threshold": 0.05}

        with self.assertRaises(ValueError):
            plot_param_sweep(
                metric_name="cka_rbf",
                plot_kind="heatmap",
                metric_param_name="rbf_sigma",
                metric_param_values=metric_param_values,
                data_param_name="t",
                data_param_values=np.array([0.1, 0.2]),
                score_grid=score_grid,
                inf_scores=None,
                sim_params=sim_params,
                logscale=False,
            )

    def test_plot_param_sweep_line_with_calibration_overlay(self):
        score_grid = np.array([[0.7, 0.8, 0.9]])
        null_scores = np.array([[[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]])
        calibration_result = compute_permutation_calibration(score_grid, null_scores)

        fig, ax = plot_param_sweep(
            metric_name="cka_rbf",
            plot_kind="line",
            metric_param_name="rbf_sigma",
            metric_param_values=np.array([0.1, 0.2, 0.3]),
            data_param_name="t",
            data_param_values=np.array([0.1]),
            score_grid=score_grid,
            inf_scores=None,
            sim_params={"cutoff_threshold": 0.05},
            logscale=False,
            calibration_result=calibration_result,
        )

        self.assertGreaterEqual(len(ax.lines), 2)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
