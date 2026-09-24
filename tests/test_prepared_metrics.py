import unittest
from unittest import mock

import torch

from manifold_repsim.metrics.prepared import _score_rbf_curve_value
from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.metrics import METRIC_SPECS
from manifold_repsim.metrics import prepare_metric_curve
from manifold_repsim.metrics import prepare_metric_pair
from manifold_repsim.metrics import score_prepared_curve
from manifold_repsim.metrics import score_prepared_metric
from manifold_repsim.metrics import supports_prepared_curve
from manifold_repsim.metrics import supports_prepared_metric


class PreparedMetricTests(unittest.TestCase):
    METRIC_CASES = {
        "cycle_knn": {"topk": 2},
        "mutual_knn": {"topk": 2},
        "mutual_knn_dist": {"topk": 2},
        "cka": {},
        "cka_rbf": {"rbf_sigma": 0.6},
        "cka_rbf_quantile": {"quantile": 0.35},
        "rbf_uka": {"rbf_sigma": 0.6},
        "softmax_rwka": {"temperature": 0.4},
        "rbf_rwka": {"rbf_sigma": 0.6},
        "rbf_rwka_symmetric": {"rbf_sigma": 0.6},
        "rbf_crwka": {"rbf_sigma": 0.6},
        "rbf_degree_crwka": {"rbf_sigma": 0.6},
        "rbf_rwka_quantile": {"quantile": 0.35},
        "rbf_rwka_diffusion_time": {
            "rbf_sigma": 0.4,
            "diffusion_time": 0.7,
        },
        "cknna": {"topk": 3},
    }
    CURVE_CASES = {
        "cycle_knn": ("topk", [1, 2, 3], {}),
        "mutual_knn": ("topk", [1, 2, 3], {}),
        "mutual_knn_dist": ("topk", [1, 2, 3], {}),
        "cknna": ("topk", [2, 3, 4], {}),
        "cka_rbf": ("rbf_sigma", [0.2, 0.6, 1.1], {}),
        "cka_rbf_quantile": ("quantile", [0.2, 0.4, 0.6], {}),
        "rbf_uka": ("rbf_sigma", [0.2, 0.6, 1.1], {}),
        "softmax_rwka": ("temperature", [0.2, 0.6, 1.1], {}),
        "rbf_rwka": ("rbf_sigma", [0.2, 0.6, 1.1], {}),
        "rbf_rwka_symmetric": ("rbf_sigma", [0.2, 0.6, 1.1], {}),
        "rbf_crwka": ("rbf_sigma", [0.2, 0.6, 1.1], {}),
        "rbf_degree_crwka": ("rbf_sigma", [0.2, 0.6, 1.1], {}),
        "rbf_rwka_quantile": ("quantile", [0.2, 0.4, 0.6], {}),
        "rbf_rwka_diffusion_time": (
            "diffusion_time",
            [0.3, 0.8, 1.4],
            {"rbf_sigma": 0.4},
        ),
    }

    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.feats_A = torch.randn(8, 5, generator=generator, dtype=torch.float64)
        self.feats_B = torch.randn(8, 3, generator=generator, dtype=torch.float64)

    def test_every_prepared_metric_matches_direct_scoring(self):
        for metric_name, kwargs in self.METRIC_CASES.items():
            with self.subTest(metric_name=metric_name):
                prepared = prepare_metric_pair(
                    metric_name,
                    self.feats_A,
                    self.feats_B,
                    **kwargs,
                )
                expected = AlignmentMetrics.measure(
                    metric_name,
                    self.feats_A,
                    self.feats_B,
                    **kwargs,
                )
                actual = score_prepared_metric(prepared)
                self.assertAlmostEqual(actual, expected, places=10)

    def test_every_prepared_metric_matches_direct_target_permutation(self):
        permutation = torch.tensor([5, 1, 7, 0, 3, 6, 2, 4])
        for metric_name, kwargs in self.METRIC_CASES.items():
            with self.subTest(metric_name=metric_name):
                prepared = prepare_metric_pair(
                    metric_name,
                    self.feats_A,
                    self.feats_B,
                    **kwargs,
                )
                expected = AlignmentMetrics.measure(
                    metric_name,
                    self.feats_A,
                    self.feats_B[permutation],
                    **kwargs,
                )
                actual = score_prepared_metric(
                    prepared,
                    target_permutation=permutation,
                )
                self.assertAlmostEqual(actual, expected, places=9)

    def test_prepared_kwargs_are_snapshotted_and_immutable(self):
        kwargs = {"rbf_sigma": 0.4, "diffusion_time": 0.7}
        prepared = prepare_metric_pair(
            "rbf_rwka_diffusion_time",
            self.feats_A,
            self.feats_B,
            **kwargs,
        )
        kwargs["diffusion_time"] = 2.0

        expected = AlignmentMetrics.measure(
            "rbf_rwka_diffusion_time",
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.4,
            diffusion_time=0.7,
        )
        self.assertAlmostEqual(score_prepared_metric(prepared), expected, places=10)
        with self.assertRaises(TypeError):
            prepared.metric_kwargs["diffusion_time"] = 3.0

    def test_every_prepared_curve_matches_a_direct_loop(self):
        registered = {
            spec.name
            for spec in METRIC_SPECS
            if spec.sweep is not None and spec.prepared_family is not None
        }
        self.assertEqual(set(self.CURVE_CASES), registered)
        for metric_name, case in self.CURVE_CASES.items():
            parameter_name, values, fixed_kwargs = case
            with self.subTest(metric_name=metric_name):
                prepared = prepare_metric_curve(
                    metric_name,
                    self.feats_A,
                    self.feats_B,
                    parameter_name,
                    values,
                    **fixed_kwargs,
                )
                actual = score_prepared_curve(prepared)
                expected = []
                for value in values:
                    kwargs = dict(fixed_kwargs)
                    kwargs[parameter_name] = value
                    expected.append(
                        AlignmentMetrics.measure(
                            metric_name,
                            self.feats_A,
                            self.feats_B,
                            **kwargs,
                        )
                    )
                torch.testing.assert_close(
                    torch.tensor(actual),
                    torch.tensor(expected),
                    rtol=1e-9,
                    atol=1e-9,
                )

    def test_hoisted_rbf_bandwidth_is_bit_identical_to_per_point_resolution(self):
        """The reused sweep bandwidth must not perturb a single result bit.

        Downstream aggregates are not all continuous in these scores, so
        equality here is exact rather than approximate.
        """
        values = [0.2, 0.6, 1.1, 2.4]
        rbf_metrics = [
            name
            for name, (parameter_name, _, _) in self.CURVE_CASES.items()
            if parameter_name == "rbf_sigma"
        ]
        self.assertTrue(rbf_metrics)
        for metric_name in rbf_metrics:
            with self.subTest(metric_name=metric_name):
                prepared = prepare_metric_curve(
                    metric_name,
                    self.feats_A,
                    self.feats_B,
                    "rbf_sigma",
                    values,
                )
                hoisted = score_prepared_curve(prepared)
                per_point = [
                    _score_rbf_curve_value(prepared, value) for value in values
                ]
                self.assertEqual(hoisted, per_point)

    def test_rbf_sigma_sweep_resolves_each_bandwidth_once(self):
        """Guard the optimization itself: the median must not creep back per point."""
        values = [0.2, 0.6, 1.1, 2.4, 3.0]
        prepared = prepare_metric_curve(
            "rbf_rwka",
            self.feats_A,
            self.feats_B,
            "rbf_sigma",
            values,
        )
        with mock.patch("torch.median", side_effect=torch.median) as median:
            score_prepared_curve(prepared)
        self.assertEqual(median.call_count, 2)

    def test_quantile_sweep_still_resolves_every_point(self):
        """A quantile sweep varies the quantile, so its bandwidth cannot be reused."""
        values = [0.2, 0.4, 0.6]
        prepared = prepare_metric_curve(
            "rbf_rwka_quantile",
            self.feats_A,
            self.feats_B,
            "quantile",
            values,
        )
        with mock.patch("torch.quantile", side_effect=torch.quantile) as quantile:
            score_prepared_curve(prepared)
        self.assertEqual(quantile.call_count, 2 * len(values))

    def test_capability_and_permutation_validation(self):
        self.assertTrue(supports_prepared_metric("cka"))
        self.assertTrue(supports_prepared_metric("mutual_knn"))
        self.assertFalse(supports_prepared_metric("svcca"))
        self.assertTrue(supports_prepared_curve("mutual_knn"))
        self.assertFalse(supports_prepared_curve("svcca"))
        with self.assertRaisesRegex(ValueError, "does not support prepared scoring"):
            prepare_metric_pair("svcca", self.feats_A, self.feats_B)

        prepared = prepare_metric_pair("cka", self.feats_A, self.feats_B)
        with self.assertRaisesRegex(ValueError, "one index per sample"):
            score_prepared_metric(prepared, target_permutation=[0, 1])
        with self.assertRaisesRegex(ValueError, "valid permutation"):
            score_prepared_metric(
                prepared,
                target_permutation=[0, 1, 2, 3, 4, 5, 6, 6],
            )


if __name__ == "__main__":
    unittest.main()
