import unittest
from unittest import mock

import numpy as np
import torch

from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.metrics import METRIC_REGISTRY, METRIC_SPECS, get_metric_spec
from manifold_repsim.metrics import center_kernel
from manifold_repsim.metrics import degree_center_random_walk_kernel
from manifold_repsim.metrics import hsic_biased
from manifold_repsim.metrics import hsic_unbiased
from manifold_repsim.metrics import normalize_random_walk_kernel
from manifold_repsim.sweeps import map_param_name_to_kwargs


class MetricRegistryTests(unittest.TestCase):
    REMOVED_METRICS = {
        "lcs_knn",
        "edit_distance_knn",
        "nn_rwka",
        "ip_nn_rwka",
        "asym_nn_rwka",
    }

    def test_every_metric_is_registered_once_and_has_an_implementation(self):
        names = [spec.name for spec in METRIC_SPECS]

        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names, AlignmentMetrics.SUPPORTED_METRICS)
        for spec in METRIC_SPECS:
            self.assertTrue(callable(getattr(AlignmentMetrics, spec.implementation)))

    def test_metric_implementations_live_in_focused_modules(self):
        expected_modules = {
            "cycle_knn": "manifold_repsim.metrics.neighbors",
            "mutual_knn": "manifold_repsim.metrics.neighbors",
            "cknna": "manifold_repsim.metrics.neighbors",
            "cka": "manifold_repsim.metrics.kernels",
            "unbiased_cka": "manifold_repsim.metrics.kernels",
            "rbf_uka": "manifold_repsim.metrics.kernels",
            "svcca": "manifold_repsim.metrics.kernels",
            "softmax_rwka": "manifold_repsim.metrics.random_walk",
            "rbf_rwka": "manifold_repsim.metrics.random_walk",
            "rbf_rwka_symmetric": "manifold_repsim.metrics.random_walk",
            "rbf_crwka": "manifold_repsim.metrics.random_walk",
            "rbf_degree_crwka": "manifold_repsim.metrics.random_walk",
            "rbf_rwka_diffusion_time": "manifold_repsim.metrics.random_walk",
        }
        for implementation, module_name in expected_modules.items():
            with self.subTest(implementation=implementation):
                self.assertEqual(
                    getattr(AlignmentMetrics, implementation).__module__,
                    module_name,
                )

    def test_interface_preserves_legacy_feature_keywords(self):
        features = np.eye(4)

        score = AlignmentMetrics.measure(
            metric="cka",
            feats_A=features,
            feats_B=features,
        )
        direct_score = AlignmentMetrics.mutual_knn(
            feats_A=features,
            feats_B=features,
            topk=2,
        )

        self.assertIsInstance(score, float)
        self.assertIsInstance(direct_score, float)

    def test_interface_preserves_legacy_kernel_keywords(self):
        K = torch.eye(4, dtype=torch.float64)
        L = K.clone()

        self.assertAlmostEqual(
            AlignmentMetrics.score_cka_kernels(K=K, L=L),
            1.0,
        )
        self.assertAlmostEqual(
            AlignmentMetrics.score_rwka_kernels(K=K, L=L, metric_name="test"),
            1.0,
        )
        self.assertAlmostEqual(
            AlignmentMetrics.score_kernel_metric(metric="cka", K=K, L=L),
            1.0,
        )
        self.assertEqual(center_kernel(K=K).shape, K.shape)
        self.assertTrue(torch.isfinite(hsic_biased(K=K, L=L)))
        self.assertTrue(torch.isfinite(hsic_unbiased(K=K, L=L)))
        self.assertEqual(normalize_random_walk_kernel(K=K).shape, K.shape)
        self.assertEqual(degree_center_random_walk_kernel(K=K).shape, K.shape)

    def test_legacy_sweep_and_kernel_metadata_are_derived_from_registry(self):
        expected_sweeps = {
            name: spec.compatibility_sweep_dict()
            for name, spec in METRIC_REGISTRY.items()
        }
        expected_reusable = {
            name for name, spec in METRIC_REGISTRY.items() if spec.reuses_kernel
        }

        self.assertEqual(AlignmentMetrics.SWEEP_PARAMS, expected_sweeps)
        self.assertEqual(
            AlignmentMetrics.KERNEL_REUSABLE_METRICS,
            expected_reusable,
        )
        self.assertFalse(AlignmentMetrics.supports_kernel_reuse("missing"))

    def test_removed_metrics_are_not_registered_or_implemented(self):
        self.assertTrue(self.REMOVED_METRICS.isdisjoint(METRIC_REGISTRY))
        for metric_name in self.REMOVED_METRICS:
            self.assertFalse(hasattr(AlignmentMetrics, metric_name))

    def test_every_metric_implementation_documents_its_formula(self):
        implementations = {spec.implementation for spec in METRIC_SPECS}
        for implementation in implementations:
            docstring = getattr(AlignmentMetrics, implementation).__doc__ or ""
            self.assertIn("Formula:", docstring, implementation)

    def test_alias_dispatch_uses_registry_target_and_fixed_arguments(self):
        features = np.eye(3)
        with mock.patch.object(
            AlignmentMetrics,
            "mutual_knn",
            return_value=0.25,
        ) as score:
            result = AlignmentMetrics.measure(
                "mutual_knn_dist",
                features,
                features,
                topk=2,
                use_distance=False,
            )

        self.assertEqual(result, 0.25)
        kwargs = score.call_args.kwargs
        self.assertEqual(kwargs["topk"], 2)
        self.assertTrue(kwargs["use_distance"])

    def test_unknown_metric_uses_existing_public_error(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized metric: missing"):
            get_metric_spec("missing")
        with self.assertRaisesRegex(ValueError, "Unrecognized metric: missing"):
            AlignmentMetrics.measure("missing", np.eye(3), np.eye(3))

    def test_parameter_defaults_come_from_registry(self):
        self.assertEqual(map_param_name_to_kwargs("topk"), {"topk": 10})
        self.assertEqual(
            map_param_name_to_kwargs("rbf_sigma"),
            {"rbf_sigma": 1.0},
        )
        self.assertEqual(
            map_param_name_to_kwargs("rbf_sigma", local=True),
            {"rbf_sigma": 0.2},
        )


if __name__ == "__main__":
    unittest.main()
