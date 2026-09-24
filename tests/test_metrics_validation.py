from __future__ import annotations

import math
import unittest
import warnings

import numpy as np
import torch

from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.metrics import compute_nearest_neighbors
from manifold_repsim.metrics import compute_rbf_distance_cache
from manifold_repsim.metrics import compute_rbf_kernel
from manifold_repsim.metrics import compute_softmax_kernel
from manifold_repsim.metrics import resolve_distance_tolerance
from manifold_repsim.metrics import resolve_rbf_base_bandwidth


class MetricValidationTests(unittest.TestCase):
    def setUp(self):
        self.features = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.5],
                [2.0, 0.5, 1.0],
                [0.5, 2.0, 1.5],
            ],
            dtype=torch.float64,
        )

    def test_pair_metrics_require_finite_2d_representations_with_matching_samples(self):
        with self.assertRaisesRegex(ValueError, "2D"):
            AlignmentMetrics.cka(self.features.reshape(2, 3, 3), self.features)
        with self.assertRaisesRegex(ValueError, "sample counts"):
            AlignmentMetrics.cka(self.features, self.features[:-1])
        with self.assertRaisesRegex(ValueError, "non-finite"):
            invalid = self.features.clone()
            invalid[0, 0] = torch.nan
            AlignmentMetrics.cka(self.features, invalid)
        with self.assertRaisesRegex(ValueError, "empty"):
            AlignmentMetrics.cka(torch.empty(0, 3), torch.empty(0, 4))

    def test_pair_metrics_allow_different_feature_widths_and_integer_inputs(self):
        narrower = np.arange(12).reshape(6, 2)
        score = AlignmentMetrics.measure("cka", self.features, narrower)

        self.assertIsInstance(score, float)
        self.assertTrue(math.isfinite(score))

    def test_neighbor_count_excludes_self_and_is_not_silently_clamped(self):
        neighbors = compute_nearest_neighbors(self.features, topk=5)
        self.assertEqual(neighbors.shape, (6, 5))
        self.assertFalse(torch.any(neighbors == torch.arange(6).reshape(-1, 1)).item())

        for invalid_topk in (0, 6, 7, 1.5, True):
            with self.subTest(topk=invalid_topk):
                with self.assertRaises((TypeError, ValueError)):
                    compute_nearest_neighbors(self.features, topk=invalid_topk)

    def test_cknna_none_uses_all_nonself_neighbors(self):
        score = AlignmentMetrics.cknna(self.features, self.features, topk=None)
        distance_agnostic = AlignmentMetrics.cknna(
            self.features,
            self.features,
            topk=3,
            distance_agnostic=True,
        )

        self.assertIsInstance(score, float)
        self.assertTrue(math.isfinite(score))
        self.assertAlmostEqual(distance_agnostic, 1.0)
        with self.assertRaisesRegex(ValueError, "at most 5"):
            AlignmentMetrics.cknna(self.features, self.features, topk=6)

    def test_unbiased_metrics_require_four_samples(self):
        small = self.features[:3]
        with self.assertRaisesRegex(ValueError, "at least 4 samples"):
            AlignmentMetrics.unbiased_cka(small, small)
        with self.assertRaisesRegex(ValueError, "at least 4 samples"):
            AlignmentMetrics.cknna(small, small, topk=2, unbiased=True)

    def test_zero_heuristic_bandwidth_uses_a_finite_fallback(self):
        constant = torch.ones(5, 3)
        kernel = compute_rbf_kernel(constant, rbf_sigma=0.5)

        torch.testing.assert_close(kernel, torch.ones(5, 5, dtype=torch.float64))
        self.assertTrue(torch.isfinite(kernel).all().item())

    def test_quantization_floor_bandwidth_falls_back_to_resolvable_distances(self):
        """Rows duplicated to within float32 precision must not set the scale.

        A representation whose rows agree to the last stored bit puts the median
        pairwise distance at one quantization step. Calibrating the bandwidth on
        that step measures float32 precision rather than geometry, and scaling it
        by a sweep's ``rbf_sigma`` used to leave a value small enough to reject.
        """
        step = 2.0**-10
        base = torch.full((1, 4), step, dtype=torch.float32)
        rows = [base.clone() for _ in range(8)]
        for index in (1, 3, 5):
            rows[index][0, 0] = torch.nextafter(
                rows[index][0, 0],
                torch.tensor(float("inf"), dtype=torch.float32),
            )
        features = torch.cat(rows + [base + 2.0**-12, base + 2.0**-11], dim=0)

        cache = compute_rbf_distance_cache(features)
        raw_median = float(cache.lower_distances.median())
        base_bandwidth = resolve_rbf_base_bandwidth(cache, median=True)

        self.assertLessEqual(raw_median, cache.distance_tolerance)
        self.assertGreater(base_bandwidth, cache.distance_tolerance)
        for rbf_sigma in (0.5, 0.05):
            kernel = compute_rbf_kernel(features, rbf_sigma=rbf_sigma)
            self.assertTrue(torch.isfinite(kernel).all().item())

    def test_uniformly_small_representations_keep_the_median_bandwidth(self):
        """The median heuristic is scale free, so smallness alone changes nothing."""
        scale = 2.0**-40
        tiny = self.features * scale
        cache = compute_rbf_distance_cache(tiny)
        base_bandwidth = resolve_rbf_base_bandwidth(cache, median=True)

        self.assertGreater(base_bandwidth, cache.distance_tolerance)
        self.assertEqual(base_bandwidth, float(cache.lower_distances.median()))
        torch.testing.assert_close(
            compute_rbf_kernel(tiny, rbf_sigma=0.5),
            compute_rbf_kernel(self.features, rbf_sigma=0.5),
        )

    def test_distance_tolerance_tracks_dtype_and_representation_scale(self):
        as_float32 = torch.ones(4, 4, dtype=torch.float32)
        as_float64 = torch.ones(4, 4, dtype=torch.float64)

        self.assertGreater(
            resolve_distance_tolerance(as_float32),
            resolve_distance_tolerance(as_float64),
        )
        self.assertAlmostEqual(
            resolve_distance_tolerance(as_float32 * 8.0),
            resolve_distance_tolerance(as_float32) * 8.0,
        )

    def test_invalid_explicit_rbf_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "rbf_sigma"):
            compute_rbf_kernel(self.features, rbf_sigma=0.0, median=False)
        with self.assertRaisesRegex(ValueError, "quantile"):
            compute_rbf_kernel(self.features, quantile=1.1)

    def test_constant_centered_kernels_return_zero_with_a_runtime_warning(self):
        constant = torch.ones(5, 3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            score = AlignmentMetrics.cka(constant, constant)

        self.assertEqual(score, 0.0)
        self.assertTrue(any(item.category is RuntimeWarning for item in caught))

    def test_softmax_kernel_validates_temperature_and_handles_constant_input(self):
        constant = torch.ones(5, 3)
        kernel = compute_softmax_kernel(constant, temperature=0.5)

        torch.testing.assert_close(
            kernel,
            torch.full((5, 5), 0.2, dtype=torch.float64),
        )
        with self.assertRaisesRegex(ValueError, "temperature"):
            compute_softmax_kernel(self.features, temperature=0.0)

    def test_svcca_is_deterministic_and_does_not_advance_rngs(self):
        target = self.features + torch.tensor(
            [0.01, -0.02, 0.03],
            dtype=self.features.dtype,
        )
        torch.manual_seed(123)
        np.random.seed(123)
        expected_torch = torch.rand(4)
        expected_numpy = np.random.rand(4)

        torch.manual_seed(123)
        np.random.seed(123)
        first = AlignmentMetrics.svcca(self.features, target, cca_dim=2)
        second = AlignmentMetrics.svcca(self.features, target, cca_dim=2)
        observed_torch = torch.rand(4)
        observed_numpy = np.random.rand(4)

        self.assertEqual(first, second)
        torch.testing.assert_close(observed_torch, expected_torch)
        np.testing.assert_array_equal(observed_numpy, expected_numpy)
        with self.assertRaisesRegex(ValueError, "cca_dim must be at most 3"):
            AlignmentMetrics.svcca(self.features, target, cca_dim=4)


if __name__ == "__main__":
    unittest.main()
