import unittest

import torch

from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.metrics import compute_rbf_kernel
from manifold_repsim.metrics import degree_center_random_walk_kernel
from manifold_repsim.metrics import power_rbf_rwka_diffusion_cache
from manifold_repsim.metrics import normalize_random_walk_kernel
from manifold_repsim.sweeps import measure_with_permutation_null


class TestRWKAKernelNormalization(unittest.TestCase):
    def setUp(self):
        self.feats_A = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
                [2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.feats_B = torch.tensor(
            [
                [0.1, 0.0],
                [0.9, 0.2],
                [0.1, 1.8],
                [2.1, 0.8],
            ],
            dtype=torch.float32,
        )

    def test_rbf_rwka_reusable_kernels_are_row_stochastic(self):
        K, L = AlignmentMetrics.compute_kernel_pair(
            "rbf_rwka",
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.5,
        )

        torch.testing.assert_close(K.sum(dim=1), torch.ones(K.shape[0], dtype=K.dtype))
        torch.testing.assert_close(L.sum(dim=1), torch.ones(L.shape[0], dtype=L.dtype))

    def test_cka_rbf_reusable_kernels_remain_raw_rbf(self):
        K, _ = AlignmentMetrics.compute_kernel_pair(
            "cka_rbf",
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.5,
        )
        raw_K = compute_rbf_kernel(self.feats_A, rbf_sigma=0.5)

        torch.testing.assert_close(K, raw_K)
        self.assertFalse(torch.allclose(K.sum(dim=1), torch.ones(K.shape[0], dtype=K.dtype)))

    def test_rbf_rwka_matches_reusable_kernel_path(self):
        direct_score = AlignmentMetrics.rbf_rwka(self.feats_A, self.feats_B, rbf_sigma=0.5)
        K, L = AlignmentMetrics.compute_kernel_pair(
            "rbf_rwka",
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.5,
        )
        reusable_score = AlignmentMetrics.score_kernel_metric("rbf_rwka", K, L)

        self.assertAlmostEqual(direct_score, reusable_score)

    def test_symmetric_rbf_rwka_reusable_kernels_use_symmetric_normalization(self):
        K, L = AlignmentMetrics.compute_kernel_pair(
            "rbf_rwka_symmetric",
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.5,
        )
        expected_K = normalize_random_walk_kernel(
            compute_rbf_kernel(self.feats_A, rbf_sigma=0.5),
            symmetric=True,
        )
        expected_L = normalize_random_walk_kernel(
            compute_rbf_kernel(self.feats_B, rbf_sigma=0.5),
            symmetric=True,
        )

        torch.testing.assert_close(K, expected_K)
        torch.testing.assert_close(L, expected_L)
        torch.testing.assert_close(K, K.T)
        torch.testing.assert_close(L, L.T)

    def test_symmetric_rbf_rwka_matches_reusable_kernel_path(self):
        direct_score = AlignmentMetrics.rbf_rwka_symmetric(
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.5,
        )
        K, L = AlignmentMetrics.compute_kernel_pair(
            "rbf_rwka_symmetric",
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.5,
        )
        reusable_score = AlignmentMetrics.score_kernel_metric(
            "rbf_rwka_symmetric",
            K,
            L,
        )

        self.assertAlmostEqual(direct_score, reusable_score)

    def test_rbf_uka_matches_raw_kernel_frobenius_alignment(self):
        K = compute_rbf_kernel(self.feats_A, rbf_sigma=0.5)
        L = compute_rbf_kernel(self.feats_B, rbf_sigma=0.5)
        expected = torch.sum(K * L) / torch.sqrt(torch.sum(K * K) * torch.sum(L * L))

        observed = AlignmentMetrics.rbf_uka(self.feats_A, self.feats_B, rbf_sigma=0.5)

        self.assertAlmostEqual(observed, expected.item())

    def test_rbf_crwka_matches_centered_symmetric_kernel_alignment(self):
        K = normalize_random_walk_kernel(
            compute_rbf_kernel(self.feats_A, rbf_sigma=0.5),
            symmetric=True,
        )
        L = normalize_random_walk_kernel(
            compute_rbf_kernel(self.feats_B, rbf_sigma=0.5),
            symmetric=True,
        )
        expected = AlignmentMetrics.score_cka_kernels(K, L, unbiased=False)

        observed = AlignmentMetrics.rbf_crwka(self.feats_A, self.feats_B, rbf_sigma=0.5)

        self.assertAlmostEqual(observed, expected)

    def test_degree_centering_matches_dense_projection_and_removes_trivial_mode(self):
        raw_kernel = compute_rbf_kernel(self.feats_A, rbf_sigma=0.5)
        degree = raw_kernel.sum(dim=1)
        mode = degree.sqrt()
        normalized = normalize_random_walk_kernel(raw_kernel, symmetric=True)
        projector = torch.eye(raw_kernel.shape[0], dtype=raw_kernel.dtype) - torch.outer(mode, mode) / torch.dot(mode, mode)
        expected = projector @ normalized @ projector

        observed = degree_center_random_walk_kernel(raw_kernel)

        torch.testing.assert_close(observed, expected)
        torch.testing.assert_close(
            projector @ mode,
            torch.zeros_like(mode),
            atol=1e-10,
            rtol=1e-10,
        )
        torch.testing.assert_close(
            observed @ mode,
            torch.zeros_like(mode),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_rbf_degree_crwka_matches_projected_kernel_alignment(self):
        K = degree_center_random_walk_kernel(
            compute_rbf_kernel(self.feats_A, rbf_sigma=0.5)
        )
        L = degree_center_random_walk_kernel(
            compute_rbf_kernel(self.feats_B, rbf_sigma=0.5)
        )
        expected = AlignmentMetrics.score_rwka_kernels(
            K,
            L,
            metric_name="Degree-centered RBF RWKA",
        )

        observed = AlignmentMetrics.rbf_degree_crwka(
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.5,
        )

        self.assertAlmostEqual(observed, expected)

    def test_new_rbf_metrics_match_reusable_paths_and_self_align(self):
        for metric_name in ("rbf_uka", "rbf_crwka", "rbf_degree_crwka"):
            with self.subTest(metric=metric_name):
                direct = AlignmentMetrics.measure(
                    metric_name,
                    self.feats_A,
                    self.feats_B,
                    rbf_sigma=0.5,
                )
                K, L = AlignmentMetrics.compute_kernel_pair(
                    metric_name,
                    self.feats_A,
                    self.feats_B,
                    rbf_sigma=0.5,
                )
                reusable = AlignmentMetrics.score_kernel_metric(metric_name, K, L)
                self_score = AlignmentMetrics.measure(
                    metric_name,
                    self.feats_A,
                    self.feats_A,
                    rbf_sigma=0.5,
                )

                self.assertAlmostEqual(direct, reusable)
                self.assertAlmostEqual(self_score, 1.0, places=6)

    def test_new_rbf_metrics_cached_permutations_match_feature_permutations(self):
        permutations = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
        for metric_name in ("rbf_uka", "rbf_crwka", "rbf_degree_crwka"):
            with self.subTest(metric=metric_name):
                observed, null_scores = measure_with_permutation_null(
                    metric_name,
                    self.feats_A,
                    self.feats_B,
                    {"rbf_sigma": 0.5},
                    permutations,
                )
                expected_observed = AlignmentMetrics.measure(
                    metric_name,
                    self.feats_A,
                    self.feats_B,
                    rbf_sigma=0.5,
                )
                expected_null = [
                    AlignmentMetrics.measure(
                        metric_name,
                        self.feats_A,
                        self.feats_B[permutation],
                        rbf_sigma=0.5,
                    )
                    for permutation in permutations
                ]

                self.assertAlmostEqual(observed, expected_observed)
                for actual, expected in zip(null_scores, expected_null):
                    self.assertAlmostEqual(actual, expected, places=6)

    def test_random_walk_symmetric_normalization_matches_diagonal_formula(self):
        K = compute_rbf_kernel(self.feats_A, rbf_sigma=0.5)
        degree = K.sum(dim=0)
        degree = torch.where(degree == 0, torch.ones_like(degree), degree)
        D_inv_sqrt = torch.diag(degree.rsqrt())

        expected = D_inv_sqrt @ K @ D_inv_sqrt
        observed = normalize_random_walk_kernel(K, symmetric=True)

        torch.testing.assert_close(observed, expected)

    def test_random_walk_asymmetric_normalization_is_row_stochastic(self):
        K = torch.tensor(
            [
                [1.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [3.0, 1.0, 2.0],
            ]
        )

        observed = normalize_random_walk_kernel(K, symmetric=False)

        torch.testing.assert_close(observed[0].sum(), torch.tensor(1.0))
        torch.testing.assert_close(observed[1], torch.zeros(3))
        torch.testing.assert_close(observed[2].sum(), torch.tensor(1.0))

    def test_rbf_rwka_diffusion_time_fractional_score_is_finite(self):
        score = AlignmentMetrics.rbf_rwka_diffusion_time(
            self.feats_A,
            self.feats_B,
            diffusion_time=0.5,
            rbf_sigma=0.2,
        )

        self.assertTrue(torch.isfinite(torch.tensor(score)))

    def test_rbf_rwka_diffusion_time_one_matches_row_walk_score(self):
        direct_score = AlignmentMetrics.rbf_rwka_diffusion_time(
            self.feats_A,
            self.feats_B,
            diffusion_time=1.0,
            rbf_sigma=0.2,
        )
        K = normalize_random_walk_kernel(
            compute_rbf_kernel(self.feats_A, rbf_sigma=0.2),
            symmetric=False,
        )
        L = normalize_random_walk_kernel(
            compute_rbf_kernel(self.feats_B, rbf_sigma=0.2),
            symmetric=False,
        )
        expected = AlignmentMetrics.score_rwka_kernels(K, L, metric_name="RBF RWKA")

        self.assertAlmostEqual(direct_score, expected, places=6)

    def test_rbf_rwka_diffusion_time_cached_path_matches_direct(self):
        direct_score = AlignmentMetrics.rbf_rwka_diffusion_time(
            self.feats_A,
            self.feats_B,
            diffusion_time=0.75,
            rbf_sigma=0.2,
        )
        K_cache, L_cache = AlignmentMetrics.compute_kernel_pair(
            "rbf_rwka_diffusion_time",
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.2,
        )
        cached_score = AlignmentMetrics.score_kernel_metric(
            "rbf_rwka_diffusion_time",
            K_cache,
            L_cache,
            diffusion_time=0.75,
        )

        self.assertAlmostEqual(direct_score, cached_score, places=6)

    def test_rbf_rwka_diffusion_time_rejects_nonpositive_time(self):
        K_cache, _ = AlignmentMetrics.compute_kernel_pair(
            "rbf_rwka_diffusion_time",
            self.feats_A,
            self.feats_B,
            rbf_sigma=0.2,
        )

        with self.assertRaises(ValueError):
            power_rbf_rwka_diffusion_cache(K_cache, 0.0)

        with self.assertRaises(ValueError):
            AlignmentMetrics.rbf_rwka_diffusion_time(
                self.feats_A,
                self.feats_B,
                diffusion_time=-1.0,
                rbf_sigma=0.2,
            )


if __name__ == "__main__":
    unittest.main()
