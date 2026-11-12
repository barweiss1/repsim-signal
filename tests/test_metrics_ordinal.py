"""Triplet and quadruplet similarity indices: exactness, ties, and the paper's claims."""

from __future__ import annotations

import itertools
import unittest
from unittest import mock

import numpy as np
import torch

from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.metrics import mutual_knn
from manifold_repsim.metrics import prepare_metric_pair
from manifold_repsim.metrics import qsi
from manifold_repsim.metrics import score_prepared_metric
from manifold_repsim.metrics import tsi
from manifold_repsim.metrics import ordinal
from manifold_repsim.metrics.ordinal import agreeing_pairs
from manifold_repsim.metrics.ordinal import ordinal_dissimilarity
from manifold_repsim.metrics.ordinal import tolerant_ranks


def _sign(value):
    return int(value > 0) - int(value < 0)


def brute_force(X, Y, quadruplets, use_distance=True):
    """The definition: every ordered tuple, exact comparison, ties as sign 0."""
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)

    def d(points, a, b):
        if use_distance:
            return float(np.linalg.norm(points[a] - points[b]))
        return float(points[a] @ points[b])

    agree = total = 0
    if quadruplets:
        for i, j, k, l in itertools.permutations(range(len(X)), 4):
            agree += _sign(d(X, i, j) - d(X, k, l)) == _sign(d(Y, i, j) - d(Y, k, l))
            total += 1
    else:
        for i, j, k in itertools.permutations(range(len(X)), 3):
            agree += _sign(d(X, i, j) - d(X, i, k)) == _sign(d(Y, i, j) - d(Y, i, k))
            total += 1
    return agree / total


def A(*rows):
    return np.array(rows, dtype=float)


class BruteForceEquivalenceTests(unittest.TestCase):
    def datasets(self, n):
        rng = np.random.default_rng(n)
        random_x, random_y = rng.normal(size=(n, 3)), rng.normal(size=(n, 2))
        duplicated = rng.normal(size=(n, 3))
        duplicated[n // 2 :] = duplicated[: n - n // 2]
        yield "random", random_x, random_y
        yield "duplicate rows", duplicated, random_y
        yield "integer grid", *(
            rng.integers(0, 3, size=(n, 2)).astype(float) for _ in range(2)
        )
        yield "constant target", random_x, np.zeros((n, 2))

    def test_fast_path_equals_the_definition_including_ties(self):
        for n in (4, 5, 7, 9):
            for label, X, Y in self.datasets(n):
                for use_distance in (True, False):
                    for quadruplets, metric in ((False, tsi), (True, qsi)):
                        with self.subTest(
                            n=n, data=label, l2=use_distance, quad=quadruplets
                        ):
                            self.assertAlmostEqual(
                                metric(X, Y, use_distance=use_distance),
                                brute_force(X, Y, quadruplets, use_distance),
                                places=12,
                            )

    def test_counting_primitive_matches_a_pairwise_loop(self):
        rng = np.random.default_rng(0)
        for size in (2, 3, 7, 40, 129):
            top = min(5, size)
            a = rng.integers(1, top + 1, size=(3, size))
            b = rng.integers(1, top + 1, size=(3, size))
            got, total = agreeing_pairs(torch.tensor(a), torch.tensor(b))
            self.assertEqual(total, size * (size - 1) // 2)
            for row in range(3):
                want = sum(
                    _sign(a[row, p] - a[row, q]) == _sign(b[row, p] - b[row, q])
                    for p, q in itertools.combinations(range(size), 2)
                )
                self.assertEqual(int(got[row]), want, (size, row))

    def test_scores_do_not_depend_on_the_chunk_size(self):
        rng = np.random.default_rng(1)
        X, Y = rng.normal(size=(30, 4)), rng.normal(size=(30, 3))
        expected = (tsi(X, Y), qsi(X, Y))
        with mock.patch.object(ordinal, "CHUNK_ENTRIES", 70):
            self.assertEqual((tsi(X, Y), qsi(X, Y)), expected)


class TieSemanticsTests(unittest.TestCase):
    """A tie is the third outcome 0: matched ties agree, one-sided ties do not."""

    def test_reference_fixtures(self):
        cases = (
            (tsi, A([0, 0], [1, 0], [0, 1]), A([0, 0], [1, 0], [1, 1]), 0.0),
            (
                tsi,
                A([0, 0], [1, 0], [0, 1], [1, 1]),
                A([0, 0], [1, 0], [0, 1], [2, 2]),
                2 / 3,
            ),
            (
                qsi,
                A([0, 0], [1, 0], [0, 1], [1, 1]),
                A([0, 0], [1, 0], [0, 1], [2, 2]),
                0.0,
            ),
            (tsi, A([0, 0], [2, 0], [3, 0]), A([0, 0], [2, 0], [1, 0]), 1 / 3),
        )
        for metric, X, Y, expected in cases:
            with self.subTest(metric=metric.__name__, expected=expected):
                self.assertAlmostEqual(metric(X, Y), expected, places=12)

    def test_paper_example_separates_triplets_from_quadruplets(self):
        """Contracting one remote cluster keeps every neighborhood but not the scale."""
        big = 1000.0
        X = A([0], [1], [big], [big + 2])
        contracted = A([0], [1], [big], [big + 1])
        self.assertAlmostEqual(tsi(X, contracted), 1.0, places=12)
        self.assertAlmostEqual(qsi(X, contracted), 1 / 3, places=12)

    def test_identical_representations_agree_completely(self):
        Z = np.random.default_rng(2).normal(size=(25, 4))
        self.assertEqual(tsi(Z, Z), 1.0)
        self.assertAlmostEqual(qsi(Z, Z), 1.0, places=12)

    def test_matched_ties_agree_and_one_sided_ties_do_not(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(8, 2))
        constant = np.zeros((8, 2))
        self.assertEqual(tsi(constant, constant), 1.0)
        self.assertAlmostEqual(qsi(constant, constant), 1.0, places=12)
        self.assertEqual(tsi(X, constant), 0.0)
        self.assertAlmostEqual(qsi(X, constant), 0.0, places=12)

    def test_duplicate_rows_are_finite_and_well_defined(self):
        X = np.random.default_rng(4).normal(size=(10, 3))
        X[5:] = X[:5]
        for score in (tsi(X, X), qsi(X, X)):
            self.assertTrue(np.isfinite(score))

    def test_ties_survive_a_large_common_offset(self):
        """Exact differences keep a zero distance zero where the expansion would not."""
        rng = np.random.default_rng(5)
        X = rng.integers(0, 4, size=(60, 3)).astype(float)
        Y = rng.integers(0, 4, size=(60, 3)).astype(float)
        expected = (tsi(X, Y), qsi(X, Y))
        for offset in (1e3, 1e6, 1e9):
            with self.subTest(offset=offset):
                self.assertEqual(
                    (
                        tsi(X + offset, Y + 0.5 * offset),
                        qsi(X + offset, Y + 0.5 * offset),
                    ),
                    expected,
                )

    def test_duplicate_rows_stay_at_exactly_zero_distance_on_continuous_data(self):
        rng = np.random.default_rng(6)
        X = torch.tensor(rng.normal(size=(200, 64)) + 1e6)
        X[100:110] = X[:10]
        matrix, _ = ordinal_dissimilarity(X)
        for i in range(10):
            self.assertEqual(float(matrix[i, 100 + i]), 0.0)


class ToleranceTests(unittest.TestCase):
    """The tolerance is float64 error, not input precision or data spread."""

    def test_tolerance_ignores_the_input_dtype(self):
        X = np.random.default_rng(7).normal(size=(40, 16)).astype(np.float32)
        _, from_float32 = ordinal_dissimilarity(torch.tensor(X))
        _, from_float64 = ordinal_dissimilarity(torch.tensor(X, dtype=torch.float64))
        self.assertEqual(from_float32, from_float64)

    def test_tolerance_is_far_below_the_spacing_of_dense_distances(self):
        """A coarse tolerance chains dense distances into spurious ties."""
        X = np.random.default_rng(8).normal(size=(200, 512)).astype(np.float32)
        matrix, tolerance = ordinal_dissimilarity(torch.tensor(X))
        upper = matrix[torch.triu_indices(200, 200, 1).unbind()]
        self.assertLess(tolerance, 1e-9 * float(upper.max()))
        ranks = tolerant_ranks(upper.unsqueeze(0), tolerance)
        self.assertEqual(int(ranks.unique().numel()), upper.numel())

    def test_independent_representations_score_about_one_half(self):
        """Lemma 1 with negligible ties; sizes chosen so a merged-tie bug shows."""
        rng = np.random.default_rng(9)
        for n in (60, 400):
            X, Y = rng.normal(size=(n, 128)), rng.normal(size=(n, 96))
            for metric in (tsi, qsi):
                with self.subTest(n=n, metric=metric.__name__):
                    self.assertAlmostEqual(metric(X, Y), 0.5, delta=0.03)


class PaperClaimTests(unittest.TestCase):
    def test_perfect_triplet_alignment_is_mutual_knn_at_every_scale(self):
        """Corollary 1, on tie-free data."""
        rng = np.random.default_rng(10)
        X = rng.normal(size=(14, 3))
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        aligned = 2.5 * X @ rotation + 7.0
        self.assertEqual(tsi(X, aligned), 1.0)
        for k in range(1, 13):
            self.assertEqual(mutual_knn(X, aligned, topk=k, use_distance=True), 1.0)

        scrambled = rng.normal(size=(14, 3))
        self.assertLess(tsi(X, scrambled), 1.0)
        self.assertTrue(
            any(
                mutual_knn(X, scrambled, topk=k, use_distance=True) < 1.0
                for k in range(1, 13)
            )
        )

    def test_invariances(self):
        rng = np.random.default_rng(11)
        X, Y = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        for metric in (tsi, qsi):
            base = metric(X, Y)
            for transformed in (X + 3.0, 4.0 * X, X @ rotation):
                self.assertAlmostEqual(metric(transformed, Y), base, places=12)
            skew = X @ np.diag([1.0, 5.0, 0.2, 2.0])
            self.assertNotAlmostEqual(metric(skew, Y), base, places=3)

    def test_outliers_move_the_score_by_at_most_their_share(self):
        """Lemma 2 / Corollary 2 with S(X_U, Y_U) = 1."""
        rng = np.random.default_rng(12)
        n, clean = 40, 36
        X = rng.normal(size=(n, 5))
        corrupted = X.copy()
        corrupted[clean:] += 1e4 * rng.normal(size=(n - clean, 5))
        falling = lambda m, k: np.prod([m - i for i in range(k)])
        for metric, degree in ((tsi, 3), (qsi, 4)):
            proportion = falling(clean, degree) / falling(n, degree)
            score = metric(corrupted, X)
            self.assertGreaterEqual(score, proportion - 1e-12)
            self.assertLessEqual(score, 1.0)

    def test_global_lower_bounds_in_one_dimension(self):
        """Table 1: no configuration of N distinct points on a line scores lower."""
        rng = np.random.default_rng(13)
        bounds = ((5, tsi, 2 / 15), (6, tsi, 1 / 6), (6, qsi, 1 / 9))
        for n, metric, bound in bounds:
            for _ in range(150):
                X, Y = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
                self.assertGreaterEqual(metric(X, Y), bound - 1e-12)


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(14)
        self.X, self.Y = rng.normal(size=(30, 5)), rng.normal(size=(30, 3))

    def test_inner_product_aliases_route_to_the_same_implementation(self):
        for plain, alias in (("tsi", "tsi_ip"), ("qsi", "qsi_ip")):
            expected = getattr(AlignmentMetrics, plain)(
                self.X, self.Y, use_distance=False
            )
            self.assertEqual(AlignmentMetrics.measure(alias, self.X, self.Y), expected)
            # The registry owns the alias's distance choice, as for mutual_knn_dist.
            self.assertEqual(
                AlignmentMetrics.measure(alias, self.X, self.Y, use_distance=True),
                expected,
            )

    def test_the_two_distance_conventions_differ(self):
        self.assertNotEqual(
            tsi(self.X, self.Y), tsi(self.X, self.Y, use_distance=False)
        )

    def test_minimum_sample_counts(self):
        X = np.random.default_rng(15).normal(size=(3, 2))
        self.assertTrue(np.isfinite(tsi(X, X)))
        with self.assertRaises(ValueError):
            tsi(X[:2], X[:2])
        with self.assertRaises(ValueError):
            qsi(X, X)

    def test_inputs_are_validated_like_the_other_pair_metrics(self):
        with self.assertRaisesRegex(ValueError, "sample counts"):
            tsi(self.X[:10], self.Y[:9])
        bad = self.X.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            qsi(bad, self.Y)

    def test_float32_and_float64_inputs_score_identically(self):
        X32, Y32 = self.X.astype(np.float32), self.Y.astype(np.float32)
        self.assertEqual(tsi(X32, Y32), tsi(X32.astype(float), Y32.astype(float)))


class SubsamplingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(16)
        self.X, self.Y = rng.normal(size=(50, 4)), rng.normal(size=(50, 4))

    def test_a_cap_at_least_the_sample_count_changes_nothing(self):
        for metric in (tsi, qsi):
            self.assertEqual(
                metric(self.X, self.Y, max_points=50), metric(self.X, self.Y)
            )
            self.assertEqual(
                metric(self.X, self.Y, max_points=500), metric(self.X, self.Y)
            )

    def test_a_cap_scores_one_seeded_shared_subset_exactly(self):
        rows = np.sort(np.random.RandomState(0).choice(50, 20, replace=False))
        for metric in (tsi, qsi):
            with self.subTest(metric=metric.__name__):
                capped = metric(self.X, self.Y, max_points=20)
                self.assertEqual(capped, metric(self.X, self.Y, max_points=20))
                self.assertAlmostEqual(
                    capped, metric(self.X[rows], self.Y[rows]), places=12
                )
                self.assertNotEqual(
                    capped, metric(self.X, self.Y, max_points=20, subsample_seed=1)
                )

    def test_the_cap_is_validated(self):
        with self.assertRaises(TypeError):
            tsi(self.X, self.Y, max_points=True)
        with self.assertRaises(ValueError):
            qsi(self.X, self.Y, max_points=3)

    def test_the_prepared_path_applies_the_same_cap(self):
        for name in ("tsi", "qsi_ip"):
            prepared = prepare_metric_pair(name, self.X, self.Y, max_points=20)
            self.assertEqual(
                score_prepared_metric(prepared),
                AlignmentMetrics.measure(name, self.X, self.Y, max_points=20),
            )


if __name__ == "__main__":
    unittest.main()
