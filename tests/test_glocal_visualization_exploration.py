from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr

from manifold_repsim.experiments.glocal_sweep_mds import exploration
from manifold_repsim.sweeps.signal_aggregation import aggregate_signal_auc


def _build_result(
    *,
    mode: str = "signal",
    lambdas: tuple[float, ...] = (0.001, 0.1),
    alphas: tuple[float, ...] = (0.25, 0.75),
    taus: tuple[float, ...] = (0.1, 0.5),
    n_batches: int = 2,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build a tiny hand-built result matching the exact NPZ schema
    `scoring.analyze_metric` writes (0-d string arrays for scalar fields, as
    `artifacts.load_npz` actually returns them)."""
    rng = np.random.RandomState(seed)
    combos = [(lam, alpha, tau) for lam in lambdas for alpha in alphas for tau in taus]
    condition_names = ["none"] + [
        f"glocal-lambda-{lam:g}-alpha-{alpha:g}-tau-{tau:g}"
        for lam, alpha, tau in combos
    ]
    n = len(condition_names)
    is_none = np.array([True] + [False] * (n - 1))
    lambda_values = np.array([np.nan] + [c[0] for c in combos])
    alpha_values = np.array([np.nan] + [c[1] for c in combos])
    tau_values = np.array([np.nan] + [c[2] for c in combos])

    if mode == "signal":
        parameter_values = np.array([0.5, 1.0, 2.0])
        parameter_name = "rbf_sigma"
    else:
        parameter_values = np.array([0.0])
        parameter_name = ""
    batch_indices = np.arange(n_batches)
    scores = rng.normal(size=(n, n_batches, parameter_values.size))
    concatenated_vectors = scores.reshape(n, -1)
    distances = squareform(pdist(concatenated_vectors))
    embedding_2d = rng.normal(size=(n, 2))
    pca_embedding_2d = rng.normal(size=(n, 2))

    return {
        "metric_id": np.array("test_metric"),
        "metric_name": np.array("test_metric"),
        "metric_mode": np.array(mode),
        "parameter_name": np.array(parameter_name),
        "parameter_values": parameter_values,
        "condition_names": np.array(condition_names),
        "is_none": is_none,
        "lambda_values": lambda_values,
        "alpha_values": alpha_values,
        "tau_values": tau_values,
        "batch_indices": batch_indices,
        "scores": scores,
        "concatenated_vectors": concatenated_vectors,
        "distances": distances,
        "embedding_2d": embedding_2d,
        "pca_embedding_2d": pca_embedding_2d,
    }


class TestComputeEmbedding(unittest.TestCase):
    def test_mds_and_pca_are_exact_passthroughs(self):
        result = _build_result()
        np.testing.assert_array_equal(
            exploration.compute_embedding(result, "mds"), result["embedding_2d"]
        )
        np.testing.assert_array_equal(
            exploration.compute_embedding(result, "pca"), result["pca_embedding_2d"]
        )

    def test_tsne_and_isomap_return_finite_shape_on_tiny_fixture(self):
        result = _build_result()
        n = len(result["condition_names"])
        for method in ("tsne", "isomap"):
            embedding = exploration.compute_embedding(result, method)
            self.assertEqual(embedding.shape, (n, 2))
            self.assertTrue(np.isfinite(embedding).all())

    def test_lda_pads_second_column_with_zeros_when_two_levels(self):
        result = _build_result()  # every parameter has exactly 2 levels
        for method in ("lda_lambda", "lda_alpha", "lda_tau"):
            embedding = exploration.compute_embedding(result, method)
            n = len(result["condition_names"])
            self.assertEqual(embedding.shape, (n, 2))
            np.testing.assert_array_equal(embedding[:, 1], np.zeros(n))
            # The first column should not be constant across the two groups.
            self.assertGreater(np.ptp(embedding[:, 0]), 0.0)

    def test_lda_fit_excludes_none_but_transform_includes_it(self):
        result = _build_result()
        baseline = exploration.compute_embedding(result, "lda_lambda")

        mutated = {key: value.copy() for key, value in result.items()}
        none_index = int(np.flatnonzero(mutated["is_none"])[0])
        mutated["concatenated_vectors"][none_index] += 1000.0
        mutated_embedding = exploration.compute_embedding(mutated, "lda_lambda")

        transformed_mask = ~result["is_none"]
        np.testing.assert_array_equal(
            baseline[transformed_mask], mutated_embedding[transformed_mask]
        )
        self.assertEqual(mutated_embedding.shape, baseline.shape)

    def test_unsupported_method_raises(self):
        result = _build_result()
        with self.assertRaisesRegex(ValueError, "Unsupported method"):
            exploration.compute_embedding(result, "not-a-method")

    def test_lda_requires_at_least_two_levels(self):
        result = _build_result(lambdas=(0.1,))
        with self.assertRaisesRegex(ValueError, "at least two distinct levels"):
            exploration.compute_embedding(result, "lda_lambda")

    def test_diffusion_map_returns_finite_shape_on_tiny_fixture(self):
        result = _build_result()
        n = len(result["condition_names"])
        embedding = exploration.compute_embedding(result, "diffusion_map")
        self.assertEqual(embedding.shape, (n, 2))
        self.assertTrue(np.isfinite(embedding).all())


def _two_cluster_distances() -> np.ndarray:
    """4 points, two tight pairs far apart from each other -- a distance
    matrix diffusion maps should trivially be able to separate."""
    return np.array(
        [
            [0.0, 0.1, 10.0, 10.0],
            [0.1, 0.0, 10.0, 10.0],
            [10.0, 10.0, 0.0, 0.1],
            [10.0, 10.0, 0.1, 0.0],
        ]
    )


class TestDiffusionMap(unittest.TestCase):
    def test_first_coordinate_separates_two_well_separated_clusters(self):
        result = {"distances": _two_cluster_distances()}
        embedding = exploration.compute_embedding(result, "diffusion_map")
        first = embedding[:, 0]
        within_cluster_gap = abs(first[0] - first[1])
        across_cluster_gap = abs(first[0] - first[2])
        self.assertLess(within_cluster_gap, across_cluster_gap)
        # Same sign within a cluster, opposite sign across clusters.
        self.assertEqual(np.sign(first[0]), np.sign(first[1]))
        self.assertNotEqual(np.sign(first[0]), np.sign(first[2]))

    def test_median_scale_knob_changes_the_embedding(self):
        result = {"distances": _two_cluster_distances()}
        narrow = exploration.compute_embedding(
            result, "diffusion_map", diffusion_median_scale=0.1
        )
        wide = exploration.compute_embedding(
            result, "diffusion_map", diffusion_median_scale=10.0
        )
        self.assertFalse(np.allclose(narrow, wide))

    def test_non_positive_median_scale_raises(self):
        result = {"distances": _two_cluster_distances()}
        for bad_scale in (0.0, -1.0):
            with self.assertRaisesRegex(ValueError, "median_scale must be positive"):
                exploration.compute_embedding(
                    result, "diffusion_map", diffusion_median_scale=bad_scale
                )

    def test_all_identical_points_raises_on_undefined_bandwidth(self):
        result = {"distances": np.zeros((4, 4))}
        with self.assertRaisesRegex(ValueError, "median pairwise distance is zero"):
            exploration.compute_embedding(result, "diffusion_map")

    def test_fewer_than_two_non_trivial_eigenvectors_zero_pads(self):
        # Two points: only one non-trivial eigenvector exists, so the
        # second column must be zero-padded rather than raising.
        result = {"distances": np.array([[0.0, 1.0], [1.0, 0.0]])}
        embedding = exploration.compute_embedding(result, "diffusion_map")
        self.assertEqual(embedding.shape, (2, 2))
        self.assertTrue(np.isfinite(embedding).all())
        np.testing.assert_array_equal(embedding[:, 1], np.zeros(2))


class TestDimensionLabels(unittest.TestCase):
    def test_signal_mode_includes_parameter_suffix(self):
        result = _build_result(mode="signal", n_batches=2)
        labels = exploration.dimension_labels(result)
        self.assertEqual(labels.shape, (2 * 3,))
        self.assertEqual(labels[0], "batch0_rbf_sigma=0.5")
        self.assertEqual(labels[3], "batch1_rbf_sigma=0.5")

    def test_fixed_mode_has_no_parameter_suffix(self):
        result = _build_result(mode="fixed", n_batches=3)
        labels = exploration.dimension_labels(result)
        self.assertEqual(labels.shape, (3,))
        np.testing.assert_array_equal(labels, ["batch0", "batch1", "batch2"])


class TestParallelCoordinatesData(unittest.TestCase):
    def test_shapes_match_concatenated_vectors(self):
        result = _build_result()
        data = exploration.parallel_coordinates_data(result, "tau")
        n, d = result["concatenated_vectors"].shape
        self.assertEqual(data["values"].shape, (n, d))
        self.assertEqual(data["dimension_labels"].shape, (d,))
        self.assertEqual(data["color_values"].shape, (n,))
        self.assertEqual(data["condition_names"].shape, (n,))

    def test_include_none_false_drops_and_realigns(self):
        result = _build_result()
        data = exploration.parallel_coordinates_data(result, "tau", include_none=False)
        expected_n = int((~result["is_none"]).sum())
        self.assertEqual(data["values"].shape[0], expected_n)
        self.assertFalse(data["is_none"].any())
        np.testing.assert_array_equal(
            data["condition_names"], result["condition_names"][~result["is_none"]]
        )

    def test_unknown_parameter_raises(self):
        result = _build_result()
        with self.assertRaisesRegex(ValueError, "Unknown parameter"):
            exploration.parallel_coordinates_data(result, "sigma")


class TestReorderedDistanceMatrix(unittest.TestCase):
    def test_hand_computed_order_none_first_then_tau_then_lambda_tiebreak(self):
        condition_names = np.array(["none", "c1", "c2", "c3", "c4"])
        is_none = np.array([True, False, False, False, False])
        lambda_values = np.array([np.nan, 0.1, 0.01, 0.1, 0.1])
        alpha_values = np.array([np.nan, 0.5, 0.5, 0.25, 0.5])
        tau_values = np.array([np.nan, 0.5, 0.5, 0.1, 0.9])
        vectors = np.arange(5 * 3, dtype=float).reshape(5, 3)
        distances = squareform(pdist(vectors))
        result = {
            "condition_names": condition_names,
            "is_none": is_none,
            "lambda_values": lambda_values,
            "alpha_values": alpha_values,
            "tau_values": tau_values,
            "distances": distances,
        }

        reordered = exploration.reordered_distance_matrix(result, "tau")

        # none(0) first, then tau=0.1(3), then tau=0.5 tied pair broken by
        # ascending lambda (2 has lambda=0.01 < 1's lambda=0.1), then tau=0.9(4).
        expected_order = np.array([0, 3, 2, 1, 4])
        np.testing.assert_array_equal(reordered["order"], expected_order)
        np.testing.assert_array_equal(
            reordered["distances"], distances[np.ix_(expected_order, expected_order)]
        )
        np.testing.assert_array_equal(
            reordered["condition_names"], condition_names[expected_order]
        )

    def test_unknown_parameter_raises(self):
        result = _build_result()
        with self.assertRaisesRegex(ValueError, "Unknown parameter"):
            exploration.reordered_distance_matrix(result, "sigma")


class TestChannelMarkersAndSizes(unittest.TestCase):
    def test_channel_markers_assigns_one_symbol_per_level(self):
        result = _build_result(lambdas=(0.001, 0.1))
        markers = exploration.channel_markers(result, "lambda")
        self.assertEqual(set(markers), {0.001, 0.1})
        self.assertEqual(len(set(markers.values())), 2)

    def test_channel_markers_raises_past_nine_levels(self):
        result = _build_result(lambdas=tuple(float(i) for i in range(10)))
        with self.assertRaisesRegex(ValueError, "Too many lambda levels"):
            exploration.channel_markers(result, "lambda")

    def test_channel_colors_assigns_one_discrete_color_per_level(self):
        result = _build_result(taus=(0.1, 0.5))
        colors = exploration.channel_colors(result, "tau")
        self.assertEqual(set(colors), {0.1, 0.5})
        # Two distinct hex colors, not a shared/interpolated value.
        self.assertEqual(len(set(colors.values())), 2)
        for value in colors.values():
            self.assertTrue(value.startswith("#"))

    def test_channel_colors_raises_past_nine_levels(self):
        result = _build_result(taus=tuple(float(i) for i in range(10)))
        with self.assertRaisesRegex(ValueError, "Too many tau levels"):
            exploration.channel_colors(result, "tau")

    def test_channel_colors_unknown_parameter_raises(self):
        result = _build_result()
        with self.assertRaisesRegex(ValueError, "Unknown parameter"):
            exploration.channel_colors(result, "sigma")

    def test_channel_sizes_none_rows_get_fixed_size(self):
        result = _build_result(alphas=(0.25, 0.75))
        sizes = exploration.channel_sizes(
            result, "alpha", low=45.0, high=150.0, none_size=95.0
        )
        none_mask = result["is_none"]
        np.testing.assert_array_equal(sizes[none_mask], np.full(none_mask.sum(), 95.0))
        self.assertTrue(np.isin(sizes[~none_mask], [45.0, 150.0]).all())

    def test_channel_sizes_single_level_uses_midpoint(self):
        result = _build_result(alphas=(0.5,))
        sizes = exploration.channel_sizes(result, "alpha", low=45.0, high=150.0)
        transformed = sizes[~result["is_none"]]
        np.testing.assert_array_equal(transformed, np.full(transformed.shape, 97.5))


class TestHoverLabels(unittest.TestCase):
    def test_none_row_is_bare_condition_name(self):
        result = _build_result()
        labels = exploration.hover_labels(result)
        self.assertEqual(labels[0], "none")

    def test_transformed_row_includes_parameter_values(self):
        result = _build_result(lambdas=(0.1,), alphas=(0.5,), taus=(0.25,))
        labels = exploration.hover_labels(result)
        self.assertIn("lambda=0.1", labels[1])
        self.assertIn("alpha=0.5", labels[1])
        self.assertIn("tau=0.25", labels[1])


def _build_correlation_metric(
    *,
    parameter_values: tuple[float, ...],
    target_lambda: tuple[float, ...],
    score_columns: tuple[tuple[float, ...], ...],
    n_batches: int = 2,
) -> dict[str, np.ndarray]:
    """A minimal metric result for parameter_correlation_heatmap_data tests:
    one `none` row plus one transformed row per `target_lambda` entry, with
    `score_columns[j]` giving every transformed row's score at
    `parameter_values[j]` (identical across batches, so batch-mean equals
    `score_columns` exactly and expected correlations are easy to reason
    about by hand or by calling scipy directly)."""
    n_transformed = len(target_lambda)
    parameter_values = np.asarray(parameter_values, dtype=float)
    assert len(score_columns) == parameter_values.size
    is_none = np.array([True] + [False] * n_transformed)
    lambda_values = np.array([np.nan] + list(target_lambda))
    alpha_values = np.full(n_transformed + 1, 0.5)
    alpha_values[0] = np.nan
    tau_values = np.full(n_transformed + 1, 0.5)
    tau_values[0] = np.nan
    scores = np.zeros((n_transformed + 1, n_batches, parameter_values.size))
    for j, column in enumerate(score_columns):
        column = np.asarray(column, dtype=float)
        assert column.shape == (n_transformed,)
        scores[1:, :, j] = column[:, None]
    return {
        "is_none": is_none,
        "lambda_values": lambda_values,
        "alpha_values": alpha_values,
        "tau_values": tau_values,
        "scores": scores,
        "parameter_values": parameter_values,
        "condition_names": np.asarray(
            ["none"] + [f"c{i}" for i in range(n_transformed)]
        ),
    }


class TestParameterCorrelationHeatmapData(unittest.TestCase):
    """target lambda = [1, 2, 3, 4] throughout.

    metric_a (P=3, parameter_values=[0.1, 1.0, 10.0]):
      col0 = [1, 2, 3, 4]   -- exact +1 correlation (pearson and spearman)
      col1 = [4, 3, 2, 1]   -- exact -1 correlation (pearson and spearman)
      col2 = [1, 4, 9, 16]  -- monotonic but non-linear: spearman exactly 1,
                               pearson strictly less than 1 (discriminates
                               the two correlation methods).

    metric_b (P=2, parameter_values=[2.0, 5.0]):
      col0 = [1, 2, 3, 4]   -- exact +1 correlation
      col1 = [8, 6, 4, 2]   -- exact linear negative slope -> exact -1
                               correlation (not a "5 - col0" mirror, so its
                               two-point AUC is not degenerately constant).
      Its own grid (P=2) is shorter than metric_a's (P=3), so its row
      exercises interpolation onto the common 3-point axis.

    metric_c (fixed, P=1, parameter_values=[0.0]):
      col0 = [1, 2, 3, 4]  -- the target itself, exact +1 correlation.
    """

    def setUp(self):
        self.target = (1.0, 2.0, 3.0, 4.0)
        self.metric_a = _build_correlation_metric(
            parameter_values=(0.1, 1.0, 10.0),
            target_lambda=self.target,
            score_columns=((1, 2, 3, 4), (4, 3, 2, 1), (1, 4, 9, 16)),
        )
        self.metric_b = _build_correlation_metric(
            parameter_values=(2.0, 5.0),
            target_lambda=self.target,
            score_columns=((1, 2, 3, 4), (8, 6, 4, 2)),
        )
        self.metric_c = _build_correlation_metric(
            parameter_values=(0.0,),
            target_lambda=self.target,
            score_columns=((1, 2, 3, 4),),
        )
        self.results = {
            "metric_a": self.metric_a,
            "metric_b": self.metric_b,
            "metric_c": self.metric_c,
        }

    def _call(self, method):
        return exploration.parameter_correlation_heatmap_data(
            self.results,
            "lambda",
            method,
            signal_metrics=("metric_a", "metric_b"),
            fixed_metrics=("metric_c",),
        )

    def test_row_labels_and_default_common_points(self):
        data = self._call("pearson")
        np.testing.assert_array_equal(
            data["rows"],
            ["metric_a", "metric_a (AUC)", "metric_b", "metric_b (AUC)", "metric_c"],
        )
        # common_points defaults to the largest present signal metric's own
        # grid size: metric_a has 3, metric_b has 2 -> 3.
        self.assertEqual(data["matrix"].shape, (5, 3))
        np.testing.assert_allclose(data["x"], [0.0, 0.5, 1.0])

    def test_metric_a_row_matches_pearson_of_each_own_column(self):
        data = self._call("pearson")
        target = np.asarray(self.target)
        expected = [
            pearsonr((1, 2, 3, 4), target).statistic,
            pearsonr((4, 3, 2, 1), target).statistic,
            pearsonr((1, 4, 9, 16), target).statistic,
        ]
        np.testing.assert_allclose(data["matrix"][0], expected, atol=1e-10)
        self.assertAlmostEqual(expected[0], 1.0)
        self.assertAlmostEqual(expected[1], -1.0)
        self.assertLess(expected[2], 1.0)

    def test_pearson_and_spearman_disagree_on_the_nonlinear_column(self):
        pearson_data = self._call("pearson")
        spearman_data = self._call("spearman")
        # col2 = target ** 2: monotonic (spearman exactly 1) but not linear
        # (pearson strictly less than 1) -- confirms the method actually
        # dispatches to a different scipy function, not just a relabeling.
        self.assertAlmostEqual(spearman_data["matrix"][0, 2], 1.0)
        self.assertLess(pearson_data["matrix"][0, 2], 1.0)

    def test_metric_b_row_is_linearly_interpolated_onto_common_axis(self):
        data = self._call("pearson")
        # metric_b's own curve is [+1, -1] at own relative positions [0, 1];
        # interpolated onto common_x = [0, 0.5, 1] this is exactly [1, 0, -1].
        np.testing.assert_allclose(data["matrix"][2], [1.0, 0.0, -1.0], atol=1e-10)

    def test_fixed_metric_row_is_constant_across_every_column(self):
        data = self._call("pearson")
        np.testing.assert_allclose(data["matrix"][4], [1.0, 1.0, 1.0], atol=1e-10)

    def test_auc_row_reuses_aggregate_signal_auc(self):
        data = self._call("pearson")
        batch_mean = np.array(
            [[1, 4, 1], [2, 3, 4], [3, 2, 9], [4, 1, 16]], dtype=float
        )
        auc = aggregate_signal_auc(self.metric_a["parameter_values"], batch_mean)
        expected = pearsonr(auc, np.asarray(self.target)).statistic
        self.assertAlmostEqual(data["matrix"][1, 0], expected)
        # An AUC row is constant across every column, same as a fixed row.
        np.testing.assert_allclose(data["matrix"][1], np.full(3, expected))

    def test_missing_metric_is_silently_skipped(self):
        data = exploration.parameter_correlation_heatmap_data(
            self.results,
            "lambda",
            "pearson",
            signal_metrics=("metric_a", "not_present"),
            fixed_metrics=("metric_c", "also_not_present"),
        )
        np.testing.assert_array_equal(
            data["rows"], ["metric_a", "metric_a (AUC)", "metric_c"]
        )

    def test_common_points_override(self):
        data = exploration.parameter_correlation_heatmap_data(
            self.results,
            "lambda",
            "pearson",
            signal_metrics=("metric_a", "metric_b"),
            fixed_metrics=("metric_c",),
            common_points=6,
        )
        self.assertEqual(data["matrix"].shape, (5, 6))
        self.assertEqual(data["x"].size, 6)

    def test_unknown_parameter_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown parameter"):
            exploration.parameter_correlation_heatmap_data(
                self.results, "sigma", "pearson"
            )

    def test_unknown_correlation_method_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown correlation method"):
            exploration.parameter_correlation_heatmap_data(
                self.results, "lambda", "kendall"
            )


_FAST_MDS_CONFIG = {
    "mds": {"random_state": 0, "n_init": 1, "max_iter": 50, "eps": 1e-3}
}


class TestConcatenateDatasetResults(unittest.TestCase):
    def setUp(self):
        # Same condition/sweep grid (default _build_result args), different
        # batch counts and different scores (different seeds) -- like two
        # real datasets analyzed under the same run configuration.
        self.result_a = _build_result(n_batches=2, seed=0)
        self.result_b = _build_result(n_batches=3, seed=1)
        self.results_by_dataset = {"dsA": self.result_a, "dsB": self.result_b}

    def test_concatenates_scores_along_the_batch_axis_in_dataset_order(self):
        merged = exploration.concatenate_dataset_results(
            self.results_by_dataset, _FAST_MDS_CONFIG
        )
        n_conditions, p = (
            self.result_a["scores"].shape[0],
            self.result_a["scores"].shape[2],
        )
        self.assertEqual(merged["scores"].shape, (n_conditions, 5, p))
        np.testing.assert_array_equal(
            merged["scores"][:, :2, :], self.result_a["scores"]
        )
        np.testing.assert_array_equal(
            merged["scores"][:, 2:, :], self.result_b["scores"]
        )

    def test_batch_indices_are_renumbered_and_batch_datasets_records_provenance(self):
        merged = exploration.concatenate_dataset_results(
            self.results_by_dataset, _FAST_MDS_CONFIG
        )
        np.testing.assert_array_equal(merged["batch_indices"], np.arange(5))
        np.testing.assert_array_equal(
            merged["batch_datasets"], ["dsA", "dsA", "dsB", "dsB", "dsB"]
        )

    def test_condition_grid_fields_pass_through_from_the_first_dataset(self):
        merged = exploration.concatenate_dataset_results(
            self.results_by_dataset, _FAST_MDS_CONFIG
        )
        for key in (
            "condition_names",
            "is_none",
            "lambda_values",
            "parameter_name",
            "parameter_values",
        ):
            np.testing.assert_array_equal(merged[key], self.result_a[key])

    def test_distances_embedding_and_pca_are_refit_on_the_wider_data(self):
        merged = exploration.concatenate_dataset_results(
            self.results_by_dataset, _FAST_MDS_CONFIG
        )
        n_conditions = self.result_a["scores"].shape[0]
        self.assertEqual(merged["concatenated_vectors"].shape[0], n_conditions)
        self.assertEqual(
            merged["concatenated_vectors"].shape[1],
            5 * self.result_a["scores"].shape[2],
        )
        self.assertEqual(merged["distances"].shape, (n_conditions, n_conditions))
        np.testing.assert_allclose(np.diag(merged["distances"]), 0.0, atol=1e-8)
        self.assertTrue(np.isfinite(merged["distances"]).all())
        self.assertEqual(merged["embedding_2d"].shape, (n_conditions, 2))
        self.assertTrue(np.isfinite(merged["embedding_2d"]).all())
        self.assertEqual(merged["pca_embedding_2d"].shape, (n_conditions, 2))
        self.assertTrue(np.isfinite(merged["pca_embedding_2d"]).all())

    def test_merged_result_composes_with_the_rest_of_the_module(self):
        merged = exploration.concatenate_dataset_results(
            self.results_by_dataset, _FAST_MDS_CONFIG
        )
        n_conditions = self.result_a["scores"].shape[0]
        np.testing.assert_array_equal(
            exploration.compute_embedding(merged, "mds"), merged["embedding_2d"]
        )
        labels = exploration.dimension_labels(merged)
        self.assertEqual(labels.shape, (5 * self.result_a["scores"].shape[2],))
        hover = exploration.hover_labels(merged)
        self.assertEqual(hover.shape, (n_conditions,))

    def test_mismatched_condition_grid_raises(self):
        # A different lambda grid changes condition_names too (condition
        # names embed their lambda value), so that's the first invariant
        # key the check reaches -- confirming a real grid mismatch is
        # caught, not this specific key.
        mismatched = _build_result(lambdas=(0.001, 0.5))  # differs from default (0.1)
        with self.assertRaisesRegex(ValueError, "'dsB'.*condition_names"):
            exploration.concatenate_dataset_results(
                {"dsA": self.result_a, "dsB": mismatched}, _FAST_MDS_CONFIG
            )

    def test_mismatched_metric_id_raises(self):
        different_metric = _build_result()
        different_metric["metric_id"] = np.array("other_metric")
        with self.assertRaisesRegex(ValueError, "metric_id"):
            exploration.concatenate_dataset_results(
                {"dsA": self.result_a, "dsB": different_metric}, _FAST_MDS_CONFIG
            )

    def test_single_dataset_still_refits_and_records_its_own_provenance(self):
        merged = exploration.concatenate_dataset_results(
            {"dsA": self.result_a}, _FAST_MDS_CONFIG
        )
        np.testing.assert_array_equal(merged["batch_datasets"], ["dsA", "dsA"])
        np.testing.assert_array_equal(merged["scores"], self.result_a["scores"])

    def test_empty_results_raises(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            exploration.concatenate_dataset_results({}, _FAST_MDS_CONFIG)


if __name__ == "__main__":
    unittest.main()
