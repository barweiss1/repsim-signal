"""Acceptance checks for the shared paper-figure style layer.

`paper_style` is what makes the two MDS experiments' paper figures look like
one another; before it existed each experiment reimplemented these helpers
and the two drifted apart. These tests pin the parts that are pure functions
-- band indexing, span geometry, channel scaling, the magma ramp, rcParams --
so the layer stops being verifiable only by rendering a PDF and looking at it.

Rendering itself is still eyeball-verified; what is checked here is the
arithmetic that decides where things land.
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manifold_repsim.experiments import paper_style


class TestRankIndex(unittest.TestCase):
    """`rank_index` keeps scatter points on their colorbar's own bands."""

    def test_each_level_maps_to_its_own_index(self):
        levels = np.array([0.0, 0.1, 0.2, 0.3])
        indices = paper_style.rank_index(levels, levels)
        np.testing.assert_array_equal(indices, np.arange(4))

    def test_repeated_values_map_to_the_same_index(self):
        levels = np.array([0.0, 0.5, 1.0])
        values = np.array([1.0, 0.0, 1.0, 0.5])
        np.testing.assert_array_equal(
            paper_style.rank_index(values, levels), [2, 0, 2, 1]
        )

    def test_float_round_trip_noise_still_matches(self):
        # Values reaching the paper module have round-tripped through NPZ
        # artifacts, so exact equality is not available.
        levels = np.linspace(0.0, 0.4, 5)
        noisy = levels + 1e-12
        np.testing.assert_array_equal(
            paper_style.rank_index(noisy, levels), np.arange(5)
        )

    def test_unknown_value_raises_instead_of_snapping(self):
        levels = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            paper_style.rank_index(np.array([0.5]), levels)

    def test_shape_is_preserved(self):
        levels = np.array([0.0, 1.0])
        values = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(paper_style.rank_index(values, levels).shape, (2, 2))

    def test_band_indices_stay_inside_the_colorbar_band_count(self):
        # The colorbar is built as BoundaryNorm(arange(n + 1), n), so valid
        # band indices are exactly 0..n-1; a point outside that range would
        # be clipped to a band whose label does not describe it.
        levels = np.linspace(0.0, 0.2, 5)
        indices = paper_style.rank_index(levels, levels)
        self.assertTrue((indices >= 0).all())
        self.assertTrue((indices < len(levels)).all())


class TestSpanGeometry(unittest.TestCase):
    """Legend columns share one vertical span so they render equal-height."""

    def test_span_is_bottom_anchored_and_scaled(self):
        # Anchored at the caller's bottom so the legend columns start level
        # with the bottom of the plot axes they annotate.
        bottom, top = paper_style.shrunk_span(0.0, 1.0, shrink=0.5)
        self.assertAlmostEqual(bottom, 0.0)
        self.assertAlmostEqual(top, 0.5)

    def test_full_shrink_is_a_no_op(self):
        bottom, top = paper_style.shrunk_span(0.2, 0.8, shrink=1.0)
        self.assertAlmostEqual(bottom, 0.2)
        self.assertAlmostEqual(top, 0.8)

    def test_default_shrink_is_the_shared_fraction(self):
        bottom, top = paper_style.shrunk_span(0.0, 1.0)
        self.assertAlmostEqual(top - bottom, paper_style.LEGEND_HEIGHT_FRACTION)

    def test_panel_metrics_are_physical_not_fractional(self):
        # Panels are sized in inches so a wider figure does not get wider
        # legend columns.
        narrow = plt.figure(figsize=(10, 4))
        wide = plt.figure(figsize=(20, 4))
        try:
            narrow_width, _ = paper_style.legend_panel_metrics(narrow)
            wide_width, _ = paper_style.legend_panel_metrics(wide)
            self.assertAlmostEqual(narrow_width, wide_width * 2)
            self.assertAlmostEqual(wide_width * 20, paper_style.LEGEND_PANEL_WIDTH_IN)
        finally:
            plt.close(narrow)
            plt.close(wide)


class TestChannelScaling(unittest.TestCase):
    """`scaled_value`/`rescale_channel` map levels onto a channel's range."""

    def test_endpoints_map_to_the_target_endpoints(self):
        levels = np.array([0.0, 1.0, 2.0])
        self.assertAlmostEqual(
            paper_style.scaled_value(0.0, levels, (10.0, 20.0)), 10.0
        )
        self.assertAlmostEqual(
            paper_style.scaled_value(2.0, levels, (10.0, 20.0)), 20.0
        )
        self.assertAlmostEqual(
            paper_style.scaled_value(1.0, levels, (10.0, 20.0)), 15.0
        )

    def test_single_level_uses_the_range_floor(self):
        levels = np.array([3.0])
        self.assertAlmostEqual(
            paper_style.scaled_value(3.0, levels, (10.0, 20.0)), 10.0
        )

    def test_rescale_channel_matches_scalar_scaling_and_shape(self):
        levels = np.array([0.0, 0.5, 1.0])
        values = np.array([[0.0, 1.0], [0.5, 0.0]])
        scaled = paper_style.rescale_channel(values, levels, (2.0, 4.0))
        self.assertEqual(scaled.shape, (2, 2))
        np.testing.assert_allclose(scaled, [[2.0, 4.0], [3.0, 2.0]])

    def test_opacity_swatches_vary_only_in_alpha(self):
        swatches = paper_style.opacity_swatches([0.3, 1.0])
        self.assertEqual(len(swatches), 2)
        for swatch in swatches:
            self.assertEqual(swatch[:3], (paper_style.OPACITY_SWATCH_GRAY,) * 3)
        self.assertAlmostEqual(swatches[0][3], 0.3)
        self.assertAlmostEqual(swatches[1][3], 1.0)


class TestMagmaRamp(unittest.TestCase):
    """Both experiments sample curve/point colors from one magma sub-range."""

    def test_ramp_returns_one_distinct_rgba_row_per_level(self):
        colors = paper_style.magma_ramp(5)
        self.assertEqual(colors.shape, (5, 4))
        self.assertEqual(len({tuple(color) for color in colors}), 5)

    def test_ramp_stays_inside_the_shared_sub_range(self):
        low, high = paper_style.MAGMA_RANGE
        expected_ends = plt.cm.magma([low, high])
        colors = paper_style.magma_ramp(7)
        np.testing.assert_allclose(colors[0], expected_ends[0])
        np.testing.assert_allclose(colors[-1], expected_ends[1])

    def test_explicit_positions_reproduce_the_ramp_endpoints(self):
        # glocal samples lambda at log positions through `magma_at`; it must
        # agree with `magma_ramp` wherever the positions coincide.
        np.testing.assert_allclose(
            paper_style.magma_at([0.0, 1.0]), paper_style.magma_ramp(2)
        )

    def test_evenly_spaced_positions_reproduce_the_full_ramp(self):
        np.testing.assert_allclose(
            paper_style.magma_at(np.linspace(0.0, 1.0, 4)),
            paper_style.magma_ramp(4),
        )


class TestPaperStyleRcParams(unittest.TestCase):
    """The shared style is applied wholesale, not per-figure by hand."""

    def setUp(self):
        self._saved = dict(plt.rcParams)
        self.addCleanup(lambda: plt.rcParams.update(self._saved))

    def test_apply_sets_every_declared_parameter(self):
        plt.rcParams.update({"font.size": 5, "axes.titlesize": 5})
        paper_style.apply_paper_style()
        for key, value in paper_style.PAPER_RCPARAMS.items():
            stored = plt.rcParams[key]
            # matplotlib normalizes a few keys (font.family) into a list.
            if isinstance(stored, list) and not isinstance(value, list):
                self.assertEqual(stored, [value], msg=key)
            else:
                self.assertEqual(stored, value, msg=key)

    def test_legend_title_fontsize_follows_the_axes_title_size(self):
        paper_style.apply_paper_style()
        self.assertEqual(
            paper_style.legend_title_fontsize(),
            paper_style.PAPER_RCPARAMS["axes.titlesize"],
        )


class TestCornerLegendWidening(unittest.TestCase):
    """The lower-left legend gets its corner by moving the data, not the legend."""

    def test_linear_axis_widens_left_only(self):
        figure, axis = plt.subplots()
        try:
            axis.set_xlim(0.0, 10.0)
            paper_style.widen_for_corner_legend(axis, logscale=False)
            left, right = axis.get_xlim()
            self.assertAlmostEqual(right, 10.0)
            self.assertAlmostEqual(
                left, -10.0 * paper_style.SIGNAL_LEGEND_LINEAR_WIDENING
            )
        finally:
            plt.close(figure)

    def test_log_axis_divides_the_left_limit(self):
        figure, axis = plt.subplots()
        try:
            axis.set_xscale("log")
            axis.set_xlim(0.1, 10.0)
            paper_style.widen_for_corner_legend(axis, logscale=True)
            left, right = axis.get_xlim()
            self.assertAlmostEqual(right, 10.0)
            self.assertAlmostEqual(left, 0.1 / paper_style.SIGNAL_LEGEND_LOG_WIDENING)
        finally:
            plt.close(figure)


class TestLegendPanels(unittest.TestCase):
    """The two legend column kinds render with one band/marker per level."""

    def test_colorbar_labels_every_band_once_at_its_center(self):
        figure = plt.figure(figsize=(4, 4))
        try:
            axis = paper_style.discrete_colorbar_panel(
                figure,
                (0.8, 0.1, 0.05, 0.8),
                paper_style.magma_ramp(4),
                ["a", "b", "c", "d"],
                "title",
            )
            np.testing.assert_allclose(axis.get_yticks(), [0.5, 1.5, 2.5, 3.5])
            self.assertEqual(
                [label.get_text() for label in axis.get_yticklabels()],
                ["a", "b", "c", "d"],
            )
        finally:
            plt.close(figure)

    def test_size_panel_positions_markers_by_rank(self):
        figure = plt.figure(figsize=(4, 4))
        try:
            axis = paper_style.size_legend_panel(
                figure,
                (0.8, 0.1, 0.05, 0.8),
                [45.0, 100.0, 150.0],
                ["0", "2", "4"],
                "title",
            )
            np.testing.assert_allclose(axis.get_yticks(), [0, 1, 2])
            self.assertEqual(
                [label.get_text() for label in axis.get_yticklabels()],
                ["0", "2", "4"],
            )
        finally:
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
