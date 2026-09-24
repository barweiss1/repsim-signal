"""The one boxplot implementation every measure figure draws through."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from manifold_repsim.resi.analysis import boxplots
from manifold_repsim.resi.analysis.config import measure_label
from manifold_repsim.resi.analysis.boxplots import (
    BOXEN_BAND_TINTS,
    BOXEN_BAND_WIDTHS,
    MEAN_FACE_COLOR,
    MEAN_MARKER,
    MEDIAN_COLOR,
    QUANTILE_MARKER,
    MIN_HEIGHT,
    PANEL_WIDTH,
    add_legend,
    category_legend,
    draw_boxen,
    draw_boxes,
    draw_series,
    grouped_panel,
    measure_panel,
    panel_size,
)
from manifold_repsim.resi.analysis.config import (
    BOXEN_BANDS,
    boxen_band_label,
    boxen_percentiles,
)


def _marker_count(ax) -> int:
    return sum(1 for line in ax.lines if line.get_marker() == QUANTILE_MARKER)


def _frame():
    return pd.DataFrame(
        {
            "metric": ["CKA"] * 3 + ["SVCCA"] * 3,
            "value": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
        }
    )


def _bands(ax):
    """The boxen rectangles on an axis, as (left, right, height) triples."""
    from matplotlib.patches import Rectangle

    return [
        (patch.get_x(), patch.get_x() + patch.get_width(), patch.get_height())
        for patch in ax.patches
        if isinstance(patch, Rectangle)
    ]


class TestBoxen(unittest.TestCase):
    """The letter-value drawing: nested bands at 10/30/50/70/90."""

    def test_the_documented_percentiles_are_the_ones_drawn(self):
        """The vocabulary and the geometry must not drift apart."""
        self.assertEqual(boxen_percentiles(), (0, 10, 30, 50, 70, 90, 100))

    def test_whiskers_reach_the_true_extremes(self):
        """A boxen must not clip the tails the way whisker percentiles do."""
        points = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        fig, ax = plt.subplots()
        draw_boxen(ax, [points], positions=[0], colors=["#123456"])
        reach = [x for line in ax.lines for x in line.get_xdata()]
        self.assertAlmostEqual(min(reach), float(points.min()))
        self.assertAlmostEqual(max(reach), float(points.max()))
        plt.close(fig)

    def test_the_extremes_are_whiskers_and_not_a_filled_band(self):
        """A band would give two single observations the weight of an interval."""
        points = np.arange(101, dtype=float)
        fig, ax = plt.subplots()
        draw_boxen(ax, [points], positions=[0], colors=["#123456"])
        # Every filled band stops at the 10th/90th; only lines go further.
        self.assertAlmostEqual(min(l for l, _r, _h in _bands(ax)), 10.0)
        self.assertAlmostEqual(max(r for _l, r, _h in _bands(ax)), 90.0)
        plt.close(fig)

    def test_the_whiskers_are_capped_at_both_ends(self):
        points = np.arange(101, dtype=float)
        fig, ax = plt.subplots()
        draw_boxen(ax, [points], positions=[0], colors=["#123456"])
        caps = [
            line
            for line in ax.lines
            if len(set(line.get_xdata())) == 1 and line.get_color() != MEDIAN_COLOR
        ]
        self.assertEqual(
            sorted(round(line.get_xdata()[0], 6) for line in caps), [0.0, 100.0]
        )
        plt.close(fig)

    def test_band_labels_name_the_percentiles_they_span(self):
        self.assertEqual(boxen_band_label(0.1, 0.9), "10-90 pct.")
        self.assertEqual(boxen_band_label(0.3, 0.7), "30-70 pct.")

    def test_bands_land_on_their_percentiles(self):
        points = np.arange(101, dtype=float)
        fig, ax = plt.subplots()
        draw_boxen(ax, [points], positions=[0], colors=["#123456"])
        bands = _bands(ax)
        self.assertEqual(len(bands), len(BOXEN_BANDS))
        for (left, right, _height), (low, high) in zip(bands, BOXEN_BANDS):
            self.assertAlmostEqual(left, low * 100)
            self.assertAlmostEqual(right, high * 100)
        plt.close(fig)

    def test_outer_bands_are_narrower_than_inner_ones(self):
        """The taper is what says 'this is the tail' without reading a legend."""
        fig, ax = plt.subplots()
        draw_boxen(ax, [np.arange(101, dtype=float)], positions=[0], colors=["#123456"])
        heights = [height for _left, _right, height in _bands(ax)]
        self.assertEqual(heights, sorted(heights))
        self.assertEqual(len(set(heights)), len(BOXEN_BAND_WIDTHS))
        plt.close(fig)

    def test_bands_are_opaque_rather_than_translucent(self):
        """Alpha would composite with the gridlines and vary by what is behind."""
        fig, ax = plt.subplots()
        draw_boxen(ax, [np.arange(101, dtype=float)], positions=[0], colors=["#1f77b4"])
        for patch in ax.patches:
            self.assertIn(patch.get_alpha(), (None, 1.0))
            self.assertEqual(patch.get_facecolor()[3], 1.0)
        plt.close(fig)

    def test_bands_lighten_outward_and_the_innermost_keeps_its_colour(self):
        """The ramp is a tint toward white, seaborn-style, not a transparency."""
        from matplotlib.colors import to_rgb

        base = "#1f77b4"
        fig, ax = plt.subplots()
        draw_boxen(ax, [np.arange(101, dtype=float)], positions=[0], colors=[base])
        # Patches are added outermost first, so lightness must fall along them.
        lightness = [sum(patch.get_facecolor()[:3]) for patch in ax.patches]
        self.assertEqual(lightness, sorted(lightness, reverse=True))
        self.assertEqual(len(set(lightness)), len(BOXEN_BAND_TINTS))
        innermost = ax.patches[-1].get_facecolor()[:3]
        for drawn, expected in zip(innermost, to_rgb(base)):
            self.assertAlmostEqual(drawn, expected)
        plt.close(fig)

    def test_the_median_is_drawn_as_a_black_line_on_top(self):
        fig, ax = plt.subplots()
        draw_boxen(ax, [np.arange(101, dtype=float)], positions=[0], colors=["#123456"])
        lines = [line for line in ax.lines if line.get_color() == MEDIAN_COLOR]
        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(lines[0].get_xdata()[0], 50.0)
        plt.close(fig)

    def test_an_empty_series_draws_nothing_rather_than_raising(self):
        fig, ax = plt.subplots()
        draw_boxen(ax, [np.array([])], positions=[0], colors=["#123456"])
        self.assertEqual(_bands(ax), [])
        plt.close(fig)

    def test_draw_series_defaults_to_the_box_and_never_mixes_the_two(self):
        """`box` must stay byte-for-byte the old drawing: no bands, real boxes."""
        for style in (None, "box"):
            fig, ax = plt.subplots()
            draw_series(
                ax,
                [np.arange(101, dtype=float)],
                positions=[0],
                colors=["#123456"],
                box_style=style,
            )
            self.assertEqual(_bands(ax), [])
            plt.close(fig)

        fig, ax = plt.subplots()
        draw_series(
            ax,
            [np.arange(101, dtype=float)],
            positions=[0],
            colors=["#123456"],
            box_style="boxen",
        )
        self.assertEqual(len(_bands(ax)), len(BOXEN_BANDS))
        plt.close(fig)

    def test_the_style_changes_the_drawing_and_not_the_ordering(self):
        """A rendering choice that reordered the axis would be a data claim."""
        frame = _frame()
        orders = []
        for style in ("box", "boxen"):
            fig, ax = plt.subplots()
            orders.append(
                measure_panel(
                    ax,
                    frame,
                    "value",
                    higher_is_better=False,
                    rank_sort="quantile90",
                    box_style=style,
                )
            )
            plt.close(fig)
        self.assertEqual(orders[0], orders[1])

    def test_the_quantile_marker_is_dropped_under_the_boxen(self):
        """The 10-90 band already ends there; a diamond would restate it."""
        frame = pd.DataFrame({"metric": ["CKA"] * 11, "value": list(range(11))})
        fig, ax = plt.subplots()
        measure_panel(
            ax,
            frame,
            "value",
            higher_is_better=False,
            rank_sort="quantile90",
            box_style="boxen",
        )
        self.assertEqual(_marker_count(ax), 0)
        plt.close(fig)

    def test_the_sorted_on_number_is_still_on_the_figure_as_a_band_edge(self):
        """Dropping the marker must not drop the statistic it stood for."""
        frame = pd.DataFrame({"metric": ["CKA"] * 11, "value": list(range(11))})
        fig, ax = plt.subplots()
        measure_panel(
            ax,
            frame,
            "value",
            higher_is_better=False,
            rank_sort="quantile90",
            box_style="boxen",
        )
        edges = [right for _left, right, _height in _bands(ax)]
        self.assertIn(9.0, [round(edge, 6) for edge in edges])
        plt.close(fig)

    def test_the_legend_names_the_bands_only_under_boxen(self):
        fig, ax = plt.subplots()
        drawn = measure_panel(ax, _frame(), "value", higher_is_better=False)
        category_legend(fig, drawn)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertNotIn("10-90 pct.", labels)
        plt.close(fig)

        fig, ax = plt.subplots()
        drawn = measure_panel(
            ax, _frame(), "value", higher_is_better=False, box_style="boxen"
        )
        category_legend(fig, drawn, box_style="boxen")
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertEqual(labels[-3:], ["30-70 pct.", "10-90 pct.", "min-max"])
        plt.close(fig)

    def test_the_legend_names_the_bands_and_not_a_marker_it_did_not_draw(self):
        fig, ax = plt.subplots()
        drawn = measure_panel(
            ax,
            _frame(),
            "value",
            higher_is_better=False,
            rank_sort="quantile90",
            box_style="boxen",
        )
        category_legend(fig, drawn, rank_sort="quantile90", box_style="boxen")
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertEqual(labels[-3:], ["30-70 pct.", "10-90 pct.", "min-max"])
        self.assertNotIn("90th pct.", labels)
        plt.close(fig)


class TestGeometry(unittest.TestCase):
    def test_width_is_fixed_and_height_grows_with_the_measure_count(self):
        """A figure must not get wider just because it has more measures."""
        small = panel_size(4)
        large = panel_size(20)
        self.assertEqual(small[0], large[0])
        self.assertGreater(large[1], small[1])

    def test_every_figure_is_taller_than_wide_at_a_realistic_field_size(self):
        width, height = panel_size(16)
        self.assertLess(width, height)

    def test_a_short_figure_does_not_collapse(self):
        self.assertEqual(panel_size(1)[1], MIN_HEIGHT)

    def test_panels_widen_the_figure_but_not_the_rows(self):
        one = panel_size(16)
        three = panel_size(16, panels=3)
        self.assertEqual(three[0], PANEL_WIDTH * 3)
        self.assertEqual(three[1], one[1])


class TestBoxStyle(unittest.TestCase):
    def _artists(self):
        figure, ax = plt.subplots()
        artists = draw_boxes(
            ax, [np.array([0.1, 0.5, 0.9])], positions=[0], colors=["#123456"]
        )
        plt.close(figure)
        return artists

    def test_medians_are_black(self):
        """They are the number read off the figure; grey competes with the fill."""
        for median in self._artists()["medians"]:
            self.assertEqual(median.get_color(), MEDIAN_COLOR)

    def test_the_requested_face_colour_is_applied(self):
        boxes = self._artists()["boxes"]
        self.assertEqual(matplotlib.colors.to_hex(boxes[0].get_facecolor()), "#123456")

    def test_outliers_are_not_drawn(self):
        figure, ax = plt.subplots()
        artists = draw_boxes(
            ax,
            [np.array([0.0] * 20 + [99.0])],
            positions=[0],
            colors=["#123456"],
        )
        plt.close(figure)
        self.assertEqual(len(artists["fliers"]), 0)


class TestMeasurePanel(unittest.TestCase):
    def _labels(self, *, higher_is_better):
        figure, ax = plt.subplots()
        measure_panel(ax, _frame(), "value", higher_is_better=higher_is_better)
        labels = [text.get_text() for text in ax.get_yticklabels()]
        plt.close(figure)
        return labels

    def test_higher_is_better_puts_the_largest_median_on_top(self):
        self.assertEqual(self._labels(higher_is_better=True)[-1], "SVCCA")

    def test_lower_is_better_puts_the_smallest_median_on_top(self):
        labels = self._labels(higher_is_better=False)
        self.assertEqual(labels[-1], measure_label("CKA"))

    def test_labels_are_the_published_abbreviations(self):
        frame = _frame().replace({"CKA": "SecondOrderCosineSimilarity"})
        figure, ax = plt.subplots()
        measure_panel(ax, frame, "value", higher_is_better=True)
        labels = [text.get_text() for text in ax.get_yticklabels()]
        plt.close(figure)
        self.assertIn("2nd-Cos", labels)

    def test_a_shared_order_is_honoured_rather_than_recomputed(self):
        """Panels that share an ordering keep a measure on the same row."""
        figure, ax = plt.subplots()
        measure_panel(
            ax, _frame(), "value", higher_is_better=True, order=["SVCCA", "CKA"]
        )
        labels = [text.get_text() for text in ax.get_yticklabels()]
        plt.close(figure)
        self.assertEqual(labels, ["SVCCA", measure_label("CKA")])

    def test_a_measure_with_no_data_is_not_drawn_but_keeps_its_row(self):
        frame = _frame()
        frame.loc[frame["metric"].eq("CKA"), "value"] = np.nan
        figure, ax = plt.subplots()
        drawn = measure_panel(ax, frame, "value", higher_is_better=True)
        plt.close(figure)
        self.assertEqual(drawn, ["SVCCA"])

    def test_an_empty_panel_draws_nothing(self):
        figure, ax = plt.subplots()
        drawn = measure_panel(ax, _frame().iloc[0:0], "value", higher_is_better=True)
        plt.close(figure)
        self.assertEqual(drawn, [])

    def test_a_display_frame_column_name_is_supported(self):
        """The recreated tables carry abbreviations under `Sim Meas.`."""
        frame = _frame().rename(columns={"metric": "Sim Meas."})
        figure, ax = plt.subplots()
        drawn = measure_panel(
            ax, frame, "value", higher_is_better=True, metric_column="Sim Meas."
        )
        plt.close(figure)
        self.assertEqual(set(drawn), {"CKA", "SVCCA"})


class TestLegend(unittest.TestCase):
    def test_the_legend_is_frameless_and_outside_the_axes_on_the_right(self):
        """Anchored at x >= 1 so it widens the figure instead of the plot area."""
        figure, _ax = plt.subplots()
        add_legend(figure, ["Neighbors"], ["#0173b2"])
        legend = figure.legends[0]
        self.assertFalse(legend.get_frame_on())
        self.assertGreaterEqual(legend.get_bbox_to_anchor().x0, 1.0)
        plt.close(figure)

    def test_the_legend_is_a_single_column(self):
        figure, _ax = plt.subplots()
        add_legend(
            figure, ["Neighbors", "RSM", "CCA"], ["#0173b2", "#de8f05", "#cc78bc"]
        )
        self.assertEqual(figure.legends[0]._ncols, 1)
        plt.close(figure)

    def test_nothing_drawn_adds_no_legend(self):
        figure, _ax = plt.subplots()
        add_legend(figure, [], [])
        self.assertEqual(figure.legends, [])
        plt.close(figure)

    def test_category_legend_lists_only_categories_present(self):
        figure, _ax = plt.subplots()
        category_legend(figure, ["CKArbfAUC", "SVCCA"])
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        plt.close(figure)
        self.assertEqual(set(labels), {"Signal", "CCA", "mean"})


class TestSingleImplementation(unittest.TestCase):
    def test_the_quantile_marker_is_drawn_only_when_it_is_sorted_on(self):
        """The marker exists to show the number the rows were ordered by.

        Drawing it under the median sort would put a statistic on the figure
        that nothing in the layout depends on.
        """
        frame = _frame()
        fig, ax = plt.subplots()
        measure_panel(ax, frame, "value", higher_is_better=False)
        self.assertEqual(_marker_count(ax), 0)
        plt.close(fig)

        fig, ax = plt.subplots()
        measure_panel(
            ax, frame, "value", higher_is_better=False, rank_sort="quantile90"
        )
        # One per measure drawn.
        self.assertEqual(_marker_count(ax), 2)
        plt.close(fig)

    def test_the_quantile_marker_is_not_drawn_under_mean_either(self):
        """`mean` needs no marker of its own: the mean is always drawn."""
        fig, ax = plt.subplots()
        measure_panel(ax, _frame(), "value", higher_is_better=False, rank_sort="mean")
        self.assertEqual(_marker_count(ax), 0)
        plt.close(fig)

    def test_the_marker_sits_at_the_unfavourable_tail(self):
        frame = pd.DataFrame({"metric": ["CKA"] * 11, "value": list(range(11))})
        fig, ax = plt.subplots()
        measure_panel(
            ax, frame, "value", higher_is_better=False, rank_sort="quantile90"
        )
        marks = [line for line in ax.lines if line.get_marker() == QUANTILE_MARKER]
        self.assertEqual(len(marks), 1)
        self.assertAlmostEqual(marks[0].get_xdata()[0], 9.0)
        plt.close(fig)

        # Higher-is-better flips which tail is the bad one.
        fig, ax = plt.subplots()
        measure_panel(ax, frame, "value", higher_is_better=True, rank_sort="quantile90")
        marks = [line for line in ax.lines if line.get_marker() == QUANTILE_MARKER]
        self.assertAlmostEqual(marks[0].get_xdata()[0], 1.0)
        plt.close(fig)

    def test_the_legend_names_the_quantile_only_when_it_is_shown(self):
        fig, ax = plt.subplots()
        drawn = measure_panel(ax, _frame(), "value", higher_is_better=False)
        category_legend(fig, drawn)
        self.assertNotIn(
            "90th pct.", [text.get_text() for text in fig.legends[0].get_texts()]
        )
        plt.close(fig)

        fig, ax = plt.subplots()
        drawn = measure_panel(
            ax, _frame(), "value", higher_is_better=False, rank_sort="quantile90"
        )
        category_legend(fig, drawn, rank_sort="quantile90")
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertEqual(labels[-1], "90th pct.")
        plt.close(fig)

    def test_a_value_panels_legend_names_the_tail_it_actually_shows(self):
        """The bad tail of a higher-is-better column is the 10th percentile."""
        fig, ax = plt.subplots()
        drawn = measure_panel(
            ax, _frame(), "value", higher_is_better=True, rank_sort="quantile90"
        )
        category_legend(fig, drawn, rank_sort="quantile90", higher_is_better=True)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertEqual(labels[-1], "10th pct.")
        plt.close(fig)

    def test_no_figure_module_calls_boxplot_outside_this_one(self):
        """The whole point of the module: one place styles a box."""
        from pathlib import Path

        root = Path(boxplots.__file__).parent
        offenders = []
        for path in list(root.glob("*.py")) + list((root / "tables").glob("*.py")):
            if path.name == "boxplots.py":
                continue
            if ".boxplot(" in path.read_text():
                offenders.append(path.name)
        self.assertEqual(offenders, [])


class TestMean(unittest.TestCase):
    """The mean marker: a white circle, black outline, drawn every time."""

    def _mean_marks(self, ax):
        return [line for line in ax.lines if line.get_marker() == MEAN_MARKER]

    def test_the_mean_is_drawn_at_the_series_mean_under_box(self):
        fig, ax = plt.subplots()
        measure_panel(ax, _frame(), "value", higher_is_better=False)
        marks = {round(line.get_xdata()[0], 6): line for line in self._mean_marks(ax)}
        self.assertIn(round(0.2, 6), marks)
        self.assertIn(round(0.8, 6), marks)
        plt.close(fig)

    def test_the_mean_is_drawn_under_boxen_too(self):
        fig, ax = plt.subplots()
        measure_panel(ax, _frame(), "value", higher_is_better=False, box_style="boxen")
        self.assertEqual(len(self._mean_marks(ax)), 2)
        plt.close(fig)

    def test_the_marker_is_a_white_circle_with_a_black_outline(self):
        fig, ax = plt.subplots()
        measure_panel(ax, _frame(), "value", higher_is_better=False)
        marks = self._mean_marks(ax)
        self.assertTrue(marks)
        for mark in marks:
            self.assertEqual(mark.get_markerfacecolor(), MEAN_FACE_COLOR)
            self.assertEqual(mark.get_markeredgecolor(), MEDIAN_COLOR)
        plt.close(fig)

    def test_the_mean_coexists_with_the_robustness_diamond(self):
        """Two different statistics, two different markers, both present."""
        fig, ax = plt.subplots()
        measure_panel(
            ax, _frame(), "value", higher_is_better=False, rank_sort="quantile90"
        )
        self.assertEqual(len(self._mean_marks(ax)), 2)
        self.assertEqual(_marker_count(ax), 2)
        plt.close(fig)

    def test_an_empty_series_draws_no_mean(self):
        fig, ax = plt.subplots()
        measure_panel(
            ax,
            pd.DataFrame({"metric": [], "value": []}),
            "value",
            higher_is_better=False,
        )
        self.assertEqual(self._mean_marks(ax), [])
        plt.close(fig)

    def test_grouped_panels_draw_a_mean_per_group_too(self):
        frame = pd.DataFrame(
            {
                "metric": ["CKA", "CKA"],
                "domain": ["graphs", "vision"],
                "value": [0.4, 0.6],
            }
        )
        fig, ax = plt.subplots()
        grouped_panel(
            ax,
            frame,
            "value",
            order=["CKA"],
            group_column="domain",
            group_order=["graphs", "vision"],
            colors={"graphs": "#0173b2", "vision": "#de8f05"},
        )
        self.assertEqual(len(self._mean_marks(ax)), 2)
        plt.close(fig)

    def test_the_legend_always_names_the_mean(self):
        fig, ax = plt.subplots()
        drawn = measure_panel(ax, _frame(), "value", higher_is_better=False)
        category_legend(fig, drawn)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertIn("mean", labels)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
