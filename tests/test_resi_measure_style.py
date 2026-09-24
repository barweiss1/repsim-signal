"""One naming and colouring scheme for every measure in every figure."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

from manifold_repsim.resi.analysis.config import (
    MEASURE_CATEGORY_COLORS,
    MEASURE_CATEGORY_ORDER,
    MEASURE_LABELS,
)
from manifold_repsim.resi.measures import RESI_MEASURE_SPECS
from manifold_repsim.resi.analysis.measure_style import (
    UNKNOWN_CATEGORY_COLOR,
    categories_present,
    label_sequence,
    measure_color,
    measure_label,
    measure_statistic,
    order_measures,
)


class TestLabels(unittest.TestCase):
    """The measure names themselves are pinned here and nowhere else.

    Other test modules ask `measure_label` what a measure is called rather than
    repeating the string, so renaming a measure is one edit here plus the
    golden LaTeX file, not a search across the suite.
    """

    def test_native_measures_use_the_published_abbreviation(self):
        self.assertEqual(measure_label("AlignedCosineSimilarity"), "AlignCos")
        self.assertEqual(measure_label("SecondOrderCosineSimilarity"), "2nd-Cos")

    def test_our_measures_are_named_as_mathematics_not_as_classes(self):
        """A reader meets `CKArbfAUC` nowhere in the paper; these are the names."""
        self.assertEqual(
            measure_label("CKArbfAUC"), r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{CKA})$"
        )
        self.assertEqual(
            measure_label("dRWKArbfAUC"),
            r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{dRWKA})$",
        )
        self.assertEqual(measure_label("CKNNATop10"), r"$\mathrm{CKNNA}_{10}$")
        self.assertEqual(measure_label("MutualKNNTop10"), r"$\mathrm{MNN}_{10}$")
        self.assertEqual(
            measure_label("CKArbfSigma02"), r"$\mathrm{CKA}_{\mathrm{RBF},0.2}$"
        )
        self.assertEqual(measure_label("CKA"), r"$\mathrm{CKA}_\mathrm{lin}$")

    def test_the_random_walk_alignments_keep_their_trailing_a(self):
        """They are alignments, like the CKA and UKA beside them on the axis."""
        for name in ("sRWKArbfAUC", "dRWKArbfAUC", "mcRWKArbfAUC", "RWKArbfAUC"):
            self.assertIn("RWKA", measure_label(name))

    def test_the_symmetric_random_walk_alignment_carries_the_plain_name(self):
        """It is the variant this work reports; the other is marked `a`.

        Naming both of them `RWKA` would put two identical rows on one axis,
        which is what the duplicate-label check at import exists to prevent.
        """
        self.assertEqual(
            measure_label("sRWKArbfAUC"),
            r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{RWKA})$",
        )
        self.assertEqual(
            measure_label("RWKArbfAUC"),
            r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{aRWKA})$",
        )

    def test_no_two_measures_share_a_label(self):
        labels = list(MEASURE_LABELS.values())
        self.assertEqual(len(labels), len(set(labels)))

    def test_boldsymbol_is_braced_so_both_renderers_accept_it(self):
        """LaTeX takes `\boldsymbol s`; Matplotlib's mathtext requires braces.

        One string has to serve the figures and the tables alike, so the braced
        spelling -- identical in LaTeX -- is the one that can.
        """
        signal = [label for label in MEASURE_LABELS.values() if "boldsymbol" in label]
        self.assertTrue(signal)
        for label in signal:
            self.assertIn(r"\boldsymbol{s}", label)

    def test_every_registered_measure_is_named(self):
        """An unnamed measure would draw an axis mixing maths with a class name."""
        for spec in RESI_MEASURE_SPECS:
            self.assertIn(spec.class_name, MEASURE_LABELS)

    def test_a_label_resolves_back_to_its_measures_category(self):
        """Colour and table block are keyed on the measure, not on the string.

        Frames reaching the table figures carry the printed label rather than
        the class name, so a label that did not resolve would paint the box
        grey and file the row under "Other".
        """
        for name in ("CKA", "CKArbfAUC", "MutualKNNTop10", "CKArbfSigma02"):
            self.assertEqual(measure_color(measure_label(name)), measure_color(name))
            self.assertNotEqual(
                measure_color(measure_label(name)), UNKNOWN_CATEGORY_COLOR
            )

    def test_an_unknown_name_passes_through(self):
        self.assertEqual(measure_label("Nonesuch"), "Nonesuch")

    def test_a_whole_axis_can_be_labelled_at_once(self):
        self.assertEqual(
            label_sequence(["AlignedCosineSimilarity", "CKArbfAUC"]),
            ["AlignCos", measure_label("CKArbfAUC")],
        )


class TestColors(unittest.TestCase):
    def test_a_manifold_measure_gets_its_category_not_a_fallback(self):
        """The tables figures used to paint every manifold measure one colour."""
        signal = MEASURE_CATEGORY_COLORS["Signal"]
        self.assertEqual(measure_color("CKArbfAUC"), signal)
        self.assertEqual(measure_color("UKArbfAUC"), signal)
        self.assertNotEqual(measure_color("CKArbfAUC"), UNKNOWN_CATEGORY_COLOR)

    def test_manifold_measures_do_not_share_one_colour(self):
        """They span three categories; one bucket would assert a family."""
        colors = {
            measure_color(name)
            for name in ("MutualKNNTop10", "CKArbfSigma05", "CKArbfAUC")
        }
        self.assertEqual(len(colors), 3)

    def test_a_full_name_and_its_abbreviation_agree(self):
        for full, short in (
            ("AlignedCosineSimilarity", "AlignCos"),
            ("SecondOrderCosineSimilarity", "2nd-Cos"),
            ("DistanceCorrelation", "DistCorr"),
        ):
            with self.subTest(measure=full):
                self.assertEqual(measure_color(full), measure_color(short))

    def test_an_unknown_measure_falls_back_rather_than_raising(self):
        self.assertEqual(measure_color("Nonesuch"), UNKNOWN_CATEGORY_COLOR)


class TestCategoriesPresent(unittest.TestCase):
    def test_only_categories_actually_drawn_are_listed(self):
        self.assertEqual(categories_present(["SVCCA"]), ["CCA"])

    def test_order_follows_the_shared_display_order(self):
        present = categories_present(["SVCCA", "CKArbfAUC", "Jaccard", "AlignCos"])
        self.assertEqual(
            present,
            [name for name in MEASURE_CATEGORY_ORDER if name in set(present)],
        )

    def test_nothing_drawn_lists_nothing(self):
        self.assertEqual(categories_present([]), [])


class TestOrdering(unittest.TestCase):
    def _frame(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "metric": ["a", "a", "b", "b", "c", "c"],
                "value": [0.1, 0.3, 0.5, 0.7, 0.9, 0.9],
            }
        )

    def test_lower_is_better_puts_the_smallest_median_last(self):
        """Position 0 is the bottom of the axis, so the best must come last."""
        order = order_measures(self._frame(), "value", higher_is_better=False)
        self.assertEqual(order[-1], "a")

    def test_higher_is_better_puts_the_largest_median_last(self):
        order = order_measures(self._frame(), "value", higher_is_better=True)
        self.assertEqual(order[-1], "c")

    def test_an_all_nan_measure_is_dropped_rather_than_placed(self):
        import numpy as np
        import pandas as pd

        frame = pd.DataFrame({"metric": ["a", "a", "b"], "value": [0.2, 0.4, np.nan]})
        self.assertEqual(order_measures(frame, "value", higher_is_better=True), ["a"])


class TestBackendIndependence(unittest.TestCase):
    def test_measure_style_imports_without_matplotlib(self):
        """It is imported by data modules, which must not need a backend."""
        code = (
            "import builtins\n"
            "_real = builtins.__import__\n"
            "def _guard(name, *args, **kwargs):\n"
            "    if name.split('.')[0] == 'matplotlib':\n"
            "        raise ImportError('matplotlib is unavailable')\n"
            "    return _real(name, *args, **kwargs)\n"
            "builtins.__import__ = _guard\n"
            "from manifold_repsim.resi.analysis import measure_style\n"
            "print(measure_style.measure_label('AlignedCosineSimilarity'))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "AlignCos")


if __name__ == "__main__":
    unittest.main()


class TestRobustnessOrdering(unittest.TestCase):
    """Sorting on the unfavourable tail rather than on the centre."""

    def _skewed(self):
        # Same median, different tails: `steady` never places badly, `spiky`
        # is usually as good but occasionally worst.
        return pd.DataFrame(
            {
                "metric": ["steady"] * 5 + ["spiky"] * 5,
                "value": [0.3, 0.4, 0.5, 0.5, 0.6] + [0.1, 0.2, 0.5, 0.9, 1.0],
            }
        )

    def test_the_statistic_is_the_bad_tail_of_the_column(self):
        frame = pd.DataFrame({"metric": ["a"] * 11, "value": list(range(11))})
        low = measure_statistic(
            frame, "value", higher_is_better=False, rank_sort="quantile90"
        )
        self.assertAlmostEqual(low["a"], 9.0)
        high = measure_statistic(
            frame, "value", higher_is_better=True, rank_sort="quantile90"
        )
        self.assertAlmostEqual(high["a"], 1.0)

    def test_a_tie_on_the_median_is_broken_by_the_tail(self):
        frame = self._skewed()
        medians = measure_statistic(frame, "value", higher_is_better=False)
        self.assertEqual(medians["steady"], medians["spiky"])
        # Ranks are best at 0, so best lands last and is drawn on top.
        self.assertEqual(
            order_measures(
                frame, "value", higher_is_better=False, rank_sort="quantile90"
            )[-1],
            "steady",
        )

    def test_the_two_statistics_can_disagree(self):
        frame = pd.DataFrame(
            {
                "metric": ["low_median"] * 5 + ["low_tail"] * 5,
                "value": [0.1, 0.1, 0.2, 0.9, 1.0] + [0.3, 0.4, 0.4, 0.5, 0.5],
            }
        )
        by_median = order_measures(frame, "value", higher_is_better=False)
        by_tail = order_measures(
            frame, "value", higher_is_better=False, rank_sort="quantile90"
        )
        self.assertEqual(by_median[-1], "low_median")
        self.assertEqual(by_tail[-1], "low_tail")

    def test_an_unknown_sort_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank_sort must be one of"):
            order_measures(
                self._skewed(), "value", higher_is_better=False, rank_sort="p99"
            )


class TestMeanOrdering(unittest.TestCase):
    """Sorting on the average standing rather than the median or the tail."""

    def test_the_statistic_is_the_arithmetic_mean(self):
        frame = pd.DataFrame({"metric": ["a"] * 5, "value": [0.0, 0.0, 0.0, 0.0, 1.0]})
        mean = measure_statistic(
            frame, "value", higher_is_better=False, rank_sort="mean"
        )
        self.assertAlmostEqual(mean["a"], 0.2)

    def test_a_skewed_outlier_moves_the_mean_but_not_the_median(self):
        """The reason to offer mean as a third option: it disagrees with median."""
        frame = pd.DataFrame(
            {
                "metric": ["steady"] * 5 + ["skewed"] * 5,
                "value": [0.3, 0.4, 0.5, 0.5, 0.6] + [0.1, 0.2, 0.3, 0.3, 5.0],
            }
        )
        medians = measure_statistic(frame, "value", higher_is_better=False)
        self.assertLess(medians["skewed"], medians["steady"])
        means = measure_statistic(
            frame, "value", higher_is_better=False, rank_sort="mean"
        )
        self.assertGreater(means["skewed"], means["steady"])
        # Ranks are best at 0, so best lands last and is drawn on top.
        self.assertEqual(
            order_measures(frame, "value", higher_is_better=False)[-1], "skewed"
        )
        self.assertEqual(
            order_measures(frame, "value", higher_is_better=False, rank_sort="mean")[
                -1
            ],
            "steady",
        )
