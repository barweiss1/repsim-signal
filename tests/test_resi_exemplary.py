"""The exemplary table and its averaged counterpart.

The exemplary table fixes one model and dataset per domain; the averaged one
keeps its rows and columns and averages every model and dataset into each cell.
The tests that matter most here are the ones pinning the *order* the averaged
table takes its means in, since every order produces a number and only one of
them weights the models equally.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from manifold_repsim.resi.analysis.config import measure_label
from manifold_repsim.resi.analysis.exemplary import (
    EXEMPLARY_CELLS,
    TASK_EVALUATIONS,
    average_display_cells,
    averaged_caption,
    build_averaged_display,
    build_averaged_table,
    build_exemplary_display,
    build_exemplary_table,
    exemplary_caption,
    select_exemplary_rows,
    write_averaged_table,
    write_exemplary_table,
)
from manifold_repsim.resi.analysis.tables.beautify import build_display_frame


def _values(domain, benchmark, dataset, architecture, quality, functional="", n=2):
    """Normalized rows for one cell, one row per measure."""
    rows = []
    for index, metric in enumerate(("CKA", "SVCCA")):
        for observation in range(n):
            rows.append(
                {
                    "domain": domain,
                    "benchmark": benchmark,
                    "dataset": dataset,
                    "architecture": architecture,
                    "observation_id": f"{dataset}-{observation}",
                    "metric": metric,
                    "quality_measure": quality,
                    "value": 0.5 + 0.1 * index,
                    "functional_similarity_measure": functional,
                    "model": "agg",
                }
            )
    return pd.DataFrame(rows)


def _domain_frame(domain, benchmark_correlation, dataset, architecture, other=None):
    """One domain's frame: a correlation benchmark plus one design benchmark."""
    frames = [
        _values(
            domain, benchmark_correlation, dataset, architecture, "spearmanr", "JSD"
        ),
        _values(
            domain,
            benchmark_correlation,
            dataset,
            architecture,
            "spearmanr",
            "AbsoluteAccDiff",
        ),
        _values(domain, "shortcut", dataset, architecture, "AUPRC"),
        # A quality measure the table does not report, to prove it is dropped.
        _values(domain, "shortcut", dataset, architecture, "violation_rate"),
    ]
    if other is not None:
        frames.append(_values(domain, "shortcut", other, architecture, "AUPRC"))
    return pd.concat(frames, ignore_index=True)


class TestCellSelection(unittest.TestCase):
    def test_only_the_named_cell_survives(self):
        frame = _domain_frame(
            "vision", "accoutput", "ImageNet100", "ResNet18", other="CIFAR100"
        )
        kept = select_exemplary_rows(frame, ("ImageNet100", "ResNet18"))
        self.assertEqual(set(kept["dataset"]), {"ImageNet100"})
        self.assertEqual(set(kept["architecture"]), {"ResNet18"})

    def test_an_absent_cell_yields_nothing_rather_than_raising(self):
        frame = _domain_frame("vision", "accoutput", "ImageNet100", "ResNet18")
        self.assertTrue(select_exemplary_rows(frame, ("CIFAR100", "ViT_L32")).empty)

    def test_an_empty_frame_is_returned_unchanged(self):
        self.assertTrue(select_exemplary_rows(pd.DataFrame(), ("a", "b")).empty)

    def test_the_checked_in_cells_are_the_ones_the_source_table_names(self):
        """These are load-bearing: the point is comparability with ReSi's table."""
        self.assertEqual(EXEMPLARY_CELLS["graphs"], ("flickr", "GraphSAGE"))
        self.assertEqual(EXEMPLARY_CELLS["language"], ("sst2", "BERT-L"))
        self.assertEqual(EXEMPLARY_CELLS["vision"], ("ImageNet100", "ResNet18"))


class TestDisplay(unittest.TestCase):
    def _frames(self):
        return {
            "vision": _domain_frame(
                "vision", "accoutput", "ImageNet100", "ResNet18", other="CIFAR100"
            ),
            "graphs": _domain_frame(
                "graphs", "output_correlation_test", "flickr", "GraphSAGE"
            ),
        }

    def test_unreported_evaluation_measures_are_dropped(self):
        """Conformity rate is computed everywhere and shown in no column here."""
        display = build_exemplary_display(self._frames())
        self.assertNotIn("Conformity Rate", set(display["Eval."].astype(str)))
        self.assertIn("AUPRC", set(display["Eval."].astype(str)))

    def test_each_task_keeps_only_its_reported_evaluations(self):
        display = build_exemplary_display(self._frames())
        for test, group in display.groupby("Test", observed=True):
            allowed = TASK_EVALUATIONS[str(test)]
            self.assertTrue(set(group["Eval."].astype(str)).issubset(set(allowed)))

    def test_the_two_correlation_tasks_stay_apart(self):
        display = build_exemplary_display(self._frames())
        tests = set(display["Test"].astype(str))
        self.assertIn("Output Corr.", tests)
        self.assertIn("Acc Corr.", tests)

    def test_a_domain_without_its_cell_contributes_nothing(self):
        frames = {"vision": _domain_frame("vision", "accoutput", "CIFAR100", "ViT_L32")}
        self.assertTrue(build_exemplary_display(frames).empty)

    def test_no_domains_yields_an_empty_frame(self):
        self.assertTrue(build_exemplary_display({}).empty)


class TestTable(unittest.TestCase):
    def _frames(self):
        return {
            "vision": _domain_frame("vision", "accoutput", "ImageNet100", "ResNet18"),
            "graphs": _domain_frame(
                "graphs", "output_correlation_test", "flickr", "GraphSAGE"
            ),
        }

    def test_columns_carry_the_four_levels(self):
        table = build_exemplary_table(self._frames())
        self.assertEqual(list(table.columns.names), ["Type", "Test", "Eval.", "Domain"])

    def test_measures_are_the_rows(self):
        table = build_exemplary_table(self._frames())
        self.assertEqual(
            sorted(table.index.get_level_values("Sim Meas.")),
            sorted(measure_label(name) for name in ("CKA", "SVCCA")),
        )

    def test_an_empty_input_yields_an_empty_table(self):
        self.assertTrue(build_exemplary_table({}).empty)

    def test_the_caption_names_every_cell(self):
        caption = exemplary_caption()
        for domain, (dataset, architecture) in EXEMPLARY_CELLS.items():
            self.assertIn(dataset, caption)
            self.assertIn(architecture, caption)


class TestWriting(unittest.TestCase):
    def _frames(self):
        return {
            "vision": _domain_frame("vision", "accoutput", "ImageNet100", "ResNet18"),
            "graphs": _domain_frame(
                "graphs", "output_correlation_test", "flickr", "GraphSAGE"
            ),
        }

    def test_csv_and_latex_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            index = write_exemplary_table(self._frames(), directory)
            self.assertEqual(len(index), 1)
            self.assertTrue(Path(index["csv_path"].iloc[0]).is_file())

    def test_the_rendered_table_declares_its_columns(self):
        """A regression guard: the layout used to be found by literal "& Dataset".

        This table's deepest levels are (Eval., Domain), so that lookup failed
        and emitted `\\begin{tabular}{}` -- a table LaTeX cannot compile.
        """
        with tempfile.TemporaryDirectory() as directory:
            index = write_exemplary_table(self._frames(), directory)
            latex_path = index["latex_path"].iloc[0]
            if not latex_path:
                self.skipTest(f"latex layout skipped: {index['note'].iloc[0]}")
            rendered = Path(latex_path).read_text(encoding="utf-8")
        self.assertNotIn(r"\begin{tabular}{}", rendered)
        self.assertIn(r"\begin{tabular}{ll|", rendered)

    def test_nothing_is_written_when_no_cell_matches(self):
        frames = {"vision": _domain_frame("vision", "accoutput", "CIFAR100", "ViT_L32")}
        with tempfile.TemporaryDirectory() as directory:
            index = write_exemplary_table(frames, directory)
            self.assertTrue(index.empty)
            self.assertEqual(list(Path(directory).glob("*.csv")), [])


def _cells(domain, benchmark, quality, cells, functional="", metric="CKA"):
    """Normalized rows for named (architecture, dataset) cells.

    `cells` maps (architecture, dataset) to the list of per-seed values for
    that cell, so a test can give one architecture more datasets or more seeds
    than another and see which weighting the averaging produces.
    """
    rows = []
    for (architecture, dataset), values in cells.items():
        for seed, value in enumerate(values):
            rows.append(
                {
                    "domain": domain,
                    "benchmark": benchmark,
                    "dataset": dataset,
                    "architecture": architecture,
                    "observation_id": f"{dataset}-{seed}",
                    "metric": metric,
                    "quality_measure": quality,
                    "value": value,
                    "functional_similarity_measure": functional,
                    "model": f"{architecture}-{dataset}-{seed}",
                }
            )
    return pd.DataFrame(rows)


def _averaged_value(display, test, domain="Vision", measure="CKA"):
    row = display[
        display["Test"].astype(str).eq(test)
        & display["Domain"].eq(domain)
        & display["Sim Meas."].eq(measure_label(measure))
    ]
    assert len(row) == 1, f"expected one cell, got {len(row)}"
    return float(row["value"].iloc[0])


class TestAveragingOrder(unittest.TestCase):
    """Each mean is taken over one dimension, outermost last.

    Every test here would also pass with a single flat mean over the raw rows
    if the fixture were balanced, which is why each fixture is deliberately
    ragged in exactly one dimension.
    """

    def test_a_model_with_more_datasets_does_not_weigh_more(self):
        """Two datasets for one model, one for the other; both count once."""
        frames = {
            "vision": _cells(
                "vision",
                "shortcut",
                "AUPRC",
                {
                    ("ResNet18", "CIFAR100"): [0.0],
                    ("ResNet18", "ImageNet100"): [1.0],
                    ("ViT_B32", "CIFAR100"): [0.0],
                },
            )
        }
        display = build_averaged_display(frames)
        # (0.5 + 0.0) / 2, not the flat (0 + 1 + 0) / 3 = 0.33.
        self.assertAlmostEqual(_averaged_value(display, "Shortcuts"), 0.25)

    def test_a_model_with_more_seeds_does_not_weigh_more(self):
        """Layer monotonicity ships one row per model; the rest ship one."""
        frames = {
            "vision": _cells(
                "vision",
                "shortcut",
                "AUPRC",
                {
                    ("ResNet18", "CIFAR100"): [0.0, 0.0, 0.0],
                    ("ViT_B32", "CIFAR100"): [1.0],
                },
            )
        }
        display = build_averaged_display(frames)
        # (0.0 + 1.0) / 2, not the flat (0 + 0 + 0 + 1) / 4 = 0.25.
        self.assertAlmostEqual(_averaged_value(display, "Shortcuts"), 0.5)

    def test_seeds_are_averaged_before_datasets(self):
        """Ragged in both dimensions at once: only one order gives 0.5."""
        frames = {
            "vision": _cells(
                "vision",
                "shortcut",
                "AUPRC",
                {
                    ("ResNet18", "CIFAR100"): [0.0, 0.0, 0.0, 0.0],
                    ("ResNet18", "ImageNet100"): [1.0],
                    ("ViT_B32", "CIFAR100"): [0.5],
                },
            )
        }
        display = build_averaged_display(frames)
        # ResNet18 is (0.0 + 1.0) / 2 = 0.5; with ViT_B32 that is 0.5 again.
        self.assertAlmostEqual(_averaged_value(display, "Shortcuts"), 0.5)

    def test_missing_values_are_skipped_rather_than_emptying_the_cell(self):
        """A model whose value never computed leaves the others' mean intact."""
        frames = {
            "vision": _cells(
                "vision",
                "shortcut",
                "AUPRC",
                {
                    ("ResNet18", "CIFAR100"): [0.4],
                    ("ViT_B32", "CIFAR100"): [float("nan")],
                },
            )
        }
        display = build_averaged_display(frames)
        self.assertAlmostEqual(_averaged_value(display, "Shortcuts"), 0.4)

    def test_a_model_missing_one_dataset_still_counts_once(self):
        """The real shape of vision's ViTs: no JSD correlation on CIFAR100."""
        frames = {
            "vision": _cells(
                "vision",
                "shortcut",
                "AUPRC",
                {
                    ("ResNet18", "CIFAR100"): [0.0],
                    ("ResNet18", "ImageNet100"): [0.0],
                    ("ViT_B32", "CIFAR100"): [float("nan")],
                    ("ViT_B32", "ImageNet100"): [1.0],
                },
            )
        }
        display = build_averaged_display(frames)
        self.assertAlmostEqual(_averaged_value(display, "Shortcuts"), 0.5)

    def test_pvalues_are_dropped_rather_than_averaged(self):
        """A mean of p-values is not a p-value, and the formatter reads it."""
        display = build_display_frame(
            _cells("vision", "shortcut", "AUPRC", {("ResNet18", "CIFAR100"): [0.4]})
        )
        display["pval"] = 0.01
        self.assertNotIn("pval", average_display_cells(display).columns)

    def test_an_empty_frame_is_returned_unchanged(self):
        self.assertTrue(average_display_cells(pd.DataFrame()).empty)


class TestAveragedTable(unittest.TestCase):
    def _frames(self):
        return {
            "vision": _domain_frame(
                "vision", "accoutput", "ImageNet100", "ResNet18", other="CIFAR100"
            ),
            "graphs": _domain_frame(
                "graphs", "output_correlation_test", "flickr", "GraphSAGE"
            ),
        }

    def test_the_columns_match_the_exemplary_table(self):
        """The point of this table is that it can be read against that one."""
        averaged = build_averaged_table(self._frames())
        exemplary = build_exemplary_table(self._frames())
        self.assertEqual(list(averaged.columns.names), list(exemplary.columns.names))

    def test_unreported_evaluation_measures_are_dropped_here_too(self):
        display = build_averaged_display(self._frames())
        self.assertNotIn("Conformity Rate", set(display["Eval."].astype(str)))

    def test_every_cell_contributes_not_only_the_exemplary_one(self):
        """`other` supplies a second dataset the exemplary table never sees."""
        frames = {
            "vision": _cells(
                "vision",
                "shortcut",
                "AUPRC",
                {
                    ("ResNet18", "ImageNet100"): [0.0],
                    ("ViT_L32", "CIFAR100"): [1.0],
                },
            )
        }
        display = build_averaged_display(frames)
        self.assertAlmostEqual(_averaged_value(display, "Shortcuts"), 0.5)

    def test_no_domains_yields_an_empty_table(self):
        self.assertTrue(build_averaged_table({}).empty)

    def test_the_caption_names_each_domain_and_its_model_count(self):
        caption = averaged_caption(self._frames())
        self.assertIn("Vision: 1 models on 2 datasets", caption)
        self.assertIn("Graph: 1 models on 1 datasets", caption)

    def test_the_caption_claims_no_counts_without_frames(self):
        self.assertNotIn("models on", averaged_caption())

    def test_csv_and_latex_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            index = write_averaged_table(self._frames(), directory)
            self.assertEqual(len(index), 1)
            self.assertEqual(index["table"].iloc[0], "averaged")
            self.assertTrue(Path(index["csv_path"].iloc[0]).is_file())

    def test_nothing_is_written_for_an_empty_input(self):
        with tempfile.TemporaryDirectory() as directory:
            index = write_averaged_table({}, directory)
            self.assertTrue(index.empty)
            self.assertEqual(list(Path(directory).glob("*.csv")), [])


if __name__ == "__main__":
    unittest.main()
