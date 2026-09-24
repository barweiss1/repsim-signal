"""Tests for the recreated ReSi result tables and rank figures.

The LaTeX golden file pins the rendered layout. It depends on the installed
pandas Styler output, so a pandas upgrade that changes that output will fail
here rather than silently changing published tables.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from manifold_repsim.resi.analysis.config import (
    MEASURE_CATEGORY_ORDER,
    measure_category,
    measure_label,
)
from manifold_repsim.resi.analysis.tables.taxonomy import (
    MEASURE_GROUP_LABEL_ORDER,
    measure_group_label,
)
from manifold_repsim.resi.analysis.tables import (
    build_correlation_table,
    build_display_frame,
    build_overview_table,
    build_value_table,
    floatify,
    pval_str,
    rank_measures,
    render_latex_table,
    separate_significance_indicator,
    summarize_ranks,
    write_appendix_report,
    write_latex_table,
)
from manifold_repsim.resi.analysis.tables.latex import latex_safe_labels
from manifold_repsim.resi.analysis.tables.figures import (
    plot_rank_distributions,
    plot_task_value_distributions,
)
from manifold_repsim.resi.analysis.tables.significance import (
    HIGHLY_SIGNIFICANT_MARKER,
    NOT_SIGNIFICANT_MARKER,
    SIGNIFICANT_MARKER,
)

PANDAS_VERSION = "2"

GOLDEN_LATEX = r"""\begin{table}[h]
\caption{Overview}
\label{tab:overview}
\centering
\resizebox{1.0\linewidth}{!}{
\rowcolors{2}{white}{Gray}
\begin{tabular}{ll||r|r|r}
\rowcolor{white}\toprule
\rowcolor{white} & Type & Grounding by Prediction & \multicolumn{2}{c}{Grounding by Design} \\
\rowcolor{white} & Test & Acc Corr. & Augmentation & Layer Mono. \\
\rowcolor{white} & Eval. & Acc Diff & AUPRC & Conformity Rate \\
\rowcolor{white} & Domain & Graph & Graph & Graph \\
\rowcolor{white} & Dataset & Cora & Cora & Cora \\
\rowcolor{white} & Arch. & GCN & GCN & GCN \\
\rowcolor{white}\midrule
\cellcolor{white}Neighbors & $\mathrm{MNN}_{10}$ & \textbf{0.80}$^{\phantom{**}}$ & \textbf{0.80} & 0.20 \\\midrule
\cellcolor{white} & $\mathrm{CKA}_\mathrm{lin}$ & 0.40$^{\phantom{**}}$ & 0.40 & \textbf{0.60} \\
\cellcolor{white}\multirow[c]{-2}{*}{RSM} & RSA & 0.60$^{\phantom{**}}$ & 0.60 & 0.40 \\
\bottomrule
\end{tabular}
}
\end{table}
"""


def _normalized_values(pvalues: dict[str, float] | None = None) -> pd.DataFrame:
    rows = []
    for benchmark, quality in (
        ("augmentation", "AUPRC"),
        ("accoutput", "correlation"),
        ("monotonicity", "violation_rate"),
    ):
        for metric, value in (
            ("CKA", 0.4),
            ("MutualKNNTop10", 0.8),
            ("RSA", 0.6),
        ):
            row = {
                "domain": "graphs",
                "benchmark": benchmark,
                "dataset": "cora",
                "architecture": "GCN",
                "metric": metric,
                "quality_measure": quality,
                "value": value,
                "model": "agg",
                # The correlation benchmark is what carries a functional
                # similarity measure; the design tests have none.
                "functional_similarity_measure": (
                    "AbsoluteAccDiff" if benchmark == "accoutput" else ""
                ),
            }
            if pvalues is not None:
                row["pval"] = pvalues[metric]
            rows.append(row)
    return pd.DataFrame(rows)


class TestDisplayFrame(unittest.TestCase):
    def test_labels_groups_and_conformity_inversion(self):
        display = build_display_frame(_normalized_values())
        self.assertEqual(display["Domain"].unique().tolist(), ["Graph"])
        self.assertEqual(display["Dataset"].unique().tolist(), ["Cora"])
        self.assertEqual(
            sorted(display["Test"].astype(str).unique()),
            ["Acc Corr.", "Augmentation", "Layer Mono."],
        )
        # Correlation rows name their functional measure here rather than
        # repeating the Spearman statistic every other correlation row carries.
        self.assertEqual(
            sorted(display["Eval."].unique()),
            ["AUPRC", "Acc Diff", "Conformity Rate"],
        )
        self.assertEqual(
            sorted(display["Measure Type"].astype(str).unique()),
            ["Neighbors", "RSM"],
        )
        self.assertEqual(
            display.loc[display["Test"].eq("Acc Corr."), "Type"].unique().tolist(),
            ["Grounding by Prediction"],
        )

        conformity = display[display["Eval."].eq("Conformity Rate")]
        # Violation rate 0.4 is reported as conformity 0.6.
        self.assertAlmostEqual(
            conformity.loc[
                conformity["Sim Meas."].eq(measure_label("CKA")), "value"
            ].iloc[0],
            0.6,
        )

    def test_measure_groups_are_the_taxonomy_the_figures_colour_by(self):
        """A measure must not sit in one group in a table and another beside it.

        The source tables put every local measure in one "Manifold" block. That
        block asserted a family that does not exist: these three measures span
        Neighbors, RSM, and Signal and finish at opposite ends of the field.
        """
        display = build_display_frame(_normalized_values())
        groups = display.set_index("Sim Meas.")["Measure Type"].astype(str)
        neighbors = groups[measure_label("MutualKNNTop10")]
        self.assertEqual(neighbors.unique().tolist(), ["Neighbors"])
        self.assertEqual(groups[measure_label("CKA")].unique().tolist(), ["RSM"])
        self.assertEqual(measure_category("CKArbfAUC"), "Signal")

    def test_the_swept_block_is_marked_as_ours_in_tables_only(self):
        """Table blocks name their author where the whole block is ours.

        Every swept AUC measure is one of ours, so that block says so. The
        figures keep the plain category name -- their legend is a legend of
        categories, not of authorship -- and three more of our measures sit in
        the Neighbors and RSM blocks, which carry no such mark.
        """
        self.assertEqual(measure_group_label("Signal"), "Signal (ours)")
        self.assertEqual(measure_group_label("Neighbors"), "Neighbors")
        self.assertIn("Signal (ours)", MEASURE_GROUP_LABEL_ORDER)
        self.assertEqual(len(MEASURE_GROUP_LABEL_ORDER), len(MEASURE_CATEGORY_ORDER))

        values = _normalized_values()
        swept = values[values["metric"].eq("MutualKNNTop10")].copy()
        swept["metric"] = "CKArbfAUC"
        display = build_display_frame(pd.concat([values, swept], ignore_index=True))
        groups = display.set_index("Sim Meas.")["Measure Type"].astype(str)
        swept_group = groups[measure_label("CKArbfAUC")]
        self.assertEqual(swept_group.unique().tolist(), ["Signal (ours)"])
        # Ours, but not in the swept block, so unmarked.
        fixed_group = groups[measure_label("MutualKNNTop10")]
        self.assertEqual(fixed_group.unique().tolist(), ["Neighbors"])
        # The figures colour by the plain category.
        self.assertEqual(measure_category("CKArbfAUC"), "Signal")

    def test_correlation_tests_are_split_by_functional_measure(self):
        """One benchmark run scores JSD, Disagreement, and AbsoluteAccDiff alike.

        ReSi splits those into separate tables; naming the test after the
        benchmark id instead labels all three after whichever measure that
        domain's benchmark happens to be called for.
        """
        rows = []
        for functional in ("JSD", "Disagreement", "AbsoluteAccDiff", "", "Unknown"):
            frame = _normalized_values()
            frame = frame[frame["benchmark"].eq("accoutput")].copy()
            frame["functional_similarity_measure"] = functional
            rows.append(frame)
        display = build_display_frame(pd.concat(rows, ignore_index=True))
        by_functional = (
            display.groupby("Functional Similarity Measure", observed=True)["Test"]
            .agg(lambda tests: sorted({str(test) for test in tests}))
            .to_dict()
        )
        # JSD and Disagreement are two metrics of one task, not two tasks.
        self.assertEqual(by_functional["JSD"], ["Output Corr."])
        self.assertEqual(by_functional["Disagreement"], ["Output Corr."])
        # Accuracy difference asks a different question, so it is its own task.
        self.assertEqual(by_functional["AbsoluteAccDiff"], ["Acc Corr."])
        # A row with no usable functional measure names the family instead of
        # claiming either task.
        self.assertEqual(by_functional[""], ["Prediction Corr."])
        self.assertEqual(by_functional["Unknown"], ["Prediction Corr."])
        self.assertEqual(
            display["Type"].dropna().unique().tolist(), ["Grounding by Prediction"]
        )

    def test_design_tests_are_not_split_by_a_stray_functional_measure(self):
        values = _normalized_values()
        values["functional_similarity_measure"] = "JSD"
        display = build_display_frame(values)
        design = display[display["Test"].astype(str).eq("Augmentation")]
        self.assertEqual(len(design), 3)

    def test_missing_required_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_display_frame(pd.DataFrame({"domain": ["graphs"]}))


class TestLatexLabelSanitising(unittest.TestCase):
    def test_underscores_in_labels_become_spaces(self):
        """An unescaped underscore opens math mode and fails to compile.

        These tables render with escape=False so the cells' \\textbf and
        significance markers survive, which leaves labels unescaped too.
        `ViT_B32` and `ViT_L32` are the architectures that hit this.
        """
        values = _normalized_values()
        values["architecture"] = "ViT_B32"
        display = build_display_frame(values)
        table = build_overview_table(display)
        # The frame the CSV is written from keeps the real name.
        self.assertIn("ViT_B32", str(table.columns.tolist()))
        rendered = render_latex_table(table, "Overview", "tab:overview")
        self.assertNotIn("ViT_B32", rendered)
        self.assertIn("ViT B32", rendered)

    def test_a_math_label_keeps_the_underscores_that_are_its_subscripts(self):
        """Sanitising used to turn `CKA_lin` into `CKA lin`, which compiles.

        The measure labels are mathematics, so their underscores are subscripts
        the author wrote, not stray characters to strip. Only labels that are
        not already math mode are sanitised.
        """
        table = build_overview_table(build_display_frame(_normalized_values()))
        rendered = render_latex_table(table, "Overview", "tab:overview")
        self.assertIn(measure_label("CKA"), rendered)
        self.assertIn(r"\mathrm{CKA}_\mathrm{lin}", rendered)

    def test_sanitising_covers_every_level_of_a_multiindex(self):
        frame = pd.DataFrame(
            [[1.0]],
            index=pd.MultiIndex.from_tuples([("a_b", "c_d")]),
            columns=pd.MultiIndex.from_tuples([("e_f", "g_h")]),
        )
        safe = latex_safe_labels(frame)
        self.assertEqual(safe.index.tolist(), [("a b", "c d")])
        self.assertEqual(safe.columns.tolist(), [("e f", "g h")])


class TestSignificanceFormatting(unittest.TestCase):
    def test_marker_thresholds(self):
        self.assertEqual(pval_str(0.001), HIGHLY_SIGNIFICANT_MARKER)
        self.assertEqual(pval_str(0.01), HIGHLY_SIGNIFICANT_MARKER)
        self.assertEqual(pval_str(0.02), SIGNIFICANT_MARKER)
        self.assertEqual(pval_str(0.05), SIGNIFICANT_MARKER)
        self.assertEqual(pval_str(0.2), NOT_SIGNIFICANT_MARKER)
        self.assertEqual(pval_str(None), NOT_SIGNIFICANT_MARKER)

    def test_split_and_rejoin_a_marked_value(self):
        marked = "-0.10" + NOT_SIGNIFICANT_MARKER
        self.assertEqual(floatify(marked), "-0.10")
        self.assertEqual(
            separate_significance_indicator(marked), NOT_SIGNIFICANT_MARKER
        )

    def test_values_without_a_marker_are_left_intact(self):
        """The source helper dropped the final character of an unmarked value."""
        self.assertEqual(floatify("0.80"), "0.80")
        self.assertEqual(separate_significance_indicator("0.80"), "")


class TestTableConstruction(unittest.TestCase):
    def setUp(self):
        self.display = build_display_frame(_normalized_values())

    def test_value_table_bolds_the_column_maximum(self):
        table = build_value_table(
            self.display[self.display["Test"].eq("Augmentation")],
        )
        column = table.columns[0]
        values = table[column].tolist()
        self.assertIn(r"\textbf{0.80}", values)
        self.assertIn("0.40", values)
        self.assertEqual(sum(value.startswith(r"\textbf") for value in values), 1)

    def test_the_default_marks_the_best_value_only(self):
        """The appendix tables and the golden fixture depend on this default."""
        table = build_value_table(
            self.display[self.display["Test"].eq("Augmentation")],
        )
        rendered = "".join(table[table.columns[0]].tolist())
        self.assertNotIn(r"\underline", rendered)
        self.assertNotIn(r"\textit", rendered)

    def test_deeper_emphasis_marks_the_runner_up_and_the_third(self):
        table = build_value_table(
            self.display[self.display["Test"].eq("Augmentation")],
            emphasis_depth=3,
        )
        values = table[table.columns[0]].tolist()
        self.assertEqual(sum(value.startswith(r"\textbf") for value in values), 1)
        self.assertEqual(sum(value.startswith(r"\underline") for value in values), 1)
        self.assertEqual(sum(value.startswith(r"\textit") for value in values), 1)

    def test_marking_follows_the_printed_value_not_the_underlying_float(self):
        """Two cells printing 0.80 must not be marked differently."""
        values = _normalized_values()
        augmentation = values["benchmark"].eq("augmentation")
        for metric, value in (("CKA", 0.804), ("MutualKNNTop10", 0.797), ("RSA", 0.1)):
            values.loc[augmentation & values["metric"].eq(metric), "value"] = value
        display = build_display_frame(values)
        table = build_value_table(
            display[display["Test"].eq("Augmentation")], emphasis_depth=3
        )
        values = table[table.columns[0]].tolist()
        self.assertEqual(sum(value.startswith(r"\textbf") for value in values), 2)
        # The tie shares first place; the next value down is still the second.
        self.assertEqual(sum(value.startswith(r"\underline") for value in values), 1)
        self.assertEqual(sum(value.startswith(r"\textit") for value in values), 0)

    def test_deeper_emphasis_keeps_the_significance_marker_outside(self):
        display = build_display_frame(
            _normalized_values({"CKA": 0.5, "MutualKNNTop10": 0.001, "RSA": 0.03})
        )
        table = build_correlation_table(
            display[display["Test"].eq("Acc Corr.")], emphasis_depth=3
        )
        rendered = "".join(table[table.columns[0]].tolist())
        self.assertIn("}" + NOT_SIGNIFICANT_MARKER, rendered)
        self.assertNotIn(NOT_SIGNIFICANT_MARKER + "}", rendered)

    def test_conformity_inversion_changes_which_measure_wins(self):
        table = build_value_table(
            self.display[self.display["Test"].eq("Layer Mono.")],
        )
        column = table.columns[0]
        best = [value for value in table[column] if value.startswith(r"\textbf")]
        # CKA has the lowest violation rate, so it has the highest conformity.
        self.assertEqual(best, [r"\textbf{0.60}"])

    def test_correlation_table_carries_significance_markers(self):
        display = build_display_frame(
            _normalized_values({"CKA": 0.5, "MutualKNNTop10": 0.001, "RSA": 0.03})
        )
        table = build_correlation_table(display[display["Test"].eq("Acc Corr.")])
        rendered = "".join(table[table.columns[0]].tolist())
        self.assertIn(HIGHLY_SIGNIFICANT_MARKER, rendered)
        self.assertIn(SIGNIFICANT_MARKER, rendered)
        self.assertIn(NOT_SIGNIFICANT_MARKER, rendered)

    def test_measure_groups_are_ordered_and_rows_are_measures(self):
        table = build_overview_table(self.display)
        groups = [index[0] for index in table.index]
        self.assertEqual(groups, ["Neighbors", "RSM", "RSM"])
        self.assertEqual(
            [index[1] for index in table.index],
            [measure_label(name) for name in ("MutualKNNTop10", "CKA", "RSA")],
        )


class TestLatexRendering(unittest.TestCase):
    def test_rendered_table_matches_the_golden_file(self):
        self.assertTrue(
            pd.__version__.startswith(PANDAS_VERSION),
            f"golden LaTeX was captured for pandas {PANDAS_VERSION}.x, "
            f"got {pd.__version__}",
        )
        table = build_overview_table(build_display_frame(_normalized_values()))
        rendered = render_latex_table(table, "Overview", "tab:overview")
        self.assertEqual(rendered, GOLDEN_LATEX)

    def test_group_labels_move_to_the_bottom_of_their_block(self):
        table = build_overview_table(build_display_frame(_normalized_values()))
        lines = render_latex_table(table, "Overview", "tab:overview").split("\n")
        rsm_rows = [line for line in lines if "RSM" in line]
        self.assertEqual(len(rsm_rows), 1)
        # A negative span places the label on the block's last row.
        self.assertIn(r"\multirow[c]{-2}{*}{RSM}", rsm_rows[0])
        self.assertIn("RSA", rsm_rows[0])

    def test_writing_creates_missing_directories(self):
        table = build_overview_table(build_display_frame(_normalized_values()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tables" / "overview.tex"
            written = write_latex_table(table, path, "Overview", "tab:overview")
            self.assertEqual(written, path)
            self.assertEqual(path.read_text(encoding="utf-8"), GOLDEN_LATEX)

    def test_empty_tables_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty table"):
            render_latex_table(pd.DataFrame(), "Overview", "tab:overview")


class TestRanking(unittest.TestCase):
    def test_best_measure_ranks_first(self):
        display = build_display_frame(_normalized_values())
        summary = summarize_ranks(rank_measures(display))
        self.assertEqual(summary["Sim Meas."].iloc[0], measure_label("MutualKNNTop10"))
        self.assertEqual(
            list(summary.columns), ["Domain", "Sim Meas.", "avg_rank", "med_rank"]
        )
        self.assertTrue((summary["med_rank"].diff().dropna() >= 0).all())

    def test_token_aware_ranking_requires_a_token_column(self):
        display = build_display_frame(_normalized_values())
        with self.assertRaisesRegex(ValueError, "Token"):
            rank_measures(display, token_aware=True)

    def test_layer_models_are_averaged_before_deduplication(self):
        rows = []
        for model, values in (
            ("layer1", {"CKA": 0.2, "RSA": 0.9}),
            ("layer2", {"CKA": 0.9, "RSA": 0.2}),
        ):
            for metric, value in values.items():
                rows.append(
                    {
                        "domain": "graphs",
                        "benchmark": "monotonicity",
                        "dataset": "cora",
                        "architecture": "GCN",
                        "metric": metric,
                        "quality_measure": "correlation",
                        "value": value,
                        "model": model,
                        "functional_similarity_measure": "",
                    }
                )
        ranked = rank_measures(build_display_frame(pd.DataFrame(rows)))
        # Each measure wins one layer and loses the other, so both average to 1.5.
        self.assertEqual(sorted(ranked["rank"].unique().tolist()), [1.5])

    def test_quality_measures_rank_separately_within_a_cell(self):
        # One benchmark reporting two quality measures is two evaluations, not
        # one pool of twice as many rows. Pooling them would let a measure's
        # AUPRC outrank another measure's correlation, and would push ranks
        # past the number of measures being compared.
        rows = [
            {
                "domain": "graphs",
                "benchmark": "output_correlation_test",
                "dataset": "cora",
                "architecture": "GCN",
                "metric": metric,
                "quality_measure": quality,
                "value": value,
                "functional_similarity_measure": "jsd",
            }
            for quality, values in (
                ("spearmanr", {"CKA": 0.9, "RSA": 0.1}),
                ("AUPRC", {"CKA": 0.2, "RSA": 0.8}),
            )
            for metric, value in values.items()
        ]
        ranked = rank_measures(build_display_frame(pd.DataFrame(rows)))
        self.assertEqual(ranked["rank"].max(), 2.0)
        by_quality = ranked.set_index(["Eval.", "Sim Meas."])["rank"]
        cka = measure_label("CKA")
        self.assertEqual(by_quality[("Spearman", cka)], 1.0)
        self.assertEqual(by_quality[("AUPRC", cka)], 2.0)

    def test_functional_similarity_measures_are_kept_apart(self):
        # A correlation test reports each quality measure once per functional
        # similarity measure. Deduplicating without that column keeps one at
        # random and silently drops the others.
        rows = [
            {
                "domain": "graphs",
                "benchmark": "output_correlation_test",
                "dataset": "cora",
                "architecture": "GCN",
                "metric": metric,
                "quality_measure": "spearmanr",
                "value": value,
                "functional_similarity_measure": functional,
            }
            for functional, values in (
                ("jsd", {"CKA": 0.9, "RSA": 0.1}),
                ("disagreement", {"CKA": 0.1, "RSA": 0.9}),
            )
            for metric, value in values.items()
        ]
        ranked = rank_measures(build_display_frame(pd.DataFrame(rows)))
        self.assertEqual(len(ranked), 4)
        self.assertEqual(
            sorted(ranked["Functional Similarity Measure"].unique()),
            ["disagreement", "jsd"],
        )
        by_functional = ranked.set_index(
            ["Functional Similarity Measure", "Sim Meas."]
        )["rank"]
        cka = measure_label("CKA")
        self.assertEqual(by_functional[("jsd", cka)], 1.0)
        self.assertEqual(by_functional[("disagreement", cka)], 2.0)


class TestRankFigure(unittest.TestCase):
    def test_one_file_is_written_after_every_panel_is_drawn(self):
        display = build_display_frame(_normalized_values())
        ranked = rank_measures(display)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "figures" / "ranks.png"
            written = plot_rank_distributions(
                ranked, path, modalities=("Graph", "Vision")
            )
            self.assertEqual(written, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            self.assertEqual(len(list(path.parent.glob("*.png"))), 1)


class TestAppendixReport(unittest.TestCase):
    """The report writes one table per test plus both NaN conventions' figures."""

    def test_writes_a_table_and_a_value_figure_per_test(self):
        with tempfile.TemporaryDirectory() as directory:
            index = write_appendix_report(_normalized_values(), directory)
            tables = Path(directory) / "tables"
            # This fixture scores AbsoluteAccDiff only, so `Acc Corr.` is the
            # correlation task it writes and `Output Corr.` is recorded missing.
            self.assertEqual(
                sorted(index["test"]),
                ["Acc Corr.", "Augmentation", "Layer Mono.", "Output Corr."],
            )
            written = index[index["csv_path"].ne("")]
            self.assertEqual(
                sorted(written["test"]),
                ["Acc Corr.", "Augmentation", "Layer Mono."],
            )
            for _, row in written.iterrows():
                self.assertTrue(Path(row["csv_path"]).is_file())
                self.assertTrue(Path(row["figure_path"]).is_file())
            self.assertTrue((tables / "published_ranks.csv").is_file())
            self.assertTrue((tables / "rank_boxplot_published.png").is_file())
            self.assertTrue((tables / "tables_index.csv").is_file())

    def test_the_two_output_measures_share_one_table_that_names_them(self):
        """Grouped, but not pooled.

        JSD and Disagreement are two metrics of one task, so they share the
        `Output Corr.` table -- but each keeps its own named `Eval.` block.
        Without that naming the columns pool two different functional measures
        under one heading, which is the defect this taxonomy has to avoid.
        Accuracy difference asks a different question and gets its own table.
        """
        rows = []
        for functional in ("JSD", "Disagreement"):
            frame = _normalized_values()
            frame = frame[frame["benchmark"].eq("accoutput")].copy()
            frame["functional_similarity_measure"] = functional
            rows.append(frame)
        values = pd.concat([_normalized_values(), *rows], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            index = write_appendix_report(values, directory)
            tables = Path(directory) / "tables"
            self.assertIn("Output Corr.", set(index["test"]))
            # The accuracy task is a separate file, not a block of this one.
            self.assertIn("Acc Corr.", set(index["test"]))
            self.assertTrue((tables / "output-corr.csv").is_file())
            self.assertTrue((tables / "output-corr_values.png").is_file())
            self.assertTrue((tables / "acc-corr.csv").is_file())
            table = pd.read_csv(tables / "output-corr.csv", header=[0, 1, 2, 3])
        levels = {name for column in table.columns for name in column}
        # Both output measures are present and named, and the accuracy one is
        # nowhere in this table.
        self.assertIn("JSD", levels)
        self.assertIn("Disagreement", levels)
        self.assertNotIn("Acc Diff", levels)

    def test_a_correlation_setting_with_no_values_is_reported_as_missing(self):
        """A missing table has to read as a recorded gap, not as an absent file.

        Graphs excludes AbsoluteAccDiff outright and language has no SmolLM2
        accuracy to difference, so those domains write no accuracy-correlation
        table at all. Nothing distinguishes that from a table nobody generated
        unless the index says so.
        """
        # The fixture scores AbsoluteAccDiff only, so the output task of the
        # same benchmark has nothing to report.
        values = _normalized_values()
        with tempfile.TemporaryDirectory() as directory:
            index = write_appendix_report(values, directory)
        missing = index[index["note"].str.startswith("not reported")]
        # One row for the absent task, naming both measures it would cover --
        # not one row per functional measure pointing at the same missing file.
        self.assertEqual(sorted(missing["test"]), ["Output Corr."])
        self.assertIn("JSD or Disagreement", missing["note"].iloc[0])
        for _, row in missing.iterrows():
            self.assertEqual(row["rows"], 0)
            self.assertEqual(row["csv_path"], "")
            self.assertEqual(row["latex_path"], "")

    def test_no_missing_split_is_reported_without_a_correlation_benchmark(self):
        values = _normalized_values()
        values = values[values["benchmark"].ne("accoutput")]
        with tempfile.TemporaryDirectory() as directory:
            index = write_appendix_report(values, directory)
        self.assertEqual(index["note"].str.startswith("not reported").sum(), 0)

    def test_a_group_holding_one_measure_still_lays_out(self):
        """Pandas writes no multirow for a one-measure group.

        The collapsed "Manifold" block almost never produced one; the shared
        taxonomy produces them routinely, and a block recovered by scanning for
        multirows loses both its separating rule and, when it comes first, the
        header/body boundary.
        """
        values = _normalized_values()
        values = values[values["metric"].eq("CKA")]
        with tempfile.TemporaryDirectory() as directory:
            index = write_appendix_report(values, directory)
            for _, row in index[index["csv_path"].ne("")].iterrows():
                self.assertTrue(Path(row["csv_path"]).is_file())
                self.assertTrue(Path(row["latex_path"]).is_file())
                self.assertEqual(row["note"], "")

    def test_published_ranking_keeps_a_measure_the_strict_filter_drops(self):
        """The two NaN conventions must be able to disagree about the field.

        The published tables exist to show what the strict matched-observation
        ranking excludes, so a measure that is NaN in one case has to survive
        here even though `filter_measures_by_coverage` would remove it.
        """
        values = _normalized_values()
        sparse = values[values["metric"].eq("CKA")].copy()
        sparse["metric"] = "Sparse"
        sparse["value"] = float("nan")
        values = pd.concat([values, sparse], ignore_index=True)
        with tempfile.TemporaryDirectory() as directory:
            write_appendix_report(values, directory)
            ranks = pd.read_csv(Path(directory) / "tables" / "published_ranks.csv")
        self.assertIn("Sparse", set(ranks["Sim Meas."]))

    def test_empty_values_write_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            index = write_appendix_report(pd.DataFrame(), directory)
            self.assertTrue(index.empty)
            self.assertFalse((Path(directory) / "tables").exists())


class TestTaskValueFigure(unittest.TestCase):
    def test_one_panel_per_evaluation_measure(self):
        display = build_display_frame(_normalized_values())
        with tempfile.TemporaryDirectory() as directory:
            path = plot_task_value_distributions(
                display, Path(directory) / "values.png", test="Augmentation"
            )
            self.assertIsNotNone(path)
            self.assertTrue(path.is_file())

    def test_nothing_to_draw_returns_none(self):
        display = build_display_frame(_normalized_values())
        with tempfile.TemporaryDirectory() as directory:
            path = plot_task_value_distributions(
                display, Path(directory) / "values.png", test="Nonexistent Test"
            )
            self.assertIsNone(path)
            self.assertFalse((Path(directory) / "values.png").exists())


class TestTablesImportIsolation(unittest.TestCase):
    def test_table_construction_does_not_require_a_plotting_backend(self):
        import subprocess
        import sys

        code = (
            "import builtins\n"
            "_real = builtins.__import__\n"
            "def _guard(name, *args, **kwargs):\n"
            "    if name.split('.')[0] == 'matplotlib':\n"
            "        raise ImportError('matplotlib is unavailable')\n"
            "    return _real(name, *args, **kwargs)\n"
            "builtins.__import__ = _guard\n"
            "from manifold_repsim.resi.analysis.tables import ("
            "build_display_frame, build_overview_table, render_latex_table)\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


class TestTableFigureBoxStyle(unittest.TestCase):
    """The appendix figures follow the run's boxplot style like every other."""

    def _frame(self):
        return pd.DataFrame(
            {
                "Domain": ["Graph"] * 6,
                "Test": ["Augmentation"] * 6,
                "Eval.": ["AUPRC"] * 6,
                "Sim Meas.": ["CKA", "CKA", "CKA", "SVCCA", "SVCCA", "SVCCA"],
                "value": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
                "rank": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            }
        )

    def test_the_style_reaches_the_task_value_figure(self):
        from manifold_repsim.resi.analysis.tables.figures import (
            plot_task_value_distributions,
        )

        with tempfile.TemporaryDirectory() as tmp:
            box = plot_task_value_distributions(
                self._frame(), Path(tmp) / "box.png", test="Augmentation"
            )
            boxen = plot_task_value_distributions(
                self._frame(),
                Path(tmp) / "boxen.png",
                test="Augmentation",
                box_style="boxen",
            )
            self.assertTrue(box.exists() and boxen.exists())
            self.assertNotEqual(box.read_bytes(), boxen.read_bytes())

    def test_the_style_reaches_the_published_rank_figure(self):
        from manifold_repsim.resi.analysis.tables.figures import (
            plot_rank_distributions,
        )

        with tempfile.TemporaryDirectory() as tmp:
            box = plot_rank_distributions(
                self._frame(), Path(tmp) / "box.png", modalities=["Graph"]
            )
            boxen = plot_rank_distributions(
                self._frame(),
                Path(tmp) / "boxen.png",
                modalities=["Graph"],
                box_style="boxen",
            )
            self.assertNotEqual(box.read_bytes(), boxen.read_bytes())

    def test_an_unknown_style_is_rejected_here_too(self):
        from manifold_repsim.resi.analysis.tables.figures import (
            plot_task_value_distributions,
        )

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "box_style must be one of"):
                plot_task_value_distributions(
                    self._frame(),
                    Path(tmp) / "bad.png",
                    test="Augmentation",
                    box_style="violin",
                )


if __name__ == "__main__":
    unittest.main()
