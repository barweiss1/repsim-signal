"""Cross-domain grouping and figures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from manifold_repsim.resi.analysis.config import measure_label
from manifold_repsim.resi.analysis.cross_domain import (
    ALL_TASKS,
    TEST_GROUPS,
    aggregate_across_domains,
    combine_case_ranks,
    combine_domain_values,
    common_functional_measures,
    drop_unpublished_quality,
    average_subtask_ranks,
    subtask_balance,
    summarize_cross_domain,
)
from manifold_repsim.resi.analysis.cross_domain_figures import (
    plot_cross_domain_ranks,
    plot_task_ranks_by_type,
    plot_task_values_by_type,
    plot_test_group,
    write_by_type_figures,
    write_cross_domain_figures,
)
from manifold_repsim.resi.analysis.cross_domain_tables import (
    build_task_table,
    render_latex,
    write_cross_domain_tables,
)


def _domain_frame(benchmark: str, quality: str, values: dict[str, float]):
    return pd.DataFrame(
        [
            {
                "benchmark": benchmark,
                "quality_measure": quality,
                "metric": metric,
                "value": value,
            }
            for metric, value in values.items()
        ]
    )


class TestTestGroups(unittest.TestCase):
    def test_every_domains_benchmark_names_reach_the_same_five_groups(self):
        """The three domains name one experiment three ways; all must land together."""
        graphs = [
            "output_correlation_test",
            "augmentation_test",
            "layer_test",
            "label_test",
            "shortcut_test",
        ]
        vision = [
            "accoutput",
            "augmentation",
            "monotonicity",
            "randomlabel",
            "shortcut",
        ]
        language = [
            "correlation",
            "augmentation",
            "monotonicity",
            "memorization",
            "shortcut",
        ]
        for names in (graphs, vision, language):
            self.assertEqual(len(names), 5)
            self.assertTrue(set(names) <= set(TEST_GROUPS))
        for a, b, c in zip(graphs, vision, language):
            self.assertEqual(TEST_GROUPS[a], TEST_GROUPS[b])
            self.assertEqual(TEST_GROUPS[b], TEST_GROUPS[c])
        self.assertEqual(len(set(TEST_GROUPS.values())), 5)

    def test_unknown_benchmark_is_dropped_rather_than_guessed(self):
        frame = _domain_frame("a_brand_new_test", "AUPRC", {"CKA": 0.5})
        combined = combine_domain_values({"graphs": frame})
        self.assertTrue(combined.empty)


class TestCombineDomainValues(unittest.TestCase):
    def test_violation_rate_is_reported_as_conformity(self):
        frame = _domain_frame("shortcut_test", "violation_rate", {"CKA": 0.25})
        combined = combine_domain_values({"graphs": frame})
        self.assertAlmostEqual(float(combined["value"].iloc[0]), 0.75)

    def test_other_quality_measures_are_untouched(self):
        frame = _domain_frame("shortcut_test", "AUPRC", {"CKA": 0.25})
        combined = combine_domain_values({"graphs": frame})
        self.assertAlmostEqual(float(combined["value"].iloc[0]), 0.25)

    def test_domains_stack_under_one_group(self):
        combined = combine_domain_values(
            {
                "graphs": _domain_frame("shortcut_test", "AUPRC", {"CKA": 0.2}),
                "vision": _domain_frame("shortcut", "AUPRC", {"CKA": 0.4}),
                "language": _domain_frame("shortcut", "AUPRC", {"CKA": 0.6}),
            }
        )
        self.assertEqual(set(combined["test_group"]), {"Shortcuts"})
        self.assertEqual(set(combined["domain"]), {"graphs", "vision", "language"})
        self.assertEqual(len(combined), 3)

    def test_empty_input_returns_an_empty_frame(self):
        self.assertTrue(combine_domain_values({}).empty)
        self.assertTrue(combine_domain_values({"graphs": pd.DataFrame()}).empty)


class TestFunctionalHarmonization(unittest.TestCase):
    """Correlation panels must pool the same functional measures everywhere."""

    def _frames(self):
        def frame(benchmark, functionals):
            return pd.DataFrame(
                [
                    {
                        "benchmark": benchmark,
                        "quality_measure": "spearmanr",
                        "metric": "CKA",
                        "value": 0.5,
                        "functional_similarity_measure": name,
                    }
                    for name in functionals
                ]
            )

        return {
            "graphs": frame("output_correlation_test", ["JSD", "Disagreement"]),
            "vision": frame("accoutput", ["JSD", "Disagreement", "AbsoluteAccDiff"]),
        }

    def test_common_measures_are_the_intersection(self):
        self.assertEqual(
            common_functional_measures(self._frames()),
            {"JSD", "Disagreement"},
        )

    def test_a_domain_only_functional_measure_is_kept_in_its_own_family(self):
        """Vision's accuracy rows survive without unbalancing the output family."""
        combined = combine_domain_values(self._frames())
        self.assertEqual(len(combined), 5)
        output = combined[combined["test_group"].eq("Output Corr.")]
        self.assertEqual(
            output.groupby("domain").size().to_dict(), {"graphs": 2, "vision": 2}
        )
        accuracy = combined[combined["test_group"].eq("Acc Corr.")]
        self.assertEqual(accuracy["domain"].tolist(), ["vision"])

    def test_harmonization_can_be_turned_back_on(self):
        """The old intersection is still reachable, it is just no longer default."""
        combined = combine_domain_values(self._frames(), harmonize_functional=True)
        self.assertEqual(len(combined), 4)
        self.assertEqual(
            combined.groupby("domain").size().to_dict(), {"graphs": 2, "vision": 2}
        )
        self.assertNotIn("Acc Corr.", set(combined["test_group"]))

    def test_rows_without_a_functional_measure_survive(self):
        frames = {
            "graphs": pd.DataFrame(
                [
                    {
                        "benchmark": "shortcut_test",
                        "quality_measure": "AUPRC",
                        "metric": "CKA",
                        "value": 0.5,
                        "functional_similarity_measure": "",
                    }
                ]
            ),
            "vision": pd.DataFrame(
                [
                    {
                        "benchmark": "accoutput",
                        "quality_measure": "spearmanr",
                        "metric": "CKA",
                        "value": 0.5,
                        "functional_similarity_measure": "JSD",
                    }
                ]
            ),
        }
        # No functional measure is shared, but the AUPRC row carries none and
        # must not be filtered out by a correlation-only concern.
        combined = combine_domain_values(frames)
        self.assertIn("graphs", set(combined["domain"]))


class TestPublishedQualityMeasures(unittest.TestCase):
    """ReSi reports prediction correlation with Spearman only."""

    def _values(self):
        def frame(benchmark):
            return pd.concat(
                [
                    _domain_frame(benchmark, quality, {"CKA": 0.5})
                    for quality in ("spearmanr", "pearsonr", "kendalltau")
                ],
                ignore_index=True,
            )

        return {
            "graphs": frame("output_correlation_test"),
            "vision": frame("accoutput"),
        }

    def _ranks(self):
        def frame(domain, benchmark):
            return pd.DataFrame(
                [
                    {
                        "domain": domain,
                        "benchmark": benchmark,
                        "case_id": f"{domain}-{quality}",
                        "quality_measure": quality,
                        "metric": "CKA",
                        "mean_normalized_rank": 0.5,
                        "mean_rank": 1.5,
                    }
                    for quality in ("spearmanr", "pearsonr", "kendalltau")
                ]
            )

        return {
            "graphs": frame("graphs", "output_correlation_test"),
            "vision": frame("vision", "accoutput"),
        }

    def test_helper_keeps_only_the_reported_quality_measures(self):
        frame = pd.DataFrame(
            {"quality_measure": ["spearmanr", "pearsonr", "kendalltau", "AUPRC"]}
        )
        kept = drop_unpublished_quality(frame)
        self.assertEqual(list(kept["quality_measure"]), ["spearmanr", "AUPRC"])

    def test_helper_ignores_a_frame_without_the_column(self):
        frame = pd.DataFrame({"metric": ["CKA"]})
        self.assertEqual(len(drop_unpublished_quality(frame)), 1)

    def test_values_drop_pearson_and_kendall(self):
        combined = combine_domain_values(self._values())
        self.assertEqual(set(combined["quality_measure"]), {"spearmanr"})

    def test_values_filter_can_be_turned_off(self):
        combined = combine_domain_values(self._values(), published_quality_only=False)
        self.assertEqual(
            set(combined["quality_measure"]),
            {"spearmanr", "pearsonr", "kendalltau"},
        )

    def test_case_ranks_drop_two_thirds_of_the_correlation_cases(self):
        """A case is keyed on the quality measure, so these are weighting units."""
        combined = combine_case_ranks(self._ranks())
        self.assertEqual(combined["case_id"].nunique(), 2)
        self.assertEqual(
            combined.groupby("domain")["case_id"].nunique().to_dict(),
            {"graphs": 1, "vision": 1},
        )
        kept = combine_case_ranks(self._ranks(), published_quality_only=False)
        self.assertEqual(kept["case_id"].nunique(), 6)

    def test_auprc_and_monotonicity_cases_are_untouched(self):
        frames = {
            "graphs": pd.DataFrame(
                [
                    {
                        "domain": "graphs",
                        "benchmark": benchmark,
                        "case_id": f"graphs-{quality}",
                        "quality_measure": quality,
                        "metric": "CKA",
                        "mean_normalized_rank": 0.5,
                        "mean_rank": 1.5,
                    }
                    for benchmark, quality in (
                        ("shortcut_test", "AUPRC"),
                        ("layer_test", "correlation"),
                        ("layer_test", "violation_rate"),
                    )
                ]
            )
        }
        combined = combine_case_ranks(frames)
        self.assertEqual(combined["case_id"].nunique(), 3)


def _case_row(
    domain, benchmark, dataset, architecture, quality, metric, rank, *, fsm=""
):
    return {
        "domain": domain,
        "benchmark": benchmark,
        "dataset": dataset,
        "architecture": architecture,
        "quality_measure": quality,
        "functional_similarity_measure": fsm,
        "case_id": f"{domain}|{benchmark}|{dataset}|{architecture}|{quality}|{fsm}",
        "metric": metric,
        "mean_normalized_rank": rank,
        # A two-measure field, so rank 1 is normalized 0.0 and rank 2 is 1.0.
        "mean_rank": 1.0 + rank,
    }


class TestSubtaskAveraging(unittest.TestCase):
    """A case counts architectures, not evidence; the subtask is the vote."""

    def test_architectures_collapse_into_one_mean(self):
        frames = {
            "vision": pd.DataFrame(
                [
                    _case_row(
                        "vision", "shortcut", "CIFAR100", arch, "AUPRC", "CKA", rank
                    )
                    for arch, rank in (("ResNet18", 0.2), ("VGG11", 0.8))
                ]
            )
        }
        subtasks = average_subtask_ranks(combine_case_ranks(frames))
        self.assertEqual(len(subtasks), 1)
        self.assertAlmostEqual(float(subtasks["mean_normalized_rank"].iloc[0]), 0.5)
        self.assertEqual(int(subtasks["n_cases"].iloc[0]), 2)

    def test_quality_measures_stay_separate_units(self):
        """An AUPRC rank and a conformity-rate rank are not the same question."""
        frames = {
            "vision": pd.DataFrame(
                [
                    _case_row(
                        "vision",
                        "shortcut",
                        "CIFAR100",
                        "ResNet18",
                        quality,
                        "CKA",
                        0.5,
                    )
                    for quality in ("AUPRC", "violation_rate")
                ]
            )
        }
        subtasks = average_subtask_ranks(combine_case_ranks(frames))
        self.assertEqual(subtasks["subtask_id"].nunique(), 2)

    def test_datasets_stay_separate_units(self):
        frames = {
            "vision": pd.DataFrame(
                [
                    _case_row(
                        "vision", "shortcut", dataset, "ResNet18", "AUPRC", "CKA", 0.5
                    )
                    for dataset in ("CIFAR100", "ImageNet100")
                ]
            )
        }
        subtasks = average_subtask_ranks(combine_case_ranks(frames))
        self.assertEqual(subtasks["subtask_id"].nunique(), 2)

    def test_an_incomplete_metric_reports_a_smaller_n_cases(self):
        """Not a filter: coverage filtering already ran, this stays visible."""
        rows = [
            _case_row("vision", "shortcut", "CIFAR100", arch, "AUPRC", "CKA", 0.4)
            for arch in ("ResNet18", "VGG11")
        ]
        rows.append(
            _case_row(
                "vision", "shortcut", "CIFAR100", "ResNet18", "AUPRC", "SVCCA", 0.9
            )
        )
        subtasks = average_subtask_ranks(
            combine_case_ranks({"vision": pd.DataFrame(rows)})
        )
        counts = subtasks.set_index("metric")["n_cases"].to_dict()
        self.assertEqual(counts, {"CKA": 2, "SVCCA": 1})

    def test_the_split_is_balanced_per_family_and_per_domain(self):
        """A correct total can hide a wrong split -- assert the split."""
        rows = []
        layout = {
            "graphs": (["cora", "flickr", "arxiv"], "shortcut_test", ["GCN", "GAT"]),
            "vision": (["CIFAR100", "ImageNet100"], "shortcut", ["ResNet18"]),
            "language": (["sst2", "mnli"], "shortcut", ["BERT-L"]),
        }
        for domain, (datasets, benchmark, architectures) in layout.items():
            for dataset in datasets:
                for architecture in architectures:
                    for quality in ("AUPRC", "violation_rate"):
                        rows.append(
                            _case_row(
                                domain,
                                benchmark,
                                dataset,
                                architecture,
                                quality,
                                "CKA",
                                0.5,
                            )
                        )
        frames = {
            domain: pd.DataFrame([r for r in rows if r["domain"] == domain])
            for domain in layout
        }
        subtasks = average_subtask_ranks(combine_case_ranks(frames))
        balance = subtask_balance(subtasks)
        self.assertEqual(int(balance.loc["Shortcuts", "graphs"]), 6)
        self.assertEqual(int(balance.loc["Shortcuts", "vision"]), 4)
        self.assertEqual(int(balance.loc["Shortcuts", "language"]), 4)
        self.assertEqual(int(balance.loc["Shortcuts", "TOTAL"]), 14)

    def test_case_ranks_carry_the_subtask_columns_through(self):
        frames = {
            "vision": pd.DataFrame(
                [
                    _case_row(
                        "vision",
                        "shortcut",
                        "CIFAR100",
                        "ResNet18",
                        "AUPRC",
                        "CKA",
                        0.5,
                    )
                ]
            )
        }
        combined = combine_case_ranks(frames)
        for column in ("dataset", "quality_measure", "functional_similarity_measure"):
            self.assertIn(column, combined.columns)

    def test_aggregate_counts_subtasks_and_the_cases_behind_them(self):
        frames = {
            "vision": pd.DataFrame(
                [
                    _case_row(
                        "vision", "shortcut", "CIFAR100", arch, "AUPRC", metric, rank
                    )
                    for arch, rank in (("ResNet18", 0.2), ("VGG11", 0.8))
                    for metric, rank in ((m, rank) for m in ("CKA", "SVCCA"))
                ]
            )
        }
        aggregate = aggregate_across_domains(
            average_subtask_ranks(combine_case_ranks(frames))
        )
        overall = aggregate[aggregate["test_group"].eq(ALL_TASKS)].set_index("metric")
        self.assertEqual(int(overall.loc["CKA", "n_subtasks"]), 1)
        self.assertEqual(int(overall.loc["CKA", "n_cases"]), 2)

    def test_empty_inputs_are_safe(self):
        empty = average_subtask_ranks(pd.DataFrame())
        self.assertTrue(empty.empty)
        self.assertIn("subtask_id", empty.columns)
        self.assertTrue(subtask_balance(empty).empty)


class TestSummary(unittest.TestCase):
    def test_summary_reports_one_row_per_domain_and_measure(self):
        combined = combine_domain_values(
            {
                "graphs": _domain_frame(
                    "shortcut_test", "AUPRC", {"CKA": 0.2, "SVCCA": 0.9}
                ),
                "vision": _domain_frame(
                    "shortcut", "AUPRC", {"CKA": 0.4, "SVCCA": 0.1}
                ),
            }
        )
        summary = summarize_cross_domain(combined)
        self.assertEqual(len(summary), 4)
        self.assertEqual(set(summary["domain"]), {"graphs", "vision"})
        row = summary[
            summary["domain"].eq("vision") & summary["metric"].eq("CKA")
        ].iloc[0]
        self.assertAlmostEqual(float(row["median"]), 0.4)

    def test_empty_summary_keeps_its_columns(self):
        summary = summarize_cross_domain(pd.DataFrame())
        self.assertTrue(summary.empty)
        self.assertIn("median", summary.columns)


class TestFigures(unittest.TestCase):
    def _combined(self):
        return combine_domain_values(
            {
                "graphs": pd.concat(
                    [
                        _domain_frame(
                            "shortcut_test", "AUPRC", {"CKA": 0.2, "SVCCA": 0.9}
                        ),
                        _domain_frame("shortcut_test", "violation_rate", {"CKA": 0.3}),
                    ]
                ),
                "vision": _domain_frame(
                    "shortcut", "AUPRC", {"CKA": 0.4, "SVCCA": 0.1}
                ),
            }
        )

    def test_one_figure_per_group_plus_an_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = write_cross_domain_figures(self._combined(), directory)
            self.assertEqual(list(index["test_group"]), ["Shortcuts"])
            written = Path(index["path"].iloc[0])
            self.assertTrue(written.is_file())
            self.assertGreater(written.stat().st_size, 0)
            self.assertTrue((directory / "cross_domain_index.csv").is_file())
            # Two quality measures reported, so two panels.
            self.assertEqual(int(index["panels"].iloc[0]), 2)

    def test_a_group_missing_a_domain_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = write_cross_domain_figures(self._combined(), Path(tmp))
        self.assertIn("only 2 of 3 domains", index["note"].iloc[0])

    def test_absent_group_draws_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_test_group(self._combined(), "Layer Mono.", Path(tmp) / "x.png")
        self.assertIsNone(path)

    def test_empty_values_write_no_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = write_cross_domain_figures(pd.DataFrame(), Path(tmp))
            self.assertTrue(index.empty)
            self.assertFalse((Path(tmp) / "cross_domain_index.csv").exists())


if __name__ == "__main__":
    unittest.main()


class TestCrossDomainTables(unittest.TestCase):
    def _values(self):
        return combine_domain_values(
            {
                "graphs": pd.concat(
                    [
                        _domain_frame(
                            "shortcut_test", "AUPRC", {"CKA": 0.2, "SVCCA": 0.9}
                        ),
                        _domain_frame(
                            "shortcut_test",
                            "violation_rate",
                            {"CKA": 0.3, "SVCCA": 0.1},
                        ),
                    ]
                ),
                "vision": _domain_frame(
                    "shortcut", "AUPRC", {"CKA": 0.4, "SVCCA": 0.1}
                ),
            }
        )

    def test_table_is_measures_by_domain_and_quality(self):
        table = build_task_table(self._values(), "Shortcuts")
        self.assertEqual(list(table.index.names), ["Group", "Sim Meas."])
        self.assertEqual(list(table.columns.names), ["Domain", "Eval."])
        # Graph reports two quality measures, Vision one.
        self.assertEqual(table.shape, (2, 3))
        cell = table.loc[("RSM", measure_label("CKA")), ("Vision", "AUPRC*")]
        self.assertEqual(cell, 0.4)

    def test_conformity_is_the_complement_of_the_violation_rate(self):
        table = build_task_table(self._values(), "Shortcuts")
        self.assertAlmostEqual(
            table.loc[("RSM", measure_label("CKA")), ("Graph", "Conformity Rate")],
            0.7,
        )

    def test_best_in_column_is_bolded_and_ties_share_it(self):
        values = combine_domain_values(
            {
                "graphs": _domain_frame(
                    "shortcut_test", "AUPRC", {"CKA": 0.804, "SVCCA": 0.799}
                )
            }
        )
        latex = render_latex(build_task_table(values, "Shortcuts"), "Shortcuts")
        # Both round to 0.80, so bolding one and not the other would show the
        # reader a winner they cannot see in the printed digits.
        self.assertEqual(latex.count(r"\textbf{0.80}"), 2)

    def test_missing_cells_render_as_a_dash(self):
        # Vision reports CKA but not SVCCA, the shape LinearRegression has on
        # the real vision run, so that cell has nothing to print.
        values = combine_domain_values(
            {
                "graphs": _domain_frame(
                    "shortcut_test", "AUPRC", {"CKA": 0.2, "SVCCA": 0.9}
                ),
                "vision": _domain_frame("shortcut", "AUPRC", {"CKA": 0.4}),
            }
        )
        table = build_task_table(values, "Shortcuts")
        self.assertTrue(table.isna().any().any())
        self.assertIn("--", render_latex(table, "Shortcuts"))

    def test_empty_table_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty table"):
            render_latex(pd.DataFrame(), "Shortcuts")

    def test_one_table_pair_per_group_plus_an_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = write_cross_domain_tables(self._values(), directory)
            self.assertEqual(list(index["test_group"]), ["Shortcuts"])
            self.assertTrue(Path(index["csv_path"].iloc[0]).is_file())
            self.assertTrue(Path(index["latex_path"].iloc[0]).is_file())
            self.assertTrue((directory / "cross_domain_tables_index.csv").is_file())

    def test_empty_values_write_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = write_cross_domain_tables(pd.DataFrame(), Path(tmp))
            self.assertTrue(index.empty)
            self.assertFalse((Path(tmp) / "cross_domain_tables_index.csv").exists())


class TestCrossDomainAggregation(unittest.TestCase):
    """Ranks pool across domains where values cannot."""

    def _ranks(self):
        rows = []
        for domain, benchmark, cases in (
            ("graphs", "shortcut_test", 4),
            ("vision", "shortcut", 12),
            ("language", "shortcut", 2),
        ):
            for case in range(cases):
                # CKA always wins its case, SVCCA always loses.
                for metric, rank in (("CKA", 0.0), ("SVCCA", 1.0)):
                    rows.append(
                        {
                            "domain": domain,
                            "benchmark": benchmark,
                            "case_id": f"{domain}-{case}",
                            "metric": metric,
                            "mean_normalized_rank": rank,
                            "mean_rank": 1.0 + rank,
                        }
                    )
        return pd.DataFrame(rows)

    def test_case_ranks_gain_their_family(self):
        combined = combine_case_ranks({"graphs": self._ranks()})
        self.assertEqual(set(combined["test_group"]), {"Shortcuts"})

    def test_aggregate_reports_overall_and_per_family(self):
        combined = combine_case_ranks({"graphs": self._ranks()})
        aggregate = aggregate_across_domains(combined)
        self.assertIn(ALL_TASKS, set(aggregate["test_group"]))
        self.assertIn("Shortcuts", set(aggregate["test_group"]))

    def test_pooled_and_balanced_differ_when_domains_are_uneven(self):
        """Vision runs 12 of 18 cases here, so weighting has to be a choice."""
        frame = self._ranks()
        # Make vision disagree with the other two about which measure wins, on
        # both scales, since the fixture carries both as a real frame does.
        vision = frame["domain"].eq("vision")
        frame.loc[vision, "mean_normalized_rank"] = (
            1 - frame.loc[vision, "mean_normalized_rank"]
        )
        # A two-measure field, so flipping rank 1 and rank 2 is 3 - r.
        frame.loc[vision, "mean_rank"] = 3.0 - frame.loc[vision, "mean_rank"]
        combined = combine_case_ranks(
            {name: frame[frame["domain"].eq(name)] for name in frame["domain"].unique()}
        )
        for rank_scale, midpoint in (("raw", 1.5), ("normalized", 0.5)):
            with self.subTest(rank_scale=rank_scale):
                aggregate = aggregate_across_domains(combined, rank_scale=rank_scale)
                overall = aggregate[aggregate["test_group"].eq(ALL_TASKS)].set_index(
                    "metric"
                )
                # 12 vision cases say CKA loses, 6 others say it wins.
                self.assertGreater(overall.loc["CKA", "pooled_mean"], midpoint)
                # One vote each: two domains say it wins, so balanced favours it.
                self.assertLess(overall.loc["CKA", "balanced_mean"], midpoint)

    def test_missing_domain_is_counted(self):
        frame = self._ranks()
        frame = frame[~(frame["domain"].eq("vision") & frame["metric"].eq("SVCCA"))]
        combined = combine_case_ranks(
            {name: frame[frame["domain"].eq(name)] for name in frame["domain"].unique()}
        )
        aggregate = aggregate_across_domains(combined)
        overall = aggregate[aggregate["test_group"].eq(ALL_TASKS)].set_index("metric")
        self.assertEqual(int(overall.loc["CKA", "n_domains"]), 3)
        self.assertEqual(int(overall.loc["SVCCA", "n_domains"]), 2)

    def test_an_accuracy_case_gets_its_own_family_rather_than_reweighting(self):
        """A rank case is a weighting unit, so an unshared one reweights domains.

        Splitting `Acc Corr.` off is what makes keeping it safe: the output
        family stays one case per domain, and the accuracy case weighs only
        against the domains that also report it.
        """

        def frame(domain, benchmark, functionals):
            return pd.DataFrame(
                [
                    {
                        "domain": domain,
                        "benchmark": benchmark,
                        "case_id": f"{domain}-{name}",
                        "metric": "CKA",
                        "mean_normalized_rank": 0.5,
                        "mean_rank": 1.5,
                        "functional_similarity_measure": name,
                    }
                    for name in functionals
                ]
            )

        frames = {
            "graphs": frame("graphs", "output_correlation_test", ["JSD"]),
            "vision": frame("vision", "accoutput", ["JSD", "AbsoluteAccDiff"]),
        }
        combined = combine_case_ranks(frames)
        # Every case survives, unlike under the old intersection.
        self.assertEqual(combined["case_id"].nunique(), 3)
        # The output family is still one case per domain -- the balance the
        # intersection used to buy by deleting vision's accuracy case.
        output = combined[combined["test_group"].eq("Output Corr.")]
        self.assertEqual(
            output.groupby("domain")["case_id"].nunique().to_dict(),
            {"graphs": 1, "vision": 1},
        )
        # And the accuracy case sits in a family only vision contributes to.
        accuracy = combined[combined["test_group"].eq("Acc Corr.")]
        self.assertEqual(accuracy["domain"].unique().tolist(), ["vision"])
        harmonized = combine_case_ranks(frames, harmonize_functional=True)
        self.assertEqual(harmonized["case_id"].nunique(), 2)

    def test_empty_inputs_are_safe(self):
        self.assertTrue(combine_case_ranks({}).empty)
        self.assertTrue(aggregate_across_domains(pd.DataFrame()).empty)

    def test_rank_figure_is_written(self):
        combined = combine_case_ranks({"graphs": self._ranks()})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ranks.png"
            written = plot_cross_domain_ranks(combined, path)
            self.assertEqual(written, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_rank_figure_of_nothing_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                plot_cross_domain_ranks(pd.DataFrame(), Path(tmp) / "x.png")
            )


class TestByTypeFigures(unittest.TestCase):
    """Type-coloured figures, each panel sorted by its own median."""

    def _ranks(self):
        rows = []
        for case in range(4):
            for metric, rank in (
                ("CKA", 0.1),
                ("SVCCA", 0.9),
                ("MutualKNNTop10", 0.5),
            ):
                rows.append(
                    {
                        "domain": "graphs",
                        "benchmark": "shortcut_test",
                        "case_id": f"c{case}",
                        "metric": metric,
                        "mean_normalized_rank": rank,
                        "mean_rank": 1.0 + 2.0 * rank,
                    }
                )
        return combine_case_ranks({"graphs": pd.DataFrame(rows)})

    def _values(self):
        return combine_domain_values(
            {
                "graphs": pd.concat(
                    [
                        _domain_frame(
                            "shortcut_test",
                            "AUPRC",
                            {"CKA": 0.9, "SVCCA": 0.1, "MutualKNNTop10": 0.5},
                        ),
                        _domain_frame(
                            "shortcut_test",
                            "violation_rate",
                            {"CKA": 0.5, "SVCCA": 0.2, "MutualKNNTop10": 0.9},
                        ),
                    ]
                )
            }
        )

    def _labels(self, column, higher_is_better, frame):
        """Y tick labels bottom-to-top for one drawn panel."""
        import matplotlib.pyplot as plt

        from manifold_repsim.resi.analysis.boxplots import measure_panel

        figure, ax = plt.subplots()
        measure_panel(ax, frame, column, higher_is_better=higher_is_better)
        labels = [text.get_text() for text in ax.get_yticklabels()]
        plt.close(figure)
        return labels

    def test_rank_panel_puts_the_lowest_rank_on_top(self):
        # Rank is best at 0, and position 0 is the bottom of the axis, so the
        # best measure has to be the last label.
        labels = self._labels("mean_normalized_rank", False, self._ranks())
        self.assertEqual(labels[-1], measure_label("CKA"))
        self.assertEqual(labels[0], "SVCCA")

    def test_value_panel_puts_the_highest_value_on_top(self):
        values = self._values()
        auprc = values[values["quality_measure"].eq("AUPRC")]
        labels = self._labels("value", True, auprc)
        self.assertEqual(labels[-1], measure_label("CKA"))
        self.assertEqual(labels[0], "SVCCA")

    def test_labels_are_the_published_abbreviations(self):
        extra = combine_domain_values(
            {
                "graphs": _domain_frame(
                    "shortcut_test",
                    "AUPRC",
                    {"SecondOrderCosineSimilarity": 0.4, "CKA": 0.5},
                )
            }
        )
        self.assertIn("2nd-Cos", self._labels("value", True, extra))

    def test_boxes_use_the_shared_measure_categories(self):
        """Signal, not Manifold: the same categories the domain figures use."""
        import matplotlib.pyplot as plt

        from manifold_repsim.resi.analysis.config import (
            MEASURE_CATEGORY_COLORS,
            measure_category,
        )

        frame = combine_domain_values(
            {
                "graphs": _domain_frame(
                    "shortcut_test", "AUPRC", {"CKArbfAUC": 0.4, "SVCCA": 0.2}
                )
            }
        )
        from manifold_repsim.resi.analysis.boxplots import measure_panel

        figure, ax = plt.subplots()
        drawn = measure_panel(ax, frame, "value", higher_is_better=True)
        plt.close(figure)
        groups = [measure_category(metric) for metric in drawn]
        self.assertIn("Signal", groups)
        self.assertNotIn("Manifold", groups)
        self.assertTrue(all(g in MEASURE_CATEGORY_COLORS for g in groups))

    def test_call_sites_pass_the_direction_matching_their_column(self):
        """The original bug was a correct helper called with the wrong flag.

        Pinning the helper's contract alone would not have caught it, so this
        asserts what each plot function actually asks for: ranks are best low,
        values are best high.
        """
        from unittest import mock

        import manifold_repsim.resi.analysis.cross_domain_figures as figures
        from manifold_repsim.resi.analysis import boxplots

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(
                figures, "measure_panel", wraps=boxplots.measure_panel
            ) as spy:
                figures.plot_task_ranks_by_type(
                    self._ranks(), "Shortcuts", Path(tmp) / "r.png"
                )
                self.assertTrue(spy.called)
                for call in spy.call_args_list:
                    self.assertIs(call.kwargs["higher_is_better"], False)

            with mock.patch.object(
                figures, "measure_panel", wraps=boxplots.measure_panel
            ) as spy:
                figures.plot_task_values_by_type(
                    self._values(), "Shortcuts", Path(tmp) / "v.png"
                )
                self.assertTrue(spy.called)
                for call in spy.call_args_list:
                    self.assertIs(call.kwargs["higher_is_better"], True)

    def test_rank_figure_is_written(self):
        ranks = self._ranks()
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_task_ranks_by_type(ranks, "Shortcuts", Path(tmp) / "r.png")
            self.assertIsNotNone(path)
            self.assertTrue(path.is_file())

    def test_value_figure_is_written_per_quality_measure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = plot_task_values_by_type(
                self._values(), "Shortcuts", Path(tmp) / "v.png"
            )
            self.assertIsNotNone(path)
            self.assertTrue(path.is_file())

    def test_absent_family_draws_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                plot_task_ranks_by_type(
                    self._ranks(), "Layer Mono.", Path(tmp) / "r.png"
                )
            )
            self.assertIsNone(
                plot_task_values_by_type(
                    self._values(), "Layer Mono.", Path(tmp) / "v.png"
                )
            )

    def test_write_by_type_covers_overall_and_each_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index = write_by_type_figures(self._values(), self._ranks(), directory)
            self.assertEqual(list(index["test_group"]), [ALL_TASKS, "Shortcuts"])
            # The overall panel is a rank aggregate; values are not pooled
            # across families, so it has no value figure.
            self.assertEqual(index.loc[0, "value_figure"], "")
            self.assertTrue(Path(index.loc[1, "value_figure"]).is_file())
            self.assertTrue((directory / "by_type_index.csv").is_file())


class TestCrossDomainRankScale(unittest.TestCase):
    """Cross-domain carries both rank scales and reports the raw one."""

    def _ranks(self):
        rows = []
        for case, (cka, svcca) in enumerate(((1.0, 2.0), (1.0, 2.0))):
            for metric, raw in (("CKA", cka), ("SVCCA", svcca)):
                rows.append(
                    {
                        "domain": "graphs",
                        "benchmark": "shortcut_test",
                        "case_id": f"c{case}",
                        "metric": metric,
                        "mean_rank": raw,
                        "mean_normalized_rank": raw - 1.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_both_scales_survive_combination_and_averaging(self):
        """Switching scale must never require re-running this layer."""
        subtasks = average_subtask_ranks(combine_case_ranks({"graphs": self._ranks()}))
        self.assertIn("mean_rank", subtasks.columns)
        self.assertIn("mean_normalized_rank", subtasks.columns)

    def test_the_default_scale_is_the_raw_rank(self):
        combined = combine_case_ranks({"graphs": self._ranks()})
        default = aggregate_across_domains(combined).set_index(["test_group", "metric"])
        raw = aggregate_across_domains(combined, rank_scale="raw").set_index(
            ["test_group", "metric"]
        )
        normalized = aggregate_across_domains(
            combined, rank_scale="normalized"
        ).set_index(["test_group", "metric"])
        self.assertAlmostEqual(
            float(default.loc[(ALL_TASKS, "CKA"), "pooled_mean"]), 1.0
        )
        pd.testing.assert_frame_equal(default, raw)
        # The same measure, one scale apart: rank 1 of 2 is normalized 0.
        self.assertAlmostEqual(
            float(normalized.loc[(ALL_TASKS, "CKA"), "pooled_mean"]), 0.0
        )

    def test_a_missing_scale_column_does_not_lose_the_rows(self):
        """A hand-built frame may carry only one scale; the other reads as NaN."""
        frame = self._ranks().drop(columns=["mean_normalized_rank"])
        combined = combine_case_ranks({"graphs": frame})
        self.assertEqual(len(combined), len(frame))
        self.assertTrue(combined["mean_normalized_rank"].isna().all())
        aggregate = aggregate_across_domains(combined)
        self.assertAlmostEqual(
            float(
                aggregate.set_index(["test_group", "metric"]).loc[
                    (ALL_TASKS, "CKA"), "pooled_mean"
                ]
            ),
            1.0,
        )
