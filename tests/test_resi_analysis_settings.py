"""Validation and resolution rules for the analysis settings file.

These pin the contract that makes a reported measure set explicit: unknown keys
are rejected rather than ignored, `measures` and `exclude_measures` cannot both
be given, and an exclusion list resolves against what the results actually
contain so one list can be shared across domains with different archives.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from manifold_repsim.resi.analysis.config import RANK_SCALES, RANK_SORTS
from manifold_repsim.resi.analysis.settings import (
    DEFAULT_MAX_NAN_FRACTION,
    DEFAULT_MIN_CASE_COVERAGE,
    AnalysisSettings,
    load_analysis_settings,
    parse_analysis_settings,
    settings_snapshot,
)

PRESENT = ("CKA", "SVCCA", "IMDScore", "PWCCA")


class TestAnalysisSettingsParsing(unittest.TestCase):
    def test_defaults_report_every_measure(self):
        settings = parse_analysis_settings({})
        self.assertIsNone(settings.measures)
        self.assertEqual(settings.exclude_measures, ())
        self.assertIsNone(settings.quality_measures)
        self.assertEqual(settings.exclude_functional_measures, ())
        self.assertFalse(settings.allow_incomplete)
        self.assertIsNone(settings.resolve_measures(PRESENT))

    def test_excluded_functional_measures_parse_and_snapshot(self):
        settings = parse_analysis_settings(
            {"exclude_functional_measures": ["AbsoluteAccDiff"]}
        )
        self.assertEqual(settings.exclude_functional_measures, ("AbsoluteAccDiff",))
        snapshot = settings_snapshot(settings, None)
        self.assertEqual(snapshot["exclude_functional_measures"], ["AbsoluteAccDiff"])

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            parse_analysis_settings({"measure": ["CKA"]})
        self.assertIn("measure", str(caught.exception))

    def test_measures_and_exclude_measures_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            parse_analysis_settings(
                {"measures": ["CKA"], "exclude_measures": ["SVCCA"]}
            )

    def test_empty_and_duplicate_lists_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_analysis_settings({"measures": []})
        with self.assertRaises(ValueError):
            parse_analysis_settings({"measures": ["CKA", "CKA"]})
        with self.assertRaises(ValueError):
            parse_analysis_settings({"measures": ["CKA", "  "]})

    def test_allow_incomplete_must_be_boolean(self):
        with self.assertRaises(ValueError):
            parse_analysis_settings({"allow_incomplete": "yes"})

    def test_a_bare_string_is_accepted_as_a_single_name(self):
        settings = parse_analysis_settings({"measures": "CKA"})
        self.assertEqual(settings.measures, ("CKA",))


class TestMeasureResolution(unittest.TestCase):
    def test_explicit_measures_pass_through_unchanged(self):
        settings = AnalysisSettings(measures=("CKA", "Missing"))
        # Returned verbatim so the caller's strict check reports the absence,
        # rather than silently narrowing to what happens to be present.
        self.assertEqual(settings.resolve_measures(PRESENT), ("CKA", "Missing"))

    def test_exclusions_resolve_against_what_is_present(self):
        settings = AnalysisSettings(exclude_measures=("IMDScore", "NeverHere"))
        self.assertEqual(settings.resolve_measures(PRESENT), ("CKA", "PWCCA", "SVCCA"))

    def test_excluding_everything_is_an_error(self):
        settings = AnalysisSettings(exclude_measures=PRESENT)
        with self.assertRaises(ValueError):
            settings.resolve_measures(PRESENT)


class TestAnalysisSettingsLoading(unittest.TestCase):
    def test_round_trip_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "analysis.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "measures": ["CKA", "SVCCA"],
                        "quality_measures": ["AUPRC"],
                        "allow_incomplete": True,
                    }
                ),
                encoding="utf-8",
            )
            settings = load_analysis_settings(path)
        self.assertEqual(settings.measures, ("CKA", "SVCCA"))
        self.assertEqual(settings.quality_measures, ("AUPRC",))
        self.assertTrue(settings.allow_incomplete)
        self.assertEqual(settings.path, path)

    def test_rank_scale_defaults_validates_and_snapshots(self):
        self.assertEqual(AnalysisSettings().rank_scale, "raw")
        self.assertEqual(parse_analysis_settings({}).rank_scale, "raw")
        self.assertEqual(
            parse_analysis_settings({"rank_scale": "normalized"}).rank_scale,
            "normalized",
        )
        with self.assertRaises(ValueError):
            parse_analysis_settings({"rank_scale": "percentile"})

        settings = parse_analysis_settings({"rank_scale": "raw"})
        self.assertEqual(settings_snapshot(settings, None)["rank_scale"], "raw")
        # A command line override is what the run actually ranked on, so that
        # is what the saved config.yaml must record.
        self.assertEqual(
            settings_snapshot(settings, None, rank_scale="normalized")["rank_scale"],
            "normalized",
        )

    def test_top_level_cli_accepts_exactly_the_analysis_rank_scales(self):
        """`resi.py` duplicates the choices to avoid importing the analysis package."""
        from manifold_repsim.resi.cli import build_parser

        base = ["analyze", "--campaign", "c.yaml", "--run-name", "r"]
        self.assertIsNone(build_parser().parse_args(base).rank_scale)
        for scale in RANK_SCALES:
            with self.subTest(scale=scale):
                args = build_parser().parse_args([*base, "--rank-scale", scale])
                self.assertEqual(args.rank_scale, scale)
        with self.assertRaises(SystemExit):
            build_parser().parse_args([*base, "--rank-scale", "percentile"])

    def test_top_level_cli_accepts_exactly_the_analysis_rank_sorts(self):
        """Duplicated for the same reason as the rank scales just above."""
        from manifold_repsim.resi.cli import build_parser

        base = ["analyze", "--campaign", "c.yaml", "--run-name", "r"]
        self.assertIsNone(build_parser().parse_args(base).rank_sort)
        for sort in RANK_SORTS:
            with self.subTest(sort=sort):
                args = build_parser().parse_args([*base, "--rank-sort", sort])
                self.assertEqual(args.rank_sort, sort)
        with self.assertRaises(SystemExit):
            build_parser().parse_args([*base, "--rank-sort", "p99"])

    def test_coverage_thresholds_default_and_validate(self):
        defaults = parse_analysis_settings({})
        self.assertEqual(defaults.max_nan_fraction, DEFAULT_MAX_NAN_FRACTION)
        self.assertEqual(defaults.min_case_coverage, DEFAULT_MIN_CASE_COVERAGE)
        chosen = parse_analysis_settings(
            {"max_nan_fraction": 0, "min_case_coverage": 1}
        )
        self.assertEqual(chosen.max_nan_fraction, 0.0)
        self.assertEqual(chosen.min_case_coverage, 1.0)
        for field in ("max_nan_fraction", "min_case_coverage"):
            # A bool is an int in Python; accepting it would silently mean 0 or 1.
            for bad in (-0.1, 1.5, "half", True, [0.5]):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises(ValueError):
                        parse_analysis_settings({field: bad})

    def test_thresholds_are_recorded_in_the_snapshot(self):
        settings = parse_analysis_settings(
            {"max_nan_fraction": 0.25, "min_case_coverage": 0.8}
        )
        snapshot = settings_snapshot(settings, ["CKA"])
        self.assertEqual(snapshot["max_nan_fraction"], 0.25)
        self.assertEqual(snapshot["min_case_coverage"], 0.8)

    def test_missing_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_analysis_settings("/nonexistent/analysis.yaml")

    def test_checked_in_configs_are_valid(self):
        configs = Path(__file__).resolve().parents[1] / "configs"
        found = sorted(configs.glob("resi_*_analysis.yaml"))
        self.assertEqual(len(found), 3)
        for path in found:
            with self.subTest(config=path.name):
                settings = load_analysis_settings(path)
                self.assertIsNotNone(settings.measures)
                self.assertEqual(len(settings.measures), len(set(settings.measures)))
                self.assertIn(settings.rank_scale, RANK_SCALES)

    def test_checked_in_configs_report_the_published_quality_measures(self):
        """ReSi reports prediction correlation with Spearman only."""
        configs = Path(__file__).resolve().parents[1] / "configs"
        for path in sorted(configs.glob("resi_*_analysis.yaml")):
            with self.subTest(config=path.name):
                settings = load_analysis_settings(path)
                self.assertIsNotNone(settings.quality_measures)
                self.assertEqual(
                    set(settings.quality_measures),
                    {"AUPRC", "violation_rate", "correlation", "spearmanr"},
                )

    def test_checked_in_configs_drop_the_retired_manifold_measures(self):
        """Retired from reporting, not from computation -- see the config comment."""
        retired = {
            "mcRWKArbfAUC",
            "dRWKArbfSigma05",
            "sRWKArbfSigma05",
            "CKArbfSigma05",
        }
        configs = Path(__file__).resolve().parents[1] / "configs"
        for path in sorted(configs.glob("resi_*_analysis.yaml")):
            with self.subTest(config=path.name):
                settings = load_analysis_settings(path)
                self.assertEqual(retired & set(settings.measures), set())
                # 9 native measures plus the 7 manifold ones that remain.
                self.assertEqual(len(settings.measures), 16)


if __name__ == "__main__":
    unittest.main()


class TestRankSortSetting(unittest.TestCase):
    def test_default_and_explicit_values(self):
        self.assertEqual(parse_analysis_settings({}).rank_sort, "median")
        self.assertEqual(
            parse_analysis_settings({"rank_sort": "quantile90"}).rank_sort,
            "quantile90",
        )

    def test_an_unknown_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank_sort must be one of"):
            parse_analysis_settings({"rank_sort": "p99"})

    def test_the_snapshot_records_the_override_not_the_file(self):
        settings = parse_analysis_settings({"rank_sort": "median"})
        snapshot = settings_snapshot(settings, None, rank_sort="quantile90")
        self.assertEqual(snapshot["rank_sort"], "quantile90")
        self.assertEqual(settings_snapshot(settings, None)["rank_sort"], "median")


class TestBoxStyleSetting(unittest.TestCase):
    def test_default_and_explicit_values(self):
        self.assertEqual(parse_analysis_settings({}).box_style, "box")
        self.assertEqual(
            parse_analysis_settings({"box_style": "boxen"}).box_style, "boxen"
        )

    def test_an_unknown_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "box_style must be one of"):
            parse_analysis_settings({"box_style": "violin"})

    def test_the_snapshot_records_the_override_not_the_file(self):
        settings = parse_analysis_settings({"box_style": "box"})
        snapshot = settings_snapshot(settings, None, box_style="boxen")
        self.assertEqual(snapshot["box_style"], "boxen")
        self.assertEqual(settings_snapshot(settings, None)["box_style"], "box")

    def test_it_is_independent_of_the_sort(self):
        """Drawing and ordering are separate choices; either combination is legal."""
        settings = parse_analysis_settings(
            {"box_style": "boxen", "rank_sort": "median"}
        )
        self.assertEqual(settings.box_style, "boxen")
        self.assertEqual(settings.rank_sort, "median")

    def test_top_level_cli_accepts_exactly_the_analysis_box_styles(self):
        """Duplicated for the same reason as the rank scales and sorts."""
        from manifold_repsim.resi.cli import build_parser
        from manifold_repsim.resi.analysis.config import BOX_STYLES

        base = ["analyze", "--campaign", "c.yaml", "--run-name", "r"]
        self.assertIsNone(build_parser().parse_args(base).box_style)
        for style in BOX_STYLES:
            with self.subTest(style=style):
                args = build_parser().parse_args([*base, "--box-style", style])
                self.assertEqual(args.box_style, style)
        with self.assertRaises(SystemExit):
            build_parser().parse_args([*base, "--box-style", "violin"])
