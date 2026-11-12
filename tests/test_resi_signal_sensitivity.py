from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from manifold_repsim.resi.signal_sensitivity import (
    BASE_METRIC,
    SignalVariant,
    load_sensitivity_config,
    parse_sensitivity_config,
    derive_variant_rows,
    reduce_signal,
    select_points,
)

GRID = np.geomspace(0.05, 10.0, 100)
CONFIG = load_sensitivity_config(
    Path(__file__).resolve().parents[1] / "configs" / "resi_signal_sensitivity.yaml"
)
VARIANTS = list(CONFIG.variants)


def _store(scores_by_pair, nan_pair=None):
    """A result store holding both directions of each pair, signal on one only."""
    rows, index = [], []
    for name, scores in scores_by_pair.items():
        signal = json.dumps({"param_values": GRID.tolist(), "scores": list(scores)})
        for position, (src, tgt) in enumerate([(f"{name}a", f"{name}b")] * 2):
            first, second = (src, tgt) if position == 0 else (tgt, src)
            index.append(f"{first}___{second}___{BASE_METRIC}")
            rows.append(
                {
                    "metric": BASE_METRIC,
                    "metric_value": float(np.mean(scores)),
                    "similarity_signal": signal if position == 0 else None,
                }
            )
    frame = pd.DataFrame(rows, index=index)
    frame["id"] = frame.index
    if nan_pair is not None:
        frame.loc[frame.index.str.startswith(nan_pair), "metric_value"] = math.nan
    return frame


class TestConfig(unittest.TestCase):
    def test_checked_in_config_declares_the_two_questions(self):
        self.assertEqual({a.name for a in CONFIG.analyses}, {"samples", "limits"})
        self.assertEqual(CONFIG.base_metric, "CKArbfAUCRerun")
        samples = next(a for a in CONFIG.analyses if a.name == "samples")
        self.assertEqual(
            [label for _k, label in samples.points], ["25", "50", "75", "100"]
        )

    def test_an_analysis_naming_an_unknown_variant_is_rejected(self):
        raw = {
            "variants": [{"key": "a", "samples": 10}],
            "analyses": {"x": {"points": [{"variant": "b", "label": "b"}]}},
        }
        with self.assertRaisesRegex(ValueError, "unknown variants"):
            parse_sensitivity_config(raw)

    def test_duplicate_or_missing_variants_are_rejected(self):
        for variants in ([], [{"key": "a", "samples": 5}, {"key": "a", "samples": 6}]):
            with self.subTest(variants=variants):
                with self.assertRaises(ValueError):
                    parse_sensitivity_config({"variants": variants, "analyses": {}})


class TestSelectPoints(unittest.TestCase):
    def test_default_variants_keep_the_documented_point_counts(self):
        counts = {v.key: len(select_points(GRID, v)) for v in VARIANTS}
        self.assertEqual(
            counts,
            {
                "samples100": 100,
                "samples75": 75,
                "samples50": 50,
                "samples25": 25,
                "left": 74,
                "right": 74,
            },
        )

    def test_sample_variants_keep_both_endpoints_and_stay_ordered(self):
        for samples in (75, 50, 25):
            kept = select_points(GRID, SignalVariant("k", samples))
            self.assertEqual((kept[0], kept[-1]), (0, 99))
            self.assertTrue(np.all(np.diff(kept) > 0))

    def test_limits_keep_only_points_inside_the_interval(self):
        left = select_points(GRID, SignalVariant("l", lower=0.2))
        right = select_points(GRID, SignalVariant("r", upper=2.5))
        self.assertGreaterEqual(GRID[left].min(), 0.2)
        self.assertLess(GRID[left[0] - 1], 0.2)
        self.assertLessEqual(GRID[right].max(), 2.5)
        self.assertGreater(GRID[right[-1] + 1], 2.5)

    def test_invalid_variants_fail_loudly(self):
        with self.assertRaises(ValueError):
            SignalVariant("x", samples=1)
        with self.assertRaises(ValueError):
            SignalVariant("x", lower=2.0, upper=1.0)
        with self.assertRaisesRegex(ValueError, "stored grid"):
            select_points(GRID, SignalVariant("x", samples=101))
        with self.assertRaisesRegex(ValueError, "no stored grid points"):
            select_points(GRID, SignalVariant("x", lower=50.0))


class TestReduceSignal(unittest.TestCase):
    def test_full_variant_is_the_plain_mean(self):
        scores = np.sin(GRID)
        full = SignalVariant("f", samples=100)
        self.assertAlmostEqual(reduce_signal(GRID, scores, full), scores.mean())

    def test_constant_curve_reduces_to_the_constant_on_every_variant(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant.key):
                self.assertAlmostEqual(
                    reduce_signal(GRID, np.full(100, 0.7), variant), 0.7
                )

    def test_clip_averages_only_the_kept_points(self):
        scores = np.where(GRID < 0.2, 0.0, 1.0)
        left = SignalVariant("l", lower=0.2)
        self.assertAlmostEqual(reduce_signal(GRID, scores, left), 1.0)
        self.assertLess(reduce_signal(GRID, scores, SignalVariant("f", 100)), 1.0)

    def test_nan_inside_the_kept_points_gives_nan_but_outside_does_not(self):
        scores = np.ones(100)
        scores[0] = math.nan
        left = SignalVariant("l", lower=0.2)
        full = SignalVariant("f", samples=100)
        self.assertEqual(reduce_signal(GRID, scores, left), 1.0)
        self.assertTrue(math.isnan(reduce_signal(GRID, scores, full)))


class TestDeriveVariantRows(unittest.TestCase):
    def test_full_variant_reproduces_the_stored_values_for_both_directions(self):
        store = _store({"p": np.sin(GRID) / 2, "q": np.cos(GRID) / 2})
        full = SignalVariant("samples100", samples=100)
        rows = derive_variant_rows(store, [full])
        self.assertEqual(len(rows), len(store))
        np.testing.assert_allclose(
            rows["metric_value"].to_numpy(), store["metric_value"].to_numpy()
        )
        self.assertTrue((rows["metric"] == full.metric_name).all())
        self.assertTrue(all(i.endswith(full.metric_name) for i in rows.index))
        self.assertTrue(rows["similarity_signal"].isna().all())

    def test_mirror_rows_take_the_value_of_their_partner(self):
        store = _store({"p": np.linspace(0, 1, 100)})
        left = SignalVariant("left", lower=0.2)
        rows = derive_variant_rows(store, [left])
        self.assertEqual(rows["metric_value"].nunique(), 1)

    def test_failed_comparisons_stay_nan(self):
        store = _store({"p": np.ones(100), "q": np.ones(100)}, nan_pair="p")
        rows = derive_variant_rows(store, VARIANTS)
        nan_rows = rows[rows.index.str.startswith("pa")]
        self.assertTrue(nan_rows["metric_value"].isna().all())
        self.assertTrue(
            rows[rows.index.str.startswith("qa")]["metric_value"].notna().all()
        )

    def test_a_value_without_any_signal_is_an_error_not_a_silent_number(self):
        store = _store({"p": np.ones(100)})
        store["similarity_signal"] = None
        with self.assertRaisesRegex(ValueError, "no stored signal"):
            derive_variant_rows(store, VARIANTS[:1])

    def test_store_without_the_base_measure_is_an_error(self):
        store = _store({"p": np.ones(100)})
        store["metric"] = "Other"
        with self.assertRaisesRegex(ValueError, "no rows"):
            derive_variant_rows(store, VARIANTS[:1])


if __name__ == "__main__":
    unittest.main()


class TestMaskedComparisons(unittest.TestCase):
    def test_comparison_the_reported_measure_could_not_score_stays_nan(self):
        signal = json.dumps({"param_values": GRID.tolist(), "scores": [0.5] * 100})
        rows = {}
        for pair, reported in (("a___b", 0.4), ("c___d", float("nan"))):
            rows[f"{pair}___CKArbfAUCRerun"] = {
                "metric": "CKArbfAUCRerun",
                "metric_value": 0.5,
                "similarity_signal": signal,
            }
            rows[f"{pair}___CKArbfAUC"] = {
                "metric": "CKArbfAUC",
                "metric_value": reported,
                "similarity_signal": None,
            }
        store = pd.DataFrame.from_dict(rows, orient="index")
        variant = SignalVariant("full", samples=100)
        plain = derive_variant_rows(store, [variant], base_metric="CKArbfAUCRerun")
        self.assertEqual(int(plain["metric_value"].notna().sum()), 2)
        masked = derive_variant_rows(
            store, [variant], base_metric="CKArbfAUCRerun", mask_metric="CKArbfAUC"
        )
        values = masked.set_index("id")["metric_value"]
        self.assertAlmostEqual(values["a___b___CKArbfAUC-full"], 0.5)
        self.assertTrue(math.isnan(values["c___d___CKArbfAUC-full"]))


class TestSensitivitySummary(unittest.TestCase):
    """Variants replace the reported CKA AUC in an otherwise unchanged field."""

    def _build(self, root: Path):
        import yaml

        from tests.test_resi_analysis import (
            _manifest_entry,
            _write_campaign_yaml,
            _write_manifest,
        )

        campaign_path = _write_campaign_yaml(
            root,
            domain="graphs",
            measures=["CKArbfAUC"],
            benchmarks=[
                {
                    "id": "augmentation",
                    "dataset": "toy",
                    "base_config": "unused.yaml",
                    "architectures": ["GCN"],
                    "baselines": [],
                }
            ],
        )
        run_dir = root / "runs" / "trial" / "graphs"
        run_dir.mkdir(parents=True)
        identifiers = ["o0", "o1", "o2", "o3"]

        def rows(metric_values):
            return pd.DataFrame(
                [
                    {
                        "similarity_measure": metric,
                        "quality_measure": "AUPRC",
                        "architecture": "GCN",
                        "identifier": identifier,
                        "representation_dataset": "toy",
                        "value": value,
                    }
                    for metric, value in metric_values.items()
                    for identifier in identifiers
                ]
            )

        baseline_dir = root / "baseline"
        compute_dir = root / "res" / "manifold" / "trial"
        for task_dir in (baseline_dir, compute_dir):
            task_dir.mkdir(parents=True)
        rows({"CKA": 0.9, "Other": 0.2}).to_csv(
            baseline_dir / "results_full.csv", index=False
        )
        pd.DataFrame({"metric": ["CKA"], "metric_value": [0.9]}).to_parquet(
            baseline_dir / "results.parquet"
        )
        rows({"CKArbfAUC": 0.5}).to_csv(compute_dir / "results_full.csv", index=False)
        pd.DataFrame({"metric": ["CKArbfAUC"], "metric_value": [0.5]}).to_parquet(
            compute_dir / "results.parquet"
        )
        compute_dir.joinpath("config.yaml").write_text(
            yaml.safe_dump(
                {
                    "raw_results_filename": "manifold/trial/results.parquet",
                    "table_creation": {
                        "full_df_filename": "manifold/trial/results_full.csv"
                    },
                }
            )
        )
        scratch = root / "res" / "manifold-sensitivity" / "signal-sensitivity" / "trial"
        scratch.mkdir(parents=True)
        rows({"CKArbfAUC-good": 0.95, "CKArbfAUC-bad": 0.1}).to_csv(
            scratch / "results_full.csv", index=False
        )
        for kind, index, task_dir, measures in (
            ("baseline", 0, baseline_dir, ["CKA", "Other"]),
            ("compute", 0, compute_dir, ["CKArbfAUC"]),
        ):
            _write_manifest(
                run_dir / f"{kind}_manifest.jsonl",
                [
                    _manifest_entry(
                        kind=kind,
                        index=index,
                        domain="graphs",
                        benchmark="augmentation",
                        dataset="toy",
                        architectures=["GCN"],
                        measures=measures,
                        task_dir=task_dir,
                    )
                ],
            )
        return campaign_path

    def test_each_variant_is_ranked_in_place_of_the_reported_measure(self):
        from manifold_repsim.resi.analysis.sensitivity import summarize_sensitivity
        from manifold_repsim.resi.analysis.settings import AnalysisSettings

        config = parse_sensitivity_config(
            {
                "base_metric": "CKArbfAUC",
                "field_measure": "CKArbfAUC",
                "variants": [
                    {"key": "good", "samples": 50},
                    {"key": "bad", "samples": 25},
                ],
                "analyses": {
                    "demo": {
                        "points": [
                            {"variant": "bad", "label": "25"},
                            {"variant": "good", "label": "50"},
                        ]
                    }
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._build(root)
            settings = AnalysisSettings(measures=("CKA", "Other", "CKArbfAUC"))
            out, summary = summarize_sensitivity(
                campaign_path, "trial", settings, config
            )
            self.assertTrue((out / "sensitivity_summary.csv").is_file())
            self.assertTrue((out / "config.yaml").is_file())
            by_variant = summary.set_index("variant")
            # CKA 0.9, Other 0.2 are fixed; the variant is 0.95 (best) or 0.1.
            self.assertAlmostEqual(by_variant.loc["good", "mean_rank"], 1.0)
            self.assertAlmostEqual(by_variant.loc["bad", "mean_rank"], 3.0)
            self.assertAlmostEqual(by_variant.loc["stored", "mean_rank"], 2.0)
            self.assertEqual(by_variant.loc["good", "position"], 1)
            self.assertEqual(by_variant.loc["bad", "position"], 3)
            self.assertEqual(set(summary["n_measures"]), {3})
            self.assertEqual(by_variant.loc["bad", "q90_rank"], 3.0)

            from manifold_repsim.resi.analysis.sensitivity_figures import (
                plot_sensitivity,
            )

            figures = plot_sensitivity(summary, config, out)
            self.assertEqual([path.name for path in figures], ["demo.png"])
            self.assertGreater(figures[0].stat().st_size, 0)

    def test_excluded_functional_measures_are_dropped_from_the_variants(self):
        from manifold_repsim.resi.analysis.sensitivity import variant_values
        from manifold_repsim.resi.campaign import read_manifest

        config = parse_sensitivity_config(
            {
                "base_metric": "CKArbfAUC",
                "variants": [{"key": "good", "samples": 50}],
                "analyses": {"demo": {"points": [{"variant": "good", "label": "50"}]}},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build(root)
            scratch = (
                root / "res" / "manifold-sensitivity" / "signal-sensitivity" / "trial"
            )
            frame = pd.read_csv(scratch / "results_full.csv")
            frame["functional_similarity_measure"] = ["", "Dropped"] * (len(frame) // 2)
            frame.to_csv(scratch / "results_full.csv", index=False)
            entries = read_manifest(
                root / "runs" / "trial" / "graphs" / "compute_manifest.jsonl"
            )
            kept = variant_values(entries, config, ["AUPRC"])
            dropped = variant_values(entries, config, ["AUPRC"], ["Dropped"])
            self.assertGreater(len(kept), len(dropped))
            self.assertNotIn("Dropped", set(dropped["functional_similarity_measure"]))

    def test_bandwidths_are_ranked_in_one_shared_field(self):
        from manifold_repsim.resi.analysis.sensitivity import summarize_bandwidths
        from manifold_repsim.resi.analysis.sensitivity_figures import plot_bandwidths
        from manifold_repsim.resi.analysis.settings import AnalysisSettings

        config = parse_sensitivity_config(
            {
                "base_metric": "CKArbfAUC",
                "variants": [{"key": "good", "samples": 50}],
                "analyses": {"demo": {"points": [{"variant": "good", "label": "50"}]}},
                "bandwidths": {
                    "reference": "CKArbfAUC",
                    "measures": [
                        {"measure": "Other", "sigma": 0.5},
                    ],
                },
            }
        )
        self.assertEqual([sigma for _m, sigma in config.bandwidths.points], [0.5])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._build(root)
            settings = AnalysisSettings(measures=("CKA", "Other", "CKArbfAUC"))
            out, table = summarize_bandwidths(campaign_path, "trial", settings, config)
            by_measure = table.set_index("measure")
            # The rest of the field is CKA (0.9); the swept reference (0.5) and
            # the single bandwidth (0.2) each meet it alone, never each other.
            self.assertEqual(set(by_measure.index), {"CKArbfAUC", "Other"})
            self.assertEqual(by_measure.loc["CKArbfAUC", "mean_rank"], 2.0)
            self.assertEqual(by_measure.loc["Other", "mean_rank"], 2.0)
            self.assertEqual(set(table["n_measures"]), {2})
            self.assertTrue(math.isnan(by_measure.loc["CKArbfAUC", "sigma"]))
            self.assertTrue((out / "bandwidth_summary.csv").is_file())
            figure = plot_bandwidths(table, config, out)
            self.assertGreater(figure.stat().st_size, 0)

    def test_paired_bandwidths_rank_the_reference_against_each_rival(self):
        from manifold_repsim.resi.analysis.sensitivity import (
            summarize_paired_bandwidths,
        )
        from manifold_repsim.resi.analysis.sensitivity_figures import (
            plot_paired_bandwidths,
        )
        from manifold_repsim.resi.analysis.settings import AnalysisSettings

        config = parse_sensitivity_config(
            {
                "base_metric": "CKArbfAUC",
                "variants": [{"key": "good", "samples": 50}],
                "analyses": {"demo": {"points": [{"variant": "good", "label": "50"}]}},
                "bandwidths": {
                    "reference": "CKArbfAUC",
                    "measures": [{"measure": "Other", "sigma": 0.5}],
                },
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            campaign_path = self._build(Path(tmp))
            settings = AnalysisSettings(measures=("CKA", "Other", "CKArbfAUC"))
            out, table = summarize_paired_bandwidths(
                campaign_path, "trial", settings, config
            )
            by_measure = table.set_index("measure")
            # One field of CKA (0.9), the reference (0.5) and Other (0.2): the
            # three measures are all present, so the reference ranks 2nd and
            # the rival 3rd, and the field is one measure larger than unpaired.
            self.assertEqual(len(table), 2)
            self.assertEqual(set(table["n_measures"]), {3})
            self.assertEqual(by_measure.loc["CKArbfAUC", "mean_rank"], 2.0)
            self.assertEqual(by_measure.loc["Other", "mean_rank"], 3.0)
            self.assertEqual(set(table["sigma"]), {0.5})
            self.assertTrue((out / "bandwidth_paired_summary.csv").is_file())
            figure = plot_paired_bandwidths(table, config, out)
            self.assertGreater(figure.stat().st_size, 0)

    def test_bandwidth_config_rejects_a_repeated_measure(self):
        with self.assertRaises(ValueError):
            parse_sensitivity_config(
                {
                    "variants": [{"key": "a", "samples": 50}],
                    "analyses": {"d": {"points": [{"variant": "a", "label": "50"}]}},
                    "bandwidths": {
                        "measures": [
                            {"measure": "X", "sigma": 0.2},
                            {"measure": "X", "sigma": 0.4},
                        ]
                    },
                }
            )

    def test_a_task_without_replay_output_is_an_error(self):
        from manifold_repsim.resi.analysis.sensitivity import summarize_sensitivity
        from manifold_repsim.resi.analysis.settings import AnalysisSettings

        config = parse_sensitivity_config(
            {
                "base_metric": "CKArbfAUC",
                "variants": [{"key": "good", "samples": 50}],
                "analyses": {"demo": {"points": [{"variant": "good", "label": "50"}]}},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._build(root)
            (
                root
                / "res"
                / "manifold-sensitivity"
                / "signal-sensitivity"
                / "trial"
                / "results_full.csv"
            ).unlink()
            with self.assertRaises(FileNotFoundError):
                summarize_sensitivity(
                    campaign_path,
                    "trial",
                    AnalysisSettings(measures=("CKA", "Other", "CKArbfAUC")),
                    config,
                )
