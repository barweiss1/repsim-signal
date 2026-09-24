"""Golden schema locks for the public ReSi analysis artifacts.

These assertions pin what campaign analysis writes: every CSV's exact column
list and order, the figure filename templates, the filename slug rule, and the
stale-directory cleanup contract. They exist so the analysis refactor in cleanup
Step 13 cannot silently change a published schema, in the same way that Step 10
pinned artifact fingerprints before reorganizing the MDS experiment.

Update a lock here only when an output format is deliberately changed, and
record that change in the cleanup plan's decision log.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from manifold_repsim.resi.analysis.plotting import _safe_name
from manifold_repsim.resi.analysis.workflows import analyze_campaign


NORMALIZED_VALUES_COLUMNS = [
    "domain",
    "benchmark",
    "dataset",
    "architecture",
    "observation_id",
    "metric",
    "quality_measure",
    "value",
    "direction",
    "source",
    "identifier",
    "representation_dataset",
    "model",
    "functional_similarity_measure",
    "source_file",
]

PER_CASE_RANKS_COLUMNS = [
    "domain",
    "benchmark",
    "dataset",
    "architecture",
    "quality_measure",
    "identifier",
    "representation_dataset",
    "functional_similarity_measure",
    "case_id",
    "metric",
    "mean_rank",
    "mean_normalized_rank",
    # Was one `n_matched_observations` count, when an observation only entered
    # the ranking if every compared measure was finite in it and the count was
    # therefore shared by the whole case. Ranking now skips a measure on the
    # observations it failed, so coverage is per measure: how many of the
    # case's observations this one was ranked in, out of how many it has.
    "n_ranked_observations",
    "n_case_observations",
    "n_compared_metrics",
]

DOMAIN_SUMMARIES_COLUMNS = [
    "domain",
    "metric",
    "mean_normalized_rank",
    "median_normalized_rank",
    "std_normalized_rank",
    "mean_rank",
    "median_rank",
    "std_rank",
    "n_cases",
]

# The task-level summary is the domain summary with its task keys inserted
# immediately after the domain column. `task` is the reporting task, which for
# the correlation benchmarks is `outcorr` or `acccorr` rather than the
# benchmark id that runs both.
RANK_SUMMARY_COLUMNS = [
    "domain",
    "task",
    "benchmark",
    "dataset",
    "metric",
    "mean_normalized_rank",
    "median_normalized_rank",
    "std_normalized_rank",
    "mean_rank",
    "median_rank",
    "std_rank",
    "n_cases",
]

# The coverage report is a concatenation of four record types with different
# fields, so its column order comes from first appearance across those frames
# rather than from any single declared list.
COVERAGE_COLUMNS = [
    "record_type",
    "kind",
    "index",
    "state",
    "domain",
    "benchmark",
    "dataset",
    "architectures",
    "rows",
    "finite_values",
    "nan_values",
    "error",
    "result_path",
    "full_csv_path",
    "metric",
    "quality_measure",
    "signal_source",
    "signal_files_declared",
    "signal_files_found",
    "excluded_model",
    "exclusion_reason",
]

SIGNALS_LONG_COLUMNS = [
    "domain",
    "benchmark",
    "dataset",
    "source",
    "signal_source",
    "source_file",
    "comparison_id",
    "canonical_comparison_id",
    "metric",
    "base_metric",
    "param_name",
    "param_index",
    "param_value",
    "score",
    "auc_value",
    "integration_method",
    "logscale",
    "setting_pair",
    "condition_pair",
    "seed_pair",
    "condition_seed_pair",
    "architecture_pair",
    "source_setting",
    "source_architecture",
    "source_train_dataset",
    "source_seed",
    "source_representation_dataset",
    "source_layer_id",
    "target_setting",
    "target_architecture",
    "target_train_dataset",
    "target_seed",
    "target_representation_dataset",
    "target_layer_id",
]

SIGNALS_BY_SEED_COLUMNS = [
    "domain",
    "benchmark",
    "dataset",
    "metric",
    "architecture_pair",
    "setting_pair",
    "condition_pair",
    "seed_pair",
    "condition_seed_pair",
    "param_name",
    "param_value",
    "logscale",
    "score",
]

SIGNALS_GROUPED_COLUMNS = [
    "domain",
    "benchmark",
    "dataset",
    "metric",
    "architecture_pair",
    "setting_pair",
    "condition_pair",
    "param_name",
    "param_value",
    "logscale",
    "mean_score",
    "std_score",
    "n_seeds",
]

TASK_INDEX_COLUMNS = [
    "task",
    "benchmark",
    "dataset",
    "state",
    "out_dir",
    "input_entries",
    "value_rows",
    "rank_rows",
    "signal_rows",
    "signal_comparisons",
    "rank_plot",
    "quality_heatmaps",
    "quality_scatter",
    "signal_plots",
    "grouped_signal_plots",
]

MEASURE_EXCLUSIONS_COLUMNS = [
    "reason",
    "metric",
    "scope",
    "case_id",
    "quality_measure",
    "nan_fraction",
    "cases_present",
    "cases_in_scope",
    "case_coverage",
    "rows_dropped",
]

DOMAIN_CSV_SCHEMAS = {
    "normalized_values.csv": NORMALIZED_VALUES_COLUMNS,
    "per_case_ranks.csv": PER_CASE_RANKS_COLUMNS,
    "domain_summaries.csv": DOMAIN_SUMMARIES_COLUMNS,
    "coverage_nan_report.csv": COVERAGE_COLUMNS,
    "signals_long.csv": SIGNALS_LONG_COLUMNS,
    "signals_by_seed.csv": SIGNALS_BY_SEED_COLUMNS,
    "signals_grouped.csv": SIGNALS_GROUPED_COLUMNS,
    "task_analysis_index.csv": TASK_INDEX_COLUMNS,
    "measure_exclusions.csv": MEASURE_EXCLUSIONS_COLUMNS,
}

# Expected non-empty row counts for the fixture below. An empty frame writes a
# zero-byte CSV with no header at all, which would let a wrong schema pass, so
# each locked output must actually contain rows.
DOMAIN_ROW_COUNTS = {
    # Two measures scored under two quality measures. Only one of those two,
    # AUPRC, is ranked (see `RANKED_QUALITY_MEASURES`), so the values frame
    # carries all four rows and the ranks carry one case's two measures.
    "normalized_values.csv": 4,
    "per_case_ranks.csv": 2,
    "domain_summaries.csv": 2,
    "coverage_nan_report.csv": 7,
    "signals_long.csv": 4,
    "signals_by_seed.csv": 4,
    "signals_grouped.csv": 2,
    "task_analysis_index.csv": 1,
}

TASK_CSV_SCHEMAS = {
    "normalized_values.csv": NORMALIZED_VALUES_COLUMNS,
    "per_case_ranks.csv": PER_CASE_RANKS_COLUMNS,
    "rank_summary.csv": RANK_SUMMARY_COLUMNS,
    "coverage_nan_report.csv": COVERAGE_COLUMNS,
    "signals_long.csv": SIGNALS_LONG_COLUMNS,
    "signals_by_seed.csv": SIGNALS_BY_SEED_COLUMNS,
    "signals_grouped.csv": SIGNALS_GROUPED_COLUMNS,
    "measure_exclusions.csv": MEASURE_EXCLUSIONS_COLUMNS,
}

BENCHMARK = "augmentation_test"
DATASET = "cora"
MEASURE = "MutualKNNAUC"
BASELINE_MEASURE = "CKA"


def _comparison_id(source_seed: int, target_seed: int) -> str:
    return (
        f"Normal__GCN__{DATASET}__{source_seed}__{DATASET}__1"
        f"___Augmentation__GCN__{DATASET}__{target_seed}__{DATASET}__1"
        f"___{MEASURE}"
    )


def _build_campaign(root: Path) -> Path:
    """Write a campaign whose analysis populates every public artifact."""
    campaign_path = root / "campaign.yaml"
    campaign_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "domain": "graphs",
                "run_root": str(root / "runs"),
                "analysis_root": str(root / "figures"),
                "quality_measures": ["AUPRC", "violation_rate"],
                "measures": [MEASURE],
                "benchmarks": [
                    {
                        "id": BENCHMARK,
                        "dataset": DATASET,
                        "base_config": "unused.yaml",
                        "architectures": ["GCN"],
                        "baselines": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    run_dir = root / "runs" / "trial" / "graphs"
    run_dir.mkdir(parents=True)
    task_dir = root / "task"
    task_dir.mkdir()
    raw = task_dir / "results.parquet"
    full = task_dir / "results_full.csv"

    signal_rows = []
    for source_seed, target_seed, scores in ((0, 1, [0.2, 0.4]), (2, 3, [0.3, 0.5])):
        signal_rows.append(
            {
                "id": _comparison_id(source_seed, target_seed),
                "metric": MEASURE,
                "metric_value": float(sum(scores)) / len(scores),
                "similarity_signal": json.dumps(
                    {
                        "base_metric": "mutual_knn",
                        "param_name": "topk",
                        "param_values": [2, 4],
                        "scores": scores,
                        "auc_value": float(sum(scores)) / len(scores),
                        "logscale": True,
                    }
                ),
            }
        )
    pd.DataFrame(signal_rows).to_parquet(raw)

    pd.DataFrame(
        {
            "similarity_measure": [MEASURE, MEASURE],
            "quality_measure": ["AUPRC", "violation_rate"],
            "architecture": ["GCN", "GCN"],
            "identifier": ["Normal", "Normal"],
            "representation_dataset": [DATASET, DATASET],
            "value": [0.8, 0.2],
        }
    ).to_csv(full, index=False)

    (run_dir / "compute_manifest.jsonl").write_text(
        json.dumps(
            {
                "kind": "compute",
                "index": 0,
                "domain": "graphs",
                "benchmark": BENCHMARK,
                "dataset": DATASET,
                "architectures": ["GCN"],
                "measures": [MEASURE],
                "config_path": str(task_dir / "config.yaml"),
                "result_path": str(raw),
                "full_csv_path": str(full),
                "source_result_path": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # A second measure is required for ranking to produce any rows at all.
    baseline_dir = root / "baseline"
    baseline_dir.mkdir()
    baseline_raw = baseline_dir / "results.parquet"
    baseline_full = baseline_dir / "results_full.csv"
    pd.DataFrame({"metric": [BASELINE_MEASURE], "metric_value": [0.4]}).to_parquet(
        baseline_raw
    )
    pd.DataFrame(
        {
            "similarity_measure": [BASELINE_MEASURE, BASELINE_MEASURE],
            "quality_measure": ["AUPRC", "violation_rate"],
            "architecture": ["GCN", "GCN"],
            "identifier": ["Normal", "Normal"],
            "representation_dataset": [DATASET, DATASET],
            "value": [0.4, 0.6],
        }
    ).to_csv(baseline_full, index=False)
    (run_dir / "baseline_manifest.jsonl").write_text(
        json.dumps(
            {
                "kind": "baseline",
                "index": 0,
                "domain": "graphs",
                "benchmark": BENCHMARK,
                "dataset": DATASET,
                "architectures": ["GCN"],
                "measures": [BASELINE_MEASURE],
                "config_path": str(baseline_dir / "config.yaml"),
                "result_path": str(baseline_raw),
                "full_csv_path": str(baseline_full),
                "source_result_path": str(baseline_raw),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return campaign_path


class TestAnalysisOutputSchemas(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        campaign_path = _build_campaign(root)
        self.out = analyze_campaign(campaign_path, "trial")
        self.task_dir = self.out / "tasks" / BENCHMARK / DATASET

    def tearDown(self):
        self._tmp.cleanup()

    def test_domain_level_csv_schemas(self):
        for filename, columns in DOMAIN_CSV_SCHEMAS.items():
            with self.subTest(output=filename):
                path = self.out / filename
                self.assertTrue(path.is_file(), f"{filename} was not written")
                frame = pd.read_csv(path)
                self.assertEqual(list(frame.columns), columns)
                if filename in DOMAIN_ROW_COUNTS:
                    self.assertEqual(len(frame), DOMAIN_ROW_COUNTS[filename])

    def test_task_level_csv_schemas(self):
        for filename, columns in TASK_CSV_SCHEMAS.items():
            with self.subTest(output=filename):
                path = self.task_dir / filename
                self.assertTrue(path.is_file(), f"{filename} was not written")
                frame = pd.read_csv(path)
                self.assertEqual(list(frame.columns), columns)

    def test_condition_and_seed_pairs_stay_distinct_from_setting_pairs(self):
        signals = pd.read_csv(self.out / "signals_long.csv")
        for column in (
            "setting_pair",
            "condition_pair",
            "seed_pair",
            "condition_seed_pair",
        ):
            self.assertIn(column, signals.columns)
        self.assertEqual(signals["setting_pair"].nunique(), 1)
        self.assertEqual(signals["seed_pair"].nunique(), 2)
        self.assertEqual(signals["condition_seed_pair"].nunique(), 2)

        grouped = pd.read_csv(self.out / "signals_grouped.csv")
        self.assertNotIn("seed_pair", grouped.columns)
        self.assertNotIn("condition_seed_pair", grouped.columns)
        self.assertEqual(grouped["n_seeds"].tolist(), [2, 2])

    def test_output_directory_layout(self):
        self.assertEqual(self.out.name, "graphs")
        self.assertEqual(self.out.parent.name, "trial")
        self.assertTrue((self.out / "rank_boxplot.png").is_file())
        self.assertTrue(self.task_dir.is_dir())
        self.assertTrue((self.task_dir / "rank_boxplot.png").is_file())
        self.assertTrue((self.task_dir / "quality_heatmaps").is_dir())
        # Every scope writes a mean-sorted companion beside the configured
        # sort, so neither ordering has to be regenerated by hand.
        self.assertTrue((self.out / "rank_boxplot_mean.png").is_file())
        self.assertTrue((self.task_dir / "rank_boxplot_mean.png").is_file())

    def test_figure_filename_templates(self):
        heatmaps = sorted(
            p.name for p in (self.task_dir / "quality_heatmaps").glob("*.png")
        )
        self.assertEqual(heatmaps, ["auprc.png", "violation_rate.png"])

        signal_plots = sorted(
            p.name for p in (self.task_dir / "signal_plots").glob("*.png")
        )
        self.assertEqual(
            signal_plots,
            [
                "graphs__augmentation_test__cora__mutualknnauc__"
                "gcn-vs-gcn__augmentation-cora-vs-normal-cora__topk.png"
            ],
        )

        grouped_plots = sorted(
            p.name for p in (self.task_dir / "grouped_signal_plots").glob("*.png")
        )
        self.assertEqual(
            grouped_plots,
            [
                "grouped_signals__graphs__augmentation_test__cora__"
                "mutualknnauc__gcn__topk__condition-pairs.png"
            ],
        )

    def test_filename_slug_rule(self):
        self.assertEqual(
            _safe_name("Normal:cora-vs-Aug:cora"), "normal-cora-vs-aug-cora"
        )
        self.assertEqual(_safe_name("A B/C"), "a-b-c")
        self.assertEqual(_safe_name("--keep.this_1--"), "keep.this_1")
        self.assertEqual(_safe_name(""), "value")
        self.assertEqual(_safe_name("///"), "value")


class TestStaleDirectoryCleanup(unittest.TestCase):
    """Signal plot directories are rebuilt, not merged, on every analysis."""

    def test_stale_signal_plots_are_removed_at_both_levels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = _build_campaign(root)
            analysis_root = root / "figures" / "trial" / "graphs"
            task_root = analysis_root / "tasks" / BENCHMARK / DATASET
            stale_files = []
            for parent in (analysis_root, task_root):
                for dirname in ("signal_plots", "grouped_signal_plots"):
                    stale_dir = parent / dirname
                    stale_dir.mkdir(parents=True, exist_ok=True)
                    stale = stale_dir / "stale.png"
                    stale.write_bytes(b"stale")
                    stale_files.append(stale)

            out = analyze_campaign(campaign_path, "trial")

            # The domain level writes no signal plots at all, so its stale
            # directories must be gone rather than left behind.
            self.assertFalse((out / "signal_plots").exists())
            self.assertFalse((out / "grouped_signal_plots").exists())
            for stale in stale_files:
                self.assertFalse(stale.exists(), f"{stale} survived regeneration")
            self.assertTrue((task_root / "signal_plots").is_dir())
            self.assertTrue((task_root / "grouped_signal_plots").is_dir())


class TestAnalysisModuleBoundaries(unittest.TestCase):
    """Public import surfaces the analysis refactor must keep working."""

    def test_data_modules_import_without_a_plotting_backend(self):
        code = (
            "import builtins\n"
            "_real = builtins.__import__\n"
            "def _guard(name, *args, **kwargs):\n"
            "    if name.split('.')[0] == 'matplotlib':\n"
            "        raise ImportError('matplotlib is unavailable')\n"
            "    return _real(name, *args, **kwargs)\n"
            "builtins.__import__ = _guard\n"
            "from manifold_repsim.resi.analysis import ("
            "loaders, ranking, signals, identifiers, normalize_manifests)\n"
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

    def test_script_wrapper_namespace_exposes_the_public_analysis_api(self):
        import scripts.analyze_resi_results as wrapper

        expected = {
            # Submodules re-exported by the package's own __all__.
            "config",
            "plotting",
            "workflows",
            # A representative name from each split responsibility.
            "analyze_campaign",
            "measure_category",
            "quality_direction",
            "NORMALIZED_COLUMNS",
            "SIGNAL_COLUMNS",
            "normalize_manifests",
            "rank_case_observations",
            "summarize_signals",
            "plot_rank_boxplot",
        }
        missing = sorted(name for name in expected if not hasattr(wrapper, name))
        self.assertEqual(missing, [])

    def test_grouped_comparisons_reject_missing_condition_columns(self):
        """Falling back to setting pairs would silently merge conditions."""
        from manifold_repsim.resi.analysis.signals import (
            summarize_grouped_signal_comparisons,
        )

        signals = pd.DataFrame(
            {
                "domain": ["graphs"],
                "benchmark": [BENCHMARK],
                "dataset": [DATASET],
                "metric": [MEASURE],
                "source_architecture": ["GCN"],
                "target_architecture": ["GCN"],
                "param_name": ["topk"],
                "param_value": [2],
                "score": [0.5],
                "logscale": [True],
                "setting_pair": ["Normal-vs-Augmentation"],
                "seed_pair": ["s0-vs-s1"],
            }
        )
        with self.assertRaisesRegex(ValueError, "condition_pair"):
            summarize_grouped_signal_comparisons(signals)


if __name__ == "__main__":
    unittest.main()
