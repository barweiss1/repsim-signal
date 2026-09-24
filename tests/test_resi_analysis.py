from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from manifold_repsim.resi.analysis.config import (
    MEASURE_CATEGORY_COLORS,
    MEASURE_CATEGORY_ORDER,
    measure_category,
    quality_direction,
)
from manifold_repsim.resi.analysis.loaders import normalize_manifests  # noqa: F401
from manifold_repsim.resi.analysis.ranking import (
    CASE_NAN_REASON,
    SCOPE_COVERAGE_REASON,
    filter_measures_by_coverage,
    normalize_case_rank,
    rank_case_observations,
    rank_scale_case_column,
    select_ranked_functional,
    summarize_domains,
    validate_rank_scale,
)
from manifold_repsim.resi.analysis.signals import (
    combine_signals,
    load_embedded_signals,
    load_legacy_signals,
    load_manifest_legacy_signals,
    signal_input_coverage,
    summarize_grouped_signal_comparisons as _summarize_grouped_signal_comparisons,
    summarize_signals,
)
from manifold_repsim.resi.analysis.boxplots import draw_boxes
from manifold_repsim.resi.analysis.measure_style import order_measures
from manifold_repsim.resi.analysis.plotting import (
    plot_auprc_vs_violation_rate,
    plot_grouped_signal_comparisons,
    plot_quality_heatmaps,
    plot_rank_boxplot,
    plot_signal_groups,
)
from manifold_repsim.resi.analysis.settings import AnalysisSettings
from manifold_repsim.resi.analysis.workflows import analyze_campaign
from manifold_repsim.resi.campaign import ManifestEntry, load_campaign


def _write_campaign_yaml(
    root: Path,
    *,
    domain: str,
    measures: list,
    benchmarks: list,
    quality_measures: list = ["AUPRC"],
    known_missing_models: list | None = None,
) -> Path:
    """Write a minimal campaign YAML under ``root`` and return its path.

    Shared by the end-to-end `analyze_campaign` tests below, which otherwise
    each hand-write the same ``version``/``run_root``/``analysis_root``
    boilerplate around a domain- and benchmark-specific payload. Not shared
    with `test_resi_analysis_schemas.py`'s own `_build_campaign`: that file is
    deliberately self-contained so shared test infrastructure can't mask an
    output-schema regression its locks exist to catch.
    """
    campaign_path = root / "campaign.yaml"
    payload = {
        "version": 1,
        "domain": domain,
        "run_root": str(root / "runs"),
        "analysis_root": str(root / "figures"),
        "quality_measures": list(quality_measures),
        "measures": list(measures),
        "benchmarks": benchmarks,
    }
    if known_missing_models is not None:
        payload["known_missing_models"] = known_missing_models
    campaign_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return campaign_path


def _manifest_entry(
    *,
    kind: str,
    index: int,
    domain: str,
    benchmark: str,
    dataset: str,
    architectures: list,
    measures: list,
    task_dir: Path,
    source_result_path: Path | None = None,
) -> dict:
    """One manifest-entry dict, with every field `read_manifest` expects."""
    return {
        "kind": kind,
        "index": index,
        "domain": domain,
        "benchmark": benchmark,
        "dataset": dataset,
        "architectures": list(architectures),
        "measures": list(measures),
        "config_path": str(task_dir / "config.yaml"),
        "result_path": str(task_dir / "results.parquet"),
        "full_csv_path": str(task_dir / "results_full.csv"),
        "source_result_path": (
            str(source_result_path) if source_result_path is not None else None
        ),
    }


def _write_manifest(path: Path, entries: list) -> None:
    """Write one manifest file from a list of entry dicts, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )


class TestManifestAnalysis(unittest.TestCase):
    def test_boxplot_orders_by_median_with_best_on_top(self):
        """Position 0 is the bottom of the axis, so the best measure is last."""
        ranks = pd.DataFrame(
            {
                "metric": ["best", "best", "best", "worse", "worse", "worse"],
                "mean_rank": [1.0, 1.0, 9.0, 2.0, 2.0, 2.0],
                "mean_normalized_rank": [0.0, 0.0, 0.9, 0.1, 0.1, 0.1],
            }
        )
        self.assertEqual(
            order_measures(ranks, "mean_normalized_rank", higher_is_better=False),
            ["worse", "best"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("matplotlib.axes.Axes.set_yticklabels") as labels:
                plot_rank_boxplot(ranks, Path(tmp) / "rank.png")
            self.assertEqual(labels.call_args.args[0][-1], "best")

    def test_measure_categories_match_paper_and_add_signal(self):
        expected = {
            "SecondOrderCosineSimilarity": "Neighbors",
            "CKA": "RSM",
            "OrthogonalProcrustesCenteredAndNormalized": "Alignment",
            "IMDScore": "Topology",
            "PWCCA": "CCA",
            "MagnitudeDifference": "Statistic",
            "MutualKNNTop10": "Neighbors",
            "CKNNATop10": "Neighbors",
            "RWKArbfSigma05": "RSM",
            "SoftmaxRWKATemp05": "RSM",
            "sRWKArbfSigma05": "RSM",
            "dRWKArbfSigma05": "RSM",
            "CKArbfSigma05": "RSM",
            "MutualKNNAUC": "Signal",
            "CKNNAAUC": "Signal",
            "RWKArbfAUC": "Signal",
            "RWKAsoftmaxAUC": "Signal",
            "CKArbfAUC": "Signal",
            "RWKA": "RSM",
            "RWKA_AUC": "Signal",
            "unknown": "Other",
        }
        self.assertEqual(
            {measure: measure_category(measure) for measure in expected},
            expected,
        )
        self.assertEqual(MEASURE_CATEGORY_ORDER[-2:], ("Signal", "Other"))
        self.assertIn("Signal", MEASURE_CATEGORY_COLORS)

    def test_boxplot_uses_legacy_matplotlib_arguments_when_required(self):
        """Matplotlib renamed `vert` to `orientation`; both spellings must work."""
        empty = {"boxes": [], "medians": [], "whiskers": [], "caps": []}

        class LegacyAxes:
            def __init__(self):
                self.call = None

            def boxplot(self, values, *, vert, **kwargs):
                self.call = (values, vert, kwargs)
                return empty

        axes = LegacyAxes()
        values = [np.array([1.0, 2.0])]
        draw_boxes(axes, values, positions=[0], colors=["#000000"])
        self.assertEqual(axes.call[0], values)
        self.assertFalse(axes.call[1])
        self.assertNotIn("orientation", axes.call[2])
        self.assertFalse(axes.call[2]["showfliers"])

    def test_legacy_jsonl_signal_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "old_signals.jsonl"
            legacy.write_text(
                json.dumps(
                    {
                        "comparison_id": "Normal__GCN__cora__0__cora__1___Normal__GCN__cora__1__cora__1___MutualKNNAUC",
                        "similarity_measure": "MutualKNNAUC",
                        "base_metric": "mutual_knn",
                        "param_name": "topk",
                        "param_values": [2, 4],
                        "scores": [0.2, 0.3],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            campaign_path = root / "campaign.yaml"
            campaign_path.write_text(
                yaml.safe_dump(
                    {
                        "version": 1,
                        "domain": "graphs",
                        "quality_measures": ["AUPRC"],
                        "measures": ["MutualKNNAUC"],
                        "legacy_signal_files": [str(legacy)],
                        "benchmarks": [
                            {
                                "id": "test",
                                "dataset": "cora",
                                "base_config": "unused.yaml",
                                "architectures": ["GCN"],
                                "baselines": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            signals = load_legacy_signals(load_campaign(campaign_path))
            self.assertEqual(len(signals), 2)
            self.assertEqual(signals["seed_pair"].unique().tolist(), ["s0-vs-s1"])
            self.assertEqual(
                signals["condition_pair"].unique().tolist(),
                ["Normal:cora-vs-Normal:cora"],
            )

    def test_campaign_analysis_merges_explicit_baseline_and_manifold_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = _write_campaign_yaml(
                root,
                domain="graphs",
                measures=["MutualKNNTop10"],
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
            for kind, metric, value, index in (
                ("baseline", "CKA", 0.4, 0),
                ("compute", "MutualKNNTop10", 0.8, 0),
            ):
                task_dir = root / kind
                task_dir.mkdir()
                pd.DataFrame({"metric": [metric], "metric_value": [value]}).to_parquet(
                    task_dir / "results.parquet"
                )
                pd.DataFrame(
                    {
                        "similarity_measure": [metric],
                        "quality_measure": ["AUPRC"],
                        "architecture": ["GCN"],
                        "identifier": ["Normal"],
                        "representation_dataset": ["toy"],
                        "value": [value],
                    }
                ).to_csv(task_dir / "results_full.csv", index=False)
                entry = _manifest_entry(
                    kind=kind,
                    index=index,
                    domain="graphs",
                    benchmark="augmentation",
                    dataset="toy",
                    architectures=["GCN"],
                    measures=[metric],
                    task_dir=task_dir,
                )
                _write_manifest(run_dir / f"{kind}_manifest.jsonl", [entry])
            out = analyze_campaign(campaign_path, "trial")
            normalized = pd.read_csv(out / "normalized_values.csv")
            self.assertEqual(set(normalized["source"]), {"baseline", "manifold"})
            ranks = (
                pd.read_csv(out / "per_case_ranks.csv")
                .set_index("metric")["mean_rank"]
                .to_dict()
            )
            self.assertEqual(ranks, {"CKA": 2.0, "MutualKNNTop10": 1.0})
            for filename in (
                "domain_summaries.csv",
                "coverage_nan_report.csv",
                "rank_boxplot.png",
                "signals_long.csv",
                "signals_by_seed.csv",
                "signals_grouped.csv",
                "task_analysis_index.csv",
            ):
                self.assertTrue((out / filename).is_file(), filename)
            task_dir = out / "tasks" / "augmentation" / "toy"
            self.assertTrue((task_dir / "normalized_values.csv").is_file())
            self.assertTrue((task_dir / "rank_summary.csv").is_file())
            self.assertTrue((task_dir / "rank_boxplot.png").is_file())
            self.assertEqual(
                len(list((task_dir / "quality_heatmaps").glob("*.png"))), 1
            )

    def test_manifest_jsonl_signals_are_canonical_and_embedded_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "results.parquet"
            full = root / "results_full.csv"
            sidecar = root / "results_signals.jsonl"
            forward = (
                "Normal__GCN__cora__0__cora__1"
                "___Augmentation__GCN__cora__1__cora__1"
                "___MutualKNNAUC"
            )
            reverse = (
                "Augmentation__GCN__cora__1__cora__1"
                "___Normal__GCN__cora__0__cora__1"
                "___MutualKNNAUC"
            )
            legacy_signal = {
                "comparison_id": forward,
                "metric": "MutualKNNAUC",
                "base_metric": "mutual_knn",
                "param_name": "topk",
                "param_values": [2, 4],
                "scores": [0.2, 0.4],
            }
            reverse_signal = {**legacy_signal, "comparison_id": reverse}
            sidecar.write_text(
                json.dumps(legacy_signal) + "\n" + json.dumps(reverse_signal) + "\n",
                encoding="utf-8",
            )
            embedded_signal = {
                "base_metric": "mutual_knn",
                "param_name": "topk",
                "param_values": [2, 4],
                "scores": [0.8, 0.9],
            }
            pd.DataFrame(
                {
                    "id": [forward],
                    "metric": ["MutualKNNAUC"],
                    "metric_value": [0.85],
                    "similarity_signal": [json.dumps(embedded_signal)],
                }
            ).to_parquet(raw)
            entry = ManifestEntry(
                kind="compute",
                index=0,
                domain="graphs",
                benchmark="augmentation_test",
                dataset="cora",
                architectures=("GCN",),
                measures=("MutualKNNAUC",),
                config_path=str(root / "config.yaml"),
                result_path=str(raw),
                full_csv_path=str(full),
                legacy_signal_paths=(str(sidecar),),
            )
            legacy = load_manifest_legacy_signals([entry])
            embedded = load_embedded_signals([entry])
            combined = combine_signals(legacy, embedded)
            self.assertEqual(len(legacy), 4)
            self.assertEqual(len(combined), 2)
            self.assertEqual(combined["signal_source"].unique().tolist(), ["embedded"])
            self.assertEqual(combined["score"].tolist(), [0.8, 0.9])
            self.assertEqual(combined["canonical_comparison_id"].nunique(), 1)
            coverage = signal_input_coverage([entry]).iloc[0]
            self.assertEqual(coverage["state"], "available_embedded")
            self.assertEqual(int(coverage["rows"]), 3)
            self.assertEqual(int(coverage["signal_files_found"]), 1)

    def test_signal_conditions_distinguish_training_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sidecar = root / "results_signals.jsonl"
            records = []
            for target_dataset in (
                "Gauss_M_ImageNet100DataModule",
                "Gauss_Max_ImageNet100DataModule",
            ):
                records.append(
                    {
                        "comparison_id": (
                            "GaussNoise__ResNet18__Gauss_S_ImageNet100DataModule__0__"
                            "Gauss_Off_ImageNet100DataModule__8"
                            f"___GaussNoise__ResNet18__{target_dataset}__1__"
                            "Gauss_Off_ImageNet100DataModule__8"
                            "___MutualKNNAUC"
                        ),
                        "metric": "MutualKNNAUC",
                        "param_name": "topk",
                        "param_values": [2],
                        "scores": [0.5],
                    }
                )
            sidecar.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            entry = ManifestEntry(
                kind="compute",
                index=0,
                domain="vision",
                benchmark="augmentation",
                dataset="ImageNet100",
                architectures=("ResNet18",),
                measures=("MutualKNNAUC",),
                config_path=str(root / "config.yaml"),
                result_path=str(root / "results.parquet"),
                full_csv_path=str(root / "results_full.csv"),
                legacy_signal_paths=(str(sidecar),),
            )
            signals = load_manifest_legacy_signals([entry])
            self.assertEqual(signals["setting_pair"].nunique(), 1)
            self.assertEqual(
                set(signals["condition_pair"]),
                {
                    "GaussNoise:Gauss_M-vs-GaussNoise:Gauss_S",
                    "GaussNoise:Gauss_Max-vs-GaussNoise:Gauss_S",
                },
            )
            self.assertEqual(signals["condition_seed_pair"].nunique(), 2)
            _, grouped = summarize_signals(signals)
            self.assertEqual(grouped["condition_pair"].nunique(), 2)

    def test_known_missing_model_is_excluded_from_signals_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = _write_campaign_yaml(
                root,
                domain="vision",
                measures=["MutualKNNAUC"],
                benchmarks=[
                    {
                        "id": "randomlabel",
                        "dataset": "ImageNet100",
                        "base_config": "unused.yaml",
                        "architectures": ["ViT_L32"],
                        "baselines": [],
                    }
                ],
                known_missing_models=[
                    {
                        "benchmark": "randomlabel",
                        "dataset": "ImageNet100",
                        "architecture": "ViT_L32",
                        "setting": "Randomlabel",
                        "train_dataset": "RandomLabel_100_IN100_DataModule",
                        "seed": 4,
                        "reason": "Upstream checkpoint has non-finite representations.",
                    }
                ],
            )
            run_dir = root / "runs" / "trial" / "vision"
            task_dir = root / "compute"
            run_dir.mkdir(parents=True)
            task_dir.mkdir()
            raw = task_dir / "results.parquet"
            full = task_dir / "results_full.csv"
            signal = json.dumps(
                {
                    "base_metric": "mutual_knn",
                    "param_name": "topk",
                    "param_values": [2],
                    "scores": [0.4],
                    "auc_value": 0.4,
                    "integration_method": "average",
                    "logscale": True,
                }
            )
            bad_id = (
                "Randomlabel__ViT_L32__RandomLabel_100_IN100_DataModule__4__ImageNet100__12"
                "___Normal__ViT_L32__ImageNet100__0__ImageNet100__12"
                "___MutualKNNAUC"
            )
            valid_id = (
                "Normal__ViT_L32__ImageNet100__0__ImageNet100__12"
                "___Normal__ViT_L32__ImageNet100__1__ImageNet100__12"
                "___MutualKNNAUC"
            )
            pd.DataFrame(
                {
                    "id": [bad_id, valid_id],
                    "metric": ["MutualKNNAUC"] * 2,
                    "metric_value": [np.nan, 0.4],
                    "similarity_signal": [signal, signal],
                }
            ).to_parquet(raw)
            pd.DataFrame(
                {
                    "similarity_measure": ["MutualKNNAUC"],
                    "quality_measure": ["AUPRC"],
                    "architecture": ["ViT_L32"],
                    "identifier": ["Normal"],
                    "representation_dataset": ["ImageNet100"],
                    "value": [0.8],
                }
            ).to_csv(full, index=False)
            entry = _manifest_entry(
                kind="compute",
                index=0,
                domain="vision",
                benchmark="randomlabel",
                dataset="ImageNet100",
                architectures=["ViT_L32"],
                measures=["MutualKNNAUC"],
                task_dir=task_dir,
            )
            _write_manifest(run_dir / "compute_manifest.jsonl", [entry])

            out = analyze_campaign(campaign_path, "trial")
            coverage = pd.read_csv(out / "coverage_nan_report.csv")
            exclusion = coverage[coverage["record_type"] == "known_missing_model"].iloc[
                0
            ]
            self.assertEqual(exclusion["state"], "excluded_known_missing")
            self.assertEqual(int(exclusion["rows"]), 1)
            self.assertIn(
                "RandomLabel_100_IN100_DataModule__4", exclusion["excluded_model"]
            )
            self.assertIn("non-finite", exclusion["exclusion_reason"])

            signals = pd.read_csv(out / "signals_long.csv")
            self.assertEqual(signals["comparison_id"].tolist(), [valid_id])
            normalized = pd.read_csv(out / "normalized_values.csv")
            self.assertEqual(normalized["value"].tolist(), [0.8])

    def test_task_outputs_isolate_repeated_benchmark_across_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = _write_campaign_yaml(
                root,
                domain="graphs",
                measures=["MutualKNNAUC"],
                benchmarks=[
                    {
                        "id": "augmentation_test",
                        "dataset": dataset,
                        "base_config": "unused.yaml",
                        "architectures": ["GCN"],
                        "baselines": [],
                    }
                    for dataset in ("cora", "flickr")
                ],
            )
            run_dir = root / "runs" / "trial" / "graphs"
            run_dir.mkdir(parents=True)
            manifest_rows = []
            for index, (dataset, value) in enumerate((("cora", 0.7), ("flickr", 0.9))):
                task_dir = root / dataset
                task_dir.mkdir()
                comparison_id = (
                    f"Normal__GCN__{dataset}__0__{dataset}__1"
                    f"___Augmentation__GCN__{dataset}__1__{dataset}__1"
                    "___MutualKNNAUC"
                )
                signal = json.dumps(
                    {
                        "base_metric": "mutual_knn",
                        "param_name": "topk",
                        "param_values": [2, 4],
                        "scores": [0.2, 0.4],
                        "logscale": True,
                    }
                )
                pd.DataFrame(
                    {
                        "id": [comparison_id],
                        "metric": ["MutualKNNAUC"],
                        "metric_value": [value],
                        "similarity_signal": [signal],
                    }
                ).to_parquet(task_dir / "results.parquet")
                pd.DataFrame(
                    {
                        "similarity_measure": ["MutualKNNAUC"],
                        "quality_measure": ["AUPRC"],
                        "architecture": ["GCN"],
                        "identifier": ["Normal"],
                        "representation_dataset": [dataset],
                        "value": [value],
                    }
                ).to_csv(task_dir / "results_full.csv", index=False)
                manifest_rows.append(
                    _manifest_entry(
                        kind="compute",
                        index=index,
                        domain="graphs",
                        benchmark="augmentation_test",
                        dataset=dataset,
                        architectures=["GCN"],
                        measures=["MutualKNNAUC"],
                        task_dir=task_dir,
                    )
                )
            _write_manifest(run_dir / "compute_manifest.jsonl", manifest_rows)
            stale_root = root / "figures" / "trial" / "graphs"
            for dirname in ("signal_plots", "grouped_signal_plots"):
                stale_dir = stale_root / dirname
                stale_dir.mkdir(parents=True, exist_ok=True)
                (stale_dir / "stale.png").write_bytes(b"stale")
            for dataset in ("cora", "flickr"):
                for dirname in ("signal_plots", "grouped_signal_plots"):
                    stale_dir = (
                        stale_root / "tasks" / "augmentation_test" / dataset / dirname
                    )
                    stale_dir.mkdir(parents=True, exist_ok=True)
                    (stale_dir / "stale.png").write_bytes(b"stale")

            out = analyze_campaign(campaign_path, "trial")
            task_index = pd.read_csv(out / "task_analysis_index.csv")
            self.assertEqual(set(task_index["dataset"]), {"cora", "flickr"})
            self.assertEqual(task_index["grouped_signal_plots"].tolist(), [1, 1])
            self.assertFalse((out / "signal_plots").exists())
            self.assertFalse((out / "grouped_signal_plots").exists())
            for dataset, expected in (("cora", 0.7), ("flickr", 0.9)):
                task_values = pd.read_csv(
                    out
                    / "tasks"
                    / "augmentation_test"
                    / dataset
                    / "normalized_values.csv"
                )
                self.assertEqual(task_values["dataset"].unique().tolist(), [dataset])
                self.assertEqual(task_values["value"].tolist(), [expected])
                self.assertEqual(
                    len(
                        list(
                            (
                                out
                                / "tasks"
                                / "augmentation_test"
                                / dataset
                                / "grouped_signal_plots"
                            ).glob("*.png")
                        )
                    ),
                    1,
                )

    def test_benchmark_level_boxplot_aggregates_across_datasets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = _write_campaign_yaml(
                root,
                domain="graphs",
                measures=["MutualKNNTop10"],
                benchmarks=[
                    {
                        "id": "label_test",
                        "dataset": dataset,
                        "base_config": "unused.yaml",
                        "architectures": ["GCN"],
                        "baselines": [],
                    }
                    for dataset in ("cora", "flickr")
                ],
            )
            run_dir = root / "runs" / "trial" / "graphs"
            run_dir.mkdir(parents=True)
            baseline_rows = []
            compute_rows = []
            for index, dataset in enumerate(("cora", "flickr")):
                for kind, metric, value in (
                    ("baseline", "CKA", 0.4),
                    ("compute", "MutualKNNTop10", 0.8),
                ):
                    task_dir = root / kind / dataset
                    task_dir.mkdir(parents=True)
                    pd.DataFrame(
                        {"metric": [metric], "metric_value": [value]}
                    ).to_parquet(task_dir / "results.parquet")
                    pd.DataFrame(
                        {
                            "similarity_measure": [metric],
                            "quality_measure": ["AUPRC"],
                            "architecture": ["GCN"],
                            "identifier": ["Normal"],
                            "representation_dataset": [dataset],
                            "value": [value],
                        }
                    ).to_csv(task_dir / "results_full.csv", index=False)
                    entry = _manifest_entry(
                        kind=kind,
                        index=index,
                        domain="graphs",
                        benchmark="label_test",
                        dataset=dataset,
                        architectures=["GCN"],
                        measures=[metric],
                        task_dir=task_dir,
                        source_result_path=(
                            task_dir / "results.parquet" if kind == "baseline" else None
                        ),
                    )
                    (baseline_rows if kind == "baseline" else compute_rows).append(
                        entry
                    )
            _write_manifest(run_dir / "baseline_manifest.jsonl", baseline_rows)
            _write_manifest(run_dir / "compute_manifest.jsonl", compute_rows)

            out = analyze_campaign(campaign_path, "trial")

            benchmark_index = pd.read_csv(out / "benchmark_analysis_index.csv")
            self.assertEqual(benchmark_index["benchmark"].tolist(), ["label_test"])
            self.assertEqual(benchmark_index["datasets"].tolist(), ["cora,flickr"])
            self.assertEqual(benchmark_index["rank_rows"].tolist(), [4])

            benchmark_dir = out / "tasks" / "label_test"
            self.assertTrue((benchmark_dir / "rank_boxplot.png").is_file())
            benchmark_ranks = pd.read_csv(benchmark_dir / "per_case_ranks.csv")
            self.assertEqual(set(benchmark_ranks["dataset"]), {"cora", "flickr"})
            self.assertEqual(set(benchmark_ranks["metric"]), {"CKA", "MutualKNNTop10"})
            rank_summary = pd.read_csv(benchmark_dir / "rank_summary.csv")
            self.assertEqual(set(rank_summary["metric"]), {"CKA", "MutualKNNTop10"})
            self.assertEqual(rank_summary["n_cases"].tolist(), [2, 2])

            # The per-(benchmark, dataset) task level still exists, unaffected,
            # one directory below the new benchmark-level aggregate.
            for dataset in ("cora", "flickr"):
                task_ranks = pd.read_csv(benchmark_dir / dataset / "per_case_ranks.csv")
                self.assertEqual(task_ranks["dataset"].unique().tolist(), [dataset])

    def test_quality_directions_and_matched_layer_ranking(self):
        self.assertEqual(quality_direction("violation_rate"), "lower")
        self.assertEqual(quality_direction("AUPRC"), "higher")
        rows = []
        # Metric M wins model 1 and loses model 2; equal weighting gives the same mean rank.
        for model, values in (
            ("m1", {"A": 0.2, "M": 0.9}),
            ("m2", {"A": 0.8, "M": 0.1}),
        ):
            for metric, value in values.items():
                rows.append(
                    {
                        "domain": "graphs",
                        "benchmark": "layer_test",
                        "dataset": "cora",
                        "architecture": "GCN",
                        "quality_measure": "correlation",
                        "identifier": "Normal",
                        "representation_dataset": "cora",
                        "functional_similarity_measure": "",
                        "observation_id": model,
                        "metric": metric,
                        "value": value,
                    }
                )
        # A field of one is not a field: with M missing, A would be handed rank
        # 1 for having been the only measure left standing.
        rows.extend(
            [
                {**rows[0], "observation_id": "m3", "metric": "A", "value": 1.0},
                {**rows[0], "observation_id": "m3", "metric": "M", "value": np.nan},
            ]
        )
        observation, cases = rank_case_observations(pd.DataFrame(rows))
        self.assertEqual(sorted(observation["observation_id"].unique()), ["m1", "m2"])
        self.assertEqual(observation["n_case_observations"].unique().tolist(), [2])
        means = cases.set_index("metric")["mean_rank"].to_dict()
        self.assertEqual(means, {"A": 1.5, "M": 1.5})
        summary = summarize_domains(cases)
        self.assertEqual(summary["n_cases"].tolist(), [1, 1])

    def test_a_missing_measure_costs_only_itself_that_observation(self):
        """ReSi's `na_option="keep"`: rank what is there, skip what is not.

        The previous convention dropped the whole observation row unless every
        compared measure was finite in it, so one measure's failure cost the
        other two their scores there as well.
        """
        rows = []
        for model, values in (
            ("m1", {"A": 0.9, "B": 0.5, "M": 0.1}),
            # M failed here. A and B were still evaluated and must still rank.
            ("m2", {"A": 0.1, "B": 0.9, "M": np.nan}),
        ):
            for metric, value in values.items():
                rows.append(
                    {
                        "domain": "graphs",
                        "benchmark": "layer_test",
                        "dataset": "cora",
                        "architecture": "GCN",
                        "quality_measure": "correlation",
                        "identifier": "Normal",
                        "representation_dataset": "cora",
                        "functional_similarity_measure": "",
                        "observation_id": model,
                        "metric": metric,
                        "value": value,
                    }
                )
        observation, cases = rank_case_observations(pd.DataFrame(rows))
        self.assertEqual(sorted(observation["observation_id"].unique()), ["m1", "m2"])
        # m2 ranked two measures, m1 three, and each row says which.
        self.assertEqual(
            observation.set_index(["observation_id", "metric"])["rank"].to_dict(),
            {
                ("m1", "A"): 1.0,
                ("m1", "B"): 2.0,
                ("m1", "M"): 3.0,
                ("m2", "A"): 2.0,
                ("m2", "B"): 1.0,
            },
        )
        by_metric = cases.set_index("metric")
        self.assertEqual(
            by_metric["mean_rank"].to_dict(), {"A": 1.5, "B": 1.5, "M": 3.0}
        )
        # M keeps the one observation it produced; A and B keep both.
        self.assertEqual(
            by_metric["n_ranked_observations"].to_dict(), {"A": 2, "B": 2, "M": 1}
        )
        self.assertEqual(by_metric["n_case_observations"].unique().tolist(), [2])
        # Normalization uses the field each observation actually ranked, so
        # B's win over one competitor and its win over two are both 0.0.
        self.assertAlmostEqual(float(by_metric.loc["B", "mean_normalized_rank"]), 0.25)

    def test_uneven_field_sizes_are_pooled_on_a_normalized_scale(self):
        """Field size alone must not decide the domain summary's order.

        Recreated baselines carry fewer native measures than the archived ones,
        so one domain pools cases that ranked different numbers of measures.
        Mid-pack is 10.0 in a 19-measure case and 17.5 in a 34-measure case, so
        pooling raw ranks would sort a mediocre measure from the small case
        above a better one from the large case.
        """
        rows = []
        # `small` places 4th of 6 in its case; `large` places 4th of 12 in a
        # disjoint case, which is the better placement despite the equal rank.
        cases = (("small-case", 6, "small"), ("large-case", 12, "large"))
        for identifier, field_size, winner in cases:
            for position in range(field_size):
                metric = winner if position == 3 else f"{identifier}-filler{position}"
                rows.append(
                    {
                        "domain": "vision",
                        "benchmark": "augmentation",
                        "dataset": "CIFAR100",
                        "architecture": "ResNet18",
                        "quality_measure": "AUPRC",
                        "identifier": identifier,
                        "representation_dataset": "CIFAR100",
                        "functional_similarity_measure": "",
                        "observation_id": "layer1",
                        "metric": metric,
                        # Descending values, so position 0 ranks first.
                        "value": 1.0 - 0.01 * position,
                    }
                )
        _, case_ranks = rank_case_observations(pd.DataFrame(rows))
        by_metric = case_ranks.set_index("metric")
        self.assertEqual(by_metric.loc["small", "mean_rank"], 4.0)
        self.assertEqual(by_metric.loc["large", "mean_rank"], 4.0)
        self.assertAlmostEqual(by_metric.loc["small", "mean_normalized_rank"], 3 / 5)
        self.assertAlmostEqual(by_metric.loc["large", "mean_normalized_rank"], 3 / 11)

        summary = summarize_domains(case_ranks).set_index("metric")
        self.assertLess(
            summary.loc["large", "mean_normalized_rank"],
            summary.loc["small", "mean_normalized_rank"],
        )
        # Identical raw ranks would have left the order to the tie-break on
        # measure name, putting `large` after `small` alphabetically.
        ordered = summarize_domains(case_ranks)["metric"].tolist()
        self.assertLess(ordered.index("large"), ordered.index("small"))

    def test_rank_scale_selects_the_sort_order_without_changing_columns(self):
        """`raw` restores the pre-normalization ordering, same columns either way."""
        rows = []
        cases = (("small-case", 6, "small"), ("large-case", 12, "large"))
        for identifier, field_size, winner in cases:
            for position in range(field_size):
                metric = winner if position == 3 else f"{identifier}-filler{position}"
                rows.append(
                    {
                        "domain": "vision",
                        "benchmark": "augmentation",
                        "dataset": "CIFAR100",
                        "architecture": "ResNet18",
                        "quality_measure": "AUPRC",
                        "identifier": identifier,
                        "representation_dataset": "CIFAR100",
                        "functional_similarity_measure": "",
                        "observation_id": "layer1",
                        "metric": metric,
                        "value": 1.0 - 0.01 * position,
                    }
                )
        _, case_ranks = rank_case_observations(pd.DataFrame(rows))

        normalized = summarize_domains(case_ranks, rank_scale="normalized")
        raw = summarize_domains(case_ranks, rank_scale="raw")
        # Switching scales must never change the schema, only the order.
        self.assertEqual(normalized.columns.tolist(), raw.columns.tolist())
        self.assertEqual(
            sorted(normalized["metric"].tolist()), sorted(raw["metric"].tolist())
        )

        normalized_order = normalized["metric"].tolist()
        raw_order = raw["metric"].tolist()
        self.assertLess(
            normalized_order.index("large"), normalized_order.index("small")
        )
        # Raw ranks tie at 4.0, so the alphabetical tie-break decides instead.
        self.assertLess(raw_order.index("large"), raw_order.index("small"))
        pd.testing.assert_frame_equal(summarize_domains(case_ranks), raw)

    def test_rank_scale_is_validated_and_defaults(self):
        self.assertEqual(validate_rank_scale(None), "raw")
        self.assertEqual(validate_rank_scale(" normalized "), "normalized")
        self.assertEqual(rank_scale_case_column("normalized"), "mean_normalized_rank")
        self.assertEqual(rank_scale_case_column(None), "mean_rank")
        with self.assertRaises(ValueError):
            validate_rank_scale("percentile")
        with self.assertRaises(ValueError):
            summarize_domains(pd.DataFrame(), rank_scale="percentile")

    def test_rank_boxplot_follows_the_selected_scale(self):
        case_ranks = pd.DataFrame(
            {
                "metric": ["a", "b"],
                "mean_normalized_rank": [0.9, 0.1],
                "mean_rank": [1.0, 9.0],
            }
        )
        # The two columns disagree on which measure is better, so the order
        # proves which one the plot actually read.
        self.assertEqual(
            order_measures(case_ranks, "mean_rank", higher_is_better=False),
            ["b", "a"],
        )
        self.assertEqual(
            order_measures(case_ranks, "mean_normalized_rank", higher_is_better=False),
            ["a", "b"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            for scale, expected in (
                ("raw", "Mean rank per case (1 = best)"),
                (
                    "normalized",
                    "Mean normalized rank per case (0 = best, 1 = worst)",
                ),
            ):
                with mock.patch("matplotlib.axes.Axes.set_xlabel") as set_xlabel:
                    plot_rank_boxplot(
                        case_ranks, Path(tmp) / f"rank-{scale}.png", rank_scale=scale
                    )
                set_xlabel.assert_called_once_with(expected)

    def test_normalized_rank_spans_the_unit_interval(self):
        """Best is 0 and worst is 1 regardless of how many measures competed."""
        self.assertEqual(normalize_case_rank(1.0, 2), 0.0)
        self.assertEqual(normalize_case_rank(2.0, 2), 1.0)
        self.assertEqual(normalize_case_rank(1.0, 34), 0.0)
        self.assertEqual(normalize_case_rank(34.0, 34), 1.0)
        # A tie at the top of a 3-measure field sits midway to second place.
        self.assertAlmostEqual(normalize_case_rank(1.5, 3), 0.25)

    def test_output_correlation_functional_measure_defines_case(self):
        rows = []
        for functional, offset in (("JSD", 0.0), ("Disagreement", 0.1)):
            for metric, value in (("A", 0.4 + offset), ("M", 0.7 + offset)):
                rows.append(
                    {
                        "domain": "vision",
                        "benchmark": "accoutput",
                        "dataset": "ImageNet100",
                        "architecture": "ResNet18",
                        "quality_measure": "spearmanr",
                        "identifier": "Normal",
                        "representation_dataset": "ImageNet100",
                        "functional_similarity_measure": functional,
                        "observation_id": functional,
                        "metric": metric,
                        "value": value,
                    }
                )
        _, cases = rank_case_observations(pd.DataFrame(rows))
        self.assertEqual(cases["functional_similarity_measure"].nunique(), 2)

    def test_embedded_signals_seed_grouping_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "results.parquet"
            full = root / "results_full.csv"
            signal = {
                "base_metric": "mutual_knn",
                "param_name": "topk",
                "param_values": [2, 4],
                "scores": [0.2, 0.4],
                "auc_value": 0.3,
                "integration_method": "average",
                "logscale": True,
            }
            comparison_id = "Normal__GCN__cora__0__cora__1___Augmentation__GCN__cora__1__cora__1___MutualKNNAUC"
            pd.DataFrame(
                {
                    "id": [comparison_id],
                    "metric": ["MutualKNNAUC"],
                    "metric_value": [0.3],
                    "similarity_signal": [json.dumps(signal)],
                }
            ).to_parquet(raw)
            entry = ManifestEntry(
                kind="compute",
                index=0,
                domain="graphs",
                benchmark="augmentation_test",
                dataset="cora",
                architectures=("GCN",),
                measures=("MutualKNNAUC",),
                config_path=str(root / "config.yaml"),
                result_path=str(raw),
                full_csv_path=str(full),
            )
            signals = load_embedded_signals([entry])
            self.assertEqual(len(signals), 2)
            self.assertEqual(signals["setting_pair"].iloc[0], "Augmentation-vs-Normal")
            by_seed, grouped = summarize_signals(signals)
            self.assertEqual(grouped["n_seeds"].tolist(), [1, 1])
            plot_rank_boxplot(
                pd.DataFrame({"metric": ["A", "M"], "mean_rank": [2.0, 1.0]}),
                root / "rank.png",
            )
            outputs = plot_signal_groups(by_seed, grouped, root / "signals")
            self.assertTrue((root / "rank.png").is_file())
            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].is_file())

    def test_grouped_signal_comparisons_average_layers_within_seed(self):
        common = {
            "domain": "vision",
            "benchmark": "randomlabel",
            "dataset": "ImageNet100",
            "metric": "MutualKNNAUC",
            "source_architecture": "ResNet18",
            "target_architecture": "ResNet18",
            "param_name": "topk",
            "logscale": True,
        }
        rows = [
            {
                **common,
                "condition_pair": "Normal-vs-RandomLabel",
                "condition_seed_pair": "s0-vs-s1",
                "param_value": 2,
                "score": 0.2,
            },
            {
                **common,
                "condition_pair": "Normal-vs-RandomLabel",
                "condition_seed_pair": "s0-vs-s1",
                "param_value": 2,
                "score": 0.4,
            },
            {
                **common,
                "condition_pair": "Normal-vs-RandomLabel",
                "condition_seed_pair": "s2-vs-s3",
                "param_value": 2,
                "score": 0.7,
            },
            {
                **common,
                "condition_pair": "Normal-vs-Normal",
                "condition_seed_pair": "s0-vs-s1",
                "param_value": 2,
                "score": 0.1,
            },
            {
                **common,
                "condition_pair": "Normal-vs-Normal",
                "condition_seed_pair": "s2-vs-s3",
                "param_value": 2,
                "score": 0.3,
            },
            {
                **common,
                "source_architecture": "ResNet18",
                "target_architecture": "VGG11",
                "condition_pair": "Normal-vs-RandomLabel",
                "condition_seed_pair": "s0-vs-s1",
                "param_value": 2,
                "score": 1.0,
            },
        ]
        signals = pd.DataFrame(rows)
        summary = _summarize_grouped_signal_comparisons(signals)
        randomlabel = summary[
            summary["condition_pair"].eq("Normal-vs-RandomLabel")
            & summary["param_value"].eq(2)
        ].iloc[0]
        self.assertAlmostEqual(randomlabel["mean_score"], 0.5)
        self.assertAlmostEqual(randomlabel["std_score"], np.std([0.3, 0.7], ddof=1))
        self.assertEqual(int(randomlabel["n_seeds"]), 2)
        self.assertEqual(summary["condition_pair"].nunique(), 2)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("matplotlib.axes.Axes.set_xscale") as set_xscale:
                outputs = plot_grouped_signal_comparisons(signals, Path(tmp))
            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].is_file())
            self.assertIn("randomlabel", outputs[0].name)
            set_xscale.assert_called_with("log")

    def test_task_quality_plots_are_conditional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for source, metric, auprc, violation in (
                ("baseline", "CKA", 0.6, 0.3),
                ("manifold", "MutualKNNTop10", 0.8, 0.1),
            ):
                for quality, value in (("AUPRC", auprc), ("violation_rate", violation)):
                    rows.append(
                        {
                            "domain": "vision",
                            "benchmark": "augmentation",
                            "dataset": "ImageNet100",
                            "architecture": "ResNet18",
                            "observation_id": "Normal|ImageNet100|||",
                            "metric": metric,
                            "quality_measure": quality,
                            "value": value,
                            "source": source,
                            "functional_similarity_measure": "",
                        }
                    )
            values = pd.DataFrame(rows)
            heatmaps = plot_quality_heatmaps(values, root / "heatmaps")
            scatter = plot_auprc_vs_violation_rate(values, root / "scatter.png")
            self.assertEqual(len(heatmaps), 2)
            self.assertTrue(all(path.is_file() for path in heatmaps))
            self.assertEqual(scatter, root / "scatter.png")
            self.assertTrue(scatter.is_file())


def _coverage_rows(cases, metrics, nan_for=()):
    """Build normalized values for `cases` x `metrics`, four observations each.

    ``nan_for`` holds ``(case, metric, observation)`` triples to make NaN.
    """
    rows = []
    for case in cases:
        for metric in metrics:
            for observation in range(4):
                rows.append(
                    {
                        "domain": "vision",
                        "benchmark": "augmentation",
                        "dataset": "CIFAR100",
                        "architecture": case,
                        "quality_measure": "AUPRC",
                        "identifier": "id",
                        "representation_dataset": "CIFAR100",
                        "functional_similarity_measure": "",
                        "observation_id": f"layer{observation}",
                        "metric": metric,
                        "value": (
                            np.nan
                            if (case, metric, observation) in nan_for
                            else 0.9 - 0.01 * observation
                        ),
                    }
                )
    return pd.DataFrame(rows)


class TestMeasureCoverageFiltering(unittest.TestCase):
    """Measures too incomplete to report are dropped, per case then per scope."""

    def test_case_filter_removes_a_measure_ranked_on_a_flattering_subset(self):
        """A mostly-NaN measure must not be ranked on the few cells it survived.

        Ranking now skips a measure on the observations it failed, so its
        failures cost it nothing at all: without this filter `Bad` is scored on
        one observation out of four and pooled as an equal.
        """
        cases = ["arch0"]
        metrics = ["Good1", "Good2", "Bad"]
        nan_for = {("arch0", "Bad", index) for index in range(3)}
        values = _coverage_rows(cases, metrics, nan_for)

        _, unfiltered = rank_case_observations(values)
        coverage = unfiltered.set_index("metric")["n_ranked_observations"].to_dict()
        self.assertEqual(coverage, {"Good1": 4, "Good2": 4, "Bad": 1})
        self.assertEqual(int(unfiltered["n_compared_metrics"].max()), 3)

        kept, report = filter_measures_by_coverage(
            values,
            max_nan_fraction=0.5,
            min_case_coverage=0.0,
            scope_columns=["domain"],
        )
        _, filtered = rank_case_observations(kept)
        # The measures that were always present keep every observation either
        # way; what the filter changes is that `Bad` no longer competes.
        self.assertEqual(int(filtered["n_ranked_observations"].max()), 4)
        self.assertEqual(sorted(filtered["metric"].unique()), ["Good1", "Good2"])
        dropped = report[report["reason"] == CASE_NAN_REASON]
        self.assertEqual(dropped["metric"].tolist(), ["Bad"])
        self.assertAlmostEqual(float(dropped["nan_fraction"].iloc[0]), 0.75)

    def test_sparse_measure_leaves_the_scope_and_is_recorded(self):
        """Below the coverage threshold a measure is removed from the scope entirely."""
        cases = [f"arch{index}" for index in range(10)]
        metrics = ["Good", "Sparse"]
        nan_for = {
            (case, "Sparse", observation)
            for case in cases[:3]
            for observation in range(4)
        }
        values = _coverage_rows(cases, metrics, nan_for)
        kept, report = filter_measures_by_coverage(
            values,
            max_nan_fraction=0.5,
            min_case_coverage=0.95,
            scope_columns=["domain"],
        )
        self.assertEqual(sorted(kept["metric"].unique()), ["Good"])
        scope_rows = report[report["reason"] == SCOPE_COVERAGE_REASON]
        self.assertEqual(scope_rows["metric"].tolist(), ["Sparse"])
        self.assertEqual(int(scope_rows["cases_present"].iloc[0]), 7)
        self.assertEqual(int(scope_rows["cases_in_scope"].iloc[0]), 10)
        self.assertAlmostEqual(float(scope_rows["case_coverage"].iloc[0]), 0.7)

    def test_case_dead_for_every_measure_does_not_exclude_the_field(self):
        """A case no measure survives must not count against the measures.

        The language run hit this: six correlation cases lost every measure to
        the per-case filter, which dragged all nineteen below the coverage
        threshold at once and emptied the ranking.
        """
        cases = [f"arch{index}" for index in range(5)]
        metrics = ["Good", "Also"]
        nan_for = {
            ("arch0", metric, observation)
            for metric in metrics
            for observation in range(4)
        }
        values = _coverage_rows(cases, metrics, nan_for)
        kept, report = filter_measures_by_coverage(
            values,
            max_nan_fraction=0.5,
            min_case_coverage=0.95,
            scope_columns=["domain"],
        )
        self.assertEqual(sorted(kept["metric"].unique()), ["Also", "Good"])
        self.assertTrue(report[report["reason"] == SCOPE_COVERAGE_REASON].empty)

    def test_coverage_denominator_counts_only_surviving_cases(self):
        """The denominator is cases still reportable, not every case attempted."""
        cases = [f"arch{index}" for index in range(5)]
        metrics = ["Good", "Sparse"]
        nan_for = {
            ("arch0", metric, observation)
            for metric in metrics
            for observation in range(4)
        }
        nan_for |= {("arch1", "Sparse", observation) for observation in range(4)}
        values = _coverage_rows(cases, metrics, nan_for)
        kept, report = filter_measures_by_coverage(
            values,
            max_nan_fraction=0.5,
            min_case_coverage=0.95,
            scope_columns=["domain"],
        )
        self.assertEqual(sorted(kept["metric"].unique()), ["Good"])
        scope_rows = report[report["reason"] == SCOPE_COVERAGE_REASON]
        self.assertEqual(scope_rows["metric"].tolist(), ["Sparse"])
        self.assertEqual(int(scope_rows["cases_in_scope"].iloc[0]), 4)
        self.assertEqual(int(scope_rows["cases_present"].iloc[0]), 3)

    def test_fully_covered_measures_are_untouched(self):
        values = _coverage_rows(["arch0", "arch1"], ["A", "B"])
        kept, report = filter_measures_by_coverage(
            values,
            max_nan_fraction=0.5,
            min_case_coverage=0.95,
            scope_columns=["domain"],
        )
        self.assertTrue(report.empty)
        self.assertEqual(len(kept), len(values))
        self.assertEqual(list(kept.columns), list(values.columns))

    def test_permissive_thresholds_disable_filtering(self):
        """`max_nan_fraction: 1` and `min_case_coverage: 0` keep every measure."""
        nan_for = {("arch0", "Bad", observation) for observation in range(4)}
        values = _coverage_rows(["arch0", "arch1"], ["Good", "Bad"], nan_for)
        kept, report = filter_measures_by_coverage(
            values,
            max_nan_fraction=1.0,
            min_case_coverage=0.0,
            scope_columns=["domain"],
        )
        self.assertTrue(report.empty)
        self.assertEqual(sorted(kept["metric"].unique()), ["Bad", "Good"])

    def test_scope_columns_select_the_granularity(self):
        """A measure can clear the domain threshold and fail one task's."""
        rows = []
        for dataset, bad_cases in (("CIFAR100", 4), ("ImageNet100", 0)):
            frame = _coverage_rows(
                [f"{dataset}-arch{index}" for index in range(4)],
                ["Good", "Patchy"],
                {
                    (f"{dataset}-arch{index}", "Patchy", observation)
                    for index in range(bad_cases)
                    for observation in range(4)
                },
            )
            frame["dataset"] = dataset
            rows.append(frame)
        values = pd.concat(rows, ignore_index=True)

        by_task, task_report = filter_measures_by_coverage(
            values,
            max_nan_fraction=0.5,
            min_case_coverage=0.95,
            scope_columns=["benchmark", "dataset"],
        )
        surviving = {
            dataset: sorted(group["metric"].unique())
            for dataset, group in by_task.groupby("dataset")
        }
        self.assertEqual(surviving["CIFAR100"], ["Good"])
        self.assertEqual(surviving["ImageNet100"], ["Good", "Patchy"])
        self.assertEqual(
            task_report[task_report["reason"] == SCOPE_COVERAGE_REASON][
                "scope"
            ].tolist(),
            ["augmentation|CIFAR100"],
        )


class TestAnalysisMeasureSelection(unittest.TestCase):
    """End-to-end effect of an analysis settings file on the written outputs."""

    MEASURES = ("CKA", "SVCCA", "MutualKNNTop10")

    def _campaign(
        self,
        root: Path,
        nan_cells: set[tuple[str, str]] = frozenset(),
        functional_measures: tuple[str, ...] = (),
    ) -> Path:
        """Write a one-benchmark campaign whose results carry three measures.

        ``nan_cells`` holds ``(quality_measure, measure)`` pairs to write as NaN,
        so a test can make a measure unreportable in part of the run.

        ``functional_measures`` writes one row set per named functional
        similarity measure. Left empty, no such column is written at all, so
        every other test sees the results unchanged.
        """
        campaign_path = root / "campaign.yaml"
        campaign_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "domain": "graphs",
                    "run_root": str(root / "runs"),
                    "analysis_root": str(root / "figures"),
                    # Two ranked quality measures so coverage has two cases to
                    # count, plus one that is reported and never ranked.
                    "quality_measures": ["AUPRC", "correlation", "violation_rate"],
                    "measures": ["MutualKNNTop10"],
                    "benchmarks": [
                        {
                            "id": "augmentation",
                            "dataset": "toy",
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
        task_dir = root / "compute"
        task_dir.mkdir()
        raw = task_dir / "results.parquet"
        full = task_dir / "results_full.csv"
        pd.DataFrame(
            {"metric": list(self.MEASURES), "metric_value": [0.4, 0.6, 0.8]}
        ).to_parquet(raw)
        # Several quality measures so a quality_measures override has something
        # to narrow, and three measures so ranking has a field to compare.
        rows = []
        for quality in ("AUPRC", "correlation", "violation_rate"):
            for measure, value in zip(self.MEASURES, (0.4, 0.6, 0.8)):
                row = {
                    "similarity_measure": measure,
                    "quality_measure": quality,
                    "architecture": "GCN",
                    "identifier": "Normal",
                    "representation_dataset": "toy",
                    "value": (np.nan if (quality, measure) in nan_cells else value),
                }
                if not functional_measures:
                    rows.append(row)
                    continue
                for functional in functional_measures:
                    rows.append({**row, "functional_similarity_measure": functional})
        pd.DataFrame(rows).to_csv(full, index=False)
        payload = {
            "kind": "compute",
            "index": 0,
            "domain": "graphs",
            "benchmark": "augmentation",
            "dataset": "toy",
            "architectures": ["GCN"],
            "measures": list(self.MEASURES),
            "config_path": str(task_dir / "config.yaml"),
            "result_path": str(raw),
            "full_csv_path": str(full),
            "source_result_path": None,
        }
        (run_dir / "compute_manifest.jsonl").write_text(
            json.dumps(payload) + "\n", encoding="utf-8"
        )
        return campaign_path

    def test_excluded_functional_measure_leaves_the_analysis(self):
        """A functional similarity measure named for exclusion is not reported.

        This exists for the asymmetry where the archive scored a functional
        measure for none of its own measures while the recreated baselines
        score it for all of theirs: the extra cases are coverable only by the
        new measures, so the established ones fall below the coverage
        threshold and vanish from the ranking.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(
                root, functional_measures=("JSD", "AbsoluteAccDiff")
            )
            settings = AnalysisSettings(
                exclude_functional_measures=("AbsoluteAccDiff",)
            )
            out = analyze_campaign(campaign_path, "trial", settings=settings)
            normalized = pd.read_csv(out / "normalized_values.csv")
            snapshot = yaml.safe_load((out / "config.yaml").read_text())
        present = set(normalized["functional_similarity_measure"].astype(str))
        self.assertIn("JSD", present)
        self.assertNotIn("AbsoluteAccDiff", present)
        self.assertEqual(snapshot["exclude_functional_measures"], ["AbsoluteAccDiff"])

    def test_coverage_threshold_drops_a_measure_and_records_why(self):
        """A measure unreportable in part of the run leaves the pooled summary.

        Ranking is within a case, so a measure absent from one case is never
        scored badly for it -- its pooled mean is simply drawn from the case it
        survived. That is the asymmetry the thresholds exist to remove.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root, nan_cells={("AUPRC", "SVCCA")})
            out = analyze_campaign(campaign_path, "trial")
            summary = pd.read_csv(out / "domain_summaries.csv")
            exclusions = pd.read_csv(out / "measure_exclusions.csv")
            snapshot = yaml.safe_load((out / "config.yaml").read_text())
        self.assertEqual(set(summary["metric"]), {"CKA", "MutualKNNTop10"})
        scope_rows = exclusions[exclusions["reason"] == SCOPE_COVERAGE_REASON]
        self.assertEqual(scope_rows["metric"].tolist(), ["SVCCA"])
        self.assertEqual(int(scope_rows["cases_present"].iloc[0]), 1)
        self.assertEqual(int(scope_rows["cases_in_scope"].iloc[0]), 2)
        case_rows = exclusions[exclusions["reason"] == CASE_NAN_REASON]
        self.assertEqual(case_rows["metric"].tolist(), ["SVCCA"])
        self.assertEqual(case_rows["quality_measure"].tolist(), ["AUPRC"])
        # The thresholds that produced this belong in the run's own record.
        self.assertEqual(snapshot["max_nan_fraction"], 0.5)
        self.assertEqual(snapshot["min_case_coverage"], 0.90)

    def test_permissive_thresholds_keep_a_partially_missing_measure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root, nan_cells={("AUPRC", "SVCCA")})
            settings = AnalysisSettings(max_nan_fraction=1.0, min_case_coverage=0.0)
            out = analyze_campaign(campaign_path, "trial", settings=settings)
            summary = pd.read_csv(out / "domain_summaries.csv")
            exclusions = pd.read_csv(out / "measure_exclusions.csv")
        self.assertIn("SVCCA", set(summary["metric"]))
        self.assertTrue(exclusions.empty)

    def test_conformity_rate_is_reported_but_never_ranked(self):
        """ReSi's quality filter, applied where it belongs.

        Leaving `violation_rate` in the ranking would give every design test
        two cases against correlation's one, doubling its weight in the domain
        summary and the boxplot. It is still a reported value, so it must
        survive everywhere that is not a rank.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root)
            out = analyze_campaign(campaign_path, "trial")
            normalized = pd.read_csv(out / "normalized_values.csv")
            ranks = pd.read_csv(out / "per_case_ranks.csv")
            published = pd.read_csv(out / "tables" / "published_case_ranks.csv")
            snapshot = yaml.safe_load((out / "config.yaml").read_text())
        self.assertIn("violation_rate", set(normalized["quality_measure"]))
        self.assertEqual(set(ranks["quality_measure"]), {"AUPRC", "correlation"})
        # The published ranking applies the same filter, so the two figures
        # rank the same cells and differ only in how they rank them.
        self.assertNotIn("Conformity Rate", set(published["Eval."].astype(str)))
        self.assertEqual(
            snapshot["ranked_quality_measures"], ["AUPRC", "spearmanr", "correlation"]
        )

    def test_output_correlation_ranks_on_jsd_and_accuracy_stays_its_own_task(self):
        """Six tasks, not five merged: JSD names one, AbsoluteAccDiff another.

        JSD and Disagreement are two readings of the same question, so ranking
        both would give output correlation two cases per cell where every other
        test has one. AbsoluteAccDiff is a different task and is kept.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(
                root, functional_measures=("JSD", "Disagreement", "AbsoluteAccDiff")
            )
            out = analyze_campaign(campaign_path, "trial")
            normalized = pd.read_csv(out / "normalized_values.csv")
            ranks = pd.read_csv(out / "per_case_ranks.csv")
            snapshot = yaml.safe_load((out / "config.yaml").read_text())
        # Disagreement is still reported; it just does not earn a case.
        self.assertIn(
            "Disagreement", set(normalized["functional_similarity_measure"].astype(str))
        )
        self.assertEqual(
            set(ranks["functional_similarity_measure"].astype(str)),
            {"JSD", "AbsoluteAccDiff"},
        )
        self.assertEqual(
            snapshot["ranked_functional_measures"], ["JSD", "AbsoluteAccDiff"]
        )

    def test_accuracy_correlation_survives_where_only_some_of_it_exists(self):
        """AbsoluteAccDiff is missing for whole domains and architectures.

        Where it exists it is reported; where it does not, the task is absent
        for that slice rather than dropping the task or padding it.
        """
        rows = []
        for architecture, has_accuracy in (("BERT-L", True), ("SmolLM2", False)):
            for functional in ("JSD", "AbsoluteAccDiff"):
                for metric, value in (("A", 0.4), ("M", 0.7)):
                    rows.append(
                        {
                            "domain": "language",
                            "benchmark": "correlation",
                            "dataset": "sst2",
                            "architecture": architecture,
                            "quality_measure": "spearmanr",
                            "identifier": "Normal",
                            "representation_dataset": "sst2",
                            "functional_similarity_measure": functional,
                            "observation_id": f"{architecture}|{functional}",
                            "metric": metric,
                            "value": (
                                np.nan
                                if functional == "AbsoluteAccDiff" and not has_accuracy
                                else value
                            ),
                        }
                    )
        kept = select_ranked_functional(pd.DataFrame(rows))
        _, cases = rank_case_observations(kept)
        pairs = set(zip(cases["architecture"], cases["functional_similarity_measure"]))
        self.assertIn(("BERT-L", "AbsoluteAccDiff"), pairs)
        self.assertIn(("SmolLM2", "JSD"), pairs)
        # SmolLM2 has no accuracy to difference, so it contributes no case
        # there -- and that does not cost BERT-L its own.
        self.assertNotIn(("SmolLM2", "AbsoluteAccDiff"), pairs)

    def test_correlation_benchmark_writes_one_task_directory_each(self):
        """The two correlation tasks get their own per-task outputs.

        Keying these on the benchmark id pooled `Output Corr.` and `Acc Corr.`
        into one ranking labelled with the benchmark, which is exactly what
        splitting them into two tasks exists to prevent.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(
                root, functional_measures=("JSD", "Disagreement", "AbsoluteAccDiff")
            )
            out = analyze_campaign(campaign_path, "trial")
            tasks = sorted(p.name for p in (out / "tasks").iterdir() if p.is_dir())
            index = pd.read_csv(out / "task_analysis_index.csv")
            outcorr = pd.read_csv(
                out / "tasks" / "outcorr" / "toy" / "normalized_values.csv"
            )
            acccorr = pd.read_csv(
                out / "tasks" / "acccorr" / "toy" / "normalized_values.csv"
            )
            summary = pd.read_csv(
                out / "tasks" / "outcorr" / "toy" / "rank_summary.csv"
            )

        self.assertEqual(tasks, ["acccorr", "outcorr"])
        self.assertNotIn("augmentation", tasks)
        self.assertEqual(sorted(set(index["task"])), ["acccorr", "outcorr"])
        # Disagreement belongs to the output task even though it is not ranked
        # there, so its values are reported beside JSD's.
        self.assertEqual(
            sorted(set(outcorr["functional_similarity_measure"].astype(str))),
            ["Disagreement", "JSD"],
        )
        self.assertEqual(
            sorted(set(acccorr["functional_similarity_measure"].astype(str))),
            ["AbsoluteAccDiff"],
        )
        self.assertEqual(summary["task"].unique().tolist(), ["outcorr"])

    def test_single_task_benchmarks_keep_their_own_directory(self):
        """Only the correlation benchmarks split; every other path is unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root)
            out = analyze_campaign(campaign_path, "trial")
            tasks = sorted(p.name for p in (out / "tasks").iterdir() if p.is_dir())
        self.assertEqual(tasks, ["augmentation"])

    def test_task_directory_names_agree_with_the_table_taxonomy(self):
        """One grouping of the functional measures, spelled in two places.

        `taxonomy` imports from `config`, so the directory names cannot live
        beside the display labels they must agree with. This is the guard.
        """
        from manifold_repsim.resi.analysis.config import CORRELATION_TASK_DIRS
        from manifold_repsim.resi.analysis.tables.taxonomy import (
            FUNCTIONAL_TEST_LABELS,
        )

        self.assertEqual(set(CORRELATION_TASK_DIRS), set(FUNCTIONAL_TEST_LABELS))
        by_dir: dict[str, set[str]] = {}
        by_label: dict[str, set[str]] = {}
        for measure, directory in CORRELATION_TASK_DIRS.items():
            by_dir.setdefault(directory, set()).add(measure)
        for measure, label in FUNCTIONAL_TEST_LABELS.items():
            by_label.setdefault(label, set()).add(measure)
        self.assertEqual(
            sorted(by_dir.values(), key=sorted),
            sorted(by_label.values(), key=sorted),
        )

    def test_settings_restrict_reported_measures_and_snapshot_the_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root)
            settings = AnalysisSettings(measures=("CKA", "MutualKNNTop10"))
            out = analyze_campaign(campaign_path, "trial", settings=settings)
            normalized = pd.read_csv(out / "normalized_values.csv")
            ranks = pd.read_csv(out / "per_case_ranks.csv")
            snapshot = yaml.safe_load((out / "config.yaml").read_text())
        self.assertEqual(set(normalized["metric"]), {"CKA", "MutualKNNTop10"})
        self.assertEqual(set(ranks["metric"]), {"CKA", "MutualKNNTop10"})
        # SVCCA was present in the results and deliberately not reported.
        self.assertNotIn("SVCCA", set(normalized["metric"]))
        self.assertEqual(snapshot["reported_measures"], ["CKA", "MutualKNNTop10"])

    def test_exclusions_resolve_against_the_loaded_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root)
            # "NeverComputed" is absent from these results, which an exclusion
            # list tolerates so one list can be shared across domains.
            settings = AnalysisSettings(exclude_measures=("SVCCA", "NeverComputed"))
            out = analyze_campaign(campaign_path, "trial", settings=settings)
            normalized = pd.read_csv(out / "normalized_values.csv")
        self.assertEqual(set(normalized["metric"]), {"CKA", "MutualKNNTop10"})

    def test_naming_an_absent_measure_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root)
            settings = AnalysisSettings(measures=("CKA", "NotComputedHere"))
            with self.assertRaises(ValueError) as caught:
                analyze_campaign(campaign_path, "trial", settings=settings)
        self.assertIn("NotComputedHere", str(caught.exception))

    def test_explicit_measures_argument_overrides_the_settings_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root)
            settings = AnalysisSettings(measures=("CKA", "MutualKNNTop10"))
            out = analyze_campaign(
                campaign_path, "trial", measures=["SVCCA", "CKA"], settings=settings
            )
            normalized = pd.read_csv(out / "normalized_values.csv")
        self.assertEqual(set(normalized["metric"]), {"CKA", "SVCCA"})

    def test_quality_measures_override_narrows_the_campaign_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root)
            settings = AnalysisSettings(quality_measures=("AUPRC",))
            out = analyze_campaign(campaign_path, "trial", settings=settings)
            normalized = pd.read_csv(out / "normalized_values.csv")
        self.assertEqual(set(normalized["quality_measure"]), {"AUPRC"})

    def test_quality_measure_outside_the_campaign_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root)
            settings = AnalysisSettings(quality_measures=("not_a_quality_measure",))
            with self.assertRaises(ValueError) as caught:
                analyze_campaign(campaign_path, "trial", settings=settings)
        self.assertIn("not_a_quality_measure", str(caught.exception))

    def test_default_settings_report_every_measure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = self._campaign(root)
            out = analyze_campaign(campaign_path, "trial")
            normalized = pd.read_csv(out / "normalized_values.csv")
            snapshot = yaml.safe_load((out / "config.yaml").read_text())
        self.assertEqual(set(normalized["metric"]), set(self.MEASURES))
        self.assertIsNone(snapshot["reported_measures"])


if __name__ == "__main__":
    unittest.main()
