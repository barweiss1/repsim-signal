from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import os
import pandas as pd
import subprocess
import sys
import yaml

from manifold_repsim.resi import runtime
from manifold_repsim.resi.campaign import (
    ManifestEntry,
    load_campaign,
    prepare_campaign,
    read_manifest,
)
from manifold_repsim.resi.measures import MANIFOLD_RESI_MEASURE_CLASSES
from manifold_repsim.resi.status import manifest_status


METRICS = ["MutualKNNTop10", "MutualKNNAUC"]


def _write_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


class TestCampaignPreparation(unittest.TestCase):
    def test_container_array_uses_configured_srun(self):
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "resi_array_container.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("#SBATCH --gpus=1", script)
        self.assertIn("#SBATCH --qos=normal", script)
        self.assertIn("$HOME/containers/resi-manifold-pytorch-24.06-v2.sqsh", script)
        self.assertIn("srun \\", script)
        self.assertIn('--container-image="$CONTAINER_IMAGE"', script)
        self.assertIn('--container-mounts="$CONTAINER_MOUNTS"', script)
        self.assertNotIn("enroot start", script)

    def test_domain_submission_scripts_select_separate_manifests(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "submit_resi_graph.sh": "graphs",
            "submit_resi_vision.sh": "vision",
            "submit_resi_language.sh": "language",
        }
        for filename, domain in expected.items():
            with self.subTest(filename=filename):
                script = (root / "scripts" / filename).read_text(encoding="utf-8")
                self.assertIn(f'submit_resi_campaign.sh" {domain} "$@"', script)

        shared = (root / "scripts" / "submit_resi_campaign.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("resi_prepare_container.sbatch", shared)
        self.assertIn("--wait", shared)
        self.assertIn("compute_manifest.jsonl", shared)
        self.assertIn("baseline_manifest.jsonl", shared)
        self.assertIn("configs/resi_graph_campaign.yaml", shared)
        self.assertIn("configs/resi_vision_campaign.yaml", shared)
        self.assertIn("configs/resi_language_campaign.yaml", shared)
        self.assertIn("%1", shared)
        self.assertIn(
            'RESI_MAX_CONCURRENT_GPUS="${RESI_MAX_CONCURRENT_GPUS:-8}"', shared
        )
        self.assertIn("%$RESI_MAX_CONCURRENT_GPUS", shared)
        self.assertIn("afterok:$baseline_job", shared)

        prepare = (root / "scripts" / "resi_prepare_container.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("srun \\", prepare)
        self.assertIn('--container-image="$CONTAINER_IMAGE"', prepare)
        self.assertIn("python scripts/resi.py prepare", prepare)

        self.assertIn('RESI_RUNTIME="${RESI_RUNTIME:-container}"', shared)
        self.assertIn("resi_array_native.sbatch", shared)
        self.assertIn("resi_prepare_native.sbatch", shared)

    def test_native_domain_submission_scripts_force_native_runtime(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "submit_resi_graph_native.sh": "graphs",
            "submit_resi_vision_native.sh": "vision",
            "submit_resi_language_native.sh": "language",
        }
        for filename, domain in expected.items():
            with self.subTest(filename=filename):
                script = (root / "scripts" / filename).read_text(encoding="utf-8")
                self.assertIn(f'submit_resi_campaign.sh" {domain} "$@"', script)
                self.assertIn("export RESI_RUNTIME=native", script)

    def test_native_scripts_use_preempt_partition_and_conda(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("resi_prepare_native.sbatch", "resi_array_native.sbatch"):
            with self.subTest(filename=filename):
                script = (root / "scripts" / filename).read_text(encoding="utf-8")
                self.assertIn("#SBATCH --partition=part-preempt", script)
                self.assertIn("#SBATCH --qos=qos-preempt", script)
                self.assertIn("#SBATCH --gres=gpu:1", script)
                self.assertIn("#SBATCH --requeue", script)
                self.assertIn("conda activate", script)
                self.assertIn('CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"', script)
                self.assertIn('CONDA_ENV="${CONDA_ENV:-resi-manifold}"', script)
                self.assertNotIn("container-image", script)

        setup = (root / "scripts" / "setup_resi_native_env.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("requirements-resi-native.txt", setup)
        self.assertIn("constraints-resi-native.txt", setup)
        self.assertIn("pip install -e", setup)
        self.assertIn("smoke_resi_container.py", setup)

    def test_checked_in_task_matrices(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            # 45 GCN/GraphSAGE/GAT tasks plus 4 cora PGNN ones, and 27
            # archived baselines plus the 4 recreated PGNN parquets.
            # PGNN is cora-only and skips augmentation_test.
            "resi_graph_campaign.yaml": ("graphs", 49, 31, 11),
            # 35 archived ImageNet100 baselines plus the 33 recreated CIFAR100
            # ones, which are per-architecture because each mfbase parquet
            # holds a single architecture.
            "resi_vision_campaign.yaml": ("vision", 68, 68, 11),
            # 10 archived BERT-L baselines plus the 8 recreated SmolLM2 ones,
            # one per benchmark/dataset since ReSi archives none for that model.
            "resi_language_campaign.yaml": ("language", 18, 18, 11),
        }
        for filename, (
            domain,
            task_count,
            baseline_count,
            metric_count,
        ) in expected.items():
            with self.subTest(filename=filename):
                campaign = load_campaign(root / "configs" / filename)
                self.assertEqual(campaign.domain, domain)
                self.assertEqual(
                    sum(len(item.architectures) for item in campaign.benchmarks),
                    task_count,
                )
                self.assertEqual(
                    sum(len(item.baselines) for item in campaign.benchmarks),
                    baseline_count,
                )
                self.assertEqual(len(campaign.measures), metric_count)
                for benchmark in campaign.benchmarks:
                    is_monotonicity = benchmark.id in {"monotonicity", "layer_test"}
                    self.assertEqual(benchmark.cache_to_mem, not is_monotonicity)

    def test_benchmark_cache_to_mem_defaults_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_path = Path(tmp) / "campaign.yaml"
            campaign = {
                "version": 1,
                "domain": "graphs",
                "quality_measures": ["AUPRC"],
                "measures": METRICS,
                "benchmarks": [
                    {
                        "id": "layer_test",
                        "dataset": "toy",
                        "base_config": "configs/base.yaml",
                        "architectures": ["GCN"],
                    }
                ],
            }
            _write_yaml(campaign_path, campaign)
            self.assertTrue(load_campaign(campaign_path).benchmarks[0].cache_to_mem)

            campaign["benchmarks"][0]["cache_to_mem"] = False
            _write_yaml(campaign_path, campaign)
            self.assertFalse(load_campaign(campaign_path).benchmarks[0].cache_to_mem)

            campaign["benchmarks"][0]["cache_to_mem"] = "false"
            _write_yaml(campaign_path, campaign)
            with self.assertRaisesRegex(ValueError, "cache_to_mem must be a boolean"):
                load_campaign(campaign_path)

    def test_known_missing_model_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_path = Path(tmp) / "campaign.yaml"
            campaign = {
                "version": 1,
                "domain": "vision",
                "quality_measures": ["AUPRC"],
                "measures": METRICS,
                "benchmarks": [
                    {
                        "id": "randomlabel",
                        "dataset": "ImageNet100",
                        "base_config": "configs/base.yaml",
                        "architectures": ["ViT_L32"],
                    }
                ],
                "known_missing_models": [
                    {
                        "benchmark": "randomlabel",
                        "dataset": "ImageNet100",
                        "architecture": "ViT_L32",
                        "setting": "Randomlabel",
                        "train_dataset": "RandomLabel_100_IN100_DataModule",
                        "seed": 4,
                        "reason": "Invalid upstream checkpoint.",
                    }
                ],
            }
            _write_yaml(campaign_path, campaign)
            missing = load_campaign(campaign_path).known_missing_models[0]
            self.assertEqual((missing.architecture, missing.seed), ("ViT_L32", 4))

            campaign["known_missing_models"][0]["seed"] = "4"
            _write_yaml(campaign_path, campaign)
            with self.assertRaisesRegex(
                ValueError, "seed must be a non-negative integer"
            ):
                load_campaign(campaign_path)

    def test_measure_sweep_overrides_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign_path = Path(tmp) / "campaign.yaml"
            base = {
                "version": 1,
                "domain": "graphs",
                "quality_measures": ["AUPRC"],
                "measures": METRICS,
                "benchmarks": [
                    {
                        "id": "augmentation_test",
                        "dataset": "toy",
                        "base_config": "configs/base.yaml",
                        "architectures": ["GCN"],
                    }
                ],
            }

            def with_sweeps(value):
                campaign = dict(base)
                campaign["measure_sweeps"] = value
                _write_yaml(campaign_path, campaign)
                return campaign_path

            good = {"MutualKNNAUC": {"min": 3, "max": 64, "num": 8, "scale": "log"}}
            campaign = load_campaign(with_sweeps(good))
            self.assertEqual(
                campaign.sweep_overrides(),
                {
                    "MutualKNNAUC": {
                        "min": 3.0,
                        "max": 64.0,
                        "num": 8,
                        "scale": "log",
                    }
                },
            )

            cases = {
                "not a configured campaign measure": {
                    "NotAMeasure": {"min": 1, "max": 2, "num": 3, "scale": "log"}
                },
                "only applies to sweeping AUC measures": {
                    "MutualKNNTop10": {
                        "min": 1,
                        "max": 2,
                        "num": 3,
                        "scale": "log",
                    }
                },
                "requires: num": {"MutualKNNAUC": {"min": 1, "max": 2, "scale": "log"}},
                "must be smaller than max": {
                    "MutualKNNAUC": {
                        "min": 5,
                        "max": 5,
                        "num": 3,
                        "scale": "log",
                    }
                },
                "log scale requires a positive min": {
                    "MutualKNNAUC": {
                        "min": 0,
                        "max": 5,
                        "num": 3,
                        "scale": "log",
                    }
                },
                "scale must be": {
                    "MutualKNNAUC": {
                        "min": 1,
                        "max": 5,
                        "num": 3,
                        "scale": "quadratic",
                    }
                },
            }
            for message, value in cases.items():
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        load_campaign(with_sweeps(value))

    def test_campaigns_without_overrides_omit_the_sweep_field(self):
        entry = ManifestEntry(
            kind="compute",
            index=0,
            domain="graphs",
            benchmark="b",
            dataset="d",
            architectures=("GCN",),
            measures=("MutualKNNAUC",),
            config_path="c.yaml",
            result_path="r.parquet",
            full_csv_path="r.csv",
        )
        payload = json.loads(entry.to_json())
        self.assertNotIn("measure_sweeps", payload)
        self.assertEqual(ManifestEntry.from_dict(payload).measure_sweeps, {})

        overridden = ManifestEntry(
            kind="compute",
            index=0,
            domain="graphs",
            benchmark="b",
            dataset="d",
            architectures=("GCN",),
            measures=("MutualKNNAUC",),
            config_path="c.yaml",
            result_path="r.parquet",
            full_csv_path="r.csv",
            measure_sweeps={
                "MutualKNNAUC": {"min": 3, "max": 64, "num": 8, "scale": "log"}
            },
        )
        round_tripped = ManifestEntry.from_dict(json.loads(overridden.to_json()))
        self.assertEqual(round_tripped.measure_sweeps, overridden.measure_sweeps)

    def test_prepare_is_deterministic_unique_and_metric_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resi = root / "resi"
            rep_sim = root / "experiments"
            (rep_sim / "results").mkdir(parents=True)
            base = {
                "excluded_measures": ["GeometryScore"],
                "experiments": [
                    {
                        "name": "stub",
                        "type": "GroupSeparationExperiment",
                        "representation_dataset": "toy",
                        "filter_key_vals": {"domain": "GRAPHS"},
                        "grouping_keys": ["identifier"],
                        "separation_keys": ["architecture"],
                    }
                ],
                "raw_results_filename": "old.parquet",
                "table_creation": {
                    "save_aggregated_df": True,
                    "save_full_df": True,
                    "full_df_filename": "old.csv",
                },
            }
            _write_yaml(resi / "configs" / "base.yaml", base)
            baseline = pd.DataFrame(
                {"metric": ["CKA", "Accuracy"], "metric_value": [0.5, 0.7]}
            )
            baseline.to_parquet(rep_sim / "results" / "baseline.parquet")
            campaign_path = root / "campaign.yaml"
            _write_yaml(
                campaign_path,
                {
                    "version": 1,
                    "domain": "graphs",
                    "run_root": str(root / "runs"),
                    "analysis_root": str(root / "figures"),
                    "quality_measures": ["AUPRC"],
                    "measures": METRICS,
                    "benchmarks": [
                        {
                            "id": "augmentation_test",
                            "dataset": "toy",
                            "base_config": "configs/base.yaml",
                            "architectures": ["GCN", "GAT"],
                            "cache_to_mem": False,
                            "baselines": [{"path": "baseline.parquet"}],
                        }
                    ],
                },
            )
            compute_path, baseline_path = prepare_campaign(
                campaign_path,
                "Trial",
                resi_dir=resi,
                rep_sim=rep_sim,
                registered_measures={"CKA"},
            )
            first = compute_path.read_bytes(), baseline_path.read_bytes()
            prepare_campaign(
                campaign_path,
                "Trial",
                resi_dir=resi,
                rep_sim=rep_sim,
                registered_measures={"CKA"},
            )
            self.assertEqual(
                first, (compute_path.read_bytes(), baseline_path.read_bytes())
            )
            compute = read_manifest(compute_path)
            baselines = read_manifest(baseline_path)
            self.assertEqual(len(compute), 2)
            self.assertEqual(
                len({entry.result_path for entry in compute + baselines}), 3
            )
            self.assertEqual(
                compute[0].legacy_signal_paths,
                (str(Path(compute[0].result_path).with_name("results_signals.jsonl")),),
            )
            self.assertEqual(baselines[0].legacy_signal_paths, ())
            old_entry = compute[0].to_json()
            old_payload = json.loads(old_entry)
            old_payload.pop("legacy_signal_paths")
            self.assertEqual(
                ManifestEntry.from_dict(old_payload).legacy_signal_paths, ()
            )
            config = yaml.safe_load(
                Path(compute[0].config_path).read_text(encoding="utf-8")
            )
            self.assertEqual(config["included_measures"], METRICS)
            self.assertNotIn("excluded_measures", config)
            self.assertEqual(
                config["experiments"][0]["filter_key_vals"]["architecture"], ["GCN"]
            )
            self.assertFalse(config["cache_to_mem"])
            self.assertFalse(config["cache_to_disk"])
            self.assertFalse(config["rerun_nans"])
            self.assertFalse(config["table_creation"]["save_aggregated_df"])
            baseline_config = yaml.safe_load(
                Path(baselines[0].config_path).read_text(encoding="utf-8")
            )
            self.assertTrue(baseline_config["only_eval"])
            self.assertFalse(baseline_config["cache_to_mem"])
            self.assertEqual(baseline_config["included_measures"], ["CKA"])

            effective = yaml.safe_load(
                (compute_path.parent / "config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(effective["run_name"], "trial")
            self.assertEqual(effective["domain"], "graphs")
            self.assertEqual(effective["measures"], METRICS)
            self.assertEqual(effective["resi_dir"], str(resi.resolve()))
            self.assertEqual(effective["rep_sim"], str(rep_sim.resolve()))
            self.assertEqual(effective["measure_sweeps"], {})
            self.assertEqual(len(effective["benchmarks"]), 1)
            self.assertEqual(
                effective["benchmarks"][0]["architectures"], ["GCN", "GAT"]
            )
            self.assertTrue((compute_path.parent / "campaign.json").is_file())


def _comparison_id(source_group: str, target_group: str, metric: str = "CKA") -> str:
    """Build a ReSi comparison id differing only in train_dataset."""

    def model(group: str) -> str:
        return "__".join(("Aug", "GCN", group, "0", "toy", "1"))

    return "___".join((model(source_group), model(target_group), metric))


def _group_campaign(root: Path, declared: list[str], present: list[str]) -> Path:
    """Write a campaign whose baseline covers only ``present`` of ``declared``."""
    resi = root / "resi"
    rep_sim = root / "experiments"
    (rep_sim / "results").mkdir(parents=True, exist_ok=True)
    _write_yaml(
        resi / "configs" / "base.yaml",
        {
            "experiments": [
                {
                    "name": "stub",
                    "type": "GroupSeparationExperiment",
                    "representation_dataset": "toy",
                    "filter_key_vals": {"domain": "GRAPHS", "train_dataset": declared},
                    "grouping_keys": ["train_dataset"],
                    "separation_keys": ["architecture"],
                }
            ],
        },
    )
    pd.DataFrame(
        {
            "id": [
                _comparison_id(source, target)
                for source in present
                for target in present
                if source != target
            ],
            "metric": ["CKA"] * (len(present) * (len(present) - 1)),
            "metric_value": [0.5] * (len(present) * (len(present) - 1)),
        }
    ).to_parquet(rep_sim / "results" / "baseline.parquet")
    campaign_path = root / "campaign.yaml"
    _write_yaml(
        campaign_path,
        {
            "version": 1,
            "domain": "graphs",
            "run_root": str(root / "runs"),
            "analysis_root": str(root / "figures"),
            "quality_measures": ["AUPRC"],
            "measures": METRICS,
            "benchmarks": [
                {
                    "id": "augmentation_test",
                    "dataset": "toy",
                    "base_config": "configs/base.yaml",
                    "architectures": ["GCN"],
                    "baselines": [{"path": "baseline.parquet"}],
                }
            ],
        },
    )
    return campaign_path


class TestBaselineGroupCoverage(unittest.TestCase):
    """A baseline must cover every group the config scoring it declares.

    ReSi evaluates quality over the declared group set, so a wholly absent group
    makes every measure NaN. This is knowable at prepare time and was how the
    recreated CIFAR100 augmentation baselines silently produced empty evals.
    """

    def _prepare(self, root: Path, campaign_path: Path):
        return prepare_campaign(
            campaign_path,
            "Trial",
            resi_dir=root / "resi",
            rep_sim=root / "experiments",
            registered_measures={"CKA"},
        )

    def test_missing_group_fails_and_names_both_sides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = _group_campaign(
                root,
                declared=["Gauss_S", "Gauss_M", "Gauss_L"],
                present=["Gauss_S", "Gauss_M"],
            )
            with self.assertRaises(ValueError) as caught:
                self._prepare(root, campaign_path)
            message = str(caught.exception)
            self.assertIn("train_dataset", message)
            self.assertIn("baseline.parquet", message)
            self.assertIn("base.yaml", message)
            self.assertIn("Gauss_L", message)

    def test_extra_group_in_baseline_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = _group_campaign(
                root,
                declared=["Gauss_S", "Gauss_M"],
                present=["Gauss_S", "Gauss_M", "Gauss_L"],
            )
            compute_path, baseline_path = self._prepare(root, campaign_path)
            self.assertTrue(compute_path.is_file())
            self.assertEqual(len(read_manifest(baseline_path)), 1)

    def test_grouping_keys_outside_the_id_grammar_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = _group_campaign(
                root,
                declared=["Gauss_S", "Gauss_M", "Gauss_L"],
                present=["Gauss_S", "Gauss_M"],
            )
            base_path = root / "resi" / "configs" / "base.yaml"
            base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            # `identifier` is not a model field, so no id can carry it and the
            # check has nothing to compare against.
            base["experiments"][0]["grouping_keys"] = ["identifier"]
            _write_yaml(base_path, base)
            compute_path, _ = self._prepare(root, campaign_path)
            self.assertTrue(compute_path.is_file())

    def test_experiments_without_grouping_keys_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign_path = _group_campaign(
                root,
                declared=["Gauss_S", "Gauss_M", "Gauss_L"],
                present=["Gauss_S", "Gauss_M"],
            )
            base_path = root / "resi" / "configs" / "base.yaml"
            base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            # Monotonicity and accoutput are not group-separation experiments.
            base["experiments"][0].pop("grouping_keys")
            _write_yaml(base_path, base)
            compute_path, _ = self._prepare(root, campaign_path)
            self.assertTrue(compute_path.is_file())


class TestPureImports(unittest.TestCase):
    """Campaign definitions and status must be usable without Torch or ReSi."""

    def test_schema_manifest_and_status_import_without_torch_or_resi(self):
        code = (
            "import builtins\n"
            "_real = builtins.__import__\n"
            "def _guard(name, *args, **kwargs):\n"
            "    if name.split('.')[0] in {'torch', 'repsim'}:\n"
            "        raise ImportError(name)\n"
            "    return _real(name, *args, **kwargs)\n"
            "builtins.__import__ = _guard\n"
            "from manifold_repsim.resi.campaign import load_campaign, read_manifest\n"
            "from manifold_repsim.resi.status import manifest_status\n"
            "from manifold_repsim.resi.cli import build_parser\n"
            "campaign = load_campaign('configs/resi_graph_campaign.yaml')\n"
            "print(campaign.domain, len(campaign.measures))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "graphs 11")


class _FakeStorer:
    def __init__(self):
        self._new_experiments = pd.DataFrame()

    def _get_comparison_id(self, src, tgt, metric_name):
        return f"{src}___{tgt}___{metric_name}"

    def add_results(
        self, src, tgt, metric, metric_value, runtime=None, overwrite=False
    ):
        ids = [self._get_comparison_id(src, tgt, metric.name)]
        if metric.is_symmetric:
            ids.append(self._get_comparison_id(tgt, src, metric.name))
        for comparison_id in ids:
            self._new_experiments = pd.concat(
                [
                    self._new_experiments,
                    pd.DataFrame(
                        {
                            "id": [comparison_id],
                            "metric": [metric.name],
                            "metric_value": [metric_value],
                        },
                        index=[comparison_id],
                    ),
                ]
            )


class TestRuntime(unittest.TestCase):
    def test_missing_nested_bert_paths_use_existing_flat_archive_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep_sim = Path(tmp) / "experiments"
            nlp_root = rep_sim / "models" / "nlp"
            flat_model = nlp_root / "memorizing" / "glue__mnli_model"
            flat_model.mkdir(parents=True)
            nested_model = nlp_root / "bert" / "memorizing" / flat_model.name
            valid_nested = nlp_root / "bert" / "standard" / "existing"
            valid_nested.mkdir(parents=True)
            missing_nested = nlp_root / "bert" / "standard" / "missing"

            flat = types.SimpleNamespace(domain="NLP", path=str(nested_model))
            valid = types.SimpleNamespace(domain="NLP", path=str(valid_nested))
            missing = types.SimpleNamespace(domain="NLP", path=str(missing_nested))
            graph = types.SimpleNamespace(domain="GRAPHS", path=str(nested_model))

            remapped = runtime.remap_missing_nlp_model_paths(
                [flat, valid, missing, graph],
                rep_sim=rep_sim,
            )

            self.assertEqual(flat.path, str(flat_model))
            self.assertEqual(valid.path, str(valid_nested))
            self.assertEqual(missing.path, str(missing_nested))
            self.assertEqual(graph.path, str(nested_model))
            self.assertEqual(remapped, {str(nested_model): str(flat_model)})

    def test_status_distinguishes_complete_incomplete_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for index, state in enumerate(("complete", "incomplete", "missing")):
                task = root / state
                result = task / "results.parquet"
                full = task / "results_full.csv"
                task.mkdir()
                if state != "missing":
                    pd.DataFrame(
                        {
                            "metric": ["M"],
                            "metric_value": [
                                0.5 if state == "complete" else float("nan")
                            ],
                        }
                    ).to_parquet(result)
                    full.write_text(
                        "similarity_measure,quality_measure,architecture,value\nM,AUPRC,GCN,0.5\n",
                        encoding="utf-8",
                    )
                entries.append(
                    {
                        "kind": "compute",
                        "index": index,
                        "domain": "graphs",
                        "benchmark": "test",
                        "dataset": "toy",
                        "architectures": ["GCN"],
                        "measures": ["M"],
                        "config_path": str(task / "config.yaml"),
                        "result_path": str(result),
                        "full_csv_path": str(full),
                        "source_result_path": None,
                    }
                )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
            )
            status = manifest_status(manifest)
            self.assertEqual(
                status["state"].tolist(), ["complete", "incomplete", "missing"]
            )
            self.assertEqual(status.loc[1, "nan_measures"], "M")

    def test_runtime_registration_does_not_write_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = root / "tracked.txt"
            marker.write_text("unchanged", encoding="utf-8")
            before = hashlib.sha256(marker.read_bytes()).hexdigest()
            registry = types.SimpleNamespace(ALL_MEASURES={"CKA": object()})
            with mock.patch.object(runtime, "_load_registry", return_value=registry):
                registered = runtime.register_manifold_measures(root)
            self.assertEqual(set(registered), set(MANIFOLD_RESI_MEASURE_CLASSES))
            self.assertEqual(hashlib.sha256(marker.read_bytes()).hexdigest(), before)

    def test_registration_applies_campaign_sweep_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = types.SimpleNamespace(ALL_MEASURES={})
            overrides = {"MutualKNNAUC": {"min": 3, "max": 9, "num": 4, "scale": "log"}}
            with mock.patch.object(runtime, "_load_registry", return_value=registry):
                registered = runtime.register_manifold_measures(Path(tmp), overrides)

            overridden = registered["MutualKNNAUC"]
            self.assertEqual(overridden.sweep_grid, overrides["MutualKNNAUC"])
            self.assertIsNone(registered["CKNNAAUC"].sweep_grid)

            source = np.random.RandomState(0).randn(12, 5).astype(np.float32)
            target = (source + 0.01).astype(np.float32)
            overridden(source, target, "nd")
            signal = overridden.last_similarity_signal
            self.assertEqual(signal["sweep_source"], "campaign")
            self.assertEqual(signal["sweep_len_requested"], 4)
            self.assertTrue(signal["logscale"])
            self.assertLessEqual(max(signal["param_values"]), 9)

            default = registered["CKNNAAUC"]
            with mock.patch.dict(
                os.environ, {"MANIFOLD_RESI_AUC_SWEEP_LEN": "5"}, clear=False
            ):
                default(source, target, "nd")
            self.assertEqual(default.last_similarity_signal["sweep_source"], "registry")
            self.assertEqual(default.last_similarity_signal["sweep_len_requested"], 5)

    def test_registration_rejects_unknown_sweep_override_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = types.SimpleNamespace(ALL_MEASURES={})
            with mock.patch.object(runtime, "_load_registry", return_value=registry):
                with self.assertRaisesRegex(ValueError, "no registered measure"):
                    runtime.register_manifold_measures(
                        Path(tmp),
                        {"NotAMeasure": {"min": 1, "max": 2, "num": 3, "scale": "log"}},
                    )

    def test_signal_hook_is_canonical_atomic_and_ignores_failures(self):
        fake_utils = types.SimpleNamespace(ExperimentStorer=_FakeStorer)
        real_import = runtime.importlib.import_module
        with mock.patch.object(
            runtime.importlib,
            "import_module",
            side_effect=lambda name: (
                fake_utils if name == "repsim.benchmark.utils" else real_import(name)
            ),
        ):
            runtime.install_signal_storage_hook()
        metric = types.SimpleNamespace(
            name="ManifoldAUC",
            is_symmetric=True,
            last_similarity_signal={"param_values": [1, 2], "scores": [0.1, 0.2]},
        )
        storer = _FakeStorer()
        storer.add_results("z", "a", metric, 0.15)
        self.assertEqual(storer._new_experiments["similarity_signal"].notna().sum(), 1)
        canonical = min(storer._new_experiments.index)
        self.assertTrue(
            pd.notna(storer._new_experiments.loc[canonical, "similarity_signal"])
        )
        failed = types.SimpleNamespace(
            name="FailedAUC", is_symmetric=True, last_similarity_signal={"stale": True}
        )
        storer.add_results("z", "a", failed, float("nan"))
        failed_rows = storer._new_experiments[
            storer._new_experiments["metric"] == "FailedAUC"
        ]
        self.assertTrue(failed_rows["similarity_signal"].isna().all())

    def test_baseline_copy_is_immutable_and_only_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.parquet"
            target = root / "out" / "results.parquet"
            full = root / "out" / "results_full.csv"
            config_path = root / "baseline.yaml"
            pd.DataFrame({"metric": ["CKA"], "metric_value": [0.5]}).to_parquet(source)
            _write_yaml(config_path, {"only_eval": True})
            entry = {
                "kind": "baseline",
                "index": 0,
                "domain": "graphs",
                "benchmark": "test",
                "dataset": "toy",
                "architectures": ["GCN"],
                "measures": ["CKA"],
                "config_path": str(config_path),
                "result_path": str(target),
                "full_csv_path": str(full),
                "source_result_path": str(source),
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

            def fake_run(path):
                config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
                self.assertTrue(config["only_eval"])
                pd.DataFrame(
                    {
                        "similarity_measure": ["CKA"],
                        "quality_measure": ["AUPRC"],
                        "architecture": ["GCN"],
                        "value": [0.8],
                    }
                ).to_csv(full, index=False)

            fake_run_module = types.SimpleNamespace(run=fake_run)
            real_import = runtime.importlib.import_module
            with (
                mock.patch.object(runtime, "_load_registry"),
                mock.patch.object(
                    runtime.importlib,
                    "import_module",
                    side_effect=lambda name: (
                        fake_run_module if name == "repsim.run" else real_import(name)
                    ),
                ),
            ):
                runtime.run_task(manifest, 0, resi_dir=root)
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(), source_hash
            )
            pd.testing.assert_frame_equal(
                pd.read_parquet(source), pd.read_parquet(target)
            )


if __name__ == "__main__":
    unittest.main()
