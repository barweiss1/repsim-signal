from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from manifold_repsim.config import load_yaml_config
from manifold_repsim.experiments.glocal_sweep_mds import (
    concatenate_batch_scores,
    discover,
    fit_signal_pca,
    load_computed_results,
    parse_glocal_transform,
    run,
    score_distances,
    validate_config,
)
from manifold_repsim.experiments.glocal_sweep_mds.discovery import load_aligned_pair
from manifold_repsim.experiments.glocal_sweep_mds.plotting import choose_mds_3d_view
from manifold_repsim.metrics import prepare_metric_curve, score_prepared_curve


class GlocalSweepMDSFixture:
    MODELS = ("model_a", "model_b")
    DATASET = "dataset_a"
    CONDITIONS = (
        "glocal-lambda-0.1-alpha-0.5-tau-0.1",
        "glocal-lambda-0.1-alpha-0.5-tau-0.5",
    )

    def __init__(self, root: Path):
        self.root = root
        self.features = root / "features"
        self.data_output = root / "results"
        self.figure_output = root / "figures"
        self._write()

    def _write(self) -> None:
        sample_ids = np.asarray([f"sample-{index}" for index in range(10)])
        labels = np.arange(10) % 3
        indices = np.arange(10)
        index_dir = self.features / "_index" / self.DATASET
        index_dir.mkdir(parents=True)
        np.savez(
            index_dir / "test.npz",
            sample_ids=sample_ids,
            labels=labels,
            sample_indices=indices,
        )
        rng = np.random.RandomState(4)
        for model_index, model in enumerate(self.MODELS):
            base = rng.normal(size=(10, 4)).astype(np.float32) + model_index * 0.2
            for transform_index, transform in enumerate(("none", *self.CONDITIONS)):
                path = self.features / model / self.DATASET / transform
                path.mkdir(parents=True)
                batches = []
                for batch_index in range(2):
                    selection = slice(batch_index * 5, (batch_index + 1) * 5)
                    features = base[selection].copy()
                    if transform != "none":
                        tau = 0.1 if transform_index == 1 else 0.5
                        features[:, 0] += tau * np.square(features[:, 1])
                    filename = f"test-batch-{batch_index:06d}.npz"
                    np.savez(
                        path / filename,
                        features=features,
                        labels=labels[selection],
                        sample_ids=sample_ids[selection],
                        sample_indices=indices[selection],
                    )
                    batches.append({"file": filename, "count": 5})
                metadata = {
                    "model": model,
                    "dataset": self.DATASET,
                    "transform": transform,
                    "transform_kind": "none" if transform == "none" else "glocal",
                    "batches": batches,
                }
                (path / "metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
            (self.features / model / self.DATASET / "global-lambda-0.1").mkdir()
            (self.features / model / self.DATASET / "naive-lambda-0.1").mkdir()

    def raw_config(self) -> dict:
        return {
            "version": 1,
            "run_name": "test_run",
            "input_root": str(self.features),
            "stages": ["compute", "plot"],
            "selection": {
                "models": "all",
                "datasets": "all",
                "batches": "all",
                "lambda": "all",
                "alpha": "all",
                "tau": "all",
            },
            "lambda_splits": [0.1],
            "metrics": [
                {
                    "id": "cka_fixed",
                    "name": "cka",
                    "mode": "fixed",
                    "kwargs": {},
                    "grid": None,
                },
                {
                    "id": "cka_rbf_signal",
                    "name": "cka_rbf",
                    "mode": "signal",
                    "kwargs": {},
                    "grid": {"values": [0.5, 1.0]},
                },
            ],
            "mds": {"random_state": 3, "n_init": 1, "max_iter": 50, "eps": 1e-6},
            "visualization": {"dpi": 40, "mds_3d_view": "auto"},
            "outputs": {
                "data_root": str(self.data_output),
                "figures_root": str(self.figure_output),
            },
            "execution": {"recompute_scores": False, "recompute_mds": False},
        }


class TestGlocalSweepMDS(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = GlocalSweepMDSFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_transform_parser_is_strict_and_finite(self):
        self.assertEqual(
            parse_glocal_transform("glocal-lambda-0.1-alpha-0.5-tau-1.0"),
            (0.1, 0.5, 1.0),
        )
        self.assertIsNone(parse_glocal_transform("global-lambda-0.1"))
        self.assertIsNone(parse_glocal_transform("naive-lambda-0.1"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            parse_glocal_transform("glocal-lambda-nan-alpha-0.5-tau-1.0")

    def test_config_requires_explicit_modes_and_valid_grids(self):
        config = validate_config(self.fixture.raw_config())
        self.assertEqual(config["metrics"][1]["parameter_name"], "rbf_sigma")
        self.assertEqual(config["metrics"][1]["parameter_values"], [0.5, 1.0])

        invalid = self.fixture.raw_config()
        invalid["metrics"][0]["grid"] = {"values": [1.0]}
        with self.assertRaisesRegex(ValueError, "must be null"):
            validate_config(invalid)

        invalid = self.fixture.raw_config()
        invalid["selection"]["tau"] = [0.1, 0.1]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_config(invalid)

        explicit = self.fixture.raw_config()
        explicit["visualization"]["mds_3d_view"] = {
            "elevation": 20,
            "azimuth": -45,
        }
        self.assertEqual(
            validate_config(explicit)["visualization"]["mds_3d_view"],
            {"elevation": 20, "azimuth": -45},
        )

        invalid = self.fixture.raw_config()
        invalid["visualization"]["mds_3d_view"] = {
            "elevation": 100,
            "azimuth": 0,
        }
        with self.assertRaisesRegex(ValueError, "between -90 and 90"):
            validate_config(invalid)

    def test_stages_must_be_unique_known_and_compute_before_plot(self):
        invalid = self.fixture.raw_config()
        invalid["stages"] = ["plot", "compute"]
        with self.assertRaisesRegex(ValueError, "compute must precede plot"):
            validate_config(invalid)

        invalid = self.fixture.raw_config()
        invalid["stages"] = ["compute", "score"]
        with self.assertRaisesRegex(ValueError, "compute and/or plot"):
            validate_config(invalid)

        invalid = self.fixture.raw_config()
        invalid["stages"] = []
        with self.assertRaisesRegex(ValueError, "compute and/or plot"):
            validate_config(invalid)

        invalid = self.fixture.raw_config()
        invalid["stages"] = ["compute", "compute"]
        with self.assertRaisesRegex(ValueError, "compute and/or plot"):
            validate_config(invalid)

        only_compute = self.fixture.raw_config()
        only_compute["stages"] = ["compute"]
        self.assertEqual(validate_config(only_compute)["stages"], ["compute"])

    def test_checked_in_config_has_requested_fixed_metrics_and_lambda_splits(self):
        config = validate_config(
            load_yaml_config(Path("configs/glocal_sweep_mds.yaml"))
        )
        fixed = {
            metric["id"]: (metric["name"], metric["kwargs"])
            for metric in config["metrics"]
            if metric["mode"] == "fixed"
        }
        self.assertEqual(
            fixed,
            {
                "cka_linear": ("cka", {}),
                "cka_rbf_sigma_05": ("cka_rbf", {"rbf_sigma": 0.5}),
                "cka_rbf_sigma_02": ("cka_rbf", {"rbf_sigma": 0.2}),
                "mutual_knn_k10": ("mutual_knn", {"topk": 10}),
            },
        )
        self.assertEqual(config["lambda_splits"], [0.001, 0.01, 0.1, 1.0])

    def test_discovery_resolves_all_and_ignores_other_transform_kinds(self):
        effective, groups = discover(validate_config(self.fixture.raw_config()))
        self.assertEqual(effective["selection"]["models"], list(self.fixture.MODELS))
        self.assertEqual(effective["selection"]["batches"], [0, 1])
        self.assertEqual(effective["selection"]["tau"], [0.1, 0.5])
        self.assertEqual(len(groups), 2)
        self.assertEqual(
            [condition.name for condition in groups[0].conditions],
            ["none", *self.fixture.CONDITIONS],
        )

    def test_exact_selection_and_complete_cartesian_grid_are_required(self):
        requested = self.fixture.raw_config()
        requested["selection"]["tau"] = [0.25]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            discover(validate_config(requested))

        requested = self.fixture.raw_config()
        requested["lambda_splits"] = [0.001]
        with self.assertRaisesRegex(ValueError, "lambda_splits"):
            discover(validate_config(requested))

        missing = (
            self.fixture.features
            / self.fixture.MODELS[1]
            / self.fixture.DATASET
            / self.fixture.CONDITIONS[1]
        )
        for path in missing.iterdir():
            path.unlink()
        missing.rmdir()
        with self.assertRaisesRegex(ValueError, "Incomplete selected glocal grid"):
            discover(validate_config(self.fixture.raw_config()))

    def test_batch_validation_rejects_misaligned_and_nonfinite_features(self):
        config = validate_config(self.fixture.raw_config())
        _, groups = discover(config)
        group = groups[0]
        source, target = load_aligned_pair(
            self.fixture.features, group, group.conditions[1], 0
        )
        self.assertEqual(source.shape, target.shape)

        target_path = group.root / group.conditions[1].name / group.batch_files[0]
        with np.load(target_path, allow_pickle=False) as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        arrays["sample_ids"][0] = "wrong"
        np.savez(target_path, **arrays)
        with self.assertRaisesRegex(ValueError, "sample_ids differs"):
            load_aligned_pair(self.fixture.features, group, group.conditions[1], 0)

        arrays["sample_ids"][0] = "sample-0"
        arrays["features"][0, 0] = np.nan
        np.savez(target_path, **arrays)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_aligned_pair(self.fixture.features, group, group.conditions[1], 0)

    def test_concatenation_and_distances_are_direct_and_batch_major(self):
        scores = np.arange(24, dtype=float).reshape(3, 2, 4)
        vectors = concatenate_batch_scores(scores)
        np.testing.assert_array_equal(vectors[1], np.arange(8, 16))
        np.testing.assert_allclose(
            score_distances(vectors), squareform(pdist(vectors, metric="euclidean"))
        )

    def test_signal_pca_centers_projects_and_pads_low_dimensional_inputs(self):
        vectors = np.asarray([[1.0], [2.0], [4.0]])
        embedding, components, explained, mean = fit_signal_pca(vectors)
        self.assertEqual(embedding.shape, (3, 2))
        self.assertEqual(components.shape, (2, 1))
        np.testing.assert_allclose(embedding.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(embedding[:, 1], 0.0)
        np.testing.assert_allclose(explained, [1.0, 0.0])
        np.testing.assert_allclose(mean, [7.0 / 3.0])

    def test_automatic_3d_view_looks_along_least_varying_direction(self):
        embedding = np.asarray(
            [
                [-2.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        elevation, _ = choose_mds_3d_view(embedding, "auto")
        self.assertAlmostEqual(abs(elevation), 90.0)
        self.assertEqual(
            choose_mds_3d_view(embedding, {"elevation": 25.0, "azimuth": -35.0}),
            (25.0, -35.0),
        )

    def test_end_to_end_keeps_models_separate_and_writes_all_outputs(self):
        config = validate_config(self.fixture.raw_config())
        results = run(config)
        self.assertEqual(
            set(results),
            {(model, self.fixture.DATASET) for model in self.fixture.MODELS},
        )

        data_run = self.fixture.data_output / "test_run"
        figure_run = self.fixture.figure_output / "test_run"
        self.assertEqual(
            load_yaml_config(data_run / "config.yaml"),
            load_yaml_config(figure_run / "config.yaml"),
        )
        saved_config = load_yaml_config(data_run / "config.yaml")
        self.assertEqual(saved_config["selection"]["tau"], [0.1, 0.5])

        for model in self.fixture.MODELS:
            group_dir = data_run / model / self.fixture.DATASET
            self.assertTrue((group_dir / "conditions.csv").is_file())
            fixed_path = group_dir / "metrics" / "cka_fixed.npz"
            signal_path = group_dir / "metrics" / "cka_rbf_signal.npz"
            split_path = (
                group_dir
                / "lambda_splits"
                / "lambda_0.1"
                / "metrics"
                / "cka_rbf_signal.npz"
            )
            with np.load(fixed_path, allow_pickle=False) as fixed:
                self.assertEqual(fixed["scores"].shape, (3, 2, 1))
                self.assertEqual(fixed["concatenated_vectors"].shape, (3, 2))
                self.assertTrue(fixed["is_none"][0])
            with np.load(signal_path, allow_pickle=False) as signal:
                self.assertEqual(signal["scores"].shape, (3, 2, 2))
                self.assertEqual(signal["concatenated_vectors"].shape, (3, 4))
                np.testing.assert_allclose(
                    signal["distances"],
                    squareform(
                        pdist(signal["concatenated_vectors"], metric="euclidean")
                    ),
                )
                self.assertEqual(signal["embedding_2d"].shape, (3, 2))
                self.assertEqual(signal["embedding_3d"].shape, (3, 3))
                self.assertEqual(signal["pca_embedding_2d"].shape, (3, 2))
                self.assertEqual(signal["pca_components_2d"].shape, (2, 4))
                self.assertEqual(signal["pca_explained_variance_ratio_2d"].shape, (2,))
                np.testing.assert_allclose(
                    squareform(pdist(signal["pca_embedding_2d"])),
                    signal["distances"],
                    atol=1e-10,
                )
            with np.load(split_path, allow_pickle=False) as split:
                self.assertEqual(split["split_lambda"].item(), 0.1)
                self.assertEqual(split["condition_names"].shape, (3,))
                self.assertTrue(split["is_none"][0])
                np.testing.assert_allclose(split["lambda_values"][1:], 0.1)

            figures = figure_run / model / self.fixture.DATASET
            self.assertTrue((figures / "mds_2d_overview.png").is_file())
            self.assertTrue((figures / "mds_3d_overview.png").is_file())
            self.assertTrue((figures / "pca_2d_overview.png").is_file())
            self.assertTrue((figures / "metrics" / "cka_fixed_2d.png").is_file())
            self.assertTrue((figures / "metrics" / "cka_rbf_signal_3d.png").is_file())
            self.assertTrue(
                (figures / "metrics" / "cka_rbf_signal_pca_2d.png").is_file()
            )
            self.assertTrue(
                (
                    figures / "signals" / "cka_rbf_signal" / "lambda_0.1_alpha_0.5.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    figures
                    / "lambda_splits"
                    / "lambda_0.1"
                    / "metrics"
                    / "cka_rbf_signal_2d.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    figures
                    / "signals_by_alpha"
                    / "cka_rbf_signal"
                    / "lambda_0.1_tau_0.1.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    figures / "lambda_splits" / "lambda_0.1" / "mds_2d_overview.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    figures / "lambda_splits" / "lambda_0.1" / "mds_3d_overview.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    figures / "lambda_splits" / "lambda_0.1" / "pca_2d_overview.png"
                ).is_file()
            )

    def test_plot_only_stage_reuses_computed_results_without_touching_features(self):
        config = validate_config(self.fixture.raw_config())
        config["stages"] = ["compute"]
        run(config)

        figure_run = self.fixture.figure_output / "test_run"
        for model in self.fixture.MODELS:
            figures = figure_run / model / self.fixture.DATASET
            self.assertFalse((figures / "mds_2d_overview.png").exists())

        # A `plot`-only rerun must reproduce every figure from the saved
        # manifest and per-metric checkpoints alone, with no access to the
        # feature root at all.
        shutil.rmtree(self.fixture.features)
        plot_config = validate_config(self.fixture.raw_config())
        plot_config["stages"] = ["plot"]
        results = run(plot_config)
        self.assertEqual(
            set(results),
            {(model, self.fixture.DATASET) for model in self.fixture.MODELS},
        )
        for model in self.fixture.MODELS:
            figures = figure_run / model / self.fixture.DATASET
            self.assertTrue((figures / "mds_2d_overview.png").is_file())
            self.assertTrue((figures / "metrics" / "cka_fixed_2d.png").is_file())

        groups, manifest_results = load_computed_results(plot_config)
        self.assertEqual({group.model for group in groups}, set(self.fixture.MODELS))
        self.assertEqual(set(manifest_results), set(results))

    def test_plot_only_stage_requires_a_prior_compute_run(self):
        config = validate_config(self.fixture.raw_config())
        config["stages"] = ["plot"]
        with self.assertRaisesRegex(FileNotFoundError, "compute"):
            run(config)

    def test_signal_scores_match_the_shared_prepared_curve(self):
        config = validate_config(self.fixture.raw_config())
        run(config)
        result_path = (
            self.fixture.data_output
            / "test_run"
            / self.fixture.MODELS[0]
            / self.fixture.DATASET
            / "metrics"
            / "cka_rbf_signal.npz"
        )
        with np.load(result_path, allow_pickle=False) as result:
            actual = result["scores"][1, 0]
        effective, groups = discover(config)
        source, target = load_aligned_pair(
            self.fixture.features, groups[0], groups[0].conditions[1], 0
        )
        prepared = prepare_metric_curve(
            "cka_rbf",
            torch.from_numpy(source),
            torch.from_numpy(target),
            "rbf_sigma",
            [0.5, 1.0],
        )
        np.testing.assert_allclose(actual, score_prepared_curve(prepared))

    def test_condition_checkpoints_resume_without_rescoring(self):
        config = validate_config(self.fixture.raw_config())
        run(config)
        config["execution"]["recompute_mds"] = True
        with mock.patch(
            "manifold_repsim.experiments.glocal_sweep_mds.scoring._score_pair",
            side_effect=AssertionError("score should be reused"),
        ):
            run(config)

    def test_effective_config_is_saved_before_metric_analysis(self):
        config = validate_config(self.fixture.raw_config())
        with mock.patch(
            "manifold_repsim.experiments.glocal_sweep_mds.workflow.analyze",
            side_effect=RuntimeError("stop after snapshot"),
        ):
            with self.assertRaisesRegex(RuntimeError, "snapshot"):
                run(config)
        self.assertTrue(
            (self.fixture.data_output / "test_run" / "config.yaml").is_file()
        )
        self.assertTrue(
            (self.fixture.figure_output / "test_run" / "config.yaml").is_file()
        )

    def test_configuration_import_does_not_load_matplotlib(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(Path("src").resolve()), str(Path.cwd())]
        )
        command = [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import manifold_repsim.experiments.glocal_sweep_mds.config; "
                "assert 'matplotlib' not in sys.modules"
            ),
        ]
        subprocess.run(
            command, check=True, env=environment, capture_output=True, text=True
        )


if __name__ == "__main__":
    unittest.main()
