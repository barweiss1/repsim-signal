import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import numpy as np
import yaml

from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.config import load_yaml_config
from manifold_repsim.experiments.permuted_gaussian_mds.dataset import (
    build_base_dataset,
)
from manifold_repsim.experiments.permuted_gaussian_mds.dataset import transform_config
from scripts.synthetic_sweep import (
    analysis_fingerprint,
    compute_fixed_similarity_matrix,
    compute_signal_similarity_matrices,
    compatible_dataset_fingerprints,
    dataset_fingerprint,
    direct_similarity_distance,
    fit_metric_mds,
    generate_dataset,
    grid_parameter_values,
    load_dataset_artifacts,
    parameter_value_label,
    run,
    scalar_base_relative_distance,
    signal_base_relative_distance,
    validate_config,
    visual_encodings,
)


def small_raw_config(root):
    return {
        "version": 1,
        "run_name": "tiny",
        "stages": ["generate", "analyze"],
        "dataset": {
            "n_points": 18,
            "dim": 2,
            "n_clusters": 3,
            "seed": 7,
            "balanced": True,
            "center_sampling": "hypersphere",
        },
        "transform": {"mode": "modify_base"},
        "base_transform": {
            "n_permute": 0,
            "cluster_mixing_probability": 0.0,
            "noise_scale": 0.1,
        },
        "grid": {
            "cluster_mixing_probability": {"min": 0.0, "max": 0.5, "num": 2},
            "noise_scale": {"min": 0.0, "max": 1.0, "num": 2},
            "n_permute": {"values": [0, 2, 3]},
        },
        "metrics": {
            "fixed": [
                {"id": "cka_lin", "name": "cka", "kwargs": {}},
                {
                    "id": "cka_rbf_sigma_02",
                    "name": "cka_rbf",
                    "kwargs": {"rbf_sigma": 0.2},
                },
                {
                    "id": "mutual_knn_dist_k3",
                    "name": "mutual_knn_dist",
                    "kwargs": {"topk": 3},
                },
            ],
            "signals": {
                "names": [
                    "cka_rbf",
                    "rbf_uka",
                    "rbf_rwka_symmetric",
                    "rbf_degree_crwka",
                ],
                "parameter": "rbf_sigma",
                "min": 0.1,
                "max": 0.5,
                "num": 2,
                "scale": "log",
                "integration": "average",
            },
        },
        "mds": {
            "random_state": 3,
            "n_init": 1,
            "max_iter": 12,
            "eps": 1e-5,
        },
        "visualization": {
            "encodings": {
                "color": "n_permute",
                "size": "cluster_mixing_probability",
                "opacity": "noise_scale",
            },
            "include_base": True,
            "dpi": 40,
        },
        "outputs": {
            "data_root": str(Path(root) / "data"),
            "figures_root": str(Path(root) / "figures"),
        },
        "execution": {
            "overwrite_dataset": False,
            "recompute_metrics": False,
            "checkpoint_signals": True,
        },
    }


class SyntheticPermutedGaussianMDSConfigTests(unittest.TestCase):
    def test_script_is_a_thin_compatibility_cli(self):
        script_path = Path("scripts/synthetic_sweep.py").resolve()
        self.assertLess(len(script_path.read_text(encoding="utf-8").splitlines()), 60)
        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=Path(tempfile.gettempdir()),
            check=False,
            capture_output=True,
            text=True,
            env={**dict(os.environ), "MPLCONFIGDIR": tempfile.gettempdir()},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Versioned YAML experiment configuration", completed.stdout)

    def test_config_import_does_not_load_plotting_backend(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "import manifold_repsim.experiments.permuted_gaussian_mds.config; "
                "assert 'matplotlib' not in sys.modules",
            ],
            cwd=Path(__file__).parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_checked_in_config_and_grid(self):
        raw = yaml.safe_load(
            Path("configs/synthetic_permuted_gaussian_mds.yaml").read_text(
                encoding="utf-8"
            )
        )
        config = validate_config(raw)
        grid = grid_parameter_values(config)

        expected_count = (
            raw["grid"]["cluster_mixing_probability"]["num"]
            * raw["grid"]["noise_scale"]["num"]
            * len(raw["grid"]["n_permute"]["values"])
        )
        self.assertEqual(grid.shape, (expected_count, 3))
        self.assertNotIn(1, grid[:, 2])
        for index, name in enumerate(("cluster_mixing_probability", "noise_scale")):
            self.assertEqual(grid[:, index].min(), raw["grid"][name]["min"])
            self.assertEqual(grid[:, index].max(), raw["grid"][name]["max"])
        self.assertEqual(
            grid[:, 2].max(),
            max(raw["grid"]["n_permute"]["values"]),
        )
        self.assertEqual(config["transform"]["mode"], "modify_base")

    def test_checked_in_mix_noise_config_holds_permutation_fixed(self):
        raw = yaml.safe_load(
            Path("configs/synthetic_permuted_gaussian_local.yaml").read_text(
                encoding="utf-8"
            )
        )
        config = validate_config(raw)
        grid = grid_parameter_values(config)
        self.assertEqual(config["run_name"], "synthetic_permuted_gaussian_local")
        self.assertEqual(config["grid"]["n_permute"], {"values": [0]})
        self.assertEqual(
            config["visualization"]["encodings"],
            {
                "color": None,
                "size": "cluster_mixing_probability",
                "opacity": "noise_scale",
            },
        )
        self.assertEqual(grid.shape, (100, 3))
        np.testing.assert_array_equal(grid[:, 2], 0)

    def test_omitted_continuous_grid_parameter_holds_it_fixed_at_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = small_raw_config(tmp)
            raw["grid"].pop("cluster_mixing_probability")
            raw["visualization"]["encodings"].pop("size")
            config = validate_config(raw)
            self.assertEqual(
                config["grid"]["cluster_mixing_probability"],
                {"min": 0.0, "max": 0.0, "num": 1},
            )
            self.assertIsNone(config["visualization"]["encodings"]["size"])
            grid = grid_parameter_values(config)
            np.testing.assert_array_equal(grid[:, 0], 0.0)

    def test_strict_keys_and_stage_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = small_raw_config(tmp)
            raw["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "Unknown configuration"):
                validate_config(raw)

            raw = small_raw_config(tmp)
            raw["stages"] = ["analyze", "generate"]
            with self.assertRaisesRegex(ValueError, "generate must precede"):
                validate_config(raw)

            raw = small_raw_config(tmp)
            raw["grid"]["n_permute"]["values"] = [0, 1, 2]
            with self.assertRaisesRegex(ValueError, "cannot contain 1"):
                validate_config(raw)

            raw = small_raw_config(tmp)
            raw["transform"]["mode"] = "unknown"
            with self.assertRaisesRegex(ValueError, "transform.mode"):
                validate_config(raw)

            raw = small_raw_config(tmp)
            raw["grid"]["cluster_mixing_probability"]["max"] = 0.75
            with self.assertRaisesRegex(ValueError, "must not exceed 0.5"):
                validate_config(raw)

    def test_visual_encodings(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = validate_config(small_raw_config(tmp))
            params = np.array([[0.0, 0.0, 0], [0.25, 0.1, 2], [0.5, 1.0, 3]])
            encodings = visual_encodings(params, config)
            np.testing.assert_allclose(encodings["color"], [0, 2, 3])
            np.testing.assert_allclose(encodings["size"], [24.0, 82.0, 140.0])
            np.testing.assert_allclose(encodings["opacity"], [0.25, 0.325, 1.0])
            self.assertEqual(
                parameter_value_label("cluster_mixing_probability", 0.5),
                r"$p_{mix}$ = 0.5",
            )
            self.assertEqual(
                parameter_value_label("noise_scale", 1.0),
                r"$\sigma_{noise}$ = 1",
            )
            self.assertEqual(
                parameter_value_label("n_permute", 9),
                r"$N_{perm}$ = 9",
            )

    def test_legend_style_defaults_to_markers_and_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = small_raw_config(tmp)
            config = validate_config(raw)
            self.assertEqual(config["visualization"]["legend_style"], "markers")

            explicit = copy.deepcopy(raw)
            explicit["visualization"]["legend_style"] = "colorbar"
            self.assertEqual(
                validate_config(explicit)["visualization"]["legend_style"],
                "colorbar",
            )

            invalid = copy.deepcopy(raw)
            invalid["visualization"]["legend_style"] = "bogus"
            with self.assertRaisesRegex(ValueError, "legend_style"):
                validate_config(invalid)

    def test_signal_sweep_fixed_parameters_default_to_base_and_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = small_raw_config(tmp)
            config = validate_config(raw)
            self.assertEqual(
                config["metrics"]["signal_sweeps"]["fixed_parameters"],
                config["base_transform"],
            )

            partial = copy.deepcopy(raw)
            partial["metrics"]["signal_sweeps"] = {
                "fixed_parameters": {"cluster_mixing_probability": 0.25}
            }
            configured = validate_config(partial)
            self.assertEqual(
                configured["metrics"]["signal_sweeps"]["fixed_parameters"],
                {
                    "cluster_mixing_probability": 0.25,
                    "noise_scale": raw["base_transform"]["noise_scale"],
                    "n_permute": raw["base_transform"]["n_permute"],
                },
            )

            invalid = copy.deepcopy(raw)
            invalid["metrics"]["signal_sweeps"] = {
                "fixed_parameters": {
                    "cluster_mixing_probability": 1.5,
                    "noise_scale": 0.1,
                    "n_permute": 0,
                }
            }
            with self.assertRaisesRegex(ValueError, "signal_sweeps.fixed_parameters"):
                validate_config(invalid)

    def test_transform_seed_and_base_seed_default_and_randomize(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = small_raw_config(tmp)
            config = validate_config(raw)
            self.assertEqual(config["transform"]["seed"], raw["dataset"]["seed"])
            self.assertEqual(
                config["transform"]["base_seed"], raw["dataset"]["seed"] + 1
            )
            self.assertNotEqual(
                config["transform"]["seed"], config["transform"]["base_seed"]
            )

            explicit = copy.deepcopy(raw)
            explicit["transform"] = {
                "mode": "modify_base",
                "seed": 5,
                "base_seed": 99,
            }
            configured = validate_config(explicit)
            self.assertEqual(configured["transform"]["seed"], 5)
            self.assertEqual(configured["transform"]["base_seed"], 99)

            randomized = copy.deepcopy(raw)
            randomized["transform"] = {"mode": "modify_base", "seed": None}
            first = validate_config(randomized)["transform"]["seed"]
            second = validate_config(randomized)["transform"]["seed"]
            self.assertNotEqual(first, second)

    def test_base_transform_uses_independent_seed_from_matching_grid_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = small_raw_config(tmp)
            raw["transform"] = {"mode": "modify_base", "seed": 11, "base_seed": 22}
            config = validate_config(raw)
            dataset, base = build_base_dataset(config)
            matching = dataset.transform_current(
                transform_config(config, **config["base_transform"])
            )
            self.assertFalse(np.allclose(base, matching))

    def test_fingerprints_ignore_stages_but_track_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = validate_config(small_raw_config(tmp))
            generate_only = copy.deepcopy(config)
            generate_only["stages"] = ["generate"]
            self.assertEqual(
                dataset_fingerprint(config),
                dataset_fingerprint(generate_only),
            )
            metric = config["metrics"]["fixed"][0]
            first = analysis_fingerprint(config, metric)
            changed = copy.deepcopy(config)
            changed["mds"]["max_iter"] += 1
            self.assertNotEqual(first, analysis_fingerprint(changed, metric))
            resampled = copy.deepcopy(config)
            resampled["transform"]["mode"] = "resample"
            self.assertNotEqual(
                dataset_fingerprint(config),
                dataset_fingerprint(resampled),
            )
            self.assertEqual(len(compatible_dataset_fingerprints(config)), 1)
            self.assertEqual(len(compatible_dataset_fingerprints(resampled)), 2)
            self.assertEqual(
                dataset_fingerprint(config),
                "6a8a0c7b32d725e0eef9bbf1a6ced3b4d8c21688c35d3e145d42afeb2f0e9a9a",
            )
            self.assertEqual(
                analysis_fingerprint(config, metric),
                "f92112d5e0527e2a13132a09384f392a480f44f775f5d428b017176848565cb1",
            )


class SyntheticPermutedGaussianMDSMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(4)
        self.datasets = rng.randn(3, 12, 3)

    def test_distance_constructions(self):
        scalar = np.array([1.0, 0.7, 0.2])
        expected_scalar = np.abs(scalar[:, None] - scalar[None, :])
        np.testing.assert_allclose(
            scalar_base_relative_distance(scalar),
            expected_scalar,
        )

        signals = np.array([[1.0, 1.0], [0.5, 0.75], [0.0, 0.0]])
        observed = signal_base_relative_distance(signals)
        self.assertAlmostEqual(
            observed[0, 1],
            np.sqrt(((1.0 - 0.5) ** 2 + (1.0 - 0.75) ** 2) / 2),
        )

        similarity = np.array([[1.0, 0.8], [0.8, 1.0]])
        np.testing.assert_allclose(
            direct_similarity_distance(similarity),
            np.array([[0.0, 0.2], [0.2, 0.0]]),
        )

    def test_optimized_fixed_metrics_match_direct_calls(self):
        specs = [
            {"id": "cka", "name": "cka", "kwargs": {}},
            {
                "id": "rbf",
                "name": "cka_rbf",
                "kwargs": {"rbf_sigma": 0.2},
            },
            {
                "id": "knn",
                "name": "mutual_knn_dist",
                "kwargs": {"topk": 3},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for spec in specs:
                observed = compute_fixed_similarity_matrix(
                    self.datasets,
                    spec,
                    Path(tmp) / spec["id"],
                )
                expected = AlignmentMetrics.measure(
                    metric=spec["name"],
                    feats_A=self.datasets[0],
                    feats_B=self.datasets[1],
                    **spec["kwargs"],
                )
                self.assertAlmostEqual(observed[0, 1], expected, places=7)

    def test_optimized_signal_metrics_match_direct_calls(self):
        names = [
            "cka_rbf",
            "rbf_uka",
            "rbf_rwka_symmetric",
            "rbf_degree_crwka",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            observed = compute_signal_similarity_matrices(
                self.datasets,
                names,
                0.4,
                Path(tmp),
            )
        for name in names:
            expected = AlignmentMetrics.measure(
                metric=name,
                feats_A=self.datasets[0],
                feats_B=self.datasets[1],
                rbf_sigma=0.4,
            )
            self.assertAlmostEqual(observed[name][0, 1], expected, places=8)

    def test_metric_mds_is_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = validate_config(small_raw_config(tmp))
            distance = np.array(
                [
                    [0.0, 0.2, 0.8, 1.0],
                    [0.2, 0.0, 0.6, 0.9],
                    [0.8, 0.6, 0.0, 0.3],
                    [1.0, 0.9, 0.3, 0.0],
                ]
            )
            first = fit_metric_mds(distance, config)
            second = fit_metric_mds(distance, config)
            np.testing.assert_allclose(first[0], second[0])
            self.assertAlmostEqual(first[2], second[2])
            first_3d = fit_metric_mds(distance, config, n_components=3)
            second_3d = fit_metric_mds(distance, config, n_components=3)
            self.assertEqual(first_3d[0].shape, (4, 3))
            np.testing.assert_allclose(first_3d[0], second_3d[0])
            self.assertAlmostEqual(first_3d[2], second_3d[2])


class SyntheticPermutedGaussianMDSEndToEndTests(unittest.TestCase):
    def test_mix_noise_only_grid_runs_without_a_color_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = small_raw_config(tmp)
            raw["run_name"] = "mix_noise_only"
            raw["grid"].pop("n_permute")
            raw["visualization"]["encodings"].pop("color")
            config = validate_config(raw)
            run(config)

            data_run = Path(config["outputs"]["data_root"]) / config["run_name"]
            figure_run = Path(config["outputs"]["figures_root"]) / config["run_name"]
            with np.load(data_run / "dataset.npz") as saved:
                self.assertEqual(saved["transformed_data"].shape[0], 4)
                np.testing.assert_array_equal(saved["grid_parameter_values"][:, 2], 0)
            saved_config = load_yaml_config(data_run / "config.yaml")
            self.assertEqual(saved_config["grid"]["n_permute"], {"values": [0]})
            self.assertIsNone(saved_config["visualization"]["encodings"]["color"])
            self.assertTrue((figure_run / "mds_base_relative_overview.png").is_file())
            self.assertTrue((figure_run / "mds_pairwise_overview.png").is_file())

    def test_colorbar_legend_style_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = small_raw_config(tmp)
            raw["run_name"] = "colorbar_legend_style"
            raw["visualization"]["legend_style"] = "colorbar"
            config = validate_config(raw)
            run(config)

            figure_run = Path(config["outputs"]["figures_root"]) / config["run_name"]
            self.assertTrue((figure_run / "mds_base_relative_overview.png").is_file())
            self.assertTrue((figure_run / "mds_pairwise_overview.png").is_file())

    def test_analyze_saves_single_parameter_signal_sweeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = validate_config(small_raw_config(tmp))
            run(config)

            figure_run = Path(config["outputs"]["figures_root"]) / config["run_name"]
            sweep_dir = figure_run / "signal_sweeps"
            for parameter in (
                "n_permute",
                "cluster_mixing_probability",
                "noise_scale",
            ):
                self.assertTrue((sweep_dir / f"{parameter}.png").is_file())

    def test_signal_sweeps_use_configured_fixed_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = small_raw_config(tmp)
            raw["run_name"] = "signal_sweep_fixed"
            raw["metrics"]["signal_sweeps"] = {
                "fixed_parameters": {
                    "cluster_mixing_probability": 0.25,
                    "noise_scale": 0.2,
                    "n_permute": 0,
                }
            }
            config = validate_config(raw)
            run(config)

            figure_run = Path(config["outputs"]["figures_root"]) / config["run_name"]
            sweep_dir = figure_run / "signal_sweeps"
            for parameter in (
                "n_permute",
                "cluster_mixing_probability",
                "noise_scale",
            ):
                self.assertTrue((sweep_dir / f"{parameter}.png").is_file())

    def test_legacy_visualization_fingerprint_migrates(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = validate_config(small_raw_config(tmp))
            config["transform"]["mode"] = "resample"
            dataset_path = generate_dataset(config)
            manifest_path = dataset_path.with_name("dataset_manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            legacy = (
                compatible_dataset_fingerprints(config) - {dataset_fingerprint(config)}
            ).pop()
            manifest["dataset_fingerprint"] = legacy
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            load_dataset_artifacts(config)

            migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                migrated["dataset_fingerprint"],
                dataset_fingerprint(config),
            )
            self.assertEqual(migrated["legacy_dataset_fingerprint"], legacy)

    def test_generate_reuse_mismatch_and_analyze(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = validate_config(small_raw_config(tmp))
            run(config)
            data_run = (
                Path(config["outputs"]["data_root"]) / config["run_name"]
            ).resolve()
            figures_run = (
                Path(config["outputs"]["figures_root"]) / config["run_name"]
            ).resolve()
            dataset_path = data_run / "dataset.npz"
            self.assertTrue(dataset_path.exists())
            data_config = load_yaml_config(data_run / "config.yaml")
            figure_config = load_yaml_config(figures_run / "config.yaml")
            self.assertEqual(data_config, figure_config)
            self.assertEqual(data_config["run_name"], "tiny")
            self.assertEqual(
                data_config["resolved_outputs"]["data_run"],
                str(data_run),
            )
            self.assertEqual(generate_dataset(config), dataset_path)
            with np.load(dataset_path) as saved:
                self.assertIn("all_color_values", saved)
                self.assertIn("all_marker_sizes", saved)
                self.assertIn("all_opacities", saved)
                for stale in (
                    "all_rgb",
                    "all_shape_indices",
                    "all_point_sizes",
                    "all_line_angles",
                    "all_line_lengths",
                ):
                    self.assertNotIn(stale, saved)
            self.assertTrue((figures_run / "parameter_grid_encodings.png").exists())

            base, transformed, labels, grid = load_dataset_artifacts(config)
            self.assertEqual(base.shape, (18, 2))
            self.assertEqual(transformed.shape, (12, 18, 2))
            self.assertEqual(labels.shape, (18,))
            self.assertEqual(grid.shape, (12, 3))

            mismatch = copy.deepcopy(config)
            mismatch["base_transform"]["noise_scale"] = 0.2
            with self.assertRaisesRegex(ValueError, "does not match"):
                generate_dataset(mismatch)

            result_paths = sorted((data_run / "metrics").glob("*.npz"))
            self.assertEqual(len(result_paths), 7)
            self.assertTrue(all(path.exists() for path in result_paths))
            manifest = json.loads(
                (data_run / "analysis_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["metric_results"]), 7)
            self.assertTrue((figures_run / "mds_base_relative_overview.png").exists())
            self.assertTrue((figures_run / "mds_pairwise_overview.png").exists())
            self.assertTrue(
                (figures_run / "mds_3d_base_relative_overview.png").exists()
            )
            self.assertTrue((figures_run / "mds_3d_pairwise_overview.png").exists())

            signal_paths = [
                data_run / "metrics" / f"{name}.npz"
                for name in (
                    "cka_rbf",
                    "rbf_uka",
                    "rbf_rwka_symmetric",
                    "rbf_degree_crwka",
                )
            ]
            self.assertEqual(
                len(list((data_run / ".scratch" / "signal_parts").glob("*.npz"))),
                2,
            )
            for path in signal_paths:
                path.unlink()
            analyze_only = copy.deepcopy(config)
            analyze_only["stages"] = ["analyze"]
            with mock.patch(
                "manifold_repsim.experiments.permuted_gaussian_mds.analysis."
                "compute_signal_similarity_matrices",
                side_effect=AssertionError("signal checkpoint was not reused"),
            ):
                run(analyze_only)
            self.assertTrue(all(path.is_file() for path in signal_paths))
            self.assertEqual(len(list((figures_run / "metrics").glob("*.png"))), 7)
            self.assertEqual(
                len(list((figures_run / "metrics_3d").glob("*.png"))),
                7,
            )


if __name__ == "__main__":
    unittest.main()
