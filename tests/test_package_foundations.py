import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import yaml

import manifold_repsim
from manifold_repsim.config import load_yaml_config, save_effective_config


class PackageFoundationTests(unittest.TestCase):
    def test_package_exposes_version(self):
        self.assertEqual(manifold_repsim.__version__, "0.1.0")

    def test_load_yaml_config_requires_mapping_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.yaml"
            path.write_text("- one\n- two\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must contain a mapping"):
                load_yaml_config(path)

    def test_save_effective_config_normalizes_common_values(self):
        @dataclass
        class DatasetConfig:
            n_points: int
            output: Path

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            saved_path = save_effective_config(
                output_dir,
                {
                    "version": 1,
                    "stages": ("generate", "analyze"),
                    "dataset": DatasetConfig(20, Path("data/example")),
                },
            )

            self.assertEqual(saved_path, output_dir / "config.yaml")
            self.assertEqual(
                yaml.safe_load(saved_path.read_text(encoding="utf-8")),
                {
                    "version": 1,
                    "stages": ["generate", "analyze"],
                    "dataset": {"n_points": 20, "output": "data/example"},
                },
            )
            self.assertEqual(list(output_dir.glob(".config.*.tmp")), [])

    def test_save_effective_config_replaces_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = save_effective_config(temp_dir, {"version": 1})
            second = save_effective_config(temp_dir, {"version": 2})

            self.assertEqual(first, second)
            self.assertEqual(load_yaml_config(second), {"version": 2})


if __name__ == "__main__":
    unittest.main()
