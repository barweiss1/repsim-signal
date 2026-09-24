"""Configuration validation, paths, and fingerprints for the MDS experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from manifold_repsim.config import save_effective_config
from synthetic_datasets.permuted_gaussian_clusters import (
    PermutedGaussianClustersDataset,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
PARAMETERS = ("cluster_mixing_probability", "noise_scale", "n_permute")
PARAMETER_LABELS = {
    "cluster_mixing_probability": r"$p_{mix}$",
    "noise_scale": r"$\sigma_{noise}$",
    "n_permute": r"$N_{perm}$",
}
POINT_SIZE_RANGE = (24.0, 140.0)
OPACITY_RANGE = (0.25, 1.0)
SIGNAL_METRICS = (
    "cka_rbf",
    "rbf_uka",
    "rbf_rwka_symmetric",
    "rbf_degree_crwka",
)
FIXED_METRICS = frozenset({"cka", "cka_rbf", "mutual_knn_dist"})
LEGEND_STYLES = frozenset({"markers", "colorbar"})

# ``None`` is a leaf; nested mappings require exactly the declared keys.
SCHEMA = {
    "version": None,
    "run_name": None,
    "stages": None,
    "dataset": {
        "n_points": None,
        "dim": None,
        "n_clusters": None,
        "seed": None,
        "balanced": None,
        "center_sampling": None,
    },
    "transform": {"mode": None, "seed": None, "base_seed": None},
    "base_transform": {name: None for name in PARAMETERS},
    "grid": {
        "cluster_mixing_probability": {"min": None, "max": None, "num": None},
        "noise_scale": {"min": None, "max": None, "num": None},
        "n_permute": {"values": None},
    },
    "metrics": {
        "fixed": None,
        "signals": {
            "names": None,
            "parameter": None,
            "min": None,
            "max": None,
            "num": None,
            "scale": None,
            "integration": None,
        },
        "signal_sweeps": {"fixed_parameters": {name: None for name in PARAMETERS}},
    },
    "mds": {
        "random_state": None,
        "n_init": None,
        "max_iter": None,
        "eps": None,
    },
    "visualization": {
        "encodings": {"color": None, "size": None, "opacity": None},
        "include_base": None,
        "dpi": None,
        "legend_style": None,
    },
    "outputs": {"data_root": None, "figures_root": None},
    "execution": {
        "overwrite_dataset": None,
        "recompute_metrics": None,
        "checkpoint_signals": None,
    },
}


def require(condition: bool, message: str) -> None:
    """Raise a configuration or artifact validation error."""
    if not condition:
        raise ValueError(message)


def _strict(value: Any, schema: dict[str, Any], field: str = "configuration") -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    unknown = sorted(set(value) - set(schema))
    missing = sorted(set(schema) - set(value))
    if unknown:
        raise ValueError(f"Unknown {field} key(s): {', '.join(unknown)}")
    if missing:
        raise ValueError(f"Missing {field} key(s): {', '.join(missing)}")
    for key, child_schema in schema.items():
        if child_schema is not None:
            _strict(value[key], child_schema, f"{field}.{key}")


def _is_int(value: Any, minimum: int = 0) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def _random_seed() -> int:
    """Draw a fresh nondeterministic seed for a ``null``-declared seed field."""
    return int.from_bytes(os.urandom(4), "big")


def validate_config(raw: Any) -> dict[str, Any]:
    """Validate the strict v1 schema and return a detached plain dictionary."""
    config = copy.deepcopy(raw)
    if isinstance(config, dict):
        grid = config.get("grid")
        base = config.get("base_transform")
        dataset_cfg = config.get("dataset")
        transform_cfg = config.get("transform")
        if isinstance(transform_cfg, dict) and isinstance(dataset_cfg, dict):
            transform_cfg.setdefault("seed", dataset_cfg.get("seed"))
            dataset_seed = dataset_cfg.get("seed")
            transform_cfg.setdefault(
                "base_seed",
                dataset_seed + 1 if isinstance(dataset_seed, int) else None,
            )
            # A ``null`` seed (whether defaulted above or written explicitly)
            # opts into a fresh nondeterministic seed for this run, so the
            # base and swept transforms can be told apart from a fixed one.
            for seed_key in ("seed", "base_seed"):
                if transform_cfg.get(seed_key) is None:
                    transform_cfg[seed_key] = _random_seed()
        if isinstance(grid, dict) and isinstance(base, dict):
            if "n_permute" not in grid and "n_permute" in base:
                grid["n_permute"] = {"values": [base["n_permute"]]}
            for field in ("cluster_mixing_probability", "noise_scale"):
                if field not in grid and field in base:
                    grid[field] = {
                        "min": base[field],
                        "max": base[field],
                        "num": 1,
                    }
        visualization = config.get("visualization")
        if isinstance(visualization, dict):
            if isinstance(visualization.get("encodings"), dict):
                for channel in ("color", "size", "opacity"):
                    visualization["encodings"].setdefault(channel, None)
            visualization.setdefault("legend_style", "markers")
        metrics_cfg = config.get("metrics")
        if isinstance(metrics_cfg, dict) and isinstance(base, dict):
            if not isinstance(metrics_cfg.get("signal_sweeps"), dict):
                metrics_cfg["signal_sweeps"] = {}
            if not isinstance(
                metrics_cfg["signal_sweeps"].get("fixed_parameters"), dict
            ):
                metrics_cfg["signal_sweeps"]["fixed_parameters"] = {}
            fixed_parameters = metrics_cfg["signal_sweeps"]["fixed_parameters"]
            for name in PARAMETERS:
                fixed_parameters.setdefault(name, base[name])
    _strict(config, SCHEMA)
    dataset = config["dataset"]
    base = config["base_transform"]
    grid = config["grid"]

    require(config["version"] == 1, "Configuration version must be 1")
    require(
        bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(config["run_name"]))),
        "run_name must be a safe path component",
    )
    stages = config["stages"]
    require(
        isinstance(stages, list)
        and bool(stages)
        and len(stages) == len(set(stages))
        and set(stages) <= {"generate", "analyze"},
        "stages must be a unique non-empty list containing generate and/or analyze",
    )
    require(
        stages != ["analyze", "generate"],
        "generate must precede analyze when both stages are selected",
    )

    for key, minimum in (("n_points", 1), ("dim", 2), ("n_clusters", 2), ("seed", 0)):
        require(
            _is_int(dataset[key], minimum),
            f"dataset.{key} must be an integer >= {minimum}",
        )
    require(
        isinstance(dataset["balanced"], bool),
        "dataset.balanced must be true or false",
    )
    require(
        dataset["center_sampling"]
        in PermutedGaussianClustersDataset.VALID_CENTER_SAMPLING,
        "dataset.center_sampling is invalid",
    )
    require(
        dataset["center_sampling"] == "hypersphere" or dataset["dim"] == 2,
        "Cartesian center sampling requires dataset.dim=2",
    )
    require(
        config["transform"]["mode"]
        in PermutedGaussianClustersDataset.VALID_TRANSFORM_MODES,
        "transform.mode must be resample or modify_base",
    )
    require(
        _is_int(config["transform"]["seed"], 0),
        "transform.seed must be an integer >= 0",
    )
    require(
        _is_int(config["transform"]["base_seed"], 0),
        "transform.base_seed must be an integer >= 0",
    )

    for field in ("cluster_mixing_probability", "noise_scale"):
        require(
            isinstance(base[field], (int, float))
            and np.isfinite(base[field])
            and base[field] >= 0,
            f"base_transform.{field} must be finite and nonnegative",
        )
    require(base["cluster_mixing_probability"] <= 1, "base mixing must not exceed 1")
    require(
        _is_int(base["n_permute"])
        and base["n_permute"] != 1
        and base["n_permute"] <= dataset["n_clusters"],
        "base_transform.n_permute must be 0 or between 2 and n_clusters",
    )

    for field in PARAMETERS[:2]:
        sweep = grid[field]
        require(
            all(isinstance(sweep[key], (int, float)) for key in ("min", "max"))
            and np.isfinite(sweep["min"])
            and np.isfinite(sweep["max"])
            and 0 <= sweep["min"] <= sweep["max"]
            and (
                (
                    sweep["min"] == sweep["max"]
                    and _is_int(sweep["num"], 1)
                    and sweep["num"] == 1
                )
                or (sweep["min"] < sweep["max"] and _is_int(sweep["num"], 2))
            ),
            f"grid.{field} must be fixed (min == max, num == 1) or a sweep "
            "(0 <= min < max, integer num >= 2)",
        )
    require(
        grid["cluster_mixing_probability"]["max"] <= 0.5,
        "grid mixing must not exceed 0.5",
    )
    permutations = grid["n_permute"]["values"]
    require(
        isinstance(permutations, list)
        and bool(permutations)
        and all(_is_int(value) for value in permutations)
        and permutations == sorted(set(permutations))
        and 1 not in permutations
        and permutations[-1] <= dataset["n_clusters"],
        "grid.n_permute.values must be sorted unique integers up to n_clusters "
        "and cannot contain 1",
    )
    require(
        len(permutations) == 1
        and permutations[0] == base["n_permute"]
        or len(permutations) > 1
        and permutations[0] == 0,
        "a fixed n_permute grid must match base_transform.n_permute; a sweep "
        "must start at 0",
    )

    fixed = config["metrics"]["fixed"]
    require(
        isinstance(fixed, list) and bool(fixed),
        "metrics.fixed must be a non-empty list",
    )
    identifiers = []
    expected_kwargs = {
        "cka": set(),
        "cka_rbf": {"rbf_sigma"},
        "mutual_knn_dist": {"topk"},
    }
    for index, specification in enumerate(fixed):
        _strict(
            specification,
            {"id": None, "name": None, "kwargs": None},
            f"metrics.fixed[{index}]",
        )
        metric_name = specification["name"]
        require(
            metric_name in FIXED_METRICS, f"Unsupported fixed metric: {metric_name}"
        )
        require(
            isinstance(specification["kwargs"], dict),
            f"metrics.fixed[{index}].kwargs must be a mapping",
        )
        require(
            bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(specification["id"]))),
            f"metrics.fixed[{index}].id is not path-safe",
        )
        identifiers.append(specification["id"])
        require(
            set(specification["kwargs"]) == expected_kwargs[metric_name],
            f"Invalid kwargs for {metric_name}",
        )
        if metric_name == "cka_rbf":
            require(
                specification["kwargs"]["rbf_sigma"] > 0, "rbf_sigma must be positive"
            )
        if metric_name == "mutual_knn_dist":
            topk = specification["kwargs"]["topk"]
            require(
                _is_int(topk, 1) and topk < dataset["n_points"],
                "topk must be in [1, n_points)",
            )
    require(
        len(identifiers) == len(set(identifiers)), "metrics.fixed ids must be unique"
    )

    signals = config["metrics"]["signals"]
    require(
        signals["names"] == list(SIGNAL_METRICS),
        f"metrics.signals.names must be {list(SIGNAL_METRICS)}",
    )
    require(signals["parameter"] == "rbf_sigma", "signals.parameter must be rbf_sigma")
    require(
        0 < signals["min"] < signals["max"] and _is_int(signals["num"], 2),
        "signals require 0 < min < max and integer num >= 2",
    )
    require(
        signals["scale"] in {"log", "linear"}, "signals.scale must be log or linear"
    )
    require(signals["integration"] == "average", "signals.integration must be average")

    sweep_fixed = config["metrics"]["signal_sweeps"]["fixed_parameters"]
    for field in ("cluster_mixing_probability", "noise_scale"):
        require(
            isinstance(sweep_fixed[field], (int, float))
            and np.isfinite(sweep_fixed[field])
            and sweep_fixed[field] >= 0,
            f"metrics.signal_sweeps.fixed_parameters.{field} must be finite and "
            "nonnegative",
        )
    require(
        sweep_fixed["cluster_mixing_probability"] <= 1,
        "metrics.signal_sweeps.fixed_parameters mixing must not exceed 1",
    )
    require(
        _is_int(sweep_fixed["n_permute"])
        and sweep_fixed["n_permute"] != 1
        and sweep_fixed["n_permute"] <= dataset["n_clusters"],
        "metrics.signal_sweeps.fixed_parameters.n_permute must be 0 or between "
        "2 and n_clusters",
    )

    for key in ("random_state", "n_init", "max_iter"):
        minimum = 0 if key == "random_state" else 1
        require(_is_int(config["mds"][key], minimum), f"mds.{key} is invalid")
    require(config["mds"]["eps"] > 0, "mds.eps must be positive")
    visual = config["visualization"]
    mapping = visual["encodings"]
    require(
        all(value is None or value in PARAMETERS for value in mapping.values()),
        "visual encodings must name data parameters or be null",
    )
    assigned = [value for value in mapping.values() if value is not None]
    swept = {
        name for name in PARAMETERS if len(parameter_grid_levels(name, config)) > 1
    }
    require(
        len(assigned) == len(set(assigned)) and set(assigned) == swept,
        "non-null color, size, and opacity encodings must map each swept "
        "parameter exactly once",
    )
    require(isinstance(visual["include_base"], bool), "include_base must be boolean")
    require(_is_int(visual["dpi"], 1), "visualization.dpi must be positive")
    require(
        visual["legend_style"] in LEGEND_STYLES,
        f"visualization.legend_style must be one of {sorted(LEGEND_STYLES)}",
    )
    for key, value in config["execution"].items():
        require(isinstance(value, bool), f"execution.{key} must be boolean")
    return config


def parameter_grid_levels(name: str, config: dict[str, Any]) -> np.ndarray:
    """Return configured grid levels for validation without importing plotting."""
    specification = config["grid"][name]
    if name == "n_permute":
        return np.asarray(specification["values"])
    return np.linspace(specification["min"], specification["max"], specification["num"])


def resolve_path(path: str | Path) -> Path:
    """Resolve experiment paths relative to the repository root."""
    value = Path(os.path.expandvars(str(path))).expanduser()
    return (REPO_ROOT / value if not value.is_absolute() else value).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one versioned YAML experiment configuration."""
    config_path = resolve_path(path)
    config = validate_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    config["_config_path"] = str(config_path)
    return config


def config_fingerprint(value: Any) -> str:
    """Return the established canonical SHA-256 configuration fingerprint."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def dataset_fingerprint(config: dict[str, Any]) -> str:
    return config_fingerprint(
        {key: config[key] for key in ("dataset", "transform", "base_transform", "grid")}
    )


def _legacy_rgb_dataset_fingerprint(config: dict[str, Any]) -> str:
    return config_fingerprint(
        {key: config[key] for key in ("dataset", "base_transform", "grid")}
        | {
            "rgb": {
                "red": "cluster_mixing_probability",
                "green": "noise_scale",
                "blue": "n_permute",
            }
        }
    )


def compatible_dataset_fingerprints(config: dict[str, Any]) -> set[str]:
    fingerprints = {dataset_fingerprint(config)}
    if config["transform"]["mode"] == "resample":
        fingerprints.add(_legacy_rgb_dataset_fingerprint(config))
    return fingerprints


def _analysis_fingerprint(
    config: dict[str, Any], metric: dict[str, Any], dataset_hash: str
) -> str:
    return config_fingerprint(
        {
            "dataset": dataset_hash,
            "metric": metric,
            "mds": config["mds"],
            "include_base": config["visualization"]["include_base"],
        }
    )


def analysis_fingerprint(config: dict[str, Any], metric: dict[str, Any]) -> str:
    return _analysis_fingerprint(config, metric, dataset_fingerprint(config))


def compatible_analysis_fingerprints(
    config: dict[str, Any], metric: dict[str, Any]
) -> set[str]:
    return {
        _analysis_fingerprint(config, metric, dataset_hash)
        for dataset_hash in compatible_dataset_fingerprints(config)
    }


def output_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    return tuple(
        resolve_path(config["outputs"][key]) / config["run_name"]
        for key in ("data_root", "figures_root")
    )


def effective_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the saved configuration including resolved output locations."""
    saved = {key: value for key, value in config.items() if not key.startswith("_")}
    data_dir, figure_dir = output_paths(config)
    saved["resolved_outputs"] = {
        "data_run": str(data_dir),
        "figures_run": str(figure_dir),
    }
    return saved


def save_run_config(config: dict[str, Any]) -> tuple[Path, Path]:
    """Write the effective config beside both data and figure run artifacts."""
    saved = effective_config(config)
    return tuple(
        save_effective_config(directory, saved) for directory in output_paths(config)
    )
