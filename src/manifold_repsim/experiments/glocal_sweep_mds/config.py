"""Strict YAML configuration for the glocal sweep MDS experiment."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from manifold_repsim.config import load_yaml_config, save_effective_config
from manifold_repsim.metrics import get_metric_spec, supports_prepared_curve
from manifold_repsim.sweeps.grid import SweepGridConfig, resolve_metric_sweep_grid

REPO_ROOT = Path(__file__).resolve().parents[4]
SELECTOR_KEYS = ("models", "datasets", "batches", "lambda", "alpha", "tau")
SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_keys(value: Any, expected: set[str], field: str) -> None:
    _require(isinstance(value, Mapping), f"{field} must be a mapping")
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    _require(not unknown, f"Unknown {field} key(s): {', '.join(unknown)}")
    _require(not missing, f"Missing {field} key(s): {', '.join(missing)}")


def _validate_selector(value: Any, field: str, kind: type) -> None:
    if value == "all":
        return
    _require(
        isinstance(value, list) and bool(value),
        f"{field} must be 'all' or a non-empty list",
    )
    if kind is str:
        valid = all(isinstance(item, str) and bool(item) for item in value)
    elif kind is int:
        valid = all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in value
        )
    else:
        valid = all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and np.isfinite(item)
            for item in value
        )
    _require(valid, f"{field} contains invalid values")
    _require(len(value) == len(set(value)), f"{field} contains duplicate values")


def _validate_metric(raw: Any, index: int) -> dict[str, Any]:
    field = f"metrics[{index}]"
    _strict_keys(raw, {"id", "name", "mode", "kwargs", "grid"}, field)
    metric = copy.deepcopy(dict(raw))
    _require(
        isinstance(metric["id"], str) and bool(SAFE_COMPONENT.fullmatch(metric["id"])),
        f"{field}.id must be a safe path component",
    )
    _require(isinstance(metric["name"], str), f"{field}.name must be a string")
    specification = get_metric_spec(metric["name"])
    _require(
        metric["mode"] in {"fixed", "signal"}, f"{field}.mode must be fixed or signal"
    )
    _require(isinstance(metric["kwargs"], Mapping), f"{field}.kwargs must be a mapping")
    if metric["mode"] == "fixed":
        _require(
            metric["grid"] is None, f"{field}.grid must be null for a fixed metric"
        )
        metric["kwargs"] = dict(metric["kwargs"])
        metric["parameter_name"] = None
        metric["parameter_values"] = [0.0]
        return metric

    _require(specification.sweep is not None, f"{metric['name']} is not sweepable")
    _require(
        supports_prepared_curve(metric["name"]),
        f"{metric['name']} has no exact prepared signal path",
    )
    grid = SweepGridConfig.from_mapping(metric["grid"])
    resolved = resolve_metric_sweep_grid(metric["name"], grid)
    metric["grid"] = {"values": resolved.values.tolist()}
    metric["kwargs"] = dict(metric["kwargs"])
    metric["parameter_name"] = resolved.parameter_name
    metric["parameter_values"] = resolved.values.tolist()
    return metric


def _validate_mds_3d_view(value: Any) -> None:
    """Validate an automatic or explicitly configured Matplotlib camera."""
    if value == "auto":
        return
    _strict_keys(value, {"elevation", "azimuth"}, "visualization.mds_3d_view")
    for key in ("elevation", "azimuth"):
        angle = value[key]
        _require(
            isinstance(angle, (int, float))
            and not isinstance(angle, bool)
            and np.isfinite(angle),
            f"visualization.mds_3d_view.{key} must be finite",
        )
    _require(
        -90 <= value["elevation"] <= 90,
        "visualization.mds_3d_view.elevation must be between -90 and 90",
    )


def validate_config(raw: Any) -> dict[str, Any]:
    """Validate the strict version-one request and resolve metric grids."""
    _strict_keys(
        raw,
        {
            "version",
            "run_name",
            "input_root",
            "stages",
            "selection",
            "lambda_splits",
            "metrics",
            "mds",
            "visualization",
            "outputs",
            "execution",
        },
        "configuration",
    )
    config = copy.deepcopy(dict(raw))
    _require(config["version"] == 1, "Configuration version must be 1")
    _require(
        isinstance(config["run_name"], str)
        and bool(SAFE_COMPONENT.fullmatch(config["run_name"])),
        "run_name must be a safe path component",
    )
    _require(
        isinstance(config["input_root"], str) and config["input_root"],
        "input_root must be a path string",
    )

    stages = config["stages"]
    _require(
        isinstance(stages, list)
        and bool(stages)
        and len(stages) == len(set(stages))
        and set(stages) <= {"compute", "plot"},
        "stages must be a unique non-empty list containing compute and/or plot",
    )
    _require(
        stages != ["plot", "compute"],
        "compute must precede plot when both stages are selected",
    )

    _strict_keys(config["selection"], set(SELECTOR_KEYS), "selection")
    for key in ("models", "datasets"):
        _validate_selector(config["selection"][key], f"selection.{key}", str)
    _validate_selector(config["selection"]["batches"], "selection.batches", int)
    for key in ("lambda", "alpha", "tau"):
        _validate_selector(config["selection"][key], f"selection.{key}", float)
    _validate_selector(config["lambda_splits"], "lambda_splits", float)

    _require(
        isinstance(config["metrics"], list) and bool(config["metrics"]),
        "metrics must be a non-empty list",
    )
    config["metrics"] = [
        _validate_metric(metric, index)
        for index, metric in enumerate(config["metrics"])
    ]
    identifiers = [metric["id"] for metric in config["metrics"]]
    _require(len(identifiers) == len(set(identifiers)), "Metric ids must be unique")

    _strict_keys(config["mds"], {"random_state", "n_init", "max_iter", "eps"}, "mds")
    for key in ("random_state", "n_init", "max_iter"):
        value = config["mds"][key]
        minimum = 0 if key == "random_state" else 1
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
            f"mds.{key} must be an integer >= {minimum}",
        )
    _require(
        isinstance(config["mds"]["eps"], (int, float)) and config["mds"]["eps"] > 0,
        "mds.eps must be positive",
    )

    _strict_keys(config["visualization"], {"dpi", "mds_3d_view"}, "visualization")
    _require(
        isinstance(config["visualization"]["dpi"], int)
        and config["visualization"]["dpi"] > 0,
        "visualization.dpi must be positive",
    )
    _validate_mds_3d_view(config["visualization"]["mds_3d_view"])
    _strict_keys(config["outputs"], {"data_root", "figures_root"}, "outputs")
    for key in ("data_root", "figures_root"):
        _require(
            isinstance(config["outputs"][key], str) and config["outputs"][key],
            f"outputs.{key} must be a path string",
        )
    _strict_keys(
        config["execution"], {"recompute_scores", "recompute_mds"}, "execution"
    )
    for key in ("recompute_scores", "recompute_mds"):
        _require(
            isinstance(config["execution"][key], bool),
            f"execution.{key} must be true or false",
        )
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one glocal experiment YAML file."""
    return validate_config(load_yaml_config(path))


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def input_root(config: Mapping[str, Any]) -> Path:
    """Return the configured feature root as an absolute path."""
    return _path(config["input_root"])


def output_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    """Return the data and figure directories for the configured run."""
    run_name = config["run_name"]
    return (
        _path(config["outputs"]["data_root"]) / run_name,
        _path(config["outputs"]["figures_root"]) / run_name,
    )


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove internal resolved metric fields from a runnable configuration."""
    value = copy.deepcopy(dict(config))
    for metric in value["metrics"]:
        metric.pop("parameter_name", None)
        metric.pop("parameter_values", None)
    return value


def save_run_config(config: Mapping[str, Any]) -> None:
    """Save the same effective request beside data and figure artifacts."""
    value = public_config(config)
    for path in output_paths(config):
        save_effective_config(path, value)


def fingerprint(value: Any) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible content."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def exact_selection(requested: Any, available: Sequence[Any], field: str) -> list[Any]:
    """Resolve an exact selector without nearest-value or subset fallback."""
    available = list(available)
    if requested == "all":
        return available
    missing = [value for value in requested if value not in available]
    _require(
        not missing,
        f"Requested {field} values are unavailable: {missing}; available: {available}",
    )
    return list(requested)
