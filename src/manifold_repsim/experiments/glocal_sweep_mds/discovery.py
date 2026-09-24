"""Discovery and validation for aligned glocal feature batches."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .config import exact_selection, fingerprint, input_root

TRANSFORM_PATTERN = re.compile(
    r"^glocal-lambda-(?P<lambda>[^/]+)-alpha-(?P<alpha>[^/]+)-tau-(?P<tau>[^/]+)$"
)
REQUIRED_BATCH_KEYS = frozenset({"features", "labels", "sample_ids", "sample_indices"})


class GroupRef(NamedTuple):
    """Minimal model/dataset identity reconstructed from a saved manifest.

    Plotting only ever reads ``.model``/``.dataset`` off a group, so a
    ``stages: [plot]``-only run can plot from this instead of a full
    ``DatasetGroup``, without re-running ``discover()`` or touching
    ``input_root``.
    """

    model: str
    dataset: str


@dataclass(frozen=True)
class Condition:
    """One none or glocal condition represented in an MDS result."""

    name: str
    lambda_value: float | None
    alpha_value: float | None
    tau_value: float | None
    is_none: bool = False

    def record(self) -> dict[str, Any]:
        """Return JSON-compatible condition metadata."""
        return {
            "name": self.name,
            "lambda": self.lambda_value,
            "alpha": self.alpha_value,
            "tau": self.tau_value,
            "is_none": self.is_none,
        }


@dataclass(frozen=True)
class DatasetGroup:
    """Resolved inputs for one model and dataset analyzed independently."""

    model: str
    dataset: str
    root: Path
    conditions: tuple[Condition, ...]
    batch_indices: tuple[int, ...]
    batch_files: tuple[str, ...]
    batch_counts: tuple[int, ...]
    source_fingerprint: str

    def condition_path(self, condition: Condition) -> Path:
        """Return the directory containing one condition's batch files."""
        return self.root / condition.name


def parse_glocal_transform(name: str) -> tuple[float, float, float] | None:
    """Parse a strict glocal transform name, returning ``None`` for other kinds."""
    match = TRANSFORM_PATTERN.fullmatch(name)
    if match is None:
        return None
    try:
        values = tuple(float(match.group(key)) for key in ("lambda", "alpha", "tau"))
    except ValueError as exc:
        raise ValueError(f"Invalid numeric glocal transform name: {name}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite glocal transform value: {name}")
    return values


def _directories(path: Path) -> list[str]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(item.name for item in path.iterdir() if item.is_dir())


def _scan_conditions(path: Path) -> dict[tuple[float, float, float], str]:
    found: dict[tuple[float, float, float], str] = {}
    for name in _directories(path):
        values = parse_glocal_transform(name)
        if values is None:
            continue
        if values in found:
            raise ValueError(
                f"Duplicate numeric glocal condition under {path}: {found[values]} and {name}"
            )
        found[values] = name
    if not found:
        raise ValueError(f"No glocal transforms found under {path}")
    return found


def _load_metadata(
    path: Path, model: str, dataset: str, transform: str
) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("model") != model or metadata.get("dataset") != dataset:
        raise ValueError(f"Model or dataset metadata mismatch: {metadata_path}")
    if metadata.get("transform") != transform:
        raise ValueError(f"Transform metadata mismatch: {metadata_path}")
    expected_kind = "none" if transform == "none" else "glocal"
    if metadata.get("transform_kind") != expected_kind:
        raise ValueError(f"Transform kind metadata mismatch: {metadata_path}")
    batches = metadata.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError(f"Metadata has no batches: {metadata_path}")
    manifest = []
    for batch in batches:
        if set(batch) != {"file", "count"}:
            raise ValueError(f"Invalid batch metadata: {metadata_path}")
        if (
            not isinstance(batch["file"], str)
            or Path(batch["file"]).name != batch["file"]
            or not isinstance(batch["count"], int)
            or isinstance(batch["count"], bool)
            or batch["count"] < 1
        ):
            raise ValueError(f"Invalid batch metadata: {metadata_path}")
        if not (path / batch["file"]).is_file():
            raise FileNotFoundError(path / batch["file"])
        manifest.append((batch["file"], batch["count"]))
    return {"metadata": metadata, "manifest": tuple(manifest), "path": metadata_path}


def _artifact_signature(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(paths):
        stat = path.stat()
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return records


def discover(config: dict[str, Any]) -> tuple[dict[str, Any], tuple[DatasetGroup, ...]]:
    """Resolve selectors and validate the complete selected transform grid."""
    root = input_root(config)
    available_models = [name for name in _directories(root) if name != "_index"]
    models = exact_selection(config["selection"]["models"], available_models, "models")
    first_model_datasets = _directories(root / models[0])
    datasets = exact_selection(
        config["selection"]["datasets"], first_model_datasets, "datasets"
    )
    for model in models[1:]:
        missing = sorted(set(datasets) - set(_directories(root / model)))
        if missing:
            raise ValueError(f"Model {model} is missing selected datasets: {missing}")

    first_scan = _scan_conditions(root / models[0] / datasets[0])
    lambdas = exact_selection(
        config["selection"]["lambda"],
        sorted({values[0] for values in first_scan}),
        "lambda",
    )
    alphas = exact_selection(
        config["selection"]["alpha"],
        sorted({values[1] for values in first_scan}),
        "alpha",
    )
    taus = exact_selection(
        config["selection"]["tau"],
        sorted({values[2] for values in first_scan}),
        "tau",
    )
    lambda_splits = exact_selection(config["lambda_splits"], lambdas, "lambda_splits")
    selected_values = list(product(lambdas, alphas, taus))

    effective = dict(config)
    effective["lambda_splits"] = list(lambda_splits)
    effective["selection"] = {
        "models": list(models),
        "datasets": list(datasets),
        "batches": config["selection"]["batches"],
        "lambda": list(lambdas),
        "alpha": list(alphas),
        "tau": list(taus),
    }
    groups = []
    resolved_batches: list[int] | None = None
    for model in models:
        for dataset in datasets:
            group_root = root / model / dataset
            if not (group_root / "none").is_dir():
                raise FileNotFoundError(group_root / "none")
            found = _scan_conditions(group_root)
            missing = [values for values in selected_values if values not in found]
            if missing:
                raise ValueError(
                    f"Incomplete selected glocal grid for {model}/{dataset}; missing: {missing}"
                )
            conditions = [Condition("none", None, None, None, True)]
            conditions.extend(
                Condition(found[values], values[0], values[1], values[2])
                for values in selected_values
            )

            metadata_records = [
                _load_metadata(
                    group_root / condition.name, model, dataset, condition.name
                )
                for condition in conditions
            ]
            manifest = metadata_records[0]["manifest"]
            for condition, record in zip(conditions[1:], metadata_records[1:]):
                if record["manifest"] != manifest:
                    raise ValueError(
                        f"Batch manifest differs from none for {model}/{dataset}/{condition.name}"
                    )
            available_batches = list(range(len(manifest)))
            batches = exact_selection(
                config["selection"]["batches"], available_batches, "batches"
            )
            if resolved_batches is None:
                resolved_batches = batches
                effective["selection"]["batches"] = list(batches)
            elif batches != resolved_batches:
                raise ValueError(
                    "Selected batch indices differ between model/dataset groups"
                )

            selected_files = [root / "_index" / dataset / "test.npz"]
            for condition, record in zip(conditions, metadata_records):
                selected_files.append(record["path"])
                selected_files.extend(
                    group_root / condition.name / manifest[index][0]
                    for index in batches
                )
            for path in selected_files:
                if not path.is_file():
                    raise FileNotFoundError(path)
            signature = _artifact_signature(selected_files, root)
            groups.append(
                DatasetGroup(
                    model=model,
                    dataset=dataset,
                    root=group_root,
                    conditions=tuple(conditions),
                    batch_indices=tuple(batches),
                    batch_files=tuple(item[0] for item in manifest),
                    batch_counts=tuple(item[1] for item in manifest),
                    source_fingerprint=fingerprint(signature),
                )
            )
    return effective, tuple(groups)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        missing = REQUIRED_BATCH_KEYS.difference(saved.files)
        if missing:
            raise ValueError(f"Batch file is missing keys {sorted(missing)}: {path}")
        return {key: saved[key].copy() for key in saved.files}


def _assert_equal(left: np.ndarray, right: np.ndarray, message: str) -> None:
    if not np.array_equal(left, right):
        raise ValueError(message)


def load_aligned_pair(
    feature_root: Path,
    group: DatasetGroup,
    condition: Condition,
    batch_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate one none-to-condition feature comparison."""
    filename = group.batch_files[batch_index]
    base_path = group.root / "none" / filename
    target_path = group.condition_path(condition) / filename
    base = _load_npz(base_path)
    target = base if condition.is_none else _load_npz(target_path)
    context = f"{group.model}/{group.dataset}/{condition.name}/batch-{batch_index}"
    for key in ("sample_ids", "sample_indices", "labels"):
        _assert_equal(base[key], target[key], f"{key} differs from none: {context}")

    with np.load(
        feature_root / "_index" / group.dataset / "test.npz", allow_pickle=False
    ) as saved:
        shared = {key: saved[key] for key in ("sample_ids", "sample_indices", "labels")}
        indices = base["sample_indices"]
        if (
            indices.ndim != 1
            or indices.size == 0
            or indices.min() < 0
            or indices.max() >= len(shared["sample_ids"])
        ):
            raise ValueError(f"sample_indices are invalid: {context}")
        for key in shared:
            _assert_equal(
                base[key],
                shared[key][indices],
                f"{key} differs from shared index: {context}",
            )

    base_features = base["features"]
    target_features = target["features"]
    if base_features.ndim != 2 or target_features.ndim != 2:
        raise ValueError(f"Features must be two-dimensional: {context}")
    if base_features.shape != target_features.shape:
        raise ValueError(f"Feature shapes differ from none: {context}")
    if base_features.shape[0] != group.batch_counts[batch_index]:
        raise ValueError(f"Feature count differs from metadata: {context}")
    if not np.isfinite(base_features).all() or not np.isfinite(target_features).all():
        raise ValueError(f"Features contain non-finite values: {context}")
    return base_features, target_features
