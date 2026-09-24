"""Shared YAML configuration helpers for reproducible runs."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


def _plain_config_value(value: Any) -> Any:
    """Convert common configuration values to safe, portable YAML values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_config_value(asdict(value))
    if isinstance(value, Enum):
        return _plain_config_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain_config_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config_value(item) for item in value]
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file whose root value must be a mapping."""
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML configuration must contain a mapping: {config_path}")
    return dict(value)


def save_effective_config(
    output_dir: str | Path,
    config: Mapping[str, Any],
) -> Path:
    """Atomically save an effective run configuration as ``config.yaml``.

    Callers are responsible for applying workflow defaults before invoking this
    function. The snapshot belongs in the same run directory as the outputs it
    describes.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    destination_dir = Path(output_dir).expanduser()
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "config.yaml"
    plain_config = _plain_config_value(config)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination_dir,
            prefix=".config.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            yaml.safe_dump(plain_config, handle, sort_keys=False)
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination
