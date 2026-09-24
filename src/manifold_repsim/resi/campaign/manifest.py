"""Task manifests: the JSONL contract between preparation and execution.

The serialized shape is a compatibility surface. Manifests are written on one
machine and read on another, and generated files already exist on the cluster,
so fields are added with defaults and never renamed or reordered.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from manifold_repsim.resi.paths import expanded_path


@dataclass(frozen=True)
class ManifestEntry:
    """One runnable ReSi task: a compute shard or a baseline import."""

    kind: str
    index: int
    domain: str
    benchmark: str
    dataset: str
    architectures: tuple[str, ...]
    measures: tuple[str, ...]
    config_path: str
    result_path: str
    full_csv_path: str
    source_result_path: str | None = None
    legacy_signal_paths: tuple[str, ...] = ()
    measure_sweeps: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize one manifest line with stable key ordering.

        An empty sweep override is omitted rather than written as ``{}`` so a
        campaign that does not override any grid produces byte-identical
        manifests to those generated before the field existed.
        """
        data = asdict(self)
        data["architectures"] = list(self.architectures)
        data["measures"] = list(self.measures)
        data["legacy_signal_paths"] = list(self.legacy_signal_paths)
        if self.measure_sweeps:
            data["measure_sweeps"] = {
                measure: dict(grid)
                for measure, grid in sorted(self.measure_sweeps.items())
            }
        else:
            data.pop("measure_sweeps")
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ManifestEntry":
        """Build an entry from one manifest line, tolerating older layouts."""
        value = dict(value)
        value["architectures"] = tuple(value.get("architectures", ()))
        value["measures"] = tuple(value.get("measures", ()))
        value["legacy_signal_paths"] = tuple(value.get("legacy_signal_paths", ()))
        value["measure_sweeps"] = dict(value.get("measure_sweeps", {}) or {})
        return cls(**value)


def write_manifest(path: str | Path, entries: list[ManifestEntry]) -> None:
    """Write one JSON object per task, in index order."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.to_json() + "\n")


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    """Read a manifest, enforcing contiguous zero-based task indices."""
    manifest_path = expanded_path(path)
    entries: list[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on {manifest_path}:{line_number}: {exc}"
                ) from exc
            entries.append(ManifestEntry.from_dict(value))
    for expected, entry in enumerate(entries):
        if entry.index != expected:
            raise ValueError(
                f"Manifest indices must be contiguous and zero-based: "
                f"expected {expected}, got {entry.index}"
            )
    return entries


__all__ = ["ManifestEntry", "read_manifest", "write_manifest"]
