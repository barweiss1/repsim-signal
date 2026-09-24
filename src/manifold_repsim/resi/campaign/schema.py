"""Campaign YAML schema: dataclasses and strict validation.

This module performs no I/O beyond reading the campaign file itself, and imports
neither Torch nor ReSi, so campaign definitions can be validated anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from manifold_repsim.resi.measures import (
    MANIFOLD_RESI_MEASURE_CLASSES,
    get_measure_spec,
)
from manifold_repsim.resi.paths import (
    DEFAULT_ANALYSIS_ROOT,
    DEFAULT_RUN_ROOT,
    REPO_ROOT,
    expanded_path,
)

# Derived from the shared catalogue so campaign validation cannot accept a
# measure the adapter does not actually register.
LOCAL_MANIFOLD_MEASURES = set(MANIFOLD_RESI_MEASURE_CLASSES)

SUPPORTED_DOMAINS = frozenset({"graphs", "vision", "language"})
CAMPAIGN_KEYS = frozenset(
    {
        "version",
        "domain",
        "measures",
        "quality_measures",
        "benchmarks",
        "run_root",
        "analysis_root",
        "legacy_signal_files",
        "known_missing_models",
        "measure_sweeps",
    }
)
BENCHMARK_KEYS = frozenset(
    {"id", "dataset", "base_config", "architectures", "baselines", "cache_to_mem"}
)
BASELINE_KEYS = frozenset({"path", "architectures", "name"})
KNOWN_MISSING_KEYS = frozenset(
    {
        "benchmark",
        "dataset",
        "architecture",
        "setting",
        "train_dataset",
        "seed",
        "reason",
    }
)
MEASURE_SWEEP_KEYS = frozenset({"min", "max", "num", "scale"})


@dataclass(frozen=True)
class BaselineSpec:
    path: str
    architectures: tuple[str, ...]
    name: str | None = None


@dataclass(frozen=True)
class BenchmarkSpec:
    id: str
    dataset: str
    base_config: str
    architectures: tuple[str, ...]
    baselines: tuple[BaselineSpec, ...]
    cache_to_mem: bool = True


@dataclass(frozen=True)
class KnownMissingModel:
    benchmark: str
    dataset: str
    architecture: str
    setting: str
    train_dataset: str
    seed: int
    reason: str


@dataclass(frozen=True)
class MeasureSweepSpec:
    """A campaign-local AUC grid override for one registered measure.

    Only generated grids are accepted. An exact value list would have to be
    valid for whatever row count each comparison happens to have, which the
    campaign cannot know when it is written.
    """

    measure: str
    minimum: float
    maximum: float
    num: int
    scale: str

    def to_grid_mapping(self) -> dict[str, Any]:
        """Return the shape accepted by the shared sweep-grid resolver."""
        return {
            "min": self.minimum,
            "max": self.maximum,
            "num": self.num,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class Campaign:
    path: Path
    domain: str
    measures: tuple[str, ...]
    quality_measures: tuple[str, ...]
    benchmarks: tuple[BenchmarkSpec, ...]
    run_root: Path
    analysis_root: Path
    legacy_signal_files: tuple[str, ...]
    known_missing_models: tuple[KnownMissingModel, ...]
    measure_sweeps: tuple[MeasureSweepSpec, ...] = ()

    def sweep_overrides(self) -> dict[str, dict[str, Any]]:
        """Return per-measure grid mappings keyed by ReSi measure class name."""
        return {spec.measure: spec.to_grid_mapping() for spec in self.measure_sweeps}


def _string_list(
    value: Any, field: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list):
        values = tuple(str(item) for item in value)
    else:
        raise ValueError(f"{field} must be a string or list of strings")
    if not allow_empty and not values:
        raise ValueError(f"{field} cannot be empty")
    if any(not item.strip() for item in values):
        raise ValueError(f"{field} contains an empty value")
    return values


def _parse_baselines(
    item: Mapping[str, Any], number: int, architectures: tuple[str, ...]
) -> tuple[BaselineSpec, ...]:
    baselines: list[BaselineSpec] = []
    for baseline_number, baseline in enumerate(item.get("baselines", [])):
        field = f"benchmarks[{number}].baselines[{baseline_number}]"
        if not isinstance(baseline, dict):
            raise ValueError(f"{field} must be a mapping")
        unknown_baseline = sorted(set(baseline) - BASELINE_KEYS)
        if unknown_baseline:
            raise ValueError(f"Unknown {field} key(s): {', '.join(unknown_baseline)}")
        baseline_architectures = _string_list(
            baseline.get("architectures", list(architectures)),
            f"{field}.architectures",
        )
        if not set(baseline_architectures).issubset(architectures):
            raise ValueError(
                f"Baseline architectures must be included in "
                f"benchmarks[{number}].architectures"
            )
        baseline_path = str(baseline.get("path", "")).strip()
        if not baseline_path:
            raise ValueError(f"{field}.path is required")
        baselines.append(
            BaselineSpec(baseline_path, baseline_architectures, baseline.get("name"))
        )
    return tuple(baselines)


def _parse_benchmarks(raw: Mapping[str, Any]) -> tuple[BenchmarkSpec, ...]:
    benchmark_values = raw.get("benchmarks")
    if not isinstance(benchmark_values, list) or not benchmark_values:
        raise ValueError("benchmarks must be a non-empty list")
    benchmarks: list[BenchmarkSpec] = []
    task_keys: set[tuple[str, str, str]] = set()
    for number, item in enumerate(benchmark_values):
        if not isinstance(item, dict):
            raise ValueError(f"benchmarks[{number}] must be a mapping")
        unknown_benchmark = sorted(set(item) - BENCHMARK_KEYS)
        if unknown_benchmark:
            raise ValueError(
                f"Unknown benchmarks[{number}] key(s): {', '.join(unknown_benchmark)}"
            )
        benchmark_id = str(item.get("id", "")).strip()
        dataset = str(item.get("dataset", "")).strip()
        base_config = str(item.get("base_config", "")).strip()
        if not benchmark_id or not dataset or not base_config:
            raise ValueError(
                f"benchmarks[{number}] requires id, dataset, and base_config"
            )
        architectures = _string_list(
            item.get("architectures"), f"benchmarks[{number}].architectures"
        )
        cache_to_mem = item.get("cache_to_mem", True)
        if not isinstance(cache_to_mem, bool):
            raise ValueError(f"benchmarks[{number}].cache_to_mem must be a boolean")
        baselines = _parse_baselines(item, number, architectures)
        for architecture in architectures:
            key = (benchmark_id, dataset, architecture)
            if key in task_keys:
                raise ValueError(f"Duplicate compute task: {key}")
            task_keys.add(key)
        benchmarks.append(
            BenchmarkSpec(
                benchmark_id,
                dataset,
                base_config,
                architectures,
                baselines,
                cache_to_mem,
            )
        )
    return tuple(benchmarks)


def _parse_known_missing(
    raw: Mapping[str, Any], benchmarks: tuple[BenchmarkSpec, ...]
) -> tuple[KnownMissingModel, ...]:
    known_missing_values = raw.get("known_missing_models", [])
    if not isinstance(known_missing_values, list):
        raise ValueError("known_missing_models must be a list")
    known_missing: list[KnownMissingModel] = []
    benchmark_lookup = {
        (benchmark.id, benchmark.dataset): benchmark for benchmark in benchmarks
    }
    for number, item in enumerate(known_missing_values):
        field = f"known_missing_models[{number}]"
        if not isinstance(item, dict):
            raise ValueError(f"{field} must be a mapping")
        unknown = sorted(set(item) - KNOWN_MISSING_KEYS)
        missing = sorted(KNOWN_MISSING_KEYS - set(item))
        if unknown:
            raise ValueError(f"Unknown {field} key(s): {', '.join(unknown)}")
        if missing:
            raise ValueError(f"{field} requires: {', '.join(missing)}")
        values = {key: str(item[key]).strip() for key in KNOWN_MISSING_KEYS - {"seed"}}
        if any(not value for value in values.values()):
            raise ValueError(f"{field} contains an empty value")
        seed = item["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{field}.seed must be a non-negative integer")
        benchmark = benchmark_lookup.get((values["benchmark"], values["dataset"]))
        if benchmark is None:
            raise ValueError(
                f"{field} does not match a configured benchmark/dataset: "
                f"{values['benchmark']}/{values['dataset']}"
            )
        if values["architecture"] not in benchmark.architectures:
            raise ValueError(
                f"{field}.architecture is not configured for {values['benchmark']}: "
                f"{values['architecture']}"
            )
        known_missing.append(
            KnownMissingModel(
                benchmark=values["benchmark"],
                dataset=values["dataset"],
                architecture=values["architecture"],
                setting=values["setting"],
                train_dataset=values["train_dataset"],
                seed=seed,
                reason=values["reason"],
            )
        )
    return tuple(known_missing)


def _parse_measure_sweeps(
    raw: Mapping[str, Any], measures: tuple[str, ...]
) -> tuple[MeasureSweepSpec, ...]:
    values = raw.get("measure_sweeps", {})
    if not isinstance(values, dict):
        raise ValueError("measure_sweeps must be a mapping of measure to grid")
    sweeps: list[MeasureSweepSpec] = []
    for measure in sorted(values):
        field = f"measure_sweeps[{measure}]"
        if measure not in measures:
            raise ValueError(f"{field} is not a configured campaign measure")
        if not get_measure_spec(measure).is_auc:
            raise ValueError(f"{field} only applies to sweeping AUC measures")
        grid = values[measure]
        if not isinstance(grid, dict):
            raise ValueError(f"{field} must be a mapping")
        unknown = sorted(set(grid) - MEASURE_SWEEP_KEYS)
        if unknown:
            raise ValueError(f"Unknown {field} key(s): {', '.join(unknown)}")
        missing = sorted(MEASURE_SWEEP_KEYS - set(grid))
        if missing:
            raise ValueError(f"{field} requires: {', '.join(missing)}")
        num = grid["num"]
        if isinstance(num, bool) or not isinstance(num, int) or num < 2:
            raise ValueError(f"{field}.num must be an integer of at least 2")
        scale = str(grid["scale"])
        if scale not in {"linear", "log"}:
            raise ValueError(f"{field}.scale must be 'linear' or 'log'")
        minimum = float(grid["min"])
        maximum = float(grid["max"])
        if minimum >= maximum:
            raise ValueError(f"{field}.min must be smaller than max")
        if scale == "log" and minimum <= 0:
            raise ValueError(f"{field} with log scale requires a positive min")
        sweeps.append(MeasureSweepSpec(measure, minimum, maximum, num, scale))
    return tuple(sweeps)


def load_campaign(path: str | Path) -> Campaign:
    """Parse and strictly validate a campaign YAML file."""
    campaign_path = expanded_path(path, root=REPO_ROOT)
    with campaign_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Campaign must be a YAML mapping: {campaign_path}")
    if raw.get("version") != 1:
        raise ValueError("Campaign 'version' must be 1")

    unknown = sorted(set(raw) - CAMPAIGN_KEYS)
    if unknown:
        raise ValueError(f"Unknown campaign key(s): {', '.join(unknown)}")

    domain = str(raw.get("domain", "")).strip().lower()
    if domain not in SUPPORTED_DOMAINS:
        raise ValueError("Campaign domain must be graphs, vision, or language")
    measures = _string_list(raw.get("measures"), "measures")
    if len(set(measures)) != len(measures):
        raise ValueError("Campaign measures must be unique")
    unknown_measures = sorted(set(measures) - LOCAL_MANIFOLD_MEASURES)
    if unknown_measures:
        raise ValueError(
            f"Campaign contains unknown local manifold measure(s): "
            f"{', '.join(unknown_measures)}"
        )
    quality_measures = _string_list(raw.get("quality_measures"), "quality_measures")
    benchmarks = _parse_benchmarks(raw)

    run_root = expanded_path(raw.get("run_root", DEFAULT_RUN_ROOT), root=REPO_ROOT)
    analysis_root = expanded_path(
        raw.get("analysis_root", DEFAULT_ANALYSIS_ROOT), root=REPO_ROOT
    )
    legacy = _string_list(
        raw.get("legacy_signal_files", []), "legacy_signal_files", allow_empty=True
    )
    known_missing = _parse_known_missing(raw, benchmarks)
    measure_sweeps = _parse_measure_sweeps(raw, measures)
    return Campaign(
        campaign_path,
        domain,
        measures,
        quality_measures,
        benchmarks,
        run_root,
        analysis_root,
        legacy,
        known_missing,
        measure_sweeps,
    )


__all__ = [
    "BaselineSpec",
    "BenchmarkSpec",
    "Campaign",
    "KnownMissingModel",
    "LOCAL_MANIFOLD_MEASURES",
    "MeasureSweepSpec",
    "load_campaign",
]
