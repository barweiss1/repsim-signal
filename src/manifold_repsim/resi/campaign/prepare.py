"""Turn a validated campaign into ReSi run configurations and task manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from manifold_repsim.config import save_effective_config
from manifold_repsim.resi.identifiers import MODEL_FIELDS, parse_comparison_id
from manifold_repsim.resi.paths import expanded_path, require_env_path, slug

from .manifest import ManifestEntry, write_manifest
from .schema import Campaign, load_campaign

RESULT_FILENAME = "results.parquet"
FULL_CSV_FILENAME = "results_full.csv"
SIGNAL_FILENAME = "results_signals.jsonl"


def result_metric_column(frame: pd.DataFrame) -> str:
    """Return whichever column holds the measure name in a ReSi result frame.

    ReSi has used both names across versions, and baseline parquets written by
    older runs are still read as inputs.
    """
    for candidate in ("metric", "similarity_measure"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(
        "Result frame has neither a 'metric' nor 'similarity_measure' column"
    )


def measures_present(frame: pd.DataFrame) -> set[str]:
    """Every measure name recorded in a ReSi result frame."""
    metric_column = result_metric_column(frame)
    return {str(value) for value in frame[metric_column].dropna().unique()}


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"ReSi config must contain a mapping: {path}")
    if not isinstance(value.get("experiments"), list) or not value["experiments"]:
        raise ValueError(f"ReSi config has no experiments: {path}")
    return value


def _filtered_config(
    base: dict[str, Any],
    architectures: Iterable[str],
    measures: Iterable[str],
    raw_filename: str,
    full_filename: str,
    *,
    only_eval: bool,
    cache_to_mem: bool,
) -> dict[str, Any]:
    # Round-trip through JSON to avoid sharing nested structures between tasks.
    config = json.loads(json.dumps(base))
    architecture_values = list(architectures)
    for experiment in config["experiments"]:
        filters = experiment.setdefault("filter_key_vals", {})
        filters["architecture"] = architecture_values
    config.pop("excluded_measures", None)
    config["included_measures"] = list(measures)
    config.update(
        cache_to_mem=cache_to_mem,
        cache_to_disk=False,
        # False: ReSi's ExperimentStorer.get_comp_result rebuilds a full
        # pd.concat([old_experiments, new_experiments]) on every call, and
        # abstract_experiment.compare_combos only reaches that call when
        # rerun_nans is True, even for comparisons it is about to skip. With
        # a large accumulated results parquet (the 100-point AUC sweep grew
        # this substantially) that made every already-computed comparison
        # cost tens of seconds to re-check instead of skipping instantly.
        # Trade-off: archived comparisons that are currently NaN stay NaN
        # until this is deliberately re-enabled for a recovery pass.
        rerun_nans=False,
        only_extract_reps=False,
        only_eval=bool(only_eval),
        raw_results_filename=raw_filename,
    )
    table = config.setdefault("table_creation", {})
    table.update(
        save_aggregated_df=False, save_full_df=True, full_df_filename=full_filename
    )
    return config


def _declared_groups(base: dict[str, Any]) -> dict[str, set[str]]:
    """Return the group values each experiment separates models by.

    Only grouping keys a comparison id can carry are returned: an id encodes the
    fields in ``MODEL_FIELDS`` and nothing else, so a campaign grouping on
    anything else cannot be checked against a parquet and is skipped rather than
    guessed at. Experiments without ``grouping_keys`` are skipped too --
    monotonicity and accoutput are not group-separation experiments.
    """
    declared: dict[str, set[str]] = {}
    for experiment in base.get("experiments") or []:
        filters = experiment.get("filter_key_vals") or {}
        for key in experiment.get("grouping_keys") or []:
            if key not in MODEL_FIELDS:
                continue
            values = filters.get(key)
            if not values:
                continue
            declared.setdefault(str(key), set()).update(str(item) for item in values)
    return declared


def _observed_groups(frame: pd.DataFrame, keys: Iterable[str]) -> dict[str, set[str]]:
    """Return the group values actually present among a result frame's models."""
    observed: dict[str, set[str]] = {str(key): set() for key in keys}
    if not observed or "id" not in frame.columns:
        return observed
    parsed = frame["id"].astype(str).map(parse_comparison_id)
    for key in observed:
        for prefix in ("source", "target"):
            field = f"{prefix}_{key}"
            observed[key].update(
                str(record[field])
                for record in parsed
                if record.get(field) not in (None, "")
            )
    return observed


def _check_baseline_groups(
    frame: pd.DataFrame, source: Path, base: dict[str, Any], base_path: Path
) -> None:
    """Fail when a baseline lacks a group the config scoring it declares.

    ReSi evaluates quality over the declared group set, so a baseline missing a
    whole group cannot produce a single cross-group comparison and every measure
    scores NaN. That is deterministic and knowable here, before any job is
    submitted, and it is exactly how the recreated CIFAR100 augmentation
    baselines were generated from a 3-group config while the campaign scored
    them against the 5-group one.
    """
    declared = _declared_groups(base)
    if not declared:
        return
    observed = _observed_groups(frame, declared)
    for key, expected in sorted(declared.items()):
        # Extra groups are fine: ReSi filters the archive down to what it wants.
        missing = sorted(expected - observed[key])
        if not missing:
            continue
        raise ValueError(
            f"Baseline covers fewer {key!r} groups than the config it is scored "
            f"against.\n"
            f"  baseline: {source}\n"
            f"    has: {', '.join(sorted(observed[key])) or '(none)'}\n"
            f"  config:   {base_path}\n"
            f"    needs: {', '.join(sorted(expected))}\n"
            f"  missing: {', '.join(missing)}"
        )


def _baseline_measures(
    frame: pd.DataFrame, source: Path, registered_measures: set[str]
) -> tuple[str, ...]:
    available = measures_present(frame)
    selected = available & registered_measures
    if not selected:
        raise ValueError(
            f"No registered representational measures found in baseline: {source}"
        )
    return tuple(sorted(selected))


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def _effective_campaign_config(
    campaign: Campaign, run_name: str, resi_root: Path, experiments_root: Path
) -> dict[str, Any]:
    """Describe the resolved run definition saved beside the manifests."""
    return {
        "campaign": str(campaign.path),
        "run_name": run_name,
        "domain": campaign.domain,
        "measures": list(campaign.measures),
        "quality_measures": list(campaign.quality_measures),
        "run_root": str(campaign.run_root),
        "analysis_root": str(campaign.analysis_root),
        "resi_dir": str(resi_root),
        "rep_sim": str(experiments_root),
        "legacy_signal_files": list(campaign.legacy_signal_files),
        "measure_sweeps": campaign.sweep_overrides(),
        "benchmarks": [
            {
                "id": benchmark.id,
                "dataset": benchmark.dataset,
                "base_config": benchmark.base_config,
                "architectures": list(benchmark.architectures),
                "cache_to_mem": benchmark.cache_to_mem,
                "baselines": [
                    {
                        "path": baseline.path,
                        "architectures": list(baseline.architectures),
                        "name": baseline.name,
                    }
                    for baseline in benchmark.baselines
                ],
            }
            for benchmark in campaign.benchmarks
        ],
        "known_missing_models": [
            {
                "benchmark": model.benchmark,
                "dataset": model.dataset,
                "architecture": model.architecture,
                "setting": model.setting,
                "train_dataset": model.train_dataset,
                "seed": model.seed,
                "reason": model.reason,
            }
            for model in campaign.known_missing_models
        ],
    }


def prepare_campaign(
    campaign_path: str | Path,
    run_name: str,
    *,
    resi_dir: str | Path | None = None,
    rep_sim: str | Path | None = None,
    registered_measures: set[str] | None = None,
) -> tuple[Path, Path]:
    """Generate ReSi configs and both task manifests for one campaign run."""
    campaign = load_campaign(campaign_path)
    safe_run_name = slug(run_name)
    resi_root = (
        expanded_path(resi_dir)
        if resi_dir is not None
        else require_env_path("RESI_DIR")
    )
    experiments_root = (
        expanded_path(rep_sim) if rep_sim is not None else require_env_path("REP_SIM")
    )
    if registered_measures is None:
        from ..runtime import registered_measure_names

        registered_measures = registered_measure_names(resi_root)

    run_dir = campaign.run_root / safe_run_name / campaign.domain
    sweep_overrides = campaign.sweep_overrides()
    compute_entries: list[ManifestEntry] = []
    baseline_entries: list[ManifestEntry] = []
    result_paths: set[str] = set()

    for benchmark in campaign.benchmarks:
        base_path = expanded_path(benchmark.base_config, root=resi_root)
        if not base_path.is_file():
            raise FileNotFoundError(f"ReSi base config not found: {base_path}")
        base = _read_config(base_path)
        for architecture in benchmark.architectures:
            relative_dir = (
                Path("manifold")
                / safe_run_name
                / campaign.domain
                / slug(benchmark.id)
                / slug(benchmark.dataset)
                / slug(architecture)
            )
            raw_relative = relative_dir / RESULT_FILENAME
            full_relative = relative_dir / FULL_CSV_FILENAME
            legacy_signal_relative = relative_dir / SIGNAL_FILENAME
            if str(raw_relative) in result_paths:
                raise ValueError(f"Generated result path collision: {raw_relative}")
            result_paths.add(str(raw_relative))
            config_path = (
                run_dir
                / "configs"
                / "compute"
                / slug(benchmark.id)
                / slug(benchmark.dataset)
                / f"{slug(architecture)}.yaml"
            )
            config = _filtered_config(
                base,
                [architecture],
                campaign.measures,
                str(raw_relative),
                str(full_relative),
                only_eval=False,
                cache_to_mem=benchmark.cache_to_mem,
            )
            _write_yaml(config_path, config)
            compute_entries.append(
                ManifestEntry(
                    kind="compute",
                    index=len(compute_entries),
                    domain=campaign.domain,
                    benchmark=benchmark.id,
                    dataset=benchmark.dataset,
                    architectures=(architecture,),
                    measures=campaign.measures,
                    config_path=str(config_path),
                    result_path=str(experiments_root / "results" / raw_relative),
                    full_csv_path=str(experiments_root / "results" / full_relative),
                    legacy_signal_paths=(
                        str(experiments_root / "results" / legacy_signal_relative),
                    ),
                    measure_sweeps=sweep_overrides,
                )
            )

        for baseline_number, baseline in enumerate(benchmark.baselines):
            source = expanded_path(baseline.path, root=experiments_root / "results")
            if not source.is_file():
                raise FileNotFoundError(f"Baseline parquet not found: {source}")
            # Read once: both the measure discovery and the group check below
            # need the same frame.
            baseline_frame = pd.read_parquet(source)
            _check_baseline_groups(baseline_frame, source, base, base_path)
            measures = tuple(
                measure
                for measure in _baseline_measures(
                    baseline_frame, source, registered_measures
                )
                if measure not in campaign.measures
            )
            if not measures:
                raise ValueError(
                    f"Baseline has no non-campaign representational measures: {source}"
                )
            baseline_name = slug(baseline.name or f"baseline-{baseline_number:03d}")
            relative_dir = (
                Path("manifold")
                / safe_run_name
                / "_baselines"
                / campaign.domain
                / slug(benchmark.id)
                / slug(benchmark.dataset)
                / baseline_name
            )
            raw_relative = relative_dir / RESULT_FILENAME
            full_relative = relative_dir / FULL_CSV_FILENAME
            if str(raw_relative) in result_paths:
                raise ValueError(f"Generated result path collision: {raw_relative}")
            result_paths.add(str(raw_relative))
            config_path = (
                run_dir
                / "configs"
                / "baseline"
                / slug(benchmark.id)
                / slug(benchmark.dataset)
                / f"{baseline_name}.yaml"
            )
            config = _filtered_config(
                base,
                baseline.architectures,
                measures,
                str(raw_relative),
                str(full_relative),
                only_eval=True,
                cache_to_mem=benchmark.cache_to_mem,
            )
            _write_yaml(config_path, config)
            baseline_entries.append(
                ManifestEntry(
                    kind="baseline",
                    index=len(baseline_entries),
                    domain=campaign.domain,
                    benchmark=benchmark.id,
                    dataset=benchmark.dataset,
                    architectures=baseline.architectures,
                    measures=measures,
                    config_path=str(config_path),
                    result_path=str(experiments_root / "results" / raw_relative),
                    full_csv_path=str(experiments_root / "results" / full_relative),
                    source_result_path=str(source),
                )
            )

    compute_manifest = run_dir / "compute_manifest.jsonl"
    baseline_manifest = run_dir / "baseline_manifest.jsonl"
    write_manifest(compute_manifest, compute_entries)
    write_manifest(baseline_manifest, baseline_entries)
    save_effective_config(
        run_dir,
        _effective_campaign_config(
            campaign, safe_run_name, resi_root, experiments_root
        ),
    )
    metadata = {
        "campaign": str(campaign.path),
        "run_name": safe_run_name,
        "domain": campaign.domain,
        "compute_tasks": len(compute_entries),
        "baseline_tasks": len(baseline_entries),
        "compute_manifest": str(compute_manifest),
        "baseline_manifest": str(baseline_manifest),
    }
    with (run_dir / "campaign.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return compute_manifest, baseline_manifest


__all__ = ["prepare_campaign", "result_metric_column"]
