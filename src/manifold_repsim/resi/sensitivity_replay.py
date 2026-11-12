"""Re-score ReSi's quality measures for signal variants without recomputing anything.

For one finished compute task this builds a scratch copy of its result store that
also holds each variant's comparison rows (derived from the stored curves), then
asks ReSi to evaluate only those variants with ``only_eval``. ReSi's own
group-separation code therefore produces the violation rate and AUPRC, with no
representation extraction and no re-implementation of its definitions. The
original result store and full CSV are never written.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .paths import expanded_path, require_env_path
from .runtime import _entry_at, _load_registry, register_manifold_measures
from .signal_sensitivity import (
    SCRATCH_ROOT,
    SensitivityConfig,
    derive_variant_rows,
    scratch_paths,
)


def _register_variant_measures(registry_module: Any, config: SensitivityConfig) -> None:
    """Add placeholder ReSi measures named after each variant.

    ReSi only needs their name and metadata to read the pre-filled rows; with
    ``only_eval`` they are never asked to score anything, and doing so is an
    error rather than a silent zero.
    """
    adapter = importlib.import_module("manifold_repsim.resi.adapter")

    def _never_scored(self, R, Rp, shape):  # noqa: ANN001
        raise RuntimeError(
            f"{type(self).__name__} is a replay placeholder and cannot score data"
        )

    for variant in config.variants:
        cls = type(
            variant.metric_name,
            (adapter._ManifoldMeasure,),
            {"_score": _never_scored, "__module__": __name__},
        )
        registry_module.ALL_MEASURES[variant.metric_name] = cls()


def replay_task(
    manifest: str | Path,
    index: int,
    config: SensitivityConfig,
    *,
    resi_dir: str | Path | None = None,
) -> Path:
    """Evaluate every variant for one finished compute task; return the CSV."""
    entry = _entry_at(manifest, index)
    if entry.kind != "compute":
        raise ValueError("Signal sensitivity replays compute tasks only")
    root = (
        expanded_path(resi_dir)
        if resi_dir is not None
        else require_env_path("RESI_DIR")
    )
    original = Path(entry.result_path)
    if not original.is_file():
        raise FileNotFoundError(f"Result store not found: {original}")
    task_config = yaml.safe_load(Path(entry.config_path).read_text())
    store_path, csv_path, results_root = scratch_paths(
        entry, task_config, config.scratch_name
    )

    store = pd.read_parquet(original)
    derived = derive_variant_rows(
        store,
        config.variants,
        base_metric=config.base_metric,
        mask_metric=config.field_measure,
    )
    store_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([store, derived]).to_parquet(store_path)

    replay_config = dict(task_config)
    replay_config["raw_results_filename"] = str(store_path.relative_to(results_root))
    replay_config["table_creation"] = {
        **task_config["table_creation"],
        "full_df_filename": str(csv_path.relative_to(results_root)),
    }
    replay_config["only_eval"] = True
    replay_config["included_measures"] = [v.metric_name for v in config.variants]
    replay_config.pop("excluded_measures", None)
    config_path = (
        results_root
        / SCRATCH_ROOT
        / config.scratch_name
        / "configs"
        / f"task_{entry.index:04d}.yaml"
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(replay_config, sort_keys=False))

    registry_module = _load_registry(root)
    register_manifold_measures(root, entry.measure_sweeps)
    _register_variant_measures(registry_module, config)
    run_module = importlib.import_module("repsim.run")
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        run_module.run(str(config_path))
    finally:
        os.chdir(previous_cwd)
    if not csv_path.is_file():
        raise RuntimeError(f"ReSi did not write the variant CSV: {csv_path}")
    return csv_path


__all__ = ["replay_task"]
