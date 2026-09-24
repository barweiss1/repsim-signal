"""Runtime integration with an external ReSi checkout.

Every ReSi import happens lazily inside these functions so that campaign
definitions, manifests, and status reports stay usable without a checkout. The
checkout itself is never modified: measures are registered into ReSi's in-memory
registry, and the only patch installed is the narrow result hook that carries
similarity signals into the canonical result row.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Mapping

from .campaign import ManifestEntry, read_manifest
from .paths import REPO_ROOT, expanded_path, require_env_path


def _add_import_roots(resi_dir: Path) -> None:
    for path in (REPO_ROOT, REPO_ROOT / "src", resi_dir):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _install_torcheval_fallback() -> None:
    try:
        importlib.import_module("torcheval.metrics.functional")
        return
    except (ImportError, ModuleNotFoundError):
        pass

    import torch

    def multiclass_accuracy(
        input: Any, target: Any, num_classes: int | None = None, **_: Any
    ):
        del num_classes
        predictions = (
            input.argmax(dim=-1)
            if getattr(input, "ndim", 1) > getattr(target, "ndim", 1)
            else input
        )
        predictions = predictions.reshape(-1)
        labels = target.reshape(-1)
        if labels.numel() == 0:
            return torch.tensor(float("nan"), device=labels.device)
        return (predictions == labels).to(dtype=torch.float32).mean()

    functional = types.ModuleType("torcheval.metrics.functional")
    functional.multiclass_accuracy = multiclass_accuracy
    metrics = types.ModuleType("torcheval.metrics")
    metrics.functional = functional
    package = types.ModuleType("torcheval")
    package.metrics = metrics
    sys.modules.setdefault("torcheval", package)
    sys.modules.setdefault("torcheval.metrics", metrics)
    sys.modules.setdefault("torcheval.metrics.functional", functional)


def _load_registry(resi_dir: Path):
    _add_import_roots(resi_dir)
    _install_torcheval_fallback()
    return importlib.import_module("repsim.measures")


def register_manifold_measures(
    resi_dir: str | Path,
    measure_sweeps: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add the manifold measures to ReSi's in-memory registry.

    Nothing is written to the ReSi checkout. When ``measure_sweeps`` is given,
    each named AUC measure receives a campaign-local sweep grid instead of the
    metric registry defaults.
    """
    root = Path(resi_dir).expanduser().resolve()
    registry_module = _load_registry(root)
    # `adapter` may already be imported from an earlier call with no ReSi
    # checkout on sys.path, in which case its measure classes were built on
    # the local fallback base class. Reload it now that the checkout's own
    # `repsim.measures.utils` is importable, so the classes ReSi registers are
    # real subclasses of its base rather than the fallback's.
    adapter = importlib.import_module("manifold_repsim.resi.adapter")
    try:
        measure_utils = importlib.import_module("repsim.measures.utils")
        first_class = getattr(adapter, adapter.MANIFOLD_RESI_MEASURE_CLASSES[0])
        if not issubclass(first_class, measure_utils.RepresentationalSimilarityMeasure):
            adapter = importlib.reload(adapter)
    except (ImportError, ModuleNotFoundError):
        # Minimal test registries need not provide ReSi's class hierarchy.
        pass
    overrides = dict(measure_sweeps or {})
    unknown = sorted(set(overrides) - set(adapter.MANIFOLD_RESI_MEASURE_CLASSES))
    if unknown:
        raise ValueError(
            f"Sweep override names no registered measure: {', '.join(unknown)}"
        )
    registered: dict[str, Any] = {}
    for class_name in adapter.MANIFOLD_RESI_MEASURE_CLASSES:
        instance = getattr(adapter, class_name)()
        grid = overrides.get(class_name)
        if grid is not None:
            if not hasattr(instance, "sweep_grid"):
                raise ValueError(
                    f"{class_name} does not sweep a parameter and cannot take a "
                    "sweep override"
                )
            instance.sweep_grid = dict(grid)
        registry_module.ALL_MEASURES[class_name] = instance
        registered[class_name] = instance
    return registered


def registered_measure_names(resi_dir: str | Path) -> set[str]:
    """Return every measure name ReSi knows about, before local registration."""
    registry_module = _load_registry(Path(resi_dir).expanduser().resolve())
    return set(registry_module.ALL_MEASURES)


def remap_missing_nlp_model_paths(
    models: list[Any],
    *,
    rep_sim: str | Path | None = None,
) -> dict[str, str]:
    """Use the flat archived BERT layout when ReSi's nested path is absent."""
    root_value = rep_sim if rep_sim is not None else os.environ.get("REP_SIM")
    if not root_value:
        return {}
    nlp_root = Path(root_value).expanduser() / "models" / "nlp"
    expected_bert_root = nlp_root / "bert"
    remapped: dict[str, str] = {}
    for model in models:
        if str(getattr(model, "domain", "")).upper() != "NLP":
            continue
        raw_path = getattr(model, "path", None)
        if not raw_path:
            continue
        source = Path(raw_path).expanduser()
        if source.exists():
            continue
        try:
            relative = source.relative_to(expected_bert_root)
        except ValueError:
            continue
        alternate = nlp_root / relative
        if not alternate.exists():
            continue
        model.path = str(alternate)
        remapped[str(source)] = str(alternate)
    return remapped


def _signal_json(metric: Any, metric_value: Any) -> str | None:
    signal = getattr(metric, "last_similarity_signal", None)
    if not isinstance(signal, dict):
        return None
    try:
        finite = math.isfinite(float(metric_value))
    except (TypeError, ValueError):
        finite = False
    if not finite:
        return None
    return json.dumps(signal, sort_keys=True, separators=(",", ":"))


def install_signal_storage_hook() -> None:
    """Patch ReSi's result storer in memory to carry the similarity signal."""
    utils = importlib.import_module("repsim.benchmark.utils")
    storer_class = utils.ExperimentStorer
    if getattr(storer_class, "_manifold_signal_hook", False):
        return
    original = storer_class.add_results

    def add_results_with_signal(
        self,
        src_single_rep,
        tgt_single_rep,
        metric,
        metric_value,
        runtime=None,
        overwrite=False,
    ):
        signal = _signal_json(metric, metric_value)
        original(
            self,
            src_single_rep,
            tgt_single_rep,
            metric,
            metric_value,
            runtime=runtime,
            overwrite=overwrite,
        )
        if signal is None or self._new_experiments.empty:
            return
        comparison_ids = [
            self._get_comparison_id(src_single_rep, tgt_single_rep, metric.name),
        ]
        if getattr(metric, "is_symmetric", False):
            comparison_ids.append(
                self._get_comparison_id(tgt_single_rep, src_single_rep, metric.name)
            )
        canonical_id = min(comparison_ids)
        if canonical_id in self._new_experiments.index:
            self._new_experiments.loc[canonical_id, "similarity_signal"] = signal

    storer_class.add_results = add_results_with_signal
    storer_class._manifold_signal_hook = True
    storer_class._manifold_original_add_results = original


def _entry_at(manifest: str | Path, index: int) -> ManifestEntry:
    entries = read_manifest(manifest)
    if index < 0 or index >= len(entries):
        raise IndexError(f"Manifest index {index} is out of range [0, {len(entries)})")
    return entries[index]


def run_task(
    manifest: str | Path, index: int, *, resi_dir: str | Path | None = None
) -> ManifestEntry:
    """Execute one manifest task through ReSi and verify its outputs exist."""
    entry = _entry_at(manifest, index)
    root = (
        expanded_path(resi_dir)
        if resi_dir is not None
        else require_env_path("RESI_DIR")
    )

    result_path = Path(entry.result_path)
    full_csv_path = Path(entry.full_csv_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    full_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if entry.kind == "baseline":
        if entry.source_result_path is None:
            raise ValueError("Baseline manifest entry has no source_result_path")
        source = Path(entry.source_result_path)
        if not source.is_file():
            raise FileNotFoundError(f"Baseline source not found: {source}")
        if not result_path.exists():
            shutil.copy2(source, result_path)
    elif entry.kind != "compute":
        raise ValueError(f"Unsupported manifest task kind: {entry.kind}")

    if entry.kind == "compute":
        register_manifold_measures(root, entry.measure_sweeps)
        install_signal_storage_hook()
    else:
        # Baseline tasks use only native measures already named in the copied
        # parquet. Loading the registry is enough to install the optional
        # torcheval fallback without instantiating any campaign metric.
        _load_registry(root)
    run_module = importlib.import_module("repsim.run")
    if entry.kind == "compute":
        remapped = remap_missing_nlp_model_paths(
            run_module.ALL_TRAINED_MODELS,
            rep_sim=os.environ.get("REP_SIM"),
        )
        if remapped:
            print(
                f"Remapped {len(remapped)} missing ReSi BERT model paths "
                "to the flat models/nlp archive layout."
            )
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        run_module.run(entry.config_path)
    finally:
        os.chdir(previous_cwd)
    if not result_path.is_file():
        raise RuntimeError(
            f"ReSi did not create the expected result parquet: {result_path}"
        )
    if not full_csv_path.is_file():
        raise RuntimeError(
            f"ReSi did not create the expected full CSV: {full_csv_path}"
        )
    return entry


__all__ = [
    "install_signal_storage_hook",
    "register_manifold_measures",
    "registered_measure_names",
    "remap_missing_nlp_model_paths",
    "run_task",
]
