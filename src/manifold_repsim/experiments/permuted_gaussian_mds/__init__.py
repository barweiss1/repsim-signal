"""Synthetic permuted-Gaussian MDS experiment public API.

Exports are resolved lazily so configuration, dataset, similarity, and MDS
consumers do not import Matplotlib unless they request plotting or workflow
functionality.
"""

from __future__ import annotations

from importlib import import_module

_EXPORT_MODULES = {
    "analysis_fingerprint": "config",
    "analyze_dataset": "analysis",
    "analyze_signals": "analysis",
    "compatible_analysis_fingerprints": "config",
    "compatible_dataset_fingerprints": "config",
    "compute_fixed_similarity_matrix": "similarity",
    "compute_signal_similarity_matrices": "similarity",
    "config_fingerprint": "config",
    "dataset_fingerprint": "config",
    "direct_similarity_distance": "similarity",
    "effective_config": "config",
    "fit_metric_mds": "mds",
    "generate_dataset": "dataset",
    "grid_parameter_values": "dataset",
    "load_config": "config",
    "load_dataset_artifacts": "dataset",
    "main": "cli",
    "output_paths": "config",
    "parameter_value_label": "plotting",
    "result_path": "analysis",
    "reusable_result": "analysis",
    "run": "workflow",
    "save_metric_result": "analysis",
    "save_run_config": "config",
    "scalar_base_relative_distance": "similarity",
    "signal_base_relative_distance": "similarity",
    "validate_config": "config",
    "validate_dissimilarity_matrix": "similarity",
    "visual_encodings": "dataset",
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str):
    """Resolve one explicitly catalogued public export on first use."""
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError:
        raise AttributeError(name) from None
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
