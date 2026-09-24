"""Top-level glocal sweep MDS workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from manifold_repsim.config import load_yaml_config

from .analysis import analyze
from ..artifacts import load_json
from .config import output_paths, save_run_config, validate_config
from .discovery import GroupRef, discover


def load_computed_results(
    config: dict[str, Any]
) -> tuple[list[GroupRef], dict[tuple[str, str], list[Path]]]:
    """Reconstruct groups and metric result paths from a saved manifest.

    Reads ``analysis_manifest.json`` written by :func:`analysis.analyze`
    instead of rescanning ``input_root``, so a ``stages: [plot]``-only run
    never touches the (potentially slow or unmounted) feature directory.
    """
    data_dir, _ = output_paths(config)
    manifest_path = data_dir / "analysis_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No analysis manifest at {manifest_path}; run with stages "
            "including 'compute' first"
        )
    manifest = load_json(manifest_path)
    groups = []
    results: dict[tuple[str, str], list[Path]] = {}
    for entry in manifest["groups"]:
        ref = GroupRef(model=entry["model"], dataset=entry["dataset"])
        groups.append(ref)
        results[(ref.model, ref.dataset)] = [
            data_dir / relative_path for relative_path in entry["metric_results"]
        ]
    return groups, results


def run(config: dict[str, Any]) -> dict[tuple[str, str], list[Path]]:
    """Execute the configured compute and/or plot stages."""
    stages = set(config["stages"])
    if "compute" in stages:
        effective, groups = discover(config)
        save_run_config(effective)
        results = analyze(effective, groups)
    else:
        data_dir, _ = output_paths(config)
        saved_path = data_dir / "config.yaml"
        if not saved_path.is_file():
            raise FileNotFoundError(
                f"No computed results at {data_dir}; run with stages "
                "including 'compute' first"
            )
        effective = validate_config(load_yaml_config(saved_path))
        groups, results = load_computed_results(effective)

    if "plot" in stages:
        from .plotting import plot_group

        for group in groups:
            plot_group(effective, group, results[(group.model, group.dataset)])
    return results
