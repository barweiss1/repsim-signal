"""Model- and dataset-local orchestration for glocal MDS artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tqdm import tqdm

from ..artifacts import save_csv, save_json
from .config import output_paths
from .discovery import DatasetGroup
from .scoring import analyze_lambda_splits, analyze_metric


def group_data_dir(data_dir: Path, group: DatasetGroup) -> Path:
    """Return the output directory for one independent model/dataset analysis."""
    return data_dir / group.model / group.dataset


def analyze(
    config: dict[str, Any], groups: tuple[DatasetGroup, ...]
) -> dict[tuple[str, str], list[Path]]:
    """Compute all configured metric artifacts without importing plotting."""
    data_dir, _ = output_paths(config)
    results: dict[tuple[str, str], list[Path]] = {}
    manifest_groups = []
    with tqdm(
        total=len(groups) * len(config["metrics"]),
        desc="Glocal metric analyses",
        unit="metric",
        disable=None,
    ) as progress:
        for group in groups:
            output_dir = group_data_dir(data_dir, group)
            save_csv(
                output_dir / "conditions.csv",
                [condition.record() for condition in group.conditions],
                ["name", "lambda", "alpha", "tau", "is_none"],
            )
            paths = []
            split_paths = []
            for metric in config["metrics"]:
                progress.set_postfix_str(
                    f"{group.model}/{group.dataset}/{metric['id']}", refresh=False
                )
                path = analyze_metric(config, group, metric, data_dir)
                paths.append(path)
                split_paths.extend(
                    analyze_lambda_splits(config, group, metric, data_dir, path)
                )
                progress.update(1)
            results[(group.model, group.dataset)] = paths
            manifest_groups.append(
                {
                    "model": group.model,
                    "dataset": group.dataset,
                    "source_fingerprint": group.source_fingerprint,
                    "batch_indices": list(group.batch_indices),
                    "conditions": [
                        condition.record() for condition in group.conditions
                    ],
                    "metric_results": [
                        str(path.relative_to(data_dir)) for path in paths
                    ],
                    "lambda_split_results": [
                        str(path.relative_to(data_dir)) for path in split_paths
                    ],
                }
            )
    save_json(
        data_dir / "analysis_manifest.json",
        {"version": 1, "groups": manifest_groups},
    )
    return results
