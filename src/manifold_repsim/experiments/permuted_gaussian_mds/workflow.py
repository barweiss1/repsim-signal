"""Workflow orchestration for the synthetic permuted-Gaussian MDS experiment."""

from __future__ import annotations

from typing import Any

from .analysis import analyze_dataset
from .config import save_run_config
from .dataset import generate_dataset


def run(config: dict[str, Any]) -> None:
    """Execute configured stages in their validated order."""
    save_run_config(config)
    stages = {
        "generate": generate_dataset,
        "analyze": analyze_dataset,
    }
    for stage in config["stages"]:
        stages[stage](config)
