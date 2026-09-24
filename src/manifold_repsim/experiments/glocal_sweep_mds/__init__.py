"""YAML-driven MDS and PCA analysis of none-to-glocal metric signals."""

from .config import load_config, validate_config
from .cli import main
from .discovery import (
    Condition,
    DatasetGroup,
    GroupRef,
    discover,
    parse_glocal_transform,
)
from .scoring import concatenate_batch_scores, fit_signal_pca, score_distances
from .workflow import load_computed_results, run

__all__ = [
    "Condition",
    "DatasetGroup",
    "GroupRef",
    "concatenate_batch_scores",
    "discover",
    "fit_signal_pca",
    "load_computed_results",
    "load_config",
    "main",
    "parse_glocal_transform",
    "run",
    "score_distances",
    "validate_config",
]
