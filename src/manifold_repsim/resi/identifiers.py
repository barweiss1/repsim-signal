"""Parsing of ReSi comparison identifiers.

ReSi builds a comparison id by joining two model identifiers and a measure name
with ``___``, and builds each model identifier by joining its fields with
``__``. It provides a builder but no parser, so the grammar is reproduced here.
That makes this module a compatibility dependency on ReSi's id format: if the
upstream join order changes, these functions must change with it.

It sits beside the campaign and analysis packages rather than inside either,
because both read ids: campaign preparation checks a baseline's group coverage
against the config it will be scored with, and analysis resolves comparisons
back to the models they came from.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

MODEL_FIELDS = (
    "setting",
    "architecture",
    "train_dataset",
    "seed",
    "representation_dataset",
    "layer_id",
)
MODEL_SEPARATOR = "__"
COMPARISON_SEPARATOR = "___"


def _parse_model_token(token: str, prefix: str) -> dict[str, Any]:
    parts = token.split(MODEL_SEPARATOR)
    result = {
        f"{prefix}_{field}": parts[index] if index < len(parts) else ""
        for index, field in enumerate(MODEL_FIELDS)
    }
    for field in ("seed", "layer_id"):
        result[f"{prefix}_{field}"] = pd.to_numeric(
            result[f"{prefix}_{field}"], errors="coerce"
        )
    return result


def parse_comparison_id(comparison_id: str) -> dict[str, Any]:
    """Split a comparison id into prefixed source and target model fields.

    Returns an empty mapping for anything that is not a three-part id, so a
    malformed row is skipped rather than raising mid-analysis.
    """
    parts = str(comparison_id).rsplit(COMPARISON_SEPARATOR, 2)
    if len(parts) != 3:
        return {}
    result = _parse_model_token(parts[0], "source")
    result.update(_parse_model_token(parts[1], "target"))
    return result


def canonical_comparison_id(comparison_id: str, metric: str) -> str:
    """Order the two model tokens so A-vs-B and B-vs-A collapse to one id."""
    parts = str(comparison_id).rsplit(COMPARISON_SEPARATOR, 2)
    if len(parts) != 3:
        return str(comparison_id)
    first, second = sorted(parts[:2])
    return COMPARISON_SEPARATOR.join((first, second, str(metric or parts[2])))


__all__ = [
    "COMPARISON_SEPARATOR",
    "MODEL_FIELDS",
    "MODEL_SEPARATOR",
    "canonical_comparison_id",
    "parse_comparison_id",
]
