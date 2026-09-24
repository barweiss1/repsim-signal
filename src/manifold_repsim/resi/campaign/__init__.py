"""ReSi campaign definitions, task manifests, and run preparation.

The schema and manifest modules are pure: they import neither Torch nor ReSi, so
campaign definitions can be validated and manifests inspected anywhere.
Preparation reaches ReSi only to discover registered baseline measures, and only
when a caller does not supply them.
"""

from __future__ import annotations

from .manifest import ManifestEntry, read_manifest, write_manifest
from .prepare import measures_present, prepare_campaign, result_metric_column
from .schema import (
    LOCAL_MANIFOLD_MEASURES,
    BaselineSpec,
    BenchmarkSpec,
    Campaign,
    KnownMissingModel,
    MeasureSweepSpec,
    load_campaign,
)

__all__ = [
    "BaselineSpec",
    "BenchmarkSpec",
    "Campaign",
    "KnownMissingModel",
    "LOCAL_MANIFOLD_MEASURES",
    "ManifestEntry",
    "MeasureSweepSpec",
    "load_campaign",
    "measures_present",
    "prepare_campaign",
    "read_manifest",
    "result_metric_column",
    "write_manifest",
]
