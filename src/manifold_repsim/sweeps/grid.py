"""Validated construction of fixed metric parameter sweep grids."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

import numpy as np

from manifold_repsim.metrics.registry import get_metric_spec
from manifold_repsim.metrics.registry import get_sweep_parameter_default

GRID_SCALES = frozenset({"linear", "log"})


@dataclass(frozen=True)
class SweepGridConfig:
    """One exact or generated sweep grid as represented in YAML.

    Exact grids define only ``values``. Generated grids define ``minimum``,
    ``maximum``, ``num``, and ``scale``. The two forms are mutually exclusive.
    """

    values: tuple[int | float, ...] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    num: int | None = None
    scale: str | None = None

    def __post_init__(self) -> None:
        exact = self.values is not None
        generated_values = (self.minimum, self.maximum, self.num, self.scale)
        generated = any(value is not None for value in generated_values)
        if exact and generated:
            raise ValueError(
                "Sweep grid 'values' cannot be combined with min, max, num, or scale."
            )
        if not exact and not all(value is not None for value in generated_values):
            raise ValueError(
                "A generated sweep grid requires min, max, num, and scale."
            )
        if exact:
            values = np.asarray(self.values, dtype=float)
            if values.ndim != 1 or values.size == 0:
                raise ValueError("Sweep grid values must be a non-empty 1D sequence.")
            if not np.isfinite(values).all():
                raise ValueError("Sweep grid values must be finite.")
            if values.size > 1 and np.any(np.diff(values) <= 0):
                raise ValueError("Sweep grid values must be strictly increasing.")
            return

        minimum = float(self.minimum)
        maximum = float(self.maximum)
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise ValueError("Sweep grid min and max must be finite.")
        if minimum >= maximum:
            raise ValueError("Sweep grid min must be smaller than max.")
        if isinstance(self.num, bool) or not isinstance(self.num, Integral):
            raise TypeError("Sweep grid num must be an integer.")
        if int(self.num) < 2:
            raise ValueError("Sweep grid num must be at least 2.")
        if self.scale not in GRID_SCALES:
            raise ValueError("Sweep grid scale must be 'linear' or 'log'.")
        if self.scale == "log" and minimum <= 0:
            raise ValueError("A log sweep grid requires a positive min.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SweepGridConfig":
        """Parse the public YAML shape using ``min`` and ``max`` keys."""
        if not isinstance(value, Mapping):
            raise TypeError("Sweep grid configuration must be a mapping.")
        allowed = {"values", "min", "max", "num", "scale"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown sweep grid fields: {sorted(unknown)}")
        values = value.get("values")
        return cls(
            values=None if values is None else tuple(values),
            minimum=value.get("min"),
            maximum=value.get("max"),
            num=value.get("num"),
            scale=value.get("scale"),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the canonical YAML-compatible request shape."""
        if self.values is not None:
            return {"values": list(self.values)}
        return {
            "min": self.minimum,
            "max": self.maximum,
            "num": self.num,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class ResolvedSweepGrid:
    """A validated fixed grid and the request that produced it."""

    metric_name: str
    parameter_name: str
    values: np.ndarray
    requested: SweepGridConfig
    source: str

    def effective_config(self) -> dict[str, Any]:
        """Return the fixed values that must be persisted for reproduction."""
        return {
            "metric": self.metric_name,
            "parameter": self.parameter_name,
            "source": self.source,
            "request": self.requested.to_mapping(),
            "values": self.values.tolist(),
        }


def _coerce_metric_values(
    parameter_name: str,
    values: np.ndarray,
    *,
    exact: bool,
) -> np.ndarray:
    """Coerce resolved grid values to the metric parameter's value domain.

    Integer-valued parameters reject non-integer *exact* values, because a
    caller listing them explicitly meant the values they wrote. Generated grids
    are rounded instead: a linear or log grid over an integer parameter lands on
    fractional points by construction, so rejecting those would leave integer
    parameters with no usable generated grid at all. Rounding can collide, so
    duplicates are removed and the grid may be shorter than ``num``.
    """
    if parameter_name != "topk":
        return values.astype(float)
    rounded = np.rint(values)
    if exact and not np.allclose(values, rounded):
        raise ValueError("Exact topk sweep values must be integers.")
    values = np.unique(rounded.astype(int))
    if np.any(values < 1):
        raise ValueError("topk sweep values must be positive.")
    return values


def resolve_metric_sweep_grid(
    metric_name: str,
    grid: SweepGridConfig | Mapping[str, Any] | None = None,
    *,
    default_num: int = 25,
    default_scale: str = "linear",
) -> ResolvedSweepGrid:
    """Resolve experiment grid settings before falling back to registry bounds."""
    spec = get_metric_spec(metric_name)
    if spec.sweep is None:
        raise ValueError(f"Metric {metric_name} does not define a sweepable parameter.")
    source = "experiment" if grid is not None else "registry"
    if grid is None:
        grid = SweepGridConfig(
            minimum=spec.sweep.minimum,
            maximum=spec.sweep.maximum,
            num=default_num,
            scale=default_scale,
        )
    elif isinstance(grid, Mapping):
        grid = SweepGridConfig.from_mapping(grid)
    elif not isinstance(grid, SweepGridConfig):
        raise TypeError("grid must be a SweepGridConfig, mapping, or None.")

    exact = grid.values is not None
    if exact:
        values = np.asarray(grid.values, dtype=float)
    elif grid.scale == "log":
        values = np.geomspace(grid.minimum, grid.maximum, num=grid.num)
    else:
        values = np.linspace(grid.minimum, grid.maximum, num=grid.num)
    values = _coerce_metric_values(spec.sweep.name, values, exact=exact)
    return ResolvedSweepGrid(
        metric_name=metric_name,
        parameter_name=spec.sweep.name,
        values=values,
        requested=grid,
        source=source,
    )


def map_param_name_to_kwargs(
    param_name: str | None,
    param_value: Any = None,
    local: bool = False,
) -> dict[str, Any]:
    """Map a sweep parameter to the metric implementation keyword."""
    if param_name is None:
        return {}
    supported = {"rbf_sigma", "quantile", "topk", "temperature", "diffusion_time"}
    if param_name not in supported:
        raise ValueError(f"Unsupported sweep parameter: {param_name}")
    value = (
        param_value
        if param_value is not None
        else get_sweep_parameter_default(param_name)
    )
    if param_name == "rbf_sigma" and local:
        value /= 5.0
    if param_name == "topk":
        value = int(value)
    return {param_name: value}


def get_param_sweep_for_metric(
    metric_name: str,
    sweep_len: int,
    sweep_values: Any = None,
    logscale: bool = False,
    sweep_config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Compatibility entry point backed by validated grid resolution."""
    spec = get_metric_spec(metric_name)
    if spec.sweep is None:
        raise ValueError(f"Metric {metric_name} does not define a sweepable parameter.")
    if sweep_values is not None:
        grid: SweepGridConfig | None = SweepGridConfig(values=tuple(sweep_values))
    elif sweep_config is not None:
        grid = SweepGridConfig(
            minimum=sweep_config["min"],
            maximum=sweep_config["max"],
            num=sweep_len,
            scale="log" if logscale else "linear",
        )
    else:
        grid = None
    return resolve_metric_sweep_grid(
        metric_name,
        grid,
        default_num=sweep_len,
        default_scale="log" if logscale else "linear",
    ).values
