"""Single source of truth for supported representation-similarity metrics."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class SweepParameterSpec:
    """Parameter metadata used to construct a metric sweep."""

    name: str
    minimum: int | float
    maximum: int | float
    default: int | float

    def compatibility_dict(self) -> dict[str, int | float | str]:
        """Return the dictionary shape exposed by the legacy interface."""
        return {"param": self.name, "min": self.minimum, "max": self.maximum}


@dataclass(frozen=True)
class MetricSpec:
    """Dispatch and sweep capabilities for one public metric name."""

    name: str
    implementation: str
    sweep: SweepParameterSpec | None = None
    dispatch_kwargs: tuple[tuple[str, Any], ...] = ()
    reuses_kernel: bool = False
    prepared_family: str | None = None

    def resolved_kwargs(self, kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Apply registry-owned dispatch arguments with compatibility precedence."""
        resolved = dict(kwargs)
        resolved.update(self.dispatch_kwargs)
        return resolved

    def compatibility_sweep_dict(self) -> dict[str, int | float | str | None]:
        """Return the dictionary shape exposed by ``AlignmentMetrics``."""
        if self.sweep is None:
            return {"param": None, "min": None, "max": None}
        return self.sweep.compatibility_dict()


def _sweep(
    name: str,
    minimum: int | float,
    maximum: int | float,
    default: int | float,
) -> SweepParameterSpec:
    return SweepParameterSpec(name, minimum, maximum, default)


TOPK = _sweep("topk", 3, 500, 10)
RBF_SIGMA = _sweep("rbf_sigma", 0.1, 3.0, 1.0)
QUANTILE = _sweep("quantile", 0.01, 0.7, 0.1)
TEMPERATURE = _sweep("temperature", 1e-3, 10.0, 0.5)
DIFFUSION_TIME = _sweep("diffusion_time", 0.01, 1e3, 1.0)


METRIC_SPECS = (
    MetricSpec("cycle_knn", "cycle_knn", sweep=TOPK, prepared_family="neighbors"),
    MetricSpec("mutual_knn", "mutual_knn", sweep=TOPK, prepared_family="neighbors"),
    MetricSpec(
        "mutual_knn_dist",
        "mutual_knn",
        sweep=TOPK,
        dispatch_kwargs=(("use_distance", True),),
        prepared_family="neighbors",
    ),
    MetricSpec("cka", "cka", reuses_kernel=True, prepared_family="linear_kernel"),
    MetricSpec(
        "cka_rbf",
        "cka",
        sweep=RBF_SIGMA,
        dispatch_kwargs=(("kernel_metric", "rbf"),),
        reuses_kernel=True,
        prepared_family="rbf_distances",
    ),
    MetricSpec(
        "cka_rbf_quantile",
        "cka",
        sweep=QUANTILE,
        dispatch_kwargs=(("kernel_metric", "rbf"),),
        reuses_kernel=True,
        prepared_family="rbf_distances",
    ),
    MetricSpec(
        "rbf_uka",
        "rbf_uka",
        sweep=RBF_SIGMA,
        reuses_kernel=True,
        prepared_family="rbf_distances",
    ),
    MetricSpec("unbiased_cka", "unbiased_cka"),
    MetricSpec("cknna", "cknna", sweep=TOPK, prepared_family="cknna"),
    MetricSpec("svcca", "svcca"),
    MetricSpec(
        "softmax_rwka",
        "softmax_rwka",
        sweep=TEMPERATURE,
        reuses_kernel=True,
        prepared_family="softmax_gram",
    ),
    MetricSpec(
        "rbf_rwka",
        "rbf_rwka",
        sweep=RBF_SIGMA,
        reuses_kernel=True,
        prepared_family="rbf_distances",
    ),
    MetricSpec(
        "rbf_rwka_symmetric",
        "rbf_rwka_symmetric",
        sweep=RBF_SIGMA,
        reuses_kernel=True,
        prepared_family="rbf_distances",
    ),
    MetricSpec(
        "rbf_crwka",
        "rbf_crwka",
        sweep=RBF_SIGMA,
        reuses_kernel=True,
        prepared_family="rbf_distances",
    ),
    MetricSpec(
        "rbf_degree_crwka",
        "rbf_degree_crwka",
        sweep=RBF_SIGMA,
        reuses_kernel=True,
        prepared_family="rbf_distances",
    ),
    MetricSpec(
        "rbf_rwka_quantile",
        "rbf_rwka",
        sweep=QUANTILE,
        dispatch_kwargs=(("rbf_sigma", 1.0),),
        reuses_kernel=True,
        prepared_family="rbf_distances",
    ),
    MetricSpec(
        "rbf_rwka_diffusion_time",
        "rbf_rwka_diffusion_time",
        sweep=DIFFUSION_TIME,
        reuses_kernel=True,
        prepared_family="diffusion_cache",
    ),
)


def _build_registry(specs: tuple[MetricSpec, ...]) -> Mapping[str, MetricSpec]:
    registry: dict[str, MetricSpec] = {}
    for spec in specs:
        if spec.name in registry:
            raise ValueError(f"Duplicate metric registration: {spec.name}")
        registry[spec.name] = spec
    return MappingProxyType(registry)


METRIC_REGISTRY = _build_registry(METRIC_SPECS)


def _build_parameter_defaults(
    specs: tuple[MetricSpec, ...],
) -> Mapping[str, int | float]:
    defaults: dict[str, int | float] = {}
    for spec in specs:
        if spec.sweep is None:
            continue
        previous = defaults.setdefault(spec.sweep.name, spec.sweep.default)
        if previous != spec.sweep.default:
            raise ValueError(
                f"Conflicting defaults for sweep parameter {spec.sweep.name}: "
                f"{previous} and {spec.sweep.default}"
            )
    return MappingProxyType(defaults)


SWEEP_PARAMETER_DEFAULTS = _build_parameter_defaults(METRIC_SPECS)


def get_metric_spec(name: str) -> MetricSpec:
    """Return a registered metric specification or raise a public API error."""
    try:
        return METRIC_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unrecognized metric: {name}") from None


def get_sweep_parameter_default(name: str) -> int | float:
    """Return the shared default value for a sweep parameter name."""
    try:
        return SWEEP_PARAMETER_DEFAULTS[name]
    except KeyError:
        raise ValueError(f"Unsupported sweep parameter: {name}") from None
