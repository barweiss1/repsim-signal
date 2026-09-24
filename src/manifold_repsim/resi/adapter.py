"""ReSi measure adapters for the manifold representation similarity metrics.

This module is importable both inside a ReSi checkout and from this repository
alone. When ReSi is unavailable a small local fallback keeps the adapter classes
smoke-testable with plain ``unittest``.

Responsibilities kept here are genuinely ReSi-specific: satisfying the measure
interface, flattening ReSi representation shapes, bounding cost by subsampling
rows, and recording the score curve behind each AUC value. Metric algorithms,
sweep-grid construction, and representation validation come from the shared
metric and sweep packages.
"""

from __future__ import annotations

import math
import os
from typing import Any, Mapping

import numpy as np
import torch

from manifold_repsim.metrics import prepare_metric_curve
from manifold_repsim.metrics import prepare_metric_pair
from manifold_repsim.metrics import score_prepared_curve
from manifold_repsim.metrics import score_prepared_metric
from manifold_repsim.sweeps.grid import resolve_metric_sweep_grid
from manifold_repsim.sweeps.signal_aggregation import integrate_metric_over_param

from .measures import MANIFOLD_RESI_MEASURE_CLASSES, RESI_MEASURE_SPECS, ResiMeasureSpec

try:  # pragma: no cover - exercised on the remote ReSi checkout.
    from repsim.measures.utils import RepresentationalSimilarityMeasure
except Exception:  # pragma: no cover - covered indirectly by local tests.

    class RepresentationalSimilarityMeasure:
        """Minimal fallback matching the ReSi constructor shape used below."""

        def __init__(self, sim_func=None, **kwargs):
            self.sim_func = sim_func
            for key, value in kwargs.items():
                setattr(self, key, value)


try:  # pragma: no cover - exercised on the remote ReSi checkout.
    from repsim.measures.utils import flatten as _resi_flatten
except Exception:  # pragma: no cover - covered indirectly by local tests.
    _resi_flatten = None


DEFAULT_AUC_SWEEP_LEN = 30
DEFAULT_AUC_INTEGRATION_METHOD = "average"
DEFAULT_AUC_LOGSCALE = True
DEFAULT_AUC_SUBSAMPLE_SEED = 0
SUPPORTED_RESI_SHAPES = frozenset({"nd", "ntd", "nchw"})
MINIMUM_TOPK_ROWS = 3
FIXED_TOPK = 10


def _as_nd(R: Any, Rp: Any, shape: str):
    """Flatten a ReSi representation pair to sample-by-feature matrices."""
    if _resi_flatten is not None:
        return _resi_flatten(R, Rp, shape=shape)

    if shape not in SUPPORTED_RESI_SHAPES:
        raise ValueError(f"Unsupported ReSi representation shape: {shape}")

    return _flatten_one(R), _flatten_one(Rp)


def _flatten_one(x: Any):
    """Collapse every dimension after the sample dimension into features."""
    if torch.is_tensor(x):
        if x.ndim < 2:
            raise ValueError(
                f"Expected at least 2 dimensions, got shape {tuple(x.shape)}"
            )
        return x.reshape(x.shape[0], -1)

    array = np.asarray(x)
    if array.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions, got shape {array.shape}")
    return array.reshape(array.shape[0], -1)


def _count_nonfinite(value: Any) -> int:
    """Count NaN and infinite entries for a diagnostic message."""
    if torch.is_tensor(value):
        return int((~torch.isfinite(value)).sum().item())
    return int(np.count_nonzero(~np.isfinite(np.asarray(value))))


def _validate_flat_pair(R: Any, Rp: Any, metric_name: str):
    """Check flattened shapes, reporting the ReSi measure and pre-flatten shapes.

    Dtype, finiteness, and device agreement are validated by the shared metric
    preparation path. Only the checks that need flattening context live here,
    because a shape complaint is far easier to act on when it names the measure
    and the shapes that produced it.
    """
    for label, value in (("source", R), ("target", Rp)):
        if len(value.shape) != 2:
            raise ValueError(
                f"{metric_name} expected a flattened 2D {label} representation, "
                f"got shape {tuple(value.shape)}"
            )
        if int(value.shape[0]) < 1 or int(value.shape[1]) < 1:
            raise ValueError(
                f"{metric_name} got empty {label} representation shape: "
                f"{tuple(value.shape)}"
            )
    if int(R.shape[0]) != int(Rp.shape[0]):
        raise ValueError(
            f"{metric_name} got mismatched sample counts after flattening: "
            f"{int(R.shape[0])} vs {int(Rp.shape[0])} "
            f"(shapes {tuple(R.shape)} vs {tuple(Rp.shape)})"
        )

    for label, value in (("source", R), ("target", Rp)):
        nonfinite = _count_nonfinite(value)
        if nonfinite:
            total = (
                int(value.numel())
                if torch.is_tensor(value)
                else int(np.asarray(value).size)
            )
            raise ValueError(
                f"{metric_name} got non-finite values in the {label} "
                f"representation after flattening: {nonfinite}/{total} values are "
                f"NaN or Inf, shape={tuple(value.shape)}"
            )
    return R, Rp


def _num_points(R: Any) -> int:
    return int(R.shape[0])


def _require_minimum_rows(label: str, n: int) -> None:
    """Raise if a comparison has too few rows for a row-count-dependent metric."""
    if n < MINIMUM_TOPK_ROWS:
        raise ValueError(
            f"{label} require at least {MINIMUM_TOPK_ROWS} representation rows."
        )


def _topk_at_most_10(R: Any) -> int:
    """Clamp the fixed neighbor count to what the row count can support."""
    n = _num_points(R)
    _require_minimum_rows("Top-k manifold metrics", n)
    return min(FIXED_TOPK, n - 1)


def _finite_float(value: Any, metric_name: str) -> float:
    """Return a scalar score, rejecting non-finite results."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{metric_name} returned a non-finite value: {result}")
    return result


def _env_int(
    name: str, default: int | None = None, minimum: int | None = None
) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value in {None, ""}:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {raw_value!r}"
        ) from exc
    if minimum is not None and value < minimum:
        raise ValueError(
            f"Environment variable {name} must be >= {minimum}, got {value}"
        )
    return value


def _auc_sweep_len() -> int:
    return int(
        _env_int("MANIFOLD_RESI_AUC_SWEEP_LEN", DEFAULT_AUC_SWEEP_LEN, minimum=1)
    )


def _max_points() -> int | None:
    return _env_int("MANIFOLD_RESI_MAX_POINTS", None, minimum=3)


def _auc_max_points() -> int | None:
    return _env_int("MANIFOLD_RESI_AUC_MAX_POINTS", _max_points(), minimum=3)


def _subsample_seed() -> int:
    return int(_env_int("MANIFOLD_RESI_SUBSAMPLE_SEED", DEFAULT_AUC_SUBSAMPLE_SEED))


def _auc_subsample_seed() -> int:
    return int(_env_int("MANIFOLD_RESI_AUC_SUBSAMPLE_SEED", _subsample_seed()))


def _take_rows(x: Any, indices: np.ndarray):
    if torch.is_tensor(x):
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=x.device)
        return x.index_select(0, index_tensor)
    return np.asarray(x)[indices]


def _subsample_pair(R: Any, Rp: Any, max_points: int | None, seed: int):
    """Draw one shared row subset so both representations stay aligned."""
    n = _num_points(R)
    if max_points is None or n <= max_points:
        return (
            R,
            Rp,
            {
                "num_points": n,
                "num_points_used": n,
                "max_points": max_points,
                "subsample_seed": None,
                "subsampled": False,
            },
        )

    rng = np.random.RandomState(seed)
    indices = np.sort(rng.choice(n, size=max_points, replace=False))
    return (
        _take_rows(R, indices),
        _take_rows(Rp, indices),
        {
            "num_points": n,
            "num_points_used": int(max_points),
            "max_points": int(max_points),
            "subsample_seed": int(seed),
            "subsampled": True,
        },
    )


def _resolve_device() -> torch.device | None:
    """Resolve the opt-in compute device for a comparison, or ``None`` to stay put.

    Kernel construction is memory-bandwidth bound, so a GPU is worth one to two
    orders of magnitude at benchmark row counts. It stays opt-in because moving
    devices perturbs the last ULP of a score, and this repository's default is to
    preserve exactly what a representation arrives as.
    """
    raw_value = os.environ.get("MANIFOLD_RESI_DEVICE")
    if raw_value in {None, ""}:
        return None
    device = torch.device(raw_value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"MANIFOLD_RESI_DEVICE={raw_value!r} requests CUDA, but no CUDA device "
            "is visible. Refusing to fall back to CPU silently, because a silent "
            "fallback would spend hours producing the result the setting meant "
            "to avoid."
        )
    return device


def _move_pair(R: Any, Rp: Any, device: torch.device | None):
    """Place both representations on the configured device, if one is configured."""
    if device is None:
        return R, Rp, {"device": None}
    source = torch.as_tensor(np.asarray(R) if not torch.is_tensor(R) else R)
    target = torch.as_tensor(np.asarray(Rp) if not torch.is_tensor(Rp) else Rp)
    return source.to(device), target.to(device), {"device": str(device)}


def _subsample_for_metric(R: Any, Rp: Any):
    return _subsample_pair(R, Rp, _max_points(), _subsample_seed())


def _subsample_for_auc(R: Any, Rp: Any):
    return _subsample_pair(R, Rp, _auc_max_points(), _auc_subsample_seed())


def _sweep_values(
    metric_name: str,
    R: Any,
    sweep_len: int = DEFAULT_AUC_SWEEP_LEN,
    logscale: bool = DEFAULT_AUC_LOGSCALE,
    grid: Mapping[str, Any] | None = None,
):
    """Resolve the AUC sweep grid, bounding neighbor counts by the row count.

    The grid comes from the shared resolver: an explicit campaign override when
    one is configured, otherwise the metric registry's default bounds. Only the
    ``topk`` clamp is applied here, because the number of available rows is a
    property of this comparison rather than of the grid.

    Returns the parameter name, its values, and the provenance recorded in the
    similarity signal so a stored curve says how its grid was chosen.
    """
    resolved = resolve_metric_sweep_grid(
        metric_name,
        grid,
        default_num=sweep_len,
        default_scale="log" if logscale else "linear",
    )
    param_name = resolved.parameter_name
    values = resolved.values
    if param_name == "topk":
        n = _num_points(R)
        _require_minimum_rows("KNN AUC manifold metrics", n)
        values = np.unique(np.clip(values, 2, n - 1))
    # The resolver's own vocabulary calls an explicit override "experiment";
    # stored signals have always said "campaign" for the same case, and
    # tests/test_resi_campaign.py pins that string, so the label is remapped
    # rather than passed through.
    source = "campaign" if resolved.source == "experiment" else resolved.source
    provenance = {
        "sweep_len_requested": int(resolved.requested.num),
        "logscale": resolved.requested.scale == "log",
        "sweep_source": source,
    }
    return param_name, values.tolist(), provenance


class _ManifoldMeasure(RepresentationalSimilarityMeasure):
    """Score one fixed-parameter manifold metric through ReSi's interface."""

    metric_name = None
    metric_kwargs = {}
    use_topk_10 = False

    def __init__(self):
        super().__init__(
            sim_func=self._score,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=False,
        )

    def _score(self, R, Rp, shape):
        R, Rp = _as_nd(R, Rp, shape)
        R, Rp = _validate_flat_pair(R, Rp, self.__class__.__name__)
        R, Rp, _ = _subsample_for_metric(R, Rp)
        R, Rp, _ = _move_pair(R, Rp, _resolve_device())
        kwargs = dict(self.metric_kwargs)
        if self.use_topk_10:
            kwargs["topk"] = _topk_at_most_10(R)
        with torch.no_grad():
            prepared = prepare_metric_pair(self.metric_name, R, Rp, **kwargs)
            return _finite_float(
                score_prepared_metric(prepared),
                self.__class__.__name__,
            )

    def __call__(self, R, Rp, shape) -> float:
        # Genuinely needed, not a duplicate of the inherited `__call__`: the
        # real ReSi base class's `__call__` is exactly this passthrough, but
        # the local fallback used when ReSi is not importable (this module's
        # own `RepresentationalSimilarityMeasure` above) defines no `__call__`
        # at all -- it only matches the constructor shape. Without this
        # override, every measure instance is uncallable under that fallback,
        # which is what every unit test that never imports ReSi runs under.
        return self._score(R, Rp, shape)


class _ManifoldAUCMeasure(_ManifoldMeasure):
    """Sweep one metric parameter and reduce the score curve to a scalar.

    The full curve is retained on ``last_similarity_signal`` so the campaign
    runtime can embed it in the result row; a scalar alone would discard the
    shape that motivates these measures.
    """

    integration_method = DEFAULT_AUC_INTEGRATION_METHOD
    logscale = DEFAULT_AUC_LOGSCALE
    # Set per instance by campaign registration to override the registry grid.
    sweep_grid: Mapping[str, Any] | None = None

    def __init__(self):
        self.last_similarity_signal = None
        super().__init__()

    def _score(self, R, Rp, shape):
        self.last_similarity_signal = None
        R, Rp = _as_nd(R, Rp, shape)
        R, Rp = _validate_flat_pair(R, Rp, self.__class__.__name__)
        R, Rp, sampling_info = _subsample_for_auc(R, Rp)
        R, Rp, device_info = _move_pair(R, Rp, _resolve_device())
        param_name, param_values, sweep_info = _sweep_values(
            self.metric_name,
            R,
            _auc_sweep_len(),
            logscale=self.logscale,
            grid=self.sweep_grid,
        )
        with torch.no_grad():
            prepared = prepare_metric_curve(
                self.metric_name,
                R,
                Rp,
                param_name,
                param_values,
                **self.metric_kwargs,
            )
            scores = [
                _finite_float(score, self.__class__.__name__)
                for score in score_prepared_curve(prepared)
            ]

        auc_value = _finite_float(
            integrate_metric_over_param(param_values, scores, self.integration_method),
            self.__class__.__name__,
        )
        self.last_similarity_signal = {
            "metric": self.__class__.__name__,
            "base_metric": self.metric_name,
            "param_name": param_name,
            "param_values": [
                int(v) if param_name == "topk" else float(v) for v in param_values
            ],
            "scores": [float(v) for v in scores],
            "auc_value": auc_value,
            "integration_method": self.integration_method,
            "max_sweep_param": max(param_values) if param_values else None,
            "num_unique_params": len(param_values),
            **sweep_info,
            **sampling_info,
            **device_info,
        }
        return auc_value


def _build_measure_class(spec: ResiMeasureSpec) -> type:
    """Create the ReSi measure class described by one catalogue entry."""
    base = _ManifoldAUCMeasure if spec.is_auc else _ManifoldMeasure
    return type(
        spec.class_name,
        (base,),
        {
            "__doc__": (
                f"ReSi measure scoring {spec.metric_name} "
                f"({'parameter sweep AUC' if spec.is_auc else 'fixed parameters'})."
            ),
            "__module__": __name__,
            "metric_name": spec.metric_name,
            "metric_kwargs": spec.kwargs(),
            "use_topk_10": spec.clamp_topk,
        },
    )


# ReSi looks measures up by class name, so every catalogue entry becomes a
# module-level class. Generating them keeps the catalogue authoritative.
for _spec in RESI_MEASURE_SPECS:
    globals()[_spec.class_name] = _build_measure_class(_spec)
del _spec


__all__ = MANIFOLD_RESI_MEASURE_CLASSES + ["MANIFOLD_RESI_MEASURE_CLASSES"]
