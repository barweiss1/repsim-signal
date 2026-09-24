"""Shared input validation, numerical safeguards, and small feature utilities."""

from __future__ import annotations

import math
import warnings
from numbers import Integral
from typing import Any

import numpy as np
import torch

NORMALIZATION_EPS = 1e-10

# Pairwise distances below this fraction of a representation's observed spread
# carry no geometry. Two sources put a floor under them: ``torch.cdist`` uses the
# ``|x|^2 + |y|^2 - 2<x,y>`` expansion at benchmark sizes, which loses about half
# the mantissa for near-identical rows, and representations stored as float32
# cannot separate rows by less than one quantization step. Both scale with the
# data, so an absolute constant cannot classify them.
DISTANCE_RELATIVE_EPS = math.sqrt(np.finfo(np.float64).eps)


def to_torch_tensor(feats: Any, name: str = "representation") -> torch.Tensor:
    """Convert array-like real values to a floating Torch tensor."""
    if torch.is_tensor(feats):
        tensor = feats
    else:
        tensor = torch.from_numpy(np.asarray(feats))
    if torch.is_complex(tensor):
        raise TypeError(f"{name} must contain real-valued features.")
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.get_default_dtype())
    return tensor


def validate_representation(
    value: Any,
    name: str = "representation",
    min_samples: int = 2,
) -> torch.Tensor:
    """Validate one finite, nonempty sample-by-feature representation."""
    tensor = to_torch_tensor(value, name=name)
    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D sample-by-feature array, got shape "
            f"{tuple(tensor.shape)}."
        )
    if tensor.shape[0] == 0 or tensor.shape[1] == 0:
        raise ValueError(f"{name} must not be empty, got shape {tuple(tensor.shape)}.")
    if tensor.shape[0] < min_samples:
        raise ValueError(
            f"{name} requires at least {min_samples} samples, got "
            f"{tensor.shape[0]}."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def validate_representation_pair(
    feats_a: Any,
    feats_b: Any,
    metric_name: str,
    min_samples: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate paired representations while allowing different feature widths."""
    feats_a = validate_representation(
        feats_a,
        name=f"{metric_name} source representation",
        min_samples=min_samples,
    )
    feats_b = validate_representation(
        feats_b,
        name=f"{metric_name} target representation",
        min_samples=min_samples,
    )
    if feats_a.shape[0] != feats_b.shape[0]:
        raise ValueError(
            f"{metric_name} requires matching sample counts, got "
            f"{feats_a.shape[0]} and {feats_b.shape[0]}."
        )
    if feats_a.device != feats_b.device:
        raise ValueError(
            f"{metric_name} requires representations on the same device, got "
            f"{feats_a.device} and {feats_b.device}."
        )
    return feats_a, feats_b


def validate_integer_in_range(
    value: Any,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    """Validate a bounded integer parameter without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    if value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}.")
    return value


def validate_topk(
    topk: Any,
    num_samples: int,
    metric_name: str,
    minimum: int = 1,
) -> int:
    """Validate a neighbor count that excludes the sample itself."""
    return validate_integer_in_range(
        topk,
        name=f"{metric_name} topk",
        minimum=minimum,
        maximum=num_samples - 1,
    )


def validate_positive_float(value: Any, name: str) -> float:
    """Return a finite positive floating-point parameter."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive value, got {value!r}.")
    return value


def validate_kernel_pair(
    K: Any,
    L: Any,
    metric_name: str,
    min_samples: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate finite square kernel matrices with matching shape and device."""
    K = to_torch_tensor(K, name=f"{metric_name} source kernel")
    L = to_torch_tensor(L, name=f"{metric_name} target kernel")
    for label, kernel in (("source", K), ("target", L)):
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError(
                f"{metric_name} {label} kernel must be square, got shape "
                f"{tuple(kernel.shape)}."
            )
        if kernel.shape[0] < min_samples:
            raise ValueError(
                f"{metric_name} requires at least {min_samples} samples, got "
                f"{kernel.shape[0]}."
            )
        if not torch.isfinite(kernel).all():
            raise ValueError(
                f"{metric_name} {label} kernel contains non-finite values."
            )
    if K.shape != L.shape:
        raise ValueError(
            f"{metric_name} requires matching kernel shapes, got "
            f"{tuple(K.shape)} and {tuple(L.shape)}."
        )
    if K.device != L.device:
        raise ValueError(f"{metric_name} requires kernels on the same device.")
    return K, L


def normalized_similarity_or_zero(
    metric_name: str,
    sim_ab: Any,
    sim_aa: Any,
    sim_bb: Any,
) -> float:
    """Normalize a similarity, returning zero for an undefined denominator."""
    values = tuple(float(value) for value in (sim_ab, sim_aa, sim_bb))
    cross, self_a, self_b = values
    if not all(math.isfinite(value) for value in values):
        warnings.warn(
            f"{metric_name} produced non-finite similarities; returning 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0
    if self_a <= NORMALIZATION_EPS or self_b <= NORMALIZATION_EPS:
        warnings.warn(
            f"{metric_name} has a zero or non-positive self-similarity; "
            "returning 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0
    score = cross / math.sqrt(self_a * self_b)
    if not math.isfinite(score):
        warnings.warn(
            f"{metric_name} produced a non-finite normalized score; returning 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0
    return float(score)


def check_division_by_zero_warning(metric_name: str, denominator: Any) -> float:
    """Compatibility helper returning epsilon for an invalid denominator."""
    denominator = float(denominator)
    if not math.isfinite(denominator) or denominator <= NORMALIZATION_EPS:
        warnings.warn(
            f"{metric_name} denominator is zero or non-finite; using "
            f"{NORMALIZATION_EPS}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return NORMALIZATION_EPS
    return denominator


def remove_outliers(
    feats: torch.Tensor,
    q: float,
    exact: bool = False,
    max_threshold: float | None = None,
) -> torch.Tensor:
    """Clamp feature magnitudes to the selected quantile threshold."""
    if q == 1:
        return feats
    if exact:
        kth = min(int(q * feats.numel()) + 1, feats.numel())
        q_value = feats.reshape(-1).abs().kthvalue(kth).values
    else:
        q_value = torch.quantile(
            feats.abs().flatten(start_dim=1),
            q,
            dim=1,
        ).mean()
    if max_threshold is not None:
        max_threshold = max(max_threshold, q_value)
    return feats.clamp(-q_value, q_value)
