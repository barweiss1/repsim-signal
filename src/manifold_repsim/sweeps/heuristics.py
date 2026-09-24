"""Experimental curve-shape heuristics kept separate from signal aggregation."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def get_cut_idx(scores, inf_score, threshold: float = 0.05) -> int:
    """Find the last curve point before it becomes close to an infinity score."""
    scores = np.asarray(scores, dtype=float)
    total_variation = np.sum(np.abs(np.diff(scores)))
    cut_index = len(scores) - 1
    if total_variation > 1e-1:
        while (
            cut_index > 0
            and np.abs(scores[cut_index] - inf_score) / total_variation < threshold
        ):
            cut_index -= 1
    return cut_index


def convolve_1d_mirror(signal, kernel) -> np.ndarray:
    """Convolve a one-dimensional signal using reflected boundary padding."""
    pad_length = len(kernel) // 2
    padded_signal = np.pad(signal, pad_width=pad_length, mode="reflect")
    return np.convolve(padded_signal, kernel, mode="valid")


def get_convex_regions(
    param_vec,
    scores,
    logscale: bool = False,
    derivative_type: str = "average",
    side_lobe: float = 0.5,
    curvature_threshold: float = 0.0,
    min_region_len: int = 1,
) -> list[list[int]]:
    """Return contiguous indices whose smoothed discrete curvature is positive."""
    x = np.asarray(param_vec, dtype=float)
    y = np.asarray(scores, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("param_values and scores must be one-dimensional.")
    if len(x) != len(y):
        raise ValueError("param_values and scores must have the same length.")
    if len(x) < 3:
        return []
    if logscale:
        if np.any(x <= 0):
            raise ValueError("param_values must be positive when logscale=True.")
        x = np.log(x)

    if side_lobe > 0:
        kernel = np.array([side_lobe, 1.0, side_lobe], dtype=float)
        y = convolve_1d_mirror(y, kernel / kernel.sum())

    if derivative_type == "average":
        first_spacing = 1.0
        second_spacing = 1.0
    elif derivative_type == "trapezoidal":
        first_spacing = np.diff(x)
        second_spacing = np.diff(0.5 * (x[1:] + x[:-1]))
    else:
        raise ValueError(f"Unsupported derivative_type: {derivative_type}")
    if np.any(first_spacing <= 0):
        raise ValueError("param_values must be strictly increasing.")

    first_derivative = np.diff(y) / first_spacing
    second_derivative = np.diff(first_derivative) / second_spacing
    convex_indices = np.where(second_derivative > curvature_threshold)[0] + 1
    if len(convex_indices) == 0:
        return []

    regions: list[list[int]] = []
    current = [int(convex_indices[0])]
    for raw_index in convex_indices[1:]:
        index = int(raw_index)
        if index == current[-1] + 1:
            current.append(index)
        else:
            if len(current) >= min_region_len:
                regions.append(current)
            current = [index]
    if len(current) >= min_region_len:
        regions.append(current)
    return regions


def calc_local_minimas(param_vec, scores, robust: bool = True):
    """Return parameter values and scores at local minima of a curve."""
    scores = np.asarray(scores)
    param_vec = np.asarray(param_vec)
    if robust:
        indices, _ = find_peaks(-scores, prominence=0.01, distance=4, width=None)
        return param_vec[indices], scores[indices]

    indices = [
        index
        for index in range(1, len(scores) - 1)
        if scores[index] < scores[index - 1] and scores[index] < scores[index + 1]
    ]
    return param_vec[indices].tolist(), scores[indices].tolist()
