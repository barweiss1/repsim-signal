"""Sweep grids, execution results, calibration, signal aggregation, and heuristics."""

from .calibration import PermutationCalibrationResult
from .calibration import compute_aggregate_permutation_calibration
from .calibration import compute_permutation_calibration
from .calibration import make_permutation_indices
from .calibration import measure_with_permutation_null
from .execution import sweep_metric_over_param
from .grid import ResolvedSweepGrid
from .grid import SweepGridConfig
from .grid import get_param_sweep_for_metric
from .grid import map_param_name_to_kwargs
from .grid import resolve_metric_sweep_grid
from .heuristics import calc_local_minimas
from .heuristics import convolve_1d_mirror
from .heuristics import get_convex_regions
from .heuristics import get_cut_idx
from .results import ParameterSweepResult
from .signal_aggregation import aggregate_signal_auc
from .signal_aggregation import aggregate_signal_variance_weighted_auc
from .signal_aggregation import calc_variance_weighted_auc_from_sweep
from .signal_aggregation import integrate_metric_over_param
from .signal_aggregation import integrate_metric_over_param_weighted

__all__ = [
    "ParameterSweepResult",
    "PermutationCalibrationResult",
    "ResolvedSweepGrid",
    "SweepGridConfig",
    "aggregate_signal_auc",
    "aggregate_signal_variance_weighted_auc",
    "calc_local_minimas",
    "calc_variance_weighted_auc_from_sweep",
    "compute_aggregate_permutation_calibration",
    "compute_permutation_calibration",
    "convolve_1d_mirror",
    "get_convex_regions",
    "get_cut_idx",
    "get_param_sweep_for_metric",
    "integrate_metric_over_param",
    "integrate_metric_over_param_weighted",
    "make_permutation_indices",
    "map_param_name_to_kwargs",
    "measure_with_permutation_null",
    "resolve_metric_sweep_grid",
    "sweep_metric_over_param",
]
