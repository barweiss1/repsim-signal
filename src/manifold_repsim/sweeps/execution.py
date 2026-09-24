"""Metric-curve execution independent of experiment and plotting concerns."""

from __future__ import annotations

from manifold_repsim.metrics import prepare_metric_curve
from manifold_repsim.metrics import score_prepared_curve

from .grid import get_param_sweep_for_metric


def sweep_metric_over_param(
    feats_A,
    feats_B,
    metric_name,
    sweep_config,
    sweep_len,
    logscale: bool = False,
    metric_kwargs=None,
):
    """Prepare shared metric state once and score a complete parameter curve."""
    parameter_name = sweep_config["param"]
    parameter_values = get_param_sweep_for_metric(
        metric_name=metric_name,
        sweep_len=sweep_len,
        logscale=logscale,
        sweep_config=sweep_config,
    )
    prepared = prepare_metric_curve(
        metric_name,
        feats_A,
        feats_B,
        parameter_name,
        parameter_values,
        **dict(metric_kwargs or {}),
    )
    return parameter_values, score_prepared_curve(prepared)
