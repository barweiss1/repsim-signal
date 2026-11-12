"""Line plots of the signal-sensitivity summary: mean and 90th-percentile rank.

The figures carry no titles and use the notation of the rank figures, since they
are made for the paper: the caption says what is plotted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from manifold_repsim.resi.signal_sensitivity import SensitivityConfig

from .config import ROBUSTNESS_QUANTILE, rank_scale_case_column

FIGURE_DPI = 180
MEAN_COLOR = "#1b6ca8"
QUANTILE_COLOR = "#c8550f"
MARKER = "o"
LINE_WIDTH = 1.8
MARKER_SIZE = 6.0
SINGLE_LABEL = r"$\mathrm{CKA}_{\mathrm{RBF},\sigma}$"
SWEPT_LABEL = r"$\mathrm{AUC}(\boldsymbol{s}_\mathrm{CKA})$"


def _columns(rank_scale: str | None) -> tuple[str, str, str]:
    normalized = rank_scale_case_column(rank_scale) == "mean_normalized_rank"
    return (
        "mean_normalized_rank" if normalized else "mean_rank",
        "q90_normalized_rank" if normalized else "q90_rank",
        "Normalized rank" if normalized else "Rank",
    )


def _percent() -> int:
    return int(round(ROBUSTNESS_QUANTILE * 100))


def plot_sensitivity(
    summary: pd.DataFrame,
    config: SensitivityConfig,
    out_dir: str | Path,
    *,
    rank_scale: str | None = None,
) -> list[Path]:
    """One figure per analysis: both rank statistics against the varied setting.

    Mean and 90th-percentile rank share an axis because they share units, so the
    gap between them shows whether a setting changes the typical standing or only
    the bad tail. Lower is better. The y-range is left to the data: the point is
    the size of the change, not where it sits in the whole field.
    """
    mean_column, quantile_column, y_label = _columns(rank_scale)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for analysis in config.analyses:
        points = summary[summary["analysis"] == analysis.name]
        if points.empty:
            continue
        ordered = [
            points[points["variant"] == key].iloc[0] for key, _label in analysis.points
        ]
        positions = list(range(len(ordered)))
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        for column, color, label in (
            (mean_column, MEAN_COLOR, "mean"),
            (quantile_column, QUANTILE_COLOR, f"{_percent()}th pct."),
        ):
            ax.plot(
                positions,
                [row[column] for row in ordered],
                marker=MARKER,
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                color=color,
                label=label,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels([label for _key, label in analysis.points])
        ax.set_xlabel(analysis.x_label)
        ax.set_ylabel(f"{y_label} (lower is better)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out / f"{analysis.name}.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def plot_bandwidths(
    table: pd.DataFrame,
    config: SensitivityConfig,
    out_dir: str | Path,
    *,
    rank_scale: str | None = None,
) -> Path:
    """Single-bandwidth CKA across sigma, with the swept measure as the baseline.

    The sweep is drawn solid with circle markers and the baseline dashed in the
    same colours, mean in blue and 90th percentile in orange, so a bandwidth
    that beats the swept measure on a statistic sits below the matching dashed
    line. The legend has one column per measure.
    """
    mean_column, quantile_column, y_label = _columns(rank_scale)
    spec = config.bandwidths
    reference = table[table["measure"] == spec.reference].iloc[0]
    points = table[table["measure"] != spec.reference].sort_values("sigma")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    statistics = (
        (mean_column, MEAN_COLOR, "mean"),
        (quantile_column, QUANTILE_COLOR, f"{_percent()}th pct."),
    )
    for column, color, label in statistics:
        ax.plot(
            points["sigma"],
            points[column],
            marker=MARKER,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=color,
            label=f"{SINGLE_LABEL}, {label}",
        )
    for column, color, label in statistics:
        ax.axhline(
            reference[column],
            color=color,
            linewidth=LINE_WIDTH,
            linestyle="--",
            label=f"{SWEPT_LABEL}, {label}",
        )
    ax.set_xticks(points["sigma"])
    ax.set_xticklabels([f"{value:g}" for value in points["sigma"]])
    ax.set_xlabel(r"Bandwidth multiplier $\sigma$")
    ax.set_ylabel(f"{y_label} (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        columnspacing=1.6,
        handlelength=2.4,
    )
    fig.tight_layout()
    path = out / "bandwidths.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_paired_bandwidths(
    table: pd.DataFrame,
    config: SensitivityConfig,
    out_dir: str | Path,
    *,
    rank_scale: str | None = None,
) -> Path:
    """Rank of the swept measure and of each single bandwidth, ranked as a pair.

    At every sigma the two measures are ranked in the same field, so the swept
    measure's rank (dashed, square markers) moves with the rival it is compared
    with; the single bandwidth (solid, circle markers) is drawn the same way.
    Colours are the statistics, mean in blue and 90th percentile in orange, as in
    :func:`plot_bandwidths`.
    """
    mean_column, quantile_column, y_label = _columns(rank_scale)
    spec = config.bandwidths
    swept = table[table["measure"] == spec.reference].sort_values("sigma")
    single = table[table["measure"] != spec.reference].sort_values("sigma")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    statistics = (
        (mean_column, MEAN_COLOR, "mean"),
        (quantile_column, QUANTILE_COLOR, f"{_percent()}th pct."),
    )
    for column, color, label in statistics:
        ax.plot(
            single["sigma"],
            single[column],
            marker=MARKER,
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            color=color,
            label=f"{SINGLE_LABEL}, {label}",
        )
    for column, color, label in statistics:
        ax.plot(
            swept["sigma"],
            swept[column],
            marker="s",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            linestyle="--",
            color=color,
            label=f"{SWEPT_LABEL}, {label}",
        )
    ax.set_xticks(single["sigma"])
    ax.set_xticklabels([f"{value:g}" for value in single["sigma"]])
    ax.set_xlabel(
        r"Bandwidth multiplier $\sigma$ of the compared $\mathrm{CKA}_{\mathrm{RBF},\sigma}$"
    )
    ax.set_ylabel(f"{y_label} (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        columnspacing=1.6,
        handlelength=2.4,
    )
    fig.tight_layout()
    path = out / "bandwidths_paired.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


__all__ = ["plot_bandwidths", "plot_paired_bandwidths", "plot_sensitivity"]
