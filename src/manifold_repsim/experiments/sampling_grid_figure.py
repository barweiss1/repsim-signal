"""Paper figure comparing the log, sqrt, and linear bandwidth sampling grids.

The CKA-signal AUC averages the similarity curve over bandwidth multipliers
``sigma`` equally spaced in ``sigma ** p``. Such a mean estimates
``integral rho d(sigma ** p)``, so the grid itself weights the curve by
``sigma ** (p - 1)``: ``p = 0`` (log), ``0.5`` (sqrt), and ``1`` (linear).
This figure shows where each grid places its points, using the same resolver
the ReSi campaigns use, so the picture cannot drift from the computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from manifold_repsim.experiments.paper_style import apply_paper_style
from manifold_repsim.experiments.paper_style import magma_ramp
from manifold_repsim.experiments.paper_style import save_figure
from manifold_repsim.sweeps.grid import resolve_metric_sweep_grid

DEFAULT_MINIMUM = 0.05
DEFAULT_MAXIMUM = 10.0
DEFAULT_NUM = 100
DEFAULT_OUTPUT_ROOT = Path("figures/paper_ready/resi")
FIGURE_SIZE: tuple[float, float] = (16.0, 5.0)
# Sigma at which the share of points is called out in panel (b).
REFERENCE_SIGMA = 1.0


@dataclass(frozen=True)
class SamplingGrid:
    """One sampling grid and how the figure names it."""

    scale: str
    exponent: float
    label: str
    values: np.ndarray

    @property
    def share_below(self) -> float:
        """Fraction of the grid's points at or below the reference sigma."""
        return float(np.mean(self.values <= REFERENCE_SIGMA))


# (campaign grid scale, exponent p, axis label), ordered as in the paper.
_GRID_SPECS: tuple[tuple[str, float, str], ...] = (
    ("log", 0.0, r"$\mathrm{AUC}_{\log}$"),
    ("sqrt", 0.5, r"$\mathrm{AUC}_{\sqrt{\sigma}}$"),
    ("linear", 1.0, r"$\mathrm{AUC}_{\mathrm{lin}}$"),
)


def build_sampling_grids(
    minimum: float = DEFAULT_MINIMUM,
    maximum: float = DEFAULT_MAXIMUM,
    num: int = DEFAULT_NUM,
) -> list[SamplingGrid]:
    """Resolve the three grids exactly as the campaigns' measure_sweeps do."""
    grids = []
    for scale, exponent, label in _GRID_SPECS:
        resolved = resolve_metric_sweep_grid(
            "cka_rbf",
            {"min": minimum, "max": maximum, "num": num, "scale": scale},
        )
        grids.append(SamplingGrid(scale, exponent, label, resolved.values))
    return grids


X_SCALES = ("log", "linear")


def _sigma_axis(axis: plt.Axes, minimum: float, maximum: float, xscale: str) -> None:
    if xscale == "log":
        axis.set_xscale("log")
        axis.set_xlim(minimum / 1.25, maximum * 1.25)
        ticks = [0.05, 0.1, 0.5, 1, 5, 10]
        ticks = [t for t in ticks if minimum / 1.25 <= t <= maximum * 1.25]
        axis.minorticks_off()
    else:
        pad = 0.02 * (maximum - minimum)
        axis.set_xlim(0.0, maximum + pad)
        ticks = [t for t in range(0, int(maximum) + 1, 2)]
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{t:g}" for t in ticks])
    axis.set_xlabel(r"bandwidth multiplier $\sigma$")


def build_figure(
    minimum: float = DEFAULT_MINIMUM,
    maximum: float = DEFAULT_MAXIMUM,
    num: int = DEFAULT_NUM,
    xscale: str = "log",
) -> plt.Figure:
    """Two panels: the sample points of each grid, and their cumulative share.

    ``xscale`` is the sigma axis: ``log`` shows the log grid evenly spaced,
    ``linear`` shows the linear grid evenly spaced.
    """
    if xscale not in X_SCALES:
        raise ValueError(f"xscale must be one of {X_SCALES}.")
    apply_paper_style()
    grids = build_sampling_grids(minimum, maximum, num)
    colors = magma_ramp(len(grids))
    figure, (points_axis, share_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, constrained_layout=True
    )

    for row, (grid, color) in enumerate(zip(grids, colors)):
        points_axis.vlines(
            grid.values, row - 0.32, row + 0.32, color=color, linewidth=1.6
        )
    points_axis.set_ylim(len(grids) - 0.4, -0.6)
    points_axis.set_yticks(range(len(grids)))
    points_axis.set_yticklabels([grid.label for grid in grids])
    points_axis.tick_params(axis="y", length=0)
    points_axis.axvline(REFERENCE_SIGMA, color="0.5", linestyle=":", linewidth=1.8)
    points_axis.set_title(f"{num} sample points per grid")
    _sigma_axis(points_axis, minimum, maximum, xscale)

    fractions = np.arange(1, num + 1) / num
    for grid, color in zip(grids, colors):
        share_axis.step(
            grid.values,
            fractions,
            where="post",
            color=color,
            label=(
                rf"$\alpha={grid.exponent:g}$: "
                f"{round(100 * grid.share_below)}% at "
                rf"$\sigma\leq{REFERENCE_SIGMA:g}$"
            ),
        )
    share_axis.axvline(REFERENCE_SIGMA, color="0.5", linestyle=":", linewidth=1.8)
    share_axis.set_ylim(0.0, 1.02)
    share_axis.set_ylabel(r"fraction of points $\leq\sigma$")
    share_axis.set_title("Where the weight sits")
    # The empty corner flips with the axis: the curves bow below the diagonal
    # on a log axis and above it on a linear one.
    share_axis.legend(
        loc="upper left" if xscale == "log" else "lower right", frameon=False
    )
    _sigma_axis(share_axis, minimum, maximum, xscale)
    return figure


def build_all(
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
    *,
    minimum: float = DEFAULT_MINIMUM,
    maximum: float = DEFAULT_MAXIMUM,
    num: int = DEFAULT_NUM,
) -> list[Path]:
    """Write the log-x and linear-x figures, each as PDF (paper) and PNG."""
    written = []
    for xscale in X_SCALES:
        for suffix in (".pdf", ".png"):
            path = Path(output_dir) / f"sampling_grids_{xscale}_x{suffix}"
            save_figure(build_figure(minimum, maximum, num, xscale), path)
            written.append(path)
    return written
