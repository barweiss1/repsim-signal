"""Publication-ready figures for the glocal sweep MDS experiment.

This module builds a small, curated set of paper figures from an already
computed run (i.e. one that ran the `compute` stage; see `workflow.run`). It
reads saved per-metric NPZ artifacts directly (via `load_computed_results`/
`load_npz`) and performs no scoring or feature loading of its own; it is a
thin composition of the existing plotting primitives in `plotting.py` and
`exploration.py`, styled for print rather than exploration. Two things are
refit rather than read straight off disk: the "all datasets" variant (see
`build_all`'s `config` argument), which reuses
`exploration.concatenate_dataset_results` -- itself a thin wrapper around
the pipeline's own MDS/PCA fitting functions -- to refit an embedding for a
concatenation the canonical per-dataset pipeline never computes; and the
diffusion-map grids (`plot_diffusion_map_grid`), which reuse
`exploration.compute_embedding`'s `"diffusion_map"` method to fit directly
from each metric's own precomputed `distances` (no embedding of that kind is
precomputed anywhere in the pipeline, unlike MDS/PCA).

Unlike the exploratory `plotting.py` pipeline, which draws a continuous
colorbar for tau, these figures encode all three swept parameters as
discrete channels: lambda as a magma color sampled at each level's own
log-scale position (see `_lambda_palette`), tau as point opacity, and alpha
as point size. `_paper_embedding_guides` renders these as side-by-side
legend columns sharing one vertical span: lambda and tau as rank-banded
colorbars (`paper_style.discrete_colorbar_panel` -- hue bands for lambda,
alpha bands for tau) and alpha as a representative-marker size column
(`paper_style.size_legend_panel`). There is no `none` legend column: `none`
is drawn as a plain circle among the regular points and ringed with
`paper_style.highlight_reference_point`, the same way the synthetic module
marks its own base condition without a separate legend entry. Every
embedding panel (MDS, PCA, and diffusion-map grids alike, since they share
`_scatter_paper_embedding`) drops its tick numbers/marks -- the embedding's
numeric scale is arbitrary -- keeping each panel to just its point cloud.

The embedding grids and the signal-sweep figures are both laid out as a
single horizontal row (1x4 and 1x3 respectively) rather than a 2x2 grid or a
3-row column.

Everything that should look identical to the synthetic experiment's paper
figures -- rcParams, the magma ramp, legend geometry, the colorbar and
size-legend panels, the signal-row styling tail -- lives in
`manifold_repsim.experiments.paper_style` and is imported rather than
reimplemented; the two experiment packages still never import each other.
What stays here is what carries meaning specific to this experiment: lambda
is log-spaced, so its colors are sampled at log positions (`_lambda_palette`
via `paper_style.magma_at`) and its labels are formatted as powers of ten
(`_lambda_magnitude`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..artifacts import load_npz
from ..paper_style import EMBEDDING_GRID_FIGSIZE
from ..paper_style import EMBEDDING_GRID_RECT
from ..paper_style import OPACITY_RANGE
from ..paper_style import SIGNAL_ROW_FIGSIZE
from ..paper_style import apply_paper_style
from ..paper_style import discrete_colorbar_panel
from ..paper_style import highlight_reference_point
from ..paper_style import legend_panel_metrics
from ..paper_style import magma_at
from ..paper_style import magma_ramp
from ..paper_style import opacity_swatches
from ..paper_style import place_panel
from ..paper_style import save_figure
from ..paper_style import shrunk_span
from ..paper_style import side_panel_bounds
from ..paper_style import size_legend_panel
from ..paper_style import style_signal_row
from .config import REPO_ROOT
from .discovery import DatasetGroup, GroupRef
from .exploration import compute_embedding, concatenate_dataset_results
from .plotting import _likely_log, _point_sizes

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "figures" / "paper_ready" / "glocal"

# The three fixed metrics that always fill the first three MDS-grid panels.
MDS_FIXED_METRICS: tuple[tuple[str, str], ...] = (
    ("cka_linear", r"$CKA_{\mathrm{lin}}$"),
    ("cka_rbf_sigma_02", r"$CKA_{\mathrm{RBF}}(\sigma=0.2)$"),
    ("mutual_knn_k10", r"Mutual $10$-NN"),
)

# One 1x4 MDS row is built per variant, swapping only the 4th (signal-
# metric) panel.
# MDS_SIGNAL_VARIANTS: tuple[tuple[str, str], ...] = (
#     ("cka_rbf", r"CKA RBF signal ($s_{CKA}(\sigma)$)"),
#     ("rbf_degree_crwka", r"Degree-centered RWKA ($s_{dRWKA}(\sigma)$)"),
# )

MDS_SIGNAL_VARIANTS: tuple[tuple[str, str], ...] = (
    ("cka_rbf", r"$s_{CKA}(\sigma)$"),
    ("rbf_degree_crwka", r"$s_{dRWKA}(\sigma)$"),
)

# One 1x3 (lambda/alpha/tau sweep) horizontal row figure per curated signal
# metric.
SIGNAL_SWEEP_METRICS: tuple[tuple[str, str], ...] = (
    ("cka_rbf", r"$s_{CKA}(\sigma)$"),
    ("rbf_degree_crwka", r"$s_{dRWKA}(\sigma)$"),
    ("rbf_rwka", r"$s_{RWKA}(\sigma)$"),
)

# Fixed reference point for the two non-swept parameters in each
# signal-sweep panel (e.g. alpha/tau held fixed while lambda is swept).
SIGNAL_SWEEP_FIXED_ALPHA = 0.25
SIGNAL_SWEEP_FIXED_LAMBDA = 0.01
SIGNAL_SWEEP_FIXED_TAU = 0.25

# Kernel-bandwidth median-scale knobs each diffusion-map grid is built at
# (see `plot_diffusion_map_grid`); one 1x4 row figure per scale, per
# `MDS_SIGNAL_VARIANTS` entry.
DIFFUSION_MAP_MEDIAN_SCALES: tuple[float, ...] = (10.0, 15.0)


def _lambda_magnitude(value: float) -> str:
    """Format one lambda value as a bare LaTeX fragment (no `$` delimiters).

    Uses power-of-ten notation (`10^{-3}`) when `value` is, within floating
    point tolerance, an exact power of ten -- glocal's lambda grid is
    typically log-spaced -- and falls back to `%g` formatting otherwise.
    """
    if value > 0:
        exponent = math.log10(value)
        rounded = round(exponent)
        if math.isclose(exponent, rounded, abs_tol=1e-9):
            return f"10^{{{int(rounded)}}}"
    return f"{value:g}"


def _lambda_label(value: float) -> str:
    """One lambda value as a bare mathtext legend entry.

    Just the magnitude, not `\\lambda=...`: every legend that uses these is
    already titled with the parameter's symbol, so repeating it on each entry
    is redundant -- and the synthetic module's equivalent legends list bare
    values too.
    """
    return rf"${_lambda_magnitude(value)}$"


def _tau_palette(tau_levels: np.ndarray) -> dict[float, tuple[float, ...]]:
    """Map each discrete tau level to one fixed color (a legend, not a
    colorbar) -- still used by `plot_signal_sweep_row`'s tau-sweep
    panel, which is unaffected by the embedding grids' lambda-color/
    tau-opacity encoding change (see `_lambda_palette`/`_tau_opacities`)."""
    colors = magma_ramp(len(tau_levels))
    return {float(value): tuple(color) for value, color in zip(tau_levels, colors)}


def _lambda_palette(lambda_levels: np.ndarray) -> dict[float, tuple[float, ...]]:
    """Map each lambda level to a magma color, sampled at that level's own
    *log-scale* position within [min, max] -- not evenly spaced by index/
    rank, so the ramp still reads correctly if the grid's log-spacing is
    ever uneven. An exact power-of-ten grid (glocal's usual case) samples in
    perfectly even ramp steps either way, since its log-positions already
    are evenly spaced.
    """
    lambda_levels = np.asarray(lambda_levels, dtype=float)
    if np.any(lambda_levels <= 0):
        raise ValueError("lambda values must be positive to sample a log-scale ramp")
    log_values = np.log10(lambda_levels)
    span = log_values.max() - log_values.min()
    if len(lambda_levels) == 1 or math.isclose(span, 0.0, abs_tol=1e-12):
        positions = np.full(len(lambda_levels), 0.5)
    else:
        positions = (log_values - log_values.min()) / span
    colors = magma_at(positions)
    return {float(value): tuple(color) for value, color in zip(lambda_levels, colors)}


def _tau_opacities(tau_levels: np.ndarray) -> dict[float, float]:
    """Map each discrete tau level to one fixed point opacity (matplotlib
    scatter's `alpha=`) across the shared `paper_style.OPACITY_RANGE`,
    mirroring `_point_sizes`'s level-to-value mapping convention but for the
    opacity channel instead of size."""
    low, high = OPACITY_RANGE
    if len(tau_levels) == 1:
        return {float(tau_levels[0]): (low + high) / 2.0}
    return {
        float(value): float(opacity)
        for value, opacity in zip(tau_levels, np.linspace(low, high, len(tau_levels)))
    }


def _scatter_paper_embedding(
    axis: plt.Axes,
    result: dict[str, np.ndarray],
    embedding: np.ndarray,
    lambda_colors: dict[float, tuple[float, ...]],
    tau_opacities: dict[float, float],
) -> None:
    """Scatter one embedding panel with explicit color (lambda) and opacity
    (tau); alpha (the hyperparameter) is point size via `_point_sizes`. One
    marker shape throughout -- lambda and tau are both encoded through
    color and opacity now, not shape.
    """
    transformed = ~result["is_none"]
    sizes = _point_sizes(result)
    for lambda_value, color in lambda_colors.items():
        for tau_value, opacity in tau_opacities.items():
            mask = (
                transformed
                & np.isclose(result["lambda_values"], lambda_value)
                & np.isclose(result["tau_values"], tau_value)
            )
            if not mask.any():
                continue
            axis.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color],
                s=sizes[mask],
                marker="o",
                alpha=opacity,
                edgecolors="none",
            )
    # `none` (no glocal transform) has no lambda/tau/alpha value to color,
    # opacity, or size it by, so it gets a neutral fill plus the shared
    # reference-point ring -- the same way the synthetic module marks its own
    # base condition, which keeps its encoded color under the identical ring.
    base = result["is_none"]
    axis.scatter(
        embedding[base, 0],
        embedding[base, 1],
        s=sizes[base],
        marker="o",
        c="0.35",
        edgecolors="none",
        zorder=5,
    )
    highlight_reference_point(
        axis, (embedding[base, 0], embedding[base, 1]), sizes[base]
    )
    # Deliberately *not* an equal aspect ratio: MDS/PCA panels can have very
    # different variance along their two axes (e.g. one dimension explains
    # almost all the spread), and forcing equal units per axis pads out the
    # low-variance one until its real structure becomes illegible. "auto"
    # lets each axis scale independently to fill the panel.
    axis.set_aspect("auto")
    # These axes' numeric scale is arbitrary (MDS/PCA/diffusion-map units),
    # so the tick numbers carry no information worth the visual clutter;
    # dropping the ticks also removes the now-empty gridlines they anchored.
    axis.set_xticks([])
    axis.set_yticks([])


def _paper_embedding_guides(
    figure: plt.Figure,
    axes: Any,
    result: dict[str, np.ndarray],
    lambda_colors: dict[float, tuple[float, ...]],
    tau_opacities: dict[float, float],
) -> None:
    """One compact legend block on the right, arranged as side-by-side
    columns sharing one vertical span -- lambda and tau as rank-banded
    colorbars, alpha as a representative-marker size column, all built from
    the shared `paper_style` primitives so the block is the same object in
    both experiments' figures. No `none` column: `none` is drawn among the
    regular points and ringed (see `_scatter_paper_embedding`), the same way
    the synthetic module marks its own base condition.
    """
    transformed = ~result["is_none"]
    axes_list = list(np.asarray(axes, dtype=object).ravel())
    left, bottom, top = side_panel_bounds(figure, axes_list)
    panel_bottom, panel_top = shrunk_span(bottom, top)
    panel_height = panel_top - panel_bottom
    panel_width, inter_gap = legend_panel_metrics(figure)

    panel_left = left
    rect = (panel_left, panel_bottom, panel_width, panel_height)
    lambda_axis = discrete_colorbar_panel(
        figure,
        rect,
        list(lambda_colors.values()),
        [f"${_lambda_magnitude(value)}$" for value in lambda_colors],
        r"$\lambda$",
    )
    panel_left = place_panel(figure, lambda_axis, rect, left) + inter_gap

    rect = (panel_left, panel_bottom, panel_width, panel_height)
    tau_axis = discrete_colorbar_panel(
        figure,
        rect,
        opacity_swatches(list(tau_opacities.values())),
        [f"{value:g}" for value in tau_opacities],
        r"$\tau$",
    )
    panel_left = place_panel(figure, tau_axis, rect, panel_left) + inter_gap

    alpha_levels = np.unique(result["alpha_values"][transformed])
    sizes = _point_sizes(result)[transformed]
    alpha_sizes = [
        float(sizes[np.isclose(result["alpha_values"][transformed], value)][0])
        for value in alpha_levels
    ]
    rect = (panel_left, panel_bottom, panel_width, panel_height)
    alpha_axis = size_legend_panel(
        figure,
        rect,
        alpha_sizes,
        [f"{value:g}" for value in alpha_levels],
        r"$\alpha$",
    )
    place_panel(figure, alpha_axis, rect, panel_left)


def _plot_embedding_grid(
    group: DatasetGroup | GroupRef,
    metric_results: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    *,
    embedding_fn: Callable[[dict[str, np.ndarray]], np.ndarray],
    file_prefix: str,
) -> None:
    """1x4 horizontal embedding rows (three fixed metrics + one curated
    signal metric each), one file per `MDS_SIGNAL_VARIANTS` entry, sharing a
    discrete lambda-color/tau-opacity/alpha-size legend. `embedding_fn` computes each
    panel's (N, 2) embedding from its metric result -- a precomputed
    passthrough for MDS/PCA, a fresh diffusion-map fit for
    `plot_diffusion_map_grid`; see `plot_mds_grid`/`plot_pca_grid`/
    `plot_diffusion_map_grid`.
    """
    reference = metric_results[MDS_FIXED_METRICS[0][0]]
    transformed = ~reference["is_none"]
    lambda_colors = _lambda_palette(np.unique(reference["lambda_values"][transformed]))
    tau_opacities = _tau_opacities(np.unique(reference["tau_values"][transformed]))

    for signal_id, signal_title in MDS_SIGNAL_VARIANTS:
        panels = MDS_FIXED_METRICS + ((signal_id, signal_title),)
        figure, axes = plt.subplots(
            1, 4, figsize=EMBEDDING_GRID_FIGSIZE, constrained_layout=True
        )
        # The legend block (see `_paper_embedding_guides`) sits in a reserved
        # right-hand margin that constrained_layout doesn't know about on its
        # own; without this it would draw the 1x4 row across the full figure
        # width and the legend would land on top of it.
        figure.get_layout_engine().set(rect=EMBEDDING_GRID_RECT)
        last_result = reference
        for axis, (metric_id, title) in zip(axes.flat, panels):
            result = metric_results[metric_id]
            _scatter_paper_embedding(
                axis, result, embedding_fn(result), lambda_colors, tau_opacities
            )
            axis.set_title(title)
            last_result = result
        _paper_embedding_guides(figure, axes, last_result, lambda_colors, tau_opacities)
        save_figure(
            figure,
            output_dir / group.model / group.dataset / f"{file_prefix}_{signal_id}.pdf",
        )


def plot_mds_grid(
    group: DatasetGroup | GroupRef,
    metric_results: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """`_plot_embedding_grid` using each metric's precomputed MDS embedding."""
    _plot_embedding_grid(
        group,
        metric_results,
        output_dir,
        embedding_fn=lambda result: np.asarray(result["embedding_2d"], dtype=float),
        file_prefix="mds_grid",
    )


def plot_pca_grid(
    group: DatasetGroup | GroupRef,
    metric_results: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """`_plot_embedding_grid` using each metric's precomputed PCA embedding."""
    _plot_embedding_grid(
        group,
        metric_results,
        output_dir,
        embedding_fn=lambda result: np.asarray(result["pca_embedding_2d"], dtype=float),
        file_prefix="pca_grid",
    )


def plot_diffusion_map_grid(
    group: DatasetGroup | GroupRef,
    metric_results: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    *,
    median_scale: float,
) -> None:
    """`_plot_embedding_grid` using a fresh diffusion-map embedding fit per
    panel (`exploration.compute_embedding`, method `"diffusion_map"`), at
    kernel bandwidth `median_scale` times each panel's own median pairwise
    distance. Unlike MDS/PCA this is not a precomputed passthrough -- every
    panel is refit here from that metric's own `distances`. See
    `DIFFUSION_MAP_MEDIAN_SCALES` for the scales `build_all` builds by
    default.
    """
    _plot_embedding_grid(
        group,
        metric_results,
        output_dir,
        embedding_fn=lambda result: compute_embedding(
            result, "diffusion_map", diffusion_median_scale=median_scale
        ),
        file_prefix=f"diffusion_grid_scale{median_scale:g}",
    )


def _fixed_signal_curve(
    result: dict[str, np.ndarray],
    *,
    lambda_value: float,
    alpha_value: float,
    tau_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the batch mean and std curve for one exact (lambda, alpha, tau)."""
    transformed = ~result["is_none"]
    mask = (
        transformed
        & np.isclose(result["lambda_values"], lambda_value)
        & np.isclose(result["alpha_values"], alpha_value)
        & np.isclose(result["tau_values"], tau_value)
    )
    index = np.flatnonzero(mask)
    if len(index) != 1:
        raise ValueError("Signal-sweep slice does not identify exactly one condition")
    batch_scores = result["scores"][index[0]]
    return batch_scores.mean(axis=0), batch_scores.std(axis=0)


def plot_signal_sweep_row(
    group: DatasetGroup | GroupRef,
    metric_results: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """One 1x3 (lambda/alpha/tau sweep) horizontal row figure per curated
    signal metric. Panel styling -- no per-panel title, the swept parameter's
    symbol as the legend title, a bottom-left legend with the x-axis widened
    to make room, and the larger signal-row fonts -- comes from the shared
    `paper_style.style_signal_row`, so these panels and the synthetic
    module's are typographically identical."""
    for metric_id, y_label in SIGNAL_SWEEP_METRICS:
        result = metric_results[metric_id]
        transformed = ~result["is_none"]
        lambdas = np.unique(result["lambda_values"][transformed])
        alphas = np.unique(result["alpha_values"][transformed])
        taus = np.unique(result["tau_values"][transformed])
        tau_colors = _tau_palette(taus)
        moderate_lambda = SIGNAL_SWEEP_FIXED_LAMBDA
        moderate_alpha = SIGNAL_SWEEP_FIXED_ALPHA
        moderate_tau = SIGNAL_SWEEP_FIXED_TAU
        x_values = result["parameter_values"]
        logscale = _likely_log(x_values)

        figure, axes = plt.subplots(
            1, 3, figsize=SIGNAL_ROW_FIGSIZE, constrained_layout=True
        )

        lambda_axis, alpha_axis, tau_axis = axes
        lambda_colors = magma_ramp(len(lambdas))
        for lambda_value, color in zip(lambdas, lambda_colors):
            mean, std = _fixed_signal_curve(
                result,
                lambda_value=lambda_value,
                alpha_value=moderate_alpha,
                tau_value=moderate_tau,
            )
            # A plain circle, not a per-lambda marker shape: color already
            # separates the curves here, so the shapes were a redundant second
            # encoding the synthetic module's equivalent panels do not use.
            lambda_axis.plot(
                x_values,
                mean,
                marker="o",
                color=color,
                label=_lambda_label(lambda_value),
            )
            lambda_axis.fill_between(
                x_values, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )

        alpha_colors = magma_ramp(len(alphas))
        for alpha_value, color in zip(alphas, alpha_colors):
            mean, std = _fixed_signal_curve(
                result,
                lambda_value=moderate_lambda,
                alpha_value=alpha_value,
                tau_value=moderate_tau,
            )
            alpha_axis.plot(
                x_values,
                mean,
                color=color,
                marker="o",
                label=f"{alpha_value:g}",
            )
            alpha_axis.fill_between(
                x_values, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )

        for tau_value in taus:
            mean, std = _fixed_signal_curve(
                result,
                lambda_value=moderate_lambda,
                alpha_value=moderate_alpha,
                tau_value=tau_value,
            )
            color = tau_colors[float(tau_value)]
            tau_axis.plot(
                x_values, mean, color=color, marker="o", label=f"{tau_value:g}"
            )
            tau_axis.fill_between(
                x_values, mean - std, mean + std, color=color, alpha=0.15, linewidth=0
            )

        style_signal_row(
            axes,
            logscale=logscale,
            legend_titles=(r"$\lambda$", r"$\alpha$", r"$\tau$"),
            y_label=y_label,
        )

        save_figure(
            figure,
            output_dir / group.model / group.dataset / f"signal_sweep_{metric_id}.pdf",
        )


def _metric_results_for_group(
    group: DatasetGroup | GroupRef, results: dict[tuple[str, str], list[Path]]
) -> dict[str, dict[str, np.ndarray]]:
    """metric_id -> loaded NPZ result for one real (model, dataset) group."""
    metric_results = {}
    for path in results[(group.model, group.dataset)]:
        result = load_npz(path)
        metric_results[result["metric_id"].item()] = result
    return metric_results


# Every metric id any curated paper figure reads, so building the "all
# datasets" variant only concatenates/refits what is actually needed.
_REQUIRED_METRIC_IDS: frozenset[str] = frozenset(
    {metric_id for metric_id, _ in MDS_FIXED_METRICS}
    | {metric_id for metric_id, _ in MDS_SIGNAL_VARIANTS}
    | {metric_id for metric_id, _ in SIGNAL_SWEEP_METRICS}
)


def _metric_results_for_all_datasets(
    model: str,
    results: dict[tuple[str, str], list[Path]],
    config: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """metric_id -> result concatenated across every real dataset for
    `model` (`exploration.concatenate_dataset_results`), for exactly the
    metrics the curated figures need. A metric missing from any of that
    model's datasets is skipped for that metric only, rather than failing
    the whole "all datasets" build.
    """
    real_datasets = sorted(
        {dataset for (candidate_model, dataset) in results if candidate_model == model}
    )
    per_dataset = {
        dataset: _metric_results_for_group(GroupRef(model, dataset), results)
        for dataset in real_datasets
    }
    merged = {}
    for metric_id in _REQUIRED_METRIC_IDS:
        results_by_dataset = {
            dataset: per_dataset[dataset][metric_id]
            for dataset in real_datasets
            if metric_id in per_dataset[dataset]
        }
        if len(results_by_dataset) != len(real_datasets):
            continue
        merged[metric_id] = concatenate_dataset_results(results_by_dataset, config)
    return merged


def build_all(
    groups: list[DatasetGroup | GroupRef],
    results: dict[tuple[str, str], list[Path]],
    output_dir: Path | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> None:
    """Build every paper-ready glocal figure for every discovered group,
    plus one "all datasets" variant per model (batches concatenated across
    every real dataset for that model, dataset name `"all"`) when `config`
    is provided -- needed to refit that concatenation's MDS embedding (see
    `exploration.concatenate_dataset_results`). Without `config`, only the
    real per-dataset groups are built.
    """
    resolved_output_dir = (
        Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT
    )
    apply_paper_style()
    for group in groups:
        metric_results = _metric_results_for_group(group, results)
        plot_mds_grid(group, metric_results, resolved_output_dir)
        plot_pca_grid(group, metric_results, resolved_output_dir)
        for median_scale in DIFFUSION_MAP_MEDIAN_SCALES:
            plot_diffusion_map_grid(
                group, metric_results, resolved_output_dir, median_scale=median_scale
            )
        plot_signal_sweep_row(group, metric_results, resolved_output_dir)

    if config is not None:
        for model in sorted({group.model for group in groups}):
            all_group = GroupRef(model=model, dataset="all")
            metric_results = _metric_results_for_all_datasets(model, results, config)
            plot_mds_grid(all_group, metric_results, resolved_output_dir)
            plot_pca_grid(all_group, metric_results, resolved_output_dir)
            for median_scale in DIFFUSION_MAP_MEDIAN_SCALES:
                plot_diffusion_map_grid(
                    all_group,
                    metric_results,
                    resolved_output_dir,
                    median_scale=median_scale,
                )
            plot_signal_sweep_row(all_group, metric_results, resolved_output_dir)
