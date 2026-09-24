"""Alternative embeddings and diagnostic views for glocal sweep MDS results.

Classical MDS does not always separate the swept parameters (lambda, alpha,
tau) well visually. This module offers additional candidate embeddings
(t-SNE, Isomap, per-parameter LDA) plus non-embedding diagnostic views (raw
parallel coordinates, a reordered distance-matrix heatmap, parameter
correlation heatmaps), all operating on an already-loaded metric result --
no scoring, no plotting imports. Consumed by
`glocal_visualization_explorer.ipynb`.

`concatenate_dataset_results` is the one exception to "no refitting": it is
an exploration-only aggregation across datasets that were each already
analyzed independently by the canonical pipeline (`workflow.run` still
never mixes datasets), and it reuses that pipeline's exact
`score_distances`/`fit_metric_mds`/`fit_signal_pca` functions rather than
reimplementing them, because there is no precomputed embedding for a
concatenation that pipeline never computes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE, Isomap

from manifold_repsim.experiments.mds import fit_metric_mds
from manifold_repsim.sweeps.signal_aggregation import aggregate_signal_auc

from .scoring import concatenate_batch_scores, fit_signal_pca, score_distances

EMBEDDING_METHODS: tuple[str, ...] = (
    "mds",
    "pca",
    "tsne",
    "isomap",
    "diffusion_map",
    "lda_lambda",
    "lda_alpha",
    "lda_tau",
)

PARAMETER_KEYS: dict[str, str] = {
    "lambda": "lambda_values",
    "alpha": "alpha_values",
    "tau": "tau_values",
}

# Tiebreak order used by reordered_distance_matrix's compound sort key.
OTHER_PARAMETERS: dict[str, tuple[str, str]] = {
    "lambda": ("alpha", "tau"),
    "alpha": ("lambda", "tau"),
    "tau": ("lambda", "alpha"),
}

CORRELATION_METHODS: tuple[str, ...] = ("pearson", "spearman")

# Signal-mode metrics correlated against each swept parameter by
# parameter_correlation_heatmap_data, and the two fixed-mode metrics shown
# alongside them as constant reference rows. These match the checked-in
# configs/glocal_sweep_mds.yaml metric ids; metrics absent from a given
# `results` dict (e.g. a different config) are silently skipped rather than
# raising, so this module stays usable against any run.
CORRELATION_SIGNAL_METRICS: tuple[str, ...] = (
    "cka_rbf",
    "rbf_uka",
    "rbf_rwka",
    "rbf_degree_crwka",
    "rbf_crwka",
    "mutual_knn",
    "mutual_knn_dist",
)
CORRELATION_FIXED_METRICS: tuple[str, ...] = ("cka_linear", "mutual_knn_k10")

# Order-matched to plotting.MARKERS = ("o","s","^","D","P","X","v","<",">")
# so a given parameter level's shape reads the same in the notebook as in
# the paper figures, just translated to plotly's marker-symbol vocabulary.
PLOTLY_SYMBOLS: tuple[str, ...] = (
    "circle",
    "square",
    "triangle-up",
    "diamond",
    "cross",
    "x",
    "triangle-down",
    "triangle-left",
    "triangle-right",
)

# Discrete per-level colors for the plotly "Color:" channel: the first 9
# entries of plotly's own qualitative "Plotly" swatch (`plotly.colors
# .qualitative.Plotly`), a categorical palette designed for maximum
# distinctness between adjacent entries -- unlike a sequential colormap
# ramp (e.g. magma), where neighboring samples can look too similar to
# tell apart at a glance. Hardcoded here (rather than imported) so this
# module stays free of any plotting-library import.
CHANNEL_COLORS: tuple[str, ...] = (
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
)


def _fit_tsne(
    vectors: np.ndarray, *, perplexity: float, random_state: int
) -> np.ndarray:
    """Fresh t-SNE fit; sklearn requires perplexity < n_samples, so clamp it
    so tiny fixtures and small real groups never raise, without silently
    distorting a genuinely valid request on larger data."""
    n_samples = vectors.shape[0]
    effective = max(1.0, min(float(perplexity), float(n_samples - 1)))
    model = TSNE(
        n_components=2, perplexity=effective, init="pca", random_state=random_state
    )
    return np.asarray(model.fit_transform(vectors), dtype=float)


def _fit_isomap(vectors: np.ndarray, *, n_neighbors: int) -> np.ndarray:
    """Fresh Isomap fit; sklearn requires n_neighbors < n_samples, clamped
    analogously to `_fit_tsne`."""
    n_samples = vectors.shape[0]
    effective = max(1, min(int(n_neighbors), n_samples - 1))
    model = Isomap(n_neighbors=effective, n_components=2)
    return np.asarray(model.fit_transform(vectors), dtype=float)


def _fit_diffusion_map(
    result: dict[str, np.ndarray],
    *,
    median_scale: float = 1.0,
    n_components: int = 2,
    diffusion_time: int = 1,
) -> np.ndarray:
    """Diffusion map embedding built directly from the metric's own
    precomputed pairwise `distances` (no refit of `concatenated_vectors`).

    Kernel bandwidth sigma is the median of the pairwise distances (over
    all condition pairs, diagonal excluded) times `median_scale` -- the one
    tunable knob this method exposes. Normalization follows the standard
    two-step diffusion-maps recipe (Coifman & Lafon, alpha=1):

    1. The Gaussian affinity kernel W = exp(-distances**2 / (2 sigma**2)) is
       density-normalized as ``D^-1 W D^-1`` (D = row sums of W), removing
       bias from non-uniform sampling density before the walk is built.
    2. That density-normalized kernel is then row-normalized into a proper
       Markov transition matrix, removing any *remaining* degree effect so
       the walk is driven purely by relative similarity, not how many
       neighbors a condition happens to have.

    The transition matrix is not symmetric, but is similar to a symmetric
    matrix (density-normalized-and-row-normalized are both symmetric-safe
    operations on a symmetric W), so its eigendecomposition is computed via
    that symmetric conjugate with `np.linalg.eigh` (exact, numerically
    stable) rather than a general nonsymmetric eigensolver. The diffusion
    coordinates are the transition matrix's eigenvectors -- scaled by their
    eigenvalue to the `diffusion_time` power -- excluding the leading
    trivial eigenvector (eigenvalue ~1, the stationary distribution, which
    carries no structure). Zero-padded when fewer than `n_components`
    non-trivial eigenvectors exist (matching `fit_signal_pca`'s convention
    for the same "fewer available components" situation).
    """
    distances = np.asarray(result["distances"], dtype=float)
    n = distances.shape[0]
    if median_scale <= 0.0:
        raise ValueError("median_scale must be positive")

    upper = distances[np.triu_indices(n, k=1)]
    median_distance = float(np.median(upper)) if upper.size else 0.0
    if median_distance <= 0.0:
        raise ValueError(
            "median pairwise distance is zero or undefined; cannot set a "
            "diffusion map kernel bandwidth from it"
        )
    sigma = median_distance * median_scale

    affinity = np.exp(-np.square(distances) / (2.0 * sigma**2))

    # Step 1: alpha=1 density normalization D^-1 W D^-1.
    degree = affinity.sum(axis=1)
    density_normalized = affinity / degree[:, None] / degree[None, :]

    # Step 2: row-normalize into a Markov transition matrix.
    row_degree = density_normalized.sum(axis=1)
    inverse_sqrt_row_degree = 1.0 / np.sqrt(row_degree)
    # density_normalized is symmetric (W and degree both are), so this
    # conjugate is exactly symmetric, not an approximation.
    symmetric_conjugate = (
        density_normalized
        * inverse_sqrt_row_degree[:, None]
        * inverse_sqrt_row_degree[None, :]
    )
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_conjugate)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Right eigenvectors of the (nonsymmetric) transition matrix, recovered
    # from the symmetric conjugate's orthonormal eigenvectors.
    transition_eigenvectors = eigenvectors * inverse_sqrt_row_degree[:, None]

    available = max(0, transition_eigenvectors.shape[1] - 1)
    effective_components = min(n_components, available)
    coordinates = np.zeros((n, n_components))
    for index in range(effective_components):
        coordinates[:, index] = (
            eigenvalues[index + 1] ** diffusion_time
            * transition_eigenvectors[:, index + 1]
        )
    return coordinates


def _fit_lda(result: dict[str, np.ndarray], parameter: str) -> np.ndarray:
    """Supervised LDA: fit only on non-`none` rows using `parameter`'s
    values as class labels, then transform every row (`none` included) so
    it still gets a plotted position.

    LDA yields at most `n_classes - 1` components. When `parameter` has only
    two distinct levels this is 1, not 2 -- the missing second column is
    zero-padded (matching `scoring.fit_signal_pca`'s existing convention for
    the same "fewer fitted components than requested" situation). That
    padded column carries no signal; it exists only so callers always get a
    plottable (N, 2) array. Do not mistake it for a real second discriminant
    axis.
    """
    if parameter not in PARAMETER_KEYS:
        raise ValueError(
            f"Unknown parameter {parameter!r}; choose one of {tuple(PARAMETER_KEYS)}"
        )
    is_none = np.asarray(result["is_none"], dtype=bool)
    vectors = np.asarray(result["concatenated_vectors"], dtype=float)
    labels = np.asarray(result[PARAMETER_KEYS[parameter]], dtype=float)[~is_none]
    # sklearn's LDA treats a float target as continuous (and refuses to fit)
    # unless it is unambiguously categorical, so fit on the levels' integer
    # codes rather than the raw parameter values themselves.
    unique_levels, class_codes = np.unique(labels, return_inverse=True)
    n_classes = len(unique_levels)
    if n_classes < 2:
        raise ValueError(f"LDA by {parameter} requires at least two distinct levels")
    model = LinearDiscriminantAnalysis(n_components=min(2, n_classes - 1))
    model.fit(vectors[~is_none], class_codes)
    embedding = np.asarray(model.transform(vectors), dtype=float)
    if embedding.shape[1] < 2:
        embedding = np.pad(embedding, ((0, 0), (0, 2 - embedding.shape[1])))
    return embedding


def compute_embedding(
    result: dict[str, np.ndarray],
    method: str,
    *,
    tsne_perplexity: float = 30.0,
    isomap_n_neighbors: int = 5,
    diffusion_median_scale: float = 1.0,
    random_state: int = 0,
) -> np.ndarray:
    """Return an (N, 2) finite embedding of every condition (`none` included).

    `"mds"`/`"pca"` pass through the precomputed `embedding_2d`/
    `pca_embedding_2d` arrays unchanged (no refit -- these are already the
    exact reusable computation the analysis pipeline wrote). `"tsne"`/
    `"isomap"` are fresh unsupervised scikit-learn fits on
    `concatenated_vectors`. `"diffusion_map"` is a fresh fit directly on
    `distances` with a median-heuristic kernel bandwidth (tunable via
    `diffusion_median_scale`); see `_fit_diffusion_map`. `"lda_lambda"`/
    `"lda_alpha"`/`"lda_tau"` are supervised projections against that
    parameter's values; see `_fit_lda`.
    """
    if method == "mds":
        return np.asarray(result["embedding_2d"], dtype=float)
    if method == "pca":
        return np.asarray(result["pca_embedding_2d"], dtype=float)
    if method == "tsne":
        vectors = np.asarray(result["concatenated_vectors"], dtype=float)
        return _fit_tsne(vectors, perplexity=tsne_perplexity, random_state=random_state)
    if method == "isomap":
        vectors = np.asarray(result["concatenated_vectors"], dtype=float)
        return _fit_isomap(vectors, n_neighbors=isomap_n_neighbors)
    if method == "diffusion_map":
        return _fit_diffusion_map(result, median_scale=diffusion_median_scale)
    if method.startswith("lda_"):
        parameter = method[len("lda_") :]
        return _fit_lda(result, parameter)
    raise ValueError(
        f"Unsupported method {method!r}; choose one of {EMBEDDING_METHODS}"
    )


def dimension_labels(result: dict[str, np.ndarray]) -> np.ndarray:
    """One label per column of `concatenated_vectors`, in the same
    batch-major order `concatenate_batch_scores` produces (batch varies
    slowest): ``"batch{b}_{parameter_name}={v:g}"`` for signal metrics
    (`parameter_values` has more than one entry), ``"batch{b}"`` for fixed
    metrics (`parameter_values` has exactly one entry). Length equals
    ``batch_indices.size * parameter_values.size ==
    concatenated_vectors.shape[1]``.
    """
    batch_indices = np.asarray(result["batch_indices"])
    parameter_values = np.asarray(result["parameter_values"])
    is_signal = parameter_values.size > 1
    parameter_name = result["parameter_name"].item() if is_signal else ""
    labels = []
    for batch in batch_indices:
        if is_signal:
            for value in parameter_values:
                labels.append(f"batch{batch}_{parameter_name}={value:g}")
        else:
            labels.append(f"batch{batch}")
    return np.asarray(labels)


def parallel_coordinates_data(
    result: dict[str, np.ndarray],
    color_parameter: str,
    *,
    include_none: bool = True,
) -> dict[str, np.ndarray]:
    """Raw per-(batch, parameter-value) signal, with no dimensionality
    reduction at all -- every dimension shown directly, colored by a chosen
    swept parameter.

    `include_none=False` drops the `none` row and re-aligns every returned
    array; the raw signal usually renders more legibly without the `none`
    baseline mixed in among a much smaller set of parameter-colored lines.
    """
    if color_parameter not in PARAMETER_KEYS:
        raise ValueError(
            f"Unknown parameter {color_parameter!r}; "
            f"choose one of {tuple(PARAMETER_KEYS)}"
        )
    values = np.asarray(result["concatenated_vectors"], dtype=float)
    color_values = np.asarray(result[PARAMETER_KEYS[color_parameter]], dtype=float)
    condition_names = np.asarray(result["condition_names"])
    is_none = np.asarray(result["is_none"], dtype=bool)

    if not include_none:
        mask = ~is_none
        values = values[mask]
        color_values = color_values[mask]
        condition_names = condition_names[mask]
        is_none = is_none[mask]

    return {
        "values": values,
        "dimension_labels": dimension_labels(result),
        "color_values": color_values,
        "condition_names": condition_names,
        "is_none": is_none,
    }


def reordered_distance_matrix(
    result: dict[str, np.ndarray],
    order_parameter: str,
) -> dict[str, np.ndarray]:
    """Sort `distances`' rows/columns by `order_parameter` -- a fast check of
    whether a metric's induced distances correlate with a parameter *before*
    trusting any embedding of it.

    `none` sorts first (via a ``-inf`` key), then ascending by
    `order_parameter`; ties are broken by the other two swept parameters.
    ``reordered["distances"] == original[np.ix_(order, order)]``.
    """
    if order_parameter not in PARAMETER_KEYS:
        raise ValueError(
            f"Unknown parameter {order_parameter!r}; "
            f"choose one of {tuple(PARAMETER_KEYS)}"
        )
    is_none = np.asarray(result["is_none"], dtype=bool)

    def key(name: str) -> np.ndarray:
        return np.where(is_none, -np.inf, np.asarray(result[PARAMETER_KEYS[name]]))

    other_a, other_b = OTHER_PARAMETERS[order_parameter]
    order = np.lexsort((key(other_b), key(other_a), key(order_parameter)))

    distances = np.asarray(result["distances"], dtype=float)
    return {
        "distances": distances[np.ix_(order, order)],
        "order": order,
        "condition_names": np.asarray(result["condition_names"])[order],
        "is_none": is_none[order],
        order_parameter: np.asarray(result[PARAMETER_KEYS[order_parameter]])[order],
    }


def channel_markers(result: dict[str, np.ndarray], parameter: str) -> dict[float, str]:
    """`plotting._encodings`'s lambda-marker mapping, generalized to any of
    lambda/alpha/tau. Maps each distinct level of `parameter` (over
    non-`none` conditions) to one `PLOTLY_SYMBOLS` entry. Raises if there
    are more distinct levels than available symbols.
    """
    if parameter not in PARAMETER_KEYS:
        raise ValueError(
            f"Unknown parameter {parameter!r}; choose one of {tuple(PARAMETER_KEYS)}"
        )
    is_none = np.asarray(result["is_none"], dtype=bool)
    levels = np.unique(np.asarray(result[PARAMETER_KEYS[parameter]])[~is_none])
    if len(levels) > len(PLOTLY_SYMBOLS):
        raise ValueError(f"Too many {parameter} levels for distinct markers")
    return {float(value): PLOTLY_SYMBOLS[index] for index, value in enumerate(levels)}


def channel_colors(result: dict[str, np.ndarray], parameter: str) -> dict[float, str]:
    """`channel_markers`'s mapping, but for discrete colors instead of
    marker symbols: maps each distinct level of `parameter` (over non-
    `none` conditions) to one `CHANNEL_COLORS` entry, so a chosen "Color:"
    parameter reads as a small discrete legend rather than a continuous
    colorbar. Raises if there are more distinct levels than available
    colors.
    """
    if parameter not in PARAMETER_KEYS:
        raise ValueError(
            f"Unknown parameter {parameter!r}; choose one of {tuple(PARAMETER_KEYS)}"
        )
    is_none = np.asarray(result["is_none"], dtype=bool)
    levels = np.unique(np.asarray(result[PARAMETER_KEYS[parameter]])[~is_none])
    if len(levels) > len(CHANNEL_COLORS):
        raise ValueError(f"Too many {parameter} levels for distinct colors")
    return {float(value): CHANNEL_COLORS[index] for index, value in enumerate(levels)}


def channel_sizes(
    result: dict[str, np.ndarray],
    parameter: str,
    *,
    low: float = 7.0,
    high: float = 13.0,
    none_size: float = 10.0,
) -> np.ndarray:
    """`plotting._point_sizes`, generalized to any of lambda/alpha/tau;
    `none` rows always get `none_size` regardless of `parameter`.

    Defaults are plotly marker *diameters* in pixels, not
    `plotting._point_sizes`' matplotlib `scatter(s=...)` *areas* in
    points**2 -- reusing the matplotlib range (45-150) verbatim here would
    make every marker 45-150px across.
    """
    if parameter not in PARAMETER_KEYS:
        raise ValueError(
            f"Unknown parameter {parameter!r}; choose one of {tuple(PARAMETER_KEYS)}"
        )
    is_none = np.asarray(result["is_none"], dtype=bool)
    values = np.asarray(result[PARAMETER_KEYS[parameter]])
    levels = np.unique(values[~is_none])
    if len(levels) == 1:
        mapped = {float(levels[0]): (low + high) / 2.0}
    else:
        mapped = {
            float(value): float(size)
            for value, size in zip(levels, np.linspace(low, high, len(levels)))
        }
    return np.asarray(
        [
            none_size if is_none_row else mapped[float(value)]
            for is_none_row, value in zip(is_none, values)
        ]
    )


def hover_labels(result: dict[str, np.ndarray]) -> np.ndarray:
    """One ``"<condition_name> (lambda=.., alpha=.., tau=..)"`` string per
    row (``"<condition_name>"`` alone for `none`), shared across every
    plotly view's hovertext."""
    condition_names = np.asarray(result["condition_names"])
    is_none = np.asarray(result["is_none"], dtype=bool)
    lambda_values = np.asarray(result["lambda_values"])
    alpha_values = np.asarray(result["alpha_values"])
    tau_values = np.asarray(result["tau_values"])
    labels = []
    for name, none_row, lambda_value, alpha_value, tau_value in zip(
        condition_names, is_none, lambda_values, alpha_values, tau_values
    ):
        if none_row:
            labels.append(str(name))
        else:
            labels.append(
                f"{name} (lambda={lambda_value:g}, alpha={alpha_value:g}, "
                f"tau={tau_value:g})"
            )
    return np.asarray(labels)


def _correlate(values: np.ndarray, targets: np.ndarray, method: str) -> float:
    if method == "pearson":
        return float(pearsonr(values, targets).statistic)
    if method == "spearman":
        return float(spearmanr(values, targets).statistic)
    raise ValueError(
        f"Unknown correlation method {method!r}; choose one of {CORRELATION_METHODS}"
    )


def _metric_correlation_row(
    result: dict[str, np.ndarray],
    parameter: str,
    method: str,
    common_points: int,
) -> np.ndarray:
    """One heatmap row: `result`'s batch-mean score at every one of its own
    `parameter_values` entries, correlated (over non-`none` conditions)
    against `parameter`, then interpolated onto a `common_points`-point
    relative sweep-position axis (0 = this metric's own sweep minimum, 1 =
    its own maximum) so metrics with different native grid sizes -- e.g. 30
    log-spaced rbf_sigma points vs. 8 k values -- become directly comparable
    rows in one heatmap. A fixed-mode metric (`parameter_values` has exactly
    one entry) has no sweep position of its own, so its single correlation
    value is simply repeated across every column.
    """
    is_none = np.asarray(result["is_none"], dtype=bool)
    transformed = ~is_none
    target = np.asarray(result[PARAMETER_KEYS[parameter]], dtype=float)[transformed]
    scores = np.asarray(result["scores"], dtype=float)[transformed]  # (n, batches, P)
    batch_mean = scores.mean(axis=1)  # (n, P)
    n_points = batch_mean.shape[1]

    curve = np.asarray(
        [_correlate(batch_mean[:, j], target, method) for j in range(n_points)]
    )
    if n_points == 1:
        return np.full(common_points, curve[0])
    if n_points == common_points:
        return curve
    own_x = np.linspace(0.0, 1.0, n_points)
    common_x = np.linspace(0.0, 1.0, common_points)
    return np.interp(common_x, own_x, curve)


def _metric_auc_correlation(
    result: dict[str, np.ndarray], parameter: str, method: str
) -> float:
    """Correlate each condition's AUC over its own sweep (via the shared
    `signal_aggregation.aggregate_signal_auc`, not a reimplementation)
    against `parameter`."""
    is_none = np.asarray(result["is_none"], dtype=bool)
    transformed = ~is_none
    target = np.asarray(result[PARAMETER_KEYS[parameter]], dtype=float)[transformed]
    scores = np.asarray(result["scores"], dtype=float)[transformed]
    batch_mean = scores.mean(axis=1)
    parameter_values = np.asarray(result["parameter_values"], dtype=float)
    auc = aggregate_signal_auc(parameter_values, batch_mean)
    return _correlate(auc, target, method)


def parameter_correlation_heatmap_data(
    results: dict[str, dict[str, np.ndarray]],
    parameter: str,
    method: str,
    *,
    signal_metrics: tuple[str, ...] = CORRELATION_SIGNAL_METRICS,
    fixed_metrics: tuple[str, ...] = CORRELATION_FIXED_METRICS,
    common_points: int | None = None,
) -> dict[str, np.ndarray]:
    """Heatmap data: how well each metric's signal tracks `parameter`.

    `results` maps metric_id -> that metric's already-loaded NPZ result
    dict; entries named in `signal_metrics`/`fixed_metrics` but absent from
    `results` are skipped. For each present signal metric this adds two
    rows: the metric's own per-sweep-position correlation curve
    (`_metric_correlation_row`), and a constant `"<metric_id> (AUC)"` row
    correlating each condition's AUC over its own sweep instead
    (`_metric_auc_correlation`). Each present fixed metric adds one constant
    row (it has no sweep of its own). `common_points` defaults to the
    largest `parameter_values` size among the present signal metrics, so a
    metric whose own grid already has that many points passes through
    unchanged and only the coarser grids are interpolated.

    Returns ``{"rows": (n_rows,) str array, "matrix": (n_rows,
    common_points) float array of correlation coefficients, "x":
    (common_points,) relative sweep-position array in [0, 1]}``.
    """
    if parameter not in PARAMETER_KEYS:
        raise ValueError(
            f"Unknown parameter {parameter!r}; choose one of {tuple(PARAMETER_KEYS)}"
        )
    if method not in CORRELATION_METHODS:
        raise ValueError(
            f"Unknown correlation method {method!r}; choose one of {CORRELATION_METHODS}"
        )

    present_signal = [
        (metric_id, results[metric_id])
        for metric_id in signal_metrics
        if metric_id in results
    ]
    if common_points is None:
        common_points = max(
            (
                int(np.asarray(result["parameter_values"]).size)
                for _, result in present_signal
            ),
            default=1,
        )
        common_points = max(common_points, 1)

    row_labels: list[str] = []
    rows: list[np.ndarray] = []
    for metric_id, result in present_signal:
        row_labels.append(metric_id)
        rows.append(_metric_correlation_row(result, parameter, method, common_points))
        row_labels.append(f"{metric_id} (AUC)")
        rows.append(
            np.full(common_points, _metric_auc_correlation(result, parameter, method))
        )
    for metric_id in fixed_metrics:
        result = results.get(metric_id)
        if result is None:
            continue
        row_labels.append(metric_id)
        rows.append(_metric_correlation_row(result, parameter, method, common_points))

    return {
        "rows": np.asarray(row_labels),
        "matrix": np.asarray(rows, dtype=float),
        "x": np.linspace(0.0, 1.0, common_points),
    }


# Fields that must be identical across every dataset being concatenated by
# concatenate_dataset_results: the condition grid and, for signal metrics,
# the metric's own sweep. If these differ, the datasets' batch rows do not
# line up and concatenating them would silently misalign the analysis.
_CONCATENATION_INVARIANT_KEYS: tuple[str, ...] = (
    "metric_id",
    "condition_names",
    "is_none",
    "lambda_values",
    "alpha_values",
    "tau_values",
    "parameter_name",
    "parameter_values",
)


def concatenate_dataset_results(
    results_by_dataset: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Concatenate one metric's per-dataset results (same model, same
    metric) along the batch axis, matching how batches within one dataset
    are already concatenated batch-major into `concatenated_vectors` --
    "all datasets" is just a wider version of the same thing.

    Every dataset must share identical condition and sweep grids
    (`_CONCATENATION_INVARIANT_KEYS`); a mismatch raises `ValueError` naming
    the first differing dataset rather than silently concatenating
    misaligned rows. `distances`, `embedding_2d`, and `pca_embedding_2d` are
    not available precomputed for this wider concatenation (the canonical
    pipeline never computes it), so they are refit here via
    `scoring.score_distances`/`fit_metric_mds`/`fit_signal_pca` -- the exact
    same functions the canonical per-dataset pipeline uses, just called on
    more data. `config` is the run's full loaded configuration (or just its
    `"mds"` sub-mapping; `fit_metric_mds` accepts either).

    Returns a result dict with the same keys as a single-dataset result,
    plus `batch_datasets`: a `(total_batches,)` string array recording which
    dataset each (renumbered, 0-based) `batch_indices` entry came from.
    """
    if not results_by_dataset:
        raise ValueError("results_by_dataset must not be empty")
    dataset_names = list(results_by_dataset)
    reference_name = dataset_names[0]
    reference = results_by_dataset[reference_name]

    for key in _CONCATENATION_INVARIANT_KEYS:
        reference_value = np.asarray(reference[key])
        # equal_nan is only valid for floating-dtype arrays (condition_names,
        # parameter_name, and metric_id are strings) -- lambda/alpha/tau
        # legitimately contain NaN on the `none` row and must still compare
        # equal there.
        equal_nan = np.issubdtype(reference_value.dtype, np.floating)
        for name in dataset_names[1:]:
            value = np.asarray(results_by_dataset[name][key])
            if value.shape != reference_value.shape or not np.array_equal(
                value, reference_value, equal_nan=equal_nan
            ):
                raise ValueError(
                    f"Dataset {name!r} has a different {key!r} than "
                    f"{reference_name!r}; cannot concatenate datasets whose "
                    "condition or sweep grids do not match"
                )

    per_dataset_scores = [
        np.asarray(results_by_dataset[name]["scores"], dtype=float)
        for name in dataset_names
    ]
    scores = np.concatenate(per_dataset_scores, axis=1)  # (n, total_batches, P)
    total_batches = scores.shape[1]
    batch_datasets = np.concatenate(
        [
            np.full(per_dataset_scores[index].shape[1], name)
            for index, name in enumerate(dataset_names)
        ]
    )

    vectors = concatenate_batch_scores(scores)
    distances = score_distances(vectors)
    embedding_2d, _, _, _ = fit_metric_mds(distances, config, n_components=2)
    pca_embedding_2d, _, _, _ = fit_signal_pca(vectors, n_components=2)

    merged = {key: np.asarray(reference[key]) for key in _CONCATENATION_INVARIANT_KEYS}
    merged.update(
        batch_indices=np.arange(total_batches),
        batch_datasets=batch_datasets,
        scores=scores,
        concatenated_vectors=vectors,
        distances=distances,
        embedding_2d=embedding_2d,
        pca_embedding_2d=pca_embedding_2d,
    )
    return merged
