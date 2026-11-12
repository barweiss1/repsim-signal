"""Similarity analysis, checkpoints, and result assembly for the MDS experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import save_json
from ..artifacts import save_npz
from .config import PARAMETERS
from .config import SIGNAL_METRICS
from .config import analysis_fingerprint
from .config import compatible_analysis_fingerprints
from .config import compatible_dataset_fingerprints
from .config import config_fingerprint
from .config import dataset_fingerprint
from .config import output_paths
from .config import require
from .config import save_run_config
from .dataset import build_base_dataset
from .dataset import load_dataset_artifacts
from .dataset import metadata
from .dataset import transform_config
from .mds import fit_metric_mds
from .plotting import plot_dataset_examples
from .plotting import plot_metric_results
from .plotting import plot_signal_sweeps
from .similarity import compute_fixed_similarity_matrix
from .similarity import compute_signal_similarity_matrices
from .similarity import direct_similarity_distance
from .similarity import scalar_base_relative_distance
from .similarity import signal_base_relative_distance


def result_path(config: dict[str, Any], metric_id: str) -> Path:
    """Return the canonical result artifact for one metric identifier."""
    return output_paths(config)[0] / "metrics" / f"{metric_id}.npz"


def reusable_result(
    path: Path,
    fingerprints: str | set[str],
    recompute: bool,
) -> bool:
    """Validate whether a result can be reused for this configuration."""
    if not path.exists() or recompute:
        return False
    expected = {fingerprints} if isinstance(fingerprints, str) else fingerprints
    with np.load(path) as saved:
        require(
            saved["analysis_fingerprint"].item() in expected,
            f"Metric result does not match config: {path}; enable recompute_metrics",
        )
    return True


def save_metric_result(
    config: dict[str, Any],
    metric_id: str,
    metric_name: str,
    fingerprint: str,
    base_similarity: np.ndarray,
    pairwise_similarity: np.ndarray,
    *,
    base_signal: np.ndarray | None = None,
    pairwise_signal: np.ndarray | None = None,
    signal_parameters: np.ndarray | None = None,
) -> Path:
    """Assemble distances, MDS embeddings, and established NPZ fields."""
    reference_distance = (
        signal_base_relative_distance(base_signal)
        if base_signal is not None
        else scalar_base_relative_distance(base_similarity)
    )
    direct_distance = direct_similarity_distance(pairwise_similarity)
    reference = fit_metric_mds(reference_distance, config)
    direct = fit_metric_mds(direct_distance, config)
    arrays = {
        "metric_id": np.array(metric_id),
        "metric_name": np.array(metric_name),
        "analysis_fingerprint": np.array(fingerprint),
        "base_similarity": base_similarity,
        "pairwise_similarity": pairwise_similarity,
        "base_relative_distance": reference_distance,
        "direct_distance": direct_distance,
    }
    for prefix, result in (("base_relative", reference), ("direct", direct)):
        arrays.update(
            {
                f"{prefix}_embedding": result[0],
                f"{prefix}_raw_stress": np.array(result[1]),
                f"{prefix}_normalized_stress": np.array(result[2]),
                f"{prefix}_n_iter": np.array(result[3]),
            }
        )
    for name, value in (
        ("base_signal", base_signal),
        ("pairwise_signal", pairwise_signal),
        ("signal_parameters", signal_parameters),
    ):
        if value is not None:
            arrays[name] = value
    path = result_path(config, metric_id)
    save_npz(path, **arrays)
    return path


def signal_values(config: dict[str, Any]) -> np.ndarray:
    """Resolve the configured fixed metric-signal grid."""
    specification = config["metrics"]["signals"]
    function = np.geomspace if specification["scale"] == "log" else np.linspace
    return function(
        specification["min"],
        specification["max"],
        specification["num"],
    )


def outlier_signal_values(config: dict[str, Any]) -> np.ndarray:
    """Resolve the outlier sweep's own bandwidth grid.

    Separate from `signal_values` because a translated point only registers
    in an RBF kernel once the bandwidth is comparable to its distance. The
    parameter sweeps need no such reach, and widening their shared grid would
    change every other synthetic signal figure.
    """
    specification = config["metrics"]["outlier_sweep"]["signals"]
    function = np.geomspace if specification["scale"] == "log" else np.linspace
    return function(
        specification["min"],
        specification["max"],
        specification["num"],
    )


def _signal_sweep_values(
    parameter: str,
    config: dict[str, Any],
) -> np.ndarray:
    """Return every configured level for one grid parameter's isolated sweep.

    Matches the parameter's own `grid` entry in the config exactly (all of
    `n_permute.values`, or `linspace(min, max, num)` for a continuous
    parameter) instead of an arbitrary fixed-count display subset, so the
    sweep shown always matches what the config actually declares.
    """
    if parameter == "n_permute":
        values = np.asarray(sorted(set(config["grid"]["n_permute"]["values"])))
        if len(values) == 1:
            n_clusters = config["dataset"]["n_clusters"]
            values = np.asarray([0] + list(range(2, n_clusters + 1)))
    else:
        sweep = config["grid"][parameter]
        values = np.linspace(sweep["min"], sweep["max"], sweep["num"])
    return values


def outlier_direction(dim: int) -> np.ndarray:
    """The fixed unit direction a swept outlier is translated along.

    The first coordinate axis. Any fixed direction would do -- the clusters
    are sampled isotropically on a hypersphere, so no axis is special -- and
    pinning it to a constant keeps the sweep reproducible and independent of
    the dataset seed.
    """
    direction = np.zeros(dim, dtype=float)
    direction[0] = 1.0
    return direction


def analyze_outlier_sweep(config: dict[str, Any], scratch: Path) -> None:
    """Signal curves for base against itself with one point moved far away.

    A robustness probe rather than a dataset transform: the comparison is the
    base data against an exact copy of itself in which a single point has
    been translated along `outlier_direction` by each configured distance.
    Everything else is identical, so whatever the signal reports is the
    effect of that one outlier alone.
    """
    outlier = config["metrics"]["outlier_sweep"]
    specification = outlier["signals"]
    sigma_values = outlier_signal_values(config)
    distances = np.asarray(outlier["distances"], dtype=float)
    point_index = int(outlier["point_index"])
    checkpoint = scratch / "outlier_sweep.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = config_fingerprint(
        {
            "dataset": dataset_fingerprint(config),
            "signals": specification,
            "distances": distances.tolist(),
            "point_index": point_index,
            "fixed": config["metrics"]["fixed"],
        }
    )
    if (
        checkpoint.exists()
        and config["execution"]["checkpoint_signals"]
        and not config["execution"]["recompute_metrics"]
    ):
        with np.load(checkpoint) as saved:
            if saved["fingerprint"].item() == checkpoint_hash:
                return

    _, base = build_base_dataset(config)
    direction = outlier_direction(base.shape[1])
    datasets = [base]
    for distance in distances:
        moved = base.copy()
        moved[point_index] = moved[point_index] + distance * direction
        datasets.append(moved)
    datasets = np.stack(datasets)

    gathered: dict[str, list[np.ndarray]] = {name: [] for name in SIGNAL_METRICS}
    for index, sigma in enumerate(sigma_values):
        matrices = compute_signal_similarity_matrices(
            datasets,
            list(SIGNAL_METRICS),
            float(sigma),
            scratch / "outlier_sweep" / f"{index:04d}",
        )
        for name in SIGNAL_METRICS:
            # Every translated copy against base, which is row 0.
            gathered[name].append(matrices[name][1:, 0])

    # The configured fixed metrics at the same comparisons. Linear CKA is the
    # signals' sigma -> inf limit, and the figures draw it per distance so the
    # curves read against where they are heading. These comparisons are not
    # conditions on the swept grid, so no per-condition result holds them.
    fixed = {
        metric["id"]: compute_fixed_similarity_matrix(
            datasets,
            metric,
            scratch / "outlier_sweep" / metric["id"],
        )[1:, 0]
        for metric in config["metrics"]["fixed"]
    }
    save_npz(
        checkpoint,
        fingerprint=np.array(checkpoint_hash),
        values=distances,
        sigma=sigma_values,
        **{f"fixed_{key}": np.asarray(value, float) for key, value in fixed.items()},
        # (distance, sigma), matching the parameter sweeps' own layout.
        **{name: np.stack(gathered[name]).T for name in SIGNAL_METRICS},
    )


def analyze_signal_shapes(config: dict[str, Any], scratch: Path) -> None:
    """Signal curves for the single comparison the shape figures draw.

    Deliberately separate from `analyze_parameter_signal_sweeps`, whose
    levels come from `grid` and so can only name conditions the swept grid
    contains. This one computes its own transform from
    ``metrics.signal_shapes.fixed_parameters``, so the shape figures can sit
    at a condition the grid never visits without perturbing the grid, the
    sweeps, or any other figure.
    """
    specification = config["metrics"]["signals"]
    sigma_values = signal_values(config)
    shape_params = config["metrics"]["signal_shapes"]["fixed_parameters"]
    checkpoint = scratch / "signal_shapes.npz"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = config_fingerprint(
        {
            "dataset": dataset_fingerprint(config),
            "signals": specification,
            "fixed_parameters": shape_params,
        }
    )
    if (
        checkpoint.exists()
        and config["execution"]["checkpoint_signals"]
        and not config["execution"]["recompute_metrics"]
    ):
        with np.load(checkpoint) as saved:
            if saved["fingerprint"].item() == checkpoint_hash:
                return

    dataset, base = build_base_dataset(config)
    transformed = dataset.transform_current(
        transform_config(config, **shape_params)
    ).copy()
    datasets = np.stack([base, transformed])
    gathered: dict[str, list[float]] = {name: [] for name in SIGNAL_METRICS}
    for index, sigma in enumerate(sigma_values):
        matrices = compute_signal_similarity_matrices(
            datasets,
            list(SIGNAL_METRICS),
            float(sigma),
            scratch / "signal_shapes" / f"{index:04d}",
        )
        for name in SIGNAL_METRICS:
            # Row 1 against row 0: the transformed condition against base.
            gathered[name].append(float(matrices[name][1, 0]))

    # The configured fixed metrics at this same comparison. The shape
    # condition need not lie on the swept grid, so their saved per-condition
    # results cannot be indexed for it; the figures need linear CKA here as
    # the signals' sigma -> inf limit, so it is measured directly.
    fixed = {
        specification["id"]: float(
            compute_fixed_similarity_matrix(
                datasets,
                specification,
                scratch / "signal_shapes" / specification["id"],
            )[1, 0]
        )
        for specification in config["metrics"]["fixed"]
    }
    save_npz(
        checkpoint,
        fingerprint=np.array(checkpoint_hash),
        sigma=sigma_values,
        **{name: np.asarray(gathered[name], dtype=float) for name in SIGNAL_METRICS},
        **{f"fixed_{key}": np.array(value) for key, value in fixed.items()},
    )


def analyze_parameter_signal_sweeps(config: dict[str, Any], scratch: Path) -> None:
    """Plot signal-vs-rbf_sigma curves for one parameter swept at a time.

    Each of ``n_permute``, ``cluster_mixing_probability``, and ``noise_scale``
    is varied alone over every level declared in its own ``grid`` entry while
    the other two stay fixed at their configured
    ``metrics.signal_sweeps.fixed_parameters`` value (defaulting to
    ``base_transform`` when omitted), mirroring the per-parameter sweeps in
    the synthetic-experiment notebooks. Results are checkpointed per
    parameter so a rerun with an unchanged dataset need not recompute them.
    """
    specification = config["metrics"]["signals"]
    sigma_values = signal_values(config)
    base_params = config["metrics"]["signal_sweeps"]["fixed_parameters"]
    parts = scratch / "signal_sweep_parts"
    parts.mkdir(parents=True, exist_ok=True)
    dataset = base = None
    curves: dict[str, dict[str, Any]] = {}
    for parameter in PARAMETERS:
        values = _signal_sweep_values(parameter, config)
        checkpoint = parts / f"{parameter}.npz"
        checkpoint_hash = config_fingerprint(
            {
                "dataset": dataset_fingerprint(config),
                "signals": specification,
                "sweep_parameter": parameter,
                "sweep_values": values.tolist(),
                "fixed_parameters": base_params,
            }
        )
        curve = None
        if (
            checkpoint.exists()
            and config["execution"]["checkpoint_signals"]
            and not config["execution"]["recompute_metrics"]
        ):
            with np.load(checkpoint) as saved:
                if saved["fingerprint"].item() == checkpoint_hash:
                    curve = {key: saved[key].copy() for key in saved.files}
        if curve is None:
            if dataset is None:
                dataset, base = build_base_dataset(config)
            overrides = dict(base_params)
            datasets = [base]
            for value in values:
                overrides[parameter] = value
                datasets.append(
                    dataset.transform_current(
                        transform_config(config, **overrides)
                    ).copy()
                )
            datasets = np.stack(datasets)
            gathered = {name: [] for name in SIGNAL_METRICS}
            for index, sigma in enumerate(sigma_values):
                matrices = compute_signal_similarity_matrices(
                    datasets,
                    list(SIGNAL_METRICS),
                    float(sigma),
                    scratch / "signal_sweep" / parameter / f"{index:04d}",
                )
                for name in SIGNAL_METRICS:
                    gathered[name].append(matrices[name][1:, 0])
            curve = {
                "values": values,
                "sigma": sigma_values,
                **{name: np.stack(gathered[name]).T for name in SIGNAL_METRICS},
            }
            if config["execution"]["checkpoint_signals"]:
                save_npz(checkpoint, fingerprint=np.array(checkpoint_hash), **curve)
        curves[parameter] = curve
    plot_signal_sweeps(config, curves)


def analyze_signals(
    config: dict[str, Any],
    datasets: np.ndarray,
    selected: np.ndarray,
    scratch: Path,
) -> list[Path]:
    """Compute or resume all registered signal metric artifacts."""
    specification = config["metrics"]["signals"]
    fingerprints = {
        name: analysis_fingerprint(config, {"name": name, "signal": specification})
        for name in SIGNAL_METRICS
    }
    paths = [result_path(config, name) for name in SIGNAL_METRICS]
    if all(
        reusable_result(
            path,
            compatible_analysis_fingerprints(
                config,
                {"name": name, "signal": specification},
            ),
            config["execution"]["recompute_metrics"],
        )
        for name, path in zip(SIGNAL_METRICS, paths)
    ):
        return paths

    parameters = signal_values(config)
    checkpoint_hashes = {
        config_fingerprint({"dataset": dataset_hash, "signals": specification})
        for dataset_hash in compatible_dataset_fingerprints(config)
    }
    checkpoint_hash = config_fingerprint(
        {"dataset": dataset_fingerprint(config), "signals": specification}
    )
    gathered = {name: [] for name in SIGNAL_METRICS}
    parts = scratch / "signal_parts"
    parts.mkdir(parents=True, exist_ok=True)
    for index, sigma in enumerate(parameters):
        checkpoint = parts / f"{index:04d}.npz"
        matrices = None
        if (
            checkpoint.exists()
            and config["execution"]["checkpoint_signals"]
            and not config["execution"]["recompute_metrics"]
        ):
            with np.load(checkpoint) as saved:
                if saved["fingerprint"].item() in checkpoint_hashes:
                    matrices = {name: saved[name].copy() for name in SIGNAL_METRICS}
        if matrices is None:
            print(f"Signal {index + 1}/{len(parameters)}: rbf_sigma={sigma:.6g}")
            matrices = compute_signal_similarity_matrices(
                datasets,
                list(SIGNAL_METRICS),
                float(sigma),
                scratch / f"signal_{index:04d}",
            )
            if config["execution"]["checkpoint_signals"]:
                save_npz(
                    checkpoint,
                    fingerprint=np.array(checkpoint_hash),
                    **matrices,
                )
        for name in SIGNAL_METRICS:
            gathered[name].append(matrices[name])

    paths = []
    for name in SIGNAL_METRICS:
        signal = np.stack(gathered[name])
        base_signal = signal[:, selected, 0].T
        pairwise_signal = signal[:, selected][:, :, selected]
        paths.append(
            save_metric_result(
                config,
                name,
                name,
                fingerprints[name],
                base_signal.mean(axis=1),
                pairwise_signal.mean(axis=0),
                base_signal=base_signal,
                pairwise_signal=pairwise_signal,
                signal_parameters=parameters,
            )
        )
    return paths


def analyze_dataset(config: dict[str, Any]) -> list[Path]:
    """Run fixed and signal metric analysis over generated datasets."""
    save_run_config(config)
    base, transformed, labels, grid = load_dataset_artifacts(config)
    plot_dataset_examples(
        base,
        transformed,
        labels,
        grid,
        output_paths(config)[1],
        config,
    )
    datasets = np.concatenate((base[None], transformed))
    selected = (
        np.arange(len(datasets))
        if config["visualization"]["include_base"]
        else np.arange(1, len(datasets))
    )
    data_dir, _ = output_paths(config)
    scratch = data_dir / ".scratch"
    paths = []
    for specification in config["metrics"]["fixed"]:
        fingerprint = analysis_fingerprint(config, specification)
        path = result_path(config, specification["id"])
        if not reusable_result(
            path,
            compatible_analysis_fingerprints(config, specification),
            config["execution"]["recompute_metrics"],
        ):
            print(f"Fixed metric: {specification['id']}")
            similarity = compute_fixed_similarity_matrix(
                datasets,
                specification,
                scratch / specification["id"],
            )
            path = save_metric_result(
                config,
                specification["id"],
                specification["name"],
                fingerprint,
                similarity[selected, 0],
                similarity[np.ix_(selected, selected)],
            )
        paths.append(path)
    paths.extend(analyze_signals(config, datasets, selected, scratch))
    analyze_parameter_signal_sweeps(config, scratch)
    analyze_signal_shapes(config, scratch)
    analyze_outlier_sweep(config, scratch)

    encodings, rows = metadata(config, grid)
    base_mask = np.array([bool(rows[index]["is_base"]) for index in selected])
    selected_encodings = {
        key: value if key.endswith("_levels") else value[selected]
        for key, value in encodings.items()
    }
    plot_metric_results(config, paths, selected_encodings, base_mask)
    save_json(
        data_dir / "analysis_manifest.json",
        {
            "version": 1,
            "dataset_fingerprint": dataset_fingerprint(config),
            "metric_results": [str(path.relative_to(data_dir)) for path in paths],
            "include_base": config["visualization"]["include_base"],
        },
    )
    print(f"Saved {len(paths)} metric analyses under {data_dir}")
    return paths
