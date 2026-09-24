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
