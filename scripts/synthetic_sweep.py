#!/usr/bin/env python3
"""Run the configured synthetic permuted-Gaussian MDS experiment."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from manifold_repsim.experiments.permuted_gaussian_mds import (  # noqa: E402,F401
    analysis_fingerprint,
    compatible_analysis_fingerprints,
    compatible_dataset_fingerprints,
    compute_fixed_similarity_matrix,
    compute_signal_similarity_matrices,
    config_fingerprint,
    dataset_fingerprint,
    direct_similarity_distance,
    fit_metric_mds,
    generate_dataset,
    grid_parameter_values,
    load_config,
    load_dataset_artifacts,
    main,
    output_paths,
    parameter_value_label,
    run,
    scalar_base_relative_distance,
    signal_base_relative_distance,
    validate_config,
    validate_dissimilarity_matrix,
    visual_encodings,
)


if __name__ == "__main__":
    main()
