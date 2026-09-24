#!/usr/bin/env python3
"""Build paper-ready figures from a computed glocal sweep MDS run."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from manifold_repsim.experiments.glocal_sweep_mds import load_computed_results
from manifold_repsim.experiments.glocal_sweep_mds.config import load_config
from manifold_repsim.experiments.glocal_sweep_mds.paper_figures import (
    DEFAULT_OUTPUT_ROOT,
    build_all,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/glocal_sweep_mds.yaml",
        help="Versioned YAML experiment configuration (must already be computed)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory to write paper-ready figures into",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    groups, results = load_computed_results(config)
    build_all(groups, results, Path(args.output_dir), config=config)


if __name__ == "__main__":
    main()
