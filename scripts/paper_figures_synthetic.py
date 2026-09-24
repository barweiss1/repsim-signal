#!/usr/bin/env python3
"""Build paper-ready figures from an analyzed synthetic MDS experiment run."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from manifold_repsim.experiments.permuted_gaussian_mds.config import load_config
from manifold_repsim.experiments.permuted_gaussian_mds.paper_figures import (
    DEFAULT_OUTPUT_ROOT,
    build_all,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/synthetic_permuted_gaussian_mds.yaml",
        help="Versioned YAML experiment configuration (must already be analyzed)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory to write paper-ready figures into",
    )
    args = parser.parse_args(argv)
    build_all(load_config(args.config), Path(args.output_dir))


if __name__ == "__main__":
    main()
