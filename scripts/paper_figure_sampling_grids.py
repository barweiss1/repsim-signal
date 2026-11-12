#!/usr/bin/env python3
"""Build the paper figure comparing the log, sqrt, and linear sampling grids."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from manifold_repsim.experiments.sampling_grid_figure import (
    DEFAULT_MAXIMUM,
    DEFAULT_MINIMUM,
    DEFAULT_NUM,
    DEFAULT_OUTPUT_ROOT,
    build_all,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--min", type=float, default=DEFAULT_MINIMUM, dest="minimum")
    parser.add_argument("--max", type=float, default=DEFAULT_MAXIMUM, dest="maximum")
    parser.add_argument("--num", type=int, default=DEFAULT_NUM)
    args = parser.parse_args(argv)
    build_all(
        Path(args.output_dir),
        minimum=args.minimum,
        maximum=args.maximum,
        num=args.num,
    )


if __name__ == "__main__":
    main()
