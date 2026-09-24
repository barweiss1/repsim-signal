#!/usr/bin/env python3
"""Run the configured glocal sweep MDS experiment."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from manifold_repsim.experiments.glocal_sweep_mds import (  # noqa: E402,F401
    concatenate_batch_scores,
    discover,
    load_config,
    main,
    parse_glocal_transform,
    run,
    score_distances,
    validate_config,
)


if __name__ == "__main__":
    main()
