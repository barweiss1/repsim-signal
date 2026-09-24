"""Path resolution shared by ReSi campaign preparation and status reporting."""

from __future__ import annotations

import os
import re
from pathlib import Path

# The campaign defaults and relative campaign paths are repository-relative, so
# this stays anchored to the checkout even though the package is importable from
# an installed location. src/manifold_repsim/resi/paths.py -> four levels up.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = REPO_ROOT / "data" / "resi" / "runs"
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "figures" / "resi"


def expanded_path(value: str | Path, *, root: Path | None = None) -> Path:
    """Expand variables and user prefixes, resolving relative paths under root."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute() and root is not None:
        path = root / path
    return path.resolve()


def require_env_path(name: str) -> Path:
    """Read a required directory path from the environment."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"{name} must point to the ReSi checkout"
            if name == "RESI_DIR"
            else f"{name} must point to ReSi's experiments directory"
        )
    path = expanded_path(value)
    if not path.is_dir():
        raise ValueError(f"{name} is not a directory: {path}")
    return path


def slug(value: str) -> str:
    """Reduce a name to a lowercase, filesystem-safe path component."""
    result = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip()).strip("-_.").lower()
    if not result:
        raise ValueError(f"Cannot create a path component from {value!r}")
    return result


__all__ = [
    "DEFAULT_ANALYSIS_ROOT",
    "DEFAULT_RUN_ROOT",
    "REPO_ROOT",
    "expanded_path",
    "require_env_path",
    "slug",
]
