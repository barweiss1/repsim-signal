#!/usr/bin/env python3
"""Smoke-test the Enroot/NGC ReSi container."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _module_version(module_name: str) -> str:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is None:
        version = getattr(module, "version", None)
    return str(version) if version is not None else "<unknown>"


def _print_import(module_name: str, label: str | None = None) -> object:
    module = importlib.import_module(module_name)
    print(f"{label or module_name} {_module_version(module_name)}")
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resi-dir",
        type=Path,
        default=Path(os.environ.get("RESI_DIR", REPO_ROOT.parent / "resi")),
        help="ReSi checkout registered in the container.",
    )
    parser.add_argument(
        "--expected-numpy",
        default=os.environ.get("EXPECTED_NUMPY_VERSION", "1.24.4"),
        help="Expected NumPy version for the NGC 24.06 container.",
    )
    args = parser.parse_args(argv)
    resi_dir = args.resi_dir.resolve()
    for path in (REPO_ROOT, REPO_ROOT / "src", resi_dir):
        path_text = os.fspath(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)

    torch = _print_import("torch")
    torchvision = _print_import("torchvision")
    numpy = _print_import("numpy")
    torch_geometric = _print_import("torch_geometric")
    cv2 = _print_import("cv2")
    albumentations = _print_import("albumentations")
    _print_import("timm")
    _print_import("medmnist")
    _print_import("datasets")
    _print_import("transformers")

    if not numpy.__version__.startswith("1."):
        raise SystemExit(f"Expected NumPy 1.x, got {numpy.__version__}")
    if args.expected_numpy and numpy.__version__ != args.expected_numpy:
        raise SystemExit(
            f"Expected NumPy {args.expected_numpy}, got {numpy.__version__}"
        )
    if not hasattr(getattr(cv2, "dnn", None), "DictValue"):
        raise SystemExit("Broken OpenCV install: cv2.dnn.DictValue is missing.")
    if not str(getattr(albumentations, "__version__", "")).startswith("1."):
        raise SystemExit(
            f"Expected Albumentations 1.x for ReSi vision transforms, got {albumentations.__version__}"
        )

    import manifold_repsim.metrics  # noqa: F401
    from manifold_repsim.resi.measures import RESI_MEASURE_SPECS
    from manifold_repsim.resi.runtime import register_manifold_measures

    registered = register_manifold_measures(resi_dir)
    expected_count = len(RESI_MEASURE_SPECS)
    if len(registered) != expected_count:
        raise SystemExit(
            f"Expected {expected_count} runtime manifold measures, got {len(registered)}"
        )
    from repsim.run import run

    import repsim.nlp  # noqa: F401
    import repsim.utils  # noqa: F401

    print("torch_cuda", getattr(torch.version, "cuda", None))
    print("torch_cuda_available", torch.cuda.is_available())
    print("torchvision_module", torchvision.__name__)
    print("torch_geometric_module", torch_geometric.__name__)
    print("repsim.run", run)
    print("manifold_repsim.metrics imported; manifold measures registered in memory.")
    print("ReSi container smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
