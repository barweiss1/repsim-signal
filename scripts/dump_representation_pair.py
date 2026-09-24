"""Dump one real ReSi representation pair to .npy for offline numerical study.

Diagnostic only. This reads models through ReSi exactly as a compute task does,
takes the final-layer representation of two named models, and writes them plus a
small JSON describing what was dumped. Nothing in the ReSi checkout is modified.

Example:

    python scripts/dump_representation_pair.py \
      --resi-dir "$RESI_DIR" \
      --representation-dataset ogbn-arxiv \
      --source GRAPHS_GAT_ogbn-arxiv_Normal_0 \
      --target GRAPHS_GAT_ogbn-arxiv_RandomLabels_100_0 \
      --out-dir /tmp/repdump
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from manifold_repsim.resi.runtime import _load_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resi-dir", required=True)
    parser.add_argument("--representation-dataset", required=True)
    parser.add_argument("--source", required=True, help="Source model id")
    parser.add_argument("--target", required=True, help="Target model id")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _final_layer(model, representation_dataset) -> tuple[np.ndarray, str]:
    """Return the final-layer representation and its ReSi shape string.

    The value is read through ``value_attr_name()`` rather than a hardcoded
    attribute, so this keeps matching whatever a real comparison is handed.
    """
    single = model.get_representation(representation_dataset).representations[-1]
    values = getattr(single, single.value_attr_name())
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values), str(getattr(single, "shape", "unknown"))


def main() -> None:
    args = _parse_args()
    resi_dir = Path(args.resi_dir).expanduser().resolve()
    _load_registry(resi_dir)
    run_module = importlib.import_module("repsim.run")

    by_id = {model.id: model for model in run_module.ALL_TRAINED_MODELS}
    missing = [name for name in (args.source, args.target) if name not in by_id]
    if missing:
        raise SystemExit(
            f"Unknown model id(s): {missing}\n"
            f"First few known ids: {sorted(by_id)[:5]}"
        )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"representation_dataset": args.representation_dataset}
    for role, model_id in (("source", args.source), ("target", args.target)):
        values, resi_shape = _final_layer(by_id[model_id], args.representation_dataset)
        path = out_dir / f"{role}.npy"
        np.save(path, values)
        meta[role] = {
            "model_id": model_id,
            "array_shape": list(values.shape),
            "resi_shape": resi_shape,
            "dtype": str(values.dtype),
            "path": str(path),
        }
        print(
            f"{role}: {model_id} array={values.shape} resi_shape={resi_shape} "
            f"dtype={values.dtype} -> {path}"
        )

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {out_dir}/meta.json")


if __name__ == "__main__":
    main()
