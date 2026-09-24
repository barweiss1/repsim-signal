#!/usr/bin/env python
"""Report whether an architecture has real (non-empty) values in archived
ReSi baseline parquets.

Usage:
  python scripts/check_baseline_architecture_coverage.py \
      --architecture PGNN \
      --parquet graphs/results_label_test_cora.parquet \
      --parquet graphs/results_shortcut_test_cora.parquet \
      --parquet graphs/results_layer_test_cora.parquet \
      --parquet graphs/results_output_correlation_test_cora.parquet

Relative --parquet paths are resolved against $REP_SIM/results, matching how
campaign baselines[].path is resolved (see docs/resi_cluster.md).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    rep_sim = os.environ.get("REP_SIM")
    if not rep_sim:
        raise SystemExit(
            "REP_SIM is not set; export it or pass an absolute --parquet path"
        )
    return Path(rep_sim) / "results" / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", required=True)
    parser.add_argument(
        "--parquet", action="append", required=True, dest="parquets", metavar="PATH"
    )
    args = parser.parse_args()

    value_column = None
    for path_str in args.parquets:
        path = _resolve(path_str)
        print(f"== {path} ==")
        if not path.is_file():
            print("  NOT FOUND")
            continue
        df = pd.read_parquet(path)
        if "architecture" not in df.columns:
            print(f"  no 'architecture' column; columns = {list(df.columns)}")
            continue
        rows = df[df["architecture"] == args.architecture]
        if value_column is None:
            value_column = "value" if "value" in df.columns else "corr"
        real = (
            int(rows[value_column].notna().sum()) if value_column in rows.columns else 0
        )
        print(f"  {len(rows)} {args.architecture} rows, {real} with a real value")
        if len(rows) and real == 0:
            print(f"  sample row: {rows.iloc[0].to_dict()}")


if __name__ == "__main__":
    main()
