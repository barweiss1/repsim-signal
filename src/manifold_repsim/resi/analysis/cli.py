from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .config import BOX_STYLES, RANK_SCALES, RANK_SORTS
from .settings import AnalysisSettings, load_analysis_settings
from .workflows import analyze_campaign


def split_measures(values: list[str] | None) -> list[str] | None:
    """Flatten repeated and comma-separated ``--measures`` values."""
    if not values:
        return None
    names = [name.strip() for value in values for name in str(value).split(",")]
    return [name for name in names if name]


def add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach the reporting options shared by both analysis entry points."""
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--analysis-config",
        metavar="PATH",
        help=(
            "Analysis settings YAML naming the measures to report "
            "(configs/resi_<domain>_analysis.yaml)."
        ),
    )
    parser.add_argument(
        "--measures",
        action="append",
        metavar="NAME[,NAME...]",
        help=(
            "Restrict the analysis to these measures (repeatable, or comma "
            "separated). Overrides --analysis-config. A named measure that is "
            "absent from the results is an error."
        ),
    )
    parser.add_argument(
        "--rank-sort",
        choices=RANK_SORTS,
        help=(
            "Sort the rank boxplots by each measure's median (default), its "
            "mean, or its 90th-percentile rank, which orders on how badly a "
            "measure does when it does badly. Overrides --analysis-config. "
            "Under quantile90 that quantile is drawn on every box; the mean "
            "is always drawn regardless of which of the three is sorted on."
        ),
    )
    parser.add_argument(
        "--box-style",
        choices=BOX_STYLES,
        help=(
            "Draw each measure's rank distribution as a quartile box with "
            "percentile whiskers (default) or as a boxen: nested bands at the "
            "10th, 30th, 50th, 70th and 90th percentiles, narrowing as they "
            "reach into the tails, with whiskers to the true min and max. "
            "Overrides --analysis-config. This changes the drawing only, "
            "never the ordering or the underlying numbers."
        ),
    )
    parser.add_argument(
        "--rank-scale",
        choices=RANK_SCALES,
        help=(
            "Rank measures on the normalized within-case rank (default) or on "
            "the raw one. Overrides --analysis-config. Both statistics are "
            "always written; this selects the sort order and the plotted axis."
        ),
    )


def print_measure_exclusions(out_dir: str | Path) -> None:
    """Report which measures the coverage thresholds dropped, and from where.

    A run that quietly reports a smaller field than asked for is the failure
    this filtering exists to prevent, so every exclusion is stated on stdout
    rather than only written to `measure_exclusions.csv`.
    """
    path = Path(out_dir) / "measure_exclusions.csv"
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("scope")]
    scope_rows = [
        row for row in rows if row.get("reason") == "case_coverage_below_threshold"
    ]
    for row in scope_rows:
        # Counts round-trip through a column that also holds blanks, so pandas
        # writes them as floats; report them as the counts they are.
        present = int(float(row["cases_present"]))
        total = int(float(row["cases_in_scope"]))
        print(
            f"excluded={row['metric']} scope={row['scope']} "
            f"cases={present}/{total} "
            f"coverage={float(row['case_coverage']):.3f}"
        )
    dropped_cases = len(rows) - len(scope_rows)
    if dropped_cases:
        print(f"dropped_cases={dropped_cases} (measures too NaN within a case)")


def resolve_settings(analysis_config: str | None) -> AnalysisSettings:
    """Load the settings file when one was named, else use the defaults."""
    if not analysis_config:
        return AnalysisSettings()
    return load_analysis_settings(analysis_config)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a prepared ReSi campaign from its explicit manifests."
    )
    add_analysis_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        output = analyze_campaign(
            args.campaign,
            args.run_name,
            allow_incomplete=args.allow_incomplete,
            measures=split_measures(args.measures),
            rank_scale=args.rank_scale,
            rank_sort=args.rank_sort,
            box_style=args.box_style,
            settings=resolve_settings(args.analysis_config),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print_measure_exclusions(output)
    print(output)


if __name__ == "__main__":
    main()


__all__ = [
    "add_analysis_arguments",
    "main",
    "print_measure_exclusions",
    "resolve_settings",
    "split_measures",
]
