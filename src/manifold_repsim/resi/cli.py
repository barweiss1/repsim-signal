"""Command line entry points for ReSi campaign preparation, execution, and analysis."""

from __future__ import annotations

import argparse
import sys

from .campaign import prepare_campaign


def build_parser() -> argparse.ArgumentParser:
    """Build the ``resi`` argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare, run, inspect, and analyze ReSi manifold campaigns."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Generate native ReSi configs and task manifests."
    )
    prepare.add_argument("--campaign", required=True)
    prepare.add_argument("--run-name", required=True)

    run = subparsers.add_parser("run-task", help="Run one zero-based manifest task.")
    run.add_argument("--manifest", required=True)
    run.add_argument("--index", required=True, type=int)

    status = subparsers.add_parser("status", help="Report manifest output completion.")
    status.add_argument("--manifest", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="Analyze explicit manifests for a campaign."
    )
    # Declared here rather than borrowed from `analysis.cli` so that building
    # the parser does not import the analysis package, keeping `prepare` and
    # `status` free of pandas and a plotting backend.
    analyze.add_argument("--campaign", required=True)
    analyze.add_argument("--run-name", required=True)
    analyze.add_argument("--allow-incomplete", action="store_true")
    analyze.add_argument(
        "--analysis-config",
        metavar="PATH",
        help=(
            "Analysis settings YAML naming the measures to report "
            "(configs/resi_<domain>_analysis.yaml)."
        ),
    )
    analyze.add_argument(
        "--measures",
        action="append",
        metavar="NAME[,NAME...]",
        help=(
            "Restrict the analysis to these measures (repeatable, or comma "
            "separated). Overrides --analysis-config. A named measure that is "
            "absent from the results is an error."
        ),
    )
    analyze.add_argument(
        "--rank-sort",
        # Spelled out for the same reason as the arguments above; the values
        # are `analysis.config.RANK_SORTS`, and a test keeps the two in sync.
        choices=("median", "quantile90", "mean"),
        help=(
            "Sort the rank boxplots by each measure's median (default), its "
            "mean, or its 90th-percentile rank, which orders on how badly a "
            "measure does when it does badly. Overrides --analysis-config. "
            "Under quantile90 that quantile is drawn on every box; the mean "
            "is always drawn regardless of which of the three is sorted on."
        ),
    )
    analyze.add_argument(
        "--box-style",
        # Spelled out for the same reason as the arguments above; the values
        # are `analysis.config.BOX_STYLES`, and a test keeps the two in sync.
        choices=("box", "boxen"),
        help=(
            "Draw each measure's rank distribution as a quartile box with "
            "percentile whiskers (default) or as a boxen: nested bands at the "
            "10th, 30th, 50th, 70th and 90th percentiles, narrowing as they "
            "reach into the tails, with whiskers to the true min and max. "
            "Overrides --analysis-config. This changes the drawing only, "
            "never the ordering or the underlying numbers."
        ),
    )
    analyze.add_argument(
        "--rank-scale",
        # Spelled out for the same reason as the arguments above; the values
        # are `analysis.config.RANK_SCALES`, and a test keeps the two in sync.
        choices=("normalized", "raw"),
        help=(
            "Rank measures on the normalized within-case rank (default) or on "
            "the raw one. Overrides --analysis-config. Both statistics are "
            "always written; this selects the sort order and the plotted axis."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Dispatch one subcommand, mapping expected failures to exit code 2.

    Execution and analysis are imported lazily so ``prepare`` and ``status`` do
    not pull in ReSi or plotting dependencies.
    """
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            compute, baseline = prepare_campaign(args.campaign, args.run_name)
            print(f"compute_manifest={compute}")
            print(f"baseline_manifest={baseline}")
            return
        if args.command == "run-task":
            from .runtime import run_task

            entry = run_task(args.manifest, args.index)
            print(f"completed index={entry.index} result={entry.result_path}")
            return
        if args.command == "status":
            from .status import manifest_status

            status = manifest_status(args.manifest)
            print(status.to_string(index=False))
            if not status.empty:
                print("\n" + status["state"].value_counts().sort_index().to_string())
            return
        if args.command == "analyze":
            from .analysis.cli import (
                print_measure_exclusions,
                resolve_settings,
                split_measures,
            )
            from .analysis.workflows import analyze_campaign

            settings = resolve_settings(args.analysis_config)
            measures = split_measures(args.measures)
            output = analyze_campaign(
                args.campaign,
                args.run_name,
                allow_incomplete=args.allow_incomplete,
                measures=measures,
                rank_scale=args.rank_scale,
                rank_sort=args.rank_sort,
                box_style=args.box_style,
                settings=settings,
            )
            reported = measures or settings.measures
            if reported:
                print(f"measures={','.join(reported)}")
            print_measure_exclusions(output)
            print(f"analysis={output}")
            return
    except (FileNotFoundError, IndexError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()


__all__ = ["build_parser", "main"]
