"""Build cross-domain comparison figures from finished per-domain analyses.

Reads each domain's `normalized_values.csv` -- the reported set, after coverage
filtering -- groups the domains' differently named benchmarks into the six
shared experiment families, and writes one value figure per family. Six rather
than five because the prediction-correlation benchmark splits into `Output
Corr.` (JSD and Disagreement) and `Acc Corr.` (AbsoluteAccDiff), which ask
different questions and are not all reported by every domain.

Ranks are averaged into subtasks before anything pools them, so every
(domain, dataset, task, evaluation) slice carries one vote regardless of how
many architectures it ran. See `analysis/cross_domain.py::SUBTASK_KEYS`.

    python scripts/resi_cross_domain.py \
      --domain graphs=figures/resi/<run>/graphs \
      --domain vision=figures/resi/<run>/vision \
      --domain language=figures/resi/<run>/language \
      --out figures/resi/cross_domain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from manifold_repsim.resi.analysis.config import (  # noqa: E402
    BOX_STYLES,
    DEFAULT_BOX_STYLE,
    DEFAULT_RANK_SCALE,
    RANK_SCALES,
    RANK_SORTS,
)
from manifold_repsim.resi.analysis.cross_domain import (  # noqa: E402
    ALL_TASKS,
    aggregate_across_domains,
    average_subtask_ranks,
    combine_case_ranks,
    combine_domain_values,
    common_functional_measures,
    functional_measure_coverage,
    subtask_balance,
    summarize_cross_domain,
)
from manifold_repsim.resi.analysis.cross_domain_figures import (  # noqa: E402
    plot_cross_domain_ranks,
    write_by_type_figures,
    write_cross_domain_figures,
)
from manifold_repsim.resi.analysis.cross_domain_tables import (  # noqa: E402
    write_cross_domain_tables,
)
from manifold_repsim.resi.analysis.exemplary import (  # noqa: E402
    write_averaged_table,
    write_exemplary_table,
)

VALUES_FILENAME = "normalized_values.csv"
RANKS_FILENAME = "per_case_ranks.csv"


def _parse_domain(argument: str) -> tuple[str, Path]:
    name, _, path = argument.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError(
            f"--domain expects <name>=<analysis-dir>, got {argument!r}"
        )
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        action="append",
        required=True,
        type=_parse_domain,
        metavar="NAME=DIR",
        help="Domain name and its analysis output directory. Repeatable.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--box-style",
        choices=BOX_STYLES,
        default=DEFAULT_BOX_STYLE,
        help=(
            "Draw each measure's rank distribution as a quartile box with "
            "percentile whiskers (default) or as a boxen: nested bands at the "
            "10th, 30th, 50th, 70th and 90th percentiles, narrowing as they "
            "reach into the tails, with whiskers to the true min and max. "
            "This changes the drawing only, never the ordering or the "
            "underlying numbers."
        ),
    )
    parser.add_argument(
        "--rank-sort",
        choices=RANK_SORTS,
        default="mean",
        help=(
            "Sort the rank boxplots by each measure's mean (default here), its "
            "median, or its 90th-percentile rank, which orders on how badly a "
            "measure does when it does badly. The quantile is drawn on every "
            "box when it is what the rows are sorted on; the mean is always "
            "drawn regardless. `mean` and `median` order on the aggregated "
            "balanced_mean, which weights the three domains equally; "
            "`quantile90` pools subtasks unweighted by domain, because a "
            "quantile has no balanced-mean analogue."
        ),
    )
    parser.add_argument(
        "--rank-scale",
        choices=RANK_SCALES,
        default=DEFAULT_RANK_SCALE,
        help=(
            "Which per-case statistic to pool: the raw within-case rank "
            "(default, 1 = best, matching the per-domain figures) or its "
            "(rank - 1)/(n - 1) rescaling. Both are written to "
            "cross_domain_subtask_ranks.csv either way. Raw ranks carry one "
            "caveat worth a caption: vision reports 15 measures where graphs "
            "and language report 16, so a raw rank there is drawn from a "
            "slightly smaller field."
        ),
    )
    args = parser.parse_args(argv)

    frames: dict[str, pd.DataFrame] = {}
    rank_frames: dict[str, pd.DataFrame] = {}
    for name, directory in args.domain:
        path = directory / VALUES_FILENAME
        if not path.is_file():
            parser.error(f"missing {VALUES_FILENAME} for {name}: {path}")
        frames[name] = pd.read_csv(path)
        rank_path = directory / RANKS_FILENAME
        if not rank_path.is_file():
            parser.error(f"missing {RANKS_FILENAME} for {name}: {rank_path}")
        rank_frames[name] = pd.read_csv(rank_path)
        print(f"{name}: {len(frames[name])} rows from {path}")

    coverage = functional_measure_coverage(frames)
    common = common_functional_measures(frames)
    if not coverage.empty:
        reported = set(coverage["functional_similarity_measure"])
        partial = sorted(reported - common)
        print(f"functional measures common to every domain: {sorted(common)}")
        if partial:
            # Reported, not dropped. AbsoluteAccDiff forms its own `Acc Corr.`
            # family, so a domain missing it contributes nothing to that family
            # rather than forcing every other domain's rows to be discarded.
            # Which domains hold it is a fact worth printing either way.
            print(
                f"reported by some domains only (kept, in their own family): {partial}"
            )
            for name in partial:
                holders = sorted(
                    coverage.loc[
                        coverage["functional_similarity_measure"].eq(name), "domain"
                    ]
                )
                print(f"  {name}: reported only by {holders}")

    values = combine_domain_values(frames)
    if values.empty:
        parser.error("no rows survived benchmark grouping; check the input domains")

    args.out.mkdir(parents=True, exist_ok=True)
    values.to_csv(args.out / "cross_domain_values.csv", index=False)
    summarize_cross_domain(values).to_csv(
        args.out / "cross_domain_summary.csv", index=False
    )
    index = write_cross_domain_figures(values, args.out)
    print(index.to_string(index=False))
    tables = write_cross_domain_tables(values, args.out)
    print()
    print(tables[["test_group", "rows", "columns"]].to_string(index=False))

    # One exemplary model and dataset per domain, every test across the top --
    # the scannable counterpart to the per-domain appendix tables. Built from
    # the unpooled per-domain frames, not from `values`, because it fixes one
    # (dataset, architecture) cell rather than aggregating over them.
    exemplary = write_exemplary_table(frames, args.out)
    print()
    if exemplary.empty:
        print("exemplary table: no domain supplied its selected cell")
    else:
        print("exemplary table (one model and dataset per domain):")
        print(exemplary[["table", "rows", "columns", "note"]].to_string(index=False))

    # The same table averaged over every model and dataset instead of read off
    # one cell, so a measure cannot look good or bad on the strength of the one
    # architecture the exemplary table happens to name.
    averaged = write_averaged_table(frames, args.out)
    print()
    if averaged.empty:
        print("averaged table: no domain supplied any reported evaluation")
    else:
        print("averaged table (every model and dataset per domain):")
        print(averaged[["table", "rows", "columns", "note"]].to_string(index=False))

    ranks = combine_case_ranks(rank_frames)
    # One point per subtask, not per case. A case count tracks how many
    # architectures a domain happens to ship, so pooling cases makes a
    # cross-domain figure mostly a vision figure.
    subtasks = average_subtask_ranks(ranks)
    subtasks.to_csv(args.out / "cross_domain_subtask_ranks.csv", index=False)
    balance = subtask_balance(subtasks)
    if not balance.empty:
        print()
        print("subtasks per family and domain (the balance this weighting buys):")
        print(balance.to_string())
    aggregate = aggregate_across_domains(subtasks, rank_scale=args.rank_scale)
    aggregate.to_csv(args.out / "cross_domain_ranks.csv", index=False)
    plot_cross_domain_ranks(
        subtasks,
        args.out / "cross_domain_ranks.png",
        rank_sort=args.rank_sort,
        rank_scale=args.rank_scale,
        box_style=args.box_style,
    )
    by_type = write_by_type_figures(
        values,
        subtasks,
        args.out,
        rank_sort=args.rank_sort,
        rank_scale=args.rank_scale,
        box_style=args.box_style,
    )
    print()
    print("type-coloured, per-panel sorted figures:")
    print(by_type.to_string(index=False))

    overall = aggregate[aggregate["test_group"].eq(ALL_TASKS)]
    print()
    print(f"aggregated across domains ({args.rank_scale} scale, lower is better):")
    print(
        overall[
            [
                "metric",
                "n_subtasks",
                "n_cases",
                "n_domains",
                "pooled_mean",
                "balanced_mean",
            ]
        ].to_string(index=False)
    )
    print(f"cross-domain outputs={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
