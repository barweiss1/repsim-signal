from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

from manifold_repsim.config import save_effective_config
from manifold_repsim.resi.campaign import load_campaign, read_manifest
from manifold_repsim.resi.paths import slug

from .config import (
    CORRELATION_TASK_DIRS,
    validate_box_style,
    validate_rank_scale,
    validate_rank_sort,
)
from .loaders import known_missing_coverage, normalize_manifests, select_measures
from .settings import AnalysisSettings, settings_snapshot
from .ranking import (
    filter_measures_by_coverage,
    rank_case_observations,
    select_ranked_functional,
    select_ranked_quality,
    summarize_domains,
)
from .signals import (
    combine_signals,
    exclude_known_missing_signals,
    load_embedded_signals,
    load_legacy_signals,
    load_manifest_legacy_signals,
    signal_input_coverage,
    summarize_signals,
)
from .plotting import (
    plot_auprc_vs_violation_rate,
    plot_grouped_signal_comparisons,
    plot_quality_heatmaps,
    plot_rank_boxplots,
    plot_signal_groups,
)


def _manifest_entries(campaign, run_name: str):
    run_dir = campaign.run_root / slug(run_name) / campaign.domain
    compute_path = run_dir / "compute_manifest.jsonl"
    baseline_path = run_dir / "baseline_manifest.jsonl"
    if not compute_path.is_file():
        raise FileNotFoundError(
            f"Compute manifest not found; run prepare first: {compute_path}"
        )
    compute = read_manifest(compute_path)
    baseline = read_manifest(baseline_path) if baseline_path.is_file() else []
    return compute, baseline


def _value_coverage(values: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "record_type",
        "kind",
        "index",
        "state",
        "domain",
        "benchmark",
        "dataset",
        "architectures",
        "metric",
        "quality_measure",
        "rows",
        "finite_values",
        "nan_values",
        "signal_source",
        "signal_files_declared",
        "signal_files_found",
        "excluded_model",
        "exclusion_reason",
        "error",
        "result_path",
        "full_csv_path",
    ]
    if values.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        values.assign(finite=np.isfinite(values["value"]), nan=values["value"].isna())
        .groupby(
            [
                "source",
                "domain",
                "benchmark",
                "dataset",
                "architecture",
                "metric",
                "quality_measure",
            ],
            as_index=False,
        )
        .agg(
            rows=("value", "size"),
            finite_values=("finite", "sum"),
            nan_values=("nan", "sum"),
        )
    )
    grouped["record_type"] = "values"
    grouped["kind"] = grouped.pop("source")
    grouped["index"] = ""
    grouped["state"] = np.where(grouped["nan_values"] > 0, "has_nan", "complete")
    grouped["architectures"] = grouped.pop("architecture")
    grouped["signal_source"] = ""
    grouped["signal_files_declared"] = ""
    grouped["signal_files_found"] = ""
    grouped["excluded_model"] = ""
    grouped["exclusion_reason"] = ""
    grouped["error"] = ""
    grouped["result_path"] = ""
    grouped["full_csv_path"] = ""
    return grouped.reindex(columns=columns)


def _rank_scope(
    values: pd.DataFrame,
    *,
    scope_columns: list[str],
    max_nan_fraction: float,
    min_case_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quality-filter, coverage-filter, and rank one reporting scope.

    Returns the values this scope reports, the measure exclusions it recorded,
    and its per-case ranks. Every scope runs the same three steps in the same
    order, and this is the only place that order is written down.

    The returned values keep every quality and functional similarity measure
    the analysis config asked for, narrowed to the measures the coverage filter
    kept -- so the conformity rate still reaches the heatmaps and the AUPRC
    scatter and Disagreement still reaches the tables, while a measure dropped
    for sparse coverage does not reappear there after leaving the ranking.
    """
    ranking_values, exclusions = filter_measures_by_coverage(
        select_ranked_functional(select_ranked_quality(values)),
        max_nan_fraction=max_nan_fraction,
        min_case_coverage=min_case_coverage,
        scope_columns=scope_columns,
    )
    _observation_ranks, case_ranks = rank_case_observations(ranking_values)
    if values.empty:
        return values, exclusions, case_ranks
    kept = set(ranking_values["metric"].astype(str))
    reported = values[values["metric"].astype(str).isin(kept)].reset_index(drop=True)
    return reported, exclusions, case_ranks


class TaskScope(NamedTuple):
    """One per-task output directory: what it is called and what it holds.

    ``name`` is the directory under ``tasks/``; ``functionals`` is the set of
    functional similarity measures belonging to it, empty for a benchmark that
    runs only one task.
    """

    name: str
    benchmark: str
    dataset: str
    functionals: frozenset[str]

    def select(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Restrict a frame carrying benchmark/dataset columns to this scope.

        An empty ``dataset`` means every dataset, which is how the per-task
        level above this one pools them.
        """
        if frame.empty or "benchmark" not in frame.columns:
            return frame
        keep = frame["benchmark"].eq(self.benchmark)
        if self.dataset:
            keep &= frame["dataset"].eq(self.dataset)
        if self.functionals and "functional_similarity_measure" in frame.columns:
            functional = frame["functional_similarity_measure"].fillna("").astype(str)
            keep &= functional.isin(self.functionals)
        return frame[keep].copy()


def task_scopes(entries, values: pd.DataFrame) -> list[TaskScope]:
    """The per-task outputs to write, keyed on the task rather than the benchmark.

    The prediction-correlation benchmark runs two of the six tasks at once, so
    keying its outputs on the benchmark id pooled `Output Corr.` and
    `Acc Corr.` into one ranking labelled with the benchmark. Those split into
    `outcorr` and `acccorr` (`CORRELATION_TASK_DIRS`), named the same way in
    every domain even though each domain names the benchmark differently.

    Every other benchmark keeps its own id and an empty functional set, so its
    path and contents are unchanged. A benchmark the run produced no values for
    still gets its directory, so an empty task reads as reported-and-empty
    rather than as an output nobody generated.
    """
    if values.empty or "functional_similarity_measure" not in values.columns:
        present = pd.DataFrame(columns=["benchmark", "dataset", "_functional"])
    else:
        present = values.assign(
            _functional=values["functional_similarity_measure"].fillna("").astype(str)
        )[["benchmark", "dataset", "_functional"]].drop_duplicates()

    scopes: list[TaskScope] = []
    for benchmark, dataset in sorted({(e.benchmark, e.dataset) for e in entries}):
        found = present[
            present["benchmark"].eq(benchmark) & present["dataset"].eq(dataset)
        ]["_functional"].tolist()
        groups: dict[str, set[str]] = {}
        for functional in found:
            # An unmapped or blank functional measure names no task of its own,
            # so it stays under the benchmark's directory rather than vanishing.
            groups.setdefault(
                CORRELATION_TASK_DIRS.get(functional, benchmark), set()
            ).add(functional)
        if not groups:
            scopes.append(TaskScope(benchmark, benchmark, dataset, frozenset()))
            continue
        for name, functionals in sorted(groups.items()):
            scopes.append(
                TaskScope(
                    name,
                    benchmark,
                    dataset,
                    frozenset() if name == benchmark else frozenset(functionals),
                )
            )
    return scopes


def _write_task_analysis(
    out_dir: Path,
    task_keys: list[TaskScope],
    values: pd.DataFrame,
    coverage: pd.DataFrame,
    signals: pd.DataFrame,
    signals_by_seed: pd.DataFrame,
    signals_grouped: pd.DataFrame,
    rank_scale: str,
    rank_sort: str,
    *,
    box_style: str,
    max_nan_fraction: float,
    min_case_coverage: float,
) -> pd.DataFrame:
    """Per-(task, dataset) outputs, each filtered and ranked in its own right.

    Ranks are recomputed here rather than sliced out of the domain-level ranking.
    A case never spans two tasks, so the two agree whenever both scopes report
    the same measures -- but a measure can clear the domain coverage threshold
    while failing one task's, and then the task's ranking must be over the
    measures that task actually reports.

    The scope is the reporting task, not the benchmark id: a benchmark running
    two tasks writes two directories. Signals carry no functional similarity
    measure, so both directories get the benchmark's signals; everything else
    is filtered to the task.
    """
    index_rows: list[dict[str, object]] = []
    for scope in task_keys:
        benchmark, dataset = scope.benchmark, scope.dataset
        task_dir = out_dir / "tasks" / slug(scope.name) / slug(dataset)
        task_dir.mkdir(parents=True, exist_ok=True)
        task_values, task_exclusions, task_ranks = _rank_scope(
            scope.select(values),
            scope_columns=["benchmark", "dataset"],
            max_nan_fraction=max_nan_fraction,
            min_case_coverage=min_case_coverage,
        )
        task_coverage = coverage[
            coverage["benchmark"].eq(benchmark) & coverage["dataset"].eq(dataset)
        ].copy()
        task_signals = signals[
            signals["benchmark"].eq(benchmark) & signals["dataset"].eq(dataset)
        ].copy()
        task_by_seed = signals_by_seed[
            signals_by_seed["benchmark"].eq(benchmark)
            & signals_by_seed["dataset"].eq(dataset)
        ].copy()
        task_grouped = signals_grouped[
            signals_grouped["benchmark"].eq(benchmark)
            & signals_grouped["dataset"].eq(dataset)
        ].copy()

        task_values.to_csv(task_dir / "normalized_values.csv", index=False)
        task_exclusions.to_csv(task_dir / "measure_exclusions.csv", index=False)
        task_ranks.to_csv(task_dir / "per_case_ranks.csv", index=False)
        rank_summary = summarize_domains(task_ranks, rank_scale=rank_scale)
        rank_summary.insert(1, "task", scope.name)
        rank_summary.insert(2, "benchmark", benchmark)
        rank_summary.insert(3, "dataset", dataset)
        rank_summary.to_csv(task_dir / "rank_summary.csv", index=False)
        task_coverage.to_csv(task_dir / "coverage_nan_report.csv", index=False)
        task_signals.to_csv(task_dir / "signals_long.csv", index=False)
        task_by_seed.to_csv(task_dir / "signals_by_seed.csv", index=False)
        task_grouped.to_csv(task_dir / "signals_grouped.csv", index=False)
        for signal_plot_dir in ("signal_plots", "grouped_signal_plots"):
            plot_path = task_dir / signal_plot_dir
            if plot_path.is_dir():
                shutil.rmtree(plot_path)

        rank_path = task_dir / "rank_boxplot.png"
        plot_rank_boxplots(
            task_ranks,
            rank_path,
            rank_scale=rank_scale,
            rank_sort=rank_sort,
            box_style=box_style,
        )
        heatmaps = plot_quality_heatmaps(task_values, task_dir / "quality_heatmaps")
        scatter = plot_auprc_vs_violation_rate(
            task_values, task_dir / "auprc_vs_violation_rate.png"
        )
        signal_plots = plot_signal_groups(
            task_by_seed, task_grouped, task_dir / "signal_plots"
        )
        grouped_signal_plots = plot_grouped_signal_comparisons(
            task_signals, task_dir / "grouped_signal_plots"
        )
        index_rows.append(
            {
                "task": scope.name,
                "benchmark": benchmark,
                "dataset": dataset,
                "state": "complete" if not task_values.empty else "no_values",
                "out_dir": str(task_dir),
                "input_entries": int(
                    (
                        task_coverage["record_type"].eq("manifest")
                        if "record_type" in task_coverage
                        else pd.Series(dtype=bool)
                    ).sum()
                ),
                "value_rows": len(task_values),
                "rank_rows": len(task_ranks),
                "signal_rows": len(task_signals),
                "signal_comparisons": (
                    task_signals["canonical_comparison_id"].nunique()
                    if "canonical_comparison_id" in task_signals
                    else 0
                ),
                "rank_plot": str(rank_path) if rank_path.is_file() else "",
                "quality_heatmaps": len(heatmaps),
                "quality_scatter": str(scatter) if scatter is not None else "",
                "signal_plots": len(signal_plots),
                "grouped_signal_plots": len(grouped_signal_plots),
            }
        )
    return pd.DataFrame(index_rows)


def _write_benchmark_analysis(
    out_dir: Path,
    benchmarks: list[TaskScope],
    values: pd.DataFrame,
    rank_scale: str,
    rank_sort: str,
    *,
    box_style: str,
    max_nan_fraction: float,
    min_case_coverage: float,
) -> pd.DataFrame:
    """Per-task rank boxplots, pooling every dataset in the domain.

    Sits one level above the per-(task, dataset) output written by
    `_write_task_analysis`: the same measure-rank boxplot, but aggregated
    across every dataset that task ran on, so e.g. label_test on
    Cora/Flickr/OGBN-Arxiv shows up as one combined view of how measures rank
    on that test across the whole domain, rather than three separate ones.

    Keyed on the task for the same reason the level below is, so `outcorr` and
    `acccorr` get a boxplot each instead of one labelled with the benchmark
    that happens to run both.
    """
    index_rows: list[dict[str, object]] = []
    for scope in benchmarks:
        benchmark = scope.benchmark
        benchmark_dir = out_dir / "tasks" / slug(scope.name)
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        _reported, benchmark_exclusions, benchmark_ranks = _rank_scope(
            scope.select(values),
            scope_columns=["benchmark"],
            max_nan_fraction=max_nan_fraction,
            min_case_coverage=min_case_coverage,
        )
        benchmark_exclusions.to_csv(
            benchmark_dir / "measure_exclusions.csv", index=False
        )
        benchmark_ranks.to_csv(benchmark_dir / "per_case_ranks.csv", index=False)
        rank_summary = summarize_domains(benchmark_ranks, rank_scale=rank_scale)
        rank_summary.insert(1, "task", scope.name)
        rank_summary.insert(2, "benchmark", benchmark)
        rank_summary.to_csv(benchmark_dir / "rank_summary.csv", index=False)
        rank_path = benchmark_dir / "rank_boxplot.png"
        plot_rank_boxplots(
            benchmark_ranks,
            rank_path,
            rank_scale=rank_scale,
            rank_sort=rank_sort,
            box_style=box_style,
        )
        if "dataset" in benchmark_ranks.columns:
            datasets = sorted(benchmark_ranks["dataset"].dropna().astype(str).unique())
        else:
            datasets = []
        index_rows.append(
            {
                "task": scope.name,
                "benchmark": benchmark,
                "datasets": ",".join(datasets),
                "state": "complete" if not benchmark_ranks.empty else "no_values",
                "out_dir": str(benchmark_dir),
                "rank_rows": len(benchmark_ranks),
                "rank_plot": str(rank_path) if rank_path.is_file() else "",
            }
        )
    return pd.DataFrame(index_rows)


def analyze_campaign(
    campaign_path: str | Path,
    run_name: str,
    *,
    allow_incomplete: bool = False,
    measures: Iterable[str] | None = None,
    rank_scale: str | None = None,
    rank_sort: str | None = None,
    box_style: str | None = None,
    settings: AnalysisSettings | None = None,
) -> Path:
    """Analyze one prepared campaign run, optionally over a fixed measure set.

    ``settings`` carries the reporting choices from an analysis settings file.
    ``measures``, ``rank_scale``, and ``allow_incomplete`` are direct overrides
    that win over it, so a command line flag beats the file it was pointed at.

    Restricting the reported measures matters when coverage is uneven across
    architectures: recreated baselines carry a reduced set, so pooling them
    with fully covered architectures would rank measures against a different
    field in each case. ``rank_scale`` decides whether that pooling ranks on
    the raw within-case rank or its normalized rescaling; both are always
    written.
    """
    settings = settings or AnalysisSettings()
    resolved_rank_scale = validate_rank_scale(
        rank_scale if rank_scale is not None else settings.rank_scale
    )
    resolved_rank_sort = validate_rank_sort(
        rank_sort if rank_sort is not None else settings.rank_sort
    )
    resolved_box_style = validate_box_style(
        box_style if box_style is not None else settings.box_style
    )
    campaign = load_campaign(campaign_path)
    compute, baseline = _manifest_entries(campaign, run_name)
    entries = [*baseline, *compute]
    values, input_coverage = normalize_manifests(
        entries,
        campaign,
        allow_incomplete=allow_incomplete or settings.allow_incomplete,
        quality_measures=settings.quality_measures,
        exclude_functional_measures=settings.exclude_functional_measures,
    )
    selected = (
        list(measures)
        if measures is not None
        else settings.resolve_measures(
            values["metric"].astype(str).unique() if not values.empty else ()
        )
    )
    # Kept unfiltered for the recreated appendix tables below: those reproduce
    # the published NaN convention, which keeps unmatched rows and every
    # measure. Handing them the filtered frame would make the two conventions
    # agree by construction and hide exactly what they are written to compare.
    selected_values = select_measures(values, selected)
    # Pooled across the domain, a measure ranked on materially fewer cases than
    # its competitors is comparing a different field, not a different measure.
    values, measure_exclusions, case_ranks = _rank_scope(
        selected_values,
        scope_columns=["domain"],
        max_nan_fraction=settings.max_nan_fraction,
        min_case_coverage=settings.min_case_coverage,
    )
    domain_summary = summarize_domains(case_ranks, rank_scale=resolved_rank_scale)

    embedded = load_embedded_signals(entries)
    manifest_legacy = load_manifest_legacy_signals(entries)
    campaign_legacy = load_legacy_signals(campaign)
    signals = combine_signals(campaign_legacy, manifest_legacy, embedded)
    signals = exclude_known_missing_signals(signals, campaign)
    signals = select_measures(signals, selected, require_all=False)
    signals_by_seed, signals_grouped = summarize_signals(signals)

    out_dir = campaign.analysis_root / slug(run_name) / campaign.domain
    out_dir.mkdir(parents=True, exist_ok=True)
    save_effective_config(
        out_dir,
        settings_snapshot(
            settings,
            selected,
            rank_scale=resolved_rank_scale,
            rank_sort=resolved_rank_sort,
            box_style=resolved_box_style,
        ),
    )
    for stale_signal_dir in ("signal_plots", "grouped_signal_plots"):
        stale_path = out_dir / stale_signal_dir
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
    values.to_csv(out_dir / "normalized_values.csv", index=False)
    measure_exclusions.to_csv(out_dir / "measure_exclusions.csv", index=False)
    case_ranks.to_csv(out_dir / "per_case_ranks.csv", index=False)
    domain_summary.to_csv(out_dir / "domain_summaries.csv", index=False)
    input_coverage = input_coverage.copy()
    input_coverage.insert(0, "record_type", "manifest")
    exclusions = known_missing_coverage(entries, campaign)
    signal_coverage = signal_input_coverage(entries)
    coverage = pd.concat(
        [input_coverage, _value_coverage(values), exclusions, signal_coverage],
        ignore_index=True,
        sort=False,
    )
    coverage.to_csv(out_dir / "coverage_nan_report.csv", index=False)
    signals.to_csv(out_dir / "signals_long.csv", index=False)
    signals_by_seed.to_csv(out_dir / "signals_by_seed.csv", index=False)
    signals_grouped.to_csv(out_dir / "signals_grouped.csv", index=False)
    plot_rank_boxplots(
        case_ranks,
        out_dir / "rank_boxplot.png",
        rank_scale=resolved_rank_scale,
        rank_sort=resolved_rank_sort,
        box_style=resolved_box_style,
    )
    task_keys = task_scopes(entries, values)
    task_index = _write_task_analysis(
        out_dir,
        task_keys,
        values,
        coverage,
        signals,
        signals_by_seed,
        signals_grouped,
        resolved_rank_scale,
        resolved_rank_sort,
        box_style=resolved_box_style,
        max_nan_fraction=settings.max_nan_fraction,
        min_case_coverage=settings.min_case_coverage,
    )
    task_index.to_csv(out_dir / "task_analysis_index.csv", index=False)
    # One entry per task, with the dataset dropped: the level above pools every
    # dataset that task ran on.
    benchmark_keys = [
        TaskScope(name, benchmark, "", functionals)
        for name, benchmark, functionals in sorted(
            {(scope.name, scope.benchmark, scope.functionals) for scope in task_keys}
        )
    ]
    benchmark_index = _write_benchmark_analysis(
        out_dir,
        benchmark_keys,
        values,
        resolved_rank_scale,
        resolved_rank_sort,
        box_style=resolved_box_style,
        max_nan_fraction=settings.max_nan_fraction,
        min_case_coverage=settings.min_case_coverage,
    )
    benchmark_index.to_csv(out_dir / "benchmark_analysis_index.csv", index=False)
    # The appendix tables and the published-mode rank figure are fed from the
    # pre-coverage-filter values, so they report every measure the analysis
    # asked for and every quality measure it kept. Both rankings now treat NaN
    # the same way; what the published one still does differently is skip the
    # coverage filters and break ties with `min`, which is what makes writing
    # both worth the second figure.
    from .tables import write_appendix_report

    write_appendix_report(
        selected_values,
        out_dir,
        token_aware=campaign.domain == "language",
        box_style=resolved_box_style,
    )
    return out_dir


__all__ = ["TaskScope", "analyze_campaign", "task_scopes"]
