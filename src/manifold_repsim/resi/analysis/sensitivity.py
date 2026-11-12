"""How the ranking of the CKA signal's AUC responds to its grid and bandwidth limits.

``resi.py sensitivity-replay`` scores each signal variant (a sample count or a
bandwidth interval) through ReSi's own evaluation and leaves one CSV per compute
task under the scratch tree. This module reads those CSVs, lets each variant
stand in for the reported CKA AUC in an otherwise unchanged field of measures,
and ranks it with the same coverage filters and within-case ranking as the main
analysis. Holding the field and its size fixed means a change in rank is the
variant's alone.

It is a data module: it imports no plotting backend.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from manifold_repsim.config import save_effective_config
from manifold_repsim.resi.campaign import ManifestEntry, load_campaign
from manifold_repsim.resi.paths import slug
from manifold_repsim.resi.signal_sensitivity import (
    SensitivityConfig,
    scratch_paths,
)

from .config import unfavourable_quantile
from .loaders import (
    manifest_entries,
    normalize_entry,
    normalize_manifests,
    select_measures,
)
from .ranking import rank_scope, summarize_domains
from .settings import AnalysisSettings

REFERENCE_VARIANT = "stored"
SUMMARY_COLUMNS = [
    "domain",
    "analysis",
    "variant",
    "label",
    "measure",
    "mean_rank",
    "q90_rank",
    "mean_normalized_rank",
    "q90_normalized_rank",
    "position",
    "n_measures",
    "n_cases",
]


def variant_values(
    entries: Iterable[ManifestEntry],
    config: SensitivityConfig,
    quality_measures: Iterable[str],
    exclude_functional_measures: Iterable[str] = (),
) -> pd.DataFrame:
    """Normalized quality values of every variant, read from the scratch CSVs.

    ``exclude_functional_measures`` drops the same correlation results the main
    analysis drops, so a variant is ranked over exactly the cases the measure it
    replaces is.

    A variant's rows are renamed to the reported measure they replace and carry
    its key in a ``variant`` column. A task whose replay has not been run is an
    error: ranking a variant over fewer tasks than the baseline would compare
    different cases.
    """
    keys = {variant.metric_name: variant.key for variant in config.variants}
    excluded = set(exclude_functional_measures)
    frames = []
    for entry in entries:
        if entry.kind != "compute":
            continue
        task_config = yaml.safe_load(Path(entry.config_path).read_text())
        _store, csv_path, _root = scratch_paths(entry, task_config, config.scratch_name)
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"No replay output for compute index {entry.index}: {csv_path}"
            )
        frame = normalize_entry(
            replace(entry, full_csv_path=str(csv_path)), tuple(quality_measures)
        )
        frame = frame[frame["metric"].isin(keys)].copy()
        if excluded:
            frame = frame[
                ~frame["functional_similarity_measure"].astype(str).isin(excluded)
            ].copy()
        frame["variant"] = frame["metric"].map(keys)
        frame["metric"] = config.field_measure
        frames.append(frame)
    if not frames:
        raise ValueError("The campaign has no compute entries to replay")
    return pd.concat(frames, ignore_index=True)


def _measure_row(
    case_ranks: pd.DataFrame, field_measure: str, rank_scale: str | None
) -> dict[str, Any]:
    """The reported measure's standing in one ranked field."""
    if case_ranks.empty:
        return {
            "mean_rank": np.nan,
            "q90_rank": np.nan,
            "mean_normalized_rank": np.nan,
            "q90_normalized_rank": np.nan,
            "position": np.nan,
            "n_measures": 0,
            "n_cases": 0,
        }
    domain_summary = summarize_domains(case_ranks, rank_scale=rank_scale)
    ordered = domain_summary.reset_index(drop=True)
    mine = case_ranks[case_ranks["metric"] == field_measure]
    quantile = unfavourable_quantile(False)
    position = ordered.index[ordered["metric"] == field_measure]
    return {
        "mean_rank": float(mine["mean_rank"].mean()) if len(mine) else np.nan,
        "q90_rank": (
            float(np.quantile(mine["mean_rank"], quantile)) if len(mine) else np.nan
        ),
        "mean_normalized_rank": (
            float(mine["mean_normalized_rank"].mean()) if len(mine) else np.nan
        ),
        "q90_normalized_rank": (
            float(np.quantile(mine["mean_normalized_rank"], quantile))
            if len(mine)
            else np.nan
        ),
        "position": int(position[0]) + 1 if len(position) else np.nan,
        "n_measures": int(len(ordered)),
        "n_cases": int(len(mine)),
    }


BANDWIDTH_COLUMNS = [
    "domain",
    "measure",
    "sigma",
    "mean_rank",
    "q90_rank",
    "mean_normalized_rank",
    "q90_normalized_rank",
    "position",
    "n_measures",
    "n_cases",
]


def _bandwidth_field(
    campaign_path: str | Path,
    run_name: str,
    settings: AnalysisSettings,
    config: SensitivityConfig,
):
    """The campaign, normalized values, candidates, and shared rest of the field."""
    spec = config.bandwidths
    if spec is None:
        raise ValueError("The sensitivity config declares no bandwidths")
    campaign = load_campaign(campaign_path)
    compute, baseline = manifest_entries(campaign, run_name)
    values, _coverage = normalize_manifests(
        [*baseline, *compute],
        campaign,
        allow_incomplete=False,
        quality_measures=settings.quality_measures,
        exclude_functional_measures=settings.exclude_functional_measures,
    )
    present = set(values["metric"].astype(str))
    selected = settings.resolve_measures(sorted(present))
    selected = list(selected) if selected is not None else sorted(present)
    candidates = [(spec.reference, np.nan), *spec.points]
    missing = [name for name, _sigma in candidates if name not in present]
    if missing:
        raise ValueError(f"Results hold no values for: {', '.join(missing)}")
    names = {name for name, _sigma in candidates}
    rest = [measure for measure in selected if measure not in names]
    return campaign, values, candidates, rest


def _target_dir(campaign, run_name: str, out_dir: str | Path | None) -> Path:
    target = (
        Path(out_dir)
        if out_dir is not None
        else campaign.analysis_root
        / "signal_sensitivity"
        / slug(run_name)
        / campaign.domain
    )
    target.mkdir(parents=True, exist_ok=True)
    return target


def summarize_bandwidths(
    campaign_path: str | Path,
    run_name: str,
    settings: AnalysisSettings,
    config: SensitivityConfig,
    *,
    out_dir: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Rank each single-bandwidth CKA, and the swept reference, in one shared field.

    The field is the reported measures minus the reference and minus every
    single-bandwidth measure, so each candidate meets the same competitors and
    the same field size, and none appears twice. The reference's row has no
    ``sigma``: it is the swept measure the bandwidths are compared with.
    """
    campaign, values, candidates, rest = _bandwidth_field(
        campaign_path, run_name, settings, config
    )
    rows = []
    for name, sigma in candidates:
        field = select_measures(values, [*rest, name])
        _kept, _exclusions, case_ranks = rank_scope(
            field,
            scope_columns=["domain"],
            max_nan_fraction=settings.max_nan_fraction,
            min_case_coverage=settings.min_case_coverage,
        )
        rows.append(
            {
                "domain": campaign.domain,
                "measure": name,
                "sigma": sigma,
                **_measure_row(case_ranks, name, settings.rank_scale),
            }
        )
    table = pd.DataFrame(rows, columns=BANDWIDTH_COLUMNS)
    target = _target_dir(campaign, run_name, out_dir)
    table.to_csv(target / "bandwidth_summary.csv", index=False)
    return target, table


def summarize_paired_bandwidths(
    campaign_path: str | Path,
    run_name: str,
    settings: AnalysisSettings,
    config: SensitivityConfig,
    *,
    out_dir: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Rank the swept reference and each single-bandwidth CKA in a field of both.

    Every bandwidth gets its own field: the common rest of the reported measures,
    the swept reference, and that one single-bandwidth measure. The reference
    therefore meets a different rival at each ``sigma``, so its rank is a function
    of the bandwidth it is compared with, which the unpaired summary (one field
    per candidate) keeps constant. Each field holds two rows, one per measure, so
    the table has two rows per ``sigma`` and the field size is the same throughout.
    """
    spec = config.bandwidths
    campaign, values, candidates, rest = _bandwidth_field(
        campaign_path, run_name, settings, config
    )
    rows = []
    for name, sigma in candidates[1:]:
        field = select_measures(values, [*rest, spec.reference, name])
        _kept, _exclusions, case_ranks = rank_scope(
            field,
            scope_columns=["domain"],
            max_nan_fraction=settings.max_nan_fraction,
            min_case_coverage=settings.min_case_coverage,
        )
        for measure in (spec.reference, name):
            rows.append(
                {
                    "domain": campaign.domain,
                    "measure": measure,
                    "sigma": sigma,
                    **_measure_row(case_ranks, measure, settings.rank_scale),
                }
            )
    table = pd.DataFrame(rows, columns=BANDWIDTH_COLUMNS)
    target = _target_dir(campaign, run_name, out_dir)
    table.to_csv(target / "bandwidth_paired_summary.csv", index=False)
    return target, table


def summarize_sensitivity(
    campaign_path: str | Path,
    run_name: str,
    settings: AnalysisSettings,
    config: SensitivityConfig,
    *,
    out_dir: str | Path | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Rank every variant in the reported field and write the summary tables.

    Returns the output directory and the per-(analysis, variant) summary. The
    unchanged stored measure is ranked too, as a reference row: the variant that
    keeps every sample must land on it.
    """
    campaign = load_campaign(campaign_path)
    compute, baseline = manifest_entries(campaign, run_name)
    entries = [*baseline, *compute]
    values, _coverage = normalize_manifests(
        entries,
        campaign,
        allow_incomplete=False,
        quality_measures=settings.quality_measures,
        exclude_functional_measures=settings.exclude_functional_measures,
    )
    selected = settings.resolve_measures(values["metric"].astype(str).unique())
    selected = list(selected) if selected is not None else None
    if selected is not None and config.field_measure not in selected:
        raise ValueError(
            f"The reported measures do not include {config.field_measure}, so there "
            "is nothing for the variants to replace"
        )
    quality = (
        settings.quality_measures
        if settings.quality_measures is not None
        else campaign.quality_measures
    )
    variants = variant_values(
        entries, config, quality, settings.exclude_functional_measures
    )
    others = values[values["metric"] != config.field_measure]

    fields = {REFERENCE_VARIANT: values}
    for variant in config.variants:
        chosen = variants[variants["variant"] == variant.key].drop(columns="variant")
        fields[variant.key] = pd.concat([others, chosen], ignore_index=True)

    ranked: dict[str, pd.DataFrame] = {}
    summaries = []
    for key, frame in fields.items():
        frame = select_measures(frame, selected)
        _kept, _exclusions, case_ranks = rank_scope(
            frame,
            scope_columns=["domain"],
            max_nan_fraction=settings.max_nan_fraction,
            min_case_coverage=settings.min_case_coverage,
        )
        ranked[key] = case_ranks
        domain_summary = summarize_domains(case_ranks, rank_scale=settings.rank_scale)
        domain_summary.insert(0, "variant", key)
        summaries.append(domain_summary)

    rows = []
    for key, case_ranks in ranked.items():
        if key == REFERENCE_VARIANT:
            rows.append(
                {
                    "domain": campaign.domain,
                    "analysis": "reference",
                    "variant": key,
                    "label": "stored",
                    "measure": config.field_measure,
                    **_measure_row(
                        case_ranks, config.field_measure, settings.rank_scale
                    ),
                }
            )
    for analysis in config.analyses:
        for key, label in analysis.points:
            rows.append(
                {
                    "domain": campaign.domain,
                    "analysis": analysis.name,
                    "variant": key,
                    "label": label,
                    "measure": config.field_measure,
                    **_measure_row(
                        ranked[key], config.field_measure, settings.rank_scale
                    ),
                }
            )
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    target = (
        Path(out_dir)
        if out_dir is not None
        else campaign.analysis_root
        / "signal_sensitivity"
        / slug(run_name)
        / campaign.domain
    )
    target.mkdir(parents=True, exist_ok=True)
    save_effective_config(
        target,
        {
            "campaign": str(campaign_path),
            "run_name": run_name,
            "field_measure": config.field_measure,
            "base_metric": config.base_metric,
            "rank_scale": settings.rank_scale,
            "reported_measures": selected,
            "variants": [
                {
                    "key": v.key,
                    "samples": v.samples,
                    "lower": v.lower,
                    "upper": v.upper,
                }
                for v in config.variants
            ],
        },
    )
    summary.to_csv(target / "sensitivity_summary.csv", index=False)
    pd.concat(summaries, ignore_index=True).to_csv(
        target / "variant_domain_summaries.csv", index=False
    )
    return target, summary


__all__ = [
    "BANDWIDTH_COLUMNS",
    "REFERENCE_VARIANT",
    "SUMMARY_COLUMNS",
    "summarize_bandwidths",
    "summarize_paired_bandwidths",
    "summarize_sensitivity",
    "variant_values",
]
