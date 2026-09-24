#!/usr/bin/env python
"""Generate patched ReSi configs for the missing baseline runs.

The campaign tooling in `manifold_repsim.resi` deliberately cannot compute
ReSi's own measures: `campaign/schema.py` validates every configured measure
against our adapter catalogue. Recreating baselines therefore means running
ReSi's own pipeline (`python -m repsim.run -c <config>`) against ReSi's own
configs -- which already exist upstream for every gap except ALBERT.

This script does not reimplement any of that. It copies the relevant upstream
configs and applies the minimum patches needed to run them safely here:

* `included_measures` -- the reduced measure set (see MEASURES). ReSi treats
  `included_measures` and `excluded_measures` as mutually exclusive, so any
  upstream `excluded_measures` key is dropped.
* `only_eval: false` -- the CIFAR100 configs ship with `only_eval: True`,
  which skips `ex.run()` and would compute nothing.
* `raw_results_filename` -- namespaced. Upstream names collide with archived
  baselines: the SmolLM2 monotonicity configs write to
  `mono_nlp_standard_cls_tok_{sst2,mnli}.parquet`, which are the BERT-L
  archive files. Writing there would corrupt data that cannot be regenerated.
* `filter_key_vals.architecture` -- narrowed to the job's single architecture.
  CIFAR100 jobs are generated from ReSi's unified config, the same file the
  campaign's baseline eval uses, so the two cannot drift apart in which
  `train_dataset` groups they expect.
* `rerun_nans` -- true only for the PGNN jobs, whose target parquets already
  hold all-NaN PGNN rows from a March 2026 run that predates the checkpoints.
  ReSi skips NaN rows by default, so without this nothing would recompute.

For PGNN the script also seeds the output parquet from the existing results
file, so the already-computed GCN/GAT/GraphSAGE comparisons are skipped and
only the PGNN block is recomputed. The source file is copied, never modified.

ALBERT is intentionally absent: no checkpoints exist under
`$REP_SIM/models/nlp/`, so its configs cannot run.

Usage (from the repo root, inside the conda env):

    export RESI_DIR=~/research/resi REP_SIM=$RESI_DIR/experiments
    python scripts/prepare_resi_baselines.py --run-name baselines_2026_08
    python scripts/prepare_resi_baselines.py --run-name baselines_2026_08 --target pgnn

Writes configs plus a `jobs.txt` under data/resi/baselines/<run-name>/, then
submit with scripts/resi_baselines.sbatch.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# The reduced comparison set. Chosen for coverage and cost: these are all
# sub-2s/comparison measures, together ~2-6% of the cost of ReSi's full 24.
# IMDScore and RSMNormDifference are excluded deliberately -- they are ~67% of
# the total runtime in every domain. PWCCA is excluded because it does not
# exist in the graph archive at all and is 55% NaN in language, which would
# gut the strict `dropna(how="any")` ranking.
MEASURES = [
    "AlignedCosineSimilarity",
    "CKA",
    "DistanceCorrelation",
    "JaccardSimilarity",
    "LinearRegression",
    "ProcrustesSizeAndShapeDistance",
    "RankSimilarity",
    "SVCCA",
    "SecondOrderCosineSimilarity",
]

# target -> (job name, upstream config path, output parquet, seed-from parquet)
# `seed_from` is relative to $REP_SIM/results and is copied to the output name
# before the run so completed comparisons are skipped.
JOBS: list[tuple[str, str, str, str, str | None]] = [
    # PGNN on Cora. Upstream configs carry no architecture filter, so PGNN is
    # picked up automatically now that its checkpoints exist. augmentation_test
    # is omitted: the paper excludes P-GNN there (DropEdge perturbs the anchor
    # distances its message passing depends on).
    (
        "pgnn",
        "pgnn_label_test_cora",
        "configs/graphs/graphs_label_test_cora.yaml",
        "mfbase_pgnn_label_test_cora.parquet",
        "graphs_label_test_cora.parquet",
    ),
    (
        "pgnn",
        "pgnn_shortcut_test_cora",
        "configs/graphs/graphs_shortcut_test_cora.yaml",
        "mfbase_pgnn_shortcut_test_cora.parquet",
        "graphs_shortcut_test_cora.parquet",
    ),
    (
        "pgnn",
        "pgnn_output_correlation_test_cora",
        "configs/graphs/graphs_output_correlation_test_cora.yaml",
        "mfbase_pgnn_output_correlation_test_cora.parquet",
        "graphs_output_correlation_test_cora.parquet",
    ),
    # layer_test has no existing PGNN rows, so nothing to seed or rerun.
    (
        "pgnn",
        "pgnn_layer_test_cora",
        "configs/graphs/graphs_layer_test_cora.yaml",
        "mfbase_pgnn_layer_test_cora.parquet",
        None,
    ),
    # SmolLM2. Upstream monotonicity configs write to the BERT-L archive
    # filenames, hence the namespaced outputs below.
    (
        "smollm",
        "smollm_memorization_sst2",
        "configs/language/mem_sst2_smollm.yaml",
        "mfbase_smollm_memorization_sst2.parquet",
        None,
    ),
    (
        "smollm",
        "smollm_memorization_mnli",
        "configs/language/mem_mnli_smollm.yaml",
        "mfbase_smollm_memorization_mnli.parquet",
        None,
    ),
    (
        "smollm",
        "smollm_shortcut_sst2",
        "configs/language/sc_sst2_smollm.yaml",
        "mfbase_smollm_shortcut_sst2.parquet",
        None,
    ),
    (
        "smollm",
        "smollm_shortcut_mnli",
        "configs/language/sc_mnli_smollm.yaml",
        "mfbase_smollm_shortcut_mnli.parquet",
        None,
    ),
    (
        "smollm",
        "smollm_monotonicity_sst2",
        "configs/language/monotonicity_nlp_standard_sst2_smollm.yaml",
        "mfbase_smollm_monotonicity_sst2.parquet",
        None,
    ),
    (
        "smollm",
        "smollm_monotonicity_mnli",
        "configs/language/monotonicity_nlp_standard_mnli_smollm.yaml",
        "mfbase_smollm_monotonicity_mnli.parquet",
        None,
    ),
    (
        "smollm",
        "smollm_correlation_sst2",
        "configs/language/correlation_nlp_standard_sst2_smollm.yaml",
        "mfbase_smollm_correlation_sst2.parquet",
        None,
    ),
    (
        "smollm",
        "smollm_correlation_mnli",
        "configs/language/correlation_nlp_standard_mnli_smollm.yaml",
        "mfbase_smollm_correlation_mnli.parquet",
        None,
    ),
]

# CIFAR100, one job per (test, architecture). The unified upstream configs
# (C100_augmentation.yaml etc.) cover 5-7 architectures each, which puts a
# single job near the sbatch's 24h wall clock -- and a SLURM TIMEOUT is not
# requeued, unlike a preemption. Narrowing the unified config to a single
# architecture splits the same work into jobs of roughly a tenth the size. The
# resulting 33 jobs match the vision campaign's 33 CIFAR100 compute tasks
# exactly.
#
# augmentation has no ViT configs upstream: the unified file notes ViT_L32 and
# ViT_B32 "do not meet clustering criteria" for that test.
_CIFAR_ARCHS = [
    "resnet18",
    "resnet34",
    "resnet101",
    "vgg11",
    "vgg19",
    "vitb32",
    "vitl32",
]
_CIFAR_TESTS = {
    "augmentation": [a for a in _CIFAR_ARCHS if not a.startswith("vit")],
    "randomlabel": _CIFAR_ARCHS,
    "shortcut": _CIFAR_ARCHS,
    "monotonicity": _CIFAR_ARCHS,
    "accoutput": _CIFAR_ARCHS,
}
# The unified config is the source wherever one exists, because the campaign's
# baseline eval is built from that same file (see base_config in
# configs/resi_vision_campaign.yaml). The per-architecture augmentation files
# drop Gauss_L and Gauss_Off, so generating from them produced baselines whose
# eval scored every measure NaN. accoutput has no unified file upstream; its
# per-architecture configs are what the campaign uses as base_config too.
_CIFAR_UNIFIED = {"augmentation", "randomlabel", "shortcut", "monotonicity"}

JOBS += [
    (
        "cifar",
        f"cifar_{test}_{arch}",
        (
            f"configs/vision/c100/C100_{test}.yaml"
            if test in _CIFAR_UNIFIED
            else f"configs/vision/c100/C100_{test}_{arch}.yaml"
        ),
        f"mfbase_cifar_{test}_{arch}.parquet",
        None,
    )
    for test, archs in _CIFAR_TESTS.items()
    for arch in archs
]

# Job name -> the architecture token its config is narrowed to.
_CIFAR_ARCHITECTURE = {
    f"cifar_{test}_{arch}": arch
    for test, archs in _CIFAR_TESTS.items()
    for arch in archs
}

TARGETS = ["pgnn", "smollm", "cifar"]


def _resolve_architecture(config: dict, token: str) -> str:
    """Map a lowercase job token to the upstream config's own spelling."""
    names: list[str] = []
    for experiment in config.get("experiments") or []:
        filters = experiment.get("filter_key_vals") or {}
        names.extend(filters.get("architecture") or [])
    for candidate in names:
        if str(candidate).replace("_", "").lower() == token:
            return candidate
    raise SystemExit(
        f"architecture {token!r} is not offered by the upstream config; "
        f"have {sorted(set(names))}"
    )


def patch(
    config: dict,
    out_parquet: str,
    rerun_nans: bool,
    name: str,
    architecture: str | None = None,
) -> dict:
    """Apply the minimum changes needed to run an upstream config here."""
    config = dict(config)
    # The per-architecture job split lives here rather than in the choice of
    # upstream file, so generator and campaign eval always share one source.
    if architecture is not None:
        resolved = _resolve_architecture(config, architecture)
        config["experiments"] = [
            {
                **experiment,
                "filter_key_vals": {
                    **(experiment.get("filter_key_vals") or {}),
                    "architecture": [resolved],
                },
            }
            for experiment in config.get("experiments") or []
        ]
    # included_measures and excluded_measures are mutually exclusive in ReSi.
    config.pop("excluded_measures", None)
    config["included_measures"] = list(MEASURES)
    # The CIFAR100 configs ship only_eval: True, which skips ex.run().
    config["only_eval"] = False
    config["only_extract_reps"] = False
    config["rerun_nans"] = bool(rerun_nans)
    config["raw_results_filename"] = out_parquet
    table = dict(config.get("table_creation") or {})
    table["save_full_df"] = True
    table["full_df_filename"] = f"{name}_full.csv"
    table["save_aggregated_df"] = False
    table.setdefault("row_index", "similarity_measure")
    table.setdefault("columns", ["quality_measure", "architecture"])
    table.setdefault("value_key", "value")
    table["filename"] = f"{name}.csv"
    config["table_creation"] = table
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--target",
        action="append",
        choices=TARGETS,
        help="limit to one target (repeatable)",
    )
    parser.add_argument("--resi-dir", default=os.environ.get("RESI_DIR"))
    parser.add_argument("--rep-sim", default=os.environ.get("REP_SIM"))
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing seeded parquet"
    )
    args = parser.parse_args()

    if not args.resi_dir:
        parser.error("set RESI_DIR or pass --resi-dir")
    if not args.rep_sim:
        parser.error("set REP_SIM or pass --rep-sim")
    resi_dir, rep_sim = Path(args.resi_dir), Path(args.rep_sim)
    if not resi_dir.is_dir():
        parser.error(f"ReSi checkout not found: {resi_dir}")
    results = rep_sim / "results"
    if not results.is_dir():
        parser.error(f"results directory not found: {results}")

    targets = args.target or TARGETS
    out_dir = REPO_ROOT / "data" / "resi" / "baselines" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = [j for j in JOBS if j[0] in targets]
    written: list[Path] = []
    for target, name, rel, out_parquet, seed_from in selected:
        src = resi_dir / rel
        if not src.is_file():
            print(f"!! missing upstream config, skipping: {src}")
            continue
        with src.open() as handle:
            config = yaml.safe_load(handle)
        patched = patch(
            config,
            out_parquet,
            rerun_nans=seed_from is not None,
            name=name,
            architecture=_CIFAR_ARCHITECTURE.get(name),
        )
        dest = out_dir / f"{name}.yaml"
        with dest.open("w") as handle:
            yaml.safe_dump(patched, handle, sort_keys=False)
        written.append(dest)

        if seed_from:
            source, target_path = results / seed_from, results / out_parquet
            if not source.is_file():
                print(f"!! seed source missing: {source}")
            elif target_path.exists() and not args.force:
                print(f"-- seed exists, keeping: {target_path.name}")
            else:
                shutil.copy2(source, target_path)
                print(f"-- seeded {target_path.name} from {seed_from}")
        print(f"[{target}] {dest.relative_to(REPO_ROOT)} -> results/{out_parquet}")

    jobs_file = out_dir / "jobs.txt"
    jobs_file.write_text("".join(f"{p}\n" for p in written))
    print(f"\n{len(written)} configs written")
    print(f"jobs list: {jobs_file}")
    print(
        f"submit:    sbatch --array=0-{len(written) - 1}%8 scripts/resi_baselines.sbatch"
    )
    print(f"           (set JOBS={jobs_file} in the environment first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
