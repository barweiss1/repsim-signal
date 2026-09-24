# Running ReSi campaigns on a SLURM cluster

The ReSi campaigns register this repository's measures in an external
[ReSi](https://github.com/mklabunde/resi) checkout at runtime and run its
benchmarks. ReSi is not vendored or modified; clone it separately.

## Layout and environment variables

The scripts assume this repository and the ReSi checkout are siblings under
one work root:

```
$WORK_ROOT/repsim-signal   # this repository (REPO_ROOT)
$WORK_ROOT/resi            # ReSi checkout (RESI_DIR)
```

| Variable | Default | Meaning |
|---|---|---|
| `REPO_ROOT` | the directory containing `scripts/` | this repository |
| `WORK_ROOT` | parent of `REPO_ROOT` | shared root, mounted into the container |
| `RESI_DIR` | `$WORK_ROOT/resi` | ReSi checkout |
| `REP_SIM` | `$RESI_DIR/experiments` | ReSi's experiments/data directory (models, datasets, representations, results) |
| `RESI_RUNTIME` | `container` | `container` (Apptainer/Pyxis) or `native` (conda) |
| `CONTAINER_IMAGE` | `$HOME/containers/resi-manifold-pytorch-24.06-v2.sqsh` | image built by `scripts/build_resi_enroot_container.sh` |
| `CONTAINER_MOUNTS` | `$WORK_ROOT:$WORK_ROOT` | add `REP_SIM` here if it lives outside `WORK_ROOT` |
| `CONDA_BASE`, `CONDA_ENV` | `$HOME/miniforge3`, `resi-manifold` | native runtime only |
| `RESI_MAX_CONCURRENT_GPUS` | `8` | array concurrency limit |
| `RESI_EXCLUDE_NODELIST` | empty | nodes to exclude, e.g. ones where `REP_SIM` is not mounted |
| `MANIFOLD_RESI_DEVICE` | `cuda` for vision/language, unset for graphs | device for kernel construction |

Paths are written into generated manifests as absolute paths, so run
`status` and `analyze` on the same cluster (and, for the container runtime,
inside the same container) where the campaign was prepared.

## One-time setup

Container runtime (NGC `nvcr.io/nvidia/pytorch:24.06-py3`):

```bash
bash scripts/build_resi_enroot_container.sh
```

Native runtime (no container support):

```bash
bash scripts/setup_resi_native_env.sh
```

`scripts/download_resi_models.sh` downloads the CIFAR100 and SmolLM2
checkpoints into `$REP_SIM/models`, and
`scripts/build_resi_nlp_sft_datasets.py` builds the SmolLM2 SFT datasets
needed by the SmolLM2 language tasks.

## Submitting a campaign

```bash
# container runtime
bash scripts/submit_resi_graph.sh <run-name>
bash scripts/submit_resi_vision.sh <run-name>
bash scripts/submit_resi_language.sh <run-name>

# native runtime
bash scripts/submit_resi_language_native.sh <run-name>
```

Each submission prepares the campaign (`scripts/resi.py prepare`) in a
blocking SLURM job, submits the baseline manifest as one sequential job, then
submits the compute manifest as a SLURM array that depends on the baselines.
Pass `--compute-only` as the third argument to skip the baseline job.

The `#SBATCH` partition, QOS and resource lines in `scripts/*.sbatch` are
examples; adjust them for your cluster.

## Preemption and resumability

ReSi's `ExperimentStorer` saves a task's results after every completed
representation-pair comparison and skips completed comparisons on restart,
so a requeued task resumes on its own. The native templates set
`#SBATCH --requeue`. The write is not atomic: if a task is killed during the
write, its `results.parquet` is unreadable and the task fails loudly on
restart. Delete that task's `results.parquet`/`results_full.csv` and
resubmit its array index.

Reuse a run name to resume a campaign; use a fresh run name after changing
tasks, architectures, benchmarks, or measures.

## Validate and analyze

```bash
.venv/bin/python scripts/resi.py status --manifest <compute-manifest>
.venv/bin/python scripts/resi.py analyze --campaign configs/resi_graph_campaign.yaml \
  --run-name <run-name> --analysis-config configs/resi_graph_analysis.yaml
.venv/bin/python scripts/resi_cross_domain.py --help
```

See [running_experiments.md](running_experiments.md) for the analysis outputs
and [resi_analysis_methods.md](resi_analysis_methods.md) for the methodology.
