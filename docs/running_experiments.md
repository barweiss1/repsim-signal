# Running Experiments

This is the single place to find how to run every experiment in this
repository.

## Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -c constraints.txt
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

`requirements.txt` installs the full environment (core numerical stack,
Torch, the ReSi benchmark runtime, vision/plotting, and language tooling), and
`constraints.txt` pins it to the exact tested versions; without it pip's
resolver can backtrack into package versions that fail to build. The
editable install makes `manifold_repsim` importable from anywhere without
touching `sys.path`; `--no-deps` is intentional, since dependency resolution
is handled by `requirements.txt`. `pyproject.toml`'s `dependencies` and
`[project.optional-dependencies]` groups (`resi`, `notebooks`, `dev`) document
which packages each area needs; they are not an alternate install path.

Container-based ReSi runs use `requirements-resi-container-ngc24.06.txt`
instead — see [docs/resi_cluster.md](resi_cluster.md).

## Tests

```bash
MPLCONFIGDIR=/tmp/manifold-repsim-mpl \
  .venv/bin/python -m unittest tests.repository_suite -v
```

`tests.repository_suite` discovers every `test_*.py` module under `tests/`.

## Synthetic permuted-Gaussian MDS

Generates a full permuted-Gaussian-cluster transform grid, computes the
configured similarity matrices over it, and writes MDS comparisons that encode
up to three swept parameters as color, point size, and opacity.

```bash
.venv/bin/python scripts/synthetic_sweep.py \
  configs/synthetic_permuted_gaussian_mds.yaml
```

Three checked-in configurations cover the supported variants:

- `configs/synthetic_permuted_gaussian_mds.yaml` — the complete grid over
  cluster-mixing probability, noise scale, and permutation count.
- `configs/synthetic_permuted_gaussian_local.yaml` — mixing/noise only. It
  omits `grid.n_permute`; validation resolves it to a singleton equal to
  `base_transform.n_permute` (currently 0 throughout its 10-by-10 grid).
- `configs/synthetic_permuted_gaussian_permutation.yaml` — noise scale swept
  against permutation count, with cluster-mixing probability held fixed.

Notes that apply to all three:

- `transform.mode` selects `modify_base` (adds noise, swaps cross-cluster
  rows, and translates clusters from known source centers) or `resample`
  (legacy resampling).
- `stages` may be `[generate]`, `[analyze]`, or `[generate, analyze]`.
  Dataset and analysis fingerprints protect checkpoint reuse, so re-running
  with `[analyze]` alone reuses a valid checkpoint instead of recomputing it.
- Visual encoding channels (`color`, `size`, `opacity`) may be omitted when
  fewer than three parameters are swept.
- Similarity calculation reuses the shared metric core (`manifold_repsim.metrics`)
  while keeping this experiment's own efficient matrix computation.

## Glocal sweep MDS and PCA

Discovers aligned feature batches under a configured input root, compares
every selected glocal condition to `none`, concatenates each metric's batch
signals in stable batch-major order, and fits separate MDS and PCA views per
model and dataset.

```bash
.venv/bin/python scripts/glocal_sweep_analysis.py configs/glocal_sweep_mds.yaml
```

- The input root is `data/glocal/features/glocal_sweep`; this directory must
  already contain the extracted feature batches (this repository does not
  extract them). Conditions must use the strict name
  `glocal-lambda-<value>-alpha-<value>-tau-<value>`; global and naive
  directories are ignored.
- `stages` selects `[compute]`, `[plot]`, or `[compute, plot]` (compute must
  precede plot when both are listed). `compute` runs discovery, similarity
  scoring, and MDS/PCA fitting, checkpointing every result under
  `outputs.data_root` and an `analysis_manifest.json` that records each
  model/dataset group's metric result paths. `plot` alone reads that manifest
  and the per-metric NPZ checkpoints directly — it never rescans
  `input_root` — so replotting after a slow `compute` run, or iterating on
  the curated `scripts/paper_figures_glocal.py` output, is fast and does not
  require the feature directory to still be present or mounted.
- Selectors for models, datasets, batches, lambda, alpha, and tau accept `all`
  or exact YAML lists; the selected Cartesian grid must be complete.
- Models and datasets are analyzed independently. Only `none`-to-`none` and
  `none`-to-each-glocal-condition comparisons are computed — never
  glocal-to-glocal, global, naive, or cross-model comparisons.
- Full-grid and configured lambda-split outputs include individual and
  overview 2D/3D MDS figures plus 2D PCA of the raw concatenated signals.
  Signal plots cover tau at fixed lambda/alpha and alpha at fixed lambda/tau,
  with batch mean and a one-standard-deviation band.
- The 3D MDS camera defaults to an automatic orthographic view along the
  least-varying embedding direction (`mds_3d_view: auto`); the YAML may fix
  `elevation`/`azimuth` instead.
- Score tensors and analysis artifacts use configuration/source fingerprints
  and reuse valid checkpoints, so re-running without changing the config
  reuses prior work instead of recomputing it.

## Notebooks

These notebooks are exploratory drivers over the same package code as the
scripted experiments above; they are supported and kept on the package API,
but are not config-driven or checkpointed the way the scripts are.

- **`synthetic_experiment.ipynb`** — interactive parameter sweeps (metric
  hyperparameter, dimension, noise, cluster/ring count) over one configured
  synthetic dataset, using `manifold_repsim.experiments.legacy.run_param_sweep`
  and `plot_param_sweep`, plus a multi-metric comparison across a chosen data
  parameter via `run_metric_comparison_across_data_param`. Edit the
  `sim_params`/`base_transform_params`/`current_transform_params` dictionaries
  at the top to choose the dataset and transform.
- **`synthetic_experiment_calibrated.ipynb`** — the same style of sweep, with
  a permutation null computed alongside each signal
  (`sim_params["permutation_test"] = True`) and aggregate AUC / variance-
  weighted-AUC calibration via `manifold_repsim.sweeps`.

## ReSi campaigns

ReSi remains a separate checkout and is never patched; this repository
integrates with it at runtime by registering manifold measures into its
in-memory registry (`src/manifold_repsim/resi/`).

```bash
export RESI_DIR=/path/to/resi
export REP_SIM="$RESI_DIR/experiments"

.venv/bin/python scripts/resi.py prepare --campaign configs/resi_graph_campaign.yaml --run-name example
.venv/bin/python scripts/resi.py run-task --manifest <baseline-manifest> --index <i>
.venv/bin/python scripts/resi.py run-task --manifest <compute-manifest> --index <i>
.venv/bin/python scripts/resi.py status --manifest <compute-manifest>
.venv/bin/python scripts/resi.py analyze --campaign configs/resi_graph_campaign.yaml --run-name example \
  --analysis-config configs/resi_graph_analysis.yaml
```

Two SLURM runtimes are supported, both through the same
`submit_resi_campaign.sh` engine: a container runtime that runs each task
inside an Apptainer/Pyxis container (`submit_resi_<domain>.sh`), and a native
runtime for clusters without container support that runs inside a conda
environment (`submit_resi_<domain>_native.sh`, after a one-time
`scripts/setup_resi_native_env.sh`). `MANIFOLD_RESI_DEVICE` defaults to
`cuda` for vision/language. See [docs/resi_cluster.md](resi_cluster.md) for
the environment variables and submission commands.

- `configs/resi_graph_campaign.yaml`, `configs/resi_vision_campaign.yaml`, and
  `configs/resi_language_campaign.yaml` are the three checked-in campaign
  definitions.
- `prepare` prints the generated `baseline_manifest` and `compute_manifest`
  paths; run every zero-based index in the baseline manifest sequentially,
  then the compute manifest (directly, or submitted as a Slurm array via
  `scripts/resi_array_container.sbatch` or
  `scripts/resi_array_native.sbatch`).
- Reuse a `--run-name` to resume a campaign; use a fresh one after changing
  tasks, architectures, benchmarks, or measures.
- Run `status` and `analyze` on the cluster where the campaign was prepared,
  because generated manifests contain server-absolute paths.
- `analyze` writes normalized values, per-case ranks, domain summaries, and
  signal tables/figures under the campaign's analysis output root. Rank
  boxplots are written at three levels: domain-wide, per task (pooling every
  dataset that test ran on), and per (task, dataset). Those levels are keyed on
  the reporting task, not the benchmark id: the prediction-correlation
  benchmark runs two of the six tasks at once, so it writes `tasks/outcorr/`
  and `tasks/acccorr/` rather than one directory named after itself. Those two
  names are the same in all three domains even though each domain names the
  benchmark differently; every other benchmark keeps its own id.
- Ranking follows ReSi: a measure missing from an observation is skipped there
  rather than costing every other measure that observation, and each task is
  ranked on one quality measure and one functional similarity measure.
  Quality: `AUPRC`, `spearmanr`, `correlation`. Functional: `JSD` names
  `Output Corr.` and `AbsoluteAccDiff` names `Acc Corr.` -- six tasks, not
  five. `violation_rate` and `Disagreement` are still reported (the appendix
  tables' conformity-rate and Disagreement columns, the quality heatmaps, the
  AUPRC scatter) and never ranked, since ranking them would give a test two
  cases per cell where every other test has one. `Acc Corr.` is reported over
  whatever domains and architectures have accuracy values -- vision throughout,
  language's BERT-L only, graphs not at all. Each box pools one point per
  (task, dataset, architecture), with that case's seeds and layer models
  already averaged into it.
- Ranks are reported raw (1 = best), matching ReSi. `rank_scale: normalized`
  switches to `(rank - 1) / (n - 1)`, which matters only when the pooled cases
  ranked different-sized fields -- the cross-domain layer's permanent
  situation, and not a per-domain one. Both statistics are always written.
- Always pass `--analysis-config`. It is what declares the reported measure set
  and quality measures; without it the analysis ranks every measure present in
  the result parquets, which is a much larger and unevenly covered field. The
  reported ordering barely changes when this is forgotten, so the mistake shows
  up in the measure count rather than in the leaders -- check that before
  trusting an output.
- `--rank-sort quantile90` orders the rank boxplots by each measure's
  90th-percentile rank instead of its median, and draws that quantile on every
  box. It answers "how badly does this measure do when it does badly" rather
  than "where does it usually land"; the two can disagree, which is the point
  of having both. Set it permanently with `rank_sort:` in the analysis config.
  Nothing but the sort order and that marker changes.
- Every scope writes two rank boxplots: `rank_boxplot.png` under the
  configured sort and `rank_boxplot_mean.png` under the mean. The reported
  sort answers "how badly does this measure do when it does badly"; the mean
  answers "where does it land on average", and the two disagree exactly when a
  measure is usually good and occasionally terrible. The companion is skipped
  when the configured sort is already `mean`.
- `--rank-sort mean` orders by each measure's average rank instead of its
  median or its 90th-percentile tail. It draws no marker of its own -- the
  mean is already shown on every box regardless of sort (see below) -- so
  nothing else about the figure changes. Use it when a single skewed outlier
  case is pulling the mean and median apart. `quantile90` stays the reported,
  primary sort (`rank_sort: quantile90` in the checked-in analysis configs).
  Getting a `mean`-sorted `rank_boxplot.png` is a presentation-only rebuild
  over an existing `analyze` output's `per_case_ranks.csv` -- call
  `plotting.plot_rank_boxplot(..., rank_sort="mean")` directly -- not a second
  `analyze` run and not something `analyze` writes on its own.
- `--box-style boxen` draws each measure's distribution as a letter-value plot
  -- nested bands at the 10th, 30th, 50th, 70th and 90th percentiles tapering
  outward, with whiskers to the true min and max -- instead of a quartile box.
  Use it when two measures share a median and you need to see which one's
  failures are concentrated in a thin tail, or when you want the extremes shown
  rather than clipped at a whisker percentile. Set it permanently with
  `box_style:` in the analysis config. It is independent of `--rank-sort`: the
  drawing changes, the ordering does not. Under `quantile90` the diamond is
  dropped on a boxen, because the 10-90 band already ends at that quantile.
- Every measure's mean is drawn as a white circle with a black outline,
  unconditionally, under both `box` and `boxen`. It has no on/off switch: the
  mean is not read off either drawing's own statistics (a box's median, a
  boxen's percentile bands), so it is always worth showing.

### Cross-domain comparison

The per-domain outputs answer "which measure won this domain". Comparing the
three requires a separate step, because each domain names the same five
experiments differently:

```bash
.venv/bin/python scripts/resi_cross_domain.py \
  --domain graphs=figures/resi/<graphs-run>/graphs \
  --domain vision=figures/resi/<vision-run>/vision \
  --domain language=figures/resi/<language-run>/language \
  --out figures/resi/cross_domain
```

- Inputs are finished per-domain analysis directories, read from
  `normalized_values.csv` and `per_case_ranks.csv`. Re-run this after any
  analysis-config change, or its outputs will disagree with the per-domain ones.
- `--rank-sort` defaults to `mean` here, not the `median` the per-domain
  figures default to. Under `mean` and `median` the shared ordering is the
  aggregated `balanced_mean`, which weights the three domains equally; under
  `quantile90` it pools subtasks unweighted by domain, because a quantile has
  no balanced-mean analogue.
- `--rank-scale` defaults to `raw` (1 = best), matching the per-domain figures.
  `normalized` rescales each case to `(rank - 1)/(n - 1)`, which matters here
  because vision reports 15 measures where graphs and language report 16 -- a
  raw rank there is drawn from a slightly smaller field. Both statistics are
  written to `cross_domain_subtask_ranks.csv` either way, so switching needs no
  re-run of this step. On the current results the two scales give the identical
  ordering, overall and per family.
- Ranks are averaged into subtasks -- (domain, dataset, task, quality measure,
  functional similarity measure) -- before anything pools them, so a domain is
  not weighted by how many architectures it happens to run. The printed balance
  table reports the resulting unit counts per family and domain.
- The prediction-correlation benchmark contributes **two** families, not one.
  `Output Corr.` covers JSD and Disagreement -- two ways of asking how far two
  models' outputs agree, and two metrics of one task the way AUPRC and
  conformity rate are for the design tests. `Acc Corr.` covers
  `AbsoluteAccDiff`, which asks a different question: how far apart the two
  models' accuracies are. Two models can agree on almost nothing and still
  score identically, so these do not pool.
- A family reported by only some domains is **kept, in its own family**, not
  dropped. `Acc Corr.` is the current example: graphs never reports it (its
  archive carries no values for ReSi's own measures, so
  `exclude_functional_measures` drops it there), and language reports it on
  BERT-L only, since SmolLM2 has no accuracy in `eval_results.json`. The
  family is therefore built from vision and language, and the figure index
  notes `only 2 of 3 domains present`. `Output Corr.` stays balanced across
  all three domains, which is what the old whole-measure intersection was
  buying at the cost of deleting vision's and language's accuracy data.
  `combine_domain_values(..., harmonize_functional=True)` restores that older
  intersection if it is ever wanted. See
  [docs/resi_analysis_methods.md](resi_analysis_methods.md) section 7.9 for the
  full per-domain breakdown and the reasoning behind each gap.
- `--rank-sort quantile90` applies here too. Note that the cross-domain rank
  figure's shared ordering is the domain-balanced mean under `median` but the
  pooled 90th percentile under `quantile90`, because a quantile has no
  balanced-mean analogue -- so that ordering is not domain-balanced.
- `--rank-sort mean` is accepted here too, since it is the same `RANK_SORTS`
  list, but it is a distinct statistic from `median`'s own balanced-mean
  substitute above: `mean` orders by the plain pooled average rank across
  every case (like `quantile90`, not domain-balanced), while `median` orders
  by `balanced_mean`, the average of each domain's own median first. The two
  can rank measures differently whenever domains contribute unevenly many
  cases. This combination has no figure checked into the repository yet and
  no dedicated test; treat it as available, not as a reported convention.
- `--box-style boxen` applies here too, and reaches both the six-panel
  `cross_domain_ranks.png` and the per-family `*_ranks_by_type.png` figures.
- It also writes `exemplary.csv` / `exemplary.tex`: ReSi's Table 3 recreated,
  fixing one (dataset, architecture) cell per domain -- the source table's own
  GraphSAGE/Flickr, BERT-L/SST2, ResNet18/ImageNet100 -- with every test across
  the top, so the two tables can be read side by side. It carries a
  `Disagreement` column the source table omits, and no significance markers,
  since p-values are not propagated into the normalized frame. Change the cells
  via `analysis/exemplary.py::EXEMPLARY_CELLS`.
- Beside it, `averaged.csv` / `averaged.tex`: the same rows and the same
  columns, with each cell averaged over every model and dataset the domain ran
  instead of read off one named cell. Read the two together -- the exemplary
  table shows what one model did, the averaged one what the domain did on
  average, and a measure that looks strong in only one of them is telling you
  something. Each cell averages seeds first, then that model's datasets, then
  the domain's models, so a model that ran more datasets or more seeds does not
  weigh more than one that ran fewer; see
  `analysis/exemplary.py::AVERAGING_STEPS`. Cells with no value are skipped
  rather than emptying the mean, which is what lets vision's ViTs contribute
  their ImageNet100 correlations despite having none on CIFAR100.
- Both of these two tables mark the best three values in each column -- bold,
  underline, italics -- where the appendix tables mark only the best, as the
  source notebooks do. Marking is on the printed two-decimal value, so cells
  showing the same number always carry the same marker, and a tie shares its
  place rather than pushing the next value down a level. Change the depth for
  both at once via `analysis/exemplary.py::EMPHASIS_DEPTH`; the appendix
  tables' `build_overview_table` default stays at one.
- Measures print as mathematics, not as the classes that implement them:
  `$\mathrm{AUC}(\boldsymbol{s}_\mathrm{CKA})$` and its siblings for the swept
  measures, `$\mathrm{CKA}_\mathrm{lin}$`, `$\mathrm{CKA}_{\mathrm{RBF},0.2}$`,
  `$\mathrm{CKNNA}_{10}$`, `$\mathrm{MNN}_{10}$` for the fixed ones. ReSi's own
  measures keep the abbreviations its notebooks print. The random-walk
  alignments keep their trailing A, and the symmetric variant -- the reported
  one -- carries the plain `RWKA`, leaving `aRWKA` for the asymmetric variant
  nothing reports. One `$...$` string serves
  both Matplotlib's mathtext and LaTeX, so a figure and the table beside it
  cannot name the same measure differently; change a name in
  `analysis/config.py::MEASURE_LABELS`, which also resolves a label back to the
  measure whose colour and table block it takes. Every CSV keeps the raw
  `metric` column, and figure filenames keep the raw names.
- Runs locally; unlike `status` and `analyze` it reads no manifests and needs no
  cluster paths.

See [docs/resi_cluster.md](resi_cluster.md) for SLURM, container, and resume
instructions.

## Generated output

Datasets, figures, manifests, checkpoints, caches, and result files produced
by any of the above are artifacts, not source. `data/`, `figures/`, and
`logs` are gitignored; do not commit generated output unless explicitly
asked to.
