# Local-manifold measures on ReSi: data, implementation, and every deviation

Working notes for the paper's experimental section. Written to be complete and
honest rather than short: every deviation from ReSi's own analysis is listed
with which of three reasons it has --

* **[missing]** the data does not exist upstream,
* **[fairness]** ReSi's choice biases the comparison and we changed it
  deliberately,
* **[no principled reason]** cost, convenience, or an oversight we have not
  fixed.

Numbers are as of 2026-08-23. Where a number depends on a re-run that has not
happened yet, that is stated inline.

---

## 1. What the experiment asks

ReSi (Representational Similarity benchmark, ICLR camera-ready) evaluates
representational similarity measures not by agreement with each other but by
whether they can *ground* a known difference between models. It contributes six
tests in three domains. A measure is scored by a **quality measure** on each
test, and measures are then ranked against each other.

We register ten local-manifold similarity measures into ReSi's measure registry
and run them through the identical benchmark harness, against ReSi's own
archived results for its native measures. The claim we want to support is
therefore comparative and within-benchmark: *given ReSi's tests, do
manifold-geometry measures ground model differences better than the established
measures?*

Nothing in the benchmark protocol -- model checkpoints, representation
extraction, group construction, quality metrics -- is reimplemented. We add
measures and we re-do the aggregation. Section 6 is the aggregation.

---

## 2. Data

### 2.1 The three domains

| Domain | Datasets | Architectures | Benchmarks |
|---|---|---|---|
| Graphs | Cora, Flickr, ogbn-arXiv | GCN, GraphSAGE, GAT (+ PGNN, Cora only) | `augmentation_test`, `label_test`, `layer_test`, `output_correlation_test`, `shortcut_test` |
| Vision | ImageNet100, CIFAR100 | ViT_L32, ViT_B32, ResNet18/34/101, VGG11/19 | `augmentation`, `randomlabel`, `shortcut`, `monotonicity`, `accoutput` |
| Language | SST2, MNLI | BERT-L, SmolLM2-1.7B | `augmentation`, `memorization`, `shortcut`, `monotonicity`, `correlation` |

The five benchmark families are the same five experiments in all three domains;
only the identifiers differ. That equivalence is written out explicitly in
`resi/analysis/cross_domain.py::TEST_GROUPS` and is the only thing that makes a
cross-domain view possible.

Four coverage holes are structural, not chosen. They are worth separating by
*what* is missing, because the three kinds have very different fixes: absent
model checkpoints, absent evaluation numbers for models that do exist, and
absent architectures upstream.

* **PGNN (graphs) -- missing checkpoints.** PGNN exists only for Cora and has
  no `Augmentation_*` checkpoints, so it joins four of the five graph
  benchmarks. **[missing]**
* **CIFAR100 augmentation -- missing architectures.** No ViT checkpoints
  upstream, so that one benchmark runs 5 architectures rather than 7.
  **[missing]**
* **SmolLM2 augmentation -- missing models.** Language augmentation is BERT-L
  only. This is missing *models*, not a missing config: ReSi's
  `augmented_sst2_models` and `augmented_mnli_models` registries contain only
  multiberts and albert entries, the registry defines exactly seven SmolLM2
  path patterns (standard, `-mem*`, `-shortcut*`), and the 70 checkpoints on
  disk account for precisely those. It is not a download gap either -- there is
  no path to download from. Closing it would mean fine-tuning SmolLM2-1.7B on
  EDA-augmented SST2 and MNLI (at minimum the rate-10 pair the other SmolLM2
  experiments use: 5 seeds x 2 datasets = 10 fine-tunes), plus registering
  those models without touching the ReSi checkout. **[missing]**
* **SmolLM2 `AbsoluteAccDiff` -- missing results, not missing models.** See
  2.1a; this one is separate and much closer to fixable.

### 2.1a The SmolLM2 accuracy gap

SmolLM2 contributes 4 cases to the correlation test where BERT-L contributes 6,
because it reports two functional similarity measures against BERT-L's three.
`AbsoluteAccDiff` is absent for it. Unlike everything in 2.1, the models here
exist and the experiment was configured to use them:
`correlation_nlp_standard_{sst2,mnli}_smollm.yaml` both set
`use_acc_comparison: true`.

What actually happens is that two cases -- `correlation | {sst2,mnli} |
smollm2-1.7b | spearmanr | AbsoluteAccDiff` -- come back with **all 16 measures
at `nan_fraction = 1.0`**. The rows exist; every value is blank. Coverage
filtering then drops the cases entirely, which is why they read as absent
rather than as NaN. This is the same shape as the graph archive gap in 7.2.

The mechanism is a lookup miss.
`repsim/utils.py::NLPModelAccuracy._extract_output` reads
`$REP_SIM/models/nlp/eval_results.json`, indexes it by model id, and on
`KeyError` logs and returns NaN. That file holds 220 model ids and **zero**
SmolLM2 entries: the accuracies were simply never computed. Nothing is wrong
with the models, the config, or our analysis.

It is the most fixable of the gaps, and also the least worth fixing:

* ReSi already ships the code. `nlp/eval.py` has an `evaluate_smollm` function
  dispatched on `architecture == "smollm2-1.7b"`, and its main loop always
  evaluates a model on its own `train_dataset` -- which for these is
  `sst2_sft`/`mnli_sft`, exactly what the correlation experiment looks up.
* But running it as shipped would not help. The lookup key is
  `eval_results[id][representation_dataset][split]`, where `split` comes from
  the registry and is `"train"` for the SFT datasets, while `nlp/config/eval.yaml`
  evaluates on `validation` / `validation_matched`. Writing the wrong split key
  reproduces the same NaN silently.
* The cost is real but bounded: `split="train"` means 67,349 (SST2) and 392,702
  (MNLI) examples per model across 20 models. Known feasible -- the correlation
  experiment already ran full forward passes over exactly these datasets for
  exactly these models, which is how JSD and Disagreement got their values --
  but it is a multi-hour GPU job.
* Results would have to be injected rather than written: the file the lookup
  reads sits inside the ReSi checkout, which this project never modifies. The
  narrow-hook pattern already used for measure registration and AUC signals
  applies.

**The payoff is small, which is why this is documented rather than fixed.**
It would add exactly two cases to the language-only ranking (38 to 40), and in
cross-domain it would add one more language subtask to the `Acc Corr.` family
-- a family graphs cannot join at all, since its archive has no
`AbsoluteAccDiff` for ReSi's own measures (7.2) and that gap is not ours to
close. So closing the SmolLM2 side sharpens one two-domain family slightly and
changes nothing else. Reconsider if the paper leans on per-domain `Acc Corr.`
tables and wants architecture parity within language; even then, graphs stays
absent.

One caution if it is ever revisited. Those two extra cases would take
`SecondOrderCosineSimilarity` in language from 36/38 = 0.947 to 38/40 = 0.950,
which passes `min_case_coverage` (the test is strict `<`). That is a
side-effect, not a justification: computing a dataset in order to rescue a
particular measure from a threshold is the same after-the-fact reasoning 7.3
warns about.

### 2.1b Coverage summary: domains x tests

One table for all of 2.1, 2.1a, 2.3, 7.2, and 7.9's coverage detail, gathered
in one place for a reader who wants the shape of the gaps before the reasons.
Every cell that is not fully "Available" points back to the section with the
full argument; nothing here is a new claim.

The prediction-correlation benchmark contributes two rows, because it answers
two questions: `Output Corr.` (JSD and Disagreement, how far two models'
outputs agree) and `Acc Corr.` (AbsoluteAccDiff, how far apart their
accuracies are). See 7.9.

| Test | Graphs (GCN, GraphSAGE, GAT, PGNN) | Vision (ViT_L32/B32, ResNet18/34/101, VGG11/19) | Language (BERT-L, SmolLM2-1.7B) |
| --- | --- | --- | --- |
| Augmentation | Available, GCN/GraphSAGE/GAT full; PGNN absent -- no `Augmentation_*` checkpoints exist for PGNN at all, Cora only (2.1, models not published by authors) | Available; ImageNet100 full 7 architectures, CIFAR100 only 5/7 -- no ViT checkpoints upstream for CIFAR100 augmentation (2.1, models not published by authors) | Available for BERT-L only; SmolLM2 absent -- ReSi's augmented-model registries list only multiberts/albert entries, no SmolLM2 path (2.1, models not published by authors; not a download gap, would need fine-tuning ourselves) |
| Random Labels | Available, all 4 architectures | Available; 2 checkpoints excluded -- `ViT_L32`/seed 4 non-finite, `ViT_B32`/seed 0 collapsed representations (2.3, data-quality gap, not a publishing gap) | Available, BERT-L and SmolLM2 both |
| Shortcuts | Available, all 4 architectures | Available, all 7 architectures | Available, BERT-L and SmolLM2 both |
| Layer Mono. | Available, all 4 architectures | Available, all 7 architectures | Available, BERT-L and SmolLM2 both |
| Output Corr. (`JSD`, `Disagreement`) | Available, both measures | Available, both measures | Available, both measures, BERT-L and SmolLM2 |
| Acc Corr. (`AbsoluteAccDiff`) | Absent entirely -- ReSi's own archive carries no `AbsoluteAccDiff` values for its native measures at all (7.2, results not published by authors) | Available, all architectures | Available for BERT-L only; SmolLM2 absent -- accuracies never written to `eval_results.json` (2.1a, results not published by authors; models exist and `use_acc_comparison: true` is set, so fixable in principle but adds only two language cases) |

ALBERT is not a cell above: it is not a test gap but an architecture that was
never part of language at all. `registry.py` is wired up for it, but its
checkpoints were never published -- confirmed directly by listing the full
731-entry contents of ReSi's own language/graph Zenodo archive
(`nlp_data.zip`, https://doi.org/10.5281/zenodo.11565486) without downloading
it, and by an exhaustive `find` for `*albert*` across both compute clusters
and their Hugging Face caches. Nothing came back on either check.

Reported *measure* count (16 for graphs/language, 15 for vision, since
`LinearRegression` falls below the coverage threshold there -- 6.6) is a
ranking detail, not a test-availability one, and is not in this table.

### 2.2 Where the baseline numbers come from

Three provenances, and they are not interchangeable:

1. **Archived.** ReSi ships result parquets for its full 24-measure suite on
   ImageNet100, BERT-L, and the three archived graph architectures. We read
   these directly; no recomputation.
2. **Recreated.** ReSi ships *no* baselines for CIFAR100, SmolLM2, or PGNN.
   `scripts/prepare_resi_baselines.py` runs ReSi's own harness for those, with
   `included_measures` set to a **reduced 9-measure set**:
   `AlignedCosineSimilarity, CKA, DistanceCorrelation, JaccardSimilarity,
   LinearRegression, ProcrustesSizeAndShapeDistance, RankSimilarity, SVCCA,
   SecondOrderCosineSimilarity`.
3. **Ours.** The ten manifold measures, computed for every architecture in
   every domain.

Consequence, and it drives Section 5.1: an architecture with archived results
can rank 24 native measures, while a recreated one can rank 9. Ranking is
within-cell, so pooling the two without care would rank measures against
different-sized fields.

### 2.3 Two known-bad vision checkpoints

Declared in `configs/resi_vision_campaign.yaml::known_missing_models`, excluded
from analysis, and reported in the coverage report rather than silently dropped:

* `ViT_L32 / randomlabel / seed 4`: final-layer representations are entirely
  non-finite upstream. **[missing]**
* `ViT_B32 / randomlabel / seed 0`: final-layer representations collapse -- over
  half of all pairwise distances sit at a single float32 quantization step, so
  no RBF bandwidth is resolvable and every kernel measure fails. Neighbour
  measures still return numbers, but only because exact ties break by index
  order, which makes those values an artifact of storage layout rather than
  geometry. **[missing]**

The second is worth a sentence in the paper: it is a failure mode that
*silently produces plausible numbers* for neighbour measures. It is the reason
`resolve_rbf_base_bandwidth` carries a representation-scaled distance floor
instead of an absolute constant.

---

## 3. The measures we contribute

All ten operate on the same input contract as ReSi's measures: two
representation matrices $X \in \mathbb{R}^{n \times p}$, $Y \in \mathbb{R}^{n \times q}$
over the same $n$ inputs, flattened through ReSi's own `flatten` helper, with
larger = more similar.

### 3.1 Fixed-parameter measures (5)

Let $K, L$ be RBF kernels
$$K_{ij} = \exp\!\left(-\frac{\lVert x_i - x_j\rVert^2}{2\sigma_X^2}\right), \qquad
\sigma_X = s \cdot \operatorname{median}_{i>j}\lVert x_i - x_j\rVert$$
with the bandwidth multiplier $s$ reported at two values, $0.5$ and $0.2$.
Neither follows from a rule: the median scaling makes the bandwidth
scale-invariant, but the multiplier on top of it is a choice, and the plain
median heuristic would be $s = 1$. Reporting a second multiplier is what
makes the first one's standing checkable -- if the two land far apart, the
reported number depends on a parameter nothing selected. The median is
taken over
distances above a representation-scaled numerical floor; if none clear it, the
bandwidth falls back to $1$.

| Name | Definition | Category |
|---|---|---|
| `CKArbfSigma05` | $\mathrm{CKA}(K,L) = \dfrac{\mathrm{HSIC}(K,L)}{\sqrt{\mathrm{HSIC}(K,K)\,\mathrm{HSIC}(L,L)}}$ with RBF kernels | RSM |
| `MutualKNNTop10` | $\dfrac1n \sum_i \dfrac{\lvert N_X(i) \cap N_Y(i)\rvert}{k}$, $k=10$, neighbours by **inner product**, self excluded | Neighbors |
| `CKNNATop10` | CKA on Gram matrices masked to shared $k$-neighbourhoods, $k=10$ | Neighbors |

`MutualKNNTop10` and `CKNNATop10` clamp $k$ to at most $n-1$ for small inputs.

### 3.2 Swept measures (4)

Each sweeps the bandwidth multiplier $s$ over a **100-point log-spaced grid on
$[0.05,\,10]$** (set per campaign in `measure_sweeps`; the adapter's own default
of 30 points is overridden), producing a *similarity signal*
$\{(s_t, \rho_t)\}_{t=1}^{100}$, reduced to one number by an **unweighted mean**:
$$\mathrm{AUC} = \frac{1}{T}\sum_{t=1}^{T} \rho_t .$$

| Name | Base kernel score |
|---|---|
| `CKArbfAUC` | RBF CKA |
| `sRWKArbfAUC` | symmetric normalized alignment $\langle S_K, S_L\rangle$ |
| `UKArbfAUC` | uncentered alignment $\langle K, L\rangle_F / (\lVert K\rVert_F \lVert L\rVert_F)$ |
| `dRWKArbfAUC` | degree-mode-removed alignment |

> **Naming honesty.** "AUC" is a misnomer: it is the mean of the curve, not a
> trapezoidal integral, and the grid is log-spaced so it is not even
> proportional to an area in $s$. It is a bandwidth-averaged similarity. The
> field name `integration_method` is recorded as `"average"` in every stored
> signal, so the artifacts are self-describing, but the paper should call it a
> *sweep-averaged score*. **[no principled reason -- inherited name]**

The full 100-point curve is retained on `last_similarity_signal` and embedded as
JSON in the result row before checkpointing, so every AUC value is auditable and
the curves themselves are a second, independent result (`signals_long.csv`).

`mcRWKArbfAUC` and `dRWKArbfAUC` were built as a deliberate pair: mean
centering annihilates the constant and both norm terms of the large-bandwidth
kernel expansion, so `mcRWKArbfAUC` should tend to linear CKA as $s \to \infty$;
degree centering removes a different mode and carries no such guarantee.
`mcRWKArbfAUC` is no longer reported (§3.3), so that comparison is available in
the parquets but is not part of the headline field.

### 3.3 Computed but not reported

`RWKArbfSigma05`, `SoftmaxRWKATemp05`, `MutualKNNAUC`, `CKNNAAUC`,
`RWKArbfAUC`, `RWKAsoftmaxAUC` are in the catalogue but excluded from all three
campaigns, to hold the reported field at a manageable size. **[no principled
reason -- cost]**

Three more are computed by every campaign and sit in the result parquets, but
are excluded from reporting by the analysis configs: `sRWKArbfSigma05` and
`dRWKArbfSigma05` -- the fixed-bandwidth random-walk variants, dropped in favour
of their swept counterparts, which answer the same question without committing
to $s = 0.5$ -- and `mcRWKArbfAUC`. Restoring any of them is a one-line config
edit and no recomputation. **[no principled reason -- field size]**

That leaves **7 manifold measures reported**: `MutualKNNTop10`, `CKNNATop10`,
`CKArbfSigma02`, `sRWKArbfAUC`, `CKArbfAUC`, `UKArbfAUC`, `dRWKArbfAUC`.
(`CKArbfSigma05` was dropped later, in favour of the `CKArbfSigma02` added to
check its undermotivated bandwidth -- see 7.3.)

Every such removal changes the size of the field a rank is drawn against, so it
changes every rank in the domain. The analysis has to be re-run; an existing
`per_case_ranks.csv` cannot be filtered after the fact.

### 3.4 Exactness

No approximation is used anywhere in the reported runs. Subsampling exists
(`MANIFOLD_RESI_MAX_POINTS`, `MANIFOLD_RESI_AUC_MAX_POINTS`) but is opt-in via
environment variables and was never set. Optimized and naive score paths are
tested to agree.

---

## 4. Quality measures (ReSi's, unchanged)

Three benchmark shapes, six quality measures, all computed by ReSi's own code.

### 4.1 Group separation (`augmentation`, `randomlabel`/`memorization`, `shortcut`)

Models are partitioned into groups that differ by design. For each group $g$,
collect intra-group similarities $A_g$ and cross-group similarities $B_g$.

* **AUPRC**: label intra-group pairs positive, run
  `sklearn.average_precision_score`, average over groups. Distance-type measures
  are sign-flipped and rescaled first.
* **Violation rate**:
  $$\mathrm{VR}_g = \frac{\lvert\{(a,b) \in A_g \times B_g : a \le b\}\rvert}{\lvert A_g\rvert\,\lvert B_g\rvert},$$
  averaged over groups. Lower is better; we report its complement (see 6.7).

NaN handling here is upstream's and is asymmetric between the two: violation
rate drops NaN similarities and returns NaN if a whole group is lost; AUPRC uses
whatever non-NaN comparisons remain. We inherit both.

### 4.2 Layer monotonicity (`monotonicity`/`layer_test`)

For each model, similarity between layer $i$ and layer $j$ should decay with
$\lvert i - j\rvert$.

* **`correlation`**: Spearman $\rho$ between similarity and layer distance,
  computed forward and backward from each anchor layer and averaged
  (`np.nanmean`). Sign-flipped for distance measures.
* **`violation_rate`**: fraction of ordered layer triples whose similarities
  violate the expected ordering.

Note that this test reports **one value per layer model**, unlike every other
test -- which is why both ranking conventions average within a cell before
pooling (6.4, 6.5).

### 4.3 Output correlation (`accoutput`/`correlation`/`output_correlation_test`)

Across all model pairs, correlate representational similarity against a
**functional** similarity measure: `JSD` (Jensen-Shannon divergence between
output distributions), `Disagreement` (prediction disagreement rate), and
`AbsoluteAccDiff` (absolute accuracy difference). ReSi computes Pearson,
Spearman, and Kendall $\tau$ for each; with ~10 models there are only 45 pairs,
so it also runs a permutation test.

**We report Spearman only, as ReSi does** -- see 7.1.

---

## 5. Pipeline and reproduction

### 5.1 Stages

```
prepare   campaign YAML -> per-task ReSi configs + JSONL manifests
run-task  one manifest index -> ReSi experiment -> results.parquet (+ full CSV)
status    manifest -> complete / incomplete / missing per index
analyze   manifests -> normalized frame -> ranks, tables, figures
```

`scripts/resi.py` is the CLI; the campaign YAML is the single declaration of
benchmarks, datasets, architectures, measures, sweeps, and baseline parquets.
Manifests are JSONL and carry server-absolute paths, so `status` and `analyze`
must run on the machine that produced them.

```bash
python scripts/resi.py prepare  --campaign configs/resi_<domain>_campaign.yaml --run-name <run>
python scripts/resi.py run-task --manifest <run-dir>/compute_manifest.jsonl --index <i>
python scripts/resi.py status   --manifest <run-dir>/compute_manifest.jsonl
python scripts/resi.py analyze  --campaign configs/resi_<domain>_campaign.yaml \
                                --run-name <run> \
                                --analysis-config configs/resi_<domain>_analysis.yaml
```

Passing `--analysis-config` is **not optional**. Without it the analysis ranks
every measure present in the parquets (33-34 in vision/graphs) instead of the
declared 19, and -- this is the trap -- the reported ordering barely changes,
because the extra measures are mostly absent from the cells that matter. The
symptom shows in the measure count, not in the leaders.

### 5.2 Compute

Two SLURM clusters, split by resource availability:

* **Container cluster**: vision and graphs, under pyxis/enroot with the NGC
  image `nvcr.io/nvidia/pytorch:24.06-py3` and
  `requirements-resi-container.txt`.
* **Native cluster**: language, in a conda environment (`resi-manifold`) on a
  shared preemptible partition. Preemption is handled by ReSi's own
  `ExperimentStorer`, which writes after each completed comparison and skips
  completed ones on restart.

`MANIFOLD_RESI_DEVICE=cuda` for vision and language; graphs run on CPU.

### 5.3 Reproduction

The external ReSi checkout is never modified: measures are registered into its
in-memory registry at runtime, and the only hook installed is the narrow one
that embeds the similarity-signal JSON into the result row. `git status` in the
checkout is clean.

Two files fully determine a reported number: the campaign YAML (what was
computed) and the analysis YAML (what is reported). Both are snapshotted into
the output directory as `config.yaml`.

---

## 6. Analysis

This is where we deviate most, so each step states the deviation inline.

### 6.1 Normalization

Every manifest entry's full CSV is read into one long frame with columns
`(domain, benchmark, dataset, architecture, observation_id, metric,
quality_measure, value, direction, source, identifier, representation_dataset,
model, functional_similarity_measure, source_file)`.

An **observation** is the atom being scored:
$$\text{observation\_id} = (\text{identifier},\ \text{representation\_dataset},\ \text{model},\ \text{functional\_similarity\_measure}).$$
`model` is non-trivial only for layer monotonicity.

Analysis reads *explicit manifest entries only*. It never reconstructs data by
globbing the result directory, because a stale parquet from an earlier run is
indistinguishable from a current one by filename alone.

### 6.2 The case

A **case** is one comparable evaluation context -- the unit inside which ranking
is meaningful:
$$\text{case} = (\text{domain},\ \text{benchmark},\ \text{dataset},\ \text{architecture},\ \text{quality\_measure},\ \text{identifier},\ \text{representation\_dataset},\ \text{functional\_similarity\_measure}).$$

Case counts are how domains get weighted, so any filter that adds or removes
cases reweights the result -- twice now that has been the actual bug (7.1, 7.2).

### 6.2b The subtask

Because a case is keyed on `architecture`, a domain's case count tracks how many
architectures it happens to ship rather than how much evidence it carries.
Vision runs 7 architectures against graphs' 3-4 and language's 1-2, which is
most of why vision held 180 of 266 cross-domain cases. Cross-domain reporting
therefore averages first, over a coarser unit:
$$\text{subtask} = (\text{domain},\ \text{dataset},\ \text{test\_group},\ \text{quality\_measure},\ \text{functional\_similarity\_measure}).$$

`architecture`, `identifier`, and `representation_dataset` are averaged over;
the quality measure and the functional similarity measure are not, because an
AUPRC rank and a conformity-rate rank answer different questions and their mean
is not a quantity. See §6.8 for what this buys.

**Figure $n$ counts subtasks**, with the cases behind them shown alongside:
a panel reading `n=7 subtasks, 28 cases` drew 7 points per box, each the mean
of 4 case-level ranks. Per-domain figures are unchanged and still count cases.

### 6.3 What gets ranked: one quality and one functional measure per task

Ranking uses the three quality measures ReSi ranks on -- `AUPRC`, `spearmanr`,
`correlation` (`RANKED_QUALITY_MEASURES`). Its rankplot notebook filters to
exactly these, and its overview table reaches the same set benchmark by
benchmark.

This is not a per-domain setting; it is fixed in code, because it is ReSi's
choice rather than one this repository makes per run. Two things fall outside
it:

* `pearsonr` and `kendalltau`, computed by the prediction-correlation
  experiment and never reported (7.1).
* `violation_rate`, which *is* reported but is a second reading of the same
  tests AUPRC already scores. A case is keyed on the quality measure, so
  ranking it too would give every design-grounded test two cases where a
  correlation test has one -- twice its published weight in the domain summary,
  the rank boxplot, and everything downstream. Dropping it from ranking took
  graphs from 98 cases to 59, vision from 146 to 92, and language from 38 to
  24, with the correlation tests untouched.

**The same rule on the other axis.** The correlation benchmark scores one set
of representations against several functional similarity measures at once, and
a case is keyed on which, so the same double-counting arrives by a second
route. `RANKED_FUNCTIONAL_MEASURES` reports each correlation task on one
measure: `Output Corr.` on `JSD` and `Acc Corr.` on `AbsoluteAccDiff`.
`Disagreement` is the second of two readings of the question `Output Corr.`
already asks, so it is reported and not ranked -- matching ReSi's own overview
table (7.8). Rows with no functional measure, meaning every design-grounded
test and layer monotonicity, are always kept.

This leaves **six tasks, not five**: `Acc Corr.` is a task of its own, reported
over whatever domains and architectures have `AbsoluteAccDiff` values rather
than merged, padded, or harmonized away.

Both filters together take the ranked case count to one per (task, dataset,
architecture):

| Domain | reported cases | after quality filter | after functional filter |
|---|---|---|---|
| Graphs | 98 | 59 | 49 |
| Vision | 146 | 92 | 80 |
| Language | 38 | 24 | 20 |

Both are applied at ranking time, not at load, so conformity rate and
`Disagreement` still reach the appendix tables, the quality heatmaps, and the
AUPRC-versus-violation scatter. The run's `config.yaml` records
`ranked_quality_measures` and `ranked_functional_measures` alongside the
reported `quality_measures`, so the distinction is on the record rather than
left for a reader to infer.

### 6.4 Ranking, repository convention (primary)

Within a case:

1. Drop non-finite cells.
2. Pivot to observations $\times$ measures.
3. Rank across measures with `method="average"`, direction from the quality
   measure. A measure missing from an observation is **skipped there** -- pandas'
   `na_option="keep"` default, and ReSi's convention. It neither scores nor
   displaces anything, and the measures present rank $1 \ldots k$ over
   themselves alone.
4. Average each measure's rank over the observations it was ranked in.
5. Normalize:
   $$\tilde r = \frac{r - 1}{k - 1} \in [0,1], \quad 0 = \text{best},$$
   where $k$ is the number of measures **that observation** ranked.

Cases with fewer than two measures contribute nothing, as does any observation
that ranked fewer than two: a field of one awards rank 1 to whichever measure
happened to survive, which reads as a win and is not one.

Step 4 is what collapses seeds. A case's observations are its layer models --
one per seed for layer monotonicity, a single aggregate row for every other
test, since AUPRC and the correlations already aggregate over seeds themselves.
Averaging first gives each case, meaning each (task, dataset, architecture,
functional similarity measure), exactly one point in the boxplot and one vote in
the domain summary, regardless of how many seeds it contains. Same reasoning as
ReSi's own layer-monotonicity averaging, applied uniformly.

`per_case_ranks.csv` records `n_ranked_observations` against
`n_case_observations`, so a mean drawn from part of a case is visible rather
than implied.

**Step 3 was previously `dropna(axis=0, how="any")`**: an observation entered
the ranking only if every compared measure was finite there. That made one
measure's failure cost every other measure the observation as well, which is a
larger distortion than the one it was guarding against -- and the thing it was
guarding against is what §6.6 is for. The two conventions differ on 1 case in
graphs, 1 in vision, and 0 in language on the current results.

**Scale.** `rank_scale: raw` is the default and what the per-domain figures
report: within one domain the field is fixed by the `measures` list and the
coverage thresholds, so a raw rank is directly readable and matches ReSi.
`normalized` is what pooling across domains needs, since vision ranks 15
measures and graphs and language 16; the cross-domain layer uses it regardless
of the setting. Both statistics are always written, so switching changes
ordering and the plotted axis, never what was measured.

### 6.5 Ranking, published convention (ReSi's, reproduced alongside)

`resi/analysis/tables/ranking.py` reproduces ReSi's notebook exactly: filter to
the three ranked quality measures (§6.3), rank within
`(Domain, Test, Eval., Dataset, Arch., model, FSM[, Token])` with
`method="min"` and `na_option="keep"`; average layer-model ranks within the
cell; deduplicate. The recreated appendix tables and `rank_boxplot_published.png`
are built from this.

Both conventions are computed and written on every run. They now agree on NaN
handling and on which quality measures are ranked; what remains different is:

| | repository | published |
| --- | --- | --- |
| ties | `method="average"` | `method="min"` |
| coverage filtering (§6.6) | applied | none |
| observation ranking one measure | skipped | ranked, awarded rank 1 |
| `Disagreement` | reported, not ranked (§6.3) | ranked as its own setting |

The last row is not an oversight in either direction: ReSi's overview table
keeps one functional measure per correlation task and its rank figure keeps all
three, so following ReSi *exactly* means the published figure keeps them.

A claim that survives both is worth more than one that needs a convention. On
the current results they agree on the top measure in every domain.

### 6.6 Coverage filtering **[fairness]**

Not in ReSi. Two thresholds, both recorded to `measure_exclusions.csv` with the
exact case ids. Both address the same asymmetry, which is the one ranking
cannot see: ranking only ever compares what is present, so a measure that fails
on the hard inputs is not penalised for it -- it is simply scored on the easy
ones, and its remaining ranks are a flattering subset rather than a partial
result.

* `max_nan_fraction = 0.5` -- within a case, drop a measure whose NaN share
  exceeds half. Its remaining observations are not a random sample of the case.
* `min_case_coverage = 0.90` -- within a reporting scope, drop a measure that
  survived fewer than 90% of the scope's cases, for the same reason one scope up.

Effect on the current runs: vision drops `LinearRegression` (116/146 = 0.795 --
wholly NaN for CIFAR100 VGG11/VGG19 in every benchmark and for randomlabel on
five ImageNet100 architectures), leaving vision at 15 measures against 16 for
graphs and language. It is the only measure any domain drops on coverage, and
all manifold measures sit at full coverage in all three domains.

This threshold is a judgement call and the paper should say so: neither 0.90
nor the 0.95 it replaced is derived from anything. What 0.90 is chosen *for* is
recorded: at 0.95 the guard also dropped language's
`SecondOrderCosineSimilarity` for missing two cases out of 38 (0.947), a
rounding-level gap rather than the materially-easier-field problem the guard
exists for. The two candidates it separates -- 0.795 and 0.947 -- are far
enough apart that any threshold between them gives the same reported set.

### 6.7 Reported orientation

`violation_rate` is the only lower-is-better quality measure. Ranking respects
that directly; figures and tables report its complement $1 - \mathrm{VR}$ as
"Conformity Rate", so every plotted axis reads higher-is-better. Matches ReSi's
own tables.

### 6.8 Cross-domain aggregation **[new -- not in ReSi]**

ReSi reports per domain. We add a cross-domain layer, which requires four
things to be true and states each:

1. **Benchmark equivalence.** The 12 domain-specific benchmark ids map onto 5
   families (`TEST_GROUPS`). A benchmark not in the map is dropped, not guessed.
2. **Correlation task split.** Correlation cases exist once per functional
   similarity measure, and those measures answer two different questions, so
   `test_group_for` sorts them into two families: `Output Corr.`
   (`{JSD, Disagreement}`, reported by every domain) and `Acc Corr.`
   (`{AbsoluteAccDiff}`, reported by vision and language only; graphs has none,
   see 7.2). Applied in *both* the value and the rank paths. This replaced an
   intersection that kept only `{JSD, Disagreement}` and discarded the accuracy
   rows outright; the split keeps them without unbalancing anything, because a
   family is weighted only over the domains that populate it.
   `harmonize_functional=True` restores the old intersection.
3. **Published quality measures only.** Spearman for correlation tests (7.1).
4. **Subtask averaging before pooling** (§6.2b, and 7.11 for why).

Ranks pool across domains as raw within-case ranks, matching the per-domain
figures. That carries one caveat worth a caption: vision reports 15 measures
where graphs and language report 16, so a raw rank there is drawn from a field
one measure smaller -- mid-pack is 8.0 against 8.5, flattering a measure present
in vision by about half a rank. `--rank-scale normalized` rescales each case to
$\tilde r = (r-1)/(n-1)$ and removes exactly that, at the cost of an axis that
no longer reads as a placing. Both statistics are written to
`cross_domain_subtask_ranks.csv` either way, so switching needs no re-run.

On the current results the two scales produce the **identical** measure
ordering, overall and in all six families, so the caveat is real in principle
and inert in fact here. Values do **not** pool across quality measures; value
figures panel by quality measure.

#### Two-stage aggregation

Stage one collapses each subtask's cases to one number per measure $m$:
$$\bar r_{s,m} = \frac{1}{\lvert C_s\rvert}\sum_{c \in C_s} \tilde r_{c,m},$$
where $C_s$ is the set of cases in subtask $s$ -- one per architecture (times
`identifier` and `representation_dataset` where those vary). Stage two pools
the $\bar r_{s,m}$, two ways because they can disagree:
$$\text{pooled}_m = \frac{1}{\lvert S\rvert}\sum_{s \in S} \bar r_{s,m},
\qquad
\text{balanced}_m = \frac{1}{\lvert D\rvert}\sum_{d \in D} \frac{1}{\lvert S_d\rvert}\sum_{s \in S_d} \bar r_{s,m}.$$

Boxplots draw the $\bar r_{s,m}$ directly, so a box holds $\lvert S\rvert$
points.

The subtask key was chosen against two alternatives on the actual data:

| grouping | units | per family | per domain |
|---|---|---|---|
| **chosen** | **70** | **14 each** | graphs 30 / vision 20 / language 20 |
| also collapse quality measure | 35 | 7 each | 15 / 10 / 10 |
| also split identifier + representation dataset | 86 | 14-18 | 30 / **36** / 20 |

Only the chosen key gives every family the same number of units. Graphs carries
30 to the others' 20 because it genuinely runs three datasets where they run
two -- a structural fact about the benchmark, not an artifact of the weighting.
After this the two stage-two weightings nearly coincide (`CKArbfAUC`: pooled
0.352, balanced 0.355), which is the point: with a near-balanced design, how
you weight stops mattering.

`n_domains` is reported alongside, because a measure missing from a domain
(LinearRegression) is a two-domain average competing against three-domain
averages. `n_cases` is retained next to `n_subtasks` so a point can be traced
back to the evaluation cells behind it.

**Averaging shrinks spread, and that is not a stronger result.** A case-level
box showed variation across evaluation cells; a subtask box shows variation
across subtasks, with the within-subtask variation averaged out. The boxes are
visibly narrower for that reason alone. With 7 points a box the 5th/95th
whiskers are effectively min/max, so read them as range, not as a tail.

**AUPRC is not calibrated across domains.** It is on $[0,1]$ everywhere but its
chance level depends on how many groups a benchmark separates, which differs by
domain. Value panels reporting AUPRC carry that annotation. Ranks are immune;
values are not.

### 6.8 Figure conventions

* Boxes use **5th/95th percentile whiskers**, not Tukey. Several benchmarks
  saturate -- vision shortcut AUPRC is exactly $1.0$ in more than half the cases
  for most measures -- which drives the IQR to zero, collapses Tukey whiskers
  onto the box, and renders the distribution as a hairline that reads as missing
  data. **[fairness -- presentation]**
* **One boxplot implementation.** `analysis/boxplots.py` owns the single
  `ax.boxplot` call, the figure geometry, the box and median styling, and the
  legend; `analysis/measure_style.py` owns the labels, colours, and ordering it
  reads. Every measure boxplot in the repository draws through them, so figures
  cannot drift apart -- which they had: three colour schemes, two copies of the
  Matplotlib orientation shim, four legend builders, and four median-styling
  loops, one of which was missing entirely so those medians rendered in
  Matplotlib's default orange.
* Measures are labelled with ReSi's published abbreviation (manifold measures
  keep their class names, which are already short) and coloured by measure
  category (Neighbors / RSM / Alignment / Topology / CCA / Statistic / Signal),
  using ReSi's own colourblind palette order plus one added hue for the swept
  measures. Our measures deliberately do
  **not** get their own colour: they land in Neighbors, RSM, and Signal, and
  they do not behave as a family (`CKArbfAUC` first, `CKNNATop10` near last).
  The recreated tables group their rows by the same categories, so a measure
  cannot sit in one block in a table and another in the boxplot beside it. The
  source tables put every local measure in one `Measure Type` = "Manifold"
  block; we do not reproduce that, since one group across three categories
  asserts a family the results contradict.
* Only the table block *labels* differ from the figure legend: the swept AUC
  block is written **`Signal (ours)`**, because every measure in it is one of
  ours. The figures keep the plain `Signal`, since their legend is a legend of
  categories rather than of authorship. Note the mark is not a complete census
  of our contributions -- `CKArbfSigma02`, `MutualKNNTop10`, and `CKNNATop10`
  are ours too and sit unmarked in the RSM and Neighbors blocks, which is the
  same point 8.3 makes: they are not a family.
* **One type scale**, in `boxplots.py`: ticks 11pt, axis labels 12pt, panel
  titles 13pt, suptitles 15pt, legends 11pt. Applied to the built artists by
  `scale_text` rather than through Matplotlib's global rcParams, which this
  package shares with the synthetic and glocal experiments' own paper styling.
  Sizes had previously been picked per call site -- 9, 10, 11, 12 and 13 all
  appear in one module -- and left at the 10pt default elsewhere, so the same
  axis label rendered at a different size depending on which figure it landed
  in.
* Per-panel sorting is by median, best on top, direction-aware. `rank_sort:
  quantile90` (or `--rank-sort quantile90`) sorts on the **unfavourable tail**
  instead -- the 90th percentile of a rank, the 10th of a quality value -- which is a robustness statement rather than a typical-case one: a
  measure whose 90th-percentile rank is low never places badly, while one with
  a good median and a long tail is only *usually* good. Under that option the
  quantile is drawn on every box as a black diamond and named in the legend,
  so the number the rows were ordered by is visible rather than asserted.
  Fixing one quantile for both directions would rank the value figures by
  their *best* cases and call it robustness, which is why the tail follows the
  column's direction.

  Two things this ordering is not. It is not domain-balanced in the
  cross-domain figure: `balanced_mean` averages the three domains equally, and
  a quantile has no such analogue, so the quantile ordering pools all 70
  subtasks and graphs' 30 carry more of the tail than language's 20. And with
  14 subtasks in a family box, the 90th percentile sits between the 12th and
  13th ordered point, so it is a coarse statistic there -- it separates
  measures reliably in the pooled panels and should be read cautiously in the
  per-family ones.
* The distribution is drawn as a quartile box with percentile whiskers by
  default. `box_style: boxen` (or `--box-style boxen`) draws it as a
  letter-value plot instead: nested bands at the 30th-70th and 10th-90th
  percentiles, each band further out narrower and paler than the one inside
  it, with the median as a black line and capped whiskers carrying on to the
  observed minimum and maximum. A box states one interval and two whisker
  ends and says nothing about the shape between them; the bands show where the
  mass actually sits, which is what separates two measures that share a median
  and a 90th percentile but differ in whether they fail gradually or
  rarely-and-badly. The taper carries the "this is the tail" reading on its
  own, so the figure survives being read without its legend.

  The whiskers reach the true extremes, not the 5th and 95th percentiles a
  plain box stops at -- those clip the tails and give the reader no way to tell
  from the figure that anything was clipped. The extremes get a whisker rather
  than a third filled band because they are two single observations: a band
  would give them the same visual weight as the 10-90 interval, and the legend
  calls them `min-max` rather than "0-100 pct." for the same reason.

  Band fills are opaque tints toward white, following seaborn's `boxenplot`,
  not alpha. Translucent bands composite with whatever is behind them, so a
  band's rendered colour would depend on the gridline or neighbouring band it
  overlapped and one percentile would not look the same in every row.

  `box_style` is orthogonal to `rank_sort`: it changes the drawing only, never
  the ordering, the filtering, or the numbers. Either style combines with
  either sort. Under `quantile90` the diamond is **not** drawn on a boxen: the
  10th-90th band already ends exactly at the sorted-on quantile, so a marker
  there would restate a number the figure has and read as a separate
  statistic. The invariant that the sorted-on number appears on the figure
  still holds, as a band edge instead of a marker, and a test pins it. A
  second test asserts the two styles produce the same ordering, because a
  rendering choice that silently reordered the axis would be making a claim
  about the data.

  `box_style` also reaches the appendix figures in `tables/` -- the per-test
  value panels and the published rank figure -- so every boxplot in a domain's
  output is drawn the same way. Those panels still take no `rank_sort`: what
  they reproduce is ReSi's published *ranking convention*, and how a
  distribution is drawn is not part of it.

* Medians are black. They are the number a reader takes off the figure, and a
  grey or default-coloured median competes with the box fill it sits on.
* Legends are a frameless, untitled swatch column to the right, anchored
  outside the axes so the saved figure widens to hold it rather than the
  plotting area shrinking -- which is what an in-axes legend does.
* Figure width is fixed and height grows with the measure count, so a figure
  never becomes wider because it has more measures and the figures stack in a
  paper without one looking stretched next to another.
* One deliberate exception: the per-family **value** figures colour by domain,
  because a row there holds one box per domain and the domain split is the
  information those panels exist to carry. They draw through the same
  primitives with the colour channel handed to the group instead.

---

## 7. Every deviation from ReSi's analysis, with reasons

### 7.1 Prediction correlation reports Spearman only **[fairness -- and we had this wrong]**

ReSi computes `pearsonr`, `spearmanr`, `kendalltau` but reports only Spearman:
`tables_and_plots.ipynb` cell 7 and `appendix_tables.ipynb` cell 7 both drop the
other two before any ranking.

Our analysis kept all three until 2026-08-23. The cost was not two extra
columns. A case is keyed on the quality measure, so three correlations produce
three cases where the paper has one, and prediction correlation carried **three
times its published weight** in every case-pooled aggregate. Cross-domain it was
156 of 370 cases against 46-56 for each of the other four families.

Fixed in two places: the three analysis configs now set
`quality_measures: [AUPRC, violation_rate, correlation, spearmanr]`, and
`cross_domain.drop_unpublished_quality` applies the same filter in the
cross-domain layer (which reads per-domain artifacts from disk that predate the
narrowing). Cross-domain case totals moved 370 -> 266, families to 46-56 each.
The 2026-08-26 conventions (6.3) took this further by the same argument, first
dropping `violation_rate` from ranking and then `Disagreement`: 149 cases now,
23-28 per family and 16 for `Acc Corr.`

### 7.2 Graphs exclude `AbsoluteAccDiff` correlations **[missing]**

ReSi's graph archive has no `AbsoluteAccDiff` correlations for its own measures
-- the rows exist, the values are blank -- while our recreated baselines and all
manifold measures do score it. That asymmetry creates 30 cases only the new
measures can cover; the 27 the natives cannot cover drag all nine natives to
141/168 = 0.839, just under `min_case_coverage`, and the graph ranking came out
containing *only manifold measures*. That reads as the natives losing when it is
actually a gap in the archive.

`exclude_functional_measures: [AbsoluteAccDiff]` in the graph analysis config.
With it the natives cover 138/138. Cross-domain, the same asymmetry is handled
by `AbsoluteAccDiff` forming its own `Acc Corr.` family, which graphs simply
does not populate -- rather than by deleting the other two domains' accuracy
rows to keep one merged correlation family comparable.

`AbsoluteAccDiff` is missing twice over, in two different ways, and the two
should not be conflated. Here in graphs it is an archive gap we cannot close:
the values were never computed for ReSi's own measures and the models are not
ours to re-evaluate. In language it is missing only for SmolLM2, because the
accuracies were never written to `eval_results.json` (2.1a) -- fixable in
principle, and low-value in practice because this graph gap keeps `Acc Corr.`
a two-domain family either way.

### 7.3 Sixteen measures reported, not twenty-four **[missing + no principled reason]**

Reported natives: `AlignedCosineSimilarity, CKA, DistanceCorrelation,
JaccardSimilarity, LinearRegression, ProcrustesSizeAndShapeDistance,
RankSimilarity, SVCCA, SecondOrderCosineSimilarity` (9) plus our 7 (§3.3).
Graphs and language report all 16; vision reports 15, because
`LinearRegression` falls below the coverage threshold there (§6.6) and is the
only measure any domain drops on coverage.

* `PWCCA` -- absent from the graph archive entirely, ~55% NaN in language.
  **[missing]**
* `IMDScore`, `RSMNormDifference` -- ~67% of total baseline runtime in every
  domain. Excluded from the recreated baselines to make CIFAR100 / SmolLM2 /
  PGNN affordable at all. **[no principled reason -- cost]**
* The remaining ~12 ReSi measures are archived for the archived architectures
  but were not recreated for the new ones, so including them would reintroduce
  exactly the field-size asymmetry the `measures` list exists to remove.
  **[no principled reason -- cost, cascading from the above]**

This is the largest honest caveat in the paper. We are not claiming to beat
ReSi's full suite; we are claiming to beat a 9-measure subset chosen for cost
and coverage, on a field of 16. The subset does contain the measures ReSi's own
tables rank near the top (`Jaccard`, `2nd-Cos`, `AlignCos`, `CKA`,
`DistCorr`), which is the mitigating fact worth stating.

Note the asymmetry in *how* the two sides were narrowed. The native field was
cut by what the archive has and what the recreated baselines could afford; our
own field was cut by us, after seeing results. Dropping three of our ten while
keeping all nine of theirs is defensible as a presentation choice -- the three
are near-duplicates of measures that remain -- but it is a choice made with
knowledge of the outcome, and a reader is entitled to know that. The full
ten-measure ranking is one config edit and one re-run away, and the parquets
are unchanged.

### 7.4 Matched-observation ranking **[withdrawn -- was a deviation]**

Ranking used to require every compared measure to be finite in an observation
before that observation was ranked at all, on the grounds that ReSi's
`na_option="keep"` rewards failure. The reasoning was one-sided: the row-wise
drop made one measure's failure cost every other measure the observation too,
which distorts more than it protects, and the asymmetry it was aimed at is what
coverage filtering (6.6) actually handles. Ranking now follows ReSi -- rank
what is present, skip what is not. See 6.4.

### 7.5 Normalized ranks **[withdrawn as the reported scale]**

Ranks were reported on a $[0,1]$ rescaling on the grounds that raw ranks are
not comparable across differently-sized fields. True across domains, and the
cross-domain layer still normalizes for exactly that reason (6.8). Within a
domain it does not apply: the `measures` list and the coverage thresholds fix
one field for every case, so the rescaling was a monotone relabelling of an
already-readable number. Per-domain outputs now report raw ranks, matching
ReSi. Both statistics are still written on every run.

### 7.6 Coverage thresholds **[fairness]** -- see 6.6.

### 7.7 Cross-domain layer **[new]** -- see 6.8. Nothing in ReSi corresponds to
this; it is an addition, not a deviation, but it introduces the domain-weighting
question that 6.8 answers with two weightings rather than one.

### 7.8 The main-table functional-measure restriction **[fixed -- was a deviation]**

ReSi's *main* table keeps only `JSD` correlations and splits `AbsoluteAccDiff`
off as a separate "Acc Corr." setting, discarding `Disagreement`. Its *appendix*
keeps all three. We used to follow the appendix everywhere, which was a choice
made without an argument.

Ranking now follows the main table: `Output Corr.` is ranked on `JSD` and
`Acc Corr.` on `AbsoluteAccDiff`, one functional similarity measure per task
(§6.3). `Disagreement` is still computed and still reported -- it keeps its
`Eval.` block in the appendix `Output Corr.` table and its panel in that test's
value figure -- it just does not earn a second case. Without this, output
correlation contributed two cases per cell where every other test contributed
one, which is the same double-counting the quality filter removes for
`violation_rate`.

These remain **six tasks, not five**. `Acc Corr.` is not merged into
`Output Corr.`; it asks a different question and is reported over whatever
domains and architectures have the values -- vision throughout, language's
BERT-L but not SmolLM2 (2.1a), graphs not at all (7.2). Verified before
adopting it: every `AbsoluteAccDiff` cell that exists in vision and language is
finite for every reported measure, so including the task disadvantages no
native measure on coverage. Graphs is the one domain where nothing is
available, which is why its config excludes the setting outright rather than
ranking a column of NaN.

The published-mode figure does **not** apply this filter, because ReSi's own
rank figure does not -- it ranks all three functional measures as separate
settings, while its overview table keeps one. That is now one of the listed
differences between the two conventions in §6.5.

### 7.9 Correlation results are labelled by functional measure **[fixed -- was a defect]**

The prediction-correlation benchmark scores one set of representations against
three functional similarity measures at once, and ReSi splits the resulting
tables on *that* measure. We keyed the split on the **benchmark id** instead:
`accoutput` (vision) -> "Acc Corr.", `output_correlation_test` (graphs) and
`correlation` (language) -> "JSD Corr.". Each domain's benchmark has exactly
one id, so every functional measure it carried inherited that one label.

The effect while it stood: vision's `published_case_ranks.csv` labelled all
three functional measures "Acc Corr." (798 rows each), and vision wrote an
`acc-corr` table where graphs wrote a `jsd-corr` one -- neither of which held
only what its name claimed. Disagreement correlations were reported in every
figure and every ranking, but they never had a table of their own. The
*ranking* was never affected: `rank_measures` carries the functional measure in
both its rank keys and its dedupe key, so the three were always ranked apart.

Now `TEST_LABELS` maps every correlation benchmark to the neutral family name
`Prediction Corr.`, and `FUNCTIONAL_TEST_LABELS` refines it per row into the
**two tasks** the benchmark actually answers:

| functional similarity measure | task | `Eval.` block |
| --- | --- | --- |
| `JSD` | `Output Corr.` | JSD |
| `Disagreement` | `Output Corr.` | Disagreement |
| `AbsoluteAccDiff` | `Acc Corr.` | Acc Diff |

A correlation row whose functional measure is blank or unrecognised keeps
`Prediction Corr.`, so an unlabelled row is visibly unlabelled rather than filed
under a measure it was not scored against.

**Two tasks, not three settings.** JSD and Disagreement both ask how far two
models' *outputs* agree and differ only in how that disagreement is quantified,
so they are two metrics of one task -- exactly what AUPRC and conformity rate
are for each design test. `AbsoluteAccDiff` asks how far apart the two models'
*accuracies* are, which is a different question: two models can agree on almost
nothing and still score identically. Grouping it with the output measures
pooled answers to two questions under one heading.

**Grouped, not pooled.** `Output Corr.` is one table and one figure covering
both its measures, because the question worth asking is how a measure does on
one *relative to* the other, and two files put that across a page turn. What
makes that safe is that each measure is a named `Eval.` block -- the same level
that separates AUPRC from conformity rate elsewhere -- so the columns never
pool two functional measures under one heading. `Acc Corr.` is a separate file
for the same reason `Shortcuts` and `Augmentation` are separate files.

This also retired a structure unique to one family: correlation tables no
longer need a `Test` column level of their own, because `Eval.` now carries
what that level was carrying. Every task uses the same
`APPENDIX_COLUMN_LEVELS`. One consequence worth naming: the old unified
correlation figure shared one measure ordering across its panels so a measure
kept its row between them, and the uniform per-task figure sorts each panel
independently, as every design test's figure already did.

#### Which correlation task each domain has

`Acc Corr.` is **absent from one of the three domains and partial in another**,
for two unrelated reasons that 7.2 and 2.1a cover in full:

| domain | Output Corr. (JSD, Disagreement) | Acc Corr. (AbsoluteAccDiff) |
| --- | --- | --- |
| graphs | yes, both | **not reported** -- `exclude_functional_measures: [AbsoluteAccDiff]`, because the archive carries no values for ReSi's own measures (7.2) |
| vision | yes, both | yes, all architectures |
| language | yes, both | yes, but on BERT-L only -- SmolLM2 has no accuracy in `eval_results.json` (2.1a) |

A domain that writes no table cannot be told apart from one nobody generated.
`tables_index.csv` therefore carries a zero-row entry for an absent task with
the note `not reported: no <measures> correlations in this domain's results`,
so the gap is recorded rather than inferred. The entry is per task rather than
per functional measure -- a domain missing `Output Corr.` is charged one row
naming both measures it would have covered, not two rows pointing at the same
missing file -- and is only emitted for a domain that runs a
prediction-correlation benchmark at all.

The cross-domain comparison follows the same split rather than working around
it. `Acc Corr.` is its own family built from the two domains that report it,
and the figure index records `only 2 of 3 domains present`; `Output Corr.`
stays balanced across all three. That replaced an intersection down to
`{JSD, Disagreement}` which deleted vision's and language's accuracy rows
entirely to keep one merged correlation family comparable (6.8).

### 7.10 "AUC" is a mean, not an integral **[no principled reason -- naming]** -- see 3.2.

### 7.11 Cross-domain ranks are averaged into subtasks before pooling **[fairness]**

ReSi has no cross-domain layer, so this is an addition rather than a deviation
-- but it is a deliberate departure from the obvious thing to do, which is to
pool cases.

Pooling cases weights a domain by how many architectures it ships. Vision runs
7, graphs 3-4, language 1-2, so vision held 180 of 266 cross-domain cases and a
"cross-domain" box was closer to a vision box. Averaging within a subtask first
(§6.2b, §6.8) gives each (domain, dataset, task, evaluation) slice one vote,
which balances the five families exactly (14 units each) and brings the two
stage-two weightings into near-agreement.

The cost is stated in §6.8 and is real: the boxes are narrower because
within-subtask variation is averaged out, and each box holds 7 points (4 for
`Acc Corr.`) rather than 23-28. Per-domain figures keep the case-level view,
where the spread across
architectures is the information the reader wants.

---

## 8. Current results

All three per-domain analyses were re-run on 2026-08-26 under the current
ranking convention (6.3-6.6): ReSi's NaN handling, raw ranks, one quality
measure and one functional similarity measure per task. Everything below is
current.

Ranked cases are now one per (task, dataset, architecture) -- graphs 49, vision
80, language 20, roughly half the previous counts, from `violation_rate` and
`Disagreement` leaving the ranking.

Reported field: **16 measures** in graphs and language; **15** in vision, where
one measure falls below the coverage threshold (6.6):

| Domain | excluded at domain scope | coverage | ranked cases |
|---|---|---|---|
| Graphs | -- | -- | 49 |
| Vision | `LinearRegression` | 64/80 = 0.800 | 80 |
| Language | -- | -- | 20 |

`LinearRegression` is the only measure any domain drops on coverage, and its
fraction barely moved under the new convention (0.795 to 0.800) -- the cases it
fails and the cases it survives were removed in near-equal proportion, which is
what you would expect if its failures are not concentrated in one test.

Language is the reason `min_case_coverage` is 0.90 rather than 0.95, and the
case is worth reading because nothing about `2nd-Cos` changed. It is missing
from the same cases it was always missing from; what changed is the
denominator. A fixed-fraction threshold is sensitive to how many cases are in
scope, so shrinking the reported scope can evict a measure that got no worse.
See 6.6 for what 0.90 separates.

### 8.1 Per-domain, raw mean rank (1 = best), top 3

| Domain | 1st | 2nd | 3rd | cases | field |
|---|---|---|---|---|---|
| Graphs | `UKArbfAUC` 5.71 | `CKArbfAUC` 5.92 | `SecondOrderCosineSimilarity` 6.11 | 49 | 16 |
| Vision | `CKArbfAUC` 6.45 | `JaccardSimilarity` 6.56 | `DistanceCorrelation` 6.89 | 80 | 15 |
| Language | `CKArbfAUC` 6.13 | `dRWKArbfAUC` 6.16 | `AlignedCosineSimilarity` 6.20 | 20 | 16 |

The reported figures sort on the 90th-percentile rank rather than the mean
(`rank_sort: quantile90`, 6.9), and on that ordering **`CKArbfAUC` is first in
all three domains**: graphs 9.2, vision 11.0, language 10.1. It is also first
or second on the mean everywhere. That is a stronger statement than the
previous convention supported, where the three domains had three different
leaders -- and it is worth being explicit that the change which produced it was
adopted for reasons argued before these numbers were seen (7.4, 7.5, 7.8).

Language's top three are separated by 0.07 of a rank over 20 cases, which is
not a real ordering; read it as a four-way tie at the top including
`DistanceCorrelation`.

### 8.2 Cross-domain (39 subtasks over 149 cases)

| Measure | pooled | balanced | $n_{\text{domains}}$ |
|---|---|---|---|
| `CKArbfAUC` | 0.352 | **0.355** | 3 |
| `DistanceCorrelation` | 0.414 | 0.411 | 3 |
| `JaccardSimilarity` | 0.424 | 0.426 | 3 |
| `SecondOrderCosineSimilarity` | 0.426 | 0.433 | 3 |
| `dRWKArbfAUC` | 0.451 | 0.443 | 3 |
| `UKArbfAUC` | 0.439 | 0.449 | 3 |
| `CKA` | 0.455 | 0.450 | 3 |
| `AlignedCosineSimilarity` | 0.470 | 0.462 | 3 |
| `RankSimilarity` | 0.474 | 0.478 | 3 |
| `ProcrustesSizeAndShapeDistance` | 0.506 | 0.505 | 3 |
| `CKArbfSigma02` | 0.512 | 0.522 | 3 |
| `sRWKArbfAUC` | 0.562 | 0.564 | 3 |
| `MutualKNNTop10` | 0.601 | 0.600 | 3 |
| `CKNNATop10` | 0.640 | 0.633 | 3 |
| `LinearRegression` | 0.638 | 0.645 | **2** |
| `SVCCA` | 0.677 | 0.672 | 3 |

Pooled and balanced agree to within 0.003 for the leader, which is what the
subtask design is for: with a near-balanced unit count, the choice of weighting
stops carrying the conclusion. `CKArbfAUC` leads by 0.056 over the next
measure, the largest gap anywhere in the table.

Per family (balanced, winner and runner-up):

| Family | 1st | 2nd |
|---|---|---|
| Output Corr. | `CKArbfAUC` 0.314 | `DistanceCorrelation` 0.328 |
| Acc Corr. | `ProcrustesSizeAndShapeDistance` 0.395 | `CKArbfAUC` 0.396 |
| Random Labels | `DistanceCorrelation` 0.414 | `MutualKNNTop10` 0.425 |
| Shortcuts | `SecondOrderCosineSimilarity` 0.305 | `CKArbfAUC` 0.321 |
| Augmentation | `DistanceCorrelation` 0.326 | `CKArbfAUC` 0.355 |
| Layer Mono. | `CKArbfAUC` 0.287 | `SecondOrderCosineSimilarity` 0.334 |

`CKArbfAUC` takes `Output Corr.` and `Layer Mono.` and is runner-up in three of
the remaining four; `Acc Corr.` is a 0.001 gap and should not be reported as a
win for either measure. `DistanceCorrelation` takes two.
`SecondOrderCosineSimilarity` takes `Shortcuts`, this time by 0.016 rather than
the 0.001 the previous convention gave it.

Subtask and case counts per family. Every family now holds 7 subtasks -- 3
graphs, 2 vision, 2 language -- except `Acc Corr.`, the one family graphs
cannot populate (7.2). Output correlation held 14 under the previous
convention, because JSD and Disagreement each earned their own subtasks:

| Family | graphs | vision | language | subtasks | cases |
|---|---|---|---|---|---|
| Output Corr. | 3 | 2 | 2 | 7 | 26 |
| Acc Corr. | 0 | 2 | 2 | 4 | 16 |
| Layer Mono. | 3 | 2 | 2 | 7 | 28 |
| Random Labels | 3 | 2 | 2 | 7 | 28 |
| Shortcuts | 3 | 2 | 2 | 7 | 28 |
| Augmentation | 3 | 2 | 2 | 7 | 23 |
| **total** | **15** | **12** | **12** | **39** | **149** |

### 8.3 Three findings that belong in the paper as caveats

* **Saturation.** In vision Shortcuts, most measures have median AUPRC
  $\ge 0.99$. Ranks manufacture a strict ordering where the values are
  effectively tied. Any rank-based headline on that benchmark is over-reading;
  this is why the value figures exist alongside the rank figures.
* **Our measures are not a family.** They spread across three of ReSi's
  categories and their outcomes spread with them: `CKArbfAUC` first of 16
  cross-domain, `CKNNATop10` second-to-last. The claim is about *specific*
  manifold measures -- RBF-kernel CKA and its bandwidth-swept variants -- not
  about manifold geometry as an approach. This is also why no figure gives them
  a shared colour.
* **One native is missing from a domain**, so `LinearRegression`'s balanced
  mean is a two-domain average competing against three-domain ones.
  `n_domains` is reported beside every number for that reason.

## 9. Artifacts a reviewer would want

Per domain, under `figures/resi/<run>/<domain>/`:

| File | Contents |
|---|---|
| `config.yaml` | resolved analysis settings actually used |
| `normalized_values.csv` | the full long frame after measure selection and coverage filtering |
| `per_case_ranks.csv` | one row per (case, measure): raw and normalized mean rank, how many of the case's observations that measure was ranked in against how many it has, field size |
| `domain_summaries.csv` | pooled per-measure statistics |
| `measure_exclusions.csv` | every dropped (measure, case) with reason and the exact fraction |
| `coverage_nan_report.csv` | per-manifest-entry state, row counts, finite/NaN counts, known-missing declarations |
| `signals_long.csv` | the full 100-point similarity curves, one row per sweep point |
| `tables/` | recreated appendix tables (LaTeX + CSV) under the published convention, one per test -- the prediction-correlation settings share `prediction-corr`, with the functional similarity measure as a column level (7.9) |
| `tables/tables_index.csv` | what was written per test, plus a zero-row entry naming any correlation setting this domain reports no values for |

Cross-domain, under `figures/resi/cross_domain/`:

| File | Contents |
|---|---|
| `cross_domain_values.csv` | the stacked value frame, after both harmonizations |
| `cross_domain_summary.csv` | median/mean/std per (family, quality measure, domain, measure) |
| `cross_domain_subtask_ranks.csv` | **one row per (subtask, measure)** -- every point the rank boxplots draw, with the `n_cases` behind it |
| `cross_domain_ranks.csv` | the stage-two aggregate: `n_subtasks`, `n_cases`, `n_domains`, pooled and balanced means |
| `<family>.png` | value panels, one box per domain per measure |
| `<family>_ranks_by_type.png`, `<family>_values_by_type.png` | category-coloured, per-panel sorted |
| `cross_domain_ranks.png` | six panels sharing one measure ordering |
| `<family>_summary.tex` / `.csv` | per-family LaTeX tables, rows blocked by the same measure taxonomy the figures colour by |

The `.tex` files render with `escape=False`, because the cells carry
`\textbf` and significance markers that have to reach LaTeX intact. That
leaves row and column labels unescaped too, so `latex_safe_labels` replaces
underscores with spaces before rendering -- `ViT_B32` would otherwise open math
mode and fail to compile. Only the rendered table is touched; the CSV beside it
keeps the raw names.

Output schemas are pinned by `tests/test_resi_analysis_schemas.py`, and the
rendered LaTeX by a golden fixture in `tests/test_resi_tables.py`, so a schema
change has to be deliberate.
