# repsim-signal

Code accompanying the paper.

This repository contains the similarity-signal representation-similarity
metrics, the synthetic manifold experiments, the glocal sweep experiment,
and the integration with the ReSi benchmark used in the paper.

## Main components

The installable package lives under `src/manifold_repsim/`:

- `metrics/`: the metric registry and the kernel, nearest-neighbor, and
  random-walk implementations.
- `sweeps/`: parameter grids, prepared score-curve (similarity signal)
  execution, permutation calibration, and signal aggregation (e.g. AUC).
- `experiments/`: shared computation, MDS fitting, artifacts, and paper
  figure styling.
- `experiments/permuted_gaussian_mds/`: the synthetic permuted-Gaussian MDS
  experiment.
- `experiments/glocal_sweep_mds/`: the glocal sweep MDS/PCA experiment.
- `resi/`: ReSi measure registration, campaign preparation/execution/status,
  and analysis (rankings, signals, tables, and figures).

`synthetic_datasets/` holds the synthetic dataset families and transforms.
`configs/` holds the YAML configuration of every experiment, and `scripts/`
the command-line entry points.

## Environment

Python 3.10.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -c constraints.txt
.venv/bin/python -m pip install -e . --no-deps --no-build-isolation
```

## Running experiments

See [docs/running_experiments.md](docs/running_experiments.md) for every
experiment. The main entry points are:

```bash
# Synthetic permuted-Gaussian MDS
.venv/bin/python scripts/synthetic_sweep.py configs/synthetic_permuted_gaussian_mds.yaml
.venv/bin/python scripts/paper_figures_synthetic.py --help

# Glocal sweep MDS/PCA
.venv/bin/python scripts/glocal_sweep_analysis.py configs/glocal_sweep_mds.yaml
.venv/bin/python scripts/paper_figures_glocal.py --help

# ReSi benchmark campaigns
.venv/bin/python scripts/resi.py --help
```

ReSi campaigns need a separate ReSi checkout and a GPU cluster; see
[docs/resi_cluster.md](docs/resi_cluster.md). The analysis methodology is
described in [docs/resi_analysis_methods.md](docs/resi_analysis_methods.md).

`synthetic_demo.ipynb` is a short, self-contained demo of the synthetic
experiment: the Gaussian-mixture base data, its noise, mixing, and
permutation transformations, and the similarity signals and their AUCs, in
the paper's figure style. It runs in about a minute on a CPU.

## Tests

```bash
MPLCONFIGDIR=/tmp/repsim-signal-mpl \
  .venv/bin/python -m unittest tests.repository_suite -v
```

Generated datasets, figures, manifests, and results are written under
`data/` and `figures/`, which are gitignored.

## License

See [LICENSE](LICENSE).
