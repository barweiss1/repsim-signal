#!/usr/bin/env bash
#SBATCH --job-name=resi-train-pgnn-cora
#SBATCH --output=logs/resi-train-pgnn-cora-%j.out
#SBATCH --error=logs/resi-train-pgnn-cora-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --qos=normal
#
# Trains ReSi's P-GNN architecture on Cora, the only dataset it's registered
# for (see repsim/benchmark/registry.py:all_trained_graph_models() and the
# ReSi paper, Appendix B.2.1: P-GNN's memory footprint made larger graphs
# like Flickr/ogbn-arxiv infeasible). Covers every (test, seed) combination
# the registry expects EXCEPT augmentation_test, which the paper explicitly
# excludes for P-GNN (DropEdge perturbs the anchor distances P-GNN's message
# passing depends on).
#
# Checkpoints land under $REP_SIM/models/graphs/cora/<setting>/, alongside
# the already-downloaded GCN/GraphSAGE/GAT checkpoints -- no path wrangling
# needed afterward.
#
# Usage (from the repository root):
#   mkdir -p logs
#   sbatch scripts/resi_train_pgnn_cora.sbatch

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"
RESI_DIR="${RESI_DIR:-$WORK_ROOT/resi}"
REP_SIM="${REP_SIM:-$RESI_DIR/experiments}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/containers/resi-manifold-pytorch-24.06-v2.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$WORK_ROOT:$WORK_ROOT}"

if [[ ! -f "$CONTAINER_IMAGE" ]]; then
  echo "Configured container image does not exist: $CONTAINER_IMAGE" >&2
  exit 1
fi
if [[ ! -d "$RESI_DIR" ]]; then
  echo "ReSi checkout not found: $RESI_DIR" >&2
  exit 1
fi

export REPO_ROOT WORK_ROOT RESI_DIR REP_SIM
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$RESI_DIR:${PYTHONPATH:-}"
    export MPLCONFIGDIR
    export REP_SIM
    cd "$RESI_DIR"
    TRAINER="$REPO_ROOT/scripts/run_graph_trainer.py"  # wraps graphs.graph_trainer with the
                                                          # torcheval fallback (see docs/resi_cluster.md)

    echo "=== Standard setting (seeds 0-9) ==="
    python "$TRAINER" -a PGNN -d cora -s 0 1 2 3 4 5 6 7 8 9

    echo "=== Label-noise test (seeds 0-4) ==="
    python "$TRAINER" -a PGNN -d cora -t label_test -s 0 1 2 3 4

    echo "=== Shortcut test (seeds 0-4) ==="
    python "$TRAINER" -a PGNN -d cora -t shortcut_test -s 0 1 2 3 4

    echo "=== Layer/monotonicity test (seeds 0-4) ==="
    python "$TRAINER" -a PGNN -d cora -t layer_test -s 0 1 2 3 4

    echo "=== Done. Checkpoint count under \$REP_SIM/models/graphs/cora: ==="
    find "$REP_SIM/models/graphs/cora" -iname "*.pt" 2>/dev/null | wc -l
  '
