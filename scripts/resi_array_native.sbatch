#!/usr/bin/env bash
#SBATCH --job-name=resi-manifold-native
#SBATCH --output=logs/resi-manifold-native-%A_%a.out
#SBATCH --error=logs/resi-manifold-native-%A_%a.err
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --array=0-7
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#
# Native (non-container) counterpart to resi_array_container.sbatch, for
# clusters without Apptainer/Pyxis container support. See
# resi_prepare_native.sbatch for the
# part-preempt/qos-preempt rationale and the fresh-activation-per-job
# reasoning (preemption/requeue can move a job to a different node).
#
# No custom checkpoint/resume logic: `python scripts/resi.py run-task` calls
# into ReSi's own ExperimentStorer, which saves results to disk (a plain
# to_parquet overwrite, not atomic) after every completed representation-pair
# comparison and skips already-computed ones on the next run. A
# preempted-and-requeued task therefore resumes on its own when SLURM reruns
# this script, in the common case -- see "Preemption and resumability" in
# docs/resi_cluster.md, including the narrow non-atomic-write failure mode and
# its recovery.
#
# Set up the environment once first with scripts/setup_resi_native_env.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"
RESI_DIR="${RESI_DIR:-$WORK_ROOT/resi}"
REP_SIM="${REP_SIM:-$RESI_DIR/experiments}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
CONDA_ENV="${CONDA_ENV:-resi-manifold}"

: "${MANIFEST:?Set MANIFEST to a generated compute_manifest.jsonl}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a zero-based SLURM array}"

if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "Miniforge/conda not found at: $CONDA_BASE (set CONDA_BASE)" >&2
  exit 1
fi
if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Repository does not exist: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -d "$RESI_DIR" ]]; then
  echo "ReSi checkout does not exist: $RESI_DIR" >&2
  exit 1
fi
if [[ ! -d "$REP_SIM" ]]; then
  echo "ReSi experiments directory does not exist: $REP_SIM" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest does not exist: $MANIFEST" >&2
  exit 1
fi

source ~/.bashrc
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

RESI_CPU_THREADS="${RESI_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$RESI_CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$RESI_CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$RESI_CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$RESI_CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$RESI_CPU_THREADS}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$RESI_DIR:${PYTHONPATH:-}"
export MANIFOLD_RESI_MAX_POINTS="${MANIFOLD_RESI_MAX_POINTS:-}"
export MANIFOLD_RESI_AUC_MAX_POINTS="${MANIFOLD_RESI_AUC_MAX_POINTS:-}"
export MANIFOLD_RESI_SUBSAMPLE_SEED="${MANIFOLD_RESI_SUBSAMPLE_SEED:-0}"
# Empty keeps representations wherever ReSi produced them. Set to "cuda" to run
# kernel construction on the allocated GPU, which dominates the AUC measures.
export MANIFOLD_RESI_DEVICE="${MANIFOLD_RESI_DEVICE:-}"
export REPO_ROOT WORK_ROOT RESI_DIR REP_SIM MANIFEST

echo "REPO_ROOT=$REPO_ROOT"
echo "WORK_ROOT=$WORK_ROOT"
echo "RESI_DIR=$RESI_DIR"
echo "REP_SIM=$REP_SIM"
echo "MANIFEST=$MANIFEST"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CONDA_ENV=$CONDA_ENV (python: $(command -v python))"
echo "MANIFOLD_RESI_MAX_POINTS=${MANIFOLD_RESI_MAX_POINTS:-disabled}"
echo "MANIFOLD_RESI_AUC_MAX_POINTS=${MANIFOLD_RESI_AUC_MAX_POINTS:-inherits general limit}"
echo "MANIFOLD_RESI_SUBSAMPLE_SEED=$MANIFOLD_RESI_SUBSAMPLE_SEED"
echo "MANIFOLD_RESI_DEVICE=${MANIFOLD_RESI_DEVICE:-unset (representations stay put)}"

cd "$REPO_ROOT"
python scripts/resi.py run-task \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID"
