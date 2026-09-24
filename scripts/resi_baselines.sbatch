#!/usr/bin/env bash
#SBATCH --job-name=resi-baselines
#SBATCH --output=logs/resi-baselines-%A_%a.out
#SBATCH --error=logs/resi-baselines-%A_%a.err
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --array=0-0
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#
# Runs one ReSi baseline config per array task, to recreate the ReSi measures
# that the archive is missing (PGNN, SmolLM2, CIFAR100).
#
# This calls ReSi's own runner directly -- `python -m repsim.run -c <config>`
# -- rather than scripts/resi.py. Our campaign tooling deliberately cannot
# compute ReSi's measures (campaign/schema.py validates against the local
# adapter catalogue), and recreating that would duplicate ReSi.
#
# Generate the configs and the jobs list first:
#   python scripts/prepare_resi_baselines.py --run-name <name> [--target pgnn]
#
# Then submit, pointing JOBS at the generated list:
#   export JOBS=data/resi/baselines/<name>/jobs.txt
#   sbatch --array=0-$(($(wc -l < "$JOBS") - 1))%8 scripts/resi_baselines.sbatch
#
# Preemption: ReSi's ExperimentStorer writes after every completed comparison
# and skips already-computed ones, so a requeued task resumes on its own. See
# "Preemption and resumability" in docs/resi_cluster.md.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"
RESI_DIR="${RESI_DIR:-$WORK_ROOT/resi}"
REP_SIM="${REP_SIM:-$RESI_DIR/experiments}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
CONDA_ENV="${CONDA_ENV:-resi-manifold}"

: "${JOBS:?Set JOBS to the jobs.txt produced by prepare_resi_baselines.py}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a zero-based SLURM array}"

for path in "$REPO_ROOT" "$RESI_DIR" "$REP_SIM"; do
  [[ -d "$path" ]] || { echo "Does not exist: $path" >&2; exit 1; }
done
[[ -f "$JOBS" ]] || { echo "Jobs list does not exist: $JOBS" >&2; exit 1; }

CONFIG="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS")"
[[ -n "$CONFIG" && -f "$CONFIG" ]] || {
  echo "No config at index $SLURM_ARRAY_TASK_ID in $JOBS" >&2; exit 1; }

RESI_CPU_THREADS="${RESI_CPU_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$RESI_CPU_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$RESI_CPU_THREADS}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$RESI_DIR:${PYTHONPATH:-}"
export REP_SIM RESI_DIR

echo "REPO_ROOT=$REPO_ROOT"
echo "RESI_DIR=$RESI_DIR"
echo "REP_SIM=$REP_SIM"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CONFIG=$CONFIG"

# Re-source the environment inside every job: a preempted task can be requeued
# onto a different node, which loses anything inherited from the submitting
# shell.
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
echo "python: $(command -v python)"

cd "$RESI_DIR"
python -m repsim.run --config "$CONFIG"
echo "completed index=$SLURM_ARRAY_TASK_ID config=$CONFIG"
