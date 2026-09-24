#!/usr/bin/env bash
#SBATCH --job-name=resi-prepare-native
#SBATCH --output=logs/resi-prepare-native-%j.out
#SBATCH --error=logs/resi-prepare-native-%j.err
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#
# Native (non-container) counterpart to resi_prepare_container.sbatch, for
# clusters without Apptainer/Pyxis container support. Runs directly inside a
# conda environment instead of an NGC image. The partition/QOS below are a
# preemptible pool; set them for your cluster (see docs/resi_cluster.md).
# Jobs there can be preempted by
# higher-priority partitions and requeued, so this re-sources the conda
# environment fresh on every run rather than relying on inherited shell
# state that a requeue onto a different node could lose. --requeue is
# defensive (many clusters already auto-requeue preempted jobs);
# `prepare` is deterministic and re-running it from scratch is cheap and
# safe, so no custom checkpoint/resume logic is needed here either -- see
# "Preemption and resumability" in docs/resi_cluster.md.
#
# Set up the environment once first with scripts/setup_resi_native_env.sh.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"
RESI_DIR="${RESI_DIR:-$WORK_ROOT/resi}"
REP_SIM="${REP_SIM:-$RESI_DIR/experiments}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
CONDA_ENV="${CONDA_ENV:-resi-manifold}"

: "${RESI_CAMPAIGN:?Set RESI_CAMPAIGN to a campaign YAML path}"
: "${RESI_CAMPAIGN_RUN_NAME:?Set RESI_CAMPAIGN_RUN_NAME}"

if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "Miniforge/conda not found at: $CONDA_BASE (set CONDA_BASE)" >&2
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
if [[ ! -f "$RESI_CAMPAIGN" ]]; then
  echo "Campaign config does not exist: $RESI_CAMPAIGN" >&2
  exit 1
fi

source ~/.bashrc
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$RESI_DIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
export REPO_ROOT WORK_ROOT RESI_DIR REP_SIM RESI_CAMPAIGN RESI_CAMPAIGN_RUN_NAME

echo "REPO_ROOT=$REPO_ROOT"
echo "WORK_ROOT=$WORK_ROOT"
echo "RESI_DIR=$RESI_DIR"
echo "REP_SIM=$REP_SIM"
echo "CONDA_ENV=$CONDA_ENV (python: $(command -v python))"

cd "$REPO_ROOT"
python scripts/resi.py prepare \
  --campaign "$RESI_CAMPAIGN" \
  --run-name "$RESI_CAMPAIGN_RUN_NAME"
