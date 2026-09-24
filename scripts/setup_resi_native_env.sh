#!/usr/bin/env bash
# One-time conda environment bootstrap for running ReSi campaigns natively
# on a SLURM cluster without container support. Run this once from a login
# node, not via SLURM -- it is just package installation, no GPU compute
# needed (see docs/resi_cluster.md).
#
# Assumes:
#   - miniforge3/conda is already installed.
#     Default location: $HOME/miniforge3 -- override with CONDA_BASE if yours
#     is elsewhere.
#   - this repository and a ReSi checkout are already cloned, as siblings
#     under one WORK_ROOT, e.g. ~/research/repsim-signal and ~/research/resi.
#
# Usage (from the manifold_repsim repo root):
#   bash scripts/setup_resi_native_env.sh
# Override the env name or conda location if needed:
#   CONDA_ENV=my-resi-env CONDA_BASE=/path/to/miniforge3 \
#     bash scripts/setup_resi_native_env.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"
RESI_DIR="${RESI_DIR:-$WORK_ROOT/resi}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
CONDA_ENV="${CONDA_ENV:-resi-manifold}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "Miniforge/conda not found at: $CONDA_BASE" >&2
  echo "Set CONDA_BASE to the directory containing etc/profile.d/conda.sh." >&2
  exit 1
fi
if [[ ! -d "$RESI_DIR" ]]; then
  echo "ReSi checkout not found: $RESI_DIR" >&2
  echo "Set RESI_DIR, or clone/rsync the ReSi checkout there first." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "Conda env '$CONDA_ENV' already exists; reusing it."
else
  echo "Creating conda env '$CONDA_ENV' (python $PYTHON_VERSION)."
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi
conda activate "$CONDA_ENV"
echo "Active python: $(command -v python) ($(python --version))"

python -m pip install --upgrade pip setuptools wheel

echo "Installing manifold_repsim's ReSi runtime requirements (pinned via constraints-resi-native.txt)."
python -m pip install --no-cache-dir \
  -c "$REPO_ROOT/constraints-resi-native.txt" \
  -r "$REPO_ROOT/requirements-resi-native.txt"

echo "Editable-installing manifold_repsim itself."
python -m pip install -e "$REPO_ROOT" --no-deps --no-build-isolation

echo "Editable-installing the ReSi checkout ($RESI_DIR)."
python -m pip install -e "$RESI_DIR"

echo "Running the smoke test."
python "$REPO_ROOT/scripts/smoke_resi_container.py" --resi-dir "$RESI_DIR"

echo "Done. Activate this environment in future sessions with:"
echo "  source $CONDA_BASE/etc/profile.d/conda.sh && conda activate $CONDA_ENV"
