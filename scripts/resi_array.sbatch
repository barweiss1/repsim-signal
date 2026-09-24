#!/bin/bash
#SBATCH --job-name=resi-manifold
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/resi-%A_%a.out

set -euo pipefail

: "${MANIFEST:?Set MANIFEST to a generated compute_manifest.jsonl}"
: "${RESI_DIR:?Set RESI_DIR to the ReSi checkout}"
: "${REP_SIM:?Set REP_SIM to the ReSi experiments directory}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
mkdir -p logs "$MPLCONFIGDIR"

python scripts/resi.py run-task \
  --manifest "$MANIFEST" \
  --index "$SLURM_ARRAY_TASK_ID"
