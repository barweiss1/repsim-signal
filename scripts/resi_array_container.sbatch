#!/usr/bin/env bash
#SBATCH --job-name=resi-manifold-container
#SBATCH --output=logs/resi-manifold-container-%A_%a.out
#SBATCH --error=logs/resi-manifold-container-%A_%a.err
#SBATCH --time=23:59:00
#SBATCH --ntasks=1
#SBATCH --array=0-7
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --qos=normal

set -euo pipefail

# Submit from the manifold_repsim checkout, or export REPO_ROOT explicitly.
REPO_ROOT="${REPO_ROOT:-$PWD}"
# The configured layout keeps manifold_repsim and ReSi below one shared root.
WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"
RESI_DIR="${RESI_DIR:-$WORK_ROOT/resi}"
REP_SIM="${REP_SIM:-$RESI_DIR/experiments}"

# SquashFS image built by scripts/build_resi_enroot_container.sh.
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/containers/resi-manifold-pytorch-24.06-v2.sqsh}"
# Preserve host paths inside the container so manifest/config paths stay valid.
# If REP_SIM lives outside WORK_ROOT, add it to CONTAINER_MOUNTS as well.
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$WORK_ROOT:$WORK_ROOT}"

: "${MANIFEST:?Set MANIFEST to a generated compute_manifest.jsonl}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a zero-based SLURM array}"

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
if [[ ! -f "$CONTAINER_IMAGE" ]]; then
  echo "Configured container image does not exist: $CONTAINER_IMAGE" >&2
  exit 1
fi

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
echo "CONTAINER_IMAGE=$CONTAINER_IMAGE"
echo "CONTAINER_MOUNTS=$CONTAINER_MOUNTS"
echo "MANIFOLD_RESI_MAX_POINTS=${MANIFOLD_RESI_MAX_POINTS:-disabled}"
echo "MANIFOLD_RESI_AUC_MAX_POINTS=${MANIFOLD_RESI_AUC_MAX_POINTS:-inherits general limit}"
echo "MANIFOLD_RESI_SUBSAMPLE_SEED=$MANIFOLD_RESI_SUBSAMPLE_SEED"
echo "MANIFOLD_RESI_DEVICE=${MANIFOLD_RESI_DEVICE:-unset (representations stay put)}"

# RESI_DIR/REP_SIM are baked in as literal text below (expanded by THIS
# outer shell, not left to inherit into the container's login shell).
# resi_prepare_container.sbatch's comment above explains why: they were
# previously only exported and relied on inheriting through srun into the
# container's `bash -l` login shell, and were observed reverting to stale
# values there (the container's own login-shell init runs before the `-c`
# body and can leave earlier values in place). PYTHONPATH was already
# re-exported explicitly and was unaffected; REP_SIM/RESI_DIR were not.
srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT:$REPO_ROOT/src:$RESI_DIR'
    export REP_SIM='$REP_SIM'
    export RESI_DIR='$RESI_DIR'
    export MPLCONFIGDIR='$MPLCONFIGDIR'
    export MANIFOLD_RESI_DEVICE='$MANIFOLD_RESI_DEVICE'
    cd '$REPO_ROOT'
    python scripts/resi.py run-task \
      --manifest '$MANIFEST' \
      --index '$SLURM_ARRAY_TASK_ID'
  "
