#!/usr/bin/env bash
#SBATCH --job-name=resi-prepare
#SBATCH --output=logs/resi-prepare-%j.out
#SBATCH --error=logs/resi-prepare-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --qos=normal

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"
RESI_DIR="${RESI_DIR:-$WORK_ROOT/resi}"
REP_SIM="${REP_SIM:-$RESI_DIR/experiments}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/containers/resi-manifold-pytorch-24.06-v2.sqsh}"
# If REP_SIM lives outside WORK_ROOT, add it to CONTAINER_MOUNTS as well.
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$WORK_ROOT:$WORK_ROOT}"

: "${RESI_CAMPAIGN:?Set RESI_CAMPAIGN to a campaign YAML path}"
: "${RESI_CAMPAIGN_RUN_NAME:?Set RESI_CAMPAIGN_RUN_NAME}"

if [[ ! -f "$CONTAINER_IMAGE" ]]; then
  echo "Configured container image does not exist: $CONTAINER_IMAGE" >&2
  exit 1
fi
if [[ ! -f "$RESI_CAMPAIGN" ]]; then
  echo "Campaign config does not exist: $RESI_CAMPAIGN" >&2
  exit 1
fi

export REPO_ROOT WORK_ROOT RESI_DIR REP_SIM RESI_CAMPAIGN RESI_CAMPAIGN_RUN_NAME
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$RESI_DIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"

# RESI_DIR/REP_SIM are baked in as literal text below (expanded by THIS
# outer shell), not left to inherit through srun into the container's login
# shell. They were previously only exported and relied on that inheritance,
# and were observed reverting to stale values there -- the container's own
# `bash -l` login-shell init runs before the `-c` body and can leave earlier
# values in place. PYTHONPATH was already re-exported explicitly inside the
# `-c` body and was unaffected; REP_SIM/RESI_DIR were not, so this brings
# them in line with the same pattern rather than relying on inheritance.
srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH='$REPO_ROOT:$REPO_ROOT/src:$RESI_DIR'
    export REP_SIM='$REP_SIM'
    export RESI_DIR='$RESI_DIR'
    export MPLCONFIGDIR='$MPLCONFIGDIR'
    cd '$REPO_ROOT'
    python scripts/resi.py prepare \
      --campaign '$RESI_CAMPAIGN' \
      --run-name '$RESI_CAMPAIGN_RUN_NAME'
  "
