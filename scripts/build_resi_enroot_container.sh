#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"
RESI_DIR="${RESI_DIR:-$WORK_ROOT/resi}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/containers/resi-manifold-pytorch-24.06-v2.sqsh}"
BASE_IMAGE="${BASE_IMAGE:-$HOME/containers/nvidia+pytorch+24.06-py3.sqsh}"
BASE_IMPORT_URI="${BASE_IMPORT_URI:-docker://nvcr.io#nvidia/pytorch:24.06-py3}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-$WORK_ROOT:$WORK_ROOT}"
SRUN_ARGS="${SRUN_ARGS:-}"

REQUIREMENTS_FILE="$REPO_ROOT/requirements-resi-container-ngc24.06.txt"
CONSTRAINTS_FILE="$REPO_ROOT/constraints-resi-container-ngc24.06.txt"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "Missing REPO_ROOT: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -d "$RESI_DIR" ]]; then
  echo "Missing RESI_DIR: $RESI_DIR" >&2
  exit 1
fi
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "Missing requirements file: $REQUIREMENTS_FILE" >&2
  exit 1
fi
if [[ ! -f "$CONSTRAINTS_FILE" ]]; then
  echo "Missing constraints file: $CONSTRAINTS_FILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$BASE_IMAGE")" "$(dirname "$CONTAINER_IMAGE")"

if [[ ! -f "$BASE_IMAGE" ]]; then
  echo "Base image missing; importing $BASE_IMPORT_URI into $(dirname "$BASE_IMAGE")"
  (
    cd "$(dirname "$BASE_IMAGE")"
    enroot import "$BASE_IMPORT_URI"
  )
fi

if [[ ! -f "$BASE_IMAGE" ]]; then
  echo "Base image import did not create expected file: $BASE_IMAGE" >&2
  exit 1
fi

echo "Removing target image: $CONTAINER_IMAGE"
rm -f "$CONTAINER_IMAGE"

export REPO_ROOT WORK_ROOT RESI_DIR
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$RESI_DIR:${PYTHONPATH:-}"

echo "Building ReSi container image: $CONTAINER_IMAGE"
echo "Base image: $BASE_IMAGE"
echo "Mounts: $CONTAINER_MOUNTS"
echo "Extra srun args: ${SRUN_ARGS:-<none>}"

# shellcheck disable=SC2086
srun $SRUN_ARGS \
  --container-image="$BASE_IMAGE" \
  --container-save="$CONTAINER_IMAGE" \
  --container-mounts="$CONTAINER_MOUNTS" \
  bash -lc '
    set -euo pipefail
    export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src:$RESI_DIR:${PYTHONPATH:-}"
    cd "$REPO_ROOT"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip uninstall -y torchaudio torcheval opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true
    python - <<'"'"'PY'"'"'
from pathlib import Path
import shutil
import site

for root in dict.fromkeys(site.getsitepackages() + [site.getusersitepackages()]):
    root_path = Path(root)
    for path in root_path.glob("cv2*"):
        if path.is_dir() or path.suffix in {".so", ".pyd", ".py"}:
            print(f"Removing stale OpenCV artifact: {path}")
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
PY
    python -m pip install --no-cache-dir -c "$REPO_ROOT/constraints-resi-container-ngc24.06.txt" -r "$REPO_ROOT/requirements-resi-container-ngc24.06.txt"
    python -m pip install -e "$RESI_DIR"
    python "$REPO_ROOT/scripts/smoke_resi_container.py" --resi-dir "$RESI_DIR"
  '

echo "Built ReSi container image: $CONTAINER_IMAGE"
