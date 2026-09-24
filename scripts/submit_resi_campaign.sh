#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "Usage: $0 <graphs|vision|language> <run-name> [--compute-only]" >&2
  exit 2
fi

DOMAIN="$1"
RUN_NAME="$2"
MODE="${3:-}"

case "$DOMAIN" in
  graphs)
    CAMPAIGN_FILE="configs/resi_graph_campaign.yaml"
    ;;
  vision)
    CAMPAIGN_FILE="configs/resi_vision_campaign.yaml"
    ;;
  language)
    CAMPAIGN_FILE="configs/resi_language_campaign.yaml"
    ;;
  *)
    echo "Unsupported ReSi domain: $DOMAIN" >&2
    exit 2
    ;;
esac
if [[ -n "$MODE" && "$MODE" != "--compute-only" ]]; then
  echo "Unknown option: $MODE" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(dirname "$SCRIPT_DIR")}"
WORK_ROOT="${WORK_ROOT:-$(dirname "$REPO_ROOT")}"
RESI_DIR="${RESI_DIR:-$WORK_ROOT/resi}"
REP_SIM="${REP_SIM:-$RESI_DIR/experiments}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$HOME/containers/resi-manifold-pytorch-24.06-v2.sqsh}"
RESI_MAX_CONCURRENT_GPUS="${RESI_MAX_CONCURRENT_GPUS:-8}"
# Vision/language representations are large enough that kernel construction
# for the AUC measures is memory-bandwidth bound and CPU-slow by 1-2 orders
# of magnitude (see MANIFOLD_RESI_DEVICE's docstring in adapter.py). Default to the allocated GPU for these two
# domains; still overridable via an already-exported MANIFOLD_RESI_DEVICE.
# Left unset for graphs, whose existing archived results were computed on
# CPU and where representations are small enough not to need this.
case "$DOMAIN" in
  vision|language)
    MANIFOLD_RESI_DEVICE="${MANIFOLD_RESI_DEVICE:-cuda}"
    ;;
esac
CAMPAIGN_PATH="$REPO_ROOT/$CAMPAIGN_FILE"

# RESI_RUNTIME selects which cluster's execution mechanism to use:
#   container -- Apptainer/Pyxis via srun --container-image.
#   native    -- a conda env activated directly inside the sbatch script,
#                for clusters without container support. Set up the env
#                once first with scripts/setup_resi_native_env.sh. The
#                native scripts default to a preemptible partition/QOS
#                (see docs/resi_cluster.md); adjust them for your cluster.
RESI_RUNTIME="${RESI_RUNTIME:-container}"
case "$RESI_RUNTIME" in
  container)
    TASK_SCRIPT="$REPO_ROOT/scripts/resi_array_container.sbatch"
    PREPARE_SCRIPT="$REPO_ROOT/scripts/resi_prepare_container.sbatch"
    # Set RESI_EXCLUDE_NODELIST to a comma-separated node list if REP_SIM
    # lives on storage that is not mounted on every node. Use --exclude, not
    # --nodelist: --nodelist means the allocation must include every listed
    # node, not "pick any one of these".
    RESI_EXCLUDE_NODELIST="${RESI_EXCLUDE_NODELIST:-}"
    ;;
  native)
    TASK_SCRIPT="$REPO_ROOT/scripts/resi_array_native.sbatch"
    PREPARE_SCRIPT="$REPO_ROOT/scripts/resi_prepare_native.sbatch"
    # No nodes are excluded by default.
    RESI_EXCLUDE_NODELIST="${RESI_EXCLUDE_NODELIST:-}"
    ;;
  *)
    echo "Unsupported RESI_RUNTIME: $RESI_RUNTIME (expected 'container' or 'native')" >&2
    exit 2
    ;;
esac
nodelist_args=()
if [[ -n "$RESI_EXCLUDE_NODELIST" ]]; then
  nodelist_args=(--exclude="$RESI_EXCLUDE_NODELIST")
fi

if [[ ! "$RESI_MAX_CONCURRENT_GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "RESI_MAX_CONCURRENT_GPUS must be a positive integer, got: $RESI_MAX_CONCURRENT_GPUS" >&2
  exit 2
fi

if [[ ! -d "$RESI_DIR" ]]; then
  echo "ReSi checkout not found: $RESI_DIR" >&2
  exit 1
fi
if [[ ! -d "$REP_SIM" ]]; then
  echo "ReSi experiments directory not found: $REP_SIM" >&2
  exit 1
fi
if [[ ! -f "$CAMPAIGN_PATH" ]]; then
  echo "Campaign config not found: $CAMPAIGN_PATH" >&2
  exit 1
fi
if [[ ! -f "$TASK_SCRIPT" ]]; then
  echo "Task script not found: $TASK_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$PREPARE_SCRIPT" ]]; then
  echo "Preparation script not found: $PREPARE_SCRIPT" >&2
  exit 1
fi
if [[ "$RESI_RUNTIME" == "container" && ! -f "$CONTAINER_IMAGE" ]]; then
  echo "Configured container image not found: $CONTAINER_IMAGE" >&2
  exit 1
fi
export REPO_ROOT WORK_ROOT RESI_DIR REP_SIM CONTAINER_IMAGE
export MANIFOLD_RESI_DEVICE
mkdir -p "$REPO_ROOT/logs"
echo "MANIFOLD_RESI_DEVICE=${MANIFOLD_RESI_DEVICE:-unset (representations stay put)}"

export RESI_CAMPAIGN="$CAMPAIGN_PATH"
export RESI_CAMPAIGN_RUN_NAME="$RUN_NAME"
echo "Submitting preparation ($RESI_RUNTIME runtime) for $DOMAIN campaign '$RUN_NAME'."
# This blocks (sbatch --wait) with NO output until the prepare job finishes,
# including while it sits queued (PD) waiting for a free node outside
# $RESI_EXCLUDE_NODELIST -- that queueing time can look identical to a hang.
# Do not Ctrl+C here: that only kills this script's local wait, it does not
# cancel the already-submitted SLURM job, and running the script again
# submits a second, duplicate prepare job racing the first. Check progress from
# another terminal instead:
echo "  squeue -u \"\$USER\""
prepare_job=$(sbatch \
  --parsable \
  --wait \
  --job-name="resi-${DOMAIN}-prepare" \
  "${nodelist_args[@]}" \
  "$PREPARE_SCRIPT")
echo "Preparation job $prepare_job completed successfully."

SAFE_RUN_NAME=$(printf '%s' "$RUN_NAME" \
  | LC_ALL=C tr '[:upper:]' '[:lower:]' \
  | sed -E 's/[^a-z0-9._-]+/-/g; s/^[-._]+//; s/[-._]+$//')
RUN_DIR="$REPO_ROOT/data/resi/runs/$SAFE_RUN_NAME/$DOMAIN"
COMPUTE_MANIFEST="$RUN_DIR/compute_manifest.jsonl"
BASELINE_MANIFEST="$RUN_DIR/baseline_manifest.jsonl"
if [[ ! -f "$COMPUTE_MANIFEST" ]]; then
  echo "Preparation did not create the compute manifest: $COMPUTE_MANIFEST" >&2
  exit 1
fi
if [[ ! -f "$BASELINE_MANIFEST" ]]; then
  echo "Preparation did not create the baseline manifest: $BASELINE_MANIFEST" >&2
  exit 1
fi

compute_count=$(awk 'NF {count++} END {print count+0}' "$COMPUTE_MANIFEST")
if [[ "$compute_count" -lt 1 ]]; then
  echo "Compute manifest is empty: $COMPUTE_MANIFEST" >&2
  exit 1
fi

dependency_args=()
baseline_job=""
if [[ "$MODE" != "--compute-only" ]]; then
  if [[ ! -f "$BASELINE_MANIFEST" ]]; then
    echo "Baseline manifest not found: $BASELINE_MANIFEST" >&2
    exit 1
  fi
  baseline_count=$(awk 'NF {count++} END {print count+0}' "$BASELINE_MANIFEST")
  if [[ "$baseline_count" -gt 0 ]]; then
    export MANIFEST="$BASELINE_MANIFEST"
    baseline_job=$(sbatch \
      --parsable \
      --job-name="resi-${DOMAIN}-baseline" \
      --array="0-$((baseline_count - 1))%1" \
      "${nodelist_args[@]}" \
      "$TASK_SCRIPT")
    dependency_args=(--dependency="afterok:$baseline_job")
    echo "Submitted $DOMAIN baseline job $baseline_job ($baseline_count sequential tasks)."
  fi
fi

export MANIFEST="$COMPUTE_MANIFEST"
compute_job=$(sbatch \
  --parsable \
  --job-name="resi-${DOMAIN}-compute" \
  "${dependency_args[@]}" \
  --array="0-$((compute_count - 1))%$RESI_MAX_CONCURRENT_GPUS" \
  "${nodelist_args[@]}" \
  "$TASK_SCRIPT")

echo "Submitted $DOMAIN compute job $compute_job ($compute_count array tasks, at most $RESI_MAX_CONCURRENT_GPUS running concurrently, excluding nodes: ${RESI_EXCLUDE_NODELIST:-none})."
if [[ -n "$baseline_job" ]]; then
  echo "Compute waits for successful completion of baseline job $baseline_job."
fi
