#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export RESI_RUNTIME=native
exec "$SCRIPT_DIR/submit_resi_campaign.sh" graphs "$@"
