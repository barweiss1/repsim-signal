#!/usr/bin/env bash
# Download the missing ReSi model checkpoints -- CIFAR100 (vision) and
# SmolLM2 (language) -- onto the cluster's storage. Plain bash; no SLURM
# or container needed, since this is just network I/O, not compute.
#
# Run this from a login node, inside tmux/screen -- SmolLM2 alone
# is ~414 GiB across two files and will take a long time over the network;
# you do not want that tied to a single SSH session that might drop.
#
#   tmux new -s resi-downloads
#   bash scripts/download_resi_models.sh
#   # Ctrl+B then D to detach; reattach later with: tmux attach -t resi-downloads
#
# Paths follow ReSi's repsim/benchmark/paths.py convention
# (VISION_MODEL_PATH = $REP_SIM/models/vision,
# NLP_SMOLLM_PATH = $REP_SIM/models/nlp/smollm), so whatever ReSi checkout
# ends up pointed at RESI_DIR below will find these without extra setup.
#
# RESI_DIR defaults to ~/research/resi -- override if your ReSi checkout
# lives elsewhere:
#   RESI_DIR=/some/other/path bash scripts/download_resi_models.sh

set -euo pipefail

RESI_DIR="${RESI_DIR:-$HOME/research/resi}"
REP_SIM="${REP_SIM:-$RESI_DIR/experiments}"

VISION_SIMBENCH_DIR="$REP_SIM/models/vision/vision_models_simbench"
SMOLLM_DIR="$REP_SIM/models/nlp/smollm"

echo "RESI_DIR = $RESI_DIR"
echo "REP_SIM  = $REP_SIM"
if [[ ! -d "$RESI_DIR" ]]; then
  echo "WARNING: $RESI_DIR does not exist yet. Creating it, but double-check" >&2
  echo "this is really where you want your ReSi checkout/data to live." >&2
fi

mkdir -p "$VISION_SIMBENCH_DIR" "$SMOLLM_DIR"

echo "############################################################"
echo "# Vision: CIFAR100 checkpoints (Zenodo, ~156 GB total)"
echo "# Resumable download, extracted then deleted one part at a time."
echo "############################################################"
declare -A CIFAR_PARTS=(
  [shortcut_c100]="https://zenodo.org/records/15148777/files/shortcut_c100.tar.gz?download=1"      # ~48.6 GB (Part 3)
  [normal_c100]="https://zenodo.org/records/15148773/files/normal_c100.tar.gz?download=1"           # ~19.4 GB (Part 4)
  [randomlabel_c100]="https://zenodo.org/records/15148781/files/randomlabel_c100.tar.gz?download=1" # ~38.8 GB (Part 5)
  [augment_c100]="https://zenodo.org/records/15148783/files/augment_c100.tar.gz?download=1"         # ~48.7 GB (Part 6)
)
for name in "${!CIFAR_PARTS[@]}"; do
  url="${CIFAR_PARTS[$name]}"
  archive="$VISION_SIMBENCH_DIR/../${name}.tar.gz"
  marker="$VISION_SIMBENCH_DIR/.${name}.done"
  if [[ -f "$marker" ]]; then
    echo "-- $name already downloaded+extracted, skipping --"
    continue
  fi
  echo "-- downloading $name --"
  wget -c "$url" -O "$archive"
  echo "-- extracting $name into $VISION_SIMBENCH_DIR --"
  tar -xzf "$archive" -C "$VISION_SIMBENCH_DIR"
  rm -f "$archive"
  touch "$marker"
done

echo "############################################################"
echo "# Done with CIFAR100."
echo "############################################################"

echo "############################################################"
echo "# Language: SmolLM2 checkpoints (~414 GiB across 2 files)"
echo "# Anonymous, resumable download; extracted then deleted."
echo "############################################################"
SMOLLM_BASE_URL="https://share.innkube.fim.uni-passau.de/public/resi-benchmark"
SMOLLM_FILES=(
  "smollm_mnli_checkpoints.tar.zst"
  "smollm_sst2_checkpoints.tar.zst"
)
for fname in "${SMOLLM_FILES[@]}"; do
  archive="$SMOLLM_DIR/../${fname}"
  marker="$SMOLLM_DIR/.${fname}.done"
  if [[ -f "$marker" ]]; then
    echo "-- $fname already downloaded+extracted, skipping --"
    continue
  fi
  echo "-- downloading $fname (~207 GiB) --"
  wget -c "$SMOLLM_BASE_URL/$fname" -O "$archive"
  echo "-- extracting $fname into $SMOLLM_DIR --"
  if tar --help 2>&1 | grep -q -- --zstd; then
    tar --zstd -xf "$archive" -C "$SMOLLM_DIR"
  else
    unzstd -c "$archive" | tar -xf - -C "$SMOLLM_DIR"
  fi
  rm -f "$archive"
  touch "$marker"
done

echo "############################################################"
echo "# All done."
echo "############################################################"
du -sh "$VISION_SIMBENCH_DIR" "$SMOLLM_DIR"