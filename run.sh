#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda:0}"
MODE="${MODE:-ablation}"
DATASET="${DATASET:-all}"
CONFIG="${CONFIG:-${ROOT}/data/config.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/results/run_$(date -u +%Y%m%dT%H%M%SZ)}"
UCF_FEATURES="${UCF_FEATURES:-${ROOT}/video-features/ucf}"
XD_FEATURES="${XD_FEATURES:-${ROOT}/video-features/xd}"

if [[ "$MODE" != "main" && "$MODE" != "ablation" ]]; then
  echo "MODE must be main or ablation" >&2
  exit 2
fi
if [[ "$DATASET" != "ucf" && "$DATASET" != "xd" && "$DATASET" != "all" ]]; then
  echo "DATASET must be ucf, xd, or all" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$ROOT"

ENTRY="algorithm.main"
if [[ "$MODE" == "ablation" ]]; then
  ENTRY="ablation.ablation"
fi

for DATA in ucf xd; do
  if [[ "$DATASET" != "all" && "$DATASET" != "$DATA" ]]; then
    continue
  fi
  FEATURES="$UCF_FEATURES"
  if [[ "$DATA" == "xd" ]]; then
    FEATURES="$XD_FEATURES"
  fi
  "$PYTHON" -m "$ENTRY" --phase score --dataset "$DATA" \
    --features "$FEATURES" --output "$OUTPUT_ROOT/$MODE" \
    --config "$CONFIG" --device "$DEVICE"
  "$PYTHON" -m "$ENTRY" --phase evaluate --dataset "$DATA" \
    --features "$FEATURES" --output "$OUTPUT_ROOT/$MODE" \
    --config "$CONFIG" --device "$DEVICE"
done
