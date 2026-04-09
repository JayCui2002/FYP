#!/usr/bin/env bash
set -euo pipefail

args=(
  python
  streamdiffusionv2_bridge.py
  --host
  0.0.0.0
  --port
  "${STREAM_PORT:-8765}"
  --real-v2
  --assets-root
  /app
  --config-path
  "${STREAM_CONFIG_PATH:-/app/input/streamv2v_config.yaml}"
  --checkpoint-folder
  "${STREAM_CHECKPOINT_FOLDER:-/models/ckpts/wan_causal_dmd_v2v}"
  --height
  "${STREAM_HEIGHT:-512}"
  --width
  "${STREAM_WIDTH:-512}"
  --step
  "${STREAM_STEP:-2}"
  --noise-scale
  "${STREAM_NOISE_SCALE:-0.85}"
  --prompt
  "${STREAM_PROMPT:-golden vase}"
)

if [[ -n "${STREAM_NEGATIVE_PROMPT:-}" ]]; then
  args+=(--negative-prompt "${STREAM_NEGATIVE_PROMPT}")
fi

if [[ -n "${STREAM_DEVICE:-}" ]]; then
  args+=(--device "${STREAM_DEVICE}")
fi

if [[ "${STREAM_USE_SR:-0}" == "1" ]]; then
  args+=(--sr)
fi

if [[ "${STREAM_STREAM_WO_BATCH:-0}" == "1" ]]; then
  args+=(--stream-wo-batch)
fi

if [[ "${STREAM_USE_TAEHV:-0}" == "1" ]]; then
  args+=(--use-taehv)
fi

if [[ -n "${STREAM_SR_MODEL_PATH:-}" ]]; then
  args+=(--sr-model-path "${STREAM_SR_MODEL_PATH}")
fi

if [[ -n "${STREAM_SR_TILE:-}" ]]; then
  args+=(--sr-tile "${STREAM_SR_TILE}")
fi

if [[ -n "${STREAM_MODEL_TYPE:-}" ]]; then
  args+=(--model-type "${STREAM_MODEL_TYPE}")
fi

if [[ -n "${STREAM_NUM_FRAMES:-}" ]]; then
  args+=(--num-frames "${STREAM_NUM_FRAMES}")
fi

exec "${args[@]}"
