#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${QUANTWM_PATHS_FILE:-${ROOT}/configs/paths.env}"
if [[ -f "$CONFIG" ]]; then source "$CONFIG"; fi
MODEL="$1"
METHOD="$2"
shift 2
GPU="${1:-0}"
if [[ $# -gt 0 ]]; then shift; fi
export QUANTWM_PYTHON="${QUANTWM_PYTHON:-python}"
exec "$QUANTWM_PYTHON" "$ROOT/quantwm/generate.py" \
  --model "$MODEL" --method "$METHOD" --gpu "$GPU" "$@"
