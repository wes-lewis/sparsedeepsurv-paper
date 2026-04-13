#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$ROOT/data/runs"
STAMP="$(date +%Y%m%d_%H%M%S)"
PYTHON="${PYTHON:-python}"
GPUS=("${@}")

if [ ${#GPUS[@]} -eq 0 ]; then
  GPUS=(0 2 4 6)
fi

cd "$ROOT"

KIPAN_DIR="$RUNS/ch3_kipan_broad_gentle_${STAMP}"
BRCA_DIR="$RUNS/ch3_brca_broad_gentle_${STAMP}"

echo "[launch] KIPAN -> $KIPAN_DIR"
"$PYTHON" analyses/ch3_broad_gentle.py \
  --dataset kipan \
  --results-dir "$KIPAN_DIR" \
  --gpus "${GPUS[@]}" \
  2>&1 | tee "${KIPAN_DIR}_stdout.log"

echo "[launch] BRCA -> $BRCA_DIR"
"$PYTHON" analyses/ch3_broad_gentle.py \
  --dataset brca \
  --results-dir "$BRCA_DIR" \
  --gpus "${GPUS[@]}" \
  2>&1 | tee "${BRCA_DIR}_stdout.log"

echo "[done]"
