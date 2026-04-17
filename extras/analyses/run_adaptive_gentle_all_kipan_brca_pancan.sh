#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${PAPER_ROOT}/data/runs/adaptive_gentle_all_kipan_brca_pancan_${TS}"

GPUS=("${@}")
if [ ${#GPUS[@]} -eq 0 ]; then
  GPUS=(0 2 4 6)
fi

mkdir -p "${RUN_ROOT}"

echo "[launcher] run root: ${RUN_ROOT}"
echo "[launcher] gpus: ${GPUS[*]}"

cd "${PAPER_ROOT}"

"${PYTHON}" analyses/ch3_kipan_adaptive_v2.py \
  --gpus "${GPUS[@]}" \
  --results-dir "${RUN_ROOT}/kipan"

"${PYTHON}" analyses/ch3_brca_adaptive_v2.py \
  --gpus "${GPUS[@]}" \
  --results-dir "${RUN_ROOT}/brca"

"${PYTHON}" analyses/ch3_pancan_adaptive_v2.py \
  --gpus "${GPUS[@]}" \
  --results-dir "${RUN_ROOT}/pancan"

echo "[launcher] complete: ${RUN_ROOT}"
