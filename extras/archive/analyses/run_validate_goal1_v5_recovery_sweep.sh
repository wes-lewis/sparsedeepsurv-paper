#!/usr/bin/env bash
# Goal-1 validation v5: Recovery sweep for L-LSPIN, L-Concrete, and MLP+STG.
#
# Context: v4 identified optimal lambda multipliers via full 0.05-10.0x sweep:
#   - L-LSPIN: x2.0 optimal on KIPAN (0.7128), x0.5 optimal on BRCA (0.5993)
#   - L-Concrete: x2.0 optimal on both datasets (but still underperforms)
#   - MLP+STG: x1.0 optimal on KIPAN (0.7339), x0.5 optimal on BRCA (0.5907)
#
# This v5 run:
#   - Sweeps fine-grained ranges around peak performance
#   - Includes best-known configs and neighbors for robustness
#   - Reduces computational cost vs v4 (full sweep)
#   - Sets up for architectural/sigma exploration if needed
#
# Expected improvement: +0.002-0.005 C-index over v4 best
set -euo pipefail

PYTHON=/home/wes/.conda/envs/musevo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(dirname "$SCRIPT_DIR")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PAPER_ROOT}/data/runs/validate_goal1_v5_recovery_sweep_${TIMESTAMP}"

KIPAN_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219/selected_showcase_configs.csv"
BRCA_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_brca_adaptive_v2_selfcontained_ste_randominit_20260406_120115/selected_showcase_configs.csv"

echo "[run_validate_goal1_v5_recovery] Results dir: ${RESULTS_DIR}"
echo "[run_validate_goal1_v5_recovery] Starting validate_models.py at $(date)"

# Fine-grained sweep around peaks from v4:
# - L-LSPIN KIPAN: peak at x2.0 → test [1.5, 1.75, 2.0, 2.25, 2.5]
# - L-LSPIN BRCA: peak at x0.5 → test [0.4, 0.45, 0.5, 0.55, 0.6]
# - L-Concrete: peak at x2.0 → test [1.5, 1.75, 2.0, 2.25, 2.5]
# - MLP+STG KIPAN: peak at x1.0 → test [0.8, 0.9, 1.0, 1.1, 1.2]
# - MLP+STG BRCA: peak at x0.5 → test [0.4, 0.45, 0.5, 0.55, 0.6]

"$PYTHON" "${SCRIPT_DIR}/validate_models.py" \
    --results-dir "${RESULTS_DIR}" \
    --mode goal1 \
    --datasets kipan brca \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --kipan-showcase "${KIPAN_SHOWCASE}" \
    --brca-showcase "${BRCA_SHOWCASE}" \
    --goal1-lambda-multipliers 1.0 \
    --stg-lambda-base 0.001 \
    --stg-lambda-multipliers 0.4 0.45 0.5 0.55 0.6 0.8 0.9 1.0 1.1 1.2 \
    --linear-gated-lambda-multipliers 0.4 0.45 0.5 0.55 0.6 1.5 1.75 2.0 2.25 2.5 \
    2>&1 | tee "${RESULTS_DIR}_stdout.log"

echo "[run_validate_goal1_v5_recovery] validate_models done at $(date)"
echo "[run_validate_goal1_v5_recovery] Results in ${RESULTS_DIR}"
