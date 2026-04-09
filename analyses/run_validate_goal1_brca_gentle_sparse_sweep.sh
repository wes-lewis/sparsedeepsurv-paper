#!/usr/bin/env bash
# BRCA-focused goal-1 sweep under the gentler gate parameterization.
#
# Purpose:
#   1. Keep gated predictor architectures close to their dense baselines.
#   2. Use open-at-init LSPIN gates (bias=0) with no extra gate dropout / gate-only WD.
#   3. Re-sweep lambda_sparse to find a more defensible patient-sparse regime on BRCA.
#
# Sweep design:
#   - LSPIN / Concrete nosmooth+smooth share goal1 lambda multipliers below.
#   - L-LSPIN / L-Concrete use the linear-gated multiplier sweep.
#   - MLP+STG gets its own lambda multipliers, tilted upward to test sparser regimes.
set -euo pipefail

PYTHON=/home/wes/.conda/envs/musevo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(dirname "$SCRIPT_DIR")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PAPER_ROOT}/data/runs/validate_goal1_brca_gentle_sparse_sweep_${TIMESTAMP}"
BRCA_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_brca_adaptive_v2_selfcontained_ste_randominit_20260406_120115/selected_showcase_configs.csv"

echo "[run_validate_goal1_brca_gentle_sparse_sweep] Results dir: ${RESULTS_DIR}"
echo "[run_validate_goal1_brca_gentle_sparse_sweep] Starting validate_models.py at $(date)"

"$PYTHON" "${SCRIPT_DIR}/validate_models.py" \
    --results-dir "${RESULTS_DIR}" \
    --mode goal1 \
    --datasets brca \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --brca-showcase "${BRCA_SHOWCASE}" \
    --lspin-init-bias 0.0 \
    --goal1-lambda-multipliers 1.0 1.25 1.5 2.0 2.5 3.0 \
    --stg-lambda-base 0.001 \
    --stg-lambda-multipliers 0.5 0.75 1.0 1.25 1.5 2.0 3.0 \
    --stg-hidden-dims 64 32 \
    --linear-gated-lambda-multipliers 0.75 1.0 1.25 1.5 2.0 3.0 \
    --gating-hidden-dim 32 \
    2>&1 | tee "${RESULTS_DIR}_stdout.log"

echo "[run_validate_goal1_brca_gentle_sparse_sweep] validate_models done at $(date)"
echo "[run_validate_goal1_brca_gentle_sparse_sweep] Results in ${RESULTS_DIR}"
