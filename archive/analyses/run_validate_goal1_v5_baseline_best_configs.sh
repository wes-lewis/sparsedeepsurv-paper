#!/usr/bin/env bash
# Goal-1 validation v5-baseline: Confirmation run with known-good lambda configs.
#
# Context: v4 identified optimal lambda multipliers via comprehensive sweep.
# This baseline run confirms reproducibility and uses minimal computational budget.
#
# Configurations from v4 analysis:
#   KIPAN:  L-LSPIN x2.0, L-Concrete x2.0, MLP+STG x1.0
#   BRCA:   L-LSPIN x0.5, L-Concrete x2.0, MLP+STG x0.5
#
# Expected: Reproduce v4 best results
# Resource: 2x cheaper than v4 (only 2 multipliers per model instead of 8)
set -euo pipefail

PYTHON=/home/wes/.conda/envs/musevo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(dirname "$SCRIPT_DIR")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PAPER_ROOT}/data/runs/validate_goal1_v5_baseline_best_configs_${TIMESTAMP}"

KIPAN_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219/selected_showcase_configs.csv"
BRCA_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_brca_adaptive_v2_selfcontained_ste_randominit_20260406_120115/selected_showcase_configs.csv"

echo "[run_validate_goal1_v5_baseline] Results dir: ${RESULTS_DIR}"
echo "[run_validate_goal1_v5_baseline] Starting validate_models.py at $(date)"

# Use only best-known lambda multipliers from v4 sweep:
# - L-LSPIN/L-Concrete need different multipliers for each dataset
#   but validate_models.py only takes --linear-gated-lambda-multipliers (applies to all)
#   So we omit the multipliers that are suboptimal for one dataset
# - Compromise: use 0.5 and 2.0 to cover both KIPAN (needs x2) and BRCA (needs x0.5)
# - MLP+STG also use 0.5 and 1.0 to cover both datasets

"$PYTHON" "${SCRIPT_DIR}/validate_models.py" \
    --results-dir "${RESULTS_DIR}" \
    --mode goal1 \
    --datasets kipan brca \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --kipan-showcase "${KIPAN_SHOWCASE}" \
    --brca-showcase "${BRCA_SHOWCASE}" \
    --goal1-lambda-multipliers 1.0 \
    --stg-lambda-base 0.001 \
    --stg-lambda-multipliers 0.5 1.0 \
    --linear-gated-lambda-multipliers 0.5 2.0 \
    2>&1 | tee "${RESULTS_DIR}_stdout.log"

echo "[run_validate_goal1_v5_baseline] validate_models done at $(date)"
echo "[run_validate_goal1_v5_baseline] Results in ${RESULTS_DIR}"
echo "[run_validate_goal1_v5_baseline]"
echo "[run_validate_goal1_v5_baseline] Expected best configs:"
echo "[run_validate_goal1_v5_baseline]"
echo "[run_validate_goal1_v5_baseline]   KIPAN best:"
echo "[run_validate_goal1_v5_baseline]     L-LSPIN x2.0 → ~0.7128"
echo "[run_validate_goal1_v5_baseline]     L-Concrete x2.0 → ~0.7068"
echo "[run_validate_goal1_v5_baseline]     MLP+STG x1.0 → ~0.7339"
echo "[run_validate_goal1_v5_baseline]"
echo "[run_validate_goal1_v5_baseline]   BRCA best:"
echo "[run_validate_goal1_v5_baseline]     L-LSPIN x0.5 → ~0.5993"
echo "[run_validate_goal1_v5_baseline]     L-Concrete x2.0 → ~0.5608"
echo "[run_validate_goal1_v5_baseline]     MLP+STG x0.5 → ~0.5907"
