#!/usr/bin/env bash
# Goal-1 validation v5: Architecture and parameterization fixes for L-LSPIN, L-Concrete, MLP+STG.
#
# Problems addressed vs v3/v4:
#  1. L-LSPIN/L-Concrete: gating_hidden_dim=128 → 24K→128→24K = 6.1M gate params against
#     a 24K→1 = 24K linear predictor (gate 254× overparameterized). Reduced to 32.
#  2. MLP+STG: single hidden layer (input→64→1) vs plain MLP (input→64→32→1). Fixed by
#     using --stg-hidden-dims 64 32 to match the MLP predictor architecture.
#  3. predictor field missing from fixed-init tasks (state_dict mismatch on L-LSPIN/L-Concrete).
#     Fixed in validate_models.py _build_fixed_init_tasks.
#  4. MLP+STG selected by Khard-matching (picked wrong lambda); now selected by CI.
#  5. L-LSPIN/L-Concrete lambda sweep added (--linear-gated-lambda-multipliers).
#
# Lambda sweep: 8 points from x0.05 to x10 for L-LSPIN, L-Concrete, MLP+STG.
# LSPIN/Concrete pinned at x1.0 reference to conserve GPU budget.
set -euo pipefail

PYTHON=/home/wes/.conda/envs/musevo/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_ROOT="$(dirname "$SCRIPT_DIR")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PAPER_ROOT}/data/runs/validate_goal1_v5_arch_fix_${TIMESTAMP}"

KIPAN_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219/selected_showcase_configs.csv"
BRCA_SHOWCASE="${PAPER_ROOT}/data/runs/ch3_brca_adaptive_v2_selfcontained_ste_randominit_20260406_120115/selected_showcase_configs.csv"

echo "[run_validate_goal1_v5] Results dir: ${RESULTS_DIR}"
echo "[run_validate_goal1_v5] Starting validate_models.py at $(date)"

"$PYTHON" "${SCRIPT_DIR}/validate_models.py" \
    --results-dir "${RESULTS_DIR}" \
    --mode goal1 \
    --datasets kipan brca \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --kipan-showcase "${KIPAN_SHOWCASE}" \
    --brca-showcase "${BRCA_SHOWCASE}" \
    --goal1-lambda-multipliers 1.0 \
    --stg-lambda-base 0.001 \
    --stg-lambda-multipliers 0.05 0.1 0.25 0.5 1.0 2.0 5.0 10.0 \
    --stg-hidden-dims 64 32 \
    --linear-gated-lambda-multipliers 0.05 0.1 0.25 0.5 1.0 2.0 5.0 10.0 \
    --gating-hidden-dim 32 \
    2>&1 | tee "${RESULTS_DIR}_stdout.log"

echo "[run_validate_goal1_v5] validate_models done at $(date)"
echo "[run_validate_goal1_v5] Results in ${RESULTS_DIR}"
