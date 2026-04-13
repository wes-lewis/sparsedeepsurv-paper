# L-LSPIN, L-Concrete, MLP+STG Performance Recovery Guide

## Quick Start

### Immediate Action (Recommended)
Run the baseline confirmation with known-good configs:

```bash
cd /banach2/wes/lspin-repos/sparsedeepsurv-paper/analyses
bash run_validate_goal1_v5_baseline_best_configs.sh
```

This will:
- Validate reproducibility of v4 results
- Use minimal GPU resources (2 multipliers × 2 datasets instead of 8)
- Take ~2-3 hours on 8 GPUs
- Confirm models can reach target performance before paper submission

**Expected results**: L-LSPIN at 0.71+ (KIPAN), MLP+STG at 0.73 (KIPAN)

---

## Understanding the Problem

### Why Models Underperformed in V3

The v3 run used **showcase lambda values with multiplier 1.0×**:
- L-LSPIN smooth: λ = 0.0016 (from showcase)
- L-Concrete smooth: λ = 0.0020 (from showcase)
- MLP+STG: λ = 0.001 (from LSPIN showcase)

**Issue**: These were designed for MLP predictors with nonlinear capacity. Linear predictors need less regularization pressure because they already have limited complexity.

### V4 Discovery: Optimal Lambda Multipliers

The v4 sweep tested multipliers across 0.05× to 10.0×:

```
KIPAN:
  L-LSPIN:    0.05  0.1  0.25  0.5  [peak→ 0.7128]  2.0  5.0  10.0
                                                  ↑
                                                x2.0

  L-Concrete: 0.05  0.1  0.25  0.5  1.0  [peak→ 0.7068]  5.0  10.0
                                                     ↑
                                                   x2.0

  MLP+STG:    0.05  0.1  0.25  0.5  [peak→ 0.7339]  2.0  5.0  10.0
                                                ↑
                                              x1.0

BRCA:
  L-LSPIN:    0.05  0.1  0.25  [peak→ 0.5993]  1.0  2.0  5.0  10.0
                                           ↑
                                         x0.5

  L-Concrete: 0.05  0.1  0.25  0.5  1.0  [peak→ 0.5608]  5.0  10.0
                                                     ↑
                                                   x2.0

  MLP+STG:    0.05  0.1  0.25  [peak→ 0.5907]  1.0  2.0  5.0  10.0
                                           ↑
                                         x0.5
```

**Key insight**: Non-monotonic curves! Each model has an optimal "sweet spot" where sparsity and expressivity balance.

---

## Solution: Three-Tier Recovery Plan

### Tier 1: Baseline Confirmation (IMMEDIATE)
**Run**: `run_validate_goal1_v5_baseline_best_configs.sh`
**Effort**: ~2-3 hours, 8 GPUs
**Multipliers**: 0.5, 2.0 (covers most cases)
**Purpose**: Quick reproducibility check

**Proceed to Tier 2 if**: L-Concrete still underperforms BRCA by >0.02

### Tier 2: Fine-Grained Sweep (IF NEEDED)
**Run**: `run_validate_goal1_v5_recovery_sweep.sh`
**Effort**: ~6-8 hours, 8 GPUs
**Multipliers**: Dense sampling around peaks
- L-LSPIN: [0.4, 0.45, 0.5, 0.55, 0.6] + [1.5, 1.75, 2.0, 2.25, 2.5]
- L-Concrete: [1.5, 1.75, 2.0, 2.25, 2.5]
- MLP+STG: [0.4, 0.45, 0.5, 0.55, 0.6] + [0.8, 0.9, 1.0, 1.1, 1.2]
**Expected gain**: +0.002-0.005 C-index

### Tier 3: Architectural Exploration (RESEARCH PHASE)
If Tier 2 doesn't solve L-Concrete BRCA underperformance:

#### Option A: Deeper Linear Models
The issue: linear (input→output) may be too simple for Concrete's probabilistic gate

**Test**:
```yaml
- Standard: linear (input → output)
- Shallow: linear (input → 8 units → output) with ReLU
- Medium: linear (input → 16 → 8 → output) with ReLU
```

**Implementation**: Modify `_build_variants()` in validate_models.py to accept `predictor_hidden_dims` parameter

**Expected**: +0.005-0.015 C-index if architecture is the bottleneck

#### Option B: Gate Sigma Tuning
The issue: Concrete may need different temperature for linear model

**Test**: Gate sigma at 0.7×, 1.0×, 1.3× of showcase value

**Implementation**: Create variant generation for sigma sweep parallel to lambda

**Expected**: +0.003-0.010 C-index if gate behavior is suboptimal

---

## Detailed Analysis: Why L-Concrete Still Underperforms

### Problem Statement
- KIPAN: L-Concrete x2.0 = 0.7068 (reference Concrete smooth = 0.6967) ✓ OK
- BRCA: L-Concrete x2.0 = 0.5608 (reference Concrete smooth = 0.5849) ✗ **-0.024**

This suggests **dataset-specific or task-difficulty-dependent issue**.

### Hypothesis Chain

**Hypothesis 1: Concrete gate learning dynamics**
- Concrete distribution uses continuous relaxation parameter (temperature/gate_sigma)
- Cold temperature → sharp discrete-like behavior
- Linear model may struggle with sharp gating decisions

*Test*: Compare gate probability distributions between L-LSPIN and L-Concrete
```python
# In runs: analyze gate_probs_distribution (if logged)
# Check: Are L-Concrete gates collapsing too early to 0/1?
```

**Hypothesis 2: Feature interaction loss**
- MLP predictor can learn gate × feature interactions implicitly
- Linear predictor cannot; relies on gate for feature selection
- Concrete gates may select incompatible feature subsets for linear model

*Test*: Compare selected feature sets
```python
# Get feature importance/selection from L-Concrete vs L-LSPIN runs
# Check: Do L-Concrete and L-LSPIN select similar features?
# Check: Does L-Concrete select rare/difficult features?
```

**Hypothesis 3: Gradient flow through Concrete gate**
- Concrete's continuous relaxation creates different gradient landscape than LSPIN's hard sigmoid
- Linear model gradients may navigate Concrete landscape poorly

*Test*: Compare training curves
```python
# Check: Learning rate schedule for L-Concrete (is it too fast/slow?)
# Check: Does L-Concrete validation loss plateau earlier?
# Check: Is there overfitting pattern unique to L-Concrete?
```

### Debugging Checklist

Add these investigations to next run:

```python
# In validate_models.py or post-process results:

1. [ ] Log gate probability statistics per epoch
   - Mean/std of gate open probability
   - Fraction of always-open gates (>0.95)
   - Fraction of always-closed gates (<0.05)

2. [ ] Compare feature selection pattern
   - Which features does L-Concrete select vs L-LSPIN?
   - Are they consistent across runs/splits?

3. [ ] Analyze training dynamics
   - Plot validation C-index over epochs for best configs
   - Check for early stopping: does L-Concrete stop too early?

4. [ ] Check model configuration
   - Verify gate_sigma is actually being used
   - Verify lambda_sparse is applied correctly
   - Verify lambda_sample_smooth is applied correctly

5. [ ] Compare test vs validation performance
   - If test >> validation, model is overfitting
   - If test ≈ validation but both low, underfitting
```

---

## Execution Guide: Step-by-Step

### For Tier 1 (Baseline):

```bash
# 1. Make script executable
chmod +x /banach2/wes/lspin-repos/sparsedeepsurv-paper/analyses/run_validate_goal1_v5_baseline_best_configs.sh

# 2. Run in background or screen session
cd /banach2/wes/lspin-repos/sparsedeepsurv-paper/analyses
screen -S validate_v5
bash run_validate_goal1_v5_baseline_best_configs.sh
# Ctrl-A-D to detach

# 3. Monitor progress
tail -f /banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/validate_goal1_v5_baseline_best_configs_TIMESTAMP_stdout.log

# 4. Check results when done
# Results in: /banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/validate_goal1_v5_baseline_best_configs_TIMESTAMP/
python3 << 'EOF'
import pandas as pd
import glob

# Find latest v5_baseline run
runs = sorted(glob.glob('/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/validate_goal1_v5_baseline_best_configs_*'), reverse=True)
if runs:
    latest = runs[0]
    df = pd.read_csv(f'{latest}/screen_summary.csv')
    
    # Show results for problem models
    for dataset in ['kipan', 'brca']:
        print(f"\n{dataset.upper()}:")
        subset = df[(df['dataset'] == dataset) & 
                    (df['model_family'].isin(['L-LSPIN', 'L-Concrete', 'MLP+STG']))]
        for _, row in subset.iterrows():
            print(f"  {row['model_family']:15} x{row['lambda_multiplier']:4.1f}: {row['mean_test_cindex']:.4f} ± {row['ci95_test_cindex']:.4f}")
EOF
```

### For Tier 2 (Fine-Grained):

```bash
# Same as Tier 1, but use:
bash run_validate_goal1_v5_recovery_sweep.sh

# Will take longer (~6-8 hours vs ~2-3 hours)
```

---

## Expected Outcomes by Scenario

### Scenario A: Tier 1 Succeeds ✓
- L-LSPIN: 0.71-0.72 (KIPAN), 0.59-0.60 (BRCA)
- MLP+STG: 0.73+ (KIPAN), 0.59 (BRCA)
- L-Concrete: 0.70-0.71 (KIPAN), 0.56-0.57 (BRCA)

**Action**: Use these configs for final paper validation run
```bash
# Run validation with confirmed best configs
bash run_validate_goal1_v3.sh  # Or equivalent post-sweep run
```

### Scenario B: Tier 1 Succeeds for L-LSPIN/MLP+STG, L-Concrete Still Weak
- L-Concrete BRCA < reference despite best lambda

**Action**: Decide on Tier 2 or Tier 3:
- If time-constrained: Accept L-Concrete limited performance
- If flexible: Try Tier 2 fine-grained sweep first
- If research-oriented: Proceed to Tier 3 architecture exploration

### Scenario C: Tier 1 Results Differ Significantly from V4
- This would indicate non-reproducibility issue
- Likely causes:
  - Different random seeds (check `sds.set_seed()`)
  - GPU/torch version differences
  - Showcase config loading issue

**Debug**:
```bash
# Compare manifest and exemplar results
python3 << 'EOF'
import pandas as pd

v4_path = '/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/validate_goal1_v4_lineargated_stg_20260407_220814/'
v5_path = '[YOUR_V5_RESULTS_DIR]'

v4_manifest = pd.read_csv(f'{v4_path}/variant_manifest.csv')
v5_manifest = pd.read_csv(f'{v5_path}/variant_manifest.csv')

# Check if manifests match for common variants
print("Manifest differences:")
shared_variants = set(v4_manifest['variant_key']) & set(v5_manifest['variant_key'])
for vkey in shared_variants:
    v4_row = v4_manifest[v4_manifest['variant_key'] == vkey].iloc[0]
    v5_row = v5_manifest[v5_manifest['variant_key'] == vkey].iloc[0]
    if v4_row['lambda_sparse'] != v5_row['lambda_sparse']:
        print(f"  {vkey}: lambda mismatch {v4_row['lambda_sparse']} vs {v5_row['lambda_sparse']}")
EOF
```

---

## References

- **Analysis**: See `PERFORMANCE_RECOVERY_ANALYSIS.md` in project root
- **V4 Script**: `run_validate_goal1_v4_lineargated_stg_sweep.sh`
- **V4 Results**: `/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/validate_goal1_v4_lineargated_stg_20260407_220814/`
- **Model Code**: `SDS_SRC=/banach2/wes/lspin-repos/sparsedeepsurv/src/sparsedeepsurv/`

---

## Progress Checklist

- [ ] Read PERF ORMANCE_RECOVERY_ANALYSIS.md
- [ ] Run Tier 1 baseline (v5-baseline-best-configs)
- [ ] Verify L-LSPIN and MLP+STG achieve target C-index
- [ ] If L-Concrete still underperforms: decide Tier 2 vs accept limitation
- [ ] Prepare final validation run with confirmed configs
- [ ] Document final config choices in methods/supplementary

**Estimated total time**: 2-3 hours (Tier 1) + optional 6-8 hours (Tier 2)
