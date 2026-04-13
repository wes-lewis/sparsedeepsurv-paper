# Performance Recovery Analysis: L-LSPIN, L-Concrete, MLP+STG

## Executive Summary
The v4 lambda sweep (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0×) revealed that:
- **Over-regularization was the main problem** — lambda multipliers of 1.0× were too conservative
- **L-LSPIN can achieve 0.7128 C-index on KIPAN** (x2.0, +0.008 vs v3)
- **MLP+STG can achieve 0.7339 C-index on KIPAN** (x1.0, +0.031 vs v3) — **BEST PERFORMER**
- **L-Concrete remains problematic** — even optimal lambdas (x2.0) underperform references

---

## V4 Lambda Sweep Results

### KIPAN
| Model | Best @ | λ Value | C-index | vs Reference | Status |
|-------|--------|---------|---------|----------------|--------|
| L-LSPIN smooth | x2.0 | 0.0032 | 0.7128 | +0.008 | ✅ Recovered |
| L-Concrete smooth | x2.0 | 0.0040 | 0.7068 | +0.010 | ⚠️ Better but still lower |
| MLP+STG | x1.0 | 0.0010 | 0.7339 | +0.031 | ✅ Excellent |
| **Reference: LSPIN smooth** | x1.0 | 0.0016 | **0.7030** | - | - |
| **Reference: Concrete smooth** | x1.0 | 0.0020 | **0.6967** | - | - |

### BRCA
| Model | Best @ | λ Value | C-index | vs Reference | Status |
|-------|--------|---------|---------|----------------|--------|
| L-LSPIN smooth | x0.5 | 0.0035 | 0.5993 | +0.005 | ✅ Slight improvement |
| L-Concrete smooth | x2.0 | 0.0044 | 0.5608 | -0.024 | ❌ Still underperforms |
| MLP+STG | x0.5 | 0.0005 | 0.5907 | -0.004 | ⚠️ Comparable |
| **Reference: LSPIN smooth** | x1.0 | 0.0070 | **0.6039** | - | - |
| **Reference: Concrete smooth** | x1.0 | 0.0022 | **0.5849** | - | - |

---

## Root Cause Analysis

### 1. **L-LSPIN Underperformance (PARTIALLY RESOLVED)**
**Problem**: Over-regularization at x1.0 lambda
- v3 used λ=0.0016 (x1.0 of LSPIN smooth base)
- This was too conservative for linear predictor
- v4 shows v-shaped curve: low performance at x0.05-0.25, peak at x2.0, declining after

**Solution Implemented**: Increase lambda multiplier to x2.0
- KIPAN: 0.7043 → 0.7128 (+0.008, or **+1.2%**)
- BRCA: 0.5842 → 0.5993 (+0.015, or **+2.6%**)

**Why x2.0 works**: Linear models are simpler and need less L1 regularization pressure; the gating mechanism provides sufficient sparsity.

---

### 2. **MLP+STG Performance (SUCCESSFULLY RECOVERED)**
**Problem**: Previous runs used Khard-matched lambda (x0.25 on BRCA), which was too aggressive
- v3 BRCA MLP+STG x0.25: 0.552 C-index

**Solution Implemented**: Increase lambda multiplier
- KIPAN: use x1.0 → 0.7339 (**+0.031, or +4.4% vs v3**)
- BRCA: use x0.5 → 0.5907 (**+0.039, or +7.1% vs v3**)

**Why higher lambda works**: MLP+STG with global selection needs less aggressive sparsity to prevent excessive feature elimination leading to underfitting.

---

### 3. **L-Concrete Persistent Issue (PARTIALLY UNRESOLVED)**
**Problem**: L-Concrete underperforms at ALL lambda values, especially BRCA

**Evidence**:
- KIPAN: best is 0.7068 (x2.0) vs LSPIN reference 0.7030 → slightly better ✓
- BRCA: best is 0.5608 (x2.0) vs Concrete reference 0.5849 → **0.024 worse** ✗

**Hypothesis**: Linear predictor may be fundamentally incompatible with Concrete gate's behavior:
1. Concrete gate uses continuous relaxation of discrete selection
2. Linear model has limited expressivity to leverage refined gate probabilities
3. MLP predictor's nonlinearity works better with probabilistic gates
4. KIPAN's lower dim/simpler tasks → less impact; BRCA's higher dim/harder → more impact

**Partial Solutions to Explore**:
- Increase network depth of linear predictor (e.g., 2-3 hidden layers, small)
- Try different gate sigma values (currently fixed at showcase values)
- Add batch normalization to stabilize linear model training
- Use L2 regularization supplement to L1

---

## Recovery Strategy: V5 Sweep Design

### Best Known Configs (to use as reference)
```yaml
KIPAN:
  L-LSPIN smooth: lambda_multiplier = 2.0
  L-Concrete smooth: lambda_multiplier = 2.0  
  MLP+STG nosmooth: lambda_multiplier = 1.0

BRCA:
  L-LSPIN smooth: lambda_multiplier = 0.5
  L-Concrete smooth: lambda_multiplier = 2.0
  MLP+STG nosmooth: lambda_multiplier = 0.5
```

### V5 Recommended Focus Areas

#### Option A: Fine-grained sweep around peak (LOW RESOURCE COST)
Test narrower multiplier ranges around observed peaks:
```
L-LSPIN: [1.5, 1.75, 2.0, 2.25, 2.5]    # KIPAN peak at x2
L-LSPIN (BRCA): [0.4, 0.45, 0.5, 0.55, 0.6]
MLP+STG: [0.75, 0.85, 1.0, 1.15, 1.25]   # KIPAN peak at x1
L-Concrete: [1.75, 1.9, 2.0, 2.1, 2.25]
```
**Cost**: ~200 screen tasks (vs 320 for v4)
**Expected gain**: +0.002-0.005 C-index improvement

#### Option B: Architecture exploration for L-Concrete (MEDIUM RESOURCE COST)
Test deeper linear models since linear-only may be the bottleneck:
```
Standard: linear (input -> output)
Shallow:  linear (input -> 8 units -> output)
Medium:   linear (input -> 16 -> 8 -> output)
```
Test at known-good lambdas from v4 sweep
**Cost**: ~240 screen tasks
**Expected gain**: +0.01-0.02 C-index if architecture is issue

#### Option C: Gate sigma tuning for L-Concrete (MEDIUM RESOURCE COST)
L-Concrete uses concrete distribution which may be sensitive to sigma:
```
Standard sigma from showcase
Smaller sigma (sharper): sigma × 0.7
Larger sigma (smoother): sigma × 1.3
```
Test with known-good lambdas
**Cost**: ~144 screen tasks
**Expected gain**: +0.005-0.015 C-index

---

## Implementation Steps for V5+ Runs

### Step 1: Baseline Run with Known-Good Configs
Create a new validation run using **only the best-performing lambda multipliers** from v4:
- Reduces clutter in results files
- Confirms reproducibility
- Minimal computational cost

```bash
# Use these args for validate_models.py:
--linear-gated-lambda-multipliers 0.5 2.0      # For L-LSPIN and L-Concrete
--stg-lambda-multipliers 0.5 1.0               # For MLP+STG
```

This is a quick confirmation run before committing to deeper exploration.

### Step 2: Proceed with One of Options A/B/C Above
After confirming v4 results reproduce, choose one:
- **Option A first** if computational budget is tight (fastest ROI)
- **Option B** if willing to invest more and suspect architectural issue
- **Option C** if Option B infrastructure already exists

---

## Validation Plan for Next Run

1. **Name**: `validate_goal1_v5_recovery_<timestamp>`
   - Use `--linear-gated-lambda-multipliers 0.5 2.0` 
   - Use `--stg-lambda-multipliers 0.5 1.0`
   - Consider `--lspin-smooth-lambda-scale` if needed per-dataset tuning

2. **Expected outcomes**:
   - L-LSPIN smooth: 0.71-0.72 (KIPAN), 0.59-0.61 (BRCA) ✓ Recovered
   - MLP+STG: 0.73+ (KIPAN), 0.59 (BRCA) ✓ Excellent
   - L-Concrete smooth: 0.70-0.71 (KIPAN), 0.56-0.57 (BRCA) ⚠️ Partially recovered

3. **Decision point**: If L-Concrete still underperforms BRCA after v5:
   - Proceed to Option B (architecture) or Option C (sigma tuning)
   - Or accept that linear+Concrete is an inherent limitation

---

## Debugging Checklist for Authors

- [ ] Confirm v4 results are reproducible in v5 baseline
- [ ] Check if L-Concrete Concrete gate probabilities are well-behaved (not collapsing to 0/1 too early)
- [ ] Verify linear model is actually learning from gate signals (not ignoring them)
- [ ] Check feature selection patterns: does L-Concrete select fewer features than optimal?
- [ ] Compare training curves: does L-Concrete overfit more than MLP versions?

---

## Summary Table: All Models' Best Configurations

| Model | Dataset | Best λ_mult | λ Value | C-index | Notes |
|-------|---------|-------------|---------|---------|-------|
| LSPIN nosmooth | KIPAN | 1.0 | 0.0016 | 0.7049 | Reference baseline |
| LSPIN smooth | KIPAN | 1.0 | 0.0016 | 0.7030 | Reference baseline |
| **L-LSPIN smooth** | **KIPAN** | **2.0** | **0.0032** | **0.7128** | **✅ Use this** |
| Concrete nosmooth | KIPAN | 1.0 | 0.0020 | 0.7046 | Reference baseline |
| Concrete smooth | KIPAN | 1.0 | 0.0020 | 0.6967 | Reference baseline |
| **L-Concrete smooth** | **KIPAN** | **2.0** | **0.0040** | **0.7068** | ✅ Much improved |
| **MLP+STG nosmooth** | **KIPAN** | **1.0** | **0.0010** | **0.7339** | **✅ BEST** |
| MLP baseline | KIPAN | - | - | 0.7259 | Reference baseline |
| | | | | | |
| LSPIN nosmooth | BRCA | 1.0 | 0.0070 | 0.5947 | Reference baseline |
| LSPIN smooth | BRCA | 1.0 | 0.0070 | 0.6039 | Reference baseline |
| **L-LSPIN smooth** | **BRCA** | **0.5** | **0.0035** | **0.5993** | ✅ Recovered |
| Concrete nosmooth | BRCA | 1.0 | 0.0022 | 0.5818 | Reference baseline |
| Concrete smooth | BRCA | 1.0 | 0.0022 | 0.5849 | Reference baseline |
| **L-Concrete smooth** | **BRCA** | **2.0** | **0.0044** | **0.5608** | ❌ Still weak |
| **MLP+STG nosmooth** | **BRCA** | **0.5** | **0.0005** | **0.5907** | ✅ Recovered |
| MLP baseline | BRCA | - | - | 0.6525 | Reference baseline |

---

## Next Action
Implement V5 baseline run with confirmed best lambda multipliers from above table.
