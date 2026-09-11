# Post-Dissertation Extension Plan

This is the tracking document for strengthening the personalization/sparsity
claims of the sparse gated survival modeling work beyond what the
dissertation-defense results support. It exists because, as of the
2026-04-18 collaborator review (see `COLLABORATOR_GUIDE.md` section 6), the
current evidence for those two claims is weak:

- **Predictive performance**: gated/sparse models (LSPIN, Concrete, MLP+STG)
  land at or below an unregularized dense MLP's C-index on both KIPAN and
  BRCA. Sparsity currently costs a little accuracy rather than buying any.
- **Personalization**: per-run gene-level overlap is above chance but not
  strongly reproducible. The current framing ("reproducible gate structure,
  not exact gene lists") is a defensible hedge, not a positive result.

Neither weakness is fixed by more lambda sweeps. Both call for experiments
that test claims a single global sparse model (Lasso-Cox, elastic-net Cox)
structurally cannot match. This document tracks seven such experiment
threads, each on its own branch off `master`, so they can be built and
evaluated independently and merged as they pan out.

## How to use this document

Each thread below has: the claim it targets, the literature precedent that
motivates it, the concrete next step, the branch it lives on, and a status.
Update status in place as work proceeds. When a thread produces a real
result (positive or clearly negative), record the outcome in the "Result"
line and decide whether it graduates into the manuscript, gets cut, or
needs a follow-up thread — add new rows for follow-ups rather than losing
the history of what was tried.

Do not treat this as a fixed checklist to clear in order. After each
thread reports a result, reassess priority for the remaining threads: a
strong result in one area (e.g., the transport test) may make a related
thread (e.g., the stability index) more or less urgent, and a null result
somewhere may surface a new thread worth adding.

## Priority order (reassess after each result)

1. `wes/pd-transport-noninterchangeability` — builds directly on existing
   scripts (`plot_kipan_feature_set_cv_transport.py`,
   `patient_subset_within_vs_outside_test.py`); cheapest path to a positive
   personalization result.
2. `wes/pd-selection-stability-index` — reframes the existing "overlap
   above chance" finding using a proper corrected-for-chance stability
   index instead of an informal percentage.
3. `wes/pd-biological-validation` — reanalysis of existing canonical runs
   (`extras/data/runs/adaptive/`) plus the tracked Hallmark gene sets; no
   new training required.
4. `wes/pd-noise-robustness` — cheap synthetic column augmentation on
   existing KIPAN/BRCA matrices; directly tests the "sparsity generalizes
   better" claim.
5. `wes/pd-synthetic-ground-truth` — heavier to build (needs a semi-
   synthetic hazard-generation harness) but the single most convincing
   thing available: proves personalization does something a global sparse
   model cannot, with ground truth to check against.
6. `wes/pd-small-sample-learning-curves` — straightforward to run, tests
   the n<<p generalization-advantage claim directly.
7. `wes/pd-external-cohort-transfer` — highest narrative payoff for
   "sparsity generalizes" but gated on getting a usable second cohort
   (Desmedt microarray set is the nearest candidate; UK Biobank needs a
   formal data-access application, flagged as a known blocker already).

## Threads

### 1. Transport / non-interchangeability test
- **Branch**: `wes/pd-transport-noninterchangeability`
- **Claim targeted**: personalized gate sets are not interchangeable
  across patients/subgroups.
- **Precedent**: instance-wise feature selection literature (INVASE, L2X)
  validates selection specificity directly, not just aggregate accuracy.
- **Next step**: extend the existing transport scripts to explicitly score
  each patient's own gate set vs. (a) a same-size global top-k set, (b)
  another patient's/subgroup's gate set, (c) a random same-size set,
  evaluated on that patient's held-out outcome. Scaffold added at
  `extras/analyses/pd_gate_set_interchangeability_test.py`.
- **Status**: scaffolded, not yet run.
- **Result**: _pending_

### 2. Formal feature-selection stability index
- **Branch**: `wes/pd-selection-stability-index`
- **Claim targeted**: selection is meaningfully more stable than chance
  and than competing sparse baselines, quantified properly.
- **Precedent**: Kuncheva stability index; Nogueira & Brown (2016)
  corrected-for-chance stability estimator.
- **Next step**: compute a corrected-for-chance stability index (with CI)
  across the existing bootstrap/rerun gate matrices, for LSPIN/Concrete vs.
  Lasso-Cox/elastic-net-Cox baselines. Scaffold added at
  `extras/analyses/pd_selection_stability_index.py`.
- **Status**: scaffolded, not yet run.
- **Result**: _pending_

### 3. Biological / clinical validation of personalized subgroups
- **Branch**: `wes/pd-biological-validation`
- **Claim targeted**: gate-derived patient subgroups carry real biological
  and prognostic signal beyond clinical stage/histology alone.
- **Precedent**: standard subtype-validation pattern in genomics survival
  papers (pathway enrichment by subgroup, log-rank survival separation).
- **Next step**: (a) log-rank test on survival curves across gate-cluster
  subgroups, adjusting for/beyond clinical stage; (b) Hallmark pathway
  enrichment comparison across subgroups using the already-tracked
  `data/gene_sets/enrichr/MSigDB_Hallmark_2020.gmt`. Scaffold added at
  `extras/analyses/pd_subgroup_biological_validation.py`.
- **Status**: scaffolded, not yet run.
- **Result**: _pending_

### 4. Noise-robustness / generalization-under-irrelevant-features test
- **Branch**: `wes/pd-noise-robustness`
- **Claim targeted**: sparsity gives a real generalization advantage, not
  just "doesn't hurt."
- **Precedent**: synthetic noise-feature injection is the standard
  robustness check in INVASE/LSPIN/Concrete-autoencoder papers.
- **Next step**: append k synthetic noise columns (k swept, e.g.
  0/50/200/1000) to existing KIPAN/BRCA matrices; compare held-out C-index
  degradation of dense MLP vs. gated-sparse models. Scaffold added at
  `extras/analyses/pd_noise_robustness_sweep.py`.
- **Status**: scaffolded, not yet run.
- **Result**: _pending_

### 5. Semi-synthetic ground-truth recovery
- **Branch**: `wes/pd-synthetic-ground-truth`
- **Claim targeted**: personalized selection recovers subgroup-specific
  true relevant features that a global sparse model cannot represent.
- **Precedent**: the standard proof-of-concept experiment for every major
  instance-wise selection method (INVASE, L2X, original LSPIN paper).
- **Next step**: build a semi-synthetic hazard generator on top of real
  KIPAN/BRCA covariance structure, with planted subgroup-specific true
  feature subsets; report precision/recall/Jaccard of selected-vs-true
  features per patient, benchmarked against Lasso-Cox and a single global
  sparse MLP. Scaffold added at
  `extras/analyses/pd_synthetic_ground_truth_recovery.py`.
- **Status**: scaffolded, not yet run. Heaviest-lift thread — needs a
  design pass on the hazard-generation harness before running.
- **Result**: _pending_

### 6. Small-sample / high-dimensional learning curves
- **Branch**: `wes/pd-small-sample-learning-curves`
- **Claim targeted**: sparsity's generalization advantage shows up most in
  the n<<p regime genomic survival data actually lives in.
- **Precedent**: Cox-nnet/DeepSurv-era argument for regularized survival
  models in high-p, low-n settings.
- **Next step**: subsample training folds (25/50/75/100%) and plot the
  train-minus-held-out C-index generalization gap for gated-sparse vs.
  dense MLP vs. Lasso-Cox. Scaffold added at
  `extras/analyses/pd_small_sample_learning_curves.py`.
- **Status**: scaffolded, not yet run.
- **Result**: _pending_

### 7. External-cohort transfer
- **Branch**: `wes/pd-external-cohort-transfer`
- **Claim targeted**: gate-selected feature sets transport better than a
  dense model's full feature set under real batch/platform shift.
- **Precedent**: standard external-validation pattern for genomic
  prognostic signatures.
- **Next step**: use the already-integrated Desmedt breast microarray
  cohort (see `sparsedeepsurv` tutorial) as the nearest available external
  BRCA target; train/select in-cohort, retrain a downstream model on the
  frozen selected feature set, evaluate on Desmedt. UK Biobank or other
  external cohorts remain a longer-term option pending data-access
  applications (known blocker, see `COLLABORATOR_GUIDE.md`). Scaffold
  added at `extras/analyses/pd_external_cohort_transfer.py`.
- **Status**: scaffolded, not yet run.
- **Result**: _pending_

## Reassessment log

Add a dated entry here each time a thread produces a result and priorities
are revisited.

- 2026-09-11: Plan created; all seven threads scaffolded, none yet run.
