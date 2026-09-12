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

**Updated 2026-09-11, twice** (see reassessment log below for full
numbers). First update after the initial real runs; second update after
thread 2 was redesigned around per-patient (not aggregate) selection
stability, which surfaced the first clearly positive, hyperparameter-
robust result of this whole round: **Concrete gates on KIPAN show real
per-patient selection reproducibility**, consistently above a
size-matched random null across a 6-cell lambda/sigma sweep. That result
does not extend to LSPIN or to BRCA. All priorities below are still
provisional — this is one positive data point on one (dataset, gate
family) pair, not a settled claim — but it changes what's worth chasing
next.

1. `wes/pd-selection-stability-index` — **run, updated**; per-patient
   redesign found KIPAN-Concrete personalization stability is real and
   sweep-robust; LSPIN and BRCA still show no effect. Highest-priority
   next step of the whole plan: understand *why* Concrete behaves
   differently from LSPIN here (its soft-gate distribution looks
   structurally different — much lower mean, more right-skewed) and
   whether BRCA's null is about sample size/dimensionality or something
   else, since that distinction determines whether this generalizes into
   a real manuscript claim or stays a KIPAN-Concrete-specific curiosity.
2. `wes/pd-small-sample-learning-curves` — **run**; Coxnet had the
   smallest generalization gap at low training fractions on both
   datasets. Same re-tuning follow-up as thread 2.
3. `wes/pd-transport-noninterchangeability` — **run**, underpowered
   (n=15-20 paired observations); increase folds/reps before treating
   the current null as conclusive.
4. `wes/pd-synthetic-ground-truth` — **run** at one effect-size setting,
   which turned out too hard for any method to solve (near-zero
   precision/recall across the board including baselines). Needs an
   effect-size/sparsity sweep before it can discriminate methods; once
   that's done this is still the most informative thread, since it is
   the only one with ground truth to check against.
5. `wes/pd-noise-robustness` — **run**, underpowered (n_reps=3); no
   signal detected yet either way.
6. `wes/pd-biological-validation` — **run**; nominally promising
   pathway-enrichment directions (Coagulation, Complement, TGF-beta) but
   nothing survived multiple-testing correction, and subgroup survival
   separation was not significant. Most likely candidate for a positive
   result with more data/subgroup-count tuning, least likely to be
   undercut by the "hyperparameters tuned for the wrong property"
   explanation that threads 1/2/6 point to.
7. `wes/pd-external-cohort-transfer` — **last priority, confirmed
   2026-09-11**: UK Biobank is not happening; restricted to public,
   no-application-required data only (Desmedt microarray is the sole
   current candidate). Held back until 1-6 clarify whether there is a
   personalization/sparsity advantage worth externally validating.

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
- **Status**: implemented and run (3-fold, 2-rep) on KIPAN and BRCA,
  both LSPIN and Concrete, `smooth` selection. Outputs at
  `<run_dir>/{kipan,brca}/pd_gate_set_interchangeability_{dataset}_3fold_2rep/`.
- **Result**: no significant own-vs-global or own-vs-foreign advantage on
  either dataset (best case KIPAN LSPIN own-vs-global: 61% positive,
  median +0.025, p=0.084; own-vs-foreign was *negative* on both datasets
  for LSPIN). n=15-20 paired subgroup observations per cell — underpowered,
  not a settled null. Next: rerun with more folds/reps for power before
  concluding.

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
- **Status (v1, superseded)**: first pass measured stability of the
  *aggregate/consensus* gene set (population-level union), run at 15
  bootstrap resamples on KIPAN/BRCA x LSPIN/Concrete. Outputs preserved
  at `<run_dir>/{kipan,brca}/pd_selection_stability_{dataset}_{family}_smooth_15boot/`
  for provenance, but this framing was the wrong unit of analysis for a
  method whose actual claim is about *individual* selection (see v2
  below) — flagged directly by feedback on the first result.
- **Status (v2, current)**: redesigned and rerun. Three sequential steps
  per (dataset, family): (0) a gate-saturation diagnostic on one
  reference fit, checking whether soft gate values sit near the 0.5 hard
  threshold (the failure mode `COLLABORATOR_GUIDE.md` calls out — gates
  near a threshold would flip from fit-to-fit noise alone, unrelated to
  the method's real ceiling); (1) *per-patient* stability — fix a
  held-out evaluation cohort once, bootstrap-refit the model 15 times on
  a disjoint training pool, and for each held-out patient compute the
  Nogueira-Brown stability of *their own* selected-gene vector across the
  15 independent fits, against a size-matched per-patient random null
  (plus the old aggregate metric kept alongside, since it's still the
  only fair comparison point for Coxnet, which has no per-patient notion
  of selection); (2) a 6-cell lambda x gate_sigma sweep (multipliers
  {1,2,4} x {1,0.5}, 6 bootstrap reps per cell) reporting per-patient
  stability, median per-patient selected count, and test C-index
  together. Outputs at
  `<run_dir>/{kipan,brca}/pd_selection_stability_v2_{dataset}_{family}_smooth/`.
- **Result (v2)**: gate saturation is *not* the explanation anywhere —
  in every (dataset, family) cell, gates were sharply bimodal (≤0.6% of
  values within 0.1 of the 0.5 threshold; most mass at 0 or 1), so
  low stability where it occurs reflects genuinely different genes
  clearing the threshold between fits, not indecisive gates. The
  headline result: **Concrete gates on KIPAN show real per-patient
  selection stability**, clearly and consistently above the size-matched
  null across every cell of the sweep (0.005-0.024 vs. null ≈0, with
  non-overlapping IQRs at the tuned operating point: median 0.023 vs.
  null -0.0001) — the first clearly positive, hyperparameter-robust
  personalization-stability result in this whole round of experiments.
  This does **not** generalize: LSPIN shows no such effect on KIPAN
  (median ≈ -0.0007, indistinguishable from null across the entire
  sweep) or on BRCA (both families ≈0 across the sweep, C-index also
  degrading at higher lambda on BRCA, down to ~0.53-0.57). Per-patient
  selected-set sizes also varied enormously by hyperparameter and
  dataset — KIPAN LSPIN selected a median ~1,179 of 6,000 genes per
  patient at the tuned operating point (~20%, not obviously "sparse" at
  the personal level even though the aggregate story calls it sparse),
  dropping to ~450 at 4x lambda without a corresponding stability gain;
  KIPAN Concrete ranged from ~620 down to ~80 across the sweep, with its
  best stability sitting at the *least* sparse end (1x lambda) rather
  than the sparsest. Aggregate-level stability (the v1 metric, kept for
  the Coxnet comparison) still favors Coxnet substantially in every
  cell (0.15-0.22 vs. the gated methods' 0.0003-0.033), which remains
  true but is now clearly labeled as a different, coarser quantity than
  the per-patient result above.
  **Implication**: there is a real, tunable personalization-stability
  story here, but it is specific to Concrete-style gates and (so far)
  to KIPAN — not a property of "the method" in general. Next steps: (a)
  investigate why Concrete's soft-gate distribution (very low mean
  probability, heavily right-skewed) behaves differently from LSPIN's
  here — this looks like it could be a genuine mechanistic difference
  between the two gate parameterizations worth explaining, not noise;
  (b) check whether BRCA's failure to show the effect is about sample
  size, dimensionality (24k vs 6k genes), or something dataset-specific,
  e.g. by rerunning at a few sample-size fractions; (c) if this holds up,
  it is a much stronger and more specific manuscript claim than
  "our method is stable" — it is "Concrete-style personalized gating
  achieves per-patient selection reproducibility a global sparse Cox
  model has no mechanism to define, on datasets with enough signal,"
  which is falsifiable and now has a first positive data point.

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
- **Status**: implemented and run for KIPAN LSPIN and BRCA LSPIN (single
  clustering pass each, `n-clusters=4`). Outputs at
  `<run_dir>/{kipan,brca}/pd_subgroup_biological_validation_{dataset}_lspin_smooth/`.
- **Result**: KIPAN formed 3 usable subgroups (n=255) with no significant
  survival separation beyond histology (log-rank p=0.41, Cox LR p=0.76).
  BRCA's default clustering cut did not yield multiple subgroups large
  enough to test at all. Hallmark enrichment surfaced plausible
  candidates (Coagulation, Complement, Protein Secretion, TGF-beta
  Signaling — all biologically sensible for kidney/breast cancer) at
  nominal p<0.05, but none survived BH correction (best q~0.07). Most
  promising thread for a future positive result (directionally
  plausible, not yet an artifact of an obviously wrong hyperparameter
  choice like threads 1/2/6) but not yet supportable as a claim. Next:
  try other cluster counts and, for BRCA, a clustering cut that actually
  separates the test set.

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
- **Status**: implemented and run (3 reps, noise_k in {0, 200, 1000}) on
  KIPAN and BRCA, LSPIN and Concrete vs. dense MLP (MLP+STG deferred —
  bespoke training loop not yet ported). Outputs at
  `<run_dir>/{kipan,brca}/pd_noise_robustness_{dataset}_3rep/`.
- **Result**: no clear robustness advantage detected for either gated
  family on either dataset; effect sizes were small relative to
  run-to-run std at only 3 reps (e.g. KIPAN LSPIN degraded slightly more
  than dense MLP as noise grew). Underpowered — "no signal detected yet,"
  not "no effect." Next: more reps before drawing a conclusion either way.

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
- **Status**: implemented and run at one effect-size/sparsity setting
  (effect_size=1.5, 8 true genes/subgroup, 4 synthetic subgroups) on
  KIPAN and BRCA. Outputs at
  `<run_dir>/{kipan,brca}/pd_synthetic_ground_truth_{dataset}/`.
- **Result**: the setting run was too hard for any method to solve —
  precision/recall/Jaccard were near zero for LSPIN, Concrete, *and* the
  Coxnet global baseline (e.g. BRCA: all three <0.0015 mean precision),
  and sanity-check C-index on the synthetic outcome was barely above
  chance (0.55-0.70). This experiment cannot yet discriminate methods.
  Next (required before this thread means anything): sweep effect size
  and/or reduce the true-feature search space (fewer total genes, or
  more true features per subgroup) so at least the strongest baseline
  clears a reasonable C-index on the synthetic outcome, then compare
  recovery quality at that setting.

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
- **Status**: implemented and run (3 reps, fractions {0.25, 0.5, 0.75,
  1.0}) on KIPAN and BRCA, LSPIN and Concrete vs. dense MLP vs. Coxnet
  (MLP+STG deferred as in thread 4). Outputs at
  `<run_dir>/{kipan,brca}/pd_small_sample_learning_curves_{dataset}_3rep/`.
- **Result**: Coxnet (sparse linear) had the smallest generalization gap
  at low training fractions on *both* datasets (e.g. BRCA at 25% data:
  Coxnet gap 0.327 vs. LSPIN 0.437 — the worst of the four — Concrete
  0.397, dense MLP 0.414). Neither gated family showed a sample-
  efficiency edge over the dense MLP baseline on either dataset. Same
  re-tuning follow-up as thread 2: rerun with lambda chosen for sample
  efficiency rather than C-index before treating this as structural.

### 7. External-cohort transfer (LAST priority, public data only)
- **Branch**: `wes/pd-external-cohort-transfer`
- **Claim targeted**: gate-selected feature sets transport better than a
  dense model's full feature set under real batch/platform shift.
- **Precedent**: standard external-validation pattern for genomic
  prognostic signatures.
- **Data constraint (2026-09-11 decision)**: UK Biobank is off the table
  -- it is not happening. This thread is restricted to cohorts that are
  publicly downloadable without an application/access process. The only
  candidate currently identified is the Desmedt breast microarray cohort
  already integrated in the `sparsedeepsurv` tutorial (GEO-derived, no
  application required). If Desmedt turns out too small or too poorly
  matched in mapped features to produce a meaningful comparison, this
  thread stops there rather than pursuing any access-gated dataset --
  do not substitute another restricted-access cohort as a workaround.
- **Next step**: train/select in-cohort on TCGA BRCA, freeze the
  consensus gate-selected feature set, map it onto the Desmedt GPL96
  platform (reusing the mygene-based probe mapping already built for the
  tutorial), retrain a downstream model on the frozen feature set, and
  evaluate on Desmedt against a size-matched dense-model comparison set.
  Scaffold at `extras/analyses/pd_external_cohort_transfer.py`.
- **Status**: scaffolded only, explicitly deferred until threads 1-6 are
  further along -- this is now confirmed last in priority order, both
  because it is the most data-constrained thread and because the
  2026-09-11 results below make it more important to first decide
  whether the underlying method has a claim worth externally validating.
- **Result**: _pending_

## Reassessment log

Add a dated entry here each time a thread produces a result and priorities
are revisited.

- 2026-09-11 (plan created): All seven threads scaffolded, none yet run.

- 2026-09-11 (threads 1, 2, 3, 4, 5, 6 run at real, if modest, scale):
  Implemented and executed all six threads against real KIPAN/BRCA data
  (thread 7 held back per the data-access restriction above). Reporting
  the results plainly rather than the outcome that was hoped for going
  in: **none of the six threads found a clear, statistically supported
  advantage for personalized/sparse gating over standard baselines at
  the scale run so far.** Specifics, so this doesn't get summarized away:

  - **Thread 1 (transport/non-interchangeability)**: own-vs-global
    C-index differences were not significant for either family on either
    dataset (KIPAN LSPIN: 61% of subgroups favored "own" over "global",
    median +0.025, Wilcoxon p=0.084; KIPAN Concrete: p=0.30; BRCA LSPIN:
    p=0.45; BRCA Concrete: p=0.78). Own-vs-*foreign* subgroup sets were
    directionally *worse* for LSPIN on both datasets (39% and 35% of
    subgroups favored "own"). n was small (15-20 paired subgroup
    observations per family/dataset from a 3-fold/2-rep run) -- this is
    underpowered, not necessarily a true null, but as run it does not
    support the non-interchangeability claim.
  - **Thread 2 (stability index)**: this is the most decisive result.
    Nogueira-Brown stability of LSPIN/Concrete hard-selection across 15
    bootstrap resamples was indistinguishable from a random-selection
    baseline of matched size on both datasets (KIPAN LSPIN: 0.0017 vs.
    random 0.0002; KIPAN Concrete: 0.025 vs. -0.001; BRCA LSPIN: 0.0010
    vs. -0.0005; BRCA Concrete: 0.0051 vs. 0.0006 -- all with heavily
    overlapping CIs). A plain elastic-net Cox (Coxnet, sparsity-matched)
    was *far* more stable on every comparison (0.16-0.22, with
    non-overlapping CIs vs. the gated methods). This directly
    contradicts the "more repeated, less interchangeable selection"
    hope from the original question -- as currently tuned, the gated
    deep models are not more reproducible than chance at the aggregate
    selection-set level, while the boring linear baseline is.
  - **Thread 3 (biological validation)**: KIPAN gate-cluster subgroups
    showed no significant survival separation beyond histology
    (log-rank p=0.41, Cox LR p=0.76 across 3 subgroups, n=255). BRCA
    clustering did not even yield multiple subgroups large enough to
    test (n_groups=1 at the default clustering cut). Hallmark pathway
    enrichment per subgroup had several nominal p<0.05 pathways
    (Coagulation, Complement, Protein Secretion, TGF-beta Signaling) but
    none survived BH correction (best q~0.07). Directionally plausible,
    not yet a supportable claim.
  - **Thread 4 (noise robustness)**: no clear robustness advantage for
    gated models under injected noise columns; effect sizes were small
    relative to run-to-run std at n_reps=3 (e.g. KIPAN LSPIN degraded
    slightly more than dense MLP as noise increased, -0.009 vs. -0.00002
    delta at k=1000; BRCA showed small positive deltas for both gated
    families but well within noise). Underpowered at 3 reps; the honest
    read is "no signal detected yet," not "no effect exists."
  - **Thread 5 (synthetic ground-truth recovery)**: at the one
    effect-size/sparsity setting run, precision/recall/Jaccard of
    selected-vs-true features were near zero for *all* methods
    including the Coxnet global baseline (e.g. BRCA: all three methods'
    mean precision <0.0015), and sanity-check C-index on the synthetic
    outcome was barely above chance (0.55-0.70). This means the
    generated problem was too hard (8 true genes per subgroup buried in
    6000-24000 real genes) for any method to solve at the available n --
    the experiment as configured cannot yet distinguish personalization
    quality, and needs the effect-size/sparsity sweep flagged in the
    script's own docstring before it says anything.
  - **Thread 6 (small-sample learning curves)**: the sparse *linear*
    baseline (Coxnet) had consistently the smallest generalization gap
    of any method at low training fractions on both datasets (e.g. BRCA
    at 25% training data: Coxnet gap 0.327 vs. LSPIN 0.437, Concrete
    0.397, dense MLP 0.414 -- LSPIN was the *worst* of the four). Gated
    models showed no sample-efficiency edge over the dense MLP baseline
    on either dataset.

  **What this changes going forward**: the pattern across five of six
  threads is the same -- a plain, non-personalized, non-deep sparse
  linear Cox model (Coxnet/elastic-net) matched or beat the gated deep
  models on stability, sample efficiency, and (where measurable)
  robustness, at the hyperparameters currently selected for the main
  paper results. That is a more fundamental issue than any single
  thread's null result: it suggests either (a) the selected
  hyperparameters for LSPIN/Concrete (tuned for predictive C-index, per
  `PERFORMANCE_RECOVERY_ANALYSIS.md`) are not the right operating point
  for a stability/robustness/personalization claim and a separate
  hyperparameter search targeting those properties directly is needed
  before re-running these threads, or (b) the personalization/sparsity
  story needs to lean on thread 3's biological-plausibility angle
  (nominally promising, not yet corrected-significant) rather than on
  stability/robustness/generalization superiority, which this round of
  evidence does not support. Recommended immediate next steps, in order:
  1. Re-run thread 2 (stability) and thread 6 (learning curves) with a
     lambda value chosen specifically to maximize stability/sample-
     efficiency (not the C-index-selected value) to test hypothesis (a)
     directly -- cheap, since both scripts already parameterize this.
  2. Increase power on thread 1 (more folds/reps) before concluding the
     non-interchangeability test is a true null.
  3. Sweep effect size on thread 5 so it can actually discriminate
     methods, then treat its result as the most informative one, since
     it is the only thread with ground truth to check against.
  4. Hold off on thread 7 until 1-3 clarify whether there is a
     personalization/sparsity advantage worth externally validating.

- 2026-09-11 (thread 2 redesigned around per-patient stability, per
  feedback that aggregate-set stability was the wrong unit of analysis
  for a personalization method): reran thread 2 as three sequential
  steps (gate-saturation diagnostic, per-patient stability against a
  size-matched null with a fixed held-out eval cohort across bootstrap
  refits, and a 6-cell lambda x gate_sigma sweep). Result: **Concrete
  gates on KIPAN show real, sweep-robust per-patient selection
  stability** (0.005-0.024 vs. a null of ≈0 in every cell; 0.023 vs.
  -0.0001 at the tuned operating point) — the first clearly positive
  result in this entire round. This does not extend to LSPIN on either
  dataset or to Concrete on BRCA; gate saturation ("stuck near 0.5") was
  ruled out as the explanation everywhere via the step-0 diagnostic
  (gates are sharply bimodal in every cell). Per-patient selected-set
  sizes also turned out to vary far more across (dataset, family,
  hyperparameter) than the aggregate framing suggested — KIPAN LSPIN
  selects ~20% of genes per patient at the tuned point, not obviously
  sparse individually despite the aggregate result implying sparsity.
  This reframes thread 2 from "gated selection isn't stable" to "one
  specific gate parameterization, on one dataset, shows a real
  stability advantage a global sparse model has no mechanism to
  produce" — promoted to top priority to understand why (Concrete vs.
  LSPIN mechanism; KIPAN vs. BRCA sample size/dimensionality) before
  deciding whether it generalizes into a manuscript claim.
