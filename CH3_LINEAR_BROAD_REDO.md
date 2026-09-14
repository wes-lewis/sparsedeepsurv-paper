# Chapter 3 broad-sweep completeness redo

Tracking doc for the `wes/ch3-linear-broad-redo` branch: closing a
completeness gap in the dissertation chapter (patient-similarity
smoothing for gated survival models), found by re-reading the actual
manuscript text and underlying scripts against post-dissertation-extension
findings.

## The gap

The published broad sweep (`ch3_kipan_broad.py` / `ch3_brca_broad_v2.py`,
1,680 total fits across both datasets, `n_reps=10`) only covers the
MLP-predictor families (LSPIN, Concrete) — confirmed directly by Figure
4's own caption ("with LSPIN and Concrete parameterizations shown
separately"). The linear-predictor families (L-LSPIN, L-Concrete) that
the chapter's own conclusion recommends, especially for BRCA, were only
attempted in a separate `ch3_broad_gentle.py` script whose history
(`COLLABORATOR_GUIDE.md`: "Replication attempts... have produced mixed
results so far") was never consolidated.

## Two real bugs found in that script, independent of the missing families

1. **Wrong lever swept for Concrete-type gates.** `gate_sigma` is a
   confirmed no-op for `_sample_concrete` (only ever reads
   `self.temperature`); v1 swept `gate_sigma` for every family including
   Concrete/L-Concrete, producing wasted, duplicate-result compute along
   that dimension, while never touching `temperature` — Concrete's actual
   lever — at all.
2. **A crash bug.** `_worker()` referenced an undefined `_SDS_SRC`,
   confirmed present in v1 too. Under the `spawn` multiprocessing start
   method the script sets, this crashes every worker with `NameError` —
   plausibly a real contributor to why the historical linear-family
   attempts never produced consolidated results.

Both fixed in `ch3_broad_gentle_v2.py` (see its module docstring for
full detail), which also added a `--pilot` mode, `--families` filter, and
per-family lever ranges, plus tracked `temperature` per run row (v1 only
tracked `gate_sigma`).

## Pilot (to find reasonable ranges before the full redo)

Real KIPAN data, cheap grid (3-5 lever values, reduced lambda/reps).
Found: unlike the synthetic ground-truth-recovery task (thread 5 in
`POST_DISSERTATION_PLAN.md`), where useful gate_sigma multipliers went up
to 8-64x, on real KIPAN data moderate ranges suffice — C-index stayed
stable (0.70-0.73) across a wide tested range for every family, with no
collapse. v1's original fixed defaults (LSPIN/L-LSPIN sigma~0.2,
Concrete/L-Concrete temperature=0.3) already sat near the sweet spot;
they just needed to actually be swept (for Concrete-type) or given a
slightly wider band (for LSPIN-type) rather than reused unexamined.

Final ranges used for the full redo: LSPIN/L-LSPIN sigma in a 2-value
band bracketing each dataset's center (KIPAN 0.15/0.25, BRCA 0.1/0.2);
Concrete/L-Concrete temperature in [0.2, 0.5] bracketing v1's fixed 0.3.

## Full redo: scale and setup

n_reps=5 (half the original protocol's 10) — chosen because the broad
sweep is a LOWESS-smoothed trend display across many grid points, not a
per-point hypothesis test, so it doesn't need the same per-point
precision a paired significance test would; not cut further given
thread 2 already showed 6 reps can flip sign on an identical setting.
Result: ~2,560 total fits across both datasets (4 families x full
lambda/smoothing grid), about 1.5x the original manuscript's 1,680 (2
families only) for double the family coverage.

Also found and fixed a real performance bug mid-run: the script set no
BLAS/OMP thread limit, so each worker process spawned ~83 threads
unconstrained. On a shared 64-core box already at system load 130+ from
other users' jobs, this was serious avoidable contention — confirmed by
killing and restarting BRCA (only ~7% progress lost) with
`OMP_NUM_THREADS=1` etc. set: per-config time dropped from ~170s to
~35-50s, a ~3-4x speedup.

## Results

Replicated the manuscript's own "lowest vs. highest smoothing, lambda
held fixed" test (Section 3.6) across all 8 sampled lambda values, for
all four families, both datasets:

**KIPAN** (outputs: `extras/data/runs/ch3_kipan_broad_gentle_v2_full/`)

| Family | Union smaller at high smoothing | ARI higher | Affinity higher |
|---|---|---|---|
| LSPIN (repro) | 0/8 | 8/8 | 6/8 |
| Concrete (repro) | 8/8 | 6/8 | 7/8 |
| L-LSPIN (new) | 0/8 | 8/8 | 6/8 |
| L-Concrete (new) | **8/8** | **8/8** | **8/8** |

**BRCA** (outputs: `extras/data/runs/ch3_brca_broad_gentle_v2_full/`)

| Family | Union smaller at high smoothing | ARI higher | Affinity higher | C-index lo→hi smoothing |
|---|---|---|---|---|
| LSPIN (repro) | 7/8 | 8/8 | 8/8 | 0.601→0.589 |
| Concrete (repro) | 8/8 | 8/8 | 4/8 | 0.609→0.605 |
| L-LSPIN (new) | 6/8 | 8/8 | 8/8 | 0.605→0.620 |
| L-Concrete (new) | **8/8** | **8/8** | **8/8** | 0.614→**0.639** |

## Interpretation

**Confirms and strengthens the chapter's central claim**: the ARI/affinity
reproducibility gain from smoothing now holds across all four families on
both datasets — previously demonstrated on only two. **L-Concrete is the
standout**, with a clean 8/8/8 result on both datasets and, on BRCA,
smoothing actually *improves* C-index (+0.025) rather than merely
preserving it. This is exactly the architecture the chapter's own
conclusion recommends for BRCA, and it now has the strongest
reproducibility evidence of any family on the dataset where that
recommendation matters most.

**Two honest complications, not smoothed over**:
1. This redo's LSPIN reproduction on KIPAN gets the *opposite*
   union-direction result from the manuscript's own published number
   (which reports contraction; this redo finds 0/8 contracting, median
   +119% expansion). ARI/affinity gains are robust in the same runs — the
   union-contraction claim specifically looks sensitive to the exact
   lambda/sigma operating point for LSPIN-type gates. This does *not*
   generalize to BRCA, where this redo's LSPIN reproduction (7/8
   contracting) is consistent with the manuscript's own already-mixed
   BRCA framing. Concrete-type gates (Concrete, L-Concrete) reproduce
   the contraction direction robustly on both datasets both times.
2. Concrete's affinity result on BRCA is weak (4/8, a coin flip) — the
   one genuinely soft spot in the full redo.

## Next steps if this goes into the manuscript

- Decide how to handle the KIPAN LSPIN union-direction discrepancy: either
  re-run at v1's exact original narrow band to check whether it's a
  hyperparameter-sensitivity artifact of this redo's wider grid, or add a
  caveat noting the sensitivity directly.
- If incorporated, Figure 4 needs a new linear-predictor panel (the
  `_save_broad_multiplot_for_families(["L-LSPIN", "L-Concrete"], ...)`
  output already exists per-dataset from this run) alongside the existing
  MLP-predictor panel.
- Consider whether L-Concrete's strong, clean result changes how strongly
  the chapter can state its BRCA architectural recommendation — this
  redo provides the reproducibility evidence that recommendation was
  previously missing.
