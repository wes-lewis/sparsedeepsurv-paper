# Is the smoothing reproducibility claim circular?

Tracking doc for `wes/ch3-smoothing-circularity-check`. Follows directly
from a methodological critique of Chapter 3's central claim (patient-
similarity smoothing improves cluster-stability/ARI and affinity
reproducibility): the smoothing penalty directly shrinks disagreement
between neighboring patients' gate vectors, and the validation metrics
are computed on those same gate vectors. Any reasonable shrinkage
regularizer mechanically reduces cross-run variance regardless of
whether what's being smoothed toward is biologically meaningful — so
"smoothing improves reproducibility" is close to the *expected* outcome
of a working regularizer, not strong evidence of biological value on its
own.

## Three checks (`extras/analyses/ch3_smoothing_circularity_check.py`)

1. **Negative-control affinity graph** — build the KNN smoothing graph
   from real PCA-projected covariates vs. from pure Gaussian noise of
   matching shape (same k, same construction code). If a meaningless
   graph gives the same reproducibility gain, the effect is generic
   shrinkage, not evidence tied to real molecular structure.
2. **Independent biology, not just internal agreement** — cluster ARI
   against real histology labels (never in the loss or the graph), for
   unsmoothed / smoothed-real / smoothed-null.
3. **Generic/housekeeping convergence check** — does the smoothing-
   stabilized consensus gene set skew toward broadly-proliferative
   Hallmark categories (a confound the chapter's own discussion already
   flags) relative to more specific ones?

KIPAN only, LSPIN and L-Concrete (one family whose broad-sweep behavior
was flagged as less robust, one "star performer"), 6 reps per condition.

## Result: a genuine, family-dependent split

| Family | cross-run ARI (base→real→null) | affinity (base→real→null) | histology-ARI (base→real→null) |
|---|---|---|---|
| LSPIN | 0.904 → 0.972 → 0.962 | 0.997 → 0.994 → 0.996 | 0.769 → 0.787 → **0.812** |
| L-Concrete | 0.920 → **0.984** → 0.741 | 0.977 → **0.986** → 0.692 | 0.793 → **0.816** → 0.709 |

**LSPIN fails the negative control.** The null (meaningless) graph does
as well or better than the real graph on both cross-run reproducibility
and — more importantly — on histology alignment. The circularity concern
is empirically confirmed for this family: its smoothing gain looks like
generic shrinkage, not evidence tied to real molecular structure.

**L-Concrete passes cleanly.** The real graph beats both the unsmoothed
baseline and the null graph on every metric, and the null graph actively
*hurts* all three — dropping below the unsmoothed baseline. A meaningless
smoothing target destabilizes training rather than merely failing to
help, which is the signature you'd want if the mechanism is doing
something real and specific to molecular neighborhoods rather than being
a free lunch that works with any graph.

**One caveat even for L-Concrete**: its real-graph-smoothed consensus set
(only 183 genes at this operating point) has a higher generic-Hallmark
fraction (12.8%) than the unsmoothed baseline (8.8%) — a real but
imprecise signal given the small gene count, worth another look with a
larger consensus set before treating it as settled either way. LSPIN
shows no such shift (moot given it already fails the main test).

## How this fits the rest of the investigation

This is not an isolated result — it reinforces a pattern that has shown
up repeatedly across this whole line of work: **L-Concrete (Concrete-
type gates) behaves as the mechanistically real, robust family** in
every test run so far —

- per-patient selection stability (`POST_DISSERTATION_PLAN.md` thread 2):
  Concrete showed real, sweep-robust stability on KIPAN; LSPIN did not
  without an extensive, only-partially-successful rescue.
- synthetic ground-truth recovery (thread 5): Concrete's recovery
  advantage replicated more consistently across effect sizes than
  LSPIN's.
- broad-sweep union-direction consistency (`CH3_LINEAR_BROAD_REDO.md`):
  Concrete/L-Concrete reproduced the union-contraction direction
  robustly on both datasets; LSPIN/L-LSPIN did not (reversed on KIPAN).
- and now: L-Concrete passes a negative control that LSPIN fails.

**LSPIN repeatedly shows signs of being more artifact-prone**, not
because of a single fluke result but as a consistent thread across
independent tests built for different purposes.

## Recommendation for the manuscript

The current framing treats LSPIN and Concrete-family reproducibility
evidence as roughly parallel support for the same smoothing claim. That
is not warranted based on this result. Options, roughly in order of
how much they'd change the chapter:

1. **Minimal**: add an explicit caveat that the reproducibility argument
   has only been validated against a negative control for Concrete-type
   gates, and note LSPIN's result could reflect generic regularization
   rather than molecular-structure-specific smoothing.
2. **Moderate**: re-run this same negative-control check on the other
   family pair (Concrete, L-LSPIN) and on BRCA, to see whether the
   split is really "gate type" (LSPIN-type vs. Concrete-type) or
   dataset/family-specific — right now this is one data point per
   family, on one dataset.
3. **Larger**: reposition the chapter's central claim to be specifically
   about Concrete-type gates, with LSPIN framed as a comparison point
   that doesn't show the same validated mechanism — consistent with the
   broader pattern above, and arguably a stronger, more defensible claim
   than the current family-agnostic framing.

## Next steps if pursued further

- Extend the negative-control check to Concrete and L-LSPIN (the two
  families not tested here) and to BRCA, to confirm the LSPIN-type vs.
  Concrete-type split generalizes rather than being specific to this one
  family pair.
- Revisit the L-Concrete generic-gene-fraction caveat with a larger
  consensus gene set (e.g. a lower gate-rate threshold) for more stable
  estimates.
- If pursuing option 2/3 above, this pairs naturally with
  `CH3_LINEAR_BROAD_REDO.md`'s own finding that Concrete-type gates were
  the more robust family in that redo too — the two documents should be
  read together.
