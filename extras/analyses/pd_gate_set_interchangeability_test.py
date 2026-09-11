#!/usr/bin/env python3
"""Test whether personalized gate sets are interchangeable across patients.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 1). Existing scripts already ask related-but-narrower questions:

- `plot_kipan_feature_set_cv_transport.py` asks whether a *consensus/union*
  gate-derived feature set transports across k-fold splits.
- `patient_subset_within_vs_outside_test.py` asks whether a selected gene's
  Cox signal is stronger within its own selecting subset than outside it.

Neither directly tests non-interchangeability: whether a patient's *own*
personalized gate set predicts that patient's outcome better than an
equally-sized feature set drawn from somewhere else. That is the direct
test of the claim that personalization is doing something real rather than
approximating one global sparse subset.

Planned comparison, per patient (or per gate-derived subgroup, to keep
variance manageable):

    (a) own gate set      -- that patient's/subgroup's personalized
                              hard-selected features
    (b) global top-k set  -- same size, selected by aggregate gate rate
                              across all patients (the "one sparse subset
                              for everyone" baseline)
    (c) foreign gate set  -- same size, another patient's/subgroup's
                              personalized gate set, applied out of group
    (d) random set        -- same size, uniformly random features

For each, fit/score a downstream Cox model restricted to that feature set
and evaluate on the target patient's/subgroup's held-out outcome (e.g. Cox
partial-likelihood contribution, or C-index within-subgroup if the
subgroup is large enough). If personalization is real and not
interchangeable, (a) should beat (b), (c), and (d) more often than chance,
with (b) as the comparison that actually matters -- beating (d) alone is
the weak, already-established result from the existing probe scripts.

This needs no new model architecture or training scheme: it reuses the
same saved gate matrices and holdout data as
`plot_kipan_feature_set_cv_transport.py` and
`patient_subset_within_vs_outside_test.py`.

Usage (once implemented):
    python pd_gate_set_interchangeability_test.py --dataset kipan
    python pd_gate_set_interchangeability_test.py --dataset brca

Status: scaffold only. TODO before this produces a result:
    1. Decide unit of comparison: per-patient (noisier, more samples) vs.
       per-gate-cluster-subgroup (less noisy, fewer samples). Start with
       subgroup-level, matching the granularity `patient_subset_within_vs_
       outside_test.py` already uses successfully.
    2. Reuse `_load_data`/`_combined_arrays`/`_selected_configs` from
       `quick_linear_gated_probe.py` for data loading and gate-matrix
       assembly, and `_cox_score_test_matrix` from
       `plot_kipan_gated_univariate_dotplot.py` for the Cox scoring step,
       rather than reimplementing either.
    3. Build the four comparison feature sets per subgroup, matched for
       size (draw (b)/(c)/(d) at the same cardinality as (a) for that
       subgroup).
    4. Score each feature set on the target subgroup's held-out fold and
       collect paired differences (a)-(b), (a)-(c), (a)-(d) across
       subgroups/folds/reps.
    5. Report a paired test (Wilcoxon signed-rank, matching the existing
       within-vs-outside script's approach) plus effect sizes, not just a
       win-rate percentage.
    6. Run for both KIPAN and BRCA; if results diverge between datasets,
       report both rather than picking the favorable one.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# See patient_subset_within_vs_outside_test.py for why this must be set
# before numpy/scipy/sklearn import (OpenBLAS/OpenMP thread-leak deadlock
# observed with repeated PCA/kNN calls in a long-lived process).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import ANALYSES_DIR, CANONICAL_RUNS, PROCESSED_DATASETS, ensure_repo_imports

ensure_repo_imports(include_analyses_dir=True)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

RUN_DEFAULTS = {
    "kipan": CANONICAL_RUNS["kipan_adaptive_gentle"],
    "brca": CANONICAL_RUNS["brca_adaptive_gentle"],
}
DATA_DEFAULTS = PROCESSED_DATASETS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kipan", "brca"], default="kipan")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument(
        "--comparison-unit",
        choices=["subgroup", "patient"],
        default="subgroup",
        help="Granularity for the own-vs-foreign comparison (see TODO 1).",
    )
    p.add_argument("--n-random-draws", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    raise NotImplementedError(
        "Scaffold only -- see module docstring TODOs for the implementation "
        "plan. Not yet wired to data loading, feature-set construction, or "
        "scoring."
    )


if __name__ == "__main__":
    main()
