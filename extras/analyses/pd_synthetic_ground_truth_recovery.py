#!/usr/bin/env python3
"""Semi-synthetic ground-truth recovery test for personalized feature
selection.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 5). This is the heaviest-lift thread in the plan, but also the
single most convincing one available: every major instance-wise feature
selection method (INVASE, Yoon et al. 2019; L2X, Chen et al. 2018; the
original LSPIN paper) validates against a semi-synthetic setup with known
ground truth precisely because held-out predictive accuracy alone cannot
distinguish "found the right personalized subsets" from "found one
adequate global subset that happens to score well."

Design:

1. Use real KIPAN/BRCA covariates (X) to preserve realistic covariance
   structure, rather than fully synthetic features -- this keeps the
   selection problem's difficulty realistic (correlated blocks, dropout
   patterns, etc.) instead of the artificially easy independent-feature
   settings some published benchmarks use.
2. Partition patients into synthetic subgroups (reuse real subgroup
   structure if available from the biological-validation thread, or
   assign synthetic subgroups directly for a cleaner ground truth) and
   assign each subgroup a distinct small true-relevant-feature subset
   (e.g. 5-10 real genes per subgroup, disjoint or partially overlapping
   across subgroups).
3. Generate synthetic survival times from a Cox/AFT hazard driven only by
   each patient's subgroup-specific true feature subset (plus censoring
   at a realistic rate matched to the real dataset), so the ground truth
   "which features actually matter for this patient" is known exactly.
4. Fit LSPIN/Concrete gated models and, as comparison baselines, Lasso-Cox
   and a single global sparse MLP (STG) on the synthetic outcome using the
   real covariates.
5. Score each method by precision/recall/Jaccard of
   selected-vs-true-relevant features, computed *per patient/subgroup*
   (not just in aggregate) -- this is the number a global sparse model
   cannot win on by construction, since it can only ever select one
   feature set for everyone.

This is the headline plot: personalized precision/recall vs. subgroup,
gated methods vs. global-sparse baselines, with the ground truth as the
reference the paper currently lacks entirely.

Usage (once implemented):
    python pd_synthetic_ground_truth_recovery.py --dataset kipan --n-subgroups 4
    python pd_synthetic_ground_truth_recovery.py --dataset brca --n-subgroups 4

Status: scaffold only, and this needs a design pass before implementation
starts (see TODO 1-2 below) -- do not just start coding the hazard
generator without settling those first, since the difficulty/realism of
the synthetic setup determines whether this experiment is convincing or
dismissable as "too easy."

TODO:
    1. Design pass: decide subgroup count/size, true-feature-subset size
       and overlap-across-subgroups policy, hazard functional form (linear
       Cox vs. mild nonlinearity), effect size (how strongly should the
       true features actually drive the hazard -- too weak and nothing
       recovers anything, too strong and every method trivially wins),
       and censoring rate (match the real dataset's censoring rate).
    2. Decide whether synthetic subgroups should be real biological
       subgroups (from thread 3, `pd_subgroup_biological_validation.py`)
       relabeled with a synthetic hazard, or freshly assigned synthetic
       partitions. Real subgroups make the covariate realism argument
       stronger; freshly assigned ones make the ground truth cleaner and
       let you control subgroup separability directly. Prefer starting
       with freshly assigned partitions for a clean first result, and
       revisit real subgroups as a robustness check.
    3. Implement `generate_synthetic_hazard(X, subgroup_labels,
       true_feature_map, effect_size, censoring_rate)` returning
       (time, event) pairs.
    4. Implement `score_recovery(selected_features_per_patient,
       true_feature_map, subgroup_labels)` -> per-subgroup and aggregate
       precision/recall/Jaccard, with bootstrap CIs.
    5. Wire up LSPIN/Concrete training on the synthetic outcome (reuse
       `sparsedeepsurv.DeepSurvGated` / `train_gated_deepsurv` directly),
       plus Lasso-Cox and a single global-sparse MLP (STG) baseline fit on
       the same synthetic outcome.
    6. Sweep effect size and subgroup separability at least coarsely (2-3
       settings) to show the result isn't a knife-edge artifact of one
       chosen difficulty level.
    7. Report both the headline precision/recall/Jaccard plot and, as a
       sanity check, that all methods achieve comparable held-out C-index
       on the synthetic outcome -- the point is that comparable predictive
       accuracy can still hide very different feature-recovery quality,
       which is exactly the gap this experiment is meant to expose.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import PROCESSED_DATASETS, ensure_repo_imports

ensure_repo_imports()

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

DATA_DEFAULTS = PROCESSED_DATASETS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kipan", "brca"], default="kipan")
    p.add_argument("--n-subgroups", type=int, default=4)
    p.add_argument("--true-features-per-subgroup", type=int, default=8)
    p.add_argument(
        "--subgroup-source",
        choices=["synthetic", "biological"],
        default="synthetic",
        help="See TODO 2 in the module docstring.",
    )
    p.add_argument("--effect-size", type=float, default=1.0)
    p.add_argument("--censoring-rate", type=float, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def generate_synthetic_hazard(
    X, subgroup_labels, true_feature_map, effect_size, censoring_rate
):
    """Generate synthetic (time, event) pairs from subgroup-specific true
    feature subsets. TODO: implement per module docstring TODO 3."""
    raise NotImplementedError("Scaffold only -- see module docstring TODOs.")


def score_recovery(selected_features_per_patient, true_feature_map, subgroup_labels):
    """Precision/recall/Jaccard of selected-vs-true features, per subgroup
    and in aggregate. TODO: implement per module docstring TODO 4."""
    raise NotImplementedError("Scaffold only -- see module docstring TODOs.")


def main() -> None:
    args = _parse_args()
    raise NotImplementedError(
        "Scaffold only -- see module docstring TODOs for the implementation "
        "plan, and do the design pass (TODO 1-2) before writing the hazard "
        "generator."
    )


if __name__ == "__main__":
    main()
