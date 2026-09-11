#!/usr/bin/env python3
"""Training-set-size learning curves: does sparsity's generalization
advantage show up in the n<<p regime this data actually lives in?

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 6).

Genomic survival data (KIPAN, BRCA gene expression) is exactly the
high-dimensional, sample-limited setting where regularized/sparse models
are traditionally argued to have an edge over dense models (this is the
core motivation cited in the Cox-nnet and DeepSurv-era survival-DL
literature). The current results don't test this directly -- they compare
methods at the full available training-set size only, which is the
setting *least* likely to show a sparsity advantage (more data narrows
the gap between sparse and dense models).

Plan: subsample training folds at multiple fractions (e.g. 25/50/75/100%
of the available training patients, holding the same held-out test fold
fixed across fractions) and, at each fraction, compute:

    - held-out C-index for gated-sparse (LSPIN, Concrete, MLP+STG) vs.
      dense MLP vs. Lasso-Cox
    - the generalization gap: train C-index minus held-out C-index, which
      more directly isolates overfitting than held-out C-index alone

Report both as curves over training fraction. The claim this experiment
targets is specifically about the *shape* of these curves: if sparsity's
generalization advantage is real, the gated models' generalization gap
should grow more slowly as training data shrinks than the dense MLP's,
even if absolute held-out C-index is comparable or slightly lower at full
sample size (consistent with the existing results).

Reuses the same processed train/test splits and known-good lambda
multipliers as `pd_noise_robustness_sweep.py` (thread 4) -- no new
architecture, just a subsampling loop around existing training code.

Usage (once implemented):
    python pd_small_sample_learning_curves.py --dataset kipan --fractions 0.25 0.5 0.75 1.0
    python pd_small_sample_learning_curves.py --dataset brca --fractions 0.25 0.5 0.75 1.0

Status: scaffold only. TODO before this produces a result:
    1. Implement stratified subsampling of the training fold at each
       fraction (stratify on event indicator at minimum, to avoid
       pathologically low event counts at small fractions -- flag if any
       fraction leaves too few events for a stable Cox fit).
    2. Reuse the known-good lambda multipliers from
       `PERFORMANCE_RECOVERY_ANALYSIS.md` / `KNOWN_GOOD_LAMBDA_MULTIPLIERS`
       in `pd_noise_robustness_sweep.py` rather than re-tuning lambda at
       every fraction (same rationale as thread 4 -- avoids confounding
       the comparison and blowing up compute). Flag as a limitation if
       optimal lambda plausibly shifts with training set size.
    3. Run enough reps per (dataset, model, fraction) for a usable CI,
       since small-fraction runs will be noisier.
    4. Compute both held-out C-index and the train-minus-held-out gap at
       each fraction; plot both curves per dataset, one line per model.
    5. If the dense MLP's curve doesn't actually degrade faster at small
       fractions, report that honestly -- it's an informative null result
       either way (e.g., it would suggest the sparsity benefit shows up
       only via noise-robustness, not sample efficiency, which narrows
       thread 4's framing rather than adding a redundant win).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import CANONICAL_RUNS, PROCESSED_DATASETS, ensure_repo_imports

ensure_repo_imports()

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

DATA_DEFAULTS = PROCESSED_DATASETS

# Same known-good lambda multipliers used by pd_noise_robustness_sweep.py
# (thread 4) -- kept in sync manually for now; consider factoring into a
# shared constants module if both threads land.
KNOWN_GOOD_LAMBDA_MULTIPLIERS = {
    "kipan": {"lspin": 2.0, "concrete": 2.0, "mlp_stg": 1.0},
    "brca": {"lspin": 0.5, "concrete": 2.0, "mlp_stg": 0.5},
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kipan", "brca"], default="kipan")
    p.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["mlp_dense", "lasso_cox", "lspin", "concrete", "mlp_stg"],
    )
    p.add_argument("--n-reps", type=int, default=10)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def subsample_training_fold(X_train, t_train, e_train, fraction: float, seed: int):
    """Stratified (on event indicator) subsample of the training fold.

    TODO: implement per module docstring TODO 1, including a minimum-
    event-count guard that raises or warns if a fraction leaves too few
    events for a stable fit.
    """
    raise NotImplementedError("Scaffold only -- see module docstring TODOs.")


def main() -> None:
    args = _parse_args()
    raise NotImplementedError(
        "Scaffold only -- see module docstring TODOs for the implementation "
        "plan. Not yet wired to subsampling, training, or curve plotting."
    )


if __name__ == "__main__":
    main()
