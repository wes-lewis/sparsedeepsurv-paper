#!/usr/bin/env python3
"""Held-out C-index degradation under injected noise features: gated-sparse
vs. dense MLP.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 4).

Current framing of sparsity's value is "doesn't cost much predictive
performance" (see `extras/archive/docs/SUMMARY_KEY_FINDINGS.md`) -- sparse
gated models land at or below the dense MLP reference on both KIPAN and
BRCA C-index. That is a weak sell on its own. Injected-noise-feature
robustness is the standard experiment in the instance-wise-selection
literature (INVASE, LSPIN, Concrete Autoencoder papers) for turning
"sparsity doesn't hurt" into "sparsity actively helps once the feature
space is not curated," which is the realistic genomic-panel-design
setting this method is pitched at.

Plan: append k synthetic noise columns (drawn to mimic the marginal
distribution of the real expression features, e.g. permuted real columns
or matched-moment Gaussian/log-normal noise -- permuted real columns are
preferable since they preserve realistic marginal scale without adding
signal) to the existing KIPAN/BRCA feature matrices, sweep k across an
order-of-magnitude range, and compare held-out C-index degradation of:

    - dense MLP (unregularized reference)
    - gated-sparse models (LSPIN, Concrete, at their existing recovered
      lambda multipliers -- see `PERFORMANCE_RECOVERY_ANALYSIS.md` for the
      known-good per-dataset configs)
    - MLP+STG (already in the existing model family lineup)

against a k=0 baseline. If sparsity's generalization advantage is real,
the gated models' degradation curve should be visibly flatter than the
dense MLP's as k grows, even though the gated models don't necessarily
win outright at k=0.

This requires new training runs (one run per (dataset, model, k)) but no
new architecture -- it reuses `sparsedeepsurv.DeepSurvGated` and the
existing dense MLP baseline code as-is, with an added noise-column
augmentation step ahead of training.

Usage (once implemented):
    python pd_noise_robustness_sweep.py --dataset kipan --noise-counts 0 50 200 1000
    python pd_noise_robustness_sweep.py --dataset brca --noise-counts 0 50 200 1000

Status: scaffold only. TODO before this produces a result:
    1. Implement `augment_with_noise_columns(X, k, mode)` with at least
       `mode="permuted_real"` (randomly permute k real columns
       independently across patients, breaking their association with
       outcome while preserving marginal distribution) as the primary
       mode; keep `mode="gaussian_matched"` as a secondary sanity check.
    2. Reuse the existing per-dataset train/val/test split artifacts under
       `data/processed/` rather than re-splitting, so noise-augmented runs
       are directly comparable to the existing canonical broad/adaptive
       results at k=0.
    3. Reuse the known-good lambda multipliers already established in
       `PERFORMANCE_RECOVERY_ANALYSIS.md` for LSPIN/Concrete/MLP+STG per
       dataset, rather than re-sweeping lambda at every noise level
       (that would confound the comparison and blow up compute).
    4. Run enough folds/reps per (dataset, model, k) to get a usable CI on
       the C-index degradation, consistent with the existing validation
       run replicate counts (see `validate_models.py`).
    5. Plot C-index (y) vs. noise count k (x), one line per model, one
       panel per dataset -- the flatness of the gated-model lines
       relative to the dense MLP line is the headline figure.
    6. If gated models also degrade substantially at high k, report that
       honestly and note where the noise tolerance limit sits, rather
       than only reporting favorable k ranges.
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

# Known-good lambda multipliers per dataset, carried over from the v4/v5
# recovery sweeps documented in PERFORMANCE_RECOVERY_ANALYSIS.md, so this
# thread doesn't have to re-tune lambda at every noise level.
KNOWN_GOOD_LAMBDA_MULTIPLIERS = {
    "kipan": {"lspin": 2.0, "concrete": 2.0, "mlp_stg": 1.0},
    "brca": {"lspin": 0.5, "concrete": 2.0, "mlp_stg": 0.5},
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kipan", "brca"], default="kipan")
    p.add_argument(
        "--noise-counts",
        type=int,
        nargs="+",
        default=[0, 50, 200, 1000],
    )
    p.add_argument(
        "--noise-mode",
        choices=["permuted_real", "gaussian_matched"],
        default="permuted_real",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=["mlp_dense", "lspin", "concrete", "mlp_stg"],
    )
    p.add_argument("--n-reps", type=int, default=5)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def augment_with_noise_columns(X, k: int, mode: str = "permuted_real"):
    """Append k synthetic noise columns to feature matrix X.

    TODO: implement both modes described in the module docstring. Must
    return an array with the same number of rows as X and k additional
    columns, and must be applied identically (same permutation/noise draw)
    to train/val/test splits of the same run so the noise columns don't
    leak information about the split.
    """
    raise NotImplementedError("Scaffold only -- see module docstring TODOs.")


def main() -> None:
    args = _parse_args()
    raise NotImplementedError(
        "Scaffold only -- see module docstring TODOs for the implementation "
        "plan. Not yet wired to training, noise augmentation, or plotting."
    )


if __name__ == "__main__":
    main()
