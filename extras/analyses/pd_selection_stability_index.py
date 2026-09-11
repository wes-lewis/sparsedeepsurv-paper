#!/usr/bin/env python3
"""Corrected-for-chance feature-selection stability index, LSPIN/Concrete
vs. sparse linear baselines.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 2).

The current internal framing of run-to-run reproducibility
(`COLLABORATOR_GUIDE.md` section 6) is informal: "overlap is greater than
random chance, but not strongly reproducible at the individual-gene
level." That is a defensible hedge but not a quantified result, and it
invites the reviewer question "compared to what, exactly, and by how
much?"

The feature-selection literature has standard answers to that question:

- Kuncheva's consistency index (Kuncheva, 2007) -- corrected-for-chance
  pairwise overlap between two same-size selected sets.
- Nogueira & Brown's stability estimator (Nogueira & Brown, 2016,
  "Measuring the Stability of Feature Selection") -- a variance-based
  stability measure across an arbitrary number of selection runs, with a
  closed-form asymptotic confidence interval, which is the more current
  standard and handles unequal selected-set sizes directly (relevant here
  since hard-selected set size can vary run to run).

Plan: compute the Nogueira-Brown stability index (with its CI) across the
already-existing repeated/bootstrap gate-matrix runs, separately for
LSPIN, Concrete, and (as the "one global sparse subset" comparison
baselines) Lasso-Cox and elastic-net-Cox fit with matched sparsity level.
Report per dataset (KIPAN, BRCA) and, if the biological-validation thread
finds meaningful subgroups, per subgroup as well -- stability may be much
higher within a molecularly homogeneous subgroup than in aggregate, which
would itself be a useful result to report.

This does not require new training runs for the gated models -- reuse the
existing repeated-run gate matrices from the canonical adaptive/broad run
directories. It does require fitting the Lasso-Cox/elastic-net-Cox
baselines across the same resampling scheme, which is new but cheap
(`sksurv.linear_model.CoxnetSurvivalAnalysis` is already an available
dependency per the broad-sweep comparisons).

Usage (once implemented):
    python pd_selection_stability_index.py --dataset kipan --gate-family lspin
    python pd_selection_stability_index.py --dataset kipan --gate-family concrete
    python pd_selection_stability_index.py --dataset kipan --baseline lasso-cox

Status: scaffold only. TODO before this produces a result:
    1. Implement `nogueira_brown_stability(selection_matrix)` (or vendor a
       small, tested implementation -- this is a well-defined ~20-line
       computation over a binary [n_runs x n_features] selection matrix,
       not model-specific) with its analytic confidence interval.
    2. Assemble the [n_runs x n_features] hard-selection matrix for each
       gate family from the existing canonical run directories (same
       source gate matrices already used by
       `render_adaptive_manuscript_figures.py`'s heatmaps).
    3. Fit Lasso-Cox and elastic-net-Cox baselines across the same
       resampling scheme, with sparsity (selected-set size) matched to the
       gated models' average hard-selected count, so stability is compared
       at equal sparsity rather than confounded by set size.
    4. Report stability index + CI per (dataset, method), plus a
       permutation null (stability of a matched-size *random* selection)
       so "above chance" has a number attached instead of being asserted.
    5. If time permits, repeat within gate-cluster subgroups (see
       biological-validation thread) to test whether stability is higher
       at that granularity.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
    p.add_argument(
        "--gate-family",
        choices=["lspin", "concrete"],
        default="lspin",
    )
    p.add_argument(
        "--baseline",
        choices=["lasso-cox", "elastic-net-cox", "none"],
        default="none",
        help="Sparse linear baseline to compute the same stability index for.",
    )
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--n-permutation-null", type=int, default=1000)
    return p.parse_args()


def nogueira_brown_stability(selection_matrix) -> tuple[float, tuple[float, float]]:
    """Compute the Nogueira & Brown (2016) stability index and its CI.

    Parameters
    ----------
    selection_matrix : array-like, shape (n_runs, n_features)
        Binary indicator of whether each feature was hard-selected in each
        run.

    Returns
    -------
    (stability, (ci_low, ci_high))

    TODO: implement. Reference: Nogueira, Sechidis & Brown, "On the
    Stability of Feature Selection Algorithms", JMLR 2018 (extends the
    2016 workshop version cited above) -- Theorem 1 gives the estimator
    and its asymptotic variance in closed form.
    """
    raise NotImplementedError("Scaffold only -- see module docstring TODOs.")


def main() -> None:
    args = _parse_args()
    raise NotImplementedError(
        "Scaffold only -- see module docstring TODOs for the implementation "
        "plan. Not yet wired to gate-matrix loading or baseline fitting."
    )


if __name__ == "__main__":
    main()
