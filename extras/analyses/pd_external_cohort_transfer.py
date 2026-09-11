#!/usr/bin/env python3
"""External-cohort transfer of gate-selected feature sets under real
batch/platform shift.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 7). Highest narrative payoff of the seven threads for the
"sparsity generalizes better" claim -- a feature set that transports to a
genuinely independent cohort (different platform, different batch
effects, different population) is a much stronger generalization claim
than anything measurable via cross-validation on one cohort -- but it is
also the thread most gated on data availability. Per
`COLLABORATOR_GUIDE.md` section 6, additional datasets (UK Biobank, etc.)
require formal data-access applications, which is a real timeline
constraint, not something to route around.

Nearest available external target without a new data-access process: the
Desmedt breast cancer microarray cohort, already integrated in the
`sparsedeepsurv` tutorial (`sparsedeepsurv/tutorial.ipynb` and
`docs/source/tutorials/breast_cancer_tutorial.ipynb`) as a BRCA-adjacent
external dataset with its own platform (microarray vs. TCGA RNA-seq) and
population. This is a real, if imperfect, external-validation setting:
different assay technology is exactly the kind of shift that a curated,
overfit-to-one-platform feature set would be expected to fail on.

Plan:

1. Select/train on the in-cohort (BRCA, TCGA) data as usual, using the
   existing canonical adaptive/broad run configs.
2. Freeze the selected feature set (global consensus feature set, since
   Desmedt won't have the same per-patient gate structure available a
   priori -- this thread tests feature-set transfer, not gate-inference
   transfer).
3. Map feature identifiers across platforms (the tutorial already handles
   GPL96 probe-ID-to-HGNC-symbol mapping via mygene -- reuse that mapping
   step rather than reimplementing it).
4. Retrain a downstream Cox model restricted to the frozen, mapped feature
   set on Desmedt (or a held-out split of it), and separately retrain one
   restricted to the dense MLP's full/near-full feature set (or an
   equal-size feature-importance-ranked subset from the dense model, for
   a fair size-matched comparison).
5. Compare C-index (and calibration, if time permits) on Desmedt held-out
   patients between the two transferred models.

Usage (once implemented):
    python pd_external_cohort_transfer.py --source-dataset brca --target-cohort desmedt

Status: scaffold only. TODO before this produces a result:
    1. Confirm probe-ID-to-symbol mapping coverage between the TCGA BRCA
       gene set and the Desmedt GPL96 platform -- some selected genes may
       simply not be measurable on that array, which caps the achievable
       feature-set size and needs to be reported as a limitation, not
       silently worked around.
    2. Decide the "frozen feature set" definition precisely: consensus
       across reps at what gate-rate threshold (reuse the threshold
       convention already used by `plot_kipan_feature_set_cv_transport.py`
       for consistency).
    3. Build the size-matched dense-model comparison set (TODO 4 above) --
       this is the fairness-critical design choice; an unmatched
       comparison (all features vs. a handful) would not isolate the
       claim being tested.
    4. Decide the Desmedt-side train/test split (or whether to do k-fold
       within Desmedt, given its likely much smaller n) for retraining the
       downstream Cox model.
    5. Report the C-index comparison plus, if patient counts allow, a
       bootstrap CI on the difference.
    6. Treat this as exploratory/best-effort: if Desmedt turns out too
       small or too poorly matched in mapped features to produce a
       meaningful comparison, say so plainly and treat UK Biobank (or
       another externally accessible cohort) as the real follow-up,
       pending the data-access process already flagged in
       `COLLABORATOR_GUIDE.md`.
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
    p.add_argument("--source-dataset", choices=["brca"], default="brca")
    p.add_argument("--target-cohort", choices=["desmedt"], default="desmedt")
    p.add_argument("--gate-rate-threshold", type=float, default=0.10)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def map_probe_ids_to_symbols(feature_ids):
    """Map TCGA gene symbols (or vice versa) to the Desmedt GPL96 probe
    namespace. TODO: reuse the mygene-based mapping already implemented in
    sparsedeepsurv's tutorial rather than reimplementing it here -- import
    or factor out that step, per module docstring TODO 1.
    """
    raise NotImplementedError("Scaffold only -- see module docstring TODOs.")


def main() -> None:
    args = _parse_args()
    raise NotImplementedError(
        "Scaffold only -- see module docstring TODOs for the implementation "
        "plan. Not yet wired to feature-set freezing, probe mapping, or "
        "downstream retraining/evaluation on Desmedt."
    )


if __name__ == "__main__":
    main()
