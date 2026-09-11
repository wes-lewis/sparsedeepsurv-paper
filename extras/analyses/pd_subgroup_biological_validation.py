#!/usr/bin/env python3
"""Biological and prognostic validation of gate-derived patient subgroups.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 3).

Existing work already goes partway here:

- `render_adaptive_manuscript_figures.py` clusters patients by hard-gate
  vector (average-linkage, cosine distance) for the heatmap figures.
- `histology_runs_test.py` shows that clustering-induced patient order
  recovers histological structure better than chance (KIPAN), via a
  permutation test on run counts.
- `quick_linear_gated_probe.py` / `plot_pancan_gated_univariate_bias.py`
  look at per-gene signal, but not at subgroup-level biological coherence.

What's missing is the pair of results that would make "personalized gating
finds real patient subgroups" a positive, freestanding claim rather than
an inference from histology-order alignment:

1. **Survival separation beyond clinical covariates.** Do gate-cluster
   subgroups have significantly different survival (log-rank test) after
   accounting for what stage/histology alone would predict? This is the
   test that shows gating adds prognostic information, not just that it
   correlates with a label we already had.
2. **Subtype-appropriate pathway enrichment.** Do different subgroups'
   selected-gene sets enrich for different, biologically plausible
   Hallmark pathways (e.g., distinct pathways for histological/molecular
   subtypes known to have different biology)? The Hallmark gene sets are
   already tracked at
   `data/gene_sets/enrichr/MSigDB_Hallmark_2020.gmt`.

Both are reanalyses of existing canonical run outputs
(`extras/data/runs/adaptive/`) -- no new model training required.

Usage (once implemented):
    python pd_subgroup_biological_validation.py --dataset kipan --analysis survival
    python pd_subgroup_biological_validation.py --dataset kipan --analysis enrichment
    python pd_subgroup_biological_validation.py --dataset brca --analysis enrichment

Status: scaffold only. TODO before this produces a result:
    1. Reuse the same hard-gate clustering used for
       `heatmap_sample_order_*.csv` (see `histology_runs_test.py` for how
       those are read) to assign a discrete subgroup label per patient,
       for a chosen k (start with the k implied by the existing heatmap
       dendrogram cut, then check sensitivity to k).
    2. Pull matched clinical covariates (stage, histology, age -- whatever
       is available in the processed dataset) and fit a Cox model with
       subgroup indicator + clinical covariates; test whether the
       subgroup term is significant, and separately run an
       unadjusted log-rank test across subgroups for a plot.
    3. For enrichment: pool each subgroup's hard-selected genes (across
       reps), run a hypergeometric/Fisher enrichment test against
       `MSigDB_Hallmark_2020.gmt` per subgroup, and check whether the
       top enriched pathways differ meaningfully (and plausibly) between
       subgroups rather than all subgroups converging on the same
       generic pathways.
    4. If KIPAN histology labels make this the natural first dataset
       (per `histology_runs_test.py` already being KIPAN-only), start
       there, then attempt the BRCA analog since BRCA lacks the same
       ready label but has PAM50-style subtype information if available
       in the processed clinical data.
    5. Report both a positive framing (subgroups found) and, honestly, a
       null framing if enrichment doesn't differentiate subgroups --
       this thread is diagnostic, not guaranteed to succeed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_RUNS_DIR, GENE_SETS_DIR, PROCESSED_DATASETS, ensure_repo_imports

ensure_repo_imports(include_analyses_dir=True)

HALLMARK_GENE_SETS = GENE_SETS_DIR / "enrichr" / "MSigDB_Hallmark_2020.gmt"

KIPAN_HEATMAP_DIR = (
    DATA_RUNS_DIR / "adaptive_gentle_all_kipan_brca_pancan_20260408_193020" / "kipan"
)
BRCA_HEATMAP_DIR = (
    DATA_RUNS_DIR / "adaptive_gentle_all_kipan_brca_pancan_20260408_193020" / "brca"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kipan", "brca"], default="kipan")
    p.add_argument(
        "--analysis",
        choices=["survival", "enrichment", "both"],
        default="both",
    )
    p.add_argument("--n-clusters", type=int, default=None, help="Override dendrogram cut k.")
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    raise NotImplementedError(
        "Scaffold only -- see module docstring TODOs for the implementation "
        "plan. Not yet wired to subgroup assignment, clinical covariates, "
        "or Hallmark enrichment testing."
    )


if __name__ == "__main__":
    main()
