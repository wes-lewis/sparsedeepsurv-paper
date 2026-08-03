#!/usr/bin/env python3
"""
Quantify histology alignment of the KIPAN feature-selection heatmaps.

The heatmaps in render_adaptive_manuscript_figures.py order patients by
average-linkage hierarchical clustering on cosine distance between hard-gate
vectors (repro_survival_pipeline.save_fig_gate_heatmap_with_histo), without
reference to histology. This script tests whether that label-blind ordering
recovers histological structure: it counts the number of maximal runs of
consecutive same-histology patients along the clustering-induced order
(heatmap_sample_order_*.csv) and compares that count to its distribution
under random permutation of patient labels, holding group sizes fixed.

Usage:
    python histology_runs_test.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_RUNS_DIR

KIPAN_HEATMAP_DIR = (
    DATA_RUNS_DIR / "adaptive_gentle_all_kipan_brca_pancan_20260408_193020" / "kipan"
)

SAMPLE_ORDER_FILES = [
    "heatmap_sample_order_lspin_nosmooth.csv",
    "heatmap_sample_order_lspin_smooth.csv",
    "heatmap_sample_order_concrete_nosmooth.csv",
    "heatmap_sample_order_concrete_smooth.csv",
    "heatmap_sample_order_llspin_nosmooth.csv",
    "heatmap_sample_order_llspin_smooth.csv",
    "heatmap_sample_order_lconcrete_nosmooth.csv",
    "heatmap_sample_order_lconcrete_smooth.csv",
]

N_PERMUTATIONS = 100_000
SEED = 0


def count_runs(labels: np.ndarray) -> int:
    return sum(1 for _ in itertools.groupby(labels))


def expected_runs(labels: np.ndarray) -> float:
    n = len(labels)
    _, counts = np.unique(labels, return_counts=True)
    s = sum(c * (c - 1) for c in counts)
    return 1 + (n - 1) * (1 - s / (n * (n - 1)))


def permutation_test(labels: np.ndarray, observed: int, rng: np.random.Generator, n_perm: int) -> tuple[float, float, float]:
    lab = labels.copy()
    perm_runs = np.empty(n_perm, dtype=int)
    for i in range(n_perm):
        rng.shuffle(lab)
        perm_runs[i] = count_runs(lab)
    p = (np.sum(perm_runs <= observed) + 1) / (n_perm + 1)
    return perm_runs.mean(), perm_runs.std(), p


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for fname in SAMPLE_ORDER_FILES:
        path = KIPAN_HEATMAP_DIR / fname
        df = pd.read_csv(path)
        labels = df["histo_group"].to_numpy()
        observed = count_runs(labels)
        expected = expected_runs(labels)
        perm_mean, perm_sd, p = permutation_test(labels, observed, rng, N_PERMUTATIONS)
        rows.append(
            {
                "config": fname.removeprefix("heatmap_sample_order_").removesuffix(".csv"),
                "n_patients": len(labels),
                "observed_runs": observed,
                "expected_runs_random": round(expected, 1),
                "permutation_mean": round(perm_mean, 1),
                "permutation_sd": round(perm_sd, 1),
                "p_value": p,
            }
        )
        print(
            f"{fname:45s} n={len(labels)} observed={observed:3d} "
            f"expected_random={expected:.1f} perm_mean={perm_mean:.1f} "
            f"perm_sd={perm_sd:.1f} p<= {p:.2e}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    out_path = KIPAN_HEATMAP_DIR / "histology_runs_test_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
