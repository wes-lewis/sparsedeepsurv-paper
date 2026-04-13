#!/usr/bin/env python3
"""
KIPAN targeted analysis — paper-repo version.

Changes vs original (run_kipan_patient_smoothing_notebook_style_paper.py):
  - knn_k default = 8
  - pick_comparison_configs: manifold_gain removed from pair scoring (circular: affinity-derived
    metric used to select configs, then same metric reported as a finding)
  - pick_heatmap_configs: aff_gain, cluster_gain, manifold_gain all removed from scoring
  - Results written to sparsedeepsurv-paper/data/runs/

Usage:
  conda activate musevo
  cd /banach2/wes/lspin-repos/sparsedeepsurv-paper
  python analyses/ch3_kipan_targeted.py [--device cuda:2] [--results-dir data/runs/ch3_kipan_targeted_knn8]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
LSPIN_ROOT = Path("/banach2/wes/lspin-pytorch")
CLEANED = LSPIN_ROOT / "cleaned_analyses"
PAPER_ROOT = Path(__file__).resolve().parents[1]
KIPAN_DATA_DEFAULT = LSPIN_ROOT / "runs" / "kipan_20260209_213604"
RESULTS_DEFAULT = PAPER_ROOT / "data" / "runs" / "ch3_kipan_targeted_knn8"

for _p in [str(LSPIN_ROOT), str(CLEANED)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Import original script (before monkey-patching) ───────────────────────────
import run_kipan_patient_smoothing_notebook_style_paper as _orig  # noqa: E402


# ── Fixed selection functions (circularity removed) ───────────────────────────

def _pick_comparison_configs(
    acc: "pd.DataFrame",
    *,
    family: str,
    x_min: float,
    x_max: float,
) -> Dict[str, Dict[str, float]]:
    """
    Select the best no-smooth / smooth config pair for a model family.

    Scoring uses only sparsity-concentration gain (khard/union ratio) and C-index.
    The original included manifold_gain (a reported outcome metric) — removed here.
    """
    import pandas as pd  # local import to avoid top-level CUDA init

    fam = acc[acc["model_family"] == family].copy()
    fam = fam[(fam["mean_Khard"] >= x_min) & (fam["mean_Khard"] <= min(x_max, 1400.0))].copy()
    if fam.empty:
        raise ValueError(f"No configs for family={family} in Khard range [{x_min}, {min(x_max, 1400.0)}]")

    fam["khard_over_union"] = fam["mean_Khard"] / fam["mean_gene_union_count"].clip(lower=1e-8)
    no_df = fam[fam["lambda_sample_smooth"] == 0.0].copy()
    sm_df = fam[fam["lambda_sample_smooth"] > 0.0].copy()

    if no_df.empty or sm_df.empty:
        raise ValueError(f"Need both no-smooth and smooth configs for family={family}")

    best = None
    best_score = -1e18
    for _, sm in sm_df.iterrows():
        target = float(sm["mean_Khard"])
        _no_df = no_df.copy()
        _no_df["match_score"] = np.abs(
            np.log10(_no_df["mean_Khard"].clip(lower=1e-6)) - np.log10(max(target, 1e-6))
        )
        no = _no_df.sort_values(["match_score", "mean_test_c"], ascending=[True, False]).iloc[0]
        c_drop = float(no["mean_test_c"] - sm["mean_test_c"])
        if c_drop > 0.03:
            continue
        ratio_gain = float(sm["khard_over_union"] - no["khard_over_union"])
        khard_match_penalty = float(
            abs(sm["mean_Khard"] - no["mean_Khard"]) / max(no["mean_Khard"], 1e-8)
        )
        # Fixed: no manifold_gain term
        score = (
            2.0 * ratio_gain
            + 1.0 * float(sm["mean_test_c"] - no["mean_test_c"])
            - 0.4 * khard_match_penalty
        )
        if score > best_score:
            best_score = score
            best = (no, sm, ratio_gain, c_drop, khard_match_penalty)

    if best is None:
        no_best = no_df.sort_values(["khard_over_union", "mean_test_c"], ascending=[False, False]).iloc[0]
        sm_best = sm_df.sort_values(["khard_over_union", "mean_test_c"], ascending=[False, False]).iloc[0]
        return {"nosmooth": no_best.to_dict(), "smooth": sm_best.to_dict()}

    no, sm, ratio_gain, c_drop, khard_match_penalty = best
    no_row = no.to_dict()
    sm_row = sm.to_dict()
    no_row.update({"pair_score": float(best_score), "ratio_gain_vs_pair": 0.0, "cindex_drop_vs_pair": 0.0, "khard_match_penalty": 0.0})
    sm_row.update({"pair_score": float(best_score), "ratio_gain_vs_pair": float(ratio_gain), "cindex_drop_vs_pair": float(c_drop), "khard_match_penalty": float(khard_match_penalty)})
    return {"nosmooth": no_row, "smooth": sm_row}


def _pick_heatmap_configs(
    acc: "pd.DataFrame",
    *,
    max_cindex_drop: float,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Select best configs for heatmap visualisation.

    Scoring uses only sparsity-concentration metrics.
    The original included aff_gain, cluster_gain, manifold_gain — all removed here.
    """
    selected: Dict = {}
    for family in ["LSPIN", "Concrete"]:
        fam = acc[(acc["model_family"] == family) & (acc["mean_Khard"] < 1400.0)].copy()
        fam["khard_over_union"] = fam["mean_Khard"] / fam["mean_gene_union_count"].clip(lower=1e-8)
        no_df = fam[np.isclose(fam["lambda_sample_smooth"], 0.0)].copy()
        sm_df = fam[fam["lambda_sample_smooth"] > 0.0].copy()
        if no_df.empty or sm_df.empty:
            raise ValueError(f"Need both no-smooth and smooth configs for family={family}")

        best = None
        best_score = -1e18
        for _, sm in sm_df.iterrows():
            nearest_idx = (no_df["mean_Khard"] - sm["mean_Khard"]).abs().idxmin()
            no = no_df.loc[nearest_idx]
            c_drop = float(no["mean_test_c"] - sm["mean_test_c"])
            if c_drop > max_cindex_drop:
                continue
            ratio_gain = float(sm["khard_over_union"] - no["khard_over_union"])
            pct_union_drop = float(
                (no["mean_gene_union_count"] - sm["mean_gene_union_count"])
                / max(no["mean_gene_union_count"], 1e-8)
            )
            pct_khard_change = float(
                abs(sm["mean_Khard"] - no["mean_Khard"]) / max(no["mean_Khard"], 1e-8)
            )
            # Fixed: no aff_gain, cluster_gain, manifold_gain
            score = 2.5 * ratio_gain + 0.75 * pct_union_drop - 0.5 * pct_khard_change
            if score > best_score:
                best_score = score
                best = (no, sm, ratio_gain, pct_union_drop, pct_khard_change, c_drop)

        if best is None:
            no = no_df.sort_values(["khard_over_union", "mean_test_c"], ascending=[False, False]).iloc[0]
            sm = sm_df.sort_values(["khard_over_union", "mean_test_c"], ascending=[False, False]).iloc[0]
            best = (no, sm, np.nan, np.nan, np.nan, float(no["mean_test_c"] - sm["mean_test_c"]))

        no, sm, ratio_gain, pct_union_drop, pct_khard_change, c_drop = best
        no_row = no.to_dict()
        sm_row = sm.to_dict()
        no_row.update({
            "pair_score": float(best_score), "ratio_gain_vs_pair": 0.0,
            "pct_union_drop_vs_pair": 0.0, "pct_khard_change_vs_pair": 0.0, "cindex_drop_vs_pair": 0.0,
        })
        sm_row.update({
            "pair_score": float(best_score), "ratio_gain_vs_pair": float(ratio_gain),
            "pct_union_drop_vs_pair": float(pct_union_drop), "pct_khard_change_vs_pair": float(pct_khard_change),
            "cindex_drop_vs_pair": float(c_drop),
        })
        selected[family] = {"nosmooth": no_row, "smooth": sm_row}
    return selected


# ── Monkey-patch before main() runs ───────────────────────────────────────────
_orig.pick_comparison_configs = _pick_comparison_configs
_orig.pick_heatmap_configs = _pick_heatmap_configs


# ── Entry point ───────────────────────────────────────────────────────────────
def _parse():
    p = argparse.ArgumentParser(description="KIPAN targeted analysis (paper repo, knn_k=8, no circular selection)")
    p.add_argument("--outdir", type=Path, default=KIPAN_DATA_DEFAULT)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--n-reps", type=int, default=12)
    return p.parse_known_args()


if __name__ == "__main__":
    args, extra = _parse()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Original script's argparse only accepts "auto"|"cpu"|"cuda" — not "cuda:N".
    # To pin a specific GPU, set CUDA_VISIBLE_DEVICES before running and pass --device cuda.
    device_arg = args.device
    if device_arg.startswith("cuda:"):
        gpu_idx = device_arg.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
        device_arg = "cuda"
        print(f"[ch3_kipan_targeted] set CUDA_VISIBLE_DEVICES={gpu_idx}")

    sys.argv = [
        "run_kipan_patient_smoothing_notebook_style_paper.py",
        "--outdir", str(args.outdir.resolve()),
        "--results-dir", str(args.results_dir.resolve()),
        "--device", device_arg,
        "--knn-k", str(args.knn_k),
        "--n-reps", str(args.n_reps),
    ] + extra

    print(f"[ch3_kipan_targeted] results → {args.results_dir.resolve()}")
    print(f"[ch3_kipan_targeted] device  = {args.device}, knn_k = {args.knn_k}")
    print(f"[ch3_kipan_targeted] Selection: pick_comparison_configs and pick_heatmap_configs "
          f"patched — affinity metrics removed from scoring")
    _orig.main()
