#!/usr/bin/env python3
"""
Chapter 3 BRCA adaptive sweep v1.

Two-phase adaptive design (identical structure to ch3_kipan_adaptive_v1):
  Phase 1: all configs × n_phase1_reps (default 3) — screening.
  Phase 2: top_n_promising configs per (family, sigma) × remaining reps.

BRCA specifics vs KIPAN:
  - Data:     tcga_brca20260214_001423 (723 train / 311 test, 24000 genes)
  - knn_k:    5 (vs 8 for KIPAN)
  - sigmas:   LSPIN [0.10, 0.15], Concrete [0.10, 0.15]
  - lambdas:  wider range calibrated to Khard ~200-1000 for BRCA
  - smooth:   [0.0, 0.1, 0.2, 0.4] (BRCA convention — higher smoothing useful)
  - khard band: [200, 900]
  - Grid:     6λ × 2σ × 4smooth × 2families = 96 configs phase 1

Usage:
  conda activate musevo
  cd /banach2/wes/lspin-repos/sparsedeepsurv-paper
  python analyses/ch3_brca_adaptive_v1.py [--gpus 0 2 6 7]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# ── Reuse all core logic from the KIPAN adaptive script ───────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ch3_kipan_adaptive_v1 import (  # noqa: E402
    _worker,
    _dispatch_phase,
    _load_partials,
    _select_promising,
    _rebuild_configs_for_phase2,
    _select_showcase_configs,
    _post_process,
)

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
LSPIN_ROOT        = Path("/banach2/wes/lspin-pytorch")
PAPER_ROOT        = Path(__file__).resolve().parents[1]
BRCA_DATA_DEFAULT = LSPIN_ROOT / "runs" / "tcga_brca20260214_001423"
RESULTS_DEFAULT   = PAPER_ROOT / "data" / "runs" / "ch3_brca_adaptive_v1"


# ── Config grid ───────────────────────────────────────────────────────────────

def _build_configs(args) -> List[Dict]:
    configs = []
    idx = 0
    for sigma in args.lspin_sigmas:
        for lam in args.lspin_lambdas:
            for smooth in args.sample_smooth_grid:
                configs.append({
                    "family_label": "LSPIN",
                    "gate_type":    "lspin_tf",
                    "gate_sigma":   float(sigma),
                    "lam":          float(lam),
                    "smooth":       float(smooth),
                    "global_cfg_idx": idx,
                })
                idx += 1
    for sigma in args.concrete_sigmas:
        for lam in args.concrete_lambdas:
            for smooth in args.sample_smooth_grid:
                configs.append({
                    "family_label": "Concrete",
                    "gate_type":    "concrete",
                    "gate_sigma":   float(sigma),
                    "lam":          float(lam),
                    "smooth":       float(smooth),
                    "global_cfg_idx": idx,
                })
                idx += 1
    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="BRCA adaptive sweep v1: 3-rep phase 1 → select promising → full phase 2"
    )
    p.add_argument("--outdir",      type=Path, default=BRCA_DATA_DEFAULT)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    p.add_argument("--gpus",        type=int,  nargs="+", default=[0, 2, 6, 7])
    p.add_argument("--seed",        type=int,  default=123)
    p.add_argument("--knn-k",       type=int,  default=5)
    p.add_argument("--n-phase1-reps",   type=int, default=3)
    p.add_argument("--n-total-reps",    type=int, default=12)
    p.add_argument("--top-n-promising", type=int, default=8,
                   help="Top N configs per (family, sigma) to carry into phase 2")

    p.add_argument("--patience",     type=int,   default=60)
    p.add_argument("--lr",           type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--max-epochs",   type=int,   default=300)

    # BRCA lambda grids — calibrated to Khard ~200–1000 at sigma=0.10
    # (based on ch3_rerun_brca_20260326_broad_balanced results)
    p.add_argument("--lspin-lambdas", type=float, nargs="+",
                   default=[0.010, 0.007, 0.005, 0.0033, 0.0025, 0.0018])
    p.add_argument("--concrete-lambdas", type=float, nargs="+",
                   default=[0.004, 0.003, 0.0022, 0.0018, 0.0015, 0.0012])

    # Two sigmas per family
    p.add_argument("--lspin-sigmas",    type=float, nargs="+", default=[0.10, 0.15])
    p.add_argument("--concrete-sigmas", type=float, nargs="+", default=[0.10, 0.15])

    # Temperatures (same as KIPAN)
    p.add_argument("--lspin-temperature",    type=float, default=0.5)
    p.add_argument("--concrete-temperature", type=float, default=0.3)

    # Smooth grid — BRCA convention uses higher smoothing values
    p.add_argument("--sample-smooth-grid", type=float, nargs="+",
                   default=[0.0, 0.1, 0.2, 0.4])

    # Evaluation
    p.add_argument("--risk-top-frac",         type=float, default=0.2)
    p.add_argument("--cluster-n-clusters",    type=int,   default=4,
                   help="BRCA has 6 subtypes; 4 clusters captures major groupings")
    p.add_argument("--global-freq-threshold", type=float, default=0.05)

    # Showcase selection Khard zone — wider than KIPAN (more features)
    p.add_argument("--khard-min", type=float, default=200.0)
    p.add_argument("--khard-max", type=float, default=900.0)

    p.add_argument("--post-process-only", action="store_true")
    return p.parse_args()


def main():
    args = _parse()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    from ch3_kipan_adaptive_v1 import _Tee
    sys.stdout = _Tee(results_dir / "run.log")

    all_configs = _build_configs(args)
    print(f"[main] {len(all_configs)} total configs in grid (BRCA)")
    print(f"[main] Data: {args.outdir}")
    print(f"[main] GPUs: {args.gpus}")
    print(f"[main] Phase 1: {args.n_phase1_reps} reps/config, "
          f"Phase 2: {args.n_total_reps - args.n_phase1_reps} more reps on top-{args.top_n_promising} per (family,sigma)")
    print(f"[main] knn_k={args.knn_k}, khard band: [{args.khard_min}, {args.khard_max}]")
    print(f"[main] Results → {results_dir}", flush=True)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    p1_dirs = sorted(results_dir.glob("_partial_p1_worker*"))
    if args.post_process_only:
        print(f"[main] --post-process-only: found {len(p1_dirs)} phase-1 partial dirs")
    else:
        if p1_dirs:
            print(f"[main] Found {len(p1_dirs)} existing phase-1 partial dirs — skipping phase 1")
        else:
            print(f"\n[main] === Phase 1: screening {len(all_configs)} configs × {args.n_phase1_reps} reps ===",
                  flush=True)
            t0 = time.time()
            p1_dirs = _dispatch_phase(
                all_configs, phase=1,
                rep_start=0, rep_end=args.n_phase1_reps,
                gpus=args.gpus, args=args, results_dir=results_dir,
            )
            print(f"[main] Phase 1 done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Load phase 1 results
    p1_data = _load_partials(p1_dirs)
    df_p1 = p1_data["runs"]

    # ── Select promising configs ───────────────────────────────────────────────
    promising_rows = _select_promising(
        df_p1,
        khard_min=args.khard_min,
        khard_max=args.khard_max,
        top_n=args.top_n_promising,
    )
    p2_configs = _rebuild_configs_for_phase2(all_configs, promising_rows)
    print(f"\n[main] Selected {len(p2_configs)} promising configs for phase 2:")
    for r in sorted(promising_rows, key=lambda x: -x["mean_test_c"]):
        print(f"  {r['model_family']} σ={r['gate_sigma']:.2f} "
              f"λ={r['lambda_sparse']:.5g} smooth={r['lambda_sample_smooth']:.4g} "
              f"Khard={r['mean_Khard']:.0f} C={r['mean_test_c']:.4f}")

    pd.DataFrame(promising_rows).to_csv(results_dir / "promising_configs_selected.csv", index=False)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    n_p2 = args.n_total_reps - args.n_phase1_reps
    p2_dirs = sorted(results_dir.glob("_partial_p2_worker*"))

    if not args.post_process_only:
        if p2_dirs:
            print(f"[main] Found {len(p2_dirs)} existing phase-2 partial dirs — skipping phase 2")
        elif n_p2 > 0 and p2_configs:
            print(f"\n[main] === Phase 2: {len(p2_configs)} configs × {n_p2} more reps ===",
                  flush=True)
            t0 = time.time()
            p2_dirs = _dispatch_phase(
                p2_configs, phase=2,
                rep_start=args.n_phase1_reps, rep_end=args.n_total_reps,
                gpus=args.gpus, args=args, results_dir=results_dir,
            )
            print(f"[main] Phase 2 done in {(time.time() - t0) / 60:.1f} min", flush=True)
        else:
            print("[main] Phase 2 skipped")

    # ── Post-process ──────────────────────────────────────────────────────────
    all_partial_dirs = p1_dirs + p2_dirs
    print(f"\n[main] Post-processing {len(all_partial_dirs)} partial dirs total", flush=True)
    _post_process(results_dir, all_partial_dirs, args)


if __name__ == "__main__":
    main()
