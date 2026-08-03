#!/usr/bin/env python3
"""
Test whether selected genes carry patient-subset-specific prognostic signal,
using a comparison that can produce a negative result.

The existing selected-gene recurrence and signal analysis
(quick_linear_gated_probe.py) compares each hard-selected gene's Cox signal,
within the patients who selected it, against random genes scored on the same
patients. Because the genes were chosen using this cohort's own survival
outcome, that comparison cannot lose: genes selected for a subset are
close to guaranteed to look better than random genes in that same subset.

This script instead compares each selected gene against itself: its Cox
signal within the patients who selected it (the "gating subset") versus its
Cox signal among the complementary held-out patients who did not select it
(the "outside" subset). If gating recovers genuinely subset-specific
prognostic signal, the within-subset signal should exceed the outside-subset
signal more often than chance. If gating were arbitrary, there would be no
reason for a gene's signal to differ between the two groups.

This needs no new model architecture or training scheme, only the same
gate matrices and holdout data already used by quick_linear_gated_probe.py,
plus one extra Cox test per gene on the complementary patient mask.

Usage:
    python patient_subset_within_vs_outside_test.py --dataset kipan
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before numpy/scipy/sklearn import. Without this, repeated
# PCA/NearestNeighbors calls across folds and reps in one long-lived process
# can leak OpenBLAS/OpenMP threads until a later call deadlocks waiting on a
# thread-pool slot that never frees (observed directly: a hung process had
# 123 threads via /proc/<pid>/status, 0% GPU utilization, and flat CPU time,
# i.e. genuinely stuck, not merely slow). Restricting each backend to a
# single thread avoids the leak entirely; per-fit cost is negligible since
# each PCA/kNN call here operates on only a few hundred patients.
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

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold

import sparsedeepsurv as sds
from plot_kipan_gated_univariate_dotplot import _cox_score_test_matrix
from quick_linear_gated_probe import _load_data, _combined_arrays, _selected_configs, _split_train_val

RUN_DEFAULTS = {
    "pancan": CANONICAL_RUNS["pancan_adaptive_gentle"],
    "kipan": CANONICAL_RUNS["kipan_adaptive_gentle"],
    "brca": CANONICAL_RUNS["brca_adaptive_gentle"],
}
DATA_DEFAULTS = PROCESSED_DATASETS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["pancan", "kipan", "brca"], default="kipan")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--families", nargs="+", default=["LSPIN", "Concrete", "L-LSPIN", "L-Concrete"],
                    choices=["LSPIN", "Concrete", "L-LSPIN", "L-Concrete"])
    p.add_argument("--selections", nargs="+", default=["nosmooth", "smooth"], choices=["nosmooth", "smooth"])
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--n-reps", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260407)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--min-selected-n", type=int, default=25)
    p.add_argument("--min-events", type=int, default=5)
    p.add_argument("--top-genes", type=int, default=500)
    return p.parse_args()


def _within_vs_outside_test(
    *,
    family: str,
    selection: str,
    fold: int,
    rep: int,
    hard: np.ndarray,
    X_holdout: np.ndarray,
    time_holdout: np.ndarray,
    event_holdout: np.ndarray,
    genes: np.ndarray,
    min_selected_n: int,
    min_events: int,
    top_genes: int,
) -> pd.DataFrame:
    gate_rate = hard.mean(axis=0)
    selected_n = hard.sum(axis=0).astype(int)
    candidate_idx = np.argsort(-gate_rate)
    candidate_idx = candidate_idx[selected_n[candidate_idx] >= int(min_selected_n)][: int(top_genes)]
    rows = []
    for gene_idx in candidate_idx:
        mask = hard[:, gene_idx] > 0.5
        outside = ~mask
        n_in, n_out = int(mask.sum()), int(outside.sum())
        ev_in, ev_out = int(event_holdout[mask].sum()), int(event_holdout[outside].sum())
        if n_in < min_selected_n or ev_in < min_events or n_out < min_selected_n or ev_out < min_events:
            continue
        within = _cox_score_test_matrix(
            X_holdout[mask][:, [gene_idx]], time_holdout[mask], event_holdout[mask], min_events=min_events,
        )
        outside_res = _cox_score_test_matrix(
            X_holdout[outside][:, [gene_idx]], time_holdout[outside], event_holdout[outside], min_events=min_events,
        )
        w = float(within["signed_neglog10_p"][0])
        o = float(outside_res["signed_neglog10_p"][0])
        if not (np.isfinite(w) and np.isfinite(o)):
            continue
        w_abs, o_abs = abs(w), abs(o)
        rows.append({
            "family": family, "selection": selection, "fold": fold, "rep": rep,
            "gene_idx": int(gene_idx), "gene": str(genes[gene_idx]),
            "n_within": n_in, "n_outside": n_out,
            "within_abs_neglog10_p": w_abs, "outside_abs_neglog10_p": o_abs,
            "within_minus_outside": w_abs - o_abs,
            "within_gt_outside": bool(w_abs > o_abs),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (results_dir / f"patient_subset_within_vs_outside_{args.dataset}_{args.n_folds}fold_{args.n_reps}rep")
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X, time, event, histo, genes = _combined_arrays(data)
    configs = _selected_configs(
        args.dataset, results_dir, args.families, args.selections,
        lambda_scale_families=args.families, lambda_scale_selections=args.selections,
        lambda_scales=[1.0], smooth_smooth_values=None,
    )
    kf = KFold(n_splits=int(args.n_folds), shuffle=True, random_state=int(args.seed))

    frames = []
    for fold, (train_all, holdout) in enumerate(kf.split(X)):
        for rep in range(int(args.n_reps)):
            seed = int(args.seed + 1000 * fold + rep)
            train_idx, val_idx = _split_train_val(train_all, seed)
            A = None
            Xt_tr = sds.as_torch(X[train_idx])
            if configs["lambda_sample_smooth"].gt(0).any():
                print(f"[knn] fold={fold} rep={rep} building adjacency (n_train={len(train_idx)})", flush=True)
                A = sds.build_knn_adjacency_csr(Xt_tr, k=int(args.knn_k), pca_dim=50, metric="cosine", symmetrize=True)
                print(f"[knn] fold={fold} rep={rep} done", flush=True)
            common = dict(
                Xt_tr=Xt_tr, tt_tr=sds.as_torch(time[train_idx]), et_tr=sds.as_torch(event[train_idx]),
                Xt_val=sds.as_torch(X[val_idx]), tt_val=sds.as_torch(time[val_idx]), et_val=sds.as_torch(event[val_idx]),
                Xt_test=sds.as_torch(X[holdout]), tt_test=sds.as_torch(time[holdout]), et_test=sds.as_torch(event[holdout]),
                input_dim=X.shape[1], A_sample_train=A, device=args.device,
                lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size),
                max_epochs=int(args.max_epochs), seed=seed,
            )
            for _, cfg in configs.iterrows():
                label = f"{cfg['family']} {cfg['selection']}"
                print(f"[run] fold={fold} rep={rep} {label}", flush=True)
                model, info = sds.run_one_model(
                    **common, gate_type=str(cfg["gate_type"]), gate_sigma=float(cfg["gate_sigma"]),
                    lam=float(cfg["lambda_sparse"]), lambda_sample_smooth=float(cfg["lambda_sample_smooth"]),
                    patience=int(cfg.get("patience", args.patience)), temperature=float(cfg["temperature"]),
                    concrete_mode=str(cfg["concrete_mode"]), predictor=str(cfg["predictor"]),
                    gating_hidden_dim=int(cfg["gating_hidden_dim"]), gate_hidden_dropout_p=float(cfg["gate_hidden_dropout_p"]),
                    risk_hidden_dims=tuple(cfg["risk_hidden_dims"]), risk_dropout_p=float(cfg["risk_dropout_p"]),
                    lspin_init_bias=float(cfg["lspin_init_bias"]), gate_weight_decay=float(cfg["gate_weight_decay"]),
                )
                print(f"[trained] fold={fold} rep={rep} {label}", flush=True)
                _, _, hard_t, _ = sds.get_gates(model, sds.as_torch(X[holdout]), device=args.device,
                                                 hard_threshold=float(args.hard_threshold), batch_size=512)
                hard = hard_t.numpy().astype(np.float32)
                print(f"[gates] fold={fold} rep={rep} {label} n_candidates={int((hard.sum(axis=0) >= args.min_selected_n).sum())}", flush=True)
                frame = _within_vs_outside_test(
                    family=str(cfg["family"]), selection=str(cfg["selection"]), fold=fold, rep=rep,
                    hard=hard, X_holdout=X[holdout], time_holdout=time[holdout], event_holdout=event[holdout],
                    genes=genes, min_selected_n=int(args.min_selected_n), min_events=int(args.min_events),
                    top_genes=int(args.top_genes),
                )
                print(f"[cox-done] fold={fold} rep={rep} {label} n_pairs={len(frame)}", flush=True)
                frames.append(frame)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(outdir / "within_vs_outside_gene_signal.csv", index=False)

    rows = []
    for (family, selection), sub in result.groupby(["family", "selection"]):
        diffs = sub["within_minus_outside"].to_numpy()
        n = len(diffs)
        frac_gt = float((diffs > 0).mean()) if n else float("nan")
        if n >= 10 and np.any(diffs != 0):
            stat, p = wilcoxon(diffs, alternative="greater")
        else:
            stat, p = float("nan"), float("nan")
        rows.append({
            "family": family, "selection": selection, "n_genes": n,
            "fraction_within_gt_outside": frac_gt,
            "median_within_minus_outside": float(np.median(diffs)) if n else float("nan"),
            "wilcoxon_stat": stat, "wilcoxon_p_greater": p,
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "within_vs_outside_summary.csv", index=False)
    print(summary.to_string(), flush=True)
    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
