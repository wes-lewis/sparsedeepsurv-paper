#!/usr/bin/env python3
"""Semi-synthetic ground-truth recovery test for personalized feature
selection.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 5). This is the single most convincing experiment type available
for the personalization claim, because it is the one place held-out
predictive accuracy is replaced with a check no global sparse model can
win on by construction: precision/recall of selected-vs-*true* features,
known exactly because the outcome is generated from them.

Design (first-pass, freshly assigned synthetic subgroups -- see module
docstring TODO history for why real-subgroup relabeling was deferred to a
later iteration of this thread rather than attempted first):

1. Use real KIPAN/BRCA covariates (X) to preserve realistic covariance
   structure.
2. Partition patients into `--n-subgroups` synthetic subgroups (uniform
   random assignment).
3. Assign each subgroup a disjoint set of `--true-features-per-subgroup`
   real genes as its true-relevant feature subset.
4. Generate synthetic survival times from a linear Cox hazard driven only
   by each patient's subgroup-specific true features (unit-weight linear
   predictor, scaled by `--effect-size`), with independent exponential
   censoring calibrated (by binary search) to hit `--censoring-rate`.
5. Fit LSPIN/Concrete (personalized) and a Coxnet "one global sparse
   subset for everyone" baseline (elastic-net Cox, alpha chosen to match
   the gated models' average per-patient selected count) on the synthetic
   outcome, using the real covariates.
6. Score precision/recall/Jaccard of selected-vs-true features, per
   subgroup and in aggregate, plus a sanity-check C-index for all methods
   on the synthetic outcome (comparable C-index despite very different
   feature-recovery quality is exactly the gap this experiment targets).

MLP+STG is deferred here as in threads 4 and 6 (bespoke training loop).

Usage:
    python pd_synthetic_ground_truth_recovery.py --dataset kipan --device cuda:2
    python pd_synthetic_ground_truth_recovery.py --dataset brca --device cuda:5

Known limitations of this first pass (flagged rather than hidden):
    - Effect size / subgroup separability is fixed per run rather than
      swept; `--effect-size` sweep across a couple of settings is the
      natural next step before treating a single run's result as
      conclusive (per the original TODO 6 in the plan).
    - Subgroups are freshly assigned, not real biological subgroups, so
      this establishes the *mechanism* (personalization can recover
      structure a global method cannot) without yet connecting it to real
      biology -- that connection is thread 3's job.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import CANONICAL_RUNS, PROCESSED_DATASETS, ensure_repo_imports

ensure_repo_imports(include_analyses_dir=True)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import ShuffleSplit

import sparsedeepsurv as sds
from quick_linear_gated_probe import _combined_arrays, _load_data, _selected_configs

RUN_DEFAULTS = {
    "kipan": CANONICAL_RUNS["kipan_adaptive_gentle"],
    "brca": CANONICAL_RUNS["brca_adaptive_gentle"],
}
DATA_DEFAULTS = PROCESSED_DATASETS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kipan", "brca"], default="kipan")
    p.add_argument("--families", nargs="+", default=["LSPIN", "Concrete"], choices=["LSPIN", "Concrete"])
    p.add_argument("--selection", choices=["nosmooth", "smooth"], default="smooth")
    p.add_argument("--n-subgroups", type=int, default=4)
    p.add_argument("--true-features-per-subgroup", type=int, default=8)
    p.add_argument("--effect-size", type=float, default=1.5)
    p.add_argument("--censoring-rate", type=float, default=0.4)
    p.add_argument("--test-fraction", type=float, default=0.30)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260407)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--coxnet-l1-ratio", type=float, default=0.9)
    p.add_argument("--coxnet-n-alphas", type=int, default=40)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument(
        "--candidate-gene-pool-size", type=int, default=None,
        help=(
            "Restrict X to this many top-variance genes before assigning true "
            "features or training (2026-09-12 recalibration): the first-pass run "
            "at full dimensionality (6000-24000 genes) made recovering "
            "true-features-per-subgroup=8 too hard for ANY method, including "
            "baselines (near-zero precision/recall, near-chance sanity-check "
            "C-index). Restricting to a moderate candidate pool tests the same "
            "question -- can personalized selection recover subgroup-specific "
            "truth a global method cannot -- as a fair, tractable feature-"
            "selection benchmark instead of an unwinnable needle-in-haystack "
            "search. Real genes are kept (not synthetic), so covariance "
            "structure among the top-variance genes is still real."
        ),
    )
    return p.parse_args()


def assign_synthetic_subgroups(n: int, n_subgroups: int, rng: np.random.Generator) -> np.ndarray:
    labels = rng.integers(0, n_subgroups, size=n)
    return labels


def assign_true_feature_map(
    n_genes: int, n_subgroups: int, k_per_subgroup: int, rng: np.random.Generator
) -> dict[int, np.ndarray]:
    """Disjoint true-relevant feature subsets, one per subgroup."""
    total_needed = n_subgroups * k_per_subgroup
    if total_needed > n_genes:
        raise ValueError(f"n_subgroups*k_per_subgroup={total_needed} exceeds n_genes={n_genes}")
    pool = rng.choice(n_genes, size=total_needed, replace=False)
    return {g: pool[g * k_per_subgroup:(g + 1) * k_per_subgroup] for g in range(n_subgroups)}


def _calibrate_censoring_scale(
    event_time: np.ndarray, target_rate: float, rng: np.random.Generator, tol: float = 0.01, max_iter: int = 40,
) -> float:
    """Binary search an exponential censoring-time scale so that, drawing
    C ~ Exp(scale) independently per patient, P(C < event_time) ~
    target_rate (i.e. the censoring rate)."""
    lo, hi = 1e-3, 1e6
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        c = rng.exponential(scale=mid, size=len(event_time))
        rate = float((c < event_time).mean())
        if abs(rate - target_rate) < tol:
            return mid
        if rate < target_rate:
            hi = mid  # need more censoring -> smaller scale -> shorter censoring times
        else:
            lo = mid
    return mid


def generate_synthetic_hazard(
    X: np.ndarray,
    subgroup_labels: np.ndarray,
    true_feature_map: dict[int, np.ndarray],
    effect_size: float,
    censoring_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (time, event) from a linear Cox hazard driven by
    each patient's subgroup-specific true features (Exponential
    proportional-hazards model: T = -log(U) * exp(-linpred)), with
    independent exponential censoring calibrated to the target rate."""
    n = X.shape[0]
    linpred = np.zeros(n, dtype=float)
    for g, feat_idx in true_feature_map.items():
        mask = subgroup_labels == g
        linpred[mask] = X[mask][:, feat_idx].sum(axis=1)
    linpred = (linpred - linpred.mean()) / (linpred.std() + 1e-8) * effect_size

    u = rng.uniform(1e-8, 1.0, size=n)
    event_time = -np.log(u) * np.exp(-linpred)

    scale = _calibrate_censoring_scale(event_time, censoring_rate, rng)
    censor_time = rng.exponential(scale=scale, size=n)

    time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(np.uint8)
    return time.astype(np.float32), event


def score_recovery(
    selected: np.ndarray, true_feature_map: dict[int, np.ndarray], subgroup_labels: np.ndarray, n_genes: int,
) -> pd.DataFrame:
    """Precision/recall/Jaccard of selected-vs-true features, per patient,
    aggregated by subgroup. `selected` is (n_patients, n_genes) binary;
    for a global (non-personalized) method, pass the same row repeated
    for every patient."""
    rows = []
    for g, feat_idx in true_feature_map.items():
        mask = subgroup_labels == g
        if not np.any(mask):
            continue
        true_mask = np.zeros(n_genes, dtype=bool)
        true_mask[feat_idx] = True
        sel_g = selected[mask].astype(bool)
        inter = (sel_g & true_mask).sum(axis=1)
        sel_count = sel_g.sum(axis=1)
        true_count = true_mask.sum()
        union = sel_count + true_count - inter
        precision = np.divide(inter, sel_count, out=np.zeros_like(inter, dtype=float), where=sel_count > 0)
        recall = inter / true_count
        jaccard = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)
        rows.append({
            "subgroup": int(g), "n_patients": int(mask.sum()),
            "precision_mean": float(precision.mean()), "recall_mean": float(recall.mean()),
            "jaccard_mean": float(jaccard.mean()), "mean_n_selected": float(sel_count.mean()),
        })
    return pd.DataFrame(rows)


def _coxnet_global_selection(
    *, X, time, event, train_idx, val_idx, target_k, l1_ratio, n_alphas,
) -> np.ndarray:
    mu, sd = X[train_idx].mean(axis=0), X[train_idx].std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xtr = (X[train_idx] - mu) / sd
    Xva = (X[val_idx] - mu) / sd
    y_tr = np.array(
        list(zip(event[train_idx].astype(bool), time[train_idx].astype(float))), dtype=[("event", "?"), ("time", "<f8")],
    )
    model = CoxnetSurvivalAnalysis(l1_ratio=float(l1_ratio), alpha_min_ratio=0.01, n_alphas=int(n_alphas), max_iter=8000)
    model.fit(Xtr, y_tr)
    coefs = model.coef_
    nnz = (np.abs(coefs) > 1e-10).sum(axis=0)
    best_j = int(np.argmin(np.abs(nnz - max(1, target_k))))
    selected_mask = (np.abs(coefs[:, best_j]) > 1e-10).astype(np.float32)
    risk_val = Xva @ coefs[:, best_j]
    c_val = sds.concordance_index(risk_val, time[val_idx], event[val_idx])
    return selected_mask, coefs[:, best_j], float(c_val)


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (results_dir / f"pd_synthetic_ground_truth_{args.dataset}")
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X, real_time, real_event, histo, genes = _combined_arrays(data)

    if args.candidate_gene_pool_size and args.candidate_gene_pool_size < X.shape[1]:
        top_idx = np.argsort(-X.var(axis=0))[: args.candidate_gene_pool_size]
        top_idx = np.sort(top_idx)
        X = X[:, top_idx]
        genes = genes[top_idx]
        print(f"[setup] restricted to top {args.candidate_gene_pool_size} variance genes", flush=True)

    n, n_genes = X.shape
    print(f"[setup] n={n} n_genes={n_genes}", flush=True)

    rng = np.random.default_rng(int(args.seed))
    subgroup_labels = assign_synthetic_subgroups(n, args.n_subgroups, rng)
    true_feature_map = assign_true_feature_map(n_genes, args.n_subgroups, args.true_features_per_subgroup, rng)
    time, event = generate_synthetic_hazard(
        X, subgroup_labels, true_feature_map, args.effect_size, args.censoring_rate, rng,
    )
    print(f"[synthetic] censoring_rate_actual={1 - event.mean():.3f} (target {args.censoring_rate})", flush=True)

    ss = ShuffleSplit(n_splits=1, test_size=float(args.test_fraction), random_state=int(args.seed))
    train_all, test_idx = next(ss.split(X))
    ss2 = ShuffleSplit(n_splits=1, test_size=float(args.val_fraction), random_state=int(args.seed) + 1)
    rel_train, rel_val = next(ss2.split(train_all))
    train_idx, val_idx = train_all[rel_train], train_all[rel_val]

    configs = _selected_configs(
        args.dataset, results_dir, args.families, [args.selection],
        lambda_scale_families=args.families, lambda_scale_selections=[args.selection],
        lambda_scales=[1.0], smooth_smooth_values=None,
    )

    recovery_frames = []
    cindex_rows = []
    mean_k_gated = []

    for _, cfg in configs.iterrows():
        label = f"{cfg['family']}_{cfg['selection']}"
        Xt_tr = sds.as_torch(X[train_idx])
        A = None
        if float(cfg["lambda_sample_smooth"]) > 0:
            A = sds.build_knn_adjacency_csr(Xt_tr, k=int(args.knn_k), pca_dim=50, metric="cosine", symmetrize=True)
        model, info = sds.run_one_model(
            Xt_tr=Xt_tr, tt_tr=sds.as_torch(time[train_idx]), et_tr=sds.as_torch(event[train_idx]),
            Xt_val=sds.as_torch(X[val_idx]), tt_val=sds.as_torch(time[val_idx]), et_val=sds.as_torch(event[val_idx]),
            Xt_test=sds.as_torch(X[test_idx]), tt_test=sds.as_torch(time[test_idx]), et_test=sds.as_torch(event[test_idx]),
            input_dim=X.shape[1], A_sample_train=A, device=args.device,
            lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size),
            max_epochs=int(args.max_epochs), seed=int(args.seed),
            gate_type=str(cfg["gate_type"]), gate_sigma=float(cfg["gate_sigma"]),
            lam=float(cfg["lambda_sparse"]), lambda_sample_smooth=float(cfg["lambda_sample_smooth"]),
            patience=int(args.patience), temperature=float(cfg["temperature"]),
            concrete_mode=str(cfg["concrete_mode"]), predictor=str(cfg["predictor"]),
            gating_hidden_dim=int(cfg["gating_hidden_dim"]), gate_hidden_dropout_p=float(cfg["gate_hidden_dropout_p"]),
            risk_hidden_dims=tuple(cfg["risk_hidden_dims"]), risk_dropout_p=float(cfg["risk_dropout_p"]),
            lspin_init_bias=float(cfg["lspin_init_bias"]), gate_weight_decay=float(cfg["gate_weight_decay"]),
        )
        cindex_rows.append({"method": label, "test_cindex": info.get("test_cindex")})
        _, _, hard_t, _ = sds.get_gates(
            model, sds.as_torch(X[test_idx]), device=args.device, hard_threshold=float(args.hard_threshold), batch_size=512,
        )
        hard = hard_t.numpy().astype(np.float32)
        mean_k_gated.append(float(hard.sum(axis=1).mean()))
        rec = score_recovery(hard, true_feature_map, subgroup_labels[test_idx], n_genes)
        rec.insert(0, "method", label)
        recovery_frames.append(rec)
        print(f"[{label}] test_cindex={info.get('test_cindex')} mean_k={mean_k_gated[-1]:.1f}", flush=True)

    target_k = int(round(np.mean(mean_k_gated))) if mean_k_gated else args.true_features_per_subgroup * 2
    coxnet_mask, coxnet_coef, coxnet_val_c = _coxnet_global_selection(
        X=X, time=time, event=event, train_idx=train_idx, val_idx=val_idx,
        target_k=target_k, l1_ratio=args.coxnet_l1_ratio, n_alphas=args.coxnet_n_alphas,
    )
    coxnet_selected_repeated = np.tile(coxnet_mask, (len(test_idx), 1))
    coxnet_risk_test = ((X[test_idx] - X[train_idx].mean(axis=0)) / (X[train_idx].std(axis=0) + 1e-8)) @ coxnet_coef
    coxnet_test_c = sds.concordance_index(coxnet_risk_test, time[test_idx], event[test_idx])
    cindex_rows.append({"method": f"coxnet_global_matched_k{target_k}", "test_cindex": coxnet_test_c})
    rec = score_recovery(coxnet_selected_repeated, true_feature_map, subgroup_labels[test_idx], n_genes)
    rec.insert(0, "method", f"coxnet_global_matched_k{target_k}")
    recovery_frames.append(rec)
    print(f"[coxnet_global] test_cindex={coxnet_test_c:.4f} k={int(coxnet_mask.sum())} (target {target_k})", flush=True)

    recovery = pd.concat(recovery_frames, ignore_index=True)
    recovery.to_csv(outdir / "synthetic_recovery_by_subgroup.csv", index=False)

    overall = recovery.groupby("method")[["precision_mean", "recall_mean", "jaccard_mean"]].mean().reset_index()
    overall.to_csv(outdir / "synthetic_recovery_overall.csv", index=False)

    cindex_df = pd.DataFrame(cindex_rows)
    cindex_df.to_csv(outdir / "synthetic_sanity_check_cindex.csv", index=False)

    print(overall.to_string(), flush=True)
    print(cindex_df.to_string(), flush=True)
    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
