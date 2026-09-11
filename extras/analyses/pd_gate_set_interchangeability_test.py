#!/usr/bin/env python3
"""Test whether personalized gate sets are interchangeable across patients.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 1). Existing scripts already ask related-but-narrower questions:

- `plot_kipan_feature_set_cv_transport.py` asks whether a *consensus/union*
  gate-derived feature set transports across k-fold splits.
- `patient_subset_within_vs_outside_test.py` asks whether a selected gene's
  Cox signal is stronger within its own selecting subset than outside it.

Neither directly tests non-interchangeability: whether a subgroup's *own*
personalized gate set predicts that subgroup's outcome better than an
equally-sized feature set drawn from somewhere else. That is the direct
test of the claim that personalization is doing something real rather
than approximating one global sparse subset.

For each independently trained model (fold x rep x gate family), holdout
patients are clustered into subgroups by their hard-gate vectors (KMeans).
For each subgroup, four same-size feature sets are compared by fitting a
Cox model on the training fold restricted to that feature set and scoring
concordance on the subgroup's held-out members only:

    (a) own      -- top-K genes by gate rate within this subgroup
    (b) global   -- top-K genes by gate rate over the whole holdout set
                    (the "one sparse subset for everyone" baseline)
    (c) foreign  -- top-K genes by gate rate within a *different* subgroup
                    from the same model, applied out of group
    (d) random   -- K uniformly random genes (averaged over several draws)

If personalization is real and not interchangeable, (a) should beat (b),
(c), and (d) more often than chance -- with (b) as the comparison that
actually matters, since beating (d) alone is the weaker result the
existing probe scripts already establish.

This reuses the same training/gating machinery as
`patient_subset_within_vs_outside_test.py` (fresh small k-fold retraining
of the selected configs, not the saved gate matrices from the main
adaptive sweep, since per-patient hard gates on a held-out fold are needed
here) and the Cox-fit/eval helper from
`plot_kipan_feature_set_cv_transport.py`.

Usage:
    python pd_gate_set_interchangeability_test.py --dataset kipan --device cuda:2
    python pd_gate_set_interchangeability_test.py --dataset brca --device cuda:2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before numpy/scipy/sklearn import -- see
# patient_subset_within_vs_outside_test.py for why (OpenBLAS/OpenMP
# thread-leak deadlock observed with repeated PCA/KMeans/kNN calls in a
# long-lived process).
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
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold

import sparsedeepsurv as sds
from plot_kipan_feature_set_cv_transport import _fit_eval_cox
from quick_linear_gated_probe import _combined_arrays, _load_data, _selected_configs, _split_train_val

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
    p.add_argument(
        "--families",
        nargs="+",
        default=["LSPIN", "Concrete", "L-LSPIN", "L-Concrete"],
        choices=["LSPIN", "Concrete", "L-LSPIN", "L-Concrete"],
    )
    p.add_argument("--selections", nargs="+", default=["smooth"], choices=["nosmooth", "smooth"])
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
    p.add_argument("--n-subgroups", type=int, default=4)
    p.add_argument("--min-subgroup-n", type=int, default=15)
    p.add_argument("--min-events", type=int, default=5)
    p.add_argument("--k-min", type=int, default=5, help="Minimum feature-set size compared.")
    p.add_argument("--k-max", type=int, default=50, help="Maximum feature-set size compared.")
    p.add_argument("--n-random-draws", type=int, default=20)
    p.add_argument("--cox-alpha", type=float, default=1.0)
    p.add_argument("--cox-max-iter", type=int, default=200)
    p.add_argument("--roc-time-quantile", type=float, default=0.60)
    return p.parse_args()


def _top_k_by_rate(hard_subset: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k genes by gate rate within a patient subset."""
    rate = hard_subset.mean(axis=0)
    order = np.argsort(-rate)
    return order[:k]


def _subgroup_feature_sets(
    hard_holdout: np.ndarray,
    subgroup_labels: np.ndarray,
    *,
    k_min: int,
    k_max: int,
) -> dict[int, np.ndarray]:
    """Own top-K feature set per subgroup, K = clip(subgroup's typical
    hard-selected count, k_min, k_max)."""
    global_rate = hard_holdout.mean(axis=0)
    out = {}
    for g in np.unique(subgroup_labels):
        mask = subgroup_labels == g
        sub_hard = hard_holdout[mask]
        typical_k = int(np.clip(np.median(sub_hard.sum(axis=1)), k_min, k_max))
        out[int(g)] = _top_k_by_rate(sub_hard, typical_k)
    return out


def _run_comparison(
    *,
    family: str,
    selection: str,
    fold: int,
    rep: int,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_idx: np.ndarray,
    holdout: np.ndarray,
    hard_holdout: np.ndarray,
    n_subgroups: int,
    min_subgroup_n: int,
    min_events: int,
    k_min: int,
    k_max: int,
    n_random_draws: int,
    cox_alpha: float,
    cox_max_iter: int,
    roc_time: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_holdout = hard_holdout.shape[0]
    n_clusters = min(n_subgroups, max(1, n_holdout // max(1, min_subgroup_n)))
    if n_clusters < 2:
        return pd.DataFrame()

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    subgroup_labels = km.fit_predict(hard_holdout)

    global_top = None  # computed per-K below, once K is known per subgroup
    own_sets = _subgroup_feature_sets(hard_holdout, subgroup_labels, k_min=k_min, k_max=k_max)
    n_genes = X.shape[1]

    rows = []
    for g, own_idx in own_sets.items():
        subgroup_mask_holdout = subgroup_labels == g
        test_idx = holdout[subgroup_mask_holdout]
        n_test = len(test_idx)
        n_events_test = int(event[test_idx].sum())
        if n_test < min_subgroup_n or n_events_test < min_events:
            continue
        k = len(own_idx)

        global_top = _top_k_by_rate(hard_holdout, k)

        other_groups = [og for og in own_sets if og != g]
        if not other_groups:
            continue
        foreign_g = rng.choice(other_groups)
        foreign_idx = own_sets[foreign_g][:k]
        if len(foreign_idx) < k:
            continue

        def _eval(feature_idx: np.ndarray) -> float:
            res, _ = _fit_eval_cox(
                X, time, event, train_idx, test_idx, feature_idx,
                alpha=cox_alpha, n_iter=cox_max_iter, roc_time=roc_time,
            )
            return float(res.get("cindex", np.nan))

        c_own = _eval(own_idx)
        c_global = _eval(global_top)
        c_foreign = _eval(foreign_idx)
        c_random_draws = [
            _eval(rng.choice(n_genes, size=k, replace=False)) for _ in range(n_random_draws)
        ]
        c_random = float(np.nanmedian(c_random_draws))

        rows.append({
            "family": family, "selection": selection, "fold": fold, "rep": rep,
            "subgroup": int(g), "n_subgroups_this_model": n_clusters,
            "k": int(k), "n_test": n_test, "n_events_test": n_events_test,
            "cindex_own": c_own, "cindex_global": c_global,
            "cindex_foreign": c_foreign, "cindex_random_median": c_random,
            "own_minus_global": c_own - c_global,
            "own_minus_foreign": c_own - c_foreign,
            "own_minus_random": c_own - c_random,
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (
        results_dir / f"pd_gate_set_interchangeability_{args.dataset}_{args.n_folds}fold_{args.n_reps}rep"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X, time, event, histo, genes = _combined_arrays(data)
    configs = _selected_configs(
        args.dataset, results_dir, args.families, args.selections,
        lambda_scale_families=args.families, lambda_scale_selections=args.selections,
        lambda_scales=[1.0], smooth_smooth_values=None,
    )
    kf = KFold(n_splits=int(args.n_folds), shuffle=True, random_state=int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    roc_time = float(np.quantile(time[event.astype(bool)], args.roc_time_quantile))

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
                _, _, hard_t, _ = sds.get_gates(
                    model, sds.as_torch(X[holdout]), device=args.device,
                    hard_threshold=float(args.hard_threshold), batch_size=512,
                )
                hard_holdout = hard_t.numpy().astype(np.float32)
                frame = _run_comparison(
                    family=str(cfg["family"]), selection=str(cfg["selection"]), fold=fold, rep=rep,
                    X=X, time=time, event=event, train_idx=train_idx, holdout=holdout,
                    hard_holdout=hard_holdout, n_subgroups=int(args.n_subgroups),
                    min_subgroup_n=int(args.min_subgroup_n), min_events=int(args.min_events),
                    k_min=int(args.k_min), k_max=int(args.k_max), n_random_draws=int(args.n_random_draws),
                    cox_alpha=float(args.cox_alpha), cox_max_iter=int(args.cox_max_iter),
                    roc_time=roc_time, rng=rng,
                )
                print(f"[compared] fold={fold} rep={rep} {label} n_subgroup_rows={len(frame)}", flush=True)
                frames.append(frame)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result.to_csv(outdir / "gate_set_interchangeability.csv", index=False)

    rows = []
    for (family, selection), sub in result.groupby(["family", "selection"]) if len(result) else []:
        summary_row = {"family": family, "selection": selection, "n_subgroup_rows": len(sub)}
        for col in ["own_minus_global", "own_minus_foreign", "own_minus_random"]:
            diffs = sub[col].dropna().to_numpy()
            n = len(diffs)
            summary_row[f"{col}_n"] = n
            summary_row[f"{col}_frac_positive"] = float((diffs > 0).mean()) if n else float("nan")
            summary_row[f"{col}_median"] = float(np.median(diffs)) if n else float("nan")
            if n >= 10 and np.any(diffs != 0):
                stat, p = wilcoxon(diffs, alternative="greater")
            else:
                stat, p = float("nan"), float("nan")
            summary_row[f"{col}_wilcoxon_p_greater"] = p
        rows.append(summary_row)
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "gate_set_interchangeability_summary.csv", index=False)
    print(summary.to_string(), flush=True)
    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
