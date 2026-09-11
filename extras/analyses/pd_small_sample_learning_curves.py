#!/usr/bin/env python3
"""Training-set-size learning curves: does sparsity's generalization
advantage show up in the n<<p regime this data actually lives in?

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 6).

Genomic survival data (KIPAN, BRCA gene expression) is exactly the
high-dimensional, sample-limited setting where regularized/sparse models
are traditionally argued to have an edge over dense models. The current
results don't test this directly -- they compare methods at the full
available training-set size only, the setting least likely to show a
sparsity advantage. This subsamples the training fold at multiple
fractions (holding the same held-out test fold fixed) and reports, for
each model:

    - held-out C-index (does absolute performance degrade less?)
    - the generalization gap: train C-index minus held-out C-index (more
      directly isolates overfitting than held-out C-index alone)

Shares infrastructure with `pd_noise_robustness_sweep.py` (thread 4):
same dense-MLP and gated-model training helpers, same reuse of
`selected_comparison_configs.csv` via `_selected_configs` for
already-tuned gated hyperparameters, so lambda isn't re-tuned at every
fraction (same rationale as thread 4 -- avoids confounding the comparison
and blowing up compute; flagged as a limitation since optimal lambda
plausibly shifts with training-set size). Adds a genuine sparse-linear
baseline (Lasso-leaning Coxnet, l1_ratio=0.9) via its own regularization
path, with the alpha selected by validation-fold C-index, matching the
sparsity-matched-baseline pattern used in
`pd_selection_stability_index.py` (thread 2).

MLP+STG is deferred here for the same reason as thread 4: bespoke
training loop not yet ported to these shared helpers.

Usage:
    python pd_small_sample_learning_curves.py --dataset kipan --device cuda:2
    python pd_small_sample_learning_curves.py --dataset brca --device cuda:5
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
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

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
    p.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    p.add_argument("--families", nargs="+", default=["LSPIN", "Concrete"], choices=["LSPIN", "Concrete"])
    p.add_argument("--selection", choices=["nosmooth", "smooth"], default="smooth")
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--test-fraction", type=float, default=0.30)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--min-events-train", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260407)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--mlp-max-epochs", type=int, default=150)
    p.add_argument("--mlp-patience", type=int, default=20)
    p.add_argument("--mlp-hidden-dims", type=int, nargs="+", default=[64, 32])
    p.add_argument("--mlp-dropout", type=float, default=0.1)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    p.add_argument("--mlp-l1-input", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--coxnet-l1-ratio", type=float, default=0.9)
    p.add_argument("--coxnet-n-alphas", type=int, default=40)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def subsample_training_fold(
    train_idx: np.ndarray, event: np.ndarray, fraction: float, seed: int, min_events: int,
) -> np.ndarray:
    """Stratified (on event indicator) subsample of the training fold.
    Raises if the requested fraction would leave too few events."""
    if fraction >= 1.0:
        return train_idx
    n_keep = max(2, int(round(len(train_idx) * fraction)))
    ev = event[train_idx].astype(int)
    if ev.sum() < 2 or (len(ev) - ev.sum()) < 2:
        rng = np.random.default_rng(seed)
        return rng.choice(train_idx, size=n_keep, replace=False)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_keep, random_state=seed)
    keep_rel, _ = next(sss.split(np.zeros_like(ev), ev))
    kept = train_idx[keep_rel]
    n_events_kept = int(event[kept].sum())
    if n_events_kept < min_events:
        raise ValueError(
            f"fraction={fraction} leaves only {n_events_kept} events (< {min_events}) in "
            f"a training fold of size {len(kept)}; skip this fraction or lower --min-events-train."
        )
    return kept


def _standardize(X_train, X_other):
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X_train - mu) / sd, (X_other - mu) / sd


def _fit_eval_coxnet(*, X, time, event, train_idx, val_idx, test_idx, l1_ratio, n_alphas):
    """Fit a Coxnet path on train, pick the alpha with the best val
    C-index, report (train_cindex, test_cindex) at that alpha."""
    Xtr, Xva = _standardize(X[train_idx], X[val_idx])
    _, Xte = _standardize(X[train_idx], X[test_idx])
    y_tr = np.array(
        list(zip(event[train_idx].astype(bool), time[train_idx].astype(float))), dtype=[("event", "?"), ("time", "<f8")],
    )
    try:
        model = CoxnetSurvivalAnalysis(l1_ratio=float(l1_ratio), alpha_min_ratio=0.01, n_alphas=int(n_alphas), max_iter=8000)
        model.fit(Xtr, y_tr)
    except Exception as exc:
        return float("nan"), float("nan"), f"fit_failed: {exc}"
    coefs = model.coef_  # (n_features, n_alphas_fit)
    best_val, best_j = -np.inf, 0
    for j in range(coefs.shape[1]):
        risk_val = Xva @ coefs[:, j]
        if np.allclose(risk_val, 0):
            continue
        c = sds.concordance_index(risk_val, time[val_idx], event[val_idx])
        if c > best_val:
            best_val, best_j = c, j
    risk_tr = Xtr @ coefs[:, best_j]
    risk_te = Xte @ coefs[:, best_j]
    c_tr = sds.concordance_index(risk_tr, time[train_idx], event[train_idx])
    c_te = sds.concordance_index(risk_te, time[test_idx], event[test_idx])
    return float(c_tr), float(c_te), "ok"


def _fit_eval_dense_mlp(*, X, time, event, train_idx, val_idx, test_idx, args, seed):
    model = sds.make_seeded_mlp(
        input_dim=X.shape[1], hidden_dims=tuple(args.mlp_hidden_dims), dropout_p=float(args.mlp_dropout), seed=seed,
    ).to(args.device)
    cfg = sds.MLPTrainConfig(
        lr=float(args.mlp_lr), weight_decay=float(args.mlp_weight_decay), lambda_l1_input=float(args.mlp_l1_input),
        batch_size=int(args.batch_size), max_epochs=int(args.mlp_max_epochs), patience=int(args.mlp_patience),
    )
    sds.train_deepsurv_mlp_l1(
        model,
        sds.as_torch(X[train_idx]), sds.as_torch(time[train_idx]), sds.as_torch(event[train_idx]),
        sds.as_torch(X[val_idx]), sds.as_torch(time[val_idx]), sds.as_torch(event[val_idx]),
        config=cfg, device=args.device,
    )
    c_tr, _ = sds.eval_mlp_cindex(model, sds.as_torch(X[train_idx]), sds.as_torch(time[train_idx]), sds.as_torch(event[train_idx]), device=args.device)
    c_te, _ = sds.eval_mlp_cindex(model, sds.as_torch(X[test_idx]), sds.as_torch(time[test_idx]), sds.as_torch(event[test_idx]), device=args.device)
    return float(c_tr), float(c_te)


def _fit_eval_gated(*, X, time, event, train_idx, val_idx, test_idx, cfg, args, seed):
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
        max_epochs=int(args.max_epochs), seed=seed,
        gate_type=str(cfg["gate_type"]), gate_sigma=float(cfg["gate_sigma"]),
        lam=float(cfg["lambda_sparse"]), lambda_sample_smooth=float(cfg["lambda_sample_smooth"]),
        patience=int(args.patience), temperature=float(cfg["temperature"]),
        concrete_mode=str(cfg["concrete_mode"]), predictor=str(cfg["predictor"]),
        gating_hidden_dim=int(cfg["gating_hidden_dim"]), gate_hidden_dropout_p=float(cfg["gate_hidden_dropout_p"]),
        risk_hidden_dims=tuple(cfg["risk_hidden_dims"]), risk_dropout_p=float(cfg["risk_dropout_p"]),
        lspin_init_bias=float(cfg["lspin_init_bias"]), gate_weight_decay=float(cfg["gate_weight_decay"]),
    )
    tr = info.get("train_cindex")
    te = info.get("test_cindex")
    return (float(tr) if tr is not None else float("nan")), (float(te) if te is not None else float("nan"))


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (results_dir / f"pd_small_sample_learning_curves_{args.dataset}_{args.n_reps}rep")
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X, time, event, histo, genes = _combined_arrays(data)

    configs = _selected_configs(
        args.dataset, results_dir, args.families, [args.selection],
        lambda_scale_families=args.families, lambda_scale_selections=[args.selection],
        lambda_scales=[1.0], smooth_smooth_values=None,
    )

    rows = []
    for rep in range(int(args.n_reps)):
        base_seed = int(args.seed + 1000 * rep)
        ss = ShuffleSplit(n_splits=1, test_size=float(args.test_fraction), random_state=base_seed)
        train_all, test_idx = next(ss.split(X))
        ss2 = ShuffleSplit(n_splits=1, test_size=float(args.val_fraction), random_state=base_seed + 1)
        rel_train, rel_val = next(ss2.split(train_all))
        train_full, val_idx = train_all[rel_train], train_all[rel_val]

        for frac in args.fractions:
            print(f"[rep {rep}] fraction={frac}", flush=True)
            try:
                train_idx = subsample_training_fold(train_full, event, float(frac), base_seed + 13, args.min_events_train)
            except ValueError as exc:
                print(f"[rep {rep}] fraction={frac} SKIPPED: {exc}", flush=True)
                continue

            c_tr, c_te = _fit_eval_dense_mlp(X=X, time=time, event=event, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, args=args, seed=base_seed)
            rows.append({"dataset": args.dataset, "rep": rep, "fraction": frac, "n_train": len(train_idx), "model": "dense_mlp", "train_cindex": c_tr, "test_cindex": c_te})
            print(f"[rep {rep}] fraction={frac} dense_mlp train={c_tr:.4f} test={c_te:.4f}", flush=True)

            c_tr, c_te, status = _fit_eval_coxnet(X=X, time=time, event=event, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, l1_ratio=args.coxnet_l1_ratio, n_alphas=args.coxnet_n_alphas)
            rows.append({"dataset": args.dataset, "rep": rep, "fraction": frac, "n_train": len(train_idx), "model": f"coxnet_l1ratio{args.coxnet_l1_ratio:g}", "train_cindex": c_tr, "test_cindex": c_te})
            print(f"[rep {rep}] fraction={frac} coxnet train={c_tr:.4f} test={c_te:.4f} ({status})", flush=True)

            for _, cfg in configs.iterrows():
                label = f"{cfg['family']}_{cfg['selection']}"
                c_tr, c_te = _fit_eval_gated(X=X, time=time, event=event, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, cfg=cfg, args=args, seed=base_seed)
                rows.append({"dataset": args.dataset, "rep": rep, "fraction": frac, "n_train": len(train_idx), "model": label, "train_cindex": c_tr, "test_cindex": c_te})
                print(f"[rep {rep}] fraction={frac} {label} train={c_tr:.4f} test={c_te:.4f}", flush=True)

    result = pd.DataFrame(rows)
    result["generalization_gap"] = result["train_cindex"] - result["test_cindex"]
    result.to_csv(outdir / "learning_curves_raw.csv", index=False)

    summary = (
        result.groupby(["model", "fraction"])[["train_cindex", "test_cindex", "generalization_gap"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(outdir / "learning_curves_summary.csv", index=False)

    print(summary.to_string(), flush=True)
    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
