#!/usr/bin/env python3
"""Held-out C-index degradation under injected noise features: gated-sparse
vs. dense MLP.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 4).

Current framing of sparsity's value is "doesn't cost much predictive
performance" (see `extras/archive/docs/SUMMARY_KEY_FINDINGS.md`) -- sparse
gated models land at or below the dense MLP reference on both KIPAN and
BRCA C-index. Injected-noise-feature robustness is the standard
experiment in the instance-wise-selection literature (INVASE, LSPIN,
Concrete Autoencoder papers) for turning "sparsity doesn't hurt" into
"sparsity actively helps once the feature space is not curated," which is
the realistic genomic-panel-design setting this method is pitched at.

Plan: append k synthetic noise columns (permuted real columns -- each
noise column is an independent random permutation of one real gene's
values across patients, which preserves realistic marginal scale/shape
while destroying any association with outcome) to the existing
KIPAN/BRCA feature matrices, sweep k, and compare held-out C-index
degradation of a dense MLP vs. LSPIN vs. Concrete (both using their
already-tuned selected hyperparameters from
`selected_comparison_configs.csv`, reused via the same
`_selected_configs` helper the other pd_ scripts use, rather than
re-tuning lambda at every noise level).

MLP+STG is deferred from this first pass: it has a bespoke training loop
in this repo (not the shared `run_one_model`/`train_deepsurv_mlp_l1`
helpers reused here) and would need that ported in separately; the
dense-MLP-vs-gated-models comparison already tests the core claim.

Usage:
    python pd_noise_robustness_sweep.py --dataset kipan --device cuda:2
    python pd_noise_robustness_sweep.py --dataset brca --device cuda:5
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
    p.add_argument("--noise-counts", type=int, nargs="+", default=[0, 200, 1000])
    p.add_argument("--families", nargs="+", default=["LSPIN", "Concrete"], choices=["LSPIN", "Concrete"])
    p.add_argument("--selection", choices=["nosmooth", "smooth"], default="smooth")
    p.add_argument("--n-reps", type=int, default=3)
    p.add_argument("--test-fraction", type=float, default=0.30)
    p.add_argument("--val-fraction", type=float, default=0.15)
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
    p.add_argument("--mlp-l1-input", type=float, default=0.0, help="0 = truly unregularized dense reference.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def augment_with_noise_columns(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Append k noise columns: independent random permutations of k
    randomly chosen real columns (with replacement across choices),
    applied identically across all rows of X so it can be called once per
    split-defining rng draw and reused consistently for train/val/test
    subsets of the same larger matrix (call this on the full combined X
    before splitting, not separately per split)."""
    if k <= 0:
        return X
    n, d = X.shape
    src_cols = rng.integers(0, d, size=k)
    noise = np.empty((n, k), dtype=X.dtype)
    for j, col in enumerate(src_cols):
        perm = rng.permutation(n)
        noise[:, j] = X[perm, col]
    return np.concatenate([X, noise], axis=1)


def _fit_dense_mlp(
    *, X, time, event, train_idx, val_idx, test_idx, args, seed,
) -> float:
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
    c_test, _ = sds.eval_mlp_cindex(
        model, sds.as_torch(X[test_idx]), sds.as_torch(time[test_idx]), sds.as_torch(event[test_idx]), device=args.device,
    )
    return float(c_test)


def _fit_gated(
    *, X, time, event, train_idx, val_idx, test_idx, cfg, args, seed,
) -> float:
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
    tc = info.get("test_cindex")
    return float(tc) if tc is not None else float("nan")


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (results_dir / f"pd_noise_robustness_{args.dataset}_{args.n_reps}rep")
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X_base, time, event, histo, genes = _combined_arrays(data)

    configs = _selected_configs(
        args.dataset, results_dir, args.families, [args.selection],
        lambda_scale_families=args.families, lambda_scale_selections=[args.selection],
        lambda_scales=[1.0], smooth_smooth_values=None,
    )

    rows = []
    for rep in range(int(args.n_reps)):
        base_seed = int(args.seed + 1000 * rep)
        rng = np.random.default_rng(base_seed)

        ss = ShuffleSplit(n_splits=1, test_size=float(args.test_fraction), random_state=base_seed)
        train_all, test_idx = next(ss.split(X_base))
        ss2 = ShuffleSplit(n_splits=1, test_size=float(args.val_fraction), random_state=base_seed + 1)
        rel_train, rel_val = next(ss2.split(train_all))
        train_idx, val_idx = train_all[rel_train], train_all[rel_val]

        for k in args.noise_counts:
            print(f"[rep {rep}] noise_k={k}", flush=True)
            noise_rng = np.random.default_rng(base_seed + 7919 * (k + 1))
            X = augment_with_noise_columns(X_base, k, noise_rng)

            c_mlp = _fit_dense_mlp(
                X=X, time=time, event=event, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                args=args, seed=base_seed,
            )
            print(f"[rep {rep}] noise_k={k} dense_mlp test_c={c_mlp:.4f}", flush=True)
            rows.append({"dataset": args.dataset, "rep": rep, "noise_k": k, "model": "dense_mlp", "test_cindex": c_mlp})

            for _, cfg in configs.iterrows():
                label = f"{cfg['family']}_{cfg['selection']}"
                c = _fit_gated(
                    X=X, time=time, event=event, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                    cfg=cfg, args=args, seed=base_seed,
                )
                print(f"[rep {rep}] noise_k={k} {label} test_c={c:.4f}", flush=True)
                rows.append({"dataset": args.dataset, "rep": rep, "noise_k": k, "model": label, "test_cindex": c})

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "noise_robustness_raw.csv", index=False)

    summary = (
        result.groupby(["model", "noise_k"])["test_cindex"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["model", "noise_k"])
    )
    # Degradation relative to that model's own k=0 mean, so the comparison
    # is about slope (robustness), not absolute level.
    baseline = summary[summary["noise_k"] == 0].set_index("model")["mean"]
    summary["delta_from_k0"] = summary.apply(lambda r: r["mean"] - baseline.get(r["model"], np.nan), axis=1)
    summary.to_csv(outdir / "noise_robustness_summary.csv", index=False)

    print(summary.to_string(), flush=True)
    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
