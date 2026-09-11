#!/usr/bin/env python3
"""Corrected-for-chance feature-selection stability index, LSPIN/Concrete
vs. sparse linear baselines.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 2).

The current internal framing of run-to-run reproducibility
(`COLLABORATOR_GUIDE.md` section 6) is informal: "overlap is greater than
random chance, but not strongly reproducible at the individual-gene
level." That invites the reviewer question "compared to what, exactly,
and by how much?" This script answers it with the standard
corrected-for-chance stability estimator from Nogueira, Sechidis & Brown,
"On the Stability of Feature Selection Algorithms" (JMLR 2018):

    hat_Phi = 1 - [ (1/d) sum_f hat_s_f^2 ] / [ kbar_d * (1 - kbar_d) ]

where, over M selection runs each choosing a binary vector over d
features: hat_p_f is the empirical selection frequency of feature f,
hat_s_f^2 = M/(M-1) * hat_p_f * (1 - hat_p_f) is the unbiased sample
variance of that Bernoulli frequency, and kbar_d is the average fraction
of features selected per run. hat_Phi = 0 for selection no more
consistent than chance at the same average sparsity, 1 for perfectly
consistent selection.

Design: bootstrap-resample the full patient cohort (with replacement) M
times. For each resample, fit a model and record its selected feature set
evaluated on that same resampled cohort (no separate holdout needed --
this measures selection consistency across independent fits, not
predictive transfer, which is what plot_kipan_feature_set_cv_transport.py
and the interchangeability test already cover):

    - LSPIN / Concrete: hard-gate rate over the bootstrap sample >=
      --gate-rate-threshold (default 0.10, matching the threshold
      convention in plot_kipan_feature_set_cv_transport.py)
    - Lasso-Cox / elastic-net-Cox baselines (sksurv CoxnetSurvivalAnalysis
      regularization path): pick the alpha along each fit's own path
      whose nonzero-coefficient count is closest to the gated models'
      average selected count, so the comparison is at matched sparsity
      rather than confounded by selected-set size
    - a random-selection null: for each bootstrap run, a uniformly random
      selection of the same size as that run's actual selection, to
      confirm hat_Phi lands near 0 for a method with no real selection
      signal

Confidence intervals: the paper gives a closed-form asymptotic variance
(its Theorem/eq. for hat_Phi's variance), but this implementation uses a
simpler, still standard substitute -- a bootstrap-of-bootstraps: resample
the M runs' selection vectors with replacement `--n-ci-resamples` times
and take the empirical percentile CI of hat_Phi. This is a pragmatic
choice, not the paper's exact estimator; note this in any manuscript text
that cites the CI.

Usage:
    python pd_selection_stability_index.py --dataset kipan --gate-family lspin --device cuda:2
    python pd_selection_stability_index.py --dataset kipan --gate-family concrete --device cuda:2
    python pd_selection_stability_index.py --dataset brca --gate-family lspin --device cuda:5
    python pd_selection_stability_index.py --dataset brca --gate-family concrete --device cuda:5
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

from _paths import ANALYSES_DIR, CANONICAL_RUNS, PROCESSED_DATASETS, ensure_repo_imports

ensure_repo_imports(include_analyses_dir=True)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis

import sparsedeepsurv as sds
from quick_linear_gated_probe import _combined_arrays, _load_data, _selected_configs, _split_train_val

RUN_DEFAULTS = {
    "kipan": CANONICAL_RUNS["kipan_adaptive_gentle"],
    "brca": CANONICAL_RUNS["brca_adaptive_gentle"],
}
DATA_DEFAULTS = PROCESSED_DATASETS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kipan", "brca"], default="kipan")
    p.add_argument("--gate-family", choices=["LSPIN", "Concrete"], default="LSPIN")
    p.add_argument("--selection", choices=["nosmooth", "smooth"], default="smooth")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--n-bootstrap", type=int, default=15)
    p.add_argument("--gate-rate-threshold", type=float, default=0.10)
    p.add_argument("--n-ci-resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260407)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--l1-ratio", type=float, default=0.9, help="Elastic-net mixing for the Coxnet baseline.")
    return p.parse_args()


def nogueira_brown_stability(selection_matrix: np.ndarray) -> float:
    """Point estimate of the Nogueira & Brown (2018) stability index.

    Parameters
    ----------
    selection_matrix : array, shape (n_runs, n_features), binary.

    Returns
    -------
    Stability index in [~0, 1] (can dip slightly negative for
    anti-correlated selections, per the original estimator).
    """
    Z = np.asarray(selection_matrix, dtype=float)
    m, d = Z.shape
    if m < 2:
        return float("nan")
    p_hat = Z.mean(axis=0)
    s2_hat = (m / (m - 1)) * p_hat * (1.0 - p_hat)
    k_bar_d = Z.sum(axis=1).mean() / d
    denom = k_bar_d * (1.0 - k_bar_d)
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - s2_hat.mean() / denom)


def stability_with_ci(
    selection_matrix: np.ndarray, *, n_resamples: int, rng: np.random.Generator
) -> tuple[float, tuple[float, float]]:
    point = nogueira_brown_stability(selection_matrix)
    m = selection_matrix.shape[0]
    if m < 4:
        return point, (float("nan"), float("nan"))
    draws = []
    for _ in range(int(n_resamples)):
        idx = rng.integers(0, m, size=m)
        draws.append(nogueira_brown_stability(selection_matrix[idx]))
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if len(draws) < 10:
        return point, (float("nan"), float("nan"))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, (float(lo), float(hi))


def _bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def _gated_bootstrap_selection(
    *,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    cfg: pd.Series,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """One bootstrap fit of the gated model; returns (binary selection
    vector over all genes, k selected)."""
    n = X.shape[0]
    boot_idx = _bootstrap_indices(n, rng)
    train_idx, val_idx = _split_train_val(np.arange(len(boot_idx)), int(rng.integers(0, 2**31 - 1)))
    Xb, tb, eb = X[boot_idx], time[boot_idx], event[boot_idx]

    Xt_tr = sds.as_torch(Xb[train_idx])
    A = None
    if float(cfg["lambda_sample_smooth"]) > 0:
        A = sds.build_knn_adjacency_csr(Xt_tr, k=int(args.knn_k), pca_dim=50, metric="cosine", symmetrize=True)

    model, info = sds.run_one_model(
        Xt_tr=Xt_tr, tt_tr=sds.as_torch(tb[train_idx]), et_tr=sds.as_torch(eb[train_idx]),
        Xt_val=sds.as_torch(Xb[val_idx]), tt_val=sds.as_torch(tb[val_idx]), et_val=sds.as_torch(eb[val_idx]),
        Xt_test=sds.as_torch(Xb[val_idx]), tt_test=sds.as_torch(tb[val_idx]), et_test=sds.as_torch(eb[val_idx]),
        input_dim=X.shape[1], A_sample_train=A, device=args.device,
        lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs), seed=int(rng.integers(0, 2**31 - 1)),
        gate_type=str(cfg["gate_type"]), gate_sigma=float(cfg["gate_sigma"]),
        lam=float(cfg["lambda_sparse"]), lambda_sample_smooth=float(cfg["lambda_sample_smooth"]),
        patience=int(args.patience), temperature=float(cfg["temperature"]),
        concrete_mode=str(cfg["concrete_mode"]), predictor=str(cfg["predictor"]),
        gating_hidden_dim=int(cfg["gating_hidden_dim"]), gate_hidden_dropout_p=float(cfg["gate_hidden_dropout_p"]),
        risk_hidden_dims=tuple(cfg["risk_hidden_dims"]), risk_dropout_p=float(cfg["risk_dropout_p"]),
        lspin_init_bias=float(cfg["lspin_init_bias"]), gate_weight_decay=float(cfg["gate_weight_decay"]),
    )
    _, _, hard_t, _ = sds.get_gates(
        model, sds.as_torch(Xb), device=args.device, hard_threshold=float(args.hard_threshold), batch_size=512,
    )
    hard = hard_t.numpy().astype(np.float32)
    rate = hard.mean(axis=0)
    selected = (rate >= float(args.gate_rate_threshold)).astype(int)
    return selected, int(selected.sum())


def _coxnet_bootstrap_selection(
    *,
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    target_k: int,
    l1_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One bootstrap fit of a Coxnet path; returns the binary selection
    vector (nonzero coefficients) at the alpha closest to target_k
    selected features."""
    n = X.shape[0]
    boot_idx = _bootstrap_indices(n, rng)
    Xb = X[boot_idx]
    mu, sd = Xb.mean(axis=0), Xb.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xb = (Xb - mu) / sd
    y = np.array(
        list(zip(event[boot_idx].astype(bool), time[boot_idx].astype(float))),
        dtype=[("event", "?"), ("time", "<f8")],
    )
    try:
        model = CoxnetSurvivalAnalysis(l1_ratio=float(l1_ratio), alpha_min_ratio=0.01, n_alphas=50, max_iter=8000)
        model.fit(Xb, y)
    except Exception:
        return np.zeros(X.shape[1], dtype=int)
    coefs = model.coef_  # shape (n_features, n_alphas)
    nnz_per_alpha = (np.abs(coefs) > 1e-10).sum(axis=0)
    best_alpha_idx = int(np.argmin(np.abs(nnz_per_alpha - max(1, target_k))))
    selected = (np.abs(coefs[:, best_alpha_idx]) > 1e-10).astype(int)
    return selected


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (
        results_dir
        / f"pd_selection_stability_{args.dataset}_{args.gate_family.lower()}_{args.selection}_{args.n_bootstrap}boot"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X, time, event, histo, genes = _combined_arrays(data)
    n_genes = X.shape[1]

    configs = _selected_configs(
        args.dataset, results_dir, [args.gate_family], [args.selection],
        lambda_scale_families=[args.gate_family], lambda_scale_selections=[args.selection],
        lambda_scales=[1.0], smooth_smooth_values=None,
    )
    cfg = configs.iloc[0]

    rng = np.random.default_rng(int(args.seed))
    gated_rows = []
    ks = []
    for b in range(int(args.n_bootstrap)):
        print(f"[gated boot {b+1}/{args.n_bootstrap}] {args.gate_family} {args.selection}", flush=True)
        selected, k = _gated_bootstrap_selection(X=X, time=time, event=event, cfg=cfg, args=args, rng=rng)
        gated_rows.append(selected)
        ks.append(k)
        print(f"[gated boot {b+1}/{args.n_bootstrap}] selected k={k}", flush=True)
    gated_matrix = np.vstack(gated_rows)
    k_bar = int(round(float(np.mean(ks)))) if ks else 10

    coxnet_rows = []
    for b in range(int(args.n_bootstrap)):
        print(f"[coxnet boot {b+1}/{args.n_bootstrap}] target_k={k_bar}", flush=True)
        coxnet_rows.append(
            _coxnet_bootstrap_selection(X=X, time=time, event=event, target_k=k_bar, l1_ratio=args.l1_ratio, rng=rng)
        )
    coxnet_matrix = np.vstack(coxnet_rows)

    random_rows = []
    for k in ks:
        sel = np.zeros(n_genes, dtype=int)
        idx = rng.choice(n_genes, size=max(1, k), replace=False)
        sel[idx] = 1
        random_rows.append(sel)
    random_matrix = np.vstack(random_rows)

    results = []
    for label, matrix in [
        (f"{args.gate_family}_{args.selection}", gated_matrix),
        (f"coxnet_l1ratio{args.l1_ratio:g}_matched_k{k_bar}", coxnet_matrix),
        (f"random_matched_k", random_matrix),
    ]:
        point, (lo, hi) = stability_with_ci(matrix, n_resamples=int(args.n_ci_resamples), rng=rng)
        results.append({
            "dataset": args.dataset, "method": label, "n_bootstrap": matrix.shape[0],
            "n_genes": n_genes, "mean_k_selected": float(matrix.sum(axis=1).mean()),
            "stability_index": point, "ci_low": lo, "ci_high": hi,
        })

    summary = pd.DataFrame(results)
    summary.to_csv(outdir / "stability_index_summary.csv", index=False)

    np.save(outdir / "gated_selection_matrix.npy", gated_matrix)
    np.save(outdir / "coxnet_selection_matrix.npy", coxnet_matrix)
    np.save(outdir / "random_selection_matrix.npy", random_matrix)
    pd.Series(genes).to_csv(outdir / "gene_names.csv", index=False, header=["gene"])

    print(summary.to_string(), flush=True)
    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
