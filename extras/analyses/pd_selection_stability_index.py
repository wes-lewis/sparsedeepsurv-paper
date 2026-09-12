#!/usr/bin/env python3
"""Stability of *per-patient* feature selection, LSPIN/Concrete vs.
sparse linear baselines.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 2).

REVISION NOTE (2026-09-11, second pass): the first version of this
script measured stability of the *aggregate/consensus* gene set (which
genes clear a population-level gate-rate threshold), because that is the
standard unit of comparison for a global sparse-linear baseline like
Lasso-Cox. Feedback on that result was correct: this model's actual claim
is about *individual* (per-patient) selection, and aggregate-set
stability doesn't test that -- a method can have a wildly unstable
aggregate union while each patient's own personal selection is highly
consistent, or vice versa. It also surfaced that the aggregate selected
sets were much larger than expected (~25-30% of genes for KIPAN LSPIN at
a 10% gate-rate threshold), which is itself a symptom of per-patient
heterogeneity inflating the union, not evidence about any single
patient's sparsity.

This version replaces that design with three things run in sequence,
each informing the next rather than a single number treated as final:

  Step 0 (diagnostic): fit one reference model at the current tuned
  hyperparameters and check the soft (pre-threshold) gate distribution on
  a held-out cohort for the "gates stuck near 0.5" failure mode flagged
  in `COLLABORATOR_GUIDE.md` -- gates sitting close to the hard threshold
  would flip between independent fits from noise alone, which would
  directly explain low reproducibility independent of anything about the
  method's ceiling.

  Step 1 (per-patient stability at current tuned hyperparameters): fix a
  held-out evaluation cohort ONCE, then fit M independent models on
  bootstrap resamples of the (disjoint) training pool and evaluate hard
  gates on that same fixed cohort each time. For each held-out patient,
  compute the Nogueira-Brown stability index of *their own* selected-gene
  vector across the M fits (not the population union), against a
  size-matched per-patient random null. This is the metric that actually
  answers "if I retrain, does this patient's personalized feature set
  stay the same" -- the real personalization-reproducibility question.
  The old aggregate-level metric is still computed and reported
  alongside, labeled explicitly as a different, coarser quantity (and the
  only one that's meaningfully comparable to a global method like
  Coxnet, which has no notion of "this patient's own selection").

  Step 2 (hyperparameter sweep, not just the paper-selected point):
  lambda and gate_sigma were tuned in this repo for predictive C-index
  (see `PERFORMANCE_RECOVERY_ANALYSIS.md`), not for selection stability,
  and step 0/1 may show gates sitting near the threshold at that
  operating point. Sweep a small grid of lambda/gate_sigma multipliers
  (fewer bootstrap reps per cell than step 1, to bound compute) and
  report per-patient stability, median per-patient selected count, and
  test C-index together, so a stability/sparsity/accuracy trade-off (if
  one exists) is visible rather than asserted from a single run.

Usage:
    python pd_selection_stability_index.py --dataset kipan --gate-family LSPIN --device cuda:2
    python pd_selection_stability_index.py --dataset kipan --gate-family LSPIN --device cuda:2 --run-sweep
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
from sklearn.model_selection import ShuffleSplit
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
    p.add_argument("--n-bootstrap", type=int, default=15, help="Bootstrap fits for step 1.")
    p.add_argument("--eval-fraction", type=float, default=0.25, help="Fixed held-out eval cohort, never resampled.")
    p.add_argument("--aggregate-gate-rate-threshold", type=float, default=0.10, help="For the coarser aggregate-set metric only.")
    p.add_argument("--n-ci-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260407)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--near-threshold-band", type=float, default=0.1, help="Step 0: flag |g_det-0.5|<band as 'stuck'.")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--l1-ratio", type=float, default=0.9)
    p.add_argument("--run-sweep", action="store_true", help="Also run step 2 (lambda/sigma sweep).")
    p.add_argument("--sweep-lambda-multipliers", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--sweep-sigma-multipliers", type=float, nargs="+", default=[1.0, 0.5])
    p.add_argument("--sweep-n-bootstrap", type=int, default=8)
    return p.parse_args()


def nogueira_brown_stability(selection_matrix: np.ndarray) -> float:
    """Point estimate of the Nogueira & Brown (2018) stability index.
    selection_matrix: (n_runs, n_items) binary."""
    Z = np.asarray(selection_matrix, dtype=float)
    m, d = Z.shape
    if m < 2 or d == 0:
        return float("nan")
    p_hat = Z.mean(axis=0)
    s2_hat = (m / (m - 1)) * p_hat * (1.0 - p_hat)
    k_bar_d = Z.sum(axis=1).mean() / d
    denom = k_bar_d * (1.0 - k_bar_d)
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - s2_hat.mean() / denom)


def stability_with_ci(selection_matrix: np.ndarray, *, n_resamples: int, rng: np.random.Generator):
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


def _fit_gated_and_eval_on_fixed_cohort(
    *, X, time, event, pool_idx, eval_idx, cfg, args, rng,
) -> tuple[np.ndarray, float]:
    """Bootstrap-resample the trainable pool, fit, evaluate hard gates on
    the FIXED eval cohort (never resampled). Returns (hard_gates_on_eval
    shape (n_eval, n_genes), test_cindex_on_eval)."""
    boot_pool = rng.choice(pool_idx, size=len(pool_idx), replace=True)
    train_rel, val_rel = _split_train_val(np.arange(len(boot_pool)), int(rng.integers(0, 2**31 - 1)))
    train_idx, val_idx = boot_pool[train_rel], boot_pool[val_rel]

    Xt_tr = sds.as_torch(X[train_idx])
    A = None
    if float(cfg["lambda_sample_smooth"]) > 0:
        A = sds.build_knn_adjacency_csr(Xt_tr, k=int(args.knn_k), pca_dim=50, metric="cosine", symmetrize=True)

    model, info = sds.run_one_model(
        Xt_tr=Xt_tr, tt_tr=sds.as_torch(time[train_idx]), et_tr=sds.as_torch(event[train_idx]),
        Xt_val=sds.as_torch(X[val_idx]), tt_val=sds.as_torch(time[val_idx]), et_val=sds.as_torch(event[val_idx]),
        Xt_test=sds.as_torch(X[eval_idx]), tt_test=sds.as_torch(time[eval_idx]), et_test=sds.as_torch(event[eval_idx]),
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
    _, g_det_t, hard_t, _ = sds.get_gates(
        model, sds.as_torch(X[eval_idx]), device=args.device, hard_threshold=float(args.hard_threshold), batch_size=512,
    )
    hard = hard_t.numpy().astype(np.float32)
    tc = info.get("test_cindex")
    return hard, (float(tc) if tc is not None else float("nan")), g_det_t.numpy()


def per_patient_stability(hard_runs: np.ndarray, rng: np.random.Generator):
    """hard_runs: (M, n_eval, n_genes) binary. Returns per-patient
    stability, per-patient median k, and a matched-size random null
    (same shapes as the per-patient outputs, minus the null being one
    array of stability values)."""
    M, n_eval, n_genes = hard_runs.shape
    stab = np.full(n_eval, np.nan)
    med_k = np.full(n_eval, np.nan)
    null_stab = np.full(n_eval, np.nan)
    for i in range(n_eval):
        Z = hard_runs[:, i, :]
        stab[i] = nogueira_brown_stability(Z)
        med_k[i] = float(np.median(Z.sum(axis=1)))
        # Matched-size random null: for each run, a random selection of
        # the SAME size as that run's real selection for this patient.
        null_Z = np.zeros_like(Z)
        for b in range(M):
            k = int(Z[b].sum())
            if k > 0:
                idx = rng.choice(n_genes, size=k, replace=False)
                null_Z[b, idx] = 1
        null_stab[i] = nogueira_brown_stability(null_Z)
    return stab, med_k, null_stab


def aggregate_stability(hard_runs: np.ndarray, threshold: float):
    """The coarser population-level metric from the first version of this
    script, kept for comparison against Coxnet (a method with no
    per-patient notion of selection). hard_runs: (M, n_eval, n_genes)."""
    M = hard_runs.shape[0]
    rate_per_run = hard_runs.mean(axis=1)  # (M, n_genes): gate rate over eval cohort, per run
    agg_selection = (rate_per_run >= threshold).astype(float)  # (M, n_genes)
    return agg_selection


def _diagnose_gate_saturation(g_det: np.ndarray, band: float) -> dict:
    frac_stuck = float((np.abs(g_det - 0.5) < band).mean())
    return {
        "frac_gates_near_0.5": frac_stuck,
        "gate_prob_mean": float(g_det.mean()),
        "gate_prob_p10": float(np.percentile(g_det, 10)),
        "gate_prob_p50": float(np.percentile(g_det, 50)),
        "gate_prob_p90": float(np.percentile(g_det, 90)),
    }


def _coxnet_bootstrap_selection(*, X, time, event, pool_idx, target_k, l1_ratio, rng):
    boot_idx = rng.choice(pool_idx, size=len(pool_idx), replace=True)
    Xb = X[boot_idx]
    mu, sd = Xb.mean(axis=0), Xb.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Xb = (Xb - mu) / sd
    y = np.array(list(zip(event[boot_idx].astype(bool), time[boot_idx].astype(float))), dtype=[("event", "?"), ("time", "<f8")])
    try:
        model = CoxnetSurvivalAnalysis(l1_ratio=float(l1_ratio), alpha_min_ratio=0.01, n_alphas=50, max_iter=8000)
        model.fit(Xb, y)
    except Exception:
        return np.zeros(X.shape[1], dtype=int)
    coefs = model.coef_
    nnz = (np.abs(coefs) > 1e-10).sum(axis=0)
    best_j = int(np.argmin(np.abs(nnz - max(1, target_k))))
    return (np.abs(coefs[:, best_j]) > 1e-10).astype(int)


def run_stability_pass(*, X, time, event, pool_idx, eval_idx, cfg, args, n_bootstrap, rng):
    """One full pass of step-1-style bootstrapping: returns per-patient
    stats, aggregate selection matrix, and mean test C-index."""
    hard_runs = []
    cindices = []
    for b in range(n_bootstrap):
        hard, c, _ = _fit_gated_and_eval_on_fixed_cohort(
            X=X, time=time, event=event, pool_idx=pool_idx, eval_idx=eval_idx, cfg=cfg, args=args, rng=rng,
        )
        hard_runs.append(hard)
        cindices.append(c)
        print(f"  [boot {b+1}/{n_bootstrap}] eval_test_c={c:.4f} mean_k_this_run={hard.sum(axis=1).mean():.1f}", flush=True)
    hard_runs = np.stack(hard_runs, axis=0)
    stab, med_k, null_stab = per_patient_stability(hard_runs, rng)
    agg_selection = aggregate_stability(hard_runs, args.aggregate_gate_rate_threshold)
    return {
        "per_patient_stability_median": float(np.nanmedian(stab)),
        "per_patient_stability_iqr_low": float(np.nanpercentile(stab, 25)),
        "per_patient_stability_iqr_high": float(np.nanpercentile(stab, 75)),
        "per_patient_null_stability_median": float(np.nanmedian(null_stab)),
        "per_patient_median_k": float(np.nanmedian(med_k)),
        "aggregate_stability": nogueira_brown_stability(agg_selection),
        "aggregate_mean_k": float(agg_selection.sum(axis=1).mean()),
        "mean_test_cindex": float(np.nanmean(cindices)),
        "n_bootstrap": n_bootstrap,
    }, hard_runs


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (
        results_dir / f"pd_selection_stability_v2_{args.dataset}_{args.gate_family.lower()}_{args.selection}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X, time, event, histo, genes = _combined_arrays(data)
    n = X.shape[0]

    ss = ShuffleSplit(n_splits=1, test_size=float(args.eval_fraction), random_state=int(args.seed))
    pool_idx, eval_idx = next(ss.split(X))
    print(f"[setup] pool_n={len(pool_idx)} eval_n={len(eval_idx)} (eval cohort fixed for all bootstrap fits)", flush=True)

    configs = _selected_configs(
        args.dataset, results_dir, [args.gate_family], [args.selection],
        lambda_scale_families=[args.gate_family], lambda_scale_selections=[args.selection],
        lambda_scales=[1.0], smooth_smooth_values=None,
    )
    base_cfg = configs.iloc[0]
    rng = np.random.default_rng(int(args.seed))

    # --- Step 0: gate-saturation diagnostic on a single reference fit ---
    print("\n[step 0] gate-saturation diagnostic (single reference fit)", flush=True)
    _, ref_c, ref_g_det = _fit_gated_and_eval_on_fixed_cohort(
        X=X, time=time, event=event, pool_idx=pool_idx, eval_idx=eval_idx, cfg=base_cfg, args=args, rng=rng,
    )
    diag = _diagnose_gate_saturation(ref_g_det, args.near_threshold_band)
    diag["reference_fit_test_cindex"] = ref_c
    pd.DataFrame([diag]).to_csv(outdir / "step0_gate_saturation_diagnostic.csv", index=False)
    print(f"[step 0] {diag}", flush=True)

    # --- Step 1: per-patient stability at current tuned hyperparameters ---
    print(f"\n[step 1] per-patient stability at tuned config (M={args.n_bootstrap})", flush=True)
    step1_summary, hard_runs_base = run_stability_pass(
        X=X, time=time, event=event, pool_idx=pool_idx, eval_idx=eval_idx, cfg=base_cfg, args=args,
        n_bootstrap=int(args.n_bootstrap), rng=rng,
    )
    step1_summary.update({"dataset": args.dataset, "gate_family": args.gate_family, "selection": args.selection,
                           "lambda_multiplier": 1.0, "sigma_multiplier": 1.0})
    print(f"[step 1] {step1_summary}", flush=True)

    # Coxnet reference (aggregate-only; no per-patient notion).
    print("\n[step 1b] Coxnet aggregate-selection reference", flush=True)
    target_k = int(round(step1_summary["aggregate_mean_k"]))
    coxnet_rows = []
    for b in range(int(args.n_bootstrap)):
        coxnet_rows.append(_coxnet_bootstrap_selection(
            X=X, time=time, event=event, pool_idx=pool_idx, target_k=target_k, l1_ratio=args.l1_ratio, rng=rng,
        ))
    coxnet_matrix = np.vstack(coxnet_rows)
    coxnet_agg_stability, (lo, hi) = stability_with_ci(coxnet_matrix, n_resamples=int(args.n_ci_resamples), rng=rng)
    coxnet_summary = {
        "dataset": args.dataset, "method": f"coxnet_matched_k{target_k}",
        "aggregate_stability": coxnet_agg_stability, "aggregate_stability_ci_low": lo, "aggregate_stability_ci_high": hi,
        "aggregate_mean_k": float(coxnet_matrix.sum(axis=1).mean()),
    }
    pd.DataFrame([coxnet_summary]).to_csv(outdir / "step1b_coxnet_aggregate_reference.csv", index=False)
    print(f"[step 1b] {coxnet_summary}", flush=True)

    pd.DataFrame([step1_summary]).to_csv(outdir / "step1_baseline_summary.csv", index=False)
    np.save(outdir / "step1_hard_runs.npy", hard_runs_base)

    # --- Step 2 (optional): lambda/sigma sweep for stability ---
    if args.run_sweep:
        print("\n[step 2] lambda/sigma sweep for stability", flush=True)
        sweep_rows = []
        for lam_mult in args.sweep_lambda_multipliers:
            for sigma_mult in args.sweep_sigma_multipliers:
                cfg = base_cfg.copy()
                cfg["lambda_sparse"] = float(base_cfg["lambda_sparse"]) * float(lam_mult)
                cfg["gate_sigma"] = float(base_cfg["gate_sigma"]) * float(sigma_mult)
                print(f"[sweep] lambda_mult={lam_mult} sigma_mult={sigma_mult} "
                      f"(lambda={cfg['lambda_sparse']:.5g}, sigma={cfg['gate_sigma']:.4g})", flush=True)
                summary, _ = run_stability_pass(
                    X=X, time=time, event=event, pool_idx=pool_idx, eval_idx=eval_idx, cfg=cfg, args=args,
                    n_bootstrap=int(args.sweep_n_bootstrap), rng=rng,
                )
                summary.update({"dataset": args.dataset, "gate_family": args.gate_family, "selection": args.selection,
                                 "lambda_multiplier": lam_mult, "sigma_multiplier": sigma_mult})
                print(f"[sweep result] {summary}", flush=True)
                sweep_rows.append(summary)
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(outdir / "step2_sweep_results.csv", index=False)
        print(sweep_df.to_string(), flush=True)

    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
