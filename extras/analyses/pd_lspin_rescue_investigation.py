#!/usr/bin/env python3
"""Investigate why LSPIN's per-patient selection stability lags Concrete's
on KIPAN, and try to rescue it with a targeted, mechanism-driven
hyperparameter search (not another blind grid).

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 2, "top priority" follow-up after the per-patient stability
redesign). Builds on `pd_selection_stability_index.py`, reusing its
Nogueira-Brown stability estimator and per-patient stability routine.

MECHANISTIC HYPOTHESIS (read the gate math before reading the sweep):

LSPIN (`gate_type="lspin_tf"`, `deepsurv_gated.py::_sample_lspin_tf`) uses
a hard-clamp deterministic gate:

    alpha = gating_net(x)                      # unbounded raw output
    g_det = clamp(a * alpha + 0.5, 0, 1)        # PIECEWISE LINEAR

and trains through a noised version of the same clamp:

    z = alpha + gate_sigma * eps,  eps ~ N(0,1)
    gate_sample = clamp(a * z + 0.5, 0, 1)

Concrete (`_sample_concrete`) instead uses a smooth Gumbel-sigmoid:

    gate_soft = sigmoid((logits + gumbel) / temperature)

The clamp has an exact, finite dead zone: once |alpha| exceeds the ramp
half-width (0.5/a with a=1.0, i.e. |alpha|>0.5), the clamp's gradient is
IDENTICALLY ZERO unless injected noise happens to push a sample back
into the ramp. Concrete's sigmoid gradient shrinks continuously in
saturation but is never exactly zero over a finite region. The
prediction: if `gate_sigma` is small relative to the ramp width (the
tuned KIPAN LSPIN value is small), genes whose alpha drifts past the
ramp early in training get almost no further corrective gradient --
which gene ends up "locked in" open or closed becomes a race decided by
early, noise-sensitive dynamics, differing between independent fits.
This predicts exactly what was observed: LSPIN gates are sharply
bimodal *within* a fit (confirmed by the thread-2 step-0 diagnostic --
not stuck near 0.5) but inconsistent *across* fits (near-zero per-patient
stability), and the aggregate selected set stays large (~20% of genes on
KIPAN) because genes that lock in early rarely get corrected back
toward zero by the sparsity penalty. This script tests that hypothesis
directly by (a) measuring dead-zone saturation fraction
(`|alpha| > 0.5/a`) alongside stability at each setting, and (b) sweeping
the three parameters the hypothesis says should matter: `gate_sigma`
(wider noise -> more escape chances), `lspin_init_bias` (less extreme
initial commitment -> less early lock-in), and `a` (narrower/wider ramp
in alpha-space -- requires bypassing the package's `make_model` helper,
which hardcodes a=1.0, so this script constructs `DeepSurvGated` and
calls `train_gated_deepsurv` directly instead).

Design (staged, not one big grid):

  Stage A: gate_sigma multipliers, WIDER than the original thread-2 sweep
  (which only tried 1x/0.5x -- narrower, the wrong direction under this
  hypothesis). Holds lambda/init_bias/a at baseline.
  Stage B: lspin_init_bias values, holding gate_sigma/lambda/a at
  baseline.
  Stage C: `a` values, holding gate_sigma/lambda/init_bias at baseline.
  Stage D: combine the best-by-median-per-patient-stability value from
  each of A/B/C into one run at higher n_bootstrap for a confident
  verdict, reported directly against the LSPIN baseline and against
  Concrete's already-measured KIPAN result (0.023 median, from
  `pd_selection_stability_v2_kipan_concrete_smooth/step1_baseline_summary.csv`).

Usage:
    python pd_lspin_rescue_investigation.py --dataset kipan --device cuda:2
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
from pd_selection_stability_index import nogueira_brown_stability, per_patient_stability
from quick_linear_gated_probe import _combined_arrays, _load_data, _selected_configs, _split_train_val

RUN_DEFAULTS = {
    "kipan": CANONICAL_RUNS["kipan_adaptive_gentle"],
    "brca": CANONICAL_RUNS["brca_adaptive_gentle"],
}
DATA_DEFAULTS = PROCESSED_DATASETS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kipan", "brca"], default="kipan")
    p.add_argument("--selection", choices=["nosmooth", "smooth"], default="smooth")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--eval-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=20260407)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--n-bootstrap-stage", type=int, default=6, help="Bootstrap reps for stages A/B/C.")
    p.add_argument("--n-bootstrap-final", type=int, default=15, help="Bootstrap reps for the final stage-D verdict.")
    p.add_argument("--sigma-multipliers", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    p.add_argument("--init-bias-values", type=float, nargs="+", default=[0.0, -1.0, -2.0, 1.0])
    p.add_argument("--a-values", type=float, nargs="+", default=[1.0, 0.5, 0.25, 2.0, 4.0])
    p.add_argument("--concrete-reference-csv", type=Path, default=None,
                    help="step1_baseline_summary.csv from the Concrete run, for the final comparison table.")
    return p.parse_args()


def _build_model_and_train(*, X, time, event, train_idx, val_idx, eval_idx, cfg, args, seed, a_override, device):
    """Construct DeepSurvGated + train_gated_deepsurv directly (bypassing
    make_model/run_one_model) so `a` can be overridden -- make_model
    hardcodes a=1.0."""
    sds.set_seed(int(seed))
    Xt_tr = sds.as_torch(X[train_idx])
    A = None
    if float(cfg["lambda_sample_smooth"]) > 0:
        A = sds.build_knn_adjacency_csr(Xt_tr, k=int(args.knn_k), pca_dim=50, metric="cosine", symmetrize=True)

    model = sds.DeepSurvGated(
        input_dim=X.shape[1], hidden_dim=64, gating_hidden_dim=int(cfg["gating_hidden_dim"]),
        temperature=float(cfg["temperature"]), llspin=False, dropout_p=0.0,
        gate_hidden_dropout_p=float(cfg["gate_hidden_dropout_p"]), gate_type=str(cfg["gate_type"]),
        concrete_mode=str(cfg["concrete_mode"]), gate_sigma=float(cfg["gate_sigma"]), a=float(a_override),
        predictor=str(cfg["predictor"]), risk_hidden_dims=tuple(cfg["risk_hidden_dims"]),
        risk_dropout_p=float(cfg["risk_dropout_p"]), lspin_init_bias=float(cfg["lspin_init_bias"]),
    ).to(device)

    gate_wd = cfg.get("gate_weight_decay", None)
    train_cfg = sds.GatedTrainConfig(
        lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs), patience=int(args.patience),
        lambda_sparse=float(cfg["lambda_sparse"]), lambda_sample_smooth=float(cfg["lambda_sample_smooth"]),
        lambda_gene_smooth=0.0,
        gate_weight_decay=(float(gate_wd) if gate_wd is not None else None),
    )
    info = sds.train_gated_deepsurv(
        model, Xt_tr, sds.as_torch(time[train_idx]), sds.as_torch(event[train_idx]),
        sds.as_torch(X[val_idx]), sds.as_torch(time[val_idx]), sds.as_torch(event[val_idx]),
        sds.as_torch(X[eval_idx]), sds.as_torch(time[eval_idx]), sds.as_torch(event[eval_idx]),
        config=train_cfg, A_sample_train=A, device=device, verbose=False,
    )
    reg_probs, g_det, hard, alpha = sds.get_gates(
        model, sds.as_torch(X[eval_idx]), device=device, hard_threshold=float(args.hard_threshold), batch_size=512,
    )
    return hard.numpy().astype(np.float32), alpha.numpy(), info.get("test_cindex")


def dead_zone_fraction(alpha: np.ndarray, a: float) -> float:
    """Fraction of (patient, gene) alpha values sitting outside the
    linear ramp -- where the LSPIN clamp gradient is exactly zero."""
    half_width = 0.5 / max(a, 1e-8)
    return float((np.abs(alpha) > half_width).mean())


def run_variant(
    *, X, time, event, pool_idx, eval_idx, base_cfg, args, n_bootstrap, rng,
    lambda_mult=1.0, sigma_mult=1.0, init_bias=None, a=1.0,
):
    cfg = base_cfg.copy()
    cfg["lambda_sparse"] = float(base_cfg["lambda_sparse"]) * float(lambda_mult)
    cfg["gate_sigma"] = float(base_cfg["gate_sigma"]) * float(sigma_mult)
    if init_bias is not None:
        cfg["lspin_init_bias"] = float(init_bias)

    hard_runs, alpha_runs, cindices = [], [], []
    for b in range(int(n_bootstrap)):
        boot_pool = rng.choice(pool_idx, size=len(pool_idx), replace=True)
        train_rel, val_rel = _split_train_val(np.arange(len(boot_pool)), int(rng.integers(0, 2**31 - 1)))
        train_idx, val_idx = boot_pool[train_rel], boot_pool[val_rel]
        hard, alpha, c = _build_model_and_train(
            X=X, time=time, event=event, train_idx=train_idx, val_idx=val_idx, eval_idx=eval_idx,
            cfg=cfg, args=args, seed=int(rng.integers(0, 2**31 - 1)), a_override=a, device=args.device,
        )
        hard_runs.append(hard)
        alpha_runs.append(alpha)
        cindices.append(c if c is not None else float("nan"))

    hard_runs = np.stack(hard_runs, axis=0)
    alpha_runs = np.stack(alpha_runs, axis=0)
    stab, med_k, null_stab = per_patient_stability(hard_runs, rng)
    dz_frac = np.mean([dead_zone_fraction(alpha_runs[b], a) for b in range(alpha_runs.shape[0])])

    return {
        "lambda_multiplier": lambda_mult, "sigma_multiplier": sigma_mult,
        "lspin_init_bias": cfg["lspin_init_bias"], "a": a,
        "gate_sigma_value": cfg["gate_sigma"], "lambda_sparse_value": cfg["lambda_sparse"],
        "per_patient_stability_median": float(np.nanmedian(stab)),
        "per_patient_stability_iqr_low": float(np.nanpercentile(stab, 25)),
        "per_patient_stability_iqr_high": float(np.nanpercentile(stab, 75)),
        "per_patient_null_stability_median": float(np.nanmedian(null_stab)),
        "per_patient_median_k": float(np.nanmedian(med_k)),
        "dead_zone_fraction": float(dz_frac),
        "mean_test_cindex": float(np.nanmean(cindices)),
        "n_bootstrap": int(n_bootstrap),
    }


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (results_dir / f"pd_lspin_rescue_{args.dataset}")
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X, time, event, histo, genes = _combined_arrays(data)

    ss = ShuffleSplit(n_splits=1, test_size=float(args.eval_fraction), random_state=int(args.seed))
    pool_idx, eval_idx = next(ss.split(X))
    print(f"[setup] pool_n={len(pool_idx)} eval_n={len(eval_idx)}", flush=True)

    configs = _selected_configs(
        args.dataset, results_dir, ["LSPIN"], [args.selection],
        lambda_scale_families=["LSPIN"], lambda_scale_selections=[args.selection],
        lambda_scales=[1.0], smooth_smooth_values=None,
    )
    base_cfg = configs.iloc[0]
    rng = np.random.default_rng(int(args.seed))
    all_rows = []

    print(f"\n[baseline] lambda_mult=1.0 sigma_mult=1.0 init_bias=baseline({base_cfg['lspin_init_bias']}) a=1.0", flush=True)
    baseline = run_variant(X=X, time=time, event=event, pool_idx=pool_idx, eval_idx=eval_idx, base_cfg=base_cfg,
                            args=args, n_bootstrap=args.n_bootstrap_stage, rng=rng)
    baseline["stage"] = "baseline"
    print(f"[baseline result] {baseline}", flush=True)
    all_rows.append(baseline)

    print("\n[stage A] gate_sigma sweep", flush=True)
    stage_a = []
    for sm in args.sigma_multipliers:
        r = run_variant(X=X, time=time, event=event, pool_idx=pool_idx, eval_idx=eval_idx, base_cfg=base_cfg,
                         args=args, n_bootstrap=args.n_bootstrap_stage, rng=rng, sigma_mult=sm)
        r["stage"] = "A_sigma"
        print(f"[stage A] sigma_mult={sm} -> {r}", flush=True)
        stage_a.append(r)
    all_rows.extend(stage_a)
    best_sigma_mult = max(stage_a, key=lambda r: r["per_patient_stability_median"])["sigma_multiplier"]

    print("\n[stage B] lspin_init_bias sweep", flush=True)
    stage_b = []
    for ib in args.init_bias_values:
        r = run_variant(X=X, time=time, event=event, pool_idx=pool_idx, eval_idx=eval_idx, base_cfg=base_cfg,
                         args=args, n_bootstrap=args.n_bootstrap_stage, rng=rng, init_bias=ib)
        r["stage"] = "B_init_bias"
        print(f"[stage B] init_bias={ib} -> {r}", flush=True)
        stage_b.append(r)
    all_rows.extend(stage_b)
    best_init_bias = max(stage_b, key=lambda r: r["per_patient_stability_median"])["lspin_init_bias"]

    print("\n[stage C] a sweep", flush=True)
    stage_c = []
    for a_val in args.a_values:
        r = run_variant(X=X, time=time, event=event, pool_idx=pool_idx, eval_idx=eval_idx, base_cfg=base_cfg,
                         args=args, n_bootstrap=args.n_bootstrap_stage, rng=rng, a=a_val)
        r["stage"] = "C_a"
        print(f"[stage C] a={a_val} -> {r}", flush=True)
        stage_c.append(r)
    all_rows.extend(stage_c)
    best_a = max(stage_c, key=lambda r: r["per_patient_stability_median"])["a"]

    print(f"\n[stage D] combined best: sigma_mult={best_sigma_mult} init_bias={best_init_bias} a={best_a}, "
          f"n_bootstrap={args.n_bootstrap_final}", flush=True)
    stage_d = run_variant(X=X, time=time, event=event, pool_idx=pool_idx, eval_idx=eval_idx, base_cfg=base_cfg,
                           args=args, n_bootstrap=args.n_bootstrap_final, rng=rng,
                           sigma_mult=best_sigma_mult, init_bias=best_init_bias, a=best_a)
    stage_d["stage"] = "D_combined"
    print(f"[stage D result] {stage_d}", flush=True)
    all_rows.append(stage_d)

    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "lspin_rescue_sweep.csv", index=False)

    concrete_ref = None
    if args.concrete_reference_csv and args.concrete_reference_csv.exists():
        concrete_ref = pd.read_csv(args.concrete_reference_csv).iloc[0].to_dict()

    print("\n" + df.to_string(), flush=True)
    if concrete_ref:
        print(f"\n[reference] Concrete KIPAN baseline: per_patient_stability_median="
              f"{concrete_ref.get('per_patient_stability_median')}, mean_test_cindex="
              f"{concrete_ref.get('mean_test_cindex')}", flush=True)
    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
