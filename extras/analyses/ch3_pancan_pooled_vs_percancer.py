#!/usr/bin/env python3
"""
PANCAN: pooled instance-wise gated models vs. per-cancer-type MLPs vs. pooled ungated MLP.

Single-fit comparison (no hyperparameter sweep, no replicates) testing whether
patient-specific sparse gating on the full 31-cancer-type PANCAN cohort gives better
marginal held-out predictive performance, per tumor type, than:
  (a) a dedicated MLP trained only on that tumor type's own training patients
      (no instance-wise selection, no cross-cancer pooling), and
  (b) a single ungated MLP trained on the pooled cohort (pooling, no per-patient
      selection).

All three model classes share exactly the same fixed PANCAN train/test split
(splits_and_core.npz) and, for a given cancer type, the same held-out test patients,
so per-cancer-type comparisons are apples-to-apples. Per-cancer MLPs additionally
carve their own 15% validation split (stratified by event where possible) out of
that cancer's own training rows, mirroring how the pooled models get their
validation split.

Usage:
    conda activate musevo
    cd sparsedeepsurv-paper
    python extras/analyses/ch3_pancan_pooled_vs_percancer.py --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import PROCESSED_DATASETS, ensure_repo_imports

ensure_repo_imports()

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit

import sparsedeepsurv as sds

PANCAN_DATA_DEFAULT = PROCESSED_DATASETS["pancan"]
RESULTS_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "runs" / "ch3_pancan_pooled_vs_percancer"

# Shared architecture / optimization settings (matched to the published MLP baseline
# and the "gentle" gated-model regime used elsewhere in this paper, single unsmoothed
# config per gate family rather than a sweep).
MLP_HIDDEN = (64, 32)
MLP_DROPOUT = 0.1
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-5
MLP_L1_INPUT = 3e-3
MLP_MAX_EPOCHS = 300
MLP_PATIENCE = 25

GATED_LR = 1e-2
GATED_WEIGHT_DECAY = 1e-5
GATED_BATCH_SIZE = 128
GATED_MAX_EPOCHS = 300
GATED_GATING_HIDDEN_DIM = 64
GATED_RISK_HIDDEN_DIMS = (64, 32)
GATED_RISK_DROPOUT_P = 0.1

LSPIN_CFG = dict(gate_type="lspin_tf", gate_sigma=0.20, lam=0.0007, temperature=0.5, patience=12)
CONCRETE_CFG = dict(gate_type="concrete", gate_sigma=0.15, lam=0.0014, temperature=0.3, patience=20)

BATCH_SIZE = 128


def stratified_val_split(events: np.ndarray, histo: Optional[np.ndarray], seed: int, val_frac: float = 0.15):
    """15% validation split, stratified by (event, histology) when possible, falling
    back to stratifying by event alone, then to an unstratified split, so this also
    works for small per-cancer cohorts."""
    n = len(events)
    rng_kwargs = dict(n_splits=1, test_size=val_frac, random_state=seed)
    if histo is not None:
        strat = np.array([f"{e}_{h}" for e, h in zip(events, histo)])
        try:
            ss = ShuffleSplit(**rng_kwargs)
            i_tr, i_val = next(ss.split(np.zeros(n), strat))
            # ShuffleSplit doesn't actually stratify; use StratifiedShuffleSplit when possible.
            from sklearn.model_selection import StratifiedShuffleSplit
            counts = pd.Series(strat).value_counts()
            if (counts >= 2).all():
                sss = StratifiedShuffleSplit(**rng_kwargs)
                i_tr, i_val = next(sss.split(np.zeros(n), strat))
            return i_tr, i_val
        except ValueError:
            pass
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        counts = pd.Series(events).value_counts()
        if (counts >= 2).all():
            sss = StratifiedShuffleSplit(**rng_kwargs)
            i_tr, i_val = next(sss.split(np.zeros(n), events))
            return i_tr, i_val
    except ValueError:
        pass
    ss = ShuffleSplit(**rng_kwargs)
    return next(ss.split(np.zeros(n)))


def train_mlp(Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, *, input_dim, seed, device):
    model = sds.make_seeded_mlp(
        input_dim=input_dim, hidden_dims=MLP_HIDDEN, dropout_p=MLP_DROPOUT, seed=seed,
    ).to(device)
    info = sds.train_deepsurv_mlp_l1(
        model, Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val,
        config=sds.MLPTrainConfig(
            lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY, lambda_l1_input=MLP_L1_INPUT,
            batch_size=BATCH_SIZE, max_epochs=MLP_MAX_EPOCHS, patience=MLP_PATIENCE,
        ),
        device=device, verbose=False,
    )
    return model, info


def train_gated(cfg: Dict, Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, Xt_test, tt_test, et_test,
                 *, input_dim, seed, device):
    model, info = sds.run_one_model(
        Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, Xt_test, tt_test, et_test,
        input_dim=input_dim,
        gate_type=cfg["gate_type"],
        gate_sigma=cfg["gate_sigma"],
        lam=cfg["lam"],
        lambda_sample_smooth=0.0,
        patience=cfg["patience"],
        A_sample_train=None,
        device=device,
        lr=GATED_LR,
        temperature=cfg["temperature"],
        concrete_mode="ste",
        weight_decay=GATED_WEIGHT_DECAY,
        batch_size=GATED_BATCH_SIZE,
        max_epochs=GATED_MAX_EPOCHS,
        predictor="mlp",
        gating_hidden_dim=GATED_GATING_HIDDEN_DIM,
        risk_hidden_dims=GATED_RISK_HIDDEN_DIMS,
        risk_dropout_p=GATED_RISK_DROPOUT_P,
        seed=seed,
    )
    return model, info


def bootstrap_cindex_ci(risk: np.ndarray, time_: np.ndarray, event: np.ndarray,
                         n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    from sparsedeepsurv import concordance_index
    rng = np.random.default_rng(seed)
    n = len(risk)
    point = concordance_index(risk, time_, event.astype(bool))
    if n < 4 or event.sum() < 2:
        return {"cindex": float(point), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if event[idx].sum() < 1:
            continue
        try:
            c = concordance_index(risk[idx], time_[idx], event[idx].astype(bool))
        except Exception:
            continue
        if not np.isnan(c):
            vals.append(c)
    if not vals:
        return {"cindex": float(point), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"cindex": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "n_boot": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=PANCAN_DATA_DEFAULT)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    ap.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--min-train-n", type=int, default=200,
                     help="Minimum train-split patients for a cancer type to get its own MLP.")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    risk_dir = results_dir / "risk_scores"
    risk_dir.mkdir(exist_ok=True)
    (results_dir / "state_dicts").mkdir(exist_ok=True)

    log_path = results_dir / "run.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a", buffering=1) as fh:
            fh.write(msg + "\n")

    log(f"=== ch3_pancan_pooled_vs_percancer  start={time.strftime('%Y-%m-%d %H:%M:%S')} device={args.device} ===")

    sds.set_all_seeds(args.seed)
    data = sds.load_pancan_split_artifacts(Path(args.outdir).resolve())
    X_train, X_test = data["X_train"], data["X_test"]
    time_train, time_test = data["time_train"], data["time_test"]
    event_train, event_test = data["event_train"], data["event_test"]
    histo_train, histo_test = data["histo_train"].astype(str), data["histo_test"].astype(str)
    input_dim = X_train.shape[1]

    log(f"PANCAN: train n={X_train.shape[0]} test n={X_test.shape[0]} features={input_dim}")
    log(f"Cancer types: {sorted(set(histo_train))}")

    # ---- Shared pooled train/val split (same for global MLP, LSPIN, Concrete) ----
    i_tr, i_val = stratified_val_split(event_train, histo_train, seed=args.seed, val_frac=0.15)
    Xt_tr = sds.as_torch(X_train[i_tr]); Xt_val = sds.as_torch(X_train[i_val]); Xt_test = sds.as_torch(X_test)
    tt_tr = sds.as_torch(time_train[i_tr]); tt_val = sds.as_torch(time_train[i_val]); tt_test = sds.as_torch(time_test)
    et_tr = sds.as_torch(event_train[i_tr]); et_val = sds.as_torch(event_train[i_val]); et_test = sds.as_torch(event_test)
    log(f"Pooled split: train={len(i_tr)} val={len(i_val)} test={X_test.shape[0]}")

    pooled_models: Dict[str, torch.nn.Module] = {}
    pooled_info: Dict[str, Dict] = {}

    t0 = time.time()
    log("[pooled MLP] training...")
    model, info = train_mlp(Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val,
                             input_dim=input_dim, seed=args.seed, device=args.device)
    pooled_models["MLP_pooled"] = model
    pooled_info["MLP_pooled"] = info
    log(f"[pooled MLP] done in {time.time() - t0:.1f}s best_epoch={info['best_epoch']} best_val_c={info['best_val_cindex']:.4f}")

    for label, cfg in (("LSPIN_pooled", LSPIN_CFG), ("Concrete_pooled", CONCRETE_CFG)):
        t0 = time.time()
        log(f"[{label}] training...")
        model, info = train_gated(cfg, Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, Xt_test, tt_test, et_test,
                                   input_dim=input_dim, seed=args.seed, device=args.device)
        pooled_models[label] = model
        pooled_info[label] = info
        log(f"[{label}] done in {time.time() - t0:.1f}s best_epoch={info['best_epoch']} "
            f"best_val_c={info['best_val_cindex']:.4f} test_c={info['test_cindex']:.4f}")

    for label, model in pooled_models.items():
        torch.save(model.state_dict(), results_dir / "state_dicts" / f"{label}.pt")

    # ---- Per-cancer-type MLPs, trained only on that cancer's own train rows ----
    type_counts = pd.Series(histo_train).value_counts()
    eligible_types = sorted(type_counts[type_counts >= args.min_train_n].index.tolist())
    log(f"Cancer types with train n >= {args.min_train_n}: {eligible_types}")

    percancer_models: Dict[str, torch.nn.Module] = {}
    percancer_info: Dict[str, Dict] = {}

    for ct in eligible_types:
        mask_tr_all = histo_train == ct
        X_ct, t_ct, e_ct = X_train[mask_tr_all], time_train[mask_tr_all], event_train[mask_tr_all]
        i_tr_ct, i_val_ct = stratified_val_split(e_ct, None, seed=args.seed, val_frac=0.15)
        Xt_tr_ct = sds.as_torch(X_ct[i_tr_ct]); Xt_val_ct = sds.as_torch(X_ct[i_val_ct])
        tt_tr_ct = sds.as_torch(t_ct[i_tr_ct]); tt_val_ct = sds.as_torch(t_ct[i_val_ct])
        et_tr_ct = sds.as_torch(e_ct[i_tr_ct]); et_val_ct = sds.as_torch(e_ct[i_val_ct])

        t0 = time.time()
        log(f"[MLP_{ct}] training on n_train={len(i_tr_ct)} n_val={len(i_val_ct)} ...")
        model, info = train_mlp(Xt_tr_ct, tt_tr_ct, et_tr_ct, Xt_val_ct, tt_val_ct, et_val_ct,
                                 input_dim=input_dim, seed=args.seed, device=args.device)
        percancer_models[ct] = model
        percancer_info[ct] = info
        torch.save(model.state_dict(), results_dir / "state_dicts" / f"MLP_{ct}.pt")
        log(f"[MLP_{ct}] done in {time.time() - t0:.1f}s best_epoch={info['best_epoch']} best_val_c={info['best_val_cindex']:.4f}")

    # ---- Evaluation: per-cancer-type held-out C-index for every model, on the SAME
    #      held-out test patients of that cancer type in all cases ----
    rows: List[Dict] = []
    risk_store: Dict[str, np.ndarray] = {}

    for label, model in pooled_models.items():
        for ct in sorted(set(histo_test)):
            m = histo_test == ct
            n_test_ct = int(m.sum())
            if n_test_ct < 5:
                continue
            c, risk = sds.eval_cindex_and_risk(model, sds.as_torch(X_test[m]), sds.as_torch(time_test[m]),
                                                sds.as_torch(event_test[m]), device=args.device)
            ci = bootstrap_cindex_ci(risk, time_test[m], event_test[m], n_boot=args.n_boot, seed=args.seed)
            rows.append({"model": label, "cancer_type": ct, "n_train_pooled": len(i_tr),
                         "n_test": n_test_ct, "n_test_events": int(event_test[m].sum()),
                         "trained_on": "pooled_31_types", **ci})
            risk_store[f"{label}__{ct}"] = risk

    for ct, model in percancer_models.items():
        m = histo_test == ct
        n_test_ct = int(m.sum())
        c, risk = sds.eval_cindex_and_risk(model, sds.as_torch(X_test[m]), sds.as_torch(time_test[m]),
                                            sds.as_torch(event_test[m]), device=args.device)
        ci = bootstrap_cindex_ci(risk, time_test[m], event_test[m], n_boot=args.n_boot, seed=args.seed)
        n_train_ct = int((histo_train == ct).sum())
        rows.append({"model": f"MLP_singlecancer", "cancer_type": ct, "n_train_pooled": n_train_ct,
                     "n_test": n_test_ct, "n_test_events": int(event_test[m].sum()),
                     "trained_on": ct, **ci})
        risk_store[f"MLP_singlecancer__{ct}"] = risk

    results = pd.DataFrame(rows)
    results.to_csv(results_dir / "percancer_cindex_results.csv", index=False)
    np.savez(risk_dir / "risk_scores.npz", **risk_store)

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "pooled_overall_test_cindex": {k: v.get("test_cindex") for k, v in pooled_info.items()},
        "pooled_info": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in pooled_info.items()},
        "eligible_types": eligible_types,
        "min_train_n": args.min_train_n,
        "seed": args.seed,
    }, indent=2, default=str))

    log(f"Wrote {results_dir / 'percancer_cindex_results.csv'}")
    log(f"Wrote {risk_dir / 'risk_scores.npz'}")
    log(f"=== done at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")


if __name__ == "__main__":
    main()
