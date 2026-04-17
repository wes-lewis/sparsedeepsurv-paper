#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid, StratifiedKFold, StratifiedShuffleSplit

import lspin_pytorch as lp

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    HAVE_LOWESS = True
except Exception:
    HAVE_LOWESS = False

try:
    from sksurv.metrics import integrated_brier_score
    from sksurv.linear_model.coxph import BreslowEstimator
    from sksurv.nonparametric import kaplan_meier_estimator
    from sksurv.util import Surv

    HAVE_IBS = True
except Exception:
    HAVE_IBS = False

MODEL_ORDER = [
    "MLP + L1",
    "MLP + L1 (|w|<cutoff zeroed)",
    "LSPIN (no smooth)",
    "LSPIN + patient smooth",
    "Concrete (no smooth)",
    "Concrete + patient smooth",
]

MODEL_PALETTE = {
    "MLP + L1": "#4e79a7",
    "MLP + L1 (|w|<cutoff zeroed)": "#59a14f",
    "LSPIN (no smooth)": "#e15759",
    "LSPIN + patient smooth": "#f28e2b",
    "Concrete (no smooth)": "#b07aa1",
    "Concrete + patient smooth": "#76b7b2",
}

CV_GRID_PRESETS = {
    "small": {
        "lambdas": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2],
        "sigmas": [0.10, 0.25, 0.50],
        "sample_smooth": [0.01, 0.05, 0.10],
    },
    "full": {
        "lambdas": [3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2],
        "sigmas": [0.10, 0.15, 0.25, 0.35, 0.50],
        "sample_smooth": [0.01, 0.025, 0.05, 0.10],
    },
}

_GPU_LOCK_HANDLE = None


class RiskOnlyWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproducible survival pipeline: compare MLP+lasso vs LSPIN variants per histology"
    )
    p.add_argument("--dataset-name", type=str, default="dataset")
    p.add_argument("--outdir", type=Path, default=Path("runs") / "kipan_20260209_213604")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--results-prefix", type=str, default="cleaned_survival")
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--val-frac", type=float, default=0.15)

    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--mlp-dropout", type=float, default=0.1)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    p.add_argument("--mlp-l1", type=float, default=3e-3)
    p.add_argument("--mlp-max-epochs", type=int, default=300)
    p.add_argument("--mlp-patience", type=int, default=25)

    p.add_argument("--lasso-cutoff", type=float, default=5e-3)
    p.add_argument("--lasso-score", type=str, default="maxabs", choices=["maxabs", "l2"])

    p.add_argument("--gate-sigma", type=float, default=0.12)
    p.add_argument("--gate-sigmas", type=float, nargs="+", default=None)
    p.add_argument("--gate-a", type=float, default=1.0)
    p.add_argument("--gating-hidden", type=int, default=128)
    p.add_argument("--lspin-hidden", type=int, default=64)

    p.add_argument("--lspin-lr", type=float, default=1e-2)
    p.add_argument("--lspin-weight-decay", type=float, default=1e-5)
    p.add_argument("--lspin-batch-size", type=int, default=128)
    p.add_argument("--lspin-max-epochs", type=int, default=220)
    p.add_argument("--lspin-patience", type=int, default=80)
    p.add_argument(
        "--lspin-lambdas",
        type=float,
        nargs="+",
        default=None,
    )
    p.add_argument("--lspin-dropout", type=float, default=0.1)
    p.add_argument("--sample-smooth", type=float, default=0.2)
    p.add_argument("--sample-smooth-grid", type=float, nargs="+", default=None)
    p.add_argument("--cv-grid-preset", type=str, default="small", choices=["small", "full"])
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--lspin-cv-folds", type=int, default=3)
    p.add_argument("--lspin-cv-max-epochs", type=int, default=140)
    p.add_argument("--lspin-cv-patience", type=int, default=35)
    p.add_argument("--n-final-seeds", type=int, default=5)
    p.add_argument("--final-seed-start", type=int, default=None)
    p.add_argument("--selection-metric", type=str, default="joint", choices=["joint", "cindex", "ibs"])
    p.add_argument("--target-k-source", type=str, default="fixed", choices=["fixed", "mlp"])
    p.add_argument("--target-k-fixed", type=float, default=120.0)
    p.add_argument("--sparsity-band-low", type=float, default=100.0)
    p.add_argument("--sparsity-band-high", type=float, default=150.0)
    p.add_argument("--include-concrete", action="store_true")
    p.add_argument("--concrete-own-cv", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def set_seed(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _query_gpu_stats() -> List[Dict[str, float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    stats = []
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != 4:
            continue
        try:
            idx = int(parts[0])
            mem_used = float(parts[1])
            mem_total = max(float(parts[2]), 1.0)
            util = float(parts[3])
        except Exception:
            continue
        stats.append(
            {
                "index": idx,
                "mem_used": mem_used,
                "mem_total": mem_total,
                "util": util,
                "mem_frac": mem_used / mem_total,
            }
        )
    return stats


def _try_lock_gpu(idx: int, lock_dir: Path) -> Optional[object]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    f = open(lock_dir / f"gpu_{idx}.lock", "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.write(f"{os.getpid()}\n")
        f.flush()
        return f
    except BlockingIOError:
        f.close()
        return None


def resolve_device(device_arg: str, *, lock_dir: Path = Path("/tmp/lspin_gpu_locks")) -> str:
    global _GPU_LOCK_HANDLE
    if device_arg == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    if device_arg == "cuda":
        return "cuda:0"
    # auto mode: choose least-used GPU and try to lock it to avoid collisions.
    stats = _query_gpu_stats()
    if not stats:
        return "cuda:0"
    stats = sorted(stats, key=lambda r: (r["mem_frac"], r["util"], r["mem_used"], r["index"]))
    for rec in stats:
        fh = _try_lock_gpu(int(rec["index"]), lock_dir)
        if fh is not None:
            _GPU_LOCK_HANDLE = fh
            return f"cuda:{int(rec['index'])}"
    # Fallback when all locks busy: still pick least-loaded card.
    return f"cuda:{int(stats[0]['index'])}"


def load_split_artifacts(outdir: Path) -> Dict[str, np.ndarray]:
    core = np.load(outdir / "splits_and_core.npz", allow_pickle=True)
    xscaled = np.load(outdir / "X_scaled.npz", allow_pickle=True)

    data = {
        "train_idx": core["train_idx"],
        "test_idx": core["test_idx"],
        "time_train": core["time_train"].astype(np.float32),
        "time_test": core["time_test"].astype(np.float32),
        "event_train": core["event_train"].astype(np.uint8),
        "event_test": core["event_test"].astype(np.uint8),
        "histo_train": core["histo_train"].astype(str),
        "histo_test": core["histo_test"].astype(str),
        "gene_names": core["gene_names"].astype(str),
        "X_train": xscaled["X_train"].astype(np.float32),
        "X_test": xscaled["X_test"].astype(np.float32),
    }

    scaler_path = outdir / "models" / "scaler.joblib"
    if scaler_path.exists():
        _ = joblib.load(scaler_path)

    return data


def make_val_split(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    histo_train: np.ndarray,
    val_frac: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    strat_key = np.array([f"{e}_{h}" for e, h in zip(event_train, histo_train)])
    # Fallback hierarchy for rare strata: (event,histo) -> event -> unstratified.
    labels = strat_key
    _, counts = np.unique(labels, return_counts=True)
    if counts.min(initial=0) < 2:
        labels = np.asarray(event_train).astype(str)
        _, counts = np.unique(labels, return_counts=True)
    if counts.min(initial=0) >= 2:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        i_tr, i_val = next(sss.split(X_train, labels))
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(X_train.shape[0])
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(idx))))
        i_val = np.sort(idx[:n_val])
        i_tr = np.sort(idx[n_val:])

    return {
        "i_tr": i_tr,
        "i_val": i_val,
        "Xt_tr": X_train[i_tr],
        "tt_tr": time_train[i_tr],
        "et_tr": event_train[i_tr],
        "Xt_val": X_train[i_val],
        "tt_val": time_train[i_val],
        "et_val": event_train[i_val],
    }


def as_torch(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def make_mlp(input_dim: int, hidden_dims: Tuple[int, ...], dropout_p: float) -> lp.DeepSurvMLP:
    return lp.DeepSurvMLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout_p=dropout_p)


def train_mlp_l1(
    model: lp.DeepSurvMLP,
    Xt_tr: torch.Tensor,
    tt_tr: torch.Tensor,
    et_tr: torch.Tensor,
    Xt_val: torch.Tensor,
    tt_val: torch.Tensor,
    et_val: torch.Tensor,
    *,
    lr: float,
    lambda_l1_input: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: str,
) -> Dict[str, float]:
    return lp.train_deepsurv_mlp_l1(
        model,
        Xt_tr,
        tt_tr,
        et_tr,
        Xt_val,
        tt_val,
        et_val,
        lr=lr,
        lambda_l1_input=lambda_l1_input,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
    )


def input_l2_scores(model: lp.DeepSurvMLP) -> np.ndarray:
    w = model.input_layer.weight.detach().cpu().numpy()
    return np.sqrt(np.sum(w * w, axis=0))


def input_maxabs_scores(model: lp.DeepSurvMLP) -> np.ndarray:
    w = model.input_layer.weight.detach().cpu().numpy()
    return np.max(np.abs(w), axis=0)


def prune_mlp_input_columns(model: lp.DeepSurvMLP, keep_mask: np.ndarray) -> lp.DeepSurvMLP:
    m = copy.deepcopy(model)
    with torch.no_grad():
        w = m.input_layer.weight
        keep = torch.as_tensor(keep_mask.astype(bool), dtype=torch.bool, device=w.device)
        w[:, ~keep] = 0.0
    return m


def make_lspin_model(
    input_dim: int,
    *,
    hidden_dim: int,
    gating_hidden_dim: int,
    gate_sigma: float,
    a: float,
    dropout_p: float,
    risk_hidden_dims: Tuple[int, ...],
    risk_dropout_p: float,
    gate_type: str = "lspin_tf",
) -> lp.DeepSurvGated:
    model = lp.DeepSurvGated(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        gating_hidden_dim=gating_hidden_dim,
        llspin=False,
        dropout_p=float(dropout_p),
        gate_type=str(gate_type),
        gate_sigma=float(gate_sigma),
        a=float(a),
        temperature=0.5,
    )
    # Match baseline MLP architecture for a fairer risk-head comparison.
    layers = []
    prev = input_dim
    for h in risk_hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        if risk_dropout_p > 0:
            layers += [nn.Dropout(float(risk_dropout_p))]
        prev = h
    layers += [nn.Linear(prev, 1)]
    model.risk_net = nn.Sequential(*layers)
    return model


def tune_lspin_for_target_k(
    *,
    lambdas: List[float],
    target_k: float,
    Xt_tr: torch.Tensor,
    tt_tr: torch.Tensor,
    et_tr: torch.Tensor,
    Xt_val: torch.Tensor,
    tt_val: torch.Tensor,
    et_val: torch.Tensor,
    Xt_test: torch.Tensor,
    tt_test: torch.Tensor,
    et_test: torch.Tensor,
    device: str,
    base_seed: int,
    input_dim: int,
    hidden_dim: int,
    gating_hidden_dim: int,
    gate_sigma: float,
    gate_a: float,
    dropout_p: float,
    risk_hidden_dims: Tuple[int, ...],
    risk_dropout_p: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    lambda_sample_smooth: float,
    A_sample_train=None,
    gate_type: str = "lspin_tf",
    log_fn: Optional[Callable[[str], None]] = None,
    log_prefix: str = "",
) -> Tuple[lp.DeepSurvGated, Dict[str, object], pd.DataFrame]:
    sweep_rows = []
    best_key = (float("inf"), float("inf"))
    best_info = None
    best_model = None

    total_lam = len(lambdas)
    for idx, lam in enumerate(lambdas):
        t0 = time.time()
        if log_fn is not None:
            log_fn(
                f"{log_prefix}tune start lambda {idx + 1}/{total_lam}: "
                f"lambda_sparse={float(lam):.6g}, gate_type={gate_type}, sample_smooth={float(lambda_sample_smooth):.6g}"
            )
        set_seed(base_seed + idx)
        model = make_lspin_model(
            input_dim,
            hidden_dim=hidden_dim,
            gating_hidden_dim=gating_hidden_dim,
            gate_sigma=gate_sigma,
            a=gate_a,
            dropout_p=dropout_p,
            risk_hidden_dims=risk_hidden_dims,
            risk_dropout_p=risk_dropout_p,
            gate_type=gate_type,
        ).to(device)

        cfg = lp.GatedTrainConfig(
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            lambda_sparse=float(lam),
            lambda_sample_smooth=float(lambda_sample_smooth),
            lambda_gene_smooth=0.0,
            target_K=None,
            lambda_Kmatch=0.0,
        )

        info = lp.train_gated_deepsurv(
            model,
            Xt_tr,
            tt_tr,
            et_tr,
            Xt_val,
            tt_val,
            et_val,
            Xt_test,
            tt_test,
            et_test,
            A_sample_train=A_sample_train,
            config=cfg,
            device=device,
            verbose=False,
        )

        k_hard_val = float(info["Khard_train_mean"])
        if len(info["history"].get("val_Khard_mean", [])):
            k_hard_val = float(info["history"]["val_Khard_mean"][-1])

        k_exp_val = float(info["Kexp_train_mean"])
        if len(info["history"].get("val_Kexp_mean", [])):
            k_exp_val = float(info["history"]["val_Kexp_mean"][-1])

        test_c = float(info.get("test_cindex", np.nan))
        val_c = float(info.get("best_val_cindex", np.nan))

        err = abs(k_hard_val - target_k)
        key = (err, -val_c)

        row = {
            "lambda_sparse": float(lam),
            "target_k": float(target_k),
            "Khard_val": k_hard_val,
            "Kexp_val": k_exp_val,
            "abs_err_target": float(err),
            "best_val_cindex": val_c,
            "test_cindex": test_c,
            "best_epoch": int(info.get("best_epoch", -1)),
        }
        sweep_rows.append(row)
        if log_fn is not None:
            dt = time.time() - t0
            log_fn(
                f"{log_prefix}tune done lambda {idx + 1}/{total_lam}: "
                f"val_c={val_c:.4f}, Khard={k_hard_val:.2f}, abs_err={err:.2f}, elapsed={dt:.1f}s"
            )

        if key < best_key:
            best_key = key
            best_info = info
            best_model = model

    if best_model is None or best_info is None:
        raise RuntimeError("LSPIN lambda sweep failed to produce a model")

    sweep_df = pd.DataFrame(sweep_rows).sort_values(["abs_err_target", "best_val_cindex"], ascending=[True, False])
    return best_model, best_info, sweep_df




def patient_affinity_consistency_from_selected_features(
    model: lp.DeepSurvGated,
    X_eval: torch.Tensor,
    *,
    device: str,
    min_genes: int = 5,
) -> float:
    X_np = X_eval.cpu().numpy().astype(float)
    if X_np.shape[0] < 4:
        return float("nan")

    G = gate_matrix(model, X_eval, device=device, view="hard")
    selected = (G.mean(axis=0) > 0.05)
    if selected.sum() < min_genes:
        return float("nan")

    X_sel = X_np[:, selected]
    # cosine similarity for patient affinity in full space and selected-feature space
    full_norm = np.linalg.norm(X_np, axis=1, keepdims=True)
    sel_norm = np.linalg.norm(X_sel, axis=1, keepdims=True)
    full = X_np / np.clip(full_norm, 1e-8, None)
    sel = X_sel / np.clip(sel_norm, 1e-8, None)
    S_full = full @ full.T
    S_sel = sel @ sel.T

    iu = np.triu_indices(S_full.shape[0], k=1)
    a = S_full[iu]
    b = S_sel[iu]
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def corr_gate_vs_sample_affinity_on_edges(
    model: lp.DeepSurvGated,
    X_eval: torch.Tensor,
    A_sample_eval,
    *,
    device: str,
) -> float:
    # Old-notebook metric: correlation between gate cosine affinity and sample-graph edge weights.
    G = gate_matrix(model, X_eval, device=device, view="hard")
    A = A_sample_eval.tocoo()
    i, j = A.row, A.col
    w = A.data.astype(np.float64)
    if len(i) == 0:
        return float("nan")
    Gi = G[i]
    Gj = G[j]
    num = (Gi * Gj).sum(axis=1)
    den = (np.linalg.norm(Gi, axis=1) * np.linalg.norm(Gj, axis=1) + 1e-12)
    cs = num / den
    cs = 0.5 * (cs + 1.0)
    if np.std(cs) < 1e-12 or np.std(w) < 1e-12:
        return float("nan")
    return float(np.corrcoef(cs, w)[0, 1])

def select_lspin_config_via_cv(
    *,
    X_train_np: np.ndarray,
    time_train_np: np.ndarray,
    event_train_np: np.ndarray,
    histo_train_np: np.ndarray,
    input_dim: int,
    device: str,
    seed: int,
    hidden_dim: int,
    gating_hidden_dim: int,
    gate_a: float,
    dropout_p: float,
    risk_hidden_dims: Tuple[int, ...],
    risk_dropout_p: float,
    lambdas: List[float],
    sigmas: List[float],
    sample_smooth_vals: List[float],
    target_k: float,
    knn_k: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    cv_folds: int,
    cv_max_epochs: int,
    cv_patience: int,
    selection_metric: str = "joint",
    gate_type: str = "lspin_tf",
    log_fn: Optional[Callable[[str], None]] = None,
    log_prefix: str = "",
) -> pd.DataFrame:
    strat_key = np.array([f"{int(e)}_{h}" for e, h in zip(event_train_np, histo_train_np)])
    labels = strat_key
    _, counts = np.unique(labels, return_counts=True)
    if counts.min(initial=0) < cv_folds:
        labels = np.asarray(event_train_np).astype(str)
        _, counts = np.unique(labels, return_counts=True)
    if counts.min(initial=0) >= cv_folds and len(np.unique(labels)) > 1:
        split_iter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(X_train_np, labels)
    else:
        split_iter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed).split(
            X_train_np, np.asarray(event_train_np).astype(str)
        )
    fold_splits = list(split_iter)
    use_sample_smooth = any(float(v) > 0.0 for v in sample_smooth_vals)
    fold_cache = []
    for fi, (itr, iva) in enumerate(fold_splits):
        Xtr_np = X_train_np[itr]
        Xva_np = X_train_np[iva]
        cache_item = {
            "fi": fi,
            "itr": itr,
            "iva": iva,
            "Xtr": as_torch(Xtr_np),
            "ttr": as_torch(time_train_np[itr]),
            "etr": as_torch(event_train_np[itr]),
            "Xva": as_torch(Xva_np),
            "tva": as_torch(time_train_np[iva]),
            "eva": as_torch(event_train_np[iva]),
            "A_sample": None,
            "A_val": None,
        }
        if use_sample_smooth:
            cache_item["A_sample"] = lp.compute_knn_affinity(Xtr_np, k=knn_k)
        if len(iva) > 3:
            k_eff = int(max(2, min(knn_k, len(iva) - 1)))
            cache_item["A_val"] = lp.compute_knn_affinity(Xva_np, k=k_eff)
        fold_cache.append(cache_item)

    grid = list(
        ParameterGrid(
            {
                "lambda_sparse": [float(x) for x in lambdas],
                "gate_sigma": [float(x) for x in sigmas],
                "lambda_sample_smooth": [float(x) for x in sample_smooth_vals],
            }
        )
    )

    rows = []
    n_cfg = len(grid)
    n_fold = len(fold_splits)
    if log_fn is not None:
        log_fn(
            f"{log_prefix}CV grid start: {n_cfg} configs x {n_fold} folds (gate_type={gate_type})"
        )
    for ci, cfg_params in enumerate(grid):
        t_cfg = time.time()
        fold_scores = []
        fold_ibs = []
        fold_khard = []
        fold_errs = []
        fold_affinity_corr = []
        fold_edge_affinity_corr = []

        for fold_item in fold_cache:
            fi = int(fold_item["fi"])
            itr = fold_item["itr"]
            iva = fold_item["iva"]
            if log_fn is not None:
                log_fn(
                    f"{log_prefix}CV cfg {ci + 1}/{n_cfg}, fold {fi + 1}/{n_fold} start: "
                    f"lambda={cfg_params['lambda_sparse']:.6g}, sigma={cfg_params['gate_sigma']:.4g}, "
                    f"smooth={cfg_params['lambda_sample_smooth']:.6g}"
                )
            set_seed(seed + 1000 * ci + fi)

            Xtr = fold_item["Xtr"]
            ttr = fold_item["ttr"]
            etr = fold_item["etr"]
            Xva = fold_item["Xva"]
            tva = fold_item["tva"]
            eva = fold_item["eva"]

            A_sample = None
            if cfg_params["lambda_sample_smooth"] > 0:
                A_sample = fold_item["A_sample"]

            model = make_lspin_model(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                gating_hidden_dim=gating_hidden_dim,
                gate_sigma=cfg_params["gate_sigma"],
                a=gate_a,
                dropout_p=dropout_p,
                risk_hidden_dims=risk_hidden_dims,
                risk_dropout_p=risk_dropout_p,
                gate_type=gate_type,
            ).to(device)

            cfg = lp.GatedTrainConfig(
                lr=lr,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epochs=cv_max_epochs,
                patience=cv_patience,
                lambda_sparse=cfg_params["lambda_sparse"],
                lambda_sample_smooth=cfg_params["lambda_sample_smooth"],
                lambda_gene_smooth=0.0,
            )
            info = lp.train_gated_deepsurv(
                model,
                Xtr,
                ttr,
                etr,
                Xva,
                tva,
                eva,
                config=cfg,
                A_sample_train=A_sample,
                device=device,
                verbose=False,
            )

            val_c = float(info.get("best_val_cindex", np.nan))
            if np.isnan(val_c):
                val_c, _ = lp.eval_cindex(RiskOnlyWrapper(model).to(device), Xva, tva, eva, device=device)
            _, kh_mean, _ = lp.eval_gates_hard_K(model, Xva, device=device, threshold=0.5)
            val_ibs = integrated_brier_overall(
                model,
                X_train=Xtr,
                t_train=ttr,
                e_train=etr,
                X_eval=Xva,
                t_eval=tva,
                e_eval=eva,
                device=device,
            )
            affinity_corr = patient_affinity_consistency_from_selected_features(
                model, Xva, device=device
            )
            if fold_item["A_val"] is not None:
                A_val = fold_item["A_val"]
                edge_aff_corr = corr_gate_vs_sample_affinity_on_edges(
                    model,
                    Xva,
                    A_val,
                    device=device,
                )
            else:
                edge_aff_corr = float("nan")

            fold_scores.append(val_c)
            fold_ibs.append(val_ibs)
            fold_khard.append(float(kh_mean))
            fold_errs.append(abs(float(kh_mean) - float(target_k)))
            fold_affinity_corr.append(affinity_corr)
            fold_edge_affinity_corr.append(edge_aff_corr)
            if log_fn is not None:
                log_fn(
                    f"{log_prefix}CV cfg {ci + 1}/{n_cfg}, fold {fi + 1}/{n_fold} done: "
                    f"val_c={val_c:.4f}, ibs={val_ibs:.4f}, Khard={float(kh_mean):.2f}"
                )

        rows.append(
            {
                **cfg_params,
                "cv_val_cindex_mean": float(np.mean(fold_scores)),
                "cv_val_cindex_std": float(np.std(fold_scores)),
                "cv_ibs_mean": float(np.nanmean(fold_ibs)),
                "cv_ibs_std": float(np.nanstd(fold_ibs)),
                "cv_khard_mean": float(np.mean(fold_khard)),
                "cv_abs_target_err_mean": float(np.mean(fold_errs)),
                "cv_affinity_corr_mean": float(np.nanmean(fold_affinity_corr)),
                "cv_affinity_corr_std": float(np.nanstd(fold_affinity_corr)),
                "cv_edge_affinity_corr_mean": float(np.nanmean(fold_edge_affinity_corr)),
                "cv_edge_affinity_corr_std": float(np.nanstd(fold_edge_affinity_corr)),
            }
        )
        if log_fn is not None:
            dt_cfg = time.time() - t_cfg
            r = rows[-1]
            log_fn(
                f"{log_prefix}CV cfg {ci + 1}/{n_cfg} done: "
                f"mean_val_c={r['cv_val_cindex_mean']:.4f}, mean_ibs={r['cv_ibs_mean']:.4f}, "
                f"mean_Khard={r['cv_khard_mean']:.2f}, elapsed={dt_cfg:.1f}s"
            )

    out = pd.DataFrame(rows)
    out["rank_cindex"] = out["cv_val_cindex_mean"].rank(method="min", ascending=False)
    out["rank_ibs"] = out["cv_ibs_mean"].rank(method="min", ascending=True, na_option="bottom")
    out["rank_target_err"] = out["cv_abs_target_err_mean"].rank(method="min", ascending=True)
    out["rank_affinity"] = out["cv_affinity_corr_mean"].rank(method="min", ascending=False, na_option="bottom")
    out["rank_edge_affinity"] = out["cv_edge_affinity_corr_mean"].rank(method="min", ascending=False, na_option="bottom")
    out["rank_joint"] = (
        out["rank_cindex"]
        + out["rank_ibs"]
        + 0.25 * out["rank_target_err"]
        + 0.20 * out["rank_affinity"]
        + 0.20 * out["rank_edge_affinity"]
    )

    if selection_metric == "cindex":
        out = out.sort_values(["cv_val_cindex_mean", "cv_ibs_mean"], ascending=[False, True])
    elif selection_metric == "ibs":
        out = out.sort_values(["cv_ibs_mean", "cv_val_cindex_mean"], ascending=[True, False])
    else:
        out = out.sort_values(["rank_joint", "cv_val_cindex_mean"], ascending=[True, False])
    out = out.reset_index(drop=True)
    if log_fn is not None:
        top = out.iloc[0].to_dict()
        log_fn(
            f"{log_prefix}CV grid complete: best lambda={top['lambda_sparse']:.6g}, "
            f"sigma={top['gate_sigma']:.4g}, smooth={top['lambda_sample_smooth']:.6g}, "
            f"val_c={top['cv_val_cindex_mean']:.4f}, ibs={top['cv_ibs_mean']:.4f}"
        )
    return out


def pick_cv_config_with_matched_sparsity(
    cv_df: pd.DataFrame,
    *,
    selection_metric: str,
    k_low: float,
    k_high: float,
) -> Dict[str, float]:
    if cv_df is None or cv_df.empty:
        raise ValueError("cv_df is empty")

    mid = 0.5 * (k_low + k_high)
    df = cv_df.copy()
    df["abs_err_mid"] = (df["cv_khard_mean"] - mid).abs()
    in_band = df[(df["cv_khard_mean"] >= k_low) & (df["cv_khard_mean"] <= k_high)].copy()
    cand = in_band if len(in_band) else df.nsmallest(10, "abs_err_mid").copy()

    if selection_metric == "cindex":
        cand = cand.sort_values(
            ["cv_val_cindex_mean", "cv_ibs_mean", "cv_edge_affinity_corr_mean", "cv_affinity_corr_mean", "abs_err_mid"],
            ascending=[False, True, False, False, True],
        )
    elif selection_metric == "ibs":
        cand = cand.sort_values(
            ["cv_ibs_mean", "cv_val_cindex_mean", "cv_edge_affinity_corr_mean", "cv_affinity_corr_mean", "abs_err_mid"],
            ascending=[True, False, False, False, True],
        )
    else:
        cand = cand.sort_values(
            ["rank_joint", "cv_val_cindex_mean", "cv_edge_affinity_corr_mean", "cv_affinity_corr_mean", "abs_err_mid"],
            ascending=[True, False, False, False, True],
        )
    return cand.iloc[0].to_dict()


def cindex_by_histology(
    model: torch.nn.Module,
    Xt: torch.Tensor,
    tt: torch.Tensor,
    et: torch.Tensor,
    histo_labels: np.ndarray,
    device: str,
    min_n: int = 10,
) -> pd.DataFrame:
    rows = []
    for h in np.unique(histo_labels):
        m = histo_labels == h
        if int(m.sum()) < min_n:
            continue
        c, _ = lp.eval_cindex(model, Xt[m], tt[m], et[m], device=device)
        rows.append({"histology": h, "n_test": int(m.sum()), "cindex": float(c)})
    return pd.DataFrame(rows)


def model_risk_scores(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: str,
    batch_size: int = 1024,
) -> np.ndarray:
    model.eval()
    risks = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device)
            out = model(xb)
            risk = out[0] if isinstance(out, (tuple, list)) else out
            risks.append(risk.detach().cpu().numpy())
    return np.concatenate(risks, axis=0).astype(float)


def integrated_brier_by_histology(
    model: torch.nn.Module,
    *,
    Xt_tr: torch.Tensor,
    tt_tr: torch.Tensor,
    et_tr: torch.Tensor,
    Xt_te: torch.Tensor,
    tt_te: torch.Tensor,
    et_te: torch.Tensor,
    histo_test: np.ndarray,
    device: str,
    min_n: int = 25,
) -> Optional[pd.DataFrame]:
    if not HAVE_IBS:
        return None

    model = RiskOnlyWrapper(model).to(device)
    risk_tr = model_risk_scores(model, Xt_tr, device=device)
    risk_te = model_risk_scores(model, Xt_te, device=device)

    y_train = Surv.from_arrays(
        event=et_tr.cpu().numpy().astype(bool),
        time=tt_tr.cpu().numpy().astype(float),
    )
    t_train = tt_tr.cpu().numpy().astype(float)
    e_train = et_tr.cpu().numpy().astype(bool)
    y_train_max = float(np.max(t_train))
    breslow = BreslowEstimator().fit(risk_tr, e_train, t_train)

    rows = []
    for h in np.unique(histo_test):
        m = histo_test == h
        n = int(m.sum())
        if n < min_n:
            continue

        y_test = Surv.from_arrays(
            event=et_te[m].cpu().numpy().astype(bool),
            time=tt_te[m].cpu().numpy().astype(float),
        )
        t_test = tt_te[m].cpu().numpy().astype(float)
        r_test = risk_te[m]

        t_lo = np.percentile(t_test, 20)
        t_hi = np.percentile(t_test, 80)
        t_hi = min(t_hi, y_train_max * 0.95)
        if t_hi <= t_lo:
            continue

        times = np.linspace(t_lo, t_hi, 40)
        if len(times) < 5:
            continue

        surv_fns = breslow.get_survival_function(r_test)
        surv_mat = np.zeros((n, len(times)), dtype=float)
        for i, fn in enumerate(surv_fns):
            surv_mat[i, :] = np.clip(fn(times), 1e-8, 1.0)

        try:
            ibs = integrated_brier_score(y_train, y_test, surv_mat, times)
        except Exception:
            continue

        rows.append({"histology": h, "n_test": n, "ibs": float(ibs)})

    if not rows:
        return None
    return pd.DataFrame(rows)


def integrated_brier_overall(
    model: torch.nn.Module,
    *,
    X_train: torch.Tensor,
    t_train: torch.Tensor,
    e_train: torch.Tensor,
    X_eval: torch.Tensor,
    t_eval: torch.Tensor,
    e_eval: torch.Tensor,
    device: str,
) -> float:
    if not HAVE_IBS:
        return float("nan")

    wrapped = RiskOnlyWrapper(model).to(device)
    risk_tr = model_risk_scores(wrapped, X_train, device=device)
    risk_ev = model_risk_scores(wrapped, X_eval, device=device)

    y_train = Surv.from_arrays(
        event=e_train.cpu().numpy().astype(bool),
        time=t_train.cpu().numpy().astype(float),
    )
    y_eval = Surv.from_arrays(
        event=e_eval.cpu().numpy().astype(bool),
        time=t_eval.cpu().numpy().astype(float),
    )
    t_tr = t_train.cpu().numpy().astype(float)
    t_ev = t_eval.cpu().numpy().astype(float)
    e_tr = e_train.cpu().numpy().astype(bool)

    t_lo = np.percentile(t_ev, 10)
    t_hi = min(np.percentile(t_ev, 90), np.max(t_tr) * 0.95)
    if t_hi <= t_lo:
        return float("nan")
    times = np.linspace(t_lo, t_hi, 40)

    breslow = BreslowEstimator().fit(risk_tr, e_tr, t_tr)
    surv_fns = breslow.get_survival_function(risk_ev)
    surv_mat = np.vstack([np.clip(fn(times), 1e-8, 1.0) for fn in surv_fns])
    try:
        return float(integrated_brier_score(y_train, y_eval, surv_mat, times))
    except Exception:
        return float("nan")


def mask_tensor_columns(X: torch.Tensor, keep_mask: np.ndarray) -> torch.Tensor:
    Xm = X.clone()
    drop = ~torch.as_tensor(keep_mask.astype(bool), dtype=torch.bool)
    Xm[:, drop] = 0.0
    return Xm


def densify_lambda_grid(lambdas: List[float], points_per_interval: int = 2) -> List[float]:
    vals = sorted({float(v) for v in lambdas if float(v) > 0})
    if len(vals) <= 1 or points_per_interval <= 0:
        return vals
    out = set(vals)
    for a, b in zip(vals[:-1], vals[1:]):
        logs = np.linspace(np.log10(a), np.log10(b), points_per_interval + 2)[1:-1]
        for lv in logs:
            out.add(float(10.0 ** lv))
    return sorted(out)


def save_fig_train_survival_curves(
    models: Dict[str, torch.nn.Module],
    *,
    X_train: torch.Tensor,
    t_train: torch.Tensor,
    e_train: torch.Tensor,
    histo_train: np.ndarray,
    device: str,
    outpath: Path,
    min_n: int = 25,
) -> None:
    t_np = t_train.cpu().numpy().astype(float)
    e_np = e_train.cpu().numpy().astype(bool)

    histos = [h for h in np.unique(histo_train) if int((histo_train == h).sum()) >= min_n]
    if not histos:
        return

    ncols = min(2, len(histos))
    nrows = int(np.ceil(len(histos) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.0 * ncols, 5.0 * nrows), squeeze=False)
    axes = axes.flatten()

    model_risks = {}
    for name, model in models.items():
        wrapped = RiskOnlyWrapper(model).to(device)
        model_risks[name] = model_risk_scores(wrapped, X_train, device=device)

    for ai, h in enumerate(histos):
        ax = axes[ai]
        m = histo_train == h
        th = t_np[m]
        eh = e_np[m]
        if eh.sum() < 3:
            ax.set_visible(False)
            continue

        km_t, km_s = kaplan_meier_estimator(eh, th)
        ax.step(km_t, km_s, where="post", c="black", lw=2.2, label="Observed KM (train)")

        t_lo = np.percentile(th, 10)
        t_hi = np.percentile(th, 90)
        if t_hi <= t_lo:
            t_hi = np.max(th)
        times = np.linspace(max(1e-6, t_lo), max(t_lo + 1e-6, t_hi), 80)

        for name, risk_all in model_risks.items():
            r = risk_all[m]
            breslow = BreslowEstimator().fit(r, eh, th)
            surv_fns = breslow.get_survival_function(r)
            surv_mat = np.vstack([fn(times) for fn in surv_fns])
            ax.plot(times, surv_mat.mean(axis=0), lw=1.8, label=name)

        ax.set_title(f"{h} (n={int(m.sum())})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)

    for j in range(len(histos), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(2, len(labels)),
        frameon=False,
    )
    fig.suptitle("Training survival curves: observed KM vs model-predicted mean", y=0.985)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def save_fig_cindex_by_histology(df: pd.DataFrame, outpath: Path, dataset_name: str) -> None:
    plt.figure(figsize=(11, 5))
    model_order = [m for m in MODEL_ORDER if m in set(df["model"])]
    ax = sns.barplot(
        data=df,
        x="histology",
        y="cindex",
        hue="model",
        hue_order=model_order,
        palette=MODEL_PALETTE,
    )
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_title(f"{dataset_name} test C-index by histology")
    ax.set_xlabel("Histology")
    ax.set_ylabel("C-index")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def save_fig_overall_cindex(df: pd.DataFrame, outpath: Path) -> None:
    plt.figure(figsize=(7.5, 4.2))
    model_order = [m for m in MODEL_ORDER if m in set(df["model"])]
    ax = sns.barplot(
        data=df,
        x="model",
        y="cindex",
        hue="model",
        hue_order=model_order,
        order=model_order,
        palette=MODEL_PALETTE,
        legend=False,
    )
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_title("Overall test C-index")
    ax.set_xlabel("")
    ax.set_ylabel("C-index")
    plt.xticks(rotation=20, ha="right")
    for p in ax.patches:
        h = p.get_height()
        ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2.0, h), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def save_fig_feature_counts(df: pd.DataFrame, outpath: Path) -> None:
    d = df.copy()
    d["label"] = d["model"] + " | " + d["metric"]
    d = d.sort_values(["model", "metric"]).reset_index(drop=True)

    y = d["value"].to_numpy(dtype=float)
    # prefer 95% CI width when available; fall back to zero (single run)
    yerr = np.zeros_like(y)
    if "ci95_low" in d.columns and "ci95_high" in d.columns:
        yerr = (d["ci95_high"].to_numpy(dtype=float) - d["ci95_low"].to_numpy(dtype=float)) / 2.0

    x = np.arange(len(d))
    colors = [MODEL_PALETTE.get(str(m), "#999999") for m in d["model"]]
    y_finite = y[np.isfinite(y)]
    if len(y_finite) == 0:
        return
    y_max = float(np.max(y_finite))
    y_med = float(np.median(y_finite))
    use_broken_axis = y_med > 0 and (y_max / max(y_med, 1e-9)) >= 4.0

    if use_broken_axis:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(12.0, 6.8),
            gridspec_kw={"height_ratios": [1, 3]},
        )
        for ax in (ax_top, ax_bot):
            ax.bar(x, y, color=colors)
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
            ax.grid(alpha=0.2, axis="y")

        low_cap = float(np.percentile(y_finite, 90) * 1.2)
        low_cap = max(low_cap, 1.0)
        high_floor = max(low_cap * 1.05, float(np.percentile(y_finite, 98)))
        ax_bot.set_ylim(0.0, low_cap)
        ax_top.set_ylim(high_floor, y_max * 1.08)
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(labeltop=False)
        ax_bot.xaxis.tick_bottom()
        dcut = 0.015
        kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, lw=1)
        ax_top.plot((-dcut, +dcut), (-dcut, +dcut), **kwargs)
        ax_top.plot((1 - dcut, 1 + dcut), (-dcut, +dcut), **kwargs)
        kwargs = dict(transform=ax_bot.transAxes, color="k", clip_on=False, lw=1)
        ax_bot.plot((-dcut, +dcut), (1 - dcut, 1 + dcut), **kwargs)
        ax_bot.plot((1 - dcut, 1 + dcut), (1 - dcut, 1 + dcut), **kwargs)

        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(d["label"], rotation=25, ha="right")
        ax_top.set_title("Feature Count Comparison (mean +/- 95% CI; broken y-axis)")
        ax_bot.set_ylabel("Count")
        for xi, yi in zip(x, y):
            target_ax = ax_top if yi >= high_floor else ax_bot
            target_ax.annotate(f"{yi:.1f}", (xi, yi), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        fig.savefig(outpath, dpi=180)
        plt.close(fig)
        return

    plt.figure(figsize=(11.5, 4.8))
    plt.bar(x, y, color=colors)
    plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
    plt.xticks(x, d["label"], rotation=25, ha="right")
    plt.ylabel("Count")
    plt.title("Feature Count Comparison (mean +/- 95% CI)")
    for xi, yi in zip(x, y):
        plt.annotate(f"{yi:.1f}", (xi, yi), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def save_fig_ibs_or_delta(
    ibs_df: Optional[pd.DataFrame],
    cindex_df: pd.DataFrame,
    outpath: Path,
    dataset_name: str,
) -> None:
    if ibs_df is not None and len(ibs_df):
        plt.figure(figsize=(11, 5))
        model_order = [m for m in MODEL_ORDER if m in set(ibs_df["model"])]
        ax = sns.barplot(
            data=ibs_df,
            x="histology",
            y="ibs",
            hue="model",
            hue_order=model_order,
            palette=MODEL_PALETTE,
        )
        ax.set_title(f"{dataset_name} integrated Brier score by histology (lower is better)")
        ax.set_xlabel("Histology")
        ax.set_ylabel("IBS")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(outpath, dpi=180)
        plt.close()
        return

    piv = cindex_df.pivot(index="histology", columns="model", values="cindex")
    if "MLP + L1" in piv.columns and "MLP + L1 (|w|<cutoff zeroed)" in piv.columns:
        piv["delta"] = piv["MLP + L1"] - piv["MLP + L1 (|w|<cutoff zeroed)"]
        ddf = piv.reset_index()[["histology", "delta"]]
        plt.figure(figsize=(8.5, 4.2))
        ax = sns.barplot(data=ddf, x="histology", y="delta", color="#cc6b5a")
        ax.axhline(0.0, c="gray", lw=1)
        ax.set_title("C-index drop after removing low-weight lasso genes")
        ax.set_xlabel("Histology")
        ax.set_ylabel("Δ C-index")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(outpath, dpi=180)
        plt.close()





def gate_matrix(
    model: lp.DeepSurvGated,
    X: torch.Tensor,
    *,
    device: str,
    view: str = "hard",
    batch_size: int = 512,
) -> np.ndarray:
    if view not in {"hard", "g_det"}:
        raise ValueError("view must be 'hard' or 'g_det'")

    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i : i + batch_size].to(device)
            alpha = model.gating_net(xb)
            g_det = torch.clamp(model.a * alpha + 0.5, 0.0, 1.0)
            if view == "hard":
                out = (g_det > 0.5).float()
            else:
                out = g_det
            outs.append(out.cpu().numpy())
    return np.concatenate(outs, axis=0)


def save_fig_gate_heatmap_with_histo(
    model: lp.DeepSurvGated,
    *,
    Xt: torch.Tensor,
    histo: np.ndarray,
    gene_names: np.ndarray,
    device: str,
    outpath: Path,
    title: str,
    view: str = "hard",
    min_frac_on: float = 0.05,
    max_frac_on: float = 0.95,
    top_genes: int = 180,
) -> Optional[pd.DataFrame]:
    def _save_histo_legend(legend_path: Path, labels: list[str], cmap: dict[str, tuple]) -> None:
        handles = [Patch(facecolor=cmap[h], edgecolor="none", label=h) for h in labels]
        ncol = 3
        nrows = max(1, int(np.ceil(len(handles) / ncol)))
        fig_h = max(2.6, 0.52 * nrows + 0.8)
        fig, ax = plt.subplots(figsize=(10.5, fig_h))
        ax.axis("off")
        ax.legend(
            handles=handles,
            title="Tumor type",
            loc="center",
            frameon=False,
            ncol=ncol,
            fontsize=9,
            title_fontsize=11,
            handlelength=1.2,
            columnspacing=1.4,
            labelspacing=0.8,
        )
        fig.tight_layout()
        fig.savefig(legend_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    G = gate_matrix(model, Xt, device=device, view=view)
    if G.ndim != 2 or G.shape[0] == 0:
        return None

    frac_on = G.mean(axis=0)
    keep = (frac_on >= min_frac_on) & (frac_on <= max_frac_on)
    if not np.any(keep):
        return None

    G = G[:, keep]
    genes = np.asarray(gene_names)[keep]

    var = G.var(axis=0)
    keep_var = var > 0
    G = G[:, keep_var]
    genes = genes[keep_var]
    if G.shape[1] == 0:
        return None

    if G.shape[1] > top_genes:
        top_idx = np.argsort(-G.var(axis=0))[:top_genes]
        G = G[:, top_idx]
        genes = genes[top_idx]

    sample_ids = np.array([f"S{i:04d}" for i in range(G.shape[0])])
    df = pd.DataFrame(G, index=sample_ids, columns=genes)
    # Guard against non-finite values that can break scipy linkage.
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    histo = np.asarray(histo).astype(str)
    hist_levels = sorted(pd.unique(histo))
    palette = sns.color_palette("tab20", n_colors=max(3, len(hist_levels)))
    color_map = {h: palette[i % len(palette)] for i, h in enumerate(hist_levels)}
    row_colors = pd.Series(histo, index=df.index).map(color_map)
    legend_path = outpath.with_name(f"{outpath.stem}_histo_legend.png")
    _save_histo_legend(legend_path, hist_levels, color_map)

    sample_order = df.index.to_numpy()
    try:
        cg = sns.clustermap(
            df,
            cmap="viridis",
            row_cluster=True,
            col_cluster=True,
            row_colors=row_colors,
            xticklabels=False,
            yticklabels=False,
            figsize=(14, 7.8),
            metric="cosine",
            method="average",
            dendrogram_ratio=(0.08, 0.06),
            colors_ratio=0.025,
            cbar_pos=None,
        )
        cg.fig.subplots_adjust(top=0.92, right=0.98)
        cg.fig.suptitle(title, y=1.01, fontsize=18)
        cg.savefig(outpath, dpi=180, bbox_inches="tight")
        sample_order = df.index.to_numpy()[cg.dendrogram_row.reordered_ind]
        plt.close(cg.fig)
    except Exception:
        # Fallback: no clustering, keep deterministic sample order so run can finish.
        fig, ax = plt.subplots(figsize=(14, 9))
        sns.heatmap(df, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False, ax=ax)
        ax.set_title(title + " (fallback: no clustering)")
        fig.tight_layout()
        fig.savefig(outpath, dpi=180, bbox_inches="tight")
        plt.close(fig)

    cluster_table = pd.DataFrame(
        {
            "sample_id": sample_order,
            "histo_group": pd.Series(histo, index=df.index).loc[sample_order].values,
        }
    )
    return cluster_table


def save_fig_parsimony_sparse_regime(
    cv_nosim: pd.DataFrame,
    cv_sim: pd.DataFrame,
    outpath: Path,
    csv_out: Path,
    *,
    k_low: float = 100.0,
    k_high: float = 150.0,
) -> None:
    pairs = matched_sparsity_pairs(cv_nosim, cv_sim)
    if pairs.empty:
        return
    pairs = pairs.copy()
    pairs["both_in_band"] = (
        (pairs["nosim_khard"] >= k_low)
        & (pairs["nosim_khard"] <= k_high)
        & (pairs["sim_khard"] >= k_low)
        & (pairs["sim_khard"] <= k_high)
    )
    pairs.to_csv(csv_out, index=False)

    rows = []
    for i, r in pairs.reset_index(drop=True).iterrows():
        rows.append({"pair_id": int(i), "model": "LSPIN (no smooth)", "cv_khard_mean": float(r["nosim_khard"]), "cv_val_cindex_mean": float(r["nosim_cindex"])})
        rows.append({"pair_id": int(i), "model": "LSPIN + patient smooth", "cv_khard_mean": float(r["sim_khard"]), "cv_val_cindex_mean": float(r["sim_cindex"])})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(9.2, 6.2))
    ax = sns.scatterplot(
        data=df,
        x="cv_khard_mean",
        y="cv_val_cindex_mean",
        hue="model",
        style="model",
        s=80,
        palette=MODEL_PALETTE,
        alpha=0.75,
    )
    ax.axvspan(k_low, k_high, color="gray", alpha=0.15, label=f"sparse regime [{int(k_low)}, {int(k_high)}]")
    ax.set_xlabel("CV mean hard-selected features per patient (Khard)")
    ax.set_ylabel("CV mean validation C-index")
    n_band = int(pairs["both_in_band"].sum())
    ax.set_title(f"Parsimony-performance frontier (matched pairs; in-band pairs={n_band})")
    for model_name in ["LSPIN (no smooth)", "LSPIN + patient smooth"]:
        sub = df[df["model"] == model_name].sort_values("cv_khard_mean")
        if len(sub) < 4:
            continue
        x = sub["cv_khard_mean"].to_numpy(dtype=float)
        y = sub["cv_val_cindex_mean"].to_numpy(dtype=float)
        color = MODEL_PALETTE.get(model_name, "#444444")
        if HAVE_LOWESS:
            frac = min(0.5, max(0.2, 12.0 / len(sub)))
            sm = sm_lowess(y, x, frac=frac, it=0, return_sorted=True)
            ax.plot(sm[:, 0], sm[:, 1], color=color, lw=2.2, alpha=0.95)
        else:
            # Fallback: smooth running median trend when LOWESS dependency is unavailable.
            k = int(max(5, min(15, len(sub) // 3)))
            y_sm = pd.Series(y).rolling(window=k, min_periods=2, center=True).median().to_numpy()
            keep = np.isfinite(y_sm)
            ax.plot(x[keep], y_sm[keep], color=color, lw=2.0, alpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def matched_sparsity_pairs(cv_nosim: pd.DataFrame, cv_sim: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if cv_nosim is None or cv_sim is None or cv_nosim.empty or cv_sim.empty:
        return pd.DataFrame()
    sim = cv_sim.copy()
    for _, r in cv_nosim.iterrows():
        d = (sim["cv_khard_mean"] - float(r["cv_khard_mean"])).abs()
        j = int(d.idxmin())
        s = sim.loc[j]
        rows.append(
            {
                "nosim_lambda_sparse": float(r["lambda_sparse"]),
                "sim_lambda_sparse": float(s["lambda_sparse"]),
                "sim_lambda_sample_smooth": float(s["lambda_sample_smooth"]),
                "nosim_khard": float(r["cv_khard_mean"]),
                "sim_khard": float(s["cv_khard_mean"]),
                "khard_abs_diff": float(abs(float(r["cv_khard_mean"]) - float(s["cv_khard_mean"]))),
                "nosim_cindex": float(r["cv_val_cindex_mean"]),
                "sim_cindex": float(s["cv_val_cindex_mean"]),
                "delta_cindex_sim_minus_nosim": float(s["cv_val_cindex_mean"] - r["cv_val_cindex_mean"]),
                "nosim_edge_affinity": float(r.get("cv_edge_affinity_corr_mean", np.nan)),
                "sim_edge_affinity": float(s.get("cv_edge_affinity_corr_mean", np.nan)),
                "delta_edge_affinity_sim_minus_nosim": float(
                    s.get("cv_edge_affinity_corr_mean", np.nan) - r.get("cv_edge_affinity_corr_mean", np.nan)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("nosim_khard").reset_index(drop=True)




def save_fig_sparsity_vs_affinity_tradeoff(
    cv_nosim: pd.DataFrame,
    cv_sim: pd.DataFrame,
    outpath: Path,
) -> None:
    d0 = cv_nosim.copy()
    d0["model"] = "LSPIN (no smooth)"
    d1 = cv_sim.copy()
    d1["model"] = "LSPIN + patient smooth"
    df = pd.concat([d0, d1], ignore_index=True)

    y_col = "cv_edge_affinity_corr_mean" if "cv_edge_affinity_corr_mean" in df.columns else "cv_affinity_corr_mean"
    if y_col == "cv_edge_affinity_corr_mean" and np.isfinite(df[y_col].to_numpy(dtype=float)).sum() == 0:
        y_col = "cv_affinity_corr_mean"
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df[np.isfinite(df["cv_khard_mean"]) & np.isfinite(df[y_col])].copy()
    plt.figure(figsize=(9.2, 6.0))
    if df.empty:
        ax = plt.gca()
        ax.text(0.5, 0.5, "No finite affinity points to plot", ha="center", va="center")
        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(outpath, dpi=180)
        plt.close()
        return
    ax = sns.scatterplot(
        data=df,
        x="cv_khard_mean",
        y=y_col,
        hue="model",
        style="model",
        s=70,
        palette=MODEL_PALETTE,
    )
    ax.set_xlabel("CV mean hard-selected features per patient (Khard)")
    if y_col == "cv_edge_affinity_corr_mean":
        ax.set_ylabel("Edge-level affinity consistency (gate cosine vs patient-kNN affinity)")
    else:
        ax.set_ylabel("Affinity consistency (corr: full-space vs selected-feature patient graph)")
    ax.set_title("Sparsity-consistency tradeoff across LSPIN settings")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def summarize_with_ci(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    value_col: str,
) -> pd.DataFrame:
    if df is None or not len(df):
        return pd.DataFrame()
    g = df.groupby(group_cols, dropna=False)[value_col]
    out = g.agg(["count", "mean", "std"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["ci95_low"] = out["mean"] - 1.96 * out["sem"]
    out["ci95_high"] = out["mean"] + 1.96 * out["sem"]
    return out


def main() -> None:
    args = parse_args()
    preset = CV_GRID_PRESETS[str(args.cv_grid_preset)]
    if args.lspin_lambdas is None:
        args.lspin_lambdas = list(preset["lambdas"])
    if args.gate_sigmas is None:
        args.gate_sigmas = list(preset["sigmas"])
    if args.sample_smooth_grid is None:
        args.sample_smooth_grid = list(preset["sample_smooth"])

    device = resolve_device(args.device)
    if isinstance(device, str) and device.startswith("cuda:"):
        try:
            torch.cuda.set_device(int(device.split(":")[1]))
        except Exception:
            pass
    set_seed(args.seed)

    outdir = args.outdir
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_dir or (Path("runs") / f"{args.results_prefix}_{args.dataset_name}_{stamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (results_dir / "run.log")

    def log(msg: str) -> None:
        tmsg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(tmsg, flush=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(tmsg + "\n")
            lf.flush()

    log(f"Starting run dataset={args.dataset_name} seed={args.seed} device={device}")
    log(
        "CV grid preset="
        f"{args.cv_grid_preset} (lambdas={len(args.lspin_lambdas)}, sigmas={len(args.gate_sigmas)}, "
        f"smooth_vals={len([v for v in args.sample_smooth_grid if float(v) > 0.0])})"
    )
    log(f"Results dir: {results_dir}")

    data = load_split_artifacts(outdir)
    log(f"Loaded split artifacts from {outdir}")
    split = make_val_split(
        data["X_train"],
        data["time_train"],
        data["event_train"],
        data["histo_train"],
        val_frac=args.val_frac,
        seed=args.seed,
    )

    Xt_tr = as_torch(split["Xt_tr"])
    tt_tr = as_torch(split["tt_tr"])
    et_tr = as_torch(split["et_tr"])

    Xt_val = as_torch(split["Xt_val"])
    tt_val = as_torch(split["tt_val"])
    et_val = as_torch(split["et_val"])

    Xt_test = as_torch(data["X_test"])
    tt_test = as_torch(data["time_test"])
    et_test = as_torch(data["event_test"])

    hidden_dims = tuple(args.mlp_hidden)

    # 1) Pilot MLP fit to set target sparsity for LSPIN hyperparameter search
    mlp_pilot = make_mlp(
        input_dim=data["X_train"].shape[1],
        hidden_dims=hidden_dims,
        dropout_p=args.mlp_dropout,
    )
    mlp_pilot_info = train_mlp_l1(
        mlp_pilot,
        Xt_tr,
        tt_tr,
        et_tr,
        Xt_val,
        tt_val,
        et_val,
        lr=args.mlp_lr,
        lambda_l1_input=args.mlp_l1,
        weight_decay=args.mlp_weight_decay,
        max_epochs=args.mlp_max_epochs,
        patience=args.mlp_patience,
        device=device,
    )
    if args.lasso_score == "l2":
        lasso_scores_pilot = input_l2_scores(mlp_pilot)
    else:
        lasso_scores_pilot = input_maxabs_scores(mlp_pilot)
    lasso_active_target = int((lasso_scores_pilot >= args.lasso_cutoff).sum())
    target_k_cv = float(args.target_k_fixed) if args.target_k_source == "fixed" else float(lasso_active_target)

    # 2) Internal CV once on train set to choose LSPIN configs
    smooth_grid = [float(v) for v in args.sample_smooth_grid if float(v) > 0.0]
    log("Starting CV for LSPIN no-smoothing configuration")
    cv_nosim = select_lspin_config_via_cv(
        X_train_np=data["X_train"],
        time_train_np=data["time_train"],
        event_train_np=data["event_train"],
        histo_train_np=data["histo_train"],
        input_dim=data["X_train"].shape[1],
        device=device,
        seed=args.seed,
        hidden_dim=args.lspin_hidden,
        gating_hidden_dim=args.gating_hidden,
        gate_a=args.gate_a,
        dropout_p=args.lspin_dropout,
        risk_hidden_dims=hidden_dims,
        risk_dropout_p=args.mlp_dropout,
        lambdas=args.lspin_lambdas,
        sigmas=args.gate_sigmas,
        sample_smooth_vals=[0.0],
        target_k=float(target_k_cv),
        knn_k=args.knn_k,
        lr=args.lspin_lr,
        weight_decay=args.lspin_weight_decay,
        batch_size=args.lspin_batch_size,
        cv_folds=args.lspin_cv_folds,
        cv_max_epochs=args.lspin_cv_max_epochs,
        cv_patience=args.lspin_cv_patience,
        selection_metric=args.selection_metric,
        log_fn=log,
        log_prefix="[cv-nosmooth] ",
    )
    best_nosim = pick_cv_config_with_matched_sparsity(
        cv_nosim,
        selection_metric=args.selection_metric,
        k_low=float(args.sparsity_band_low),
        k_high=float(args.sparsity_band_high),
    )
    best_nosim_cindex = cv_nosim.sort_values("cv_val_cindex_mean", ascending=False).iloc[0].to_dict()
    best_nosim_ibs = cv_nosim.sort_values("cv_ibs_mean", ascending=True).iloc[0].to_dict()
    log(
        "Finished no-smoothing CV: "
        f"lambda={best_nosim['lambda_sparse']}, sigma={best_nosim['gate_sigma']}, "
        f"khard={best_nosim['cv_khard_mean']:.2f}, cv_c={best_nosim['cv_val_cindex_mean']:.4f}"
    )

    log("Starting CV for LSPIN smoothing configuration")
    cv_sim = select_lspin_config_via_cv(
        X_train_np=data["X_train"],
        time_train_np=data["time_train"],
        event_train_np=data["event_train"],
        histo_train_np=data["histo_train"],
        input_dim=data["X_train"].shape[1],
        device=device,
        seed=args.seed + 77,
        hidden_dim=args.lspin_hidden,
        gating_hidden_dim=args.gating_hidden,
        gate_a=args.gate_a,
        dropout_p=args.lspin_dropout,
        risk_hidden_dims=hidden_dims,
        risk_dropout_p=args.mlp_dropout,
        lambdas=args.lspin_lambdas,
        sigmas=args.gate_sigmas,
        sample_smooth_vals=smooth_grid if len(smooth_grid) else [args.sample_smooth],
        target_k=float(target_k_cv),
        knn_k=args.knn_k,
        lr=args.lspin_lr,
        weight_decay=args.lspin_weight_decay,
        batch_size=args.lspin_batch_size,
        cv_folds=args.lspin_cv_folds,
        cv_max_epochs=args.lspin_cv_max_epochs,
        cv_patience=args.lspin_cv_patience,
        selection_metric=args.selection_metric,
        log_fn=log,
        log_prefix="[cv-smooth] ",
    )
    best_sim = pick_cv_config_with_matched_sparsity(
        cv_sim,
        selection_metric=args.selection_metric,
        k_low=float(args.sparsity_band_low),
        k_high=float(args.sparsity_band_high),
    )
    best_sim_cindex = cv_sim.sort_values("cv_val_cindex_mean", ascending=False).iloc[0].to_dict()
    best_sim_ibs = cv_sim.sort_values("cv_ibs_mean", ascending=True).iloc[0].to_dict()
    log(
        "Finished smoothing CV: "
        f"lambda={best_sim['lambda_sparse']}, sigma={best_sim['gate_sigma']}, "
        f"smooth={best_sim['lambda_sample_smooth']}, khard={best_sim['cv_khard_mean']:.2f}, "
        f"cv_c={best_sim['cv_val_cindex_mean']:.4f}"
    )

    concrete_best_nosim = None
    concrete_best_sim = None
    concrete_cv_nosim = None
    concrete_cv_sim = None
    if args.include_concrete and args.concrete_own_cv:
        concrete_lambdas = densify_lambda_grid([float(x) for x in args.lspin_lambdas], points_per_interval=2)
        log(f"Concrete CV lambda grid size={len(concrete_lambdas)}")
        log("Starting CV for Concrete no-smoothing configuration")
        concrete_cv_nosim = select_lspin_config_via_cv(
            X_train_np=data["X_train"],
            time_train_np=data["time_train"],
            event_train_np=data["event_train"],
            histo_train_np=data["histo_train"],
            input_dim=data["X_train"].shape[1],
            device=device,
            seed=args.seed + 131,
            hidden_dim=args.lspin_hidden,
            gating_hidden_dim=args.gating_hidden,
            gate_a=args.gate_a,
            dropout_p=args.lspin_dropout,
            risk_hidden_dims=hidden_dims,
            risk_dropout_p=args.mlp_dropout,
            lambdas=concrete_lambdas,
            sigmas=[0.1],
            sample_smooth_vals=[0.0],
            target_k=float(target_k_cv),
            knn_k=args.knn_k,
            lr=args.lspin_lr,
            weight_decay=args.lspin_weight_decay,
            batch_size=args.lspin_batch_size,
            cv_folds=args.lspin_cv_folds,
            cv_max_epochs=args.lspin_cv_max_epochs,
            cv_patience=args.lspin_cv_patience,
            selection_metric=args.selection_metric,
            gate_type="concrete",
            log_fn=log,
            log_prefix="[cv-concrete-nosmooth] ",
        )
        concrete_best_nosim = pick_cv_config_with_matched_sparsity(
            concrete_cv_nosim,
            selection_metric=args.selection_metric,
            k_low=float(args.sparsity_band_low),
            k_high=float(args.sparsity_band_high),
        )
        log(
            "Finished Concrete no-smoothing CV: "
            f"lambda={concrete_best_nosim['lambda_sparse']}, khard={concrete_best_nosim['cv_khard_mean']:.2f}, "
            f"cv_c={concrete_best_nosim['cv_val_cindex_mean']:.4f}"
        )

        log("Starting CV for Concrete smoothing configuration")
        concrete_cv_sim = select_lspin_config_via_cv(
            X_train_np=data["X_train"],
            time_train_np=data["time_train"],
            event_train_np=data["event_train"],
            histo_train_np=data["histo_train"],
            input_dim=data["X_train"].shape[1],
            device=device,
            seed=args.seed + 171,
            hidden_dim=args.lspin_hidden,
            gating_hidden_dim=args.gating_hidden,
            gate_a=args.gate_a,
            dropout_p=args.lspin_dropout,
            risk_hidden_dims=hidden_dims,
            risk_dropout_p=args.mlp_dropout,
            lambdas=concrete_lambdas,
            sigmas=[0.1],
            sample_smooth_vals=smooth_grid if len(smooth_grid) else [args.sample_smooth],
            target_k=float(target_k_cv),
            knn_k=args.knn_k,
            lr=args.lspin_lr,
            weight_decay=args.lspin_weight_decay,
            batch_size=args.lspin_batch_size,
            cv_folds=args.lspin_cv_folds,
            cv_max_epochs=args.lspin_cv_max_epochs,
            cv_patience=args.lspin_cv_patience,
            selection_metric=args.selection_metric,
            gate_type="concrete",
            log_fn=log,
            log_prefix="[cv-concrete-smooth] ",
        )
        concrete_best_sim = pick_cv_config_with_matched_sparsity(
            concrete_cv_sim,
            selection_metric=args.selection_metric,
            k_low=float(args.sparsity_band_low),
            k_high=float(args.sparsity_band_high),
        )
        log(
            "Finished Concrete smoothing CV: "
            f"lambda={concrete_best_sim['lambda_sparse']}, smooth={concrete_best_sim['lambda_sample_smooth']}, "
            f"khard={concrete_best_sim['cv_khard_mean']:.2f}, cv_c={concrete_best_sim['cv_val_cindex_mean']:.4f}"
        )

    # 3) Final training/eval across multiple random seeds
    final_seed_start = args.seed if args.final_seed_start is None else int(args.final_seed_start)
    final_seeds = [int(final_seed_start + i) for i in range(int(args.n_final_seeds))]
    if len(final_seeds) == 0:
        raise ValueError("n_final_seeds must be >= 1")

    overall_rows = []
    by_h_rows = []
    feature_rows = []
    ibs_parts = []

    models_first_seed = None
    lspin_nosim_first = None
    lspin_sim_first = None
    concrete_nosim_first = None
    concrete_sim_first = None
    lspin_nosim_info = None
    lspin_sim_info = None
    concrete_nosim_info = None
    concrete_sim_info = None
    sweep_nosim = None
    sweep_sim = None
    mlp_info = None

    A_sample_train = lp.compute_knn_affinity(split["Xt_tr"], k=args.knn_k)

    for s in final_seeds:
        log(f"Starting final-seed run: seed={s}")
        set_seed(s)
        mlp = make_mlp(
            input_dim=data["X_train"].shape[1],
            hidden_dims=hidden_dims,
            dropout_p=args.mlp_dropout,
        )
        mlp_info_seed = train_mlp_l1(
            mlp,
            Xt_tr,
            tt_tr,
            et_tr,
            Xt_val,
            tt_val,
            et_val,
            lr=args.mlp_lr,
            lambda_l1_input=args.mlp_l1,
            weight_decay=args.mlp_weight_decay,
            max_epochs=args.mlp_max_epochs,
            patience=args.mlp_patience,
            device=device,
        )

        if args.lasso_score == "l2":
            lasso_scores = input_l2_scores(mlp)
        else:
            lasso_scores = input_maxabs_scores(mlp)
        keep_mask = lasso_scores >= args.lasso_cutoff
        lasso_active_n = int(keep_mask.sum())
        mlp_pruned = prune_mlp_input_columns(mlp, keep_mask).to(device)
        Xt_tr_pruned = mask_tensor_columns(Xt_tr, keep_mask)
        Xt_test_pruned = mask_tensor_columns(Xt_test, keep_mask)

        lspin_nosim, lspin_nosim_info_seed, sweep_nosim_seed = tune_lspin_for_target_k(
            lambdas=[float(best_nosim["lambda_sparse"])],
            target_k=float(target_k_cv),
            Xt_tr=Xt_tr,
            tt_tr=tt_tr,
            et_tr=et_tr,
            Xt_val=Xt_val,
            tt_val=tt_val,
            et_val=et_val,
            Xt_test=Xt_test,
            tt_test=tt_test,
            et_test=et_test,
            device=device,
            base_seed=s,
            input_dim=data["X_train"].shape[1],
            hidden_dim=args.lspin_hidden,
            gating_hidden_dim=args.gating_hidden,
        gate_sigma=float(best_nosim["gate_sigma"]),
        gate_a=args.gate_a,
        dropout_p=args.lspin_dropout,
        risk_hidden_dims=hidden_dims,
        risk_dropout_p=args.mlp_dropout,
            lr=args.lspin_lr,
            weight_decay=args.lspin_weight_decay,
            batch_size=args.lspin_batch_size,
            max_epochs=args.lspin_max_epochs,
            patience=args.lspin_patience,
            lambda_sample_smooth=0.0,
            A_sample_train=A_sample_train,
            gate_type="lspin_tf",
            log_fn=log,
            log_prefix=f"[seed={s}][lspin-nosmooth] ",
        )

        lspin_sim, lspin_sim_info_seed, sweep_sim_seed = tune_lspin_for_target_k(
            lambdas=[float(best_sim["lambda_sparse"])],
            target_k=float(target_k_cv),
            Xt_tr=Xt_tr,
            tt_tr=tt_tr,
            et_tr=et_tr,
            Xt_val=Xt_val,
            tt_val=tt_val,
            et_val=et_val,
            Xt_test=Xt_test,
            tt_test=tt_test,
            et_test=et_test,
            device=device,
            base_seed=s + 100,
            input_dim=data["X_train"].shape[1],
            hidden_dim=args.lspin_hidden,
            gating_hidden_dim=args.gating_hidden,
        gate_sigma=float(best_sim["gate_sigma"]),
        gate_a=args.gate_a,
        dropout_p=args.lspin_dropout,
        risk_hidden_dims=hidden_dims,
        risk_dropout_p=args.mlp_dropout,
            lr=args.lspin_lr,
            weight_decay=args.lspin_weight_decay,
            batch_size=args.lspin_batch_size,
            max_epochs=args.lspin_max_epochs,
            patience=args.lspin_patience,
            lambda_sample_smooth=float(best_sim["lambda_sample_smooth"]),
            A_sample_train=A_sample_train,
            gate_type="lspin_tf",
            log_fn=log,
            log_prefix=f"[seed={s}][lspin-smooth] ",
        )

        concrete_nosim = None
        concrete_sim = None
        concrete_nosim_info_seed = None
        concrete_sim_info_seed = None
        if args.include_concrete:
            c_nosim = concrete_best_nosim if concrete_best_nosim is not None else best_nosim
            c_sim = concrete_best_sim if concrete_best_sim is not None else best_sim
            concrete_nosim, concrete_nosim_info_seed, _ = tune_lspin_for_target_k(
                lambdas=[float(c_nosim["lambda_sparse"])],
                target_k=float(target_k_cv),
                Xt_tr=Xt_tr,
                tt_tr=tt_tr,
                et_tr=et_tr,
                Xt_val=Xt_val,
                tt_val=tt_val,
                et_val=et_val,
                Xt_test=Xt_test,
                tt_test=tt_test,
                et_test=et_test,
                device=device,
                base_seed=s + 200,
                input_dim=data["X_train"].shape[1],
                hidden_dim=args.lspin_hidden,
                gating_hidden_dim=args.gating_hidden,
                gate_sigma=0.1,
                gate_a=args.gate_a,
                dropout_p=args.lspin_dropout,
                risk_hidden_dims=hidden_dims,
                risk_dropout_p=args.mlp_dropout,
                lr=args.lspin_lr,
                weight_decay=args.lspin_weight_decay,
                batch_size=args.lspin_batch_size,
                max_epochs=args.lspin_max_epochs,
                patience=args.lspin_patience,
                lambda_sample_smooth=0.0,
                A_sample_train=A_sample_train,
                gate_type="concrete",
                log_fn=log,
                log_prefix=f"[seed={s}][concrete-nosmooth] ",
            )

            concrete_sim, concrete_sim_info_seed, _ = tune_lspin_for_target_k(
                lambdas=[float(c_sim["lambda_sparse"])],
                target_k=float(target_k_cv),
                Xt_tr=Xt_tr,
                tt_tr=tt_tr,
                et_tr=et_tr,
                Xt_val=Xt_val,
                tt_val=tt_val,
                et_val=et_val,
                Xt_test=Xt_test,
                tt_test=tt_test,
                et_test=et_test,
                device=device,
                base_seed=s + 300,
                input_dim=data["X_train"].shape[1],
                hidden_dim=args.lspin_hidden,
                gating_hidden_dim=args.gating_hidden,
                gate_sigma=0.1,
                gate_a=args.gate_a,
                dropout_p=args.lspin_dropout,
                risk_hidden_dims=hidden_dims,
                risk_dropout_p=args.mlp_dropout,
                lr=args.lspin_lr,
                weight_decay=args.lspin_weight_decay,
                batch_size=args.lspin_batch_size,
                max_epochs=args.lspin_max_epochs,
                patience=args.lspin_patience,
                lambda_sample_smooth=float(c_sim["lambda_sample_smooth"]),
                A_sample_train=A_sample_train,
                gate_type="concrete",
                log_fn=log,
                log_prefix=f"[seed={s}][concrete-smooth] ",
            )

        models = {
            "MLP + L1": mlp,
            "MLP + L1 (|w|<cutoff zeroed)": mlp_pruned,
            "LSPIN (no smooth)": RiskOnlyWrapper(lspin_nosim).to(device),
            "LSPIN + patient smooth": RiskOnlyWrapper(lspin_sim).to(device),
        }
        if args.include_concrete and concrete_nosim is not None and concrete_sim is not None:
            models["Concrete (no smooth)"] = RiskOnlyWrapper(concrete_nosim).to(device)
            models["Concrete + patient smooth"] = RiskOnlyWrapper(concrete_sim).to(device)

        for name, model in models.items():
            Xt_eval = Xt_test_pruned if name == "MLP + L1 (|w|<cutoff zeroed)" else Xt_test
            Xt_eval_tr = Xt_tr_pruned if name == "MLP + L1 (|w|<cutoff zeroed)" else Xt_tr
            c, _ = lp.eval_cindex(model, Xt_eval, tt_test, et_test, device=device)
            overall_rows.append({"seed": s, "model": name, "cindex": float(c)})
            d = cindex_by_histology(model, Xt_eval, tt_test, et_test, data["histo_test"], device=device)
            if len(d):
                d["seed"] = s
                d["model"] = name
                by_h_rows.append(d)

            d_ibs = integrated_brier_by_histology(
                model,
                Xt_tr=Xt_eval_tr,
                tt_tr=tt_tr,
                et_tr=et_tr,
                Xt_te=Xt_eval,
                tt_te=tt_test,
                et_te=et_test,
                histo_test=data["histo_test"],
                device=device,
            )
            if d_ibs is not None and len(d_ibs):
                d_ibs["seed"] = s
                d_ibs["model"] = name
                ibs_parts.append(d_ibs)

        _, kh_nosim_mean, kh_nosim_med = lp.eval_gates_hard_K(lspin_nosim, Xt_test, device=device, threshold=0.5)
        _, kh_sim_mean, kh_sim_med = lp.eval_gates_hard_K(lspin_sim, Xt_test, device=device, threshold=0.5)
        feature_rows.extend(
            [
                {"seed": s, "model": "MLP active genes", "metric": "global genes with |w|>=cutoff", "value": float(lasso_active_n)},
                {"seed": s, "model": "LSPIN (no smooth)", "metric": "hard-selected genes / patient (mean)", "value": float(kh_nosim_mean)},
                {"seed": s, "model": "LSPIN (no smooth)", "metric": "hard-selected genes / patient (median)", "value": float(kh_nosim_med)},
                {"seed": s, "model": "LSPIN + patient smooth", "metric": "hard-selected genes / patient (mean)", "value": float(kh_sim_mean)},
                {"seed": s, "model": "LSPIN + patient smooth", "metric": "hard-selected genes / patient (median)", "value": float(kh_sim_med)},
            ]
        )
        if args.include_concrete and concrete_nosim is not None and concrete_sim is not None:
            _, kh_conc_nosim_mean, kh_conc_nosim_med = lp.eval_gates_hard_K(
                concrete_nosim, Xt_test, device=device, threshold=0.5
            )
            _, kh_conc_sim_mean, kh_conc_sim_med = lp.eval_gates_hard_K(
                concrete_sim, Xt_test, device=device, threshold=0.5
            )
            feature_rows.extend(
                [
                    {"seed": s, "model": "Concrete (no smooth)", "metric": "hard-selected genes / patient (mean)", "value": float(kh_conc_nosim_mean)},
                    {"seed": s, "model": "Concrete (no smooth)", "metric": "hard-selected genes / patient (median)", "value": float(kh_conc_nosim_med)},
                    {"seed": s, "model": "Concrete + patient smooth", "metric": "hard-selected genes / patient (mean)", "value": float(kh_conc_sim_mean)},
                    {"seed": s, "model": "Concrete + patient smooth", "metric": "hard-selected genes / patient (median)", "value": float(kh_conc_sim_med)},
                ]
            )

        if models_first_seed is None:
            models_first_seed = models
            lspin_nosim_first = lspin_nosim
            lspin_sim_first = lspin_sim
            concrete_nosim_first = concrete_nosim
            concrete_sim_first = concrete_sim
            mlp_info = mlp_info_seed
            lspin_nosim_info = lspin_nosim_info_seed
            lspin_sim_info = lspin_sim_info_seed
            concrete_nosim_info = concrete_nosim_info_seed
            concrete_sim_info = concrete_sim_info_seed
            sweep_nosim = sweep_nosim_seed
            sweep_sim = sweep_sim_seed
        log(f"Finished final-seed run: seed={s}")

    df_overall_seed = pd.DataFrame(overall_rows)
    df_cindex_h_seed = pd.concat(by_h_rows, ignore_index=True) if by_h_rows else pd.DataFrame(columns=["seed", "histology", "n_test", "cindex", "model"])
    df_ibs_seed = pd.concat(ibs_parts, ignore_index=True) if ibs_parts else None
    df_feature_seed = pd.DataFrame(feature_rows)

    df_overall_ci = summarize_with_ci(df_overall_seed, group_cols=["model"], value_col="cindex")
    df_cindex_h_ci = summarize_with_ci(df_cindex_h_seed, group_cols=["histology", "model"], value_col="cindex")
    df_feature_counts = summarize_with_ci(df_feature_seed, group_cols=["model", "metric"], value_col="value")
    if len(df_feature_counts):
        df_feature_counts["value"] = df_feature_counts["mean"]
    df_ibs_ci = summarize_with_ci(df_ibs_seed, group_cols=["histology", "model"], value_col="ibs") if df_ibs_seed is not None else None

    # Use seed-level frames for seaborn error bars; also keep CI tables.
    df_overall = df_overall_seed.copy()
    df_cindex_h = df_cindex_h_seed.copy()
    df_ibs = df_ibs_seed.copy() if df_ibs_seed is not None else None

    # persist seed-level and CI tables
    df_overall.to_csv(results_dir / "overall_cindex_by_seed.csv", index=False)
    df_overall_ci.to_csv(results_dir / "overall_cindex_ci.csv", index=False)
    df_cindex_h.to_csv(results_dir / "cindex_by_histology_by_seed.csv", index=False)
    df_cindex_h_ci.to_csv(results_dir / "cindex_by_histology_ci.csv", index=False)
    # backward-compatible filenames now point to CI means
    df_overall_ci[["model", "mean"]].rename(columns={"mean": "cindex"}).to_csv(results_dir / "overall_cindex.csv", index=False)
    df_cindex_h_ci[["histology", "model", "mean"]].rename(columns={"mean": "cindex"}).to_csv(results_dir / "cindex_by_histology.csv", index=False)
    df_feature_counts.to_csv(results_dir / "feature_count_alignment.csv", index=False)
    df_feature_seed.to_csv(results_dir / "feature_count_alignment_by_seed.csv", index=False)
    sweep_nosim.to_csv(results_dir / "lspin_nosmooth_lambda_sweep.csv", index=False)
    sweep_sim.to_csv(results_dir / "lspin_smooth_lambda_sweep.csv", index=False)
    cv_nosim.to_csv(results_dir / "lspin_nosmooth_cv_grid.csv", index=False)
    cv_sim.to_csv(results_dir / "lspin_smooth_cv_grid.csv", index=False)
    if concrete_cv_nosim is not None and len(concrete_cv_nosim):
        concrete_cv_nosim.to_csv(results_dir / "concrete_nosmooth_cv_grid.csv", index=False)
    if concrete_cv_sim is not None and len(concrete_cv_sim):
        concrete_cv_sim.to_csv(results_dir / "concrete_smooth_cv_grid.csv", index=False)
    matched_df = matched_sparsity_pairs(cv_nosim, cv_sim)
    if len(matched_df):
        matched_df.to_csv(results_dir / "matched_sparsity_smooth_vs_nosmooth.csv", index=False)
    if df_ibs is not None:
        df_ibs.to_csv(results_dir / "ibs_by_histology_by_seed.csv", index=False)
        if df_ibs_ci is not None and len(df_ibs_ci):
            df_ibs_ci.to_csv(results_dir / "ibs_by_histology_ci.csv", index=False)
            df_ibs_ci[["histology", "model", "mean"]].rename(columns={"mean": "ibs"}).to_csv(results_dir / "ibs_by_histology.csv", index=False)

    # four PNG figures
    save_fig_cindex_by_histology(
        df_cindex_h,
        results_dir / "fig1_cindex_by_histology_all_models.png",
        dataset_name=args.dataset_name,
    )
    save_fig_overall_cindex(df_overall, results_dir / "fig2_overall_cindex_all_models.png")
    save_fig_feature_counts(df_feature_counts, results_dir / "fig3_feature_count_alignment.png")
    save_fig_ibs_or_delta(
        df_ibs,
        df_cindex_h,
        results_dir / "fig4_ibs_or_lasso_delta.png",
        dataset_name=args.dataset_name,
    )
    if models_first_seed is not None:
        save_fig_train_survival_curves(
            models_first_seed,
            X_train=as_torch(data["X_train"]),
            t_train=as_torch(data["time_train"]),
            e_train=as_torch(data["event_train"]),
            histo_train=data["histo_train"],
            device=device,
            outpath=results_dir / "fig5_train_survival_curves_km_vs_predicted.png",
        )
    save_fig_parsimony_sparse_regime(
        cv_nosim=cv_nosim,
        cv_sim=cv_sim,
        outpath=results_dir / "fig6_parsimony_in_sparse_regime.png",
        csv_out=results_dir / "parsimony_sparse_regime_100_150.csv",
        k_low=float(args.sparsity_band_low),
        k_high=float(args.sparsity_band_high),
    )
    save_fig_sparsity_vs_affinity_tradeoff(
        cv_nosim=cv_nosim,
        cv_sim=cv_sim,
        outpath=results_dir / "fig9_sparsity_vs_affinity_tradeoff.png",
    )
    if lspin_nosim_first is not None:
        cluster_nosim = save_fig_gate_heatmap_with_histo(
            lspin_nosim_first,
            Xt=Xt_test,
            histo=data["histo_test"],
            gene_names=data["gene_names"],
            device=device,
            outpath=results_dir / "fig7_gate_heatmap_nosmooth_with_histo.png",
            title=f"{args.dataset_name}: LSPIN no smooth gate heatmap (histo_group colorbar)",
            view="hard",
        )
        if cluster_nosim is not None:
            cluster_nosim.to_csv(results_dir / "heatmap_nosmooth_sample_order.csv", index=False)
    if lspin_sim_first is not None:
        cluster_sim = save_fig_gate_heatmap_with_histo(
            lspin_sim_first,
            Xt=Xt_test,
            histo=data["histo_test"],
            gene_names=data["gene_names"],
            device=device,
            outpath=results_dir / "fig8_gate_heatmap_smooth_with_histo.png",
            title=f"{args.dataset_name}: LSPIN + patient kernel gate heatmap (histo_group colorbar)",
            view="hard",
        )
        if cluster_sim is not None:
            cluster_sim.to_csv(results_dir / "heatmap_smooth_sample_order.csv", index=False)

    def compact_info(info: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if info is None:
            return None
        keep = {}
        for k in [
            "best_epoch",
            "best_val_obj",
            "best_val_cindex",
            "test_cindex",
            "Kexp_train_mean",
            "Khard_train_mean",
            "Kexp_test_mean",
            "Khard_test_mean",
        ]:
            if k not in info:
                continue
            v = info[k]
            if isinstance(v, torch.Tensor):
                v = float(v.detach().cpu().item())
            elif isinstance(v, np.generic):
                v = v.item()
            keep[k] = v
        return keep

    summary = {
        "dataset_name": str(args.dataset_name),
        "device": device,
        "source_outdir": str(outdir),
        "results_dir": str(results_dir),
        "seed": int(args.seed),
        "final_seeds": final_seeds,
        "selection_metric": str(args.selection_metric),
        "deterministic_repro_mode": True,
        "target_k_source": str(args.target_k_source),
        "target_k_cv_used": float(target_k_cv),
        "sparsity_band_low": float(args.sparsity_band_low),
        "sparsity_band_high": float(args.sparsity_band_high),
        "val_frac": float(args.val_frac),
        "lasso_cutoff": float(args.lasso_cutoff),
        "lasso_score": str(args.lasso_score),
        "mlp_active_genes_reference_from_pilot": int(lasso_active_target),
        "mlp_active_genes_mean_over_final_seeds": float(
            df_feature_seed.loc[
                (df_feature_seed["model"] == "MLP active genes")
                & (df_feature_seed["metric"] == "global genes with |w|>=cutoff"),
                "value",
            ].mean()
        ),
        "mlp_train_info_first_seed": mlp_info,
        "lspin_nosmooth_best": {
            "best_epoch": int(lspin_nosim_info.get("best_epoch", -1)),
            "best_val_obj": lspin_nosim_info.get("best_val_obj", None),
            "best_val_cindex": lspin_nosim_info.get("best_val_cindex", None),
            "test_cindex": lspin_nosim_info.get("test_cindex", None),
            "best_lambda_sparse": float(sweep_nosim.iloc[0]["lambda_sparse"]),
            "abs_err_vs_target_k_reference": float(sweep_nosim.iloc[0]["abs_err_target"]),
            "cv_selected_sigma": float(best_nosim["gate_sigma"]),
            "cv_selected_sample_smooth": float(best_nosim["lambda_sample_smooth"]),
            "cv_selected_khard_mean": float(best_nosim["cv_khard_mean"]),
            "cv_val_cindex_mean": float(best_nosim["cv_val_cindex_mean"]),
            "cv_ibs_mean": float(best_nosim["cv_ibs_mean"]),
            "cv_affinity_corr_mean": float(best_nosim.get("cv_affinity_corr_mean", np.nan)),
            "cv_edge_affinity_corr_mean": float(best_nosim.get("cv_edge_affinity_corr_mean", np.nan)),
        },
        "lspin_nosmooth_best_by_cindex": {
            "lambda_sparse": float(best_nosim_cindex["lambda_sparse"]),
            "gate_sigma": float(best_nosim_cindex["gate_sigma"]),
            "lambda_sample_smooth": float(best_nosim_cindex["lambda_sample_smooth"]),
            "cv_val_cindex_mean": float(best_nosim_cindex["cv_val_cindex_mean"]),
            "cv_ibs_mean": float(best_nosim_cindex["cv_ibs_mean"]),
        },
        "lspin_nosmooth_best_by_ibs": {
            "lambda_sparse": float(best_nosim_ibs["lambda_sparse"]),
            "gate_sigma": float(best_nosim_ibs["gate_sigma"]),
            "lambda_sample_smooth": float(best_nosim_ibs["lambda_sample_smooth"]),
            "cv_val_cindex_mean": float(best_nosim_ibs["cv_val_cindex_mean"]),
            "cv_ibs_mean": float(best_nosim_ibs["cv_ibs_mean"]),
        },
        "lspin_smooth_best": {
            "best_epoch": int(lspin_sim_info.get("best_epoch", -1)),
            "best_val_obj": lspin_sim_info.get("best_val_obj", None),
            "best_val_cindex": lspin_sim_info.get("best_val_cindex", None),
            "test_cindex": lspin_sim_info.get("test_cindex", None),
            "best_lambda_sparse": float(sweep_sim.iloc[0]["lambda_sparse"]),
            "abs_err_vs_target_k_reference": float(sweep_sim.iloc[0]["abs_err_target"]),
            "cv_selected_sigma": float(best_sim["gate_sigma"]),
            "cv_selected_sample_smooth": float(best_sim["lambda_sample_smooth"]),
            "cv_selected_khard_mean": float(best_sim["cv_khard_mean"]),
            "cv_val_cindex_mean": float(best_sim["cv_val_cindex_mean"]),
            "cv_ibs_mean": float(best_sim["cv_ibs_mean"]),
            "cv_affinity_corr_mean": float(best_sim.get("cv_affinity_corr_mean", np.nan)),
            "cv_edge_affinity_corr_mean": float(best_sim.get("cv_edge_affinity_corr_mean", np.nan)),
        },
        "lspin_smooth_best_by_cindex": {
            "lambda_sparse": float(best_sim_cindex["lambda_sparse"]),
            "gate_sigma": float(best_sim_cindex["gate_sigma"]),
            "lambda_sample_smooth": float(best_sim_cindex["lambda_sample_smooth"]),
            "cv_val_cindex_mean": float(best_sim_cindex["cv_val_cindex_mean"]),
            "cv_ibs_mean": float(best_sim_cindex["cv_ibs_mean"]),
        },
        "lspin_smooth_best_by_ibs": {
            "lambda_sparse": float(best_sim_ibs["lambda_sparse"]),
            "gate_sigma": float(best_sim_ibs["gate_sigma"]),
            "lambda_sample_smooth": float(best_sim_ibs["lambda_sample_smooth"]),
            "cv_val_cindex_mean": float(best_sim_ibs["cv_val_cindex_mean"]),
            "cv_ibs_mean": float(best_sim_ibs["cv_ibs_mean"]),
        },
        "concrete_enabled": bool(args.include_concrete),
        "concrete_own_cv_enabled": bool(args.include_concrete and args.concrete_own_cv),
        "cv_grid_preset": str(args.cv_grid_preset),
        "concrete_nosmooth_first_seed": compact_info(concrete_nosim_info),
        "concrete_smooth_first_seed": compact_info(concrete_sim_info),
        "concrete_nosmooth_cv_selected": None
        if concrete_best_nosim is None
        else {
            "lambda_sparse": float(concrete_best_nosim["lambda_sparse"]),
            "gate_sigma": float(concrete_best_nosim["gate_sigma"]),
            "lambda_sample_smooth": float(concrete_best_nosim["lambda_sample_smooth"]),
            "cv_khard_mean": float(concrete_best_nosim["cv_khard_mean"]),
            "cv_val_cindex_mean": float(concrete_best_nosim["cv_val_cindex_mean"]),
        },
        "concrete_smooth_cv_selected": None
        if concrete_best_sim is None
        else {
            "lambda_sparse": float(concrete_best_sim["lambda_sparse"]),
            "gate_sigma": float(concrete_best_sim["gate_sigma"]),
            "lambda_sample_smooth": float(concrete_best_sim["lambda_sample_smooth"]),
            "cv_khard_mean": float(concrete_best_sim["cv_khard_mean"]),
            "cv_val_cindex_mean": float(concrete_best_sim["cv_val_cindex_mean"]),
        },
        "gated_train_defaults": {
            "lr": float(args.lspin_lr),
            "weight_decay": float(args.lspin_weight_decay),
            "batch_size": int(args.lspin_batch_size),
            "max_epochs": int(args.lspin_max_epochs),
            "patience": int(args.lspin_patience),
            "lambda_sparse": float(sweep_nosim.iloc[0]["lambda_sparse"]),
            "lambda_sample_smooth": float(args.sample_smooth),
            "lambda_gene_smooth": 0.0,
            "dropout_p": float(args.lspin_dropout),
        },
        "cv_settings": {
            "folds": int(args.lspin_cv_folds),
            "max_epochs": int(args.lspin_cv_max_epochs),
            "patience": int(args.lspin_cv_patience),
            "selection_metric": str(args.selection_metric),
            "lspin_lambdas": [float(x) for x in args.lspin_lambdas],
            "gate_sigmas": [float(x) for x in args.gate_sigmas],
            "sample_smooth_grid": [float(x) for x in args.sample_smooth_grid],
        },
        "overall_cindex_ci": df_overall_ci.to_dict(orient="records"),
        "generated_files": sorted([p.name for p in results_dir.iterdir()]),
    }

    with open(results_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log("Wrote run_summary.json")

    print("Saved results to:", results_dir)
    log("Run complete")
    print("Overall C-index (mean ± 95% CI across seeds):")
    show_overall = df_overall_ci.copy()
    show_overall["summary"] = show_overall.apply(
        lambda r: f"{r['mean']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]",
        axis=1,
    )
    print(show_overall.sort_values("mean", ascending=False)[["model", "count", "summary"]].to_string(index=False))
    print("\nBy-histology C-index (mean ± 95% CI across seeds):")
    show_h = df_cindex_h_ci.copy()
    show_h["summary"] = show_h.apply(
        lambda r: f"{r['mean']:.4f} [{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]",
        axis=1,
    )
    print(show_h.sort_values(["histology", "mean"], ascending=[True, False])[["histology", "model", "count", "summary"]].to_string(index=False))


if __name__ == "__main__":
    main()
