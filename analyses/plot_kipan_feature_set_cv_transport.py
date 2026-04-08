#!/usr/bin/env python3
"""K-fold transport test for adaptive gate-derived feature sets.

This is a post-hoc reproducibility analysis: it does not retrain the adaptive
gated neural models. Instead, it uses their saved gates to define compact
feature sets, then asks whether those feature sets support survival prediction
across independent K-fold splits using a regularized Cox model.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ANALYSES_DIR = Path(__file__).resolve().parent
SDS_SRC = Path("/banach2/wes/lspin-repos/sparsedeepsurv/src")

for path in [SDS_SRC]:
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import cumulative_dynamic_auc, integrated_brier_score

import sparsedeepsurv as sds


KIPAN_RUN_DEFAULT = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
    "ch3_kipan_adaptive_v2_selfcontained_ste_lspinmoderate_randominit_20260405_081219"
)
KIPAN_DATA_DEFAULT = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/processed/"
    "kipan_20260209_213604"
)
BRCA_RUN_DEFAULT = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/runs/"
    "ch3_brca_adaptive_v2_selfcontained_ste_randominit_20260406_120115"
)
BRCA_DATA_DEFAULT = Path(
    "/banach2/wes/lspin-repos/sparsedeepsurv-paper/data/processed/"
    "tcga_brca20260214_001423"
)


@dataclass(frozen=True)
class Config:
    family: str
    selection: str
    gate_type: str
    gate_sigma: float
    lambda_sparse: float
    lambda_sample_smooth: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Define feature sets from adaptive gates and test their "
            "k-fold transportability."
        )
    )
    p.add_argument("--dataset", choices=["kipan", "brca"], default="kipan")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--families", nargs="+", default=["LSPIN", "Concrete"])
    p.add_argument("--selections", nargs="+", default=["smooth"])
    p.add_argument(
        "--feature-set-mode",
        choices=["consensus", "union"],
        default="consensus",
        help=(
            "consensus: mean gate rate across models and samples must exceed threshold; "
            "union: any saved model's sample gate rate can exceed threshold."
        ),
    )
    p.add_argument("--gate-rate-threshold", type=float, default=0.10)
    p.add_argument(
        "--max-features",
        type=int,
        default=200,
        help="Optional cap: keep the top genes by gate rate within each gate-derived feature set.",
    )
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--model-kind", choices=["cox", "mlp"], default="mlp")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260406)
    p.add_argument("--cox-alpha", type=float, default=1.0)
    p.add_argument("--cox-max-iter", type=int, default=200)
    p.add_argument("--mlp-hidden-dims", type=str, default="64,32")
    p.add_argument(
        "--linear-predictor",
        action="store_true",
        help="Use a one-layer linear DeepSurv predictor for the MLP transport test.",
    )
    p.add_argument("--mlp-dropout", type=float, default=0.0)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-l1", type=float, default=0.0)
    p.add_argument("--mlp-max-epochs", type=int, default=120)
    p.add_argument("--mlp-patience", type=int, default=12)
    p.add_argument("--roc-time-quantile", type=float, default=0.60)
    return p.parse_args()


def _resolve_default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.dataset == "brca":
        return args.results_dir or BRCA_RUN_DEFAULT, args.data_dir or BRCA_DATA_DEFAULT
    return args.results_dir or KIPAN_RUN_DEFAULT, args.data_dir or KIPAN_DATA_DEFAULT


def _load_dataset(dataset: str, data_dir: Path) -> dict:
    if dataset == "brca":
        return sds.load_brca_split_artifacts(data_dir)
    return sds.load_kipan_split_artifacts(data_dir)


def _close(series: pd.Series, value: float, tol: float = 1e-9) -> pd.Series:
    return np.isclose(pd.to_numeric(series, errors="coerce").astype(float), float(value), atol=tol)


def _read_rows(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("_partial_p*_worker*/run_rows.csv")):
        df = pd.read_csv(p)
        if df.empty:
            continue
        df["partial_dir"] = str(p.parent)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No partial run_rows.csv files found under {results_dir}")
    out = pd.concat(rows, ignore_index=True)
    out["model_family"] = out["model_family"].replace({"HardSigmoid": "LSPIN"})
    return out


def _state_path(row: pd.Series) -> Path:
    return (
        Path(str(row["partial_dir"]))
        / "state_dicts"
        / f"cfg{int(row['global_cfg_idx'])}_rep{int(row['rep_id'])}.pt"
    )


def _selected_configs(results_dir: Path, rows: pd.DataFrame, families: list[str], selections: list[str]) -> list[Config]:
    selected = pd.read_csv(results_dir / "selected_comparison_configs.csv")
    configs: list[Config] = []
    for family in families:
        for selection in selections:
            hit = selected[
                (selected["family"].astype(str) == family)
                & (selected["selection"].astype(str) == selection)
            ]
            if hit.empty:
                raise ValueError(f"Missing selected config for family={family}, selection={selection}")
            r = hit.iloc[0]
            matching = rows[
                (rows["model_family"].astype(str) == family)
                & _close(rows["gate_sigma"], float(r["gate_sigma"]))
                & _close(rows["lambda_sparse"], float(r["lambda_sparse"]))
                & _close(rows["lambda_sample_smooth"], float(r["lambda_sample_smooth"]))
            ]
            if matching.empty:
                raise ValueError(f"No run rows matched {family}/{selection}")
            configs.append(Config(
                family=family,
                selection=selection,
                gate_type=str(matching["gate_type"].iloc[0]),
                gate_sigma=float(r["gate_sigma"]),
                lambda_sparse=float(r["lambda_sparse"]),
                lambda_sample_smooth=float(r["lambda_sample_smooth"]),
            ))
    return configs


def _make_model(cfg: Config, input_dim: int) -> torch.nn.Module:
    return sds.make_model(
        input_dim=input_dim,
        gate_type=cfg.gate_type,
        gate_sigma=cfg.gate_sigma,
        temperature=0.3 if cfg.gate_type == "concrete" else 0.5,
        concrete_mode="ste" if cfg.gate_type == "concrete" else "relaxed",
    )


def _matching_saved_rows(rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hit = rows[
        (rows["model_family"].astype(str) == cfg.family)
        & _close(rows["gate_sigma"], cfg.gate_sigma)
        & _close(rows["lambda_sparse"], cfg.lambda_sparse)
        & _close(rows["lambda_sample_smooth"], cfg.lambda_sample_smooth)
    ].copy()
    hit["state_dict_path"] = hit.apply(_state_path, axis=1).astype(str)
    hit = hit[hit["state_dict_path"].map(lambda p: Path(p).exists())].copy()
    if hit.empty:
        raise FileNotFoundError(f"No saved checkpoints found for {cfg}")
    return hit.sort_values(["phase", "rep_id", "seed"])


def _gate_feature_set(
    cfg: Config,
    saved_rows: pd.DataFrame,
    X: np.ndarray,
    gene_names: np.ndarray,
    *,
    mode: str,
    threshold: float,
    hard_threshold: float,
    device: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    x_t = torch.tensor(X, dtype=torch.float32)
    input_dim = X.shape[1]
    per_model_rates = []
    model_meta = []

    for _, row in saved_rows.iterrows():
        model = _make_model(cfg, input_dim=input_dim).to(device)
        try:
            state = torch.load(str(row["state_dict_path"]), map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(str(row["state_dict_path"]), map_location=device)
        model.load_state_dict(state)
        _, _, hard, _ = sds.get_gates(
            model,
            x_t,
            device=device,
            hard_threshold=hard_threshold,
            batch_size=512,
        )
        rates = hard.numpy().astype(float).mean(axis=0)
        per_model_rates.append(rates)
        model_meta.append({
            "family": cfg.family,
            "selection": cfg.selection,
            "seed": int(row["seed"]),
            "phase": int(row["phase"]),
            "rep_id": int(row["rep_id"]),
            "state_dict_path": str(row["state_dict_path"]),
        })

    rate_matrix = np.vstack(per_model_rates)
    mean_rate = rate_matrix.mean(axis=0)
    max_rate = rate_matrix.max(axis=0)
    selected = mean_rate >= threshold if mode == "consensus" else max_rate >= threshold
    feat = pd.DataFrame({
        "family": cfg.family,
        "selection": cfg.selection,
        "feature_set": f"{cfg.family}_{cfg.selection}_{mode}_rate{threshold:g}",
        "gene": gene_names.astype(str),
        "gene_idx": np.arange(len(gene_names), dtype=int),
        "mean_gate_rate": mean_rate,
        "max_model_gate_rate": max_rate,
        "selected": selected,
        "n_saved_models": len(saved_rows),
    })
    meta = pd.DataFrame(model_meta)
    return feat, meta


def _surv_y(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    return np.array(list(zip(event.astype(bool), time.astype(float))), dtype=[("event", "?"), ("time", "<f8")])


def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd


def _safe_times(time_train: np.ndarray, time_test: np.ndarray, event_train: np.ndarray) -> np.ndarray:
    event_times = np.asarray(time_train)[np.asarray(event_train).astype(bool)]
    if len(event_times) < 5:
        return np.array([], dtype=float)
    times = np.quantile(event_times, [0.25, 0.5, 0.75]).astype(float)
    upper = min(float(np.max(time_train)), float(np.max(time_test))) * 0.98
    lower = max(float(np.min(time_train)), float(np.min(time_test))) * 1.02
    times = times[(times > lower) & (times < upper)]
    return np.unique(times)


def _eval_risk_metrics(
    *,
    risk_train: np.ndarray,
    risk_test: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    roc_time: float,
) -> tuple[dict, pd.DataFrame]:
    ytr = _surv_y(time_train, event_train)
    yte = _surv_y(time_test, event_test)
    out: dict[str, float | str] = {
        "status": "ok",
        "cindex": sds.concordance_index(risk_test, time_test, event_test),
    }

    times = _safe_times(time_train, time_test, event_train)
    if len(times):
        try:
            aucs, mean_auc = cumulative_dynamic_auc(ytr, yte, risk_test, times)
            out["dynamic_auc_mean"] = float(mean_auc)
            out["dynamic_auc_median_time"] = float(aucs[np.argmin(np.abs(times - np.median(times)))])
        except Exception:
            out["dynamic_auc_mean"] = np.nan
            out["dynamic_auc_median_time"] = np.nan
        try:
            # Calibrate arbitrary risk scores to survival curves via a 1D Cox model.
            cal = CoxPHSurvivalAnalysis(alpha=1e-6, n_iter=100, tol=1e-7)
            cal.fit(risk_train.reshape(-1, 1), ytr)
            surv_fns = cal.predict_survival_function(risk_test.reshape(-1, 1))
            surv = np.asarray([[fn(t) for t in times] for fn in surv_fns], dtype=float)
            out["integrated_brier_score"] = float(integrated_brier_score(ytr, yte, surv, times))
        except Exception:
            out["integrated_brier_score"] = np.nan
    else:
        out["dynamic_auc_mean"] = np.nan
        out["dynamic_auc_median_time"] = np.nan
        out["integrated_brier_score"] = np.nan

    roc_df = pd.DataFrame()
    valid = ~((time_test <= roc_time) & (~event_test.astype(bool)))
    binary = (time_test[valid] <= roc_time) & event_test[valid].astype(bool)
    if valid.sum() >= 10 and binary.any() and (~binary).any():
        try:
            fpr, tpr, _ = roc_curve(binary.astype(int), risk_test[valid])
            roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            out["naive_roc_auc_at_horizon"] = float(sklearn_auc(fpr, tpr))
            out["roc_horizon_time"] = float(roc_time)
            out["roc_n"] = int(valid.sum())
        except Exception:
            out["naive_roc_auc_at_horizon"] = np.nan
    else:
        out["naive_roc_auc_at_horizon"] = np.nan
        out["roc_horizon_time"] = float(roc_time)
        out["roc_n"] = int(valid.sum())
    return out, roc_df


def _fit_eval_cox(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_idx: np.ndarray,
    *,
    alpha: float,
    n_iter: int,
    roc_time: float,
) -> tuple[dict, pd.DataFrame]:
    if len(feature_idx) < 1:
        return {"status": "skipped_empty_feature_set"}, pd.DataFrame()
    xtr, xte = _standardize_train_test(X[train_idx][:, feature_idx], X[test_idx][:, feature_idx])
    ytr = _surv_y(time[train_idx], event[train_idx])
    model = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=n_iter, tol=1e-7)
    try:
        model.fit(xtr, ytr)
        risk_train = model.predict(xtr).astype(float)
        risk = model.predict(xte).astype(float)
    except Exception as exc:
        return {"status": f"fit_failed: {exc}"}, pd.DataFrame()

    return _eval_risk_metrics(
        risk_train=risk_train,
        risk_test=risk,
        time_train=time[train_idx],
        event_train=event[train_idx],
        time_test=time[test_idx],
        event_test=event[test_idx],
        roc_time=roc_time,
    )


def _parse_hidden_dims(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(s).replace(" ", "").split(",") if x)


def _fit_eval_mlp(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    histo: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_idx: np.ndarray,
    *,
    hidden_dims: tuple[int, ...],
    dropout: float,
    lr: float,
    l1: float,
    max_epochs: int,
    patience: int,
    seed: int,
    device: str,
    roc_time: float,
) -> tuple[dict, pd.DataFrame]:
    if len(feature_idx) < 1:
        return {"status": "skipped_empty_feature_set"}, pd.DataFrame()

    xtr_all, xte = _standardize_train_test(X[train_idx][:, feature_idx], X[test_idx][:, feature_idx])
    tr_time = time[train_idx]
    tr_event = event[train_idx]
    tr_histo = histo[train_idx]
    split_labels = np.array([f"{h}|{int(e)}" for h, e in zip(tr_histo, tr_event)], dtype=str)
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        inner_train, inner_val = next(splitter.split(xtr_all, split_labels))
    except Exception:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train_idx))
        cut = max(1, int(0.8 * len(perm)))
        inner_train, inner_val = perm[:cut], perm[cut:]

    model = sds.make_seeded_mlp(
        input_dim=len(feature_idx),
        hidden_dims=hidden_dims,
        dropout_p=dropout,
        seed=seed,
    )
    cfg = sds.MLPTrainConfig(
        lr=lr,
        lambda_l1_input=l1,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=128,
    )
    try:
        info = sds.train_deepsurv_mlp_l1(
            model,
            torch.tensor(xtr_all[inner_train], dtype=torch.float32),
            torch.tensor(tr_time[inner_train], dtype=torch.float32),
            torch.tensor(tr_event[inner_train].astype(float), dtype=torch.float32),
            torch.tensor(xtr_all[inner_val], dtype=torch.float32),
            torch.tensor(tr_time[inner_val], dtype=torch.float32),
            torch.tensor(tr_event[inner_val].astype(float), dtype=torch.float32),
            config=cfg,
            device=device,
        )
        _, risk_train = sds.eval_mlp_cindex(
            model,
            torch.tensor(xtr_all, dtype=torch.float32),
            torch.tensor(tr_time, dtype=torch.float32),
            torch.tensor(tr_event.astype(float), dtype=torch.float32),
            device=device,
        )
        _, risk_test = sds.eval_mlp_cindex(
            model,
            torch.tensor(xte, dtype=torch.float32),
            torch.tensor(time[test_idx], dtype=torch.float32),
            torch.tensor(event[test_idx].astype(float), dtype=torch.float32),
            device=device,
        )
    except Exception as exc:
        return {"status": f"fit_failed: {exc}"}, pd.DataFrame()

    out, roc_df = _eval_risk_metrics(
        risk_train=risk_train,
        risk_test=risk_test,
        time_train=tr_time,
        event_train=tr_event,
        time_test=time[test_idx],
        event_test=event[test_idx],
        roc_time=roc_time,
    )
    out["best_val_cindex"] = float(info.get("best_val_cindex", np.nan))
    out["best_epoch"] = int(info.get("best_epoch", -1))
    out["epochs_ran"] = int(info.get("epochs_ran", -1))
    return out, roc_df


def _plot_metrics(metrics: pd.DataFrame, out_png: Path, *, dataset: str) -> None:
    ok = metrics[metrics["status"] == "ok"].copy()
    if ok.empty:
        return
    labels = ok["feature_set_label"].drop_duplicates().tolist()
    metrics_to_plot = [
        ("cindex", "Held-out C-index", False),
        ("dynamic_auc_mean", "Mean dynamic AUC", False),
        ("integrated_brier_score", "Integrated Brier score", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))
    for ax, (col, ylabel, lower_better) in zip(axes, metrics_to_plot):
        data = [ok.loc[ok["feature_set_label"] == label, col].dropna().values for label in labels]
        ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#dce8ef", "edgecolor": "0.25"},
                   medianprops={"color": "0.1", "linewidth": 1.4})
        for i, vals in enumerate(data, start=1):
            if len(vals):
                jitter = np.linspace(-0.08, 0.08, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color="#d95f02",
                           edgecolor="0.25", linewidth=0.3, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", color="0.9")
        if lower_better:
            ax.set_title("lower is better")
        else:
            ax.set_title("higher is better")
    fig.suptitle(f"{dataset.upper()} gate-derived feature-set transport across k-folds", y=1.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(roc_df: pd.DataFrame, out_png: Path) -> None:
    if roc_df.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for label, sub in roc_df.groupby("feature_set_label", sort=False):
        # Interpolate each fold onto a common FPR grid.
        grid = np.linspace(0, 1, 101)
        curves = []
        for _, fold in sub.groupby(["fold", "random_id"], dropna=False):
            fold = fold.sort_values("fpr")
            curves.append(np.interp(grid, fold["fpr"], fold["tpr"]))
        if not curves:
            continue
        mean_tpr = np.mean(np.vstack(curves), axis=0)
        ax.plot(grid, mean_tpr, linewidth=2.0, label=label)
    ax.plot([0, 1], [0, 1], color="0.55", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Naive horizon ROC from cross-fold risk scores")
    ax.legend(frameon=False)
    ax.grid(color="0.92")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    results_dir, data_dir = _resolve_default_paths(args)
    outdir = args.outdir or results_dir
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_dataset(args.dataset, data_dir)
    X = np.vstack([data["X_train"], data["X_test"]]).astype(np.float32)
    time = np.concatenate([data["time_train"], data["time_test"]]).astype(float)
    event = np.concatenate([data["event_train"], data["event_test"]]).astype(bool)
    histo = np.concatenate([data["histo_train"], data["histo_test"]]).astype(str)
    gene_names = np.asarray(data["gene_names"]).astype(str)

    rows = _read_rows(results_dir)
    configs = _selected_configs(results_dir, rows, args.families, args.selections)
    feature_tables = []
    model_tables = []
    feature_sets: list[tuple[str, np.ndarray]] = []
    for cfg in configs:
        saved = _matching_saved_rows(rows, cfg)
        feat, meta = _gate_feature_set(
            cfg,
            saved,
            X,
            gene_names,
            mode=args.feature_set_mode,
            threshold=args.gate_rate_threshold,
            hard_threshold=args.hard_threshold,
            device=args.device,
        )
        label = f"{cfg.family} {cfg.selection}"
        feat["feature_set_label"] = label
        meta["feature_set_label"] = label
        selected_feat = feat.loc[feat["selected"]].copy()
        if args.max_features and args.max_features > 0 and len(selected_feat) > args.max_features:
            rank_col = "mean_gate_rate" if args.feature_set_mode == "consensus" else "max_model_gate_rate"
            keep_gene_idx = set(
                selected_feat.sort_values(rank_col, ascending=False)
                .head(args.max_features)["gene_idx"].astype(int)
            )
            feat["selected_before_cap"] = feat["selected"]
            feat["selected"] = feat["gene_idx"].astype(int).isin(keep_gene_idx)
            selected_feat = feat.loc[feat["selected"]].copy()
        else:
            feat["selected_before_cap"] = feat["selected"]
        idx = selected_feat["gene_idx"].astype(int).to_numpy()
        feature_sets.append((label, idx))
        feature_tables.append(feat)
        model_tables.append(meta)
        print(f"[info] {label}: selected {len(idx)} genes from {len(saved)} saved checkpoints", flush=True)

    feature_df = pd.concat(feature_tables, ignore_index=True)
    model_df = pd.concat(model_tables, ignore_index=True)

    strat = np.array([f"{h}|{int(e)}" for h, e in zip(histo, event)], dtype=str)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = list(cv.split(X, strat))
    event_times = time[event]
    roc_time = float(np.quantile(event_times, args.roc_time_quantile))
    rng = np.random.default_rng(args.seed)
    all_features = np.arange(X.shape[1], dtype=int)

    metric_rows = []
    roc_rows = []
    hidden_dims = tuple() if args.linear_predictor else _parse_hidden_dims(args.mlp_hidden_dims)
    for label, feat_idx in feature_sets:
        excluded_gated = set(int(i) for i in feat_idx)
        random_pool = np.asarray(
            [int(i) for i in all_features if int(i) not in excluded_gated],
            dtype=int,
        )
        if len(random_pool) < len(feat_idx):
            raise ValueError(
                f"Random pool for {label} has only {len(random_pool)} genes, "
                f"but needs {len(feat_idx)} genes."
            )
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            if args.model_kind == "cox":
                res, roc = _fit_eval_cox(
                    X,
                    time,
                    event,
                    train_idx,
                    test_idx,
                    feat_idx,
                    alpha=args.cox_alpha,
                    n_iter=args.cox_max_iter,
                    roc_time=roc_time,
                )
            else:
                res, roc = _fit_eval_mlp(
                    X,
                    time,
                    event,
                    histo,
                    train_idx,
                    test_idx,
                    feat_idx,
                    hidden_dims=hidden_dims,
                    dropout=args.mlp_dropout,
                    lr=args.mlp_lr,
                    l1=args.mlp_l1,
                    max_epochs=args.mlp_max_epochs,
                    patience=args.mlp_patience,
                    seed=args.seed + 1000 * fold_id,
                    device=args.device,
                    roc_time=roc_time,
                )
            res.update({
                "feature_set_label": label,
                "feature_set_type": "gated",
                "model_kind": args.model_kind,
                "fold": fold_id,
                "random_id": -1,
                "n_features": int(len(feat_idx)),
                "random_pool_excludes_gated": True,
                "random_pool_size": int(len(random_pool)),
                "random_gated_overlap_n": 0,
            })
            metric_rows.append(res)
            if not roc.empty:
                roc["feature_set_label"] = label
                roc["feature_set_type"] = "gated"
                roc["fold"] = fold_id
                roc["random_id"] = -1
                roc_rows.append(roc)

            for random_id in range(args.n_random):
                rand_idx = rng.choice(random_pool, size=len(feat_idx), replace=False)
                if args.model_kind == "cox":
                    rres, rroc = _fit_eval_cox(
                        X,
                        time,
                        event,
                        train_idx,
                        test_idx,
                        rand_idx,
                        alpha=args.cox_alpha,
                        n_iter=args.cox_max_iter,
                        roc_time=roc_time,
                    )
                else:
                    rres, rroc = _fit_eval_mlp(
                        X,
                        time,
                        event,
                        histo,
                        train_idx,
                        test_idx,
                        rand_idx,
                        hidden_dims=hidden_dims,
                        dropout=args.mlp_dropout,
                        lr=args.mlp_lr,
                        l1=args.mlp_l1,
                        max_epochs=args.mlp_max_epochs,
                        patience=args.mlp_patience,
                        seed=args.seed + 1000 * fold_id + 100 * (random_id + 1),
                        device=args.device,
                        roc_time=roc_time,
                    )
                rres.update({
                    "feature_set_label": f"{label} random matched",
                    "feature_set_type": "random_matched",
                    "source_feature_set_label": label,
                    "model_kind": args.model_kind,
                    "fold": fold_id,
                    "random_id": random_id,
                    "n_features": int(len(rand_idx)),
                    "random_pool_excludes_gated": True,
                    "random_pool_size": int(len(random_pool)),
                    "random_gated_overlap_n": int(len(set(int(i) for i in rand_idx) & excluded_gated)),
                })
                metric_rows.append(rres)
                if not rroc.empty:
                    rroc["feature_set_label"] = f"{label} random matched"
                    rroc["feature_set_type"] = "random_matched"
                    rroc["source_feature_set_label"] = label
                    rroc["fold"] = fold_id
                    rroc["random_id"] = random_id
                    roc_rows.append(rroc)

    metrics = pd.DataFrame(metric_rows)
    roc_df = pd.concat(roc_rows, ignore_index=True) if roc_rows else pd.DataFrame()
    summary = metrics[metrics["status"] == "ok"].groupby(
        ["feature_set_label", "feature_set_type", "model_kind", "n_features"], as_index=False
    ).agg(
        n_evals=("cindex", "count"),
        cindex_mean=("cindex", "mean"),
        cindex_sd=("cindex", "std"),
        dynamic_auc_mean=("dynamic_auc_mean", "mean"),
        integrated_brier_score_mean=("integrated_brier_score", "mean"),
        naive_roc_auc_at_horizon_mean=("naive_roc_auc_at_horizon", "mean"),
    )

    predictor_tag = "linear" if args.linear_predictor and args.model_kind == "mlp" else args.model_kind
    prefix = f"{args.dataset}_feature_set_cv_transport_{predictor_tag}_{args.feature_set_mode}_rate{args.gate_rate_threshold:g}"
    feature_csv = outdir / f"{prefix}_features.csv"
    models_csv = outdir / f"{prefix}_models_used.csv"
    metrics_csv = outdir / f"{prefix}_fold_metrics.csv"
    summary_csv = outdir / f"{prefix}_summary.csv"
    roc_csv = outdir / f"{prefix}_roc_curves.csv"
    fig_png = outdir / f"fig_{prefix}.png"
    roc_png = outdir / f"fig_{prefix}_roc.png"

    feature_df.to_csv(feature_csv, index=False)
    model_df.to_csv(models_csv, index=False)
    metrics.to_csv(metrics_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    roc_df.to_csv(roc_csv, index=False)
    _plot_metrics(metrics, fig_png, dataset=args.dataset)
    _plot_roc(roc_df, roc_png)

    print(f"[done] wrote {feature_csv}", flush=True)
    print(f"[done] wrote {models_csv}", flush=True)
    print(f"[done] wrote {metrics_csv}", flush=True)
    print(f"[done] wrote {summary_csv}", flush=True)
    print(f"[done] wrote {roc_csv}", flush=True)
    print(f"[done] wrote {fig_png}", flush=True)
    print(f"[done] wrote {roc_png}", flush=True)


if __name__ == "__main__":
    main()
