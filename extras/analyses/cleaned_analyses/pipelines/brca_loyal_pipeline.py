#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import NearestNeighbors

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import lspin_pytorch as lp


class RiskOnlyWrapper(torch.nn.Module):
    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        out = self.base(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TCGA BRCA notebook-loyal LSPIN pipeline")
    p.add_argument("--outdir", type=Path, default=Path("runs") / "tcga_brca20260214_001423")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-sparsity-grid", action="store_true")
    p.add_argument("--skip-sparsity-grid", action="store_true")
    p.add_argument("--copy1-lambda-sparse", type=float, default=3.59e-3)
    p.add_argument("--copy1-lambda-smooth", type=float, default=1.6681005372000593e-05)
    p.add_argument("--copy1-gate-sigma", type=float, default=0.2)
    p.add_argument("--copy1-lr", type=float, default=1e-4)
    p.add_argument("--copy1-max-epochs", type=int, default=250)
    p.add_argument("--copy1-patience", type=int, default=80)
    p.add_argument("--copy1-target-k", type=float, default=30.0)
    p.add_argument("--copy1-lambda-kmatch", type=float, default=0.0)
    p.add_argument("--copy1-knn-k", type=int, default=15)
    p.add_argument("--bar-n-seeds", type=int, default=5)
    return p.parse_args()


def resolve_device(d: str) -> str:
    if d == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if d == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return d


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_torch(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def load_data(outdir: Path):
    core = np.load(outdir / "splits_and_core.npz", allow_pickle=True)
    xscaled = np.load(outdir / "X_scaled.npz", allow_pickle=True)
    return {
        "X_train": xscaled["X_train"].astype(np.float32),
        "X_test": xscaled["X_test"].astype(np.float32),
        "time_train": core["time_train"].astype(np.float32),
        "time_test": core["time_test"].astype(np.float32),
        "event_train": core["event_train"].astype(np.uint8),
        "event_test": core["event_test"].astype(np.uint8),
        "histo_train": core["histo_train"].astype(str),
        "histo_test": core["histo_test"].astype(str),
        "gene_names": core["gene_names"].astype(str),
    }


def build_knn_adjacency_csr(X, k=5, pca_dim=50, metric="cosine", symmetrize=True):
    if hasattr(X, "detach"):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float32)
    p = min(pca_dim, X.shape[0] - 1, X.shape[1])
    Xr = PCA(n_components=p, random_state=0).fit_transform(X) if p >= 2 else X
    nn = NearestNeighbors(n_neighbors=min(k + 1, Xr.shape[0]), metric=metric)
    nn.fit(Xr)
    idx = nn.kneighbors(Xr, return_distance=False)
    idx = idx[:, 1:] if idx.shape[1] > 1 else idx
    rows = np.repeat(np.arange(X.shape[0]), idx.shape[1])
    cols = idx.reshape(-1)
    import scipy.sparse as sp
    A = sp.csr_matrix((np.ones(rows.shape[0], dtype=np.float32), (rows, cols)), shape=(X.shape[0], X.shape[0]))
    if symmetrize:
        A = ((A + A.T) > 0).astype(np.float32).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def make_lspin_tf(input_dim, gate_sigma=0.5, a=1.0):
    return lp.DeepSurvGated(
        input_dim=input_dim,
        hidden_dim=64,
        gating_hidden_dim=128,
        llspin=False,
        dropout_p=0.0,
        gate_type="lspin_tf",
        gate_sigma=float(gate_sigma),
        a=float(a),
        temperature=0.5,
    )


@torch.no_grad()
def get_tf_gates(model, X, device="cuda", hard_threshold=0.5, batch_size=512):
    model.eval()
    X = X.to(device)
    reg_probs, g_dets, hards, alphas = [], [], [], []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size]
        alpha = model.gating_net(xb)
        g_det = torch.clamp(model.a * alpha + 0.5, 0.0, 1.0)
        denom = (model.gate_sigma * (2.0**0.5) + 1e-8)
        reg_prob = 0.5 - 0.5 * torch.erf(((-1.0 / (2.0 * model.a) - alpha) / denom))
        hard = (g_det > hard_threshold).to(g_det.dtype)
        reg_probs.append(reg_prob.detach().cpu())
        g_dets.append(g_det.detach().cpu())
        hards.append(hard.detach().cpu())
        alphas.append(alpha.detach().cpu())
    return torch.cat(reg_probs), torch.cat(g_dets), torch.cat(hards), torch.cat(alphas)


def cindex_by_histology(model, Xt, tt, et, histo_labels, device):
    rows = []
    for h in np.unique(histo_labels):
        m = histo_labels == h
        if m.sum() < 10:
            continue
        c, _ = lp.eval_cindex(model, Xt[m], tt[m], et[m], device=device)
        rows.append({"histology": h, "n_test": int(m.sum()), "cindex": float(c)})
    return pd.DataFrame(rows)


def cindex_by_histology_with_ci(model, Xt, tt, et, histo_labels, device, n_boot=400, seed=123):
    rows = []
    rng = np.random.default_rng(seed)
    for h in np.unique(histo_labels):
        m = histo_labels == h
        n = int(m.sum())
        if n < 10:
            continue
        Xt_h = Xt[m]
        tt_h = tt[m]
        et_h = et[m]
        c_main, _ = lp.eval_cindex(model, Xt_h, tt_h, et_h, device=device)

        boot_vals = []
        for _ in range(int(n_boot)):
            idx = rng.integers(0, n, size=n)
            try:
                c_b, _ = lp.eval_cindex(model, Xt_h[idx], tt_h[idx], et_h[idx], device=device)
            except Exception:
                continue
            if np.isfinite(c_b):
                boot_vals.append(float(c_b))

        if len(boot_vals) >= 20:
            ci_lo = float(np.percentile(boot_vals, 2.5))
            ci_hi = float(np.percentile(boot_vals, 97.5))
        else:
            ci_lo = np.nan
            ci_hi = np.nan

        rows.append(
            {
                "histology": h,
                "n_test": n,
                "cindex": float(c_main),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "n_boot_ok": int(len(boot_vals)),
            }
        )
    return pd.DataFrame(rows)


def neighbor_gate_similarity(G, A):
    A = A.tocoo()
    sims = []
    for i, j in zip(A.row, A.col):
        if i == j:
            continue
        sims.append(np.dot(G[i], G[j]) / (np.linalg.norm(G[i]) * np.linalg.norm(G[j]) + 1e-8))
    return np.asarray(sims, dtype=float)


def train_mlp_with_l1_sweep(
    Xt_tr,
    tt_tr,
    et_tr,
    Xt_val,
    tt_val,
    et_val,
    Xt_test,
    tt_test,
    et_test,
    *,
    input_dim,
    device,
    seed,
):
    # Copy1-like plus slightly broader L1 settings to avoid random-performance failure.
    candidates = [
        {"lr": 1e-3, "lambda_l1_input": 1e-1, "weight_decay": 1e-5, "patience": 25, "max_epochs": 300, "dropout_p": 0.1},
        {"lr": 1e-3, "lambda_l1_input": 1e-2, "weight_decay": 1e-5, "patience": 25, "max_epochs": 300, "dropout_p": 0.1},
        {"lr": 1e-2, "lambda_l1_input": 1e-10, "weight_decay": 1e-5, "patience": 25, "max_epochs": 300, "dropout_p": 0.1},
        {"lr": 3e-4, "lambda_l1_input": 1e-2, "weight_decay": 1e-5, "patience": 40, "max_epochs": 300, "dropout_p": 0.1},
    ]
    rows = []
    best = None
    for i, cfg in enumerate(candidates):
        set_all_seeds(seed + i)
        m = lp.DeepSurvMLP(input_dim=input_dim, hidden_dims=(64, 32), dropout_p=float(cfg["dropout_p"]))
        info = lp.train_deepsurv_mlp_l1(
            m,
            Xt_tr,
            tt_tr,
            et_tr,
            Xt_val,
            tt_val,
            et_val,
            lr=float(cfg["lr"]),
            lambda_l1_input=float(cfg["lambda_l1_input"]),
            weight_decay=float(cfg["weight_decay"]),
            max_epochs=int(cfg["max_epochs"]),
            patience=int(cfg["patience"]),
            device=device,
        )
        c_test, _ = lp.eval_cindex(m, Xt_test, tt_test, et_test, device=device)
        row = {
            "run_id": i,
            **cfg,
            "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
            "test_cindex": float(c_test),
            "best_epoch": int(info.get("best_epoch", -1)),
        }
        rows.append(row)
        if best is None:
            best = (row, m, info)
        else:
            # prioritize validation C-index then test C-index as tie-break
            if (row["best_val_cindex"] > best[0]["best_val_cindex"]) or (
                abs(row["best_val_cindex"] - best[0]["best_val_cindex"]) <= 1e-8
                and row["test_cindex"] > best[0]["test_cindex"]
            ):
                best = (row, m, info)
    return best[1], best[2], best[0], pd.DataFrame(rows)


def train_mlp_with_cfg(
    Xt_tr,
    tt_tr,
    et_tr,
    Xt_val,
    tt_val,
    et_val,
    *,
    input_dim,
    device,
    seed,
    cfg,
):
    set_all_seeds(seed)
    m = lp.DeepSurvMLP(input_dim=input_dim, hidden_dims=(64, 32), dropout_p=float(cfg["dropout_p"]))
    info = lp.train_deepsurv_mlp_l1(
        m,
        Xt_tr,
        tt_tr,
        et_tr,
        Xt_val,
        tt_val,
        et_val,
        lr=float(cfg["lr"]),
        lambda_l1_input=float(cfg["lambda_l1_input"]),
        weight_decay=float(cfg["weight_decay"]),
        max_epochs=int(cfg["max_epochs"]),
        patience=int(cfg["patience"]),
        device=device,
    )
    return m, info


def run_one_lspin(Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, Xt_test, tt_test, et_test, *,
                  input_dim, gate_sigma, lam, lambda_sample_smooth, patience, A_sample_train, device,
                  lr=1e-2, weight_decay=1e-5, batch_size=128, max_epochs=200, target_K=None, lambda_Kmatch=0.0):
    model = make_lspin_tf(input_dim=input_dim, gate_sigma=gate_sigma, a=1.0).to(device)
    info = lp.train_gated_deepsurv(
        model,
        Xt_tr, tt_tr, et_tr,
        Xt_val, tt_val, et_val,
        Xt_test, tt_test, et_test,
        A_sample_train=A_sample_train,
        config=lp.GatedTrainConfig(
            lr=float(lr),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            max_epochs=int(max_epochs),
            patience=patience,
            lambda_sparse=float(lam),
            lambda_sample_smooth=float(lambda_sample_smooth),
            lambda_gene_smooth=0.0,
            target_K=target_K,
            lambda_Kmatch=float(lambda_Kmatch),
        ),
        device=device,
        verbose=False,
    )
    return model, info


def train_lspin_across_seeds(
    seeds,
    lam,
    gate_sigma,
    lambda_sample_smooth,
    Xt_tr,
    tt_tr,
    et_tr,
    Xt_val,
    tt_val,
    et_val,
    Xt_test,
    tt_test,
    et_test,
    histo_test,
    gene_names,
    device,
    topK=30,
    knn_k=15,
    lr=1e-2,
    max_epochs=200,
    patience=60,
    target_K=None,
    lambda_Kmatch=0.0,
):
    sets_by_seed = []
    rows = []
    models = []
    for s in seeds:
        set_all_seeds(s)
        A_train = build_knn_adjacency_csr(Xt_tr, k=int(knn_k), pca_dim=50)
        m, info = run_one_lspin(
            Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, Xt_test, tt_test, et_test,
            input_dim=Xt_tr.shape[1], gate_sigma=gate_sigma, lam=lam,
            lambda_sample_smooth=lambda_sample_smooth, patience=patience,
            A_sample_train=A_train, device=device, lr=lr, max_epochs=max_epochs,
            target_K=target_K, lambda_Kmatch=lambda_Kmatch,
        )
        models.append(m)
        _, gdet, _, _ = get_tf_gates(m, Xt_test, device=device, batch_size=512)
        G = gdet.numpy()
        per_h = {}
        for h in np.unique(histo_test):
            idx = np.where(histo_test == h)[0]
            if idx.size < 5:
                continue
            top_idx = np.argsort(G[idx].mean(axis=0))[::-1][:topK]
            per_h[h] = set(np.asarray(gene_names)[top_idx])
        sets_by_seed.append(per_h)
        rows.append({"seed": int(s), "test_cindex": float(info.get("test_cindex", np.nan))})

    pair_rows = []
    histos = sorted(set().union(*[set(d.keys()) for d in sets_by_seed]))
    for h in histos:
        for i in range(len(sets_by_seed)):
            for j in range(i + 1, len(sets_by_seed)):
                if h not in sets_by_seed[i] or h not in sets_by_seed[j]:
                    continue
                a, b = sets_by_seed[i][h], sets_by_seed[j][h]
                jac = 0.0 if (not a or not b) else (len(a & b) / len(a | b))
                pair_rows.append({"histology": h, "run_i": i, "run_j": j, "jaccard": float(jac)})

    return {
        "models": models,
        "df_runs": pd.DataFrame(rows),
        "df_pairwise": pd.DataFrame(pair_rows),
    }


def summarize_seed_histology(df_seed: pd.DataFrame) -> pd.DataFrame:
    if df_seed.empty:
        return pd.DataFrame(columns=["model", "histology", "n_test", "mean_cindex", "std_cindex", "q05_cindex", "q95_cindex", "n_seed"])
    g = (
        df_seed.groupby(["model", "histology", "n_test"], as_index=False)
        .agg(
            mean_cindex=("cindex", "mean"),
            std_cindex=("cindex", "std"),
            q05_cindex=("cindex", lambda s: float(np.quantile(s, 0.05))),
            q95_cindex=("cindex", lambda s: float(np.quantile(s, 0.95))),
            n_seed=("seed", "nunique"),
        )
    )
    g["std_cindex"] = g["std_cindex"].fillna(0.0)
    return g


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_all_seeds(args.seed)

    d = load_data(args.outdir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.results_dir or (Path("runs") / f"brca_loyal_{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    X_train = d["X_train"]
    X_test = d["X_test"]
    y_event = d["event_train"]
    y_histo = d["histo_train"]
    strat_key = np.array([f"{e}_{h}" for e, h in zip(y_event, y_histo)])
    ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    i_tr, i_val = next(ss.split(X_train, strat_key))

    Xt_tr, Xt_val, Xt_test = as_torch(X_train[i_tr]), as_torch(X_train[i_val]), as_torch(X_test)
    tt_tr, tt_val, tt_test = as_torch(d["time_train"][i_tr]), as_torch(d["time_train"][i_val]), as_torch(d["time_test"])
    et_tr, et_val, et_test = as_torch(d["event_train"][i_tr]), as_torch(d["event_train"][i_val]), as_torch(d["event_test"])

    # MLP baseline: sweep a few Copy1-like settings and use the best by val C-index
    mlp, mlp_info, mlp_best_cfg, mlp_sweep_df = train_mlp_with_l1_sweep(
        Xt_tr,
        tt_tr,
        et_tr,
        Xt_val,
        tt_val,
        et_val,
        Xt_test,
        tt_test,
        et_test,
        input_dim=X_train.shape[1],
        device=device,
        seed=args.seed,
    )
    mlp_sweep_df.to_csv(out / "mlp_l1_sweep_results.csv", index=False)

    # Copy1-inspired LSPIN regime
    copy1_lam = float(args.copy1_lambda_sparse)
    copy1_smooth = float(args.copy1_lambda_smooth)
    copy1_sigma = float(args.copy1_gate_sigma)
    copy1_lr = float(args.copy1_lr)
    copy1_max_epochs = int(args.copy1_max_epochs)
    copy1_patience = int(args.copy1_patience)
    copy1_target_k = float(args.copy1_target_k)
    copy1_lambda_kmatch = float(args.copy1_lambda_kmatch)
    copy1_knn_k = int(args.copy1_knn_k)

    A_train = build_knn_adjacency_csr(Xt_tr, k=copy1_knn_k, pca_dim=50)
    model_sim, info_sim = run_one_lspin(
        Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, Xt_test, tt_test, et_test,
        input_dim=X_train.shape[1], gate_sigma=copy1_sigma, lam=copy1_lam,
        lambda_sample_smooth=copy1_smooth, patience=copy1_patience, A_sample_train=A_train, device=device,
        lr=copy1_lr, max_epochs=copy1_max_epochs, target_K=copy1_target_k, lambda_Kmatch=copy1_lambda_kmatch,
    )
    model_nosim, info_nosim = run_one_lspin(
        Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, Xt_test, tt_test, et_test,
        input_dim=X_train.shape[1], gate_sigma=copy1_sigma, lam=copy1_lam,
        lambda_sample_smooth=0.0, patience=copy1_patience, A_sample_train=A_train, device=device,
        lr=copy1_lr, max_epochs=copy1_max_epochs, target_K=copy1_target_k, lambda_Kmatch=copy1_lambda_kmatch,
    )

    # Marginal C-index by histology from seeded retrains (mean +/- SD across seeds)
    bar_seeds = [args.seed + i for i in range(int(args.bar_n_seeds))]
    rows_bar = []
    for s in bar_seeds:
        mlp_s, _ = train_mlp_with_cfg(
            Xt_tr,
            tt_tr,
            et_tr,
            Xt_val,
            tt_val,
            et_val,
            input_dim=X_train.shape[1],
            device=device,
            seed=int(s),
            cfg=mlp_best_cfg,
        )
        d_mlp = cindex_by_histology(mlp_s, Xt_test, tt_test, et_test, d["histo_test"], device)
        d_mlp["model"] = "MLP + L1"
        d_mlp["seed"] = int(s)
        rows_bar.append(d_mlp)

        set_all_seeds(int(s))
        model_sim_s, _ = run_one_lspin(
            Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, Xt_test, tt_test, et_test,
            input_dim=X_train.shape[1], gate_sigma=copy1_sigma, lam=copy1_lam,
            lambda_sample_smooth=copy1_smooth, patience=copy1_patience, A_sample_train=A_train, device=device,
            lr=copy1_lr, max_epochs=copy1_max_epochs, target_K=copy1_target_k, lambda_Kmatch=copy1_lambda_kmatch,
        )
        d_sim = cindex_by_histology(RiskOnlyWrapper(model_sim_s).to(device), Xt_test, tt_test, et_test, d["histo_test"], device)
        d_sim["model"] = "LSPIN + sample smooth"
        d_sim["seed"] = int(s)
        rows_bar.append(d_sim)

        set_all_seeds(int(s))
        model_nosim_s, _ = run_one_lspin(
            Xt_tr, tt_tr, et_tr, Xt_val, tt_val, et_val, Xt_test, tt_test, et_test,
            input_dim=X_train.shape[1], gate_sigma=copy1_sigma, lam=copy1_lam,
            lambda_sample_smooth=0.0, patience=copy1_patience, A_sample_train=A_train, device=device,
            lr=copy1_lr, max_epochs=copy1_max_epochs, target_K=copy1_target_k, lambda_Kmatch=copy1_lambda_kmatch,
        )
        d_nosim = cindex_by_histology(RiskOnlyWrapper(model_nosim_s).to(device), Xt_test, tt_test, et_test, d["histo_test"], device)
        d_nosim["model"] = "LSPIN no smooth"
        d_nosim["seed"] = int(s)
        rows_bar.append(d_nosim)

    df_c_seed = pd.concat(rows_bar, ignore_index=True) if rows_bar else pd.DataFrame()
    df_c_seed.to_csv(out / "marginal_test_cindex_by_histology_by_seed.csv", index=False)
    df_c = summarize_seed_histology(df_c_seed)
    df_c.to_csv(out / "marginal_test_cindex_by_histology.csv", index=False)

    model_order = ["LSPIN + sample smooth", "LSPIN no smooth", "MLP + L1"]
    hist_order = sorted(df_c["histology"].unique().tolist()) if len(df_c) else []
    x = np.arange(len(hist_order), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, model_name in enumerate(model_order):
        sub = df_c[df_c["model"] == model_name].set_index("histology").reindex(hist_order)
        y = sub["mean_cindex"].to_numpy(float)
        err = sub["std_cindex"].fillna(0.0).to_numpy(float)
        xpos = x + (i - 1) * width
        ax.bar(xpos, y, width=width, label=model_name)
        finite = np.isfinite(y) & np.isfinite(err)
        if np.any(finite):
            ax.errorbar(
                xpos[finite],
                y[finite],
                yerr=err[finite],
                fmt="none",
                ecolor="black",
                capsize=3,
                lw=1,
            )
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(hist_order, rotation=30, ha="right")
    ax.set_ylabel("C-index")
    ax.set_title("Marginal test C-index by histology (mean +/- SD across seeds)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_marginal_test_cindex_by_histology.png", dpi=180)
    plt.close(fig)

    # Neighbor gate coherence like notebook
    _, gdet_sim, _, _ = get_tf_gates(model_sim, Xt_test, device=device)
    _, gdet_nosim, _, _ = get_tf_gates(model_nosim, Xt_test, device=device)
    G_sim = zscore(gdet_sim.numpy(), axis=0, nan_policy="omit")
    G_nosim = zscore(gdet_nosim.numpy(), axis=0, nan_policy="omit")
    G_sim = np.nan_to_num(G_sim)
    G_nosim = np.nan_to_num(G_nosim)
    A_test = build_knn_adjacency_csr(Xt_test, k=10, pca_dim=50)
    sim_sims = neighbor_gate_similarity(G_sim, A_test)
    nosim_sims = neighbor_gate_similarity(G_nosim, A_test)
    pd.DataFrame({
        "model": ["LSPIN + sample smooth", "LSPIN no smooth"],
        "mean_neighbor_similarity": [float(np.mean(sim_sims)), float(np.mean(nosim_sims))],
        "median_neighbor_similarity": [float(np.median(sim_sims)), float(np.median(nosim_sims))],
    }).to_csv(out / "neighbor_gate_similarity_summary.csv", index=False)

    # Seed stability block close to notebook cells 127+
    seeds = [0, 1, 2, 3, 4]
    res_sim = train_lspin_across_seeds(
        seeds=seeds, lam=7e-3, gate_sigma=0.2, lambda_sample_smooth=0.15,
        Xt_tr=Xt_tr, tt_tr=tt_tr, et_tr=et_tr, Xt_val=Xt_val, tt_val=tt_val, et_val=et_val,
        Xt_test=Xt_test, tt_test=tt_test, et_test=et_test, histo_test=d["histo_test"],
        gene_names=d["gene_names"], device=device, topK=30, knn_k=copy1_knn_k,
        lr=1e-2, max_epochs=200, patience=80, target_K=None, lambda_Kmatch=0.0,
    )
    res_nosim = train_lspin_across_seeds(
        seeds=seeds, lam=5e-3, gate_sigma=0.2, lambda_sample_smooth=0.0,
        Xt_tr=Xt_tr, tt_tr=tt_tr, et_tr=et_tr, Xt_val=Xt_val, tt_val=tt_val, et_val=et_val,
        Xt_test=Xt_test, tt_test=tt_test, et_test=et_test, histo_test=d["histo_test"],
        gene_names=d["gene_names"], device=device, topK=30, knn_k=copy1_knn_k,
        lr=1e-2, max_epochs=200, patience=60, target_K=None, lambda_Kmatch=0.0,
    )
    res_sim["df_pairwise"].to_csv(out / "seed_pairwise_jaccard_sim.csv", index=False)
    res_nosim["df_pairwise"].to_csv(out / "seed_pairwise_jaccard_nosim.csv", index=False)

    do_sparsity_grid = True
    if args.run_sparsity_grid:
        do_sparsity_grid = True
    if args.skip_sparsity_grid:
        do_sparsity_grid = False

    if do_sparsity_grid:
        lambdas = np.geomspace(1e-3, 5e-2, 12)
        rows = []
        for lam in lambdas:
            for smooth in [0.0, 0.1]:
                run = train_lspin_across_seeds(
                    seeds=[args.seed + int(lam * 1e6) + i for i in range(8)],
                    lam=float(lam), gate_sigma=0.1, lambda_sample_smooth=float(smooth),
                    Xt_tr=Xt_tr, tt_tr=tt_tr, et_tr=et_tr, Xt_val=Xt_val, tt_val=tt_val, et_val=et_val,
                    Xt_test=Xt_test, tt_test=tt_test, et_test=et_test, histo_test=d["histo_test"],
                    gene_names=d["gene_names"], device=device, topK=30, knn_k=copy1_knn_k,
                    lr=1e-3, max_epochs=200, patience=60, target_K=None, lambda_Kmatch=0.0,
                )
                rows.append({
                    "lambda_sparse": float(lam),
                    "lambda_sample_smooth": float(smooth),
                    "mean_test_cindex": float(run["df_runs"]["test_cindex"].mean()),
                    "std_test_cindex": float(run["df_runs"]["test_cindex"].std(ddof=0)),
                    "mean_jaccard": float(run["df_pairwise"]["jaccard"].mean()) if len(run["df_pairwise"]) else np.nan,
                })
        pd.DataFrame(rows).to_csv(out / "sparsity_grid_summary.csv", index=False)

    mlp_test_cindex, _ = lp.eval_cindex(mlp, Xt_test, tt_test, et_test, device=device)
    summary = {
        "results_dir": str(out),
        "seed": int(args.seed),
        "device": device,
        "mlp_best_val_cindex": float(mlp_info.get("best_val_cindex", np.nan)),
        "mlp_test_cindex": float(mlp_test_cindex),
        "mlp_selected_lr": float(mlp_best_cfg["lr"]),
        "mlp_selected_lambda_l1_input": float(mlp_best_cfg["lambda_l1_input"]),
        "mlp_selected_patience": int(mlp_best_cfg["patience"]),
        "mlp_selected_max_epochs": int(mlp_best_cfg["max_epochs"]),
        "lspin_sim_test_cindex": float(info_sim.get("test_cindex", np.nan)),
        "lspin_nosim_test_cindex": float(info_nosim.get("test_cindex", np.nan)),
        "copy1_lambda_sparse": copy1_lam,
        "copy1_lambda_smooth": copy1_smooth,
        "copy1_gate_sigma": copy1_sigma,
        "copy1_lr": copy1_lr,
        "copy1_target_k": copy1_target_k,
        "copy1_lambda_kmatch": copy1_lambda_kmatch,
        "copy1_knn_k": copy1_knn_k,
    }
    with open(out / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved results to:", out)


if __name__ == "__main__":
    main()
