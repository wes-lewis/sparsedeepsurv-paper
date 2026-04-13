#!/usr/bin/env python3
"""
KIPAN adaptive sweep v1.

Two-phase design:
  Phase 1: all configs × n_phase1_reps (default 3) — identifies promising configs.
  Phase 2: top_n_promising configs per (family, sigma) × remaining reps to reach
           n_total_reps (default 12).

Promising = highest mean C-index among configs with mean_Khard in [khard_min, khard_max].
Selection is per (family, gate_sigma) and is pre-stated (independent of affinity/cluster).

Grid (same as targeted_v3, 80 configs total):
  LSPIN:    5 lambdas × 2 sigmas × 4 smooth = 40
  Concrete: 5 lambdas × 2 sigmas × 4 smooth = 40

Usage:
  conda activate musevo
  cd /banach2/wes/lspin-repos/sparsedeepsurv-paper
  python analyses/ch3_kipan_adaptive_v1.py [--gpus 0 2 6 7]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Make sparsedeepsurv importable from source if not installed
_SDS_SRC = "/banach2/wes/lspin-repos/sparsedeepsurv/src"
if _SDS_SRC not in sys.path:
    sys.path.insert(0, _SDS_SRC)

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
LSPIN_ROOT       = Path("/banach2/wes/lspin-pytorch")
CLEANED          = LSPIN_ROOT / "cleaned_analyses"
PAPER_ROOT       = Path(__file__).resolve().parents[1]
KIPAN_DATA_DEFAULT = LSPIN_ROOT / "runs" / "kipan_20260209_213604"
RESULTS_DEFAULT    = PAPER_ROOT / "data" / "runs" / "ch3_kipan_adaptive_v1"


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(
    worker_id: int,
    configs: List[Dict],
    outdir_str: str,
    results_dir_str: str,
    phase: int,                  # 1 or 2
    rep_start: int,
    rep_end: int,
    seed: int,
    knn_k: int,
    device: str,
    patience: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    lspin_temperature: float,
    concrete_temperature: float,
    risk_top_frac: float,
    cluster_n_clusters: int,
    global_freq_threshold: float,
) -> str:
    import sys as _sys
    _sds_src = "/banach2/wes/lspin-repos/sparsedeepsurv/src"
    if _sds_src not in _sys.path:
        _sys.path.insert(0, _sds_src)

    import numpy as np
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import ShuffleSplit
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import sparsedeepsurv as sds

    data = sds.load_kipan_split_artifacts(Path(outdir_str).resolve())
    X_train = data["X_train"]
    X_test  = data["X_test"]
    strat_key = np.array([
        f"{e}_{h}" for e, h in zip(data["event_train"], data["histo_train"])
    ])
    ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    i_tr, i_val = next(ss.split(X_train, strat_key))

    Xt_tr   = sds.as_torch(X_train[i_tr])
    Xt_val  = sds.as_torch(X_train[i_val])
    Xt_test = sds.as_torch(X_test)
    tt_tr   = sds.as_torch(data["time_train"][i_tr])
    tt_val  = sds.as_torch(data["time_train"][i_val])
    tt_test = sds.as_torch(data["time_test"])
    et_tr   = sds.as_torch(data["event_train"][i_tr])
    et_val  = sds.as_torch(data["event_train"][i_val])
    et_test = sds.as_torch(data["event_test"])

    A_train = sds.build_knn_adjacency_csr(
        Xt_tr, k=knn_k, pca_dim=50, metric="cosine", symmetrize=True
    )

    run_rows:     List[Dict] = []
    aff_rows:     List[Dict] = []
    risk_rows:    List[Dict] = []
    cluster_rows: List[Dict] = []
    hist_rows:    List[pd.DataFrame] = []

    temperature_map = {"lspin_tf": lspin_temperature, "concrete": concrete_temperature}

    n_cfg = len(configs)
    for cfg_i, cfg in enumerate(configs):
        family_label = cfg["family_label"]
        gate_type    = cfg["gate_type"]
        gate_sigma   = float(cfg["gate_sigma"])
        temperature  = temperature_map[gate_type]
        lam          = float(cfg["lam"])
        smooth       = float(cfg["smooth"])
        cfg_seeds    = [
            int(seed + 10000 * cfg["global_cfg_idx"] + k)
            for k in range(rep_start, rep_end)
        ]

        t0 = time.time()
        print(
            f"[worker {worker_id}|p{phase}] [{cfg_i + 1}/{n_cfg}] "
            f"{family_label} σ={gate_sigma} λ={lam:.4g} smooth={smooth:.4g} "
            f"reps={rep_start}..{rep_end - 1}",
            flush=True,
        )

        aff_vecs:     List[np.ndarray] = []
        risk_vecs:    List[np.ndarray] = []
        cluster_vecs: List[np.ndarray] = []

        for run_id, s in enumerate(cfg_seeds):
            sds.set_all_seeds(s)
            try:
                model, info = sds.run_one_model(
                    Xt_tr, tt_tr, et_tr,
                    Xt_val, tt_val, et_val,
                    Xt_test, tt_test, et_test,
                    input_dim=X_train.shape[1],
                    gate_type=gate_type,
                    gate_sigma=gate_sigma,
                    lam=lam,
                    lambda_sample_smooth=smooth,
                    patience=patience,
                    A_sample_train=A_train,
                    device=device,
                    lr=lr,
                    temperature=temperature,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                )
            except Exception as exc:
                print(f"[worker {worker_id}|p{phase}]   seed={s} FAILED: {exc}", flush=True)
                continue

            _, g_det, hard_g, _ = sds.get_gates(
                model, Xt_test, device=device, hard_threshold=0.5, batch_size=512,
            )
            G_det  = g_det.numpy()
            G_hard = hard_g.numpy()

            gpars    = sds.global_parsimony_metrics(G_hard, freq_threshold=global_freq_threshold)
            risk_vec = sds.model_risk_scores(sds.RiskOnlyWrapper(model).to(device), Xt_test, device=device)

            run_rows.append({
                "model_family":          family_label,
                "gate_type":             gate_type,
                "gate_sigma":            gate_sigma,
                "lambda_sparse":         lam,
                "lambda_sample_smooth":  smooth,
                "seed":                  int(s),
                "rep_id":                int(rep_start + run_id),
                "phase":                 phase,
                "test_cindex":           float(info.get("test_cindex", np.nan)),
                "best_val_cindex":       float(info.get("best_val_cindex", np.nan)),
                "best_epoch":            int(info.get("best_epoch", -1)),
                "mean_Khard":            float(G_hard.sum(axis=1).mean()),
                "manifold_alignment":    sds.stable_manifold_alignment(model, Xt_test, device=device),
                **gpars,
            })

            d_hist = sds.cindex_by_histology(
                sds.RiskOnlyWrapper(model).to(device),
                Xt_test, tt_test, et_test, data["histo_test"], device=device,
            )
            if len(d_hist):
                d_hist["model_family"]         = family_label
                d_hist["gate_sigma"]           = gate_sigma
                d_hist["lambda_sparse"]        = lam
                d_hist["lambda_sample_smooth"] = smooth
                d_hist["seed"]                 = int(s)
                hist_rows.append(d_hist)

            aff_vecs.append(sds.affinity_upper_vec(
                sds.gate_affinity_matrix(G_hard, normalize="01", zero_diag=True)
            ))
            risk_vecs.append(risk_vec.astype(np.float32))
            cluster_vecs.append(
                sds.cluster_labels_from_gates(G_det, n_clusters=cluster_n_clusters, seed=s)
            )

        for i in range(len(aff_vecs)):
            for j in range(i + 1, len(aff_vecs)):
                lo_i, hi_i = sds.risk_group_indices(risk_vecs[i], risk_top_frac)
                lo_j, hi_j = sds.risk_group_indices(risk_vecs[j], risk_top_frac)
                hi_stat = sds.topk_overlap_stats(hi_i, hi_j, universe=len(risk_vecs[i]))
                lo_stat = sds.topk_overlap_stats(lo_i, lo_j, universe=len(risk_vecs[i]))
                base = {
                    "model_family": family_label, "gate_type": gate_type,
                    "gate_sigma": gate_sigma,
                    "lambda_sparse": lam, "lambda_sample_smooth": smooth,
                    "run_i": int(rep_start + i), "run_j": int(rep_start + j),
                }
                aff_rows.append({
                    **base,
                    "affinity_corr": sds.affinity_corr_from_vec(aff_vecs[i], aff_vecs[j]),
                })
                risk_rows.append({
                    **base,
                    "risk_corr":                 sds.vector_corr(risk_vecs[i], risk_vecs[j]),
                    "top_risk_overlap_ratio":    float(hi_stat["overlap_ratio"]),
                    "top_risk_jaccard":          float(hi_stat["jaccard"]),
                    "bottom_risk_overlap_ratio": float(lo_stat["overlap_ratio"]),
                    "bottom_risk_jaccard":       float(lo_stat["jaccard"]),
                })
                cluster_rows.append({
                    **base,
                    "cluster_ari": float(adjusted_rand_score(cluster_vecs[i], cluster_vecs[j])),
                    "cluster_nmi": float(normalized_mutual_info_score(cluster_vecs[i], cluster_vecs[j])),
                })

        print(f"[worker {worker_id}|p{phase}]   done in {time.time() - t0:.1f}s", flush=True)

    partial_dir = Path(results_dir_str) / f"_partial_p{phase}_worker{worker_id}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(partial_dir / "run_rows.csv", index=False)
    pd.DataFrame(aff_rows).to_csv(partial_dir / "aff_rows.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(partial_dir / "risk_rows.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(partial_dir / "cluster_rows.csv", index=False)
    if hist_rows:
        pd.concat(hist_rows, ignore_index=True).to_csv(partial_dir / "hist_rows.csv", index=False)
    else:
        pd.DataFrame().to_csv(partial_dir / "hist_rows.csv", index=False)

    print(f"[worker {worker_id}|p{phase}] wrote → {partial_dir}", flush=True)
    return str(partial_dir)


# ── Phase helpers ──────────────────────────────────────────────────────────────

def _dispatch_phase(
    configs: List[Dict],
    phase: int,
    rep_start: int,
    rep_end: int,
    gpus: List[int],
    args,
    results_dir: Path,
) -> List[Path]:
    """Distribute configs across GPUs and return list of partial dirs."""
    buckets: List[List[Dict]] = [[] for _ in gpus]
    for i, cfg in enumerate(configs):
        buckets[i % len(gpus)].append(cfg)

    worker_kwargs = dict(
        outdir_str=str(args.outdir.resolve()),
        results_dir_str=str(results_dir),
        phase=phase,
        rep_start=rep_start,
        rep_end=rep_end,
        seed=args.seed,
        knn_k=args.knn_k,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lspin_temperature=args.lspin_temperature,
        concrete_temperature=args.concrete_temperature,
        risk_top_frac=args.risk_top_frac,
        cluster_n_clusters=args.cluster_n_clusters,
        global_freq_threshold=args.global_freq_threshold,
    )

    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    partial_dirs: List[Path] = []

    with ProcessPoolExecutor(max_workers=len(gpus), mp_context=ctx) as ex:
        futs = {}
        for wid, (gpu, bucket) in enumerate(zip(gpus, buckets)):
            if not bucket:
                continue
            f = ex.submit(
                _worker,
                wid, bucket, **worker_kwargs,
                device=f"cuda:{gpu}",
            )
            futs[f] = (wid, gpu)

        for f in as_completed(futs):
            wid, gpu = futs[f]
            try:
                pdir = f.result()
                partial_dirs.append(Path(pdir))
                print(f"[phase {phase}] worker {wid} (GPU {gpu}) done → {pdir}", flush=True)
            except Exception as e:
                print(f"[phase {phase}] worker {wid} (GPU {gpu}) FAILED: {e}", flush=True)

    return partial_dirs


def _load_partials(partial_dirs: List[Path]) -> Dict[str, pd.DataFrame]:
    """Merge partial CSV files across workers."""
    def _merge(fname):
        frames = []
        for d in partial_dirs:
            p = d / fname
            if p.exists():
                df = pd.read_csv(p)
                if not df.empty:
                    frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return {
        "runs":    _merge("run_rows.csv"),
        "aff":     _merge("aff_rows.csv"),
        "risk":    _merge("risk_rows.csv"),
        "cluster": _merge("cluster_rows.csv"),
        "hist":    _merge("hist_rows.csv"),
    }


# ── Promising config selection ─────────────────────────────────────────────────

def _select_promising(
    df_p1: pd.DataFrame,
    *,
    khard_min: float,
    khard_max: float,
    top_n: int,
) -> List[Dict]:
    """
    For each (family, gate_sigma): select top_n configs by mean C-index
    with mean_Khard in [khard_min, khard_max].
    If fewer than top_n in-band configs exist, take all in-band ones.
    Returns a flat list of config dicts ready for phase 2.
    """
    if df_p1.empty or "model_family" not in df_p1.columns:
        return []

    group_cols = ["model_family", "gate_type", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"]
    acc = (
        df_p1.groupby(group_cols)
        .agg(mean_test_c=("test_cindex", "mean"), mean_Khard=("mean_Khard", "mean"))
        .reset_index()
    )

    selected_rows = []
    for family in acc["model_family"].unique():
        for sigma in sorted(acc.loc[acc["model_family"] == family, "gate_sigma"].unique()):
            sub = acc[
                (acc["model_family"] == family) &
                np.isclose(acc["gate_sigma"], sigma)
            ].copy()
            in_band = sub[
                (sub["mean_Khard"] >= khard_min) & (sub["mean_Khard"] <= khard_max)
            ]
            pool = in_band if not in_band.empty else sub
            top = pool.nlargest(top_n, "mean_test_c")
            selected_rows.append(top)

    if not selected_rows:
        return []

    sel = pd.concat(selected_rows, ignore_index=True)
    return sel[group_cols + ["mean_test_c", "mean_Khard"]].to_dict("records")


def _rebuild_configs_for_phase2(
    all_configs: List[Dict],
    promising_rows: List[Dict],
) -> List[Dict]:
    """
    Filter all_configs to only those matching the (family, sigma, lam, smooth)
    combinations in promising_rows.
    """
    keys = {
        (r["model_family"], float(r["gate_sigma"]), float(r["lambda_sparse"]), float(r["lambda_sample_smooth"]))
        for r in promising_rows
    }
    return [
        c for c in all_configs
        if (c["family_label"], float(c["gate_sigma"]), float(c["lam"]), float(c["smooth"])) in keys
    ]


# ── Showcase config selection (same rule as v3) ────────────────────────────────

def _select_showcase_configs(
    acc: pd.DataFrame,
    *,
    khard_min: float,
    khard_max: float,
) -> Dict:
    selected = {}
    for family in acc["model_family"].unique():
        selected[family] = {}
        fam = acc[acc["model_family"] == family].copy()
        for sigma in sorted(fam["gate_sigma"].unique()):
            sig_df = fam[np.isclose(fam["gate_sigma"], sigma)].copy()
            no_df  = sig_df[np.isclose(sig_df["lambda_sample_smooth"], 0.0)].copy()
            sm_df  = sig_df[sig_df["lambda_sample_smooth"] > 0.0].copy()
            if no_df.empty or sm_df.empty:
                continue
            in_band = no_df[
                (no_df["mean_Khard"] >= khard_min) & (no_df["mean_Khard"] <= khard_max)
            ]
            if not in_band.empty:
                no_row = in_band.loc[in_band["mean_test_c"].idxmax()].to_dict()
            else:
                mid = (khard_min + khard_max) / 2
                no_row = no_df.loc[(no_df["mean_Khard"] - mid).abs().idxmin()].to_dict()
            sm_nearest_idx = (sm_df["mean_Khard"] - no_row["mean_Khard"]).abs().idxmin()
            sm_row = sm_df.loc[sm_nearest_idx].to_dict()
            selected[family][sigma] = {"nosmooth": no_row, "smooth": sm_row}
    return selected


# ── Post-processing ────────────────────────────────────────────────────────────

def _post_process(
    results_dir: Path,
    all_partial_dirs: List[Path],
    args,
) -> Dict:
    import matplotlib
    matplotlib.use("Agg")

    data = _load_partials(all_partial_dirs)
    df_runs    = data["runs"]
    df_aff     = data["aff"]
    df_risk    = data["risk"]
    df_cluster = data["cluster"]
    df_hist    = data["hist"]

    df_runs.to_csv(results_dir / "all_runs_raw.csv",         index=False)
    df_aff.to_csv(results_dir  / "all_affinity_pairs.csv",   index=False)
    df_risk.to_csv(results_dir / "all_risk_pairs.csv",       index=False)
    df_cluster.to_csv(results_dir / "all_cluster_pairs.csv", index=False)
    df_hist.to_csv(results_dir / "all_histology_runs.csv",   index=False)

    group_cols = ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"]

    def _summarize(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby(group_cols)
            .agg(
                mean_test_c=("test_cindex", "mean"),
                std_test_c=("test_cindex", "std"),
                mean_Khard=("mean_Khard", "mean"),
                std_Khard=("mean_Khard", "std"),
                mean_manifold_alignment=("manifold_alignment", "mean"),
                std_manifold_alignment=("manifold_alignment", "std"),
                mean_gene_union_count=("gene_union_count", "mean"),
                mean_gene_freq_ge_threshold_count=("gene_freq_ge_threshold_count", "mean"),
                n_runs=("seed", "count"),
            )
            .reset_index()
        )

    acc = _summarize(df_runs)

    def _summarize_aff(df_a: pd.DataFrame) -> pd.DataFrame:
        merged = df_a.merge(
            df_runs[group_cols].drop_duplicates(),
            on=[c for c in group_cols if c in df_a.columns],
            how="left",
        ) if "gate_sigma" not in df_a.columns else df_a
        return (
            merged.groupby(group_cols)["affinity_corr"]
            .agg(mean_affinity_corr="mean", std_affinity_corr="std")
            .reset_index()
        )

    def _summarize_pair(df_p: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        existing = [c for c in group_cols if c in df_p.columns]
        return df_p.groupby(existing).agg(**{f"mean_{m}": (m, "mean") for m in metrics if m in df_p.columns}).reset_index()

    aff     = _summarize_aff(df_aff)
    risk    = _summarize_pair(df_risk, [
        "risk_corr", "top_risk_overlap_ratio", "top_risk_jaccard",
        "bottom_risk_overlap_ratio", "bottom_risk_jaccard",
    ])
    cluster = _summarize_pair(df_cluster, ["cluster_ari", "cluster_nmi"])

    acc.to_csv(results_dir / "accuracy_summary.csv",   index=False)
    aff.to_csv(results_dir / "affinity_summary.csv",   index=False)
    risk.to_csv(results_dir / "risk_summary.csv",      index=False)
    cluster.to_csv(results_dir / "cluster_summary.csv", index=False)

    # ── Supplement table ──────────────────────────────────────────────────────
    supp = acc.copy()
    for df_extra in [aff, risk, cluster]:
        merge_on = [c for c in group_cols if c in supp.columns and c in df_extra.columns]
        supp = supp.merge(df_extra, on=merge_on, how="left")
    supp = supp.sort_values(group_cols).reset_index(drop=True)
    supp.to_csv(results_dir / "supp_all_configs_metrics.csv", index=False)
    print(f"[post-process] supp table: {len(supp)} configs × {len(supp.columns)} columns")

    # ── LOESS figures ─────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.nonparametric.smoothers_lowess import lowess

    def _loess_fig(sub_acc: pd.DataFrame, *, y_col: str, title: str, outpath: Path) -> None:
        sub = sub_acc[np.isfinite(sub_acc["mean_Khard"]) & np.isfinite(sub_acc[y_col])].copy()
        if sub.empty:
            return
        fig, ax = plt.subplots(figsize=(9.8, 6.0))
        palette = sns.color_palette("tab10", n_colors=max(3, sub["lambda_sample_smooth"].nunique()))
        for idx, sm in enumerate(sorted(sub["lambda_sample_smooth"].unique())):
            part = sub[np.isclose(sub["lambda_sample_smooth"], sm)].sort_values("mean_Khard")
            x = part["mean_Khard"].to_numpy(float)
            y = part[y_col].to_numpy(float)
            color = palette[idx]
            ax.scatter(x, y, s=28, alpha=0.45, color=color, edgecolors="none", zorder=3)
            if len(part) > 1:
                xl = np.log10(np.clip(x, 1e-6, None))
                grid = np.linspace(xl.min(), xl.max(), max(60, len(xl) * 10))
                bw = max((xl.max() - xl.min()) * 0.4, 0.055)
                means = [float(np.sum(np.exp(-0.5 * ((xl - g) / bw) ** 2) * y) /
                               np.clip(np.sum(np.exp(-0.5 * ((xl - g) / bw) ** 2)), 1e-12, None))
                         for g in grid]
                means_sm = lowess(np.array(means), grid, frac=0.4, return_sorted=False)
                ax.plot(10 ** grid, means_sm, color=color, linewidth=2.6,
                        label=f"smooth={sm:g}", zorder=2)
        ax.set_xscale("log")
        ax.set_xlabel("Mean Khard")
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)

    for family in acc["model_family"].unique():
        for sigma in sorted(acc[acc["model_family"] == family]["gate_sigma"].unique()):
            sub = acc[(acc["model_family"] == family) & np.isclose(acc["gate_sigma"], sigma)]
            tag = f"{family.lower()}_sigma{sigma:.2f}".replace(".", "p")
            for y_col in ["mean_test_c", "mean_Khard", "mean_manifold_alignment", "mean_gene_union_count"]:
                _loess_fig(
                    sub, y_col=y_col,
                    title=f"{family} σ={sigma} — {y_col}",
                    outpath=results_dir / f"fig_{tag}_{y_col}_vs_khard.png",
                )

    # ── Apply showcase selection rule ─────────────────────────────────────────
    selected = _select_showcase_configs(acc, khard_min=args.khard_min, khard_max=args.khard_max)
    sel_rows = []
    for family, sigma_dict in selected.items():
        for sigma, pair in sigma_dict.items():
            for which, row in pair.items():
                sel_rows.append({
                    "family": family, "gate_sigma": sigma, "selection": which,
                    **{k: v for k, v in row.items() if not isinstance(v, dict)},
                })
    pd.DataFrame(sel_rows).to_csv(results_dir / "selected_showcase_configs.csv", index=False)

    print(f"\n[post-process] Showcase configs (Khard [{args.khard_min}, {args.khard_max}]):")
    for family, sigma_dict in selected.items():
        for sigma, pair in sigma_dict.items():
            no = pair["nosmooth"]
            sm = pair["smooth"]
            print(f"  {family} σ={sigma}: "
                  f"no-smooth λ={no.get('lambda_sparse', '?'):.5g} "
                  f"Khard={no.get('mean_Khard', 0):.0f} C={no.get('mean_test_c', 0):.4f}  |  "
                  f"smooth λ={sm.get('lambda_sparse', '?'):.5g} "
                  f"smooth={sm.get('lambda_sample_smooth', '?'):.4g} "
                  f"Khard={sm.get('mean_Khard', 0):.0f} C={sm.get('mean_test_c', 0):.4f}")

    print(f"[post-process] done → {results_dir}", flush=True)
    return selected


# ── Config grid ───────────────────────────────────────────────────────────────

def _build_configs(args) -> List[Dict]:
    configs = []
    idx = 0
    for sigma in args.lspin_sigmas:
        for lam in args.lspin_lambdas:
            for smooth in args.sample_smooth_grid:
                configs.append({
                    "family_label": "LSPIN",
                    "gate_type":    "lspin_tf",
                    "gate_sigma":   float(sigma),
                    "lam":          float(lam),
                    "smooth":       float(smooth),
                    "global_cfg_idx": idx,
                })
                idx += 1
    for sigma in args.concrete_sigmas:
        for lam in args.concrete_lambdas:
            for smooth in args.sample_smooth_grid:
                configs.append({
                    "family_label": "Concrete",
                    "gate_type":    "concrete",
                    "gate_sigma":   float(sigma),
                    "lam":          float(lam),
                    "smooth":       float(smooth),
                    "global_cfg_idx": idx,
                })
                idx += 1
    return configs


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="KIPAN adaptive sweep v1: 3-rep phase 1 → select promising → full phase 2"
    )
    p.add_argument("--outdir",      type=Path, default=KIPAN_DATA_DEFAULT)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    p.add_argument("--gpus",        type=int,  nargs="+", default=[0, 2, 6, 7])
    p.add_argument("--seed",        type=int,  default=123)
    p.add_argument("--knn-k",       type=int,  default=8)
    p.add_argument("--n-phase1-reps", type=int, default=3,
                   help="Number of seeds per config in phase 1 (screening)")
    p.add_argument("--n-total-reps",  type=int, default=12,
                   help="Total seeds per promising config (phase 1 + phase 2)")
    p.add_argument("--top-n-promising", type=int, default=8,
                   help="Top N configs per (family, sigma) to carry into phase 2")

    p.add_argument("--patience",     type=int,   default=60)
    p.add_argument("--lr",           type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--max-epochs",   type=int,   default=300)

    # Lambda grids (same as targeted_v3)
    p.add_argument("--lspin-lambdas", type=float, nargs="+",
                   default=[0.0018, 0.0014, 0.0011, 0.0009, 0.0007])
    p.add_argument("--concrete-lambdas", type=float, nargs="+",
                   default=[0.005, 0.004, 0.003, 0.0025, 0.002])

    # Two sigmas per family (same as targeted_v3)
    p.add_argument("--lspin-sigmas",    type=float, nargs="+", default=[0.25, 0.20])
    p.add_argument("--concrete-sigmas", type=float, nargs="+", default=[0.10, 0.15])

    # Temperatures
    p.add_argument("--lspin-temperature",    type=float, default=0.5)
    p.add_argument("--concrete-temperature", type=float, default=0.3)

    # Smooth grid
    p.add_argument("--sample-smooth-grid", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.15])

    # Evaluation
    p.add_argument("--risk-top-frac",          type=float, default=0.2)
    p.add_argument("--cluster-n-clusters",     type=int,   default=3)
    p.add_argument("--global-freq-threshold",  type=float, default=0.05)

    # Showcase selection Khard zone
    p.add_argument("--khard-min", type=float, default=250.0)
    p.add_argument("--khard-max", type=float, default=550.0)

    p.add_argument("--post-process-only", action="store_true",
                   help="Skip training; merge existing partial dirs and post-process")
    return p.parse_args()


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_path):
        self._log = open(log_path, "a", buffering=1)
        self._stdout = sys.stdout
    def write(self, msg):
        self._stdout.write(msg)
        self._log.write(msg)
    def flush(self):
        self._stdout.flush()
        self._log.flush()
    def fileno(self):
        return self._stdout.fileno()


def main():
    args = _parse()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = _Tee(results_dir / "run.log")

    all_configs = _build_configs(args)
    print(f"[main] {len(all_configs)} total configs in grid")
    print(f"[main] GPUs: {args.gpus}")
    print(f"[main] Phase 1: {args.n_phase1_reps} reps/config, "
          f"Phase 2: {args.n_total_reps - args.n_phase1_reps} more reps on top-{args.top_n_promising} per (family,sigma)")
    print(f"[main] Khard showcase band: [{args.khard_min}, {args.khard_max}]")
    print(f"[main] Results → {results_dir}", flush=True)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    p1_dirs = sorted(results_dir.glob("_partial_p1_worker*"))
    if args.post_process_only:
        print(f"[main] --post-process-only: found {len(p1_dirs)} phase-1 partial dirs")
    else:
        if p1_dirs:
            print(f"[main] Found {len(p1_dirs)} existing phase-1 partial dirs — skipping phase 1")
        else:
            print(f"\n[main] === Phase 1: screening {len(all_configs)} configs × {args.n_phase1_reps} reps ===",
                  flush=True)
            t0 = time.time()
            p1_dirs = _dispatch_phase(
                all_configs, phase=1,
                rep_start=0, rep_end=args.n_phase1_reps,
                gpus=args.gpus, args=args, results_dir=results_dir,
            )
            print(f"[main] Phase 1 done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # Load phase 1 results
    p1_data = _load_partials(p1_dirs)
    df_p1 = p1_data["runs"]

    # ── Select promising configs ───────────────────────────────────────────────
    promising_rows = _select_promising(
        df_p1,
        khard_min=args.khard_min,
        khard_max=args.khard_max,
        top_n=args.top_n_promising,
    )
    p2_configs = _rebuild_configs_for_phase2(all_configs, promising_rows)
    print(f"\n[main] Selected {len(p2_configs)} promising configs for phase 2:")
    for r in sorted(promising_rows, key=lambda x: -x["mean_test_c"]):
        print(f"  {r['model_family']} σ={r['gate_sigma']:.2f} "
              f"λ={r['lambda_sparse']:.5g} smooth={r['lambda_sample_smooth']:.4g} "
              f"Khard={r['mean_Khard']:.0f} C={r['mean_test_c']:.4f}")

    pd.DataFrame(promising_rows).to_csv(results_dir / "promising_configs_selected.csv", index=False)

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    n_p2 = args.n_total_reps - args.n_phase1_reps
    p2_dirs = sorted(results_dir.glob("_partial_p2_worker*"))

    if not args.post_process_only:
        if p2_dirs:
            print(f"[main] Found {len(p2_dirs)} existing phase-2 partial dirs — skipping phase 2")
        elif n_p2 > 0 and p2_configs:
            print(f"\n[main] === Phase 2: {len(p2_configs)} configs × {n_p2} more reps ===",
                  flush=True)
            t0 = time.time()
            p2_dirs = _dispatch_phase(
                p2_configs, phase=2,
                rep_start=args.n_phase1_reps, rep_end=args.n_total_reps,
                gpus=args.gpus, args=args, results_dir=results_dir,
            )
            print(f"[main] Phase 2 done in {(time.time() - t0) / 60:.1f} min", flush=True)
        else:
            print("[main] Phase 2 skipped (n_total_reps <= n_phase1_reps or no promising configs)")

    # ── Post-process (combine all phases) ─────────────────────────────────────
    all_partial_dirs = p1_dirs + p2_dirs
    print(f"\n[main] Post-processing {len(all_partial_dirs)} partial dirs total", flush=True)
    _post_process(results_dir, all_partial_dirs, args)


if __name__ == "__main__":
    main()
