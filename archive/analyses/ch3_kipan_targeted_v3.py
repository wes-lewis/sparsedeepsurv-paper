#!/usr/bin/env python3
"""
KIPAN targeted analysis v3 — multi-sigma, wide grid, supp table.

Design:
  - 5 lambda values per family × 2 gate sigmas × 4 smooth values = 80 configs total
  - 8 seeds per config, 4 GPUs in parallel (~2.3h sweep + ~30min showcase)
  - Post-processing generates a supplement table of all configs × all metrics
  - Showcase config selection uses a pre-stated rule: highest mean C-index among
    no-smooth configs with mean Khard in [khard_min, khard_max], per (family, sigma).
    Smooth counterpart = same family+sigma, nearest Khard, lambda_sample_smooth > 0.
    No affinity or cluster metrics used in selection.
  - Showcase re-run (--run-showcase) calls the original targeted script on just
    the 4 selected configs to generate heatmaps and bridge comparison figures.

Usage:
  conda activate musevo
  cd /banach2/wes/lspin-repos/sparsedeepsurv-paper
  python analyses/ch3_kipan_targeted_v3.py [--gpus 0 2 6 7] [--run-showcase]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
LSPIN_ROOT  = Path("/banach2/wes/lspin-pytorch")
CLEANED     = LSPIN_ROOT / "cleaned_analyses"
PAPER_ROOT  = Path(__file__).resolve().parents[1]
KIPAN_DATA_DEFAULT = LSPIN_ROOT / "runs" / "kipan_20260209_213604"
RESULTS_DEFAULT    = PAPER_ROOT / "data" / "runs" / "ch3_kipan_targeted_v3"

for _p in [str(LSPIN_ROOT), str(CLEANED)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(
    worker_id: int,
    configs: List[Dict],
    outdir_str: str,
    results_dir_str: str,
    seed: int,
    knn_k: int,
    device: str,
    n_reps: int,
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
        gate_sigma   = float(cfg["gate_sigma"])   # per-config sigma
        temperature  = temperature_map[gate_type]
        lam          = float(cfg["lam"])
        smooth       = float(cfg["smooth"])
        cfg_seeds    = [int(seed + 10000 * cfg["global_cfg_idx"] + k) for k in range(n_reps)]

        t0 = time.time()
        print(
            f"[worker {worker_id}] [{cfg_i + 1}/{n_cfg}] "
            f"{family_label} σ={gate_sigma} λ={lam:.4g} smooth={smooth:.4g}",
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
                print(f"[worker {worker_id}]   seed={s} FAILED: {exc}", flush=True)
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
                "run_id":                int(run_id),
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
                    "run_i": i, "run_j": j,
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

        print(f"[worker {worker_id}]   done in {time.time() - t0:.1f}s", flush=True)

    partial_dir = Path(results_dir_str) / f"_partial_worker{worker_id}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(run_rows).to_csv(partial_dir / "run_rows.csv", index=False)
    pd.DataFrame(aff_rows).to_csv(partial_dir / "aff_rows.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(partial_dir / "risk_rows.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(partial_dir / "cluster_rows.csv", index=False)
    if hist_rows:
        pd.concat(hist_rows, ignore_index=True).to_csv(partial_dir / "hist_rows.csv", index=False)
    else:
        pd.DataFrame().to_csv(partial_dir / "hist_rows.csv", index=False)

    print(f"[worker {worker_id}] wrote partial results → {partial_dir}", flush=True)
    return str(partial_dir)


# ── Config selection (stated rule, no circularity) ─────────────────────────────

def _select_showcase_configs(
    acc: pd.DataFrame,
    *,
    khard_min: float,
    khard_max: float,
) -> Dict:
    """
    Selection rule (pre-stated, independent of affinity/cluster/manifold metrics):

    For each (family, gate_sigma):
      1. No-smooth config: highest mean test C-index among configs with
         lambda_sample_smooth == 0 and mean_Khard in [khard_min, khard_max].
      2. Smooth config: same family+sigma, lambda_sample_smooth > 0,
         nearest mean_Khard to the selected no-smooth config.

    If no no-smooth config falls in the Khard range, relax to the full range
    and pick the config whose mean_Khard is closest to the midpoint.

    Returns dict: family → {sigma → {"nosmooth": row_dict, "smooth": row_dict}}
    """
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

            # Step 1: best no-smooth config in Khard range
            in_band = no_df[
                (no_df["mean_Khard"] >= khard_min) & (no_df["mean_Khard"] <= khard_max)
            ]
            if not in_band.empty:
                no_row = in_band.loc[in_band["mean_test_c"].idxmax()].to_dict()
            else:
                # Fallback: closest to midpoint
                mid = (khard_min + khard_max) / 2
                no_row = no_df.loc[(no_df["mean_Khard"] - mid).abs().idxmin()].to_dict()

            # Step 2: nearest-Khard smooth config (same family+sigma)
            sm_nearest_idx = (sm_df["mean_Khard"] - no_row["mean_Khard"]).abs().idxmin()
            sm_row = sm_df.loc[sm_nearest_idx].to_dict()

            selected[family][sigma] = {"nosmooth": no_row, "smooth": sm_row}

    return selected


# ── Post-processing ────────────────────────────────────────────────────────────

def _post_process(results_dir: Path, args) -> Dict:
    """
    Merge partial CSVs → supp table → apply selection rule.
    Returns the selected showcase configs dict.
    """
    import matplotlib
    matplotlib.use("Agg")

    partial_dirs = sorted(results_dir.glob("_partial_worker*"))
    print(f"[post-process] merging {len(partial_dirs)} partial result dirs")

    df_runs    = pd.concat([pd.read_csv(d / "run_rows.csv")     for d in partial_dirs], ignore_index=True)
    df_aff     = pd.concat([pd.read_csv(d / "aff_rows.csv")     for d in partial_dirs], ignore_index=True)
    df_risk    = pd.concat([pd.read_csv(d / "risk_rows.csv")    for d in partial_dirs], ignore_index=True)
    df_cluster = pd.concat([pd.read_csv(d / "cluster_rows.csv") for d in partial_dirs], ignore_index=True)
    hist_list  = [pd.read_csv(d / "hist_rows.csv") for d in partial_dirs]
    df_hist    = pd.concat([f for f in hist_list if not f.empty], ignore_index=True) if hist_list else pd.DataFrame()

    df_runs.to_csv(results_dir / "all_runs_raw.csv",             index=False)
    df_aff.to_csv(results_dir  / "all_affinity_pairs.csv",       index=False)
    df_risk.to_csv(results_dir / "all_risk_pairs.csv",           index=False)
    df_cluster.to_csv(results_dir / "all_cluster_pairs.csv",     index=False)
    df_hist.to_csv(results_dir / "all_histology_runs.csv",       index=False)

    # Summarize — gate_sigma is a new grouping dimension; the broad_v3 summarize_runs
    # groups by (model_family, lambda_sample_smooth, lambda_sparse). We add gate_sigma
    # by computing summaries manually here so the supp table is correct.
    def _summarize_with_sigma(df: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"]
        agg = (
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
        return agg

    acc = _summarize_with_sigma(df_runs)

    # Affinity summary with sigma grouping
    def _summarize_aff_with_sigma(df_aff: pd.DataFrame, df_runs: pd.DataFrame) -> pd.DataFrame:
        merged = df_aff.merge(
            df_runs[["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"]].drop_duplicates(),
            on=["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"],
            how="left",
        ) if "gate_sigma" in df_aff.columns else df_aff
        group_cols = ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"]
        return (
            merged.groupby(group_cols)["affinity_corr"]
            .agg(mean_affinity_corr="mean", std_affinity_corr="std")
            .reset_index()
        )

    def _summarize_pair_with_sigma(df_pair: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        group_cols = ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"]
        existing = [c for c in group_cols if c in df_pair.columns]
        agg_dict = {m: "mean" for m in metrics if m in df_pair.columns}
        return df_pair.groupby(existing).agg(**{f"mean_{m}": (m, "mean") for m in agg_dict}).reset_index()

    aff     = _summarize_aff_with_sigma(df_aff, df_runs)
    risk    = _summarize_pair_with_sigma(df_risk, [
        "risk_corr", "top_risk_overlap_ratio", "top_risk_jaccard",
        "bottom_risk_overlap_ratio", "bottom_risk_jaccard",
    ])
    cluster = _summarize_pair_with_sigma(df_cluster, ["cluster_ari", "cluster_nmi"])

    acc.to_csv(results_dir / "accuracy_summary.csv",  index=False)
    aff.to_csv(results_dir / "affinity_summary.csv",  index=False)
    risk.to_csv(results_dir / "risk_summary.csv",     index=False)
    cluster.to_csv(results_dir / "cluster_summary.csv", index=False)

    # ── Supplement table: all configs × all scalar metrics ────────────────────
    supp = acc.copy()
    for df_extra, merge_cols_extra, col_prefix in [
        (aff,     ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"], ""),
        (risk,    ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"], ""),
        (cluster, ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"], ""),
    ]:
        merge_on = [c for c in merge_cols_extra if c in supp.columns and c in df_extra.columns]
        supp = supp.merge(df_extra, on=merge_on, how="left")

    supp = supp.sort_values(
        ["model_family", "gate_sigma", "lambda_sparse", "lambda_sample_smooth"]
    ).reset_index(drop=True)
    supp.to_csv(results_dir / "supp_all_configs_metrics.csv", index=False)
    print(f"[post-process] supp table: {len(supp)} configs × {len(supp.columns)} columns")

    # ── LOESS figures (per family × sigma combination) ────────────────────────
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.nonparametric.smoothers_lowess import lowess

    def _loess_fig(sub_acc: pd.DataFrame, *, y_col: str, title: str, outpath: Path, frac: float) -> None:
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
                bw = max((xl.max() - xl.min()) * max(frac, 0.18) * 0.38, 0.055)
                means = [float(np.sum(np.exp(-0.5 * ((xl - g) / bw) ** 2) * y) /
                               np.clip(np.sum(np.exp(-0.5 * ((xl - g) / bw) ** 2)), 1e-12, None))
                         for g in grid]
                means_sm = lowess(np.array(means), grid, frac=max(frac * 0.75, 0.2), return_sorted=False)
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
            sigma_tag = f"sigma{sigma:.2f}".replace(".", "p")
            for y_col in ["mean_test_c", "mean_Khard", "mean_manifold_alignment", "mean_gene_union_count"]:
                _loess_fig(
                    sub, y_col=y_col,
                    title=f"{family} σ={sigma} — {y_col}",
                    outpath=results_dir / f"fig_{family.lower()}_{sigma_tag}_{y_col}_vs_khard.png",
                    frac=0.55,
                )

    # ── Apply selection rule ───────────────────────────────────────────────────
    selected = _select_showcase_configs(
        acc,
        khard_min=float(args.khard_min),
        khard_max=float(args.khard_max),
    )

    # Write selection summary
    sel_rows = []
    for family, sigma_dict in selected.items():
        for sigma, pair in sigma_dict.items():
            for which, row in pair.items():
                sel_rows.append({
                    "family": family, "gate_sigma": sigma, "selection": which,
                    **{k: v for k, v in row.items() if not isinstance(v, dict)},
                })
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(results_dir / "selected_showcase_configs.csv", index=False)

    print(f"\n[post-process] Selected showcase configs (Khard range [{args.khard_min}, {args.khard_max}]):")
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


# ── Showcase re-run (targeted script on 4 selected configs) ───────────────────

def _run_showcase(selected: Dict, results_dir: Path, args) -> None:
    """
    Re-run the original targeted script on just the selected configs.
    This generates heatmaps and bridge comparison figures for the paper.
    The sigma is passed directly; if two families selected different sigmas,
    each gets its own sub-run.
    """
    showcase_dir = results_dir / "showcase"
    showcase_dir.mkdir(parents=True, exist_ok=True)

    # Collect selected (family, sigma) combinations
    # Pick the sigma with the best no-smooth C-index per family
    best_per_family: Dict[str, Dict] = {}
    for family, sigma_dict in selected.items():
        best_sigma = None
        best_c = -1.0
        for sigma, pair in sigma_dict.items():
            c = float(pair["nosmooth"].get("mean_test_c", 0.0))
            if c > best_c:
                best_c = c
                best_sigma = sigma
        if best_sigma is not None:
            best_per_family[family] = {
                "sigma": best_sigma,
                **selected[family][best_sigma],
            }

    if not best_per_family:
        print("[showcase] No configs selected — skipping showcase re-run.", flush=True)
        return

    lspin_no_lam  = best_per_family["LSPIN"]["nosmooth"].get("lambda_sparse")
    lspin_sm_lam  = best_per_family["LSPIN"]["smooth"].get("lambda_sparse")
    lspin_sm_val  = best_per_family["LSPIN"]["smooth"].get("lambda_sample_smooth")
    lspin_sigma   = best_per_family["LSPIN"]["sigma"]

    conc_no_lam   = best_per_family["Concrete"]["nosmooth"].get("lambda_sparse")
    conc_sm_lam   = best_per_family["Concrete"]["smooth"].get("lambda_sparse")
    conc_sm_val   = best_per_family["Concrete"]["smooth"].get("lambda_sample_smooth")
    conc_sigma    = best_per_family["Concrete"]["sigma"]

    print(f"\n[showcase] Re-running targeted script:", flush=True)
    print(f"  LSPIN σ={lspin_sigma}: λ_no={lspin_no_lam:.5g} λ_sm={lspin_sm_lam:.5g} smooth={lspin_sm_val:g}")
    print(f"  Concrete σ={conc_sigma}: λ_no={conc_no_lam:.5g} λ_sm={conc_sm_lam:.5g} smooth={conc_sm_val:g}")
    print(f"  Output → {showcase_dir}", flush=True)

    import subprocess
    cmd = [
        sys.executable,
        str(CLEANED / "run_kipan_patient_smoothing_notebook_style_paper.py"),
        "--outdir",     str(args.outdir.resolve()),
        "--results-dir", str(showcase_dir),
        "--device",     "cuda",
        "--knn-k",      str(args.knn_k),
        "--n-reps",     str(args.n_reps),
        "--seed",       str(args.seed),
        "--lspin-lambdas",    str(lspin_no_lam), str(lspin_sm_lam),
        "--concrete-lambdas", str(conc_no_lam),  str(conc_sm_lam),
        "--lspin-gate-sigma",    str(lspin_sigma),
        "--concrete-gate-sigma", str(conc_sigma),
        "--sample-smooth-grid", "0.0", str(lspin_sm_val),
    ]

    # Use CUDA_VISIBLE_DEVICES to pin to the first GPU in the list
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"[showcase] done in {elapsed / 60:.1f} min → {showcase_dir}", flush=True)
    else:
        print(f"[showcase] FAILED (returncode={result.returncode}) in {elapsed / 60:.1f} min", flush=True)


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(
        description="KIPAN targeted v3: wide multi-sigma sweep + supp table + showcase selection"
    )
    p.add_argument("--outdir",      type=Path, default=KIPAN_DATA_DEFAULT)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    p.add_argument("--gpus",        type=int,  nargs="+", default=[0, 2, 6, 7])
    p.add_argument("--seed",        type=int,  default=123)
    p.add_argument("--knn-k",       type=int,  default=8)
    p.add_argument("--n-reps",      type=int,  default=8)
    p.add_argument("--patience",    type=int,  default=60)
    p.add_argument("--lr",          type=float, default=1e-2)
    p.add_argument("--weight-decay",type=float, default=1e-5)
    p.add_argument("--batch-size",  type=int,  default=128)
    p.add_argument("--max-epochs",  type=int,  default=300)

    # Lambda grids — target zone Khard ~200–650 for LSPIN, ~120–500 for Concrete
    p.add_argument("--lspin-lambdas", type=float, nargs="+",
                   default=[0.0018, 0.0014, 0.0011, 0.0009, 0.0007])
    p.add_argument("--concrete-lambdas", type=float, nargs="+",
                   default=[0.005, 0.004, 0.003, 0.0025, 0.002])

    # Two sigmas per family
    p.add_argument("--lspin-sigmas",    type=float, nargs="+", default=[0.25, 0.20])
    p.add_argument("--concrete-sigmas", type=float, nargs="+", default=[0.10, 0.15])

    # Temperature (single value per family, not swept)
    p.add_argument("--lspin-temperature",    type=float, default=0.5)
    p.add_argument("--concrete-temperature", type=float, default=0.3)

    # Smooth grid — 4 values for dose-response curve
    p.add_argument("--sample-smooth-grid", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.15])

    # Evaluation params
    p.add_argument("--risk-top-frac",         type=float, default=0.2)
    p.add_argument("--cluster-n-clusters",    type=int,   default=3)
    p.add_argument("--global-freq-threshold", type=float, default=0.05)

    # Selection rule Khard zone
    p.add_argument("--khard-min", type=float, default=250.0)
    p.add_argument("--khard-max", type=float, default=550.0)

    # Run showcase re-run after selection
    p.add_argument("--run-showcase", action="store_true",
                   help="After sweep, re-run selected configs through the full targeted pipeline")

    # Post-process only (skip training, merge existing partial dirs)
    p.add_argument("--post-process-only", action="store_true")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args = _parse()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if not args.post_process_only:
        # Build config list: family × lambda × sigma × smooth
        # global_cfg_idx is stable and used for seed derivation
        families_spec = [
            ("LSPIN",    "lspin_tf",  args.lspin_lambdas,    args.lspin_sigmas),
            ("Concrete", "concrete",  args.concrete_lambdas,  args.concrete_sigmas),
        ]
        all_configs: List[Dict] = []
        cfg_idx = 0
        for family_label, gate_type, lambdas, sigmas in families_spec:
            for sigma in sigmas:
                for smooth in args.sample_smooth_grid:
                    for lam in lambdas:
                        cfg_idx += 1
                        all_configs.append({
                            "family_label":   family_label,
                            "gate_type":      gate_type,
                            "gate_sigma":     sigma,
                            "lam":            lam,
                            "smooth":         smooth,
                            "global_cfg_idx": cfg_idx,
                        })

        gpus     = args.gpus
        n_workers = len(gpus)
        batches: List[List[Dict]] = [[] for _ in range(n_workers)]
        for i, cfg in enumerate(all_configs):
            batches[i % n_workers].append(cfg)

        n_lspin    = sum(1 for c in all_configs if c["family_label"] == "LSPIN")
        n_concrete = sum(1 for c in all_configs if c["family_label"] == "Concrete")
        print(f"[ch3_kipan_targeted_v3] {len(all_configs)} configs total "
              f"({n_lspin} LSPIN, {n_concrete} Concrete) → {n_workers} workers on GPUs {gpus}")
        print(f"  LSPIN lambdas: {args.lspin_lambdas}")
        print(f"  LSPIN sigmas:  {args.lspin_sigmas}")
        print(f"  Concrete lambdas: {args.concrete_lambdas}")
        print(f"  Concrete sigmas:  {args.concrete_sigmas}")
        print(f"  Smooth grid: {args.sample_smooth_grid}")
        print(f"  n_reps={args.n_reps}  knn_k={args.knn_k}  khard zone=[{args.khard_min}, {args.khard_max}]")
        for i, (gpu, batch) in enumerate(zip(gpus, batches)):
            print(f"  worker {i}: GPU cuda:{gpu}, {len(batch)} configs")

        worker_kwargs = dict(
            outdir_str=str(args.outdir.resolve()),
            results_dir_str=str(args.results_dir.resolve()),
            seed=args.seed,
            knn_k=args.knn_k,
            n_reps=args.n_reps,
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

        t_start = time.time()
        futures = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for worker_id, (gpu, batch) in enumerate(zip(gpus, batches)):
                fut = pool.submit(
                    _worker,
                    worker_id,
                    batch,
                    device=f"cuda:{gpu}",
                    **worker_kwargs,
                )
                futures[fut] = worker_id

            for fut in as_completed(futures):
                wid = futures[fut]
                try:
                    partial_path = fut.result()
                    print(f"[main] worker {wid} finished → {partial_path}", flush=True)
                except Exception as exc:
                    print(f"[main] worker {wid} FAILED: {exc}", flush=True)
                    raise

        print(f"[main] all workers done in {(time.time() - t_start) / 60:.1f} min")

    selected = _post_process(args.results_dir, args)

    if args.run_showcase:
        _run_showcase(selected, args.results_dir, args)
