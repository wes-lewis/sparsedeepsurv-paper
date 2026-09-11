#!/usr/bin/env python3
"""Biological and prognostic validation of gate-derived patient subgroups.

Post-dissertation extension thread (see `POST_DISSERTATION_PLAN.md`,
thread 3).

Existing work already goes partway here:

- `render_adaptive_manuscript_figures.py` clusters patients by hard-gate
  vector (average-linkage, cosine distance) for the heatmap figures.
- `histology_runs_test.py` shows that clustering-induced patient order
  recovers histological structure better than chance (KIPAN), via a
  permutation test on run counts.
- `quick_linear_gated_probe.py` / `plot_pancan_gated_univariate_bias.py`
  look at per-gene signal, but not at subgroup-level biological coherence.

This script adds the two results that turn "gating finds real patient
subgroups" into a positive, freestanding claim rather than an inference
from histology-order alignment:

1. **Survival separation beyond histology.** A log-rank test across
   gate-cluster subgroups, and a Cox model with subgroup indicators
   alongside histology dummies (a likelihood-ratio test of whether adding
   subgroup improves on histology alone). Note: there is no tumor-stage
   field in the processed split artifacts (only `histo_{train,test}`,
   which for KIPAN is itself the pooled cancer subtype label), so
   histology is the covariate this adjusts for -- reported as such rather
   than implying stage adjustment that isn't available.
2. **Subtype-appropriate pathway enrichment.** A hypergeometric enrichment
   test of each subgroup's consensus hard-selected genes against
   `data/gene_sets/enrichr/MSigDB_Hallmark_2020.gmt`, with the
   dataset's own measured gene panel (not the whole genome) as the
   background, and Benjamini-Hochberg correction across pathways within
   each subgroup.

This trains one model per (dataset, family, selection) on a held-out test
split (no k-fold sweep needed here -- the point is one clean clustering +
enrichment pass, matching the granularity of the existing heatmap
figures), then reanalyzes it. No dependency on the k-fold retraining used
by the other pd_ scripts.

Usage:
    python pd_subgroup_biological_validation.py --dataset kipan --gate-family LSPIN --device cuda:2
    python pd_subgroup_biological_validation.py --dataset brca --gate-family LSPIN --device cuda:5
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

from _paths import CANONICAL_RUNS, GENE_SETS_DIR, PROCESSED_DATASETS, ensure_repo_imports

ensure_repo_imports(include_analyses_dir=True)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sparsedeepsurv")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sparsedeepsurv-cache")

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import hypergeom
from sklearn.model_selection import ShuffleSplit

import sparsedeepsurv as sds
from quick_linear_gated_probe import _combined_arrays, _load_data, _selected_configs

HALLMARK_GENE_SETS = GENE_SETS_DIR / "enrichr" / "MSigDB_Hallmark_2020.gmt"

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
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument("--test-fraction", type=float, default=0.30)
    p.add_argument("--gate-rate-threshold", type=float, default=0.10)
    p.add_argument("--min-subgroup-n", type=int, default=15)
    p.add_argument("--top-pathways", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260407)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--hard-threshold", type=float, default=0.5)
    return p.parse_args()


def _load_hallmark_gene_sets(path: Path) -> dict[str, set[str]]:
    sets = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if genes:
                sets[name] = genes
    return sets


def _hypergeometric_enrichment(
    selected_genes: set[str], background: set[str], gene_sets: dict[str, set[str]]
) -> pd.DataFrame:
    N = len(background)
    n = len(selected_genes & background)
    rows = []
    for pathway, genes in gene_sets.items():
        K_genes = genes & background
        K = len(K_genes)
        if K == 0 or n == 0:
            continue
        k = len(selected_genes & K_genes)
        if k == 0:
            continue
        # P(X >= k) for X ~ Hypergeom(N, K, n)
        pval = float(hypergeom.sf(k - 1, N, K, n))
        rows.append({
            "pathway": pathway, "n_selected_in_pathway": k, "pathway_size_in_background": K,
            "n_selected_total": n, "background_size": N, "pvalue": pval,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("pvalue").reset_index(drop=True)
    df["pvalue_bh"] = _benjamini_hochberg(df["pvalue"].to_numpy())
    return df


def _benjamini_hochberg(pvals_sorted_ascending: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values for an array already sorted ascending."""
    m = len(pvals_sorted_ascending)
    ranks = np.arange(1, m + 1)
    adj = pvals_sorted_ascending * m / ranks
    # Enforce monotonicity from the largest p-value down.
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    return np.clip(adj, 0.0, 1.0)


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (
        results_dir / f"pd_subgroup_biological_validation_{args.dataset}_{args.gate_family.lower()}_{args.selection}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X, time, event, histo, genes = _combined_arrays(data)
    background = {g.upper() for g in genes}

    configs = _selected_configs(
        args.dataset, results_dir, [args.gate_family], [args.selection],
        lambda_scale_families=[args.gate_family], lambda_scale_selections=[args.selection],
        lambda_scales=[1.0], smooth_smooth_values=None,
    )
    cfg = configs.iloc[0]

    ss = ShuffleSplit(n_splits=1, test_size=float(args.test_fraction), random_state=int(args.seed))
    train_all, test_idx = next(ss.split(X))
    ss2 = ShuffleSplit(n_splits=1, test_size=0.15, random_state=int(args.seed) + 1)
    rel_train, rel_val = next(ss2.split(train_all))
    train_idx, val_idx = train_all[rel_train], train_all[rel_val]

    Xt_tr = sds.as_torch(X[train_idx])
    A = None
    if float(cfg["lambda_sample_smooth"]) > 0:
        print("[knn] building adjacency", flush=True)
        A = sds.build_knn_adjacency_csr(Xt_tr, k=int(args.knn_k), pca_dim=50, metric="cosine", symmetrize=True)

    print("[train] fitting model on train split", flush=True)
    model, info = sds.run_one_model(
        Xt_tr=Xt_tr, tt_tr=sds.as_torch(time[train_idx]), et_tr=sds.as_torch(event[train_idx]),
        Xt_val=sds.as_torch(X[val_idx]), tt_val=sds.as_torch(time[val_idx]), et_val=sds.as_torch(event[val_idx]),
        Xt_test=sds.as_torch(X[test_idx]), tt_test=sds.as_torch(time[test_idx]), et_test=sds.as_torch(event[test_idx]),
        input_dim=X.shape[1], A_sample_train=A, device=args.device,
        lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs), seed=int(args.seed),
        gate_type=str(cfg["gate_type"]), gate_sigma=float(cfg["gate_sigma"]),
        lam=float(cfg["lambda_sparse"]), lambda_sample_smooth=float(cfg["lambda_sample_smooth"]),
        patience=int(args.patience), temperature=float(cfg["temperature"]),
        concrete_mode=str(cfg["concrete_mode"]), predictor=str(cfg["predictor"]),
        gating_hidden_dim=int(cfg["gating_hidden_dim"]), gate_hidden_dropout_p=float(cfg["gate_hidden_dropout_p"]),
        risk_hidden_dims=tuple(cfg["risk_hidden_dims"]), risk_dropout_p=float(cfg["risk_dropout_p"]),
        lspin_init_bias=float(cfg["lspin_init_bias"]), gate_weight_decay=float(cfg["gate_weight_decay"]),
    )
    print(f"[trained] test_c={info.get('test_c')}", flush=True)

    _, _, hard_t, _ = sds.get_gates(
        model, sds.as_torch(X[test_idx]), device=args.device, hard_threshold=float(args.hard_threshold), batch_size=512,
    )
    hard = hard_t.numpy().astype(np.float32)

    # Cluster test-set patients by hard-gate vector, matching the heatmap
    # figures' metric/method (cosine distance, average linkage), cut to a
    # fixed number of flat clusters.
    dist = pdist(hard, metric="cosine")
    dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="average")
    subgroup = fcluster(Z, t=int(args.n_clusters), criterion="maxclust")

    surv_df = pd.DataFrame({
        "time": time[test_idx], "event": event[test_idx].astype(int),
        "subgroup": subgroup.astype(str), "histo": histo[test_idx],
    })
    surv_df.to_csv(outdir / "subgroup_survival_table.csv", index=False)

    # 1a. Unadjusted log-rank across subgroups.
    subgroup_counts = surv_df["subgroup"].value_counts()
    keep_groups = subgroup_counts[subgroup_counts >= args.min_subgroup_n].index
    lr_df = surv_df[surv_df["subgroup"].isin(keep_groups)]
    logrank_result = {"n_groups": int(lr_df["subgroup"].nunique()), "n_patients": int(len(lr_df))}
    if lr_df["subgroup"].nunique() >= 2:
        lr = multivariate_logrank_test(lr_df["time"], lr_df["subgroup"], lr_df["event"])
        logrank_result["logrank_p"] = float(lr.p_value)
        logrank_result["logrank_stat"] = float(lr.test_statistic)
    else:
        logrank_result["logrank_p"] = float("nan")
        logrank_result["logrank_stat"] = float("nan")

    # 1b. Cox likelihood-ratio test: histo-only vs. histo + subgroup.
    cox_result = {}
    try:
        design = pd.get_dummies(surv_df[["histo", "subgroup"]], drop_first=True)
        histo_only = pd.get_dummies(surv_df[["histo"]], drop_first=True)
        full = pd.concat([surv_df[["time", "event"]], design], axis=1)
        reduced = pd.concat([surv_df[["time", "event"]], histo_only], axis=1)
        # Drop degenerate columns (constant, or perfectly collinear) that
        # would make the Cox fit singular.
        full = full.loc[:, full.nunique(dropna=False) > 1]
        reduced = reduced.loc[:, reduced.nunique(dropna=False) > 1]
        cph_full = CoxPHFitter(penalizer=0.01).fit(full, duration_col="time", event_col="event")
        cph_reduced = CoxPHFitter(penalizer=0.01).fit(reduced, duration_col="time", event_col="event")
        ll_full = cph_full.log_likelihood_
        ll_reduced = cph_reduced.log_likelihood_
        df_diff = full.shape[1] - reduced.shape[1]
        from scipy.stats import chi2
        lr_stat = 2 * (ll_full - ll_reduced)
        cox_result = {
            "cox_lr_stat": float(lr_stat), "cox_lr_df": int(df_diff),
            "cox_lr_p": float(chi2.sf(lr_stat, max(df_diff, 1))) if df_diff > 0 else float("nan"),
            "cox_full_loglik": float(ll_full), "cox_reduced_loglik": float(ll_reduced),
        }
    except Exception as exc:
        cox_result = {"error": str(exc)}

    survival_summary = {**logrank_result, **cox_result}
    pd.DataFrame([survival_summary]).to_csv(outdir / "subgroup_survival_summary.csv", index=False)
    print("[survival]", survival_summary, flush=True)

    # 2. Pathway enrichment per subgroup.
    gene_sets = _load_hallmark_gene_sets(HALLMARK_GENE_SETS)
    enrichment_frames = []
    for g in sorted(set(subgroup)):
        mask = subgroup == g
        if mask.sum() < args.min_subgroup_n:
            continue
        rate = hard[mask].mean(axis=0)
        selected_genes = {genes[i].upper() for i in np.where(rate >= args.gate_rate_threshold)[0]}
        if not selected_genes:
            continue
        enr = _hypergeometric_enrichment(selected_genes, background, gene_sets)
        if enr.empty:
            continue
        enr.insert(0, "subgroup", int(g))
        enr.insert(1, "n_subgroup_patients", int(mask.sum()))
        enr.insert(2, "n_selected_genes", len(selected_genes))
        enrichment_frames.append(enr.head(int(args.top_pathways)))

    enrichment = pd.concat(enrichment_frames, ignore_index=True) if enrichment_frames else pd.DataFrame()
    enrichment.to_csv(outdir / "subgroup_pathway_enrichment_top.csv", index=False)
    print(enrichment.to_string() if not enrichment.empty else "[enrichment] no pathways passed filtering", flush=True)
    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
