#!/usr/bin/env python3
"""
Chapter 3 KIPAN broad sweep — paper-repo version, multi-GPU.

Changes vs original (run_kipan_patient_smoothing_broad_v3.py):
  - knn_k default = 8
  - Multi-GPU: config grid split across GPUs 0, 2, 6, 7 using multiprocessing
    (no circular selection issue in the broad script — rank_top_configs uses
    only C-index and khard_over_union, not affinity metrics)
  - Results written to sparsedeepsurv-paper/data/runs/

Usage:
  conda activate musevo
  cd /banach2/wes/lspin-repos/sparsedeepsurv-paper
  python analyses/ch3_kipan_broad.py [--gpus 0 2 6 7]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
LSPIN_ROOT = Path("/banach2/wes/lspin-pytorch")
CLEANED = LSPIN_ROOT / "cleaned_analyses"
PAPER_ROOT = Path(__file__).resolve().parents[1]
KIPAN_DATA_DEFAULT = PAPER_ROOT / "data" / "processed" / "kipan_20260209_213604"
RESULTS_DEFAULT = PAPER_ROOT / "data" / "runs" / "ch3_kipan_broad_knn8"

_SDS_SRC = "/banach2/wes/lspin-repos/sparsedeepsurv/src"
for _p in [str(LSPIN_ROOT), str(CLEANED), _SDS_SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Worker (runs in a subprocess — all heavy imports are local) ────────────────

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
    lspin_gate_sigma: float,
    lspin_temperature: float,
    concrete_gate_sigma: float,
    concrete_temperature: float,
    concrete_mode: str,
    risk_top_frac: float,
    cluster_n_clusters: int,
    global_freq_threshold: float,
) -> str:
    """
    Train all configs in `configs` on `device`.
    Returns path to a partial results CSV (raw rows) written to a temp dir.
    """
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

    # Per-worker log file (worker stdout goes here via Tee)
    _partial_dir = Path(results_dir_str) / f"_partial_worker{worker_id}"
    _partial_dir.mkdir(parents=True, exist_ok=True)
    _log = open(_partial_dir / "worker.log", "w", buffering=1)
    class _WorkerTee:
        def __init__(self, orig, log):
            self._orig, self._log = orig, log
        def write(self, msg):
            self._orig.write(msg); self._log.write(msg)
        def flush(self):
            self._orig.flush(); self._log.flush()
        def fileno(self):
            return self._orig.fileno()
    _sys.stdout = _WorkerTee(_sys.stdout, _log)

    # Load data fresh in this worker (avoids cross-process CUDA init issues)
    data = sds.load_kipan_split_artifacts(Path(outdir_str).resolve())

    X_train = data["X_train"]
    X_test  = data["X_test"]
    strat_key = np.array([f"{e}_{h}" for e, h in zip(data["event_train"], data["histo_train"])])
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

    A_train = sds.build_knn_adjacency_csr(Xt_tr, k=knn_k, pca_dim=50, metric="cosine", symmetrize=True)

    run_rows:     List[Dict] = []
    aff_rows:     List[Dict] = []
    risk_rows:    List[Dict] = []
    cluster_rows: List[Dict] = []
    hist_rows:    List[pd.DataFrame] = []

    n_cfg = len(configs)
    for cfg_i, cfg in enumerate(configs):
        family_label = cfg["family_label"]
        gate_type    = cfg["gate_type"]
        gate_sigma   = lspin_gate_sigma if family_label == "LSPIN" else concrete_gate_sigma
        temperature  = lspin_temperature if family_label == "LSPIN" else concrete_temperature
        lam          = float(cfg["lam"])
        smooth       = float(cfg["smooth"])
        cfg_seeds    = [int(seed + 10000 * cfg["global_cfg_idx"] + k) for k in range(n_reps)]

        t0 = time.time()
        print(f"[worker {worker_id}] [{cfg_i + 1}/{n_cfg}] {family_label} "
              f"lam={lam:.4g} smooth={smooth:.4g}", flush=True)

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
                    concrete_mode=(concrete_mode if gate_type == "concrete" else "relaxed"),
                )
            except Exception as exc:
                print(f"[worker {worker_id}]   seed={s} FAILED: {exc}", flush=True)
                continue

            _, g_det, hard_g, _ = sds.get_gates(
                model, Xt_test, device=device, hard_threshold=0.5, batch_size=512,
            )
            G_det  = g_det.numpy()
            G_hard = hard_g.numpy()

            gpars = sds.global_parsimony_metrics(G_hard, freq_threshold=global_freq_threshold)
            risk_vec = sds.model_risk_scores(sds.RiskOnlyWrapper(model).to(device), Xt_test, device=device)

            run_rows.append({
                "model_family": family_label,
                "gate_type":    gate_type,
                "gate_sigma":   gate_sigma,
                "lambda_sparse":       lam,
                "lambda_sample_smooth": smooth,
                "seed":        int(s),
                "run_id":      int(run_id),
                "test_cindex": float(info.get("test_cindex", np.nan)),
                "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
                "best_epoch":  int(info.get("best_epoch", -1)),
                "mean_Khard":  float(G_hard.sum(axis=1).mean()),
                "manifold_alignment": sds.stable_manifold_alignment(model, Xt_test, device=device),
                **gpars,
            })

            # histology breakdown
            d_hist = sds.cindex_by_histology(
                sds.RiskOnlyWrapper(model).to(device),
                Xt_test, tt_test, et_test, data["histo_test"], device=device,
            )
            if len(d_hist):
                d_hist["model_family"]         = family_label
                d_hist["lambda_sparse"]        = lam
                d_hist["lambda_sample_smooth"] = smooth
                d_hist["seed"]                 = int(s)
                hist_rows.append(d_hist)

            aff_vecs.append(sds.affinity_upper_vec(sds.gate_affinity_matrix(G_hard, normalize="01", zero_diag=True)))
            risk_vecs.append(risk_vec.astype(np.float32))
            cluster_vecs.append(sds.cluster_labels_from_gates(G_det, n_clusters=cluster_n_clusters, seed=s))

        # Pairwise metrics across seeds
        for i in range(len(aff_vecs)):
            for j in range(i + 1, len(aff_vecs)):
                lo_i, hi_i = sds.risk_group_indices(risk_vecs[i], risk_top_frac)
                lo_j, hi_j = sds.risk_group_indices(risk_vecs[j], risk_top_frac)
                hi_stat = sds.topk_overlap_stats(hi_i, hi_j, universe=len(risk_vecs[i]))
                lo_stat = sds.topk_overlap_stats(lo_i, lo_j, universe=len(risk_vecs[i]))
                base = {
                    "model_family": family_label, "gate_type": gate_type,
                    "lambda_sparse": lam, "lambda_sample_smooth": smooth,
                    "run_i": i, "run_j": j,
                }
                aff_rows.append({**base, "affinity_corr": sds.affinity_corr_from_vec(aff_vecs[i], aff_vecs[j])})
                risk_rows.append({
                    **base,
                    "risk_corr": sds.vector_corr(risk_vecs[i], risk_vecs[j]),
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

    # Write partial results to worker-specific dir
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


# ── Post-processing (runs in main process after all workers finish) ────────────

def _post_process(results_dir: Path, args) -> None:
    """Merge partial CSVs and run all post-processing from the original broad script."""
    import matplotlib
    matplotlib.use("Agg")

    import sparsedeepsurv.summary as sds_summary

    summarize_runs          = sds_summary.summarize_runs
    summarize_affinity      = sds_summary.summarize_affinity
    summarize_pairwise_metric = sds_summary.summarize_pairwise_metric
    build_merged_summary    = sds_summary.build_merged_summary
    rank_top_configs        = sds_summary.rank_top_configs
    save_smoothing_legend   = sds_summary.save_smoothing_legend
    save_family_loess       = sds_summary.save_family_loess
    save_broad_multiplot    = sds_summary.save_broad_multiplot

    # Observation binned figures module — imports may vary by version; make optional
    import sys as _sys
    for _p in [str(LSPIN_ROOT), str(CLEANED)]:
        if _p not in _sys.path:
            _sys.path.insert(0, _p)
    try:
        import cleaned_analyses.make_patient_smoothing_observation_binned_figures as _obs_mod
        if not hasattr(_obs_mod, "load_config_ratio"):
            _obs_mod.load_config_ratio = _obs_mod.load_config_axes
        from cleaned_analyses.make_patient_smoothing_observation_binned_figures import (
            save_figure as save_observation_binned_figure,
            save_legend as save_observation_binned_legend,
            load_config_axes as load_config_ratio,
            merge_run_obs,
            summarize as summarize_observation_bins,
        )
        _have_obs_figs = True
    except ImportError as _e:
        print(f"[post-process] observation-binned figures module not importable: {_e} — skipping", flush=True)
        _have_obs_figs = False

    partial_dirs = sorted(results_dir.glob("_partial_worker*"))
    print(f"[post-process] merging {len(partial_dirs)} partial result dirs")

    df_runs     = pd.concat([pd.read_csv(d / "run_rows.csv")     for d in partial_dirs], ignore_index=True)
    df_aff      = pd.concat([pd.read_csv(d / "aff_rows.csv")     for d in partial_dirs], ignore_index=True)
    df_risk     = pd.concat([pd.read_csv(d / "risk_rows.csv")    for d in partial_dirs], ignore_index=True)
    df_cluster  = pd.concat([pd.read_csv(d / "cluster_rows.csv") for d in partial_dirs], ignore_index=True)
    hist_frames = [pd.read_csv(d / "hist_rows.csv") for d in partial_dirs]
    df_hist     = pd.concat([f for f in hist_frames if not f.empty], ignore_index=True) if hist_frames else pd.DataFrame()

    df_runs.to_csv(results_dir / "notebook_style_models_runs.csv", index=False)
    df_aff.to_csv(results_dir / "notebook_style_models_affinity_pairs.csv", index=False)
    df_risk.to_csv(results_dir / "notebook_style_models_risk_pairs.csv", index=False)
    df_cluster.to_csv(results_dir / "notebook_style_models_cluster_pairs.csv", index=False)
    df_hist.to_csv(results_dir / "notebook_style_models_histology_runs.csv", index=False)

    acc     = summarize_runs(df_runs)
    aff     = summarize_affinity(df_aff, df_runs)
    risk    = summarize_pairwise_metric(df_risk, df_runs, metrics=[
        "risk_corr", "top_risk_overlap_ratio", "top_risk_jaccard",
        "bottom_risk_overlap_ratio", "bottom_risk_jaccard"
    ])
    cluster = summarize_pairwise_metric(df_cluster, df_runs, metrics=["cluster_ari", "cluster_nmi"])

    acc.to_csv(results_dir / "notebook_style_models_accuracy_summary.csv", index=False)
    aff.to_csv(results_dir / "notebook_style_models_affinity_summary.csv", index=False)
    risk.to_csv(results_dir / "notebook_style_models_risk_summary.csv", index=False)
    cluster.to_csv(results_dir / "notebook_style_models_cluster_summary.csv", index=False)

    merged = build_merged_summary(acc, aff, risk, cluster)
    merged.to_csv(results_dir / "kipan_broad_tuning_merged_summary.csv", index=False)

    top = rank_top_configs(merged, x_min=float(args.plot_x_min), x_max=float(args.plot_x_max))
    top.to_csv(results_dir / "kipan_broad_tuning_top_configs_in_band.csv", index=False)

    smooth_vals = sorted(acc["lambda_sample_smooth"].unique().tolist())
    save_smoothing_legend(smooth_vals, results_dir / "fig_notebook_style_smoothing_legend.png")

    for family in ["LSPIN", "Concrete"]:
        for y_col, y_label in [
            ("mean_test_c", "Mean test C-index across seeds"),
            ("mean_manifold_alignment", "Mean alignment to test patient manifold"),
            ("mean_gene_union_count", "Mean union of genes selected across test patients"),
        ]:
            save_family_loess(
                acc, family=family, y_col=y_col, y_label=y_label,
                outpath=results_dir / f"fig_{family.lower()}_{y_col}_vs_khard.png",
                title=f"KIPAN {family} broad: {y_label} vs sparsity",
                x_min=float(args.plot_x_min), x_max=float(args.plot_x_max),
                frac=float(args.lowess_frac),
            )

    # 2-row × 6-col multiplot: LSPIN/Concrete rows, one panel per metric
    try:
        save_broad_multiplot(
            merged,
            dataset_label="KIPAN",
            outpath=results_dir / "fig_broad_multiplot_smoothing_vs_khard.png",
            x_min=float(args.plot_x_min),
            x_max=float(args.plot_x_max),
            frac=float(args.lowess_frac),
        )
    except Exception as exc:
        print(f"[post-process] broad multiplot failed (non-fatal): {exc}", flush=True)

    # Observation-binned tradeoff figures (same as original broad script)
    if _have_obs_figs:
        try:
            config_axes = load_config_ratio(results_dir)
            obs_data = merge_run_obs(results_dir, config_axes)
            obs_summaries = {}
            for metric, df_metric in obs_data.items():
                s_df = summarize_observation_bins(df_metric, bins=14, bootstrap=2000)
                obs_summaries[metric] = s_df
                s_df.to_csv(results_dir / f"observation_binned_{metric}_summary.csv", index=False)
            save_observation_binned_legend(results_dir / "fig_observation_binned_ratio_legend.png")
            save_observation_binned_figure(
                obs_data, obs_summaries,
                results_dir / "fig_observation_binned_ratio_tradeoffs.png",
                "KIPAN",
            )
        except Exception as exc:
            print(f"[post-process] observation-binned figures failed (non-fatal): {exc}", flush=True)

    print(f"[post-process] done → {results_dir}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse():
    p = argparse.ArgumentParser(description="KIPAN broad sweep (paper repo, knn_k=8, multi-GPU)")
    p.add_argument("--outdir",      type=Path, default=KIPAN_DATA_DEFAULT)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    p.add_argument("--gpus",        type=int,  nargs="+", default=[0, 2, 6, 7])
    p.add_argument("--seed",        type=int,  default=123)
    p.add_argument("--knn-k",       type=int,  default=8)
    p.add_argument("--n-reps",      type=int,  default=10)
    p.add_argument("--patience",    type=int,  default=60)
    p.add_argument("--lr",          type=float, default=1e-2)
    p.add_argument("--weight-decay",type=float, default=1e-5)
    p.add_argument("--batch-size",  type=int,  default=128)
    p.add_argument("--max-epochs",  type=int,  default=300)
    # Gate params (same as original defaults)
    p.add_argument("--lspin-gate-sigma",    type=float, default=0.25)
    p.add_argument("--lspin-temperature",   type=float, default=0.5)
    p.add_argument("--concrete-gate-sigma", type=float, default=0.1)
    p.add_argument("--concrete-temperature",type=float, default=0.3)
    p.add_argument("--concrete-mode", choices=["relaxed", "ste"], default="ste")
    # Lambda grids (same as original defaults)
    p.add_argument("--positive-smooth-grid", type=float, nargs="+",
                   default=[0.05, 0.1, 0.15, 0.2])
    p.add_argument("--lspin-nosmooth-lambdas", type=float, nargs="+",
                   default=[0.0018, 0.0016, 0.0015, 0.0013, 0.0012, 0.0011,
                             0.0010, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005])
    p.add_argument("--lspin-smooth-lambdas", type=float, nargs="+",
                   default=[0.0015, 0.0013, 0.0012, 0.0011, 0.0010, 0.0009, 0.0008, 0.0007])
    p.add_argument("--concrete-nosmooth-lambdas", type=float, nargs="+",
                   default=[0.005, 0.004, 0.003, 0.0025, 0.002, 0.0017,
                             0.001442249570307409, 0.0012, 0.001, 0.00085, 0.0007, 0.0006])
    p.add_argument("--concrete-smooth-lambdas", type=float, nargs="+",
                   default=[0.003, 0.0025, 0.002, 0.0017,
                             0.001442249570307409, 0.0012, 0.001, 0.00085])
    # Eval params
    p.add_argument("--risk-top-frac",        type=float, default=0.2)
    p.add_argument("--cluster-n-clusters",   type=int,   default=3)
    p.add_argument("--global-freq-threshold",type=float, default=0.05)
    # Plot params (used in post-processing)
    p.add_argument("--plot-x-min",  type=float, default=5.0)
    p.add_argument("--plot-x-max",  type=float, default=2500.0)
    p.add_argument("--lowess-frac", type=float, default=0.55)
    p.add_argument("--post-process-only", action="store_true",
                   help="Skip training; merge existing partial dirs and post-process only")
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


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args = _parse()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = _Tee(args.results_dir / "run.log")

    if args.post_process_only:
        print(f"[main] --post-process-only: running post-processing on {args.results_dir}", flush=True)
        _post_process(args.results_dir, args)
        sys.exit(0)

    # Build the full ordered config list (preserving original global_cfg_idx for seed derivation)
    families_spec = [
        ("LSPIN",    "lspin_tf",  args.lspin_nosmooth_lambdas,    args.lspin_smooth_lambdas),
        ("Concrete", "concrete",  args.concrete_nosmooth_lambdas,  args.concrete_smooth_lambdas),
    ]
    all_configs: List[Dict] = []
    cfg_idx = 0
    for family_label, gate_type, no_lams, sm_lams in families_spec:
        sweep = [(0.0, lam) for lam in no_lams] + \
                [(smooth, lam) for smooth in args.positive_smooth_grid for lam in sm_lams]
        for smooth, lam in sweep:
            cfg_idx += 1
            all_configs.append({
                "family_label": family_label,
                "gate_type":    gate_type,
                "lam":          lam,
                "smooth":       smooth,
                "global_cfg_idx": cfg_idx,
            })

    gpus = args.gpus
    n_workers = len(gpus)
    # Distribute configs round-robin across workers
    batches: List[List[Dict]] = [[] for _ in range(n_workers)]
    for i, cfg in enumerate(all_configs):
        batches[i % n_workers].append(cfg)

    print(f"[ch3_kipan_broad] {len(all_configs)} configs → {n_workers} workers on GPUs {gpus}")
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
        lspin_gate_sigma=args.lspin_gate_sigma,
        lspin_temperature=args.lspin_temperature,
        concrete_gate_sigma=args.concrete_gate_sigma,
        concrete_temperature=args.concrete_temperature,
        concrete_mode=args.concrete_mode,
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
    _post_process(args.results_dir, args)
