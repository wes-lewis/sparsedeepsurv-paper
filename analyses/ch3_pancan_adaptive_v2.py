#!/usr/bin/env python3
"""
Chapter 3 TCGA Pan-Cancer adaptive sweep v2 — independent-init phase-2 replicates.

Two-phase adaptive design aligned with the current KIPAN/BRCA runs:
  Phase 1: all configs x n_phase1_reps (default 3) - screening.
  Phase 2: one family-level matched pair per model family (nosmooth + smooth)
           x remaining reps, all from fresh random initialization.

Grid (PanCan adaptive v2, 80 configs total):
  LSPIN:       5 lambdas × 2 sigmas × 4 smooth = 40
  Concrete: 5 lambdas × 2 sigmas × 4 smooth = 40

Usage:
  conda activate musevo
  cd /banach2/wes/lspin-repos/sparsedeepsurv-paper
  python analyses/ch3_pancan_adaptive_v2.py [--gpus 0 2 6 7]
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
PAPER_ROOT       = Path(__file__).resolve().parents[1]
PANCAN_DATA_DEFAULT = PAPER_ROOT / "data" / "processed" / "tcga_pancan_xena_20260330_top5000"
RESULTS_DEFAULT     = PAPER_ROOT / "data" / "runs" / "ch3_pancan_adaptive_v2"


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
    concrete_mode: str,
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
    import torch
    from pathlib import Path
    from sklearn.model_selection import ShuffleSplit
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import sparsedeepsurv as sds

    # Build partial_dir early so logs/partials are written incrementally
    partial_dir = Path(results_dir_str) / f"_partial_p{phase}_worker{worker_id}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    (partial_dir / "state_dicts").mkdir(exist_ok=True)
    worker_log_path = partial_dir / "worker.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with worker_log_path.open("a", buffering=1) as fh:
            fh.write(msg + "\n")

    log(
        f"[worker {worker_id}|p{phase}] start device={device} "
        f"n_configs={len(configs)} reps={rep_start}..{rep_end - 1} seed={seed}"
    )

    data = sds.load_pancan_split_artifacts(Path(outdir_str).resolve())
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
        global_cfg_idx = cfg["global_cfg_idx"]
        cfg_seeds    = [
            int(seed + 10000 * global_cfg_idx + k)
            for k in range(rep_start, rep_end)
        ]

        t0 = time.time()
        log(
            f"[worker {worker_id}|p{phase}] [{cfg_i + 1}/{n_cfg}] "
            f"{family_label} σ={gate_sigma} λ={lam:.4g} smooth={smooth:.4g} "
            f"reps={rep_start}..{rep_end - 1}",
        )

        aff_vecs:     List[np.ndarray] = []
        risk_vecs:    List[np.ndarray] = []
        cluster_vecs: List[np.ndarray] = []
        ok_reps = 0

        for run_id, s in enumerate(cfg_seeds):
            rep_id = rep_start + run_id
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
                    concrete_mode=(concrete_mode if gate_type == "concrete" else "relaxed"),
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    seed=s,
                )
            except Exception as exc:
                log(f"[worker {worker_id}|p{phase}]   seed={s} FAILED: {exc}")
                continue

            # Keep checkpoints for post hoc gate-interpretability analyses.
            sd_path = partial_dir / "state_dicts" / f"cfg{global_cfg_idx}_rep{rep_id}.pt"
            torch.save(model.state_dict(), sd_path)

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
                "rep_id":                rep_id,
                "global_cfg_idx":        global_cfg_idx,
                "phase":                 phase,
                "test_cindex":           float(info.get("test_cindex", np.nan)),
                "best_val_cindex":       float(info.get("best_val_cindex", np.nan)),
                "best_epoch":            int(info.get("best_epoch", -1)),
                "mean_Ksoft":            float(G_det.sum(axis=1).mean()),
                "mean_Khard":            float(G_hard.sum(axis=1).mean()),
                "manifold_alignment":    sds.stable_manifold_alignment(model, Xt_test, device=device),
                **gpars,
            })
            ok_reps += 1
            log(
                f"[worker {worker_id}|p{phase}]   seed={s} ok "
                f"test_c={float(info.get('test_cindex', np.nan)):.4f} "
                f"val_c={float(info.get('best_val_cindex', np.nan)):.4f} "
                f"epoch={int(info.get('best_epoch', -1))} "
                f"Ksoft={float(G_det.sum(axis=1).mean()):.1f} "
                f"Khard={float(G_hard.sum(axis=1).mean()):.1f}"
            )

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

        log(
            f"[worker {worker_id}|p{phase}]   done in {time.time() - t0:.1f}s "
            f"ok_reps={ok_reps}/{len(cfg_seeds)}"
        )

    pd.DataFrame(run_rows).to_csv(partial_dir / "run_rows.csv", index=False)
    pd.DataFrame(aff_rows).to_csv(partial_dir / "aff_rows.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(partial_dir / "risk_rows.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(partial_dir / "cluster_rows.csv", index=False)
    if hist_rows:
        pd.concat(hist_rows, ignore_index=True).to_csv(partial_dir / "hist_rows.csv", index=False)
    else:
        pd.DataFrame().to_csv(partial_dir / "hist_rows.csv", index=False)

    log(
        f"[worker {worker_id}|p{phase}] wrote → {partial_dir} "
        f"runs={len(run_rows)} aff={len(aff_rows)} risk={len(risk_rows)} "
        f"cluster={len(cluster_rows)} hist_frames={len(hist_rows)}"
    )
    return str(partial_dir)


# ── Phase helpers ─────────────────────────────────────────────────────────────
# (Shared logic imported from kipan_adaptive_v2 where possible)

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ch3_kipan_adaptive_v2 import (  # noqa: E402
    _load_partials,
    _select_promising,
    _rebuild_configs_for_phase2,
    _select_showcase_configs,
    _post_process,
    _Tee,
)


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
    print(
        f"[phase {phase}] dispatching {len(configs)} configs across {len(gpus)} GPUs: "
        + ", ".join(f"gpu{gpu}={len(bucket)}" for gpu, bucket in zip(gpus, buckets)),
        flush=True,
    )

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
        concrete_mode=args.concrete_mode,
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

    print(f"[phase {phase}] completed partial_dirs={len(partial_dirs)}", flush=True)
    return partial_dirs


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
        description="TCGA Pan-Cancer adaptive sweep v2: 3-rep phase 1 → 9 more random-init reps on a matched pair per family"
    )
    p.add_argument("--outdir",      type=Path, default=PANCAN_DATA_DEFAULT)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    p.add_argument("--gpus",        type=int,  nargs="+", default=[0, 2, 6, 7])
    p.add_argument("--seed",        type=int,  default=123)
    p.add_argument("--knn-k",       type=int,  default=5)
    p.add_argument("--n-phase1-reps", type=int, default=3,
                   help="Number of seeds per config in phase 1 (screening)")
    p.add_argument("--n-total-reps",  type=int, default=12,
                   help="Total seeds per promising config (phase 1 + phase 2)")
    p.add_argument("--top-n-promising", type=int, default=1,
                   help="Retained for compatibility; phase 2 now carries one relative-Khard-matched pair per family")

    p.add_argument("--patience",     type=int,   default=60)
    p.add_argument("--lr",           type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size",   type=int,   default=128)
    p.add_argument("--max-epochs",   type=int,   default=300)

    # Lambda grids (PanCan adaptive v2 — same as v1)
    p.add_argument("--lspin-lambdas", type=float, nargs="+",
                   default=[0.0009, 0.0007, 0.00055, 0.0004, 0.0003])
    p.add_argument("--concrete-lambdas", type=float, nargs="+",
                   default=[0.0014, 0.0012, 0.0010, 0.0008, 0.0006])

    # Two sigmas per family
    p.add_argument("--lspin-sigmas",    type=float, nargs="+", default=[0.25, 0.20])
    p.add_argument("--concrete-sigmas", type=float, nargs="+", default=[0.10, 0.15])

    # Temperatures
    p.add_argument("--lspin-temperature",    type=float, default=0.5)
    p.add_argument("--concrete-temperature", type=float, default=0.3)
    p.add_argument("--concrete-mode", choices=["relaxed", "ste"], default="ste")

    # Smooth grid
    p.add_argument("--sample-smooth-grid", type=float, nargs="+",
                   default=[0.0, 0.05, 0.1, 0.15])

    # Evaluation
    p.add_argument("--risk-top-frac",          type=float, default=0.2)
    p.add_argument("--cluster-n-clusters",     type=int,   default=5)
    p.add_argument("--global-freq-threshold",  type=float, default=0.05)

    # Showcase selection Khard zone
    p.add_argument("--khard-min", type=float, default=250.0)
    p.add_argument("--khard-max", type=float, default=600.0)

    # Manuscript-style heatmap rendering
    p.add_argument("--heatmap-top-genes", type=int, default=180)
    p.add_argument("--heatmap-min-frac-on", type=float, default=0.05)
    p.add_argument("--heatmap-max-frac-on", type=float, default=0.95)
    p.add_argument("--skip-heatmaps", action="store_true")

    p.add_argument("--post-process-only", action="store_true",
                   help="Skip training; merge existing partial dirs and post-process")
    return p.parse_args()


def main():
    args = _parse()
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    sys.stdout = _Tee(results_dir / "run.log")

    all_configs = _build_configs(args)
    print(f"[main] {len(all_configs)} total configs in grid (PanCan)")
    print(f"[main] GPUs: {args.gpus}")
    print(f"[main] Phase 1: {args.n_phase1_reps} reps/config, "
          f"Phase 2: {args.n_total_reps - args.n_phase1_reps} more reps on one relative-Khard-matched pair per family")
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
            print(f"\n[main] === Phase 2: {len(p2_configs)} configs × {n_p2} more reps (random init) ===",
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
    try:
        from render_adaptive_manuscript_figures import render as _render_adaptive_manuscript_figures

        _render_adaptive_manuscript_figures(
            results_dir=results_dir,
            outdir=args.outdir,
            dataset="pancan",
            args=args,
        )
        print(f"[main] notebook-style selected figures done -> {results_dir}", flush=True)
    except Exception as exc:
        print(f"[main] WARNING: notebook-style selected figure render failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
