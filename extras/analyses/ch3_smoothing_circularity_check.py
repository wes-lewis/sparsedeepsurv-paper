#!/usr/bin/env python3
"""Is the smoothing-improves-ARI/affinity-reproducibility claim circular?

Post-dissertation manuscript-completeness thread (see
CH3_LINEAR_BROAD_REDO.md and the discussion that followed it). The
patient-similarity smoothing penalty directly shrinks disagreement
between neighboring (KNN-graph) patients' gate-probability vectors:

    R_sample = (1/|E|) sum_(i,j) in E ||pi_i - pi_j||^2

and the chapter then validates it using cluster-stability (ARI) and
affinity-reproducibility metrics computed on those same gate-probability
vectors, across independently-trained runs, on held-out test patients.
That is not literally circular (the penalty acts on training patients
within one run; the metrics are cross-run, held-out-patient measures of
whether the induced bias generalizes and reproduces) -- but any
reasonable smoothness/shrinkage regularizer will mechanically reduce
cross-run variance as a near-definitional consequence of the bias-
variance tradeoff, regardless of whether what's being smoothed toward is
biologically meaningful or just noise. Pushed far enough, neighbors
converge to the same gate vector and ARI/affinity go to ~1 by
construction. So "smoothing improves ARI/affinity" is close to the
*expected* outcome of a correctly-implemented regularizer, not strong
evidence it is doing something biologically valuable.

Three checks, all in one script since they share the same training loop:

  1. NEGATIVE-CONTROL AFFINITY GRAPH (the decisive one). Build the KNN
     graph from real PCA-projected covariates (as normal) vs. from pure
     Gaussian noise of the same shape (same k, same construction code,
     only the input feature space is meaningless). If a meaningless
     graph gives the same ARI/affinity reproducibility gain, the effect
     is generic shrinkage, not evidence the mechanism is doing anything
     tied to real molecular structure.

  2. INDEPENDENT BIOLOGICAL SIGNAL, not just internal agreement. Compute
     each run's cluster ARI against REAL histology labels (never in the
     loss function or the KNN graph construction) rather than only
     against other runs' clusters. Does smoothing with the REAL graph
     increase alignment with real biology beyond both the unsmoothed
     baseline and the null-graph-smoothed condition? That is a
     fundamentally different, harder-to-fake claim than internal
     cross-run agreement.

  3. GENERIC/HOUSEKEEPING CONVERGENCE CHECK. If smoothing's real effect
     is nudging everyone toward a handful of broadly-expressed,
     generically-"prognostic" proliferation/cell-cycle genes (a known
     confound in this data -- see the chapter's own discussion), high
     reproducibility is bad news dressed as a good number. Checks
     whether the smoothing-stabilized consensus gene set is
     disproportionately enriched for generic proliferative Hallmark
     categories relative to more specific ones, compared across
     conditions.

Design: for each of {LSPIN, L-Concrete} (one family whose smoothing
union-direction was flagged as less robust in the broad-sweep redo, one
"star performer" family), three conditions at matched (lambda_sparse,
lambda_sample_smooth) -- unsmoothed, smoothed with the real graph,
smoothed with a null graph -- each run --n-reps times with independent
seeds. KIPAN only (fastest, has a clean 3-histology label for check #2).

Usage:
    python ch3_smoothing_circularity_check.py --device cuda:0
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

import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import LabelEncoder


def _deterministic_offset(*parts: str, modulus: int = 100000) -> int:
    """Deterministic seed offset from string parts (Python's built-in hash()
    is randomized per-process for strings unless PYTHONHASHSEED is fixed)."""
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(digest, 16) % modulus

import sparsedeepsurv as sds
from quick_linear_gated_probe import _load_data, _selected_configs

HALLMARK_GENE_SETS = GENE_SETS_DIR / "enrichr" / "MSigDB_Hallmark_2020.gmt"
RUN_DEFAULTS = {"kipan": CANONICAL_RUNS["kipan_adaptive_gentle"]}
DATA_DEFAULTS = PROCESSED_DATASETS

# Broadly proliferative / cell-cycle / generic-growth Hallmark categories --
# the chapter's own discussion flags these as a known confound ("proliferative
# and cell-cycle programs are widespread"). Anything not in this set is
# treated as more specific/informative for the enrichment-shift check.
GENERIC_HALLMARK_CATEGORIES = {
    "E2F Targets", "G2M Checkpoint", "Myc Targets V1", "Myc Targets V2",
    "Mitotic Spindle", "MTORC1 Signaling", "Oxidative Phosphorylation",
    "DNA Repair", "Unfolded Protein Response", "Protein Secretion",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["kipan"], default="kipan")
    p.add_argument("--families", nargs="+", default=["LSPIN", "L-Concrete"], choices=["LSPIN", "Concrete", "L-LSPIN", "L-Concrete"])
    p.add_argument("--n-reps", type=int, default=6)
    p.add_argument("--n-clusters", type=int, default=3)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260414)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hard-threshold", type=float, default=0.5)
    p.add_argument("--gene-rate-threshold", type=float, default=0.10)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def _load_hallmark_gene_sets(path: Path) -> dict[str, set[str]]:
    sets = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sets[parts[0]] = {g.strip().upper() for g in parts[2:] if g.strip()}
    return sets


def generic_fraction(selected_genes: set[str], gene_sets: dict[str, set[str]]) -> tuple[float, float]:
    """Of the selected genes that fall in ANY Hallmark category, what
    fraction fall in a GENERIC (proliferative/growth) category vs. a more
    specific one? Returns (generic_fraction, frac_hallmark_annotated)."""
    generic_genes = set().union(*(gene_sets[c] for c in GENERIC_HALLMARK_CATEGORIES if c in gene_sets))
    all_hallmark_genes = set().union(*gene_sets.values())
    annotated = selected_genes & all_hallmark_genes
    if not annotated:
        return float("nan"), 0.0
    generic_hit = annotated & generic_genes
    return len(generic_hit) / len(annotated), len(annotated) / max(1, len(selected_genes))


def _build_null_graph_like(Xt_tr: torch.Tensor, k: int) -> "scipy.sparse.csr_matrix":
    """Same KNN construction, same k, but the input feature space is pure
    Gaussian noise of matching shape -- a negative control with no
    molecular meaning, only the same graph *shape* (degree, symmetry)."""
    noise = torch.randn_like(Xt_tr)
    return sds.build_knn_adjacency_csr(noise, k=k, pca_dim=50, metric="cosine", symmetrize=True)


def run_condition(
    *, X, time, event, train_idx, val_idx, test_idx, cfg, A, n_reps, args, base_seed, histo_test_encoded,
) -> dict:
    hard_mats, cluster_labels_list, affinity_vecs, cindices = [], [], [], []
    for rep in range(n_reps):
        seed = base_seed + rep
        Xt_tr = sds.as_torch(X[train_idx])
        model, info = sds.run_one_model(
            Xt_tr=Xt_tr, tt_tr=sds.as_torch(time[train_idx]), et_tr=sds.as_torch(event[train_idx]),
            Xt_val=sds.as_torch(X[val_idx]), tt_val=sds.as_torch(time[val_idx]), et_val=sds.as_torch(event[val_idx]),
            Xt_test=sds.as_torch(X[test_idx]), tt_test=sds.as_torch(time[test_idx]), et_test=sds.as_torch(event[test_idx]),
            input_dim=X.shape[1], A_sample_train=A, device=args.device,
            lr=float(args.lr), weight_decay=float(args.weight_decay), batch_size=int(args.batch_size),
            max_epochs=int(args.max_epochs), seed=seed,
            gate_type=str(cfg["gate_type"]), gate_sigma=float(cfg["gate_sigma"]),
            lam=float(cfg["lambda_sparse"]), lambda_sample_smooth=float(cfg["lambda_sample_smooth"]),
            patience=int(args.patience), temperature=float(cfg["temperature"]),
            concrete_mode=str(cfg["concrete_mode"]), predictor=str(cfg["predictor"]),
            gating_hidden_dim=int(cfg["gating_hidden_dim"]), gate_hidden_dropout_p=float(cfg["gate_hidden_dropout_p"]),
            risk_hidden_dims=tuple(cfg["risk_hidden_dims"]), risk_dropout_p=float(cfg["risk_dropout_p"]),
            lspin_init_bias=float(cfg["lspin_init_bias"]), gate_weight_decay=float(cfg["gate_weight_decay"]),
        )
        g_det, hard = None, None
        _, g_det_t, hard_t, _ = sds.get_gates(
            model, sds.as_torch(X[test_idx]), device=args.device, hard_threshold=float(args.hard_threshold), batch_size=512,
        )
        g_det, hard = g_det_t.numpy(), hard_t.numpy()
        hard_mats.append(hard)
        affinity_vecs.append(sds.affinity_upper_vec(sds.gate_affinity_matrix(hard, normalize="01", zero_diag=True)))
        cluster_labels_list.append(sds.cluster_labels_from_gates(g_det, n_clusters=int(args.n_clusters), seed=seed))
        cindices.append(info.get("test_cindex"))

    # Cross-run reproducibility (the manuscript's existing metric).
    ari_pairs, aff_pairs = [], []
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            ari_pairs.append(adjusted_rand_score(cluster_labels_list[i], cluster_labels_list[j]))
            aff_pairs.append(sds.affinity_corr_from_vec(affinity_vecs[i], affinity_vecs[j]))

    # Direction 2: agreement with REAL histology, never in the loss or graph.
    histology_ari = [adjusted_rand_score(cl, histo_test_encoded) for cl in cluster_labels_list]

    # Direction 3 setup: aggregate (consensus) hard-selected genes across reps.
    mean_rate = np.mean([h.mean(axis=0) for h in hard_mats], axis=0)

    return {
        "cross_run_ari_mean": float(np.nanmean(ari_pairs)),
        "cross_run_affinity_mean": float(np.nanmean(aff_pairs)),
        "histology_ari_mean": float(np.nanmean(histology_ari)),
        "histology_ari_std": float(np.nanstd(histology_ari)),
        "mean_test_cindex": float(np.nanmean(cindices)),
        "mean_khard": float(np.mean([h.sum(axis=1).mean() for h in hard_mats])),
        "gene_rate": mean_rate,
    }


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir or RUN_DEFAULTS[args.dataset]
    data_dir = args.data_dir or DATA_DEFAULTS[args.dataset]
    outdir = args.outdir or (results_dir / f"ch3_smoothing_circularity_{args.dataset}")
    outdir.mkdir(parents=True, exist_ok=True)

    data = _load_data(args.dataset, data_dir)
    X = np.vstack([data["X_train"], data["X_test"]]).astype(np.float32)
    time = np.concatenate([data["time_train"], data["time_test"]]).astype(np.float32)
    event = np.concatenate([data["event_train"], data["event_test"]]).astype(np.uint8)
    histo = np.concatenate([data["histo_train"], data["histo_test"]]).astype(str)
    genes = np.asarray(data["gene_names"]).astype(str)
    n = X.shape[0]

    from sklearn.model_selection import ShuffleSplit
    ss = ShuffleSplit(n_splits=1, test_size=0.30, random_state=int(args.seed))
    train_all, test_idx = next(ss.split(X))
    ss2 = ShuffleSplit(n_splits=1, test_size=0.15, random_state=int(args.seed) + 1)
    rel_train, rel_val = next(ss2.split(train_all))
    train_idx, val_idx = train_all[rel_train], train_all[rel_val]

    histo_test_encoded = LabelEncoder().fit_transform(histo[test_idx])
    print(f"[setup] n={n} test_n={len(test_idx)} histology categories: {np.unique(histo[test_idx])}", flush=True)

    gene_sets = _load_hallmark_gene_sets(HALLMARK_GENE_SETS)

    all_rows = []
    for family in args.families:
        selection = "smooth"
        configs_smooth = _selected_configs(
            args.dataset, results_dir, [family], [selection],
            lambda_scale_families=[family], lambda_scale_selections=[selection],
            lambda_scales=[1.0], smooth_smooth_values=None,
        )
        configs_nosmooth = _selected_configs(
            args.dataset, results_dir, [family], ["nosmooth"],
            lambda_scale_families=[family], lambda_scale_selections=["nosmooth"],
            lambda_scales=[1.0], smooth_smooth_values=None,
        )
        cfg_smooth = configs_smooth.iloc[0]
        cfg_nosmooth = configs_nosmooth.iloc[0]

        rng = np.random.default_rng(int(args.seed))
        Xt_tr_full = sds.as_torch(X[train_idx])
        print(f"\n[{family}] building real graph", flush=True)
        A_real = sds.build_knn_adjacency_csr(Xt_tr_full, k=int(args.knn_k), pca_dim=50, metric="cosine", symmetrize=True)
        print(f"[{family}] building null (noise) graph", flush=True)
        A_null = _build_null_graph_like(Xt_tr_full, k=int(args.knn_k))

        conditions = [
            ("unsmoothed", None, cfg_nosmooth),
            ("smoothed_real_graph", A_real, cfg_smooth),
            ("smoothed_null_graph", A_null, cfg_smooth),
        ]
        for cond_name, A, cfg in conditions:
            print(f"[{family}] condition={cond_name} lam={cfg['lambda_sparse']:.5g} "
                  f"smooth={cfg['lambda_sample_smooth']:.5g}", flush=True)
            base_seed = int(args.seed) + 1000 * _deterministic_offset(family, cond_name)
            result = run_condition(
                X=X, time=time, event=event, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                cfg=cfg, A=A, n_reps=int(args.n_reps), args=args, base_seed=base_seed,
                histo_test_encoded=histo_test_encoded,
            )
            gene_rate = result.pop("gene_rate")
            selected_genes = {genes[i].upper() for i in np.where(gene_rate >= args.gene_rate_threshold)[0]}
            generic_frac, hallmark_frac = generic_fraction(selected_genes, gene_sets)
            row = {
                "family": family, "condition": cond_name, "n_selected_genes": len(selected_genes),
                "generic_hallmark_fraction": generic_frac, "hallmark_annotated_fraction": hallmark_frac,
                **result,
            }
            all_rows.append(row)
            print(f"[{family}] [{cond_name}] {row}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "circularity_check_summary.csv", index=False)
    print("\n" + df.to_string(index=False), flush=True)
    print(f"\n[done] wrote {outdir}", flush=True)


if __name__ == "__main__":
    main()
