#!/usr/bin/env python3
"""
Broad gated sweep under the gentler baseline-like regime.

This broad rerun is meant to replace the older broad maps for KIPAN / BRCA when
we want smoothing-vs-sparsity trends under the updated training setup:
  - all four gated families: LSPIN, Concrete, L-LSPIN, L-Concrete
  - baseline-like predictors
  - lspin_init_bias=0, zero gate dropout, zero gate-specific weight decay
  - narrow sigma tuning around the current successful regime

Outputs include separate broad multiplots for:
  - MLP predictor families: LSPIN, Concrete
  - Linear predictor families: L-LSPIN, L-Concrete
"""
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import PAPER_ROOT, ensure_repo_imports, sparsedeepsurv_src

ensure_repo_imports()

_SDS_SRC_PATH = sparsedeepsurv_src()
_SDS_SRC = str(_SDS_SRC_PATH) if _SDS_SRC_PATH is not None else None


KIPAN_DATA_DEFAULT = PAPER_ROOT / "data" / "processed" / "kipan_20260209_213604"
BRCA_DATA_DEFAULT = PAPER_ROOT / "data" / "processed" / "tcga_brca20260214_001423"
CACHE_ROOT = PAPER_ROOT / "data" / "cache" / "ch3_broad_gentle"


DATASET_SPECS = {
    "kipan": {
        "dataset_label": "KIPAN",
        "outdir": KIPAN_DATA_DEFAULT,
        "results_dir": PAPER_ROOT / "data" / "runs" / "ch3_kipan_broad_gentle",
        "loader_name": "load_kipan_split_artifacts",
        "knn_k": 8,
        "sample_smooth_grid": [0.0, 0.025, 0.05, 0.1],
        "cluster_n_clusters": 3,
        "plot_x_min": 10.0,
        "plot_x_max": 3000.0,
        "families": {
            "LSPIN": {
                "gate_type": "lspin_tf",
                "predictor": "mlp",
                "sigmas": [0.22, 0.20],
                "lambdas": [0.0035, 0.00525, 0.007, 0.00875, 0.0105, 0.014, 0.021, 0.028],
                "temperature": 0.5,
                "patience": 12,
                "lr": 0.002,
            },
            "Concrete": {
                "gate_type": "concrete",
                "predictor": "mlp",
                "sigmas": [0.15, 0.12],
                "lambdas": [0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.006],
                "temperature": 0.3,
                "patience": 20,
                "lr": 0.001,
            },
            "L-LSPIN": {
                "gate_type": "lspin_tf",
                "predictor": "linear",
                "sigmas": [0.22, 0.20],
                "lambdas": [0.0035, 0.00525, 0.007, 0.00875, 0.0105, 0.014, 0.021, 0.028],
                "temperature": 0.5,
                "patience": 20,
                "lr": 0.002,
            },
            "L-Concrete": {
                "gate_type": "concrete",
                "predictor": "linear",
                "sigmas": [0.15, 0.12],
                "lambdas": [0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.006],
                "temperature": 0.3,
                "patience": 20,
                "lr": 0.001,
            },
        },
    },
    "brca": {
        "dataset_label": "BRCA",
        "outdir": BRCA_DATA_DEFAULT,
        "results_dir": PAPER_ROOT / "data" / "runs" / "ch3_brca_broad_gentle",
        "loader_name": "load_brca_split_artifacts",
        "knn_k": 5,
        "sample_smooth_grid": [0.0, 0.1, 0.2, 0.4],
        "cluster_n_clusters": 4,
        "plot_x_min": 10.0,
        "plot_x_max": 3000.0,
        "families": {
            "LSPIN": {
                "gate_type": "lspin_tf",
                "predictor": "mlp",
                "sigmas": [0.15, 0.12],
                "lambdas": [0.0035, 0.00525, 0.007, 0.00875, 0.0105, 0.014, 0.021, 0.028],
                "temperature": 0.5,
                "patience": 12,
                "lr": 0.002,
            },
            "Concrete": {
                "gate_type": "concrete",
                "predictor": "mlp",
                "sigmas": [0.15, 0.12],
                "lambdas": [0.0011, 0.00165, 0.0022, 0.00275, 0.0033, 0.0044, 0.0055, 0.0066],
                "temperature": 0.3,
                "patience": 20,
                "lr": 0.001,
            },
            "L-LSPIN": {
                "gate_type": "lspin_tf",
                "predictor": "linear",
                "sigmas": [0.15, 0.12],
                "lambdas": [0.0035, 0.00525, 0.007, 0.00875, 0.0105, 0.014, 0.021, 0.028],
                "temperature": 0.5,
                "patience": 35,
                "lr": 0.002,
            },
            "L-Concrete": {
                "gate_type": "concrete",
                "predictor": "linear",
                "sigmas": [0.15, 0.12],
                "lambdas": [0.0011, 0.00165, 0.0022, 0.00275, 0.0033, 0.0044, 0.0055, 0.0066],
                "temperature": 0.3,
                "patience": 20,
                "lr": 0.001,
            },
        },
    },
}


COMMON_GATED_DEFAULTS = {
    "gating_hidden_dim": 64,
    "risk_hidden_dims": (64, 32),
    "risk_dropout_p": 0.1,
    "gate_hidden_dropout_p": 0.0,
    "gate_weight_decay": 0.0,
    "lspin_init_bias": 0.0,
}


def _nice_log_ticks(x_lo: float, x_hi: float) -> List[float]:
    candidates = [5, 10, 20, 50, 100, 200, 300, 500, 700, 1000, 2000, 5000]
    ticks = [t for t in candidates if x_lo * 0.95 <= t <= x_hi * 1.05]
    if len(ticks) >= 3:
        if len(ticks) > 4:
            idx = np.round(np.linspace(0, len(ticks) - 1, 4)).astype(int)
            ticks = [ticks[i] for i in idx]
        return ticks
    vals = np.geomspace(max(x_lo, 1e-6), x_hi, num=3)
    return [round(float(v)) for v in vals]


def _short_metric_label(label: str) -> str:
    return {
        "Mean test C-index across seeds": "Mean test C-index",
        "Mean alignment to test patient manifold": "Manifold alignment",
        "Mean run-to-run risk correlation": "Risk reproducibility",
        "Mean run-to-run cluster ARI": "Cluster stability (ARI)",
        "Mean union of genes selected across test patients": "Mean global union",
        "Feature-use efficiency (mean Khard / union)": "Feature-use efficiency\n(mean Khard / union)",
        "Mean run-to-run affinity correlation": "Affinity reproducibility",
    }.get(label, label)


def _kernel_smooth_shared(x_log: np.ndarray, y: np.ndarray, frac_val: float):
    from statsmodels.nonparametric.smoothers_lowess import lowess

    if len(x_log) == 1:
        grid = np.array([x_log[0], x_log[0]])
        return grid, np.array([y[0], y[0]]), np.array([0.0, 0.0])
    grid = np.linspace(x_log.min(), x_log.max(), max(120, min(220, len(x_log) * 28)))
    x_span = max(x_log.max() - x_log.min(), 1e-6)
    bandwidth = max(x_span * max(frac_val, 0.18) * 0.38, 0.055)
    means, spreads = [], []
    for gx in grid:
        w = np.exp(-0.5 * ((x_log - gx) / bandwidth) ** 2)
        w_sum = np.clip(w.sum(), 1e-12, None)
        mu = float(np.sum(w * y) / w_sum)
        var = float(np.sum(w * (y - mu) ** 2) / w_sum)
        n_eff = float((w_sum ** 2) / np.clip(np.sum(w ** 2), 1e-12, None))
        spread = math.sqrt(max(var, 0.0)) * (0.65 + 0.35 / math.sqrt(max(n_eff, 1.0)))
        means.append(mu)
        spreads.append(spread)
    means = np.asarray(means, dtype=float)
    spreads = np.asarray(spreads, dtype=float)
    mean_sm = lowess(means, grid, frac=max(frac_val * 0.75, 0.2), return_sorted=False)
    spread_sm = lowess(spreads, grid, frac=max(frac_val * 0.9, 0.25), return_sorted=False)
    return grid, mean_sm, np.clip(spread_sm, 0.0, None)


def _plot_loess_panel(
    ax,
    subdf: pd.DataFrame,
    y_col: str,
    y_label: str,
    x_min: float,
    x_max: float,
    frac: float,
    palette,
    smooth_vals: List[float],
    show_xlabel: bool = True,
) -> None:
    subdf = subdf.copy()
    subdf["mean_Khard"] = pd.to_numeric(subdf["mean_Khard"], errors="coerce")
    subdf[y_col] = pd.to_numeric(subdf[y_col], errors="coerce")
    subdf = subdf[np.isfinite(subdf["mean_Khard"]) & np.isfinite(subdf[y_col])]
    subdf = subdf[(subdf["mean_Khard"] >= x_min) & (subdf["mean_Khard"] <= x_max)]

    if subdf.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_axis_off()
        return

    x_all = subdf["mean_Khard"].to_numpy(float)
    x_lo = max(float(np.nanmin(x_all)) * 0.88, x_min)
    x_hi = min(float(np.nanmax(x_all)) * 1.08, x_max)

    for idx, smooth in enumerate(smooth_vals):
        part = subdf[subdf["lambda_sample_smooth"] == smooth].sort_values("mean_Khard")
        if part.empty:
            continue
        color = palette[idx]
        x = part["mean_Khard"].to_numpy(float)
        y = part[y_col].to_numpy(float)
        ax.scatter(x, y, s=14, alpha=0.40, color=color, edgecolors="none", zorder=3)
        if len(part) == 1:
            ax.plot(x, y, color=color, linewidth=2.0, zorder=2)
            continue
        x_log = np.log10(np.clip(x, 1e-9, None))
        grid, mean_sm, spread_sm = _kernel_smooth_shared(x_log, y, frac)
        x_grid = 10 ** grid
        ax.fill_between(
            x_grid,
            mean_sm - spread_sm,
            mean_sm + spread_sm,
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x_grid, mean_sm, color=color, linewidth=2.0, zorder=2)

    from matplotlib.ticker import FixedFormatter, FixedLocator

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ticks = _nice_log_ticks(x_lo, x_hi)
    labels = [f"{int(t):d}" if t >= 10 else f"{t:g}" for t in ticks]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.tick_params(axis="x", labelsize=11, rotation=45)
    ax.tick_params(axis="y", labelsize=11)
    if show_xlabel:
        ax.set_xlabel("mean Khard", fontsize=12)
    ax.set_ylabel(_short_metric_label(y_label), fontsize=12)
    ax.grid(alpha=0.22)


def _save_broad_multiplot_for_families(
    df: pd.DataFrame,
    *,
    dataset_label: str,
    families: List[str],
    outpath: Path,
    x_min: float,
    x_max: float,
    frac: float,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    metrics = [
        ("mean_test_c", "Mean test C-index across seeds"),
        ("mean_risk_corr", "Mean run-to-run risk correlation"),
        ("mean_manifold_alignment", "Mean alignment to test patient manifold"),
        ("mean_affinity_corr", "Mean run-to-run affinity correlation"),
        ("mean_cluster_ari", "Mean run-to-run cluster ARI"),
        ("feat_efficiency", "Feature-use efficiency (mean Khard / union)"),
    ]

    df = df.copy()
    if "mean_Khard" in df.columns and "mean_gene_union_count" in df.columns:
        df["feat_efficiency"] = df["mean_Khard"] / df["mean_gene_union_count"].replace(0, np.nan)
    metrics = [(c, l) for c, l in metrics if c in df.columns]
    if not metrics:
        return

    smooth_vals = sorted(df["lambda_sample_smooth"].dropna().unique().tolist())
    palette = sns.color_palette("tab10", n_colors=max(3, len(smooth_vals)))
    n_rows = len(families)
    n_cols = len(metrics)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.6 * n_cols, max(3.3 * n_rows, 6.25)),
        squeeze=False,
    )

    for row_idx, family in enumerate(families):
        subdf_fam = df[df["model_family"] == family].copy()
        for col_idx, (y_col, y_label) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            _plot_loess_panel(
                ax,
                subdf_fam,
                y_col,
                y_label,
                x_min=x_min,
                x_max=x_max,
                frac=frac,
                palette=palette,
                smooth_vals=smooth_vals,
                show_xlabel=(row_idx == n_rows - 1),
            )
            if row_idx == 0:
                ax.set_title(_short_metric_label(y_label), fontsize=12, pad=3)
        axes[row_idx][0].set_ylabel(
            f"{family}\n{_short_metric_label(metrics[0][1])}",
            fontsize=12,
        )

    legend_handles = [
        Line2D([0], [0], color=palette[i], lw=2.2, label=f"smooth={s:g}")
        for i, s in enumerate(smooth_vals)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(smooth_vals), 6),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.02),
    )

    fam_text = ", ".join(families)
    fig.suptitle(
        f"{dataset_label} broad sweep — smoothing effect across metrics ({fam_text})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_configs(args) -> List[Dict]:
    spec = DATASET_SPECS[args.dataset]
    selected_families = set(args.families) if args.families else None
    smooth_scaled_families = set(args.smooth_lambda_scale_families) if args.smooth_lambda_scale_families else set()
    smooth_grid = [float(x) for x in (args.sample_smooth_grid if args.sample_smooth_grid else spec["sample_smooth_grid"])]
    if 0.0 not in smooth_grid:
        smooth_grid = [0.0] + smooth_grid
    smooth_grid = sorted(set(smooth_grid))
    nosmooth_lambdas_override = [float(x) for x in args.nosmooth_lambdas] if args.nosmooth_lambdas else None
    smooth_lambdas_override = [float(x) for x in args.smooth_lambdas] if args.smooth_lambdas else None
    lspin_nosmooth_lambdas = [float(x) for x in args.lspin_nosmooth_lambdas] if args.lspin_nosmooth_lambdas else None
    lspin_smooth_lambdas = [float(x) for x in args.lspin_smooth_lambdas] if args.lspin_smooth_lambdas else None
    lspin_sigmas_override = [float(x) for x in args.lspin_sigmas] if args.lspin_sigmas else None
    all_configs: List[Dict] = []
    cfg_idx = 0
    for family_label, family in spec["families"].items():
        if selected_families is not None and family_label not in selected_families:
            continue
        family_lr = float(args.lr_override) if args.lr_override is not None else float(family["lr"])
        sigma_grid = lspin_sigmas_override if (family_label in {"LSPIN", "L-LSPIN"} and lspin_sigmas_override is not None) else family["sigmas"]
        for sigma in sigma_grid:
            for smooth in smooth_grid:
                if family_label == "LSPIN" and lspin_nosmooth_lambdas is not None:
                    lam_grid = lspin_nosmooth_lambdas if smooth == 0.0 else (lspin_smooth_lambdas if lspin_smooth_lambdas is not None else lspin_nosmooth_lambdas)
                elif nosmooth_lambdas_override is not None:
                    lam_grid = nosmooth_lambdas_override if smooth == 0.0 else (smooth_lambdas_override if smooth_lambdas_override is not None else nosmooth_lambdas_override)
                else:
                    lam_grid = family["lambdas"]
                for lam in lam_grid:
                    lam_eff = float(lam)
                    if smooth > 0 and family_label in smooth_scaled_families:
                        lam_eff *= float(args.smooth_lambda_scale)
                    cfg_idx += 1
                    all_configs.append(
                        {
                            "global_cfg_idx": cfg_idx,
                            "model_family": family_label,
                            "gate_type": family["gate_type"],
                            "predictor": family["predictor"],
                            "gate_sigma": float(sigma),
                            "lambda_sparse": lam_eff,
                            "lambda_sparse_base": float(lam),
                            "lambda_sample_smooth": float(smooth),
                            "temperature": float(family["temperature"]),
                            "patience": int(args.patience_override if args.patience_override is not None else family["patience"]),
                            "lr": family_lr,
                            "gating_hidden_dim": int(COMMON_GATED_DEFAULTS["gating_hidden_dim"]),
                            "gate_hidden_dropout_p": float(COMMON_GATED_DEFAULTS["gate_hidden_dropout_p"]),
                            "risk_hidden_dims": tuple(COMMON_GATED_DEFAULTS["risk_hidden_dims"]) if family["predictor"] == "mlp" else (),
                            "risk_dropout_p": float(COMMON_GATED_DEFAULTS["risk_dropout_p"]) if family["predictor"] == "mlp" else 0.0,
                            "lspin_init_bias": float(COMMON_GATED_DEFAULTS["lspin_init_bias"]),
                            "gate_weight_decay": float(COMMON_GATED_DEFAULTS["gate_weight_decay"]),
                        }
                    )
    return all_configs


def _cfg_stem(global_cfg_idx: int) -> str:
    return f"cfg{global_cfg_idx:04d}"


def _append_df(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _load_completed_cfg_ids(partial_dir: Path) -> set[int]:
    done_dir = partial_dir / "done_cfgs"
    if not done_dir.exists():
        return set()
    out: set[int] = set()
    for marker in done_dir.glob("cfg*.done"):
        m = re.fullmatch(r"cfg(\d+)\.done", marker.name)
        if m:
            out.add(int(m.group(1)))
    return out


def _write_cfg_frame(cfg_dir: Path, stem: str, suffix: str, df: pd.DataFrame) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / f"{stem}_{suffix}.csv").unlink(missing_ok=True)
    df.to_csv(cfg_dir / f"{stem}_{suffix}.csv", index=False)


def _broad_prefix(dataset: str) -> str:
    return f"{dataset}_broad_gentle"


def _prepare_shared_cache(args) -> Dict[str, str]:
    import sparsedeepsurv as sds
    from scipy.sparse import load_npz, save_npz
    from sklearn.model_selection import ShuffleSplit

    spec = DATASET_SPECS[args.dataset]
    cache_dir = CACHE_ROOT / args.dataset / f"seed{int(args.seed)}_knn{int(spec['knn_k'])}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_cache = cache_dir / "split_indices.npz"
    adj_cache = cache_dir / "train_knn_adj.npz"

    if not split_cache.exists():
        loader_name = spec["loader_name"]
        data = getattr(sds, loader_name)(args.outdir.resolve())
        X_train = data["X_train"]
        strat_key = np.array([f"{e}_{h}" for e, h in zip(data["event_train"], data["histo_train"])])
        ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=int(args.seed))
        i_tr, i_val = next(ss.split(X_train, strat_key))
        np.savez_compressed(split_cache, i_tr=i_tr, i_val=i_val)
        print(
            f"[cache] wrote split cache -> {split_cache} "
            f"(train={len(i_tr)} val={len(i_val)})",
            flush=True,
        )
    else:
        with np.load(split_cache) as cached:
            print(
                f"[cache] reusing split cache -> {split_cache} "
                f"(train={len(cached['i_tr'])} val={len(cached['i_val'])})",
                flush=True,
            )

    if not adj_cache.exists():
        loader_name = spec["loader_name"]
        data = getattr(sds, loader_name)(args.outdir.resolve())
        with np.load(split_cache) as cached:
            i_tr = cached["i_tr"]
        Xt_tr = sds.as_torch(data["X_train"][i_tr])
        A_train = sds.build_knn_adjacency_csr(
            Xt_tr, k=int(spec["knn_k"]), pca_dim=50, metric="cosine", symmetrize=True
        )
        save_npz(adj_cache, A_train)
        print(f"[cache] wrote train adjacency cache -> {adj_cache}", flush=True)
    else:
        A_train = load_npz(adj_cache)
        print(
            f"[cache] reusing train adjacency cache -> {adj_cache} "
            f"(shape={A_train.shape}, nnz={A_train.nnz})",
            flush=True,
        )

    return {
        "split_cache_npz_str": str(split_cache.resolve()),
        "adj_cache_npz_str": str(adj_cache.resolve()),
    }


def _read_nonempty_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    return df if not df.empty else None


def _load_reference_metric_frames(results_dir: Path, dataset: str, families: set[str] | None) -> Dict[str, pd.DataFrame]:
    prefix = _broad_prefix(dataset)
    metric_files = {
        "run_rows": results_dir / f"{prefix}_runs.csv",
        "aff_rows": results_dir / f"{prefix}_affinity_pairs.csv",
        "risk_rows": results_dir / f"{prefix}_risk_pairs.csv",
        "cluster_rows": results_dir / f"{prefix}_cluster_pairs.csv",
        "hist_rows": results_dir / f"{prefix}_histology_runs.csv",
    }
    out: Dict[str, pd.DataFrame] = {}
    for key, path in metric_files.items():
        frame = _read_nonempty_csv(path)
        if frame is None:
            continue
        if families is not None and "model_family" in frame.columns:
            frame = frame[frame["model_family"].astype(str).isin(families)].copy()
        if not frame.empty:
            out[key] = frame
    return out


def _parse_reference_specs(items: List[str]) -> List[tuple[Path, set[str] | None]]:
    out: List[tuple[Path, set[str] | None]] = []
    for item in items:
        if "::" in item:
            path_str, fam_str = item.split("::", 1)
            families = {x.strip() for x in fam_str.split(",") if x.strip()}
            out.append((Path(path_str), families if families else None))
        else:
            out.append((Path(item), None))
    return out


def _worker(
    worker_id: int,
    configs: List[Dict],
    *,
    outdir_str: str,
    results_dir_str: str,
    split_cache_npz_str: str,
    adj_cache_npz_str: str,
    dataset: str,
    seed: int,
    knn_k: int,
    device: str,
    n_reps: int,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    concrete_mode: str,
    risk_top_frac: float,
    cluster_n_clusters: int,
    global_freq_threshold: float,
) -> str:
    import sys as _sys

    if _SDS_SRC and _SDS_SRC not in _sys.path:
        _sys.path.insert(0, _SDS_SRC)

    import sparsedeepsurv as sds
    from scipy.sparse import load_npz
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    partial_dir = Path(results_dir_str) / f"_partial_worker{worker_id}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    worker_log = partial_dir / "worker.log"
    cfg_dir = partial_dir / "config_rows"
    done_dir = partial_dir / "done_cfgs"
    done_dir.mkdir(parents=True, exist_ok=True)
    completed_cfg_ids = _load_completed_cfg_ids(partial_dir)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with worker_log.open("a", buffering=1) as fh:
            fh.write(msg + "\n")

    loader_name = DATASET_SPECS[dataset]["loader_name"]
    data = getattr(sds, loader_name)(Path(outdir_str).resolve())
    X_train = data["X_train"]
    X_test = data["X_test"]
    split_cache = np.load(split_cache_npz_str)
    i_tr = split_cache["i_tr"]
    i_val = split_cache["i_val"]
    split_cache.close()

    Xt_tr = sds.as_torch(X_train[i_tr])
    Xt_val = sds.as_torch(X_train[i_val])
    Xt_test = sds.as_torch(X_test)
    tt_tr = sds.as_torch(data["time_train"][i_tr])
    tt_val = sds.as_torch(data["time_train"][i_val])
    tt_test = sds.as_torch(data["time_test"])
    et_tr = sds.as_torch(data["event_train"][i_tr])
    et_val = sds.as_torch(data["event_train"][i_val])
    et_test = sds.as_torch(data["event_test"])

    A_train = load_npz(adj_cache_npz_str)

    n_cfg = len(configs)
    log(
        f"[worker {worker_id}] start dataset={dataset} device={device} "
        f"n_configs={n_cfg} completed={len(completed_cfg_ids)} "
        f"split_cache={Path(split_cache_npz_str).name} adj_cache={Path(adj_cache_npz_str).name}"
    )
    for cfg_i, cfg in enumerate(configs):
        family_label = cfg["model_family"]
        gate_type = cfg["gate_type"]
        predictor = cfg["predictor"]
        gate_sigma = float(cfg["gate_sigma"])
        temperature = float(cfg["temperature"])
        lam = float(cfg["lambda_sparse"])
        smooth = float(cfg["lambda_sample_smooth"])
        patience = int(cfg["patience"])
        lr = float(cfg["lr"])
        gating_hidden_dim = int(cfg["gating_hidden_dim"])
        gate_hidden_dropout = float(cfg["gate_hidden_dropout_p"])
        risk_hidden_dims = tuple(int(x) for x in cfg["risk_hidden_dims"])
        risk_dropout_p = float(cfg["risk_dropout_p"])
        lspin_init_bias = float(cfg["lspin_init_bias"])
        gate_weight_decay = float(cfg["gate_weight_decay"])
        global_cfg_idx = int(cfg["global_cfg_idx"])
        stem = _cfg_stem(global_cfg_idx)
        cfg_seeds = [int(seed + 10000 * global_cfg_idx + k) for k in range(n_reps)]

        if global_cfg_idx in completed_cfg_ids:
            log(
                f"[worker {worker_id}] [{cfg_i + 1}/{n_cfg}] skip completed "
                f"{family_label} predictor={predictor} cfg={global_cfg_idx}"
            )
            continue

        t0 = time.time()
        log(
            f"[worker {worker_id}] [{cfg_i + 1}/{n_cfg}] {family_label} predictor={predictor} "
            f"sigma={gate_sigma:.3g} lambda={lam:.4g} smooth={smooth:.4g} cfg={global_cfg_idx}"
        )

        run_rows_cfg: List[Dict] = []
        aff_rows_cfg: List[Dict] = []
        risk_rows_cfg: List[Dict] = []
        cluster_rows_cfg: List[Dict] = []
        hist_rows_cfg: List[pd.DataFrame] = []
        aff_vecs: List[np.ndarray] = []
        risk_vecs: List[np.ndarray] = []
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
                    concrete_mode=(concrete_mode if gate_type == "concrete" else "relaxed"),
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    predictor=predictor,
                    gating_hidden_dim=gating_hidden_dim,
                    gate_hidden_dropout_p=gate_hidden_dropout,
                    risk_hidden_dims=risk_hidden_dims,
                    risk_dropout_p=risk_dropout_p,
                    lspin_init_bias=lspin_init_bias,
                    gate_weight_decay=gate_weight_decay,
                    seed=s,
                )
            except Exception as exc:
                log(f"[worker {worker_id}]   seed={s} FAILED: {exc}")
                continue

            _, g_det, hard_g, _ = sds.get_gates(
                model, Xt_test, device=device, hard_threshold=0.5, batch_size=512
            )
            G_det = g_det.numpy()
            G_hard = hard_g.numpy()
            gpars = sds.global_parsimony_metrics(G_hard, freq_threshold=global_freq_threshold)
            risk_vec = sds.model_risk_scores(sds.RiskOnlyWrapper(model).to(device), Xt_test, device=device)

            run_rows_cfg.append(
                {
                    "global_cfg_idx": global_cfg_idx,
                    "model_family": family_label,
                    "gate_type": gate_type,
                    "predictor": predictor,
                    "gate_sigma": gate_sigma,
                    "lambda_sparse": lam,
                    "lambda_sparse_base": float(cfg.get("lambda_sparse_base", lam)),
                    "lambda_sample_smooth": smooth,
                    "seed": int(s),
                    "run_id": int(run_id),
                    "test_cindex": float(info.get("test_cindex", np.nan)),
                    "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
                    "best_epoch": int(info.get("best_epoch", -1)),
                    "mean_Khard": float(G_hard.sum(axis=1).mean()),
                    "manifold_alignment": sds.stable_manifold_alignment(model, Xt_test, device=device),
                    **gpars,
                }
            )

            d_hist = sds.cindex_by_histology(
                sds.RiskOnlyWrapper(model).to(device),
                Xt_test,
                tt_test,
                et_test,
                data["histo_test"],
                device=device,
            )
            if len(d_hist):
                d_hist["global_cfg_idx"] = global_cfg_idx
                d_hist["model_family"] = family_label
                d_hist["predictor"] = predictor
                d_hist["lambda_sparse"] = lam
                d_hist["lambda_sample_smooth"] = smooth
                d_hist["seed"] = int(s)
                hist_rows_cfg.append(d_hist)

            aff_vecs.append(
                sds.affinity_upper_vec(sds.gate_affinity_matrix(G_hard, normalize="01", zero_diag=True))
            )
            risk_vecs.append(risk_vec.astype(np.float32))
            cluster_vecs.append(sds.cluster_labels_from_gates(G_det, n_clusters=cluster_n_clusters, seed=s))

        for i in range(len(aff_vecs)):
            for j in range(i + 1, len(aff_vecs)):
                lo_i, hi_i = sds.risk_group_indices(risk_vecs[i], risk_top_frac)
                lo_j, hi_j = sds.risk_group_indices(risk_vecs[j], risk_top_frac)
                hi_stat = sds.topk_overlap_stats(hi_i, hi_j, universe=len(risk_vecs[i]))
                lo_stat = sds.topk_overlap_stats(lo_i, lo_j, universe=len(risk_vecs[i]))
                base = {
                    "model_family": family_label,
                    "gate_type": gate_type,
                    "predictor": predictor,
                    "lambda_sparse": lam,
                    "lambda_sparse_base": float(cfg.get("lambda_sparse_base", lam)),
                    "lambda_sample_smooth": smooth,
                    "run_i": i,
                    "run_j": j,
                }
                aff_rows_cfg.append(
                    {**base, "global_cfg_idx": global_cfg_idx, "affinity_corr": sds.affinity_corr_from_vec(aff_vecs[i], aff_vecs[j])}
                )
                risk_rows_cfg.append(
                    {
                        **base,
                        "global_cfg_idx": global_cfg_idx,
                        "risk_corr": sds.vector_corr(risk_vecs[i], risk_vecs[j]),
                        "top_risk_overlap_ratio": float(hi_stat["overlap_ratio"]),
                        "top_risk_jaccard": float(hi_stat["jaccard"]),
                        "bottom_risk_overlap_ratio": float(lo_stat["overlap_ratio"]),
                        "bottom_risk_jaccard": float(lo_stat["jaccard"]),
                    }
                )
                cluster_rows_cfg.append(
                    {
                        **base,
                        "global_cfg_idx": global_cfg_idx,
                        "cluster_ari": float(adjusted_rand_score(cluster_vecs[i], cluster_vecs[j])),
                        "cluster_nmi": float(normalized_mutual_info_score(cluster_vecs[i], cluster_vecs[j])),
                    }
                )

        run_df_cfg = pd.DataFrame(run_rows_cfg)
        aff_df_cfg = pd.DataFrame(aff_rows_cfg)
        risk_df_cfg = pd.DataFrame(risk_rows_cfg)
        cluster_df_cfg = pd.DataFrame(cluster_rows_cfg)
        hist_df_cfg = (
            pd.concat(hist_rows_cfg, ignore_index=True) if hist_rows_cfg else pd.DataFrame()
        )

        _write_cfg_frame(cfg_dir, stem, "run_rows", run_df_cfg)
        _write_cfg_frame(cfg_dir, stem, "aff_rows", aff_df_cfg)
        _write_cfg_frame(cfg_dir, stem, "risk_rows", risk_df_cfg)
        _write_cfg_frame(cfg_dir, stem, "cluster_rows", cluster_df_cfg)
        _write_cfg_frame(cfg_dir, stem, "hist_rows", hist_df_cfg)

        _append_df(partial_dir / "run_rows.csv", run_df_cfg)
        _append_df(partial_dir / "aff_rows.csv", aff_df_cfg)
        _append_df(partial_dir / "risk_rows.csv", risk_df_cfg)
        _append_df(partial_dir / "cluster_rows.csv", cluster_df_cfg)
        _append_df(partial_dir / "hist_rows.csv", hist_df_cfg)

        (done_dir / f"{stem}.done").write_text("", encoding="utf-8")
        completed_cfg_ids.add(global_cfg_idx)
        log(f"[worker {worker_id}]   done in {time.time() - t0:.1f}s")

    log(f"[worker {worker_id}] wrote partial results -> {partial_dir}")
    return str(partial_dir)


def _post_process(results_dir: Path, args) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import sparsedeepsurv.summary as sds_summary

    partial_dirs = sorted(results_dir.glob("_partial_worker*"))
    if not partial_dirs:
        raise FileNotFoundError(f"No partial worker dirs found in {results_dir}")

    print(f"[post-process] merging {len(partial_dirs)} partial result dirs", flush=True)
    def _collect_metric_frames(metric_name: str) -> List[pd.DataFrame]:
        frames: List[pd.DataFrame] = []
        for d in partial_dirs:
            cfg_rows_dir = d / "config_rows"
            metric_frames: List[pd.DataFrame] = []
            if cfg_rows_dir.exists():
                for path in sorted(cfg_rows_dir.glob(f"*_{metric_name}.csv")):
                    frame = _read_nonempty_csv(path)
                    if frame is not None:
                        metric_frames.append(frame)
            if not metric_frames:
                frame = _read_nonempty_csv(d / f"{metric_name}.csv")
                if frame is not None:
                    metric_frames.append(frame)
            frames.extend(metric_frames)
        return frames

    run_frames = _collect_metric_frames("run_rows")
    aff_frames = _collect_metric_frames("aff_rows")
    risk_frames = _collect_metric_frames("risk_rows")
    cluster_frames = _collect_metric_frames("cluster_rows")
    hist_frames = _collect_metric_frames("hist_rows")

    run_frames = [f for f in run_frames if f is not None]
    aff_frames = [f for f in aff_frames if f is not None]
    risk_frames = [f for f in risk_frames if f is not None]
    cluster_frames = [f for f in cluster_frames if f is not None]
    hist_frames = [f for f in hist_frames if f is not None]

    if not run_frames:
        raise RuntimeError(
            "No successful runs were written by any worker. "
            "Check worker logs for CUDA / environment launch failures."
        )

    df_runs = pd.concat(run_frames, ignore_index=True)
    df_aff = pd.concat(aff_frames, ignore_index=True) if aff_frames else pd.DataFrame()
    df_risk = pd.concat(risk_frames, ignore_index=True) if risk_frames else pd.DataFrame()
    df_cluster = pd.concat(cluster_frames, ignore_index=True) if cluster_frames else pd.DataFrame()
    df_hist = pd.concat([f for f in hist_frames if not f.empty], ignore_index=True) if hist_frames else pd.DataFrame()

    ref_specs: List[tuple[Path, set[str] | None]] = []
    ref_specs.extend((p, set(args.reference_families) if args.reference_families else None) for p in args.reference_results_dir)
    ref_specs.extend(_parse_reference_specs(args.reference_spec))
    for ref_dir, ref_families in ref_specs:
        ref_dir = ref_dir.resolve()
        ref_frames = _load_reference_metric_frames(ref_dir, args.dataset, ref_families)
        if not ref_frames:
            print(f"[post-process] no reference broad outputs found in {ref_dir}", flush=True)
            continue
        print(
            f"[post-process] merging reference results from {ref_dir}"
            + (f" families={sorted(ref_families)}" if ref_families else ""),
            flush=True,
        )
        if "run_rows" in ref_frames:
            df_runs = pd.concat([df_runs, ref_frames["run_rows"]], ignore_index=True)
        if "aff_rows" in ref_frames:
            df_aff = pd.concat([df_aff, ref_frames["aff_rows"]], ignore_index=True)
        if "risk_rows" in ref_frames:
            df_risk = pd.concat([df_risk, ref_frames["risk_rows"]], ignore_index=True)
        if "cluster_rows" in ref_frames:
            df_cluster = pd.concat([df_cluster, ref_frames["cluster_rows"]], ignore_index=True)
        if "hist_rows" in ref_frames:
            df_hist = pd.concat([df_hist, ref_frames["hist_rows"]], ignore_index=True)

    prefix = _broad_prefix(args.dataset)
    df_runs.to_csv(results_dir / f"{prefix}_runs.csv", index=False)
    df_aff.to_csv(results_dir / f"{prefix}_affinity_pairs.csv", index=False)
    df_risk.to_csv(results_dir / f"{prefix}_risk_pairs.csv", index=False)
    df_cluster.to_csv(results_dir / f"{prefix}_cluster_pairs.csv", index=False)
    df_hist.to_csv(results_dir / f"{prefix}_histology_runs.csv", index=False)

    acc = sds_summary.summarize_runs(df_runs)
    aff = sds_summary.summarize_affinity(df_aff, df_runs)
    risk = sds_summary.summarize_pairwise_metric(
        df_risk,
        df_runs,
        metrics=[
            "risk_corr",
            "top_risk_overlap_ratio",
            "top_risk_jaccard",
            "bottom_risk_overlap_ratio",
            "bottom_risk_jaccard",
        ],
    )
    cluster = sds_summary.summarize_pairwise_metric(df_cluster, df_runs, metrics=["cluster_ari", "cluster_nmi"])

    acc.to_csv(results_dir / f"{prefix}_accuracy_summary.csv", index=False)
    aff.to_csv(results_dir / f"{prefix}_affinity_summary.csv", index=False)
    risk.to_csv(results_dir / f"{prefix}_risk_summary.csv", index=False)
    cluster.to_csv(results_dir / f"{prefix}_cluster_summary.csv", index=False)

    merged = sds_summary.build_merged_summary(acc, aff, risk, cluster)
    merged.to_csv(results_dir / f"{prefix}_merged_summary.csv", index=False)
    top = sds_summary.rank_top_configs(
        merged,
        x_min=float(args.plot_x_min),
        x_max=float(args.plot_x_max),
    )
    top.to_csv(results_dir / f"{prefix}_top_configs_in_band.csv", index=False)
    sds_summary.save_smoothing_legend(
        sorted(acc["lambda_sample_smooth"].unique().tolist()),
        results_dir / "fig_notebook_style_smoothing_legend.png",
    )

    for family in ["LSPIN", "Concrete", "L-LSPIN", "L-Concrete"]:
        for y_col, y_label in [
            ("mean_test_c", "Mean test C-index across seeds"),
            ("mean_manifold_alignment", "Mean alignment to test patient manifold"),
            ("mean_gene_union_count", "Mean union of genes selected across test patients"),
        ]:
            sds_summary.save_family_loess(
                acc,
                family=family,
                y_col=y_col,
                y_label=y_label,
                outpath=results_dir / f"fig_{family.lower().replace('-', '').replace(' ', '_')}_{y_col}_vs_khard.png",
                title=f"{DATASET_SPECS[args.dataset]['dataset_label']} {family} broad: {y_label} vs sparsity",
                x_min=float(args.plot_x_min),
                x_max=float(args.plot_x_max),
                frac=float(args.lowess_frac),
            )

    _save_broad_multiplot_for_families(
        merged,
        dataset_label=DATASET_SPECS[args.dataset]["dataset_label"],
        families=["LSPIN", "Concrete"],
        outpath=results_dir / "fig_broad_multiplot_mlp_predictor_smoothing_vs_khard.png",
        x_min=float(args.plot_x_min),
        x_max=float(args.plot_x_max),
        frac=float(args.lowess_frac),
    )
    _save_broad_multiplot_for_families(
        merged,
        dataset_label=DATASET_SPECS[args.dataset]["dataset_label"],
        families=["L-LSPIN", "L-Concrete"],
        outpath=results_dir / "fig_broad_multiplot_linear_predictor_smoothing_vs_khard.png",
        x_min=float(args.plot_x_min),
        x_max=float(args.plot_x_max),
        frac=float(args.lowess_frac),
    )
    print(f"[post-process] done -> {results_dir}", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gentle broad gated sweep for KIPAN / BRCA")
    p.add_argument("--dataset", choices=["kipan", "brca"], required=True)
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--families", nargs="+", default=None, choices=["LSPIN", "Concrete", "L-LSPIN", "L-Concrete"])
    p.add_argument(
        "--smooth-lambda-scale-families",
        nargs="+",
        default=[],
        choices=["LSPIN", "Concrete", "L-LSPIN", "L-Concrete"],
        help="Families whose smoothed (lambda_sample_smooth > 0) configs should use scaled-up lambda_sparse.",
    )
    p.add_argument(
        "--smooth-lambda-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to lambda_sparse only for smoothed configs in --smooth-lambda-scale-families.",
    )
    p.add_argument(
        "--sample-smooth-grid",
        type=float,
        nargs="+",
        default=None,
        help="Optional override for sample smoothing values. 0.0 is auto-included for no-smooth.",
    )
    p.add_argument(
        "--lspin-nosmooth-lambdas",
        type=float,
        nargs="+",
        default=None,
        help="Optional override lambda grid for LSPIN at lambda_sample_smooth=0.",
    )
    p.add_argument(
        "--lspin-smooth-lambdas",
        type=float,
        nargs="+",
        default=None,
        help="Optional override lambda grid for LSPIN at lambda_sample_smooth>0. "
             "If omitted while --lspin-nosmooth-lambdas is set, the no-smooth grid is reused for smooth points.",
    )
    p.add_argument(
        "--lspin-sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Optional override sigma grid for LSPIN/L-LSPIN families.",
    )
    p.add_argument(
        "--nosmooth-lambdas",
        type=float,
        nargs="+",
        default=None,
        help="Optional override lambda grid at lambda_sample_smooth=0 for all selected families.",
    )
    p.add_argument(
        "--smooth-lambdas",
        type=float,
        nargs="+",
        default=None,
        help="Optional override lambda grid at lambda_sample_smooth>0 for all selected families. "
             "If omitted with --nosmooth-lambdas, the no-smooth grid is reused.",
    )
    p.add_argument(
        "--patience-override",
        type=int,
        default=None,
        help="Optional early-stopping patience override applied to selected families/configs.",
    )
    p.add_argument(
        "--lr-override",
        type=float,
        default=None,
        help="Optional learning-rate override applied to selected families/configs.",
    )
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 2, 4, 6])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-reps", type=int, default=8)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--concrete-mode", choices=["relaxed", "ste"], default="ste")
    p.add_argument("--risk-top-frac", type=float, default=0.2)
    p.add_argument("--global-freq-threshold", type=float, default=0.05)
    p.add_argument("--plot-x-min", type=float, default=None)
    p.add_argument("--plot-x-max", type=float, default=None)
    p.add_argument("--lowess-frac", type=float, default=0.55)
    p.add_argument(
        "--reference-results-dir",
        type=Path,
        nargs="*",
        default=[],
        help="Completed broad run dirs whose CSV outputs should be merged in during post-processing.",
    )
    p.add_argument(
        "--reference-families",
        nargs="+",
        default=None,
        choices=["LSPIN", "Concrete", "L-LSPIN", "L-Concrete"],
        help="Optional family filter applied when merging --reference-results-dir outputs.",
    )
    p.add_argument(
        "--reference-spec",
        action="append",
        default=[],
        help="Per-reference merge spec of the form '/path/to/run::Family1,Family2'. "
             "If no ::family list is given, all families from that reference are merged.",
    )
    p.add_argument("--post-process-only", action="store_true")
    args = p.parse_args()
    spec = DATASET_SPECS[args.dataset]
    if args.outdir is None:
        args.outdir = spec["outdir"]
    if args.results_dir is None:
        args.results_dir = spec["results_dir"]
    if args.plot_x_min is None:
        args.plot_x_min = spec["plot_x_min"]
    if args.plot_x_max is None:
        args.plot_x_max = spec["plot_x_max"]
    return args


class _Tee:
    def __init__(self, log_path: Path):
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


def main() -> None:
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    args = _parse()
    spec = DATASET_SPECS[args.dataset]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(args.results_dir / "run.log")

    if args.post_process_only:
        print(f"[main] --post-process-only on {args.results_dir}", flush=True)
        _post_process(args.results_dir, args)
        return

    all_configs = _build_configs(args)
    cache_kwargs = _prepare_shared_cache(args)
    gpus = args.gpus
    n_workers = len(gpus)
    batches: List[List[Dict]] = [[] for _ in range(n_workers)]
    for i, cfg in enumerate(all_configs):
        batches[i % n_workers].append(cfg)

    print(
        f"[ch3_broad_gentle] dataset={args.dataset} {len(all_configs)} configs -> "
        f"{n_workers} workers on GPUs {gpus}",
        flush=True,
    )
    if args.families:
        print(f"[ch3_broad_gentle] restricted families: {args.families}", flush=True)
    if args.smooth_lambda_scale_families and float(args.smooth_lambda_scale) != 1.0:
        print(
            f"[ch3_broad_gentle] scaling smoothed lambda_sparse by {args.smooth_lambda_scale:g} "
            f"for families {args.smooth_lambda_scale_families}",
            flush=True,
        )
    for i, (gpu, batch) in enumerate(zip(gpus, batches)):
        print(f"  worker {i}: GPU cuda:{gpu}, {len(batch)} configs", flush=True)

    worker_kwargs = dict(
        outdir_str=str(args.outdir.resolve()),
        results_dir_str=str(args.results_dir.resolve()),
        dataset=args.dataset,
        seed=args.seed,
        knn_k=int(spec["knn_k"]),
        n_reps=args.n_reps,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        concrete_mode=args.concrete_mode,
        risk_top_frac=args.risk_top_frac,
        cluster_n_clusters=int(spec["cluster_n_clusters"]),
        global_freq_threshold=args.global_freq_threshold,
        **cache_kwargs,
    )

    t_start = time.time()
    futures = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for worker_id, (gpu, batch) in enumerate(zip(gpus, batches)):
            futures[
                pool.submit(
                    _worker,
                    worker_id,
                    batch,
                    device=f"cuda:{gpu}",
                    **worker_kwargs,
                )
            ] = worker_id
        for fut in as_completed(futures):
            wid = futures[fut]
            part_dir = fut.result()
            print(f"[main] worker {wid} finished -> {part_dir}", flush=True)

    print(f"[main] training done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    _post_process(args.results_dir, args)


if __name__ == "__main__":
    main()
