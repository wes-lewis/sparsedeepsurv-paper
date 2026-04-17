#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

FAMILY_ORDER = ["LSPIN", "Concrete"]
COLORS = {
    ("LSPIN", "nosmooth"): "#9ec1e6",
    ("LSPIN", "smooth"): "#1f5a99",
    ("Concrete", "nosmooth"): "#a7d99b",
    ("Concrete", "smooth"): "#2f7d32",
}
BASE_RUN_METRICS = [
    ("test_cindex", "Test C-index", (0.5, None)),
    ("manifold_alignment", "Manifold alignment", (None, None)),
]
PAIR_METRICS = [
    ("affinity_corr", "Affinity reproducibility", (None, None), "notebook_style_models_affinity_pairs.csv"),
    ("risk_corr", "Risk reproducibility", (None, None), "notebook_style_models_risk_pairs.csv"),
    ("cluster_ari", "Cluster stability (ARI)", (None, None), "notebook_style_models_cluster_pairs.csv"),
]
RATIO_METRIC = ("ratio_efficiency", "mean_Khard / union", (None, None))
X_SPECS = {
    "ratio_x": {
        "label": "mean_Khard / union",
        "filename": "fig_observation_binned_ratio_tradeoffs.png",
        "legend": "fig_observation_binned_ratio_legend.png",
        "title": "observation-level stability and performance vs feature-use efficiency",
        "include_ratio_panel": False,
    },
    "khard_x": {
        "label": "mean_Khard",
        "filename": "fig_observation_binned_khard_tradeoffs.png",
        "legend": "fig_observation_binned_khard_legend.png",
        "title": "observation-level metrics vs mean per-patient selected genes",
        "include_ratio_panel": True,
    },
    "union_x": {
        "label": "union",
        "filename": "fig_observation_binned_union_tradeoffs.png",
        "legend": "fig_observation_binned_union_legend.png",
        "title": "observation-level metrics vs global gene union",
        "include_ratio_panel": True,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Observation-level binned smoothing figures")
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--dataset-label", type=str, default=None)
    p.add_argument("--bins", type=int, default=14)
    p.add_argument("--bootstrap", type=int, default=2000)
    return p.parse_args()


def bootstrap_interval(values: np.ndarray, n_boot: int) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(0)
    boot = np.empty(n_boot, dtype=float)
    n = len(vals)
    for i in range(n_boot):
        boot[i] = float(rng.choice(vals, size=n, replace=True).mean())
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, float(lo), float(hi)


def smooth_curve(y: np.ndarray, weights: np.ndarray, window: int = 3) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(y) <= 2:
        return y.copy()
    out = np.empty_like(y)
    half = window // 2
    for i in range(len(y)):
        lo = max(0, i - half)
        hi = min(len(y), i + half + 1)
        out[i] = float(np.average(y[lo:hi], weights=weights[lo:hi]))
    return out


def load_config_axes(results_dir: Path) -> pd.DataFrame:
    acc = pd.read_csv(results_dir / 'notebook_style_models_accuracy_summary.csv')
    acc['ratio_x'] = acc['mean_Khard'] / acc['mean_gene_union_count'].clip(lower=1e-8)
    acc['khard_x'] = acc['mean_Khard']
    acc['union_x'] = acc['mean_gene_union_count']
    acc['smooth_group'] = np.where(np.isclose(acc['lambda_sample_smooth'], 0.0), 'nosmooth', 'smooth')
    return acc[[
        'model_family', 'lambda_sample_smooth', 'lambda_sparse', 'smooth_group',
        'ratio_x', 'khard_x', 'union_x'
    ]]


def merge_run_obs(results_dir: Path, config_axes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    runs = pd.read_csv(results_dir / 'notebook_style_models_runs.csv')
    merged_runs = runs.merge(config_axes, on=['model_family', 'lambda_sample_smooth', 'lambda_sparse'], how='left')
    out = {
        'test_cindex': merged_runs[['model_family', 'smooth_group', 'ratio_x', 'khard_x', 'union_x', 'test_cindex']].rename(columns={'test_cindex': 'value'}),
        'manifold_alignment': merged_runs[['model_family', 'smooth_group', 'ratio_x', 'khard_x', 'union_x', 'manifold_alignment']].rename(columns={'manifold_alignment': 'value'}),
        'ratio_efficiency': merged_runs[['model_family', 'smooth_group', 'ratio_x', 'khard_x', 'union_x']].assign(value=merged_runs['ratio_x'].to_numpy()),
    }
    for col, _, _, fname in PAIR_METRICS:
        pairs = pd.read_csv(results_dir / fname)
        merged = pairs.merge(config_axes, on=['model_family', 'lambda_sample_smooth', 'lambda_sparse'], how='left')
        out[col] = merged[['model_family', 'smooth_group', 'ratio_x', 'khard_x', 'union_x', col]].rename(columns={col: 'value'})
    return out


def summarize(obs: pd.DataFrame, x_key: str, bins: int, bootstrap: int) -> pd.DataFrame:
    rows = []
    for family in FAMILY_ORDER:
        fam = obs[obs['model_family'] == family].copy()
        if fam.empty:
            continue
        unique_x = np.sort(fam[x_key].dropna().unique())
        if len(unique_x) < 2:
            continue
        q = min(bins, max(2, len(unique_x)))
        fam['bin'] = pd.qcut(fam[x_key], q=q, duplicates='drop')
        for smooth_group in ['nosmooth', 'smooth']:
            sub = fam[fam['smooth_group'] == smooth_group]
            if sub.empty:
                continue
            for _, bdf in sub.groupby('bin', observed=False):
                if bdf.empty:
                    continue
                mean, lo, hi = bootstrap_interval(bdf['value'].to_numpy(), bootstrap)
                rows.append({
                    'model_family': family,
                    'smooth_group': smooth_group,
                    'x_mid': float(bdf[x_key].mean()),
                    'n_obs': int(len(bdf)),
                    'mean': mean,
                    'lo': lo,
                    'hi': hi,
                })
    if not rows:
        return pd.DataFrame(columns=['model_family', 'smooth_group', 'x_mid', 'n_obs', 'mean', 'lo', 'hi'])
    return pd.DataFrame(rows).sort_values(['model_family', 'smooth_group', 'x_mid']).reset_index(drop=True)


def save_legend(outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 1.2))
    handles = [
        Line2D([0], [0], color=COLORS[("LSPIN", "nosmooth")], lw=3, label='LSPIN: no smooth'),
        Line2D([0], [0], color=COLORS[("LSPIN", "smooth")], lw=3, label='LSPIN: patient smooth'),
        Line2D([0], [0], color=COLORS[("Concrete", "nosmooth")], lw=3, label='Concrete: no smooth'),
        Line2D([0], [0], color=COLORS[("Concrete", "smooth")], lw=3, label='Concrete: patient smooth'),
    ]
    ax.legend(handles=handles, loc='center', ncol=2, frameon=False)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _dense_kernel_curve(x: np.ndarray, y: np.ndarray, *, n_grid: int = 180) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    if len(x) == 1:
        return x.copy(), y.copy(), y.copy(), y.copy()
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_lo = float(x.min())
    x_hi = float(x.max())
    grid = np.linspace(x_lo, x_hi, n_grid)
    span = max(x_hi - x_lo, 1e-8)
    bw = max(span * 0.16, 0.01)
    mean = np.empty_like(grid)
    lo = np.empty_like(grid)
    hi = np.empty_like(grid)
    for i, gx in enumerate(grid):
        w = np.exp(-0.5 * ((x - gx) / bw) ** 2)
        w_sum = np.clip(w.sum(), 1e-12, None)
        mu = float(np.sum(w * y) / w_sum)
        var = float(np.sum(w * (y - mu) ** 2) / w_sum)
        n_eff = float((w_sum ** 2) / np.clip(np.sum(w ** 2), 1e-12, None))
        spread = np.sqrt(max(var, 0.0)) * (0.75 + 0.25 / np.sqrt(max(n_eff, 1.0)))
        mean[i] = mu
        lo[i] = mu - spread
        hi[i] = mu + spread
    return grid, smooth_curve(mean, np.ones_like(mean), window=7), smooth_curve(lo, np.ones_like(lo), window=7), smooth_curve(hi, np.ones_like(hi), window=7)


def save_figure(obs: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame], outpath: Path, dataset_label: str, x_key: str) -> None:
    spec = X_SPECS[x_key]
    metrics = BASE_RUN_METRICS + [(m, l, y) for m, l, y, _ in PAIR_METRICS]
    if spec['include_ratio_panel']:
        metrics = metrics + [RATIO_METRIC]
    fig, axes = plt.subplots(2, len(metrics), figsize=(3.6 * len(metrics), 7.8), sharex=False)
    if len(metrics) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for r, family in enumerate(FAMILY_ORDER):
        family_x = []
        for metric, _, _ in metrics:
            df = obs.get(metric, pd.DataFrame())
            fam = df[df['model_family'] == family]
            if not fam.empty:
                family_x.extend(fam[x_key].tolist())
        x_lo = min(family_x) * 0.92 if family_x else None
        x_hi = max(family_x) * 1.05 if family_x else None
        for c, (metric, title, ylim) in enumerate(metrics):
            ax = axes[r, c]
            df_obs = obs.get(metric, pd.DataFrame())
            fam_obs = df_obs[df_obs['model_family'] == family]
            df_sum = summaries.get(metric, pd.DataFrame())
            fam_sum = df_sum[df_sum['model_family'] == family]
            for smooth_group in ['nosmooth', 'smooth']:
                sub_obs = fam_obs[fam_obs['smooth_group'] == smooth_group].copy()
                if sub_obs.empty:
                    continue
                color = COLORS[(family, smooth_group)]
                grid, ys, los, his = _dense_kernel_curve(sub_obs[x_key].to_numpy(), sub_obs['value'].to_numpy())
                if len(grid):
                    ax.fill_between(grid, los, his, color=color, alpha=0.16, zorder=1)
                    ax.plot(grid, ys, color=color, lw=2.7, zorder=2)
                sub_sum = fam_sum[fam_sum['smooth_group'] == smooth_group].sort_values('x_mid')
                if not sub_sum.empty:
                    ax.scatter(sub_sum['x_mid'], sub_sum['mean'], s=18, color=color, alpha=0.4, edgecolors='none', zorder=3)
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.text(-0.33, 0.5, family, transform=ax.transAxes, rotation=90, va='center', ha='center', fontsize=11)
            ax.set_ylabel(title)
            ax.set_xlabel(spec['label'])
            ax.grid(alpha=0.25)
            if x_lo is not None and x_hi is not None:
                ax.set_xlim(x_lo, x_hi)
            if ylim[0] is not None or ylim[1] is not None:
                ax.set_ylim(*ylim)
            if metric == 'test_cindex':
                ax.axhline(0.5, ls='--', lw=1, color='gray')
    fig.suptitle(f'{dataset_label}: {spec["title"]}', y=1.02, fontsize=15)
    fig.text(0.5, 0.975, 'Lines show smoothed trends over all runs or within-config seed pairs; points show binned mean anchors.', ha='center', fontsize=10)
    fig.tight_layout(rect=[0.04, 0.05, 1, 0.93])
    fig.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    dataset_label = args.dataset_label or results_dir.name
    config_axes = load_config_axes(results_dir)
    obs = merge_run_obs(results_dir, config_axes)
    save_legend(results_dir / X_SPECS['ratio_x']['legend'])
    save_legend(results_dir / X_SPECS['khard_x']['legend'])
    save_legend(results_dir / X_SPECS['union_x']['legend'])
    for x_key, spec in X_SPECS.items():
        summaries = {}
        for metric, df in obs.items():
            s = summarize(df, x_key, args.bins, args.bootstrap)
            summaries[metric] = s
            s.to_csv(results_dir / f'observation_binned_{x_key}_{metric}_summary.csv', index=False)
        save_figure(obs, summaries, results_dir / spec['filename'], dataset_label, x_key)
    print(f'Saved observation-level binned figures to {results_dir}')

if __name__ == '__main__':
    main()
