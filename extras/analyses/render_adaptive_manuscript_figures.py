#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import ANALYSES_DIR, PAPER_ROOT, ensure_repo_imports

ensure_repo_imports(include_lspin_pytorch=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from matplotlib.patches import Patch
from scipy.stats import ttest_ind

import sparsedeepsurv as sds
from cleaned_analyses.pipelines import repro_survival_pipeline as rp
from cleaned_analyses import rerender_patient_smoothing_boxplots as boxplots


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render notebook-style selected figures for adaptive runs.")
    p.add_argument("--dataset", choices=["kipan", "brca", "pancan"], required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--lspin-temperature", type=float, default=0.5)
    p.add_argument("--concrete-temperature", type=float, default=0.3)
    p.add_argument("--heatmap-top-genes", type=int, default=180)
    p.add_argument("--heatmap-min-frac-on", type=float, default=0.05)
    p.add_argument("--heatmap-max-frac-on", type=float, default=0.95)
    p.add_argument("--skip-heatmaps", action="store_true")
    return p.parse_args()


def _norm_family(x: str) -> str:
    x = str(x)
    if x in {"HardSigmoid", "LSPIN"}:
        return "LSPIN"
    return x


def _model_label(family: str, selection: str) -> str:
    if family == "LSPIN":
        return "LSPIN (no smooth)" if selection == "nosmooth" else "LSPIN + patient smooth"
    if family == "L-LSPIN":
        return "L-LSPIN (no smooth)" if selection == "nosmooth" else "L-LSPIN + patient smooth"
    if family == "Concrete":
        return "Concrete (no smooth)" if selection == "nosmooth" else "Concrete + patient smooth"
    if family == "L-Concrete":
        return "L-Concrete (no smooth)" if selection == "nosmooth" else "L-Concrete + patient smooth"
    raise ValueError(f"Unexpected family: {family}")


def _predictor_group(family: str) -> str:
    return "linear" if family in {"L-LSPIN", "L-Concrete"} else "mlp"


def _mean_ci(series: pd.Series) -> Tuple[float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(vals) == 0:
        return float("nan"), float("nan")
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, 0.0
    ci = float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))
    return mean, ci


def _load_dataset(dataset: str, outdir: Path) -> Dict[str, np.ndarray]:
    if dataset == "kipan":
        return sds.load_kipan_split_artifacts(outdir)
    if dataset == "brca":
        return sds.load_brca_split_artifacts(outdir)
    if dataset == "pancan":
        return sds.load_pancan_split_artifacts(outdir)
    raise ValueError(dataset)


def _infer_global_cfg_idx(dataset: str, row: pd.Series) -> Optional[int]:
    if dataset == "kipan":
        from ch3_kipan_adaptive_v2 import _build_configs
        args = SimpleNamespace(
            lspin_sigmas=[0.22, 0.20],
            llspin_sigmas=[0.22, 0.20],
            concrete_sigmas=[0.15, 0.12],
            lconcrete_sigmas=[0.15, 0.12],
            lspin_lambdas=[0.014, 0.0105, 0.00875, 0.007, 0.00525],
            llspin_lambdas=[0.021, 0.014, 0.0105, 0.00875, 0.007],
            concrete_lambdas=[0.004, 0.003, 0.0025, 0.002, 0.0015],
            lconcrete_lambdas=[0.004, 0.003, 0.0025, 0.002, 0.0015],
            gated_hidden_dim=64,
            risk_hidden_dims=[64, 32],
            risk_dropout_p=0.1,
            gate_hidden_dropout_p=0.0,
            gate_weight_decay=0.0,
            lspin_init_bias=0.0,
            lspin_temperature=0.5,
            concrete_temperature=0.3,
            lspin_patience=12,
            llspin_patience=20,
            concrete_patience=20,
            lconcrete_patience=20,
            sample_smooth_grid=[0.0, 0.025, 0.05, 0.1],
        )
    elif dataset == "brca":
        from ch3_brca_adaptive_v2 import _build_configs
        args = SimpleNamespace(
            lspin_sigmas=[0.15, 0.12],
            llspin_sigmas=[0.15, 0.12],
            concrete_sigmas=[0.15, 0.12],
            lconcrete_sigmas=[0.15, 0.12],
            lspin_lambdas=[0.014, 0.0105, 0.00875, 0.007, 0.00525],
            llspin_lambdas=[0.021, 0.014, 0.0105, 0.00875, 0.007],
            concrete_lambdas=[0.0055, 0.0044, 0.0033, 0.00275, 0.0022],
            lconcrete_lambdas=[0.0066, 0.0044, 0.0033, 0.00275, 0.00165],
            gated_hidden_dim=64,
            risk_hidden_dims=[64, 32],
            risk_dropout_p=0.1,
            gate_hidden_dropout_p=0.0,
            gate_weight_decay=0.0,
            lspin_init_bias=0.0,
            lspin_temperature=0.5,
            concrete_temperature=0.3,
            lspin_patience=12,
            llspin_patience=35,
            concrete_patience=20,
            lconcrete_patience=20,
            sample_smooth_grid=[0.0, 0.1, 0.2, 0.4],
        )
    elif dataset == "pancan":
        from ch3_pancan_adaptive_v2 import _build_configs
        args = SimpleNamespace(
            lspin_sigmas=[0.20, 0.17],
            llspin_sigmas=[0.20, 0.17],
            concrete_sigmas=[0.15, 0.12],
            lconcrete_sigmas=[0.15, 0.12],
            lspin_lambdas=[0.0012, 0.0009, 0.0007, 0.0006, 0.0005],
            llspin_lambdas=[0.0025, 0.0018, 0.0014, 0.0011, 0.0009],
            concrete_lambdas=[0.002, 0.0017, 0.0014, 0.0012, 0.0010],
            lconcrete_lambdas=[0.0030, 0.0023, 0.0018, 0.0014, 0.0012],
            gated_hidden_dim=64,
            risk_hidden_dims=[64, 32],
            risk_dropout_p=0.1,
            gate_hidden_dropout_p=0.0,
            gate_weight_decay=0.0,
            lspin_init_bias=0.0,
            lspin_temperature=0.5,
            concrete_temperature=0.3,
            lspin_patience=12,
            llspin_patience=20,
            concrete_patience=20,
            lconcrete_patience=20,
            sample_smooth_grid=[0.0, 0.05, 0.1, 0.15],
        )
    else:
        return None
    configs = _build_configs(args)
    family = str(row["family"])
    gate_type = "lspin_tf" if "LSPIN" in family else "concrete"
    for cfg in configs:
        if (
            cfg["family_label"] == family
            and cfg["gate_type"] == gate_type
            and np.isclose(float(cfg["gate_sigma"]), float(row["gate_sigma"]))
            and np.isclose(float(cfg["lam"]), float(row["lambda_sparse"]))
            and np.isclose(float(cfg["smooth"]), float(row["lambda_sample_smooth"]))
        ):
            return int(cfg["global_cfg_idx"])
    return None


def _prepare_split(data: Dict[str, np.ndarray], *, seed: int, knn_k: int):
    X_train = data["X_train"]
    X_test = data["X_test"]
    strat_key = np.array([f"{e}_{h}" for e, h in zip(data["event_train"], data["histo_train"])])
    ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    i_tr, i_val = next(ss.split(X_train, strat_key))

    Xt_tr = sds.as_torch(X_train[i_tr])
    Xt_val = sds.as_torch(X_train[i_val])
    Xt_test = sds.as_torch(X_test)
    tt_tr = sds.as_torch(data["time_train"][i_tr])
    tt_val = sds.as_torch(data["time_train"][i_val])
    tt_test = sds.as_torch(data["time_test"])
    et_tr = sds.as_torch(data["event_train"][i_tr])
    et_val = sds.as_torch(data["event_train"][i_val])
    et_test = sds.as_torch(data["event_test"])
    A_train = sds.build_knn_adjacency_csr(
        Xt_tr, k=knn_k, pca_dim=50, metric="cosine", symmetrize=True
    )
    return {
        "Xt_tr": Xt_tr,
        "Xt_val": Xt_val,
        "Xt_test": Xt_test,
        "tt_tr": tt_tr,
        "tt_val": tt_val,
        "tt_test": tt_test,
        "et_tr": et_tr,
        "et_val": et_val,
        "et_test": et_test,
        "A_train": A_train,
        "input_dim": X_train.shape[1],
    }


def _choose_representative_seed(df_runs: pd.DataFrame, *, family: str, gate_sigma: float, lam: float, smooth: float) -> int:
    sub = df_runs[
        (df_runs["model_family"] == family)
        & np.isclose(df_runs["gate_sigma"], gate_sigma)
        & np.isclose(df_runs["lambda_sparse"], lam)
        & np.isclose(df_runs["lambda_sample_smooth"], smooth)
    ].copy()
    if sub.empty:
        raise ValueError(f"No runs found for family={family}, sigma={gate_sigma}, lambda={lam}, smooth={smooth}")
    cols = ["mean_Khard", "gene_union_count", "test_cindex"]
    means = sub[cols].mean()
    stds = sub[cols].std(ddof=0).replace(0, 1.0)
    z = ((sub[cols] - means) / stds).pow(2).sum(axis=1)
    return int(sub.loc[z.idxmin(), "seed"])


def _family_model_kwargs(dataset: str, family: str, row: Optional[pd.Series] = None) -> Dict[str, object]:
    predictor = "linear" if family in {"L-LSPIN", "L-Concrete"} else "mlp"
    gate_type = "lspin_tf" if "LSPIN" in family else "concrete"
    base = {
        "predictor": predictor,
        "gate_type": gate_type,
        "gating_hidden_dim": 64,
        "gate_hidden_dropout_p": 0.0,
        "risk_hidden_dims": (64, 32) if predictor == "mlp" else (),
        "risk_dropout_p": 0.1 if predictor == "mlp" else 0.0,
        "lspin_init_bias": 0.0,
        "gate_weight_decay": 0.0,
        "patience": 20,
        "temperature": 0.5 if gate_type == "lspin_tf" else 0.3,
    }
    if family == "LSPIN":
        base["patience"] = 12
    elif family == "L-LSPIN":
        base["patience"] = 35 if dataset == "brca" else 20
    elif family in {"Concrete", "L-Concrete"}:
        base["patience"] = 20
    if row is not None:
        for key in [
            "predictor",
            "gating_hidden_dim",
            "gate_hidden_dropout_p",
            "risk_dropout_p",
            "lspin_init_bias",
            "gate_weight_decay",
            "temperature",
            "patience",
        ]:
            if key in row and pd.notna(row[key]):
                base[key] = row[key]
        if "risk_hidden_dims" in row and pd.notna(row["risk_hidden_dims"]):
            val = row["risk_hidden_dims"]
            if isinstance(val, str):
                val = val.strip()
                if val in {"()", "[]", ""}:
                    base["risk_hidden_dims"] = ()
                else:
                    base["risk_hidden_dims"] = tuple(int(x) for x in ast.literal_eval(val))
            else:
                base["risk_hidden_dims"] = tuple(val)
    base["gating_hidden_dim"] = int(base["gating_hidden_dim"])
    base["gate_hidden_dropout_p"] = float(base["gate_hidden_dropout_p"])
    base["risk_dropout_p"] = float(base["risk_dropout_p"])
    base["lspin_init_bias"] = float(base["lspin_init_bias"])
    base["gate_weight_decay"] = float(base["gate_weight_decay"])
    base["temperature"] = float(base["temperature"])
    base["patience"] = int(base["patience"])
    base["risk_hidden_dims"] = tuple(int(x) for x in base["risk_hidden_dims"])
    return base


def _build_selected_metric_table(selected_cfg: pd.DataFrame, df_runs: pd.DataFrame, df_aff: pd.DataFrame, df_risk: pd.DataFrame, df_cluster: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, sel in selected_cfg.iterrows():
        family = str(sel["family"])
        sigma = float(sel["gate_sigma"])
        lam = float(sel["lambda_sparse"])
        smooth = float(sel["lambda_sample_smooth"])
        model = str(sel["model"])
        run_sub = df_runs[
            (df_runs["model_family"] == family)
            & np.isclose(df_runs["gate_sigma"], sigma)
            & np.isclose(df_runs["lambda_sparse"], lam)
            & np.isclose(df_runs["lambda_sample_smooth"], smooth)
        ]
        aff_sub = df_aff[
            (df_aff["model_family"] == family)
            & np.isclose(df_aff["gate_sigma"], sigma)
            & np.isclose(df_aff["lambda_sparse"], lam)
            & np.isclose(df_aff["lambda_sample_smooth"], smooth)
        ]
        risk_sub = df_risk[
            (df_risk["model_family"] == family)
            & np.isclose(df_risk["gate_sigma"], sigma)
            & np.isclose(df_risk["lambda_sparse"], lam)
            & np.isclose(df_risk["lambda_sample_smooth"], smooth)
        ]
        cluster_sub = df_cluster[
            (df_cluster["model_family"] == family)
            & np.isclose(df_cluster["gate_sigma"], sigma)
            & np.isclose(df_cluster["lambda_sparse"], lam)
            & np.isclose(df_cluster["lambda_sample_smooth"], smooth)
        ]
        rows.append(
            {
                "family": family,
                "selection": str(sel["selection"]),
                "model": model,
                "mean_Khard": _mean_ci(run_sub["mean_Khard"])[0],
                "ci_Khard": _mean_ci(run_sub["mean_Khard"])[1],
                "mean_khard_over_union": _mean_ci(run_sub["mean_Khard"] / run_sub["gene_union_count"].clip(lower=1e-8))[0],
                "ci_khard_over_union": _mean_ci(run_sub["mean_Khard"] / run_sub["gene_union_count"].clip(lower=1e-8))[1],
                "mean_test_c": _mean_ci(run_sub["test_cindex"])[0],
                "ci_test_c": _mean_ci(run_sub["test_cindex"])[1],
                "mean_manifold_alignment": _mean_ci(run_sub["manifold_alignment"])[0],
                "ci_manifold_alignment": _mean_ci(run_sub["manifold_alignment"])[1],
                "mean_gene_union_count": _mean_ci(run_sub["gene_union_count"])[0],
                "ci_gene_union_count": _mean_ci(run_sub["gene_union_count"])[1],
                "mean_gene_freq_ge_threshold_count": _mean_ci(run_sub["gene_freq_ge_threshold_count"])[0],
                "ci_gene_freq_ge_threshold_count": _mean_ci(run_sub["gene_freq_ge_threshold_count"])[1],
                "mean_effective_gene_count": _mean_ci(run_sub["effective_gene_count"])[0],
                "ci_effective_gene_count": _mean_ci(run_sub["effective_gene_count"])[1],
                "mean_affinity_corr": _mean_ci(aff_sub["affinity_corr"])[0],
                "ci_affinity_corr": _mean_ci(aff_sub["affinity_corr"])[1],
                "mean_risk_corr": _mean_ci(risk_sub["risk_corr"])[0],
                "ci_risk_corr": _mean_ci(risk_sub["risk_corr"])[1],
                "mean_top_risk_overlap_ratio": _mean_ci(risk_sub["top_risk_overlap_ratio"])[0],
                "ci_top_risk_overlap_ratio": _mean_ci(risk_sub["top_risk_overlap_ratio"])[1],
                "mean_cluster_ari": _mean_ci(cluster_sub["cluster_ari"])[0],
                "ci_cluster_ari": _mean_ci(cluster_sub["cluster_ari"])[1],
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        model_order = [
            "LSPIN (no smooth)",
            "LSPIN + patient smooth",
            "Concrete (no smooth)",
            "Concrete + patient smooth",
            "L-LSPIN (no smooth)",
            "L-LSPIN + patient smooth",
            "L-Concrete (no smooth)",
            "L-Concrete + patient smooth",
        ]
        out["model"] = pd.Categorical(out["model"], categories=model_order, ordered=True)
        out = out.sort_values("model").reset_index(drop=True)
    return out


def _pick_matched_comparison_configs(results_dir: Path, *, required_n_runs: int = 12) -> pd.DataFrame:
    acc = pd.read_csv(results_dir / "accuracy_summary.csv")
    acc["model_family"] = acc["model_family"].map(_norm_family)
    acc = acc[acc["n_runs"].astype(int) >= int(required_n_runs)].copy()

    rows = []
    for family in sorted(acc["model_family"].dropna().unique()):
        fam = acc[acc["model_family"] == family].copy()
        no_df = fam[np.isclose(fam["lambda_sample_smooth"], 0.0)].copy()
        sm_df = fam[fam["lambda_sample_smooth"] > 0.0].copy()
        if no_df.empty or sm_df.empty:
            continue

        pair_candidates = []
        for _, sm in sm_df.iterrows():
            cand = no_df.copy()
            cand["khard_gap_abs"] = (cand["mean_Khard"] - float(sm["mean_Khard"])).abs()
            cand["khard_gap_rel"] = cand["khard_gap_abs"] / cand["mean_Khard"].clip(lower=1e-8)
            cand["sigma_gap"] = (cand["gate_sigma"] - float(sm["gate_sigma"])).abs()
            no = cand.sort_values(
                ["khard_gap_rel", "khard_gap_abs", "sigma_gap", "mean_test_c"],
                ascending=[True, True, True, False],
            ).iloc[0]
            c_drop = float(no["mean_test_c"] - sm["mean_test_c"])
            pair_candidates.append(
                {
                    "no": no,
                    "sm": sm,
                    "khard_gap_rel": float(no["khard_gap_rel"]),
                    "khard_gap_abs": float(no["khard_gap_abs"]),
                    "sigma_gap": float(no["sigma_gap"]),
                    "c_drop": c_drop,
                }
            )

        if not pair_candidates:
            continue

        best = min(
            pair_candidates,
            key=lambda p: (
                p["khard_gap_rel"],
                p["khard_gap_abs"],
                p["sigma_gap"],
                -float((float(p["no"]["mean_test_c"]) + float(p["sm"]["mean_test_c"])) / 2.0),
                -float(p["no"]["mean_test_c"]),
            ),
        )
        no = best["no"]
        sm = best["sm"]
        for selection, row in [("nosmooth", no), ("smooth", sm)]:
            rr = dict(row)
            rows.append(
                {
                    "family": family,
                    "gate_sigma": float(rr["gate_sigma"]),
                    "selection": selection,
                    **rr,
                    "model": _model_label(family, selection),
                }
            )
    return pd.DataFrame(rows)


def _write_notebook_style_files(results_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(results_dir / "all_runs_raw.csv")
    aff = pd.read_csv(results_dir / "all_affinity_pairs.csv")
    risk = pd.read_csv(results_dir / "all_risk_pairs.csv")
    cluster = pd.read_csv(results_dir / "all_cluster_pairs.csv")
    selected = pd.read_csv(results_dir / "selected_showcase_configs.csv")

    for df in [runs, aff, risk, cluster, selected]:
        df["model_family"] = df["model_family"].map(_norm_family)
    selected["family"] = selected["family"].map(_norm_family)

    runs.to_csv(results_dir / "notebook_style_models_runs.csv", index=False)
    aff.to_csv(results_dir / "notebook_style_models_affinity_pairs.csv", index=False)
    risk.to_csv(results_dir / "notebook_style_models_risk_pairs.csv", index=False)
    cluster.to_csv(results_dir / "notebook_style_models_cluster_pairs.csv", index=False)

    acc = pd.read_csv(results_dir / "accuracy_summary.csv")
    aff_sum = pd.read_csv(results_dir / "affinity_summary.csv")
    risk_sum = pd.read_csv(results_dir / "risk_summary.csv")
    cluster_sum = pd.read_csv(results_dir / "cluster_summary.csv")
    for df in [acc, aff_sum, risk_sum, cluster_sum]:
        df["model_family"] = df["model_family"].map(_norm_family)
    acc.to_csv(results_dir / "notebook_style_models_accuracy_summary.csv", index=False)
    aff_sum.to_csv(results_dir / "notebook_style_models_affinity_summary.csv", index=False)
    risk_sum.to_csv(results_dir / "notebook_style_models_risk_summary.csv", index=False)
    cluster_sum.to_csv(results_dir / "notebook_style_models_cluster_summary.csv", index=False)

    selected_cfg = _pick_matched_comparison_configs(results_dir)
    if selected_cfg.empty:
        selected_cfg = selected.copy()
        selected_cfg["model"] = [
            _model_label(str(fam), str(sel))
            for fam, sel in zip(selected_cfg["family"], selected_cfg["selection"])
        ]
    selected_cfg.to_csv(results_dir / "selected_comparison_configs.csv", index=False)

    selected_metric_table = _build_selected_metric_table(selected_cfg, runs, aff, risk, cluster)
    selected_metric_table.to_csv(results_dir / "selected_comparison_metrics_summary.csv", index=False)
    return runs, aff, risk, cluster, selected_cfg


def _retrain_selected_model(
    *,
    dataset: str,
    row: pd.Series,
    split: Dict[str, object],
    seed: int,
    device: str,
    args: argparse.Namespace,
):
    family = str(row["family"])
    model_kwargs = _family_model_kwargs(dataset, family, row)
    gate_type = str(model_kwargs["gate_type"])
    temperature = float(model_kwargs["temperature"])
    concrete_mode = str(getattr(args, "concrete_mode", "ste"))

    sds.set_all_seeds(seed)
    model, info = sds.run_one_model(
        split["Xt_tr"],
        split["tt_tr"],
        split["et_tr"],
        split["Xt_val"],
        split["tt_val"],
        split["et_val"],
        split["Xt_test"],
        split["tt_test"],
        split["et_test"],
        input_dim=int(split["input_dim"]),
        gate_type=gate_type,
        gate_sigma=float(row["gate_sigma"]),
        lam=float(row["lambda_sparse"]),
        lambda_sample_smooth=float(row["lambda_sample_smooth"]),
        patience=int(model_kwargs["patience"]),
        A_sample_train=split["A_train"],
        device=device,
        lr=float(args.lr),
        temperature=temperature,
        concrete_mode=(concrete_mode if gate_type == "concrete" else "relaxed"),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        max_epochs=int(args.max_epochs),
        predictor=str(model_kwargs["predictor"]),
        gating_hidden_dim=int(model_kwargs["gating_hidden_dim"]),
        gate_hidden_dropout_p=float(model_kwargs["gate_hidden_dropout_p"]),
        risk_hidden_dims=tuple(model_kwargs["risk_hidden_dims"]),
        risk_dropout_p=float(model_kwargs["risk_dropout_p"]),
        lspin_init_bias=float(model_kwargs["lspin_init_bias"]),
        gate_weight_decay=float(model_kwargs["gate_weight_decay"]),
    )
    return model, info


def _load_phase1_model_if_available(
    *,
    dataset: str,
    results_dir: Path,
    row: pd.Series,
    runs: pd.DataFrame,
    input_dim: int,
    device: str,
    args: argparse.Namespace,
):
    family = str(row["family"])
    concrete_mode = str(getattr(args, "concrete_mode", "ste"))

    sub = runs[
        (runs["model_family"] == family)
        & np.isclose(runs["gate_sigma"], float(row["gate_sigma"]))
        & np.isclose(runs["lambda_sparse"], float(row["lambda_sparse"]))
        & np.isclose(runs["lambda_sample_smooth"], float(row["lambda_sample_smooth"]))
        & (runs["phase"] == 1)
    ].copy()
    if sub.empty:
        return None, None

    global_cfg_idx = None
    if "global_cfg_idx" in sub.columns:
        global_cfg_idx = int(sub.iloc[0]["global_cfg_idx"])
    else:
        global_cfg_idx = _infer_global_cfg_idx(dataset, row)
    if global_cfg_idx is None:
        return None, None

    cols = ["mean_Khard", "gene_union_count", "test_cindex"]
    means = sub[cols].mean()
    stds = sub[cols].std(ddof=0).replace(0, 1.0)
    z = ((sub[cols] - means) / stds).pow(2).sum(axis=1)
    best = sub.loc[z.idxmin()]
    sd_path = None
    for part_dir in sorted(results_dir.glob("_partial_p1_worker*")):
        cand = part_dir / "state_dicts" / f"cfg{int(global_cfg_idx)}_rep{int(best['rep_id'])}.pt"
        if cand.exists():
            sd_path = cand
            break
    if sd_path is None:
        return None, None

    model_kwargs = _family_model_kwargs(dataset, family, sub.iloc[0])

    model = sds.make_model(
        input_dim=int(input_dim),
        gate_type=str(model_kwargs["gate_type"]),
        gate_sigma=float(row["gate_sigma"]),
        temperature=float(model_kwargs["temperature"]),
        concrete_mode=(concrete_mode if str(model_kwargs["gate_type"]) == "concrete" else "relaxed"),
        predictor=str(model_kwargs["predictor"]),
        gating_hidden_dim=int(model_kwargs["gating_hidden_dim"]),
        gate_hidden_dropout_p=float(model_kwargs["gate_hidden_dropout_p"]),
        risk_hidden_dims=tuple(model_kwargs["risk_hidden_dims"]),
        risk_dropout_p=float(model_kwargs["risk_dropout_p"]),
        lspin_init_bias=float(model_kwargs["lspin_init_bias"]),
    ).to(device)
    sd = __import__("torch").load(sd_path, map_location=device)
    model.load_state_dict(sd)
    info = {
        "seed": int(best["seed"]),
        "state_dict_path": str(sd_path),
        "test_cindex": float(best["test_cindex"]),
        "best_val_cindex": float(best["best_val_cindex"]),
    }
    return model, info


def _save_heatmaps(
    *,
    dataset: str,
    results_dir: Path,
    outdir: Path,
    selected_cfg: pd.DataFrame,
    runs: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    data = _load_dataset(dataset, outdir)
    split = _prepare_split(data, seed=int(args.seed), knn_k=int(args.knn_k))
    device = sds.resolve_device(str(getattr(args, "device", "auto")))

    heatmap_rows = []
    stem_family = {
        "LSPIN": "lspin",
        "L-LSPIN": "llspin",
        "Concrete": "concrete",
        "L-Concrete": "lconcrete",
    }
    for _, row in selected_cfg.iterrows():
        family = str(row["family"])
        selection = str(row["selection"])
        model, info = _load_phase1_model_if_available(
            dataset=dataset,
            results_dir=results_dir,
            row=row,
            runs=runs,
            input_dim=int(split["input_dim"]),
            device=device,
            args=args,
        )
        if model is None:
            seed = _choose_representative_seed(
                runs,
                family=family,
                gate_sigma=float(row["gate_sigma"]),
                lam=float(row["lambda_sparse"]),
                smooth=float(row["lambda_sample_smooth"]),
            )
            model, info = _retrain_selected_model(
                dataset=dataset,
                row=row,
                split=split,
                seed=seed,
                device=device,
                args=args,
            )
        else:
            seed = int(info["seed"])
        stem = f"{stem_family[family]}_{selection}"
        cluster = rp.save_fig_gate_heatmap_with_histo(
            model,
            Xt=split["Xt_test"],
            histo=data["histo_test"],
            gene_names=data["gene_names"],
            device=device,
            outpath=results_dir / f"fig_gate_heatmap_{stem}.png",
            title=(
                f"{family} {'no smooth' if selection == 'nosmooth' else 'patient smooth'} heatmap\n"
                f"meanK={float(row['mean_Khard']):.0f}, union={float(row['mean_gene_union_count']):.0f}, C={float(row['mean_test_c']):.3f}"
            ),
            view="hard",
            min_frac_on=float(args.heatmap_min_frac_on),
            max_frac_on=float(args.heatmap_max_frac_on),
            top_genes=int(args.heatmap_top_genes),
        )
        if cluster is not None:
            cluster.to_csv(results_dir / f"heatmap_sample_order_{stem}.csv", index=False)
        heatmap_rows.append(
            {
                "family": family,
                "selection": selection,
                "model": _model_label(family, selection),
                "lambda_sparse": float(row["lambda_sparse"]),
                "lambda_sample_smooth": float(row["lambda_sample_smooth"]),
                "gate_sigma": float(row["gate_sigma"]),
                "representative_seed": int(seed),
                "test_cindex": float(info.get("test_cindex", np.nan)),
                "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
            }
        )
    pd.DataFrame(heatmap_rows).to_csv(results_dir / "heatmap_models_summary.csv", index=False)
    selected_cfg[
        [
            "family",
            "selection",
            "gate_sigma",
            "lambda_sparse",
            "lambda_sample_smooth",
            "mean_Khard",
            "mean_gene_union_count",
            "mean_test_c",
        ]
    ].to_csv(results_dir / "selected_best_heatmap_configs.csv", index=False)


def _selected_metric_df(results_dir: Path, metric: str, *, families: list[str], aggregate_pairs: bool = False) -> pd.DataFrame:
    cfg = pd.read_csv(results_dir / "selected_comparison_configs.csv")
    cfg = cfg[cfg["family"].isin(families)].copy()
    runs = pd.read_csv(results_dir / "notebook_style_models_runs.csv")
    runs["khard_over_union"] = runs["mean_Khard"] / runs["gene_union_count"].clip(lower=1e-8)
    pair_map = {
        "affinity_corr": pd.read_csv(results_dir / "notebook_style_models_affinity_pairs.csv"),
        "risk_corr": pd.read_csv(results_dir / "notebook_style_models_risk_pairs.csv"),
        "cluster_ari": pd.read_csv(results_dir / "notebook_style_models_cluster_pairs.csv"),
    }
    source = pair_map.get(metric, runs)
    rows = []
    for _, row in cfg.iterrows():
        sub = source[
            (source["model_family"] == row["family"])
            & np.isclose(source["lambda_sparse"], row["lambda_sparse"])
            & np.isclose(source["lambda_sample_smooth"], row["lambda_sample_smooth"])
        ].copy()
        if sub.empty:
            continue
        if aggregate_pairs and {"run_i", "run_j"}.issubset(sub.columns):
            left = sub[["run_i", metric]].rename(columns={"run_i": "run_id"})
            right = sub[["run_j", metric]].rename(columns={"run_j": "run_id"})
            sub = (
                pd.concat([left, right], ignore_index=True)
                .groupby("run_id", as_index=False)[metric]
                .mean()
            )
        sub["family"] = row["family"]
        sub["selection"] = row["selection"]
        sub["model"] = row["model"]
        sub["value"] = sub[metric]
        rows.append(sub[["family", "selection", "model", "value"]])
    if not rows:
        return pd.DataFrame(columns=["family", "selection", "model", "value"])
    return pd.concat(rows, ignore_index=True)


def _add_predictor_boxplot(ax, metric_df: pd.DataFrame, families: list[str], title: str) -> None:
    x = np.arange(len(families))
    positions = []
    data_groups = []
    colors = []
    color_map = {
        ("LSPIN", "nosmooth"): "#9ec1e6",
        ("LSPIN", "smooth"): "#1f5a99",
        ("L-LSPIN", "nosmooth"): "#9ec1e6",
        ("L-LSPIN", "smooth"): "#1f5a99",
        ("Concrete", "nosmooth"): "#a7d99b",
        ("Concrete", "smooth"): "#2f7d32",
        ("L-Concrete", "nosmooth"): "#a7d99b",
        ("L-Concrete", "smooth"): "#2f7d32",
    }
    labels = {
        "LSPIN": "LSPIN",
        "L-LSPIN": "L-LSPIN",
        "Concrete": "Concrete",
        "L-Concrete": "L-Concrete",
    }
    for fam_idx, fam in enumerate(families):
        for sel, offset in [("nosmooth", -0.18), ("smooth", 0.18)]:
            sub = metric_df[(metric_df["family"] == fam) & (metric_df["selection"] == sel)]["value"].dropna()
            if sub.empty:
                continue
            positions.append(fam_idx + offset)
            data_groups.append(sub.to_numpy())
            colors.append(color_map[(fam, sel)])
    if data_groups:
        boxplots._draw_colored_boxplots(ax, data_groups, positions, colors, width=0.30)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[f] for f in families])
    ax.grid(axis="y", alpha=0.22)


def _annotate_predictor_significance(ax, metric_df: pd.DataFrame, families: list[str]) -> None:
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    extra_top = 0.18 * yr if yr > 0 else 0.1
    ax.set_ylim(y0, y1 + extra_top)
    y0, y1 = ax.get_ylim()
    yr = y1 - y0
    top = y1 - 0.06 * yr
    step = 0.10 * yr
    for fam_idx, fam in enumerate(families):
        no = metric_df[(metric_df["family"] == fam) & (metric_df["selection"] == "nosmooth")]["value"].dropna().to_numpy()
        sm = metric_df[(metric_df["family"] == fam) & (metric_df["selection"] == "smooth")]["value"].dropna().to_numpy()
        if len(no) < 2 or len(sm) < 2:
            continue
        p = float(ttest_ind(no, sm, equal_var=False, nan_policy="omit").pvalue)
        if not np.isfinite(p):
            label = "n/a"
        elif p < 0.001:
            label = "***"
        elif p < 0.01:
            label = "**"
        elif p < 0.05:
            label = "*"
        else:
            label = "ns"
        x1, x2 = fam_idx - 0.18, fam_idx + 0.18
        y = top - fam_idx * step
        ax.plot([x1, x1, x2, x2], [y - 0.018 * yr, y, y, y - 0.018 * yr], color="#333333", lw=1.0, clip_on=False)
        ax.text((x1 + x2) / 2.0, y + 0.010 * yr, label, ha="center", va="bottom", fontsize=10, clip_on=False)


def _save_predictor_split_boxplots(results_dir: Path, dataset_label: str) -> None:
    metric_specs = [
        ("affinity_corr", "Affinity reproducibility", True),
        ("test_cindex", "Test C-index", False),
        ("cluster_ari", "Cluster stability (ARI)", True),
        ("khard_over_union", "Khard / Union", False),
    ]
    predictor_specs = [
        ("mlp", ["LSPIN", "Concrete"], "fig_selected_stability_metrics_with_cindex_boxplot_mlp_predictor.png"),
        ("linear", ["L-LSPIN", "L-Concrete"], "fig_selected_stability_metrics_with_cindex_boxplot_linear_predictor.png"),
    ]
    for predictor_label, families, outname in predictor_specs:
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
        for ax, (metric, title, aggregate_pairs) in zip(axes.ravel(), metric_specs):
            metric_df = _selected_metric_df(
                results_dir,
                metric,
                families=families,
                aggregate_pairs=aggregate_pairs,
            )
            _add_predictor_boxplot(ax, metric_df, families, title)
            if not metric_df.empty:
                _annotate_predictor_significance(ax, metric_df, families)
            if metric == "test_cindex":
                ymin, ymax = ax.get_ylim()
                pad = (ymax - ymin) * 0.08 if np.isfinite(ymin) and np.isfinite(ymax) else 0.02
                ax.set_ylim(max(0.5, ymin - pad), ymax + pad)
                ax.axhline(0.5, ls="--", lw=1, color="gray", alpha=0.8)
        if predictor_label == "mlp":
            handles = [
                Patch(facecolor="#9ec1e6", edgecolor="#444444", label="LSPIN: no smooth"),
                Patch(facecolor="#1f5a99", edgecolor="#444444", label="LSPIN: patient smooth"),
                Patch(facecolor="#a7d99b", edgecolor="#444444", label="Concrete: no smooth"),
                Patch(facecolor="#2f7d32", edgecolor="#444444", label="Concrete: patient smooth"),
            ]
            fig.suptitle(f"{dataset_label}: selected MLP-predictor adaptive configs", y=0.985, fontsize=14)
        else:
            handles = [
                Patch(facecolor="#9ec1e6", edgecolor="#444444", label="L-LSPIN: no smooth"),
                Patch(facecolor="#1f5a99", edgecolor="#444444", label="L-LSPIN: patient smooth"),
                Patch(facecolor="#a7d99b", edgecolor="#444444", label="L-Concrete: no smooth"),
                Patch(facecolor="#2f7d32", edgecolor="#444444", label="L-Concrete: patient smooth"),
            ]
            fig.suptitle(f"{dataset_label}: selected linear-predictor adaptive configs", y=0.985, fontsize=14)
        fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.83))
        fig.text(
            0.5,
            0.915,
            "Matched smooth vs no-smooth selections within each gate family, with Khard matching prioritized before performance.",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.84])
        fig.savefig(results_dir / outname, dpi=180, bbox_inches="tight")
        plt.close(fig)


def render(results_dir: Path, outdir: Path, dataset: str, args: argparse.Namespace) -> None:
    runs, aff, risk, cluster, selected_cfg = _write_notebook_style_files(results_dir)
    dataset_label = dataset.upper() if dataset != "pancan" else "PanCan"
    boxplots.save_selected_metric_boxplots(results_dir, dataset_label, "selected_stability")
    boxplots.save_selected_metric_boxplots(results_dir, dataset_label, "selected_stability_cindex")
    _save_predictor_split_boxplots(results_dir, dataset_label)
    if not bool(getattr(args, "skip_heatmaps", False)):
        _save_heatmaps(
            dataset=dataset,
            results_dir=results_dir,
            outdir=outdir,
            selected_cfg=selected_cfg,
            runs=runs,
            args=args,
        )


def main() -> None:
    args = _parse_args()
    render(args.results_dir.resolve(), args.outdir.resolve(), args.dataset, args)


if __name__ == "__main__":
    main()
