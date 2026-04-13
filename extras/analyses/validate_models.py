#!/usr/bin/env python3
"""
Validation runner for SparseDeepSurv paper experiments.

Implements two experiment families:
  - Goal 0: gate-type comparison (LSPIN vs Concrete)
  - Goal 1: sparse-vs-MLP comparison on KIPAN and BRCA,
    including an MLP+STG global-selection baseline

The script intentionally uses the source trees directly instead of relying on
editable installs, so each worker repeats the required `sys.path.insert(...)`.
"""
from __future__ import annotations

import argparse
import ast
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import CANONICAL_RUNS, PAPER_ROOT, ensure_repo_imports

ensure_repo_imports()

import sparsedeepsurv as sds

RESULTS_DEFAULT = PAPER_ROOT / "data" / "runs" / "validate_models"
KIPAN_DATA_DEFAULT = PAPER_ROOT / "data" / "processed" / "kipan_20260209_213604"
BRCA_DATA_DEFAULT = PAPER_ROOT / "data" / "processed" / "tcga_brca20260214_001423"
KIPAN_SHOWCASE_DEFAULT = CANONICAL_RUNS["kipan_adaptive_gentle"] / "selected_showcase_configs.csv"
BRCA_SHOWCASE_DEFAULT = CANONICAL_RUNS["brca_adaptive_gentle"] / "selected_showcase_configs.csv"

DATASET_DEFAULTS = {
    "kipan": {
        "data_path": KIPAN_DATA_DEFAULT,
        "showcase_path": KIPAN_SHOWCASE_DEFAULT,
        "knn_k": 8,
        "lspin_main_sigma": 0.22,   # matches KIPAN adaptive run sigma exactly
        "lspin_alt_sigma": 0.22,    # only one sigma in KIPAN showcase; kept for API compat
        "concrete_sigma": 0.15,     # matches KIPAN adaptive run sigma exactly
    },
    "brca": {
        "data_path": BRCA_DATA_DEFAULT,
        "showcase_path": BRCA_SHOWCASE_DEFAULT,
        "knn_k": 5,
        "lspin_main_sigma": 0.10,
        "lspin_alt_sigma": 0.15,
        "concrete_sigma": 0.10,
    },
}

# Per-(dataset, model_family) training hyperparameters derived from tuning sweeps.
# Fields here override the CLI defaults for that family. Missing keys fall back to args.
# Concrete/L-Concrete: from validate_goal1_tuning_20260408_082651
# LSPIN/L-LSPIN: from validate_lspin_patience_20260408_124221
GATED_BEST_CONFIGS: Dict[tuple, Dict] = {
    ("kipan", "Concrete"):   {"gated_lr": 1e-3, "gating_hidden_dim": 64,  "lr_schedule": None,     "temp_schedule": None,     "gate_hidden_dropout_p": 0.0, "patience": 20, "gate_weight_decay": None},
    ("kipan", "L-Concrete"): {"gated_lr": 1e-3, "gating_hidden_dim": 64,  "lr_schedule": None,     "temp_schedule": None,     "gate_hidden_dropout_p": 0.0, "patience": 20, "gate_weight_decay": None},
    ("kipan", "LSPIN"):      {"gated_lr": 2e-3, "gating_hidden_dim": 64,  "lr_schedule": None,     "temp_schedule": None,     "gate_hidden_dropout_p": 0.0, "patience": 12, "gate_weight_decay": 0.0},
    ("kipan", "L-LSPIN"):    {"gated_lr": 2e-3, "gating_hidden_dim": 64,  "lr_schedule": None,     "temp_schedule": None,     "gate_hidden_dropout_p": 0.0, "patience": 12, "gate_weight_decay": 0.0},
    ("brca",  "Concrete"):   {"gated_lr": 1e-3, "gating_hidden_dim": 64,  "lr_schedule": None,     "temp_schedule": None,     "gate_hidden_dropout_p": 0.0, "patience": 20, "gate_weight_decay": None},
    ("brca",  "L-Concrete"): {"gated_lr": 1e-3, "gating_hidden_dim": 64,  "lr_schedule": None,     "temp_schedule": None,     "gate_hidden_dropout_p": 0.0, "patience": 20, "gate_weight_decay": None},
    ("brca",  "LSPIN"):      {"gated_lr": 2e-3, "gating_hidden_dim": 64,  "lr_schedule": None,     "temp_schedule": None,     "gate_hidden_dropout_p": 0.0, "patience": 10, "gate_weight_decay": 0.0},
    ("brca",  "L-LSPIN"):    {"gated_lr": 2e-3, "gating_hidden_dim": 64,  "lr_schedule": None,     "temp_schedule": None,     "gate_hidden_dropout_p": 0.0, "patience": 35, "gate_weight_decay": 0.0},
}

# Per-(dataset, model_family) lambda_sparse anchors derived from tuning sweeps.
# These replace the showcase-CSV lambda_base for the sweep, so multipliers bracket
# a region we have evidence is near-optimal. The sweep+select mechanism is unchanged —
# this just ensures the evaluated range is well-centred on what we learned.
#
# KIPAN LSPIN/L-LSPIN: showcase lambda=0.0016; best from reg sweep=0.007 (4.4x off —
#   max multiplier of 3x would never reach it, so anchor is shifted).
# All Concrete/BRCA LSPIN entries already close to showcase; kept for symmetry.
GATED_LAMBDA_ANCHORS: Dict[tuple, float] = {
    ("kipan", "LSPIN"):      0.007,
    ("kipan", "L-LSPIN"):    0.007,
    # Concrete anchors match showcase closely; include to be explicit
    ("kipan", "Concrete"):   0.002,
    ("kipan", "L-Concrete"): 0.002,
    ("brca",  "LSPIN"):      0.007,
    ("brca",  "L-LSPIN"):    0.007,
    ("brca",  "Concrete"):   0.0022,
    ("brca",  "L-Concrete"): 0.0022,
}

SHOWCASE_FAMILY_ALIASES = {
    "lspin": {"hardsigmoid", "lspin"},
    "hardsigmoid": {"hardsigmoid", "lspin"},
    "concrete": {"concrete"},
}


class _Tee:
    """Write to both stdout and a log file simultaneously."""
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

def _load_dataset(dataset_name: str, data_path: Path) -> Dict[str, np.ndarray]:
    if dataset_name == "kipan":
        return sds.load_kipan_split_artifacts(data_path)
    if dataset_name == "brca":
        return sds.load_brca_split_artifacts(data_path)
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _pick_showcase_row(
    showcase_csv: Path,
    *,
    family: str,
    selection: str,
    sigma_target: float,
) -> Dict[str, object]:
    df = pd.read_csv(showcase_csv)
    family_key = family.lower()
    aliases = SHOWCASE_FAMILY_ALIASES.get(family_key, {family_key})
    sub = df[
        (df["family"].astype(str).str.lower().isin(aliases))
        & (df["selection"].astype(str).str.lower() == selection.lower())
    ].copy()
    if sub.empty:
        raise ValueError(
            f"No showcase rows found in {showcase_csv} for family={family}, selection={selection}"
        )
    sub["sigma_dist"] = (sub["gate_sigma"].astype(float) - float(sigma_target)).abs()
    row = sub.sort_values(["sigma_dist", "mean_test_c"], ascending=[True, False]).iloc[0].to_dict()
    row["sigma_target"] = float(sigma_target)
    row["sigma_match"] = "exact" if abs(float(row["gate_sigma"]) - float(sigma_target)) < 1e-12 else "nearest"
    return row


def _goal0_sweep_variants(
    lspin_no: Dict[str, object],
    concrete_no: Dict[str, object],
    lambda_multipliers: Iterable[float],
) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = []
    families = [
        ("LSPIN", "lspin_tf", float(lspin_no["gate_sigma"]), float(lspin_no["lambda_sparse"])),
        ("Concrete", "concrete", float(concrete_no["gate_sigma"]), float(concrete_no["lambda_sparse"])),
    ]
    for family_label, gate_type, gate_sigma, base_lambda in families:
        for mult in lambda_multipliers:
            lam = float(base_lambda) * float(mult)
            mult_label = str(mult).replace(".", "p")
            variants.append({
                "experiment": "goal0",
                "variant_key": f"goal0_{family_label.lower()}_lamx{mult_label}",
                "variant_label": f"{family_label} x{float(mult):.2g}",
                "model_family": family_label,
                "model_kind": "gated",
                "gate_type": gate_type,
                "gate_sigma": gate_sigma,
                "lambda_sparse": lam,
                "lambda_sample_smooth": 0.0,
                "selection": "goal0_sweep",
                "sigma_match": "exact",
                "lambda_base": float(base_lambda),
                "lambda_multiplier": float(mult),
                "goal0_target_reference": bool(family_label == "LSPIN" and abs(float(mult) - 1.0) < 1e-12),
            })
    return variants


def _goal1_sweep_variants(
    lspin_no: Dict[str, object],
    lspin_sm: Dict[str, object],
    lspin_alt: Dict[str, object],
    concrete_no: Dict[str, object],
    concrete_sm: Dict[str, object],
    lambda_multipliers: Iterable[float],
    *,
    stg_lambda_base: float = None,
    stg_lambda_multipliers: Iterable[float] = None,
    lspin_smooth_lambda_scale: float = 1.0,
    concrete_smooth_lambda_scale: float = 1.0,
    linear_gated_lambda_multipliers: Iterable[float] = (1.0,),
) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = [
        {
            "experiment": "goal1",
            "variant_key": "mlp",
            "variant_label": "MLP",
            "model_family": "MLP",
            "model_kind": "mlp",
            "gate_type": None,
            "gate_sigma": np.nan,
            "lambda_sparse": np.nan,
            "lambda_sample_smooth": np.nan,
            "selection": "reference",
            "sigma_match": "na",
            "lambda_base": np.nan,
            "lambda_multiplier": np.nan,
            "goal0_target_reference": False,
            "goal1_group": "dense",
            "goal1_target_reference": False,
        },
        {
            "experiment": "goal1",
            "variant_key": "goal1_rsf",
            "variant_label": "RSF",
            "model_family": "RSF",
            "model_kind": "rsf",
            "gate_type": None,
            "gate_sigma": np.nan,
            "lambda_sparse": np.nan,
            "lambda_sample_smooth": np.nan,
            "selection": "reference",
            "sigma_match": "na",
            "lambda_base": np.nan,
            "lambda_multiplier": np.nan,
            "goal0_target_reference": False,
            "goal1_group": "dense",
            "goal1_target_reference": False,
        },
    ]

    # MLP+STG sweep — lambda base independent of LSPIN when stg_lambda_base is set
    stg_base = float(stg_lambda_base) if stg_lambda_base is not None else float(lspin_no["lambda_sparse"])
    stg_mults = list(stg_lambda_multipliers) if stg_lambda_multipliers is not None else list(lambda_multipliers)
    for mult in stg_mults:
        lam = stg_base * float(mult)
        mult_label = str(mult).replace(".", "p")
        variants.append({
            "experiment": "goal1",
            "variant_key": f"goal1_nosmooth_stg_lamx{mult_label}",
            "variant_label": f"MLP+STG nosmooth x{float(mult):.2g}",
            "model_family": "MLP+STG",
            "model_kind": "stg",
            "gate_type": "global_stg",
            "gate_sigma": float(lspin_no["gate_sigma"]),
            "lambda_sparse": lam,
            "lambda_sample_smooth": 0.0,
            "selection": "goal1_nosmooth_sweep",
            "sigma_match": "na",
            "lambda_base": stg_base,
            "lambda_multiplier": float(mult),
            "goal0_target_reference": False,
            "goal1_group": "nosmooth",
            "goal1_target_reference": False,
        })

    # Linear Cox (ungated, no hidden layers, L1-penalised)
    variants.append({
        "experiment": "goal1",
        "variant_key": "goal1_linear_cox",
        "variant_label": "Linear Cox",
        "model_family": "LinearCox",
        "model_kind": "linear_cox",
        "gate_type": None,
        "gate_sigma": np.nan,
        "lambda_sparse": np.nan,
        "lambda_sample_smooth": np.nan,
        "selection": "reference",
        "sigma_match": "na",
        "lambda_base": np.nan,
        "lambda_multiplier": np.nan,
        "goal0_target_reference": False,
        "goal1_group": "dense",
        "goal1_target_reference": False,
    })

    # Apply scale to smooth lambdas (allows tightening sparsity for smooth variants
    # when the showcase smooth config doesn't reduce Khard relative to nosmooth)
    lspin_sm_lam = float(lspin_sm["lambda_sparse"]) * float(lspin_smooth_lambda_scale)
    concrete_sm_lam = float(concrete_sm["lambda_sparse"]) * float(concrete_smooth_lambda_scale)

    grouped_families = [
        ("nosmooth", "LSPIN",     "lspin_tf", float(lspin_no["gate_sigma"]),    float(lspin_no["lambda_sparse"]),  0.0,                                            True),
        ("nosmooth", "Concrete",  "concrete",  float(concrete_no["gate_sigma"]), float(concrete_no["lambda_sparse"]), 0.0,                                           False),
        ("smooth",   "LSPIN",     "lspin_tf", float(lspin_sm["gate_sigma"]),    lspin_sm_lam,                       float(lspin_sm["lambda_sample_smooth"]),        True),
        ("smooth",   "Concrete",  "concrete",  float(concrete_sm["gate_sigma"]), concrete_sm_lam,                    float(concrete_sm["lambda_sample_smooth"]),     False),
    ]

    for group_label, family_label, gate_type, gate_sigma, base_lambda, smooth, is_ref in grouped_families:
        for mult in lambda_multipliers:
            lam = float(base_lambda) * float(mult)
            mult_label = str(mult).replace(".", "p")
            variants.append({
                "experiment": "goal1",
                "variant_key": f"goal1_{group_label}_{family_label.lower()}_lamx{mult_label}",
                "variant_label": f"{family_label} {group_label} x{float(mult):.2g}",
                "model_family": family_label,
                "model_kind": "gated",
                "predictor": "mlp",
                "gate_type": gate_type,
                "gate_sigma": gate_sigma,
                "lambda_sparse": lam,
                "lambda_sample_smooth": float(smooth),
                "selection": f"goal1_{group_label}_sweep",
                "sigma_match": "exact",
                "lambda_base": float(base_lambda),
                "lambda_multiplier": float(mult),
                "goal0_target_reference": False,
                "goal1_group": group_label,
                "goal1_target_reference": bool(is_ref and abs(float(mult) - 1.0) < 1e-12),
            })

    variants.append({
        "experiment": "goal1",
        "variant_key": "goal1_lspin_alt_nosmooth",
        "variant_label": f"LSPIN sigma={float(lspin_alt['gate_sigma']):.2f}",
        "model_family": "LSPIN",
        "model_kind": "gated",
        "predictor": "mlp",
        "gate_type": "lspin_tf",
        "gate_sigma": float(lspin_alt["gate_sigma"]),
        "lambda_sparse": float(lspin_alt["lambda_sparse"]),
        "lambda_sample_smooth": float(lspin_alt["lambda_sample_smooth"]),
        "selection": str(lspin_alt["selection"]),
        "sigma_match": str(lspin_alt["sigma_match"]),
        "lambda_base": float(lspin_alt["lambda_sparse"]),
        "lambda_multiplier": 1.0,
        "goal0_target_reference": False,
        "goal1_group": "alt_sigma",
        "goal1_target_reference": False,
    })

    # L-LSPIN smooth: linear predictor + LSPIN gate + patient-similarity smoothing
    # Swept over linear_gated_lambda_multipliers so the best lambda can be selected per dataset.
    lg_mults = list(linear_gated_lambda_multipliers)
    for mult in lg_mults:
        mult_label = str(mult).replace(".", "p")
        lam_ll = lspin_sm_lam * float(mult)
        vkey = "goal1_smooth_llspin" if abs(float(mult) - 1.0) < 1e-12 else f"goal1_smooth_llspin_lamx{mult_label}"
        vlabel = "L-LSPIN smooth" if abs(float(mult) - 1.0) < 1e-12 else f"L-LSPIN smooth x{float(mult):.2g}"
        variants.append({
            "experiment": "goal1",
            "variant_key": vkey,
            "variant_label": vlabel,
            "model_family": "L-LSPIN",
            "model_kind": "gated",
            "predictor": "linear",
            "gate_type": "lspin_tf",
            "gate_sigma": float(lspin_sm["gate_sigma"]),
            "lambda_sparse": lam_ll,
            "lambda_sample_smooth": float(lspin_sm["lambda_sample_smooth"]),
            "selection": "goal1_smooth_sweep",
            "sigma_match": "exact",
            "lambda_base": lspin_sm_lam,
            "lambda_multiplier": float(mult),
            "goal0_target_reference": False,
            "goal1_group": "linear_gated",
            "goal1_target_reference": False,
        })

    # L-Concrete smooth: linear predictor + Concrete gate + patient-similarity smoothing
    for mult in lg_mults:
        mult_label = str(mult).replace(".", "p")
        lam_lc = concrete_sm_lam * float(mult)
        vkey = "goal1_smooth_lconcrete" if abs(float(mult) - 1.0) < 1e-12 else f"goal1_smooth_lconcrete_lamx{mult_label}"
        vlabel = "L-Concrete smooth" if abs(float(mult) - 1.0) < 1e-12 else f"L-Concrete smooth x{float(mult):.2g}"
        variants.append({
            "experiment": "goal1",
            "variant_key": vkey,
            "variant_label": vlabel,
            "model_family": "L-Concrete",
            "model_kind": "gated",
            "predictor": "linear",
            "gate_type": "concrete",
            "gate_sigma": float(concrete_sm["gate_sigma"]),
            "lambda_sparse": lam_lc,
            "lambda_sample_smooth": float(concrete_sm["lambda_sample_smooth"]),
            "selection": "goal1_smooth_sweep",
            "sigma_match": "exact",
            "lambda_base": concrete_sm_lam,
            "lambda_multiplier": float(mult),
            "goal0_target_reference": False,
            "goal1_group": "linear_gated",
            "goal1_target_reference": False,
        })

    # lambda=0 floor variants: gate fully open, should recover MLP / Linear Cox performance.
    # One variant per gate type × predictor combination. Used as sanity-check that the gated
    # architecture matches its ungated baseline when no sparsity pressure is applied.
    floor_variants = [
        ("lspin_tf",  "mlp",    "LSPIN",       float(lspin_no["gate_sigma"])),
        ("concrete",  "mlp",    "Concrete",     float(concrete_no["gate_sigma"])),
        ("lspin_tf",  "linear", "L-LSPIN",      float(lspin_sm["gate_sigma"])),
        ("concrete",  "linear", "L-Concrete",   float(concrete_sm["gate_sigma"])),
    ]
    for gate_type, predictor, family, gate_sigma in floor_variants:
        variants.append({
            "experiment": "goal1",
            "variant_key": f"goal1_floor_{family.lower().replace('-', '')}",
            "variant_label": f"{family} floor (λ=0)",
            "model_family": family,
            "model_kind": "gated",
            "predictor": predictor,
            "gate_type": gate_type,
            "gate_sigma": gate_sigma,
            "lambda_sparse": 0.0,
            "lambda_sample_smooth": 0.0,
            "selection": "floor",
            "sigma_match": "exact",
            "lambda_base": 0.0,
            "lambda_multiplier": 0.0,
            "goal0_target_reference": False,
            "goal1_group": "floor",
            "goal1_target_reference": False,
        })

    return variants


def _build_variants(
    dataset_name: str,
    showcase_csv: Path,
    mode: str,
    goal0_lambda_multipliers: Iterable[float],
    goal1_lambda_multipliers: Iterable[float],
    *,
    stg_lambda_base: float = None,
    stg_lambda_multipliers: Iterable[float] = None,
    lspin_smooth_lambda_scale: float = 1.0,
    concrete_smooth_lambda_scale: float = 1.0,
    linear_gated_lambda_multipliers: Iterable[float] = (1.0,),
) -> List[Dict[str, object]]:
    defaults = DATASET_DEFAULTS[dataset_name]
    lspin_no = _pick_showcase_row(
        showcase_csv, family="LSPIN", selection="nosmooth", sigma_target=defaults["lspin_main_sigma"]
    )
    lspin_sm = _pick_showcase_row(
        showcase_csv, family="LSPIN", selection="smooth", sigma_target=defaults["lspin_main_sigma"]
    )
    lspin_alt = _pick_showcase_row(
        showcase_csv, family="LSPIN", selection="nosmooth", sigma_target=defaults["lspin_alt_sigma"]
    )
    concrete_no = _pick_showcase_row(
        showcase_csv, family="Concrete", selection="nosmooth", sigma_target=defaults["concrete_sigma"]
    )
    concrete_sm = _pick_showcase_row(
        showcase_csv, family="Concrete", selection="smooth", sigma_target=defaults["concrete_sigma"]
    )

    # Override lambda_sparse anchors in showcase rows where our tuning sweeps showed
    # the showcase value is far from the optimal regime.
    def _apply_lambda_anchor(row: Dict, family: str) -> Dict:
        anchor = GATED_LAMBDA_ANCHORS.get((dataset_name, family))
        if anchor is not None:
            row = dict(row)
            row["lambda_sparse"] = float(anchor)
        return row

    lspin_no  = _apply_lambda_anchor(lspin_no,  "LSPIN")
    lspin_sm  = _apply_lambda_anchor(lspin_sm,  "LSPIN")
    lspin_alt = _apply_lambda_anchor(lspin_alt, "LSPIN")
    concrete_no = _apply_lambda_anchor(concrete_no, "Concrete")
    concrete_sm = _apply_lambda_anchor(concrete_sm, "Concrete")

    goal0 = _goal0_sweep_variants(lspin_no, concrete_no, goal0_lambda_multipliers)

    goal1 = _goal1_sweep_variants(
        lspin_no,
        lspin_sm,
        lspin_alt,
        concrete_no,
        concrete_sm,
        goal1_lambda_multipliers,
        stg_lambda_base=stg_lambda_base,
        stg_lambda_multipliers=stg_lambda_multipliers,
        lspin_smooth_lambda_scale=lspin_smooth_lambda_scale,
        concrete_smooth_lambda_scale=concrete_smooth_lambda_scale,
        linear_gated_lambda_multipliers=linear_gated_lambda_multipliers,
    )

    # Inject per-(dataset, model_family) best training configs into each variant
    for v in goal0 + goal1:
        cfg = GATED_BEST_CONFIGS.get((dataset_name, v.get("model_family", "")), {})
        if cfg:
            v["_best_cfg"] = cfg

    if mode == "goal0":
        return goal0
    if mode == "goal1":
        return goal1
    if mode == "all":
        return goal0 + goal1
    raise ValueError(f"Unknown mode: {mode}")


def _make_tasks(args: argparse.Namespace, variants_by_dataset: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for dataset_name in args.datasets:
        for seed_offset in range(args.n_seeds):
            run_seed = int(args.seed_base + seed_offset)
            for variant in variants_by_dataset[dataset_name]:
                task = {
                    "dataset": dataset_name,
                    "data_path": str(DATASET_DEFAULTS[dataset_name]["data_path"]),
                    "knn_k": int(DATASET_DEFAULTS[dataset_name]["knn_k"]),
                    "run_seed": run_seed,
                    "split_seed": run_seed,
                    "train_seed": run_seed,
                    "seed_offset": seed_offset,
                    "task_stage": "screen",
                    "save_init_artifact": True,
                    "load_init_artifact": "",
                    **variant,
                }
                tasks.append(task)
    tasks.sort(key=lambda row: (row["dataset"], row["experiment"], row["variant_key"], row["run_seed"]))
    return tasks


def _split_dataset(data: Dict[str, np.ndarray], split_seed: int, knn_k: int) -> Dict[str, object]:
    from sklearn.model_selection import ShuffleSplit

    X_train = data["X_train"]
    strat_key = np.array([f"{e}_{h}" for e, h in zip(data["event_train"], data["histo_train"])], dtype=object)
    ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=split_seed)
    i_tr, i_val = next(ss.split(X_train, strat_key))

    return {
        "Xt_tr": sds.as_torch(X_train[i_tr]),
        "Xt_val": sds.as_torch(X_train[i_val]),
        "Xt_test": sds.as_torch(data["X_test"]),
        "tt_tr": sds.as_torch(data["time_train"][i_tr]),
        "tt_val": sds.as_torch(data["time_train"][i_val]),
        "tt_test": sds.as_torch(data["time_test"]),
        "et_tr": sds.as_torch(data["event_train"][i_tr]),
        "et_val": sds.as_torch(data["event_train"][i_val]),
        "et_test": sds.as_torch(data["event_test"]),
        "histo_test": data["histo_test"],
        "input_dim": int(X_train.shape[1]),
        "A_train": sds.build_knn_adjacency_csr(
            sds.as_torch(X_train[i_tr]),
            k=int(knn_k),
            pca_dim=50,
            metric="cosine",
            symmetrize=True,
        ),
    }


def _affinity_artifact_path(artifacts_dir: Path, task: Dict[str, object]) -> Path:
    stem = (
        f"{task['dataset']}__{task['experiment']}__{task['task_stage']}"
        f"__{task['variant_key']}__seed{task['run_seed']}"
    )
    return artifacts_dir / f"{stem}_aff.npy"


def _init_artifact_path(artifacts_dir: Path, task: Dict[str, object]) -> Path:
    stem = (
        f"{task['dataset']}__{task['experiment']}__{task['task_stage']}"
        f"__{task['variant_key']}__seed{task['run_seed']}"
    )
    return artifacts_dir / f"{stem}_init.pt"


def _run_mlp_task(task: Dict[str, object], split: Dict[str, object], device: str, args) -> Dict[str, object]:
    import torch

    train_seed = int(task.get("train_seed", task["run_seed"]))
    is_linear_cox = str(task.get("model_kind", "mlp")) == "linear_cox"
    hidden_dims = () if is_linear_cox else tuple(args.mlp_hidden)
    dropout_p = 0.0 if is_linear_cox else float(args.mlp_dropout)
    model = sds.make_seeded_mlp(
        input_dim=int(split["input_dim"]),
        hidden_dims=hidden_dims,
        dropout_p=dropout_p,
        seed=train_seed,
    ).to(device)

    init_artifact = ""
    if str(task.get("load_init_artifact", "")):
        try:
            state = torch.load(str(task["load_init_artifact"]), map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(str(task["load_init_artifact"]), map_location=device)
        model.load_state_dict(state)
        init_artifact = str(task["load_init_artifact"])
    elif bool(task.get("save_init_artifact", False)):
        init_path = _init_artifact_path(Path(task["artifacts_dir"]), task)
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, init_path)
        init_artifact = str(init_path)

    l1 = float(args.linear_cox_l1 if (is_linear_cox and args.linear_cox_l1 is not None) else args.mlp_l1)
    info = sds.train_deepsurv_mlp_l1(
        model,
        split["Xt_tr"], split["tt_tr"], split["et_tr"],
        split["Xt_val"], split["tt_val"], split["et_val"],
        config=sds.MLPTrainConfig(
            lr=float(args.mlp_lr),
            weight_decay=float(args.mlp_weight_decay),
            lambda_l1_input=l1,
            batch_size=int(args.batch_size),
            max_epochs=int(args.mlp_max_epochs),
            patience=int(args.mlp_patience),
        ),
        device=device,
        verbose=False,
    )
    test_cindex, _ = sds.eval_mlp_cindex(
        model,
        split["Xt_test"], split["tt_test"], split["et_test"],
        device=device,
    )
    return {
        "test_cindex": float(test_cindex),
        "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
        "best_epoch": int(info.get("best_epoch", info.get("epochs_ran", -1))),
        "mean_Ksoft": np.nan,
        "mean_Khard": np.nan,
        "gene_union_count": np.nan,
        "gene_freq_ge_threshold_count": np.nan,
        "effective_gene_count": np.nan,
        "affinity_artifact": "",
        "init_artifact": init_artifact,
    }


def _run_stg_task(
    task: Dict[str, object],
    split: Dict[str, object],
    device: str,
    args,
    artifacts_dir: Path,
) -> Dict[str, object]:
    import torch

    stg_hidden_dims = list(args.stg_hidden_dims) if hasattr(args, "stg_hidden_dims") and args.stg_hidden_dims else [int(args.stg_hidden_dim)]
    model = sds.DeepSurvSTG(
        input_dim=int(split["input_dim"]),
        hidden_dim=stg_hidden_dims[0],
        hidden_dims=stg_hidden_dims,
        gate_sigma=float(args.stg_sigma),
        a=float(args.stg_a),
        init_alpha=float(args.stg_init_alpha),
    ).to(device)

    init_artifact = ""
    if str(task.get("load_init_artifact", "")):
        try:
            state = torch.load(str(task["load_init_artifact"]), map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(str(task["load_init_artifact"]), map_location=device)
        model.load_state_dict(state)
        init_artifact = str(task["load_init_artifact"])
    elif bool(task.get("save_init_artifact", False)):
        init_path = _init_artifact_path(Path(task["artifacts_dir"]), task)
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, init_path)
        init_artifact = str(init_path)

    info = sds.train_gated_deepsurv(
        model,
        split["Xt_tr"], split["tt_tr"], split["et_tr"],
        split["Xt_val"], split["tt_val"], split["et_val"],
        split["Xt_test"], split["tt_test"], split["et_test"],
        A_sample_train=None,
        config=sds.GatedTrainConfig(
            lr=float(args.stg_lr),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.batch_size),
            max_epochs=int(args.max_epochs),
            patience=int(args.patience),
            lambda_sparse=float(task["lambda_sparse"]),
            lambda_sample_smooth=0.0,
            lambda_gene_smooth=0.0,
        ),
        device=device,
        verbose=False,
    )
    _, g_det, hard_g, _ = sds.get_gates(model, split["Xt_test"], device=device, hard_threshold=0.5, batch_size=512)
    G_det = g_det.numpy()
    G_hard = hard_g.numpy()
    gpars = sds.global_parsimony_metrics(G_hard, freq_threshold=float(args.global_freq_threshold))
    aff_vec = sds.affinity_upper_vec(sds.gate_affinity_matrix(G_hard, normalize="01", zero_diag=True))
    aff_path = _affinity_artifact_path(artifacts_dir, task)
    np.save(aff_path, aff_vec)

    return {
        "test_cindex": float(info.get("test_cindex", np.nan)),
        "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
        "best_epoch": int(info.get("best_epoch", -1)),
        "mean_Ksoft": float(G_det.sum(axis=1).mean()),
        "mean_Khard": float(G_hard.sum(axis=1).mean()),
        "affinity_artifact": str(aff_path),
        "init_artifact": init_artifact,
        **gpars,
    }


def _run_rsf_task(
    task: Dict[str, object],
    split: Dict[str, object],
    args,
) -> Dict[str, object]:
    from sksurv.ensemble import RandomSurvivalForest

    rng = int(task.get("train_seed", task["run_seed"]))
    X_tr = split["Xt_tr"].cpu().numpy()
    t_tr = split["tt_tr"].cpu().numpy()
    e_tr = split["et_tr"].cpu().numpy().astype(bool)
    X_test = split["Xt_test"].cpu().numpy()
    t_test = split["tt_test"].cpu().numpy()
    e_test = split["et_test"].cpu().numpy().astype(bool)

    y_tr = np.array(
        [(bool(e), float(t)) for e, t in zip(e_tr, t_tr)],
        dtype=[("event", bool), ("time", float)],
    )
    y_test = np.array(
        [(bool(e), float(t)) for e, t in zip(e_test, t_test)],
        dtype=[("event", bool), ("time", float)],
    )

    rsf = RandomSurvivalForest(
        n_estimators=int(args.rsf_n_estimators),
        min_samples_leaf=int(args.rsf_min_samples_leaf),
        max_features=str(args.rsf_max_features),
        random_state=rng,
        n_jobs=1,
    )
    rsf.fit(X_tr, y_tr)
    test_cindex = float(rsf.score(X_test, y_test))

    return {
        "test_cindex": test_cindex,
        "best_val_cindex": float("nan"),
        "best_epoch": -1,
        "mean_Ksoft": float("nan"),
        "mean_Khard": float("nan"),
        "gene_union_count": float("nan"),
        "gene_freq_ge_threshold_count": float("nan"),
        "effective_gene_count": float("nan"),
        "affinity_artifact": "",
        "init_artifact": "",
    }


def _run_gated_task(
    task: Dict[str, object],
    split: Dict[str, object],
    device: str,
    args,
    artifacts_dir: Path,
) -> Dict[str, object]:
    import torch

    best_cfg = task.get("_best_cfg", {})

    temperature = float(args.lspin_temperature)
    if task["gate_type"] == "concrete":
        temperature = float(args.concrete_temperature)

    _predictor = str(task.get("predictor", "mlp"))
    model = sds.make_model(
        input_dim=int(split["input_dim"]),
        gate_type=str(task["gate_type"]),
        gate_sigma=float(task["gate_sigma"]),
        temperature=temperature,
        concrete_mode=(str(args.concrete_mode) if str(task["gate_type"]) == "concrete" else "relaxed"),
        predictor=_predictor,
        gating_hidden_dim=int(best_cfg.get("gating_hidden_dim", getattr(args, "gating_hidden_dim", 128))),
        gate_hidden_dropout_p=float(best_cfg.get("gate_hidden_dropout_p", 0.0)),
        risk_hidden_dims=tuple(getattr(args, "risk_hidden", [64, 32])) if _predictor == "mlp" else (),
        risk_dropout_p=float(getattr(args, "risk_dropout", 0.1)) if _predictor == "mlp" else 0.0,
        lspin_init_bias=float(getattr(args, "lspin_init_bias", -2.0)),
    ).to(device)

    init_artifact = ""
    if str(task.get("load_init_artifact", "")):
        try:
            state = torch.load(str(task["load_init_artifact"]), map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(str(task["load_init_artifact"]), map_location=device)
        model.load_state_dict(state)
        init_artifact = str(task["load_init_artifact"])
    elif bool(task.get("save_init_artifact", False)):
        init_path = _init_artifact_path(Path(task["artifacts_dir"]), task)
        torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, init_path)
        init_artifact = str(init_path)

    info = sds.train_gated_deepsurv(
        model,
        split["Xt_tr"], split["tt_tr"], split["et_tr"],
        split["Xt_val"], split["tt_val"], split["et_val"],
        split["Xt_test"], split["tt_test"], split["et_test"],
        A_sample_train=split["A_train"],
        config=sds.GatedTrainConfig(
            lr=float(best_cfg.get("gated_lr", args.gated_lr)),
            weight_decay=float(args.weight_decay),
            gate_weight_decay=best_cfg.get("gate_weight_decay"),
            batch_size=int(args.batch_size),
            max_epochs=int(args.max_epochs),
            patience=int(best_cfg.get("patience", args.patience)),
            lambda_sparse=float(task["lambda_sparse"]),
            lambda_sample_smooth=float(task["lambda_sample_smooth"]),
            lambda_gene_smooth=0.0,
            lr_schedule=best_cfg.get("lr_schedule"),
            lr_warmup_epochs=10 if best_cfg.get("lr_schedule") == "cosine" else 0,
            temp_schedule=best_cfg.get("temp_schedule"),
            temp_init=0.8 if best_cfg.get("temp_schedule") is not None else None,
            temp_final=0.05 if best_cfg.get("temp_schedule") is not None else None,
        ),
        device=device,
        verbose=False,
    )
    _, g_det, hard_g, _ = sds.get_gates(model, split["Xt_test"], device=device, hard_threshold=0.5, batch_size=512)
    G_det = g_det.numpy()
    G_hard = hard_g.numpy()
    gpars = sds.global_parsimony_metrics(G_hard, freq_threshold=float(args.global_freq_threshold))
    aff_vec = sds.affinity_upper_vec(sds.gate_affinity_matrix(G_hard, normalize="01", zero_diag=True))
    aff_path = _affinity_artifact_path(artifacts_dir, task)
    np.save(aff_path, aff_vec)

    row = {
        "test_cindex": float(info.get("test_cindex", np.nan)),
        "best_val_cindex": float(info.get("best_val_cindex", np.nan)),
        "best_epoch": int(info.get("best_epoch", -1)),
        "mean_Ksoft": float(G_det.sum(axis=1).mean()),
        "mean_Khard": float(G_hard.sum(axis=1).mean()),
        "affinity_artifact": str(aff_path),
        "init_artifact": init_artifact,
        **gpars,
    }

    if task["dataset"] == "kipan":
        d_hist = sds.cindex_by_histology(
            sds.RiskOnlyWrapper(model).to(device),
            split["Xt_test"], split["tt_test"], split["et_test"], split["histo_test"],
            device=device,
        )
        row["_histology_rows"] = d_hist.to_dict("records")

    return row


def _worker(worker_id: int, tasks: List[Dict[str, object]], device: str, args_dict: Dict[str, object], results_dir: str) -> str:
    import sys as _sys

    for _path in [SDS_SRC]:
        if _path not in _sys.path:
            _sys.path.insert(0, _path)

    import sparsedeepsurv as _sds

    global sds
    sds = _sds

    args = argparse.Namespace(**args_dict)
    results_path = Path(results_dir)
    artifacts_dir = results_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stage_label = str(tasks[0].get("task_stage", "screen")) if tasks else "screen"
    partial_dir = results_path / f"_partial_{stage_label}_worker{worker_id}"
    partial_dir.mkdir(parents=True, exist_ok=True)
    worker_log_path = partial_dir / "worker.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with worker_log_path.open("a", buffering=1) as fh:
            fh.write(msg + "\n")

    dataset_cache: Dict[str, Dict[str, np.ndarray]] = {}
    split_cache: Dict[tuple[str, int], Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    hist_rows: List[Dict[str, object]] = []
    log(f"[worker {worker_id}|{stage_label}] start device={device} n_tasks={len(tasks)}")

    for idx, task in enumerate(tasks, start=1):
        task = {**task, "artifacts_dir": str(artifacts_dir)}
        dataset_name = str(task["dataset"])
        if dataset_name not in dataset_cache:
            dataset_cache[dataset_name] = _load_dataset(dataset_name, Path(str(task["data_path"])))

        split_key = (dataset_name, int(task.get("split_seed", task["run_seed"])))
        if split_key not in split_cache:
            split_cache[split_key] = _split_dataset(
                dataset_cache[dataset_name],
                int(task.get("split_seed", task["run_seed"])),
                int(task["knn_k"]),
            )

        split = split_cache[split_key]
        sds.set_all_seeds(int(task.get("train_seed", task["run_seed"])))
        t0 = time.time()
        log(
            f"[worker {worker_id}] [{idx}/{len(tasks)}] "
            f"{dataset_name} {task['experiment']} {task['variant_key']} "
            f"split={task.get('split_seed', task['run_seed'])} "
            f"train={task.get('train_seed', task['run_seed'])} "
            f"device={device}"
        )

        try:
            if task["model_kind"] in ("mlp", "linear_cox"):
                metrics = _run_mlp_task(task, split, device, args)
            elif task["model_kind"] == "stg":
                metrics = _run_stg_task(task, split, device, args, artifacts_dir)
            elif task["model_kind"] == "rsf":
                metrics = _run_rsf_task(task, split, args)
            else:
                metrics = _run_gated_task(task, split, device, args, artifacts_dir)
        except Exception as exc:
            rows.append({
                **task,
                "test_cindex": np.nan,
                "best_val_cindex": np.nan,
                "best_epoch": np.nan,
                "mean_Ksoft": np.nan,
                "mean_Khard": np.nan,
                "gene_union_count": np.nan,
                "gene_freq_ge_threshold_count": np.nan,
                "effective_gene_count": np.nan,
                "affinity_artifact": "",
                "init_artifact": str(task.get("load_init_artifact", "")),
                "status": "failed",
                "error": str(exc),
                "elapsed_sec": float(time.time() - t0),
            })
            log(
                f"[worker {worker_id}|{stage_label}]   FAILED "
                f"{task['dataset']} {task['variant_key']} "
                f"split={task.get('split_seed', task['run_seed'])} "
                f"train={task.get('train_seed', task['run_seed'])} "
                f"elapsed={time.time() - t0:.1f}s error={exc}"
            )
            continue

        row = {
            **task,
            **metrics,
            "status": "ok",
            "error": "",
            "elapsed_sec": float(time.time() - t0),
        }
        hist_recs = row.pop("_histology_rows", [])
        rows.append(row)
        log(
            f"[worker {worker_id}|{stage_label}]   ok "
            f"{task['dataset']} {task['variant_key']} "
            f"test_c={float(row['test_cindex']):.4f} "
            f"val_c={float(row['best_val_cindex']):.4f} "
            f"epoch={int(row['best_epoch'])} "
            f"Ksoft={float(row['mean_Ksoft']):.1f} "
            f"Khard={float(row['mean_Khard']):.1f} "
            f"elapsed={float(row['elapsed_sec']):.1f}s"
        )
        for rec in hist_recs:
            hist_rows.append({
                **task,
                "seed": int(task["run_seed"]),
                **rec,
            })

    pd.DataFrame(rows).to_csv(partial_dir / "runs.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(partial_dir / "histology.csv", index=False)
    log(f"[worker {worker_id}|{stage_label}] wrote → {partial_dir} runs={len(rows)} hist={len(hist_rows)}")
    return str(partial_dir)


def _summarize_runs(df_runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family", "gate_type",
        "lambda_sparse", "lambda_base", "lambda_multiplier", "gate_sigma", "goal1_group",
    ]
    df_runs = df_runs.copy()
    for col in group_cols:
        if col not in df_runs.columns:
            df_runs[col] = np.nan
    out = (
        df_runs[df_runs["status"] == "ok"]
        .groupby(group_cols, dropna=False)
        .agg(
            n_runs=("run_seed", "size"),
            mean_test_cindex=("test_cindex", "mean"),
            std_test_cindex=("test_cindex", "std"),
            mean_best_val_cindex=("best_val_cindex", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            std_best_epoch=("best_epoch", "std"),
            mean_Ksoft=("mean_Ksoft", "mean"),
            std_Ksoft=("mean_Ksoft", "std"),
            mean_Khard=("mean_Khard", "mean"),
            std_Khard=("mean_Khard", "std"),
            mean_gene_union_count=("gene_union_count", "mean"),
            std_gene_union_count=("gene_union_count", "std"),
            mean_effective_gene_count=("effective_gene_count", "mean"),
            std_effective_gene_count=("effective_gene_count", "std"),
        )
        .reset_index()
        .sort_values(["dataset", "experiment", "variant_key"])
    )
    for mean_col, std_col, ci_col in [
        ("mean_test_cindex", "std_test_cindex", "ci95_test_cindex"),
        ("mean_Ksoft", "std_Ksoft", "ci95_Ksoft"),
        ("mean_Khard", "std_Khard", "ci95_Khard"),
        ("mean_gene_union_count", "std_gene_union_count", "ci95_gene_union_count"),
        ("mean_effective_gene_count", "std_effective_gene_count", "ci95_effective_gene_count"),
    ]:
        out[ci_col] = 1.96 * out[std_col].astype(float) / np.sqrt(out["n_runs"].clip(lower=1).astype(float))
    return out


def _summarize_split_blocks(df_runs: pd.DataFrame) -> pd.DataFrame:
    if df_runs.empty:
        return pd.DataFrame()
    cols = [
        "dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family",
        "goal1_group", "split_seed",
    ]
    df_runs = df_runs.copy()
    for col in cols:
        if col not in df_runs.columns:
            df_runs[col] = np.nan
    block = (
        df_runs[df_runs["status"] == "ok"]
        .groupby(cols, dropna=False)
        .agg(
            n_reps=("train_seed", "size"),
            split_mean_test_cindex=("test_cindex", "mean"),
            split_sd_test_cindex=("test_cindex", "std"),
            split_mean_Ksoft=("mean_Ksoft", "mean"),
            split_sd_Ksoft=("mean_Ksoft", "std"),
            split_mean_Khard=("mean_Khard", "mean"),
            split_sd_Khard=("mean_Khard", "std"),
        )
        .reset_index()
    )
    return block


def _summarize_within_between(df_block: pd.DataFrame) -> pd.DataFrame:
    if df_block.empty:
        return pd.DataFrame()
    cols = ["dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family", "goal1_group"]
    out = (
        df_block.groupby(cols, dropna=False)
        .agg(
            n_splits=("split_seed", "size"),
            mean_of_split_means_test_cindex=("split_mean_test_cindex", "mean"),
            between_split_sd_test_cindex=("split_mean_test_cindex", "std"),
            mean_within_split_sd_test_cindex=("split_sd_test_cindex", "mean"),
            mean_of_split_means_Ksoft=("split_mean_Ksoft", "mean"),
            between_split_sd_Ksoft=("split_mean_Ksoft", "std"),
            mean_within_split_sd_Ksoft=("split_sd_Ksoft", "mean"),
            mean_of_split_means_Khard=("split_mean_Khard", "mean"),
            between_split_sd_Khard=("split_mean_Khard", "std"),
            mean_within_split_sd_Khard=("split_sd_Khard", "mean"),
        )
        .reset_index()
    )
    return out


def _summarize_affinity(df_aff: pd.DataFrame) -> pd.DataFrame:
    if df_aff.empty:
        return pd.DataFrame()
    return (
        df_aff.groupby(
            ["dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family", "gate_type"],
            dropna=False,
        )
        .agg(
            mean_affinity_corr=("affinity_corr", "mean"),
            std_affinity_corr=("affinity_corr", "std"),
            n_pairs=("affinity_corr", "size"),
        )
        .reset_index()
    )


def _select_goal0_matched(summary_runs: pd.DataFrame, summary_aff: pd.DataFrame) -> pd.DataFrame:
    if summary_runs.empty:
        return pd.DataFrame()

    goal0 = summary_runs[
        (summary_runs["experiment"] == "goal0")
        & (summary_runs["task_stage"] == "screen")
    ].copy()
    if goal0.empty:
        return pd.DataFrame()

    aff_lookup = (
        summary_aff[
            (summary_aff["experiment"] == "goal0")
            & (summary_aff["task_stage"] == "screen")
        ].copy()
        if not summary_aff.empty else pd.DataFrame()
    )
    rows: List[Dict[str, object]] = []
    family_order = ["LSPIN", "Concrete"]

    for dataset_name, sub in goal0.groupby("dataset", dropna=False):
        ref = sub[
                (sub["model_family"] == "LSPIN")
                & np.isclose(sub["lambda_multiplier"].astype(float), 1.0)
            ].copy()
        if ref.empty:
            ref = sub[sub["model_family"] == "LSPIN"].copy()
        if ref.empty:
            continue
        ref_row = ref.sort_values(["mean_test_cindex"], ascending=[False]).iloc[0]
        target_khard = float(ref_row["mean_Khard"])

        for family in family_order:
            fam = sub[sub["model_family"] == family].copy()
            if fam.empty:
                continue
            fam["khard_delta"] = (fam["mean_Khard"].astype(float) - target_khard).abs()
            fam = fam.sort_values(["khard_delta", "mean_test_cindex"], ascending=[True, False])
            pick = fam.iloc[0].to_dict()
            pick["target_mean_Khard"] = target_khard
            pick["khard_delta"] = float(abs(float(pick["mean_Khard"]) - target_khard))
            pick["goal0_reference_variant_key"] = str(ref_row["variant_key"])
            rows.append(pick)

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected

    if not aff_lookup.empty:
        selected = selected.merge(
            aff_lookup[
                ["dataset", "task_stage", "variant_key", "mean_affinity_corr", "std_affinity_corr", "n_pairs"]
            ],
            on=["dataset", "task_stage", "variant_key"],
            how="left",
        )

    return selected.sort_values(["dataset", "model_family"]).reset_index(drop=True)


def _select_goal1_matched(summary_runs: pd.DataFrame, summary_aff: pd.DataFrame) -> pd.DataFrame:
    if summary_runs.empty:
        return pd.DataFrame()

    goal1 = summary_runs[
        (summary_runs["experiment"] == "goal1")
        & (summary_runs["task_stage"] == "screen")
    ].copy()
    if goal1.empty:
        return pd.DataFrame()

    aff_lookup = (
        summary_aff[
            (summary_aff["experiment"] == "goal1")
            & (summary_aff["task_stage"] == "screen")
        ].copy()
        if not summary_aff.empty else pd.DataFrame()
    )
    rows: List[Dict[str, object]] = []
    family_order = ["LSPIN", "Concrete", "MLP+STG"]

    # Dense reference models (MLP, LinearCox): always passthrough (single variant each).
    passthrough_families = {"MLP", "LinearCox"}
    passthrough = goal1[goal1["model_family"].isin(passthrough_families)].copy()
    for _, row in passthrough.iterrows():
        picked = row.to_dict()
        picked["target_mean_Khard"] = np.nan
        picked["khard_delta"] = np.nan
        picked["goal1_reference_variant_key"] = ""
        rows.append(picked)

    # L-LSPIN, L-Concrete, and MLP+STG: select best by test C-index per dataset.
    # MLP+STG uses global gates (not patient-specific) so Khard-matching against LSPIN
    # is not the right criterion — CI-based selection is used instead.
    for ci_family in ["L-LSPIN", "L-Concrete", "MLP+STG"]:
        ci_all = goal1[goal1["model_family"] == ci_family].copy()
        if ci_all.empty:
            continue
        for dataset_name, ci_ds in ci_all.groupby("dataset", dropna=False):
            if len(ci_ds) == 1:
                picked = ci_ds.iloc[0].to_dict()
            else:
                picked = ci_ds.sort_values("mean_test_cindex", ascending=False).iloc[0].to_dict()
            picked["target_mean_Khard"] = np.nan
            picked["khard_delta"] = np.nan
            picked["goal1_reference_variant_key"] = ""
            rows.append(picked)

    # LSPIN and Concrete: Khard-matched selection (patient-specific gated models)
    khard_family_order = ["LSPIN", "Concrete"]
    for dataset_name, sub_dataset in goal1.groupby("dataset", dropna=False):
        for group_label in ["nosmooth", "smooth"]:
            sub = sub_dataset[sub_dataset["variant_key"].astype(str).str.contains(f"goal1_{group_label}_")].copy()
            if sub.empty:
                continue
            ref = sub[
                (sub["model_family"] == "LSPIN")
                & np.isclose(sub["lambda_multiplier"].astype(float), 1.0)
            ].copy()
            if ref.empty:
                ref = sub[sub["model_family"] == "LSPIN"].copy()
            if ref.empty:
                continue
            ref_row = ref.sort_values(["mean_test_cindex"], ascending=[False]).iloc[0]
            target_khard = float(ref_row["mean_Khard"])

            for family in khard_family_order:
                fam = sub[sub["model_family"] == family].copy()
                if fam.empty:
                    continue
                fam["khard_delta"] = (fam["mean_Khard"].astype(float) - target_khard).abs()
                fam = fam.sort_values(["khard_delta", "mean_test_cindex"], ascending=[True, False])
                pick = fam.iloc[0].to_dict()
                pick["goal1_group"] = group_label
                pick["target_mean_Khard"] = target_khard
                pick["khard_delta"] = float(abs(float(pick["mean_Khard"]) - target_khard))
                pick["goal1_reference_variant_key"] = str(ref_row["variant_key"])
                rows.append(pick)

    alt_rows = goal1[goal1["variant_key"] == "goal1_lspin_alt_nosmooth"].copy()
    for _, row in alt_rows.iterrows():
        picked = row.to_dict()
        picked["target_mean_Khard"] = np.nan
        picked["khard_delta"] = np.nan
        picked["goal1_reference_variant_key"] = ""
        rows.append(picked)

    selected = pd.DataFrame(rows)
    if selected.empty:
        return selected

    if not aff_lookup.empty:
        selected = selected.merge(
            aff_lookup[
                ["dataset", "task_stage", "variant_key", "mean_affinity_corr", "std_affinity_corr", "n_pairs"]
            ],
            on=["dataset", "task_stage", "variant_key"],
            how="left",
        )

    return selected.sort_values(["dataset", "goal1_group", "model_family"]).reset_index(drop=True)


def _compute_affinity_pairs(df_runs: pd.DataFrame) -> pd.DataFrame:
    gated = df_runs[(df_runs["status"] == "ok")].copy()
    if gated.empty:
        return pd.DataFrame()
    gated["affinity_artifact"] = gated["affinity_artifact"].fillna("")
    gated = gated[gated["affinity_artifact"].astype(str).ne("")].copy()
    if gated.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    group_cols = ["dataset", "experiment", "task_stage", "variant_key", "variant_label", "model_family", "gate_type"]
    for keys, sub in gated.groupby(group_cols, dropna=False):
        sub = sub.sort_values("run_seed")
        vecs = []
        seeds = []
        for _, row in sub.iterrows():
            aff_path = Path(str(row["affinity_artifact"]))
            if not aff_path.exists():
                continue
            vecs.append(np.load(aff_path))
            seeds.append(int(row["run_seed"]))
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                rows.append({
                    "dataset": keys[0],
                    "experiment": keys[1],
                    "task_stage": keys[2],
                    "variant_key": keys[3],
                    "variant_label": keys[4],
                    "model_family": keys[5],
                    "gate_type": keys[6],
                    "seed_i": seeds[i],
                    "seed_j": seeds[j],
                    "affinity_corr": sds.affinity_corr_from_vec(vecs[i], vecs[j]),
                })
    return pd.DataFrame(rows)


def _choose_init_run(screen_runs: pd.DataFrame, selected_row: pd.Series) -> pd.Series | None:
    sub = screen_runs[
        (screen_runs["dataset"] == selected_row["dataset"])
        & (screen_runs["experiment"] == selected_row["experiment"])
        & (screen_runs["task_stage"] == "screen")
        & (screen_runs["variant_key"] == selected_row["variant_key"])
        & (screen_runs["status"] == "ok")
        & screen_runs["init_artifact"].astype(str).ne("")
    ].copy()
    if sub.empty:
        return None
    if np.isfinite(selected_row.get("mean_Khard", np.nan)):
        sub["khard_delta"] = (sub["mean_Khard"].astype(float) - float(selected_row["mean_Khard"])).abs()
    else:
        sub["khard_delta"] = 0.0
    sub = sub.sort_values(["khard_delta", "test_cindex"], ascending=[True, False])
    return sub.iloc[0]


def _choose_eval_split_runs(
    screen_runs: pd.DataFrame,
    selected_row: pd.Series,
    n_eval_splits: int,
) -> pd.DataFrame:
    sub = screen_runs[
        (screen_runs["dataset"] == selected_row["dataset"])
        & (screen_runs["experiment"] == selected_row["experiment"])
        & (screen_runs["task_stage"] == "screen")
        & (screen_runs["variant_key"] == selected_row["variant_key"])
        & (screen_runs["status"] == "ok")
        & screen_runs["init_artifact"].astype(str).ne("")
    ].copy()
    if sub.empty:
        return sub
    if np.isfinite(selected_row.get("mean_Khard", np.nan)):
        sub["khard_delta"] = (sub["mean_Khard"].astype(float) - float(selected_row["mean_Khard"])).abs()
    else:
        sub["khard_delta"] = 0.0
    sub = sub.sort_values(["khard_delta", "test_cindex"], ascending=[True, False])
    sub = sub.drop_duplicates(subset=["split_seed"], keep="first")
    return sub.head(int(n_eval_splits)).reset_index(drop=True)


def _recover_best_cfg(init_row: pd.Series) -> Dict:
    raw = init_row.get("_best_cfg", None)
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                return dict(parsed)
        except (ValueError, SyntaxError):
            pass
    return dict(
        GATED_BEST_CONFIGS.get(
            (str(init_row.get("dataset", "")), str(init_row.get("model_family", ""))),
            {},
        )
    )


def _build_fixed_init_tasks(
    args: argparse.Namespace,
    selected_df: pd.DataFrame,
    screen_runs: pd.DataFrame,
) -> List[Dict[str, object]]:
    if selected_df.empty or args.n_reps_per_eval_split <= 0 or args.n_eval_splits <= 0:
        return []
    tasks: List[Dict[str, object]] = []
    seen_keys = set()
    for _, row in selected_df.iterrows():
        key = (row["dataset"], row["experiment"], row["variant_key"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        eval_split_rows = _choose_eval_split_runs(screen_runs, row, args.n_eval_splits)
        if eval_split_rows.empty:
            continue
        dataset_name = str(row["dataset"])
        for split_block_id, (_, init_row) in enumerate(eval_split_rows.iterrows()):
            split_seed = int(init_row["split_seed"])
            best_cfg = _recover_best_cfg(init_row)
            for rep_offset in range(args.n_reps_per_eval_split):
                train_seed = int(args.eval_train_seed_base + split_block_id * 1000 + rep_offset)
                run_seed = train_seed
                tasks.append({
                    "dataset": dataset_name,
                    "data_path": str(DATASET_DEFAULTS[dataset_name]["data_path"]),
                    "knn_k": int(DATASET_DEFAULTS[dataset_name]["knn_k"]),
                    "run_seed": run_seed,
                    "split_seed": split_seed,
                    "train_seed": train_seed,
                    "seed_offset": rep_offset,
                    "split_block_id": split_block_id,
                    "task_stage": "fixed_init_eval",
                    "save_init_artifact": False,
                    "load_init_artifact": str(init_row["init_artifact"]),
                    "fixed_init_source_seed": int(init_row["run_seed"]),
                    "fixed_init_source_variant_key": str(init_row["variant_key"]),
                    "dataset_selected_mean_Khard": float(row.get("mean_Khard", np.nan)),
                    "dataset_target_mean_Khard": float(row.get("target_mean_Khard", np.nan)),
                    "dataset_khard_delta": float(row.get("khard_delta", np.nan)),
                    "dataset_selected_summary_stage": str(row.get("task_stage", "screen")),
                    "dataset_selected_summary_file": f"{row['experiment']}_matched_summary.csv",
                    "goal1_group": row.get("goal1_group", np.nan),
                    "experiment": str(row["experiment"]),
                    "variant_key": str(row["variant_key"]),
                    "variant_label": str(row["variant_label"]),
                    "model_family": str(row["model_family"]),
                    "model_kind": str(init_row["model_kind"]),
                    "predictor": str(init_row.get("predictor", "mlp")),
                    "gate_type": init_row["gate_type"],
                    "gate_sigma": init_row["gate_sigma"],
                    "lambda_sparse": init_row["lambda_sparse"],
                    "lambda_sample_smooth": init_row["lambda_sample_smooth"],
                    "selection": init_row["selection"],
                    "sigma_match": init_row.get("sigma_match", "na"),
                    "lambda_base": init_row.get("lambda_base", np.nan),
                    "lambda_multiplier": init_row.get("lambda_multiplier", np.nan),
                    "goal0_target_reference": init_row.get("goal0_target_reference", False),
                    "_best_cfg": best_cfg,
                })
    tasks.sort(key=lambda row: (row["dataset"], row["experiment"], row["variant_key"], row["run_seed"]))
    return tasks


def _run_task_batch(
    tasks: List[Dict[str, object]],
    args: argparse.Namespace,
    results_dir: Path,
) -> List[Path]:
    if not tasks:
        return []
    shards: List[List[Dict[str, object]]] = [[] for _ in args.devices]
    for idx, task in enumerate(tasks):
        shards[idx % len(args.devices)].append(task)
    stage_label = str(tasks[0].get("task_stage", "screen"))
    print(
        f"[main] dispatching {len(tasks)} {stage_label} tasks across {len(args.devices)} devices: "
        + ", ".join(f"{device}={len(shard)}" for device, shard in zip(args.devices, shards)),
        flush=True,
    )

    partial_dirs: List[Path] = []
    with ProcessPoolExecutor(max_workers=len(args.devices)) as ex:
        futures = {}
        for worker_id, (device, shard) in enumerate(zip(args.devices, shards)):
            if not shard:
                continue
            fut = ex.submit(
                _worker,
                worker_id,
                shard,
                device,
                vars(args),
                str(results_dir),
            )
            futures[fut] = (worker_id, device, len(shard))
        for fut in as_completed(futures):
            worker_id, device, n_tasks = futures[fut]
            try:
                partial_dir = Path(fut.result())
                partial_dirs.append(partial_dir)
                print(
                    f"[main] worker {worker_id} device={device} done "
                    f"tasks={n_tasks} → {partial_dir}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[main] worker {worker_id} device={device} FAILED "
                    f"tasks={n_tasks}: {exc}",
                    flush=True,
                )
    print(f"[main] completed {stage_label} partial_dirs={len(partial_dirs)}", flush=True)
    return partial_dirs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate SparseDeepSurv gate types and sparse-vs-MLP baselines")
    p.add_argument("--mode", choices=["goal0", "goal1", "all"], default="all")
    p.add_argument("--datasets", nargs="+", choices=["kipan", "brca"], default=["kipan", "brca"])
    p.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    p.add_argument("--devices", nargs="+", default=["cuda:0"])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed-base", type=int, default=123)
    p.add_argument("--n-eval-splits", type=int, default=4)
    p.add_argument("--n-reps-per-eval-split", type=int, default=3)
    p.add_argument("--eval-train-seed-base", type=int, default=100123)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--gated-lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--lspin-temperature", type=float, default=0.5)
    p.add_argument(
        "--lspin-init-bias",
        type=float,
        default=0.0,
        help="Initial bias for the final LSPIN gate layer. Default 0.0 keeps the gate near-open at init; "
             "set to -2.0 to reproduce the older TensorFlow-faithful closed-at-init setup.",
    )
    p.add_argument("--concrete-temperature", type=float, default=0.3)
    p.add_argument("--concrete-mode", choices=["relaxed", "ste"], default="ste")
    p.add_argument("--global-freq-threshold", type=float, default=0.05)

    p.add_argument(
        "--risk-hidden", type=int, nargs="+", default=[64, 32],
        help="Hidden layer widths for the gated MLP risk network. Default [64, 32] matches "
             "the standalone MLP baseline for a fair architectural comparison.",
    )
    p.add_argument(
        "--risk-dropout", type=float, default=0.1,
        help="Dropout probability in the gated MLP risk network (default 0.1, matching MLP baseline).",
    )
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--mlp-dropout", type=float, default=0.1)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    p.add_argument("--mlp-l1", type=float, default=3e-3)
    p.add_argument("--mlp-max-epochs", type=int, default=300)
    p.add_argument("--mlp-patience", type=int, default=25)
    p.add_argument("--stg-hidden-dim", type=int, default=64)
    p.add_argument(
        "--stg-hidden-dims",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Hidden layer widths for MLP+STG risk network (overrides --stg-hidden-dim). "
             "Default [64, 32] matches the plain MLP architecture.",
    )
    p.add_argument(
        "--gating-hidden-dim",
        type=int,
        default=128,
        help="Hidden width of the two-layer gating network (input → H → input). "
             "Default 128. Reduce to 32 or 64 when using a linear predictor on high-dimensional "
             "data (e.g. BRCA 24K features) to avoid a massively overparameterized gate.",
    )
    p.add_argument("--stg-lr", type=float, default=1e-2)
    p.add_argument("--stg-sigma", type=float, default=0.5)
    p.add_argument("--stg-a", type=float, default=1.0)
    p.add_argument("--stg-init-alpha", type=float, default=0.0)
    p.add_argument(
        "--stg-lambda-base", type=float, default=None,
        help="Override STG goal-1 lambda base (default: derived from LSPIN nosmooth showcase lambda). "
             "Use a small value (e.g. 0.001) to prevent STG Khard collapse.",
    )
    p.add_argument(
        "--stg-lambda-multipliers",
        type=float,
        nargs="+",
        default=None,
        help="Lambda sweep multipliers for MLP+STG (default: same as --goal1-lambda-multipliers). "
             "Override to explore a different range, e.g. 0.1 0.25 0.5 1.0 2.0 5.0",
    )
    p.add_argument(
        "--linear-gated-lambda-multipliers",
        type=float,
        nargs="+",
        default=None,
        help="Lambda sweep multipliers for L-LSPIN and L-Concrete (default: [1.0], single point). "
             "Set to e.g. 0.1 0.25 0.5 1.0 2.0 5.0 to sweep and select best CI per dataset.",
    )
    p.add_argument("--rsf-n-estimators", type=int, default=200)
    p.add_argument("--rsf-min-samples-leaf", type=int, default=15)
    p.add_argument("--rsf-max-features", type=str, default="sqrt")
    p.add_argument(
        "--lspin-smooth-lambda-scale", type=float, default=1.0,
        help="Scale factor applied to smooth-LSPIN lambda_sparse from the showcase "
             "(e.g. 1.5 to push Khard lower than nosmooth).",
    )
    p.add_argument(
        "--concrete-smooth-lambda-scale", type=float, default=1.0,
        help="Scale factor applied to smooth-Concrete lambda_sparse from the showcase.",
    )
    p.add_argument(
        "--linear-cox-l1", type=float, default=None,
        help="L1 penalty for linear Cox; default: same as --mlp-l1.",
    )

    p.add_argument("--kipan-showcase", type=Path, default=KIPAN_SHOWCASE_DEFAULT)
    p.add_argument("--brca-showcase", type=Path, default=BRCA_SHOWCASE_DEFAULT)
    p.add_argument("--kipan-data", type=Path, default=KIPAN_DATA_DEFAULT)
    p.add_argument("--brca-data", type=Path, default=BRCA_DATA_DEFAULT)
    p.add_argument(
        "--goal0-lambda-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        help="Lambda sweep multipliers for Goal 0 matched-sparsity gate comparison",
    )
    p.add_argument(
        "--goal1-lambda-multipliers",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        help="Lambda sweep multipliers for Goal 1 matched-sparsity sparse-model comparison",
    )
    p.add_argument("--smoke-test", action="store_true", help="Run only one task per dataset for import/CLI validation")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(args.results_dir / "run.log")

    DATASET_DEFAULTS["kipan"]["showcase_path"] = args.kipan_showcase.resolve()
    DATASET_DEFAULTS["brca"]["showcase_path"] = args.brca_showcase.resolve()
    DATASET_DEFAULTS["kipan"]["data_path"] = args.kipan_data.resolve()
    DATASET_DEFAULTS["brca"]["data_path"] = args.brca_data.resolve()

    lg_mults = args.linear_gated_lambda_multipliers if args.linear_gated_lambda_multipliers is not None else [1.0]
    variants_by_dataset = {
        dataset_name: _build_variants(
            dataset_name,
            Path(DATASET_DEFAULTS[dataset_name]["showcase_path"]),
            args.mode,
            args.goal0_lambda_multipliers,
            args.goal1_lambda_multipliers,
            stg_lambda_base=args.stg_lambda_base,
            stg_lambda_multipliers=args.stg_lambda_multipliers,
            lspin_smooth_lambda_scale=float(args.lspin_smooth_lambda_scale),
            concrete_smooth_lambda_scale=float(args.concrete_smooth_lambda_scale),
            linear_gated_lambda_multipliers=lg_mults,
        )
        for dataset_name in args.datasets
    }
    manifest_rows = []
    for dataset_name, variants in variants_by_dataset.items():
        for variant in variants:
            manifest_rows.append({"dataset": dataset_name, **{k: v for k, v in variant.items() if k != "_best_cfg"}})
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(args.results_dir / "variant_manifest.csv", index=False)
    print(f"[main] wrote variant_manifest.csv rows={len(manifest_df)}", flush=True)

    tasks = _make_tasks(args, variants_by_dataset)
    if args.smoke_test:
        kept = []
        seen = set()
        for task in tasks:
            key = (task["dataset"], task["experiment"])
            if key in seen:
                continue
            kept.append(task)
            seen.add(key)
        tasks = kept

    print(f"[main] mode={args.mode} datasets={args.datasets} screen_tasks={len(tasks)}")
    print(f"[main] devices={args.devices}")
    print(f"[main] results={args.results_dir}")

    screen_partial_dirs = sorted(
        partial_dir
        for partial_dir in args.results_dir.glob("_partial_screen_worker*")
        if (partial_dir / "runs.csv").exists()
    )
    if screen_partial_dirs:
        print(
            f"[main] found {len(screen_partial_dirs)} reusable screen partial dirs; reusing screen outputs",
            flush=True,
        )
    else:
        screen_partial_dirs = _run_task_batch(tasks, args, args.results_dir)

    run_frames = []
    hist_frames = []
    for partial_dir in sorted(screen_partial_dirs):
        run_csv = partial_dir / "runs.csv"
        hist_csv = partial_dir / "histology.csv"
        if run_csv.exists():
            run_frames.append(pd.read_csv(run_csv))
        if hist_csv.exists() and hist_csv.stat().st_size > 1:
            try:
                hist_frames.append(pd.read_csv(hist_csv))
            except pd.errors.EmptyDataError:
                pass

    df_screen_runs = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    df_screen_hist = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    df_screen_aff = _compute_affinity_pairs(df_screen_runs) if len(df_screen_runs) else pd.DataFrame()
    df_screen_aff_summary = _summarize_affinity(df_screen_aff) if len(df_screen_aff) else pd.DataFrame()
    df_screen_summary = _summarize_runs(df_screen_runs) if len(df_screen_runs) else pd.DataFrame()
    df_goal0_matched = _select_goal0_matched(df_screen_summary, df_screen_aff_summary) if len(df_screen_summary) else pd.DataFrame()
    df_goal1_matched = _select_goal1_matched(df_screen_summary, df_screen_aff_summary) if len(df_screen_summary) else pd.DataFrame()
    print(
        f"[main] screen complete ok_runs={int((df_screen_runs['status'] == 'ok').sum()) if len(df_screen_runs) else 0} "
        f"goal0_matched={len(df_goal0_matched)} goal1_matched={len(df_goal1_matched)}",
        flush=True,
    )

    fixed_init_tasks = _build_fixed_init_tasks(
        args,
        pd.concat([df_goal0_matched, df_goal1_matched], ignore_index=True) if len(df_goal0_matched) or len(df_goal1_matched) else pd.DataFrame(),
        df_screen_runs,
    )
    if args.smoke_test and fixed_init_tasks:
        kept = []
        seen = set()
        for task in fixed_init_tasks:
            key = (task["dataset"], task["experiment"], task["split_seed"])
            if key in seen:
                continue
            kept.append(task)
            seen.add(key)
        fixed_init_tasks = kept

    print(f"[main] fixed_init_eval_tasks={len(fixed_init_tasks)}")
    fixed_partial_dirs = sorted(
        partial_dir
        for partial_dir in args.results_dir.glob("_partial_fixed_init_eval_worker*")
        if (partial_dir / "runs.csv").exists()
    )
    if fixed_partial_dirs:
        print(
            f"[main] found {len(fixed_partial_dirs)} reusable fixed-init partial dirs; reusing eval outputs",
            flush=True,
        )
    else:
        fixed_partial_dirs = _run_task_batch(fixed_init_tasks, args, args.results_dir)

    all_run_frames = [df_screen_runs] if len(df_screen_runs) else []
    all_hist_frames = [df_screen_hist] if len(df_screen_hist) else []
    for partial_dir in sorted(fixed_partial_dirs):
        run_csv = partial_dir / "runs.csv"
        hist_csv = partial_dir / "histology.csv"
        if run_csv.exists():
            all_run_frames.append(pd.read_csv(run_csv))
        if hist_csv.exists() and hist_csv.stat().st_size > 1:
            try:
                all_hist_frames.append(pd.read_csv(hist_csv))
            except pd.errors.EmptyDataError:
                pass

    df_runs = pd.concat(all_run_frames, ignore_index=True) if all_run_frames else pd.DataFrame()
    df_hist = pd.concat(all_hist_frames, ignore_index=True) if all_hist_frames else pd.DataFrame()
    df_aff = _compute_affinity_pairs(df_runs) if len(df_runs) else pd.DataFrame()
    df_aff_summary = _summarize_affinity(df_aff) if len(df_aff) else pd.DataFrame()
    df_summary = _summarize_runs(df_runs) if len(df_runs) else pd.DataFrame()
    df_split_blocks = _summarize_split_blocks(df_runs) if len(df_runs) else pd.DataFrame()
    df_within_between = _summarize_within_between(df_split_blocks) if len(df_split_blocks) else pd.DataFrame()

    def _stage_summary(experiment: str) -> pd.DataFrame:
        sub = df_summary[
            (df_summary["experiment"] == experiment)
            & (df_summary["task_stage"] == "fixed_init_eval")
        ].copy()
        if sub.empty:
            return sub
        if not df_aff_summary.empty and "experiment" in df_aff_summary.columns:
            aff_sub = df_aff_summary[
                (df_aff_summary["experiment"] == experiment)
                & (df_aff_summary["task_stage"] == "fixed_init_eval")
            ].copy()
        else:
            aff_sub = pd.DataFrame()
        if not aff_sub.empty:
            sub = sub.merge(
                aff_sub[
                    ["dataset", "task_stage", "variant_key", "mean_affinity_corr", "std_affinity_corr", "n_pairs"]
                ],
                on=["dataset", "task_stage", "variant_key"],
                how="left",
            )
        return sub

    df_goal0_fixed = _stage_summary("goal0")
    df_goal1_fixed = _stage_summary("goal1")
    print(
        f"[main] fixed-init complete ok_runs={int(((df_runs['status'] == 'ok') & (df_runs['task_stage'] == 'fixed_init_eval')).sum()) if len(df_runs) else 0} "
        f"split_blocks={len(df_split_blocks)} within_between={len(df_within_between)}",
        flush=True,
    )

    df_runs.to_csv(args.results_dir / "runs.csv", index=False)
    df_hist.to_csv(args.results_dir / "histology.csv", index=False)
    df_aff.to_csv(args.results_dir / "affinity_pairs.csv", index=False)
    df_aff_summary.to_csv(args.results_dir / "affinity_summary.csv", index=False)
    df_summary.to_csv(args.results_dir / "summary.csv", index=False)
    df_split_blocks.to_csv(args.results_dir / "split_block_summary.csv", index=False)
    df_within_between.to_csv(args.results_dir / "within_between_split_summary.csv", index=False)
    df_screen_summary.to_csv(args.results_dir / "screen_summary.csv", index=False)
    df_goal0_matched.to_csv(args.results_dir / "goal0_matched_summary.csv", index=False)
    df_goal1_matched.to_csv(args.results_dir / "goal1_matched_summary.csv", index=False)
    df_goal0_fixed.to_csv(args.results_dir / "goal0_fixed_init_summary.csv", index=False)
    df_goal1_fixed.to_csv(args.results_dir / "goal1_fixed_init_summary.csv", index=False)

    print(
        f"[main] wrote variant_manifest.csv, runs.csv, screen_summary.csv, summary.csv, "
        f"split_block_summary.csv, within_between_split_summary.csv, "
        f"goal0_matched_summary.csv, goal1_matched_summary.csv, "
        f"goal0_fixed_init_summary.csv, goal1_fixed_init_summary.csv to {args.results_dir}"
    )


if __name__ == "__main__":
    main()
