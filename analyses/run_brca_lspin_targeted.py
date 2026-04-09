#!/usr/bin/env python3
"""
Targeted BRCA reruns for LSPIN/L-LSPIN sanity checks.

Two presets are supported:
  - strict_floor:
      Re-run BRCA floor variants with a maximally permissive gate setup:
      zero init bias, zero gate sigma, zero gate hidden dropout, zero gate
      weight decay.
  - gentle_actual:
      Re-run selected BRCA "real" LSPIN variants with zero init bias and
      floor-like gate regularization (no gate dropout / no gate weight decay),
      while keeping the original lambda_sparse and gate_sigma values > 0.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PAPER_ROOT = Path(__file__).resolve().parents[1]
VALIDATE_MODELS_PATH = PAPER_ROOT / "analyses" / "validate_models.py"


def _load_validate_models():
    spec = importlib.util.spec_from_file_location("validate_models_targeted", VALIDATE_MODELS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_args(vm, results_dir: Path, device: str, lspin_init_bias: float) -> SimpleNamespace:
    return SimpleNamespace(
        mode="goal1",
        datasets=["brca"],
        results_dir=results_dir,
        devices=[device],
        n_seeds=5,
        seed_base=123,
        n_eval_splits=0,
        n_reps_per_eval_split=0,
        eval_train_seed_base=100123,
        batch_size=128,
        max_epochs=300,
        patience=60,
        gated_lr=1e-2,
        weight_decay=1e-5,
        lspin_temperature=0.5,
        lspin_init_bias=float(lspin_init_bias),
        concrete_temperature=0.3,
        concrete_mode="ste",
        global_freq_threshold=0.05,
        risk_hidden=[64, 32],
        risk_dropout=0.1,
        mlp_hidden=[64, 32],
        mlp_dropout=0.1,
        mlp_lr=1e-3,
        mlp_weight_decay=1e-5,
        mlp_l1=3e-3,
        mlp_max_epochs=300,
        mlp_patience=25,
        stg_hidden_dim=64,
        stg_hidden_dims=[64, 32],
        gating_hidden_dim=32,
        stg_lr=1e-2,
        stg_sigma=0.5,
        stg_a=1.0,
        stg_init_alpha=0.0,
        stg_lambda_base=0.001,
        stg_lambda_multipliers=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
        linear_gated_lambda_multipliers=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0],
        rsf_n_estimators=200,
        rsf_min_samples_leaf=15,
        rsf_max_features="sqrt",
        lspin_smooth_lambda_scale=1.0,
        concrete_smooth_lambda_scale=1.0,
        linear_cox_l1=None,
        kipan_showcase=vm.KIPAN_SHOWCASE_DEFAULT,
        brca_showcase=vm.BRCA_SHOWCASE_DEFAULT,
        kipan_data=vm.KIPAN_DATA_DEFAULT,
        brca_data=vm.BRCA_DATA_DEFAULT,
        goal0_lambda_multipliers=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
        goal1_lambda_multipliers=[1.0],
        smoke_test=False,
    )


def _build_tasks(vm, args: SimpleNamespace) -> list[dict]:
    vm.DATASET_DEFAULTS["brca"]["showcase_path"] = Path(args.brca_showcase).resolve()
    vm.DATASET_DEFAULTS["brca"]["data_path"] = Path(args.brca_data).resolve()
    variants_by_dataset = {
        "brca": vm._build_variants(
            "brca",
            Path(vm.DATASET_DEFAULTS["brca"]["showcase_path"]),
            args.mode,
            args.goal0_lambda_multipliers,
            args.goal1_lambda_multipliers,
            stg_lambda_base=args.stg_lambda_base,
            stg_lambda_multipliers=args.stg_lambda_multipliers,
            lspin_smooth_lambda_scale=float(args.lspin_smooth_lambda_scale),
            concrete_smooth_lambda_scale=float(args.concrete_smooth_lambda_scale),
            linear_gated_lambda_multipliers=args.linear_gated_lambda_multipliers,
        )
    }
    return vm._make_tasks(args, variants_by_dataset)


def _override_best_cfg(task: dict, *, gate_hidden_dropout_p: float, gate_weight_decay: float) -> dict:
    task = dict(task)
    best_cfg = dict(task.get("_best_cfg", {}))
    best_cfg["gate_hidden_dropout_p"] = float(gate_hidden_dropout_p)
    best_cfg["gate_weight_decay"] = float(gate_weight_decay)
    task["_best_cfg"] = best_cfg
    return task


def _strict_floor_tasks(tasks: list[dict]) -> list[dict]:
    keep = []
    for task in tasks:
        if task["dataset"] != "brca":
            continue
        if task["experiment"] != "goal1" or task.get("goal1_group") != "floor":
            continue
        if task["model_family"] not in {"LSPIN", "L-LSPIN"}:
            continue
        mod = _override_best_cfg(task, gate_hidden_dropout_p=0.0, gate_weight_decay=0.0)
        mod["gate_sigma"] = 0.0
        keep.append(mod)
    return keep


def _gentle_actual_tasks(tasks: list[dict]) -> list[dict]:
    variant_keys = {
        "goal1_nosmooth_lspin_lamx1p0",
        "goal1_smooth_lspin_lamx1p0",
        "goal1_smooth_llspin_lamx0p75",
    }
    keep = []
    for task in tasks:
        if task["dataset"] != "brca":
            continue
        if task["experiment"] != "goal1":
            continue
        if task["variant_key"] not in variant_keys:
            continue
        if task["model_family"] not in {"LSPIN", "L-LSPIN"}:
            continue
        keep.append(_override_best_cfg(task, gate_hidden_dropout_p=0.0, gate_weight_decay=0.0))
    return keep


def _write_outputs(vm, results_dir: Path, partial_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_runs = pd.read_csv(partial_dir / "runs.csv")
    df_hist = (
        pd.read_csv(partial_dir / "histology.csv")
        if (partial_dir / "histology.csv").exists() and (partial_dir / "histology.csv").stat().st_size > 1
        else pd.DataFrame()
    )
    df_aff = vm._compute_affinity_pairs(df_runs) if len(df_runs) else pd.DataFrame()
    df_aff_summary = vm._summarize_affinity(df_aff) if len(df_aff) else pd.DataFrame()
    df_summary = vm._summarize_runs(df_runs) if len(df_runs) else pd.DataFrame()

    df_runs.to_csv(results_dir / "runs.csv", index=False)
    df_hist.to_csv(results_dir / "histology.csv", index=False)
    df_aff.to_csv(results_dir / "affinity_pairs.csv", index=False)
    df_aff_summary.to_csv(results_dir / "affinity_summary.csv", index=False)
    df_summary.to_csv(results_dir / "screen_summary.csv", index=False)
    return df_runs, df_summary


def main() -> None:
    p = argparse.ArgumentParser(description="Targeted BRCA LSPIN/L-LSPIN reruns")
    p.add_argument("--preset", choices=["strict_floor", "gentle_actual"], required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--lspin-init-bias", type=float, default=0.0)
    args_cli = p.parse_args()

    vm = _load_validate_models()
    args = _make_args(vm, args_cli.results_dir, args_cli.device, args_cli.lspin_init_bias)
    tasks = _build_tasks(vm, args)

    if args_cli.preset == "strict_floor":
        tasks = _strict_floor_tasks(tasks)
    else:
        tasks = _gentle_actual_tasks(tasks)

    args_cli.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tasks).drop(columns=["_best_cfg"], errors="ignore").to_csv(
        args_cli.results_dir / "task_manifest.csv",
        index=False,
    )

    print(f"[targeted] preset={args_cli.preset} device={args_cli.device} tasks={len(tasks)}")
    partial_dir = Path(vm._worker(0, tasks, args_cli.device, vars(args), str(args_cli.results_dir)))
    print(f"[targeted] partial_dir={partial_dir}")
    _df_runs, df_summary = _write_outputs(vm, args_cli.results_dir, partial_dir)
    if not df_summary.empty:
        cols = ["dataset", "model_family", "variant_key", "mean_test_cindex", "ci95_test_cindex", "mean_Khard", "n_runs"]
        print(df_summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
