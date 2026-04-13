#!/usr/bin/env python3
"""
Chapter 3 KIPAN MLP+L1 reference run.

Trains MLP+L1 (no pruning) using the same data split, seed scheme, and
hyperparameters as the KIPAN targeted run, then saves per-seed C-indexes.
Output is a CSV that can be overlaid on the broad sweep LOESS plots.

Usage:
  conda activate musevo
  cd sparsedeepsurv-paper
  python analyses/ch3_kipan_mlp_reference.py [--device cuda:0] [--results-dir data/runs/ch3_kipan_mlp_reference]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ANALYSES_DIR = Path(__file__).resolve().parent
if str(ANALYSES_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSES_DIR))
from _paths import PAPER_ROOT, PROCESSED_DATASETS, ensure_repo_imports, lspin_pytorch_root

LSPIN_ROOT = lspin_pytorch_root()
if LSPIN_ROOT is None:
    raise FileNotFoundError(
        "Could not locate the optional lspin-pytorch checkout. "
        "Set LSPIN_PYTORCH_ROOT to run analyses/ch3_kipan_mlp_reference.py."
    )
CLEANED = LSPIN_ROOT / "cleaned_analyses"
KIPAN_DATA_DEFAULT = PROCESSED_DATASETS["kipan"]
RESULTS_DEFAULT = PAPER_ROOT / "data" / "runs" / "ch3_kipan_mlp_reference"

ensure_repo_imports(include_analyses_dir=True, include_lspin_pytorch=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KIPAN MLP+L1 reference (paper repo)")
    p.add_argument("--outdir", type=Path, default=KIPAN_DATA_DEFAULT)
    p.add_argument("--results-dir", type=Path, default=RESULTS_DEFAULT)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-reps", type=int, default=12)
    # MLP hyperparameters — match targeted run defaults
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--mlp-dropout", type=float, default=0.1)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-5)
    p.add_argument("--mlp-l1", type=float, default=3e-3)
    p.add_argument("--mlp-max-epochs", type=int, default=300)
    p.add_argument("--mlp-patience", type=int, default=25)
    return p.parse_args()


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a")

    def log(msg: str) -> None:
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()

    return log


def main() -> None:
    args = parse_args()

    # GPU pinning: accept cuda:N, convert to CUDA_VISIBLE_DEVICES
    device_arg = args.device
    if device_arg.startswith("cuda:"):
        gpu_idx = device_arg.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
        device_arg = "cuda"
        print(f"[mlp_reference] set CUDA_VISIBLE_DEVICES={gpu_idx}")

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(results_dir / "run.log")

    import pandas as pd
    import lspin_pytorch as lp
    from cleaned_analyses.pipelines import brca_loyal_pipeline as bl
    from cleaned_analyses.pipelines import repro_survival_pipeline as rp
    from sklearn.model_selection import ShuffleSplit
    from kipan_lasso_vs_lspin_histology import load_kipan_split_artifacts

    device = bl.resolve_device(device_arg)
    bl.set_all_seeds(args.seed)
    log(f"Using device: {device}")

    # ── Load data (same as targeted run) ─────────────────────────────────────
    data = load_kipan_split_artifacts(args.outdir.resolve())
    X_train = data["X_train"]
    X_test = data["X_test"]
    strat_key = np.array([
        f"{e}_{h}" for e, h in zip(data["event_train"], data["histo_train"])
    ])
    ss = ShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    i_tr, i_val = next(ss.split(X_train, strat_key))
    split = {
        "Xt_tr": X_train[i_tr],
        "Xt_val": X_train[i_val],
        "tt_tr": data["time_train"][i_tr],
        "tt_val": data["time_train"][i_val],
        "et_tr": data["event_train"][i_tr],
        "et_val": data["event_train"][i_val],
    }

    Xt_tr = bl.as_torch(split["Xt_tr"])
    Xt_val = bl.as_torch(split["Xt_val"])
    Xt_test = bl.as_torch(X_test)
    tt_tr = bl.as_torch(split["tt_tr"])
    tt_val = bl.as_torch(split["tt_val"])
    tt_test = bl.as_torch(data["time_test"])
    et_tr = bl.as_torch(split["et_tr"])
    et_val = bl.as_torch(split["et_val"])
    et_test = bl.as_torch(data["event_test"])

    input_dim = X_train.shape[1]
    hidden_dims = tuple(args.mlp_hidden)

    # ── Train MLP+L1 over seeds ───────────────────────────────────────────────
    # Seed scheme: seed + 500000 + k, matching the targeted run's mlp_seeds
    mlp_seeds = [args.seed + 500000 + k for k in range(args.n_reps)]
    rows = []
    t0 = time.time()
    for k, seed in enumerate(mlp_seeds):
        log(f"[{k+1}/{args.n_reps}] seed={seed}")
        bl.set_all_seeds(seed)

        mlp = rp.make_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_p=args.mlp_dropout,
        ).to(device)
        info = rp.train_mlp_l1(
            mlp,
            Xt_tr, tt_tr, et_tr,
            Xt_val, tt_val, et_val,
            lr=args.mlp_lr,
            lambda_l1_input=args.mlp_l1,
            weight_decay=args.mlp_weight_decay,
            max_epochs=args.mlp_max_epochs,
            patience=args.mlp_patience,
            device=device,
        )
        c_full, _ = lp.eval_cindex(
            rp.RiskOnlyWrapper(mlp).to(device),
            Xt_test, tt_test, et_test,
            device=device,
        )
        rows.append({
            "seed": seed,
            "model": "MLP + L1",
            "cindex": float(c_full),
            "best_epoch": int(info.get("best_epoch", -1)),
            "best_val_loss": float(info.get("best_val_loss", float("nan"))),
        })
        log(f"  cindex={c_full:.4f}  epochs={info.get('best_epoch', '?')}")

    elapsed = time.time() - t0
    log(f"Done in {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    out_csv = results_dir / "mlp_reference_cindex.csv"
    df.to_csv(out_csv, index=False)
    log(f"Saved {len(df)} rows → {out_csv}")

    mean_c = df["cindex"].mean()
    std_c = df["cindex"].std()
    log(f"MLP+L1 C-index: {mean_c:.4f} ± {std_c:.4f} (n={len(df)})")


if __name__ == "__main__":
    main()
