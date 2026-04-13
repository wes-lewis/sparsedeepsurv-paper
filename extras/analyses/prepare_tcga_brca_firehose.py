#!/usr/bin/env python3
"""Prepare the canonical BRCA processed dataset from raw TCGA Firehose inputs.

This script mirrors the preprocessing path used to create the BRCA artifact
consumed by the paper analyses:

1. Load Firehose RNA-seq, GISTIC CNV, and clinical files.
2. Collapse sample-level molecular matrices to one row per patient.
3. Concatenate RNA and CNV features after prefixing their namespaces.
4. Parse overall survival and subtype labels from the clinical table.
5. Drop missing survival times, mean-impute remaining feature NaNs,
   remove duplicate feature names by keeping the highest-variance copy,
   and retain the top-variance feature set.
6. Create the canonical train/test split and train-only standardization.

The output layout matches the processed artifact expected by the analysis
drivers under ``data/processed/tcga_brca20260214_001423``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import PAPER_ROOT, lspin_pytorch_root


DATASET_NAME = "tcga_brca20260214_001423"
LOW_VAR_THRESHOLD = 1e-4
DEFAULT_TOP_P = 24000
DEFAULT_TEST_SIZE = 0.30
DEFAULT_RANDOM_STATE = 42


def _default_firehose_file(filename: str) -> Path | None:
    root = lspin_pytorch_root()
    candidates = []
    if root is not None:
        candidates.append(root)
    candidates.extend(
        [
            PAPER_ROOT.parents[2] / "lspin-pytorch",
            PAPER_ROOT.parent / "lspin-pytorch",
        ]
    )
    for candidate in candidates:
        path = candidate / "data" / "TCGA" / filename
        if path.exists():
            return path
    return None


DEFAULT_RNA_FILE = _default_firehose_file(
    "Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz"
)
DEFAULT_CNV_FILE = _default_firehose_file(
    "Human__TCGA_BRCA__BI__SCNA__SNP_6.0__01_28_2016__BI__Gene__Firehose_GISTIC2.cct.gz"
)
DEFAULT_CLIN_FILE = _default_firehose_file(
    "Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi"
)
DEFAULT_OUTDIR = PAPER_ROOT / "data" / "processed" / DATASET_NAME


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rna-file", type=Path, default=DEFAULT_RNA_FILE)
    p.add_argument("--cnv-file", type=Path, default=DEFAULT_CNV_FILE)
    p.add_argument("--clin-file", type=Path, default=DEFAULT_CLIN_FILE)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--top-p", type=int, default=DEFAULT_TOP_P)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _require_file(path: Path | None, cli_name: str, label: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"{label} path is not configured. Set LSPIN_PYTORCH_ROOT or pass --{cli_name}."
        )
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _tcga_patient_id(sample_barcode: str) -> str:
    return str(sample_barcode)[:12].replace("-", ".")


def _collapse_to_patient_level(df_samples_by_features: pd.DataFrame) -> pd.DataFrame:
    patient_ids = [_tcga_patient_id(sample_id) for sample_id in df_samples_by_features.index]
    collapsed = df_samples_by_features.copy()
    collapsed["patient"] = patient_ids
    collapsed = collapsed.sort_index()
    collapsed = collapsed.groupby("patient").head(1).drop(columns=["patient"])
    collapsed.index.name = "patient"
    return collapsed


def _load_expression(path: Path) -> pd.DataFrame:
    expr_gx = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    expr = expr_gx.T
    return _collapse_to_patient_level(expr)


def _load_cnv(path: Path) -> pd.DataFrame:
    cnv_gx = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    cnv = cnv_gx.T
    return _collapse_to_patient_level(cnv)


def _load_clinical(path: Path) -> pd.DataFrame:
    clin_long = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    if "attrib_name" not in clin_long.columns:
        raise ValueError(f"Clinical file missing attrib_name column: {path}")
    clin = clin_long.set_index("attrib_name").T
    clin.index.name = "patient"

    if "histological_type" in clin.columns:
        replacement_targets = [np.nan, "mixedhistology(pleasespecify)", "other,specify", "other"]
        mask = clin["histological_type"].isin(replacement_targets)
        clin.loc[mask, "histological_type"] = "other"

    if "overallsurvival" in clin.columns:
        parts = clin["overallsurvival"].astype(str).str.split(",", n=1, expand=True)
        clin["time"] = pd.to_numeric(parts[0], errors="coerce")
        clin["event"] = pd.to_numeric(parts[1], errors="coerce").eq(1)
    else:
        clin["time"] = pd.to_numeric(clin.get("overall_survival"), errors="coerce")
        clin["event"] = pd.to_numeric(clin.get("status"), errors="coerce").fillna(0).astype(int).astype(bool)

    if "PAM50" in clin.columns:
        clin["histo"] = clin["PAM50"].astype(str)
    elif "histological_type" in clin.columns:
        clin["histo"] = clin["histological_type"].astype(str)
    else:
        clin["histo"] = "all"

    return clin


def _dedup_keep_highvar_pos(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.to_numpy()
    variances = np.var(df.to_numpy(dtype=np.float32, copy=False), axis=0)
    tmp = pd.DataFrame(
        {
            "colname": cols,
            "var": variances,
            "pos": np.arange(len(cols)),
        }
    )
    keep_pos = tmp.sort_values("var", ascending=False).drop_duplicates("colname")["pos"].to_numpy()
    keep_pos.sort()
    return df.iloc[:, keep_pos]


def _write_manifest(outdir: Path, manifest: dict) -> None:
    (outdir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _public_input_record(path: Path, source_type: str) -> dict:
    return {
        "source_type": source_type,
        "filename": path.name,
    }


def main() -> None:
    args = _parse_args()

    rna_file = _require_file(args.rna_file, "rna-file", "RNA file")
    cnv_file = _require_file(args.cnv_file, "cnv-file", "CNV file")
    clin_file = _require_file(args.clin_file, "clin-file", "clinical file")

    outdir = args.outdir
    if outdir.exists() and any(outdir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory already exists and is non-empty: {outdir}")
    if outdir.exists() and args.overwrite:
        shutil.rmtree(outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)

    expr_patient = _load_expression(rna_file)
    cnv_patient = _load_cnv(cnv_file)
    clin = _load_clinical(clin_file)

    common_patients = expr_patient.index.intersection(cnv_patient.index)
    expr_aligned = expr_patient.loc[common_patients].copy()
    cnv_aligned = cnv_patient.loc[common_patients].copy()

    expr_aligned.columns = [f"RNA_{col}" for col in expr_aligned.columns]
    cnv_aligned.columns = [f"CNV_{col}" for col in cnv_aligned.columns]
    x_full = pd.concat([expr_aligned, cnv_aligned], axis=1)

    common_with_clin = x_full.index.intersection(clin.index)
    expr2 = x_full.loc[common_with_clin].copy()
    clin2 = clin.loc[common_with_clin].copy()

    valid_time_mask = clin2["time"].notna()
    expr2 = expr2.loc[valid_time_mask]
    clin2 = clin2.loc[valid_time_mask]
    expr2 = expr2.loc[clin2.index]

    pre_low_var_feature_count = int(expr2.shape[1])
    feature_variances_before = expr2.var(axis=0)
    low_var_mask = feature_variances_before < LOW_VAR_THRESHOLD
    expr3 = expr2.loc[:, ~low_var_mask].copy()

    nan_cols = expr3.columns[expr3.isna().any(axis=0)]
    expr_clean = expr3.copy()
    if len(nan_cols):
        col_means = expr_clean.loc[:, nan_cols].mean(axis=0)
        expr_clean.loc[:, nan_cols] = expr_clean.loc[:, nan_cols].fillna(col_means)

    feature_count_before_dedup = int(expr_clean.shape[1])
    expr_clean = _dedup_keep_highvar_pos(expr_clean)
    deduped_feature_count = int(expr_clean.shape[1])

    final_variances = expr_clean.var(axis=0).sort_values(ascending=False)
    top_features = final_variances.head(args.top_p).index
    expr_model = expr_clean.loc[:, top_features].copy()

    gene_names = expr_model.columns.to_numpy(dtype=str)
    patient_ids = expr_model.index.to_numpy(dtype=str)
    x = expr_model.to_numpy(np.float32)
    time = clin2.loc[expr_model.index, "time"].to_numpy(np.float32)
    event = clin2.loc[expr_model.index, "event"].to_numpy(bool)
    histo = clin2.loc[expr_model.index, "histo"].astype(str).to_numpy()

    idx = np.arange(expr_model.shape[0])
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=histo,
    )

    x_train_raw, x_test_raw = x[train_idx], x[test_idx]
    time_train, time_test = time[train_idx], time[test_idx]
    event_train, event_test = event[train_idx], event[test_idx]
    histo_train, histo_test = histo[train_idx], histo[test_idx]
    patient_train, patient_test = patient_ids[train_idx], patient_ids[test_idx]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    np.savez_compressed(
        outdir / "splits_and_core.npz",
        train_idx=train_idx,
        test_idx=test_idx,
        time_train=time_train,
        time_test=time_test,
        event_train=event_train.astype(np.uint8),
        event_test=event_test.astype(np.uint8),
        histo_train=histo_train.astype(str),
        histo_test=histo_test.astype(str),
        patient_train=patient_train.astype(str),
        patient_test=patient_test.astype(str),
        gene_names=gene_names.astype(str),
    )
    np.savez_compressed(outdir / "X_scaled.npz", X_train=x_train, X_test=x_test)
    joblib.dump(scaler, outdir / "models" / "scaler.joblib")

    selected_features = pd.DataFrame(
        {
            "feature": final_variances.head(args.top_p).index.astype(str),
            "variance": final_variances.head(args.top_p).to_numpy(),
        }
    )
    selected_features.to_csv(outdir / "tables" / "selected_features_top_variance.csv", index=False)

    subtype_counts = pd.Series(histo, name="count").value_counts().rename_axis("histo").reset_index()
    subtype_counts.to_csv(outdir / "tables" / "subtype_counts.csv", index=False)

    manifest = {
        "dataset": "TCGA_BRCA",
        "dataset_name": DATASET_NAME,
        "provenance_note": (
            "Regenerated from TCGA Firehose BRCA RNA, CNV, and clinical inputs using the "
            "patient-level collapse and top-variance feature-selection path preserved from "
            "lspin-pytorch/TCGA_BRCA-pt1.ipynb."
        ),
        "inputs": {
            "rna_file": _public_input_record(rna_file, "firehose_rnaseq"),
            "cnv_file": _public_input_record(cnv_file, "firehose_gistic_cnv"),
            "clinical_file": _public_input_record(clin_file, "firehose_clinical"),
        },
        "preprocessing": {
            "patient_level_collapse": "Keep first sample per patient after barcode sort.",
            "feature_prefixes": ["RNA_", "CNV_"],
            "low_variance_threshold": LOW_VAR_THRESHOLD,
            "nan_imputation": "Column-mean imputation after low-variance filtering.",
            "duplicate_feature_rule": "Keep highest-variance duplicate per feature name.",
            "top_variance_features_retained": int(args.top_p),
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "stratify_label": "PAM50 if available, else histological_type, else all.",
            "standardization": "StandardScaler fit on training data only.",
        },
        "shapes": {
            "expr_patient": list(expr_patient.shape),
            "cnv_patient": list(cnv_patient.shape),
            "combined_before_clinical_alignment": list(x_full.shape),
            "aligned_after_survival_filter": list(expr2.shape),
            "feature_count_before_low_variance_filter": pre_low_var_feature_count,
            "features_removed_low_variance": int(low_var_mask.sum()),
            "feature_count_before_dedup": feature_count_before_dedup,
            "feature_count_after_dedup": deduped_feature_count,
            "final_model_matrix": list(expr_model.shape),
            "X_train": list(x_train.shape),
            "X_test": list(x_test.shape),
        },
        "event_counts": {
            "train_events": int(event_train.sum()),
            "test_events": int(event_test.sum()),
            "total_events": int(event.sum()),
        },
        "outputs": [
            "splits_and_core.npz",
            "X_scaled.npz",
            "models/scaler.joblib",
            "tables/selected_features_top_variance.csv",
            "tables/subtype_counts.csv",
        ],
    }
    _write_manifest(outdir, manifest)

    print(f"[ok] wrote BRCA processed artifact to {outdir}")
    print(f"[shape] final model matrix: {expr_model.shape}")
    print(f"[shape] X_train: {x_train.shape} | X_test: {x_test.shape}")
    print(f"[events] train: {int(event_train.sum())} | test: {int(event_test.sum())}")


if __name__ == "__main__":
    main()
