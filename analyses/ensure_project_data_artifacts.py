#!/usr/bin/env python3
"""Stage split artifacts under sparsedeepsurv-paper/data/processed.

This is a transitional reproducibility helper: it makes the analysis data
objects project-local instead of letting scripts silently read legacy
`/banach2/wes/lspin-pytorch/runs/...` paths. It copies the already-prepared
split artifacts plus a small provenance manifest.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PAPER_ROOT / "data" / "processed"

LEGACY_SOURCES = {
    "kipan": Path("/banach2/wes/lspin-pytorch/runs/kipan_20260209_213604"),
    "brca": Path("/banach2/wes/lspin-pytorch/runs/tcga_brca20260214_001423"),
    "pancan": Path("/banach2/wes/lspin-pytorch/runs/tcga_pancan_xena_20260330_top5000"),
}
LOCAL_NAMES = {
    "kipan": "kipan_20260209_213604",
    "brca": "tcga_brca20260214_001423",
    "pancan": "tcga_pancan_xena_20260330_top5000",
}
REQUIRED_FILES = ["splits_and_core.npz", "X_scaled.npz"]
OPTIONAL_FILES = ["models/scaler.joblib"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--datasets", nargs="+", choices=sorted(LEGACY_SOURCES), default=sorted(LEGACY_SOURCES))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = _parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        source = LEGACY_SOURCES[dataset]
        dest = args.out_root / LOCAL_NAMES[dataset]
        missing = [rel for rel in REQUIRED_FILES if not (source / rel).exists()]
        if missing:
            raise FileNotFoundError(f"{dataset}: missing required source artifacts under {source}: {missing}")
        for rel in REQUIRED_FILES:
            _copy_file(source / rel, dest / rel, overwrite=args.overwrite)
        for rel in OPTIONAL_FILES:
            if (source / rel).exists():
                _copy_file(source / rel, dest / rel, overwrite=args.overwrite)
        manifest = {
            "dataset": dataset,
            "source": str(source),
            "destination": str(dest),
            "required_files": REQUIRED_FILES,
            "optional_files_copied": [rel for rel in OPTIONAL_FILES if (source / rel).exists()],
            "note": (
                "Project-local copy of prepared split artifacts. Regenerate from raw source data "
                "before publication if these legacy-prepared artifacts are not acceptable as provenance."
            ),
        }
        (dest / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[ok] {dataset}: {dest}")


if __name__ == "__main__":
    main()
