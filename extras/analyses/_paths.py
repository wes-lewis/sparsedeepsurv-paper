from __future__ import annotations

import os
import sys
from pathlib import Path


ANALYSES_DIR = Path(__file__).resolve().parent
PAPER_ROOT = ANALYSES_DIR.parent
DATA_DIR = PAPER_ROOT / "data"
DATA_RUNS_DIR = DATA_DIR / "runs"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
GENE_SETS_DIR = DATA_DIR / "gene_sets"
FIGURES_DIR = PAPER_ROOT / "figures"
TABLES_DIR = PAPER_ROOT / "tables"


def _candidate_paths(*paths: Path | None) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        if p is None:
            continue
        s = str(p)
        if s not in seen:
            out.append(p)
            seen.add(s)
    return out


def _existing_path(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def sparsedeepsurv_src() -> Path | None:
    env = os.environ.get("SPARSEDEEPSURV_SRC")
    candidates = _candidate_paths(
        Path(env) if env else None,
        PAPER_ROOT.parent / "sparsedeepsurv" / "src",
    )
    return _existing_path(candidates)


def ensure_repo_imports(*, include_analyses_dir: bool = False) -> None:
    to_add: list[Path] = []
    sds_src = sparsedeepsurv_src()
    if sds_src is not None:
        to_add.append(sds_src)
    if include_analyses_dir:
        to_add.append(ANALYSES_DIR)
    for p in to_add:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def canonical_run_dir(name: str) -> Path:
    return DATA_RUNS_DIR / name


def processed_data_dir(name: str) -> Path:
    return DATA_PROCESSED_DIR / name


CANONICAL_RUNS = {
    "validate_goal1_gentle_all_kipan_brca": canonical_run_dir(
        "validate_goal1_gentle_all_kipan_brca_20260408_175906"
    ),
    "kipan_adaptive_gentle": canonical_run_dir(
        "adaptive_gentle_all_kipan_brca_pancan_20260408_193020"
    ) / "kipan",
    "brca_adaptive_gentle": canonical_run_dir(
        "adaptive_gentle_all_kipan_brca_pancan_20260408_193020"
    ) / "brca",
    "pancan_adaptive_gentle": canonical_run_dir(
        "adaptive_gentle_all_kipan_brca_pancan_20260408_193020"
    ) / "pancan",
    "kipan_broad_v2": canonical_run_dir("ch3_kipan_broad_v2_selfcontained_ste_20260407_012336"),
    "brca_broad_v2": canonical_run_dir("ch3_brca_broad_v2_recovered_20260326"),
    "kipan_broad_gentle": canonical_run_dir("ch3_kipan_broad_gentle_20260409_161803"),
    "brca_broad_gentle": canonical_run_dir("ch3_brca_broad_gentle_20260409_161803"),
}


PROCESSED_DATASETS = {
    "kipan": processed_data_dir("kipan_20260209_213604"),
    "brca": processed_data_dir("tcga_brca20260214_001423"),
    "pancan": processed_data_dir("tcga_pancan_xena_20260330_top5000"),
}
