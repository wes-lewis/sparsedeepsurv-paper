"""
Smoke test: rerender_boxplots.py runs without error given frozen CSV.
"""
import pytest
from pathlib import Path
import pandas as pd
import tempfile

from analyses.rerender_boxplots import main


def test_rerender_boxplots_runs(tmp_path):
    # Minimal fake metrics CSV
    df = pd.DataFrame({
        "model_label": ["LSPIN + patient smooth"] * 10 + ["MLP + L1"] * 10,
        "cindex": [0.68] * 10 + [0.62] * 10,
    })
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "selected_comparison_metrics_summary.csv").write_text(df.to_csv(index=False))

    out_dir = tmp_path / "figures"
    main(data_dir=data_dir, out_dir=out_dir)
    assert (out_dir / "targeted_cindex_boxplot.pdf").exists()
