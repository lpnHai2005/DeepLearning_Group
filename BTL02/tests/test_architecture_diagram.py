# Phụ trách: TV4 – Huỳnh Viễn Thông
"""Kiểm thử unit test cho module vẽ sơ đồ kiến trúc CNN và pipeline (src/visual/architecture_diagram.py).

Nhiệm vụ T24.
"""
import csv
from pathlib import Path

import pytest

from src.models.cnn import build_model, n_params
from src.visual.architecture_diagram import (
    generate_all,
    plot_baseline_architecture,
    plot_model_comparison,
    plot_pipeline_overview,
)


def test_plot_baseline_architecture(tmp_path):
    out_png = tmp_path / "test_arch.png"
    out_csv = tmp_path / "test_arch.csv"

    p_png, p_csv = plot_baseline_architecture(out_png, out_csv)
    assert p_png.exists() and p_png.stat().st_size > 0
    assert p_csv.exists() and p_csv.stat().st_size > 0

    with open(p_csv, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) >= 15  # Baseline có 15 lớp theo describe()

    # Kiểm tra tổng tham số khớp với hàm n_params thực tế
    model = build_model("baseline")
    assert int(reader[-1]["params_cumulative"]) == n_params(model) == 5867847


def test_plot_model_comparison(tmp_path):
    out_png = tmp_path / "test_comp.png"
    out_csv = tmp_path / "test_comp.csv"

    p_png, p_csv = plot_model_comparison(out_png, out_csv)
    assert p_png.exists() and p_png.stat().st_size > 0
    assert p_csv.exists() and p_csv.stat().st_size > 0

    with open(p_csv, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 5  # 5 mô hình: baseline, small, medium, medium_01, large
    model_names = {r["model_name"] for r in reader}
    assert "baseline" in model_names and "medium_01" in model_names


def test_plot_pipeline_overview(tmp_path):
    out_png = tmp_path / "test_pipe.png"
    out_csv = tmp_path / "test_pipe.csv"

    p_png, p_csv = plot_pipeline_overview(out_png, out_csv)
    assert p_png.exists() and p_png.stat().st_size > 0
    assert p_csv.exists() and p_csv.stat().st_size > 0

    with open(p_csv, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 6  # 6 chặng của pipeline


def test_generate_all(tmp_path):
    fig_dir = tmp_path / "figures"
    manifest = tmp_path / "tracking" / "figures_manifest.csv"

    res = generate_all(out_dir=fig_dir, manifest_path=manifest, root=tmp_path)
    assert len(res) == 3

    assert manifest.exists()
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    fig_ids = {r["fig_id"] for r in rows}
    assert fig_ids == {"arch_cnn_baseline", "arch_cnn_comparison", "pipeline_overview"}
    for r in rows:
        assert r["task_id"] == "T24"
        assert len(r["sha256_hinh"]) == 64
