# Phụ trách: TV4 – Huỳnh Viễn Thông
"""Unit test cho src/visual/filters.py và src/visualize_filters.py.

Nhiệm vụ T20.
Chạy từ thư mục BTL02: python -m pytest tests/test_filters.py -v
Tensor ngẫu nhiên chỉ dùng trong test này, không xuất hiện trong kết quả báo cáo chính thức.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.cnn import build_model
from src.visual.filters import (
    activation_max,
    feature_maps,
    grad_cam,
    plot_weights,
)


@pytest.fixture
def tiny_model():
    """Tạo mô hình CNN nhỏ gọn để kiểm thử tốc độ cao."""
    return build_model("baseline", blocks=[[8], [16]], fc_units=32).eval()


def test_plot_weights_first_layer(tiny_model, tmp_path):
    out_png = tmp_path / "weights_first.png"
    out_csv = tmp_path / "weights_first.csv"
    fig, w = plot_weights(tiny_model, layer="first", out_path=out_png, data_path=out_csv)

    assert out_png.exists() and out_png.stat().st_size > 0
    assert out_csv.exists() and out_csv.stat().st_size > 0
    assert w.shape == (8, 3, 3)  # 8 bộ lọc 3x3 của conv đầu


def test_plot_weights_last_layer(tiny_model, tmp_path):
    out_png = tmp_path / "weights_last.png"
    out_csv = tmp_path / "weights_last.csv"
    fig, w = plot_weights(tiny_model, layer="last", out_path=out_png, data_path=out_csv)

    assert out_png.exists()
    assert out_csv.exists()
    assert w.shape == (16, 3, 3)  # 16 bộ lọc 3x3 ở lớp conv cuối


def test_feature_maps(tiny_model, tmp_path):
    x = torch.randn(1, 1, 48, 48)
    out_png = tmp_path / "fmaps.png"
    out_npz = tmp_path / "fmaps.npz"
    fig, fmaps = feature_maps(tiny_model, x, layer="last", top_k=8, out_path=out_png, data_path=out_npz)

    assert out_png.exists()
    assert out_npz.exists()
    assert fmaps.shape[0] == 8  # top 8 kênh
    assert np.isfinite(fmaps).all()


def test_activation_max_filter(tiny_model, tmp_path):
    out_png = tmp_path / "actmax_filter.png"
    out_npz = tmp_path / "actmax_filter.npz"
    fig, imgs = activation_max(
        tiny_model,
        target_type="filter",
        target_indices=[0, 1],
        steps=10,
        out_path=out_png,
        data_path=out_npz,
    )

    assert out_png.exists()
    assert out_npz.exists()
    assert imgs.shape == (2, 48, 48)
    assert (imgs >= 0.0).all() and (imgs <= 1.0).all()


def test_activation_max_class(tiny_model, tmp_path):
    out_png = tmp_path / "actmax_class.png"
    out_npz = tmp_path / "actmax_class.npz"
    fig, imgs = activation_max(
        tiny_model,
        target_type="class",
        target_indices=[0, 1, 2],
        steps=10,
        out_path=out_png,
        data_path=out_npz,
    )

    assert out_png.exists()
    assert out_npz.exists()
    assert imgs.shape == (3, 48, 48)
    assert (imgs >= 0.0).all() and (imgs <= 1.0).all()


def test_grad_cam(tiny_model, tmp_path):
    x = torch.randn(1, 1, 48, 48)
    out_png = tmp_path / "grad_cam.png"
    out_csv = tmp_path / "grad_cam.csv"
    fig, cam = grad_cam(tiny_model, x, target_class=3, out_path=out_png, data_path=out_csv)

    assert out_png.exists()
    assert out_csv.exists()
    assert cam.shape == (48, 48)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6


def test_visualize_filters_cli(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    btl02_dir = Path(__file__).resolve().parent.parent
    out_dir = tmp_path / "figures_test"
    cmd = [
        sys.executable,
        str(btl02_dir / "src" / "visualize_filters.py"),
        "--model", "baseline",
        "--out-dir", str(out_dir),
        "--device", "cpu",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=str(btl02_dir))
    assert res.returncode == 0, f"Lỗi chạy CLI: {res.stderr}\nSTDOUT: {res.stdout}"
    assert (out_dir / "filters_weights_first.png").exists()
    assert (out_dir / "filters_feature_maps.png").exists()
    assert (out_dir / "filters_actmax_filter.png").exists()
    assert (out_dir / "filters_grad_cam.png").exists()


def test_load_or_build_model_reads_config_model_kwargs(tmp_path):
    """Xác nhận nạp đúng checkpoint có model_kwargs (vd large_bn: bn=True, activation=leaky_relu) từ config.json."""
    import json
    from src.visualize_filters import load_or_build_model

    run_dir = tmp_path / "E7-001_s0"
    run_dir.mkdir()
    kwargs = {"blocks": [[8], [16]], "fc_units": 32, "bn": True, "activation": "leaky_relu"}
    m = build_model("large", **kwargs).eval()
    ckpt_path = run_dir / "best.pt"
    torch.save({"model": m.state_dict()}, ckpt_path)
    (run_dir / "config.json").write_text(
        json.dumps({"exp_id": "E7-001_s0", "cfg": {"model": "large", "model_kwargs": kwargs}}),
        encoding="utf-8",
    )

    # Tự phát hiện config.json nằm cạnh best.pt mà không cần truyền --model large hay --config
    loaded = load_or_build_model(model_name="baseline", checkpoint_path=str(ckpt_path), device="cpu")
    assert any(isinstance(mod, nn.BatchNorm2d) for mod in loaded.modules())
    assert any(isinstance(mod, nn.LeakyReLU) for mod in loaded.modules())
    for p1, p2 in zip(m.parameters(), loaded.parameters()):
        assert torch.allclose(p1, p2)


def test_visualize_filters_cli_with_run_dir_large_bn(tmp_path):
    """Kiểm tra chạy CLI qua --run-dir nạp checkpoint large_bn từ config.json."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    btl02_dir = Path(__file__).resolve().parent.parent
    run_dir = tmp_path / "E7_large_bn"
    run_dir.mkdir()
    kwargs = {"blocks": [[8], [16]], "fc_units": 32, "bn": True, "activation": "relu"}
    m = build_model("large", **kwargs).eval()
    torch.save({"model": m.state_dict()}, run_dir / "best.pt")
    (run_dir / "config.json").write_text(
        json.dumps({"exp_id": "E7-test_s0", "cfg": {"model": "large", "model_kwargs": kwargs}}),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_large_bn"
    cmd = [
        sys.executable,
        str(btl02_dir / "src" / "visualize_filters.py"),
        "--run-dir", str(run_dir),
        "--method", "weights",
        "--out-dir", str(out_dir),
        "--device", "cpu",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=str(btl02_dir))
    assert res.returncode == 0, f"Lỗi chạy CLI --run-dir: {res.stderr}\nSTDOUT: {res.stdout}"
    assert (out_dir / "filters_weights_first.png").exists()
    assert (out_dir / "filters_weights_last.png").exists()
