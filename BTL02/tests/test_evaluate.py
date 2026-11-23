"""Unit test cho src/engine/evaluate.py và src/visual/plots.py.

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T08.
Chạy từ thư mục BTL02: python -m pytest tests/test_evaluate.py -v
Dữ liệu giả lập CHỈ dùng trong unit test này, không bao giờ dùng cho báo cáo.
"""
import csv

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine.evaluate import EMOTIONS, metrics, predict, save_predictions
from src.visual.plots import (
    MANIFEST_HEADER,
    plot_confusion_matrix,
    plot_curves,
    plot_error_histogram,
    plot_run_comparison,
    plot_sample_grid,
    register_figure,
)


def test_metrics_khop_sklearn():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 7, size=120)
    probs = rng.dirichlet(np.ones(7), size=120)
    y_pred = probs.argmax(axis=1)

    res = metrics(y_true, probs)
    assert res["acc"] == pytest.approx(accuracy_score(y_true, y_pred), abs=1e-12)
    assert res["macro_f1"] == pytest.approx(f1_score(y_true, y_pred, labels=list(range(7)), average="macro", zero_division=0), abs=1e-12)
    assert np.array_equal(res["cm"], confusion_matrix(y_true, y_pred, labels=list(range(7))))
    assert list(res["per_class"].keys()) == EMOTIONS
    assert np.allclose(res["cm_norm"].sum(axis=1), 1.0)


def test_predict_thuong_va_tta():
    torch.manual_seed(0)
    x = torch.randn(20, 1, 48, 48)
    y = torch.randint(0, 7, (20,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 7))

    for mode in (False, True, "flip", "ten_crop"):
        yt, yp = predict(model, loader, tta=mode, device="cpu", amp=False)
        assert yt.shape == (20,) and yt.dtype == np.int64
        assert yp.shape == (20, 7) and yp.dtype == np.float32
        assert np.allclose(yp.sum(axis=1), 1.0, atol=1e-5)
        assert np.array_equal(yt, y.numpy())


def test_save_predictions(tmp_path):
    yt = np.array([0, 3, 6], dtype=np.int64)
    yp = np.array([[0.8, 0.2, 0, 0, 0, 0, 0], [0, 0, 0, 0.9, 0.1, 0, 0], [0, 0, 0, 0, 0, 0, 1.0]], dtype=np.float32)
    p = save_predictions(tmp_path / "preds.npz", yt, yp)
    data = np.load(p)
    assert np.array_equal(data["y_true"], yt)
    assert np.array_equal(data["y_pred"], np.array([0, 3, 6]))


def test_plots_xuat_hinh_kem_csv_va_manifest(tmp_path):
    hist_csv = tmp_path / "history.csv"
    hist_rows = [
        {"epoch": 1, "train_loss": 1.8, "val_loss": 1.7, "train_acc": 0.30, "val_acc": 0.32, "val_macro_f1": 0.28, "lr": 0.01, "epoch_time_s": 1.2},
        {"epoch": 2, "train_loss": 1.4, "val_loss": 1.5, "train_acc": 0.45, "val_acc": 0.44, "val_macro_f1": 0.40, "lr": 0.005, "epoch_time_s": 1.1},
    ]
    with open(hist_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist_rows[0].keys()))
        w.writeheader()
        w.writerows(hist_rows)

    png, csv_p = plot_curves(hist_csv, tmp_path / "curves.png")
    assert png.exists() and png.stat().st_size > 0
    assert csv_p.exists()
    with open(csv_p, encoding="utf-8", newline="") as f:
        plotted = list(csv.DictReader(f))
    assert len(plotted) == 2 and float(plotted[1]["val_acc"]) == pytest.approx(0.44)

    rng = np.random.default_rng(7)
    imgs = rng.integers(0, 255, size=(12, 48, 48), dtype=np.uint8)
    yt = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4])
    yp = np.array([0, 2, 2, 3, 6, 5, 6, 4, 1, 0, 3, 4])
    probs = np.eye(7, dtype=np.float32)[yp]
    res = metrics(yt, yp)

    cm_png, cm_csv = plot_confusion_matrix(res["cm"], tmp_path / "cm.png")
    err_png, err_csv = plot_error_histogram(yt, yp, tmp_path / "err.png")
    grid_png, grid_csv = plot_sample_grid(imgs, yt, yp, tmp_path / "wrong.png", y_prob=probs, correct=False)
    for p in (cm_png, cm_csv, err_png, err_csv, grid_png, grid_csv):
        assert p.exists() and p.stat().st_size > 0

    exp_csv = tmp_path / "experiments.csv"
    with open(exp_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exp_id", "task_id", "thanh_vien", "val_acc", "val_macro_f1", "best_val_loss", "trang_thai_ky_thuat"])
        w.writeheader()
        w.writerow({"exp_id": "E3-001_s0", "task_id": "T14", "thanh_vien": "TV3", "val_acc": "0.66", "val_macro_f1": "0.60", "best_val_loss": "0.95", "trang_thai_ky_thuat": "Completed"})
        w.writerow({"exp_id": "E3-002_s0", "task_id": "T14", "thanh_vien": "TV3", "val_acc": "0.68", "val_macro_f1": "0.62", "best_val_loss": "0.91", "trang_thai_ky_thuat": "Completed"})
    cmp_png, cmp_csv = plot_run_comparison(exp_csv, tmp_path / "cmp.png")
    assert cmp_png.exists() and cmp_csv.exists()

    manifest = tmp_path / "figures_manifest.csv"
    row = register_figure(manifest, "FIG-001", png, hist_csv, "src/visual/plots.py", "python src/visual/plots.py", exp_ids="E3-001_s0", task_id="T08", dung_o="Ch.4", root=tmp_path)
    assert row["fig_id"] == "FIG-001" and len(row["sha256_hinh"]) == 64
    with open(manifest, encoding="utf-8", newline="") as f:
        m_rows = list(csv.DictReader(f))
    assert len(m_rows) == 1 and list(m_rows[0].keys()) == MANIFEST_HEADER
