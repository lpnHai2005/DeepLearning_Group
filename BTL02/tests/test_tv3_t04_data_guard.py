# Phụ trách: TV3 – Võ Hoàng Bảo
"""Kiểm tra bổ sung cho T04 (C1): hằng EMOTIONS, chống rò rỉ mean/std từ val/test và tách biệt PrivateTest.

Chạy từ thư mục BTL02: python -m pytest tests/test_tv3_t04_data_guard.py -v
"""
import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.fer2013 import EMOTIONS, EXPECTED_COUNTS, load_fer2013
from src.data.loader import make_loaders, make_test_loader
from src.engine.evaluate import EMOTIONS as EVAL_EMOTIONS
from src.visual.filters import EMOTIONS as FILTER_EMOTIONS

BTL02 = Path(__file__).resolve().parents[1]
REAL_CSV = BTL02 / "data" / "fer2013.csv"


def _pixels(value: int) -> str:
    return " ".join([str(value)] * (48 * 48))


def _write_csv(path: Path, val_pixel: int, test_pixel: int) -> Path:
    rows = [
        {"emotion": 0, "pixels": _pixels(20), "Usage": "Training"},
        {"emotion": 1, "pixels": _pixels(80), "Usage": "Training"},
        {"emotion": 2, "pixels": _pixels(140), "Usage": "Training"},
        {"emotion": 3, "pixels": _pixels(200), "Usage": "Training"},
        {"emotion": 4, "pixels": _pixels(val_pixel), "Usage": "PublicTest"},
        {"emotion": 5, "pixels": _pixels(test_pixel), "Usage": "PrivateTest"},
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["emotion", "pixels", "Usage"])
        w.writeheader()
        w.writerows(rows)
    return path


def test_thu_tu_nhan_emotions_thong_nhat_toan_repo():
    expected = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    assert list(EMOTIONS) == expected
    assert list(EVAL_EMOTIONS) == expected
    assert list(FILTER_EMOTIONS) == expected


def test_chong_ro_ri_mean_std_tu_val_va_test(tmp_path):
    csv_a = _write_csv(tmp_path / "a.csv", val_pixel=0, test_pixel=0)
    csv_b = _write_csv(tmp_path / "b.csv", val_pixel=255, test_pixel=255)

    cfg_a = {"csv_path": str(csv_a), "batch_size": 4, "seed": 3, "augmentation": "none", "device": "cpu"}
    cfg_b = {"csv_path": str(csv_b), "batch_size": 4, "seed": 3, "augmentation": "none", "device": "cpu"}

    tr_a, va_a = make_loaders(cfg_a)
    tr_b, va_b = make_loaders(cfg_b)

    assert torch.equal(tr_a.dataset.mean, tr_b.dataset.mean)
    assert torch.equal(tr_a.dataset.std, tr_b.dataset.std)
    xa, ya = next(iter(tr_a))
    xb, yb = next(iter(tr_b))
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert len(va_a.dataset) == 1 and len(va_b.dataset) == 1


def test_make_loaders_khong_bao_gio_chua_private_test(tmp_path):
    csv_p = _write_csv(tmp_path / "split.csv", val_pixel=111, test_pixel=250)
    cfg = {"csv_path": str(csv_p), "batch_size": 10, "seed": 0, "augmentation": "none", "device": "cpu"}

    train_loader, val_loader = make_loaders(cfg)
    test_loader = make_test_loader(cfg)

    train_labels = torch.cat([y for _, y in train_loader]).tolist()
    val_labels = torch.cat([y for _, y in val_loader]).tolist()
    test_labels = torch.cat([y for _, y in test_loader]).tolist()

    assert set(train_labels) == {0, 1, 2, 3}
    assert val_labels == [4]
    assert test_labels == [5]


@pytest.mark.skipif(not REAL_CSV.is_file(), reason="Không có data/fer2013.csv cục bộ")
def test_fer2013_that_du_so_anh_va_dtype():
    data = load_fer2013(REAL_CSV)
    for split, expected_n in EXPECTED_COUNTS.items():
        x, y = data[split]
        assert x.shape == (expected_n, 48, 48)
        assert x.dtype == np.uint8
        assert y.shape == (expected_n,)
        assert y.dtype == np.int64
        assert int(y.min()) == 0 and int(y.max()) == 6
