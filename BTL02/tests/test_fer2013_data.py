# Phụ trách: TV1 – Đoàn Trung Kiên
"""Unit tests cho đọc dữ liệu, augmentation và loader FER2013; fixture chỉ ở tmpdir."""

import csv

import numpy as np
import pytest
import torch

from src.data.fer2013 import EMOTIONS, load_fer2013
from src.data.loader import make_loaders, make_test_loader


def _pixels(value):
    return " ".join([str(value)] * (48 * 48))


@pytest.fixture
def fer_csv(tmp_path):
    rows = [
        {"emotion": 0, "pixels": _pixels(0), "Usage": "Training"},
        {"emotion": 1, "pixels": _pixels(255), "Usage": "Training"},
        {"emotion": 2, "pixels": _pixels(128), "Usage": "PublicTest"},
        {"emotion": 3, "pixels": _pixels(200), "Usage": "PrivateTest"},
    ]
    path = tmp_path / "fer2013_fixture.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pixels", "Usage"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_load_fer2013_preserves_split_shape_dtype_and_labels(fer_csv):
    data = load_fer2013(fer_csv)

    assert list(EMOTIONS) == ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    assert list(data) == ["train", "val", "test"]
    assert data["train"][0].shape == (2, 48, 48)
    assert data["train"][0].dtype == np.uint8
    assert data["train"][1].dtype == np.int64
    assert data["train"][1].tolist() == [0, 1]
    assert data["val"][1].tolist() == [2]
    assert data["test"][1].tolist() == [3]


@pytest.mark.parametrize(
    ("usage", "label", "pixel_text", "message"),
    [
        ("Other", 0, _pixels(0), "Usage"),
        ("Training", 7, _pixels(0), "Nhãn ngoài khoảng"),
        ("Training", 0, "1 2 malformed", "pixel"),
        ("Training", 0, _pixels(256), "Pixel ngoài khoảng"),
    ],
)
def test_load_fer2013_rejects_invalid_rows(tmp_path, usage, label, pixel_text, message):
    path = tmp_path / "invalid.csv"
    rows = [
        {"emotion": label, "pixels": pixel_text, "Usage": usage},
        {"emotion": 0, "pixels": _pixels(0), "Usage": "PublicTest"},
        {"emotion": 0, "pixels": _pixels(0), "Usage": "PrivateTest"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pixels", "Usage"])
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match=message):
        load_fer2013(path)


def test_make_loaders_fit_stats_only_on_training_and_test_is_separate(fer_csv):
    config = {"csv_path": str(fer_csv), "batch_size": 2, "seed": 7, "augmentation": "none",
              "resize": 24, "channels": 3, "device": "cpu"}
    train_loader, val_loader = make_loaders(config)
    assert len(train_loader.dataset) == 2
    assert len(val_loader.dataset) == 1
    assert train_loader.dataset.mean.shape == (3, 1, 1)
    assert torch.allclose(train_loader.dataset.mean, torch.full((3, 1, 1), 0.5))
    assert torch.allclose(train_loader.dataset.std, torch.full((3, 1, 1), 0.5))
    x_train, y_train = next(iter(train_loader))
    assert x_train.shape == (2, 3, 24, 24)
    assert y_train.dtype == torch.int64
    assert torch.allclose(next(iter(val_loader))[0].mean(), torch.tensor((128 / 255 - 0.5) / 0.5), atol=1e-5)

    test_loader = make_test_loader(config)
    assert len(test_loader.dataset) == 1
    assert next(iter(test_loader))[1].tolist() == [3]


def test_augmentation_is_reproducible_for_same_seed(fer_csv):
    config = {"csv_path": str(fer_csv), "batch_size": 2, "seed": 19, "augmentation": "flip_affine"}
    first = next(iter(make_loaders(config)[0]))[0]
    second = next(iter(make_loaders(config)[0]))[0]
    assert torch.equal(first, second)
