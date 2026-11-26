# Phụ trách: TV1 – Đoàn Trung Kiên
"""Tạo loader in-memory; mean/std chỉ tính từ Training."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.augment import build_augmentation
from src.data.fer2013 import EMOTIONS, load_fer2013

BTL02 = Path(__file__).resolve().parents[2]


def _cfg(config: Mapping, *keys, default=None):
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _data_path(config: Mapping) -> Path:
    raw = _cfg(config, "csv_path", "data_path", "dataset_path", default="data/fer2013.csv")
    path = Path(raw)
    return path if path.is_absolute() else (BTL02 / path).resolve()


def _image_options(config: Mapping) -> tuple[int, int]:
    size = _cfg(config, "resize", "image_size", "input_size", default=48)
    if isinstance(size, (tuple, list)):
        if len(size) != 2 or size[0] != size[1]:
            raise ValueError("FER2013 chỉ hỗ trợ ảnh vuông; resize phải là số hoặc (N, N)")
        size = size[0]
    size = int(size)
    channels = int(_cfg(config, "channels", "in_channels", "input_channels", default=1))
    if size <= 0:
        raise ValueError("Kích thước ảnh phải lớn hơn 0")
    if channels not in (1, 3):
        raise ValueError("FER2013 hỗ trợ ảnh 1 kênh hoặc lặp xám thành 3 kênh")
    return size, channels


def _as_float_tensor(images: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(images)).unsqueeze(1).to(torch.float32).div_(255.0)
    if image_size != 48:
        tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor


def _training_stats(images: np.ndarray, image_size: int, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    train = _as_float_tensor(images, image_size)
    mean = train.mean(dim=(0, 2, 3))
    std = train.std(dim=(0, 2, 3), unbiased=False).clamp_min(1e-6)
    if channels == 3:
        mean = mean.repeat(3)
        std = std.repeat(3)
    return mean.view(channels, 1, 1), std.view(channels, 1, 1)


class _FerDataset(Dataset):
    def __init__(self, images, labels, image_size, channels, mean, std, augmentation=None):
        self.images = torch.from_numpy(np.ascontiguousarray(images))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).to(torch.int64)
        self.image_size = image_size
        self.channels = channels
        self.mean = mean
        self.std = std
        self.augmentation = augmentation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index].unsqueeze(0).to(torch.float32).div_(255.0)
        if self.channels == 3:
            image = image.repeat(3, 1, 1)
        if self.image_size != 48:
            image = F.interpolate(
                image.unsqueeze(0), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            ).squeeze(0)
        if self.augmentation is not None:
            image = self.augmentation(image)
        image = (image - self.mean) / self.std
        return image, self.labels[index]


def _make_dataset(split, image_size, channels, mean, std, augmentation=None):
    images, labels = split
    return _FerDataset(images, labels, image_size, channels, mean, std, augmentation)


def _loader(dataset, config, *, shuffle: bool, seed: int, sampler=None):
    batch_size = int(_cfg(config, "batch_size", default=128))
    workers = int(_cfg(config, "num_workers", default=0))
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0")
    if workers < 0:
        raise ValueError("num_workers không được âm")
    if os.name == "nt":
        workers = 0
    generator = torch.Generator().manual_seed(seed)
    device = str(_cfg(config, "device", default="cpu"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(_cfg(config, "pin_memory", default=device.startswith("cuda"))),
        generator=generator,
    )


def _training_sampler(labels, config: Mapping, seed: int):
    """Tạo sampler train có seed; trọng số lấy riêng từ nhãn Training."""
    name = str(_cfg(config, "sampler", default="none")).strip().lower()
    if name in {"", "none", "shuffle"}:
        return None
    if name not in {"weighted_random", "weightedrandomsampler"}:
        raise ValueError(f"Sampler không hỗ trợ: {name!r}")

    y = torch.as_tensor(labels, dtype=torch.int64)
    if y.numel() == 0 or torch.any(y < 0) or torch.any(y >= len(EMOTIONS)):
        raise ValueError("Nhãn Training không hợp lệ để tạo WeightedRandomSampler")
    counts = torch.bincount(y, minlength=len(EMOTIONS))
    if torch.any(counts == 0):
        raise ValueError("Mỗi lớp phải có mẫu trong Training để tạo WeightedRandomSampler")
    sample_weights = counts.to(torch.double).reciprocal()[y]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(y),
        replacement=True,
        generator=generator,
    )


def _build_datasets(config: Mapping):
    data = load_fer2013(_data_path(config))
    image_size, channels = _image_options(config)
    mean, std = _training_stats(data["train"][0], image_size, channels)
    return data, image_size, channels, mean, std


def make_loaders(config: Mapping):
    """Trả `(train_loader, val_loader)`; PrivateTest không được đưa vào `fit()`."""
    data, image_size, channels, mean, std = _build_datasets(config)
    seed = int(_cfg(config, "seed", default=0))
    torch.manual_seed(seed)
    augmentation = _cfg(config, "augmentation", "augment", default="flip_affine")
    train_ds = _make_dataset(data["train"], image_size, channels, mean, std, build_augmentation(augmentation))
    val_ds = _make_dataset(data["val"], image_size, channels, mean, std)
    sampler = _training_sampler(data["train"][1], config, seed)
    return (
        _loader(train_ds, config, shuffle=True, seed=seed, sampler=sampler),
        _loader(val_ds, config, shuffle=False, seed=seed),
    )


def make_test_loader(config: Mapping):
    """Tạo riêng loader PrivateTest; chỉ gọi trong quy trình `--eval-test` cuối."""
    data, image_size, channels, mean, std = _build_datasets(config)
    test_ds = _make_dataset(data["test"], image_size, channels, mean, std)
    seed = int(_cfg(config, "seed", default=0))
    return _loader(test_ds, config, shuffle=False, seed=seed)
