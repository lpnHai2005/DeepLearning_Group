# Phụ trách: TV1 – Đoàn Trung Kiên
"""Các augmentation FER2013 áp dụng trên tensor xám trước chuẩn hóa."""

from __future__ import annotations

from collections.abc import Mapping

from torchvision import transforms
from torchvision.transforms import InterpolationMode

_ALIASES = {
    "": "flip_affine",
    "default": "flip_affine",
    "baseline": "flip_affine",
    "no_aug": "none",
    "no-aug": "none",
    "no augmentation": "none",
    "flip+affine": "flip_affine",
    "flip_affine": "flip_affine",
    "randomerasing": "random_erasing",
    "+randomerasing": "random_erasing",
}


def build_augmentation(config: str | Mapping | None = None) -> transforms.Compose:
    """Tạo biến đổi cho `none`, `flip`, `affine`, `flip_affine`, `random_erasing`."""
    if isinstance(config, Mapping):
        mode = config.get("mode", config.get("name", "flip_affine"))
        erase_probability = float(config.get("p", 0.5))
    else:
        mode = config or "flip_affine"
        erase_probability = 0.5

    key = str(mode).strip().lower()
    mode = _ALIASES.get(key, key)
    if mode not in {"none", "flip", "affine", "flip_affine", "random_erasing"}:
        raise ValueError(f"Augmentation không hỗ trợ: {mode!r}")
    if not 0.0 <= erase_probability <= 1.0:
        raise ValueError("Xác suất RandomErasing phải nằm trong [0, 1]")

    operations = []
    if mode in {"flip", "affine", "flip_affine", "random_erasing"}:
        operations.append(transforms.RandomHorizontalFlip(p=0.5))
    if mode in {"affine", "flip_affine", "random_erasing"}:
        operations.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )
        )
    if mode == "random_erasing":
        operations.append(transforms.RandomErasing(p=erase_probability, value=0))
    return transforms.Compose(operations)
