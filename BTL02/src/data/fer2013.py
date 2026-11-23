# Phụ trách: TV1 – Đoàn Trung Kiên
"""Đọc và tách FER2013 theo Usage mà không đổi thứ tự nhãn Kaggle."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
EXPECTED_COUNTS = {"train": 28_709, "val": 3_589, "test": 3_589}
IMAGE_SIZE = 48
PIXELS_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE


def load_fer2013(csv_path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Trả train/val/test thành (x uint8[N,48,48], y int64[N]).

    Hàm chỉ đọc dữ liệu. Chuẩn hóa, augmentation và cách dùng test được xử lý
    riêng trong loader.py để `fit()` không nhận được PrivateTest.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy FER2013 CSV: {path}")

    images: dict[str, list[np.ndarray]] = {split: [] for split in USAGE_TO_SPLIT.values()}
    labels: dict[str, list[int]] = {split: [] for split in USAGE_TO_SPLIT.values()}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"emotion", "pixels", "Usage"}
        headers = set(reader.fieldnames or ())
        if missing := required - headers:
            raise ValueError(f"Thiếu cột bắt buộc trong {path.name}: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            if not row or not any(value for value in row.values() if value is not None):
                continue

            usage = (row.get("Usage") or "").strip()
            if usage not in USAGE_TO_SPLIT:
                raise ValueError(f"Usage không hợp lệ ở dòng {row_number}: {usage!r}")
            split = USAGE_TO_SPLIT[usage]

            try:
                label = int(row["emotion"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Nhãn emotion không phải số nguyên ở dòng {row_number}") from exc
            if not 0 <= label < len(EMOTIONS):
                raise ValueError(f"Nhãn ngoài khoảng 0–6 ở dòng {row_number}: {label}")

            pixel_text = row.get("pixels") or ""
            pixel_values = pixel_text.split()
            if len(pixel_values) != PIXELS_PER_IMAGE:
                raise ValueError(
                    f"Dòng {row_number} có {len(pixel_values)} pixel; cần đúng {PIXELS_PER_IMAGE}"
                )
            try:
                values = np.fromiter((int(value) for value in pixel_values), dtype=np.int64)
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Pixel không phải số nguyên ở dòng {row_number}") from exc
            if np.any((values < 0) | (values > 255)):
                raise ValueError(f"Pixel ngoài khoảng 0–255 ở dòng {row_number}")

            images[split].append(values.astype(np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE))
            labels[split].append(label)

    empty = [split for split, items in images.items() if not items]
    if empty:
        raise ValueError(f"Không có ảnh cho split(s): {', '.join(empty)}")

    return {
        split: (
            np.stack(images[split]).astype(np.uint8, copy=False),
            np.asarray(labels[split], dtype=np.int64),
        )
        for split in USAGE_TO_SPLIT.values()
    }
