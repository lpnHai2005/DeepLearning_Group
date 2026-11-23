# Phụ trách: TV1 – Đoàn Trung Kiên
"""Sinh EDA FER2013 và CSV nguồn cho từng hình từ bộ dữ liệu thật."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.augment import build_augmentation
from src.data.fer2013 import EMOTIONS, load_fer2013

SPLITS = ("train", "val", "test")
SPLIT_LABELS = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def _write_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, png_path: Path, csv_path: Path, fields: list[str], rows: list[dict]) -> None:
    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    _write_rows(csv_path, fields, rows)


def _sample_indices(labels: np.ndarray, per_class: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in range(len(EMOTIONS)):
        found = np.flatnonzero(labels == label)
        if len(found):
            selected.extend(rng.choice(found, size=min(per_class, len(found)), replace=False).tolist())
    return selected


def generate_eda(
    data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str | Path,
    *,
    seed: int = 0,
    samples_per_class: int = 3,
) -> list[Path]:
    """Tạo năm hình và các CSV dữ liệu tương ứng; không ghi vào tracking."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    expected = set(SPLITS)
    if set(data) != expected:
        raise ValueError(f"Cần đủ split {sorted(expected)}")
    for split, (images, labels) in data.items():
        if images.ndim != 3 or images.shape[1:] != (48, 48) or images.dtype != np.uint8:
            raise ValueError(f"Ảnh {split} phải có dạng uint8 (N, 48, 48)")
        if labels.ndim != 1 or len(images) != len(labels):
            raise ValueError(f"Nhãn {split} không khớp số ảnh")

    created: list[Path] = []

    # Phân bố lớp theo ba split.
    count_rows = [
        {"split": SPLIT_LABELS[split], "label": i, "emotion": EMOTIONS[i], "count": int((labels == i).sum())}
        for split, (_, labels) in data.items()
        for i in range(len(EMOTIONS))
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(EMOTIONS))
    width = 0.25
    for j, split in enumerate(SPLITS):
        _, labels = data[split]
        counts = [(labels == i).sum() for i in range(len(EMOTIONS))]
        ax.bar(x + (j - 1) * width, counts, width, label=SPLIT_LABELS[split])
    ax.set_xticks(x, EMOTIONS, rotation=20, ha="right")
    ax.set_ylabel("Số ảnh")
    ax.set_title("Phân bố nhãn FER2013 theo split")
    ax.legend()
    paths = [output / "eda_class_distribution.png", output / "eda_class_distribution.csv"]
    _save(fig, *paths, ["split", "label", "emotion", "count"], count_rows)
    created.extend(paths)

    # Ảnh mẫu có nhãn; CSV chứa từng pixel để làm nguồn trực tiếp của lưới ảnh.
    sample_rows: list[dict] = []
    selected_by_split: dict[str, list[int]] = {}
    for split in SPLITS:
        images, labels = data[split]
        selected = _sample_indices(labels, samples_per_class, seed)
        selected_by_split[split] = selected
        for idx in selected:
            label = int(labels[idx])
            sample_rows.extend(
                {"split": SPLIT_LABELS[split], "index_in_split": idx, "label": label,
                 "emotion": EMOTIONS[label], "y": y, "x": x, "pixel": int(images[idx, y, x])}
                for y in range(48) for x in range(48)
            )
    columns = max(1, max((len(v) for v in selected_by_split.values()), default=0))
    fig, axes = plt.subplots(len(SPLITS), columns, figsize=(max(8, columns * 1.4), 6), squeeze=False)
    for row_idx, split in enumerate(SPLITS):
        images, labels = data[split]
        selected = selected_by_split[split]
        for col_idx in range(columns):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(selected):
                idx = selected[col_idx]
                label = int(labels[idx])
                ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{SPLIT_LABELS[split]}\n{EMOTIONS[label]} (y={label})", fontsize=8)
    paths = [output / "eda_samples.png", output / "eda_samples.csv"]
    _save(fig, *paths, ["split", "index_in_split", "label", "emotion", "y", "x", "pixel"], sample_rows)
    created.extend(paths)

    # Ảnh trung bình theo lớp, chỉ tính trên Training.
    train_images, train_labels = data["train"]
    mean_rows: list[dict] = []
    fig, axes = plt.subplots(1, len(EMOTIONS), figsize=(14, 2.5))
    for label, ax in enumerate(axes):
        members = train_images[train_labels == label]
        ax.axis("off")
        if len(members):
            mean_image = members.mean(axis=0, dtype=np.float64)
            ax.imshow(mean_image, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{EMOTIONS[label]}\nn={len(members)}", fontsize=8)
            mean_rows.extend(
                {"split": "Training", "label": label, "emotion": EMOTIONS[label],
                 "y": y, "x": x, "mean_pixel": round(float(mean_image[y, x]), 6)}
                for y in range(48) for x in range(48)
            )
        else:
            ax.set_title(f"{EMOTIONS[label]}\nchưa có mẫu", fontsize=8)
    paths = [output / "eda_mean_images.png", output / "eda_mean_images.csv"]
    _save(fig, *paths, ["split", "label", "emotion", "y", "x", "mean_pixel"], mean_rows)
    created.extend(paths)

    # Độ sáng và ảnh toàn đen/trắng theo split.
    quality_rows = []
    for split, (images, _) in data.items():
        pixel_means = images.astype(np.float32).mean(axis=(1, 2))
        quality_rows.append({
            "split": SPLIT_LABELS[split], "n_images": len(images),
            "mean_brightness_0_255": round(float(pixel_means.mean()), 6) if len(images) else "",
            "black_images": int(np.all(images == 0, axis=(1, 2)).sum()),
            "white_images": int(np.all(images == 255, axis=(1, 2)).sum()),
            "near_black_images": int(((images <= 2).mean(axis=(1, 2)) >= 0.99).sum()),
            "near_white_images": int(((images >= 253).mean(axis=(1, 2)) >= 0.99).sum()),
        })
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [r["split"] for r in quality_rows]
    brightness = [r["mean_brightness_0_255"] or 0 for r in quality_rows]
    ax.bar(labels, brightness, color="#4c78a8")
    ax.set_ylabel("Cường độ pixel trung bình (0–255)")
    ax.set_title("Độ sáng trung bình theo split")
    for i, row in enumerate(quality_rows):
        ax.text(i, brightness[i], f"đen={row['black_images']}  trắng={row['white_images']}", ha="center", va="bottom", fontsize=8)
    paths = [output / "eda_image_quality.png", output / "eda_image_quality.csv"]
    _save(fig, *paths, ["split", "n_images", "mean_brightness_0_255", "black_images", "white_images",
                        "near_black_images", "near_white_images"], quality_rows)
    created.extend(paths)

    # Minh họa các augmentation trên ảnh Training; cố định seed và khôi phục RNG toàn cục.
    available = [i for i in range(len(EMOTIONS)) if np.any(train_labels == i)]
    chosen = [int(np.flatnonzero(train_labels == i)[0]) for i in available]
    variants = (
        ("gốc", None),
        ("flip_affine", build_augmentation("flip_affine")),
        ("random_erasing", build_augmentation({"mode": "random_erasing", "p": 1.0})),
    )
    aug_rows: list[dict] = []
    fig, axes = plt.subplots(max(1, len(chosen)), len(variants), figsize=(7, max(2.2, len(chosen) * 1.15)), squeeze=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for row_idx, idx in enumerate(chosen):
            image = torch.from_numpy(train_images[idx].copy()).float().unsqueeze(0).div_(255.0)
            label = int(train_labels[idx])
            for col_idx, (variant, transform) in enumerate(variants):
                changed = image if transform is None else transform(image)
                ax = axes[row_idx, col_idx]
                ax.imshow(changed.squeeze(0).cpu().numpy(), cmap="gray", vmin=0, vmax=1)
                ax.axis("off")
                ax.set_title(f"{EMOTIONS[label]}: {variant}", fontsize=8)
                pixels = changed.squeeze(0).mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
                aug_rows.extend(
                    {"source_index_in_split": idx, "label": label, "emotion": EMOTIONS[label],
                     "variant": variant, "y": y, "x": x, "pixel": int(pixels[y, x])}
                    for y in range(48) for x in range(48)
                )
    paths = [output / "eda_augmentation.png", output / "eda_augmentation.csv"]
    _save(fig, *paths, ["source_index_in_split", "label", "emotion", "variant", "y", "x", "pixel"], aug_rows)
    created.extend(paths)

    return created


def run_eda(csv_path: str | Path, output_dir: str | Path, *, seed: int = 0) -> list[Path]:
    """Nạp FER2013 cục bộ và tạo các tệp EDA."""
    return generate_eda(load_fer2013(csv_path), output_dir, seed=seed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_ROOT / "data" / "fer2013.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT / "figures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not args.csv.is_file():
        parser.error(f"Không tìm thấy FER2013 CSV: {args.csv}; EDA cần dữ liệu gốc thật.")
    created = run_eda(args.csv, args.output_dir, seed=args.seed)
    print(f"Đã tạo {len(created) // 2} hình và CSV nguồn trong {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
