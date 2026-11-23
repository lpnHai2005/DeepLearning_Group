# Phụ trách: TV1 – Đoàn Trung Kiên
"""Unit test cho EDA TV1; fixture không xuất ra ngoài thư mục tạm."""

import csv

from src.data.eda import generate_eda
from src.data.fer2013 import load_fer2013


def _pixels(value):
    return " ".join([str(value)] * (48 * 48))


def test_eda_writes_png_and_source_csv_for_all_figures(tmp_path):
    csv_path = tmp_path / "fer2013_fixture.csv"
    rows = [
        {"emotion": 0, "pixels": _pixels(0), "Usage": "Training"},
        {"emotion": 1, "pixels": _pixels(255), "Usage": "Training"},
        {"emotion": 2, "pixels": _pixels(128), "Usage": "PublicTest"},
        {"emotion": 3, "pixels": _pixels(200), "Usage": "PrivateTest"},
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pixels", "Usage"])
        writer.writeheader()
        writer.writerows(rows)

    output = tmp_path / "eda"
    created = generate_eda(load_fer2013(csv_path), output, seed=5, samples_per_class=1)

    assert len(created) == 10
    assert all(path.is_file() and path.stat().st_size > 0 for path in created)
    assert {path.name for path in created} == {
        "eda_class_distribution.png", "eda_class_distribution.csv",
        "eda_samples.png", "eda_samples.csv",
        "eda_mean_images.png", "eda_mean_images.csv",
        "eda_image_quality.png", "eda_image_quality.csv",
        "eda_augmentation.png", "eda_augmentation.csv",
    }
    with (output / "eda_class_distribution.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 21
    assert sum(int(row["count"]) for row in rows if row["split"] == "Training") == 2
    with (output / "eda_samples.csv").open(encoding="utf-8", newline="") as handle:
        sample_pixels = list(csv.DictReader(handle))
    assert len(sample_pixels) == 4 * 48 * 48
    assert {row["split"] for row in sample_pixels} == {"Training", "PublicTest", "PrivateTest"}
