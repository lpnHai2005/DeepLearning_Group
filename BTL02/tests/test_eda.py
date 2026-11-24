# Phụ trách: TV1 – Đoàn Trung Kiên
"""Unit test cho EDA TV1; fixture không xuất ra ngoài thư mục tạm."""

import csv

from src.data.eda import generate_eda, register_eda_figures
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
    assert {"percent"}.issubset(rows[0])
    totals = {}
    for split in {row["split"] for row in rows}:
        totals[split] = sum(float(row["percent"]) for row in rows if row["split"] == split)
    assert all(abs(total - 100.0) < 1e-5 for total in totals.values())
    with (output / "eda_samples.csv").open(encoding="utf-8", newline="") as handle:
        sample_pixels = list(csv.DictReader(handle))
    assert len(sample_pixels) == 4 * 48 * 48
    assert {row["split"] for row in sample_pixels} == {"Training", "PublicTest", "PrivateTest"}
    with (output / "eda_image_quality.csv").open(encoding="utf-8", newline="") as handle:
        quality = {row["split"]: row for row in csv.DictReader(handle)}
    assert quality["Training"]["black_images"] == "1"
    assert quality["Training"]["white_images"] == "1"
    assert quality["PublicTest"]["mean_brightness_0_255"] == "128.0"


def test_register_eda_figures_ghi_moi_hinh_mot_dong(tmp_path):
    from src.utils.io import sha256_file

    out = tmp_path / "figures"
    out.mkdir()
    created = []
    for name in ("eda_a", "eda_b"):
        (out / f"{name}.png").write_bytes(name.encode())
        (out / f"{name}.csv").write_text("x\n1\n", encoding="utf-8")
        created += [out / f"{name}.png", out / f"{name}.csv"]
    manifest = tmp_path / "tracking" / "figures_manifest.csv"
    for _ in range(2):  # chạy lại không nhân đôi dòng
        register_eda_figures(created, manifest, tmp_path / "data" / "fer2013.csv", out, seed=3, root=tmp_path)
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = {row["fig_id"]: row for row in csv.DictReader(handle)}
    assert set(rows) == {"eda_a", "eda_b"}
    assert rows["eda_a"]["file_hinh"] == "figures/eda_a.png" and rows["eda_a"]["file_du_lieu_dau_vao"] == "figures/eda_a.csv"
    assert rows["eda_a"]["task_id"] == "T05" and rows["eda_a"]["sha256_hinh"] == sha256_file(out / "eda_a.png")
    assert rows["eda_a"]["lenh_chay"].endswith("--csv data/fer2013.csv --output-dir figures --seed 3")
