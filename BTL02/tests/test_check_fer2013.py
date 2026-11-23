# Phụ trách: TV1 – Đoàn Trung Kiên
"""Unit tests cho kiểm tra schema, Usage và SHA-256 T03 bằng CSV fixture."""

import csv
import hashlib
from datetime import date

from src.data.check_fer2013 import inspect_fer2013


def _pixels(value=0):
    return " ".join([str(value)] * (48 * 48))


def _fixture(path):
    rows = [
        {"emotion": 0, "pixels": _pixels(), "Usage": "Training"},
        {"emotion": 1, "pixels": _pixels(1), "Usage": "Training"},
        {"emotion": 2, "pixels": _pixels(2), "Usage": "PublicTest"},
        {"emotion": 3, "pixels": _pixels(3), "Usage": "PrivateTest"},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pixels", "Usage"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_check_reports_split_counts_and_sha256_for_fixture(tmp_path):
    path = _fixture(tmp_path / "fixture.csv")
    report = inspect_fer2013(path, expected_counts={"Training": 2, "PublicTest": 1, "PrivateTest": 1})

    assert report["status"] == "PASS"
    assert report["total_rows"] == 4
    assert report["observed_split_counts"] == {"Training": 2, "PublicTest": 1, "PrivateTest": 1}
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_check_records_download_date(tmp_path):
    path = _fixture(tmp_path / "fixture.csv")
    report = inspect_fer2013(
        path,
        expected_counts={"Training": 2, "PublicTest": 1, "PrivateTest": 1},
        downloaded_on=date(2026, 11, 24),
    )

    assert report["download_date"] == "2026-11-24"


def test_check_flags_unexpected_split(tmp_path):
    path = _fixture(tmp_path / "fixture.csv")
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([0, _pixels(), "Unknown"])
    report = inspect_fer2013(path, expected_counts={"Training": 2, "PublicTest": 1, "PrivateTest": 1})

    assert report["status"] == "FAIL"
    assert report["unexpected_usage"] == ["Unknown"]


def test_check_flags_unexpected_column(tmp_path):
    path = tmp_path / "extra_column.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pixels", "Usage", "extra"])
        writer.writeheader()
        writer.writerow({"emotion": 0, "pixels": _pixels(), "Usage": "Training", "extra": "x"})
        writer.writerow({"emotion": 0, "pixels": _pixels(), "Usage": "PublicTest", "extra": "x"})
        writer.writerow({"emotion": 0, "pixels": _pixels(), "Usage": "PrivateTest", "extra": "x"})
    report = inspect_fer2013(
        path, expected_counts={"Training": 1, "PublicTest": 1, "PrivateTest": 1}
    )

    assert report["status"] == "FAIL"
    assert report["unexpected_columns"] == ["extra"]
