# Phụ trách: TV1 – Đoàn Trung Kiên
"""Kiểm tra cột, Usage, số mẫu và SHA-256 của FER2013 gốc."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

EXPECTED_COUNTS = {"Training": 28_709, "PublicTest": 3_589, "PrivateTest": 3_589}
REQUIRED_COLUMNS = {"emotion", "pixels", "Usage"}
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_fer2013(
    csv_path: str | Path,
    expected_counts: dict[str, int] | None = None,
    downloaded_on: date | None = None,
) -> dict:
    """Đếm Usage và kiểm tra schema; không nạp ảnh vào RAM hoặc sửa dữ liệu."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy FER2013 CSV: {path}")

    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(columns))
        if missing:
            raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}")
        for row_number, row in enumerate(reader, start=2):
            if row.get(None) is not None:
                raise ValueError(f"Dòng {row_number} có nhiều trường hơn header")
            usage = (row.get("Usage") or "").strip()
            if not usage:
                raise ValueError(f"Usage rỗng ở dòng {row_number}")
            counts[usage] += 1

    expected = expected_counts or EXPECTED_COUNTS
    observed = {key: counts.get(key, 0) for key in EXPECTED_COUNTS}
    unexpected = sorted(set(counts) - set(EXPECTED_COUNTS))
    unexpected_columns = sorted(set(columns) - REQUIRED_COLUMNS)
    valid = observed == expected and not unexpected and not unexpected_columns
    return {
        "status": "PASS" if valid else "FAIL",
        "file": path.name,
        "download_date": downloaded_on.isoformat() if downloaded_on else None,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "columns": columns,
        "required_columns": sorted(REQUIRED_COLUMNS),
        "unexpected_columns": unexpected_columns,
        "expected_split_counts": expected,
        "observed_split_counts": observed,
        "unexpected_usage": unexpected,
        "total_rows": sum(counts.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_ROOT / "data" / "fer2013.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "results" / "data_check.json")
    parser.add_argument(
        "--download-date",
        type=date.fromisoformat,
        required=True,
        help="Ngày tải/đưa bản FER2013 này vào dự án, định dạng YYYY-MM-DD.",
    )
    args = parser.parse_args(argv)

    try:
        result = inspect_fer2013(args.csv, downloaded_on=args.download_date)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Lỗi kiểm tra FER2013: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
