"""Đọc/ghi config JSON, history CSV, summary.csv; đường dẫn tính từ vị trí file. (TV2)"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path, rows, fields=None):
    """rows: list[dict]. fields cố định thứ tự cột; khóa thiếu để trống, khóa thừa bỏ qua."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]), restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
