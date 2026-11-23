"""Tiện ích ghi file an toàn. Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T07."""
import csv
import hashlib
import json
import os
from pathlib import Path

import torch

BTL02 = Path(__file__).resolve().parents[2]


def rel(path, root=BTL02):
    """Đường dẫn tương đối (dấu /) từ BTL02; ngoài BTL02 thì giữ nguyên tuyệt đối."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    """Ghi qua file tạm rồi đổi tên: không để lại file JSON dở khi bị ngắt."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_csv_row(path, header, row):
    """Nối 1 dòng vào CSV (tạo kèm header nếu chưa có). Cột lạ trong row gây lỗi, cột thiếu để trống."""
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) not in (b"\n", b"\r")
        if needs_newline:
            with open(path, "a", encoding="utf-8", newline="") as f:
                f.write("\n")
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
