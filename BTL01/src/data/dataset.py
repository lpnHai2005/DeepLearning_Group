"""Đọc CSV, scale, tạo DataLoader. (TV1 – T1.4)

load_split trả về numpy ở thang GỐC. make_loader scale x và y trước khi đưa vào mô hình.
"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.gaussian import X_SCALE, Y_SCALE

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SPLIT_FILES = {
    "train_pool": "train_pool.csv",
    "val": "val.csv",
    "test": "test.csv",
}


def load_split(name):
    """name in {'train_pool', 'val', 'test'} -> (x: (N,2), y: (N,))."""
    if name not in SPLIT_FILES:
        raise ValueError(f"name phải thuộc {tuple(SPLIT_FILES)}, nhận được {name!r}")

    path = DATA_DIR / SPLIT_FILES[name]
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy split {name!r}: {path}. Chạy src/generate_data.py trước.")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["x1", "x2", "y"]:
            raise ValueError(f"CSV {path} cần có header x1,x2,y")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Split {name!r} không có mẫu: {path}")
    try:
        values = np.asarray([[float(row["x1"]), float(row["x2"]), float(row["y"])] for row in rows],
                            dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV chứa giá trị không phải số: {path}") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"CSV chứa NaN hoặc vô cực: {path}")
    return values[:, :2].copy(), values[:, 2].copy()


def subset(x, y, n):
    """Lấy n mẫu đầu của pool (tập con lồng nhau cho Thực nghiệm A)."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim != 2 or x.shape[1] != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x phải có shape (N, 2), y phải có shape (N,), cùng số mẫu")
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
        raise TypeError("n phải là số nguyên")
    if n < 0 or n > len(x):
        raise ValueError(f"n phải nằm trong [0, {len(x)}], nhận được {n}")
    return x[:n], y[:n]


def make_loader(x, y, batch_size, shuffle, seed):
    """x, y thang gốc -> DataLoader (đã scale), shuffle dùng generator có seed."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x phải có shape (N, 2), y phải có shape (N,), cùng số mẫu")
    if len(x) == 0:
        raise ValueError("Không thể tạo DataLoader rỗng")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x/y phải chứa giá trị hữu hạn")
    if not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, (bool, np.bool_)):
        raise TypeError("batch_size phải là số nguyên")
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed phải là số nguyên")

    x_scaled = (x / X_SCALE).astype(np.float32, copy=False)
    y_scaled = (y / Y_SCALE).astype(np.float32, copy=False)
    dataset = TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(y_scaled))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), generator=generator)
