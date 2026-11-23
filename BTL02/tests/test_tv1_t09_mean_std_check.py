# Phụ trách: TV1 – Đoàn Trung Kiên
"""Đối chiếu mean/std của bảng baseline T09 với phép tính pandas độc lập."""

import pandas as pd
import pytest

from src.make_tables import baseline_stats


def test_baseline_mean_and_sample_std_match_pandas():
    rows = [
        {"task_id": "T11", "trang_thai_ky_thuat": "Completed", "val_acc": "0.651"},
        {"task_id": "T11", "trang_thai_ky_thuat": "Completed", "val_acc": "0.664"},
        {"task_id": "T11", "trang_thai_ky_thuat": "Completed", "val_acc": "0.658"},
        {"task_id": "T11", "trang_thai_ky_thuat": "Failed", "val_acc": "0.999"},
    ]
    expected = pd.Series([0.651, 0.664, 0.658], dtype="float64")

    mean, std = baseline_stats(rows)

    assert mean == pytest.approx(expected.mean())
    assert std == pytest.approx(expected.std(ddof=1))
