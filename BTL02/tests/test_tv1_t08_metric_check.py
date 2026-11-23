# Phụ trách: TV1 – Đoàn Trung Kiên
"""Đối chiếu metric T08 với sklearn trên nhãn giả lập trong unit test."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data.fer2013 import EMOTIONS
from src.engine.evaluate import metrics


def test_metrics_match_sklearn_and_keep_kaggle_label_order():
    y_true = np.array([0, 0, 1, 2, 2, 2, 4, 6], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1, 2, 2, 6, 6], dtype=np.int64)
    result = metrics(y_true, y_pred)

    assert result["acc"] == pytest.approx(accuracy_score(y_true, y_pred))
    assert result["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, labels=np.arange(7), average="macro", zero_division=0)
    )
    assert np.array_equal(result["cm"], confusion_matrix(y_true, y_pred, labels=np.arange(7)))
    assert result["labels"] == EMOTIONS
    assert list(result["per_class"]) == EMOTIONS
