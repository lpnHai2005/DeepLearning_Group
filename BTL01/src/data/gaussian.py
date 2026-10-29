"""Hàm Gaussian 2D cần xấp xỉ và các hằng số của bài toán. (TV1 – T1.2)

Hằng số là của đề bài, không fit từ dữ liệu nên dùng để scale không gây rò rỉ.
"""
import numpy as np

MU = np.zeros(2)
C = np.array([[0.3, -0.5], [-0.5, 2.0]])
C_INV = np.linalg.inv(C)
NORM = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(C)))   # = y_max ≈ 0.2690

X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)
X_SCALE = np.array([2.0, 4.0])   # x / X_SCALE -> [-1, 1]
Y_SCALE = NORM                    # y / Y_SCALE -> [0, 1]


def gaussian_pdf(x):
    """x: (N, 2) -> (N,)."""
    raise NotImplementedError("TV1 - T1.2")


def sample_uniform(n, rng):
    """Lấy n điểm đều trên miền chữ nhật. rng: np.random.Generator."""
    raise NotImplementedError("TV1 - T1.3")
