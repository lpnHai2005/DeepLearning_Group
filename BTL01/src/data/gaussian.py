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
    """Tính mật độ Gaussian cho x có shape (N, 2), trả kết quả shape (N,)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x phải có shape (N, 2), nhận được {x.shape}")

    delta = x - MU
    mahalanobis_sq = np.einsum("ni,ij,nj->n", delta, C_INV, delta)
    return NORM * np.exp(-0.5 * mahalanobis_sq)


def sample_uniform(n, rng):
    """Lấy n điểm đều trên miền chữ nhật. rng: np.random.Generator."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 0:
        raise ValueError("n phải là số nguyên không âm")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng phải là numpy.random.Generator")

    x1 = rng.uniform(X1_RANGE[0], X1_RANGE[1], size=n)
    x2 = rng.uniform(X2_RANGE[0], X2_RANGE[1], size=n)
    return np.column_stack((x1, x2))
