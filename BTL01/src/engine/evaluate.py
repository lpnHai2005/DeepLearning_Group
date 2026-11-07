"""Dự đoán và tính chỉ số trên thang y gốc. (TV3 – T3.1)"""
import numpy as np
import torch

from src.data.gaussian import X1_RANGE, X2_RANGE, X_SCALE, Y_SCALE, gaussian_pdf


def predict(model, x):
    """x: (N,2) thang gốc -> y_hat: (N,) thang gốc."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x phải có shape (N, 2), nhận được {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("x phải chứa các giá trị hữu hạn")

    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model phải có ít nhất một tham số để xác định device") from exc

    was_training = model.training
    model.eval()
    with torch.no_grad():
        x_scaled = torch.as_tensor(x / X_SCALE, dtype=torch.float32, device=device)
        y_scaled = model(x_scaled)
        if y_scaled.ndim != 1 or y_scaled.shape[0] != len(x):
            raise ValueError(f"model phải trả shape (N,), nhận được {tuple(y_scaled.shape)}")
        y_hat = y_scaled.detach().cpu().numpy().astype(np.float64) * Y_SCALE
    model.train(was_training)
    return y_hat


def metrics(y, y_hat):
    """-> dict(mse, mae, rmse, r2)."""
    y = np.asarray(y, dtype=np.float64)
    y_hat = np.asarray(y_hat, dtype=np.float64)
    if y.ndim != 1 or y_hat.ndim != 1 or y.shape != y_hat.shape:
        raise ValueError(f"y và y_hat phải cùng shape (N,), nhận được {y.shape} và {y_hat.shape}")
    if y.size == 0 or not np.isfinite(y).all() or not np.isfinite(y_hat).all():
        raise ValueError("y và y_hat phải không rỗng và chứa các giá trị hữu hạn")

    residual = y_hat - y
    mse = float(np.mean(residual ** 2))
    mae = float(np.mean(np.abs(residual)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum(residual ** 2))
    r2 = 1.0 if ss_tot == 0.0 and ss_res == 0.0 else (0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot)
    return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse)), "r2": float(r2)}


def predict_grid(model, n=100):
    """Dự đoán trên lưới n×n phủ miền; trả (X1, X2, Y_hat)."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 2:
        raise ValueError("n phải là số nguyên ít nhất bằng 2")
    x1 = np.linspace(*X1_RANGE, int(n))
    x2 = np.linspace(*X2_RANGE, int(n))
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    points = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    grid_y = predict(model, points).reshape(grid_x1.shape)
    return grid_x1, grid_x2, grid_y


def evaluate_grid(model, n=100):
    """Đánh giá sai số xấp xỉ trên lưới đều n×n toàn miền -> dict(mse, mae, rmse, r2, max_abs_err)."""
    grid_x1, grid_x2, grid_y_hat = predict_grid(model, n=n)
    points = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    y_true = gaussian_pdf(points)
    y_hat = grid_y_hat.ravel()
    out = metrics(y_true, y_hat)
    out["max_abs_err"] = float(np.max(np.abs(y_hat - y_true)))
    return out

