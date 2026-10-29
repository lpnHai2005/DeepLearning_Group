"""Dự đoán và tính chỉ số trên thang y GỐC. (TV3 – T3.1)"""


def predict(model, x):
    """x: (N,2) thang gốc -> y_hat: (N,) thang gốc."""
    raise NotImplementedError("TV3 - T3.1")


def metrics(y, y_hat):
    """-> dict(mse, mae, rmse, r2)."""
    raise NotImplementedError("TV3 - T3.1")


def predict_grid(model, n=100):
    """Dự đoán trên lưới n×n phủ miền; trả (X1, X2, Y_hat)."""
    raise NotImplementedError("TV3 - T3.1")
