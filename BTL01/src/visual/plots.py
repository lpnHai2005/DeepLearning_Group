"""Hàm vẽ dùng chung, một style duy nhất (màu train/val/test cố định, nhãn trục, 300 dpi). (TV3 – T3.2)

Lưu ý Matplotlib >= 3.7: dùng fig.add_subplot(projection="3d"), KHÔNG dùng fig.gca(projection="3d").
Cung cấp: loss curve, y–ŷ (parity), bề mặt thật vs xấp xỉ (cùng trục z), bản đồ sai số |y−ŷ|,
đồ thị lỗi theo biến khảo sát (A/B/C) có error bar, và thời gian huấn luyện.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.gaussian import X1_RANGE, X2_RANGE, Y_SCALE, gaussian_pdf
from src.engine.evaluate import metrics, predict_grid

SPLIT_COLORS = {
    "train": "#2563eb",
    "val": "#f97316",
    "test": "#16a34a",
    "baseline": "#4b5563",
    "gap": "#7c3aed",
}
DPI = 300
Z_LIMITS = (0.0, 0.28)


def _ensure_parent(out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _load_history_arrays(history):
    """Nhận đường dẫn CSV, list[dict] hoặc dict[str, array] -> (epochs, train_loss, val_loss)."""
    if isinstance(history, (str, Path)):
        with Path(history).open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        epochs = np.asarray([int(r["epoch"]) for r in rows], dtype=np.int64)
        train_loss = np.asarray([float(r["train_loss"]) for r in rows], dtype=np.float64)
        val_loss = np.asarray([float(r["val_loss"]) for r in rows], dtype=np.float64)
        return epochs, train_loss, val_loss
    if isinstance(history, list):
        epochs = np.asarray([int(r.get("epoch", i + 1)) for i, r in enumerate(history)], dtype=np.int64)
        train_loss = np.asarray([float(r["train_loss"]) for r in history], dtype=np.float64)
        val_loss = np.asarray([float(r["val_loss"]) for r in history], dtype=np.float64)
        return epochs, train_loss, val_loss
    if isinstance(history, dict):
        train_loss = np.asarray(history["train_loss"], dtype=np.float64)
        val_loss = np.asarray(history["val_loss"], dtype=np.float64)
        epochs = np.asarray(history.get("epoch", np.arange(1, len(train_loss) + 1)), dtype=np.int64)
        return epochs, train_loss, val_loss
    raise TypeError("history phải là đường dẫn CSV, list[dict] hoặc dict")


def plot_loss_curve(history, out_path, best_epoch=None, scale_to_original=True,
                    title="Training and validation loss across epochs"):
    """Vẽ đường hội tụ train/val loss theo epoch trên thang log."""
    out_path = _ensure_parent(out_path)
    epochs, train_loss, val_loss = _load_history_arrays(history)
    factor = (Y_SCALE ** 2) if scale_to_original else 1.0
    tr = train_loss * factor
    va = val_loss * factor

    if best_epoch is None and len(va) > 0:
        best_epoch = int(epochs[int(np.argmin(va))])

    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    ax.plot(epochs, tr, color=SPLIT_COLORS["train"], lw=1.6, alpha=0.88, label="Train MSE")
    ax.plot(epochs, va, color=SPLIT_COLORS["val"], lw=1.6, alpha=0.88, label="Validation MSE")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="#dc2626", ls="--", lw=1.2, label=f"Best epoch ({best_epoch})")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on original y scale (log)" if scale_to_original else "Scaled MSE (log)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_parity(y_true, y_pred, out_path, split="test", title=None):
    """Vẽ biểu đồ phân tán y thật vs y dự đoán kèm đường y = x và hộp chỉ số."""
    out_path = _ensure_parent(out_path)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    m = metrics(y_true, y_pred)

    color = SPLIT_COLORS.get(split, SPLIT_COLORS["test"])
    lo = min(0.0, float(min(y_true.min(), y_pred.min())))
    hi = max(float(Y_SCALE), float(max(y_true.max(), y_pred.max()))) * 1.04

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    ax1.plot([lo, hi], [lo, hi], ls="--", color="#111827", lw=1.3, label="Ideal (ŷ = y)")
    ax1.scatter(y_true, y_pred, s=26, alpha=0.78, color=color, edgecolors="white", linewidth=0.4,
                label=f"{split.capitalize()} samples (n={len(y_true)})")
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("True density y = f(x1, x2)")
    ax1.set_ylabel("Predicted density ŷ")
    ax1.set_title(title or f"True vs. predicted density ({split} set)")
    ax1.grid(True, alpha=0.22)
    ax1.legend(loc="upper left")

    stats_text = (
        f"MSE  = {m['mse']:.3e}\n"
        f"RMSE = {m['rmse']:.3e}\n"
        f"MAE  = {m['mae']:.3e}\n"
        f"R²   = {m['r2']:.6f}"
    )
    ax1.text(0.96, 0.06, stats_text, transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=9.5, family="monospace",
             bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"})

    residuals = y_pred - y_true
    ax2.axhline(0.0, ls="--", color="#111827", lw=1.2)
    ax2.scatter(y_true, residuals, s=26, alpha=0.78, color=color, edgecolors="white", linewidth=0.4)
    ax2.set_xlabel("True density y = f(x1, x2)")
    ax2.set_ylabel("Residual (ŷ − y)")
    ax2.set_title(f"Residuals vs. true density ({split} set)")
    ax2.grid(True, alpha=0.22)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_surface_comparison(model, out_path, n=100, title=None):
    """Vẽ bề mặt 3D hàm Gaussian thật và bề mặt MLP xấp xỉ trên cùng giới hạn trục z."""
    out_path = _ensure_parent(out_path)
    grid_x1, grid_x2, y_hat = predict_grid(model, n=n)
    points = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    y_true = gaussian_pdf(points).reshape(grid_x1.shape)
    m = metrics(y_true.ravel(), y_hat.ravel())

    fig = plt.figure(figsize=(13.5, 5.8), constrained_layout=True)
    ax_true = fig.add_subplot(1, 2, 1, projection="3d")
    surf_true = ax_true.plot_surface(
        grid_x1, grid_x2, y_true, cmap="viridis", vmin=0.0, vmax=Y_SCALE,
        linewidth=0, antialiased=True
    )
    ax_true.set_xlim(*X1_RANGE)
    ax_true.set_ylim(*X2_RANGE)
    ax_true.set_zlim(*Z_LIMITS)
    ax_true.set_xlabel("x1")
    ax_true.set_ylabel("x2")
    ax_true.set_zlabel("Density f(x1, x2)")
    ax_true.set_title("True 2D Gaussian surface f(x1, x2)")

    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
    surf_pred = ax_pred.plot_surface(
        grid_x1, grid_x2, y_hat, cmap="viridis", vmin=0.0, vmax=Y_SCALE,
        linewidth=0, antialiased=True
    )
    ax_pred.set_xlim(*X1_RANGE)
    ax_pred.set_ylim(*X2_RANGE)
    ax_pred.set_zlim(*Z_LIMITS)
    ax_pred.set_xlabel("x1")
    ax_pred.set_ylabel("x2")
    ax_pred.set_zlabel("Predicted ŷ(x1, x2)")
    ax_pred.set_title(
        title or f"MLP approximation (grid MSE={m['mse']:.2e}, R²={m['r2']:.5f})"
    )
    fig.colorbar(surf_pred, ax=[ax_true, ax_pred], shrink=0.68, pad=0.04, label="Density")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_error_map(model, out_path, n=100, sample_x=None, title=None):
    """Vẽ đường đồng mức thật vs xấp xỉ (trái) và bản đồ nhiệt sai số tuyệt đối |y − ŷ| (phải)."""
    out_path = _ensure_parent(out_path)
    grid_x1, grid_x2, y_hat = predict_grid(model, n=n)
    points = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    y_true = gaussian_pdf(points).reshape(grid_x1.shape)
    abs_err = np.abs(y_hat - y_true)
    m = metrics(y_true.ravel(), y_hat.ravel())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=True)

    levels = np.linspace(0.02, 0.25, 8)
    c_true = ax1.contour(grid_x1, grid_x2, y_true, levels=levels, colors="#1e293b", linewidths=1.2)
    c_pred = ax1.contour(grid_x1, grid_x2, y_hat, levels=levels, colors="#f97316", linestyles="--", linewidths=1.2)
    ax1.clabel(c_true, inline=True, fontsize=7.5, fmt="%.2f")
    ax1.plot([], [], color="#1e293b", lw=1.2, label="True contours f(x)")
    ax1.plot([], [], color="#f97316", lw=1.2, ls="--", label="Predicted contours ŷ(x)")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_title("Iso-density contours: True vs. MLP")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper right", fontsize=8.5)

    cf = ax2.contourf(grid_x1, grid_x2, abs_err, levels=28, cmap="magma")
    if sample_x is not None:
        sample_x = np.asarray(sample_x, dtype=np.float64)
        ax2.scatter(sample_x[:, 0], sample_x[:, 1], s=14, c="#38bdf8", edgecolors="black",
                    linewidth=0.35, alpha=0.85, label="Test samples (n=100)")
        ax2.legend(loc="upper right", fontsize=8.5)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_title(
        title or f"Absolute error |y − ŷ| (max={abs_err.max():.2e}, MAE={m['mae']:.2e})"
    )
    fig.colorbar(cf, ax=ax2, label="Absolute error |y − ŷ|")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_error_vs_param(x, train_mean, train_std, val_mean, val_std, out_path,
                        xlabel="", ylabel="MSE on original y scale (log)",
                        title="", baseline_mse=None, xticks=None):
    """Vẽ lỗi train/val (mean ± std) theo biến khảo sát (A, B hoặc C) trên thang log."""
    out_path = _ensure_parent(out_path)
    x = np.asarray(x)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.errorbar(x, train_mean, yerr=train_std, label="Train MSE",
                color=SPLIT_COLORS["train"], marker="o", capsize=3, lw=1.6)
    ax.errorbar(x, val_mean, yerr=val_std, label="Validation MSE",
                color=SPLIT_COLORS["val"], marker="s", capsize=3, lw=1.6)
    if baseline_mse is not None:
        if np.ndim(baseline_mse) == 0:
            ax.axhline(float(baseline_mse), ls="--", color=SPLIT_COLORS["baseline"],
                       label="Constant mean predictor (val)")
        else:
            ax.plot(x, baseline_mse, ls="--", color=SPLIT_COLORS["baseline"],
                    label="Constant mean predictor (val)")
    ax.set_yscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path
