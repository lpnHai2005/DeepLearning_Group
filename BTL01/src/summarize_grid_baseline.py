"""Vẽ bằng chứng chọn lr và activation từ lưới baseline; không đọc tập test. (TV2 – T2.5)

Chạy sau grid_baseline.json:  python src/summarize_grid_baseline.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.summarize_expC import read_summary  # noqa: E402

EXP = ROOT / "results" / "grid_baseline"
FIG = ROOT / "figures"
# (nhãn, màu, hệ số lệch trục x để các chấm seed không đè lên nhau)
SERIES = {"sigmoid": ("Sigmoid", "#2a78d6", 0.93), "tanh": ("Tanh", "#eb6834", 1.0), "relu": ("ReLU", "#1baf7a", 1.07)}


def collect(grouped):
    """-> {activation: {lr: mảng val_mse theo seed}}."""
    out = {act: {} for act in SERIES}
    for value, runs in grouped.items():
        lr, act = value.split("_")
        out[act][float(lr)] = np.array([float(r["val_mse"]) for r in runs])
    return out


def plot(data, best, path):
    lrs = sorted(next(iter(data.values())))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for act, (label, color, shift) in SERIES.items():
        x = np.array(lrs) * shift
        ax.plot(x, [data[act][lr].mean() for lr in lrs], color=color, lw=2, marker="o", ms=8, label=label)
        for xi, lr in zip(x, lrs):
            ax.scatter([xi] * len(data[act][lr]), data[act][lr], color=color, s=18, alpha=0.45, zorder=3)
    act, lr = best
    ax.annotate(f"Selected: {SERIES[act][0]}, lr={lr:g}", (lr * SERIES[act][2], data[act][lr].mean()),
                xytext=(-12, -30), textcoords="offset points", ha="right", fontsize=10,
                arrowprops={"arrowstyle": "-", "color": "#52514e"})
    ax.set(xscale="log", yscale="log", xlabel="Learning rate (Adam)", ylabel="Validation MSE, original y scale (log)",
           title="Baseline selection: large marker = mean of 3 seeds, small dots = each seed")
    ax.set_xticks(lrs, [f"{lr:g}" for lr in lrs])
    ax.minorticks_off()
    ax.grid(True, which="major", alpha=0.22)
    ax.legend(title="Activation")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    data = collect(read_summary(EXP))
    assert all(len(v) == 3 and all(len(s) == 3 for s in v.values()) for v in data.values()), "cần đủ 3 lr × 3 seed"
    means = {(act, lr): s.mean() for act, v in data.items() for lr, s in v.items()}
    best = min(means, key=means.get)      # chỉ theo val, trùng configs/baseline.json
    FIG.mkdir(parents=True, exist_ok=True)
    plot(data, best, FIG / "grid_baseline_val_mse.png")
    for (act, lr), m in sorted(means.items(), key=lambda kv: kv[1]):
        print(f"{act:8s} lr={lr:<6g} val_mse={m:.3e}  seeds=" + " ".join(f"{v:.2e}" for v in data[act][lr]))


if __name__ == "__main__":
    main()
