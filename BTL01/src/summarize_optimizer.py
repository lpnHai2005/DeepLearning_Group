"""Chọn lr tốt nhất của từng optimizer theo val, vẽ train/val loss theo epoch; không đọc tập test. (TV2 – T2.8)

Chạy sau grid_optimizer.json:  python src/summarize_optimizer.py
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

from src.data.summarize_expA import mean_std  # noqa: E402
from src.summarize_expC import SMOOTH, mean_curve, read_history, read_summary  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

EXP = ROOT / "results" / "grid_optimizer"
FIG = ROOT / "figures"
STATS = ("train_mse", "val_mse", "best_epoch", "epochs_run")
SERIES = {"sgd": ("SGD", "#2a78d6"), "momentum": ("Momentum", "#eb6834"), "nesterov": ("Nesterov", "#1baf7a"),
          "rmsprop": ("RMSProp", "#eda100"), "adam": ("Adam", "#e87ba4"), "nadam": ("Nadam", "#008300")}


def aggregate(grouped):
    out = []
    for value, runs in grouped.items():
        opt, lr = value.split("_")
        row = {"optimizer": opt, "lr": float(lr), "n_seeds": len(runs)}
        for s in STATS:
            row[f"{s}_mean"], row[f"{s}_std"] = mean_std([float(r[s]) for r in runs])
        out.append(row)
    return out


def best_per_optimizer(rows):
    """Mỗi optimizer lấy lr có mean val MSE thấp nhất: so sánh ở lr tốt nhất của từng cái, không ở lr chung."""
    return [min((r for r in rows if r["optimizer"] == opt), key=lambda r: r["val_mse_mean"]) for opt in SERIES]


def plot_curves(best, grouped, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    for r in best:
        value = f"{r['optimizer']}_{r['lr']:g}"
        hists = [read_history(EXP / value / f"seed_{run['seed']}") for run in grouped[value]]
        label, color = SERIES[r["optimizer"]]
        for ax, key in zip(axes, ("train_loss", "val_loss")):
            y = mean_curve(hists, key)
            ax.plot(np.arange(1, len(y) + 1), y, color=color, lw=2, label=f"{label} (lr={r['lr']:g})")
    for ax, title in zip(axes, ("Train loss", "Validation loss")):
        ax.set(yscale="log", xlabel="Epoch", title=title)
        ax.grid(True, which="major", alpha=0.22)
    axes[0].set_ylabel("MSE on original y scale (log)")
    fig.suptitle(f"Best lr per optimizer: mean of 3 seeds, {SMOOTH}-epoch rolling median")
    axes[0].legend()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    grouped = read_summary(EXP)
    rows = aggregate(grouped)
    best = best_per_optimizer(rows)
    fields = ["optimizer", "lr", "n_seeds"] + [f"{s}_{t}" for s in STATS for t in ("mean", "std")]
    write_csv(EXP / "aggregate.csv", rows, fields)
    FIG.mkdir(parents=True, exist_ok=True)
    plot_curves(best, grouped, FIG / "optimizer_loss.png")

    lrs = sorted({r["lr"] for r in rows})
    for r in sorted(best, key=lambda r: r["val_mse_mean"]):
        edge = "  <- lr o bien luoi" if r["lr"] in (lrs[0], lrs[-1]) else ""
        print(f"{r['optimizer']:9s} lr={r['lr']:<6g} val_mse={r['val_mse_mean']:.3e}±{r['val_mse_std']:.1e}  "
              f"train_mse={r['train_mse_mean']:.3e}  best_epoch={r['best_epoch_mean']:.0f}{edge}")


if __name__ == "__main__":
    main()
