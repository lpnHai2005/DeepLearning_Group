"""Tổng hợp và vẽ kết quả Thực nghiệm C (độ sâu) + đối chứng cùng số tham số; không đọc tập test. (TV2 – T2.6)

Chạy sau expC.json và expC_ctrl.json:  python src/summarize_expC.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_split, subset  # noqa: E402
from src.data.gaussian import Y_SCALE  # noqa: E402
from src.data.summarize_expA import mean_std  # noqa: E402
from src.engine.evaluate import metrics  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "figures"
STATS = ("train_mse", "val_mse", "train_mae", "val_mae", "train_rmse", "val_rmse",
         "best_epoch", "epochs_run", "train_time_s", "time_per_epoch_ms")
N_TRAIN = 1000
COLORS = {"deep": "#2563eb", "wide": "#f97316"}


def read_summary(exp_dir):
    with (exp_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["value"]].append(r)
    return grouped


def aggregate(grouped, family):
    out = []
    for value, runs in grouped.items():
        widths = [int(w) for w in value.split("-")]
        row = {"family": family, "value": value, "depth": len(widths), "width": widths[0],
               "n_params": int(runs[0]["n_params"]), "n_seeds": len(runs)}
        for s in STATS:
            row[f"{s}_mean"], row[f"{s}_std"] = mean_std([float(r[s]) for r in runs])
        out.append(row)
    return sorted(out, key=lambda r: r["n_params"])


def read_history(run_dir):
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: np.asarray([float(r[k]) for r in rows]) for k in rows[0]}


def grad_table(exp_dir, grouped):
    """Chuẩn gradient từng lớp: epoch 1 và trung vị qua các epoch, trung bình 3 seed."""
    out = []
    for value, runs in grouped.items():
        hists = [read_history(exp_dir / value / f"seed_{r['seed']}") for r in runs]
        layers = sorted((k for k in hists[0] if k.startswith("grad_W")), key=lambda k: int(k[6:]))
        for k in layers:
            out.append({"value": value, "layer": int(k[6:]), "n_layers": len(layers),
                        "grad_epoch1": float(np.mean([h[k][0] for h in hists])),
                        "grad_median": float(np.mean([np.median(h[k]) for h in hists]))})
    return out


def plot_error(deep, wide, base_mse, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    c = [r["depth"] for r in deep]
    for split, color, marker in (("train", "#2563eb", "o"), ("val", "#f97316", "s")):
        a.errorbar(c, [r[f"{split}_mse_mean"] for r in deep], yerr=[r[f"{split}_mse_std"] for r in deep],
                   label=f"{split.capitalize()} MSE", color=color, marker=marker, capsize=3)
    a.axhline(base_mse, ls="--", color="#4b5563", label="Constant mean predictor (val)")
    a.set(yscale="log", xticks=c, xlabel="Number of hidden layers C (10 neurons each)",
          ylabel="MSE on original y scale (log)", title="Experiment C: error versus depth")
    a.grid(True, which="both", alpha=0.22)
    a.legend()

    shared = [r for r in deep if r["depth"] == 1]          # C=1 (B=10) là điểm chung của hai họ
    for fam, rows, label in (("deep", deep, "Deeper: C layers × 10"), ("wide", shared + wide, "Wider: 1 layer × B")):
        b.errorbar([r["n_params"] for r in rows], [r["val_mse_mean"] for r in rows],
                   yerr=[r["val_mse_std"] for r in rows], color=COLORS[fam], marker="o", capsize=3, label=label)
        for r in rows:
            b.annotate(r["value"].replace("-", "×") if fam == "deep" and r["depth"] > 1 else f"B={r['width']}",
                       (r["n_params"], r["val_mse_mean"]), textcoords="offset points", xytext=(5, 6), fontsize=8)
    b.set(yscale="log", xlabel="Number of parameters", ylabel="Validation MSE (log)",
          title="Depth versus width at equal parameter count")
    b.grid(True, which="both", alpha=0.22)
    b.legend()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_curves(exp_dir, grouped, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    values = sorted(grouped, key=lambda v: len(v.split("-")))
    for i, value in enumerate(values):
        color = cmap(i / max(1, len(values) - 1))
        for j, r in enumerate(grouped[value]):
            h = read_history(exp_dir / value / f"seed_{r['seed']}")
            ax.plot(h["epoch"], h["val_loss"] * Y_SCALE ** 2, color=color, lw=1.0, alpha=0.8,
                    label=f"C={len(value.split('-'))}" if j == 0 else None)   # đổi loss về thang y gốc
    ax.set(yscale="log", xlabel="Epoch", ylabel="Validation MSE, original y scale (log)",
           title="Experiment C: validation loss per epoch (3 seeds per depth)")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_time(deep, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    c = [r["depth"] for r in deep]
    for ax, key, label in ((a, "train_time_s", "Total training time (s)"),
                           (b, "time_per_epoch_ms", "Time per epoch (ms)")):
        ax.bar(c, [r[f"{key}_mean"] for r in deep], yerr=[r[f"{key}_std"] for r in deep],
               color="#2563eb", capsize=4)
        ax.set(xticks=c, xlabel="Number of hidden layers C", ylabel=label)
        ax.grid(True, axis="y", alpha=0.22)
    a.set_title("Total time (depends on epochs until early stop)")
    b.set_title("Cost per epoch (A=1000, batch 32, 1 CPU thread)")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grads(grads, path):
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    by_value = defaultdict(list)
    for g in grads:
        by_value[g["value"]].append(g)
    values = sorted(by_value, key=lambda v: len(v.split("-")))
    for i, value in enumerate(values):
        rows = sorted(by_value[value], key=lambda g: g["layer"])
        color = cmap(i / max(1, len(values) - 1))
        x = [g["layer"] for g in rows]
        ax.plot(x, [g["grad_median"] for g in rows], marker="o", color=color, label=f"C={len(rows) - 1}, median")
        ax.plot(x, [g["grad_epoch1"] for g in rows], marker="x", ls=":", color=color, label=f"C={len(rows) - 1}, epoch 1")
    ax.set(yscale="log", xlabel="Weight matrix index (1 = first layer, last = output layer)",
           ylabel="‖∂L/∂W‖ (log)", title="Per-layer gradient norm (mean of 3 seeds)")
    ax.set_xticks(range(1, 6))
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    dirs = {"deep": RES / "expC", "wide": RES / "expC_ctrl"}
    grouped = {fam: read_summary(d) for fam, d in dirs.items()}
    deep, wide = aggregate(grouped["deep"], "deep"), aggregate(grouped["wide"], "wide")
    # 4B+1 cho 1 tầng; 30 + 110(C−1) + 11 cho C tầng rộng 10 (CHI_TIET_NHIEM_VU 5.1)
    assert [r["n_params"] for r in deep] == [41 + 110 * (r["depth"] - 1) for r in deep], deep
    assert all(r["n_params"] == 4 * r["width"] + 1 for r in wide), wide

    x_pool, y_pool = load_split("train_pool")
    _, y_tr = subset(x_pool, y_pool, N_TRAIN)
    _, y_val = load_split("val")
    base_mse = metrics(y_val, np.full_like(y_val, y_tr.mean()))["mse"]

    fields = ["family", "value", "depth", "width", "n_params", "n_seeds"]
    fields += [f"{s}_{t}" for s in STATS for t in ("mean", "std")]
    write_csv(dirs["deep"] / "aggregate.csv", deep + wide, fields)
    grads = grad_table(dirs["deep"], grouped["deep"])
    write_csv(dirs["deep"] / "grad_norms.csv", grads)

    FIG.mkdir(parents=True, exist_ok=True)
    plot_error(deep, wide, base_mse, FIG / "expC_error.png")
    plot_curves(dirs["deep"], grouped["deep"], FIG / "expC_val_curves.png")
    plot_time(deep, FIG / "expC_time.png")
    plot_grads(grads, FIG / "expC_grad_norms.png")
    print(f"constant-mean val MSE = {base_mse:.4e}")
    for r in deep + wide:
        print(f"{r['family']:4s} {r['value']:12s} params={r['n_params']:3d}  "
              f"val_mse={r['val_mse_mean']:.3e}±{r['val_mse_std']:.1e}  "
              f"best_epoch={r['best_epoch_mean']:.0f}  t/epoch={r['time_per_epoch_ms_mean']:.1f}ms")


if __name__ == "__main__":
    main()
