"""Tổng hợp và vẽ kết quả Thực nghiệm A; không đọc tập test. (TV1 – T1.6)"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_split, subset  # noqa: E402
from src.engine.evaluate import metrics  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

METRICS = ("mse", "mae", "rmse")


def read_runs(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Không có run nào trong {path}; chạy expA trước.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["value"])].append(row)
    return grouped


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def build_aggregate(grouped):
    x_pool, y_pool = load_split("train_pool")
    x_val, y_val = load_split("val")
    rows = []
    for n_train in sorted(grouped):
        x_train, y_train = subset(x_pool, y_pool, n_train)
        const_pred = np.full_like(y_val, y_train.mean())
        base_metrics = metrics(y_val, const_pred)
        aggregate = {"n_train": n_train, "n_seeds": len(grouped[n_train]),
                     "baseline_val_mse": base_metrics["mse"]}
        for metric in METRICS:
            for split in ("train", "val"):
                mean, std = mean_std([float(run[f"{split}_{metric}"]) for run in grouped[n_train]])
                aggregate[f"{split}_{metric}_mean"] = mean
                aggregate[f"{split}_{metric}_std"] = std
        rows.append(aggregate)
    return rows


def plot_validation_mse(rows, out_path):
    x = np.asarray([r["n_train"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for split, label, color, marker in (("train", "Train MSE", "#2563eb", "o"),
                                        ("val", "Validation MSE", "#f97316", "s")):
        means = np.asarray([r[f"{split}_mse_mean"] for r in rows])
        stds = np.asarray([r[f"{split}_mse_std"] for r in rows])
        ax.errorbar(x, means, yerr=stds, label=label, color=color, marker=marker,
                    capsize=3, linewidth=1.6)
    baseline = np.asarray([r["baseline_val_mse"] for r in rows])
    ax.plot(x, baseline, "--", color="#4b5563", label="Constant mean predictor")
    ax.set_yscale("log")
    ax.set_xlabel("Number of training samples A")
    ax.set_ylabel("MSE on original y scale (log)")
    ax.set_title("Experiment A: error versus training set size")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_train_val_gap(rows, out_path):
    x = np.asarray([r["n_train"] for r in rows])
    gaps = np.asarray([r["val_mse_mean"] - r["train_mse_mean"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    ax.axhline(0.0, color="#4b5563", linewidth=0.8)
    ax.plot(x, gaps, marker="o", color="#7c3aed", linewidth=1.6)
    ax.set_xlabel("Number of training samples A")
    ax.set_ylabel("Validation MSE − train MSE")
    ax.set_title("Experiment A: train–validation gap")
    ax.grid(True, alpha=0.22)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    results_dir = ROOT / "results" / "expA"
    figure_dir = ROOT / "figures"
    grouped = read_runs(results_dir / "summary.csv")
    rows = build_aggregate(grouped)
    fields = ["n_train", "n_seeds", "baseline_val_mse"]
    fields += [f"{split}_{metric}_{stat}" for split in ("train", "val")
               for metric in METRICS for stat in ("mean", "std")]
    write_csv(results_dir / "aggregate.csv", rows, fields)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_validation_mse(rows, figure_dir / "expA_validation_mse.png")
    plot_train_val_gap(rows, figure_dir / "expA_train_val_gap.png")
    print(f"A aggregate: {results_dir / 'aggregate.csv'}")
    print(f"A figures: {figure_dir}")


if __name__ == "__main__":
    main()
