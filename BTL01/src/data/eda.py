"""Tạo thống kê và hình EDA có thể tái lập từ các split CSV. (TV1 – T1.5)"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_split  # noqa: E402
from src.data.gaussian import X1_RANGE, X2_RANGE, gaussian_pdf  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

SPLIT_COLORS = {"train_pool": "#2563eb", "val": "#f97316", "test": "#16a34a"}
SPLIT_LABELS = {"train_pool": "Train pool", "val": "Validation", "test": "Test"}


def build_summary(splits):
    rows = []
    for split, (x, y) in splits.items():
        for name, values in (("x1", x[:, 0]), ("x2", x[:, 1]), ("y", y)):
            rows.append({
                "split": split,
                "variable": name,
                "n": int(values.size),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "fraction_y_below_0_01": float(np.mean(y < 0.01)) if name == "y" else "",
            })
    return rows


def plot_surface_and_contour(out_path):
    x1 = np.linspace(*X1_RANGE, 120)
    x2 = np.linspace(*X2_RANGE, 160)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    points = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    z = gaussian_pdf(points).reshape(grid_x1.shape)

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surface = ax3d.plot_surface(grid_x1, grid_x2, z, cmap="viridis", linewidth=0, antialiased=True)
    ax3d.set_title("True 2D Gaussian surface")
    ax3d.set_xlabel("x1")
    ax3d.set_ylabel("x2")
    ax3d.set_zlabel("Density f(x1, x2)")
    fig.colorbar(surface, ax=ax3d, shrink=0.68, pad=0.08, label="Density")

    ax2d = fig.add_subplot(1, 2, 2)
    filled = ax2d.contourf(grid_x1, grid_x2, z, levels=28, cmap="viridis")
    ax2d.contour(grid_x1, grid_x2, z, levels=12, colors="black", linewidths=0.45, alpha=0.65)
    ax2d.set_title("Density contours")
    ax2d.set_xlabel("x1")
    ax2d.set_ylabel("x2")
    fig.colorbar(filled, ax=ax2d, label="Density")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_split_scatter(splits, out_path):
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for split, (x, y) in splits.items():
        ax.scatter(x[:, 0], x[:, 1], y, s=9, alpha=0.55,
                   color=SPLIT_COLORS[split], label=SPLIT_LABELS[split], depthshade=False)
    ax.set_title("Generated train, validation, and test samples")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Density f(x1, x2)")
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(splits, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    variables = (("x1", 0), ("x2", 1))
    for ax, (variable, column) in zip(axes[:2], variables):
        for split, (x, _) in splits.items():
            ax.hist(x[:, column], bins=36, density=True, histtype="step", linewidth=1.6,
                    color=SPLIT_COLORS[split], label=SPLIT_LABELS[split])
        ax.set_title(f"Distribution of {variable}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2)

    ax = axes[2]
    for split, (_, y) in splits.items():
        ax.hist(y, bins=45, histtype="step", linewidth=1.6,
                color=SPLIT_COLORS[split], label=SPLIT_LABELS[split], log=True)
    ax.set_title("Distribution of y (log count)")
    ax.set_xlabel("y = f(x1, x2)")
    ax.set_ylabel("Count (log scale)")
    ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    splits = {name: load_split(name) for name in ("train_pool", "val", "test")}
    figure_dir = ROOT / "figures"
    summary_dir = ROOT / "results" / "eda"
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    write_csv(summary_dir / "data_summary.csv", build_summary(splits),
              fields=["split", "variable", "n", "min", "max", "mean", "std", "fraction_y_below_0_01"])
    plot_surface_and_contour(figure_dir / "eda_surface_contour.png")
    plot_split_scatter(splits, figure_dir / "eda_scatter_3d.png")
    plot_histograms(splits, figure_dir / "eda_histograms.png")
    print(f"EDA summary: {summary_dir / 'data_summary.csv'}")
    print(f"EDA figures: {figure_dir}")


if __name__ == "__main__":
    main()
