"""Tổng hợp và vẽ kết quả Thực nghiệm B (độ rộng tầng ẩn B = 1..10 + mở rộng). (TV3 – T3.3, T3.4)

Lưu ý chống rò rỉ: quá trình huấn luyện (run_experiment.py) và chọn checkpoint (best.pt) chỉ dùng
validation. Tập test chỉ được đánh giá hậu kiểm (post-hoc) trên các trọng số best.pt đã đóng băng
để trình bày đủ 3 đường Train / Validation / Test theo yêu cầu của đề bài.
Chạy sau expB.json:  python src/summarize_expB.py
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_split, subset  # noqa: E402
from src.data.gaussian import X1_RANGE, X2_RANGE, Y_SCALE, gaussian_pdf  # noqa: E402
from src.data.summarize_expA import mean_std  # noqa: E402
from src.engine.evaluate import metrics, predict, predict_grid  # noqa: E402
from src.model.mlp import MLP  # noqa: E402
from src.utils.io import write_csv  # noqa: E402
from src.visual.plots import DPI, SPLIT_COLORS, Z_LIMITS  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "figures"
N_TRAIN = 1000
STATS = (
    "train_mse", "val_mse", "test_mse",
    "train_mae", "val_mae", "test_mae",
    "train_rmse", "val_rmse", "test_rmse",
    "test_r2",
    "best_epoch", "epochs_run",
    "train_time_s", "time_per_epoch_ms",
)
SURFACE_WIDTHS = (1, 2, 3, 5, 10)


def read_summary(summary_path):
    with Path(summary_path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Không có run nào trong {summary_path}; chạy expB trước.")
    grouped = defaultdict(list)
    for r in rows:
        grouped[int(r["value"])].append(r)
    return grouped


def enrich_posthoc_test(grouped, exp_dir, x_te, y_te):
    """Đánh giá hậu kiểm (post-hoc) tập test từ best.pt đã đóng băng mà không sửa file summary.csv gốc."""
    for width, runs in grouped.items():
        for r in runs:
            if r.get("test_mse") not in (None, ""):
                continue
            ckpt = exp_dir / str(width) / f"seed_{r['seed']}" / "best.pt"
            model = MLP(hidden_sizes=[width], activation="sigmoid")
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            te_m = metrics(y_te, predict(model, x_te))
            for k, v in te_m.items():
                r[f"test_{k}"] = v


def build_aggregate(grouped, base_val_mse, base_test_mse, group_tag="main"):
    out = []
    for width in sorted(grouped):
        runs = grouped[width]
        n_params = int(runs[0]["n_params"])
        assert n_params == 4 * width + 1, f"B={width}: n_params={n_params} != {4 * width + 1}"
        row = {
            "group": group_tag,
            "width": width,
            "n_params": n_params,
            "n_seeds": len(runs),
            "baseline_val_mse": base_val_mse,
            "baseline_test_mse": base_test_mse,
        }
        for s in STATS:
            row[f"{s}_mean"], row[f"{s}_std"] = mean_std([float(r[s]) for r in runs])
        gaps = [float(r["val_mse"]) - float(r["train_mse"]) for r in runs]
        row["gap_mse_mean"], row["gap_mse_std"] = mean_std(gaps)
        out.append(row)
    return out


def plot_expB_error(main_rows, base_mse, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.3), constrained_layout=True)

    b_main = np.asarray([r["width"] for r in main_rows])
    for split, label, color, marker, ls in (
        ("train", "Train MSE", SPLIT_COLORS["train"], "o", "-"),
        ("val", "Validation MSE", SPLIT_COLORS["val"], "s", "-"),
        ("test", "Test MSE (post-hoc)", SPLIT_COLORS["test"], "^", "--"),
    ):
        means = np.asarray([r[f"{split}_mse_mean"] for r in main_rows])
        stds = np.asarray([r[f"{split}_mse_std"] for r in main_rows])
        ax1.errorbar(b_main, means, yerr=stds, label=label, color=color, marker=marker,
                     ls=ls, capsize=3, lw=1.6, alpha=0.9)
    ax1.axhline(base_mse, ls=":", color=SPLIT_COLORS["baseline"], lw=1.4,
                label="Constant mean predictor (val)")
    ax1.set_yscale("log")
    ax1.set_xticks(b_main)
    ax1.set_xlabel("Number of hidden neurons B (1 hidden layer)")
    ax1.set_ylabel("MSE on original y scale (log)")
    ax1.set_title("Experiment B: Train, Validation & Test MSE vs. width B")
    ax1.grid(True, which="both", alpha=0.22)
    ax1.legend(fontsize=8.8)

    gaps_mean = np.asarray([r["gap_mse_mean"] for r in main_rows])
    gaps_std = np.asarray([r["gap_mse_std"] for r in main_rows])
    ax2.axhline(0.0, color=SPLIT_COLORS["baseline"], lw=0.9, ls="--")
    ax2.errorbar(b_main, gaps_mean, yerr=gaps_std, color=SPLIT_COLORS["gap"], marker="o",
                 capsize=3, lw=1.6, label="Val MSE − Train MSE (mean ± std)")
    ax2.set_xticks(b_main)
    ax2.set_xlabel("Number of hidden neurons B (1 hidden layer)")
    ax2.set_ylabel("Validation MSE − Train MSE")
    ax2.set_title("Experiment B: train–validation gap across width B")
    ax2.grid(True, alpha=0.22)
    ax2.legend()

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_expB_surfaces(exp_dir, out_path, seed=0, n=90):
    """Vẽ bề mặt thật và bề mặt xấp xỉ ở B = 1, 2, 3, 5, 10 (cùng seed, cùng giới hạn trục z). (TV3 – T3.4)"""
    x1 = np.linspace(*X1_RANGE, n)
    x2 = np.linspace(*X2_RANGE, n)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    points = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    y_true = gaussian_pdf(points).reshape(grid_x1.shape)

    fig = plt.figure(figsize=(15.5, 9.5), constrained_layout=True)

    ax0 = fig.add_subplot(2, 3, 1, projection="3d")
    surf = ax0.plot_surface(grid_x1, grid_x2, y_true, cmap="viridis",
                            vmin=0.0, vmax=Y_SCALE, linewidth=0, antialiased=True)
    ax0.set_xlim(*X1_RANGE)
    ax0.set_ylim(*X2_RANGE)
    ax0.set_zlim(*Z_LIMITS)
    ax0.set_xlabel("x1")
    ax0.set_ylabel("x2")
    ax0.set_zlabel("Density")
    ax0.set_title(f"True 2D Gaussian (max={y_true.max():.3f})")

    for idx, b in enumerate(SURFACE_WIDTHS, start=2):
        ckpt_path = exp_dir / str(b) / f"seed_{seed}" / "best.pt"
        model = MLP(hidden_sizes=[b], activation="sigmoid")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        _, _, y_hat = predict_grid(model, n=n)
        m = metrics(y_true.ravel(), y_hat.ravel())

        ax = fig.add_subplot(2, 3, idx, projection="3d")
        surf = ax.plot_surface(grid_x1, grid_x2, y_hat, cmap="viridis",
                               vmin=0.0, vmax=Y_SCALE, linewidth=0, antialiased=True)
        ax.set_xlim(*X1_RANGE)
        ax.set_ylim(*X2_RANGE)
        ax.set_zlim(*Z_LIMITS)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("ŷ(x1, x2)")
        ax.set_title(
            f"B = {b} ({4 * b + 1} params, seed {seed})\n"
            f"Grid MSE = {m['mse']:.2e}, R² = {m['r2']:.3f}"
        )

    fig.colorbar(surf, ax=fig.axes, shrink=0.55, pad=0.03, label="Density f(x1, x2)")
    fig.suptitle(
        "Experiment B: Surface evolution with hidden layer width B (1 hidden layer, sigmoid, A=1000)",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    exp_dir = RES / "expB"
    ctrl_dir = RES / "expC_ctrl"
    grouped_main = read_summary(exp_dir / "summary.csv")
    grouped_ctrl = read_summary(ctrl_dir / "summary.csv") if (ctrl_dir / "summary.csv").exists() else {}

    x_pool, y_pool = load_split("train_pool")
    _, y_tr = subset(x_pool, y_pool, N_TRAIN)
    _, y_val = load_split("val")
    x_te, y_te = load_split("test")
    base_val_mse = metrics(y_val, np.full_like(y_val, y_tr.mean()))["mse"]
    base_test_mse = metrics(y_te, np.full_like(y_te, y_tr.mean()))["mse"]

    enrich_posthoc_test(grouped_main, exp_dir, x_te, y_te)
    if grouped_ctrl:
        enrich_posthoc_test(grouped_ctrl, ctrl_dir, x_te, y_te)

    main_rows = build_aggregate(grouped_main, base_val_mse, base_test_mse, group_tag="expB")
    ctrl_rows = build_aggregate(grouped_ctrl, base_val_mse, base_test_mse, group_tag="expC_ctrl") if grouped_ctrl else []
    all_rows = main_rows + ctrl_rows

    fields = ["group", "width", "n_params", "n_seeds", "baseline_val_mse", "baseline_test_mse"]
    fields += [f"{s}_{t}" for s in STATS for t in ("mean", "std")]
    fields += ["gap_mse_mean", "gap_mse_std"]
    write_csv(exp_dir / "aggregate.csv", all_rows, fields)

    FIG.mkdir(parents=True, exist_ok=True)
    plot_expB_error(main_rows, base_val_mse, FIG / "expB_error.png")
    plot_expB_surfaces(exp_dir, FIG / "expB_surfaces.png", seed=0, n=90)

    print(f"Constant-mean val MSE = {base_val_mse:.4e}, test MSE = {base_test_mse:.4e}")
    for r in all_rows:
        print(
            f"[{r['group']:9s}] B={r['width']:2d} (params={r['n_params']:3d})  "
            f"train_mse={r['train_mse_mean']:.3e}±{r['train_mse_std']:.1e}  "
            f"val_mse={r['val_mse_mean']:.3e}±{r['val_mse_std']:.1e}  "
            f"test_mse={r['test_mse_mean']:.3e}±{r['test_mse_std']:.1e}  "
            f"best_ep={r['best_epoch_mean']:.0f}"
        )


if __name__ == "__main__":
    main()
