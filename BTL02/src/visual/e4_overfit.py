# Phụ trách: TV4 – Huỳnh Viễn Thông
"""Vẽ biểu đồ đường cong Train/Val và khoảng cách Overfit cho toàn bộ 10 biến thể E4 từ history.csv.

Phụ trách: TV4 – Huỳnh Viễn Thông. Nhiệm vụ T15 (phối hợp C2).

Sinh:
- figures/e4_overfit.png: Lưới 2x3 tách rõ 3 nhóm (Augmentation | Dropout | Weight Decay & Label Smoothing):
    + Hàng trên: Train Loss (nét đứt) vs Val Loss (nét liền) theo epoch.
    + Hàng dưới: Khoảng cách Overfit theo Accuracy (Train Acc - Val Acc, %) theo epoch.
- figures/e4_overfit.csv: dữ liệu từng epoch khớp 100% với biểu đồ (phục vụ Tester T15 đối chiếu).
- figures/e4_overfit_summary.csv: tổng hợp khoảng cách overfit tại best_epoch và epoch cuối của từng biến thể.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visual.plots import _apply_style, _write_csv, register_figure

PANEL_SPECS = [
    ("Augmentation", ["aug_none", "aug_flip", "aug_flip_affine", "aug_random_erasing"]),
    ("Dropout", ["baseline", "dropout_0", "dropout_0_25", "dropout_0_5"]),
    ("Weight Decay & Label Smoothing", ["baseline", "weight_decay_0", "weight_decay_1e-3", "label_smoothing_0_1"]),
]

DISPLAY_NAMES = {
    "baseline": "baseline (wd=5e-4, dr=0.25/0.5)",
    "aug_none": "aug_none",
    "aug_flip": "aug_flip",
    "aug_flip_affine": "aug_flip_affine",
    "aug_random_erasing": "aug_random_erasing",
    "dropout_0": "dropout_0 (0.0/0.0)",
    "dropout_0_25": "dropout_0_25 (0.25/0.25)",
    "dropout_0_5": "dropout_0_5 (0.5/0.5)",
    "weight_decay_0": "weight_decay_0",
    "weight_decay_1e-3": "weight_decay_1e-3",
    "label_smoothing_0_1": "label_smoothing_0_1",
}


def _load_history(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            ep = int(r["epoch"])
            tr_l = float(r["train_loss"])
            va_l = float(r["val_loss"])
            tr_a = float(r["train_acc"])
            va_a = float(r["val_acc"])
            rows.append(
                {
                    "epoch": ep,
                    "train_loss": tr_l,
                    "val_loss": va_l,
                    "loss_gap": va_l - tr_l,
                    "train_acc": tr_a,
                    "val_acc": va_a,
                    "acc_gap": tr_a - va_a,
                }
            )
    return rows


def collect_e4_runs(experiments_csv: Path, baseline_id: str = "baseline-002_s0"):
    runs_by_variant = {}
    with experiments_csv.open("r", encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    for r in all_rows:
        if r["exp_id"] == baseline_id and r.get("trang_thai_ky_thuat") == "Completed":
            runs_by_variant["baseline"] = (r["exp_id"], "baseline", ROOT / r["history_path"], int(r["best_epoch"]))
            break

    for r in all_rows:
        if not r["exp_id"].startswith("E4-") or r.get("trang_thai_ky_thuat") != "Completed":
            continue
        hp = json.loads(r.get("hyperparams_json") or "{}")
        variant = hp.get("variant") or r["exp_id"]
        if variant not in runs_by_variant:
            runs_by_variant[variant] = (r["exp_id"], variant, ROOT / r["history_path"], int(r["best_epoch"]))
    return runs_by_variant


def plot_e4_overfit(experiments_csv: Path, out_png: Path, out_csv: Path, summary_csv: Path, manifest_path: Path):
    _apply_style()
    runs_by_variant = collect_e4_runs(experiments_csv)
    if len(runs_by_variant) <= 1:
        raise ValueError("Không tìm thấy run E4 hợp lệ nào trong experiments.csv")

    palette = ["#c53030", "#d69e2e", "#2b6cb0", "#2f855a", "#805ad5"]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8), sharey="row")

    # Cache loaded histories
    hist_cache = {}
    for var, (exp_id, _, hist_path, _) in runs_by_variant.items():
        hist_cache[var] = _load_history(hist_path)

    for col_idx, (panel_title, variants) in enumerate(PANEL_SPECS):
        ax_loss = axes[0, col_idx]
        ax_gap = axes[1, col_idx]
        for idx, var in enumerate(variants):
            if var not in runs_by_variant:
                continue
            exp_id, _, _, _ = runs_by_variant[var]
            hist = hist_cache[var]
            color = "#2d3748" if var == "baseline" else palette[idx % len(palette)]
            epochs = [h["epoch"] for h in hist]
            tr_l = [h["train_loss"] for h in hist]
            va_l = [h["val_loss"] for h in hist]
            agap = [h["acc_gap"] * 100.0 for h in hist]

            lbl = f"{DISPLAY_NAMES.get(var, var)} ({exp_id})"
            ax_loss.plot(epochs, va_l, color=color, linewidth=1.8, label=lbl)
            ax_loss.plot(epochs, tr_l, color=color, linewidth=1.0, linestyle="--", alpha=0.6)
            ax_gap.plot(epochs, agap, color=color, linewidth=1.8, label=lbl)

        ax_loss.set_title(f"{panel_title}\nTrain Loss (đứt) vs Val Loss (liền)", fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        if col_idx == 0:
            ax_loss.set_ylabel("Cross-Entropy Loss")
        ax_loss.legend(fontsize=7.8, loc="upper right")

        ax_gap.axhline(0.0, color="#718096", linewidth=1.0, linestyle=":")
        ax_gap.set_title(f"{panel_title}\nOverfit Gap (Train Acc − Val Acc)")
        ax_gap.set_xlabel("Epoch")
        if col_idx == 0:
            ax_gap.set_ylabel("Khoảng cách Acc (% — thấp hơn = ít overfit hơn)")
        ax_gap.legend(fontsize=7.8, loc="upper left")

    fig.suptitle("E4 – So sánh đường cong Train/Val và mức độ Overfitting (10 biến thể vs Baseline)", fontweight="bold", fontsize=13, y=1.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Xuất CSV đầy đủ (không lặp dòng baseline, giữ thứ tự chuẩn baseline + 10 biến thể E4)
    ordered_vars = ["baseline"] + [
        "aug_none", "aug_flip", "aug_flip_affine", "aug_random_erasing",
        "dropout_0", "dropout_0_25", "dropout_0_5",
        "weight_decay_0", "weight_decay_1e-3", "label_smoothing_0_1",
    ]
    var_to_group = {
        "baseline": "baseline",
        "aug_none": "augmentation", "aug_flip": "augmentation",
        "aug_flip_affine": "augmentation", "aug_random_erasing": "augmentation",
        "dropout_0": "dropout", "dropout_0_25": "dropout", "dropout_0_5": "dropout",
        "weight_decay_0": "regularization", "weight_decay_1e-3": "regularization",
        "label_smoothing_0_1": "regularization",
    }

    csv_rows = []
    summary_rows = []
    exp_ids = []
    for var in ordered_vars:
        if var not in runs_by_variant:
            continue
        exp_id, _, _, best_ep = runs_by_variant[var]
        exp_ids.append(exp_id)
        hist = hist_cache[var]
        grp = var_to_group.get(var, "other")
        for h in hist:
            csv_rows.append(
                {
                    "exp_id": exp_id,
                    "group_panel": grp,
                    "variant": var,
                    "epoch": h["epoch"],
                    "train_loss": round(h["train_loss"], 6),
                    "val_loss": round(h["val_loss"], 6),
                    "loss_gap": round(h["loss_gap"], 6),
                    "train_acc": round(h["train_acc"], 6),
                    "val_acc": round(h["val_acc"], 6),
                    "acc_gap": round(h["acc_gap"], 6),
                }
            )
        best_row = next((h for h in hist if h["epoch"] == best_ep), hist[-1])
        last_row = hist[-1]
        summary_rows.append(
            {
                "exp_id": exp_id,
                "group_panel": grp,
                "variant": var,
                "best_epoch": best_ep,
                "best_val_acc": round(best_row["val_acc"], 6),
                "train_acc_at_best": round(best_row["train_acc"], 6),
                "acc_gap_at_best": round(best_row["acc_gap"], 6),
                "best_val_loss": round(best_row["val_loss"], 6),
                "loss_gap_at_best": round(best_row["loss_gap"], 6),
                "final_acc_gap": round(last_row["acc_gap"], 6),
                "final_loss_gap": round(last_row["loss_gap"], 6),
            }
        )

    _write_csv(
        out_csv,
        ["exp_id", "group_panel", "variant", "epoch", "train_loss", "val_loss", "loss_gap", "train_acc", "val_acc", "acc_gap"],
        csv_rows,
    )
    _write_csv(
        summary_csv,
        [
            "exp_id", "group_panel", "variant", "best_epoch", "best_val_acc", "train_acc_at_best",
            "acc_gap_at_best", "best_val_loss", "loss_gap_at_best", "final_acc_gap", "final_loss_gap",
        ],
        summary_rows,
    )

    register_figure(
        manifest_path=manifest_path,
        fig_id="e4_overfit",
        file_hinh=out_png,
        file_du_lieu_dau_vao=out_csv,
        script_sinh_hinh="src/visual/e4_overfit.py",
        lenh_chay="python src/visual/e4_overfit.py",
        exp_ids=";".join(exp_ids),
        task_id="T15",
        dung_o="E4: Đường cong Train/Val và khoảng cách overfit đủ 10 biến thể (Augmentation, Dropout, Weight Decay, Label Smoothing) vs Baseline",
        root=ROOT,
    )
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Vẽ đường cong train/val so overfit cho đủ 10 biến thể E4.")
    parser.add_argument("--experiments", type=Path, default=ROOT / "tracking/experiments.csv")
    parser.add_argument("--out-png", type=Path, default=ROOT / "figures/e4_overfit.png")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "figures/e4_overfit.csv")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "figures/e4_overfit_summary.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "tracking/figures_manifest.csv")
    args = parser.parse_args()

    summary = plot_e4_overfit(args.experiments, args.out_png, args.out_csv, args.summary_csv, args.manifest)
    print(f"=== E4 OVERFIT SUMMARY ({len(summary)} runs) ===")
    for s in summary:
        print(
            f"  {s['exp_id']:15s} | {s['group_panel']:14s} | {s['variant']:20s} | ep={s['best_epoch']:3d} | "
            f"val_acc={s['best_val_acc']*100:5.2f}% | acc_gap@best={s['acc_gap_at_best']*100:+6.2f}% | "
            f"final_acc_gap={s['final_acc_gap']*100:+6.2f}% | final_loss_gap={s['final_loss_gap']:+.4f}"
        )


if __name__ == "__main__":
    main()
