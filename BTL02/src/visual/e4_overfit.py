# Phụ trách: TV4 – Huỳnh Viễn Thông
"""Vẽ biểu đồ đường cong Train/Val và khoảng cách Overfit cho thí nghiệm E4 từ history.csv.

Phụ trách: TV4 – Huỳnh Viễn Thông. Nhiệm vụ T15 (phối hợp C2).

Sinh:
- figures/e4_overfit.png: 4 biểu đồ (Train/Val Loss, Loss Gap, Train/Val Acc, Acc Gap) theo epoch.
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
    selected = []
    seen_variants = set()
    with experiments_csv.open("r", encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    # Thêm mốc đối chứng Baseline (seed 0)
    for r in all_rows:
        if r["exp_id"] == baseline_id and r.get("trang_thai_ky_thuat") == "Completed":
            selected.append((r["exp_id"], "baseline (wd=5e-4, ls=0)", ROOT / r["history_path"], int(r["best_epoch"])))
            break

    # Gom toàn bộ các run E4 đã Completed (mỗi biến thể lấy run seed 0)
    for r in all_rows:
        if not r["exp_id"].startswith("E4-") or r.get("trang_thai_ky_thuat") != "Completed":
            continue
        hp = json.loads(r.get("hyperparams_json") or "{}")
        variant = hp.get("variant") or r["exp_id"]
        if variant in seen_variants:
            continue
        seen_variants.add(variant)
        selected.append((r["exp_id"], variant, ROOT / r["history_path"], int(r["best_epoch"])))
    return selected


def plot_e4_overfit(experiments_csv: Path, out_png: Path, out_csv: Path, summary_csv: Path, manifest_path: Path):
    _apply_style()
    runs = collect_e4_runs(experiments_csv)
    if not runs:
        raise ValueError("Không tìm thấy run E4 hợp lệ nào trong experiments.csv")

    colors = ["#2d3748", "#d69e2e", "#2b6cb0", "#2f855a", "#c53030", "#805ad5", "#dd6b20", "#319795", "#b83280", "#4a5568", "#718096"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    ax_loss, ax_lgap = axes[0]
    ax_acc, ax_agap = axes[1]

    csv_rows = []
    summary_rows = []
    exp_ids = []

    for idx, (exp_id, variant, hist_path, best_ep) in enumerate(runs):
        exp_ids.append(exp_id)
        color = colors[idx % len(colors)]
        hist = _load_history(hist_path)
        epochs = [h["epoch"] for h in hist]
        tr_l = [h["train_loss"] for h in hist]
        va_l = [h["val_loss"] for h in hist]
        lgap = [h["loss_gap"] for h in hist]
        tr_a = [h["train_acc"] * 100.0 for h in hist]
        va_a = [h["val_acc"] * 100.0 for h in hist]
        agap = [h["acc_gap"] * 100.0 for h in hist]

        lbl = f"{variant} ({exp_id})"
        ax_loss.plot(epochs, va_l, color=color, linewidth=1.8, label=f"{variant} (Val)")
        ax_loss.plot(epochs, tr_l, color=color, linewidth=1.1, linestyle="--", alpha=0.65)
        ax_lgap.plot(epochs, lgap, color=color, linewidth=1.8, label=lbl)

        ax_acc.plot(epochs, va_a, color=color, linewidth=1.8, label=f"{variant} (Val)")
        ax_acc.plot(epochs, tr_a, color=color, linewidth=1.1, linestyle="--", alpha=0.65)
        ax_agap.plot(epochs, agap, color=color, linewidth=1.8, label=lbl)

        for h in hist:
            csv_rows.append(
                {
                    "exp_id": exp_id,
                    "variant": variant,
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
                "variant": variant,
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

    ax_loss.set_title("Train Loss (nét đứt) vs Val Loss (nét liền)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend(fontsize=8)

    ax_lgap.axhline(0.0, color="#718096", linewidth=1.0, linestyle=":")
    ax_lgap.set_title("Khoảng cách Overfit theo Loss (Val Loss − Train Loss)")
    ax_lgap.set_xlabel("Epoch")
    ax_lgap.set_ylabel("Loss Gap (thấp hơn = ít overfit hơn)")
    ax_lgap.legend(fontsize=8)

    ax_acc.set_title("Train Acc (nét đứt) vs Val Acc (nét liền)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend(fontsize=8)

    ax_agap.axhline(0.0, color="#718096", linewidth=1.0, linestyle=":")
    ax_agap.set_title("Khoảng cách Overfit theo Accuracy (Train Acc − Val Acc)")
    ax_agap.set_xlabel("Epoch")
    ax_agap.set_ylabel("Acc Gap (% — thấp hơn = ít overfit hơn)")
    ax_agap.legend(fontsize=8)

    fig.suptitle("E4 – So sánh đường cong Train/Val và mức độ Overfitting (FER2013)", fontweight="bold", y=1.01)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    _write_csv(
        out_csv,
        ["exp_id", "variant", "epoch", "train_loss", "val_loss", "loss_gap", "train_acc", "val_acc", "acc_gap"],
        csv_rows,
    )
    _write_csv(
        summary_csv,
        [
            "exp_id", "variant", "best_epoch", "best_val_acc", "train_acc_at_best",
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
        dung_o="E4: Đường cong Train/Val và khoảng cách overfit (weight decay, label smoothing vs baseline)",
        root=ROOT,
    )
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Vẽ đường cong train/val so overfit cho E4.")
    parser.add_argument("--experiments", type=Path, default=ROOT / "tracking/experiments.csv")
    parser.add_argument("--out-png", type=Path, default=ROOT / "figures/e4_overfit.png")
    parser.add_argument("--out-csv", type=Path, default=ROOT / "figures/e4_overfit.csv")
    parser.add_argument("--summary-csv", type=Path, default=ROOT / "figures/e4_overfit_summary.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "tracking/figures_manifest.csv")
    args = parser.parse_args()

    summary = plot_e4_overfit(args.experiments, args.out_png, args.out_csv, args.summary_csv, args.manifest)
    print("=== E4 OVERFIT SUMMARY ===")
    for s in summary:
        print(
            f"  {s['exp_id']:15s} | {s['variant']:25s} | best_ep={s['best_epoch']:3d} | "
            f"val_acc={s['best_val_acc']*100:5.2f}% | acc_gap@best={s['acc_gap_at_best']*100:+5.2f}% | "
            f"final_acc_gap={s['final_acc_gap']*100:+5.2f}% | final_loss_gap={s['final_loss_gap']:+.4f}"
        )


if __name__ == "__main__":
    main()
