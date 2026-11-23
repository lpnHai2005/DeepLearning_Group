"""Vẽ biểu đồ đánh giá và theo dõi thực nghiệm FER2013 (xuất kèm CSV và ghi figures_manifest.csv).

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T08.

Mọi hình sinh ra luôn kèm file CSV dữ liệu đã vẽ theo đúng quy chuẩn mục 9.4 của PHAN_CONG_NHIEM_VU.md.
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # chạy được cả dạng `python src/visual/plots.py`

from src.engine.evaluate import EMOTIONS  # noqa: E402
from src.utils.io import BTL02, sha256_file  # noqa: E402

MANIFEST_HEADER = [
    "fig_id",
    "file_hinh",
    "file_du_lieu_dau_vao",
    "script_sinh_hinh",
    "lenh_chay",
    "exp_ids",
    "task_id",
    "thoi_gian_sinh",
    "sha256_hinh",
    "dung_o",
]


def _apply_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )


def _rel_path(path, root):
    p = Path(path).resolve()
    try:
        return p.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def register_figure(
    manifest_path,
    fig_id,
    file_hinh,
    file_du_lieu_dau_vao,
    script_sinh_hinh,
    lenh_chay,
    exp_ids="",
    task_id="",
    dung_o="",
    root=BTL02,
):
    """Ghi hoặc cập nhật 1 dòng truy vết vào tracking/figures_manifest.csv theo đúng fig_id."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    img_path = Path(file_hinh)
    if not img_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file hình để tính sha256: {img_path}")

    existing = []
    if manifest_path.exists() and manifest_path.stat().st_size > 0:
        with open(manifest_path, encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))

    row = {
        "fig_id": fig_id,
        "file_hinh": _rel_path(img_path, root),
        "file_du_lieu_dau_vao": _rel_path(file_du_lieu_dau_vao, root) if Path(file_du_lieu_dau_vao).exists() else str(file_du_lieu_dau_vao),
        "script_sinh_hinh": str(script_sinh_hinh),
        "lenh_chay": str(lenh_chay),
        "exp_ids": str(exp_ids),
        "task_id": str(task_id),
        "thoi_gian_sinh": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sha256_hinh": sha256_file(img_path),
        "dung_o": str(dung_o),
    }

    updated = False
    for i, r in enumerate(existing):
        if r.get("fig_id") == fig_id:
            existing[i] = row
            updated = True
            break
    if not updated:
        existing.append(row)

    _write_csv(manifest_path, MANIFEST_HEADER, existing)
    return row


def plot_curves(history_csv, out_png, out_csv=None, title=None):
    """Vẽ đường cong Train/Val loss, Train/Val acc (+ macro-F1) và learning rate từ history.csv."""
    _apply_style()
    history_csv = Path(history_csv)
    out_png = Path(out_png)
    out_csv = Path(out_csv) if out_csv is not None else out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with open(history_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{history_csv} không có dữ liệu epoch nào")

    epochs = [int(float(r["epoch"])) for r in rows]
    tr_loss = [float(r["train_loss"]) for r in rows]
    va_loss = [float(r["val_loss"]) for r in rows]
    tr_acc = [float(r["train_acc"]) for r in rows]
    va_acc = [float(r["val_acc"]) for r in rows]
    va_f1 = [float(r["val_macro_f1"]) for r in rows]
    lrs = [float(r["lr"]) for r in rows]

    best_idx = max(range(len(va_acc)), key=lambda i: (va_acc[i], -i))
    best_ep = epochs[best_idx]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(epochs, tr_loss, label="Train loss", linewidth=1.8)
    axes[0].plot(epochs, va_loss, label="Val loss", linewidth=1.8)
    axes[0].axvline(best_ep, color="gray", linestyle="--", linewidth=1.0, label=f"Best epoch ({best_ep})")
    axes[0].set_title("Loss theo epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, tr_acc, label="Train acc", linewidth=1.8)
    axes[1].plot(epochs, va_acc, label="Val acc", linewidth=1.8)
    axes[1].plot(epochs, va_f1, label="Val macro-F1", linewidth=1.4, linestyle=":")
    axes[1].axvline(best_ep, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"Accuracy & Macro-F1 (best val={va_acc[best_idx]:.4f})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()

    axes[2].plot(epochs, lrs, color="#2b6cb0", linewidth=1.8)
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    if max(lrs) > 0 and min(lrs) > 0 and max(lrs) / min(lrs) >= 10:
        axes[2].set_yscale("log")

    if title:
        fig.suptitle(title, y=1.02, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    csv_cols = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "val_macro_f1", "lr"]
    out_rows = [{k: rows[i][k] for k in csv_cols} for i in range(len(rows))]
    _write_csv(out_csv, csv_cols, out_rows)
    return out_png, out_csv


def plot_confusion_matrix(cm, out_png, out_csv=None, labels=EMOTIONS, normalize=True, title="Ma trận nhầm lẫn"):
    """Vẽ ma trận nhầm lẫn chuẩn hóa theo hàng kèm số lượng trong từng ô."""
    _apply_style()
    cm_arr = np.asarray(cm, dtype=np.int64)
    n = len(labels)
    if cm_arr.shape != (n, n):
        raise ValueError(f"Kích thước cm {cm_arr.shape} không khớp {n} lớp")

    row_sum = cm_arr.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_arr.astype(np.float64), row_sum, out=np.zeros_like(cm_arr, dtype=np.float64), where=row_sum > 0)
    disp = cm_pct * 100.0 if normalize else cm_arr.astype(np.float64)

    out_png = Path(out_png)
    out_csv = Path(out_csv) if out_csv is not None else out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(disp, cmap="Blues", vmin=0.0, vmax=100.0 if normalize else max(float(disp.max()), 1.0))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Tỷ lệ (%)" if normalize else "Số mẫu")
    ax.grid(False)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(title)

    thresh = disp.max() * 0.55
    for i in range(n):
        for j in range(n):
            txt = f"{cm_pct[i, j] * 100:.1f}%\n({cm_arr[i, j]})" if normalize else f"{cm_arr[i, j]}"
            color = "white" if disp[i, j] > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for i, tl in enumerate(labels):
        for j, pl in enumerate(labels):
            rows.append(
                {
                    "true_label": tl,
                    "pred_label": pl,
                    "count": int(cm_arr[i, j]),
                    "row_pct": round(float(cm_pct[i, j] * 100.0), 4),
                }
            )
    _write_csv(out_csv, ["true_label", "pred_label", "count", "row_pct"], rows)
    return out_png, out_csv


def _to_display_image(img):
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr)
    return arr


def plot_sample_grid(
    images,
    y_true,
    y_pred,
    out_png,
    out_csv=None,
    y_prob=None,
    labels=EMOTIONS,
    correct=False,
    max_samples=16,
    title=None,
):
    """Vẽ lưới ảnh ca đoán đúng (correct=True) hoặc sai (correct=False) từ đúng tập đánh giá."""
    _apply_style()
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(images) != len(yt) or len(yt) != len(yp):
        raise ValueError("Số lượng images, y_true, y_pred không khớp")

    mask = (yt == yp) if correct else (yt != yp)
    idxs = np.flatnonzero(mask)[:max_samples]

    out_png = Path(out_png)
    out_csv = Path(out_csv) if out_csv is not None else out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    n_show = len(idxs)
    ncols = min(4, max(n_show, 1))
    nrows = max(1, int(np.ceil(max(n_show, 1) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.7))
    axes_arr = np.atleast_1d(axes).ravel()

    csv_rows = []
    for k, ax in enumerate(axes_arr):
        ax.axis("off")
        if k >= n_show:
            continue
        idx = int(idxs[k])
        disp = _to_display_image(images[idx])
        ax.imshow(disp, cmap="gray" if disp.ndim == 2 else None)
        t_name = labels[int(yt[idx])]
        p_name = labels[int(yp[idx])]
        conf = float(y_prob[idx, yp[idx]]) if y_prob is not None else None
        conf_txt = f" ({conf * 100:.1f}%)" if conf is not None else ""
        color = "#1f7a1f" if yt[idx] == yp[idx] else "#b91c1c"
        ax.set_title(f"#{idx}\nT:{t_name} | P:{p_name}{conf_txt}", fontsize=8.5, color=color)
        csv_rows.append(
            {
                "sample_idx": idx,
                "true_idx": int(yt[idx]),
                "true_label": t_name,
                "pred_idx": int(yp[idx]),
                "pred_label": p_name,
                "correct": int(yt[idx] == yp[idx]),
                "confidence": round(conf, 6) if conf is not None else "",
            }
        )

    if n_show == 0:
        axes_arr[0].text(0.5, 0.5, "Không có mẫu phù hợp", ha="center", va="center")

    default_title = "Mẫu dự đoán ĐÚNG" if correct else "Mẫu dự đoán SAI"
    fig.suptitle(title or default_title, y=1.01, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    _write_csv(out_csv, ["sample_idx", "true_idx", "true_label", "pred_idx", "pred_label", "correct", "confidence"], csv_rows)
    return out_png, out_csv


def plot_error_histogram(y_true, y_pred, out_png, out_csv=None, labels=EMOTIONS, title="Tỷ lệ sai và Recall theo lớp"):
    """Vẽ biểu đồ tỷ lệ ca sai (%) và recall (%) từng lớp cảm xúc."""
    _apply_style()
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(yt) != len(yp):
        raise ValueError("y_true và y_pred không cùng độ dài")

    out_png = Path(out_png)
    out_csv = Path(out_csv) if out_csv is not None else out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    recalls, err_rates = [], []
    for i, name in enumerate(labels):
        m = yt == i
        sup = int(m.sum())
        corr = int((yp[m] == i).sum()) if sup > 0 else 0
        errs = sup - corr
        rec = (corr / sup * 100.0) if sup > 0 else 0.0
        err_r = (errs / sup * 100.0) if sup > 0 else 0.0
        recalls.append(rec)
        err_rates.append(err_r)
        csv_rows.append(
            {
                "label": name,
                "support": sup,
                "correct": corr,
                "errors": errs,
                "error_rate": round(err_r, 4),
                "recall": round(rec, 4),
            }
        )

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, recalls, width=w, label="Recall (%)", color="#2b6cb0")
    b2 = ax.bar(x + w / 2, err_rates, width=w, label="Tỷ lệ sai (%)", color="#c53030")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Phần trăm (%)")
    ax.set_title(title)
    ax.legend()

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    _write_csv(out_csv, ["label", "support", "correct", "errors", "error_rate", "recall"], csv_rows)
    return out_png, out_csv


def plot_run_comparison(experiments_csv, out_png, exp_ids=None, metric="val_acc", out_csv=None, title=None):
    """Vẽ đồ thị cột ngang so sánh các run từ tracking/experiments.csv."""
    _apply_style()
    experiments_csv = Path(experiments_csv)
    out_png = Path(out_png)
    out_csv = Path(out_csv) if out_csv is not None else out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    with open(experiments_csv, encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    wanted = set(exp_ids) if exp_ids is not None else None
    rows = []
    for r in all_rows:
        if wanted is not None and r["exp_id"] not in wanted:
            continue
        val_raw = (r.get(metric) or "").strip()
        if not val_raw:
            continue
        rows.append(r)
    if not rows:
        raise ValueError(f"Không có run hợp lệ có chỉ số '{metric}' trong {experiments_csv}")

    labels = [r["exp_id"] for r in rows]
    vals = [float(r[metric]) for r in rows]
    order = np.argsort(vals)
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    ordered_rows = [rows[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.42 * len(labels) + 1.2)))
    bars = ax.barh(labels, vals, color="#2f855a")
    ax.set_xlabel(metric)
    ax.set_title(title or f"So sánh các run theo {metric}")
    hi = max(vals) if vals else 1.0
    ax.set_xlim(0.0, hi * 1.15 if hi > 0 else 1.0)
    for bar, v in zip(bars, vals):
        ax.text(v + hi * 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    csv_cols = ["exp_id", "task_id", "thanh_vien", "val_acc", "val_macro_f1", "best_val_loss", "trang_thai_ky_thuat"]
    _write_csv(out_csv, csv_cols, [{k: r.get(k, "") for k in csv_cols} for r in ordered_rows])
    return out_png, out_csv


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sinh biểu đồ curves.png + curves.csv cho một run_dir")
    ap.add_argument("--run-dir", required=True, help="Thư mục run chứa history.csv")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)
    run_dir = Path(a.run_dir)
    png, csv_p = plot_curves(run_dir / "history.csv", run_dir / "curves.png", title=a.title or run_dir.name)
    print(f"Đã ghi: {png} và {csv_p}")


if __name__ == "__main__":
    main()
