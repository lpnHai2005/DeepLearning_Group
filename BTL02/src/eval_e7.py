"""Tổng hợp mô hình cuối E7 (3 seed, TTA, ensemble) trên PublicTest và xuất ca đúng/sai cho Grad-CAM (T20).

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T18.

Dùng (từ thư mục BTL02):
    python src/eval_e7.py --run-dirs results/E7/E7-001_s0 results/E7/E7-001_s1 results/E7/E7-001_s2
    python src/eval_e7.py --run-dirs results/E7/E7-001_s0 results/E7/E7-001_s1 results/E7/E7-001_s2 --tta none flip ten_crop

Quy tắc:
- CHỈ đánh giá trên PublicTest (val_loader từ make_loaders); tuyệt đối KHÔNG gọi make_test_loader / PrivateTest.
- Mọi bảng và biểu đồ đều xuất kèm file CSV và đăng ký vào tracking/figures_manifest.csv.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # chạy được cả dạng `python src/eval_e7.py`

from src.engine.evaluate import EMOTIONS, metrics, predict, save_predictions  # noqa: E402
from src.run_experiment import _collect_loader_images, _load_checkpoint_model  # noqa: E402
from src.utils.io import BTL02, rel, sha256_file, write_json  # noqa: E402
from src.visual.plots import (  # noqa: E402
    _apply_style,
    plot_confusion_matrix,
    plot_error_histogram,
    plot_sample_grid,
    register_figure,
)


def _nll_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    probs = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1.0)
    idx = np.arange(len(y_true))
    return float(-np.mean(np.log(probs[idx, np.asarray(y_true, dtype=np.int64)])))


def _mean_std_fmt(vals: list[float]) -> tuple[float, float | None, str]:
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) >= 2:
        std = float(arr.std(ddof=1))
        return round(mean, 6), round(std, 6), f"{mean:.4f} ± {std:.4f}"
    return round(mean, 6), None, f"{mean:.4f}"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return path


def _select_gradcam_cases(y_true: np.ndarray, y_prob: np.ndarray, per_class_correct: int = 2, max_wrong: int = 16) -> list[dict]:
    """Chọn các ca đúng tự tin cao (đều 7 lớp) và ca sai tự tin cao để phục vụ Grad-CAM (T20) & phân tích lỗi (T23)."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_prob, dtype=np.float32)
    pred = yp.argmax(axis=1).astype(np.int64)
    conf = yp[np.arange(len(yt)), pred]
    true_p = yp[np.arange(len(yt)), yt]

    rows = []
    # Ca đúng: chọn top tự tin cao nhất cho từng lớp cảm xúc
    for cls_idx, cls_name in enumerate(EMOTIONS):
        mask = (yt == cls_idx) & (pred == cls_idx)
        idxs = np.flatnonzero(mask)
        if len(idxs) == 0:
            continue
        order = idxs[np.argsort(-conf[idxs])][:per_class_correct]
        for i in order:
            rows.append(
                {
                    "case_type": "correct",
                    "sample_idx": int(i),
                    "true_idx": int(yt[i]),
                    "true_label": cls_name,
                    "pred_idx": int(pred[i]),
                    "pred_label": EMOTIONS[int(pred[i])],
                    "confidence": round(float(conf[i]), 6),
                    "true_prob": round(float(true_p[i]), 6),
                    "margin": round(float(conf[i] - true_p[i]), 6),
                }
            )

    # Ca sai: ưu tiên các ca mô hình đoán sai với độ tự tin cao nhất
    wrong_idxs = np.flatnonzero(yt != pred)
    if len(wrong_idxs) > 0:
        wrong_order = wrong_idxs[np.argsort(-conf[wrong_idxs])][:max_wrong]
        for i in wrong_order:
            rows.append(
                {
                    "case_type": "wrong",
                    "sample_idx": int(i),
                    "true_idx": int(yt[i]),
                    "true_label": EMOTIONS[int(yt[i])],
                    "pred_idx": int(pred[i]),
                    "pred_label": EMOTIONS[int(pred[i])],
                    "confidence": round(float(conf[i]), 6),
                    "true_prob": round(float(true_p[i]), 6),
                    "margin": round(float(conf[i] - true_p[i]), 6),
                }
            )
    return rows


def _plot_tta_ensemble_bars(summary_rows: list[dict], out_png: Path, out_csv: Path | None = None):
    """Vẽ biểu đồ cột so sánh Single-seed (mean ± std) và Ensemble theo từng chế độ TTA trên PublicTest."""
    _apply_style()
    out_png = Path(out_png)
    out_csv = Path(out_csv) if out_csv is not None else out_png.with_suffix(".csv")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    modes = []
    for r in summary_rows:
        if r["tta"] not in modes:
            modes.append(r["tta"])

    single_map = {r["tta"]: r for r in summary_rows if r["eval_type"] == "single_mean"}
    ens_map = {r["tta"]: r for r in summary_rows if r["eval_type"] == "ensemble"}

    x = np.arange(len(modes))
    w = 0.35
    s_acc = [float(single_map[m]["val_acc_mean"]) for m in modes]
    s_std = [float(single_map[m]["val_acc_std"] or 0.0) for m in modes]
    e_acc = [float(ens_map[m]["val_acc_mean"]) for m in modes]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    b1 = ax.bar(x - w / 2, s_acc, yerr=s_std, capsize=4, width=w, label="Đơn lẻ (mean ± std)", color="#3182ce")
    b2 = ax.bar(x + w / 2, e_acc, width=w, label="Ensemble", color="#2f855a")
    ax.set_xticks(x)
    ax.set_xticklabels([f"TTA: {m}" for m in modes])
    lo = max(0.0, min(s_acc + e_acc) - 0.05)
    hi = min(1.0, max(s_acc + e_acc) + 0.05)
    ax.set_ylim(lo, hi if hi > lo else 1.0)
    ax.set_ylabel("Validation Accuracy (PublicTest)")
    ax.set_title("So sánh E7 đơn lẻ (mean ± std) và Ensemble theo chế độ TTA")
    ax.legend()

    for bar, v in zip(b1, s_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)
    for bar, v in zip(b2, e_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    cols = ["tta", "eval_type", "n_models", "val_acc_mean", "val_acc_std", "val_macro_f1_mean", "val_macro_f1_std", "val_nll"]
    _write_csv(out_csv, cols, [{k: r.get(k, "") for k in cols} for r in summary_rows])
    return out_png, out_csv


def _write_markdown_table(path: Path, summary_rows: list[dict], seed_rows: list[dict]) -> Path:
    lines = [
        "**Bảng 4.7b: Đánh giá TTA và Ensemble mô hình cuối E7 trên PublicTest (Validation)**",
        "",
        "| Chế độ TTA | Loại đánh giá | Số seed | Val acc (mean ± std) | Val macro-F1 (mean ± std) | Val NLL |",
        "|---|---|---:|---|---|---:|",
    ]
    for r in summary_rows:
        kind = "Đơn lẻ (trung bình)" if r["eval_type"] == "single_mean" else "**Ensemble**"
        lines.append(
            f"| `{r['tta']}` | {kind} | {r['n_models']} | {r['val_acc_fmt']} | {r['val_macro_f1_fmt']} | {r['val_nll']:.4f} |"
        )
    lines.extend(
        [
            "",
            "**Chi tiết từng seed trên PublicTest**",
            "",
            "| Mã run | Seed | Chế độ TTA | Val acc | Val macro-F1 | Val NLL |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for r in seed_rows:
        lines.append(
            f"| `{r['exp_id']}` | {r['seed']} | `{r['tta']}` | {r['val_acc']:.4f} | {r['val_macro_f1']:.4f} | {r['val_nll']:.4f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def evaluate_e7(
    run_dirs,
    tta_modes=("none", "flip", "ten_crop"),
    out_dir=None,
    figures_dir=None,
    tables_dir=None,
    root=BTL02,
    device=None,
    register_figures=True,
    cli_command=None,
):
    """Đánh giá đầy đủ 3 seed, TTA và Ensemble cho E7 trên PublicTest (CHỈ validation, không chạm PrivateTest)."""
    if not run_dirs:
        raise SystemExit("Cần truyền ít nhất 1 thư mục run qua --run-dirs.")
    root = Path(root)
    resolved_dirs = [Path(d) if Path(d).is_absolute() else (root / d) for d in run_dirs]

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_dir) if out_dir else (root / "results" / "E7")
    if not out_path.is_absolute():
        out_path = root / out_path
    fig_path = Path(figures_dir) if figures_dir else (root / "figures")
    if not fig_path.is_absolute():
        fig_path = root / fig_path
    tbl_path = Path(tables_dir) if tables_dir else (root / "report" / "tables")
    if not tbl_path.is_absolute():
        tbl_path = root / tbl_path
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path.mkdir(parents=True, exist_ok=True)
    tbl_path.mkdir(parents=True, exist_ok=True)

    loaded = []
    input_sig = None
    for rd in resolved_dirs:
        model, meta, cfg, ckpt_path = _load_checkpoint_model(rd, dev)
        sig = (int(cfg.get("resize", 48)), int(cfg.get("channels", 1)))
        if input_sig is None:
            input_sig = sig
        elif sig != input_sig:
            raise SystemExit(f"Các checkpoint không đồng nhất kích thước/kênh đầu vào: {input_sig} vs {sig} ({rd})")
        loaded.append((model, meta, cfg, ckpt_path, rd))

    loader_cfg = dict(loaded[0][2])
    loader_cfg["device"] = str(dev)
    loader_cfg["augmentation"] = "none"

    from src.data.loader import make_loaders  # CHỈ dùng val_loader (PublicTest), tuyệt đối không chạm PrivateTest

    _, val_loader = make_loaders(loader_cfg)
    images = _collect_loader_images(val_loader)
    use_amp = bool(loader_cfg.get("amp", True)) and dev.type == "cuda"

    seed_rows = []
    summary_rows = []
    ens_outputs = {}
    y_true_ref = None

    for tta_mode in tta_modes:
        probs_list = []
        accs, f1s, nlls = [], [], []
        for model, meta, cfg, ckpt_path, rd in loaded:
            yt, yp = predict(model, val_loader, tta=tta_mode, device=dev, amp=use_amp)
            if y_true_ref is None:
                y_true_ref = yt
            elif not np.array_equal(y_true_ref, yt):
                raise RuntimeError("Thứ tự nhãn PublicTest giữa các lần duyệt không khớp!")
            m = metrics(yt, yp)
            nll = _nll_from_probs(yt, yp)
            probs_list.append(yp)
            accs.append(m["acc"])
            f1s.append(m["macro_f1"])
            nlls.append(nll)
            seed_rows.append(
                {
                    "exp_id": meta.get("exp_id", rd.name),
                    "seed": cfg.get("seed", ""),
                    "tta": tta_mode,
                    "val_acc": round(m["acc"], 6),
                    "val_macro_f1": round(m["macro_f1"], 6),
                    "val_nll": round(nll, 6),
                    "weights_sha256": sha256_file(ckpt_path),
                }
            )

        acc_mean, acc_std, acc_fmt = _mean_std_fmt(accs)
        f1_mean, f1_std, f1_fmt = _mean_std_fmt(f1s)
        summary_rows.append(
            {
                "tta": tta_mode,
                "eval_type": "single_mean",
                "n_models": len(loaded),
                "val_acc_mean": acc_mean,
                "val_acc_std": acc_std if acc_std is not None else "",
                "val_acc_fmt": acc_fmt,
                "val_macro_f1_mean": f1_mean,
                "val_macro_f1_std": f1_std if f1_std is not None else "",
                "val_macro_f1_fmt": f1_fmt,
                "val_nll": round(float(np.mean(nlls)), 6),
            }
        )

        y_prob_ens = np.mean(probs_list, axis=0).astype(np.float32)
        ens_m = metrics(y_true_ref, y_prob_ens)
        ens_nll = _nll_from_probs(y_true_ref, y_prob_ens)
        pred_file = save_predictions(out_path / f"val_predictions_ens_{tta_mode}.npz", y_true_ref, y_prob_ens)
        ens_outputs[tta_mode] = {
            "y_prob": y_prob_ens,
            "metrics": ens_m,
            "nll": ens_nll,
            "predictions_path": rel(pred_file, root),
        }
        summary_rows.append(
            {
                "tta": tta_mode,
                "eval_type": "ensemble",
                "n_models": len(loaded),
                "val_acc_mean": round(ens_m["acc"], 6),
                "val_acc_std": "",
                "val_acc_fmt": f"{ens_m['acc']:.4f}",
                "val_macro_f1_mean": round(ens_m["macro_f1"], 6),
                "val_macro_f1_std": "",
                "val_macro_f1_fmt": f"{ens_m['macro_f1']:.4f}",
                "val_nll": round(ens_nll, 6),
            }
        )

    best_tta = max(tta_modes, key=lambda m: (ens_outputs[m]["metrics"]["acc"], ens_outputs[m]["metrics"]["macro_f1"]))
    best_out = ens_outputs[best_tta]
    best_prob = best_out["y_prob"]
    best_pred = best_prob.argmax(axis=1).astype(np.int64)
    best_m = best_out["metrics"]

    best_pred_file = save_predictions(out_path / "e7_val_best_predictions.npz", y_true_ref, best_prob)

    per_class_rows = []
    for label in EMOTIONS:
        st = best_m["per_class"][label]
        per_class_rows.append(
            {
                "label": label,
                "precision": round(st["precision"], 6),
                "recall": round(st["recall"], 6),
                "f1": round(st["f1"], 6),
                "support": st["support"],
            }
        )
    per_class_csv = _write_csv(out_path / "e7_val_per_class.csv", ["label", "precision", "recall", "f1", "support"], per_class_rows)

    gradcam_rows = _select_gradcam_cases(y_true_ref, best_prob)
    gradcam_csv = _write_csv(
        out_path / "e7_gradcam_cases.csv",
        ["case_type", "sample_idx", "true_idx", "true_label", "pred_idx", "pred_label", "confidence", "true_prob", "margin"],
        gradcam_rows,
    )

    summary_csv = _write_csv(
        tbl_path / "E7_tta_ensemble.csv",
        ["tta", "eval_type", "n_models", "val_acc_mean", "val_acc_std", "val_acc_fmt", "val_macro_f1_mean", "val_macro_f1_std", "val_macro_f1_fmt", "val_nll"],
        summary_rows,
    )
    summary_md = _write_markdown_table(tbl_path / "E7_tta_ensemble.md", summary_rows, seed_rows)

    exp_ids = [meta.get("exp_id", rd.name) for _, meta, _, _, rd in loaded]
    exp_ids_joined = ";".join(exp_ids)
    cmd_str = cli_command or (
        f"python src/eval_e7.py --run-dirs {' '.join(rel(rd, root) for _, _, _, _, rd in loaded)} "
        f"--tta {' '.join(tta_modes)}"
    )

    bar_png, bar_csv = _plot_tta_ensemble_bars(summary_rows, fig_path / "e7_val_tta_ensemble.png")
    cm_png, cm_csv = plot_confusion_matrix(
        best_m["cm"],
        fig_path / "e7_val_confusion.png",
        title=f"Ma trận nhầm lẫn PublicTest — E7 Ensemble ({len(loaded)} seed, TTA={best_tta}, acc={best_m['acc']*100:.2f}%)",
    )
    err_png, err_csv = plot_error_histogram(
        y_true_ref,
        best_pred,
        fig_path / "e7_val_error_hist.png",
        title=f"Recall và tỷ lệ sai theo lớp trên PublicTest — E7 Ensemble (TTA={best_tta})",
    )
    ok_png, ok_csv = plot_sample_grid(
        images,
        y_true_ref,
        best_pred,
        fig_path / "e7_val_correct_grid.png",
        y_prob=best_prob,
        correct=True,
        max_samples=16,
        title=f"PublicTest — Mẫu dự đoán ĐÚNG (E7 Ensemble, TTA={best_tta})",
    )
    bad_png, bad_csv = plot_sample_grid(
        images,
        y_true_ref,
        best_pred,
        fig_path / "e7_val_wrong_grid.png",
        y_prob=best_prob,
        correct=False,
        max_samples=16,
        title=f"PublicTest — Mẫu dự đoán SAI (E7 Ensemble, TTA={best_tta})",
    )

    if register_figures:
        manifest_path = root / "tracking" / "figures_manifest.csv"
        for fid, png_p, csv_p, usage in [
            ("e7_val_tta_ensemble", bar_png, bar_csv, "Báo cáo Ch.4 & Slide E7"),
            ("e7_val_confusion", cm_png, cm_csv, "Báo cáo Ch.4 ma trận nhầm lẫn E7"),
            ("e7_val_error_hist", err_png, err_csv, "Báo cáo Ch.4 phân tích lỗi E7"),
            ("e7_val_correct_grid", ok_png, ok_csv, "Báo cáo Ch.4 ca đúng E7"),
            ("e7_val_wrong_grid", bad_png, bad_csv, "Báo cáo Ch.4 ca sai E7"),
        ]:
            register_figure(
                manifest_path,
                fid,
                png_p,
                csv_p,
                "src/eval_e7.py",
                cmd_str,
                exp_ids=exp_ids_joined,
                task_id="T18",
                dung_o=usage,
                root=root,
            )

    report = {
        "task_id": "T18",
        "split": "PublicTest",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": cmd_str,
        "exp_ids": exp_ids,
        "tta_modes": list(tta_modes),
        "best_tta": best_tta,
        "best_val_acc": best_m["acc"],
        "best_val_macro_f1": best_m["macro_f1"],
        "best_val_nll": best_out["nll"],
        "summary": summary_rows,
        "per_seed": seed_rows,
        "per_class": best_m["per_class"],
        "artifacts": {
            "best_predictions": rel(best_pred_file, root),
            "per_class_csv": rel(per_class_csv, root),
            "gradcam_cases_csv": rel(gradcam_csv, root),
            "summary_csv": rel(summary_csv, root),
            "summary_md": rel(summary_md, root),
        },
    }
    write_json(out_path / "e7_val_summary.json", report)
    print(
        f"[T18 eval_e7] Hoàn tất đánh giá {len(loaded)} seed trên PublicTest | "
        f"Best Ensemble (TTA={best_tta}): val_acc={best_m['acc']:.4f}, macro-F1={best_m['macro_f1']:.4f}"
    )
    return report


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Đánh giá 3 seed, TTA và Ensemble mô hình cuối E7 trên PublicTest (T18)")
    ap.add_argument("--run-dirs", nargs="+", required=True, help="Danh sách thư mục run chứa config.json và best.pt")
    ap.add_argument("--tta", nargs="+", default=["none", "flip", "ten_crop"], choices=["none", "flip", "ten_crop"])
    ap.add_argument("--out-dir", default=None, help="Thư mục lưu kết quả tổng hợp (mặc định results/E7)")
    ap.add_argument("--device", default=None, help="Thiết bị chạy (cuda/cpu)")
    a = ap.parse_args(argv)
    cmd_str = "python src/eval_e7.py " + " ".join(argv if argv is not None else sys.argv[1:])
    evaluate_e7(run_dirs=a.run_dirs, tta_modes=tuple(a.tta), out_dir=a.out_dir, device=a.device, cli_command=cmd_str)


if __name__ == "__main__":
    main()
