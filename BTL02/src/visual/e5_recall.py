# Phụ trách: TV4 – Huỳnh Viễn Thông
"""Tính recall từng lớp độc lập trên tập PublicTest cho run E5 (WeightedRandomSampler) và xuất hình kèm CSV.

Phụ trách: TV4 – Huỳnh Viễn Thông. Nhiệm vụ T16 (phối hợp C2).

Lưu ý quan trọng:
- CHỈ đánh giá trên tập PublicTest (val_loader), TUYỆT ĐỐI KHÔNG truy cập PrivateTest.
- Lưu tệp dự đoán val_predictions.npz ngay trong run_dir để Tester (TV3) kiểm thử độc lập recall Disgust/Fear.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fer2013 import EMOTIONS
from src.data.loader import make_loaders
from src.engine.evaluate import metrics, predict, save_predictions
from src.models.cnn import build_model
from src.visual.plots import plot_error_histogram, register_figure


def evaluate_run_per_class(run_dir: Path, data_csv: Path):
    meta = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = meta.get("cfg", meta)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    loader_cfg = dict(cfg)
    loader_cfg["data_path"] = str(data_csv)
    _, val_loader = make_loaders(loader_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], **cfg.get("model_kwargs", {}))
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    model.load_state_dict(state)
    model.to(device)

    y_true, y_prob = predict(model, val_loader, tta=False, device=device, amp=cfg.get("amp", True))
    y_pred = y_prob.argmax(axis=1)

    pred_path = save_predictions(run_dir / "val_predictions.npz", y_true, y_prob)
    res = metrics(y_true, y_prob, labels=EMOTIONS)

    # Đối chiếu độc lập recall = tp / support bằng numpy thuần
    rows = []
    for i, name in enumerate(EMOTIONS):
        mask = y_true == i
        support = int(mask.sum())
        tp = int((y_pred[mask] == i).sum())
        rec_indep = (tp / support) if support > 0 else 0.0
        pc = res["per_class"][name]
        assert abs(rec_indep - pc["recall"]) < 1e-12, f"Lệch recall độc lập ở lớp {name}"
        rows.append(
            {
                "label": name,
                "support": support,
                "tp": tp,
                "precision": round(pc["precision"], 6),
                "recall": round(pc["recall"], 6),
                "f1": round(pc["f1"], 6),
            }
        )

    out_csv = run_dir / "per_class_recall.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "support", "tp", "precision", "recall", "f1"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    return cfg, res, rows, y_true, y_pred, pred_path, out_csv


def main():
    parser = argparse.ArgumentParser(description="Tính recall từng lớp độc lập cho E5 (PublicTest).")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/E5/E5-001_s0")
    parser.add_argument("--data-csv", type=Path, default=ROOT / "data/fer2013.csv")
    parser.add_argument("--out-png", type=Path, default=ROOT / "figures/e5_wrs_recall.png")
    parser.add_argument("--manifest", type=Path, default=ROOT / "tracking/figures_manifest.csv")
    args = parser.parse_args()

    cfg, res, rows, y_true, y_pred, pred_path, per_class_csv = evaluate_run_per_class(args.run_dir, args.data_csv)
    exp_id = cfg.get("exp_id", args.run_dir.name)
    variant = cfg.get("variant", "weighted_random_sampler")

    # Đối chiếu với metrics.json gốc do trainer ghi
    metrics_json = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    acc_diff = abs(res["acc"] - float(metrics_json["best_val_acc"]))
    f1_diff = abs(res["macro_f1"] - float(metrics_json["best_val_macro_f1"]))

    png_path, fig_csv = plot_error_histogram(
        y_true,
        y_pred,
        out_png=args.out_png,
        labels=EMOTIONS,
        title=f"E5 ({exp_id} - {variant}): Recall và tỷ lệ sai theo lớp (PublicTest)",
    )
    register_figure(
        manifest_path=args.manifest,
        fig_id="e5_wrs_recall",
        file_hinh=png_path,
        file_du_lieu_dau_vao=fig_csv,
        script_sinh_hinh="src/visual/e5_recall.py",
        lenh_chay=f"python src/visual/e5_recall.py --run-dir results/E5/{exp_id}",
        exp_ids=exp_id,
        task_id="T16",
        dung_o="E5: Recall từng lớp của WeightedRandomSampler (PublicTest)",
        root=ROOT,
    )

    print(f"=== {exp_id} ({variant}) - PublicTest ===")
    print(f"Acc={res['acc']:.6f} (diff={acc_diff:.2e}) | Macro-F1={res['macro_f1']:.6f} (diff={f1_diff:.2e})")
    for r in rows:
        print(
            f"  {r['label']:8s} | support={r['support']:3d} | tp={r['tp']:3d} | "
            f"prec={r['precision']*100:5.2f}% | recall={r['recall']*100:5.2f}% | f1={r['f1']*100:5.2f}%"
        )
    print(f"Saved: {pred_path}, {per_class_csv}, {png_path}, {fig_csv}")


if __name__ == "__main__":
    main()
