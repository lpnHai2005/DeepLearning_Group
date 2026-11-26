"""Phân tích E6 (transfer learning): bảng so sánh các run và giải thích "fail case" VGG16 ở 48 px bằng bản đồ đặc trưng.

Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T17.

Mọi số trong bảng sinh từ tracking/experiments.csv, results/E6/<run>/history.csv, metrics.json và từ việc chạy thật
VGG16 torchvision (trọng số ngẫu nhiên, chỉ đo hình dạng, không cần mạng). Không chạy PrivateTest.

Dùng (từ thư mục BTL02):
    python src/e6_analysis.py [--out report/tables/E6_analysis.md]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.make_tables import baseline_stats, load_experiments  # noqa: E402
from src.models.transfer import build_transfer  # noqa: E402

# Thứ tự trình bày: VGG16 hai cỡ ảnh, rồi ResNet18 từ đầu / pretrained fine-tune / pretrained đóng băng.
VARIANTS = ["vgg16_48_ft", "vgg16_96_ft", "resnet18_96_scratch", "resnet18_96_ft", "resnet18_96_frozen"]
LABELS = {
    "vgg16_48_ft": "VGG16 pretrained, fine-tune, 48 px",
    "vgg16_96_ft": "VGG16 pretrained, fine-tune, 96 px (30 epoch)",
    "resnet18_96_scratch": "ResNet18 từ đầu, 96 px",
    "resnet18_96_ft": "ResNet18 pretrained, fine-tune, 96 px",
    "resnet18_96_frozen": "ResNet18 pretrained, đóng băng backbone, 96 px",
}
THRESHOLDS = (0.60, 0.65)


def vgg_feature_maps(size):
    """Đo thật trên VGG16: kích thước sau từng MaxPool, sau AdaptiveAvgPool, tỉ lệ ô của bản đồ conv5 mà pool5 dùng tới,
    số giá trị khác nhau trong đầu ra avgpool (mỗi kênh) và số tham số FC1."""
    m = build_transfer("vgg16").eval()
    pools = [l for l in m.features if isinstance(l, nn.MaxPool2d)]
    shapes, ins = [], {}
    hooks = [p.register_forward_hook(lambda mod, i, o: shapes.append(tuple(o.shape[1:]))) for p in pools]
    hooks.append(pools[-1].register_forward_hook(lambda mod, i, o: ins.__setitem__("pool5_in", i[0])))
    hooks.append(m.avgpool.register_forward_hook(lambda mod, i, o: ins.__setitem__("avg_out", o)))
    with torch.no_grad():
        m(torch.randn(1, 3, size, size))
    for h in hooks:
        h.remove()
    side_in = ins["pool5_in"].shape[-1]
    used = (side_in // 2) * 2  # MaxPool 2x2 stride 2, ceil_mode=False: hàng/cột lẻ cuối bị bỏ
    avg = ins["avg_out"][0]  # (512, 7, 7)
    flat = avg.reshape(avg.shape[0], -1)
    distinct = max(len(torch.unique(flat[c])) for c in range(flat.shape[0]))
    fc1 = m.classifier[0]
    rf, jump = 1, 1  # vùng tiếp nhận của conv cuối (conv5_3), tính từ kernel/stride của features
    for layer in list(m.features)[:-1]:
        if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
            k = layer.kernel_size if isinstance(layer.kernel_size, int) else layer.kernel_size[0]
            st = layer.stride if isinstance(layer.stride, int) else layer.stride[0]
            rf, jump = rf + (k - 1) * jump, jump * st
    return {
        "rf_conv5": rf,
        "size": size,
        "pools": shapes,
        "conv5_side": side_in,
        "conv5_used": f"{used * used}/{side_in * side_in}",
        "avgpool": tuple(avg.shape),
        "distinct_per_channel": distinct,
        "fc1_in": fc1.in_features,
        "fc1_params": sum(p.numel() for p in fc1.parameters()),
        "pool5_cells": shapes[-1][1] * shapes[-1][2],
        # avgpool là phép tuyến tính của pool5: FC1 tương đương một lớp Linear(512 * số ô pool5, 4096)
        "fc1_effective_params": shapes[-1][1] * shapes[-1][2] * 512 * fc1.out_features + fc1.out_features,
        "total_params": sum(p.numel() for p in m.parameters()),
    }


def _history(root, run_dir):
    path = Path(root) / run_dir / "history.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _first_epoch(hist, thr):
    return next((int(r["epoch"]) for r in hist if r["val_acc"] >= thr), None)


def run_rows(experiments_csv, root=ROOT):
    """Một dòng cho mỗi biến thể E6 Completed (lấy run mới nhất nếu chạy lại)."""
    rows = load_experiments(experiments_csv)
    latest = {}
    for r in rows:
        if r.get("task_id") != "T17" or r.get("trang_thai_ky_thuat") != "Completed":
            continue
        latest[json.loads(r["hyperparams_json"])["variant"]] = r
    out = []
    for v in VARIANTS:
        r = latest.get(v)
        if r is None:
            continue
        hist = _history(root, r["run_dir"])
        metrics = json.loads((Path(root) / r["run_dir"] / "metrics.json").read_text(encoding="utf-8"))
        out.append({
            "variant": v, "exp_id": r["exp_id"], "thanh_vien": r["thanh_vien"],
            "val_acc": float(r["val_acc"]), "val_macro_f1": float(r["val_macro_f1"]),
            "best_val_loss": float(r["best_val_loss"]), "best_epoch": int(float(r["best_epoch"])),
            "epochs_run": int(metrics["epochs_run"]), "epochs_max": json.loads(r["hyperparams_json"])["epochs"],
            "ep1_val_acc": hist[0]["val_acc"],
            **{f"ep_to_{int(t * 100)}": _first_epoch(hist, t) for t in THRESHOLDS},
            "train_acc_cuoi": float(r["train_acc_cuoi"]),
            "gap": float(r["train_acc_cuoi"]) - float(r["val_acc"]),
            "n_params": int(float(r["n_params"])), "epoch_time_s": float(r["epoch_time_s_tb"]),
            "gpu": r["may_chay"].split("GeForce ")[-1].replace(" Laptop GPU", ""),
            "weights_mb": float(r["weights_size_mb"]) if r.get("weights_size_mb") else None,
        })
    return out, baseline_stats(rows)


def _ep(x):
    return "không đạt" if x is None else str(x)


def render(runs, base, vgg48, vgg96):
    mean, std = base
    lines = ["# E6 – Transfer learning: bảng tổng hợp và giải thích VGG16 ở 48 px", "",
             "_Sinh tự động bằng `python src/e6_analysis.py` từ `tracking/experiments.csv`, `results/E6/*/history.csv`, "
             "`metrics.json` và từ việc chạy thật VGG16 torchvision. Không sửa tay. Mỗi biến thể 1 seed (seed 0)._", "",
             f"Baseline (T11, 3 seed): val acc {mean:.4f} ± {std:.4f}.", "",
             "## Bảng E6.1 – Toàn bộ biến thể", "",
             "| Cấu hình | Run | Val acc | Δ baseline | Val macro-F1 | Best val loss | Best epoch / chạy / tối đa "
             "| Train acc cuối | Train − Val | Tham số | Weights (MB) | s/epoch (GPU) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        d = r["val_acc"] - mean
        tag = "< std" if abs(d) < std else ("cao hơn" if d > 0 else "thấp hơn")
        lines.append(
            f"| {LABELS[r['variant']]} | {r['exp_id']} ({r['thanh_vien']}) | {r['val_acc']:.4f} | {d:+.4f} ({tag}) "
            f"| {r['val_macro_f1']:.4f} | {r['best_val_loss']:.4f} | {r['best_epoch']} / {r['epochs_run']} / {r['epochs_max']} "
            f"| {r['train_acc_cuoi']:.4f} | {r['gap']:+.4f} | {r['n_params']:,} | {r['weights_mb']:.1f} "
            f"| {r['epoch_time_s']:.1f} ({r['gpu']}) |")
    by = {r["variant"]: r for r in runs}
    res = [by[v] for v in ("resnet18_96_scratch", "resnet18_96_ft", "resnet18_96_frozen") if v in by]
    if res:
        lines += ["", "## Bảng E6.2 – ResNet18 96 px: từ đầu so với pretrained", "",
                  "| Cấu hình | Val acc | Val macro-F1 | Best val loss | Val acc epoch 1 | Epoch đầu tiên val ≥ 0.60 "
                  "| Epoch đầu tiên val ≥ 0.65 | Best epoch | Train − Val |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in res:
            lines.append(f"| {LABELS[r['variant']]} | {r['val_acc']:.4f} | {r['val_macro_f1']:.4f} | {r['best_val_loss']:.4f} "
                         f"| {r['ep1_val_acc']:.4f} | {_ep(r['ep_to_60'])} | {_ep(r['ep_to_65'])} | {r['best_epoch']} "
                         f"| {r['gap']:+.4f} |")
    lines += ["", "## Bảng E6.3 – Kích thước bản đồ đặc trưng VGG16 (đo bằng forward thật)", "",
              "| Đầu vào | Sau pool1 → pool5 | Bản đồ conv5 vào pool5 | Vùng tiếp nhận conv5_3 (px) | Ô conv5 pool5 dùng tới | Sau AdaptiveAvgPool "
              "| Giá trị khác nhau / kênh trong 7×7 | Ô pool5 (thông tin thật) | FC1: đầu vào, tham số | FC1: tham số hiệu dụng |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for g in (vgg48, vgg96):
        chain = " → ".join(f"{s[0]}×{s[1]}×{s[2]}" for s in g["pools"])
        lines.append(f"| {g['size']}×{g['size']}×3 | {chain} | {g['conv5_side']}×{g['conv5_side']} | {g['rf_conv5']} | {g['conv5_used']} "
                     f"| {'×'.join(map(str, g['avgpool']))} | {g['distinct_per_channel']} | {g['pool5_cells']} "
                     f"| {g['fc1_in']:,}, {g['fc1_params']:,} | {g['fc1_effective_params']:,} |")
    lines += ["", "_AdaptiveAvgPool chỉ lặp/trung bình các ô của pool5 nên là phép tuyến tính; FC1 tương đương Linear(512 × số ô pool5, "
              "4096), tham số hiệu dụng = 512 × số ô pool5 × 4096 + 4096. Ở 48 px cả 49 ô của bản đồ 7×7 là bản sao của "
              "một vector 512 chiều, nên FC1 chỉ thấy 512 số thật dù có 25,088 đầu vào._", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--experiments", default=str(ROOT / "tracking" / "experiments.csv"))
    ap.add_argument("--out", default=str(ROOT / "report" / "tables" / "E6_analysis.md"))
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    runs, base = run_rows(a.experiments)
    text = render(runs, base, vgg_feature_maps(48), vgg_feature_maps(96))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print("đã ghi", a.out)


if __name__ == "__main__":
    main()
