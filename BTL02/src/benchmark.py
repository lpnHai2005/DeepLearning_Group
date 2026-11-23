# Phụ trách: TV3 – Võ Hoàng Bảo
"""Đo ngân sách tính toán thật (1 epoch với loader thật + fit()) và lưu results/_bench/bench.csv.

Nhiệm vụ T07 (C1): nối loader (T04) với mô hình (T06) và trainer (T07), chạy thử 1 epoch trên máy riêng
để xác minh mục 2.5 của PHAN_CONG_NHIEM_VU.md (thời gian/epoch và VRAM thực tế).

Chạy từ thư mục BTL02:
    python src/benchmark.py [--out results/_bench/bench.csv]
"""
import argparse
import csv
import platform
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import make_loaders  # noqa: E402
from src.engine.trainer import fit  # noqa: E402
from src.models import build_model, build_transfer, n_params  # noqa: E402

BENCH_CASES = [
    {
        "name": "baseline_48x48x1",
        "kind": "cnn",
        "model": "baseline",
        "resize": 48,
        "channels": 1,
        "batch_size": 128,
    },
    {
        "name": "medium_01_bn_48x48x1",
        "kind": "cnn",
        "model": "medium_01",
        "model_kwargs": {"bn": True},
        "resize": 48,
        "channels": 1,
        "batch_size": 128,
    },
    {
        "name": "resnet18_96x96x3",
        "kind": "transfer",
        "model": "resnet18",
        "resize": 96,
        "channels": 3,
        "batch_size": 128,
    },
]


def run_benchmark(out_csv: Path, cases=BENCH_CASES, csv_path="data/fer2013.csv", amp=True):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    rows = []

    with tempfile.TemporaryDirectory(prefix="btl02_bench_") as tmp:
        tmp_dir = Path(tmp)
        for case in cases:
            cfg = {
                "csv_path": csv_path,
                "resize": case["resize"],
                "channels": case["channels"],
                "batch_size": case["batch_size"],
                "augmentation": "flip_affine",
                "epochs": 1,
                "optimizer": "sgd",
                "lr": 0.01,
                "scheduler": "constant",
                "early_stopping_patience": 0,
                "amp": amp and device == "cuda",
                "device": device,
                "seed": 0,
                "exp_id": f"bench_{case['name']}",
            }
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            train_loader, val_loader = make_loaders(cfg)
            if case["kind"] == "cnn":
                model = build_model(case["model"], **case.get("model_kwargs", {}))
            else:
                model = build_transfer(case["model"], pretrained=False, freeze_backbone=False)

            res = fit(model, train_loader, val_loader, cfg, tmp_dir / case["name"])
            peak_alloc_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device == "cuda" else 0.0
            peak_resv_mb = round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1) if device == "cuda" else 0.0

            row = {
                "model_name": case["name"],
                "hostname": platform.node(),
                "gpu": gpu_name,
                "input_shape": f"{case['resize']}x{case['resize']}x{case['channels']}",
                "batch_size": case["batch_size"],
                "amp": cfg["amp"],
                "n_params": n_params(model),
                "epoch_time_s": round(res["epoch_time_s_tb"], 3),
                "total_run_s": res["train_time_s"],
                "peak_vram_alloc_mb": peak_alloc_mb,
                "peak_vram_reserved_mb": peak_resv_mb,
                "val_acc_ep1": round(res["best_val_acc"], 4),
                "status": res["status"],
            }
            rows.append(row)
            print(
                f"{row['model_name']} | params={row['n_params']:,} | epoch={row['epoch_time_s']}s | "
                f"vram_alloc={row['peak_vram_alloc_mb']}MB | vram_resv={row['peak_vram_reserved_mb']}MB | "
                f"val_acc={row['val_acc_ep1']}"
            )

    fields = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return out_csv, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "_bench" / "bench.csv")
    ap.add_argument("--csv", default="data/fer2013.csv")
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args(argv)
    out, _ = run_benchmark(a.out, csv_path=a.csv, amp=not a.no_amp)
    print(f"Đã lưu benchmark vào {out}")


if __name__ == "__main__":
    main()
