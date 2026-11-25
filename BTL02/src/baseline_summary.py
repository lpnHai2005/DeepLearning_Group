"""Tổng hợp baseline 3 seed (T11): mean ± std và dung sai tái lập, sinh từ tracking/experiments.csv.

Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T11.

Chỉ lấy run thỏa cả 4 điều kiện: exp_id dạng `baseline-NNN_sK`, task T11, trạng thái Completed, và mọi khóa trong
`base` của configs/baseline.json trùng với hyperparams_json của run (nên run 60 epoch cũ và run `_probe` bị loại).
Các run hợp lệ phải cùng một số NNN; nếu có nhiều số NNN cùng khớp config thì báo lỗi, không tự chọn.

Dung sai tái lập = độ lệch chuẩn mẫu (ddof=1) của val_acc qua các seed (mục 5.2 của PHAN_CONG_NHIEM_VU.md).
sha256 của baseline.json tính trên nội dung đã đổi CRLF thành LF, để Windows (autocrlf) và Linux ra cùng một số.

Dùng (từ thư mục BTL02):
    python src/baseline_summary.py [--out results/baseline/baseline_summary.json]
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_RE = re.compile(r"^baseline-(\d{3})_s(\d+)$")
METRICS = ("val_acc", "val_macro_f1", "best_val_loss")


def config_sha256_lf(path):
    """sha256 của file config sau khi chuẩn hóa xuống dòng về LF."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def select_runs(rows, base_cfg, task_id="T11"):
    """Trả các dòng baseline Completed có hyperparams khớp base_cfg, cùng một số NNN, sắp theo seed."""
    by_num = {}
    for r in rows:
        m = EXP_RE.match(r.get("exp_id", ""))
        if not m or r.get("task_id") != task_id or r.get("trang_thai_ky_thuat") != "Completed":
            continue
        hp = json.loads(r.get("hyperparams_json") or "{}")
        if all(hp.get(k) == v for k, v in base_cfg.items()):
            by_num.setdefault(m.group(1), []).append(r)
    if not by_num:
        raise SystemExit("không có run baseline Completed nào khớp configs/baseline.json")
    if len(by_num) > 1:
        raise SystemExit(f"nhiều bộ run cùng khớp config: {sorted(by_num)}; cần xử lý tay trước khi tổng hợp")
    runs = next(iter(by_num.values()))
    seeds = [int(r["seed"]) for r in runs]
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"trùng seed trong bộ baseline: {sorted(seeds)}")
    return sorted(runs, key=lambda r: int(r["seed"]))


def summarize(runs, config_path, config_sha):
    out = {
        "task_id": "T11",
        "config_path": config_path,
        "config_sha256_lf": config_sha,
        "n_seeds": len(runs),
        "runs": [
            {
                "exp_id": r["exp_id"],
                "seed": int(r["seed"]),
                "thanh_vien": r["thanh_vien"],
                "may_chay": r["may_chay"],
                "git_commit": r["git_commit"],
                "best_epoch": int(r["best_epoch"]),
                **{k: float(r[k]) for k in METRICS},
            }
            for r in runs
        ],
    }
    for k in METRICS:
        vals = [float(r[k]) for r in runs]
        out[k] = {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else None}
    out["dung_sai_tai_lap_val_acc"] = out["val_acc"]["std"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--experiments", default="tracking/experiments.csv")
    ap.add_argument("--out", default="results/baseline/baseline_summary.json")
    ap.add_argument("--min-seeds", type=int, default=3)
    args = ap.parse_args(argv)

    cfg_path = ROOT / args.config
    base_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))["base"]
    with open(ROOT / args.experiments, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    runs = select_runs(rows, base_cfg)
    if len(runs) < args.min_seeds:
        raise SystemExit(f"mới có {len(runs)} seed ({[r['exp_id'] for r in runs]}), cần {args.min_seeds}")

    out = summarize(runs, args.config, config_sha256_lf(cfg_path))
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for k in METRICS:
        print(f"{k}: {out[k]['mean']:.4f} ± {out[k]['std']:.4f}")
    print(f"dung sai tái lập (std val_acc) = {out['dung_sai_tai_lap_val_acc']:.4f}")
    print(f"runs: {', '.join(r['exp_id'] for r in out['runs'])}")
    print(f"sha256(LF) {args.config} = {out['config_sha256_lf']}")
    print(f"đã ghi {args.out}")
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
