"""Chạy một thí nghiệm từ file cấu hình. (TV2 – T2.3)

    python src/run_experiment.py --config configs/expA.json
    python src/run_experiment.py --config configs/final.json --eval-test   # test chỉ dùng ở bước này

Khóa bắt đầu bằng '_' trong JSON (vd '_note') bị bỏ qua.
Mỗi lần chạy lưu results/<exp>/<giá trị>/seed_<k>/{config.json, history.csv, metrics.json, best.pt}
và gom vào results/<exp>/summary.csv.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # để chạy được cả `python src/...` lẫn `python -m src...`


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="đường dẫn tới configs/*.json")
    ap.add_argument("--eval-test", action="store_true", help="đánh giá trên tập test (chỉ dùng ở bước cuối)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raise NotImplementedError(f"TV2 - T2.3: chạy thí nghiệm '{cfg['exp']}'")


if __name__ == "__main__":
    main()
