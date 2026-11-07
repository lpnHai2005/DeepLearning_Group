"""Chạy một thí nghiệm từ file cấu hình. (TV2 – T2.3)

    python src/run_experiment.py --config configs/expA.json
    python src/run_experiment.py --config configs/final.json --eval-test   # test chỉ dùng ở bước này

Khóa bắt đầu bằng '_' trong JSON (vd '_note') bị bỏ qua.
'vary' có nhiều khóa thì chạy tích Descartes (vd lưới lr × activation cho baseline).
Mỗi lần chạy lưu results/<exp>/<giá trị>/seed_<k>/{config.json, history.csv, metrics.json, best.pt}
và gom vào results/<exp>/summary.csv. results/<exp>/run.log ghi mỗi run một dòng ngay khi run xong,
nên chạy dở mà dừng vẫn biết đã xong những run nào; chạy lại thì ghi đè, giống summary.csv.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # để chạy được cả `python src/...` lẫn `python -m src...`

import torch  # noqa: E402

from src.data.dataset import load_split, make_loader, subset  # noqa: E402
from src.engine.evaluate import metrics, predict  # noqa: E402
from src.engine.trainer import fit  # noqa: E402
from src.model.mlp import MLP, n_params  # noqa: E402
from src.utils.io import save_json, write_csv  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

SUMMARY_COLS = ["exp", "value", "seed", "n_params", "train_mse", "train_mae", "train_rmse",
                "val_mse", "val_mae", "val_rmse", "test_mse", "test_mae", "test_rmse", "test_r2",
                "best_epoch", "epochs_run", "train_time_s", "time_per_epoch_ms"]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def fmt(v):
    return "-".join(map(str, v)) if isinstance(v, list) else str(v)


def expand(cfg):
    """-> (tên giá trị, tham số đầy đủ) cho từng tổ hợp trong 'vary'; vary rỗng -> ('base', base).

    'cases' (danh sách dict) thay cho 'vary' khi các khóa phải đổi cùng nhau, không theo tích Descartes
    (vd n_train cùng max_epochs/patience để giữ ngân sách bước cập nhật); tên giá trị lấy từ khóa đầu.
    """
    if "cases" in cfg:
        for case in cfg["cases"]:
            yield fmt(next(iter(case.values()))), {**cfg["base"], **case}
        return
    keys = list(cfg["vary"])
    for combo in itertools.product(*(cfg["vary"][k] for k in keys)):
        yield "_".join(map(fmt, combo)) or "base", {**cfg["base"], **dict(zip(keys, combo))}


def run(cfg, eval_test=False):
    torch.set_num_threads(1)           # cùng số luồng cho mọi lần chạy: so sánh thời gian công bằng, tái lập
    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"
    log_path.write_text("", encoding="utf-8")
    x_pool, y_pool = load_split("train_pool")
    x_val, y_val = load_split("val")
    if eval_test:
        x_te, y_te = load_split("test")
    rows = []
    for name, p in expand(cfg):
        x_tr, y_tr = subset(x_pool, y_pool, p["n_train"])
        for seed in cfg["seeds"]:
            run_dir = out / name / f"seed_{seed}"
            set_seed(seed)
            model = MLP(p["hidden_sizes"], p["activation"])
            info = fit(model, make_loader(x_tr, y_tr, p["batch_size"], True, seed),
                       make_loader(x_val, y_val, p["batch_size"], False, seed), p, run_dir)
            train_metrics = metrics(y_tr, predict(model, x_tr))
            val_metrics = metrics(y_val, predict(model, x_val))
            row = {"exp": cfg["exp"], "value": name, "seed": seed, "n_params": n_params(model),
                   **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"val_{k}": v for k, v in val_metrics.items()}, **info,
                   "time_per_epoch_ms": 1000 * info["train_time_s"] / info["epochs_run"]}
            if eval_test:
                row.update({f"test_{k}": v for k, v in metrics(y_te, predict(model, x_te)).items()})
            save_json(run_dir / "config.json", {**p, "seed": seed})
            save_json(run_dir / "metrics.json", row)
            rows.append(row)
            line = (f"[{cfg['exp']}] {name} seed={seed}  val_mse={row['val_mse']:.3e}  "
                    f"best_epoch={info['best_epoch']}/{info['epochs_run']}  {info['train_time_s']:.1f}s")
            print(line, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                print(line, file=f)
    write_csv(out / "summary.csv", rows, SUMMARY_COLS)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="đường dẫn tới configs/*.json")
    ap.add_argument("--eval-test", action="store_true", help="đánh giá trên tập test (chỉ dùng ở bước cuối)")
    args = ap.parse_args()
    run(load_config(args.config), args.eval_test)


if __name__ == "__main__":
    main()
