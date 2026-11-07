"""Hậu kiểm (post-hoc) tập test cho từng run của A/B/C, ghi lại để truy vết từng seed. (Plan 1.1)

Chống rò rỉ: run_experiment.py chỉ train/chọn checkpoint theo validation; script này CHỈ nạp lại
best.pt đã đóng băng rồi đo trên test.csv, nên số test không thể ảnh hưởng tới việc chọn cấu hình.
Kết quả được dùng để báo cáo (bảng/đồ thị train–val–test), không dùng để chọn mô hình.

    python src/posthoc_test.py            # chạy sau run_experiment.py; chạy lại run_experiment thì chạy lại script này

Với mỗi results/<exp>/:
  - điền các cột test_* đang trống trong summary.csv (mỗi run 1 dòng),
  - thêm test_* vào metrics.json của từng seed,
  - ghi posthoc_test.log (mỗi run 1 dòng, kèm thời điểm chạy).
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.data.dataset import load_split  # noqa: E402
from src.engine.evaluate import metrics, predict  # noqa: E402
from src.model.mlp import MLP  # noqa: E402
from src.run_experiment import SUMMARY_COLS  # noqa: E402
from src.utils.io import save_json, write_csv  # noqa: E402

RES = ROOT / "results"
EXPS = ("expA", "expA_steps", "expB", "expB_wide", "expB_A200", "expC", "expC_ctrl", "grid_batch")


def load_run_model(run_dir):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = MLP(cfg["hidden_sizes"], cfg["activation"])
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    return model


def posthoc_exp(exp_dir, x_te, y_te):
    """-> list các dòng summary đã có test_*; ghi lại summary.csv, metrics.json, posthoc_test.log."""
    with (exp_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    stamp = datetime.now().isoformat(timespec="seconds")
    lines = [f"# post-hoc test trên best.pt đã đóng băng (không dùng để chọn cấu hình) — {stamp}"]
    for r in rows:
        run_dir = exp_dir / r["value"] / f"seed_{r['seed']}"
        te = metrics(y_te, predict(load_run_model(run_dir), x_te))
        r.update({f"test_{k}": v for k, v in te.items()})
        m_path = run_dir / "metrics.json"
        m = json.loads(m_path.read_text(encoding="utf-8"))
        save_json(m_path, {**m, **{f"test_{k}": v for k, v in te.items()}})
        lines.append(f"[{r['exp']}] {r['value']} seed={r['seed']}  val_mse={float(r['val_mse']):.3e}  "
                     f"test_mse={te['mse']:.3e}  test_mae={te['mae']:.3e}  "
                     f"test_rmse={te['rmse']:.3e}  test_r2={te['r2']:.6f}")
    write_csv(exp_dir / "summary.csv", rows, SUMMARY_COLS)
    (exp_dir / "posthoc_test.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main():
    x_te, y_te = load_split("test")
    for name in EXPS:
        exp_dir = RES / name
        if not (exp_dir / "summary.csv").exists():
            print(f"skip {name}: no summary.csv")
            continue
        rows = posthoc_exp(exp_dir, x_te, y_te)
        print(f"{name}: {len(rows)} run -> summary.csv, metrics.json, posthoc_test.log")


if __name__ == "__main__":
    main()
