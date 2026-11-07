"""Config viết tay phải khớp với log: mỗi giá trị đã chọn là cấu hình có val MSE trung bình thấp nhất.

Chạy lại thực nghiệm mà kết quả đổi, hoặc sửa config sai, thì test này báo ngay.
Bỏ qua (skip) khi chưa có results/ tương ứng, ví dụ trên bản clone chưa chạy thực nghiệm.
"""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
SEARCH_EXPS = ("expA", "expA_steps", "expB", "expB_wide", "expB_A200", "expC", "expC_ctrl")


def best_by_val(exp):
    """-> (value, mean val_mse) có val MSE trung bình thấp nhất trong results/<exp>/summary.csv."""
    path = RES / exp / "summary.csv"
    if not path.exists():
        pytest.skip(f"chưa có {path}")
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups[r["value"]].append(float(r["val_mse"]))
    return min(((v, statistics.mean(m)) for v, m in groups.items()), key=lambda t: t[1])


def config(name):
    return json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))["base"]


def test_baseline_activation_lr_from_grid_baseline():
    value, _ = best_by_val("grid_baseline")          # tên dạng "<lr>_<activation>"
    lr, act = value.split("_")
    base = config("baseline")
    assert (base["lr"], base["activation"]) == (float(lr), act)


def test_baseline_optimizer_from_grid_optimizer():
    value, _ = best_by_val("grid_optimizer")         # tên dạng "<optimizer>_<lr>"
    opt, lr = value.split("_")
    base = config("baseline")
    assert (base["optimizer"], base["lr"]) == (opt, float(lr))


def test_baseline_batch_from_grid_batch():
    value, _ = best_by_val("grid_batch")
    assert config("baseline")["batch_size"] == int(value)


def test_final_is_lowest_val_over_all_experiments():
    best = min((best_by_val(e) + (e,) for e in SEARCH_EXPS if (RES / e / "summary.csv").exists()),
               key=lambda t: t[1])
    value, _, exp = best
    run_cfg = json.loads(next((RES / exp / value).glob("seed_*/config.json")).read_text(encoding="utf-8"))
    final = json.loads((ROOT / "configs" / "final.json").read_text(encoding="utf-8"))
    assert final["vary"]["hidden_sizes"] == [run_cfg["hidden_sizes"]], (exp, value)
    for k in ("n_train", "activation", "optimizer", "lr", "batch_size"):
        assert final["base"][k] == run_cfg[k], k
