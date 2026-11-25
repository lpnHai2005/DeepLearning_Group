"""Unit test cho src/make_tables.py.

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T09.
Chạy từ thư mục BTL02: python -m pytest tests/test_make_tables.py -v
"""
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from src.make_tables import (
    baseline_stats,
    compare_to_baseline,
    export_tables,
    summarize_rows,
    update_experiment_log,
)

REAL_HEADER = (Path(__file__).resolve().parents[1] / "tracking" / "experiments.csv").read_text(encoding="utf-8").splitlines()[0].split(",")


def _row(exp_id, task_id, variant, seed, val_acc, f1=0.6, loss=0.9, status="Completed"):
    r = {c: "" for c in REAL_HEADER}
    r.update(
        {
            "exp_id": exp_id,
            "task_id": task_id,
            "thanh_vien": "TV3",
            "epoch_time_s_tb": "12.5",
            "hyperparams_json": json.dumps({"variant": variant, "seed": seed}),
            "seed": str(seed),
            "val_acc": "" if val_acc is None else str(val_acc),
            "val_macro_f1": "" if f1 is None else str(f1),
            "best_val_loss": "" if loss is None else str(loss),
            "trang_thai_ky_thuat": status,
        }
    )
    return r


def test_mean_std_va_xuat_bang_khop_pandas(tmp_path):
    exp_csv = tmp_path / "experiments.csv"
    data = [
        _row("B0-001_s0", "T11", "baseline", 0, 0.6500, 0.60, 1.00),
        _row("B0-001_s1", "T11", "baseline", 1, 0.6600, 0.61, 0.98),
        _row("B0-001_s2", "T11", "baseline", 2, 0.6550, 0.605, 0.99),
        _row("E3-001_s0", "T14", "adamw_1e-3", 0, 0.6800, 0.64, 0.90),
        _row("E3-002_s0", "T14", "bad_lr", 0, None, None, None, status="Failed"),
    ]
    with open(exp_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REAL_HEADER)
        w.writeheader()
        w.writerows(data)

    written = export_tables(exp_csv, tmp_path / "tables")
    names = {p.name for p in written}
    assert {"summary_all.csv", "summary_all.md", "B0_summary.csv", "B0_summary.md", "E3_summary.csv", "E3_summary.md"} <= names

    df = pd.read_csv(tmp_path / "tables" / "B0_summary.csv")
    assert len(df) == 1
    assert df.loc[0, "val_acc_mean"] == pytest.approx(0.655, abs=1e-6)
    assert df.loc[0, "val_acc_std"] == pytest.approx(0.005, abs=1e-6)
    assert df.loc[0, "val_acc_fmt"] == "0.6550 ± 0.0050"

    bm, bs = baseline_stats(data)
    assert bm == pytest.approx(0.655) and bs == pytest.approx(0.005)
    assert "< std baseline" in compare_to_baseline(0.657, bm, bs)
    assert "cao hơn" in compare_to_baseline(0.680, bm, bs)

    rows_sum = summarize_rows(data, bm, bs)
    failed_row = [r for r in rows_sum if r["variant"] == "bad_lr"][0]
    assert failed_row["n_runs"] == 1 and failed_row["n_completed"] == 0 and failed_row["val_acc_fmt"] == ""


def test_update_experiment_log(tmp_path):
    exp_csv = tmp_path / "experiments.csv"
    r = _row("E3-001_s0", "T14", "adamw_1e-3", 0, 0.6812, 0.635, 0.912)
    r.update({"best_epoch": "42", "tong_thoi_gian_s": "510.2", "run_dir": "results/E3/E3-001_s0"})
    with open(exp_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REAL_HEADER)
        w.writeheader()
        w.writerow(r)

    log_md = tmp_path / "NHAT_KY.md"
    log_md.write_text("### [E3-001_s0] AdamW — TV3\n- Giả thuyết: thử AdamW\n- Kết quả: (chờ điền)\n", encoding="utf-8")
    n = update_experiment_log(log_md, exp_csv)
    text = log_md.read_text(encoding="utf-8")
    assert n == 1
    assert "val_acc=0.6812" in text and "results/E3/E3-001_s0" in text


def test_baseline_stats_va_summarize_tach_probe_va_cau_hinh_cu(tmp_path):
    cfg_path = tmp_path / "baseline.json"
    base_cfg = {"model": "baseline", "epochs": 100, "lr": 0.01}
    cfg_path.write_text(json.dumps({"group": "baseline", "task_id": "T11", "base": base_cfg}), encoding="utf-8")

    def _full_row(exp_id, seed, val_acc, epochs=100, variant="baseline"):
        r = _row(exp_id, "T11", variant, seed, val_acc)
        r["hyperparams_json"] = json.dumps({**base_cfg, "epochs": epochs, "seed": seed, "variant": variant})
        return r

    data = [
        _full_row("baseline-001_s1", 1, 0.65645, epochs=60),
        _full_row("_probe-001_s0", 0, 0.68153, epochs=100, variant="baseline_ep100"),
        _full_row("baseline-002_s0", 0, 0.6815268877124547, epochs=100),
        _full_row("baseline-002_s1", 1, 0.6740039008080245, epochs=100),
        _full_row("baseline-002_s2", 2, 0.6890498746168849, epochs=100),
    ]
    bm, bs = baseline_stats(data, baseline_config=cfg_path)
    assert bm == pytest.approx(0.6815268877124547, abs=1e-9)
    assert bs == pytest.approx(0.007522986904430207, abs=1e-9)

    rows_sum = summarize_rows(data, bm, bs)
    by_var = {r["variant"]: r for r in rows_sum}
    assert "baseline [baseline-001]" in by_var
    assert "baseline [baseline-002]" in by_var
    assert by_var["baseline [baseline-002]"]["n_runs"] == 3
    assert by_var["baseline [baseline-002]"]["val_acc_fmt"] == "0.6815 ± 0.0075"
    assert by_var["baseline [baseline-001]"]["n_runs"] == 1
