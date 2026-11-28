"""Test luật chọn run và công thức của src/baseline_summary.py (T11). Dữ liệu giả lập chỉ nằm trong test.

Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T11.
"""
import json
import statistics

import pytest

from src.baseline_summary import config_sha256_lf, select_runs, summarize

BASE = {"model": "baseline", "epochs": 100, "lr": 0.01}


def _row(exp_id, seed, val_acc, epochs=100, status="Completed", task="T11"):
    hp = dict(BASE, epochs=epochs, seed=seed, exp_id=exp_id, variant="baseline")
    return {"exp_id": exp_id, "task_id": task, "trang_thai_ky_thuat": status, "seed": str(seed),
            "hyperparams_json": json.dumps(hp), "thanh_vien": "TV0", "may_chay": "m", "git_commit": "c",
            "best_epoch": "90", "val_acc": str(val_acc), "val_macro_f1": "0.6", "best_val_loss": "0.9"}


def test_chi_lay_bo_khop_config():
    rows = [
        _row("baseline-001_s0", 0, 0.66, epochs=60),
        _row("_probe-001_s0", 0, 0.68),
        _row("baseline-002_s2", 2, 0.69),
        _row("baseline-002_s0", 0, 0.68),
        _row("baseline-002_s1", 1, 0.67),
        _row("baseline-003_s0", 0, 0.10, status="Failed"),
    ]
    runs = select_runs(rows, BASE)
    assert [r["exp_id"] for r in runs] == ["baseline-002_s0", "baseline-002_s1", "baseline-002_s2"]


def test_nhieu_bo_cung_khop_thi_bao_loi():
    rows = [_row("baseline-002_s0", 0, 0.68), _row("baseline-003_s0", 0, 0.68)]
    with pytest.raises(SystemExit):
        select_runs(rows, BASE)


def test_mean_std_mau():
    runs = [_row(f"baseline-002_s{s}", s, v) for s, v in enumerate([0.68, 0.67, 0.69])]
    out = summarize(runs, "configs/baseline.json", "x")
    assert out["val_acc"]["mean"] == pytest.approx(0.68)
    assert out["val_acc"]["std"] == pytest.approx(statistics.stdev([0.68, 0.67, 0.69]))
    assert out["dung_sai_tai_lap_val_acc"] == out["val_acc"]["std"]


def test_sha256_khong_phu_thuoc_xuong_dong(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(b'{\n  "x": 1\n}\n')
    b.write_bytes(b'{\r\n  "x": 1\r\n}\r\n')
    assert config_sha256_lf(a) == config_sha256_lf(b)
