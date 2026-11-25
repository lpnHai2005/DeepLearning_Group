"""Unit test cho src/make_tables.py.

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T09.
Chạy từ thư mục BTL02: python -m pytest tests/test_make_tables.py -v
"""
import csv
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.make_tables import (
    baseline_stats,
    compare_to_baseline,
    export_tables,
    filter_report_rows,
    summarize_rows,
    update_experiment_log,
)

REAL_HEADER = (Path(__file__).resolve().parents[1] / "tracking" / "experiments.csv").read_text(encoding="utf-8").splitlines()[0].split(",")


def _row(
    exp_id,
    task_id,
    variant,
    seed,
    val_acc,
    f1=0.6,
    loss=0.9,
    status="Completed",
    train_acc=0.72,
    n_params=5867847,
    may_chay="Hbao | NVIDIA GeForce RTX 3050 Laptop GPU",
    config_path="",
):
    r = {c: "" for c in REAL_HEADER}
    r.update(
        {
            "exp_id": exp_id,
            "task_id": task_id,
            "thanh_vien": "TV3",
            "epoch_time_s_tb": "12.5",
            "may_chay": may_chay,
            "config_path": config_path,
            "hyperparams_json": json.dumps({"variant": variant, "seed": seed}),
            "seed": str(seed),
            "n_params": "" if n_params is None else str(n_params),
            "train_acc_cuoi": "" if train_acc is None else str(train_acc),
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
        _row("E3-002_s0", "T14", "bad_lr", 0, None, None, None, status="Failed", train_acc=None),
    ]
    with open(exp_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REAL_HEADER)
        w.writeheader()
        w.writerows(data)

    written = export_tables(exp_csv, tmp_path / "tables", root=tmp_path)
    names = {p.name for p in written}
    assert {
        "summary_all.csv",
        "summary_all.md",
        "summary_all_report.md",
        "B0_summary.csv",
        "B0_summary.md",
        "B0_report.md",
        "E3_summary.csv",
        "E3_summary.md",
        "E3_report.md",
        "BTL02_tables_report.docx",
    } <= names

    df = pd.read_csv(tmp_path / "tables" / "B0_summary.csv")
    assert len(df) == 1
    assert df.loc[0, "val_acc_mean"] == pytest.approx(0.655, abs=1e-6)
    assert df.loc[0, "val_acc_std"] == pytest.approx(0.005, abs=1e-6)
    assert df.loc[0, "val_acc_fmt"] == "0.6550 ± 0.0050"
    assert df.loc[0, "so_voi_baseline_tu_tinh"] == "tham chiếu"

    bm, bs = baseline_stats(data)
    assert bm == pytest.approx(0.655) and bs == pytest.approx(0.005)
    assert "< std baseline" in compare_to_baseline(0.657, bm, bs)
    assert "cao hơn" in compare_to_baseline(0.680, bm, bs)
    assert compare_to_baseline(0.655, bm, bs, is_baseline=True) == "tham chiếu"

    rows_sum = summarize_rows(data, bm, bs, root=tmp_path)
    failed_row = [r for r in rows_sum if r["variant"] == "bad_lr"][0]
    assert failed_row["n_runs"] == 1 and failed_row["n_completed"] == 0 and failed_row["val_acc_fmt"] == ""

    # Kiểm tra bảng E3 báo cáo có dòng Baseline (tham chiếu) ở đầu và có file Word hợp lệ
    e3_rep = (tmp_path / "tables" / "E3_report.md").read_text(encoding="utf-8")
    assert "Bảng 4.3:" in e3_rep
    assert "| Baseline (tham chiếu) |" in e3_rep
    assert "| adamw_1e-3 |" in e3_rep
    with zipfile.ZipFile(tmp_path / "tables" / "BTL02_tables_report.docx") as zf:
        doc_xml = zf.read("word/document.xml").decode("utf-8")
        assert "Bảng 4.3:" in doc_xml and "Baseline (tham chiếu)" in doc_xml


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


def test_gop_y_c2_loc_nhom_tam_thu_tu_config_gpu_va_word(tmp_path):
    (tmp_path / "configs").mkdir()
    base_cfg = {"model": "baseline", "epochs": 100, "lr": 0.01}
    cfg_base = tmp_path / "configs" / "baseline.json"
    cfg_base.write_text(json.dumps({"group": "baseline", "task_id": "T11", "base": base_cfg}), encoding="utf-8")

    cfg_e1 = tmp_path / "configs" / "E1_arch.json"
    cfg_e1.write_text(
        json.dumps(
            {
                "group": "E1",
                "task_id": "T12",
                "base": base_cfg,
                "variants": {"small": {}, "medium": {}, "baseline_gap": {}},
            }
        ),
        encoding="utf-8",
    )

    def _full_row(exp_id, task_id, seed, val_acc, epochs=100, variant="baseline", may_chay="M1 | NVIDIA GeForce RTX 4060 Laptop GPU", cfg_p="configs/baseline.json"):
        r = _row(exp_id, task_id, variant, seed, val_acc, may_chay=may_chay, config_path=cfg_p)
        r["hyperparams_json"] = json.dumps({**base_cfg, "epochs": epochs, "seed": seed, "variant": variant})
        return r

    data = [
        _full_row("_qa-001_s0", "T07", 0, 0.38, epochs=2, variant="qa"),
        _full_row("_probe-001_s0", "T11", 0, 0.68153, epochs=100, variant="baseline_ep100"),
        _full_row("_nb-001_s0", "T10", 0, 0.387, epochs=2, variant="repro_2ep"),
        _full_row("baseline-001_s1", "T11", 1, 0.65645, epochs=60),
        _full_row("baseline-002_s0", "T11", 0, 0.6815268877124547, may_chay="A | NVIDIA GeForce RTX 4060 Laptop GPU"),
        _full_row("baseline-002_s1", "T11", 1, 0.6740039008080245, may_chay="B | NVIDIA GeForce RTX 3050 Laptop GPU"),
        _full_row("baseline-002_s2", "T11", 2, 0.6890498746168849, may_chay="C | NVIDIA GeForce RTX 3060 Laptop GPU"),
        # Cố tình đảo thứ tự E1 trong experiments.csv để kiểm tra sort theo E1_arch.json
        _full_row("E1-003_s0", "T12", 0, 0.6829, variant="baseline_gap", cfg_p="configs/E1_arch.json"),
        _full_row("E1-001_s0", "T12", 0, 0.6915, variant="small", cfg_p="configs/E1_arch.json"),
        _full_row("E1-002_s0", "T12", 0, 0.6951, variant="medium", cfg_p="configs/E1_arch.json"),
    ]
    exp_csv = tmp_path / "experiments.csv"
    with open(exp_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REAL_HEADER)
        w.writeheader()
        w.writerows(data)

    filtered = filter_report_rows(data, baseline_config=cfg_base)
    f_ids = {r["exp_id"] for r in filtered}
    assert "_qa-001_s0" not in f_ids and "_probe-001_s0" not in f_ids and "_nb-001_s0" not in f_ids
    assert "baseline-001_s1" not in f_ids
    assert {"baseline-002_s0", "baseline-002_s1", "baseline-002_s2"} <= f_ids

    bm, bs = baseline_stats(data, baseline_config=cfg_base)
    assert bm == pytest.approx(0.6815268877124547, abs=1e-9)
    assert bs == pytest.approx(0.007522986904430207, abs=1e-9)

    written = export_tables(exp_csv, tmp_path / "tables", baseline_config=cfg_base, root=tmp_path)
    names = {p.name for p in written}
    assert "_qa_summary.csv" not in names and "_probe_summary.csv" not in names and "_nb_summary.csv" not in names

    # Nhiều GPU -> không gộp trung bình epoch_time_s
    base_rep = (tmp_path / "tables" / "baseline_report.md").read_text(encoding="utf-8")
    assert "— (nhiều GPU)" in base_rep
    assert "| tham chiếu |" in base_rep

    # E1 xếp đúng thứ tự trong config: Baseline -> small -> medium -> baseline_gap
    e1_rep = (tmp_path / "tables" / "E1_report.md").read_text(encoding="utf-8")
    idx_base = e1_rep.index("| Baseline (tham chiếu) |")
    idx_small = e1_rep.index("| small |")
    idx_med = e1_rep.index("| medium |")
    idx_gap = e1_rep.index("| baseline_gap |")
    assert idx_base < idx_small < idx_med < idx_gap
    assert "train_acc_cuoi" in e1_rep  # có ghi chú aug + dropout
