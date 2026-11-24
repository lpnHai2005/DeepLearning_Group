"""Test cho src/run_experiment.py. Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T07.

Chạy từ thư mục BTL02: python -m pytest tests/test_run_experiment.py
make_loaders giả (dữ liệu ngẫu nhiên) chỉ tồn tại trong test; loader thật là nhiệm vụ T04 (TV1).
"""
import csv
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src import run_experiment

REAL_HEADER = (Path(__file__).resolve().parents[1] / "tracking" / "experiments.csv").read_text(encoding="utf-8").splitlines()[0]
SMALL = {"blocks": [[4], [8]], "fc_units": 16}  # CNN rất nhỏ để test chạy nhanh trên CPU


def fake_make_loaders(cfg):
    g = torch.Generator().manual_seed(cfg["seed"])
    x = torch.randn(64, 1, 48, 48, generator=g)
    y = torch.randint(0, 7, (64,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True, generator=g), DataLoader(TensorDataset(x[:32], y[:32]), batch_size=32)


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "tracking").mkdir()
    (tmp_path / "tracking" / "members.csv").write_text("ma,ho_ten\nTV1,A\nTV2,B\n", encoding="utf-8")
    (tmp_path / "tracking" / "experiments.csv").write_text(REAL_HEADER + "\n", encoding="utf-8")
    mod = types.ModuleType("src.data.loader")
    mod.make_loaders = fake_make_loaders
    monkeypatch.setitem(sys.modules, "src.data.loader", mod)
    return tmp_path


def write_config(root, variants, seeds=(0,), **base):
    spec = {"group": "E9", "task_id": "T07", "base": {"model": "baseline", "model_kwargs": SMALL, "epochs": 2, "amp": False,
            "device": "cpu", "scheduler": "constant", "early_stopping_patience": 0, **base}, "variants": variants, "seeds": list(seeds)}
    p = root / "E9.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def rows(root):
    with open(root / "tracking" / "experiments.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_run_hop_le_ghi_du_file_va_mot_dong(root):
    cfg = write_config(root, {"sgd": {}, "adamw": {"optimizer": "adamw", "lr": 0.001}}, seeds=(0, 1))
    res = run_experiment.run(cfg, "TV2", root=root)
    assert len(res) == 4 and all(r["status"] == "Completed" for r in res)
    r = rows(root)
    assert [x["exp_id"] for x in r] == ["E9-001_s0", "E9-001_s1", "E9-002_s0", "E9-002_s1"]  # seed chung số thứ tự
    d = root / "results" / "E9" / "E9-001_s0"
    assert {p.name for p in d.iterdir()} == {"config.json", "run.log", "history.csv", "metrics.json", "best.pt",
                                         "curves.png", "curves.csv"}
    meta = json.loads((d / "config.json").read_text(encoding="utf-8"))
    for k in ("member", "hostname", "os_user", "gpu", "started_at", "finished_at", "duration_s", "torch", "n_params"):
        assert meta[k] not in (None, ""), k
    assert {"git_user", "git_commit", "dataset_sha256"} <= set(meta)  # rỗng nếu thư mục test không phải repo git/không có dữ liệu
    assert meta["member"] == "TV2" and meta["finished_at"] >= meta["started_at"]
    row = r[0]
    assert row["thanh_vien"] == "TV2" and row["trang_thai_ky_thuat"] == "Completed" and row["weights_quyet_dinh"] == "KEEP"
    assert row["weights_path"] == "results/E9/E9-001_s0/best.pt" and row["run_dir"] == "results/E9/E9-001_s0"
    assert json.loads(row["hyperparams_json"])["variant"] == "sgd"
    assert int(row["n_params"]) == meta["n_params"] and float(row["epoch_time_s_tb"]) > 0 and row["test_acc"] == ""
    assert set(r[0]) == set(REAL_HEADER.split(","))
    assert row["curve_path"] == "results/E9/E9-001_s0/curves.png"
    hist = (d / "history.csv").read_text(encoding="utf-8").splitlines()
    with open(d / "curves.csv", encoding="utf-8", newline="") as f:
        plotted = list(csv.DictReader(f))
    assert len(plotted) == len(hist) - 1 == 2  # mỗi epoch của history.csv đúng 1 dòng dữ liệu vẽ
    with open(root / "tracking" / "figures_manifest.csv", encoding="utf-8", newline="") as f:
        figs = {x["fig_id"]: x for x in csv.DictReader(f)}
    assert len(figs) == 4  # 4 run, mỗi run 1 hình
    fig = figs["curves_E9-001_s0"]
    assert fig["file_hinh"] == row["curve_path"] and fig["exp_ids"] == "E9-001_s0" and fig["task_id"] == "T07"
    assert fig["file_du_lieu_dau_vao"] == "results/E9/E9-001_s0/history.csv" and len(fig["sha256_hinh"]) == 64


def test_config_sai_ghi_run_failed_va_chay_tiep(root):
    cfg = write_config(root, {"hong": {"model": "khong_co"}, "tot": {}})
    res = run_experiment.run(cfg, "TV1", root=root)
    assert [x["status"] for x in res] == ["Failed", "Completed"]
    bad = root / "results" / "E9" / "E9-001_s0"
    assert not list(bad.glob("*.pt*")) and "Traceback" in (bad / "run.log").read_text(encoding="utf-8")
    r = rows(root)
    assert r[0]["trang_thai_ky_thuat"] == "Failed" and r[0]["weights_ma_ly_do"] == "EXCEPTION:ValueError" and r[0]["weights_path"] == ""
    assert r[1]["trang_thai_ky_thuat"] == "Completed"
    assert r[0]["curve_path"] == "" and not (bad / "curves.png").exists()  # run lỗi trước epoch 1: không có hình
    assert r[1]["curve_path"].endswith("curves.png")


def test_chay_lai_dung_exp_id_moi(root):
    cfg = write_config(root, {"a": {}})
    run_experiment.run(cfg, "TV1", root=root)
    run_experiment.run(cfg, "TV1", root=root)
    assert [x["exp_id"] for x in rows(root)] == ["E9-001_s0", "E9-002_s0"]


def test_cung_seed_hai_lan_cung_history(root):
    cfg = write_config(root, {"a": {}, "b": {}})
    run_experiment.run(cfg, "TV1", root=root)
    h = [(root / "results" / "E9" / f"E9-00{i}_s0" / "history.csv").read_text(encoding="utf-8") for i in (1, 2)]
    strip = lambda t: [",".join(l.split(",")[:-1]) for l in t.splitlines()]  # bỏ cột epoch_time_s
    assert strip(h[0]) == strip(h[1])


def test_member_khong_hop_le_va_only_sai(root):
    cfg = write_config(root, {"a": {}})
    with pytest.raises(SystemExit):
        run_experiment.run(cfg, "TV9", root=root)
    with pytest.raises(SystemExit):
        run_experiment.run(cfg, "TV1", only=["khong_co"], root=root)
    assert rows(root) == []


def test_only_chi_chay_variant_duoc_chon(root):
    cfg = write_config(root, {"a": {}, "b": {}})
    run_experiment.run(cfg, "TV1", only=["b"], root=root)
    assert [json.loads(x["hyperparams_json"])["variant"] for x in rows(root)] == ["b"]


def test_chia_seed_nhieu_may_dung_chung_so_thu_tu(root):
    cfg = write_config(root, {"base": {}}, seeds=(0, 1, 2))
    run_experiment.run(cfg, "TV2", root=root, seeds=[0])  # máy TV2
    run_experiment.run(cfg, "TV1", root=root, seeds=[1])  # máy TV1 (đã pull experiments.csv)
    run_experiment.run(cfg, "TV1", root=root, seeds=[2])
    assert [x["exp_id"] for x in rows(root)] == ["E9-001_s0", "E9-001_s1", "E9-001_s2"]
    run_experiment.run(cfg, "TV2", root=root, seeds=[0])  # chạy lại seed đã có: số mới, không ghi đè
    assert rows(root)[-1]["exp_id"] == "E9-002_s0"


def test_khong_dung_lai_so_cua_cfg_khac_hoac_run_loi(root):
    a = write_config(root, {"x": {}})
    run_experiment.run(a, "TV2", root=root, seeds=[0])
    b = write_config(root, {"x": {"lr": 0.001}})  # cùng tên variant nhưng cfg khác
    run_experiment.run(b, "TV1", root=root, seeds=[1])
    bad = write_config(root, {"y": {"model": "khong_co"}})
    run_experiment.run(bad, "TV1", root=root, seeds=[0])
    run_experiment.run(bad, "TV1", root=root, seeds=[1])  # số có run Failed thì không gom seed vào đó
    assert [x["exp_id"] for x in rows(root)] == ["E9-001_s0", "E9-002_s1", "E9-003_s0", "E9-004_s1"]
