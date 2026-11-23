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
    assert {p.name for p in d.iterdir()} == {"config.json", "run.log", "history.csv", "metrics.json", "best.pt"}
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


def test_config_sai_ghi_run_failed_va_chay_tiep(root):
    cfg = write_config(root, {"hong": {"model": "khong_co"}, "tot": {}})
    res = run_experiment.run(cfg, "TV1", root=root)
    assert [x["status"] for x in res] == ["Failed", "Completed"]
    bad = root / "results" / "E9" / "E9-001_s0"
    assert not list(bad.glob("*.pt*")) and "Traceback" in (bad / "run.log").read_text(encoding="utf-8")
    r = rows(root)
    assert r[0]["trang_thai_ky_thuat"] == "Failed" and r[0]["weights_ma_ly_do"] == "EXCEPTION:ValueError" and r[0]["weights_path"] == ""
    assert r[1]["trang_thai_ky_thuat"] == "Completed"


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
