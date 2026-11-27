"""Unit test cho cờ --eval-test và cơ chế khóa TEST_DONE.json (nhiệm vụ T19).

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T19.

Lưu ý: Toàn bộ test dùng dữ liệu tổng hợp trong tmp_path, tuyệt đối KHÔNG đọc fer2013.csv
và KHÔNG chạm tới results/E7/TEST_DONE.json thật.
"""
import csv
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src import run_experiment
from src.models import build_model

REAL_HEADER = (Path(__file__).resolve().parents[1] / "tracking" / "experiments.csv").read_text(encoding="utf-8").splitlines()[0]
SMALL = {"blocks": [[4], [8]], "fc_units": 16}


def _make_fake_run(root: Path, exp_id: str, seed: int) -> Path:
    rd = root / "results" / "E7" / exp_id
    rd.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    cfg = {
        "model": "baseline",
        "model_kwargs": SMALL,
        "seed": seed,
        "amp": False,
        "device": "cpu",
        "resize": 48,
        "channels": 1,
    }
    model = build_model("baseline", **SMALL)
    torch.save({"model": model.state_dict()}, rd / "best.pt")
    meta = {
        "exp_id": exp_id,
        "task_id": "T18",
        "group": "E7",
        "variant": "e7_final",
        "member": "TV3",
        "cfg": cfg,
    }
    (rd / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return rd


@pytest.fixture
def env_root(tmp_path, monkeypatch):
    (tmp_path / "tracking").mkdir()
    (tmp_path / "tracking" / "members.csv").write_text("ma,ho_ten\nTV1,A\nTV2,B\nTV3,C\nTV4,D\n", encoding="utf-8")
    (tmp_path / "tracking" / "experiments.csv").write_text(REAL_HEADER + "\n", encoding="utf-8")

    g = torch.Generator().manual_seed(2026)
    x = torch.randn(35, 1, 48, 48, generator=g)
    y = torch.arange(35, dtype=torch.int64) % 7

    def fake_make_test_loader(cfg):
        return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)

    mod = types.ModuleType("src.data.loader")
    mod.make_test_loader = fake_make_test_loader
    monkeypatch.setitem(sys.modules, "src.data.loader", mod)
    return tmp_path


def test_eval_test_chay_mot_lan_va_khoa_lan_hai(env_root):
    r0 = _make_fake_run(env_root, "E7-001_s0", seed=0)
    r1 = _make_fake_run(env_root, "E7-001_s1", seed=1)
    r2 = _make_fake_run(env_root, "E7-001_s2", seed=2)
    lock_path = env_root / "results" / "E7" / "TEST_DONE.json"

    res = run_experiment.eval_test(
        run_dirs=[r0, r1, r2],
        member="TV3",
        tta="flip",
        lock_file=lock_path,
        root=env_root,
        device="cpu",
    )

    assert lock_path.is_file()
    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock_data["locked"] is True
    assert lock_data["task_id"] == "T19"
    assert lock_data["split"] == "PrivateTest"
    assert lock_data["n_models"] == 3
    assert lock_data["n_samples"] == 35
    assert lock_data["tta"] == "flip"
    assert len(lock_data["models"]) == 3

    # Đối chiếu độc lập với sklearn trên file dự đoán đã lưu
    npz = np.load(env_root / lock_data["predictions_path"])
    yt, yp, yprob = npz["y_true"], npz["y_pred"], npz["y_prob"]
    assert yprob.shape == (35, 7)
    assert np.isclose(res["test_acc"], accuracy_score(yt, yp), atol=1e-12)
    assert np.isclose(res["test_macro_f1"], f1_score(yt, yp, average="macro", zero_division=0), atol=1e-12)

    # Kiểm tra log, bảng per-class, 4 hình kèm CSV và dòng experiments.csv
    assert (env_root / lock_data["log_path"]).is_file()
    assert (env_root / lock_data["per_class_csv"]).is_file()
    for fig_rel in lock_data["figures"].values():
        p = env_root / fig_rel
        assert p.is_file() and p.with_suffix(".csv").is_file()
    with open(env_root / "tracking" / "figures_manifest.csv", encoding="utf-8", newline="") as f:
        figs = {r["fig_id"]: r for r in csv.DictReader(f)}
    assert {"e7_test_confusion", "e7_test_error_hist", "e7_test_correct_grid", "e7_test_wrong_grid"} <= set(figs)
    with open(env_root / "tracking" / "experiments.csv", encoding="utf-8", newline="") as f:
        exp_rows = list(csv.DictReader(f))
    assert len(exp_rows) == 1
    assert exp_rows[0]["exp_id"] == "E7-TEST" and exp_rows[0]["task_id"] == "T19"
    assert np.isclose(float(exp_rows[0]["test_acc"]), res["test_acc"], atol=1e-12)

    # Lần chạy thứ hai BẮT BUỘC bị từ chối và không ghi đè TEST_DONE.json
    lock_before = lock_path.read_text(encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        run_experiment.eval_test(
            run_dirs=[r0],
            member="TV3",
            tta="none",
            lock_file=lock_path,
            root=env_root,
            device="cpu",
        )
    assert "TỪ CHỐI CHẠY --eval-test" in str(exc.value)
    assert lock_path.read_text(encoding="utf-8") == lock_before


def test_eval_test_tu_choi_member_sai_va_thieu_run_dirs(env_root):
    r0 = _make_fake_run(env_root, "E7-001_s0", seed=0)
    with pytest.raises(SystemExit):
        run_experiment.eval_test(run_dirs=[r0], member="TV9", root=env_root, device="cpu")
    with pytest.raises(SystemExit):
        run_experiment.eval_test(run_dirs=[], member="TV3", root=env_root, device="cpu")


def test_eval_test_tu_choi_checkpoint_khac_kich_thuoc(env_root):
    r0 = _make_fake_run(env_root, "E7-001_s0", seed=0)
    r1 = _make_fake_run(env_root, "E7-001_s1", seed=1)
    p1 = r1 / "config.json"
    meta1 = json.loads(p1.read_text(encoding="utf-8"))
    meta1["cfg"]["resize"] = 96
    p1.write_text(json.dumps(meta1), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_experiment.eval_test(run_dirs=[r0, r1], member="TV3", root=env_root, device="cpu")
