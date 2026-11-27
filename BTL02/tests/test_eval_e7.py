"""Unit test cho src/eval_e7.py (tổng hợp 3 seed, TTA, Ensemble trên PublicTest — nhiệm vụ T18).

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T18.

Lưu ý: Dùng hoàn toàn dữ liệu tổng hợp trong tmp_path, tuyệt đối KHÔNG đọc fer2013.csv
và đảm bảo src/eval_e7.py KHÔNG bao giờ gọi make_test_loader / PrivateTest.
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

from src import eval_e7
from src.engine.evaluate import predict
from src.models import build_model

SMALL = {"blocks": [[4], [8]], "fc_units": 16}


def _make_run(root: Path, exp_id: str, seed: int):
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
    meta = {"exp_id": exp_id, "task_id": "T18", "group": "E7", "variant": "e7_final", "member": "TV3", "cfg": cfg}
    (rd / "config.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return rd, model


@pytest.fixture
def fake_val_env(tmp_path, monkeypatch):
    (tmp_path / "tracking").mkdir()
    g = torch.Generator().manual_seed(77)
    x = torch.randn(42, 1, 48, 48, generator=g)
    y = torch.arange(42, dtype=torch.int64) % 7
    val_loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)

    def fake_make_loaders(cfg):
        return val_loader, val_loader

    def forbidden_test_loader(cfg):
        raise AssertionError("eval_e7.py TUYỆT ĐỐI không được gọi make_test_loader / PrivateTest!")

    mod = types.ModuleType("src.data.loader")
    mod.make_loaders = fake_make_loaders
    mod.make_test_loader = forbidden_test_loader
    monkeypatch.setitem(sys.modules, "src.data.loader", mod)
    return tmp_path, val_loader


def test_eval_e7_3_seed_tta_ensemble_va_gradcam(fake_val_env):
    root, val_loader = fake_val_env
    r0, m0 = _make_run(root, "E7-001_s0", seed=0)
    r1, m1 = _make_run(root, "E7-001_s1", seed=1)
    r2, m2 = _make_run(root, "E7-001_s2", seed=2)

    report = eval_e7.evaluate_e7(
        run_dirs=[r0, r1, r2],
        tta_modes=("none", "flip", "ten_crop"),
        root=root,
        device="cpu",
    )

    assert report["split"] == "PublicTest"
    assert report["exp_ids"] == ["E7-001_s0", "E7-001_s1", "E7-001_s2"]
    assert len(report["per_seed"]) == 9  # 3 seed x 3 TTA
    assert len(report["summary"]) == 6   # 3 TTA x (single_mean, ensemble)

    # Tự tính độc lập ensemble TTA='flip' và kiểm khớp tuyệt đối với npz đã lưu
    yt0, p0 = predict(m0, val_loader, tta="flip", device="cpu", amp=False)
    _, p1 = predict(m1, val_loader, tta="flip", device="cpu", amp=False)
    _, p2 = predict(m2, val_loader, tta="flip", device="cpu", amp=False)
    expected_prob = np.mean([p0, p1, p2], axis=0).astype(np.float32)

    npz = np.load(root / "results" / "E7" / "val_predictions_ens_flip.npz")
    assert np.array_equal(npz["y_true"], yt0)
    assert np.allclose(npz["y_prob"], expected_prob, atol=1e-6)
    ens_flip_row = [r for r in report["summary"] if r["tta"] == "flip" and r["eval_type"] == "ensemble"][0]
    assert np.isclose(ens_flip_row["val_acc_mean"], accuracy_score(yt0, expected_prob.argmax(axis=1)), atol=1e-6)
    assert np.isclose(ens_flip_row["val_macro_f1_mean"], f1_score(yt0, expected_prob.argmax(axis=1), average="macro", zero_division=0), atol=1e-6)

    # Kiểm tra mean ± std của single_mean (ddof=1)
    single_none = [r for r in report["per_seed"] if r["tta"] == "none"]
    sum_none = [r for r in report["summary"] if r["tta"] == "none" and r["eval_type"] == "single_mean"][0]
    accs = [r["val_acc"] for r in single_none]
    assert np.isclose(sum_none["val_acc_mean"], np.mean(accs), atol=1e-5)
    assert np.isclose(sum_none["val_acc_std"], np.std(accs, ddof=1), atol=1e-5)

    # Kiểm tra file bảng, Grad-CAM cases, 5 hình kèm CSV và figures_manifest.csv
    assert (root / "report" / "tables" / "E7_tta_ensemble.csv").is_file()
    assert (root / "report" / "tables" / "E7_tta_ensemble.md").is_file()
    assert (root / "results" / "E7" / "e7_val_summary.json").is_file()
    with open(root / "results" / "E7" / "e7_gradcam_cases.csv", encoding="utf-8", newline="") as f:
        gc_rows = list(csv.DictReader(f))
    assert len(gc_rows) > 0
    assert set(r["case_type"] for r in gc_rows) <= {"correct", "wrong"}

    with open(root / "tracking" / "figures_manifest.csv", encoding="utf-8", newline="") as f:
        figs = {r["fig_id"]: r for r in csv.DictReader(f)}
    expected_figs = {"e7_val_tta_ensemble", "e7_val_confusion", "e7_val_error_hist", "e7_val_correct_grid", "e7_val_wrong_grid"}
    assert expected_figs <= set(figs)
    for fid in expected_figs:
        p = root / figs[fid]["file_hinh"]
        assert p.is_file() and p.with_suffix(".csv").is_file()
