"""Test cho src/engine/trainer.py. Phụ trách: TV2 – Lê Phan Nguyên Hải. Nhiệm vụ T07.

Chạy từ thư mục BTL02: python -m pytest tests/test_trainer.py
Dữ liệu ngẫu nhiên chỉ dùng trong test này (tập tách được tuyến tính), không bao giờ ra kết quả hay hình báo cáo.
"""
import csv
import json
import os
import time

import pytest
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine import trainer
from src.engine.trainer import fit, macro_f1, scan_orphans
from src.utils.io import sha256_file


def make_data(n=140, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 1, 4, 4, generator=g)
    w = torch.randn(16, 7, generator=g)
    return TensorDataset(x, (x.flatten(1) @ w).argmax(1))


def loaders():
    return (DataLoader(make_data(140, 0), batch_size=35), DataLoader(make_data(70, 1), batch_size=35))


def model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Flatten(), nn.Linear(16, 7))


CFG = dict(epochs=3, optimizer="sgd", lr=0.1, scheduler="constant", amp=False, device="cpu", early_stopping_patience=0)


class Flaky:
    """Bọc loader: lần duyệt thứ `fail_on` (= epoch thứ fail_on) gây NaN hoặc ném lỗi."""
    def __init__(self, base, fail_on, how):
        self.base, self.fail_on, self.how, self.n = base, fail_on, how, 0

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        self.n += 1
        if self.n == self.fail_on:
            if self.how == "nan":
                for x, y in self.base:
                    yield x * float("nan"), y
                return
            raise self.how()
        yield from self.base


def pt_files(d):
    return list(d.glob("*.pt*"))


def read_history(d):
    with open(d / "history.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_metrics(d):
    return json.loads((d / "metrics.json").read_text(encoding="utf-8"))


def test_loss_giam_va_run_hop_le_giu_weights(tmp_path):
    tl, vl = loaders()
    r = fit(model(), tl, vl, CFG, tmp_path)
    hist = read_history(tmp_path)
    assert [float(h["train_loss"]) for h in hist][-1] < float(hist[0]["train_loss"])  # 1 bước train giảm loss
    assert r["status"] == "Completed" and r["weights_quyet_dinh"] == "KEEP" and r["weights_ma_ly_do"] == "OK"
    assert (tmp_path / "best.pt").exists() and r["weights_sha256"] == sha256_file(tmp_path / "best.pt")
    assert not list(tmp_path.glob("*.tmp"))
    log = (tmp_path / "run.log").read_text(encoding="utf-8").splitlines()
    assert len([l for l in log if "/3 |" in l]) == len(hist) == 3  # số dòng epoch trong run.log = history.csv
    assert " | START | " in log[0] and " | END | " in log[-1]
    assert r["best_val_acc"] == max(float(h["val_acc"]) for h in hist) == read_metrics(tmp_path)["best_val_acc"]
    assert torch.load(tmp_path / "best.pt", weights_only=True).keys() == model().state_dict().keys()


def test_macro_f1_khop_sklearn():
    g = torch.Generator().manual_seed(1)
    y, p = torch.randint(0, 7, (500,), generator=g), torch.randint(0, 7, (500,), generator=g)
    conf = torch.bincount(y * 7 + p, minlength=49).reshape(7, 7)
    assert macro_f1(conf) == pytest.approx(f1_score(y, p, average="macro", labels=range(7), zero_division=0))


def test_early_stopping(tmp_path):
    tl, vl = loaders()
    r = fit(model(), tl, vl, {**CFG, "epochs": 20, "lr": 0.0, "early_stopping_patience": 2}, tmp_path)
    assert r["stopped_by"] == "early_stopping" and r["epochs_run"] == 3 and r["best_epoch"] == 1  # lr=0: val_acc không đổi


def test_nan_loss_xoa_weights_giu_log(tmp_path):
    tl, vl = loaders()
    r = fit(model(), Flaky(tl, 2, "nan"), vl, CFG, tmp_path)
    assert r["status"] == "Failed" and r["weights_ma_ly_do"] == "NAN_LOSS@epoch2" and r["weights_quyet_dinh"] == "DISCARD"
    assert pt_files(tmp_path) == []  # best.pt của epoch 1 cũng bị xóa
    assert (tmp_path / "run.log").exists() and len(read_history(tmp_path)) == 1  # giữ log và history


def test_exception_ghi_traceback(tmp_path):
    tl, vl = loaders()
    r = fit(model(), Flaky(tl, 2, RuntimeError), vl, CFG, tmp_path)
    assert r["status"] == "Failed" and r["weights_ma_ly_do"] == "EXCEPTION:RuntimeError" and pt_files(tmp_path) == []
    log = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert " | ERROR | " in log and "Traceback" in log


def test_ctrl_c_ghi_aborted_va_khong_nem(tmp_path):
    tl, vl = loaders()
    r = fit(model(), Flaky(tl, 2, KeyboardInterrupt), vl, CFG, tmp_path)
    assert r["status"] == "Aborted" and r["weights_ma_ly_do"] == "INTERRUPTED@epoch1" and pt_files(tmp_path) == []
    assert " | END | " in (tmp_path / "run.log").read_text(encoding="utf-8")


def test_reload_mismatch(tmp_path, monkeypatch):
    tl, vl = loaders()
    monkeypatch.setattr(trainer, "_reload_val_acc", lambda *a: 0.0)
    r = fit(model(), tl, vl, CFG, tmp_path)
    assert r["weights_ma_ly_do"] == "RELOAD_MISMATCH" and r["weights_quyet_dinh"] == "DISCARD" and pt_files(tmp_path) == []
    assert r["status"] == "Completed"  # chạy xong về kỹ thuật nhưng weights không đáng tin


def test_no_checkpoint(tmp_path):
    tl, vl = loaders()
    r = fit(model(), tl, vl, {**CFG, "epochs": 0}, tmp_path)
    assert r["weights_ma_ly_do"] == "NO_CHECKPOINT" and r["best_epoch"] == 0


def test_val_acc_thap_van_keep(tmp_path):
    tl, vl = loaders()
    r = fit(model(), tl, vl, {**CFG, "epochs": 1, "lr": 0.0}, tmp_path)  # không học gì, val_acc thấp
    assert r["weights_quyet_dinh"] == "KEEP"  # luật không xét metric


def test_khong_ghi_de_run_cu(tmp_path):
    tl, vl = loaders()
    fit(model(), tl, vl, CFG, tmp_path)
    with pytest.raises(FileExistsError):
        fit(model(), tl, vl, CFG, tmp_path)


def test_cung_seed_cung_history(tmp_path):
    def once(d):
        torch.manual_seed(0)
        tl, vl = loaders()
        fit(model(), tl, vl, CFG, d)
        return [(h["train_loss"], h["val_loss"], h["val_acc"]) for h in read_history(d)]
    assert once(tmp_path / "a") == once(tmp_path / "b")


def test_scan_orphans(tmp_path):
    d = tmp_path / "E1" / "E1-001_s0"
    d.mkdir(parents=True)
    (d / "run.log").write_text("t | E1-001_s0 | START | x\nt | E1-001_s0 | 1/5 | ok\n", encoding="utf-8")
    (d / "history.csv").write_text("epoch,train_loss,val_loss,train_acc,val_acc,val_macro_f1,lr,epoch_time_s\n"
                                   "1,1,1,0.1,0.2,0.1,0.01,1.0\n2,1,1,0.1,0.3,0.1,0.01,1.0\n", encoding="utf-8")
    (d / "best.pt").write_bytes(b"do")
    (d / "best.pt.tmp").write_bytes(b"do")
    old = time.time() - 7200
    os.utime(d / "run.log", (old, old))
    ok = tmp_path / "E1" / "E1-002_s0"  # run đang chạy (log mới): không đụng
    ok.mkdir()
    (ok / "run.log").write_text("t | E1-002_s0 | START | x\n", encoding="utf-8")
    (ok / "best.pt").write_bytes(b"x")
    assert scan_orphans(tmp_path) == [d]
    assert pt_files(d) == [] and (ok / "best.pt").exists()
    m = read_metrics(d)
    assert m["status"] == "Aborted" and m["weights_ma_ly_do"] == "INTERRUPTED@epoch2" and m["best_epoch"] == 2
    assert scan_orphans(tmp_path) == []  # lần quét sau không xử lý lại


def test_grad_norms_tung_lop(tmp_path):
    """T13: log_grad_norms ghi chuẩn gradient từng lớp có trọng số mỗi epoch, khớp chuẩn tính tay ở epoch 1."""
    tl, vl = loaders()
    net = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 7))
    torch.manual_seed(0)
    ref = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 7))
    ref.load_state_dict(net.state_dict())
    fit(net, tl, vl, {**CFG, "log_grad_norms": True}, tmp_path)
    with open(tmp_path / "grad_norms.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["epoch"], r["layer"]) for r in rows] == [(str(e), n) for e in (1, 2, 3) for n in ("1", "4")]
    assert all(float(r["grad_norm_mean"]) > 0 and r["n_steps"] == str(len(tl)) for r in rows)
    # tính tay epoch 1: cùng trọng số ban đầu, cùng thứ tự batch, SGD giống fit
    opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    ref.train()
    sums = {"1": 0.0, "4": 0.0}
    for x, y in tl:
        opt.zero_grad()
        nn.CrossEntropyLoss()(ref(x), y).backward()
        sums["1"] += ref[1].weight.grad.norm().item()
        sums["4"] += ref[4].weight.grad.norm().item()
        opt.step()
    for r in rows[:2]:
        assert float(r["grad_norm_mean"]) == pytest.approx(sums[r["layer"]] / len(tl), rel=1e-5)


def test_grad_norms_khong_doi_ket_qua_train(tmp_path):
    """Bật log_grad_norms không đổi bước cập nhật: history giống run không bật; mặc định không ghi file."""
    tl, vl = loaders()
    fit(model(), tl, vl, CFG, tmp_path / "a")
    fit(model(), tl, vl, {**CFG, "log_grad_norms": True}, tmp_path / "b")
    assert not (tmp_path / "a" / "grad_norms.csv").exists()
    cols = ["train_loss", "val_loss", "train_acc", "val_acc"]
    assert [[h[c] for c in cols] for h in read_history(tmp_path / "a")] == \
           [[h[c] for c in cols] for h in read_history(tmp_path / "b")]
