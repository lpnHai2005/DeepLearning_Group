"""T2.1–T2.2: số tham số đúng công thức, best.pt nạp lại cho đúng val loss đã log, chạy lại ra cùng số."""
import csv

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.engine.trainer import OPTS, fit, validate
from src.model.mlp import MLP, n_params
from src.utils.seed import set_seed

CFG = {"optimizer": "adam", "lr": 0.01, "max_epochs": 30, "patience": 5}


@pytest.mark.parametrize("hidden, expected", [([1], 5), ([10], 41), ([10] * 2, 151), ([10] * 4, 371)])
def test_n_params(hidden, expected):
    # [2,B,1]: 4B+1; C tầng rộng 10: 30 + 110(C−1) + 11
    assert n_params(MLP(hidden)) == expected


def _loaders(seed=0):
    g = torch.Generator().manual_seed(123)
    x = torch.rand(120, 2, generator=g) * 2 - 1
    y = torch.exp(-(x ** 2).sum(1) * 3)
    tr = DataLoader(TensorDataset(x[:100], y[:100]), batch_size=16, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    return tr, DataLoader(TensorDataset(x[100:], y[100:]), batch_size=16)


def _run(tmp_path):
    set_seed(0)
    model = MLP([4])
    tr, va = _loaders()
    return model, va, fit(model, tr, va, CFG, tmp_path)


def test_fit_checkpoint_and_history(tmp_path):
    model, va, info = _run(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "history.csv", encoding="utf-8")))
    assert len(rows) == info["epochs_run"] and {"train_loss", "val_loss", "grad_W1", "grad_W2"} <= set(rows[0])
    logged = float(rows[info["best_epoch"] - 1]["val_loss"])
    assert logged == min(float(r["val_loss"]) for r in rows)

    fresh = MLP([4])
    fresh.load_state_dict(torch.load(tmp_path / "best.pt"))
    crit = torch.nn.MSELoss()
    assert validate(fresh, va, crit) == pytest.approx(logged, rel=1e-6)
    assert validate(model, va, crit) == pytest.approx(logged, rel=1e-6)   # fit trả model ở trạng thái best


def test_reproducible(tmp_path):
    a = _run(tmp_path / "a")[2]
    b = _run(tmp_path / "b")[2]
    assert a["best_val_loss"] == pytest.approx(b["best_val_loss"], abs=1e-6) and a["best_epoch"] == b["best_epoch"]


def test_shape_guard():
    with pytest.raises(AssertionError):
        validate(MLP([2]), DataLoader(TensorDataset(torch.zeros(4, 2), torch.zeros(4, 1))), torch.nn.MSELoss())


@pytest.mark.parametrize("name", OPTS)
def test_every_optimizer_lowers_val_loss(name, tmp_path):
    set_seed(0)
    model = MLP([4])
    tr, va = _loaders()
    start = validate(model, va, torch.nn.MSELoss())
    info = fit(model, tr, va, {**CFG, "optimizer": name, "lr": 0.05}, tmp_path)
    assert info["best_val_loss"] < start


def test_nesterov_differs_from_momentum():
    p = torch.nn.Parameter(torch.ones(1))
    assert OPTS["nesterov"]([p], lr=0.1).defaults["nesterov"] and not OPTS["momentum"]([p], lr=0.1).defaults["nesterov"]
