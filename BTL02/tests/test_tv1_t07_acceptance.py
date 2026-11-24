# Phụ trách: TV1 – Đoàn Trung Kiên
"""Kiểm tra độc lập trainer cho T07 bằng dữ liệu tổng hợp chỉ trong unit test."""

import csv

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine.trainer import fit


def _inputs():
    generator = torch.Generator().manual_seed(73)
    x_train = torch.randn(8, 1, 4, 4, generator=generator)
    y_train = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0])
    x_val = torch.randn(4, 1, 4, 4, generator=generator)
    y_val = torch.tensor([0, 1, 2, 3])
    return x_train, y_train, x_val, y_val


def _loaders(fail_after_first_epoch=False):
    x_train, y_train, x_val, y_val = _inputs()
    generator = torch.Generator().manual_seed(19)
    train = DataLoader(TensorDataset(x_train, y_train), batch_size=8, shuffle=True, generator=generator)
    if fail_after_first_epoch:
        train = _FailOnSecondPass(train)
    val = DataLoader(TensorDataset(x_val, y_val), batch_size=4, shuffle=False)
    return train, val


class _FailOnSecondPass:
    def __init__(self, base):
        self.base = base
        self.passes = 0

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        self.passes += 1
        if self.passes == 2:
            raise RuntimeError("Injected failure after first completed epoch")
        yield from self.base


def _model():
    torch.manual_seed(31)
    return nn.Sequential(nn.Flatten(), nn.Linear(16, 7))


def _config(**overrides):
    return {"epochs": 2, "optimizer": "sgd", "lr": 0.1, "scheduler": "constant",
            "amp": False, "device": "cpu", "early_stopping_patience": 0, **overrides}


def _history(run_dir):
    with (run_dir / "history.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_two_epochs_repeat_with_same_seed(tmp_path):
    outputs = []
    for name in ("run_a", "run_b"):
        result = fit(_model(), *_loaders(), _config(seed=11), tmp_path / name)
        rows = _history(tmp_path / name)
        assert result["status"] == "Completed"
        assert len(rows) == 2
        outputs.append([{key: value for key, value in row.items() if key != "epoch_time_s"} for row in rows])

    assert outputs[0] == outputs[1]


def test_failure_after_checkpoint_removes_partial_weights(tmp_path):
    run_dir = tmp_path / "failed_run"
    result = fit(_model(), *_loaders(fail_after_first_epoch=True), _config(seed=11), run_dir)

    assert result["status"] == "Failed"
    assert result["weights_quyet_dinh"] == "DISCARD"
    assert result["weights_ma_ly_do"] == "EXCEPTION:RuntimeError"
    assert len(_history(run_dir)) == 1
    assert not list(run_dir.glob("*.pt*"))
    assert "Injected failure" in (run_dir / "run.log").read_text(encoding="utf-8")


def test_invalid_optimizer_records_failure_without_checkpoint(tmp_path):
    run_dir = tmp_path / "invalid_config"
    result = fit(_model(), *_loaders(), _config(optimizer="not-a-real-optimizer"), run_dir)

    assert result["status"] == "Failed"
    assert result["weights_quyet_dinh"] == "DISCARD"
    assert result["weights_ma_ly_do"] == "EXCEPTION:ValueError"
    assert not list(run_dir.glob("*.pt*"))


def test_cli_seeds_override_reaches_runner():
    import json
    import subprocess
    import sys
    from pathlib import Path

    btl02_dir = Path(__file__).resolve().parents[1]
    probe = """
import json
from src import run_experiment

def capture(config, member, only=None, root=run_experiment.ROOT, seeds=None):
    print(json.dumps({"config": config, "member": member, "only": only, "seeds": seeds}))

run_experiment.run = capture
run_experiment.main([
    "--config", "configs/baseline.json",
    "--member", "TV4",
    "--seeds", "1", "2",
])
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=btl02_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "config": "configs/baseline.json",
        "member": "TV4",
        "only": None,
        "seeds": [1, 2],
    }
