"""Kiểm tra khung repo: mọi module import được và các file cấu hình hợp lệ."""
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "src.data.gaussian", "src.data.dataset", "src.generate_data",
    "src.model.mlp", "src.engine.trainer", "src.engine.evaluate",
    "src.utils.seed", "src.utils.io", "src.utils.timer",
    "src.visual.plots", "src.visual.draw_mlp",
    "src.run_experiment", "src.make_report_assets", "src.summarize_expB", "src.summarize_expC", "src.summarize_optimizer", "src.summarize_grid_baseline",
]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


@pytest.mark.parametrize("cfg", sorted((ROOT / "configs").glob("*.json")))
def test_config_valid(cfg):
    d = json.loads(cfg.read_text(encoding="utf-8"))
    for key in ("exp", "vary", "base", "seeds", "out_dir"):
        assert key in d, f"{cfg.name} thiếu khóa {key}"


def test_no_forbidden_frameworks():
    banned = ("tensorflow", "keras")
    for py in (ROOT / "src").rglob("*.py"):
        text = py.read_text(encoding="utf-8").lower()
        for b in banned:
            assert f"import {b}" not in text and f"from {b}" not in text, f"{py} dùng {b}"


def test_run_log_has_one_line_per_run_and_is_rewritten(tmp_path):
    from src.run_experiment import run
    cfg = {"exp": "t", "vary": {"lr": [0.01, 0.1]}, "seeds": [0, 1], "out_dir": str(tmp_path),
           "base": {"n_train": 20, "hidden_sizes": [2], "activation": "tanh", "optimizer": "sgd",
                    "batch_size": 8, "max_epochs": 1, "patience": 1}}
    for _ in range(2):   # lần 2 phải ghi đè, không nối thêm
        run(cfg)
    lines = (tmp_path / "run.log").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and lines[0].startswith("[t] 0.01 seed=0  val_mse=")


def test_mean_curve_stops_at_shortest_seed():
    import numpy as np

    from src.summarize_expC import mean_curve
    from src.data.gaussian import Y_SCALE
    y = mean_curve([{"val_loss": np.ones(50)}, {"val_loss": np.ones(40)}], "val_loss")
    assert y.shape == (40,) and np.allclose(y, Y_SCALE ** 2)
