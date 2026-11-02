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
    "src.run_experiment", "src.make_report_assets", "src.summarize_expC",
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
