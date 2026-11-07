"""Ghi môi trường chạy (Python, thư viện, CPU/GPU, OS) ra results/env.txt để báo cáo Ch.3 và QA tái lập. (Plan 1.5)

    python src/env_info.py
"""
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "numpy", "pandas", "scipy", "matplotlib", "pytest")


def collect():
    lines = [f"date: {datetime.now().isoformat(timespec='seconds')}",
             f"os: {platform.platform()}",
             f"machine: {platform.machine()}  processor: {platform.processor() or 'n/a'}",
             f"python: {sys.version.split()[0]}"]
    for p in PACKAGES:
        try:
            lines.append(f"{p}: {version(p)}")
        except PackageNotFoundError:
            lines.append(f"{p}: not installed")
    lines += [f"torch threads used by run_experiment.py: 1 (torch.set_num_threads(1))",
              f"cuda available: {torch.cuda.is_available()}",
              f"device used: cpu"]
    return lines


if __name__ == "__main__":
    out = ROOT / "results" / "env.txt"
    out.write_text("\n".join(collect()) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
