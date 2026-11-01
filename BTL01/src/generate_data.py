"""Sinh dữ liệu: pool train 2000 (seed 2026), val 200 (seed 2027), test 100 (seed 2028). (TV1 – T1.3)

Chạy: python src/generate_data.py   -> data/train_pool.csv, val.csv, test.csv, meta.json
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.gaussian import C, MU, NORM, gaussian_pdf, sample_uniform  # noqa: E402
from src.utils.io import save_json, write_csv  # noqa: E402

SEEDS = {"train_pool": 2026, "val": 2027, "test": 2028}
SIZES = {"train_pool": 2000, "val": 200, "test": 100}


def main():
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    split_stats = {}

    for name, size in SIZES.items():
        rng = np.random.default_rng(SEEDS[name])
        x = sample_uniform(size, rng)
        y = gaussian_pdf(x)
        rows = [
            {"x1": float(xi[0]), "x2": float(xi[1]), "y": float(yi)}
            for xi, yi in zip(x, y)
        ]
        filename = "train_pool.csv" if name == "train_pool" else f"{name}.csv"
        write_csv(data_dir / filename, rows, fields=["x1", "x2", "y"])

        split_stats[name] = {
            "size": int(size),
            "seed": int(SEEDS[name]),
            "x1": {"min": float(x[:, 0].min()), "max": float(x[:, 0].max()),
                   "mean": float(x[:, 0].mean()), "std": float(x[:, 0].std())},
            "x2": {"min": float(x[:, 1].min()), "max": float(x[:, 1].max()),
                   "mean": float(x[:, 1].mean()), "std": float(x[:, 1].std())},
            "y": {"min": float(y.min()), "max": float(y.max()),
                  "mean": float(y.mean()), "std": float(y.std()),
                  "fraction_below_0_01": float(np.mean(y < 0.01))},
        }

    metadata = {
        "seeds": {name: int(seed) for name, seed in SEEDS.items()},
        "sizes": {name: int(size) for name, size in SIZES.items()},
        "mu": MU.tolist(),
        "covariance": C.tolist(),
        "domain": {"x1": [-2.0, 2.0], "x2": [-4.0, 4.0]},
        "y_max": float(NORM),
        "splits": split_stats,
    }
    save_json(data_dir / "meta.json", metadata)
    print(f"Generated data at {data_dir}")
    for name, stats in split_stats.items():
        print(f"  {name}: {stats['size']} samples, seed={stats['seed']}")


if __name__ == "__main__":
    main()
