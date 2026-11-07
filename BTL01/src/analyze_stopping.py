"""Minh chứng chọn patience = 200 và max_epochs = 2000 (Plan 1.4). Không train lại, không đọc test.

1) Mô phỏng patience: results/grid_optimizer/adam_0.01 chạy đủ 2000 epoch không dừng sớm (patience = 2000),
   nên từ history.csv tính lại đúng logic early stopping của trainer.fit cho từng patience P:
   dừng ở epoch nào, best_epoch, val MSE tốt nhất -> results/stopping/patience_sim.csv.
2) Kiểm tra trần epoch: với mọi run của A/B/C, đếm run chạy chạm max_epochs (early stopping chưa kích hoạt)
   và run có best_epoch nằm trong 200 epoch cuối -> results/stopping/epoch_cap.csv.

    python src/analyze_stopping.py
"""
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.gaussian import Y_SCALE  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

RES = ROOT / "results"
FIG = ROOT / "figures"
FULL_RUN = RES / "grid_optimizer" / "adam_0.01"     # baseline Adam lr 0.01, 2000 epoch không dừng sớm
PATIENCES = (25, 50, 100, 200, 400, 800)
CAP_EXPS = ("expA", "expA_steps", "expB", "expB_wide", "expB_A200", "expC", "expC_ctrl", "grid_batch")


def simulate(val_loss, patience):
    """Lặp lại đúng trainer.fit: cập nhật best khi va < best, dừng khi wait >= patience. -> (stop, best_ep, best)."""
    best, best_ep, wait = math.inf, 0, 0
    for epoch, va in enumerate(val_loss, start=1):
        if va < best:
            best, best_ep, wait = va, epoch, 0
        else:
            wait += 1
            if wait >= patience:
                return epoch, best_ep, best
    return len(val_loss), best_ep, best


def read_val_loss(run_dir):
    with (run_dir / "history.csv").open(newline="", encoding="utf-8") as f:
        return np.asarray([float(r["val_loss"]) for r in csv.DictReader(f)])


def patience_table():
    runs = sorted(FULL_RUN.glob("seed_*"))
    rows = []
    for run_dir in runs:
        v = read_val_loss(run_dir)
        full_best = float(v.min()) * Y_SCALE ** 2
        for p in PATIENCES + (len(v),):
            stop, best_ep, best = simulate(v, p)
            mse = best * Y_SCALE ** 2
            rows.append({"seed": run_dir.name[5:], "patience": p, "stop_epoch": stop, "best_epoch": best_ep,
                         "best_val_mse": mse, "ratio_to_full_2000": mse / full_best})
    return rows


def cap_table():
    rows = []
    for name in CAP_EXPS:
        path = RES / name / "summary.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            runs = list(csv.DictReader(f))
        hit, late = 0, 0
        for r in runs:
            cfg = json.loads((RES / name / r["value"] / f"seed_{r['seed']}" / "config.json").read_text(encoding="utf-8"))
            cap = cfg["max_epochs"]
            hit += int(r["epochs_run"]) >= cap
            late += int(r["best_epoch"]) > cap - cfg["patience"]
        rows.append({"exp": name, "n_runs": len(runs), "hit_max_epochs": hit, "best_in_last_patience": late,
                     "pct_hit": 100 * hit / len(runs)})
    return rows


def plot(sim, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ps = sorted({r["patience"] for r in sim})
    labels = [str(p) if p < 2000 else "none (2000)" for p in ps]
    for key, ax, ylab in (("ratio_to_full_2000", a, "Best val MSE / best val MSE of full 2000-epoch run"),
                          ("stop_epoch", b, "Epoch at which training stops")):
        # từng seed là 1 điểm (tỉ lệ luôn ≥ 1 nên không vẽ ± std đối xứng), đường nối trung bình
        for i, p in enumerate(ps):
            ax.scatter([i] * 3, [r[key] for r in sim if r["patience"] == p], color="#93c5fd", s=18, zorder=2)
        mean = [np.mean([r[key] for r in sim if r["patience"] == p]) for p in ps]
        ax.plot(range(len(ps)), mean, marker="o", color="#2563eb", label="mean of 3 seeds", zorder=3)
        ax.axvline(ps.index(200), ls="--", color="#f97316", label="chosen patience = 200")
        ax.set(xticks=range(len(ps)), xticklabels=labels, xlabel="Early-stopping patience (epochs)", ylabel=ylab)
        ax.grid(True, alpha=0.22)
        ax.legend()
    a.set_yscale("log")
    a.set_title("Quality loss from stopping early (baseline [10], Adam lr 0.01)")
    b.set_title("Training cost (dots = seeds)")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    out = RES / "stopping"
    sim, cap = patience_table(), cap_table()
    write_csv(out / "patience_sim.csv", sim)
    write_csv(out / "epoch_cap.csv", cap)
    plot(sim, FIG / "stopping_patience.png")
    for p in PATIENCES:
        rs = [r for r in sim if r["patience"] == p]
        print(f"patience={p:4d}  stop_epoch={np.mean([r['stop_epoch'] for r in rs]):6.0f}  "
              f"val_mse/full={np.mean([r['ratio_to_full_2000'] for r in rs]):.3f}")
    for r in cap:
        print(f"{r['exp']:10s} runs={r['n_runs']:3d}  hit_max_epochs={r['hit_max_epochs']:3d}  "
              f"best_in_last_patience={r['best_in_last_patience']:3d}")


if __name__ == "__main__":
    main()
