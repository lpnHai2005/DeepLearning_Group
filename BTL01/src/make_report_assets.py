"""Gom results/*/summary.csv thành bảng (mean ± std), đồng bộ log (run.log, baseline, final),
chọn mô hình cuối theo validation, đánh giá tập test đúng 1 lần và xuất hình cho báo cáo.
(TV3 – T3.5, T3.6 & TV4 – T4.3)
"""
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_split, subset  # noqa: E402
from src.data.summarize_expA import mean_std  # noqa: E402
from src.engine.evaluate import evaluate_grid, metrics, predict  # noqa: E402
from src.model.mlp import MLP, n_params  # noqa: E402
from src.utils.io import save_json, write_csv  # noqa: E402
from src.visual.plots import (  # noqa: E402
    plot_error_map,
    plot_loss_curve,
    plot_parity,
    plot_surface_comparison,
)

RES = ROOT / "results"
FIG = ROOT / "figures"


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_run_log_from_summary(exp_dir):
    """Đảm bảo mọi thư mục thực nghiệm đều có run.log đồng nhất định dạng với run_experiment.py."""
    summary_path = Path(exp_dir) / "summary.csv"
    if not summary_path.exists():
        return
    rows = _read_csv(summary_path)
    lines = [
        f"[{r['exp']}] {r['value']} seed={r['seed']}  val_mse={float(r['val_mse']):.3e}  "
        f"best_epoch={r['best_epoch']}/{r['epochs_run']}  {float(r['train_time_s']):.1f}s"
        for r in rows
    ]
    (Path(exp_dir) / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_baseline_dir():
    """Đảm bảo results/baseline/ có đầy đủ 3 seed, summary.csv và run.log từ cấu hình baseline [10]."""
    base_dir = RES / "baseline"
    src_dir = RES / "expB" / "10"
    if not src_dir.exists():
        return
    base_dir.mkdir(parents=True, exist_ok=True)
    expb_rows = [r for r in _read_csv(RES / "expB" / "summary.csv") if r["value"] == "10"]
    out_rows = []
    for r in expb_rows:
        seed = r["seed"]
        s_src = src_dir / f"seed_{seed}"
        s_dst = base_dir / "base" / f"seed_{seed}"
        s_dst.mkdir(parents=True, exist_ok=True)
        for fname in ("best.pt", "history.csv", "config.json"):
            shutil.copy2(s_src / fname, s_dst / fname)
        row_copy = dict(r)
        row_copy["exp"] = "baseline"
        row_copy["value"] = "base"
        save_json(s_dst / "metrics.json", row_copy)
        out_rows.append(row_copy)
    if out_rows:
        write_csv(base_dir / "summary.csv", out_rows, list(out_rows[0].keys()))
        _write_run_log_from_summary(base_dir)


def _eval_posthoc_test_for_group(exp_dir, x_te, y_te):
    """Tính hậu kiểm (post-hoc) chỉ số test trên các best.pt đã đóng băng cho từng nhóm giá trị."""
    rows = _read_csv(Path(exp_dir) / "summary.csv")
    by_val = defaultdict(list)
    for r in rows:
        val = r["value"]
        seed = r["seed"]
        run_dir = Path(exp_dir) / val / f"seed_{seed}"
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        model = MLP(cfg["hidden_sizes"], cfg["activation"])
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
        te_m = metrics(y_te, predict(model, x_te))
        by_val[val].append(te_m)
    out = {}
    for val, mlist in by_val.items():
        out[val] = {}
        for k in ("mse", "mae", "rmse", "r2"):
            out[val][f"test_{k}_mean"], out[val][f"test_{k}_std"] = mean_std([m[k] for m in mlist])
    return out


def _rows_from_summary(exp_name, label):
    """Gom thực nghiệm bổ sung trực tiếp từ summary.csv (đã có test_* từng run nhờ posthoc_test.py)."""
    path = RES / exp_name / "summary.csv"
    if not path.exists():
        return []
    groups = defaultdict(list)
    for r in _read_csv(path):
        groups[r["value"]].append(r)
    out = []
    for value, runs in groups.items():
        cfg = json.loads((RES / exp_name / value / f"seed_{runs[0]['seed']}" / "config.json").read_text(encoding="utf-8"))
        hs = ",".join(map(str, cfg["hidden_sizes"]))
        row = {"exp": label, "config": f"A={cfg['n_train']}, [{hs}]", "n_train": cfg["n_train"],
               "hidden_sizes": hs.replace(",", "-"), "n_params": int(runs[0]["n_params"]), "n_seeds": len(runs)}
        for s in ("train_mse", "val_mse", "test_mse", "val_mae", "test_mae", "val_rmse", "test_rmse", "test_r2"):
            row[f"{s}_mean"], row[f"{s}_std"] = mean_std([float(r[s]) for r in runs])
        out.append(row)
    return sorted(out, key=lambda r: (r["n_train"], r["n_params"]))


def build_summary_all():
    """Gom bảng tổng hợp của A, B, C (và C_ctrl) kèm hậu kiểm test về results/summary_all.csv."""
    x_te, y_te = load_split("test")
    test_a = _eval_posthoc_test_for_group(RES / "expA", x_te, y_te)
    test_b = _eval_posthoc_test_for_group(RES / "expB", x_te, y_te)
    test_c = _eval_posthoc_test_for_group(RES / "expC", x_te, y_te)
    test_c_ctrl = _eval_posthoc_test_for_group(RES / "expC_ctrl", x_te, y_te)

    rows_out = []

    for r in _read_csv(RES / "expA" / "aggregate.csv"):
        val_key = str(r["n_train"])
        tm = test_a[val_key]
        rows_out.append({
            "exp": "A",
            "config": f"A={r['n_train']}, [10]",
            "n_train": int(r["n_train"]),
            "hidden_sizes": "10",
            "n_params": 41,
            "n_seeds": int(r["n_seeds"]),
            "train_mse_mean": float(r["train_mse_mean"]),
            "train_mse_std": float(r["train_mse_std"]),
            "val_mse_mean": float(r["val_mse_mean"]),
            "val_mse_std": float(r["val_mse_std"]),
            "test_mse_mean": tm["test_mse_mean"],
            "test_mse_std": tm["test_mse_std"],
            "val_mae_mean": float(r["val_mae_mean"]),
            "val_mae_std": float(r["val_mae_std"]),
            "test_mae_mean": tm["test_mae_mean"],
            "test_mae_std": tm["test_mae_std"],
            "val_rmse_mean": float(r["val_rmse_mean"]),
            "val_rmse_std": float(r["val_rmse_std"]),
            "test_rmse_mean": tm["test_rmse_mean"],
            "test_rmse_std": tm["test_rmse_std"],
            "test_r2_mean": tm["test_r2_mean"],
            "test_r2_std": tm["test_r2_std"],
        })

    for r in _read_csv(RES / "expB" / "aggregate.csv"):
        if r["group"] != "expB":
            continue
        val_key = str(r["width"])
        tm = test_b[val_key]
        rows_out.append({
            "exp": "B",
            "config": f"B={r['width']}, [{r['width']}]",
            "n_train": 1000,
            "hidden_sizes": str(r["width"]),
            "n_params": int(r["n_params"]),
            "n_seeds": int(r["n_seeds"]),
            "train_mse_mean": float(r["train_mse_mean"]),
            "train_mse_std": float(r["train_mse_std"]),
            "val_mse_mean": float(r["val_mse_mean"]),
            "val_mse_std": float(r["val_mse_std"]),
            "test_mse_mean": tm["test_mse_mean"],
            "test_mse_std": tm["test_mse_std"],
            "val_mae_mean": float(r["val_mae_mean"]),
            "val_mae_std": float(r["val_mae_std"]),
            "test_mae_mean": tm["test_mae_mean"],
            "test_mae_std": tm["test_mae_std"],
            "val_rmse_mean": float(r["val_rmse_mean"]),
            "val_rmse_std": float(r["val_rmse_std"]),
            "test_rmse_mean": tm["test_rmse_mean"],
            "test_rmse_std": tm["test_rmse_std"],
            "test_r2_mean": tm["test_r2_mean"],
            "test_r2_std": tm["test_r2_std"],
        })

    for r in _read_csv(RES / "expC" / "aggregate.csv"):
        exp_label = "C" if r["family"] == "deep" else "C_ctrl"
        cfg_label = f"C={r['depth']} [{r['value'].replace('-', ',')}]" if exp_label == "C" else f"Ctrl [{r['width']}]"
        tm = test_c[r["value"]] if exp_label == "C" else test_c_ctrl[r["value"]]
        rows_out.append({
            "exp": exp_label,
            "config": cfg_label,
            "n_train": 1000,
            "hidden_sizes": r["value"],
            "n_params": int(r["n_params"]),
            "n_seeds": int(r["n_seeds"]),
            "train_mse_mean": float(r["train_mse_mean"]),
            "train_mse_std": float(r["train_mse_std"]),
            "val_mse_mean": float(r["val_mse_mean"]),
            "val_mse_std": float(r["val_mse_std"]),
            "test_mse_mean": tm["test_mse_mean"],
            "test_mse_std": tm["test_mse_std"],
            "val_mae_mean": float(r["val_mae_mean"]),
            "val_mae_std": float(r["val_mae_std"]),
            "test_mae_mean": tm["test_mae_mean"],
            "test_mae_std": tm["test_mae_std"],
            "val_rmse_mean": float(r["val_rmse_mean"]),
            "val_rmse_std": float(r["val_rmse_std"]),
            "test_rmse_mean": tm["test_rmse_mean"],
            "test_rmse_std": tm["test_rmse_std"],
            "test_r2_mean": tm["test_r2_mean"],
            "test_r2_std": tm["test_r2_std"],
        })

    for exp_name, label in (("expA_steps", "A_steps"), ("expB_wide", "B"), ("expB_A200", "B_A200")):
        rows_out += _rows_from_summary(exp_name, label)

    fields = [
        "exp", "config", "n_train", "hidden_sizes", "n_params", "n_seeds",
        "train_mse_mean", "train_mse_std",
        "val_mse_mean", "val_mse_std",
        "test_mse_mean", "test_mse_std",
        "val_mae_mean", "val_mae_std",
        "test_mae_mean", "test_mae_std",
        "val_rmse_mean", "val_rmse_std",
        "test_rmse_mean", "test_rmse_std",
        "test_r2_mean", "test_r2_std",
    ]
    write_csv(RES / "summary_all.csv", rows_out, fields)
    return rows_out


def evaluate_final_candidates():
    """Chọn mô hình theo validation MSE và đánh giá trên tập test độc lập (100 mẫu) + lưới 100×100."""
    x_pool, y_pool = load_split("train_pool")
    x_tr, y_tr = subset(x_pool, y_pool, 1000)
    x_val, y_val = load_split("val")
    x_te, y_te = load_split("test")

    final_dir = RES / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # 1. Baseline tầm thường: dự đoán hằng số = mean(y_train)
    const_val = metrics(y_val, np.full_like(y_val, y_tr.mean()))
    const_test = metrics(y_te, np.full_like(y_te, y_tr.mean()))

    # 2. Đánh giá các ứng viên đại diện (đã chọn bằng validation) trên test và lưới dày 100×100
    candidates = [
        ("Baseline C=1 [10]", RES / "expC" / "10", [10]),
        ("Ctrl B=93 [93]", RES / "expC_ctrl" / "93", [93]),
        ("Selected C=3 [10,10,10]", RES / "expC" / "10-10-10", [10, 10, 10]),
    ]

    expc_summary = {(r["value"], int(r["seed"])): r for r in _read_csv(RES / "expC" / "summary.csv")}
    ctrl_summary = {(r["value"], int(r["seed"])): r for r in _read_csv(RES / "expC_ctrl" / "summary.csv")}

    seed_rows = []
    agg_rows = [{
        "candidate": "Constant mean predictor",
        "hidden_sizes": "const",
        "n_params": 1,
        "val_mse_mean": const_val["mse"], "val_mse_std": 0.0,
        "test_mse_mean": const_test["mse"], "test_mse_std": 0.0,
        "test_mae_mean": const_test["mae"], "test_mae_std": 0.0,
        "test_rmse_mean": const_test["rmse"], "test_rmse_std": 0.0,
        "test_r2_mean": const_test["r2"], "test_r2_std": 0.0,
        "grid_mse_mean": np.nan, "grid_mse_std": np.nan,
        "grid_mae_mean": np.nan, "grid_mae_std": np.nan,
        "grid_rmse_mean": np.nan, "grid_rmse_std": np.nan,
        "grid_r2_mean": np.nan, "grid_r2_std": np.nan,
        "grid_max_abs_err_mean": np.nan, "grid_max_abs_err_std": np.nan,
    }]

    best_seed_info = None
    final_log_lines = []

    for label, base_dir, hidden_sizes in candidates:
        val_name = "-".join(map(str, hidden_sizes))
        cand_runs = []
        for seed in (0, 1, 2):
            run_dir = base_dir / f"seed_{seed}"
            ckpt_path = run_dir / "best.pt"
            model = MLP(hidden_sizes=hidden_sizes, activation="sigmoid")
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)

            tr_m = metrics(y_tr, predict(model, x_tr))
            va_m = metrics(y_val, predict(model, x_val))
            te_m = metrics(y_te, predict(model, x_te))
            gr_m = evaluate_grid(model, n=100)

            orig = expc_summary.get((val_name, seed)) or ctrl_summary.get((val_name, seed))
            row = {
                "exp": "final",
                "candidate": label,
                "value": val_name,
                "hidden_sizes": val_name,
                "seed": seed,
                "n_params": n_params(model),
                "train_mse": tr_m["mse"],
                "train_mae": tr_m["mae"],
                "train_rmse": tr_m["rmse"],
                "val_mse": va_m["mse"],
                "val_mae": va_m["mae"],
                "val_rmse": va_m["rmse"],
                "test_mse": te_m["mse"],
                "test_mae": te_m["mae"],
                "test_rmse": te_m["rmse"],
                "test_r2": te_m["r2"],
                "grid_mse": gr_m["mse"],
                "grid_mae": gr_m["mae"],
                "grid_rmse": gr_m["rmse"],
                "grid_r2": gr_m["r2"],
                "grid_max_abs_err": gr_m["max_abs_err"],
                "best_epoch": int(orig["best_epoch"]),
                "epochs_run": int(orig["epochs_run"]),
                "train_time_s": float(orig["train_time_s"]),
                "time_per_epoch_ms": float(orig["time_per_epoch_ms"]),
            }
            seed_rows.append(row)
            cand_runs.append((row, model, run_dir))

            if label == "Selected C=3 [10,10,10]":
                dst_seed_dir = final_dir / val_name / f"seed_{seed}"
                dst_seed_dir.mkdir(parents=True, exist_ok=True)
                for fname in ("best.pt", "history.csv", "config.json"):
                    shutil.copy2(run_dir / fname, dst_seed_dir / fname)
                save_json(dst_seed_dir / "metrics.json", row)
                final_log_lines.append(
                    f"[final] {val_name} seed={seed}  val_mse={row['val_mse']:.3e}  "
                    f"test_mse={row['test_mse']:.3e}  best_epoch={row['best_epoch']}/{row['epochs_run']}  "
                    f"{row['train_time_s']:.1f}s"
                )

        agg = {
            "candidate": label,
            "hidden_sizes": val_name,
            "n_params": cand_runs[0][0]["n_params"],
        }
        for k in (
            "val_mse", "test_mse", "test_mae", "test_rmse", "test_r2",
            "grid_mse", "grid_mae", "grid_rmse", "grid_r2", "grid_max_abs_err",
        ):
            agg[f"{k}_mean"], agg[f"{k}_std"] = mean_std([r[0][k] for r in cand_runs])
        agg_rows.append(agg)

        if label == "Selected C=3 [10,10,10]":
            best_seed_info = min(cand_runs, key=lambda item: item[0]["val_mse"])

    write_csv(final_dir / "summary.csv", seed_rows)
    write_csv(final_dir / "aggregate.csv", agg_rows)
    (final_dir / "run.log").write_text("\n".join(final_log_lines) + "\n", encoding="utf-8")

    # Lưu trọng số và hình trực quan hóa của mô hình được chọn (theo validation MSE)
    best_row, best_model, best_run_dir = best_seed_info
    shutil.copy2(best_run_dir / "best.pt", final_dir / "best.pt")
    shutil.copy2(best_run_dir / "config.json", final_dir / "config.json")
    save_json(final_dir / "metrics.json", best_row)

    y_te_hat = predict(best_model, x_te)
    plot_parity(
        y_te, y_te_hat, FIG / "final_parity_test.png", split="test",
        title=f"Selected MLP [10, 10, 10] (seed {best_row['seed']}): True vs. predicted on test set",
    )
    plot_surface_comparison(
        best_model, FIG / "final_surface_comparison.png", n=100,
        title=f"Selected MLP [10, 10, 10] (grid MSE={best_row['grid_mse']:.2e}, R²={best_row['grid_r2']:.6f})",
    )
    plot_error_map(
        best_model, FIG / "final_error_map.png", n=100, sample_x=x_te,
        title=f"Absolute error |y − ŷ| (max={best_row['grid_max_abs_err']:.2e}, MAE={best_row['grid_mae']:.2e})",
    )
    plot_loss_curve(
        best_run_dir / "history.csv", FIG / "final_loss_curve.png",
        title=f"Selected MLP [10, 10, 10] (seed {best_row['seed']}): Training and validation loss",
    )
    return agg_rows, best_row


def sync_all_run_logs():
    for exp_name in ("expA", "expB", "expC", "expC_ctrl", "grid_baseline", "grid_optimizer"):
        exp_dir = RES / exp_name
        if exp_dir.exists() and not (exp_dir / "run.log").exists():
            _write_run_log_from_summary(exp_dir)


def main():
    ensure_baseline_dir()
    sync_all_run_logs()
    summary_all = build_summary_all()
    agg_rows, best_row = evaluate_final_candidates()
    print(f"Wrote {len(summary_all)} configurations to {RES / 'summary_all.csv'}")
    print("\n=== Final Test & Dense-Grid Evaluation (mean ± std across 3 seeds) ===")
    for r in agg_rows:
        print(
            f"{r['candidate']:26s} params={r['n_params']:3d}  "
            f"val_mse={r['val_mse_mean']:.3e}±{r['val_mse_std']:.1e}  "
            f"test_mse={r['test_mse_mean']:.3e}±{r['test_mse_std']:.1e}  "
            f"test_rmse={r['test_rmse_mean']:.3e}±{r['test_rmse_std']:.1e}  "
            f"test_r2={r['test_r2_mean']:.6f}"
        )
    print(
        f"\nBest single checkpoint (by val_mse): C=3 [10,10,10] seed={best_row['seed']}  "
        f"val_mse={best_row['val_mse']:.3e}  test_mse={best_row['test_mse']:.3e}  "
        f"test_mae={best_row['test_mae']:.3e}  test_rmse={best_row['test_rmse']:.3e}  "
        f"test_r2={best_row['test_r2']:.6f}  grid_mse={best_row['grid_mse']:.3e}"
    )


if __name__ == "__main__":
    main()
