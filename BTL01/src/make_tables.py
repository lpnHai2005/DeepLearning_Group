"""Sinh mọi bảng số liệu cho báo cáo từ results/*/summary.csv (Plan 1.6). Không gõ số bằng tay.

    python src/make_tables.py

Với mỗi thực nghiệm có summary.csv -> report/tables/<exp>.md (bảng Markdown, mean ± std qua các seed,
dán thẳng vào Word) và report/tables/<exp>.csv (số thô). Mỗi bảng ghi nguồn để truy vết.
Cột test là hậu kiểm (src/posthoc_test.py), không dùng để chọn cấu hình.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.summarize_expA import mean_std  # noqa: E402
from src.utils.io import write_csv  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "report" / "tables"
EXPS = ("baseline", "grid_baseline", "grid_optimizer", "grid_batch", "expA", "expA_steps", "expB", "expB_wide",
        "expB_A200", "expC", "expC_ctrl", "final")
COLS = ("train_mse", "val_mse", "test_mse", "val_mae", "test_mae", "test_rmse", "test_r2",
        "best_epoch", "epochs_run", "train_time_s")


def sort_key(value):
    """Sắp giá trị số theo số (200 < 1000), còn lại theo chữ; '10-10' sắp theo số tầng."""
    head = value.split("_")[0].split("-")
    try:
        return (0, len(head), float(head[0]), value)
    except ValueError:
        return (1, 0, 0.0, value)


def aggregate(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["value"]].append(r)
    out = []
    for value in sorted(grouped, key=sort_key):
        runs = grouped[value]
        row = {"value": value, "n_params": runs[0]["n_params"], "n_seeds": len(runs)}
        for c in COLS:
            vals = [float(r[c]) for r in runs if r.get(c) not in (None, "")]
            if len(vals) == len(runs):
                row[f"{c}_mean"], row[f"{c}_std"] = mean_std(vals)
        out.append(row)
    return out


def fmt(mean, std, col):
    if col in ("best_epoch", "epochs_run"):
        return f"{mean:.0f} ± {std:.0f}"
    if col == "train_time_s":
        return f"{mean:.1f} ± {std:.1f}"
    if col == "test_r2":
        return f"{mean:.5f} ± {std:.5f}"
    return f"{mean:.2e} ± {std:.1e}"


def to_markdown(name, agg):
    cols = [c for c in COLS if f"{c}_mean" in agg[0]]
    lines = [f"<!-- sinh tự động bởi src/make_tables.py từ results/{name}/summary.csv; không sửa tay -->",
             "| value | params | seeds | " + " | ".join(cols) + " |",
             "|" + "---|" * (3 + len(cols))]
    for r in agg:
        cells = [fmt(r[f"{c}_mean"], r[f"{c}_std"], c) for c in cols]
        lines.append(f"| {r['value']} | {r['n_params']} | {r['n_seeds']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"Nguồn: `results/{name}/summary.csv` (mỗi run 1 dòng; log: `results/{name}/run.log`"
                 + (f", `results/{name}/posthoc_test.log`" if (RES / name / "posthoc_test.log").exists() else "")
                 + "). Giá trị: trung bình ± độ lệch chuẩn (ddof=1) qua các seed; MSE/MAE trên thang y gốc.")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in EXPS:
        path = RES / name / "summary.csv"
        if not path.exists():
            print(f"skip {name}: no summary.csv")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            agg = aggregate(list(csv.DictReader(f)))
        write_csv(OUT / f"{name}.csv", agg)
        (OUT / f"{name}.md").write_text(to_markdown(name, agg), encoding="utf-8")
        print(f"{name}: {len(agg)} rows -> report/tables/{name}.md")


if __name__ == "__main__":
    main()
