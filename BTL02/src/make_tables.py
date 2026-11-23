"""Tổng hợp bảng kết quả thực nghiệm E1–E7 từ tracking/experiments.csv (mean ± std, so sánh baseline)
và điền mục Kết quả trong report/NHAT_KY_THUC_NGHIEM.md.

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T09.

Dùng (từ thư mục BTL02):
    python src/make_tables.py [--sync-orphans] [--update-log report/NHAT_KY_THUC_NGHIEM.md]
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # chạy được cả dạng `python src/make_tables.py`

from src.engine.trainer import scan_orphans  # noqa: E402
from src.run_experiment import collect_row  # noqa: E402
from src.utils.io import BTL02, append_csv_row  # noqa: E402

SUMMARY_COLS = [
    "group",
    "task_id",
    "variant",
    "n_runs",
    "n_completed",
    "val_acc_mean",
    "val_acc_std",
    "val_acc_fmt",
    "val_macro_f1_mean",
    "val_macro_f1_std",
    "val_macro_f1_fmt",
    "best_val_loss_mean",
    "epoch_time_s_tb_mean",
    "so_voi_baseline_tu_tinh",
]


def sync_orphans(root=BTL02):
    """Quét các run bị kill cứng trong results/ và nối dòng còn thiếu vào tracking/experiments.csv."""
    root = Path(root)
    csv_path = root / "tracking" / "experiments.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        known = {r["exp_id"] for r in reader}
    added = []
    for run_dir in scan_orphans(root / "results"):
        if run_dir.name not in known:
            append_csv_row(csv_path, header, collect_row(run_dir, root))
            known.add(run_dir.name)
            added.append(run_dir.name)
    return added


def load_experiments(experiments_csv):
    """Đọc toàn bộ các dòng từ tracking/experiments.csv."""
    with open(experiments_csv, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _group_of(row):
    exp_id = (row.get("exp_id") or "").strip()
    if "-" in exp_id:
        return exp_id.split("-", 1)[0]
    return (row.get("task_id") or "OTHER").strip() or "OTHER"


def _variant_of(row):
    hp = (row.get("hyperparams_json") or "").strip()
    if hp:
        try:
            obj = json.loads(hp)
            if obj.get("variant"):
                return str(obj["variant"])
        except json.JSONDecodeError:
            pass
    exp_id = (row.get("exp_id") or "").strip()
    return re.sub(r"_s\d+$", "", exp_id)


def _float_or_none(val):
    s = (val or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def baseline_stats(rows, baseline_task="T11"):
    """Tính (mean, std) val_acc của các run baseline (T11 hoặc group 'B0'/'baseline') đã Completed."""
    vals = []
    for r in rows:
        if r.get("trang_thai_ky_thuat") != "Completed":
            continue
        grp = _group_of(r).lower()
        if r.get("task_id") == baseline_task or grp in ("baseline", "b0"):
            v = _float_or_none(r.get("val_acc"))
            if v is not None:
                vals.append(v)
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
    return float(arr.mean()), std


def compare_to_baseline(val_acc, base_mean, base_std):
    """So sánh thuần số học với baseline (không diễn giải định tính)."""
    if val_acc is None or base_mean is None:
        return ""
    diff = val_acc - base_mean
    if base_std is not None and abs(diff) < base_std:
        return f"{diff:+.4f} (< std baseline)"
    tag = "cao hơn" if diff > 0 else ("thấp hơn" if diff < 0 else "bằng")
    return f"{diff:+.4f} ({tag})"


def _fmt_mean_std(vals):
    if not vals:
        return None, None, ""
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) >= 2:
        std = float(arr.std(ddof=1))
        return round(mean, 6), round(std, 6), f"{mean:.4f} ± {std:.4f}"
    return round(mean, 6), 0.0, f"{mean:.4f}"


def summarize_rows(rows, base_mean=None, base_std=None):
    """Nhóm theo (group, task_id, variant) và tính mean ± std cho các chỉ số chính."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[(_group_of(r), r.get("task_id", ""), _variant_of(r))].append(r)

    summary = []
    for (grp, tid, var), items in sorted(buckets.items()):
        completed = [x for x in items if x.get("trang_thai_ky_thuat") == "Completed"]
        accs = [v for x in completed if (v := _float_or_none(x.get("val_acc"))) is not None]
        f1s = [v for x in completed if (v := _float_or_none(x.get("val_macro_f1"))) is not None]
        losses = [v for x in completed if (v := _float_or_none(x.get("best_val_loss"))) is not None]
        times = [v for x in completed if (v := _float_or_none(x.get("epoch_time_s_tb"))) is not None]

        acc_m, acc_s, acc_fmt = _fmt_mean_std(accs)
        f1_m, f1_s, f1_fmt = _fmt_mean_std(f1s)
        loss_m, _, _ = _fmt_mean_std(losses)
        time_m, _, _ = _fmt_mean_std(times)

        summary.append(
            {
                "group": grp,
                "task_id": tid,
                "variant": var,
                "n_runs": len(items),
                "n_completed": len(completed),
                "val_acc_mean": "" if acc_m is None else acc_m,
                "val_acc_std": "" if acc_s is None else acc_s,
                "val_acc_fmt": acc_fmt,
                "val_macro_f1_mean": "" if f1_m is None else f1_m,
                "val_macro_f1_std": "" if f1_s is None else f1_s,
                "val_macro_f1_fmt": f1_fmt,
                "best_val_loss_mean": "" if loss_m is None else loss_m,
                "epoch_time_s_tb_mean": "" if time_m is None else time_m,
                "so_voi_baseline_tu_tinh": compare_to_baseline(acc_m, base_mean, base_std),
            }
        )
    return summary


def _write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _write_markdown(path, cols, rows, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_Chưa có dữ liệu thực nghiệm._")
    else:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_tables(experiments_csv, out_dir):
    """Sinh các bảng CSV và Markdown tổng hợp toàn bộ & theo từng nhóm từ tracking/experiments.csv."""
    experiments_csv = Path(experiments_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_experiments(experiments_csv)
    base_mean, base_std = baseline_stats(rows)
    summary = summarize_rows(rows, base_mean=base_mean, base_std=base_std)

    written = []
    all_csv = out_dir / "summary_all.csv"
    all_md = out_dir / "summary_all.md"
    _write_csv(all_csv, SUMMARY_COLS, summary)
    _write_markdown(all_md, SUMMARY_COLS, summary, "Tổng hợp kết quả thực nghiệm")
    written.extend([all_csv, all_md])

    by_group = defaultdict(list)
    for r in summary:
        by_group[r["group"]].append(r)

    for grp, grp_rows in sorted(by_group.items()):
        g_csv = out_dir / f"{grp}_summary.csv"
        g_md = out_dir / f"{grp}_summary.md"
        _write_csv(g_csv, SUMMARY_COLS, grp_rows)
        _write_markdown(g_md, SUMMARY_COLS, grp_rows, f"Tổng hợp nhóm {grp}")
        written.extend([g_csv, g_md])

    return written


def update_experiment_log(md_path, experiments_csv):
    """Tự điền dòng `- Kết quả: ...` cho từng mục `### [<exp_id>]` trong NHAT_KY_THUC_NGHIEM.md."""
    md_path = Path(md_path)
    if not md_path.exists():
        return 0
    rows = {r["exp_id"]: r for r in load_experiments(experiments_csv)}
    lines = md_path.read_text(encoding="utf-8").splitlines()
    current_id = None
    updated = 0
    out_lines = []
    for line in lines:
        m = re.match(r"^###\s+\[([^\]]+)\]", line)
        if m:
            current_id = m.group(1).strip()
            out_lines.append(line)
            continue
        if current_id and line.lstrip().startswith("- Kết quả:"):
            r = rows.get(current_id)
            if r is not None:
                line = (
                    f"- Kết quả: status={r.get('trang_thai_ky_thuat','')}, "
                    f"best_val_loss={r.get('best_val_loss','')}, "
                    f"val_acc={r.get('val_acc','')}, "
                    f"val_macro_f1={r.get('val_macro_f1','')}, "
                    f"best_epoch={r.get('best_epoch','')}, "
                    f"time_s={r.get('tong_thoi_gian_s','')}, "
                    f"run_dir={r.get('run_dir','')}"
                )
                updated += 1
            current_id = None
        out_lines.append(line)
    md_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return updated


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tổng hợp bảng kết quả thực nghiệm từ tracking/experiments.csv")
    ap.add_argument("--experiments", default=str(BTL02 / "tracking" / "experiments.csv"))
    ap.add_argument("--out-dir", default=str(BTL02 / "report" / "tables"))
    ap.add_argument("--sync-orphans", action="store_true", help="Quét run bị kill trong results/ trước khi tổng hợp")
    ap.add_argument("--update-log", default=None, help="Đường dẫn NHAT_KY_THUC_NGHIEM.md cần tự điền dòng Kết quả")
    a = ap.parse_args(argv)

    if a.sync_orphans:
        added = sync_orphans(BTL02)
        print(f"Đã đồng bộ orphan runs: {added}")

    written = export_tables(a.experiments, a.out_dir)
    print(f"Đã xuất {len(written)} file bảng vào {a.out_dir}")
    if a.update_log:
        n = update_experiment_log(a.update_log, a.experiments)
        print(f"Đã cập nhật {n} mục Kết quả trong {a.update_log}")


if __name__ == "__main__":
    main()
