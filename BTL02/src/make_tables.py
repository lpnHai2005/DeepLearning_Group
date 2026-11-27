"""Tổng hợp bảng kết quả thực nghiệm E1–E7 từ tracking/experiments.csv (mean ± std, so sánh baseline),
xuất bảng tra cứu (CSV/MD), bảng báo cáo gọn (MD + Word .docx) và điền mục Kết quả trong
report/NHAT_KY_THUC_NGHIEM.md.

Phụ trách: TV3 – Võ Hoàng Bảo. Nhiệm vụ T09.

Dùng (từ thư mục BTL02):
    python src/make_tables.py [--sync-orphans] [--include-internal] [--update-log report/NHAT_KY_THUC_NGHIEM.md]
"""
import argparse
import csv
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

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
    "n_params",
    "val_acc_mean",
    "val_acc_std",
    "val_acc_fmt",
    "val_macro_f1_mean",
    "val_macro_f1_std",
    "val_macro_f1_fmt",
    "best_val_loss_mean",
    "train_val_gap_mean",
    "gpu",
    "epoch_time_s_tb_mean",
    "so_voi_baseline_tu_tinh",
]

REPORT_COLS = [
    ("variant_display", "Cấu hình"),
    ("n_completed", "Seed"),
    ("val_acc_fmt", "Val acc (mean ± std)"),
    ("val_macro_f1_fmt", "Val macro-F1"),
    ("best_val_loss_fmt", "Best val loss"),
    ("n_params_fmt", "Số tham số"),
    ("train_val_gap_fmt", "Train–Val*"),
    ("gpu_time_fmt", "GPU (s/ep)"),
    ("so_voi_baseline_tu_tinh", "So với baseline"),
]

GROUP_ORDER = ["baseline", "B0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]

TABLE_CAPTIONS = {
    "baseline": "Bảng 4.0: Kết quả mô hình Baseline đóng băng (3 seed, 100 epoch)",
    "B0": "Bảng 4.0: Kết quả mô hình Baseline đóng băng",
    "E1": "Bảng 4.1: Kết quả thực nghiệm E1 – Khảo sát kiến trúc CNN",
    "E2": "Bảng 4.2: Kết quả thực nghiệm E2 – Khảo sát hàm kích hoạt",
    "E3": "Bảng 4.3: Kết quả thực nghiệm E3 – Khảo sát thuật toán tối ưu, lịch learning rate và batch size",
    "E4": "Bảng 4.4: Kết quả thực nghiệm E4 – Khảo sát data augmentation và điều chuẩn",
    "E5": "Bảng 4.5: Kết quả thực nghiệm E5 – Xử lý mất cân bằng lớp",
    "E6": "Bảng 4.6: Kết quả thực nghiệm E6 – Transfer learning (VGG16 và ResNet18)",
    "E7": "Bảng 4.7: Kết quả thực nghiệm E7 – Mô hình cuối, TTA và ensemble",
    "summary_all": "Bảng 4.8: Tổng hợp toàn bộ kết quả thực nghiệm (Baseline và E1–E7)",
}

REPORT_FOOTNOTE = (
    "* Ghi chú: `Train–Val` = `train_acc_cuoi − val_acc`; `train_acc_cuoi` được đo trong chế độ "
    "huấn luyện (có bật data augmentation và dropout), chỉ dùng để so sánh xu hướng quá khớp giữa "
    "các cấu hình. Ký hiệu `—` thể hiện không xác định (1 seed không có độ lệch chuẩn mẫu, hoặc "
    "thời gian không lấy trung bình qua nhiều GPU khác nhau)."
)


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


def _series_of(row):
    exp_id = (row.get("exp_id") or "").strip()
    return re.sub(r"_s\d+$", "", exp_id) if exp_id else ""


def _parse_hp(row):
    hp = (row.get("hyperparams_json") or "").strip()
    if not hp:
        return {}
    try:
        obj = json.loads(hp)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _variant_of(row):
    obj = _parse_hp(row)
    if obj.get("variant"):
        return str(obj["variant"])
    exp_id = (row.get("exp_id") or "").strip()
    return re.sub(r"_s\d+$", "", exp_id)


def _float_or_none(val):
    s = str(val if val is not None else "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _int_or_none(val):
    v = _float_or_none(val)
    return int(round(v)) if v is not None else None


def _short_gpu(may_chay):
    raw = (may_chay or "").strip()
    if not raw:
        return ""
    gpu = raw.split("|", 1)[1].strip() if "|" in raw else raw
    gpu = re.sub(r"^NVIDIA\s+GeForce\s+", "", gpu, flags=re.I)
    gpu = re.sub(r"\s+Laptop\s+GPU$", "", gpu, flags=re.I)
    gpu = re.sub(r"\s+GPU$", "", gpu, flags=re.I)
    return gpu.strip()


def _load_baseline_cfg(baseline_config=None):
    cfg_path = Path(baseline_config) if baseline_config is not None else (BTL02 / "configs" / "baseline.json")
    if not cfg_path.exists():
        return None
    try:
        spec = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(spec, dict) and isinstance(spec.get("base"), dict):
            return spec["base"]
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _official_baseline_rows(rows, baseline_task="T11", baseline_config=None):
    """Trả danh sách dòng run thuộc bộ baseline chính thức (khớp configs/baseline.json)."""
    base_cfg = _load_baseline_cfg(baseline_config)
    if base_cfg:
        matched_by_num = defaultdict(list)
        for r in rows:
            if r.get("trang_thai_ky_thuat") != "Completed":
                continue
            exp_id = (r.get("exp_id") or "").strip()
            m = re.match(r"^baseline-(\d{3})_s(\d+)$", exp_id)
            if not m or r.get("task_id") != baseline_task:
                continue
            hp = _parse_hp(r)
            if all(hp.get(k) == v for k, v in base_cfg.items()):
                matched_by_num[int(m.group(1))].append(r)
        if matched_by_num:
            return matched_by_num[max(matched_by_num)]
    return []


def filter_report_rows(rows, baseline_task="T11", baseline_config=None, include_internal=False):
    """Lọc dòng đưa vào bảng báo cáo validation: bỏ nhóm tạm `_...`, dòng đánh giá PrivateTest (`T19` / `E7-TEST`) và run baseline cấu hình cũ (60 epoch)."""
    official = _official_baseline_rows(rows, baseline_task=baseline_task, baseline_config=baseline_config)
    official_ids = {r.get("exp_id") for r in official}

    kept = []
    for r in rows:
        grp = _group_of(r)
        if not include_internal and grp.startswith("_"):
            continue
        exp_id = (r.get("exp_id") or "").strip()
        tid = (r.get("task_id") or "").strip()
        # Bỏ dòng đánh giá PrivateTest (T19 / E7-TEST) khỏi bảng tổng hợp validation E1–E7 (QA-03)
        if tid == "T19" or exp_id.upper() == "E7-TEST" or (
            _float_or_none(r.get("test_acc")) is not None and _float_or_none(r.get("val_acc")) is None
        ):
            continue
        if official_ids and (grp.lower() in ("baseline", "b0") or tid == baseline_task):
            if exp_id not in official_ids:
                continue
        kept.append(r)
    return kept


def baseline_stats(rows, baseline_task="T11", baseline_config=None):
    """Tính (mean, std) val_acc của bộ run baseline chính thức đã Completed.

    Đồng bộ luật với src/baseline_summary.py: nếu có file configs/baseline.json và tồn tại
    các run `baseline-NNN_sK` khớp toàn bộ khóa trong `base` của config, chỉ tính trên bộ đó
    (loại run 60 epoch cũ và run `_probe`). Khi chạy trên fixture tối giản trong unit test,
    tự động fallback về các dòng Completed thuộc `baseline_task` hoặc group `baseline`/`b0`
    (luôn loại nhóm tạm bắt đầu bằng `_`).
    """
    official = _official_baseline_rows(rows, baseline_task=baseline_task, baseline_config=baseline_config)
    if official:
        vals = [v for r in official if (v := _float_or_none(r.get("val_acc"))) is not None]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            std = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
            return float(arr.mean()), std

    vals = []
    for r in rows:
        if r.get("trang_thai_ky_thuat") != "Completed":
            continue
        grp = _group_of(r).lower()
        if grp.startswith("_"):
            continue
        if r.get("task_id") == baseline_task or grp in ("baseline", "b0"):
            v = _float_or_none(r.get("val_acc"))
            if v is not None:
                vals.append(v)
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) >= 2 else 0.0
    return float(arr.mean()), std


def compare_to_baseline(val_acc, base_mean, base_std, is_baseline=False):
    """So sánh thuần số học với baseline; chính dòng baseline ghi 'tham chiếu'."""
    if is_baseline:
        return "tham chiếu"
    if val_acc is None or base_mean is None:
        return ""
    diff = val_acc - base_mean
    if base_std is not None and base_std > 0 and abs(diff) < base_std:
        return f"{diff:+.4f} (< std baseline)"
    tag = "cao hơn" if diff > 0 else ("thấp hơn" if diff < 0 else "bằng")
    return f"{diff:+.4f} ({tag})"


def _fmt_mean_std(vals):
    """Trả (mean, std_or_None, fmt_4_decimals). 1 mẫu -> std=None (không in 0.0)."""
    if not vals:
        return None, None, ""
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) >= 2:
        std = float(arr.std(ddof=1))
        return round(mean, 6), round(std, 6), f"{mean:.4f} ± {std:.4f}"
    return round(mean, 6), None, f"{mean:.4f}"


def _variant_order_map(rows, root=BTL02):
    """Đọc thứ tự `variants` từ file config JSON tương ứng của từng nhóm để giữ đúng thứ tự khoa học."""
    order = {}
    root = Path(root)
    config_paths = set()
    for r in rows:
        cp = (r.get("config_path") or "").strip()
        if cp.endswith(".json"):
            config_paths.add(cp)
    for cfg_file in (root / "configs").glob("*.json") if (root / "configs").is_dir() else []:
        config_paths.add(str(cfg_file))

    for cp in sorted(config_paths):
        p = Path(cp)
        if not p.is_absolute():
            p = root / cp
        if not p.is_file():
            continue
        try:
            spec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        grp = spec.get("group")
        variants = spec.get("variants")
        if grp and isinstance(variants, dict):
            order.setdefault(grp, {})
            for idx, vname in enumerate(variants.keys()):
                order[grp].setdefault(vname, idx)
    return order


def _group_sort_key(grp):
    if grp in GROUP_ORDER:
        return (0, GROUP_ORDER.index(grp), grp)
    if grp.startswith("_"):
        return (2, 0, grp)
    return (1, 0, grp)


def _config_sig(r):
    """Chữ ký cấu hình cốt lõi (bỏ qua seed/exp_id và cờ hệ thống) để chỉ tách series khi thực sự đổi siêu tham số."""
    hp = _parse_hp(r)
    if not hp:
        return ""
    ignore = {"seed", "exp_id", "num_workers", "amp", "log_grad_norms"}
    core = {k: v for k, v in hp.items() if k not in ignore}
    return json.dumps(core, sort_keys=True, ensure_ascii=False)


def summarize_rows(rows, base_mean=None, base_std=None, root=BTL02, baseline_task="T11"):
    """Nhóm theo (group, task_id, variant) và tính các chỉ số theo thứ tự config.

    Chỉ tách theo series khi các run Completed của cùng (group, task_id, variant) có cấu hình
    siêu tham số cốt lõi khác nhau (ví dụ 60 epoch vs 100 epoch); các run chạy lại do Aborted/Failed
    (cùng biến thể, cùng siêu tham số, khác mã số run như E1-009_s1 vs E1-001_s0/s2) được gộp chung.
    """
    sigs_per_key = defaultdict(set)
    for r in rows:
        if r.get("trang_thai_ky_thuat") != "Completed":
            continue
        key = (_group_of(r), r.get("task_id", ""), _variant_of(r))
        sig = _config_sig(r)
        if sig:
            sigs_per_key[key].add(sig)

    buckets = defaultdict(list)
    raw_var_for_bucket = {}
    for r in rows:
        grp = _group_of(r)
        tid = r.get("task_id", "")
        var = _variant_of(r)
        ser = _series_of(r)
        split_series = len(sigs_per_key[(grp, tid, var)]) > 1 and bool(ser)
        label_var = f"{var} [{ser}]" if split_series else var
        bkey = (grp, tid, label_var)
        buckets[bkey].append(r)
        raw_var_for_bucket[bkey] = var

    order_map = _variant_order_map(rows, root=root)

    def _bucket_sort(bkey):
        grp, tid, label_var = bkey
        raw_v = raw_var_for_bucket[bkey]
        v_idx = order_map.get(grp, {}).get(raw_v, 10_000)
        return (_group_sort_key(grp), v_idx, label_var, tid)

    summary = []
    for bkey in sorted(buckets.keys(), key=_bucket_sort):
        grp, tid, var = bkey
        items = buckets[bkey]
        completed = [x for x in items if x.get("trang_thai_ky_thuat") == "Completed"]
        accs = [v for x in completed if (v := _float_or_none(x.get("val_acc"))) is not None]
        f1s = [v for x in completed if (v := _float_or_none(x.get("val_macro_f1"))) is not None]
        losses = [v for x in completed if (v := _float_or_none(x.get("best_val_loss"))) is not None]
        times = [v for x in completed if (v := _float_or_none(x.get("epoch_time_s_tb"))) is not None]
        params_vals = [v for x in completed if (v := _int_or_none(x.get("n_params"))) is not None]
        gaps = []
        for x in completed:
            ta = _float_or_none(x.get("train_acc_cuoi"))
            va = _float_or_none(x.get("val_acc"))
            if ta is not None and va is not None:
                gaps.append(ta - va)

        gpus = sorted({g for x in completed if (g := _short_gpu(x.get("may_chay")))})
        if len(gpus) == 1:
            gpu_label = gpus[0]
            same_gpu = True
        elif len(gpus) > 1:
            gpu_label = f"nhiều GPU ({'/'.join(gpus)})"
            same_gpu = False
        else:
            gpu_label = ""
            same_gpu = True

        acc_m, acc_s, acc_fmt = _fmt_mean_std(accs)
        f1_m, f1_s, f1_fmt = _fmt_mean_std(f1s)
        loss_m, _, _ = _fmt_mean_std(losses)
        gap_m, _, _ = _fmt_mean_std(gaps)
        time_m = round(float(np.mean(times)), 1) if (times and same_gpu) else None
        n_par = params_vals[0] if params_vals else None

        is_base = grp.lower() in ("baseline", "b0") and (tid == baseline_task or not tid)
        summary.append(
            {
                "group": grp,
                "task_id": tid,
                "variant": var,
                "n_runs": len(items),
                "n_completed": len(completed),
                "n_params": "" if n_par is None else n_par,
                "val_acc_mean": "" if acc_m is None else acc_m,
                "val_acc_std": "" if acc_s is None else acc_s,
                "val_acc_fmt": acc_fmt,
                "val_macro_f1_mean": "" if f1_m is None else f1_m,
                "val_macro_f1_std": "" if f1_s is None else f1_s,
                "val_macro_f1_fmt": f1_fmt,
                "best_val_loss_mean": "" if loss_m is None else round(loss_m, 4),
                "train_val_gap_mean": "" if gap_m is None else round(gap_m, 4),
                "gpu": gpu_label,
                "epoch_time_s_tb_mean": "" if time_m is None else time_m,
                "so_voi_baseline_tu_tinh": compare_to_baseline(acc_m, base_mean, base_std, is_baseline=is_base),
            }
        )
    return summary


def _to_report_row(r, variant_override=None):
    """Chuyển 1 dòng summary thành dòng báo cáo gọn tiếng Việt với số làm tròn chuẩn."""
    loss_v = _float_or_none(r.get("best_val_loss_mean"))
    gap_v = _float_or_none(r.get("train_val_gap_mean"))
    n_par = _int_or_none(r.get("n_params"))
    time_v = _float_or_none(r.get("epoch_time_s_tb_mean"))
    gpu = (r.get("gpu") or "").strip()

    if gpu.startswith("nhiều GPU"):
        gpu_time = "— (nhiều GPU)"
    elif gpu and time_v is not None:
        gpu_time = f"{gpu} ({time_v:.1f}s)"
    elif time_v is not None:
        gpu_time = f"{time_v:.1f}s"
    elif gpu:
        gpu_time = gpu
    else:
        gpu_time = "—"

    return {
        "variant_display": variant_override or r.get("variant", ""),
        "n_completed": str(r.get("n_completed", 0)),
        "val_acc_fmt": r.get("val_acc_fmt") or "—",
        "val_macro_f1_fmt": r.get("val_macro_f1_fmt") or "—",
        "best_val_loss_fmt": f"{loss_v:.4f}" if loss_v is not None else "—",
        "n_params_fmt": f"{n_par:,}" if n_par is not None else "—",
        "train_val_gap_fmt": f"{gap_v:+.4f}" if gap_v is not None else "—",
        "gpu_time_fmt": gpu_time,
        "so_voi_baseline_tu_tinh": r.get("so_voi_baseline_tu_tinh") or "—",
    }


def _write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _write_markdown(path, cols, rows, title, footnote=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_Chưa có dữ liệu thực nghiệm._")
    else:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    if footnote:
        lines.extend(["", footnote])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_markdown(path, report_rows, caption, footnote=REPORT_FOOTNOTE):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k, _ in REPORT_COLS]
    headers = [h for _, h in REPORT_COLS]
    lines = [f"**{caption}**", ""]
    if not report_rows:
        lines.append("_Chưa có dữ liệu thực nghiệm._")
    else:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for r in report_rows:
            lines.append("| " + " | ".join(str(r.get(k, "—")) for k in keys) + " |")
    if footnote:
        lines.extend(["", f"_{footnote}_"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _docx_cell(text, bold=False, shaded=False):
    shd = '<w:shd w:val="clear" w:color="auto" w:fill="E8EEF5"/>' if shaded else ""
    rpr = "<w:rPr><w:b/><w:sz w:val=\"18\"/></w:rPr>" if bold else "<w:rPr><w:sz w:val=\"18\"/></w:rPr>"
    return (
        f"<w:tc><w:tcPr>{shd}</w:tcPr>"
        f"<w:p><w:r>{rpr}<w:t xml:space=\"preserve\">{xml_escape(str(text))}</w:t></w:r></w:p></w:tc>"
    )


def _docx_paragraph(text, bold=False, italic=False, size=20):
    tags = []
    if bold:
        tags.append("<w:b/>")
    if italic:
        tags.append("<w:i/>")
    tags.append(f'<w:sz w:val="{size}"/>')
    return (
        f"<w:p><w:r><w:rPr>{''.join(tags)}</w:rPr>"
        f'<w:t xml:space="preserve">{xml_escape(str(text))}</w:t></w:r></w:p>'
    )


def export_docx(tables, out_docx, footnote=REPORT_FOOTNOTE):
    """Xuất các bảng báo cáo sang file Word (.docx) chuẩn OpenXML bằng thư viện chuẩn Python."""
    out_docx = Path(out_docx)
    out_docx.parent.mkdir(parents=True, exist_ok=True)
    keys = [k for k, _ in REPORT_COLS]
    headers = [h for _, h in REPORT_COLS]

    body_parts = [_docx_paragraph("BTL02 – Tổng hợp bảng kết quả thực nghiệm FER2013", bold=True, size=28)]
    tbl_pr = (
        "<w:tblPr>"
        '<w:tblW w:w="0" w:type="auto"/>'
        "<w:tblBorders>"
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="888888"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="888888"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '<w:left w:val="none"/><w:right w:val="none"/><w:insideV w:val="none"/>'
        "</w:tblBorders>"
        "</w:tblPr>"
    )

    for caption, report_rows in tables:
        body_parts.append(_docx_paragraph(caption, bold=True, size=22))
        if not report_rows:
            body_parts.append(_docx_paragraph("Chưa có dữ liệu thực nghiệm.", italic=True, size=18))
            continue
        rows_xml = ["<w:tr>" + "".join(_docx_cell(h, bold=True, shaded=True) for h in headers) + "</w:tr>"]
        for r in report_rows:
            rows_xml.append("<w:tr>" + "".join(_docx_cell(r.get(k, "—")) for k in keys) + "</w:tr>")
        body_parts.append(f"<w:tbl>{tbl_pr}{''.join(rows_xml)}</w:tbl>")
        body_parts.append(_docx_paragraph("", size=12))

    if footnote:
        body_parts.append(_docx_paragraph(footnote, italic=True, size=17))

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body_parts)}"
        '<w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/></w:sectPr>'
        "</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(out_docx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
    return out_docx


def export_tables(experiments_csv, out_dir, include_internal=False, baseline_config=None, root=BTL02):
    """Sinh bảng tra cứu (CSV/MD), bảng báo cáo gọn (MD) có baseline tham chiếu và Word (.docx)."""
    experiments_csv = Path(experiments_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_experiments(experiments_csv)
    base_mean, base_std = baseline_stats(all_rows, baseline_config=baseline_config)
    rows = filter_report_rows(all_rows, baseline_config=baseline_config, include_internal=include_internal)
    summary = summarize_rows(rows, base_mean=base_mean, base_std=base_std, root=root)

    # Xác định dòng baseline tham chiếu để chèn đầu các bảng E1-E7
    base_summary_row = next(
        (r for r in summary if r["group"].lower() in ("baseline", "b0") and r["n_completed"] > 0),
        None,
    )

    written = []
    # 1. Bảng tra cứu thô tổng hợp (CSV + MD)
    all_csv = out_dir / "summary_all.csv"
    all_md = out_dir / "summary_all.md"
    _write_csv(all_csv, SUMMARY_COLS, summary)
    _write_markdown(all_md, SUMMARY_COLS, summary, TABLE_CAPTIONS["summary_all"], footnote=REPORT_FOOTNOTE)
    written.extend([all_csv, all_md])

    # 2. Bảng báo cáo gọn tổng hợp (MD) — chỉ đưa các biến thể có ít nhất 1 run Completed
    all_report_rows = [
        _to_report_row(
            r,
            variant_override="Baseline (tham chiếu)"
            if r["group"].lower() in ("baseline", "b0")
            else f"{r['group']} – {r['variant']}",
        )
        for r in summary
        if r["n_completed"] > 0
    ]
    all_rep_md = out_dir / "summary_all_report.md"
    _write_report_markdown(all_rep_md, all_report_rows, TABLE_CAPTIONS["summary_all"])
    written.append(all_rep_md)

    by_group = defaultdict(list)
    for r in summary:
        by_group[r["group"]].append(r)

    docx_tables = []
    for grp in sorted(by_group.keys(), key=_group_sort_key):
        grp_rows = by_group[grp]
        caption = TABLE_CAPTIONS.get(grp, f"Bảng tổng hợp nhóm {grp}")

        g_csv = out_dir / f"{grp}_summary.csv"
        g_md = out_dir / f"{grp}_summary.md"
        _write_csv(g_csv, SUMMARY_COLS, grp_rows)
        _write_markdown(g_md, SUMMARY_COLS, grp_rows, caption, footnote=REPORT_FOOTNOTE)
        written.extend([g_csv, g_md])

        # Bảng báo cáo gọn cho nhóm: chèn dòng Baseline tham chiếu ở đầu các bảng E1-E7, bỏ dòng n_completed=0
        rep_rows = []
        if re.match(r"^E[1-7]$", grp) and base_summary_row is not None:
            rep_rows.append(_to_report_row(base_summary_row, variant_override="Baseline (tham chiếu)"))
        for r in grp_rows:
            if r["n_completed"] <= 0:
                continue
            v_over = "Baseline (tham chiếu)" if grp.lower() in ("baseline", "b0") else None
            rep_rows.append(_to_report_row(r, variant_override=v_over))

        g_rep_md = out_dir / f"{grp}_report.md"
        _write_report_markdown(g_rep_md, rep_rows, caption)
        written.append(g_rep_md)
        docx_tables.append((caption, rep_rows))

    docx_tables.append((TABLE_CAPTIONS["summary_all"], all_report_rows))
    docx_path = export_docx(docx_tables, out_dir / "BTL02_tables_report.docx")
    written.append(docx_path)

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
    ap.add_argument("--include-internal", action="store_true", help="Gồm cả nhóm tạm bắt đầu bằng '_' (_qa, _probe, _nb)")
    ap.add_argument("--update-log", default=None, help="Đường dẫn NHAT_KY_THUC_NGHIEM.md cần tự điền dòng Kết quả")
    a = ap.parse_args(argv)

    if a.sync_orphans:
        added = sync_orphans(BTL02)
        print(f"Đã đồng bộ orphan runs: {added}")

    written = export_tables(a.experiments, a.out_dir, include_internal=a.include_internal)
    print(f"Đã xuất {len(written)} file bảng vào {a.out_dir}")
    if a.update_log:
        n = update_experiment_log(a.update_log, a.experiments)
        print(f"Đã cập nhật {n} mục Kết quả trong {a.update_log}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows (cp1252) không in được tiếng Việt
    main()
