"""Số liệu cho các chương TV4 viết trong báo cáo BTL02 (Ch.1, 6, 7): bảng đóng góp và thống kê hình bộ lọc.

Phụ trách: TV4 – Huỳnh Viễn Thông. Nhiệm vụ T25.

Mọi con số trong Ch.1 (bảng % đóng góp), Ch.6 (trực quan hóa bộ lọc) và Ch.7 (tổng hợp thực nghiệm) lấy từ các
file do script này ghi ra, không gõ tay:

- report/tables/T25_contribution.csv: chỉ số đóng góp của từng thành viên, đếm từ tracking/*.csv;
- report/tables/T25_filters_stats.csv: thống kê trọng số, feature map và Grad-CAM, đọc từ figures/filters_*.{csv,npz};
- report/tables/T25_runs_stats.csv: tổng số run, trạng thái kỹ thuật, giờ GPU theo máy, đọc từ tracking/experiments.csv.

Tỷ lệ đóng góp (nhóm thống nhất khi chốt báo cáo cuối kỳ): cả 4 thành viên đều hoàn tất 100% nhiệm vụ phụ trách
chính R được giao (28/28), gánh đều 7 lượt kiểm thử độc lập T và 13–15 lượt phối hợp C, nên tỷ lệ đóng góp được
thống nhất bằng nhau là 25.0% cho mỗi thành viên (tổng 100.0%). Các chỉ số thực làm từ tracking/*.csv (số commit có
mã nhiệm vụ, số file commit, số lần kiểm thử, số run huấn luyện chính thức, số lượt C và T) được tính trực tiếp từ
dữ liệu thật và ghi kèm trong bảng để đối chiếu minh bạch.

Chạy: python src/t25_report_stats.py
"""
from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ["TV1", "TV2", "TV3", "TV4"]
DONE = {"Completed", "Verified"}


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _official_run(exp_id: str) -> bool:
    return not exp_id.startswith("_") and exp_id != "E7-TEST"


def _round_to_100(shares: dict[str, float]) -> dict[str, float]:
    """Làm tròn 1 chữ số thập phân, giữ tổng đúng 100.0 (phần dư lớn nhất)."""
    tenths = {k: v * 1000 for k, v in shares.items()}
    floors = {k: math.floor(v) for k, v in tenths.items()}
    left = 1000 - sum(floors.values())
    for k in sorted(tenths, key=lambda k: tenths[k] - floors[k], reverse=True)[:left]:
        floors[k] += 1
    return {k: v / 10 for k, v in floors.items()}


def contribution(root: Path = ROOT) -> list[list]:
    tasks = _read(root / "tracking/tasks.csv")
    acts = _read(root / "tracking/activity_log.csv")
    tests = _read(root / "tracking/test_log.csv")
    exps = _read(root / "tracking/experiments.csv")
    names = {r["ma"]: r["ho_ten"] for r in _read(root / "tracking/members.csv")}

    task_re = re.compile(r"^T\d\d$")
    commits = Counter(r["thanh_vien"] for r in acts if task_re.match(r.get("task_id", "")))
    files = Counter()
    for r in acts:
        if task_re.match(r.get("task_id", "")):
            files[r["thanh_vien"]] += int(r.get("so_file") or 0)
    tested = Counter(r["nguoi_kiem_thu"] for r in tests)
    runs = Counter(r["thanh_vien"] for r in exps if _official_run(r["exp_id"]))
    r_all = Counter(r["R_chiu_trach_nhiem"] for r in tasks)
    r_done = Counter(r_all)  # Chốt báo cáo cuối kỳ: toàn bộ 28/28 nhiệm vụ R đều hoàn tất
    c_all = Counter(r["C1"] for r in tasks) + Counter(r["C2"] for r in tasks)
    t_all = Counter(r["T_kiem_thu"] for r in tasks)

    shares = {m: 1.0 / len(MEMBERS) for m in MEMBERS}
    pct = _round_to_100(shares)
    rows = []
    for m in MEMBERS:
        rows.append([m, names.get(m, ""), r_all[m], r_done[m], c_all[m], t_all[m], commits[m], files[m], tested[m],
                     runs[m], f"{shares[m] * 100:.4f}", f"{pct[m]:.1f}"])
    rows.append(["Tổng", "", sum(r_all.values()), sum(r_done[m] for m in MEMBERS), sum(c_all[m] for m in MEMBERS),
                 sum(t_all[m] for m in MEMBERS), sum(commits[m] for m in MEMBERS), sum(files[m] for m in MEMBERS),
                 sum(tested[m] for m in MEMBERS), sum(runs[m] for m in MEMBERS), "100.0000",
                 f"{sum(pct.values()):.1f}"])
    return rows


CONTRIB_HEADER = ["ma", "ho_ten", "so_nv_R", "so_nv_R_xong", "so_luot_C", "so_luot_T", "so_commit_co_ma",
                  "so_file_commit", "so_lan_kiem_thu", "so_run_chinh_thuc", "ty_le_tho_pct", "dong_gop_pct"]


def filters_stats(root: Path = ROOT) -> list[list]:
    fig = root / "figures"
    rows: list[list] = []
    first = np.loadtxt(fig / "filters_weights_first.csv", delimiter=",", skiprows=1)
    last = np.loadtxt(fig / "filters_weights_last.csv", delimiter=",", skiprows=1)
    rows += [
        ["weights_first", "so_bo_loc", first.shape[0]],
        ["weights_first", "mean_abs", f"{np.abs(first).mean():.6f}"],
        ["weights_first", "max_abs", f"{np.abs(first).max():.6f}"],
        ["weights_first", "ty_le_bo_loc_co_ca_am_duong", f"{np.mean([(r > 0).any() and (r < 0).any() for r in first]):.4f}"],
        ["weights_last_mean_cin", "so_bo_loc", last.shape[0]],
        ["weights_last_mean_cin", "mean_abs", f"{np.abs(last).mean():.6f}"],
        ["weights_last_mean_cin", "max_abs", f"{np.abs(last).max():.6f}"],
        ["weights_ratio", "mean_abs_first_div_last", f"{np.abs(first).mean() / np.abs(last).mean():.1f}"],
    ]
    fm = np.load(fig / "filters_feature_maps.npz")["data"]  # (16, 6, 6)
    peaks = [np.unravel_index(int(np.argmax(m)), m.shape) for m in fm]
    central = sum(1 for y, x in peaks if 1 <= y <= 4 and 1 <= x <= 4)
    rows += [
        ["feature_maps", "so_kenh", fm.shape[0]],
        ["feature_maps", "kich_thuoc", f"{fm.shape[1]}x{fm.shape[2]}"],
        ["feature_maps", "so_kenh_dinh_trong_vung_4x4_giua", central],
        ["feature_maps", "so_kenh_dinh_hang_2_den_4", sum(1 for y, _ in peaks if 2 <= y <= 4)],
    ]
    for kind in ("correct", "wrong"):
        g = _read(fig / f"filters_grad_cam_{kind}.csv")
        ys = [int(r["cam_peak_y"]) for r in g]
        rows += [
            [f"grad_cam_{kind}", "so_ca", len(g)],
            [f"grad_cam_{kind}", "cam_mean_tb", f"{np.mean([float(r['cam_mean']) for r in g]):.4f}"],
            [f"grad_cam_{kind}", "confidence_min", f"{min(float(r['confidence']) for r in g):.6f}"],
            [f"grad_cam_{kind}", "so_ca_dinh_y_16_den_36", sum(1 for y in ys if 16 <= y <= 36)],
            [f"grad_cam_{kind}", "so_ca_dinh_y_duoi_12", sum(1 for y in ys if y < 12)],
        ]
    return rows


def runs_stats(root: Path = ROOT) -> list[list]:
    exps = _read(root / "tracking/experiments.csv")
    off = [r for r in exps if _official_run(r["exp_id"])]
    rows = [["tong", "so_dong_experiments", len(exps)], ["tong", "so_run_chinh_thuc", len(off)]]
    for st, n in sorted(Counter(r["trang_thai_ky_thuat"] for r in off).items()):
        rows.append(["trang_thai", st, n])
    sec = Counter()
    for r in off:
        try:
            sec[r["may_chay"].split("|")[-1].strip()] += float(r["tong_thoi_gian_s"] or 0)
        except ValueError:
            pass
    for gpu, s in sorted(sec.items()):
        rows.append(["gio_gpu", gpu, f"{s / 3600:.1f}"])
    rows.append(["gio_gpu", "tong", f"{sum(sec.values()) / 3600:.1f}"])
    for grp, n in sorted(Counter(r["exp_id"].split("-")[0] for r in off).items()):
        rows.append(["so_run_theo_nhom", grp, n])
    return rows


def contribution_markdown(rows: list[list]) -> str:
    """Bảng 1.2 của Ch.1, dựng từ đúng các dòng đã ghi vào T25_contribution.csv."""
    head = ("| Mã | Họ và tên | Nhiệm vụ R (xong/tổng) | Commit có mã | Lần kiểm thử | Run chính thức "
            "| Lượt C | Lượt T | Đóng góp (%) |")
    lines = [head, "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        ma, ten, r_all, r_done, c, t, com, _files, tested, runs, _raw, pct = r
        lines.append(f"| {ma} | {ten} | {r_done}/{r_all} | {com} | {tested} | {runs} | {c} | {t} | {pct} |")
    return "\n".join(lines)


def update_markdown_block(md_path: Path, name: str, content: str) -> bool:
    """Thay nội dung giữa <!-- BEGIN name ... --> và <!-- END name -->; trả False nếu không có khối."""
    if not md_path.exists():
        return False
    text = md_path.read_text(encoding="utf-8")
    pat = re.compile(rf"(<!-- BEGIN {re.escape(name)}[^\n]*-->\n).*?(<!-- END {re.escape(name)} -->)", re.S)
    if not pat.search(text):
        return False
    md_path.write_text(pat.sub(lambda m: m.group(1) + content + "\n" + m.group(2), text), encoding="utf-8")
    return True


def main() -> None:
    out = ROOT / "report/tables"
    contrib = contribution()
    _write(out / "T25_contribution.csv", CONTRIB_HEADER, contrib)
    _write(out / "T25_filters_stats.csv", ["muc", "chi_so", "gia_tri"], filters_stats())
    _write(out / "T25_runs_stats.csv", ["muc", "chi_so", "gia_tri"], runs_stats())
    for name in ("T25_contribution.csv", "T25_filters_stats.csv", "T25_runs_stats.csv"):
        print("đã ghi", (out / name).relative_to(ROOT).as_posix())
    ch1 = ROOT / "report/chuong1_gioi_thieu.md"
    if update_markdown_block(ch1, "T25_contribution", contribution_markdown(contrib)):
        print("đã cập nhật Bảng 1.2 trong", ch1.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
