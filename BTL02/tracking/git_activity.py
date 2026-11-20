"""Sinh nhật ký hoạt động từ lịch sử git, không ai phải nhập tay.

Phụ trách: TV2 – Lê Phan Nguyên Hải (nhóm trưởng).
Mỗi commit thành một dòng: giờ, tác giả, thành viên (đối chiếu cột git_user của members.csv; một người có nhiều tên git thì ngăn cách bằng dấu ;),
mã nhiệm vụ (lấy từ thông điệp commit dạng `feat(T07): ...`), số file, danh sách file.
Chạy: python tracking/git_activity.py [đường_dẫn_trong_repo] [file_ra.csv]
Mặc định đọc thư mục BTL02 và ghi tracking/activity_log.csv (file sinh ra, không commit).
"""
import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SEP = "\x1e"


def main(path=ROOT, out=HERE / "activity_log.csv"):
    with open(HERE / "members.csv", encoding="utf-8-sig") as f:
        git_user = {u.strip().lower(): r["ma"] for r in csv.DictReader(f) for u in (r.get("git_user") or "").split(";") if u.strip()}
    raw = subprocess.run(["git", "log", "--name-only", f"--format={SEP}%H\x1f%an\x1f%aI\x1f%s", "--", str(path)],
                         capture_output=True, text=True, encoding="utf-8", cwd=ROOT, check=True).stdout
    rows = []
    for rec in filter(None, raw.split(SEP)):
        head, *files = rec.strip("\n").split("\n")
        h, author, when, subject = head.split("\x1f")
        files = [x for x in files if x]
        task = re.search(r"\bT\d{2}\b", subject)
        rows.append([h[:8], when, author, git_user.get(author.lower(), "(chưa gán trong members.csv)"),
                     task.group(0) if task else "(không có mã nhiệm vụ)", subject, len(files), "; ".join(files)])
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["commit", "thoi_gian", "tac_gia_git", "thanh_vien", "task_id", "thong_diep", "so_file", "files"])
        w.writerows(rows)
    print(f"{len(rows)} commit -> {out}")
    for k, v in Counter(r[3] for r in rows).most_common():
        print(f"  {k}: {v} commit")
    return rows


if __name__ == "__main__":
    a = sys.argv
    main(Path(a[1]) if len(a) > 1 else ROOT, Path(a[2]) if len(a) > 2 else HERE / "activity_log.csv")
