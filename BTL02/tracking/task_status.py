"""Tự suy cột trang_thai và ngay_cap_nhat của tasks.csv từ git và test_log.csv, không ai phải gõ trạng thái.

Phụ trách: TV2 – Lê Phan Nguyên Hải (nhóm trưởng).

Luật (xét riêng từng nhiệm vụ, theo sự kiện mới nhất):
- Verified / Failed: dòng kiểm thử mới nhất của nhiệm vụ trong test_log.csv có ket_qua Pass / Fail,
  và không có commit `done(Txx)` nào sau dòng đó.
- Completed: R báo xong bằng commit `done(Txx): ...` (được phép là commit rỗng: git commit --allow-empty),
  sau lần kiểm thử gần nhất nếu có. Nghĩa là đang chờ T kiểm thử (lần đầu hoặc kiểm lại sau Fail).
- In Progress: đã có ít nhất một commit mang mã Txx trong thông điệp.
- Pending: chưa có gì.
ngay_cap_nhat = ngày (giờ Việt Nam) của sự kiện mới nhất liên quan đến nhiệm vụ; không có sự kiện thì giữ nguyên.

Chạy: python tracking/task_status.py   (workflow .github/workflows/activity-log.yml tự chạy sau mỗi push)
"""
import csv
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
VN = timezone(timedelta(hours=7))
TASK_RE = re.compile(r"\bT\d{2}\b")
DONE_RE = re.compile(r"^done\(([^)]*)\)")


def _t(iso):
    return datetime.fromisoformat(iso)


def read_commits():
    """Danh sách (thời điểm, thông điệp) của mọi commit trong repo, bỏ commit của bot.

    Không lọc theo thư mục: commit rỗng `done(Txx)` không chạm file nào, lọc `-- BTL02` sẽ bỏ sót nó.
    """
    raw = subprocess.run(["git", "log", "--format=%aI\x1f%an\x1f%s"], capture_output=True,
                         text=True, encoding="utf-8", cwd=ROOT, check=True).stdout
    out = []
    for line in filter(None, raw.splitlines()):
        when, author, subject = line.split("\x1f", 2)
        if author != "github-actions[bot]":
            out.append((_t(when), subject))
    return out


def derive(task_ids, commits, tests):
    """Trả {task_id: (trang_thai, thời điểm sự kiện mới nhất hoặc None)}.

    commits: [(datetime, subject)], tests: [dict có task_id, thoi_gian, ket_qua] (dòng của test_log.csv).
    """
    touched, done, last_test = {}, {}, {}
    for when, subject in commits:
        m = DONE_RE.match(subject)
        for tid in TASK_RE.findall(subject):
            touched[tid] = max(touched.get(tid, when), when)
            if m and tid in TASK_RE.findall(m.group(1)):
                done[tid] = max(done.get(tid, when), when)
    for r in tests:
        if r.get("ket_qua") not in ("Pass", "Fail") or not r.get("thoi_gian"):
            continue
        when = _t(r["thoi_gian"])
        if r["task_id"] not in last_test or when >= last_test[r["task_id"]][0]:
            last_test[r["task_id"]] = (when, r["ket_qua"])

    result = {}
    for tid in task_ids:
        events = [x for x in (touched.get(tid), done.get(tid), last_test.get(tid, (None,))[0]) if x is not None]
        latest = max(events) if events else None
        if tid in last_test and (tid not in done or last_test[tid][0] >= done[tid]):
            status = "Verified" if last_test[tid][1] == "Pass" else "Failed"
        elif tid in done:
            status = "Completed"
        elif tid in touched:
            status = "In Progress"
        else:
            status = "Pending"
        result[tid] = (status, latest)
    return result


def main(tasks_csv=HERE / "tasks.csv", test_csv=HERE / "test_log.csv"):
    with open(tasks_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    with open(test_csv, encoding="utf-8", newline="") as f:
        tests = list(csv.DictReader(f))
    status = derive([r["task_id"] for r in rows], read_commits(), tests)
    changed = []
    for r in rows:
        s, latest = status[r["task_id"]]
        day = latest.astimezone(VN).date().isoformat() if latest else r["ngay_cap_nhat"]
        if (r["trang_thai"], r["ngay_cap_nhat"]) != (s, day):
            changed.append(f"{r['task_id']}: {r['trang_thai']} -> {s}")
        r["trang_thai"], r["ngay_cap_nhat"] = s, day
    with open(tasks_csv, "w", encoding="utf-8", newline="") as f:  # không BOM để IMPORTDATA của Sheets đọc đúng
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(changed)} nhiệm vụ đổi trạng thái" + "".join(f"\n  {c}" for c in changed))
    return status


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252, print tiếng Việt sẽ lỗi
    main()
