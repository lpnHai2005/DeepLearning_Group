"""Sinh slide báo cáo tiến độ/phân công BTL02 (Slide_TienDo_BTL02.pptx) để nhập vào Google Slides.

Phụ trách: TV4 (nhóm trưởng).
Số liệu trên slide đều đọc từ tracking/*.csv hoặc quét thư mục lúc chạy; không có số gõ tay.
Muốn slide cập nhật: sửa CSV rồi chạy lại. Chạy: python tracking/build_slides.py
"""
import csv
import subprocess
from collections import Counter
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "report" / "Slide_TienDo_BTL02.pptx"
NAVY, GRAY, RED, GREEN, LIGHT = (RGBColor(0x1F, 0x4E, 0x78), RGBColor(0x55, 0x55, 0x55),
                                 RGBColor(0xC0, 0x39, 0x2B), RGBColor(0x2E, 0x7D, 0x32), RGBColor(0xEA, 0xF1, 0xF8))


def read(name):
    with open(HERE / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


tasks, exps, tests = read("tasks.csv"), read("experiments.csv"), read("test_log.csv")
figs = read("figures_manifest.csv")
NAMES = {m["ma"]: m["ho_ten"] + (" (nhóm trưởng)" if m.get("vai_tro") else "") for m in read("members.csv")}
MEMBERS = ["TV1", "TV2", "TV3", "TV4"]
status = Counter(t["trang_thai"] for t in tasks)


def n_commits():
    r = subprocess.run(["git", "log", "--oneline", "--", str(ROOT)], capture_output=True, text=True, cwd=ROOT)
    return len(r.stdout.split())


def count_files(sub, pattern):
    d = ROOT / sub
    return sum(1 for f in d.rglob(pattern) if f.is_file()) if d.exists() else 0


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, s, size=16, bold=False, color=GRAY, align=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.size, p.font.bold, p.font.color.rgb = Pt(size), bold, color
        p.space_after = Pt(6)
        if align:
            p.alignment = align
    return tb


def new(title, note=None):
    s = prs.slides.add_slide(BLANK)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.95))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    text(s, 0.5, 0.18, 12.3, 0.6, title, 26, True, RGBColor(255, 255, 255))
    if note:
        s.notes_slide.notes_text_frame.text = note
    return s


def table(slide, x, y, w, rows, widths, size=11, row_h=0.36):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))).table
    for j, cw in enumerate(widths):
        t.columns[j].width = Inches(cw)
    for i, r in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.text = str(v)
            for p in c.text_frame.paragraphs:
                p.font.size = Pt(size)
                p.font.bold = i == 0
                p.font.color.rgb = RGBColor(255, 255, 255) if i == 0 else RGBColor(0x22, 0x22, 0x22)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (LIGHT if i % 2 == 0 else RGBColor(255, 255, 255))
    return t


# 1. Tiêu đề
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()
text(s, 0.8, 2.0, 11.5, 1.6, "BTL02 – Nhận dạng cảm xúc khuôn mặt FER2013 bằng CNN (PyTorch)", 36, True, RGBColor(255, 255, 255))
text(s, 0.8, 3.7, 11.5, 1.0, "Phân công ngang/dọc, nhật ký, kiểm thử độc lập và truy vết kết quả", 22, False, RGBColor(0xDD, 0xE8, 0xF3))
text(s, 0.8, 5.2, 11.5, 1.2, ["Cập nhật 2026-11-20",
                              "Trạng thái: chưa có kết quả thực nghiệm. Slide này báo cáo kế hoạch và tiến độ thật, không có số liệu minh họa."],
     16, False, RGBColor(0xDD, 0xE8, 0xF3))

# 2. Hiện trạng
inv = [("Source code .py (src/)", count_files("src", "*.py")),
       ("Notebook .ipynb (notebooks/)", count_files("notebooks", "*.ipynb")),
       ("Thư mục results/ (run, log, weights)", count_files("results", "*")),
       ("Thư mục figures/ (hình sinh từ code)", count_files("figures", "*")),
       ("data/fer2013.csv", "có" if (ROOT / "data" / "fer2013.csv").exists() else "chưa có")]
s = new("Hiện trạng project: kiểm kê tại thời điểm báo cáo", "Số đếm do build_slides.py quét thư mục lúc chạy.")
table(s, 0.6, 1.3, 7.2, [["Hạng mục", "Số file / trạng thái"]] + [[a, b] for a, b in inv], [5.0, 2.2], 14, 0.5)
text(s, 8.2, 1.3, 4.7, 5.5, ["Đã có", "• Đề bài, báo cáo mẫu, 4 notebook mẫu Keras (chỉ tham khảo)",
                             "• Kế hoạch phân tích + thực nghiệm E1–E7",
                             "• Khung theo dõi: tracking/*.csv, Google Sheets, slide này", "",
                             "Chưa có", "• Code PyTorch, dữ liệu, log, weights, hình, kết quả",
                             f"• Commit của BTL02: {n_commits()}", "",
                             "Kết luận: mọi số liệu thực nghiệm đang là \"chưa thực hiện\"."], 15)

# 3. Hai chiều
mods = []
for t in tasks:
    if t["module"] not in mods:
        mods.append(t["module"])
rows = [["Module (chiều ngang)", "Giai đoạn", "Số nhiệm vụ", "Mã nhiệm vụ"]]
for m in mods:
    ts = [t for t in tasks if t["module"] == m]
    rows.append([m, "/".join(sorted({t["giai_doan"] for t in ts})), len(ts), ", ".join(t["task_id"] for t in ts)])
s = new("Phân công hai chiều: module (ngang) × vai trò (dọc)")
table(s, 0.5, 1.2, 12.3, rows, [4.2, 1.3, 1.5, 5.3], 12, 0.42)
text(s, 0.5, 4.8, 12.3, 2.4, ["Chiều dọc: mỗi nhiệm vụ có 1 R (chịu trách nhiệm chính), 2 C (cùng thực hiện), 1 T (kiểm thử độc lập).",
                              "Mỗi nhiệm vụ đủ 4 người khác nhau. Người làm T xoay vòng để chia đều việc kiểm thử.",
                              f"Tổng: {len(tasks)} nhiệm vụ, {len(mods)} module."], 16)


# 4-5. Ma trận
def matrix(part, subset):
    s = new(f"Ma trận phân công ({part}/2)  –  R: chính, C: phối hợp, T: kiểm thử")
    rows = [["ID", "Nhiệm vụ", "R", "C", "T", "Phụ thuộc"]]
    for t in subset:
        rows.append([t["task_id"], t["ten_nhiem_vu"], t["R_chiu_trach_nhiem"], f"{t['C1']}, {t['C2']}", t["T_kiem_thu"], t["phu_thuoc"] or "–"])
    table(s, 0.3, 1.1, 12.7, rows, [0.6, 7.0, 0.7, 1.4, 0.7, 2.3], 10, 0.4)


half = (len(tasks) + 1) // 2
matrix(1, tasks[:half])
matrix(2, tasks[half:])

# 6. Quy tắc kiểm thử + tải
rows = [["Thành viên", "Làm R", "Làm C", "Làm T"]]
for m in MEMBERS:
    rows.append([f"{m} {NAMES[m]}", sum(t["R_chiu_trach_nhiem"] == m for t in tasks),
                 sum(m in (t["C1"], t["C2"]) for t in tasks), sum(t["T_kiem_thu"] == m for t in tasks)])
s = new("Kiểm thử độc lập và phân bổ vai trò", "Phân bố đếm từ tasks.csv.")
table(s, 0.5, 1.3, 5.6, rows, [3.2, 0.8, 0.8, 0.8], 12, 0.55)
text(s, 6.5, 1.2, 6.4, 5.8, ["Quy tắc của người kiểm thử (T)",
                             "• Tự chạy trên bản clone sạch, tự tính lại số từ file gốc",
                             "• Không xác nhận dựa trên báo cáo của người làm",
                             "• Tự chọn ít nhất 1 cấu hình để chạy lại (không do R chỉ định)",
                             "• Ghi Kiem_thu: lệnh, số gốc, số tái tính, Pass/Fail",
                             "• Chỉ T mới đặt Verified, và chỉ khi Pass",
                             "• Dung sai tái lập = std 3 seed baseline (chưa có số, chờ T11)"], 15)

# 7. Chuỗi truy vết
s = new("Chuỗi truy vết từ nhiệm vụ đến báo cáo")
steps = ["Nhiệm vụ\n(task_id)", "Thành viên\n(R, C)", "Source code\n(commit)", "Log\n(run.log)", "Kết quả\n(experiments,\nhistory.csv)", "Kiểm thử\n(Kiem_thu)", "Báo cáo\nSlide"]
w, gap = 1.62, 0.17
for i, st in enumerate(steps):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.4 + i * (w + gap)), Inches(2.1), Inches(w), Inches(1.4))
    sh.fill.solid()
    sh.fill.fore_color.rgb = NAVY
    sh.line.fill.background()
    sh.text_frame.text = st
    for p in sh.text_frame.paragraphs:
        p.font.size, p.font.color.rgb, p.alignment = Pt(13), RGBColor(255, 255, 255), PP_ALIGN.CENTER
text(s, 0.5, 4.0, 12.3, 3.0, ["Khóa liên kết: task_id nối Phan_cong ↔ Kiem_thu ↔ commit git (mã nhiệm vụ trong thông điệp); exp_id nối Thuc_nghiem ↔ Hinh_ve.",
                              "Mọi hình ghi script, file dữ liệu đầu vào và lệnh chạy ở tab Hinh_ve.",
                              "Không ghi Completed khi chưa có bằng chứng; không ghi Verified khi chưa có dòng Pass."], 16)

# 8. Quy chuẩn train & log
s = new("Quy chuẩn mỗi lần train và ghi log")
text(s, 0.5, 1.2, 6.2, 5.8, ["Thư mục results/<nhóm>/<exp_id>/",
                             "• config.json: hyperparameters, seed, commit, sha256 dữ liệu, torch/GPU",
                             "• run.log: mỗi epoch 1 dòng (thời gian, epoch, train/val loss, metric, lr, trạng thái)",
                             "• history.csv: dữ liệu theo epoch, ghi ngay từng epoch",
                             "• metrics.json, curves.png + curves.csv",
                             "• best.pt chỉ khi run hợp lệ"], 15)
text(s, 7.0, 1.2, 5.9, 5.8, ["Nguyên tắc",
                             "• Seed cố định; PublicTest để chọn mô hình",
                             "• PrivateTest chỉ chạy 1 lần (TEST_DONE.json)",
                             "• Thành công kỹ thuật tách khỏi chất lượng mô hình",
                             "• Metric thấp vẫn được ghi nhận, không bị loại",
                             "• Notebook: Restart & Run All, giữ output của bản chạy làm bằng chứng"], 15)

# 9. Weights
s = new("Quản lý model weights: tự động theo luật")
text(s, 0.5, 1.2, 12.3, 5.8, ["• Cuối mỗi run, code tự quyết giữ hoặc xóa best.pt theo luật cố định, không ai điền bảng",
                              "• Giữ (KEEP, mã OK): chạy xong, loss hữu hạn, nạp lại best.pt tính lại val_acc khớp history.csv",
                              "• Xóa (DISCARD): lỗi (EXCEPTION), NaN, bị ngắt (INTERRUPTED), thiếu checkpoint, tính lại không khớp",
                              "• Luật không xét metric: run chạy đúng nhưng điểm thấp vẫn được giữ để phân tích",
                              "• Kết quả ghi vào dòng của run: đường dẫn, sha256, dung lượng, quyết định, mã lý do",
                              "• Chỉ đụng thư mục của chính run đang kết thúc, nên không thể xóa nhầm weights của baseline hay mô hình cuối",
                              "• Giới hạn: lý do là mã phân loại; nguyên nhân sâu xem traceback trong run.log",
                              f"Hiện tại: {len(exps)} run, chưa có weights nào vì chưa có code train."], 17)

# 10. Kế hoạch thực nghiệm (từ PHAN_CONG_NHIEM_VU.md mục 3.3)
plan = [("E1 Kiến trúc", "T12", 8), ("E2 Activation", "T13", 7), ("E3 Tối ưu", "T14", 12), ("E4 Augmentation & điều chuẩn", "T15", 10),
        ("E5 Mất cân bằng lớp", "T16", 3), ("E6 Transfer learning", "T17", 5), ("E7 Mô hình cuối", "T18", 4)]
by = {t["task_id"]: t for t in tasks}
rows = [["Nhóm", "Nhiệm vụ", "R", "Số run theo kế hoạch", "Đã chạy", "Kết quả"]]
for n, tid, k in plan:
    done = sum(1 for e in exps if e["task_id"] == tid)
    rows.append([n, tid, by[tid]["R_chiu_trach_nhiem"], f"~{k}", done, "chưa thực hiện" if not done else "xem Thuc_nghiem"])
s = new("Kế hoạch thực nghiệm E1–E7 (số run là ước tính của kế hoạch, chưa chạy)")
table(s, 0.5, 1.3, 12.3, rows, [4.0, 1.4, 1.0, 2.6, 1.3, 2.0], 13, 0.5)
text(s, 0.5, 5.6, 12.3, 1.5, ["Baseline: 3 seed (T11). Mỗi nhóm chỉ đổi một yếu tố so với baseline đóng băng."], 15)

# 11. Biểu đồ
s = new("Biểu đồ Train Loss / Validation Loss")
text(s, 0.5, 1.4, 12.3, 5.5, [f"Chưa có biểu đồ nào ({len(figs)} dòng trong Hinh_ve, {len(exps)} run).",
                              "Hình sẽ được sinh bởi plots.py từ history.csv của từng run (Train/Val loss theo epoch).",
                              "Mỗi hình kèm file CSV dữ liệu đã vẽ, script và lệnh chạy.",
                              "Slide này không dùng hình minh họa hay đường cong dựng sẵn."], 20)

# 12. Tiến độ & nhật ký
rows = [["Trạng thái", "Số nhiệm vụ"]] + [[k, status.get(k, 0)] for k in ["Pending", "In Progress", "Completed", "Failed", "Verified"]]
s = new("Tiến độ thực tế và nhật ký hoạt động", "Đếm từ tasks.csv; số commit đếm bằng git lúc build.")
table(s, 0.5, 1.3, 4.0, rows, [2.4, 1.6], 14, 0.5)
text(s, 5.0, 1.3, 7.9, 4.0, ["Nhật ký hoạt động không nhập tay:",
                             "• Commit git: giờ, tác giả, file sửa, mã nhiệm vụ trong thông điệp (feat(T07): ...)",
                             "• Mỗi lần train: run.log và dòng thực nghiệm do code tự ghi (ai chạy, giờ, máy, thời gian)",
                             "• Kiểm thử: người kiểm thử ghi 1 dòng Pass/Fail",
                             f"Hiện tại: {n_commits()} commit của BTL02, {len(exps)} run, {len(tests)} dòng kiểm thử, {status.get('Verified', 0)} nhiệm vụ Verified."], 15)

# 13. Việc tiếp theo
s = new("Khoảng trống và việc ưu tiên tiếp theo")
text(s, 0.5, 1.2, 12.3, 5.9, ["1. Nhóm xem lại ma trận phân công; mỗi người điền MSSV, hostname, GPU",
                              "2. Commit khung theo dõi và tài liệu",
                              "3. T02: khung repo, requirements.txt, README (bảng phân công theo file)",
                              "4. T03: tải fer2013.csv, kiểm toàn vẹn, ghi sha256",
                              "5. T04, T06, T07: loader, model, trainer có log/history/weights an toàn; đo lại ngân sách tính toán",
                              "6. T08, T10: evaluate/plots (Train/Val loss từ CSV), notebook tái lập",
                              "7. T11: baseline 3 seed, chốt dung sai tái lập",
                              "8. E1–E6, E7 và test PrivateTest 1 lần, trực quan hóa bộ lọc, báo cáo, QA, zip",
                              "",
                              "Chưa xác minh: bảng thời gian/VRAM trong kế hoạch, phân bố lớp công bố, độ chính xác của báo cáo mẫu."], 17)

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
