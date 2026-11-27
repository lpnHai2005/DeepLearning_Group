# Phụ trách: TV1 – Đoàn Trung Kiên
"""Sinh 3 slide phần TV1 cho T26: dữ liệu FER2013, augmentation/điều chuẩn (E4), mất cân bằng lớp (E5).

Phụ trách: TV1 – Đoàn Trung Kiên (C1 của T26; R là TV4, người ghép vào Slide_BTL02).
Mọi số trên slide và trong speaker notes đều đọc từ CSV/JSON trong repo lúc chạy; không có số gõ tay.
Script đồng thời sinh bảng dò nguồn TV1_slide_nguon.md: mỗi số hiển thị → file nguồn → dòng/cột.

Chạy từ thư mục BTL02:  python report/slides/build_slides_tv1.py
Kết quả:  report/slides/Slide_BTL02_TV1.pptx, report/slides/TV1_slide_nguon.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # BTL02/
OUT_PPTX = HERE / "Slide_BTL02_TV1.pptx"
OUT_TRACE = HERE / "TV1_slide_nguon.md"

# Bảng màu: cùng tông navy với Slide_TienDo_BTL02 (tracking/build_slides.py) để TV4 ghép không lệch.
NAVY = RGBColor(0x1F, 0x4E, 0x78)
INK = RGBColor(0x22, 0x2B, 0x36)
GRAY = RGBColor(0x5F, 0x6B, 0x78)
MUTED = RGBColor(0xA9, 0xBC, 0xCF)
TINT = RGBColor(0xEE, 0xF3, 0xF8)
ACCENT = RGBColor(0xC2, 0x41, 0x0C)  # cam đất: chỉ dùng để nhấn lớp hiếm / biến thể đang nói tới
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"


# ---------------------------------------------------------------- đọc nguồn
def read_csv(rel):
    with open(ROOT / rel, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


SRC_CHECK = "results/data_check.json"
SRC_DIST = "figures/eda_class_distribution.csv"
SRC_E4 = "report/tables/E4_summary.csv"
SRC_GAP = "figures/e4_overfit_summary.csv"
SRC_E5 = "report/tables/E5_summary.csv"
SRC_RECALL = "results/E5/E5-001_s0/per_class_recall.csv"
FIG_AUG = "figures/eda_augmentation.png"

check = read_json(SRC_CHECK)
dist = read_csv(SRC_DIST)
e4 = {r["variant"]: r for r in read_csv(SRC_E4)}
gap = {r["variant"]: r for r in read_csv(SRC_GAP)}
e5 = {r["variant"]: r for r in read_csv(SRC_E5)}
recall = read_csv(SRC_RECALL)
for _r in recall:  # file lưu recall dạng tỷ lệ 0–1, số đúng ở cột tp; đổi sang % một lần ở đây
    _r["recall"] = float(_r["recall"]) * 100
    _r["correct"] = _r["tp"]


# ---------------------------------------------------------------- định dạng kiểu Việt (giống Chương 2 và mục E4/E5)
def vi_int(n):
    return f"{int(n):,}".replace(",", ".")


def vi_dec(x, nd):
    s = f"{float(x):.{nd}f}".replace(".", ",")
    return s.replace("-", "−")


def vi_signed(x, nd):
    s = vi_dec(abs(x), nd)
    return ("+" if x >= 0 else "−") + s


TRACE: list[tuple[str, str, str, str, str]] = []


def t(slide_no, label, shown, src, where):
    """Ghi lại một số hiển thị và nguồn của nó; trả về chính chuỗi hiển thị."""
    TRACE.append((slide_no, label, shown, src, where))
    return shown


# ---------------------------------------------------------------- tính số (chỉ phép tính trên giá trị đọc từ nguồn)
train_rows = [r for r in dist if r["split"] == "Training"]
pub_rows = {r["emotion"]: r for r in dist if r["split"] == "PublicTest"}
happy = next(r for r in train_rows if r["emotion"] == "Happy")
disgust = next(r for r in train_rows if r["emotion"] == "Disgust")
imbalance_ratio = int(happy["count"]) / int(disgust["count"])

ce, cw, wrs = e5["ce"], e5["class_weight"], e5["weighted_random_sampler"]
d_acc_cw = (float(cw["val_acc_mean"]) - float(ce["val_acc_mean"])) * 100
d_acc_wrs = (float(wrs["val_acc_mean"]) - float(ce["val_acc_mean"])) * 100
d_f1_wrs = (float(wrs["val_macro_f1_mean"]) - float(ce["val_macro_f1_mean"])) * 100
rec = {r["label"]: r for r in recall}

E4_ORDER = ["aug_none", "aug_flip", "aug_flip_affine", "aug_random_erasing",
            "dropout_0", "dropout_0_25", "dropout_0_5",
            "weight_decay_0", "weight_decay_1e-3", "label_smoothing_0_1"]


def gap_pp(variant):
    return float(gap[variant]["acc_gap_at_best"]) * 100


# ---------------------------------------------------------------- dựng slide
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def style_run(run, size, bold=False, color=INK, italic=False):
    f = run.font
    f.name, f.size, f.bold, f.italic = FONT, Pt(size), bold, italic
    f.color.rgb = color


def text(slide, x, y, w, h, parts, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, space_after=0, name=None):
    """parts: chuỗi, hoặc list đoạn; mỗi đoạn là chuỗi hoặc list (text, {size,bold,color,italic})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = parts if isinstance(parts, list) else [parts]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        runs = para if isinstance(para, list) else [(para, {})]
        for s, opt in runs:
            r = p.add_run()
            r.text = s
            style_run(r, opt.get("size", size), opt.get("bold", bold), opt.get("color", color),
                      opt.get("italic", False))
    return tb


def card(slide, x, y, w, h, fill=TINT, name=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.08
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def header(slide, kicker, title):
    text(slide, 0.6, 0.42, 12.1, 0.35, kicker, size=14, color=ACCENT, bold=True, name="Kicker")
    text(slide, 0.6, 0.78, 12.1, 0.7, title, size=30, color=NAVY, bold=True, name="Title")


def footer(slide, sources):
    text(slide, 0.6, 6.92, 10.4, 0.35, "Nguồn: " + " · ".join(sources), size=10, color=GRAY, name="Source")
    text(slide, 11.0, 6.92, 1.73, 0.35, "TV1 · Đoàn Trung Kiên", size=10, color=GRAY,
         align=PP_ALIGN.RIGHT, name="Owner")


def style_axis_text(axis, size=12):
    axis.tick_labels.font.size = Pt(size)
    axis.tick_labels.font.name = FONT
    axis.tick_labels.font.color.rgb = GRAY
    axis.format.line.color.rgb = MUTED


def bar_chart(slide, x, y, w, h, categories, values, colors, labels, horizontal=False, label_size=12,
              axis_size=12, gap_width=45, name=None):
    data = CategoryChartData()
    data.categories = categories
    data.add_series("s", values)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), data)
    if name:
        gf.name = name
    ch = gf.chart
    ch.has_legend = False
    ch.has_title = False
    plot = ch.plots[0]
    plot.gap_width = gap_width
    plot.vary_by_categories = False
    va, ca = ch.value_axis, ch.category_axis
    va.visible = False
    va.has_major_gridlines = False
    va.has_minor_gridlines = False
    style_axis_text(ca, axis_size)
    if horizontal:
        ca.reverse_order = True  # danh mục đầu tiên nằm trên cùng
    # Chừa chỗ cho nhãn ở đầu cột; nếu có giá trị âm, đặt tên danh mục sát mép trái để không đè nhãn
    top = max(values)
    va.maximum_scale = top * 1.32
    if min(values) < 0:
        va.minimum_scale = -top * 0.3
        ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    else:
        va.minimum_scale = 0
    ser = plot.series[0]
    ser.invert_if_negative = False
    for i, (c, lab) in enumerate(zip(colors, labels)):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = c
        dpt = ser._element.get_or_add_dPt_for_point(i)
        inv = dpt.find(qn("c:invertIfNegative"))
        if inv is None:  # theo schema, invertIfNegative đứng ngay sau c:idx
            inv = OxmlElement("c:invertIfNegative")
            dpt.find(qn("c:idx")).addnext(inv)
        inv.set("val", "0")
        dl = pt.data_label
        dl.has_text_frame = True
        dl.text_frame.text = lab
        for para in dl.text_frame.paragraphs:
            for run in para.runs:
                style_run(run, label_size, bold=(c == ACCENT or c == NAVY), color=INK)
        dl.position = XL_LABEL_POSITION.OUTSIDE_END
    return ch


# ============================================================ Slide 1: dữ liệu
s1 = prs.slides.add_slide(BLANK)
n_total = check["total_rows"]
ratio_txt = t("1", "Tỷ lệ Happy/Disgust trên Training", vi_dec(imbalance_ratio, 2),
              SRC_DIST, "count Training/Happy ÷ count Training/Disgust")
header(s1, "DỮ LIỆU FER2013",
       f"{t('1', 'Tổng số ảnh', vi_int(n_total), SRC_CHECK, 'total_rows')} ảnh xám 48×48, "
       f"Happy nhiều gấp {ratio_txt} lần Disgust")

ROLES = {"Training": "Huấn luyện + tính mean/std chuẩn hóa",
         "PublicTest": "Validation: chọn cấu hình, so sánh run",
         "PrivateTest": "Chỉ đánh giá mô hình cuối, một lần (T19)"}
cy = 1.95
for split, role in ROLES.items():
    n = check["observed_split_counts"][split]
    card(s1, 0.6, cy, 3.9, 1.22, name=f"Card {split}")
    text(s1, 0.85, cy + 0.14, 1.9, 0.55,
         t("1", f"Số ảnh {split}", vi_int(n), SRC_CHECK, f"observed_split_counts.{split}"),
         size=30, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    text(s1, 2.75, cy + 0.2, 1.6, 0.45, split, size=16, bold=True, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    text(s1, 0.85, cy + 0.72, 3.5, 0.4, role, size=13, color=GRAY)
    cy += 1.42

text(s1, 0.6, cy + 0.02, 3.9, 0.7,
     [[("Kiểm tra toàn vẹn: ", {"bold": True, "color": INK}),
       (t("1", "Trạng thái kiểm tra dữ liệu", check["status"], SRC_CHECK, "status"), {"bold": True, "color": NAVY}),
       (" · SHA-256 ", {}),
       (t("1", "SHA-256 (rút gọn)", check["sha256"][:8] + "…" + check["sha256"][-5:], SRC_CHECK, "sha256"), {}),
       (", mọi run dùng cùng bản dữ liệu", {})]],
     size=13, color=GRAY)

cats = [r["emotion"] for r in train_rows]
vals = [float(r["percent"]) for r in train_rows]
cols = [ACCENT if r["emotion"] in ("Disgust",) else (NAVY if r["emotion"] == "Happy" else MUTED) for r in train_rows]
labs = [t("1", f"Training {r['emotion']} (%, số ảnh)",
          f"{vi_dec(r['percent'], 2)}%\n{vi_int(r['count'])}", SRC_DIST,
          f"split=Training, emotion={r['emotion']}: percent, count") for r in train_rows]
text(s1, 5.0, 1.95, 7.7, 0.4, "Phân bố nhãn trên Training (tỷ lệ % và số ảnh)", size=16, bold=True, color=INK)
bar_chart(s1, 4.9, 2.35, 7.85, 3.9, cats, vals, cols, labs, label_size=12, axis_size=13, gap_width=40,
          name="Chart class distribution")
dis_pub = pub_rows["Disgust"]
text(s1, 5.0, 6.28, 7.7, 0.55,
     [[("Disgust hiếm như nhau ở validation: ", {"bold": True, "color": INK}),
       (t("1", "PublicTest Disgust (số ảnh)", vi_int(dis_pub["count"]), SRC_DIST, "split=PublicTest, emotion=Disgust: count"), {}),
       ("/", {}),
       (t("1", "Tổng PublicTest", vi_int(check["observed_split_counts"]["PublicTest"]), SRC_CHECK, "observed_split_counts.PublicTest"), {}),
       (" ảnh (", {}),
       (t("1", "PublicTest Disgust (%)", vi_dec(dis_pub["percent"], 2) + "%", SRC_DIST, "split=PublicTest, emotion=Disgust: percent"), {}),
       ("), nên accuracy một mình không đủ: báo cáo thêm macro-F1 và recall lớp hiếm.", {})]],
     size=13, color=GRAY)
footer(s1, [SRC_CHECK, SRC_DIST, "src/data/check_fer2013.py, src/data/eda.py"])
s1.notes_slide.notes_text_frame.text = (
    f"Bộ FER2013 có {vi_int(n_total)} ảnh xám 48×48, chia sẵn ba phần theo cột Usage. "
    f"Training {vi_int(check['observed_split_counts']['Training'])} ảnh dùng để huấn luyện và cũng là phần duy nhất dùng để tính mean, std chuẩn hóa, "
    f"PublicTest {vi_int(check['observed_split_counts']['PublicTest'])} ảnh làm validation để chọn cấu hình, "
    f"còn PrivateTest {vi_int(check['observed_split_counts']['PrivateTest'])} ảnh nhóm khóa lại, chỉ chạy đúng một lần cho mô hình cuối. "
    f"Trước mọi thí nghiệm, script check_fer2013 kiểm tra schema, số ảnh từng split và ghi SHA-256, kết quả {check['status']}, "
    f"nên mọi run đều chạy trên cùng một bản dữ liệu.\n"
    f"Điểm khó chính là mất cân bằng: Happy chiếm {vi_dec(happy['percent'], 2)}% Training với {vi_int(happy['count'])} ảnh, "
    f"Disgust chỉ {vi_dec(disgust['percent'], 2)}% với {vi_int(disgust['count'])} ảnh, chênh khoảng {ratio_txt} lần. "
    f"Validation cũng lệch y như vậy, Disgust chỉ có {vi_int(dis_pub['count'])} ảnh. "
    f"Vì thế nhóm không chỉ nhìn accuracy mà theo dõi thêm macro-F1 và recall lớp hiếm; phần E5 sẽ quay lại vấn đề này.\n"
    f"(Chuyển ý) Với dữ liệu ít và lệch như vậy, câu hỏi tiếp theo là augmentation và điều chuẩn giúp được bao nhiêu."
)

# ============================================================ Slide 2: augmentation & E4
s2 = prs.slides.add_slide(BLANK)
fa_gap = t("2", "Gap tại best epoch aug_flip_affine (điểm %)", vi_dec(gap_pp("aug_flip_affine"), 2), SRC_GAP,
           "variant=aug_flip_affine: acc_gap_at_best × 100")
none_gap = t("2", "Gap tại best epoch aug_none (điểm %)", vi_dec(gap_pp("aug_none"), 2), SRC_GAP,
             "variant=aug_none: acc_gap_at_best × 100")
header(s2, "AUGMENTATION & ĐIỀU CHUẨN · E4",
       f"Flip + affine: gap Train–Val từ {none_gap} còn {fa_gap} điểm %")

# Ảnh minh họa augmentation (sinh từ code, đã đăng ký figures_manifest)
img_h = 4.55
img_w = img_h * 852 / 1273
pic = s2.shapes.add_picture(str(ROOT / FIG_AUG), Inches(0.6), Inches(1.85), height=Inches(img_h))
pic.name = "Augmentation examples"
text(s2, 0.6, 1.85 + img_h + 0.1, img_w + 0.2, 0.3,
     "Gốc · flip + affine · + random erasing",
     size=11, color=GRAY, name="Image caption")

cx = 0.6 + img_w + 0.45
cw_ = 13.333 - 0.6 - cx
text(s2, cx, 1.85, cw_, 0.4, "Khoảng cách Train–Val accuracy tại best epoch (điểm %)", size=16, bold=True, color=INK)
text(s2, cx, 2.22, cw_, 0.32, "Nhãn: khoảng cách · val accuracy (PublicTest). Mỗi biến thể 1 seed.",
     size=12, color=GRAY)

cats, vals, cols, labs = [], [], [], []
bg = gap["baseline"] if "baseline" in gap else None
rows = [("baseline (flip_affine)", float(bg["acc_gap_at_best"]) * 100,
         f"{vi_dec(float(bg['acc_gap_at_best']) * 100, 2)} · {vi_dec(bg['best_val_acc'], 4)}",
         NAVY, "baseline")] if bg else []
for v in E4_ORDER:
    g = gap_pp(v)
    acc = vi_dec(e4[v]["val_acc_mean"], 4)
    color = ACCENT if v in ("aug_none", "aug_flip") else (NAVY if v == "aug_flip_affine" else MUTED)
    rows.append((v, g, f"{vi_dec(g, 2)} · {acc}", color, v))
for name, g, lab, color, key in rows:
    cats.append(name)
    vals.append(g)
    cols.append(color)
    if key == "baseline":
        t("2", "Gap baseline-002_s0 (điểm %)", vi_dec(g, 2), SRC_GAP, "exp_id=baseline-002_s0: acc_gap_at_best × 100")
        t("2", "Val acc baseline-002_s0", vi_dec(bg["best_val_acc"], 4), SRC_GAP, "exp_id=baseline-002_s0: best_val_acc")
    else:
        t("2", f"Gap {key} (điểm %)", vi_dec(g, 2), SRC_GAP, f"variant={key}: acc_gap_at_best × 100")
        t("2", f"Val acc {key}", vi_dec(e4[key]["val_acc_mean"], 4), SRC_E4, f"variant={key}: val_acc_mean")
    labs.append(lab)
bar_chart(s2, cx - 0.1, 2.5, cw_ + 0.1, 3.65, cats, vals, cols, labs, horizontal=True,
          label_size=11, axis_size=12, gap_width=35, name="Chart E4 gap")

best_f1 = max(E4_ORDER, key=lambda v: float(e4[v]["val_macro_f1_mean"]))
text(s2, cx, 6.22, cw_, 0.6,
     [[("Giảm gap ≠ tăng chất lượng: ", {"bold": True, "color": INK}),
       ("random erasing có gap nhỏ nhất nhưng macro-F1 ", {}),
       (t("2", "Macro-F1 aug_random_erasing", vi_dec(e4["aug_random_erasing"]["val_macro_f1_mean"], 4), SRC_E4,
          "variant=aug_random_erasing: val_macro_f1_mean"), {}),
       ("; macro-F1 cao nhất E4 là ", {}),
       (best_f1, {}),
       (" (", {}),
       (t("2", f"Macro-F1 {best_f1} (cao nhất E4)", vi_dec(e4[best_f1]["val_macro_f1_mean"], 4), SRC_E4,
          f"max val_macro_f1_mean → variant={best_f1}"), {}),
       (").", {})]],
     size=13, color=GRAY)
footer(s2, [SRC_GAP, SRC_E4, FIG_AUG, "configs/E4_aug_reg.json"])
s2.notes_slide.notes_text_frame.text = (
    "Augmentation chỉ gắn vào loader huấn luyện. Cấu hình mặc định của nhóm là flip ngang cộng affine nhẹ: "
    "xoay tối đa 10 độ, dịch 10%, co giãn 0,9 đến 1,1; biến thể mạnh hơn thêm random erasing, như ba cột ảnh bên trái.\n"
    f"E4 gồm {len(E4_ORDER)} biến thể, mỗi biến thể một seed, so với baseline. Thước đo là khoảng cách giữa train accuracy và "
    "val accuracy tại chính epoch có val accuracy tốt nhất. "
    f"Không augmentation thì khoảng cách lên {none_gap} điểm %, chỉ flip còn "
    f"{vi_dec(gap_pp('aug_flip'), 2)}; flip cộng affine kéo xuống {fa_gap}, ngang baseline, và val accuracy đạt "
    f"{vi_dec(e4['aug_flip_affine']['val_acc_mean'], 4)}.\n"
    f"Lưu ý: random erasing có khoảng cách nhỏ nhất, {vi_dec(gap_pp('aug_random_erasing'), 2)} điểm %, nhưng macro-F1 chỉ "
    f"{vi_dec(e4['aug_random_erasing']['val_macro_f1_mean'], 4)}. Giảm overfit không tự động là dự đoán tốt hơn. "
    f"Với dropout, {best_f1} có macro-F1 cao nhất E4 là {vi_dec(e4[best_f1]['val_macro_f1_mean'], 4)} nhưng gap "
    f"{vi_dec(gap_pp(best_f1), 2)} điểm %. "
    "Weight decay và label smoothing không tạo khác biệt nhất quán. "
    "Vì mỗi biến thể chỉ một seed, chênh lệch nhỏ không được coi là có ý nghĩa thống kê.\n"
    "(Chuyển ý) Augmentation giải quyết overfit, nhưng chưa giải quyết lớp hiếm. Đó là E5."
)

# ============================================================ Slide 3: E5
s3 = prs.slides.add_slide(BLANK)
f1_delta_txt = t("3", "Δ macro-F1 WRS so với CE (điểm %)", vi_signed(d_f1_wrs, 2), SRC_E5,
                 "(weighted_random_sampler − ce): val_macro_f1_mean × 100")
header(s3, "MẤT CÂN BẰNG LỚP · E5",
       f"Sampler tăng macro-F1 {f1_delta_txt} điểm %, accuracy giữ nguyên")

# Bảng 3 phương pháp
METHODS = [("CE thường", ce, "ce", None),
           ("Class weight", cw, "class_weight", d_acc_cw),
           ("WeightedRandomSampler", wrs, "weighted_random_sampler", d_acc_wrs)]
tx, ty, tw = 0.6, 1.95, 6.2
tbl = s3.shapes.add_table(4, 4, Inches(tx), Inches(ty), Inches(tw), Inches(2.2)).table
for j, wdt in enumerate([2.5, 1.15, 1.3, 1.25]):
    tbl.columns[j].width = Inches(wdt)
heads = ["Phương pháp (seed 0)", "Val acc", "Δ acc vs CE", "Macro-F1"]
for j, h in enumerate(heads):
    c = tbl.cell(0, j)
    c.fill.solid()
    c.fill.fore_color.rgb = NAVY
    c.text_frame.text = h
    r = c.text_frame.paragraphs[0].runs[0]
    style_run(r, 14, bold=True, color=WHITE)
    c.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
for i, (label, row, key, dacc) in enumerate(METHODS, start=1):
    hi = key == "weighted_random_sampler"
    vals_ = [label,
             t("3", f"Val acc {key}", vi_dec(row["val_acc_mean"], 4), SRC_E5, f"variant={key}: val_acc_mean"),
             "mốc" if dacc is None else t("3", f"Δ acc {key} vs CE (điểm %)", vi_signed(dacc, 2), SRC_E5,
                                         f"({key} − ce): val_acc_mean × 100"),
             t("3", f"Macro-F1 {key}", vi_dec(row["val_macro_f1_mean"], 4), SRC_E5, f"variant={key}: val_macro_f1_mean")]
    for j, v in enumerate(vals_):
        c = tbl.cell(i, j)
        c.fill.solid()
        c.fill.fore_color.rgb = TINT if hi else WHITE
        c.text_frame.text = v
        r = c.text_frame.paragraphs[0].runs[0]
        style_run(r, 14, bold=hi, color=ACCENT if (hi and j == 3) else INK)
        c.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
for i in range(4):
    tbl.rows[i].height = Inches(0.5)

# Hai thẻ recall lớp hiếm
def recall_card(x, label):
    r = rec[label]
    card(s3, x, 4.35, 2.95, 1.6, name=f"Card recall {label}")
    text(s3, x + 0.25, 4.47, 2.5, 0.65,
         t("3", f"Recall {label} (WRS)", vi_dec(r["recall"], 2) + "%", SRC_RECALL, f"label={label}: recall × 100"),
         size=32, bold=True, color=ACCENT, anchor=MSO_ANCHOR.MIDDLE)
    text(s3, x + 0.25, 5.15, 2.5, 0.7,
         [[(f"Recall {label}", {"bold": True, "color": INK})],
          [(t("3", f"Đúng/support {label}", f"{vi_int(r['correct'])}/{vi_int(r['support'])}", SRC_RECALL,
              f"label={label}: tp/support"), {}), (" ảnh PublicTest", {})]],
         size=13, color=GRAY)


recall_card(0.6, "Disgust")
recall_card(3.85, "Fear")
text(s3, 0.6, 6.12, 6.2, 0.65,
     "Recall của CE và class weight: chưa thực hiện tái tính (repo chưa có val_predictions.npz của "
     "E5-002/E5-003), nên chưa so sánh recall giữa ba phương pháp.",
     size=12, color=GRAY, name="Caveat")

# Biểu đồ recall từng lớp của WRS
text(s3, 7.25, 1.95, 5.5, 0.4, "Recall từng lớp, WeightedRandomSampler", size=16, bold=True, color=INK)
text(s3, 7.25, 2.32, 5.5, 0.32, "PublicTest, run E5-001_s0, tính lại từ dự đoán đã lưu", size=12, color=GRAY)
cats = [r["label"] for r in recall]
vals = [float(r["recall"]) for r in recall]
cols = [ACCENT if r["label"] in ("Disgust", "Fear") else MUTED for r in recall]
labs = [t("3", f"Recall {r['label']} (WRS, biểu đồ)", vi_dec(r["recall"], 1) + "%", SRC_RECALL, f"label={r['label']}: recall × 100")
        for r in recall]
bar_chart(s3, 7.15, 2.65, 5.6, 4.1, cats, vals, cols, labs, label_size=12, axis_size=12, gap_width=40,
          name="Chart E5 recall")
footer(s3, [SRC_E5, SRC_RECALL, "configs/E5_imbalance.json", "src/visual/e5_recall.py"])
s3.notes_slide.notes_text_frame.text = (
    "E5 so ba cách xử lý mất cân bằng trên cùng cấu hình baseline: cross-entropy thường, cross-entropy có trọng số lớp, "
    "và WeightedRandomSampler, tức lấy mẫu lại để mỗi batch cân bằng hơn.\n"
    f"Class weight làm accuracy giảm {vi_dec(abs(d_acc_cw), 2)} điểm % và macro-F1 còn {vi_dec(cw['val_macro_f1_mean'], 4)}. "
    f"Sampler giữ accuracy gần như CE, {vi_signed(d_acc_wrs, 2)} điểm %, nhưng macro-F1 tăng từ "
    f"{vi_dec(ce['val_macro_f1_mean'], 4)} lên {vi_dec(wrs['val_macro_f1_mean'], 4)}, tức {f1_delta_txt} điểm %. "
    "Nghĩa là mô hình phân bổ dự đoán đều hơn giữa các lớp mà không đánh đổi accuracy.\n"
    f"Trên PublicTest, sampler nhận đúng {rec['Disgust']['correct']}/{rec['Disgust']['support']} ảnh Disgust, recall "
    f"{vi_dec(rec['Disgust']['recall'], 2)}%. Fear vẫn là lớp khó nhất, recall {vi_dec(rec['Fear']['recall'], 2)}%, "
    f"thấp hơn hẳn Happy {vi_dec(rec['Happy']['recall'], 2)}% và Surprise {vi_dec(rec['Surprise']['recall'], 2)}%.\n"
    "Giới hạn cần nói rõ: mỗi phương pháp một seed, và repo chưa có file dự đoán của CE và class weight, "
    "nên nhóm chưa so sánh recall từng lớp giữa ba phương pháp; chỉ kết luận trên accuracy và macro-F1.\n"
    "(Chuyển ý) Phần tiếp theo: mô hình và kết quả cuối."
)

prs.save(OUT_PPTX)

# ---------------------------------------------------------------- bảng dò nguồn
lines = [
    "# Dò nguồn số liệu: slide phần TV1 (T26)",
    "",
    "Sinh tự động bởi `report/slides/build_slides_tv1.py`; mỗi dòng là một số hiển thị trên",
    "`report/slides/Slide_BTL02_TV1.pptx` (cả tiêu đề, bảng, biểu đồ) và vị trí của nó trong file nguồn.",
    "Đường dẫn tính từ thư mục `BTL02/`. Phép tính (÷, −, × 100) ghi rõ trong cột vị trí.",
    "",
    "| Slide | Số liệu | Hiển thị | File nguồn | Dòng/cột |",
    "|---|---|---:|---|---|",
]
seen = set()
for row in TRACE:
    if row in seen:
        continue
    seen.add(row)
    sl, label, shown, src, where = row
    lines.append(f"| {sl} | {label} | {shown.replace(chr(10), ' / ')} | `{src}` | {where} |")
lines += ["", f"Tổng: {len(seen)} số liệu, 100% có nguồn trong repo."]
OUT_TRACE.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Đã ghi {OUT_PPTX.relative_to(ROOT)} ({len(prs.slides)} slide) và {OUT_TRACE.relative_to(ROOT)} ({len(seen)} số).")
