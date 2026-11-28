"""Ghép các chương markdown thành report/BaoCao_BTL02.docx (và .pdf qua Microsoft Word).

Phụ trách: TV4 – Huỳnh Viễn Thông. Nhiệm vụ T25.

Nội dung từng chương do người viết chương sở hữu (file .md trong report/); script này chỉ ghép và chuẩn hóa hình thức,
không đổi số liệu:
- bỏ các ghi chú quy trình (xác nhận review, phản hồi kiểm thử) ở đầu chương, giữ dòng "Người viết";
- chuyển mục "Nguồn và lệnh tái lập" của từng chương sang Phụ lục A (đã viết trong phu_luc.md);
- thống nhất dấu thập phân là dấu chấm (Chương 2 và mục E4, E5 viết theo kiểu dấu phẩy);
- đưa E4, E5 vào Chương 2 (mục 2.6, 2.7) và đánh lại số bảng/hình của phần này;
- chèn ảnh cho các chú thích "Hình x.y" chỉ ghi đường dẫn;
- định dạng theo quy chuẩn CLAUDE.md: Times New Roman, căn trái, giãn dòng 1.25, không gạch đầu dòng, không gạch
  ngang dài, không in đậm giữa câu, bảng viền mảnh không tô nền, hàng không bị tách qua trang.

Chạy: python report/build_report.py [--pdf]
"""
from __future__ import annotations

import argparse
import copy
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import docx
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

REPORT = Path(__file__).resolve().parent
BTL02 = REPORT.parent
FONT = "Times New Roman"
BODY_PT = 13
TABLE_PT = 11
TEXT_WIDTH_CM = 16.0

CHAPTERS = [
    "chuong1_gioi_thieu.md",
    "chuong2_du_lieu.md",
    "chuong4_E4_E5.md",          # ghép vào Chương 2 thành mục 2.6, 2.7
    "chuong3_E1_E2_E6.md",
    "chuong4_E3_E7.md",
    "chuong5_ma_nguon.md",
    "chuong6_truc_quan_bo_loc.md",
    "chuong7_thao_luan.md",
    "chuong8_ket_luan.md",
    "tai_lieu_tham_khao.md",
    "phu_luc.md",
]

COVER_BLANK_LINES = 7  # số dòng trống trước "Giảng viên" để khung bìa kín một trang
COVER_AFTER_PT = 8  # giãn sau mỗi đoạn của bìa (mặc định Word, như file mẫu)

MEMBERS = [
    ("TV1", "Đoàn Trung Kiên", "3123411166", "Thành viên"),
    ("TV2", "Lê Phan Nguyên Hải", "3123411081", "Nhóm trưởng"),
    ("TV3", "Võ Hoàng Bảo", "3123411030", "Thành viên"),
    ("TV4", "Huỳnh Viễn Thông", "3123411287", "Thành viên"),
]

# Chú thích "Hình x.y" chỉ ghi đường dẫn: chèn ảnh ngay trước chú thích.
FIGURE_MAP = {
    "Hình 3.1": ["figures/arch_cnn_baseline.png"],
    "Hình 3.3": ["figures/pipeline_overview.png"],
    "Hình 3.4": ["figures/e2_grad_norms.png"],
    "Hình 4.1": ["figures/e3_comparison.png"],
    "Hình 4.2": ["figures/e7_val_tta_ensemble.png"],
    "Hình 4.3": ["figures/e7_test_confusion.png"],
    "Hình 4.4": ["figures/e7_test_error_hist.png"],
    "Hình 4.5": ["figures/e7_test_correct_grid.png", "figures/e7_test_wrong_grid.png"],
}

# Hình 3.2 chỉ được nhắc trong chú thích Bảng 3.2: chèn hình riêng sau đoạn mở đầu bằng anchor.
AFTER_ANCHOR = {
    "chuong3_E1_E2_E6.md": [
        ("`small` nặng nhất", ["figures/arch_cnn_comparison.png"],
         "**Hình 3.2.** So sánh kiến trúc và số tham số của các CNN (`figures/arch_cnn_comparison.png`, "
         "dữ liệu `figures/arch_cnn_comparison.csv`)."),
    ],
}

# Bảng chưa có chú thích: (file, thứ tự bảng trong file) -> chú thích.
TABLE_CAPTIONS = {
    ("chuong2_du_lieu.md", 0): "**Bảng 2.1.** Số ảnh theo split (`results/data_check.json`).",
    ("chuong2_du_lieu.md", 1): "**Bảng 2.2.** Phân bố nhãn trên Training (`figures/eda_class_distribution.csv`).",
    ("chuong2_du_lieu.md", 2): "**Bảng 2.3.** Độ sáng và ảnh bất thường theo split (`figures/eda_image_quality.csv`).",
    ("chuong4_E4_E5.md", 0): "**Bảng 2.4.** Kết quả E4 trên PublicTest, seed 0 (`report/tables/E4_summary.csv`, "
                             "`figures/e4_overfit_summary.csv`).",
    ("chuong4_E4_E5.md", 1): "**Bảng 2.5.** Kết quả E5 trên PublicTest, seed 0 (`report/tables/E5_summary.csv`).",
    ("chuong4_E4_E5.md", 2): "**Bảng 2.6.** Recall Disgust và Fear trên PublicTest.",
    ("chuong4_E3_E7.md", 5): "**Bảng 4.6.** Các cặp lớp hay nhầm trên PrivateTest (`figures/e7_test_confusion.csv`).",
}

# Thay thế văn bản theo file (đánh lại số, sửa tham chiếu chéo khi ghép).
REPLACE = {
    "chuong2_du_lieu.md": [
        ("Các con số trong Bảng 2.2", "Các con số trong Bảng 2.3"),
    ],
    "chuong4_E4_E5.md": [
        ("# Bổ sung Chương 4. Thực nghiệm E4 và E5\n", ""),
        ("## E4. Data augmentation và điều chuẩn", "## 2.6. Thực nghiệm E4: Data augmentation và điều chuẩn"),
        ("## E5. Xử lý mất cân bằng lớp", "## 2.7. Thực nghiệm E5: Xử lý mất cân bằng lớp"),
        ("Bảng E4-1", "Bảng 2.4"), ("Bảng E5-1", "Bảng 2.5"), ("Bảng E5-2", "Bảng 2.6"),
        ("Hình E4-1", "Hình 2.5"), ("Hình E5-1", "Hình 2.6"),
    ],
    "chuong4_E3_E7.md": [
        ("để Chương 5 (`T20`", "để Chương 6 (`T20`"),
        ("Trung bình cộng không trọng số của $F_1$", "Trung bình cộng không trọng số của F1"),
    ],
}

KEEP_QUOTES = ("Người viết", "Mọi độ chính xác", "Các bảng trong chương giữ nguyên")
DROP_SECTIONS = ("Nguồn và lệnh tái lập", "Phụ lục 3.A")
DECIMAL_COMMA_FILES = {"chuong2_du_lieu.md", "chuong4_E4_E5.md"}


# ----------------------------------------------------------------------------------------- tiền xử lý văn bản
def unify_decimal(text: str) -> str:
    """Kiểu Việt (1.234,5) -> kiểu dấu chấm (1,234.5); chỉ áp cho số, không chạm đường dẫn hay mã."""
    text = re.sub(r"(?<=\d),(?=\d)", "\x00", text)
    text = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", ",", text)
    return text.replace("\x00", ".")


def _dash_repl(m: re.Match) -> str:
    left, right = m.group(1), m.group(2)
    return f"{left} đến {right}" if re.search(r"\d", left) and re.search(r"\d", right) else f"{left}-{right}"


def clean_dashes(line: str, heading: bool = False) -> str:
    line = re.sub(r"\b(TV\d) [–—] ((?:[A-ZĐÂĂÊÔƠƯÁÀẢÃẠ][\wÀ-ỹ]*)(?: [A-ZĐÂĂÊÔƠƯÁÀẢÃẠ][\wÀ-ỹ]*){1,4})", r"\1 (\2)", line)
    line = re.sub(r"\) \(nhiệm vụ (T\d+)\)", r"), nhiệm vụ \1", line)
    line = re.sub(r"\s+[—–]\s+", ": " if heading else ", ", line)
    line = re.sub(r"(\S+?)[–—](\S+)", _dash_repl, line)
    return line.replace("—", "-").replace("–", "-")


def preprocess(name: str, text: str) -> str:
    for old, new in REPLACE.get(name, []):
        if old not in text:
            raise SystemExit(f"[{name}] không tìm thấy chuỗi cần thay: {old!r}")
        text = text.replace(old, new)
    if name in DECIMAL_COMMA_FILES:
        text = unify_decimal(text)
    text = re.sub(r"\n\$\$(.+?)\$\$", lambda m: "\n" + latex_to_text(m.group(1)), text)
    text = re.sub(r"^ *\$\$(.+?)\$\$ *$", lambda m: "     " + latex_to_text(m.group(1)), text, flags=re.M)
    lines, out, skip = text.split("\n"), [], False
    quote: list[str] = []

    def flush_quote():
        para = " ".join(q.strip() for q in quote).strip()
        if para.startswith(KEEP_QUOTES):
            out.append("> " + para)
            out.append("")
        quote.clear()

    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and line.startswith("## ") and any(s in line for s in DROP_SECTIONS):
            skip = True
            continue
        if skip and not in_code and (line.startswith("# ") or (line.startswith("## ") and not any(s in line for s in DROP_SECTIONS))):
            skip = False
        if skip:
            continue
        if line.startswith(">"):
            body = line[1:].strip()
            if body == "":
                flush_quote()
            else:
                quote.append(body)
            continue
        if quote:
            flush_quote()
        if line.strip() == "---":
            continue
        out.append(line)
    if quote:
        flush_quote()
    text = "\n".join(out)
    return text


def latex_to_text(expr: str) -> str:
    return ("Macro-F1 = (1/7) × Σ(c = 0..6) [2 × Precision_c × Recall_c / (Precision_c + Recall_c)]"
            if "Macro-F1" in expr else re.sub(r"[\\{}$]", "", expr))


# ----------------------------------------------------------------------------------------- docx helpers
def set_run_font(run, size=BODY_PT, bold=None, italic=None):
    run.font.name = FONT
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic


def fmt_par(p, before=0, after=4, spacing=1.25, align=WD_ALIGN_PARAGRAPH.LEFT, left=0.0, first=0.0):
    pf = p.paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), spacing
    p.alignment = align
    if left:
        pf.left_indent = Cm(left)
    if first:
        pf.first_line_indent = Cm(first)


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|(?<![\w*])\*(?!\s)[^*]+?(?<!\s)\*(?!\w))")
URL_RE = re.compile(r"(https?://[^\s)]+)")


def add_hyperlink(p, url, size=BODY_PT):
    """Link bấm được (chữ xanh, gạch chân) như khi chèn link trong Word."""
    rid = p.part.relate_to(url, RT.HYPERLINK, is_external=True)
    h = OxmlElement("w:hyperlink")
    h.set(qn("r:id"), rid)
    r = p.add_run(url)
    set_run_font(r, size)
    r.font.color.rgb = RGBColor(0x05, 0x63, 0xC1)
    r.font.underline = True
    h.append(r._r)
    p._p.append(h)


def add_inline(p, text, size=BODY_PT, keep_bold=False, italic=False):
    """Viết văn bản có **đậm**, *nghiêng*, `mã`, link; bỏ in đậm giữa câu trừ khi keep_bold."""
    if URL_RE.search(text):
        for chunk in URL_RE.split(text):
            if URL_RE.fullmatch(chunk):
                add_hyperlink(p, chunk, size)
            elif chunk:
                add_inline(p, chunk, size, keep_bold, italic)
        return
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            set_run_font(p.add_run(part[2:-2].replace("`", "")), size, bold=keep_bold, italic=italic)
        elif part.startswith("`") and part.endswith("`"):
            set_run_font(p.add_run(part[1:-1]), size, italic=italic)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            set_run_font(p.add_run(part[1:-1].replace("`", "")), size, italic=True)
        else:
            set_run_font(p.add_run(part), size, italic=italic)


def plain(text: str) -> str:
    return re.sub(r"\*\*|`", "", text)


def add_field(p, instr: str, size=BODY_PT):
    def el(tag, **attrs):
        e = OxmlElement(tag)
        for k, v in attrs.items():
            e.set(qn(k), v)
        return e
    r1 = p.add_run()
    r1._r.append(el("w:fldChar", **{"w:fldCharType": "begin"}))
    r2 = p.add_run()
    it = el("w:instrText", **{"xml:space": "preserve"})
    it.text = instr
    r2._r.append(it)
    r3 = p.add_run()
    r3._r.append(el("w:fldChar", **{"w:fldCharType": "separate"}))
    r4 = p.add_run("Nhấn chuột phải, chọn Update Field để cập nhật.")
    set_run_font(r4, size)
    r5 = p.add_run()
    r5._r.append(el("w:fldChar", **{"w:fldCharType": "end"}))


class Builder:
    def __init__(self):
        self.doc = docx.Document()
        self._setup()
        self.table_idx = 0
        self.fig_count = 0

    # ---------------------------------------------------------------- styles, trang
    def _setup(self):
        d = self.doc
        sec = d.sections[0]
        sec.page_width, sec.page_height = Cm(21), Cm(29.7)
        sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Cm(3), Cm(2), Cm(2), Cm(2)
        normal = d.styles["Normal"]
        normal.font.name, normal.font.size = FONT, Pt(BODY_PT)
        normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
        normal.paragraph_format.line_spacing = 1.25
        normal.paragraph_format.space_after = Pt(4)
        for name, size, before, after in (("Heading 1", 14, 14, 6), ("Heading 2", 13, 10, 4), ("Heading 3", 13, 8, 4)):
            st = d.styles[name]
            st.font.name, st.font.size, st.font.bold, st.font.italic = FONT, Pt(size), True, False
            st.font.color.rgb = RGBColor(0, 0, 0)
            rpr = st.element.get_or_add_rPr()
            rf = rpr.get_or_add_rFonts()
            for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(att), FONT)
            for att in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if rf.get(qn(att)) is not None:
                    del rf.attrib[qn(att)]
            pf = st.paragraph_format
            pf.space_before, pf.space_after, pf.line_spacing = Pt(before), Pt(after), 1.25
            pf.keep_with_next = True
        # số trang ở chân trang
        fp = sec.footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_field(fp, "PAGE", 11)

    # ---------------------------------------------------------------- bìa và mục lục
    def cover(self):
        """Bìa theo mẫu bìa bài tập cá nhân của nhóm (khung viền đôi quanh trang, logo SGU, chữ 14/18 pt)."""
        d = self.doc
        border = parse_xml(
            f'<w:pBdr {nsdecls("w")}>'
            '<w:top w:val="thinThickSmallGap" w:sz="24" w:space="1" w:color="auto"/>'
            '<w:left w:val="thinThickSmallGap" w:sz="24" w:space="4" w:color="auto"/>'
            '<w:bottom w:val="thickThinSmallGap" w:sz="24" w:space="1" w:color="auto"/>'
            '<w:right w:val="thickThinSmallGap" w:sz="24" w:space="4" w:color="auto"/>'
            '</w:pBdr>')

        def line(parts=(), before=0, indent_cm=0.0, center=True):
            """parts: danh sách (văn bản, cỡ chữ, đậm). Đoạn rỗng vẫn mang khung để khung liền một khối."""
            p = d.add_paragraph()
            fmt_par(p, before=before, after=COVER_AFTER_PT, spacing=1.08,
                    align=WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT, first=indent_cm)
            p._p.get_or_add_pPr().append(copy.deepcopy(border))
            for text, size, bold in parts:
                set_run_font(p.add_run(text), size, bold=bold)
            if not parts:
                set_run_font(p.add_run(""), 14)
            return p

        p = line(before=10)
        p.runs[0].add_break()
        set_run_font(p.add_run("TRƯỜNG ĐẠI HỌC SÀI GÒN"), 14)
        line([("KHOA CÔNG NGHỆ THÔNG TIN", 14, True)], before=10)
        line()
        logo = line()
        logo.add_run().add_picture(str(REPORT / "assets" / "logo_sgu.png"), width=Cm(5.08))
        line()
        line([("CHỦ ĐỀ", 18, True), (" Bài tập lớn 02: Nhận dạng cảm xúc khuôn mặt FER2013 bằng mạng "
                                    "nơ-ron tích chập (PyTorch)", 18, False)])
        line()
        line([("Nhóm sinh viên thực hiện:", 14, False)])
        for ma, ten, mssv, vai in MEMBERS:
            suffix = " (nhóm trưởng)" if vai == "Nhóm trưởng" else ""
            m = line([(ten, 14, False)], indent_cm=3.5, center=False)
            m.paragraph_format.tab_stops.add_tab_stop(Cm(9.37))  # MSSV thẳng một cột
            m.runs[-1].add_tab()
            set_run_font(m.add_run(f"{mssv}{suffix}"), 14)
        for _ in range(COVER_BLANK_LINES):
            line()
        line([("Giảng viên: TS. Đỗ Như Tài", 14, False)])
        last = line([("Thành phố Hồ Chí Minh, tháng 11 năm 2026", 14, False)])
        last.runs[-1].add_break(WD_BREAK.PAGE)
        p = d.add_paragraph()
        fmt_par(p, after=10, align=WD_ALIGN_PARAGRAPH.CENTER)
        set_run_font(p.add_run("MỤC LỤC"), 14, bold=True)
        p = d.add_paragraph()
        add_field(p, 'TOC \\o "1-2" \\h \\z \\u')
        self.page_break()

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    # ---------------------------------------------------------------- khối nội dung
    def heading(self, text, level):
        text = clean_dashes(plain(text), heading=True)
        h = self.doc.add_heading(level=level)
        if level == 1 and self._started:
            h.paragraph_format.page_break_before = True  # tránh trang trắng khi trang trước vừa đầy
        self._started = True
        set_run_font(h.add_run(text), {1: 14, 2: 13, 3: 13}[level], bold=True)
        h.alignment = WD_ALIGN_PARAGRAPH.LEFT

    _started = False

    def paragraph(self, text, left=0.0, first=0.0, italic=False):
        text = clean_dashes(text)
        p = self.doc.add_paragraph()
        fmt_par(p, left=left, first=first)
        keep = text.startswith("**Kết luận")
        add_inline(p, text, keep_bold=keep, italic=italic)
        return p

    def caption(self, text):
        text = clean_dashes(text)
        m = CAPTION_RE.match(text)
        p = self.doc.add_paragraph()
        fmt_par(p, before=2, after=6, spacing=1.15, align=WD_ALIGN_PARAGRAPH.CENTER)
        if m:
            extra = m.group(2).strip().rstrip(".").strip()
            body = m.group(3).strip()
            if extra:  # "**Bảng 4.1 (tương ứng ...).** Nội dung" -> "Bảng 4.1. Nội dung (tương ứng ...)"
                body = body.rstrip(".") + " " + extra + "."
            set_run_font(p.add_run(m.group(1) + ". "), 12, bold=True)
            add_inline(p, body, size=12, italic=True)
        else:
            add_inline(p, text, size=12, italic=True)
        if text.startswith("**Bảng"):
            p.paragraph_format.keep_with_next = True
        return p

    def image(self, rel_path):
        path = (BTL02 / rel_path).resolve()
        if not path.exists():
            raise SystemExit(f"thiếu hình {rel_path}")
        with Image.open(path) as im:
            w, h = im.size
        width = min(15.5, 12.0 * w / h)
        p = self.doc.add_paragraph()
        fmt_par(p, before=4, after=2, spacing=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Cm(width))
        self.fig_count += 1

    def code(self, lines):
        for i, ln in enumerate(lines):
            p = self.doc.add_paragraph()
            fmt_par(p, after=0 if i < len(lines) - 1 else 6, spacing=1.0, left=1.0)
            set_run_font(p.add_run(clean_dashes(ln) if ln.strip() else " "), TABLE_PT)

    def _format_table(self, table, headers, rows, widths_cm, aligns, size=TABLE_PT):
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT}
        all_rows = [headers] + rows
        for ri, data in enumerate(all_rows):
            row = table.rows[ri]
            trpr = row._tr.get_or_add_trPr()
            if ri == 0:
                trpr.append(parse_xml(r"<w:tblHeader %s/>" % nsdecls("w")))
            trpr.append(parse_xml(r"<w:cantSplit %s/>" % nsdecls("w")))
            for ci, val in enumerate(data):
                cell = row.cells[ci]
                cell.width = Cm(widths_cm[ci])
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                p = cell.paragraphs[0]
                fmt_par(p, before=2, after=2, spacing=1.0,
                        align=WD_ALIGN_PARAGRAPH.CENTER if ri == 0 else amap[aligns[ci]])
                if ri < min(2, len(all_rows) - 1):  # không để hàng tiêu đề nằm lẻ cuối trang
                    p.paragraph_format.keep_with_next = True
                add_inline(p, clean_dashes(val.strip()) if val.strip() not in ("—", "–") else "-",
                           size=size, keep_bold=(ri == 0))
        tbl_pr = table._tbl.tblPr
        tbl_pr.append(parse_xml(f"""
            <w:tblBorders {nsdecls("w")}>
              <w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>
              <w:left w:val="single" w:sz="6" w:space="0" w:color="000000"/>
              <w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>
              <w:right w:val="single" w:sz="6" w:space="0" w:color="000000"/>
              <w:insideH w:val="single" w:sz="4" w:space="0" w:color="7F7F7F"/>
              <w:insideV w:val="single" w:sz="4" w:space="0" w:color="7F7F7F"/>
            </w:tblBorders>"""))
        grid = table._tbl.tblGrid
        for gc, wcm in zip(grid.findall(qn("w:gridCol")), widths_cm):
            gc.set(qn("w:w"), str(int(Cm(wcm).twips)))

    def table(self, md_rows):
        cells = [[c.replace(" ± ", " ± ") for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))]
                 for r in md_rows]  # giữ "mean ± std" trên một dòng
        header, sep, body = cells[0], cells[1], cells[2:]
        n = len(header)
        body = [(r + [""] * n)[:n] for r in body]
        aligns = []
        for s in sep:
            s = s.strip()
            aligns.append("r" if s.endswith(":") and not s.startswith(":") else "c" if s.startswith(":") and s.endswith(":") else "l")
        widths, size = column_widths([[plain(c) for c in r] for r in [header] + body])
        t = self.doc.add_table(rows=1 + len(body), cols=n)
        self._format_table(t, header, body, widths, aligns, size=size)
        sp = self.doc.add_paragraph()
        fmt_par(sp, after=2, spacing=1.0)


NUMERIC_TOKEN = re.compile(r"^[\d.,%+\-−± ()×/e]+$")


def column_widths(rows, total=TEXT_WIDTH_CM, pad=0.35):
    """Trả (độ rộng cột cm, cỡ chữ). Số (kể cả "mean ± std") không bị ngắt; tên/mã dài được phép xuống dòng.
    Bảng nhiều cột được thu cỡ chữ 11 -> 10 -> 9 pt cho tới khi vừa khổ giấy."""
    n = len(rows[0])
    for size in (TABLE_PT, 10, 9):
        cpc = 0.23 * size / TABLE_PT
        mins, avgs = [], []
        for ci in range(n):
            longest = 1
            for ri, r in enumerate(rows):
                for w in r[ci].strip().split(" "):
                    cap = len(w) if NUMERIC_TOKEN.match(w) else (7 if ri == 0 else 10)
                    longest = max(longest, min(len(w), cap))
            mins.append(longest * cpc + pad)
            avgs.append(max(1.0, sum(len(r[ci].strip()) for r in rows) / len(rows)))
        if sum(mins) <= total:
            break
    if sum(mins) >= total:
        return [total * m / sum(mins) for m in mins], size
    extra = total - sum(mins)
    grow = [max(0.0, a * cpc + pad - m) for a, m in zip(avgs, mins)]
    if sum(grow) <= 0:
        return [m + extra / n for m in mins], size
    if sum(grow) <= extra:
        return [m + g + (extra - sum(grow)) * g / sum(grow) for m, g in zip(mins, grow)], size
    return [m + extra * g / sum(grow) for m, g in zip(mins, grow)], size


# ----------------------------------------------------------------------------------------- parser markdown
CAPTION_RE = re.compile(r"\*\*((?:Hình|Bảng) (?:[A-Z]|\d+)\.\d+)(.*?)\*\*\s*(.*)", re.S)
LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
ALPHA = "abcdefghijklmnopqrstuvwxyz"
ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]


def render_file(b: Builder, name: str, text: str):
    lines = text.split("\n")
    i, n = 0, len(lines)
    table_no = 0
    anchors = list(AFTER_ANCHOR.get(name, []))
    counters: dict[int, int] = {}
    para_buf: list[str] = []
    pending_caption: str | None = None

    def emit_para(txt):
        nonlocal pending_caption
        txt = txt.strip()
        if not txt:
            return
        m_cap = CAPTION_RE.match(txt)
        if m_cap:
            label = m_cap.group(1)
            for png in FIGURE_MAP.get(label, []):
                b.image(png)
            b.caption(txt)
            pending_caption = label if label.startswith("Bảng") else None
        elif txt.startswith("> "):
            pending_caption = None
            b.paragraph(txt[2:], italic=True)
        else:
            pending_caption = None
            b.paragraph(txt)
        for a in list(anchors):
            if txt.startswith(a[0]):
                for png in a[1]:
                    b.image(png)
                b.caption(a[2])
                anchors.remove(a)

    def flush():
        if para_buf:
            emit_para(" ".join(s.strip() for s in para_buf))
            para_buf.clear()

    while i < n:
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            flush()
            j = i + 1
            block = []
            while j < n and not lines[j].strip().startswith("```"):
                block.append(lines[j])
                j += 1
            b.code(block)
            i = j + 1
            continue
        if s.startswith("|"):
            flush()
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            if pending_caption is None:
                cap = TABLE_CAPTIONS.get((name, table_no))
                if cap:
                    b.caption(cap)
            pending_caption = None
            b.table(block)
            table_no += 1
            continue
        m_img = re.match(r"!\[[^\]]*\]\(([^)]+)\)", s)
        if m_img:
            flush()
            b.image(str((REPORT / m_img.group(1)).resolve().relative_to(BTL02)).replace("\\", "/"))
            i += 1
            continue
        m_h = re.match(r"^(#{1,3})\s+(.*)$", line)
        if m_h:
            flush()
            counters.clear()
            b.heading(m_h.group(2), len(m_h.group(1)))
            i += 1
            continue
        m_l = LIST_RE.match(line)
        if m_l:
            flush()
            indent = len(m_l.group(1).replace("\t", "    "))
            level = 0 if indent < 2 else 1
            marker, content = m_l.group(2), m_l.group(3)
            for k in [k for k in counters if k > level]:
                del counters[k]
            j = i + 1
            while j < n and lines[j].strip() and not LIST_RE.match(lines[j]) and lines[j].startswith(" ") \
                    and not lines[j].strip().startswith(("```", "|")):
                content += " " + lines[j].strip()
                j += 1
            counters[level] = counters.get(level, 0) + 1
            k = counters[level]
            if marker[0].isdigit():
                label = marker
            elif level == 0:
                label = ALPHA[(k - 1) % 26] + ")"
            else:
                label = ROMAN[(k - 1) % 12] + ")"
            if CAPTION_RE.match(content):
                emit_para(content)
            else:
                b.paragraph(f"{label} {content}", left=0.75 * (level + 1), first=-0.6)
            i = j
            continue
        if not s:
            flush()
            counters.clear()
            i += 1
            continue
        if line.startswith("     ") and not para_buf:  # dòng thụt (công thức) sau mục danh sách
            p = b.paragraph(s, left=1.5)
            for r in p.runs:
                r.font.italic = True
            i += 1
            continue
        para_buf.append(line)
        i += 1
    flush()


def build(out_docx: Path) -> Path:
    b = Builder()
    b.cover()
    for name in CHAPTERS:
        text = (REPORT / name).read_text(encoding="utf-8")
        text = re.sub(r"<!--.*?-->\n?", "", text, flags=re.S)
        render_file(b, name, preprocess(name, text))
    set_metadata(b.doc)
    b.doc.save(out_docx)
    return out_docx


PS_EXPORT = r"""
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
  $doc = $word.Documents.Open('%DOCX%')
  foreach ($t in $doc.TablesOfContents) { $t.Update() }
  $doc.Fields.Update() | Out-Null
  foreach ($t in $doc.TablesOfContents) { $t.Update() }
  $doc.Save()
  $doc.ExportAsFixedFormat('%PDF%', 17)
  $doc.Close()
} finally { $word.Quit() }
"""


def set_metadata(doc) -> None:
    """Thuộc tính file ghi theo nhóm, không lấy tên tài khoản Office của máy dựng báo cáo."""
    cp = doc.core_properties
    cp.author = cp.last_modified_by = "Nhóm BTL02"
    cp.title = "Báo cáo BTL02: Nhận dạng cảm xúc khuôn mặt FER2013 bằng CNN (PyTorch)"
    cp.comments = ""
    # Khuôn mặc định của python-docx khai báo chế độ tương thích Word cũ -> Word mở ở "Compatibility Mode"
    settings = doc.settings.element
    compat = settings.find(qn("w:compat"))
    if compat is None:
        compat = OxmlElement("w:compat")
        settings.append(compat)
    for cs in compat.findall(qn("w:compatSetting")):
        if cs.get(qn("w:name")) == "compatibilityMode":
            compat.remove(cs)
    cs = OxmlElement("w:compatSetting")
    for k, v in (("w:name", "compatibilityMode"), ("w:uri", "http://schemas.microsoft.com/office/word"), ("w:val", "15")):
        cs.set(qn(k), v)
    compat.append(cs)


def export_pdf(docx_path: Path) -> Path:
    pdf = docx_path.with_suffix(".pdf")
    script = PS_EXPORT.replace("%DOCX%", str(docx_path)).replace("%PDF%", str(pdf))
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8-sig") as f:
        f.write(script)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", f.name], check=True)
    Path(f.name).unlink(missing_ok=True)
    return pdf


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ghép BaoCao_BTL02.docx từ các chương markdown")
    ap.add_argument("--out", default=str(REPORT / "BaoCao_BTL02.docx"))
    ap.add_argument("--pdf", action="store_true", help="cập nhật mục lục và xuất PDF bằng Microsoft Word")
    a = ap.parse_args(argv)
    out = build(Path(a.out).resolve())
    print("đã ghi", out)
    if a.pdf:
        print("đã ghi", export_pdf(out))
        d = docx.Document(out)  # Word ghi đè "Last modified by" khi lưu mục lục: đặt lại
        set_metadata(d)
        d.save(out)


if __name__ == "__main__":
    sys.exit(main())
