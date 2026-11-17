"""Build the Word deliverables from the markdown sources.

    python tools/build_docs.py

Turns docs/项目书.md and docs/报名表填写.md into formatted .docx files
(A4, Chinese fonts, real tables, embedded figures). No external converter.
"""
from __future__ import annotations

import os
import re
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT2 = RGBColor(0x2E, 0x86, 0xC1)
GREY = RGBColor(0x59, 0x59, 0x59)
INK = RGBColor(0x22, 0x22, 0x22)
CJK = "PingFang SC"
MONO = "Menlo"


# --------------------------------------------------------------------- helpers
def set_cjk(run, font_name: str = CJK) -> None:
    run.font.name = font_name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), font_name)
    rFonts.set(qn("w:hAnsi"), font_name)
    rFonts.set(qn("w:eastAsia"), font_name)


def shade(paragraph, hex_fill: str) -> None:
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    pPr.append(shd)


def cell_shade(cell, hex_fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def add_rich(paragraph, text: str, size: float = 10.5, color=INK,
             bold_all: bool = False) -> None:
    """Render **bold**, `code` and plain text runs."""
    tokens = re.split(r"(\*\*.+?\*\*|`[^`]+`)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            r = paragraph.add_run(tok[2:-2])
            r.bold = True
        elif tok.startswith("`") and tok.endswith("`") and len(tok) > 2:
            r = paragraph.add_run(tok[1:-1])
            r.font.name = MONO
            r._element.get_or_add_rPr()
            r.font.size = Pt(size - 1)
            r.font.color.rgb = RGBColor(0xB0, 0x3A, 0x2E)
            set_cjk(r, MONO)
            continue
        else:
            r = paragraph.add_run(tok)
        r.font.size = Pt(size)
        r.font.color.rgb = color
        if bold_all:
            r.bold = True
        set_cjk(r)


def style_doc(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = CJK
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    normal.paragraph_format.line_spacing = 1.55
    normal.paragraph_format.space_after = Pt(4)

    for name, size, color in (("Heading 1", 16, ACCENT), ("Heading 2", 13, ACCENT2),
                              ("Heading 3", 11.5, INK), ("Title", 30, ACCENT)):
        st = doc.styles[name]
        st.font.name = CJK
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)
        st.paragraph_format.keep_with_next = True
        st.paragraph_format.space_before = Pt(14 if name != "Title" else 0)
        st.paragraph_format.space_after = Pt(8 if name != "Title" else 6)


def cover(doc: Document, title: str, subtitle: str, meta: list[str],
          logo: str | None = None) -> None:
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.font.size = Pt(34)
    r.bold = True
    r.font.color.rgb = ACCENT
    set_cjk(r)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(subtitle)
    r.font.size = Pt(14)
    r.font.color.rgb = GREY
    set_cjk(r)

    doc.add_paragraph()
    if logo and os.path.exists(logo):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(logo, width=Cm(9))

    for _ in range(3):
        doc.add_paragraph()
    for line in meta:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(line)
        r.font.size = Pt(11)
        r.font.color.rgb = GREY
        set_cjk(r)
    doc.add_page_break()


def add_table(doc: Document, rows: list[list[str]]) -> None:
    header, body = rows[0], rows[1:]
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    trPr = t.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)

    for i, cell_text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        add_rich(p, cell_text, size=9.5, color=RGBColor(0xFF, 0xFF, 0xFF), bold_all=True)
        cell_shade(cell, "1F4E79")
    for row in body:
        cells = t.add_row().cells
        for i, val in enumerate(row[: len(header)]):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            add_rich(p, val, size=9.5)
    doc.add_paragraph()


def add_code(doc: Document, lines: list[str]) -> None:
    for n, line in enumerate(lines):
        p = doc.add_paragraph()
        p.paragraph_format.keep_together = True
        p.paragraph_format.keep_with_next = n < len(lines) - 1
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.left_indent = Cm(0.4)
        p.paragraph_format.line_spacing = 1.15
        shade(p, "F2F4F7")
        r = p.add_run(line if line else " ")
        r.font.name = MONO
        r.font.size = Pt(8.5)
        set_cjk(r, MONO)
    doc.add_paragraph()


def add_image(doc: Document, path: str, caption: str | None) -> None:
    if not os.path.exists(path):
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Cm(16.2))
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(caption)
        r.font.size = Pt(9)
        r.font.color.rgb = GREY
        set_cjk(r)


def add_quote(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.space_before = Pt(2)
    shade(p, "EEF4FA")
    add_rich(p, text, size=9.5, color=RGBColor(0x33, 0x44, 0x55))


def add_bullet(doc: Document, text: str, level: int = 0) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.75 + 0.6 * level)
    p.paragraph_format.space_after = Pt(2)
    add_rich(p, text)


def add_number(doc: Document, text: str, number: str = "1.") -> None:
    """Manual numbering: Word's List Number style keeps counting across lists."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.first_line_indent = Cm(-0.55)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"{number} ")
    r.font.size = Pt(10.5)
    r.font.color.rgb = INK
    set_cjk(r)
    add_rich(p, text)


# ---------------------------------------------------------------- md renderer
def render_markdown(doc: Document, md: str, base: str) -> None:
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        if s.startswith("```"):
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i].rstrip())
                i += 1
            i += 1
            add_code(doc, code)
            continue

        if s.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
            if m:
                add_image(doc, os.path.join(base, m.group(2)), m.group(1))
            i += 1
            continue

        if s.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= set("|-: "):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c or "---") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                add_table(doc, rows)
            continue

        if s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            text = s.lstrip("#").strip()
            if level == 1:
                doc.add_heading(text, level=1)
            elif level == 2:
                doc.add_heading(text, level=2)
            else:
                doc.add_heading(text, level=3)
            i += 1
            continue

        if s == "---":
            i += 1
            continue

        if s.startswith(">"):
            add_quote(doc, s.lstrip("> ").strip())
            i += 1
            continue

        if re.match(r"^[-*]\s+", s):
            add_bullet(doc, re.sub(r"^[-*]\s+", "", s))
            i += 1
            continue

        m_num = re.match(r"^(\d+)[.)]\s+", s)
        if m_num:
            add_number(doc, s[m_num.end():], number=m_num.group(1) + ".")
            i += 1
            continue

        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0)
        add_rich(p, s)
        i += 1


# ------------------------------------------------------------------- builders
def build_book() -> str:
    src = os.path.join(DOCS, "项目书.md")
    out = os.path.join(DOCS, "BeatDancer项目书_2026地瓜机器人黑客松.docx")
    md = open(src, encoding="utf-8").read()
    # drop the md title block (title + subtitle + meta + rule): the cover replaces it
    lines = md.splitlines()
    cut = 0
    for i, line in enumerate(lines[:12]):
        if line.strip() == "---":
            cut = i + 1
            break
    md = "\n".join(lines[cut:])

    doc = Document()
    style_doc(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for attr in ("left_margin", "right_margin"):
        setattr(sec, attr, Cm(2.3))
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.2)

    cover(doc, "BeatDancer", "给机器人一副耳朵，它自己会编舞",
          ["2026 地瓜机器人黑客松（DDC2026）参赛项目书",
           "上海张江 · 2026 年 12 月 5–7 日",
           "项目主页：github.com/〔你的用户名〕/beatdancer"],
          logo=os.path.join(DOCS, "fig_pipeline.png"))

    render_markdown(doc, md, DOCS)
    doc.save(out)
    return out


def build_form() -> str:
    src = os.path.join(DOCS, "报名表填写.md")
    out = os.path.join(DOCS, "报名表填写_可复制版.docx")
    md = open(src, encoding="utf-8").read()
    md = re.sub(r"^# .*?\n", "", md, count=1)

    doc = Document()
    style_doc(doc)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for attr in ("left_margin", "right_margin"):
        setattr(sec, attr, Cm(2.2))

    p = doc.add_paragraph()
    r = p.add_run("2026 地瓜机器人黑客松 · 报名表填写内容")
    r.font.size = Pt(20)
    r.bold = True
    r.font.color.rgb = ACCENT
    set_cjk(r)

    p = doc.add_paragraph()
    r = p.add_run("灰色说明是给你的提示，不要粘进表单；黑色文本框里的内容可以直接复制。")
    r.font.size = Pt(9.5)
    r.font.color.rgb = GREY
    set_cjk(r)

    render_markdown(doc, md, DOCS)
    doc.save(out)
    return out


if __name__ == "__main__":
    print(build_book())
    print(build_form())
