"""Render the bilingual post-stack inversion guide to PDF.

English and Russian are laid out side by side, column for column, so a mixed
team can work from one page and point at the same paragraph.  DejaVu Sans is
registered because ReportLab's built-in fonts carry no Cyrillic -- with
Helvetica every Russian character renders as a black box.
"""
from __future__ import annotations

import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether,
                                NextPageTemplate, PageBreak, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guide_content as C  # noqa: E402

# Screenshots are optional: the guide has to build on a machine that has never
# run the app, so a missing image is a skipped figure and a note, not a crash.
SHOT_DIR = os.environ.get(
    "GUIDE_SHOTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots"))

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DJV", f"{FONT_DIR}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJV-B", f"{FONT_DIR}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DJV-M", f"{FONT_DIR}/DejaVuSansMono.ttf"))
# DejaVu ships no oblique for Sans here, so italic falls back to regular rather
# than letting ReportLab synthesise a missing face.
pdfmetrics.registerFontFamily("DJV", normal="DJV", bold="DJV-B",
                              italic="DJV", boldItalic="DJV-B")

INK = colors.HexColor("#1b2430")
MUTED = colors.HexColor("#5b6673")
ACCENT = colors.HexColor("#b4423c")
ACCENT_2 = colors.HexColor("#2d5f8a")
RULE = colors.HexColor("#d5dae0")
PANEL = colors.HexColor("#f4f6f8")
NOTE_BG = colors.HexColor("#fdf4e7")
NOTE_EDGE = colors.HexColor("#e0a458")

PAGE_W, PAGE_H = A4
MARGIN = 16 * mm
BODY_W = PAGE_W - 2 * MARGIN
COL_W = (BODY_W - 6 * mm) / 2


def style(name, size=8.6, leading=11.6, font="DJV", colour=INK, **kw):
    return ParagraphStyle(name, fontName=font, fontSize=size, leading=leading,
                          textColor=colour, **kw)


S = {
    # Ragged right: at an 86 mm column width, justifying long Russian words
    # opens rivers of white space rather than tidying the block.
    "body": style("body", leading=11.0, alignment=TA_LEFT),
    "body_ru": style("body_ru", leading=11.0, alignment=TA_LEFT),
    "h1": style("h1", 16, 19, "DJV-B", ACCENT, spaceBefore=0, spaceAfter=1),
    "h1ru": style("h1ru", 11.5, 14, "DJV", MUTED, spaceAfter=6),
    "h2": style("h2", 10.5, 13, "DJV-B", ACCENT_2, spaceBefore=4, spaceAfter=0),
    "h2ru": style("h2ru", 9, 11.5, "DJV", MUTED, spaceAfter=4),
    "lead": style("lead", 9.4, 13),
    "small": style("small", 7.6, 10, colour=MUTED),
    "label": style("label", 7.0, 9, "DJV-B", ACCENT_2),
    "code": style("code", 7.6, 11, "DJV-M"),
    "cell": style("cell", 7.6, 10),
    "gloss": style("gloss", 7.2, 9.2),
    "cellb": style("cellb", 7.6, 10, "DJV-B"),
    "stepnum": style("stepnum", 18, 20, "DJV-B", colors.white),
    "steptitle": style("steptitle", 11.5, 14, "DJV-B", colors.white),
    "steptitleru": style("steptitleru", 9.5, 12, "DJV", colors.HexColor("#e8eef4")),
    "cover": style("cover", 30, 35, "DJV-B", colors.white),
    "coverru": style("coverru", 19, 24, "DJV", colors.HexColor("#dbe4ec")),
    "coversub": style("coversub", 11, 15, "DJV", colors.HexColor("#c3d0dc")),
}


def two_col(left, right, pad=0):
    """Side-by-side English / Russian block."""
    t = Table([[left, right]], colWidths=[COL_W, COL_W])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 3 * mm),
        ("LEFTPADDING", (1, 0), (1, 0), 3 * mm),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), pad),
        ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
        ("LINEBEFORE", (1, 0), (1, 0), 0.4, RULE),
    ]))
    return t


def para(text, st="body"):
    return Paragraph(text, S[st])


def bullets(items, st="body"):
    return [Paragraph(f"•&nbsp;&nbsp;{it}", ParagraphStyle(
        "b", parent=S[st], leftIndent=8, firstLineIndent=-8, spaceAfter=3))
        for it in items]


def numbered(items, st="body"):
    return [Paragraph(f"<b>{i}.</b>&nbsp;&nbsp;{it}", ParagraphStyle(
        "n", parent=S[st], leftIndent=12, firstLineIndent=-12, spaceAfter=5))
        for i, it in enumerate(items, 1)]


def note_box(en, ru):
    inner = two_col(para(en), para(ru))
    t = Table([[inner]], colWidths=[BODY_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NOTE_BG),
        ("LINEBEFORE", (0, 0), (0, 0), 2.2, NOTE_EDGE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def code_box(text):
    body = Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"])
    t = Table([[body]], colWidths=[BODY_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("BOX", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def render_blocks(blocks, story):
    for block in blocks:
        kind = block[0]
        if kind == "h1":
            story.append(Spacer(1, 4))
            story.append(Paragraph(block[1], S["h1"]))
            story.append(Paragraph(block[2], S["h1ru"]))
            story.append(rule())
        elif kind == "h2":
            story.append(Spacer(1, 6))
            story.append(Paragraph(block[1], S["h2"]))
            story.append(Paragraph(block[2], S["h2ru"]))
        elif kind == "p":
            story.append(two_col(para(block[1]), para(block[2])))
            story.append(Spacer(1, 5))
        elif kind == "bullets":
            story.append(two_col(bullets(block[1]), bullets(block[2])))
            story.append(Spacer(1, 3))
        elif kind == "numbered":
            story.append(two_col(numbered(block[1]), numbered(block[2])))
            story.append(Spacer(1, 3))
        elif kind == "note":
            story.append(note_box(block[1], block[2]))
            story.append(Spacer(1, 6))
        elif kind == "code":
            story.append(code_box(block[1]))
            story.append(Spacer(1, 6))
        elif kind == "resulttable":
            story.append(result_table())
            story.append(Spacer(1, 6))
        elif kind == "figure":
            # block[5] is a height cap in *millimetres*, matching how the step
            # figures declare theirs -- passing raw points here silently
            # shrank a full-width figure to a third of its size.
            fig = figure(block[1], block[2], block[3],
                         max_w=block[4] if len(block) > 4 else 1.0,
                         max_h=(block[5] if len(block) > 5 else 118) * mm)
            if fig is not None:
                story.append(fig)


def figure(name, cap_en, cap_ru, max_w=1.0, max_h=118 * mm):
    """A screenshot with a bilingual caption, scaled to fit the text column.

    Returned as a KeepTogether so a caption never orphans onto the next page
    from the picture it describes.
    """
    path = name if os.path.isabs(name) else os.path.join(SHOT_DIR, name)
    if not os.path.isfile(path):
        return None
    try:
        from reportlab.lib.utils import ImageReader
        iw, ih = ImageReader(path).getSize()
    except Exception:  # noqa: BLE001 - unreadable image is a skipped figure
        return None
    if not iw or not ih:
        return None

    target_w = BODY_W * max_w
    scale = min(target_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale

    img = Image(path, width=w, height=h)
    frame = Table([[img]], colWidths=[w])
    frame.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    holder = Table([[frame]], colWidths=[BODY_W])
    holder.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    caption = two_col(para(cap_en, "small"), para(cap_ru, "small"))
    return KeepTogether([holder, caption, Spacer(1, 8)])


def rule(colour=RULE, thickness=0.6, space=5):
    t = Table([[""]], colWidths=[BODY_W], rowHeights=[thickness])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colour)]))
    return KeepTogether([t, Spacer(1, space)])


# --------------------------------------------------------------------------
# Step cards
# --------------------------------------------------------------------------

def step_card(step):
    head_inner = Table(
        [[Paragraph(str(step["n"]), S["stepnum"]),
          [Paragraph(step["en_title"], S["steptitle"]),
           Paragraph(step["ru_title"], S["steptitleru"])]]],
        colWidths=[13 * mm, BODY_W - 13 * mm - 12])
    head_inner.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    head = Table([[head_inner]], colWidths=[BODY_W])
    head.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_2),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    rows = []
    for label_en, label_ru, key_en, key_ru in (
            ("WHAT IT DOES", "ЧТО ДЕЛАЕТ", "en_does", "ru_does"),
            ("WHAT YOU DO", "ЧТО ДЕЛАТЬ", "en_do", "ru_do"),
            ("WHAT TO CHECK", "ЧТО ПРОВЕРИТЬ", "en_check", "ru_check")):
        rows.append([
            [Paragraph(label_en, S["label"]), Spacer(1, 1), para(step[key_en])],
            [Paragraph(label_ru, S["label"]), Spacer(1, 1), para(step[key_ru])],
        ])
    body = Table(rows, colWidths=[COL_W, COL_W])
    body.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (0, -1), 7),
        ("RIGHTPADDING", (0, 0), (0, -1), 3 * mm),
        ("LEFTPADDING", (1, 0), (1, -1), 3 * mm),
        ("RIGHTPADDING", (1, 0), (1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEBEFORE", (1, 0), (1, -1), 0.4, RULE),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
    ]))
    card = KeepTogether([head, body, Spacer(1, 5)])

    # Figures follow the card rather than living inside it: a step card plus a
    # tall screenshot rarely fits one page, and KeepTogether would then push the
    # whole thing over, leaving half a page empty.
    out = [card]
    # Captions are prefixed with the step, because a figure regularly lands on
    # the page after its card -- a reader opening at that page would otherwise
    # meet a picture with nothing saying which step it belongs to.
    tag_en = f'<font color="#2d5f8a"><b>Step {step["n"]} — {step["en_title"]}.</b></font> '
    tag_ru = f'<font color="#2d5f8a"><b>Шаг {step["n"]} — {step["ru_title"]}.</b></font> '
    for key, cap_en, cap_ru in (("fig", "fig_en", "fig_ru"),
                                ("fig2", "fig2_en", "fig2_ru")):
        name = step.get(key)
        if not name:
            continue
        fig = figure(name, tag_en + step.get(cap_en, ""), tag_ru + step.get(cap_ru, ""),
                     max_h=step.get(key + "_h", 92) * mm)
        if fig is not None:
            out.append(fig)
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def method_table():
    spec = C.METHOD_TABLE
    widths = [w for _, w in spec["head"]]
    total = sum(widths)
    widths = [BODY_W * w / total for w in widths]
    data = [[Paragraph(h, S["cellb"]) for h, _ in spec["head"]]]
    for row in spec["rows"]:
        data.append([Paragraph(c.replace("\n", "<br/>"), S["cell"]) for c in row])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT_2),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


RESULT_ROWS = [
    ("Low-frequency model only\nТолько низкочастотная модель", "0.319", "0.333", False),
    ("Coloured / Цветная", "0.337", "0.490", False),
    ("Sparse-spike / Разреженно-импульсная", "0.293", "0.374", False),
    ("Model-based / На основе модели", "0.384", "0.494", False),
    ("Bayesian / Байесовская", "0.395", "0.499", True),
]


def result_table():
    head = ["", "Bulk shift only\nТолько сдвиг", "With the stretch\nС растяжением"]
    data = [[Paragraph(h.replace("\n", "<br/>"), S["cellb"]) for h in head]]
    for name, a, b, best in RESULT_ROWS:
        st = "cellb" if best else "cell"
        data.append([Paragraph(name.replace("\n", "<br/>"), S[st]),
                     Paragraph(a, S[st]), Paragraph(b, S[st])])
    t = Table(data, colWidths=[BODY_W * 0.52, BODY_W * 0.24, BODY_W * 0.24], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT_2),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, len(RESULT_ROWS)), (-1, len(RESULT_ROWS)),
         colors.HexColor("#eef4fa")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def glossary_table():
    data = [[Paragraph("English", S["cellb"]), Paragraph("Русский", S["cellb"]),
             Paragraph("Meaning / Значение", S["cellb"])]]  # header stays at cell size
    for en, ru, mean_en, mean_ru in C.GLOSSARY:
        data.append([Paragraph(en, S["gloss"]), Paragraph(ru, S["gloss"]),
                     Paragraph(f'{mean_en}<br/><font color="#5b6673">{mean_ru}</font>',
                               S["gloss"])])
    # Nineteen rows of two-language glosses: the row padding is what decides
    # whether this lands on one page or spills a single row onto a second.
    t = Table(data, colWidths=[BODY_W * 0.25, BODY_W * 0.27, BODY_W * 0.48], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT_2),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 2.4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def workflow_map():
    """One-glance map of the eleven steps, grouped by what they are for."""
    groups = [
        ("PREPARE / ПОДГОТОВКА", ACCENT_2, [(1, "Data / Данные"),
                                            (2, "Seismic viewer / Просмотр"),
                                            (3, "Log QC / КК каротажа"),
                                            (4, "Correlation / Корреляция")]),
        ("CALIBRATE / КАЛИБРОВКА", colors.HexColor("#4a7c59"), [(5, "Well tie / Привязка"),
                                                                (6, "Wavelet / Импульс"),
                                                                (7, "Низкочастотная модель")]),
        ("INVERT / ИНВЕРСИЯ", ACCENT, [(8, "Inversion / Инверсия")]),
        ("VERIFY & USE / ПРОВЕРКА", colors.HexColor("#6b5b95"),
         [(9, "Validation / Валидация"),
          (10, "Rock property / Свойства породы"),
          (11, "Results / Результаты")]),
    ]
    rows = []
    for title, colour, items in groups:
        chips = []
        for n, label in items:
            chip = Table([[Paragraph(f"<b>{n}</b>&nbsp; {label}", S["cell"])]],
                         colWidths=[(BODY_W - 34 * mm) / 4 - 3])
            chip.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PANEL),
                ("BOX", (0, 0), (-1, -1), 0.4, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            chips.append(chip)
        while len(chips) < 4:
            chips.append("")
        band = Paragraph(f'<font color="#ffffff"><b>{title}</b></font>',
                         ParagraphStyle("g", fontName="DJV-B", fontSize=7, leading=9))
        inner = Table([chips], colWidths=[(BODY_W - 34 * mm) / 4] * 4)
        inner.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        rows.append([band, inner])
    t = Table(rows, colWidths=[34 * mm, BODY_W - 34 * mm])
    style_cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, -1), 6),
        ("RIGHTPADDING", (0, 0), (0, -1), 6),
        ("LEFTPADDING", (1, 0), (1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("BOX", (0, 0), (-1, -1), 0.6, RULE),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    for i, (_t, colour, _items) in enumerate(groups):
        style_cmds.append(("BACKGROUND", (0, i), (0, i), colour))
    t.setStyle(TableStyle(style_cmds))
    return t


# --------------------------------------------------------------------------
# Page furniture
# --------------------------------------------------------------------------

def cover_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#16283c"))
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, PAGE_H - 118 * mm, PAGE_W, 3, fill=1, stroke=0)
    # A quiet reference to a seismic section: stacked sinusoidal traces.
    canvas.setStrokeColor(colors.HexColor("#26405c"))
    canvas.setLineWidth(0.7)
    import math
    for k in range(26):
        y0 = 26 * mm + k * 2.6 * mm
        path = canvas.beginPath()
        path.moveTo(MARGIN, y0)
        for x in range(int(MARGIN), int(PAGE_W - MARGIN), 4):
            amp = 3.2 * mm * math.sin((x / 46.0) + k * 0.55) * math.cos(k * 0.22 + x / 320.0)
            path.lineTo(x, y0 + amp)
        canvas.drawPath(path)
    canvas.restoreState()


def content_page(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - MARGIN + 5 * mm, PAGE_W - MARGIN, PAGE_H - MARGIN + 5 * mm)
    canvas.setFont("DJV", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 6.6 * mm,
                      "Post-Stack Seismic Inversion — user guide")
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 6.6 * mm,
                           "Инверсия суммарных сейсмических данных — руководство")
    canvas.line(MARGIN, MARGIN - 4 * mm, PAGE_W - MARGIN, MARGIN - 4 * mm)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(PAGE_W / 2, MARGIN - 8.5 * mm, str(canvas.getPageNumber()))
    canvas.restoreState()


def build(out_path):
    doc = BaseDocTemplate(out_path, pagesize=A4,
                          leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN,
                          title="Post-Stack Seismic Inversion — user guide / руководство",
                          author="Seismic Inversion toolkit",
                          subject="Bilingual EN/RU user guide")
    frame = Frame(MARGIN, MARGIN, BODY_W, PAGE_H - 2 * MARGIN, id="f")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=cover_page),
        PageTemplate(id="body", frames=[frame], onPage=content_page),
    ])

    story = []

    # ---- cover -----------------------------------------------------------
    story.append(Spacer(1, 58 * mm))
    story.append(Paragraph(C.TITLE_EN, S["cover"]))
    story.append(Paragraph(C.TITLE_RU, S["coverru"]))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(C.SUB_EN, S["coversub"]))
    story.append(Paragraph(C.SUB_RU, S["coversub"]))
    story.append(Spacer(1, 14 * mm))
    story.append(Paragraph(
        '<font color="#8fa6bb">Eleven steps, four inversion methods, and the QC that '
        'makes the result defensible.<br/>'
        'Одиннадцать шагов, четыре метода инверсии и контроль качества, '
        'позволяющий защитить результат.</font>',
        ParagraphStyle("cn", fontName="DJV", fontSize=9.5, leading=14)))
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())

    # ---- overview --------------------------------------------------------
    render_blocks(C.OVERVIEW, story)

    story.append(Spacer(1, 8))
    story.append(Paragraph("The workflow at a glance", S["h2"]))
    story.append(Paragraph("Рабочий процесс в целом", S["h2ru"]))
    story.append(Spacer(1, 3))
    story.append(workflow_map())
    story.append(Spacer(1, 5))
    story.append(two_col(
        para("Follow the steps in order. Each consumes what the previous one produced, and "
             "changing something early clears everything downstream that depended on it.",
             "small"),
        para("Выполняйте шаги по порядку. Каждый использует результат предыдущего, а "
             "изменение на раннем шаге очищает все зависимые результаты.", "small")))
    story.append(Spacer(1, 12))

    # ---- the eleven steps ------------------------------------------------
    story.append(Paragraph("The eleven steps", S["h1"]))
    story.append(Paragraph("Одиннадцать шагов", S["h1ru"]))
    story.append(rule())
    for step in C.STEPS:
        story.extend(step_card(step))

    # ---- methods ---------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Choosing an inversion method", S["h1"]))
    story.append(Paragraph("Выбор метода инверсии", S["h1ru"]))
    story.append(rule())
    story.append(two_col(
        para("All four share one interface and one set of QC panels, so you can run several "
             "and compare them on the same evidence. Step 9 settles the choice on the only "
             "evidence that generalises: which engine best predicts a log it was not given."),
        para("Все четыре метода используют общий интерфейс и общие панели контроля качества, "
             "поэтому их можно запустить и сравнить на одних и тех же данных. Шаг 9 решает "
             "вопрос выбора по единственному обобщаемому критерию: какой алгоритм лучше "
             "предсказывает кривую, которая ему не была предоставлена.")))
    story.append(Spacer(1, 7))
    story.append(method_table())
    story.append(Spacer(1, 8))
    story.append(note_box(
        "Coloured inversion returns <b>relative</b> impedance — it has no absolute level. "
        "The other three splice onto the low-frequency model to give an absolute answer, so "
        "they need that model built first (step 7).",
        "Цветная инверсия даёт <b>относительный</b> импеданс — без абсолютного уровня. "
        "Остальные три метода сшиваются с низкочастотной моделью и дают абсолютные "
        "значения, поэтому сначала нужно построить эту модель (шаг 7)."))

    # ---- multi-attribute prediction --------------------------------------
    story.append(PageBreak())
    render_blocks(C.MULTIATTRIBUTE, story)

    # ---- pitfalls --------------------------------------------------------
    story.append(PageBreak())
    render_blocks(C.PITFALLS, story)

    # ---- results ---------------------------------------------------------
    story.append(PageBreak())
    render_blocks(C.RESULTS, story)

    # ---- glossary --------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Glossary", S["h1"]))
    story.append(Paragraph("Словарь терминов", S["h1ru"]))
    story.append(rule())
    story.append(glossary_table())

    doc.build(story)
    return out_path


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "guide.pdf"
    print("written:", build(out))
