"""
pdf_export.py — Converts the LLM/local report markdown text into a
professional-looking PDF using ReportLab and saves it to the target directory.
"""

import os
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


# ── Colour palette ──────────────────────────────────────────────────────────
CYAN   = colors.HexColor("#00d4ff")
BLUE   = colors.HexColor("#0066ff")
GREEN  = colors.HexColor("#00ff88")
PURPLE = colors.HexColor("#a855f7")
DARK   = colors.HexColor("#020b18")
PANEL  = colors.HexColor("#0a1628")
BORDER = colors.HexColor("#0d2a4a")
WHITE  = colors.HexColor("#e0f4ff")
MUTED  = colors.HexColor("#5a7a9a")
RED    = colors.HexColor("#ff6b6b")

PAGE_W, PAGE_H = A4


def _build_styles():
    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=CYAN,
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=10,
            textColor=MUTED,
            spaceAfter=12,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=CYAN,
            spaceBefore=14,
            spaceAfter=4,
            borderPad=4,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            textColor=WHITE,
            spaceAfter=4,
            leading=14,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=9,
            textColor=WHITE,
            spaceAfter=3,
            leftIndent=16,
            leading=14,
            bulletIndent=6,
        ),
        "code": ParagraphStyle(
            "code",
            fontName="Courier",
            fontSize=8,
            textColor=GREEN,
            spaceAfter=4,
            leftIndent=10,
            backColor=PANEL,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=CYAN,
            alignment=TA_CENTER,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            fontName="Helvetica",
            fontSize=8,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
        "footer": ParagraphStyle(
            "footer",
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }
    return styles


def _clean_emoji(text: str) -> str:
    """Remove emoji characters that ReportLab's built-in fonts can't render."""
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"
        "\U00002600-\U000027BF"
        "\U0001FA00-\U0001FA9F"
        "\u2639-\u263A"
        "\u2764"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text).strip()


def _escape_xml(text: str) -> str:
    """Escape XML special chars for ReportLab Paragraph."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def _parse_markdown_to_flowables(report_text: str, styles: dict) -> list:
    """
    Convert a simple markdown string into ReportLab flowables.
    Supports: # H1, ## H2, ### H3, - bullets, | tables, --- dividers, code `...`, bold **...**
    """
    flowables = []
    lines = report_text.split("\n")

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = _clean_emoji(raw)

        # --- Horizontal rule
        if re.match(r"^---+$", line.strip()):
            flowables.append(HRFlowable(width="100%", thickness=1, color=BORDER, spaceAfter=6))
            i += 1
            continue

        # --- H1
        if line.startswith("# "):
            text = _escape_xml(line[2:].strip())
            flowables.append(Paragraph(text, styles["h1"]))
            flowables.append(HRFlowable(width="100%", thickness=1, color=CYAN, spaceAfter=6))
            i += 1
            continue

        # --- H2
        if line.startswith("## "):
            text = _escape_xml(line[3:].strip())
            flowables.append(Paragraph(text, styles["h2"]))
            i += 1
            continue

        # --- H3
        if line.startswith("### "):
            text = _escape_xml(line[4:].strip())
            flowables.append(Paragraph(f"<b>{text}</b>", styles["body"]))
            i += 1
            continue

        # --- Markdown table (collect all rows)
        if "|" in line and line.strip().startswith("|"):
            table_lines = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip().startswith("|"):
                table_lines.append(_clean_emoji(lines[i]))
                i += 1

            # Filter out separator rows (|---|---|)
            data_rows = [
                r for r in table_lines
                if not re.match(r"^\s*\|[-| :]+\|\s*$", r)
            ]

            if data_rows:
                table_data = []
                for row_idx, row_line in enumerate(data_rows):
                    cells = [c.strip() for c in row_line.strip().strip("|").split("|")]
                    style = styles["table_header"] if row_idx == 0 else styles["table_cell"]
                    table_data.append([Paragraph(_escape_xml(c), style) for c in cells])

                col_count = max(len(r) for r in table_data)
                available = PAGE_W - 4 * cm
                col_w = available / col_count

                tbl = Table(table_data, colWidths=[col_w] * col_count, repeatRows=1)
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), PANEL),
                    ("BACKGROUND", (0, 1), (-1, -1), DARK),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK, PANEL]),
                    ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
                    ("TOPPADDING",    (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
                    ("LINEABOVE",     (0, 0), (-1, 0),  1, CYAN),
                    ("LINEBELOW",     (0, 0), (-1, 0),  1, CYAN),
                ]))
                flowables.append(tbl)
                flowables.append(Spacer(1, 8))
            continue

        # --- Bullet point
        if line.startswith("- ") or line.startswith("* "):
            text = line[2:].strip()
            # Convert **bold** inline
            text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _escape_xml(text))
            flowables.append(Paragraph(f"• {text}", styles["bullet"]))
            i += 1
            continue

        # --- Empty line
        if not line.strip():
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # --- Normal paragraph (with inline bold/code)
        text = _escape_xml(line)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`(.+?)`", r'<font name="Courier" color="#00ff88">\1</font>', text)
        flowables.append(Paragraph(text, styles["body"]))
        i += 1

    return flowables


def _on_first_page(canvas, doc):
    """Draw dark background on every page."""
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Top accent bar
    canvas.setFillColor(CYAN)
    canvas.rect(0, PAGE_H - 6, PAGE_W, 6, fill=1, stroke=0)
    # Bottom bar
    canvas.setFillColor(BORDER)
    canvas.rect(0, 0, PAGE_W, 20, fill=1, stroke=0)
    # Footer text
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(
        PAGE_W / 2, 7,
        f"DataDoctor AI  •  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  •  Page {doc.page}",
    )
    canvas.restoreState()


_on_later_pages = _on_first_page  # same treatment


def generate_pdf(
    report_text: str,
    pipeline_result: dict,
    output_dir: str,
    filename: str,
) -> str:
    """
    Generate a PDF report and save it to output_dir/filename.
    Returns the full path of the saved PDF.

    Parameters
    ----------
    report_text     : markdown-formatted report string
    pipeline_result : dict returned by run_pipeline()
    output_dir      : absolute path where the PDF will be saved
    filename        : PDF filename (e.g. 'DataDoctor_RandomForest_classification_....pdf')
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    styles = _build_styles()
    best = pipeline_result["best_model"]
    task = pipeline_result["task"]
    metric = "Accuracy" if task == "classification" else "R2 Score"

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    story = []

    # ── Cover block ──
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("DataDoctor AI", styles["title"]))
    story.append(Paragraph("Automated Machine Learning Pipeline Report", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=CYAN, spaceAfter=10))

    # ── Summary metrics table ──
    summary_data = [
        [
            Paragraph("BEST MODEL", styles["table_header"]),
            Paragraph(f"{metric}", styles["table_header"]),
            Paragraph("TASK", styles["table_header"]),
            Paragraph("CV FOLDS", styles["table_header"]),
        ],
        [
            Paragraph(best["model"], styles["table_cell"]),
            Paragraph(f"{best['cv_mean']:.4f} +/- {best['cv_std']:.4f}", styles["table_cell"]),
            Paragraph(task.upper(), styles["table_cell"]),
            Paragraph(str(pipeline_result["cv_folds"]), styles["table_cell"]),
        ],
    ]
    summary_tbl = Table(summary_data, colWidths=[(PAGE_W - 4 * cm) / 4] * 4)
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  PANEL),
        ("BACKGROUND",    (0, 1), (-1, -1), DARK),
        ("GRID",          (0, 0), (-1, -1), 0.5, BORDER),
        ("LINEABOVE",     (0, 0), (-1, 0),  2, CYAN),
        ("LINEBELOW",     (0, -1), (-1, -1), 1, CYAN),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 0.5 * cm))

    # ── Report body ──
    story += _parse_markdown_to_flowables(report_text, styles)

    doc.build(story, onFirstPage=_on_first_page, onLaterPages=_on_later_pages)
    return output_path
