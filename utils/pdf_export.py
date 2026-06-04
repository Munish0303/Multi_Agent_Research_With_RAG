"""
Convert a markdown research report to a clean, readable PDF.
Compatible with fpdf2 >= 2.7  (tested on 2.8.x).
"""

import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos


# ---------------------------------------------------------------------------
# PDF class with header / footer
# ---------------------------------------------------------------------------

class ReportPDF(FPDF):
    def __init__(self, title: str = "Research Report"):
        super().__init__()
        # title is rendered in the header on every page — sanitise it too
        self.report_title = _to_latin1(title)
        self.set_margins(left=20, top=20, right=20)

    def header(self):
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 6, self.report_title, align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(220, 220, 220)
        self.line(self.l_margin, self.get_y(),
                  self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(160, 160, 160)
        self.cell(0, 8, f"Page {self.page_no()}",
                  align="C", new_x=XPos.RIGHT, new_y=YPos.TOP)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Unicode characters LLMs commonly emit that the built-in latin-1 Helvetica
# font cannot encode. Mapped to safe ASCII equivalents so PDF export never
# crashes on smart quotes, dashes, arrows, symbols, etc.
_UNICODE_MAP = {
    "—": "-",  "–": "-",  "―": "-",  "‒": "-",   # — – ― ‒
    "‐": "-",  "‑": "-",                                    # ‐ ‑
    "‘": "'",  "’": "'",  "‚": "'",  "‛": "'",   # ‘ ’ ‚ ‛
    "“": '"',  "”": '"',  "„": '"',  "‟": '"',   # “ ” „ ‟
    "…": "...",                                                  # …
    "•": "*",  "◦": "-",  "·": "-",  "‣": "*",   # • ◦ · ‣
    "→": "->", "←": "<-", "↔": "<->","⇒": "=>",  # → ← ↔ ⇒
    "°": " deg", "′": "'", "″": '"',                  # ° ′ ″
    "€": "EUR ", "£": "GBP ", "¥": "JPY ",            # € £ ¥
    "©": "(c)", "®": "(r)", "™": "(tm)",              # © ® ™
    "×": "x",  "÷": "/",  "±": "+/-",                 # × ÷ ±
    "≈": "~",  "≤": "<=", "≥": ">=", "≠": "!=",  # ≈ ≤ ≥ ≠
    "½": "1/2","¼": "1/4","¾": "3/4",                 # ½ ¼ ¾
    " ": " ",  "​": "",   "﻿": "",   " ": " ",   # nbsp, zwsp, bom, thin
    "−": "-",                                                    # − minus sign
}


def _to_latin1(text: str) -> str:
    """Replace common Unicode chars with ASCII, then drop anything still
    outside latin-1 so fpdf's Helvetica font can always render the text."""
    for uni, ascii_eq in _UNICODE_MAP.items():
        text = text.replace(uni, ascii_eq)
    # Final safety net: discard any remaining non-latin-1 characters
    return text.encode("latin-1", "ignore").decode("latin-1")


def _strip_inline(text: str) -> str:
    """Remove common inline markdown and normalise Unicode for the latin-1 font."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)    # **bold**
    text = re.sub(r"\*(.+?)\*",     r"\1", text)     # *italic*
    text = re.sub(r"`(.+?)`",       r"\1", text)     # `code`
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # [link](url)
    return _to_latin1(text)


def _write(pdf: FPDF, text: str, line_height: float = 6) -> None:
    """Write a body paragraph, resetting colour/font safely."""
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, line_height, _strip_inline(text),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export_pdf(markdown_text: str, output_path: str, topic: str) -> None:
    """
    Render *markdown_text* as a PDF and write it to *output_path*.

    Supported markdown elements:
      # H1  ## H2  ### H3  - lists  * lists  1. numbered  --- rule  plain text
    """
    short_title = f"Research: {topic[:60]}"
    pdf = ReportPDF(title=short_title)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ── Cover block ───────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(25, 70, 150)
    pdf.cell(0, 12, "Research Report",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(70, 70, 70)
    # Wrap long topics manually (sanitise Unicode for the latin-1 font)
    pdf.multi_cell(0, 8, _to_latin1(topic),
                   align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_draw_color(25, 70, 150)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.set_line_width(0.2)
    pdf.set_draw_color(220, 220, 220)
    pdf.ln(6)

    # ── Body ──────────────────────────────────────────────────────────────
    for raw in markdown_text.splitlines():
        line = raw.rstrip()

        # H1
        if re.match(r"^# ", line):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 15)
            pdf.set_text_color(25, 70, 150)
            pdf.multi_cell(0, 9, _strip_inline(line[2:]),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_draw_color(180, 200, 235)
            pdf.line(pdf.l_margin, pdf.get_y(),
                     pdf.w - pdf.r_margin, pdf.get_y())
            pdf.set_draw_color(220, 220, 220)
            pdf.ln(3)

        # H2
        elif re.match(r"^## ", line):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(35, 90, 170)
            pdf.multi_cell(0, 8, _strip_inline(line[3:]),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

        # H3
        elif re.match(r"^### ", line):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 7, _strip_inline(line[4:]),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)

        # Indented sub-bullet (2+ spaces then - or *)
        elif re.match(r"^\s{2,}[\-\*] ", line):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(80, 80, 80)
            text = re.sub(r"^\s+[\-\*] ", "", line)
            pdf.set_x(pdf.l_margin + 8)
            pdf.multi_cell(0, 6, f"  - {_strip_inline(text)}",
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Unordered list (- or *)
        elif re.match(r"^[\-\*] ", line):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(55, 55, 55)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(0, 6, f"* {_strip_inline(line[2:])}",
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Ordered list
        elif re.match(r"^\d+\. ", line):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(55, 55, 55)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(0, 6, _strip_inline(line),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Horizontal rule
        elif re.match(r"^-{3,}$", line) or re.match(r"^={3,}$", line):
            pdf.ln(3)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.l_margin, pdf.get_y(),
                     pdf.w - pdf.r_margin, pdf.get_y())
            pdf.set_draw_color(220, 220, 220)
            pdf.ln(4)

        # Blank line
        elif line.strip() == "":
            pdf.ln(3)

        # Plain paragraph
        else:
            _write(pdf, line)

    pdf.output(output_path)
