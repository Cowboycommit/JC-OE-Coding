"""Shared formatting utilities for Word documents."""

from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from typing import Optional, Tuple


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple.

    Args:
        hex_color: Color in hex format (e.g., "#1f77b4" or "1f77b4")

    Returns:
        Tuple of (red, green, blue) integers
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def set_cell_margins(cell, top: int = 100, bottom: int = 100,
                     left: int = 180, right: int = 180) -> None:
    """Set cell margins in twentieths of a point.

    Args:
        cell: Table cell to modify
        top: Top margin in twips
        bottom: Bottom margin in twips
        left: Left margin in twips
        right: Right margin in twips
    """
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')

    for margin_name, value in [('top', top), ('bottom', bottom),
                                ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{margin_name}')
        node.set(qn('w:w'), str(value))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)

    tcPr.append(tcMar)


def set_cell_border(cell, top: Optional[dict] = None, bottom: Optional[dict] = None,
                    left: Optional[dict] = None, right: Optional[dict] = None) -> None:
    """Set cell borders with customizable styles.

    Args:
        cell: Table cell to modify
        top, bottom, left, right: Border specs with keys 'sz', 'val', 'color'
    """
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')

    border_specs = {
        'top': top or {'sz': 4, 'val': 'single', 'color': '000000'},
        'bottom': bottom or {'sz': 4, 'val': 'single', 'color': '000000'},
        'left': left or {'sz': 4, 'val': 'single', 'color': '000000'},
        'right': right or {'sz': 4, 'val': 'single', 'color': '000000'},
    }

    for border_name, spec in border_specs.items():
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:sz'), str(spec.get('sz', 4)))
        border.set(qn('w:val'), spec.get('val', 'single'))
        border.set(qn('w:color'), spec.get('color', '000000').lstrip('#'))
        tcBorders.append(border)

    tcPr.append(tcBorders)


def set_cell_background(cell, color: str) -> None:
    """Set cell background color.

    Args:
        cell: Table cell to modify
        color: Hex color string (e.g., "#1f77b4")
    """
    tc = cell._element
    tcPr = tc.get_or_add_tcPr()
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), color.lstrip('#'))
    shading.set(qn('w:val'), 'clear')
    tcPr.append(shading)


def format_paragraph(paragraph, font_name: str = "Times New Roman",
                     font_size: int = 12, bold: bool = False,
                     italic: bool = False, alignment: str = "left",
                     space_before: int = 0, space_after: int = 6,
                     line_spacing: float = 1.5,
                     first_line_indent: Optional[float] = None) -> None:
    """Apply comprehensive formatting to a paragraph.

    Args:
        paragraph: Paragraph to format
        font_name: Font family name
        font_size: Font size in points
        bold: Whether text is bold
        italic: Whether text is italic
        alignment: Text alignment ('left', 'center', 'right', 'justify')
        space_before: Space before paragraph in points
        space_after: Space after paragraph in points
        line_spacing: Line spacing multiplier
        first_line_indent: First line indent in inches
    """
    # Set paragraph formatting
    pf = paragraph.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line_spacing

    if first_line_indent is not None:
        pf.first_line_indent = Inches(first_line_indent)

    # Set alignment
    alignments = {
        'left': WD_ALIGN_PARAGRAPH.LEFT,
        'center': WD_ALIGN_PARAGRAPH.CENTER,
        'right': WD_ALIGN_PARAGRAPH.RIGHT,
        'justify': WD_ALIGN_PARAGRAPH.JUSTIFY,
    }
    pf.alignment = alignments.get(alignment, WD_ALIGN_PARAGRAPH.LEFT)

    # Set run formatting
    for run in paragraph.runs:
        run.font.name = font_name
        run.font.size = Pt(font_size)
        run.font.bold = bold
        run.font.italic = italic
        run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)


def format_table(table, header_bg: str = "#1f77b4", header_fg: str = "#ffffff",
                 font_name: str = "Times New Roman", font_size: int = 11,
                 header_font_size: int = 11, align_center: bool = True) -> None:
    """Apply comprehensive formatting to a table.

    Args:
        table: Table to format
        header_bg: Header background color (hex)
        header_fg: Header text color (hex)
        font_name: Font family name
        font_size: Body font size in points
        header_font_size: Header font size in points
        align_center: Whether to center-align the table
    """
    if align_center:
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

    header_rgb = hex_to_rgb(header_fg)

    for row_idx, row in enumerate(table.rows):
        is_header = row_idx == 0

        for cell in row.cells:
            # Set borders and margins
            set_cell_border(cell)
            set_cell_margins(cell)

            # Header row styling
            if is_header:
                set_cell_background(cell, header_bg)

            # Format cell paragraphs
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.name = font_name
                    run.font.size = Pt(header_font_size if is_header else font_size)
                    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)

                    if is_header:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(*header_rgb)


def set_document_margins(document, margin_inches: float = 0.75) -> None:
    """Set document margins for all sections.

    Args:
        document: Document to modify
        margin_inches: Margin size in inches
    """
    for section in document.sections:
        section.left_margin = Inches(margin_inches)
        section.right_margin = Inches(margin_inches)
        section.top_margin = Inches(margin_inches)
        section.bottom_margin = Inches(margin_inches)
