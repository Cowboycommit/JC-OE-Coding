"""Generate reporting and visualization standards documentation."""

from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ProjectConfig
from utils.document import (
    create_document, set_document_defaults, add_heading,
    add_paragraph, add_bullet_point, add_table, save_document
)


def generate_reporting_standards(config: ProjectConfig) -> str:
    """Generate reporting and visualization standards documentation.

    Args:
        config: Project configuration

    Returns:
        Path to generated document
    """
    doc = create_document(config.formatting.margin_inches)
    set_document_defaults(doc, config.formatting.font_name, config.formatting.font_size_body)

    fmt = config.formatting
    colors = fmt.colors

    # Title
    add_heading(doc, f"{config.framework_name} - Reporting & Visualization Standards", level=0,
                font_name=fmt.font_name)

    add_paragraph(doc, f"Version: {config.version}", font_name=fmt.font_name,
                  font_size=fmt.font_size_body, italic=True)

    # 1. Introduction
    add_heading(doc, "1. Introduction", level=1, font_name=fmt.font_name)

    intro_text = config.custom_sections.get('reporting_intro') or (
        f"This document establishes standards for reporting and visualization within {config.framework_name}. "
        f"Consistent reporting ensures clarity, comparability, and professional presentation of results."
    )
    add_paragraph(doc, intro_text, font_name=fmt.font_name, font_size=fmt.font_size_body)

    # 2. Color Palette
    add_heading(doc, "2. Color Palette", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "Use the following standardized color palette for all visualizations:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    color_rows = [
        ["Primary", colors.get('primary', '#1f77b4'), "Main accent, headers, key data"],
        ["Secondary", colors.get('secondary', '#f0f2f6'), "Backgrounds, secondary elements"],
        ["Success", colors.get('success', '#28a745'), "Positive indicators, success states"],
        ["Warning", colors.get('warning', '#ffc107'), "Caution indicators, warnings"],
        ["Danger", colors.get('danger', '#dc3545'), "Error states, negative indicators"],
        ["Dark", colors.get('dark', '#262730'), "Text, borders, dark elements"],
        ["Light", colors.get('light', '#ffffff'), "Backgrounds, light elements"],
    ]
    add_table(doc, ["Color Name", "Hex Value", "Usage"], color_rows,
              header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 3. Typography
    add_heading(doc, "3. Typography Standards", level=1, font_name=fmt.font_name)

    add_heading(doc, "3.1 Document Typography", level=2, font_name=fmt.font_name)

    type_rows = [
        ["Title", fmt.font_name, str(fmt.font_size_title), "Bold"],
        ["Heading 1", fmt.font_name, str(fmt.font_size_h1), "Bold"],
        ["Heading 2", fmt.font_name, str(fmt.font_size_h2), "Bold"],
        ["Heading 3", fmt.font_name, str(fmt.font_size_h3), "Bold"],
        ["Body Text", fmt.font_name, str(fmt.font_size_body), "Regular"],
        ["Code", "Courier New", "10", "Regular"],
    ]
    add_table(doc, ["Element", "Font", "Size (pt)", "Weight"], type_rows,
              header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    add_heading(doc, "3.2 Visualization Typography", level=2, font_name=fmt.font_name)
    add_bullet_point(doc, "Chart titles: 14pt, bold", font_name=fmt.font_name)
    add_bullet_point(doc, "Axis labels: 11pt, regular", font_name=fmt.font_name)
    add_bullet_point(doc, "Tick labels: 10pt, regular", font_name=fmt.font_name)
    add_bullet_point(doc, "Legend text: 10pt, regular", font_name=fmt.font_name)
    add_bullet_point(doc, "Annotations: 9pt, italic", font_name=fmt.font_name)

    # 4. Chart Types
    add_heading(doc, "4. Recommended Chart Types", level=1, font_name=fmt.font_name)

    add_heading(doc, "4.1 Frequency Analysis", level=2, font_name=fmt.font_name)
    add_bullet_point(doc, "Bar charts for categorical frequency distributions", font_name=fmt.font_name)
    add_bullet_point(doc, "Horizontal bars when category labels are long", font_name=fmt.font_name)
    add_bullet_point(doc, "Include data labels for exact values", font_name=fmt.font_name)

    add_heading(doc, "4.2 Relationship Analysis", level=2, font_name=fmt.font_name)
    add_bullet_point(doc, "Heatmaps for correlation or co-occurrence matrices", font_name=fmt.font_name)
    add_bullet_point(doc, "Scatter plots for continuous variable relationships", font_name=fmt.font_name)
    add_bullet_point(doc, "Network diagrams for complex relationships", font_name=fmt.font_name)

    add_heading(doc, "4.3 Distribution Analysis", level=2, font_name=fmt.font_name)
    add_bullet_point(doc, "Histograms for continuous distributions", font_name=fmt.font_name)
    add_bullet_point(doc, "Box plots for comparing distributions across groups", font_name=fmt.font_name)
    add_bullet_point(doc, "Violin plots for detailed distribution shapes", font_name=fmt.font_name)

    add_heading(doc, "4.4 Trend Analysis", level=2, font_name=fmt.font_name)
    add_bullet_point(doc, "Line charts for time series data", font_name=fmt.font_name)
    add_bullet_point(doc, "Area charts for cumulative trends", font_name=fmt.font_name)
    add_bullet_point(doc, "Include confidence intervals where appropriate", font_name=fmt.font_name)

    # 5. Table Standards
    add_heading(doc, "5. Table Formatting Standards", level=1, font_name=fmt.font_name)

    add_heading(doc, "5.1 General Guidelines", level=2, font_name=fmt.font_name)
    add_bullet_point(doc, "Use header row with distinct background color", font_name=fmt.font_name)
    add_bullet_point(doc, "Align text left, numbers right", font_name=fmt.font_name)
    add_bullet_point(doc, "Use consistent decimal places", font_name=fmt.font_name)
    add_bullet_point(doc, "Include units in column headers", font_name=fmt.font_name)

    add_heading(doc, "5.2 Numeric Formatting", level=2, font_name=fmt.font_name)
    num_format = [
        ["Counts", "1,234", "Thousands separator, no decimals"],
        ["Percentages", "45.6%", "One decimal place"],
        ["Decimals", "0.123", "Three significant figures"],
        ["Large Numbers", "1.2M", "Abbreviate with suffix"],
        ["Currency", "$1,234.56", "Symbol, two decimals"],
    ]
    add_table(doc, ["Type", "Example", "Format Rule"], num_format,
              header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 6. Accessibility
    add_heading(doc, "6. Accessibility Guidelines", level=1, font_name=fmt.font_name)

    add_bullet_point(doc, "Ensure sufficient color contrast (WCAG 2.1 AA minimum)", font_name=fmt.font_name)
    add_bullet_point(doc, "Do not rely solely on color to convey information", font_name=fmt.font_name)
    add_bullet_point(doc, "Provide alt text for all images and charts", font_name=fmt.font_name)
    add_bullet_point(doc, "Use patterns or shapes in addition to colors", font_name=fmt.font_name)
    add_bullet_point(doc, "Ensure text is readable at standard zoom levels", font_name=fmt.font_name)
    add_bullet_point(doc, "Test visualizations with colorblind simulation tools", font_name=fmt.font_name)

    # 7. Export Specifications
    add_heading(doc, "7. Export Specifications", level=1, font_name=fmt.font_name)

    export_specs = [
        ["Print/PDF", "300 DPI", "CMYK", "Vector (PDF) preferred"],
        ["Web/Screen", "72-96 DPI", "RGB", "PNG for static, SVG for scalable"],
        ["Presentation", "150 DPI", "RGB", "PNG or embedded"],
        ["Raw Data", "N/A", "N/A", "CSV or Excel"],
    ]
    add_table(doc, ["Medium", "Resolution", "Color Space", "Format"], export_specs,
              header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 8. Report Structure
    add_heading(doc, "8. Standard Report Structure", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "All analysis reports should follow this structure:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    structure = [
        "Executive Summary (1 page maximum)",
        "Introduction and Objectives",
        "Methodology Overview",
        "Key Findings (with visualizations)",
        "Detailed Results",
        "Limitations and Considerations",
        "Conclusions and Recommendations",
        "Appendices (technical details, raw data)",
    ]
    for i, item in enumerate(structure, 1):
        add_bullet_point(doc, f"{i}. {item}", font_name=fmt.font_name)

    # Save document
    output_path = Path(config.project_root) / config.docs_dir / "Reporting_Visualization_Standards.docx"
    save_document(doc, str(output_path))

    return str(output_path)
