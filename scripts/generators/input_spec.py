"""Generate input data specification documentation."""

from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ProjectConfig
from utils.document import (
    create_document, set_document_defaults, add_heading,
    add_paragraph, add_bullet_point, add_table, add_code_block,
    save_document
)


def generate_input_specification(config: ProjectConfig) -> str:
    """Generate input data specification documentation.

    Args:
        config: Project configuration

    Returns:
        Path to generated document
    """
    doc = create_document(config.formatting.margin_inches)
    set_document_defaults(doc, config.formatting.font_name, config.formatting.font_size_body)

    fmt = config.formatting
    df = config.data_fields

    # Title
    add_heading(doc, f"{config.framework_name} - Input Data Specification", level=0,
                font_name=fmt.font_name)

    add_paragraph(doc, f"Version: {config.version}", font_name=fmt.font_name,
                  font_size=fmt.font_size_body, italic=True)

    # 1. Overview
    add_heading(doc, "1. Overview", level=1, font_name=fmt.font_name)

    overview_text = config.custom_sections.get('input_spec_overview') or (
        f"This document specifies the input data requirements for {config.framework_name}. "
        f"Following these specifications ensures proper data processing and analysis."
    )
    add_paragraph(doc, overview_text, font_name=fmt.font_name, font_size=fmt.font_size_body)

    # 2. Supported Formats
    add_heading(doc, "2. Supported File Formats", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "The framework accepts data in the following formats:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    formats = [
        ["CSV", ".csv", "Comma-separated values, UTF-8 encoded"],
        ["Excel", ".xlsx, .xls", "Microsoft Excel workbooks"],
        ["JSON", ".json", "JSON arrays or objects with records"],
        ["Parquet", ".parquet", "Apache Parquet columnar format"],
    ]
    add_table(doc, ["Format", "Extensions", "Description"], formats,
              header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 3. Required Fields
    add_heading(doc, "3. Required Fields", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "The following fields are required in all input data:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    if df.required_fields:
        req_rows = []
        for field in df.required_fields:
            req_rows.append([
                field.get('name', ''),
                field.get('type', 'string'),
                field.get('description', '')
            ])
        add_table(doc, ["Field Name", "Type", "Description"], req_rows,
                  header_bg=fmt.table_header_bg, font_name=fmt.font_name)
    else:
        # Default required fields
        add_table(doc, ["Field Name", "Type", "Description"], [
            ["response", "string", "The text content to be analyzed"]
        ], header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 4. Optional Fields
    add_heading(doc, "4. Optional Fields", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "The following optional fields can enhance analysis:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    if df.optional_fields:
        opt_rows = []
        for field in df.optional_fields:
            opt_rows.append([
                field.get('name', ''),
                field.get('type', 'string'),
                field.get('description', '')
            ])
        add_table(doc, ["Field Name", "Type", "Description"], opt_rows,
                  header_bg=fmt.table_header_bg, font_name=fmt.font_name)
    else:
        # Default optional fields
        default_optional = [
            ["id", "string/integer", "Unique identifier for each record"],
            ["timestamp", "datetime", "When the data was collected"],
            ["category", "string", "Category or grouping variable"],
            ["weight", "float", "Sampling weight for weighted analysis"],
        ]
        add_table(doc, ["Field Name", "Type", "Description"], default_optional,
                  header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 5. Data Quality Requirements
    add_heading(doc, "5. Data Quality Requirements", level=1, font_name=fmt.font_name)

    add_heading(doc, "5.1 Text Content", level=2, font_name=fmt.font_name)
    add_bullet_point(doc, "Text fields should be UTF-8 encoded", font_name=fmt.font_name)
    add_bullet_point(doc, "Avoid excessive use of special characters", font_name=fmt.font_name)
    add_bullet_point(doc, "Empty or null values will be excluded from analysis", font_name=fmt.font_name)
    add_bullet_point(doc, "Minimum recommended text length: 10 characters", font_name=fmt.font_name)

    add_heading(doc, "5.2 Missing Values", level=2, font_name=fmt.font_name)
    add_paragraph(doc, "Missing values should be represented as:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)
    add_bullet_point(doc, "Empty cells (CSV/Excel)", font_name=fmt.font_name)
    add_bullet_point(doc, "null values (JSON)", font_name=fmt.font_name)
    add_bullet_point(doc, "NA or NaN (where supported)", font_name=fmt.font_name)

    # 6. File Size Limits
    add_heading(doc, "6. File Size Recommendations", level=1, font_name=fmt.font_name)

    size_limits = [
        ["Small", "< 10,000 rows", "Immediate processing"],
        ["Medium", "10,000 - 100,000 rows", "Standard processing"],
        ["Large", "100,000 - 1,000,000 rows", "Batch processing recommended"],
        ["Very Large", "> 1,000,000 rows", "Chunked processing required"],
    ]
    add_table(doc, ["Dataset Size", "Row Count", "Processing Mode"], size_limits,
              header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 7. Example Data
    add_heading(doc, "7. Example Data Format", level=1, font_name=fmt.font_name)

    add_heading(doc, "7.1 CSV Example", level=2, font_name=fmt.font_name)
    csv_example = '''id,response,timestamp,category
1,"This is the first response text.",2024-01-15,feedback
2,"Another example of input data.",2024-01-16,survey
3,"Third response for demonstration.",2024-01-17,feedback'''
    add_code_block(doc, csv_example)

    add_heading(doc, "7.2 JSON Example", level=2, font_name=fmt.font_name)
    json_example = '''[
  {"id": 1, "response": "This is the first response text.", "category": "feedback"},
  {"id": 2, "response": "Another example of input data.", "category": "survey"},
  {"id": 3, "response": "Third response for demonstration.", "category": "feedback"}
]'''
    add_code_block(doc, json_example)

    # 8. Data Preparation Checklist
    add_heading(doc, "8. Data Preparation Checklist", level=1, font_name=fmt.font_name)

    checklist = [
        "Ensure all required fields are present",
        "Verify text encoding is UTF-8",
        "Remove or handle duplicate records",
        "Check for and handle missing values",
        "Validate date/time formats if applicable",
        "Review data for sensitive information",
        "Test with a small sample before full processing",
    ]
    for item in checklist:
        add_bullet_point(doc, item, font_name=fmt.font_name)

    # Save document
    output_path = Path(config.project_root) / config.docs_dir / "Input_Data_Specification.docx"
    save_document(doc, str(output_path))

    return str(output_path)
