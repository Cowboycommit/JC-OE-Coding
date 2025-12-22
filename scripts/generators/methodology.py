"""Generate methodology documentation from project introspection."""

from pathlib import Path
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ProjectConfig
from utils.document import (
    create_document, set_document_defaults, add_heading,
    add_paragraph, add_bullet_point, add_table, add_code_block,
    save_document
)
from utils.introspection import discover_project, get_dependencies


def generate_methodology_documentation(config: ProjectConfig) -> str:
    """Generate methodology documentation based on project introspection.

    Args:
        config: Project configuration

    Returns:
        Path to generated document
    """
    # Discover project structure
    discovery = discover_project(
        config.project_root,
        config.source_dirs
    )
    dependencies = get_dependencies(config.project_root)

    # Create document
    doc = create_document(config.formatting.margin_inches)
    set_document_defaults(doc, config.formatting.font_name, config.formatting.font_size_body)

    fmt = config.formatting

    # Title
    add_heading(doc, f"{config.framework_name} - Methodology Documentation", level=0,
                font_name=fmt.font_name)

    # Version info
    add_paragraph(doc, f"Version: {config.version}", font_name=fmt.font_name,
                  font_size=fmt.font_size_body, italic=True)

    # 1. Introduction
    add_heading(doc, "1. Introduction", level=1, font_name=fmt.font_name)

    intro_text = config.custom_sections.get('methodology_intro') or (
        f"This document provides comprehensive methodology documentation for {config.framework_name}. "
        f"It covers the technical implementation, analytical approaches, and usage guidelines for the framework."
    )
    add_paragraph(doc, intro_text, font_name=fmt.font_name, font_size=fmt.font_size_body)

    if config.description:
        add_paragraph(doc, config.description, font_name=fmt.font_name, font_size=fmt.font_size_body)

    # 2. Project Structure
    add_heading(doc, "2. Project Structure", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "The project is organized into the following modules:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    for module in discovery['modules']:
        module_desc = module.docstring.split('\n')[0] if module.docstring else "No description available"
        add_bullet_point(doc, f"{module.name}: {module_desc}", font_name=fmt.font_name)

    # 3. Core Components
    add_heading(doc, "3. Core Components", level=1, font_name=fmt.font_name)

    classes = discovery['classes']
    if classes:
        add_heading(doc, "3.1 Classes", level=2, font_name=fmt.font_name)

        for cls in classes:
            add_heading(doc, cls.name, level=3, font_name=fmt.font_name)

            if cls.docstring:
                add_paragraph(doc, cls.docstring, font_name=fmt.font_name,
                              font_size=fmt.font_size_body)

            add_paragraph(doc, f"Location: {cls.module} (line {cls.line_number})",
                          font_name=fmt.font_name, font_size=fmt.font_size_body, italic=True)

            if cls.bases:
                add_paragraph(doc, f"Inherits from: {', '.join(cls.bases)}",
                              font_name=fmt.font_name, font_size=fmt.font_size_body)

            if cls.methods:
                # Filter to public methods
                public_methods = [m for m in cls.methods if not m.name.startswith('_') or m.name == '__init__']
                if public_methods:
                    add_paragraph(doc, "Methods:", font_name=fmt.font_name,
                                  font_size=fmt.font_size_body, bold=True)

                    method_rows = []
                    for method in public_methods[:10]:  # Limit to first 10
                        params = ', '.join([p['name'] for p in method.parameters])
                        desc = method.docstring.split('\n')[0] if method.docstring else "-"
                        method_rows.append([method.name, params, desc[:50]])

                    if method_rows:
                        add_table(doc, ["Method", "Parameters", "Description"], method_rows,
                                  header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 3.2 Key Functions
    functions = [f for f in discovery['functions'] if not f.name.startswith('_')]
    if functions:
        add_heading(doc, "3.2 Key Functions", level=2, font_name=fmt.font_name)

        func_rows = []
        for func in functions[:20]:  # Limit display
            params = ', '.join([p['name'] for p in func.parameters])
            desc = func.docstring.split('\n')[0] if func.docstring else "-"
            func_rows.append([func.name, func.module, desc[:40]])

        add_table(doc, ["Function", "Module", "Description"], func_rows,
                  header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 4. Dependencies
    add_heading(doc, "4. Dependencies", level=1, font_name=fmt.font_name)

    runtime_deps = [d for d in dependencies if d.category == 'runtime']
    dev_deps = [d for d in dependencies if d.category == 'dev']

    if runtime_deps:
        add_heading(doc, "4.1 Runtime Dependencies", level=2, font_name=fmt.font_name)

        dep_rows = []
        for dep in runtime_deps:
            version = dep.version or "any"
            desc = dep.description or "-"
            dep_rows.append([dep.name, version, desc])

        add_table(doc, ["Package", "Version", "Description"], dep_rows,
                  header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    if dev_deps:
        add_heading(doc, "4.2 Development Dependencies", level=2, font_name=fmt.font_name)

        dep_rows = []
        for dep in dev_deps:
            version = dep.version or "any"
            desc = dep.description or "-"
            dep_rows.append([dep.name, version, desc])

        add_table(doc, ["Package", "Version", "Description"], dep_rows,
                  header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 5. Usage Guidelines
    add_heading(doc, "5. Usage Guidelines", level=1, font_name=fmt.font_name)

    usage_text = config.custom_sections.get('usage_guidelines') or (
        f"This section provides guidelines for using {config.framework_name} effectively."
    )
    add_paragraph(doc, usage_text, font_name=fmt.font_name, font_size=fmt.font_size_body)

    # Add installation if we have dependencies
    if dependencies:
        add_heading(doc, "5.1 Installation", level=2, font_name=fmt.font_name)
        add_paragraph(doc, "Install the required dependencies using pip:",
                      font_name=fmt.font_name, font_size=fmt.font_size_body)
        add_code_block(doc, "pip install -r requirements.txt")

    # 6. API Reference
    add_heading(doc, "6. API Reference", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "This section provides detailed API documentation for the core classes and functions.",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    for cls in classes[:5]:  # Limit to top 5 classes
        add_heading(doc, f"6.{classes.index(cls)+1} {cls.name}", level=2, font_name=fmt.font_name)

        if cls.docstring:
            add_paragraph(doc, cls.docstring, font_name=fmt.font_name, font_size=fmt.font_size_body)

        for method in cls.methods:
            if not method.name.startswith('_') or method.name == '__init__':
                params = ', '.join([
                    f"{p['name']}: {p.get('type', 'Any')}" for p in method.parameters
                ])
                signature = f"{method.name}({params})"
                if method.return_type:
                    signature += f" -> {method.return_type}"

                add_paragraph(doc, signature, font_name="Courier New",
                              font_size=10, bold=True)

                if method.docstring:
                    add_paragraph(doc, method.docstring, font_name=fmt.font_name,
                                  font_size=fmt.font_size_body)

    # Save document
    output_path = Path(config.project_root) / config.docs_dir / "Methodology_Documentation.docx"
    save_document(doc, str(output_path))

    return str(output_path)
