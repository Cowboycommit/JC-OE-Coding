"""Documentation toolkit utilities."""

from .formatting import (
    set_cell_border,
    set_cell_margins,
    set_cell_background,
    format_paragraph,
    format_table,
)
from .document import (
    create_document,
    add_heading,
    add_paragraph,
    add_bullet_point,
    add_table,
    set_document_defaults,
)
from .introspection import (
    discover_project,
    get_classes,
    get_functions,
    get_modules,
    get_dependencies,
)
from .config_loader import load_config, ProjectConfig

__all__ = [
    # Formatting
    "set_cell_border",
    "set_cell_margins",
    "set_cell_background",
    "format_paragraph",
    "format_table",
    # Document
    "create_document",
    "add_heading",
    "add_paragraph",
    "add_bullet_point",
    "add_table",
    "set_document_defaults",
    # Introspection
    "discover_project",
    "get_classes",
    "get_functions",
    "get_modules",
    "get_dependencies",
    # Config
    "load_config",
    "ProjectConfig",
]
