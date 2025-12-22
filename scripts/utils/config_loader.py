"""Configuration loader for documentation toolkit."""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from pathlib import Path


@dataclass
class FormattingConfig:
    """Document formatting specifications."""
    font_name: str = "Times New Roman"
    font_size_body: int = 12
    font_size_h1: int = 16
    font_size_h2: int = 14
    font_size_h3: int = 12
    font_size_title: int = 18
    margin_inches: float = 0.75
    line_spacing: float = 1.5
    table_header_bg: str = "#1f77b4"
    table_header_fg: str = "#ffffff"
    colors: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#1f77b4",
        "secondary": "#f0f2f6",
        "success": "#28a745",
        "warning": "#ffc107",
        "danger": "#dc3545",
        "dark": "#262730",
        "light": "#ffffff",
    })


@dataclass
class DataFieldConfig:
    """Data field specifications for input data documentation."""
    required_fields: List[Dict[str, str]] = field(default_factory=list)
    optional_fields: List[Dict[str, str]] = field(default_factory=list)
    template_file: str = "input_data_template.xlsx"


@dataclass
class ProjectConfig:
    """Project configuration for documentation generation."""
    # Project identification
    project_name: str = "My Project"
    framework_name: str = "My Framework"
    description: str = ""
    version: str = "1.0"
    author: str = ""
    github_url: str = ""

    # Paths (relative to project root)
    project_root: str = "."
    source_dirs: List[str] = field(default_factory=lambda: ["src", "lib"])
    docs_dir: str = "documentation"
    data_dir: str = "data"

    # What to generate
    generate_methodology: bool = True
    generate_input_spec: bool = True
    generate_reporting_standards: bool = True
    generate_tools_review: bool = True
    generate_benchmark: bool = True

    # Nested configs
    formatting: FormattingConfig = field(default_factory=FormattingConfig)
    data_fields: DataFieldConfig = field(default_factory=DataFieldConfig)

    # Custom content sections (loaded from templates)
    custom_sections: Dict[str, Any] = field(default_factory=dict)

    # Auto-discovered metadata (populated at runtime)
    discovered_classes: List[Dict[str, Any]] = field(default_factory=list)
    discovered_functions: List[Dict[str, Any]] = field(default_factory=list)
    discovered_modules: List[Dict[str, Any]] = field(default_factory=list)
    dependencies: List[Dict[str, str]] = field(default_factory=list)

    def get_docs_path(self) -> Path:
        """Get absolute path to documentation directory."""
        return Path(self.project_root) / self.docs_dir

    def get_source_paths(self) -> List[Path]:
        """Get absolute paths to source directories."""
        root = Path(self.project_root)
        return [root / src for src in self.source_dirs if (root / src).exists()]


def load_config(config_path: str) -> ProjectConfig:
    """Load project configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ProjectConfig instance with loaded values
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    # Handle nested configurations
    formatting_data = data.pop('formatting', {})
    data_fields_data = data.pop('data_fields', {})

    # Build formatting config
    formatting = FormattingConfig(**formatting_data) if formatting_data else FormattingConfig()

    # Build data fields config
    data_fields = DataFieldConfig(**data_fields_data) if data_fields_data else DataFieldConfig()

    # Build main config
    config = ProjectConfig(
        formatting=formatting,
        data_fields=data_fields,
        **{k: v for k, v in data.items() if k in ProjectConfig.__dataclass_fields__}
    )

    # Resolve project root to absolute path
    if not os.path.isabs(config.project_root):
        config.project_root = str(config_path.parent / config.project_root)

    return config


def save_config_template(output_path: str) -> None:
    """Save a template configuration file.

    Args:
        output_path: Path to save the template YAML file
    """
    template = """# Documentation Toolkit Configuration
# Copy this file and customize for your project

# Project Identification
project_name: "My Project"
framework_name: "My Analysis Framework"
description: "A brief description of what your project does"
version: "1.0"
author: "Your Name or Organization"
github_url: "https://github.com/username/repo"

# Paths (relative to this config file's directory)
project_root: "."
source_dirs:
  - "src"
  - "lib"
  - "helpers"
docs_dir: "documentation"
data_dir: "data"

# Which documents to generate
generate_methodology: true
generate_input_spec: true
generate_reporting_standards: true
generate_tools_review: true
generate_benchmark: true

# Formatting specifications
formatting:
  font_name: "Times New Roman"
  font_size_body: 12
  font_size_h1: 16
  font_size_h2: 14
  font_size_h3: 12
  font_size_title: 18
  margin_inches: 0.75
  line_spacing: 1.5
  table_header_bg: "#1f77b4"
  table_header_fg: "#ffffff"
  colors:
    primary: "#1f77b4"
    secondary: "#f0f2f6"
    success: "#28a745"
    warning: "#ffc107"
    danger: "#dc3545"

# Data field specifications (for input data documentation)
data_fields:
  template_file: "input_data_template.xlsx"
  required_fields:
    - name: "response"
      type: "string"
      description: "The text response to analyze"
  optional_fields:
    - name: "id"
      type: "string/integer"
      description: "Unique identifier for each record"
    - name: "timestamp"
      type: "datetime"
      description: "When the response was collected"

# Custom content sections (optional overrides)
custom_sections:
  methodology_intro: |
    This section can contain custom introduction text
    that will be used in the methodology documentation.

  analysis_description: |
    Describe your specific analysis approach here.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)

    print(f"Template configuration saved to: {output_path}")
