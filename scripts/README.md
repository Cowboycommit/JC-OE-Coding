# Documentation Generator Toolkit

A generic, portable documentation generator that automatically creates project documentation by introspecting the codebase.

## Features

- **Automatic Code Discovery**: Scans your codebase to find classes, functions, and modules
- **Dependency Detection**: Reads requirements.txt and pyproject.toml to document dependencies
- **Configurable Output**: YAML-based configuration for project-specific customization
- **Multiple Document Types**: Generates methodology, input specs, reporting standards, and tools review
- **Portable**: Copy to any Python project with similar structure

## Quick Start

### 1. Create Configuration

```bash
python scripts/generate_docs.py --init
```

This creates a `docs_config.yaml` template. Edit it with your project details.

### 2. Generate Documentation

```bash
python scripts/generate_docs.py --config docs_config.yaml
```

### 3. Generate Specific Document

```bash
python scripts/generate_docs.py --config docs_config.yaml --only methodology
```

## Configuration

The YAML configuration file supports:

```yaml
# Project identification
project_name: "My Project"
framework_name: "My Framework"
version: "1.0"
github_url: "https://github.com/..."

# Paths
project_root: "."
source_dirs: ["src", "lib"]
docs_dir: "documentation"

# What to generate
generate_methodology: true
generate_input_spec: true
generate_reporting_standards: true
generate_tools_review: true

# Formatting
formatting:
  font_name: "Times New Roman"
  font_size_body: 12
  # ... more options

# Data field specs (for input spec doc)
data_fields:
  required_fields:
    - name: "response"
      type: "string"
      description: "Text to analyze"
```

## Document Types

| Type | Description |
|------|-------------|
| `methodology` | Technical documentation, API reference, class/function docs |
| `input_spec` | Input data format requirements and examples |
| `reporting` | Visualization and reporting standards |
| `tools_review` | Review of dependencies and their licenses |

## Project Structure

```
scripts/
├── generate_docs.py      # Main CLI entry point
├── config/
│   └── project_config.yaml  # Project-specific config
├── generators/           # Document generators
│   ├── methodology.py
│   ├── input_spec.py
│   ├── reporting.py
│   └── tools_review.py
├── utils/                # Shared utilities
│   ├── config_loader.py  # YAML config loading
│   ├── document.py       # Word document helpers
│   ├── formatting.py     # Formatting utilities
│   └── introspection.py  # Code analysis
└── templates/            # Custom templates (optional)
```

## Using in Another Project

1. Copy the entire `scripts/` folder to your project
2. Run `python scripts/generate_docs.py --init` to create a config
3. Edit the config with your project details
4. Run `python scripts/generate_docs.py --config your_config.yaml`

## Requirements

- Python 3.8+
- python-docx
- pyyaml
