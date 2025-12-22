#!/usr/bin/env python3
"""
Documentation Generator CLI

A generic documentation toolkit that automatically generates project documentation
by introspecting the codebase and using configuration files.

Usage:
    python generate_docs.py --config project_config.yaml
    python generate_docs.py --init  # Create template config
    python generate_docs.py --config project_config.yaml --only methodology
"""

import argparse
import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config, save_config_template, ProjectConfig
from utils.introspection import discover_project, get_dependencies
from generators.methodology import generate_methodology_documentation
from generators.input_spec import generate_input_specification
from generators.reporting import generate_reporting_standards
from generators.tools_review import generate_tools_review
from generators.benchmark import generate_benchmark_standards


def print_banner():
    """Print CLI banner."""
    print("=" * 60)
    print("  Documentation Generator Toolkit")
    print("  Automatic documentation from code introspection")
    print("=" * 60)
    print()


def print_discovery_summary(config: ProjectConfig):
    """Print summary of discovered project elements."""
    print("Project Discovery Summary")
    print("-" * 40)

    discovery = discover_project(config.project_root, config.source_dirs)
    dependencies = get_dependencies(config.project_root)

    print(f"  Modules found:    {len(discovery['modules'])}")
    print(f"  Classes found:    {len(discovery['classes'])}")
    print(f"  Functions found:  {len(discovery['functions'])}")
    print(f"  Dependencies:     {len(dependencies)}")
    print()


def generate_all(config: ProjectConfig, only: str = None) -> dict:
    """Generate all configured documentation.

    Args:
        config: Project configuration
        only: If specified, only generate this document type

    Returns:
        Dictionary mapping document type to output path
    """
    results = {}

    generators = {
        'methodology': (config.generate_methodology, generate_methodology_documentation),
        'input_spec': (config.generate_input_spec, generate_input_specification),
        'reporting': (config.generate_reporting_standards, generate_reporting_standards),
        'tools_review': (config.generate_tools_review, generate_tools_review),
        'benchmark': (config.generate_benchmark, generate_benchmark_standards),
    }

    for doc_type, (enabled, generator) in generators.items():
        if only and doc_type != only:
            continue

        if enabled or only == doc_type:
            print(f"Generating {doc_type}...", end=" ")
            try:
                path = generator(config)
                results[doc_type] = path
                print(f"✓ {path}")
            except Exception as e:
                print(f"✗ Error: {e}")
                results[doc_type] = None

    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate project documentation automatically",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --init                          Create template configuration
  %(prog)s --config docs_config.yaml       Generate all documentation
  %(prog)s --config docs_config.yaml --only methodology
                                            Generate only methodology doc
  %(prog)s --config docs_config.yaml --list
                                            Show project discovery summary

Document types for --only:
  methodology   - Methodology and API documentation
  input_spec    - Input data specification
  reporting     - Reporting and visualization standards
  tools_review  - Open-source tools review
  benchmark     - Benchmark standards with technique references
        """
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to project configuration YAML file'
    )

    parser.add_argument(
        '--init',
        action='store_true',
        help='Create a template configuration file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='docs_config.yaml',
        help='Output path for --init (default: docs_config.yaml)'
    )

    parser.add_argument(
        '--only',
        type=str,
        choices=['methodology', 'input_spec', 'reporting', 'tools_review', 'benchmark'],
        help='Generate only the specified document type'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List discovered project elements without generating'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress banner and verbose output'
    )

    args = parser.parse_args()

    if not args.quiet:
        print_banner()

    # Handle --init
    if args.init:
        save_config_template(args.output)
        print(f"\nNext steps:")
        print(f"  1. Edit {args.output} with your project details")
        print(f"  2. Run: python {sys.argv[0]} --config {args.output}")
        return 0

    # Require config for other operations
    if not args.config:
        parser.print_help()
        print("\nError: --config is required (or use --init to create one)")
        return 1

    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
        print(f"Project: {config.project_name} v{config.version}")
        print(f"Root: {config.project_root}")
        print()
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {args.config}")
        print(f"Run with --init to create a template configuration")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Ensure documentation directory exists
    docs_path = Path(config.project_root) / config.docs_dir
    docs_path.mkdir(parents=True, exist_ok=True)

    # Handle --list
    if args.list:
        print_discovery_summary(config)
        return 0

    # Generate documentation
    print("Generating Documentation")
    print("-" * 40)

    results = generate_all(config, args.only)

    # Summary
    print()
    print("Generation Complete")
    print("-" * 40)
    success = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"  Generated: {success}/{total} documents")

    if success < total:
        print("  Some documents failed to generate. Check errors above.")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
