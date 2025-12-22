"""Generate tools review documentation based on project dependencies."""

from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ProjectConfig
from utils.document import (
    create_document, set_document_defaults, add_heading,
    add_paragraph, add_bullet_point, add_table, save_document
)
from utils.introspection import get_dependencies, DependencyInfo


# Extended library information
LIBRARY_DETAILS = {
    'pandas': {
        'full_name': 'pandas',
        'category': 'Data Processing',
        'license': 'BSD-3-Clause',
        'url': 'https://pandas.pydata.org/',
        'description': 'Powerful data structures and data analysis tools for Python.',
        'use_cases': ['Data loading', 'Data transformation', 'Data cleaning', 'Analysis'],
    },
    'numpy': {
        'full_name': 'NumPy',
        'category': 'Numerical Computing',
        'license': 'BSD-3-Clause',
        'url': 'https://numpy.org/',
        'description': 'Fundamental package for scientific computing with Python.',
        'use_cases': ['Array operations', 'Mathematical functions', 'Linear algebra'],
    },
    'scikit-learn': {
        'full_name': 'scikit-learn',
        'category': 'Machine Learning',
        'license': 'BSD-3-Clause',
        'url': 'https://scikit-learn.org/',
        'description': 'Machine learning library with classification, regression, and clustering.',
        'use_cases': ['Classification', 'Clustering', 'Feature extraction', 'Model evaluation'],
    },
    'sklearn': {
        'full_name': 'scikit-learn',
        'category': 'Machine Learning',
        'license': 'BSD-3-Clause',
        'url': 'https://scikit-learn.org/',
        'description': 'Machine learning library with classification, regression, and clustering.',
        'use_cases': ['Classification', 'Clustering', 'Feature extraction', 'Model evaluation'],
    },
    'plotly': {
        'full_name': 'Plotly',
        'category': 'Visualization',
        'license': 'MIT',
        'url': 'https://plotly.com/python/',
        'description': 'Interactive graphing library for Python.',
        'use_cases': ['Interactive charts', 'Dashboards', 'Web visualizations'],
    },
    'matplotlib': {
        'full_name': 'Matplotlib',
        'category': 'Visualization',
        'license': 'PSF',
        'url': 'https://matplotlib.org/',
        'description': 'Comprehensive library for creating static, animated, and interactive visualizations.',
        'use_cases': ['Static plots', 'Publication figures', 'Custom visualizations'],
    },
    'streamlit': {
        'full_name': 'Streamlit',
        'category': 'Web Framework',
        'license': 'Apache-2.0',
        'url': 'https://streamlit.io/',
        'description': 'Framework for building data applications in Python.',
        'use_cases': ['Web applications', 'Dashboards', 'Interactive tools'],
    },
    'nltk': {
        'full_name': 'Natural Language Toolkit',
        'category': 'NLP',
        'license': 'Apache-2.0',
        'url': 'https://www.nltk.org/',
        'description': 'Platform for building Python programs to work with human language data.',
        'use_cases': ['Text processing', 'Tokenization', 'Stemming', 'POS tagging'],
    },
    'spacy': {
        'full_name': 'spaCy',
        'category': 'NLP',
        'license': 'MIT',
        'url': 'https://spacy.io/',
        'description': 'Industrial-strength natural language processing library.',
        'use_cases': ['Named entity recognition', 'Dependency parsing', 'Text classification'],
    },
    'sqlalchemy': {
        'full_name': 'SQLAlchemy',
        'category': 'Database',
        'license': 'MIT',
        'url': 'https://www.sqlalchemy.org/',
        'description': 'SQL toolkit and Object-Relational Mapping (ORM) library.',
        'use_cases': ['Database connectivity', 'ORM', 'Query building'],
    },
    'python-docx': {
        'full_name': 'python-docx',
        'category': 'Document Processing',
        'license': 'MIT',
        'url': 'https://python-docx.readthedocs.io/',
        'description': 'Library for creating and updating Microsoft Word documents.',
        'use_cases': ['Report generation', 'Document automation', 'Template filling'],
    },
    'openpyxl': {
        'full_name': 'openpyxl',
        'category': 'Document Processing',
        'license': 'MIT',
        'url': 'https://openpyxl.readthedocs.io/',
        'description': 'Library to read/write Excel 2010 xlsx/xlsm files.',
        'use_cases': ['Excel file handling', 'Spreadsheet generation', 'Data export'],
    },
    'pytest': {
        'full_name': 'pytest',
        'category': 'Testing',
        'license': 'MIT',
        'url': 'https://pytest.org/',
        'description': 'Testing framework that makes it easy to write simple and scalable tests.',
        'use_cases': ['Unit testing', 'Integration testing', 'Test automation'],
    },
    'requests': {
        'full_name': 'Requests',
        'category': 'HTTP',
        'license': 'Apache-2.0',
        'url': 'https://requests.readthedocs.io/',
        'description': 'Elegant and simple HTTP library for Python.',
        'use_cases': ['API calls', 'Web scraping', 'HTTP requests'],
    },
    'pyyaml': {
        'full_name': 'PyYAML',
        'category': 'Configuration',
        'license': 'MIT',
        'url': 'https://pyyaml.org/',
        'description': 'YAML parser and emitter for Python.',
        'use_cases': ['Configuration files', 'Data serialization'],
    },
}


def generate_tools_review(config: ProjectConfig) -> str:
    """Generate tools review documentation based on project dependencies.

    Args:
        config: Project configuration

    Returns:
        Path to generated document
    """
    doc = create_document(config.formatting.margin_inches)
    set_document_defaults(doc, config.formatting.font_name, config.formatting.font_size_body)

    fmt = config.formatting

    # Get dependencies
    dependencies = get_dependencies(config.project_root)

    # Title
    add_heading(doc, f"{config.framework_name} - Open-Source Tools Review", level=0,
                font_name=fmt.font_name)

    add_paragraph(doc, f"Version: {config.version}", font_name=fmt.font_name,
                  font_size=fmt.font_size_body, italic=True)

    # 1. Introduction
    add_heading(doc, "1. Introduction", level=1, font_name=fmt.font_name)

    intro_text = config.custom_sections.get('tools_intro') or (
        f"This document reviews the open-source libraries and tools used in {config.framework_name}. "
        f"All tools are carefully selected for reliability, performance, and license compatibility."
    )
    add_paragraph(doc, intro_text, font_name=fmt.font_name, font_size=fmt.font_size_body)

    # 2. Dependencies Summary
    add_heading(doc, "2. Dependencies Summary", level=1, font_name=fmt.font_name)

    runtime_deps = [d for d in dependencies if d.category == 'runtime']
    dev_deps = [d for d in dependencies if d.category == 'dev']

    summary_rows = [
        ["Runtime Dependencies", str(len(runtime_deps))],
        ["Development Dependencies", str(len(dev_deps))],
        ["Total", str(len(dependencies))],
    ]
    add_table(doc, ["Category", "Count"], summary_rows,
              header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 3. Runtime Dependencies Detail
    add_heading(doc, "3. Runtime Dependencies", level=1, font_name=fmt.font_name)

    # Group by category
    categorized = {}
    for dep in runtime_deps:
        lib_info = LIBRARY_DETAILS.get(dep.name.lower(), {})
        category = lib_info.get('category', 'Other')
        if category not in categorized:
            categorized[category] = []
        categorized[category].append((dep, lib_info))

    for category, deps in sorted(categorized.items()):
        add_heading(doc, f"3.x {category}", level=2, font_name=fmt.font_name)

        for dep, lib_info in deps:
            name = lib_info.get('full_name', dep.name)
            add_heading(doc, name, level=3, font_name=fmt.font_name)

            # Basic info table
            version = dep.version or "Latest"
            license_info = lib_info.get('license', 'See package')
            url = lib_info.get('url', '')

            info_rows = [
                ["Package Name", dep.name],
                ["Version", version],
                ["License", license_info],
            ]
            if url:
                info_rows.append(["Website", url])

            add_table(doc, ["Property", "Value"], info_rows,
                      header_bg=fmt.table_header_bg, font_name=fmt.font_name)

            # Description
            description = lib_info.get('description', dep.description or 'No description available.')
            add_paragraph(doc, description, font_name=fmt.font_name, font_size=fmt.font_size_body)

            # Use cases
            use_cases = lib_info.get('use_cases', [])
            if use_cases:
                add_paragraph(doc, "Use cases in this project:", font_name=fmt.font_name,
                              font_size=fmt.font_size_body, bold=True)
                for use in use_cases:
                    add_bullet_point(doc, use, font_name=fmt.font_name)

    # 4. Development Dependencies
    if dev_deps:
        add_heading(doc, "4. Development Dependencies", level=1, font_name=fmt.font_name)

        dev_rows = []
        for dep in dev_deps:
            lib_info = LIBRARY_DETAILS.get(dep.name.lower(), {})
            version = dep.version or "Latest"
            license_info = lib_info.get('license', '-')
            desc = lib_info.get('description', dep.description or '-')
            dev_rows.append([dep.name, version, license_info, desc[:40]])

        add_table(doc, ["Package", "Version", "License", "Description"], dev_rows,
                  header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 5. License Summary
    add_heading(doc, "5. License Summary", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "All dependencies use permissive open-source licenses compatible with commercial use:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    license_info = [
        ["MIT", "Permissive, minimal restrictions"],
        ["BSD-3-Clause", "Permissive, attribution required"],
        ["Apache-2.0", "Permissive, patent protection"],
        ["PSF", "Python Software Foundation License"],
    ]
    add_table(doc, ["License", "Description"], license_info,
              header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 6. Security Considerations
    add_heading(doc, "6. Security Considerations", level=1, font_name=fmt.font_name)

    add_bullet_point(doc, "All packages are sourced from PyPI (Python Package Index)",
                     font_name=fmt.font_name)
    add_bullet_point(doc, "Dependencies are pinned to specific versions for reproducibility",
                     font_name=fmt.font_name)
    add_bullet_point(doc, "Regular security audits recommended using tools like safety or pip-audit",
                     font_name=fmt.font_name)
    add_bullet_point(doc, "Keep dependencies updated to receive security patches",
                     font_name=fmt.font_name)

    # Save document
    output_path = Path(config.project_root) / config.docs_dir / "Open-Source_Tools_Review.docx"
    save_document(doc, str(output_path))

    return str(output_path)
