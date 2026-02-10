# AI Agent Prompt: Generate Documentation Suite (01–07)

## Context

You are writing a comprehensive documentation suite for the **Open-Ended Coding Analysis Framework** — a Python-based ML pipeline for qualitative analysis of open-ended text responses (surveys, customer feedback, interview transcripts, focus groups). The framework is production-ready and includes:

- **5 ML algorithms**: TF-IDF + K-Means, LDA, LSTM, BERT, SVM
- **Text preprocessing**: Data-type presets (survey, social media, reviews), negation preservation, domain-specific stopwords
- **Sentiment analysis**: VADER (survey data), Twitter-RoBERTa (social media), Review-BERT (product reviews)
- **LLM-enhanced labelling**: AI-refined code labels and descriptions
- **15 essential outputs**: Complete analysis package for researchers
- **Visualizations**: Word clouds, network diagrams, sunburst charts (Plotly, NetworkX, Matplotlib)
- **Two Streamlit UIs**: `app.py` (main user-facing) and `app_lite.py` (engineering/lite view)
- **Jupyter notebook**: `ml_open_coding_analysis.ipynb` for interactive exploration

### Tech Stack

| Category | Tools |
|----------|-------|
| Data processing | pandas (>=2.0) |
| ML/Clustering | scikit-learn |
| NLP | nltk, gensim |
| Embeddings | SentenceTransformers (optional) |
| Visualization | Plotly, Matplotlib, Seaborn, NetworkX |
| Web UI | Streamlit |
| Deep learning | PyTorch (LSTM, BERT) |

All dependencies use permissive licensing (MIT, Apache 2.0, BSD).

### Key Source Modules

- `src/data_loader.py` — data ingestion and validation
- `src/content_quality.py` — content quality assessment
- `src/text_processor.py` — text preprocessing pipeline
- `src/ml_pipeline.py` — ML algorithm execution
- `src/visualization.py` — chart and report generation
- `app.py` — main Streamlit application
- `app_lite.py` — engineering/lite Streamlit view

---

## Task

Create **seven markdown files** in the `documentation/` folder, numbered `01` through `07`. Each file must be thorough, technically detailed, well-structured with a table of contents, and written for a mixed audience of analysts, developers, and maintainers. Use tables, code blocks, checklists, and cross-references between documents where appropriate.

---

## File Specifications

### 01_open_source_tools_review.md

**Title**: Open Source Tools Review
**Purpose**: Evaluate and justify every open-source tool/library used in the framework.

**Must include**:
- Executive overview of the technology strategy
- Category-by-category tool comparison tables (Data Processing, ML/Clustering, NLP/Topic Modelling, Embeddings, Visualization, Web UI, Deep Learning, Utilities)
- For each category: selected tool vs. alternatives considered, with columns for version, strengths, weaknesses, and use-case fit
- Justification paragraph for each selected tool explaining why it was chosen over alternatives
- Licensing and compatibility matrix
- Integration architecture showing how tools connect in the pipeline
- Dependency management notes (pip, version pinning, optional vs. required)
- Version compatibility and upgrade guidance

### 02_benchmark_standards.md

**Title**: Benchmark Standards & Gold-Standard Outputs
**Purpose**: Define quality benchmarks, validation metrics, and gold-standard criteria for all ML outputs.

**Must include**:
- Definition of "gold-standard outputs" in the context of ML-based qualitative analysis
- Benchmark standards broken down by technique (TF-IDF + K-Means, LDA, NMF, Semantic Embeddings)
- For each technique: target metrics with acceptable/good/excellent thresholds (e.g., Silhouette Score, Coherence Score, Calinski-Harabasz Index, Davies-Bouldin Index)
- Gold-standard output checklists for the 15 essential deliverables
- Validation metrics reference table with formulas/definitions, interpretation guidance, and acceptable ranges
- Content quality scoring criteria and thresholds
- Authoritative citations from academic literature for each metric threshold
- Known limitations and caveats

### 03_input_data_specification.md

**Title**: Input Data Specification
**Purpose**: Fully specify what input data the framework expects and how datasets must be structured.

**Must include**:
- Overview of supported data sources and use cases
- Dataset requirements (minimum/recommended row counts, size limits)
- Required columns (especially `response`) with flexible column-name mapping
- Optional columns (ID, weight, timestamp, demographic, category fields)
- Supported file formats (CSV, Excel, JSON) with encoding requirements (UTF-8)
- Data type specifications and auto-detection behaviour
- Constraints and assumptions (e.g., one response per row, language requirements)
- ID, weight, and time field requirements with examples
- Sample data schema with a minimal working example
- Cross-references to Document 04 for formatting details

### 04_data_formatting_rules.md

**Title**: Data Formatting Rules
**Purpose**: Provide specific, prescriptive rules for formatting and preparing input data.

**Must include**:
- Naming conventions for columns and files
- Character encoding rules (UTF-8 enforcement, BOM handling)
- Date and time format specifications
- Missing values policy (how NaN/null/blank are handled)
- Categorical encoding rules
- Text content rules (min/max length, special characters, HTML/markup handling)
- CSV-specific formatting rules (delimiters, quoting, line endings)
- Excel-specific formatting rules (sheet naming, cell formatting, merged cells)
- JSON-specific formatting rules (structure, nesting, arrays)
- A template/schema reference
- Minimal sample dataset showing a correct input file
- Validation rules with error messages
- Cross-references to Documents 03 and 06

### 05_reporting_and_visualization_standards.md

**Title**: Reporting and Visualization Standards
**Purpose**: Establish standards for all output reports, dashboards, and visualizations.

**Must include**:
- Reporting templates guidance (executive summary, detailed analysis, technical report)
- Visualization standards for each chart type (word clouds, bar charts, sunburst charts, network diagrams, scatter/embedding plots, heatmaps)
- Colour scheme standards (primary palette, sequential/diverging/categorical palettes, accessibility considerations)
- Table formatting standards for output data
- Textual inference summary guidelines (how to write AI-generated narrative summaries)
- Dashboard/UX integration standards for Streamlit components
- Export format specifications (CSV, Excel, JSON, PNG/SVG for images)
- Accessibility guidelines (WCAG compliance, colour-blind safe palettes, alt text)
- References to Plotly, Matplotlib, and Seaborn best practices

### 06_validation_and_demonstration.md

**Title**: Validation and Demonstration
**Purpose**: Provide validation procedures, test cases, and demonstration walkthroughs.

**Must include**:
- At least 2 validation examples per ML method (TF-IDF + K-Means, LDA, NMF) with:
  - Dataset description and objective
  - Step-by-step instructions for running via Streamlit UI and programmatically
  - Expected outputs (files generated, visualizations produced)
  - Acceptance criteria with specific metric thresholds
- A video walkthrough template (structured guide, not actual recordings)
- Test suite validation procedures (unit tests, integration tests, end-to-end tests)
- Sample datasets description and location
- Troubleshooting guide for common validation failures
- Cross-references to Document 02 for benchmark thresholds

### 07_documentation_and_handover.md

**Title**: Documentation and Handover
**Purpose**: Serve as the definitive handover guide for analysts, developers, and maintainers.

**Must include**:
- Methodology documentation per technique (TF-IDF + K-Means, LDA, NMF, Semantic Embeddings, Sentiment Analysis) covering:
  - Objectives and use cases
  - Algorithm description with parameters and defaults
  - Preprocessing steps
  - Output interpretation guide
  - Limitations and when not to use
- Developer notes:
  - Architecture overview and module map
  - API reference for key functions
  - Extension points (adding new algorithms, visualizations, data sources)
  - Testing strategy and CI/CD notes
- Analyst/client guidance:
  - How to interpret results
  - Common questions and answers
  - Best practices for different data types
- Handover checklist (environment setup, dependencies, configuration, data, testing)
- Long-term maintenance notes (dependency updates, monitoring, scaling considerations)

---

## Style & Formatting Requirements

1. **Markdown only** — no HTML unless necessary for complex tables
2. **Table of contents** at the top of every document with anchor links
3. **Consistent header hierarchy**: `#` for title, `##` for major sections, `###` for subsections, `####` for details
4. **Tables** for comparisons, metrics, and reference data
5. **Code blocks** with language tags (```python, ```bash, ```json) for all code examples
6. **Cross-references** between documents using relative links (e.g., `[Benchmark Standards](./02_benchmark_standards.md)`)
7. **Version info** in each document header (version, date, related source modules)
8. **Professional tone** — clear, precise, written for a technical audience but accessible to non-developers
9. **No emojis** unless used sparingly for visual markers in checklists (e.g., ✓)
10. **Thorough** — each document should be comprehensive (aim for 3,000–10,000 words per document depending on topic complexity)
