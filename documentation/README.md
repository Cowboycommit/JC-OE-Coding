# Open-Ended Coding Analysis Framework - Documentation Suite

Welcome to the comprehensive documentation for the Open-Ended Coding Analysis Framework. This documentation suite provides detailed guidance on implementing, validating, and deploying a machine learning pipeline for open-ended coding analysis.

## Quick Start

Get up and running with the framework in three simple steps:

### Installation
```bash
pip install -r requirements.txt
```

### Run the Streamlit Application (Recommended)
```bash
streamlit run app.py
```

### Run the Engineering/Lite View
```bash
streamlit run app_lite.py
```

### Run the Jupyter Notebook
```bash
jupyter notebook ml_open_coding_analysis.ipynb
```

---

## Key Features

- **6 ML Algorithms**: TF-IDF+K-Means, LDA, NMF, LSTM, BERT, SVM
- **Text Preprocessing**: Data-type presets, negation preservation, domain stopwords
- **Sentiment Analysis**: VADER (survey), Twitter-RoBERTa (social), Review-BERT (reviews)
- **LLM-Enhanced Labels**: AI-refined code labels and descriptions
- **15 Essential Outputs**: Complete analysis package for researchers

---

## Documentation Index

This suite contains seven comprehensive documents covering all aspects of the framework:

1. **[01_open_source_tools_review.md](./01_open_source_tools_review.md)** - OSS Tools Analysis
   - Evaluation of open-source tools for the analysis pipeline
   - Tool selection criteria and recommendations

2. **[02_benchmark_standards.md](./02_benchmark_standards.md)** - Quality Benchmarks
   - Performance and quality standards
   - Evaluation metrics and success criteria

3. **[03_input_data_specification.md](./03_input_data_specification.md)** - Data Requirements
   - Input data format and structure specifications
   - Data schema and requirements documentation

4. **[04_data_formatting_rules.md](./04_data_formatting_rules.md)** - Formatting Rules
   - Rules for formatting and preparing data
   - Data validation and transformation guidelines

5. **[05_reporting_and_visualization_standards.md](./05_reporting_and_visualization_standards.md)** - Visualization Standards
   - Standards for reporting and data visualization
   - Dashboard and output format specifications

6. **[06_validation_and_demonstration.md](./06_validation_and_demonstration.md)** - Validation Examples
   - Validation methodologies and examples
   - Demonstration use cases and test scenarios

7. **[07_documentation_and_handover.md](./07_documentation_and_handover.md)** - Handover Guide
   - Complete documentation and handover procedures
   - Knowledge transfer and deployment guidelines

---

## Coverage Checklist

The following matrix confirms that each requirement is comprehensively covered:

- [x] **Open-Source Tool Evaluation** - Document 01
- [x] **Benchmark Standards Definition** - Document 02
- [x] **Input Data Specification** - Document 03
- [x] **Data Formatting Rules** - Document 04
- [x] **Reporting & Visualization Standards** - Document 05
- [x] **Validation & Demonstration** - Document 06
- [x] **Documentation & Handover** - Document 07
- [x] **ML Pipeline Implementation** - Documents 02-06
- [x] **Quality Assurance** - Document 06
- [x] **Deployment Readiness** - Document 07

---

## Document Descriptions

### Document 01: Open Source Tools Review
Provides a comprehensive evaluation of open-source tools available for the coding analysis pipeline. This document guides the selection of appropriate libraries and frameworks, including analysis of their capabilities, limitations, and integration potential.

### Document 02: Benchmark Standards
Establishes the quality and performance standards for the framework. Defines metrics for success, acceptable performance thresholds, and evaluation criteria that the ML pipeline must meet.

### Document 03: Input Data Specification
Details the required input data format, structure, and schema. Essential for understanding what data the framework expects and how to prepare your datasets.

### Document 04: Data Formatting Rules
Provides specific rules and procedures for formatting, cleaning, and validating data before it enters the pipeline. Critical for ensuring data quality and consistency.

### Document 05: Reporting and Visualization Standards
Establishes standards for output reporting and data visualization. Covers dashboard design, chart types, metrics presentation, and visualization best practices.

### Document 06: Validation and Demonstration
Contains validation methodologies, test cases, and demonstration examples. Shows how to validate pipeline outputs and demonstrates proper usage patterns.

### Document 07: Documentation and Handover
Provides complete documentation standards and handover procedures for deploying the framework. Ensures knowledge transfer and operational readiness.

---

## Cross-Reference Guide

Use this guide to quickly find the documentation you need:

| Question | Document/Location |
|----------|-------------------|
| What tools should I use? | 01 - Open Source Tools Review |
| How do I measure success? | 02 - Benchmark Standards |
| What data format do I need? | 03 - Input Data Specification |
| How do I prepare my data? | 04 - Data Formatting Rules |
| How should results be displayed? | 05 - Reporting & Visualization Standards |
| How do I test the system? | 06 - Validation & Demonstration |
| How do I deploy this? | 07 - Documentation & Handover |
| **How do I preprocess text?** | **app.py → Text Processor page** |
| **Which sentiment model to use?** | **app.py → Configuration page** |
| **What ML algorithm is best?** | **app.py → About page (algorithm table)** |
| **How does the pipeline work?** | **app_lite.py (Engineering View)** |

---

## Framework Overview

The Open-Ended Coding Analysis Framework is a comprehensive machine learning pipeline designed to analyze, process, and visualize open-ended coding data. The framework consists of:

- **Data Pipeline**: Ingestion, validation, and formatting of input data
- **Text Preprocessing**: Data-type presets, negation preservation, domain stopwords
- **ML Analysis Engine**: 6 algorithms (TF-IDF, LDA, NMF, LSTM, BERT, SVM)
- **Sentiment Analysis**: Data-type-specific models (VADER, Twitter-RoBERTa, Review-BERT)
- **LLM Enhancement**: AI-refined code labels and descriptions
- **Quality Assurance**: Validation, rigor diagnostics, and QA reports
- **Visualization Layer**: Word clouds, network diagrams, sunburst charts
- **Streamlit UIs**: Main app (user-facing) and Engineering View (pipeline documentation)
- **Notebook Interface**: Jupyter notebook for exploration and analysis

Each component is documented in detail across the documentation suite to ensure successful implementation and deployment.

---

## Getting Started

1. **First Time?** Start with [01_open_source_tools_review.md](./01_open_source_tools_review.md)
2. **Setting Up?** Follow [03_input_data_specification.md](./03_input_data_specification.md)
3. **Preparing Data?** Consult [04_data_formatting_rules.md](./04_data_formatting_rules.md)
4. **Testing?** Review [06_validation_and_demonstration.md](./06_validation_and_demonstration.md)
5. **Deploying?** Refer to [07_documentation_and_handover.md](./07_documentation_and_handover.md)

---

## Version Information

- **Documentation Suite Version**: 1.4
- **Framework Version**: 1.4.0
- **Last Updated**: 2026-02-23
- **Framework Status**: Production Ready
- **Maintenance**: Active

---

## Support and Questions

For questions or issues related to specific aspects of the framework:

- **Tool Selection**: See Document 01
- **Performance Issues**: Consult Document 02 for benchmarks
- **Data Problems**: Check Documents 03-04
- **Output Questions**: Review Document 05
- **Validation Issues**: Refer to Document 06
- **Deployment Questions**: See Document 07

---

## Document Maintenance

This documentation suite is maintained as a living document. Updates and revisions are made to reflect changes in the framework, best practices, and user feedback. All updates are documented with timestamps and version information.

For the most current version of this documentation, please refer to the repository's main branch.
