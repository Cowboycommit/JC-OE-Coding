# Open-Ended Coding Analysis

A comprehensive Python framework for analyzing open-ended qualitative data through systematic coding, theme identification, and hierarchical categorization.

## 🎯 Two Approaches Available

### 1. **Traditional Keyword-Based Coding** (`open_ended_coding_analysis.ipynb`)
Manual coding with predefined code frames and keywords

### 2. **ML-Based Open Coding** (`ml_open_coding_analysis.ipynb`) ⭐ NEW!
Automated theme discovery using machine learning with **15 essential outputs**

### 3. **Streamlit Web UI** (`app.py`) 🌐 NEW!
Interactive web interface for ML-based coding - no coding required!

## Features

### Traditional Approach
- **Code Frames**: Systematic coding structures for categorizing qualitative data
- **Theme Analysis**: Identification and analysis of recurring patterns and themes
- **Multi-level Categorization**: Hierarchical classification of coded data
- **Data Loading**: Support for multiple data sources (CSV, Excel, JSON)
- **Interactive Visualizations**: Rich visualizations using Plotly and Seaborn
- **Robust Error Handling**: Comprehensive logging and error management
- **Code Quality**: Automated testing and linting via Makefile

### ML-Based Approach
- 🤖 **Automatic Theme Discovery**: Uses TF-IDF, LDA, LSTM, BERT, and SVM clustering
- 📊 **15 Essential Outputs**: Complete analysis package for researchers
- 🎯 **Confidence Scoring**: Probabilistic code assignments with quality metrics
- 📈 **Advanced Analytics**: Co-occurrence analysis, network diagrams, segmentation
- 💾 **Multiple Export Formats**: CSV, Excel, Markdown with comprehensive results
- 📝 **Executive Summaries**: Auto-generated stakeholder reports
- ✅ **Quality Assurance**: Built-in validation and error detection
- 🏷️ **LLM-Enhanced Labels**: AI-refined code labels and descriptions

### Text Preprocessing
- 🔧 **Quick Presets**: One-click configurations for different data types (General, Social Media, Reviews, News)
- 🛡️ **Negation Preservation**: Keeps "not", "never" for accurate sentiment/topic analysis
- 📋 **Domain Stopwords**: Removes survey-specific noise words ("response", "survey", "participant")
- 🐦 **Social Media Handling**: URL, @mention, #hashtag standardization, slang expansion
- 📊 **Quality Reports**: Detailed statistics on filtered records and preprocessing effects

### Sentiment Analysis
- 📊 **Data-Type Specific Models**: Optimal model selection based on your data
  - Survey/General: VADER (rule-based, fast)
  - Twitter/Social Media: Twitter-RoBERTa (transformer-based)
  - Long-form Reviews: Review-BERT (transformer-based)
- 😊 **Sentiment Classification**: Positive, Neutral, Negative with confidence scores
- 📈 **Integrated Results**: Sentiment appears alongside topic codes in results

### Streamlit Web UI
- 🌐 **No Coding Required**: User-friendly web interface for non-programmers
- 📤 **Drag & Drop Upload**: Upload CSV/Excel files or use sample datasets
- 🔧 **Text Processor**: Comprehensive preprocessing with data-type presets
- ⚙️ **Interactive Configuration**: Visual parameter adjustment with algorithm guidance
- 📊 **Sentiment Analysis**: Optional integrated sentiment detection
- 📈 **Rich Visualizations**: Word clouds, network diagrams, sunburst charts
- 💾 **One-Click Export**: Download complete results packages
- 📱 **Responsive Design**: Works on desktop and mobile
- 🎨 **Professional Styling**: Publication-ready visualizations

## What This System Does

- Assists thematic coding by clustering responses based on word patterns
- Generates suggested codes with confidence scores
- Flags responses requiring human review
- Provides diagnostic metrics for quality assessment

## What This System Cannot Do

- Cannot replace human qualitative judgment
- Cannot understand context, irony, or cultural nuance
- Cannot determine causal relationships
- Cannot detect sarcasm or non-literal language

## Where Human Judgment is Required

- Validating auto-generated code labels
- Reviewing low-confidence assignments
- Interpreting co-occurrence patterns
- Deciding on final code structure

## Known Limitations

- English-only language processing
- Response-level granularity only
- Requires minimum 20-30 responses
- Bag-of-words assumption (word order ignored)

## Ethical Considerations

- Algorithmic bias may reflect biases in method design
- Automated coding should not be used for high-stakes decisions without validation
- Researchers remain responsible for interpretations

## Quick Start

### Installation

**For Users (Production):**
```bash
# Install production dependencies
pip install -r requirements.txt
```

**For Developers:**
```bash
# Install development dependencies (includes production dependencies)
pip install -r requirements-dev.txt
```

The `requirements-dev.txt` includes additional tools for:
- Testing (pytest, coverage)
- Code quality (black, flake8, pylint, mypy)
- Documentation (Sphinx)
- Development utilities (debuggers, profilers)
- Notebook tools (Jupyter extensions)

### Running the Analysis

**Option 1: Streamlit Web UI (Easiest - Recommended for Non-Programmers)**
```bash
streamlit run app.py
```
Then open your browser to `http://localhost:8501`

**Option 2: ML-Based Open Coding Notebook**
```bash
jupyter notebook ml_open_coding_analysis.ipynb
```

**Option 3: Traditional Keyword-Based Coding**
```bash
jupyter notebook open_ended_coding_analysis.ipynb
```

## Project Structure

```
.
├── app.py                           # Main Streamlit web UI (user-facing)
├── app_lite.py                      # Engineering/Lite UI (pipeline documentation)
├── open_ended_coding_analysis.ipynb  # Traditional keyword-based coding
├── ml_open_coding_analysis.ipynb     # ML-based automatic coding
├── helpers/                         # Helper modules for Streamlit
│   ├── __init__.py
│   ├── formatting.py               # Formatting utilities
│   ├── analysis.py                 # Analysis orchestration
│   └── ui_utils.py                 # Shared UI components and utilities (NEW)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Data loading (CSV, Excel, JSON)
│   ├── text_preprocessor.py        # Enhanced text preprocessing (NLTK)
│   ├── gold_standard_preprocessing.py  # Industry-standard text normalization
│   ├── sentiment_analysis.py       # Data-type-specific sentiment models
│   ├── embeddings.py               # TF-IDF, BERT, LSTM, Word2Vec
│   ├── cluster_interpretation.py   # Code labeling with LLM enhancement
│   ├── method_visualizations.py    # Word clouds, network diagrams
│   ├── rigor_diagnostics.py        # Validity assessment
│   ├── code_frame.py               # Code frame management
│   ├── theme_analyzer.py           # Theme identification
│   └── category_manager.py         # Categorization system
├── data/
│   ├── *.csv                       # Sample datasets (9 datasets)
│   └── stopwords_domain.txt        # Domain-specific stopwords
├── documentation/                  # 7-document comprehensive suite
├── tests/                          # Unit and integration tests (21 files)
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies
├── Makefile                        # Build and test automation
└── README.md                       # This file
```

## Usage

### 1. Data Loading

The notebook supports loading data from multiple sources:

```python
from src.data_loader import DataLoader

loader = DataLoader()

# From CSV
df = loader.load_csv('data/responses.csv')

# From Excel
df = loader.load_excel('data/responses.xlsx', sheet_name='Survey')

# From JSON (standard or JSON Lines)
df = loader.load_json('data/responses.json', lines=True)
```

### 2. Define Code Frames

Create systematic coding structures:

```python
from src.code_frame import CodeFrame

# Create code frame
code_frame = CodeFrame("Research Analysis", "Analysis of survey responses")

# Add codes
code_frame.add_code(
    'POSITIVE',
    'Positive Sentiment',
    description='Positive experiences',
    keywords=['good', 'great', 'love', 'excellent']
)

# Apply codes to data
df['codes'] = df['response'].apply(code_frame.apply_codes)
```

### 3. Identify Themes

Analyze recurring patterns:

```python
from src.theme_analyzer import ThemeAnalyzer

analyzer = ThemeAnalyzer()

# Define themes
analyzer.define_theme(
    'THEME_SATISFACTION',
    'User Satisfaction',
    'Themes related to user satisfaction',
    associated_codes=['POSITIVE', 'ENGAGEMENT']
)

# Identify themes in data
df = analyzer.identify_themes(df)
```

### 4. Categorize Data

Apply multi-level categorization:

```python
from src.category_manager import CategoryManager

cat_manager = CategoryManager()

# Create categories
cat_manager.create_category(
    'CAT_HIGH_ENGAGEMENT',
    'High Engagement',
    criteria={'codes_required': ['POSITIVE', 'ACTIVE']},
    level=1
)

# Apply categorization
df = cat_manager.categorize(df)
```

## Streamlit Web UI (NEW!)

The `app.py` provides an intuitive web interface for ML-based coding - **perfect for non-programmers**!

### 🌐 Features

- **📥 Download Data Template**: Get a formatted Excel template with instructions and examples
- **📤 Data Upload**: Drag and drop CSV/Excel files
- **🔍 Data Preview**: Instantly see your data with column information
- **⚙️ Interactive Configuration**:
  - Select text column visually
  - Adjust number of codes with sliders
  - Choose ML algorithm from dropdown
  - Set confidence thresholds
- **🚀 One-Click Analysis**: Run complete ML analysis with progress tracking
- **📊 Rich Visualizations**:
  - Interactive frequency charts
  - Co-occurrence heatmaps
  - Network diagrams
  - Distribution plots
  - Confidence score analysis
- **💾 Easy Exports**: Download complete results as Excel with one click
- **📝 Executive Summaries**: Auto-generated insights and reports

### 🎯 How to Use

1. **Launch the app**:
   ```bash
   streamlit run app.py
   ```

2. **Download the template** (optional): Click "📥 Download Template" to get a formatted Excel file with instructions and examples

3. **Upload your data**: Click "Browse files" and select your CSV/Excel file

4. **Configure analysis**:
   - Select the column containing responses
   - Choose number of themes (3-30)
   - Pick ML algorithm (TF-IDF+K-Means recommended)
   - Set confidence threshold (0.3 works well)

5. **Run analysis**: Click "Start Analysis" and watch the magic happen!

6. **Explore results**:
   - View key metrics and insights
   - Examine code frequencies and examples
   - Explore interactive visualizations
   - Download complete results package

### 📱 Interface Sections (8 Pages)

1. **📤 Data Upload**: Load sample datasets or upload CSV/Excel files
2. **🔧 Text Processor**: Preprocess text with data-type presets (General, Social Media, Reviews, News)
3. **⚙️ Configuration**: Select ML algorithm, code count, and enable sentiment analysis
4. **🚀 Run Analysis**: Execute ML coding with real-time progress tracking
5. **📊 Results Overview**: View metrics, codebook, word cloud, and sentiment results
6. **📈 Visualizations**: Charts, heatmaps, word clouds, sunburst, network diagrams
7. **💾 Export Results**: Download Excel package, CSV, or Markdown summary
8. **ℹ️ About**: Complete feature documentation and getting started guide

## ML-Based Open Coding (NEW!)

The `ml_open_coding_analysis.ipynb` notebook provides a comprehensive ML-powered framework with **15 essential outputs**:

### Key Features

1. **Code Assignments** - Response-level codes with confidence scores
2. **Codebook** - Auto-generated code definitions with examples
3. **Frequency Tables** - Statistical distribution of codes
4. **Quality Metrics** - Confidence scores and reliability measures
5. **Binary Matrix** - Code presence/absence for statistical analysis
6. **Representative Quotes** - Top examples for each code
7. **Co-Occurrence Analysis** - Code relationship patterns and networks
8. **Descriptive Statistics** - Comprehensive summary statistics
9. **Segmentation Analysis** - Code patterns across demographics
10. **QA Report** - Validation and error analysis
11. **Visualizations** - Interactive charts, heatmaps, network diagrams
12. **Multiple Exports** - CSV, Excel, JSON formats
13. **Method Documentation** - Transparent methodology and reproducibility
14. **Uncoded Responses** - Edge cases and low-confidence items
15. **Executive Summary** - High-level insights for stakeholders

### Quick Example

```python
# Initialize ML coder
from ml_open_coding_analysis import MLOpenCoder, OpenCodingResults

coder = MLOpenCoder(
    n_codes=10,              # Number of themes to discover
    method='tfidf_kmeans',   # Algorithm: 'tfidf_kmeans', 'lda', or 'nmf'
    min_confidence=0.3       # Confidence threshold
)

# Fit on your data
coder.fit(df['response'])

# Generate complete results package
results = OpenCodingResults(df, coder, response_col='response')

# Get all 15 outputs
assignments = results.get_code_assignments()
codebook = results.get_codebook_detailed()
frequency = results.get_frequency_table()
quality = results.get_quality_metrics()
# ... and 11 more outputs!

# Export everything
from ml_open_coding_analysis import ResultsExporter
exporter = ResultsExporter(results)
exporter.export_all()
exporter.export_excel('results.xlsx')
```

### Supported ML Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **TF-IDF + K-Means** | Fast bag-of-words clustering | General use, quick exploration |
| **LDA** | Probabilistic topic modeling | Overlapping themes, academic research |
| **LSTM + K-Means** | Sequential pattern recognition | Order-dependent text, narratives |
| **BERT + K-Means** | Semantic embedding clustering | Nuanced meaning, synonyms |
| **SVM Spectral** | Kernel-based clustering | Complex, non-linear boundaries |

### Output Formats

- **CSV**: Individual files for each output type
- **Excel**: Single workbook with multiple sheets
- **JSON**: Structured data for APIs and further processing
- **Markdown**: Executive summaries and documentation

## Development

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_data_loader.py -v

# Run with coverage
make test-coverage
```

### Code Quality

```bash
# Run linting
make lint

# Format code
make format

# Type checking
make typecheck

# Run all quality checks
make quality
```

### Building Documentation

```bash
make docs
```

## Data Format

### Input Data Requirements

Your data should be in CSV, Excel, or JSON format with at least one column containing text responses:

| id | response | metadata |
|----|----------|----------|
| 1 | "Text response..." | Optional |
| 2 | "Another response..." | Optional |

### 📥 Data Template Available!

To help you format your data correctly, we provide a comprehensive Excel template with:
- **Instructions sheet** - Detailed formatting guidelines
- **Data entry sheet** - Pre-formatted columns ready for your data
- **Sample data sheet** - 15+ examples showing proper format

**Download the template:**
- **Via Streamlit UI**: Click the "📥 Download Template" button on the Data Upload page
- **Direct path**: `documentation/input_data_template.xlsx`
- **Generate fresh copy**: Run `python scripts/create_data_template.py`

**Required column:**
- `response` - Your text responses (minimum 5 characters)

**Optional columns (add as needed):**
- `id` - Unique response identifier
- `respondent_id` - Participant identifier
- `timestamp` - Collection date/time (YYYY-MM-DD format)
- `topic` - Response category or topic
- Any other demographic or grouping variables

### Output Files

The analysis generates several output files in the `output/` directory:

- `coded_data.csv`: Full dataset with applied codes, themes, and categories
- `code_summary.csv`: Summary statistics for each code
- `theme_summary.csv`: Theme frequency and associations
- `category_summary.csv`: Category distribution and hierarchy

## Visualization Examples

The notebook includes several interactive visualizations:

1. **Code Distribution**: Bar charts showing code frequencies
2. **Hierarchical Sunburst**: Nested code structure visualization
3. **Theme Networks**: Co-occurrence patterns between themes
4. **Category Analysis**: Multi-level category distributions
5. **Comprehensive Dashboard**: Combined analytics view

## Error Handling

The framework includes robust error handling:

- File validation before loading
- Empty data detection
- Logging of all operations
- Graceful degradation for missing data

## Advanced Features

### Intercoder Reliability

```python
from src.reliability import calculate_cohens_kappa

kappa = calculate_cohens_kappa(coder1_codes, coder2_codes)
print(f"Cohen's Kappa: {kappa:.3f}")
```

### Export to Multiple Formats

```python
# Export to Excel with multiple sheets
with pd.ExcelWriter('output/analysis_results.xlsx') as writer:
    df.to_excel(writer, sheet_name='Coded Data')
    code_summary.to_excel(writer, sheet_name='Code Summary')
    theme_summary.to_excel(writer, sheet_name='Themes')
```

## Configuration

### Logging

Configure logging in the notebook or via environment variable:

```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Visualization Themes

Customize visualization appearance:

```python
import plotly.io as pio
pio.templates.default = "plotly_dark"
```

## Best Practices

1. **Start with Sample Data**: Test your code frames on a subset before full analysis
2. **Iterative Refinement**: Review and refine codes/themes based on initial results
3. **Document Decisions**: Keep notes on coding decisions and rationale
4. **Regular Backups**: Save intermediate results frequently
5. **Version Control**: Use git to track changes to code frames and categories

## Troubleshooting

### Common Issues

**Issue**: Import errors when running notebook
```bash
# Solution: Ensure all dependencies are installed
pip install -r requirements.txt --upgrade
```

**Issue**: Empty visualizations
```bash
# Solution: Check that codes are being applied correctly
print(df['codes'].value_counts())
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-analysis`)
3. Make your changes
4. Run tests (`make test`)
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/new-analysis`)
7. Create a Pull Request

## Testing

The project includes comprehensive tests:

- Unit tests for all core components
- Integration tests for complete workflows
- Data validation tests
- Error handling tests

Run the full test suite:

```bash
make test-all
```

## License

MIT License - See LICENSE file for details

## Citation

If you use this framework in your research, please cite:

```
@software{open_ended_coding,
  title = {Open-Ended Coding Analysis Framework},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/Cowboycommit/JC-OE-Coding}
}
```

## Support

- Documentation: See the notebook and inline comments
- Issues: Open an issue on GitHub
- Questions: Check existing issues or open a discussion

## Acknowledgments

Built with:
- Pandas for data manipulation
- Plotly and Seaborn for visualizations
- Jupyter for interactive analysis

## Roadmap

### Completed Features

- [x] Machine learning-assisted coding (ML-based notebook)
- [x] Advanced NLP integration (topic modeling with LDA, LSTM, BERT, SVM)
- [x] Comprehensive export formats (CSV, Excel, Markdown)
- [x] Executive summaries and stakeholder reports
- [x] Web-based dashboard (Streamlit UI)
- [x] Sentiment analysis integration (VADER, Twitter-RoBERTa, Review-BERT)
- [x] Text preprocessing with data-type presets
- [x] LLM-enhanced code labels and descriptions
- [x] Semantic word clouds with meaning-based coloring

### Planned Features

- [ ] Real-time collaborative coding
- [ ] REST API for programmatic access
- [ ] Support for qualitative data software export formats (NVivo, Atlas.ti)
- [ ] Multi-language support (currently English only)

## Version History

### v1.3.1 (2026) - Codebase Refactoring
- **Shared UI Utilities**: Created `helpers/ui_utils.py` with common patterns
  - Centralized code label extraction logic
  - Standardized progress mapping and stage tracking
  - Reusable error recovery UI components
  - Shared stage metadata rendering
  - Session state management utilities
- **Reduced Redundancy**: Eliminated duplicate code across app.py and app_lite.py
- **Consistency**: All interfaces now use the same shared analysis functions

### v1.3.0 (2026) - Enhanced Preprocessing & Sentiment
- **Text Processor**: Comprehensive preprocessing with data-type presets
  - Quick Presets: General, Social Media, Reviews, News
  - Negation preservation for accurate sentiment/topic analysis
  - Domain-specific stopwords for noise reduction
  - Gold Standard processing (Unicode, HTML, contractions, slang)
- **Sentiment Analysis**: Integrated with data-type-specific models
  - Survey: VADER (rule-based)
  - Twitter: Twitter-RoBERTa (transformer)
  - Reviews: Review-BERT (transformer)
- **Enhanced ML Algorithms**: Added LSTM, BERT, and SVM clustering
- **LLM-Enhanced Labels**: AI-refined code labels and descriptions
- **Semantic Word Clouds**: Color-coded by word meaning similarity
- **Network Diagrams**: Cluster relationship visualization
- **Engineering View** (app_lite.py): Pipeline documentation UI

### v1.2.0 (2024) - Streamlit Web UI
- Added Streamlit web application for no-code analysis
- Drag-and-drop file upload (CSV/Excel)
- Interactive parameter configuration
- Real-time progress tracking
- Rich interactive visualizations (Plotly)
- One-click export of complete results
- Formatted helper modules for analysis and formatting
- Mobile-responsive design
- Professional UI with custom styling

### v1.1.0 (2024) - ML-Based Open Coding
- Added ML-based automatic coding notebook
- Implemented 15 essential outputs for comprehensive analysis
- Added support for TF-IDF, LDA, and K-Means algorithms
- Automatic theme discovery and code generation
- Confidence scoring and quality metrics
- Co-occurrence analysis and network visualizations
- Multiple export formats (CSV, Excel, JSON)
- Executive summary generation
- Method documentation and reproducibility support

### v1.0.0 (2024)
- Initial release
- Core coding, theming, and categorization features
- Multiple data source support
- Interactive visualizations
- Comprehensive testing framework
