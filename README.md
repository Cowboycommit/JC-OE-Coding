# Open-Ended Coding Analysis

A comprehensive Python framework for analyzing open-ended qualitative data through systematic coding, theme identification, and hierarchical categorization.

## Features

- **Code Frames**: Systematic coding structures for categorizing qualitative data
- **Theme Analysis**: Identification and analysis of recurring patterns and themes
- **Multi-level Categorization**: Hierarchical classification of coded data
- **Data Loading**: Support for multiple data sources (CSV, Excel, SQLite, PostgreSQL)
- **Interactive Visualizations**: Rich visualizations using Plotly and Seaborn
- **Robust Error Handling**: Comprehensive logging and error management
- **Code Quality**: Automated testing and linting via Makefile

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Launch Jupyter notebook
jupyter notebook open_ended_coding_analysis.ipynb
```

## Project Structure

```
.
├── open_ended_coding_analysis.ipynb  # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Data loading utilities
│   ├── code_frame.py                # Code frame management
│   ├── theme_analyzer.py            # Theme identification
│   └── category_manager.py          # Categorization system
├── data/
│   ├── sample_responses.csv         # Sample data
│   └── README.md                    # Data documentation
├── output/                          # Analysis results
├── tests/                           # Unit tests
│   ├── test_data_loader.py
│   ├── test_code_frame.py
│   ├── test_theme_analyzer.py
│   └── test_category_manager.py
├── requirements.txt                 # Python dependencies
├── Makefile                         # Build and test automation
└── README.md                        # This file
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

# From SQLite
df = loader.load_from_sqlite('data/survey.db', 'SELECT * FROM responses')

# From PostgreSQL
df = loader.load_from_postgres(
    'postgresql://user:pass@localhost:5432/db',
    'SELECT * FROM responses'
)
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

Your data should be in CSV, Excel, or database format with at least one column containing text responses:

| id | response | metadata |
|----|----------|----------|
| 1 | "Text response..." | Optional |
| 2 | "Another response..." | Optional |

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
- Database connection error management
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

### Database Export

```python
# Save results to database
from sqlalchemy import create_engine

engine = create_engine('postgresql://localhost/analysis_db')
df.to_sql('coded_responses', engine, if_exists='replace')
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

**Issue**: Database connection failures
```bash
# Solution: Verify connection string and credentials
# Test connection separately before running full analysis
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
  year = {2024},
  url = {https://github.com/yourusername/open-ended-coding}
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
- SQLAlchemy for database connectivity
- Jupyter for interactive analysis

## Roadmap

### Planned Features

- [ ] Machine learning-assisted coding
- [ ] Advanced NLP integration (topic modeling, sentiment analysis)
- [ ] Real-time collaborative coding
- [ ] Web-based dashboard
- [ ] API for programmatic access
- [ ] Support for qualitative data software export formats (NVivo, Atlas.ti)

## Version History

### v1.0.0 (2024)
- Initial release
- Core coding, theming, and categorization features
- Multiple data source support
- Interactive visualizations
- Comprehensive testing framework
