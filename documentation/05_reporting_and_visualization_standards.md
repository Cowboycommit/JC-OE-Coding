# Reporting and Visualization Standards

**Agent-E: Reporting & Visualization Standards**
**Open-Ended Coding Analysis Framework**
**Version:** 1.0
**Last Updated:** 2026-02-23

---

## Table of Contents

1. [Overview](#overview)
2. [Reporting Templates Guidance](#reporting-templates-guidance)
3. [Visualization Standards](#visualization-standards)
4. [Color Scheme Standards](#color-scheme-standards)
5. [Table Formatting Standards](#table-formatting-standards)
6. [Textual Inference Summaries](#textual-inference-summaries)
7. [Dashboard/UX Integration](#dashboardux-integration)
8. [Export Format Specifications](#export-format-specifications)
9. [Accessibility Guidelines](#accessibility-guidelines)
10. [References](#references)

---

## Overview

This document establishes standards for all reporting and visualization outputs in the Open-Ended Coding Analysis Framework. Consistent formatting ensures professional presentation, reproducibility, and ease of interpretation across different stakeholders.

**Scope:**
- Interactive visualizations (Plotly, NetworkX)
- Static charts (Matplotlib, Seaborn)
- Word clouds and textual displays
- Executive summaries and reports
- Streamlit dashboard components
- Export formats (CSV, Excel, JSON, images)

**Related Documentation:**
- [Data Processing Standards](./03_data_processing_standards.md)
- [Benchmark Standards](./02_benchmark_standards.md)
- [Agent Implementation Guide](./06_agent_implementation_guide.md)

---

## Reporting Templates Guidance

### 1. Word/DOCX Report Structure

**Library:** `python-docx`

#### Standard Report Sections

| Section | Description | Required |
|---------|-------------|----------|
| **Cover Page** | Project title, date, analyst name, framework version | Yes |
| **Executive Summary** | 1-2 page overview, key findings, top themes | Yes |
| **Methodology** | Data source, sample size, coding approach, confidence thresholds | Yes |
| **Findings** | Detailed analysis by theme/category | Yes |
| **Visualizations** | Charts embedded as images (PNG, 300 DPI minimum) | Yes |
| **Appendices** | Full code mapping, raw statistics, data dictionary | Optional |
| **References** | Citations, documentation links | Optional |

#### Document Formatting Specifications

```python
# Example python-docx formatting
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Title styling
title = doc.add_heading('Open-Ended Coding Analysis Report', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Section headings (Heading 1)
# Font: Calibri 16pt, Bold, Color: #1f77b4
heading1 = doc.add_heading('Executive Summary', 1)

# Body text
# Font: Calibri 11pt, Color: Black, Line spacing: 1.15
paragraph = doc.add_paragraph('Analysis overview...')
paragraph.style.font.size = Pt(11)

# Table styling
table = doc.add_table(rows=3, cols=3)
table.style = 'Light Grid Accent 1'
```

**Image Insertion Guidelines:**
- Resolution: 300 DPI minimum for print, 150 DPI for digital-only
- Format: PNG with transparent backgrounds where appropriate
- Width: 6.5 inches (standard page width with margins)
- Caption: Below image, italicized, 10pt font

### 2. PowerPoint/Presentation Recommendations

**Library:** `python-pptx`

#### Slide Structure

1. **Title Slide**: Project name, date, framework branding
2. **Agenda/Overview**: 3-5 key sections
3. **Methodology Summary**: 1 slide with visual diagram
4. **Key Findings**: 1 slide per major theme (max 5-7 themes)
5. **Detailed Analysis**: Supporting slides as needed
6. **Recommendations**: Actionable insights
7. **Appendix**: Technical details, full tables

#### Slide Design Standards

| Element | Specification |
|---------|--------------|
| **Template** | 16:9 widescreen format |
| **Font (Title)** | Calibri 32pt, Bold, #1f77b4 |
| **Font (Body)** | Calibri 18pt, #333333 |
| **Background** | White or light gray (#f7f7f7) |
| **Chart Size** | 60-70% of slide area |
| **Bullet Points** | Maximum 5 per slide, 2-3 levels deep |
| **Logo Placement** | Bottom right corner, 0.5" x 0.5" |

**Best Practices:**
- One key message per slide
- Use visuals over text (70/30 ratio)
- Animations: Minimal, professional only (fade in/out)
- Consistent chart types across similar data

### 3. HTML Report Structure

**Libraries:** `jinja2`, `plotly` (for interactive charts)

#### Template Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Open-Ended Coding Analysis Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #ffffff;
            color: #333333;
        }
        h1 { color: #1f77b4; border-bottom: 3px solid #1f77b4; }
        h2 { color: #2c5f8d; margin-top: 2em; }
        .metric-card {
            background: #f7f7f7;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background-color: #1f77b4;
            color: white;
            padding: 12px;
            text-align: left;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <h1>{{ project_title }}</h1>
    <div class="metric-card">
        <h3>Executive Summary</h3>
        {{ executive_summary }}
    </div>

    <h2>Key Findings</h2>
    {{ plotly_chart_div }}

    <h2>Detailed Results</h2>
    {{ results_table }}
</body>
</html>
```

**Interactive Features:**
- Plotly charts with hover tooltips
- Collapsible sections for detailed data
- Export buttons (CSV, Excel, JSON)
- Print-friendly CSS media queries

---

## Visualization Standards

### 1. Frequency Charts (Bar Charts)

**Primary Use:** Display code/theme frequency distributions

#### Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Library** | Plotly Express (`px.bar`) | Interactivity, professional appearance |
| **Orientation** | Horizontal (for >5 categories) | Easier label reading |
| **Color** | Primary blue (#1f77b4) | Brand consistency |
| **Bar Width** | 0.8 (80% of available space) | Optimal visual balance |
| **Label Position** | Outside bars (right side) | Clarity |
| **Grid Lines** | Horizontal, light gray (#e0e0e0) | Readability without clutter |
| **Font Size** | Title: 18pt, Axis: 12pt, Labels: 10pt | Accessibility |

#### Example Code

```python
import plotly.express as px
import pandas as pd

# Data preparation
df = pd.DataFrame({
    'Code': ['Customer Service', 'Product Quality', 'Pricing', 'Delivery', 'Other'],
    'Frequency': [145, 98, 76, 54, 23]
}).sort_values('Frequency', ascending=True)

# Create chart
fig = px.bar(
    df,
    x='Frequency',
    y='Code',
    orientation='h',
    title='Code Frequency Distribution',
    color_discrete_sequence=['#1f77b4'],
    text='Frequency'
)

# Styling
fig.update_traces(
    textposition='outside',
    textfont_size=10,
    marker_line_color='#0d3d5c',
    marker_line_width=1.5
)

fig.update_layout(
    font=dict(family='Arial', size=12, color='#333333'),
    title_font_size=18,
    plot_bgcolor='white',
    xaxis=dict(
        title='Frequency Count',
        gridcolor='#e0e0e0',
        showgrid=True
    ),
    yaxis=dict(title='', showgrid=False),
    margin=dict(l=20, r=20, t=60, b=40),
    height=max(400, len(df) * 40)  # Dynamic height
)

fig.show()
```

**Interpretation Notes:**
- Always sort by frequency (descending for vertical, ascending for horizontal)
- Include counts on bars for exact values
- Use logarithmic scale if range exceeds 2 orders of magnitude

### 2. Heatmaps (Co-occurrence Visualization)

**Primary Use:** Show relationships between codes/themes

#### Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Library** | Plotly (`plotly.graph_objects.Heatmap`) or Seaborn | Interactivity vs. publication quality |
| **Color Scale** | Blues (sequential) or RdBu (diverging) | Professional, colorblind-friendly |
| **Annotation** | Values displayed in cells | Precise interpretation |
| **Cell Size** | Auto-adjust to matrix size | Readability |
| **Diagonal** | Highlighted or removed | Self-co-occurrence not meaningful |

#### Example Code

```python
import plotly.graph_objects as go
import numpy as np

# Example co-occurrence matrix
codes = ['Service', 'Quality', 'Price', 'Delivery', 'Support']
co_occurrence = np.array([
    [0, 45, 23, 12, 34],
    [45, 0, 56, 8, 19],
    [23, 56, 0, 15, 7],
    [12, 8, 15, 0, 22],
    [34, 19, 7, 22, 0]
])

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=co_occurrence,
    x=codes,
    y=codes,
    colorscale='Blues',
    text=co_occurrence,
    texttemplate='%{text}',
    textfont={"size": 10},
    colorbar=dict(
        title='Co-occurrence<br>Count',
        titleside='right',
        tickmode='linear',
        tick0=0,
        dtick=20
    ),
    hoverongaps=False,
    hovertemplate='<b>%{y}</b> & <b>%{x}</b><br>Count: %{z}<extra></extra>'
))

fig.update_layout(
    title='Code Co-occurrence Matrix',
    xaxis_title='',
    yaxis_title='',
    font=dict(family='Arial', size=12),
    width=600,
    height=600,
    plot_bgcolor='white'
)

fig.show()
```

**Interpretation Notes:**
- Symmetric matrices: Only show lower/upper triangle
- Normalize by row/column if categories have vastly different frequencies
- Include legend explaining co-occurrence meaning

### 3. Network Diagrams

**Primary Use:** Visualize code relationships and theme clustering

#### Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Library** | NetworkX + Plotly or Pyvis | Professional layouts, interactivity |
| **Layout Algorithm** | Fruchterman-Reingold (spring) | Natural clustering visualization |
| **Node Size** | Proportional to frequency (50-500px) | Visual hierarchy |
| **Node Color** | By cluster/theme (#1f77b4, #ff7f0e, #2ca02c) | Category distinction |
| **Edge Width** | Proportional to co-occurrence (1-10px) | Relationship strength |
| **Edge Color** | Light gray (#cccccc) with transparency | Non-distracting |
| **Labels** | Code names, 10pt font | Readability |

#### Example Code

```python
import networkx as nx
import plotly.graph_objects as go

# Create network
G = nx.Graph()
edges = [
    ('Service', 'Quality', 45),
    ('Service', 'Support', 34),
    ('Quality', 'Price', 56),
    ('Price', 'Delivery', 15),
    ('Delivery', 'Support', 22)
]
for node1, node2, weight in edges:
    G.add_edge(node1, node2, weight=weight)

# Layout
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Create edge traces
edge_trace = []
for edge in G.edges(data=True):
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    weight = edge[2]['weight']

    edge_trace.append(go.Scatter(
        x=[x0, x1, None],
        y=[y0, y1, None],
        mode='lines',
        line=dict(width=weight/10, color='#cccccc'),
        hoverinfo='none',
        showlegend=False
    ))

# Create node trace
node_trace = go.Scatter(
    x=[pos[node][0] for node in G.nodes()],
    y=[pos[node][1] for node in G.nodes()],
    mode='markers+text',
    text=list(G.nodes()),
    textposition='top center',
    marker=dict(
        size=[G.degree(node) * 10 for node in G.nodes()],
        color='#1f77b4',
        line=dict(width=2, color='#0d3d5c')
    ),
    textfont=dict(size=10, color='#333333')
)

# Combine
fig = go.Figure(data=edge_trace + [node_trace])
fig.update_layout(
    title='Code Co-occurrence Network',
    showlegend=False,
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white',
    width=800,
    height=600
)

fig.show()
```

**Interpretation Notes:**
- Clusters indicate frequently co-occurring codes
- Isolates may represent unique response patterns
- Edge thickness shows relationship strength

### 4. Word Clouds

**Primary Use:** Visual summary of open-ended text responses

#### Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Library** | WordCloud | Standard, well-supported |
| **Dimensions** | 1200x600 px (2:1 ratio) | Optimal display |
| **Background** | White | Professional appearance |
| **Colormap** | 'Blues' or custom brand colors | Consistency |
| **Max Words** | 100 | Avoid clutter |
| **Min Font Size** | 10 | Readability |
| **Max Font Size** | 100 | Visual hierarchy |
| **Relative Scaling** | 0.5 | Balance between frequent and rare words |
| **Stopwords** | Custom + NLTK defaults | Context-specific filtering |

#### Example Code

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Custom stopwords (extend NLTK defaults)
custom_stopwords = set([
    'said', 'would', 'could', 'also', 'really', 'like', 'just',
    # Add domain-specific stopwords
])

# Generate word cloud
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    colormap='Blues',
    max_words=100,
    min_font_size=10,
    max_font_size=100,
    relative_scaling=0.5,
    stopwords=custom_stopwords,
    collocations=True,  # Include bigrams
    normalize_plurals=True
).generate(text_data)

# Display
plt.figure(figsize=(12, 6), dpi=150)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Response Word Cloud', fontsize=18, color='#1f77b4', pad=20)
plt.tight_layout(pad=0)
plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
```

**Best Practices:**
- Pre-process text: lowercase, remove punctuation, lemmatize
- Use frequency weights, not just presence
- Avoid word clouds as sole analysis method (complement with charts)
- Consider separate clouds for positive/negative sentiment

### 5. Distribution Plots

**Primary Use:** Show statistical distributions (confidence scores, response lengths)

#### Specifications

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Library** | Seaborn (`sns.histplot`) or Plotly | Statistical focus |
| **Chart Type** | Histogram + KDE overlay | Distribution shape + smoothed trend |
| **Bins** | 20-30 (auto-calculated) | Balance detail and clarity |
| **Color** | Primary blue with transparency | Professional |
| **KDE Line** | Dark blue, 2px width | Emphasis |
| **Axes** | Labeled with units | Clear interpretation |

#### Example Code

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example: Confidence score distribution
confidence_scores = np.random.beta(8, 2, 1000)  # Example data

# Create plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

sns.histplot(
    confidence_scores,
    bins=30,
    kde=True,
    color='#1f77b4',
    alpha=0.6,
    edgecolor='#0d3d5c',
    linewidth=1.2,
    kde_kws={'linewidth': 2, 'color': '#0d3d5c'},
    ax=ax
)

# Styling
ax.set_xlabel('Confidence Score', fontsize=12, color='#333333')
ax.set_ylabel('Frequency', fontsize=12, color='#333333')
ax.set_title('Confidence Score Distribution', fontsize=16, color='#1f77b4', pad=15)
ax.grid(axis='y', alpha=0.3, color='#e0e0e0')
ax.set_axisbelow(True)

# Add mean line
mean_score = np.mean(confidence_scores)
ax.axvline(mean_score, color='#ff7f0e', linestyle='--', linewidth=2,
           label=f'Mean: {mean_score:.2f}')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
```

**Interpretation Notes:**
- Report mean, median, standard deviation
- Identify skewness (left, right, or normal)
- Flag outliers or bimodal distributions
- Compare against benchmark thresholds

---

## Color Scheme Standards

### Primary Color Palette

Consistent colors ensure brand recognition and professional appearance.

| Color Name | Hex Code | RGB | Use Case |
|------------|----------|-----|----------|
| **Primary Blue** | `#1f77b4` | (31, 119, 180) | Main charts, headings, primary elements |
| **Dark Blue** | `#0d3d5c` | (13, 61, 92) | Borders, emphasis, hover states |
| **Secondary Orange** | `#ff7f0e` | (255, 127, 14) | Highlights, secondary metrics, warnings |
| **Green (Positive)** | `#2ca02c` | (44, 160, 44) | Success states, positive trends |
| **Red (Negative)** | `#d62728` | (214, 39, 40) | Errors, negative trends, alerts |
| **Gray (Neutral)** | `#7f7f7f` | (127, 127, 127) | Supporting text, inactive states |
| **Light Gray (Background)** | `#f7f7f7` | (247, 247, 247) | Backgrounds, cards, dividers |
| **Grid Lines** | `#e0e0e0` | (224, 224, 224) | Chart grids, borders |

### Multi-Category Color Schemes

For charts with multiple categories (>2):

**Categorical (up to 10 categories):**
```python
CATEGORY_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf'   # Cyan
]
```

**Sequential (heatmaps, gradients):**
- Blues: `['#deebf7', '#9ecae1', '#4292c6', '#084594']`
- Greens: `['#e5f5e0', '#a1d99b', '#31a354', '#006d2c']`

**Diverging (showing deviation from center):**
- Blue-Red: `['#2166ac', '#4393c3', '#f7f7f7', '#d6604d', '#b2182b']`

### Accessibility Considerations

- **Contrast Ratio**: Minimum 4.5:1 for text, 3:1 for large text (WCAG AA)
- **Colorblind-Friendly**: Avoid red-green combinations; use blue-orange or shape/pattern differentiation
- **Testing Tool**: Use Color Oracle or browser extensions to simulate color vision deficiencies

---

## Table Formatting Standards

### Standard Table Structure

All tabular outputs (CSV exports, HTML reports, Streamlit displays) should follow these conventions:

#### Column Headers

| Element | Specification |
|---------|--------------|
| **Capitalization** | Title Case (e.g., "Code Name", "Frequency Count") |
| **Alignment** | Left for text, right for numbers |
| **Font Weight** | Bold |
| **Background** | Primary blue (#1f77b4) with white text |
| **Padding** | 12px top/bottom, 10px left/right |

#### Table Cells

| Element | Specification |
|---------|--------------|
| **Text Alignment** | Left for strings, right for numbers |
| **Number Format** | Integers: no decimals; Floats: 2 decimals; Percentages: 1 decimal |
| **Padding** | 10px top/bottom, 10px left/right |
| **Borders** | Bottom border only (1px, #ddd) for row separation |
| **Zebra Striping** | Alternate rows: white and #f7f7f7 |
| **Hover State** | Light blue background (#e3f2fd) |

#### Example Table (Markdown)

```markdown
| Code Name | Frequency | Percentage | Avg. Confidence |
|-----------|----------:|-----------:|----------------:|
| Customer Service | 145 | 36.3% | 0.87 |
| Product Quality | 98 | 24.5% | 0.92 |
| Pricing | 76 | 19.0% | 0.85 |
| Delivery | 54 | 13.5% | 0.79 |
| Other | 27 | 6.8% | 0.65 |
```

#### Example Table (HTML)

```html
<table class="results-table">
    <thead>
        <tr>
            <th style="text-align: left;">Code Name</th>
            <th style="text-align: right;">Frequency</th>
            <th style="text-align: right;">Percentage</th>
            <th style="text-align: right;">Avg. Confidence</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Customer Service</td>
            <td style="text-align: right;">145</td>
            <td style="text-align: right;">36.3%</td>
            <td style="text-align: right;">0.87</td>
        </tr>
        <!-- Additional rows -->
    </tbody>
</table>
```

### Number Formatting Guidelines

| Data Type | Format | Example |
|-----------|--------|---------|
| **Integer Counts** | No decimals, comma separators for 1000+ | `1,234` |
| **Percentages** | 1 decimal place, % symbol | `36.3%` |
| **Confidence Scores** | 2 decimal places, range 0-1 | `0.87` |
| **Currency** | 2 decimal places, $ symbol | `$123.45` |
| **Large Numbers** | Abbreviated with suffix (K, M, B) | `1.2M` |
| **Decimal Values** | 2-3 decimal places max | `3.142` |

### Sorting Standards

- **Default Sort**: Most frequent to least (descending)
- **Interactive Tables**: Allow user sorting on any column
- **Ties**: Secondary sort by alphabetical order

---

## Textual Inference Summaries

### Executive Summary Structure

Auto-generated executive summaries should follow this template:

#### Template

```markdown
## Executive Summary

**Project:** [Project Name]
**Analysis Date:** [Date]
**Total Responses:** [N]
**Codes Identified:** [N unique codes]
**Average Confidence:** [X.XX]

### Key Findings

1. **[Top Theme Name]** emerged as the dominant theme, mentioned in [N] responses ([X]% of total). [Brief interpretation: e.g., "indicating widespread concern about..."]

2. **[Second Theme]** was the second most common theme ([N] mentions, [X]%), with particularly strong representation in [demographic/segment if applicable].

3. **Co-occurrence Analysis** revealed strong relationships between [Theme A] and [Theme B] ([N] co-occurrences), suggesting [interpretation].

4. **Confidence Analysis** showed [X]% of codes were assigned with high confidence (>0.8), indicating [interpretation of data quality].

5. **[Notable Pattern]**: [Describe any unexpected findings, trends, or outliers]

### Recommendations

Based on the analysis, we recommend:

- [Action item 1 based on top theme]
- [Action item 2 addressing secondary findings]
- [Action item 3 for areas requiring further investigation]

### Data Quality Assessment

- **High Confidence Codes**: [X]% (>[threshold])
- **Low Confidence Codes**: [X]% (<[threshold]) - recommend manual review
- **Uncoded Responses**: [X]% - [interpretation]
```

### Key Metrics to Highlight

Always include these metrics in summaries:

1. **Total Response Count**: Overall sample size
2. **Unique Code Count**: Number of distinct themes identified
3. **Top 3-5 Themes**: Most frequent codes with counts and percentages
4. **Average Confidence Score**: Overall coding quality indicator
5. **Co-occurrence Insights**: Most common theme combinations
6. **Coverage Rate**: Percentage of responses successfully coded
7. **Quality Flags**: Any data quality issues identified

### Interpretation Guidelines

**Frequency Interpretation:**
- **Dominant** (>40%): "The overwhelming majority..."
- **Common** (20-40%): "A substantial portion..."
- **Moderate** (10-20%): "A notable segment..."
- **Emerging** (5-10%): "An emerging theme..."
- **Rare** (<5%): "A small but potentially significant group..."

**Confidence Interpretation:**
- **High** (>0.8): "Clear, unambiguous coding"
- **Moderate** (0.6-0.8): "Generally confident with some uncertainty"
- **Low** (<0.6): "Uncertain, recommend manual review"

**Co-occurrence Interpretation:**
- Report as: "[Theme A] and [Theme B] co-occurred in [N] responses ([X]% of [Theme A] mentions)"
- Interpret strength: >50% = strong relationship, 25-50% = moderate, <25% = weak

### Automated Summary Generation

```python
def generate_executive_summary(results_df, confidence_threshold=0.8):
    """
    Generate executive summary from coding results.

    Parameters:
    - results_df: DataFrame with columns ['response_id', 'code', 'confidence']
    - confidence_threshold: Minimum confidence for "high quality" designation

    Returns:
    - str: Formatted executive summary in Markdown
    """
    total_responses = results_df['response_id'].nunique()
    total_codes = results_df['code'].nunique()
    avg_confidence = results_df['confidence'].mean()

    # Top themes
    theme_counts = results_df['code'].value_counts()
    top_themes = theme_counts.head(5)

    # Confidence distribution
    high_confidence_pct = (results_df['confidence'] > confidence_threshold).mean() * 100

    summary = f"""## Executive Summary

**Total Responses:** {total_responses:,}
**Codes Identified:** {total_codes}
**Average Confidence:** {avg_confidence:.2f}

### Key Findings

1. **{top_themes.index[0]}** emerged as the dominant theme, mentioned in {top_themes.iloc[0]} responses ({top_themes.iloc[0]/total_responses*100:.1f}% of total).

2. **{top_themes.index[1]}** was the second most common ({top_themes.iloc[1]} mentions, {top_themes.iloc[1]/total_responses*100:.1f}%).

### Data Quality

- **High Confidence Codes**: {high_confidence_pct:.1f}%
- **Average Confidence**: {avg_confidence:.2f}

[Additional analysis...]
"""
    return summary
```

---

## Dashboard/UX Integration

### Streamlit Best Practices

The framework's Streamlit app (`app.py`) should follow these UX standards:

#### Layout Structure

```python
import streamlit as st

# Page config
st.set_page_config(
    page_title="Open-Ended Coding Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background-color: #f7f7f7;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }

    /* Button styling */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #0d3d5c;
    }
</style>
""", unsafe_allow_html=True)
```

#### Page Sections

| Section | Purpose | Components |
|---------|---------|------------|
| **Header** | Branding, navigation | Title, logo, description |
| **Sidebar** | Configuration, inputs | File upload, parameters, filters |
| **Main Content** | Analysis display | Tabs: Overview, Charts, Tables, Export |
| **Footer** | Metadata | Version, links, credits |

#### Interactive Element Guidelines

**File Uploaders:**
```python
uploaded_file = st.file_uploader(
    "Upload Response Data (CSV/Excel)",
    type=['csv', 'xlsx'],
    help="File should contain at least one column with open-ended text responses"
)
```

**Parameter Inputs:**
```python
col1, col2 = st.columns(2)
with col1:
    confidence_threshold = st.slider(
        "Minimum Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Codes below this confidence will be flagged for review"
    )

with col2:
    max_codes = st.number_input(
        "Maximum Codes per Response",
        min_value=1,
        max_value=10,
        value=3,
        help="Limit the number of codes assigned to each response"
    )
```

**Progress Indicators:**
```python
with st.spinner('Analyzing responses...'):
    progress_bar = st.progress(0)
    for i, response in enumerate(responses):
        # Process response
        progress_bar.progress((i + 1) / len(responses))
    st.success('Analysis complete!')
```

**Tabs for Organization:**
```python
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Charts", "📋 Data Table", "💾 Export"])

with tab1:
    # Executive summary
    st.markdown(executive_summary)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responses", total_responses)
    col2.metric("Unique Codes", unique_codes)
    col3.metric("Avg. Confidence", f"{avg_confidence:.2f}")
    col4.metric("Coverage", f"{coverage_pct:.1f}%")

with tab2:
    # Visualizations
    st.plotly_chart(freq_chart, use_container_width=True)
    st.plotly_chart(heatmap, use_container_width=True)

with tab3:
    # Data table with filtering
    st.dataframe(results_df, use_container_width=True)

with tab4:
    # Export options
    st.download_button("Download CSV", csv_data, "results.csv", "text/csv")
    st.download_button("Download Excel", excel_data, "results.xlsx", "application/vnd.ms-excel")
```

### Mobile Responsiveness

**Design Principles:**
- Use `use_container_width=True` for all charts and dataframes
- Column layouts: Collapse to single column on mobile
- Font sizes: Minimum 14px for body text
- Button sizes: Minimum 44px height for touch targets
- Test on viewport widths: 320px (mobile), 768px (tablet), 1024px+ (desktop)

**Responsive Columns:**
```python
# Desktop: 4 columns, Mobile: 1 column
if st.session_state.get('mobile_view', False):
    # Stack vertically
    st.metric("Total Responses", total_responses)
    st.metric("Unique Codes", unique_codes)
else:
    # Horizontal layout
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Responses", total_responses)
    col2.metric("Unique Codes", unique_codes)
```

### Performance Optimization

- **Caching**: Use `@st.cache_data` for expensive computations
- **Lazy Loading**: Load charts only when tab is selected
- **Pagination**: For tables with >1000 rows
- **Image Optimization**: Use PNG for charts, compress to <500KB

```python
@st.cache_data
def load_and_process_data(file_path):
    """Cache data loading to avoid re-processing on widget interactions"""
    df = pd.read_csv(file_path)
    # Processing...
    return df
```

---

## Export Format Specifications

### 1. CSV Export

**File Naming Convention:** `[project_name]_[export_type]_[YYYY-MM-DD].csv`

Example: `customer_feedback_coded_results_2026-02-23.csv`

#### Column Specifications

| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `response_id` | Integer | Unique identifier | `1`, `2`, `3` |
| `response_text` | String | Original text (quoted) | `"Great service!"` |
| `code` | String | Assigned code/theme | `Customer Service` |
| `confidence` | Float (2 decimals) | Confidence score | `0.87` |
| `timestamp` | ISO 8601 | Processing time | `2026-02-23T10:30:00` |
| `metadata_*` | Variable | Any additional metadata | `metadata_segment: Premium` |

#### Encoding and Format

```python
import pandas as pd

df.to_csv(
    'export.csv',
    index=False,
    encoding='utf-8-sig',  # Excel-friendly UTF-8
    quoting=csv.QUOTE_NONNUMERIC,  # Quote text fields
    line_terminator='\n',  # Unix-style line endings
    float_format='%.2f'  # 2 decimal places
)
```

**Best Practices:**
- Always include header row
- Escape double quotes in text: `"She said ""hello"""`
- Use UTF-8 encoding with BOM for Excel compatibility
- No commas in numeric values (use as separator only)

### 2. Excel Export (Multi-sheet)

**File Naming Convention:** `[project_name]_full_report_[YYYY-MM-DD].xlsx`

#### Sheet Organization

| Sheet Name | Contents | Sorting |
|------------|----------|---------|
| **Summary** | Executive metrics, top themes | N/A |
| **Coded Results** | Full coding output | Response ID ascending |
| **Frequency Table** | Code counts and percentages | Frequency descending |
| **Co-occurrence** | Co-occurrence matrix | Alphabetical |
| **Quality Metrics** | Confidence distribution, flags | N/A |
| **Data Dictionary** | Column definitions | N/A |

#### Formatting Specifications

```python
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Create writer
with pd.ExcelWriter('export.xlsx', engine='openpyxl') as writer:
    # Write sheets
    summary_df.to_excel(writer, sheet_name='Summary', index=False)
    results_df.to_excel(writer, sheet_name='Coded Results', index=False)
    frequency_df.to_excel(writer, sheet_name='Frequency Table', index=False)

    # Format workbook
    workbook = writer.book

    # Format each sheet
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]

        # Header formatting
        header_fill = PatternFill(start_color='1f77b4', end_color='1f77b4', fill_type='solid')
        header_font = Font(bold=True, color='FFFFFF', size=12)

        for cell in worksheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center', vertical='center')

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = get_column_letter(column[0].column)

            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass

            adjusted_width = min(max_length + 2, 50)  # Cap at 50
            worksheet.column_dimensions[column_letter].width = adjusted_width

        # Freeze top row
        worksheet.freeze_panes = 'A2'
```

**Cell Formatting:**
- **Headers**: Bold, white text, blue background (#1f77b4)
- **Numbers**: Right-aligned, 2 decimal places
- **Percentages**: Percent format (0.0%)
- **Dates**: `YYYY-MM-DD` format
- **Text**: Left-aligned, wrap text for long entries

### 3. JSON Export

**File Naming Convention:** `[project_name]_data_[YYYY-MM-DD].json`

#### Schema Structure

```json
{
    "metadata": {
        "project_name": "Customer Feedback Analysis",
        "export_date": "2026-02-23T10:30:00Z",
        "framework_version": "1.3.1",
        "total_responses": 500,
        "total_codes": 12,
        "average_confidence": 0.85
    },
    "parameters": {
        "confidence_threshold": 0.7,
        "max_codes_per_response": 3,
        "model_used": "gpt-4",
        "temperature": 0.0
    },
    "results": [
        {
            "response_id": 1,
            "response_text": "Great customer service!",
            "codes": [
                {
                    "code": "Customer Service",
                    "confidence": 0.92,
                    "reasoning": "Direct mention of customer service quality"
                }
            ],
            "timestamp": "2026-02-23T10:15:00Z"
        }
    ],
    "summary_statistics": {
        "code_frequencies": {
            "Customer Service": 145,
            "Product Quality": 98,
            "Pricing": 76
        },
        "confidence_distribution": {
            "mean": 0.85,
            "median": 0.87,
            "std": 0.12,
            "min": 0.45,
            "max": 0.99
        },
        "co_occurrence_matrix": {
            "Customer Service": {
                "Product Quality": 45,
                "Pricing": 23
            }
        }
    }
}
```

**Export Code:**

```python
import json
from datetime import datetime

export_data = {
    "metadata": {
        "project_name": project_name,
        "export_date": datetime.utcnow().isoformat() + 'Z',
        "framework_version": "1.3.1",
        "total_responses": len(results_df),
        "total_codes": results_df['code'].nunique(),
        "average_confidence": float(results_df['confidence'].mean())
    },
    "results": results_df.to_dict(orient='records'),
    # ... additional sections
}

with open('export.json', 'w', encoding='utf-8') as f:
    json.dump(export_data, f, indent=2, ensure_ascii=False)
```

**Best Practices:**
- Use ISO 8601 for timestamps
- Include schema version for compatibility
- Validate against JSON schema before export
- Use `ensure_ascii=False` for international characters

### 4. Image Exports

**Resolution and Format Standards:**

| Use Case | Format | Resolution | Size Guidelines |
|----------|--------|------------|-----------------|
| **Web Display** | PNG | 150 DPI | 800-1200px width |
| **Print** | PNG | 300 DPI | 2400-3600px width |
| **Vector (logos, diagrams)** | SVG | N/A (vector) | Optimize paths |
| **Presentations** | PNG | 150 DPI | 1920x1080 (16:9) |
| **Reports (embedded)** | PNG | 150-200 DPI | 6.5 inches width |

#### Export Code Examples

**Matplotlib/Seaborn:**
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
# ... create plot ...

# Save for web
fig.savefig(
    'chart_web.png',
    dpi=150,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none',
    format='png',
    transparent=False
)

# Save for print
fig.savefig(
    'chart_print.png',
    dpi=300,
    bbox_inches='tight',
    facecolor='white',
    format='png'
)

# Save as SVG
fig.savefig(
    'chart_vector.svg',
    format='svg',
    bbox_inches='tight'
)

plt.close()
```

**Plotly:**
```python
import plotly.graph_objects as go

fig = go.Figure(...)

# PNG export
fig.write_image(
    'chart.png',
    width=1200,
    height=600,
    scale=2  # 2x for high DPI
)

# SVG export
fig.write_image('chart.svg', format='svg')

# HTML (interactive)
fig.write_html(
    'chart.html',
    include_plotlyjs='cdn',
    config={'displayModeBar': True, 'displaylogo': False}
)
```

**File Naming:**
- Format: `[chart_type]_[description]_[date].png`
- Examples:
  - `frequency_bar_top_themes_2026-02-23.png`
  - `heatmap_cooccurrence_2026-02-23.png`
  - `network_code_relationships_2026-02-23.svg`

### 5. Complete Export Package

When providing full results to stakeholders, bundle all outputs:

**Directory Structure:**
```
project_name_export_2026-02-23/
├── README.md                          # Export documentation
├── executive_summary.pdf              # PDF summary report
├── data/
│   ├── coded_results.csv              # Main results
│   ├── full_report.xlsx               # Multi-sheet workbook
│   └── data_export.json               # Structured data
├── visualizations/
│   ├── frequency_bar.png
│   ├── cooccurrence_heatmap.png
│   ├── network_diagram.png
│   ├── wordcloud.png
│   └── confidence_distribution.png
└── metadata/
    ├── processing_log.txt             # Analysis log
    ├── parameters.json                # Configuration used
    └── data_dictionary.csv            # Column definitions
```

**Automated Package Creation:**

```python
import os
import shutil
from datetime import datetime

def create_export_package(project_name, results_df, charts):
    """Create comprehensive export package"""

    # Create directory
    timestamp = datetime.now().strftime('%Y-%m-%d')
    package_name = f"{project_name}_export_{timestamp}"
    os.makedirs(package_name, exist_ok=True)
    os.makedirs(f"{package_name}/data", exist_ok=True)
    os.makedirs(f"{package_name}/visualizations", exist_ok=True)
    os.makedirs(f"{package_name}/metadata", exist_ok=True)

    # Export data
    results_df.to_csv(f"{package_name}/data/coded_results.csv", index=False)
    # ... Excel, JSON exports ...

    # Save charts
    for chart_name, fig in charts.items():
        fig.write_image(f"{package_name}/visualizations/{chart_name}.png")

    # Create README
    readme_content = f"""# {project_name} - Analysis Export

**Export Date:** {timestamp}
**Framework Version:** 1.3.1

## Contents

- `data/`: All analysis results in multiple formats
- `visualizations/`: Charts and graphs (PNG, 300 DPI)
- `metadata/`: Processing details and documentation

## Quick Start

1. Review `executive_summary.pdf` for overview
2. Open `data/full_report.xlsx` for interactive data exploration
3. Visualizations are ready for presentations/reports

For questions, refer to the framework documentation.
"""

    with open(f"{package_name}/README.md", 'w') as f:
        f.write(readme_content)

    # Zip package
    shutil.make_archive(package_name, 'zip', package_name)

    return f"{package_name}.zip"
```

---

## Accessibility Guidelines

### Visual Accessibility

**Color Blindness:**
- Use colorblind-safe palettes (avoid red-green combinations)
- Include patterns/textures in addition to color
- Test charts with Color Oracle or similar tools

**Screen Readers:**
- Provide alt text for all images: `<img alt="Frequency bar chart showing Customer Service (145 mentions) as top theme">`
- Use semantic HTML: `<table>`, `<thead>`, `<th>` for data tables
- ARIA labels for interactive elements

**Text Contrast:**
- Minimum 4.5:1 ratio for normal text (WCAG AA)
- Minimum 3:1 ratio for large text (18pt+ or 14pt bold)
- Test with WebAIM Contrast Checker

### Keyboard Navigation

**Streamlit Apps:**
- Ensure all controls accessible via Tab key
- Provide keyboard shortcuts for common actions
- Skip navigation links for long pages

**Interactive Charts:**
- Plotly charts have built-in keyboard support
- Provide non-interactive alternatives (data tables)

### Cognitive Accessibility

**Clear Language:**
- Use plain language in summaries (8th-grade reading level)
- Define technical terms
- Use consistent terminology

**Visual Hierarchy:**
- Clear headings (H1, H2, H3)
- Chunked information (avoid walls of text)
- Whitespace for visual breathing room

---

## References

### Software Libraries

- **Plotly** (v5.0+): [https://plotly.com/python/](https://plotly.com/python/)
- **Matplotlib** (v3.5+): [https://matplotlib.org/](https://matplotlib.org/)
- **Seaborn** (v0.12+): [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
- **WordCloud** (v1.8+): [https://github.com/amueller/word_cloud](https://github.com/amueller/word_cloud)
- **NetworkX** (v3.0+): [https://networkx.org/](https://networkx.org/)
- **python-docx** (v0.8+): [https://python-docx.readthedocs.io/](https://python-docx.readthedocs.io/)
- **python-pptx** (v0.6+): [https://python-pptx.readthedocs.io/](https://python-pptx.readthedocs.io/)
- **openpyxl** (v3.0+): [https://openpyxl.readthedocs.io/](https://openpyxl.readthedocs.io/)
- **Streamlit** (v1.20+): [https://docs.streamlit.io/](https://docs.streamlit.io/)

### Design Resources

- **Color Brewer** (colorblind-safe palettes): [https://colorbrewer2.org/](https://colorbrewer2.org/)
- **WebAIM Contrast Checker**: [https://webaim.org/resources/contrastchecker/](https://webaim.org/resources/contrastchecker/)
- **Color Oracle** (colorblindness simulator): [https://colororacle.org/](https://colororacle.org/)
- **WCAG 2.1 Guidelines**: [https://www.w3.org/WAI/WCAG21/quickref/](https://www.w3.org/WAI/WCAG21/quickref/)

### Related Framework Documentation

- [Overview and Philosophy](./01_overview_and_philosophy.md)
- [Benchmark Standards](./02_benchmark_standards.md)
- [Data Processing Standards](./03_data_processing_standards.md)
- [Confidence Scoring Guidelines](./04_confidence_scoring_guidelines.md)
- [Agent Implementation Guide](./06_agent_implementation_guide.md)
- [Testing and Validation Procedures](./07_testing_and_validation_procedures.md)

---

**Document Control:**
- **Owner**: Agent-E (Reporting & Visualization)
- **Review Cycle**: Quarterly
- **Next Review**: March 2026
- **Change Log**: See Git history for detailed changes

---

*This documentation is part of the Open-Ended Coding Analysis Framework. For questions or contributions, see the project repository.*
