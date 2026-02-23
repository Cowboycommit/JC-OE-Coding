# Dataset Assessment Report
## Open-Ended Coding Analysis Framework

**Report Date:** February 11, 2026
**Assessment Scope:** Evaluating which datasets best demonstrate the full feature set of the ML-powered qualitative analysis framework

---

## Executive Summary

This report assesses the **6 curated datasets** available in the project, optimized for demonstrating the framework's capabilities. The datasets were rationalized from 12 to 6, removing pre-classified benchmark datasets that don't align with the tool's open-ended qualitative analysis purpose.

> **Dataset Expansion (February 2026):** All 6 datasets expanded to 1,000 rows each using seed-based variation (synonym substitution, phrase recombination, style variation) via `scripts/expand_datasets.py`. Original smaller datasets retained for backward compatibility. Added optimal sampling feature for targeted code counts (5/10/15/20 codes).

### Curated Dataset Inventory

| Dataset | Records | Avg Length | Best For |
|---------|---------|------------|----------|
| Psychology_Wellbeing_Study_1000.csv | 1,000 | 95 chars | **Best quality** - Rich emotional content, natural themes |
| Healthcare_Patient_Feedback_1000.csv | 1,000 | 69 chars | **Domain-specific** - Patient experience patterns |
| Market_Research_Survey_1000.csv | 1,000 | 60 chars | **Business use** - Consumer insights, demographics |
| Remote_Work_Experiences_1000.csv | 1,000 | 77 chars | **Quick demos** - Remote work themes |
| cricket_responses_1000.csv | 1,000 | 79 chars | **Topic variety** - 40+ natural topics |
| fashion_responses_1000.csv | 1,000 | 69 chars | **Theme diversity** - 45+ natural topics |

---

## Project Features to Demonstrate

The framework provides **15 essential outputs** for qualitative analysis:

1. **Code Assignments** - Response-level codes with confidence scores
2. **Codebook** - Auto-generated code definitions with examples
3. **Frequency Tables** - Statistical distribution of codes
4. **Quality Metrics** - Confidence scores and reliability measures
5. **Binary Matrix** - Code presence/absence for statistical analysis
6. **Representative Quotes** - Top examples for each code
7. **Co-Occurrence Analysis** - Code relationship patterns and networks
8. **Descriptive Statistics** - Comprehensive summary statistics
9. **Segmentation Analysis** - Code patterns across demographics/groups
10. **QA Report** - Validation and error analysis
11. **Visualizations** - Charts, heatmaps, network diagrams, word clouds
12. **Multiple Exports** - CSV, Excel, JSON formats
13. **Method Documentation** - Transparent methodology
14. **Uncoded Responses** - Edge cases and low-confidence items
15. **Executive Summaries** - High-level stakeholder reports

### ML Algorithms Supported (All 6 Methods)
- TF-IDF + K-Means (default)
- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)
- BERT + K-Means (semantic)
- LSTM + K-Means (sequence-aware)
- SVM Spectral Clustering (non-linear boundaries)

---

## Complete Dataset Inventory

### Survey Datasets (1,000 rows each)

| Dataset | Records | Segmentation Column | Key Themes |
|---------|---------|---------------------|------------|
| Healthcare_Patient_Feedback_1000.csv | 1,000 | `department` | Wait times, staff, communication, cleanliness |
| Market_Research_Survey_1000.csv | 1,000 | `demographic_segment` | Quality, price, service, loyalty |
| Psychology_Wellbeing_Study_1000.csv | 1,000 | `age_group` | Burnout, balance, mental health, growth |

### Topic Datasets (1,000 rows each)

| Dataset | Records | Topic Column | Content Focus |
|---------|---------|--------------|---------------|
| Remote_Work_Experiences_1000.csv | 1,000 | `topic` | Remote work themes (30+ topics) |
| cricket_responses_1000.csv | 1,000 | `topic` | Cricket perspectives (40+ topics) |
| fashion_responses_1000.csv | 1,000 | `topic` | Fashion opinions (45+ topics) |

---

## Detailed Dataset Assessments

### Tier 1: Excellent Demo Datasets

#### 1. Healthcare_Patient_Feedback_1000.csv
**Overall Rating: 5/5 Stars**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Text Quality | Excellent | Natural imperfections, authentic typos |
| Realism | Excellent | Genuine patient voice with emotional expression |
| Theme Clarity | Excellent | Clear healthcare-specific themes emerge |
| Segmentation Support | Excellent | `department` column for cross-segment analysis |
| Preprocessing Challenge | High | Intentional errors test pipeline robustness |

**Sample Response:**
> "The ER wait was wayyy too long!!! Had to wait 3 hours & nobody told us what was happening"

**Key Themes Discovered:**
- Wait times and access issues
- Staff professionalism and bedside manner
- Cleanliness and facilities
- Treatment effectiveness
- Communication clarity
- Pain management

**Best For Demonstrating:**
- Theme discovery in domain-specific feedback
- Segmentation analysis by department
- Text preprocessing handling
- Co-occurrence between themes (e.g., wait time + communication)

---

#### 2. Market_Research_Survey_1000.csv
**Overall Rating: 5/5 Stars**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Text Quality | Excellent | Mix of minimal and expansive responses |
| Realism | Excellent | Typos, colloquialisms, varied opinion types |
| Theme Clarity | Excellent | Consumer-focused themes well separated |
| Segmentation Support | Excellent | `demographic_segment` (Young Adults, Middle Aged, Seniors) |
| Business Relevance | High | Actionable market insights |

**Sample Response:**
> "The product quality is AMAZING!!! I loooove how durable it is & would definately recommend"

**Key Themes Discovered:**
- Product quality and durability
- Customer service responsiveness
- Pricing and value perception
- Store experience and layout
- Brand loyalty patterns
- Product innovation requests

**Best For Demonstrating:**
- Demographic-based segmentation analysis
- Consumer sentiment extraction
- Business-actionable theme identification
- Representative quote selection for reports

---

#### 3. Psychology_Wellbeing_Study_1000.csv
**Overall Rating: 5/5 Stars**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Text Quality | Excellent | Longer, narrative-style responses |
| Realism | Excellent | Authentic emotional expression, personal stories |
| Theme Complexity | High | Overlapping, nuanced themes |
| Segmentation Support | Good | `age_group` column |
| Sensitivity | Medium | Mental health content requires careful handling |

**Sample Response:**
> "I've been feeling soooo burnt out lately... work just never stops & I can't find time for myself"

**Key Themes Discovered:**
- Work burnout and stress
- Work-life balance boundaries
- Social connection and loneliness
- Mental health treatment approaches
- Self-care practices
- Gratitude and life satisfaction
- Personal growth and coping strategies

**Best For Demonstrating:**
- Complex, nuanced theme discovery
- Handling overlapping concepts
- Age-based segmentation patterns
- Qualitative research depth

---

### Tier 2: Topic Datasets (1,000 rows each)

#### 4. Remote_Work_Experiences_1000.csv
**Overall Rating: 4.5/5 Stars**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Text Quality | Excellent | Natural remote work feedback |
| Topic Variety | High | 30+ distinct themes |
| Realism | Excellent | Authentic workplace experiences |
| Demo Suitability | High | Quick, effective demonstrations |

**Best For Demonstrating:**
- Quick demos and testing
- Workplace theme discovery
- Work-life balance patterns

**Sample:**
> "Working from home has improved my productivity, though I miss the spontaneous conversations with colleagues"

---

#### 5. cricket_responses_1000.csv
**Overall Rating: 4.5/5 Stars**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Topic Diversity | Very High | 40+ cricket-related topics |
| Theme Clarity | Good | Clear sporting themes emerge |
| Enthusiasm | High | Authentic fan perspectives |

**Best For Demonstrating:**
- High topic variety analysis
- Sports/enthusiasm detection
- Cultural content handling

**Sample:**
> "The IPL has revolutionized cricket with its T20 format and brought international players together"

---

#### 6. fashion_responses_1000.csv
**Overall Rating: 4.5/5 Stars**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Theme Diversity | Very High | 45+ fashion topics |
| Content Range | Excellent | Sustainability, trends, personal style |
| Opinion Variety | High | Mixed perspectives |

**Best For Demonstrating:**
- Theme diversity and complexity
- Lifestyle/cultural analysis
- Sustainability discussions

**Sample:**
> "Fast fashion needs to change - I'm trying to buy more sustainable pieces even if they cost more"

---

## Feature-to-Dataset Mapping

| Project Feature | Best Dataset(s) | Why |
|-----------------|-----------------|-----|
| **Theme Discovery** | Healthcare, Market Research, Psychology | Authentic text with discoverable themes |
| **Segmentation Analysis** | Market Research (demographics), Healthcare (departments) | Built-in segment columns |
| **Topic Variety Analysis** | Cricket (40+ topics), Fashion (45+ topics) | High natural topic diversity |
| **Confidence Scoring** | Psychology (nuanced themes) | Overlapping concepts test confidence |
| **Co-occurrence Analysis** | Healthcare | Related issues co-occur (wait + communication) |
| **Representative Quotes** | All 6 datasets | Authentic, quotable responses |
| **Text Preprocessing** | Healthcare, Market Research | Intentional errors test robustness |
| **Quick Demonstrations** | Remote Work Experiences | Balanced content, use optimal sampling |
| **Algorithm Comparison** | Psychology (nuanced) | Tests separation of subtle themes |
| **Executive Reporting** | Market Research | Business-relevant insights |
| **Lifestyle/Cultural** | Fashion, Cricket | Domain-specific theme extraction |

---

## Recommended Demo Configurations

### Configuration 1: Full Feature Showcase
**Dataset:** Healthcare_Patient_Feedback_1000.csv
```
- Algorithm: TF-IDF + K-Means
- Number of codes: 6-8
- Segmentation: By department
- Outputs: All 15 outputs enabled
```

**Rationale:** Demonstrates all core features with authentic healthcare data. Clear themes emerge, segmentation shows meaningful patterns, preprocessing handles intentional errors.

---

### Configuration 2: Business Use Case Demo
**Dataset:** Market_Research_Survey_1000.csv
```
- Algorithm: TF-IDF + K-Means
- Number of codes: 5-7
- Segmentation: By demographic_segment
- Focus: Executive summary, actionable insights
```

**Rationale:** Shows business relevance with consumer feedback analysis and demographic breakdown.

---

### Configuration 3: Research/Academic Demo
**Dataset:** Psychology_Wellbeing_Study_1000.csv
```
- Algorithm: NMF or LDA
- Number of codes: 7-10
- Segmentation: By age_group
- Focus: Codebook generation, method documentation
```

**Rationale:** Demonstrates nuanced theme extraction for academic research contexts.

---

### Configuration 4: Topic Variety Demo
**Dataset:** cricket_responses.csv
```
- Algorithm: TF-IDF + K-Means or LDA
- Number of codes: 8-12
- Focus: High topic diversity handling
```

**Rationale:** Demonstrates framework's ability to discover and organize 40+ natural topics.

---

### Configuration 5: Lifestyle/Cultural Demo
**Dataset:** fashion_responses.csv
```
- Algorithm: NMF or LDA
- Number of codes: 8-10
- Focus: Theme diversity, sustainability discussions
```

**Rationale:** Shows cultural content analysis with 45+ diverse fashion-related themes.

---

## Summary Recommendations

### For General Demonstrations
Use the **six 1,000-row datasets** with optimal sampling:
1. **Healthcare_Patient_Feedback_1000** - Best overall demonstration
2. **Market_Research_Survey_1000** - Business context
3. **Psychology_Wellbeing_Study_1000** - Research context

### Optimal Sampling for Demonstrations
All datasets support optimal sampling for targeted code counts:
| Target Codes | Sample Size | Responses per Code |
|---|---|---|
| 5 codes | 150 responses | 30 per code |
| 10 codes | 300 responses | 30 per code |
| 15 codes | 500 responses | 33 per code |
| 20 codes | 700 responses | 35 per code |

### For Specific Features
| Feature to Highlight | Use This Dataset |
|---------------------|------------------|
| Preprocessing pipeline | Healthcare (has intentional errors) |
| Demographic segmentation | Market Research |
| Complex theme extraction | Psychology Wellbeing |
| Quick demos | Remote_Work_Experiences_1000 |
| Topic variety | cricket_responses_1000 (40+ topics) |
| Theme diversity | fashion_responses_1000 (45+ topics) |

### Dataset Column Structure

All datasets follow a consistent structure optimized for the analysis pipeline:

| Column | Description | Required |
|--------|-------------|----------|
| `id` | Unique row identifier | Yes |
| `response` | Primary text for analysis | Yes |
| `*_id` | Respondent/participant identifier | Yes |
| `timestamp` | Response timestamp | Yes |
| `topic/segment` | Category/segmentation column | Yes |

### Datasets Removed (Rationalization)
The following datasets were removed as they don't align with open-ended qualitative analysis:
- **AG News Classification** (2,000 rows) - Pre-classified news categories
- **GoEmotions Multi-Label** (2,000 rows) - Pre-labeled emotion classification
- **SNIPS Intent Classification** (2,000 rows) - Templated voice commands
- **SemEval Twitter Sentiment** (2,000 rows) - Pre-labeled sentiment
- **SST-2 Sentiment Dataset** (150 rows) - Too short (34 chars avg)
- **SST-5 Sentiment Dataset** (75 rows) - Too short (47 chars avg)

---

## Conclusion

The **6 curated datasets** (1,000 rows each, 6,000 total) provide optimal coverage for the framework's capabilities:

### All Datasets (1,000 rows each)
- **Psychology_Wellbeing_Study_1000** - Best overall quality, rich emotional content
- **Healthcare_Patient_Feedback_1000** - Domain-specific feedback patterns
- **Market_Research_Survey_1000** - Business/consumer insights
- **Remote_Work_Experiences_1000** - Remote work themes (ideal for quick demos)
- **cricket_responses_1000** - Topic variety (40+ natural topics)
- **fashion_responses_1000** - Theme diversity (45+ natural topics)

All datasets feature:
- Authentic, open-ended qualitative text expanded from curated seeds
- 60-95 character average length (optimal for theme discovery)
- Consistent column structure for pipeline compatibility
- Natural topic variety for demonstrating ML clustering
- 1,000 rows each - sufficient for all 6 ML methods at good accuracy
- Built-in optimal sampling support (150/300/500/700 rows for 5/10/15/20 codes)

---

*Report updated February 2026 following dataset expansion to 1,000 rows and concordance alignment.*
