# Dataset Assessment Report
## Open-Ended Coding Analysis Framework

**Report Date:** January 18, 2026
**Assessment Scope:** Evaluating which datasets best demonstrate the full feature set of the ML-powered qualitative analysis framework

---

## Executive Summary

This report assesses the **6 curated datasets** available in the project, optimized for demonstrating the framework's capabilities. The datasets were rationalized from 12 to 6, removing pre-classified benchmark datasets that don't align with the tool's open-ended qualitative analysis purpose.

> **Dataset Rationalization (January 2026):** Removed 6 low-value datasets (AG News, GoEmotions, SNIPS, SemEval, SST-2, SST-5) that were pre-labeled classification datasets unsuitable for theme discovery. Expanded demo datasets to 200 rows each for better ML clustering results.

### Curated Dataset Inventory

| Dataset | Records | Avg Length | Best For |
|---------|---------|------------|----------|
| Psychology_Wellbeing_Study_300.csv | 300 | 95 chars | **Best quality** - Rich emotional content, natural themes |
| Healthcare_Patient_Feedback_300.csv | 300 | 69 chars | **Domain-specific** - Patient experience patterns |
| Market_Research_Survey_300.csv | 300 | 60 chars | **Business use** - Consumer insights, demographics |
| Remote_Work_Experiences_200.csv | 200 | 77 chars | **Quick demos** - Remote work themes |
| cricket_responses.csv | 200 | 79 chars | **Topic variety** - 40+ natural topics |
| fashion_responses.csv | 200 | 69 chars | **Theme diversity** - 45+ natural topics |

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

### ML Algorithms Supported
- TF-IDF + K-Means (default)
- Latent Dirichlet Allocation (LDA)
- Non-negative Matrix Factorization (NMF)
- BERT + K-Means (semantic)
- LSTM + K-Means (sequence-aware)

---

## Complete Dataset Inventory

### Survey Datasets (Purpose-Built for Testing)

| Dataset | Size | Records | Segmentation Column | Key Themes |
|---------|------|---------|---------------------|------------|
| Healthcare_Patient_Feedback_300.csv | 34 KB | 300 | `department` | Wait times, staff, communication, cleanliness |
| Market_Research_Survey_300.csv | 32 KB | 300 | `demographic_segment` | Quality, price, service, loyalty |
| Psychology_Wellbeing_Study_300.csv | 39 KB | 300 | `age_group` | Burnout, balance, mental health, growth |

### Demo Datasets (200 rows each)

| Dataset | Size | Records | Topic Column | Content Focus |
|---------|------|---------|--------------|---------------|
| Remote_Work_Experiences_200.csv | 22 KB | 200 | `topic` | Remote work themes (30+ topics) |
| cricket_responses.csv | 21 KB | 200 | `topic` | Cricket perspectives (40+ topics) |
| fashion_responses.csv | 19 KB | 200 | `topic` | Fashion opinions (45+ topics) |

---

## Detailed Dataset Assessments

### Tier 1: Excellent Demo Datasets

#### 1. Healthcare_Patient_Feedback_300.csv
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

#### 2. Market_Research_Survey_300.csv
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

#### 3. Psychology_Wellbeing_Study_300.csv
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

### Tier 2: Demo Datasets

#### 4. Remote_Work_Experiences_200.csv
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

#### 5. cricket_responses.csv
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

#### 6. fashion_responses.csv
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
| **Quick Demonstrations** | Remote Work Experiences | Balanced size and content |
| **Algorithm Comparison** | Psychology (nuanced) | Tests separation of subtle themes |
| **Executive Reporting** | Market Research | Business-relevant insights |
| **Lifestyle/Cultural** | Fashion, Cricket | Domain-specific theme extraction |

---

## Recommended Demo Configurations

### Configuration 1: Full Feature Showcase
**Dataset:** Healthcare_Patient_Feedback_300.csv
```
- Algorithm: TF-IDF + K-Means
- Number of codes: 6-8
- Segmentation: By department
- Outputs: All 15 outputs enabled
```

**Rationale:** Demonstrates all core features with authentic healthcare data. Clear themes emerge, segmentation shows meaningful patterns, preprocessing handles intentional errors.

---

### Configuration 2: Business Use Case Demo
**Dataset:** Market_Research_Survey_300.csv
```
- Algorithm: TF-IDF + K-Means
- Number of codes: 5-7
- Segmentation: By demographic_segment
- Focus: Executive summary, actionable insights
```

**Rationale:** Shows business relevance with consumer feedback analysis and demographic breakdown.

---

### Configuration 3: Research/Academic Demo
**Dataset:** Psychology_Wellbeing_Study_300.csv
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
Use the **three 300-row test datasets** in order:
1. **Healthcare_Patient_Feedback_300** - Best overall demonstration
2. **Market_Research_Survey_300** - Business context
3. **Psychology_Wellbeing_Study_300** - Research context

### For Specific Features
| Feature to Highlight | Use This Dataset |
|---------------------|------------------|
| Preprocessing pipeline | Healthcare (has intentional errors) |
| Demographic segmentation | Market Research |
| Complex theme extraction | Psychology Wellbeing |
| Quick demos | Remote_Work_Experiences_200 |
| Topic variety | cricket_responses (40+ topics) |
| Theme diversity | fashion_responses (45+ topics) |

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

The **6 curated datasets** provide optimal coverage for the framework's capabilities:

### Primary Analysis Datasets (300 rows each)
- **Psychology_Wellbeing_Study_300** - Best overall quality, rich emotional content
- **Healthcare_Patient_Feedback_300** - Domain-specific feedback patterns
- **Market_Research_Survey_300** - Business/consumer insights

### Demo Datasets (200 rows each)
- **Remote_Work_Experiences_200** - Remote work themes (ideal for quick demos)
- **cricket_responses** - Topic variety (40+ natural topics)
- **fashion_responses** - Theme diversity (45+ natural topics)

All datasets feature:
- Authentic, open-ended qualitative text
- 60-95 character average length (optimal for theme discovery)
- Consistent column structure for pipeline compatibility
- Natural topic variety for demonstrating ML clustering
- Suitable size for meaningful analysis (200-300 rows)

---

*Report updated January 2026 following dataset rationalization and optimization.*
