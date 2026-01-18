# Dataset Assessment Report
## Open-Ended Coding Analysis Framework

**Report Date:** January 18, 2026
**Assessment Scope:** Evaluating which datasets best demonstrate the full feature set of the ML-powered qualitative analysis framework

---

## Executive Summary

This report assesses **9 datasets** available in the project to determine which best demonstrate the framework's capabilities. The analysis considers text quality, realism, theme clarity, and alignment with key project features.

> **Note:** 4 low-quality datasets were removed following this assessment (Remote Work Survey, Fashion Trends Survey, Cricket Lovers Survey, IMDB Movie Reviews).

### Top Recommendations

| Rank | Dataset | Records | Best For |
|------|---------|---------|----------|
| 1 | Healthcare_Patient_Feedback_300.csv | 300 | **Primary demo** - Authentic text, clear themes, segmentation |
| 2 | Market_Research_Survey_300.csv | 300 | **Primary demo** - Demographics, consumer insights |
| 3 | Psychology_Wellbeing_Study_300.csv | 300 | **Complex themes** - Nuanced emotional analysis |
| 4 | GoEmotions Multi-Label.csv | 2,000 | **Multi-label demo** - 27 emotion categories |
| 5 | SNIPS Intent Classification.csv | 2,001 | **Classification accuracy** - Intent detection |

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

### Benchmark Datasets (Classification/Sentiment)

| Dataset | Size | Records | Label Column | Classes |
|---------|------|---------|--------------|---------|
| GoEmotions Multi-Label.csv | 179 KB | 2,000 | `emotions` | 27 emotions (multi-label) |
| SNIPS Intent Classification.csv | 129 KB | 2,001 | `intent` | 7 intent classes |
| AG News Classification.csv | 487 KB | 2,000 | `category` | 4 news categories |
| SemEval Twitter Sentiment.csv | 218 KB | 2,116 | `sentiment` | 3 sentiment classes |
| SST-5 Sentiment Dataset.csv | 4.5 KB | 75 | `sentiment` | 5 sentiment levels |
| SST-2 Sentiment Dataset.csv | 6.8 KB | 150 | `sentiment` | Binary (pos/neg) |

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

### Tier 2: Good Demo Datasets

#### 4. GoEmotions Multi-Label.csv
**Overall Rating: 4/5 Stars**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Text Quality | Good | Reddit-style, raw social media text |
| Multi-label Support | Excellent | Multiple emotions per response |
| Topic Diversity | Very High | Wide-ranging subjects |
| Theme Focus | Moderate | Scattered rather than domain-focused |

**Best For Demonstrating:**
- Multi-label classification capability
- Emotion detection features
- Handling informal online discourse
- Algorithm robustness with messy data

**Sample:**
> "Whoa this is really creepy" → emotions: disgust

---

#### 5. SNIPS Intent Classification.csv
**Overall Rating: 4/5 Stars**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Size | Large | 2,001 responses |
| Classification Clarity | Excellent | 7 well-defined intent classes |
| Text Type | Conversational | Voice assistant queries |

**Best For Demonstrating:**
- Classification accuracy metrics
- Intent/action-oriented coding
- Algorithm comparison on structured tasks

**Sample:**
> "Book a restaurant in Nevada for two people" → BookRestaurant

---

#### 6. AG News Classification.csv
**Overall Rating: 4/5 Stars**

| Criterion | Score | Notes |
|-----------|-------|-------|
| Size | Large | 2,000 articles, 487 KB |
| Text Length | Long | Full news article paragraphs |
| Category Clarity | Excellent | Business, Sports, Sci/Tech |

**Best For Demonstrating:**
- Long-form text processing
- Topic modeling at scale
- BERT/semantic embedding advantages

---

### Tier 3: Specialized/Limited Datasets

#### 7. SemEval Twitter Sentiment.csv
**Rating: 3.5/5 Stars** - Good for sentiment analysis demo, 2,116 tweets

#### 8. SST-5 Sentiment Dataset.csv
**Rating: 3/5 Stars** - Fine-grained 5-class sentiment, only 75 samples

#### 9. SST-2 Sentiment Dataset.csv
**Rating: 3/5 Stars** - Binary sentiment, 150 samples

---

## Feature-to-Dataset Mapping

| Project Feature | Best Dataset(s) | Why |
|-----------------|-----------------|-----|
| **Theme Discovery** | Healthcare, Market Research, Psychology | Authentic text with discoverable themes |
| **Segmentation Analysis** | Market Research (demographics), Healthcare (departments) | Built-in segment columns |
| **Multi-label Classification** | GoEmotions | Multiple emotions per response |
| **Confidence Scoring** | Psychology (nuanced themes) | Overlapping concepts test confidence |
| **Co-occurrence Analysis** | Healthcare | Related issues co-occur (wait + communication) |
| **Representative Quotes** | All Tier 1 datasets | Authentic, quotable responses |
| **Text Preprocessing** | Healthcare, Market Research | Intentional errors test robustness |
| **Long-form Text** | AG News | Paragraph-level content |
| **Short-form Text** | Twitter Sentiment | Tweet-length responses |
| **Sentiment Analysis** | SST-5 (5-class), SST-2 (binary), Twitter | Pre-labeled sentiment data |
| **Algorithm Comparison** | Psychology (nuanced) | Tests separation of subtle themes |
| **Executive Reporting** | Market Research | Business-relevant insights |

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

### Configuration 4: Multi-label/Emotion Demo
**Dataset:** GoEmotions Multi-Label.csv
```
- Algorithm: BERT + K-Means (semantic)
- Focus: Emotion detection, multi-label handling
- Advanced: Compare with TF-IDF for baseline
```

**Rationale:** Showcases multi-label capability and semantic embedding advantages.

---

### Configuration 5: Scale/Performance Demo
**Dataset:** AG News Classification.csv (2,000 rows)
```
- Algorithm: TF-IDF + K-Means (fast), then BERT (quality)
- Focus: Processing speed, large dataset handling
```

**Rationale:** Demonstrates performance with larger datasets.

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
| Multi-label classification | GoEmotions |
| Large-scale processing | AG News or SNIPS |
| Sentiment analysis | SST-5 or SemEval Twitter |

### Datasets Removed (Low Quality)
The following datasets were removed from the project after this assessment:
- **Remote Work Survey Data** - Too pre-processed, lacks authenticity
- **Fashion Trends Survey Data** - Same issue
- **Cricket Lovers Survey Data** - Same issue
- **IMDB Movie Reviews** - Only 36 samples, too small

---

## Conclusion

The **Healthcare_Patient_Feedback_300**, **Market_Research_Survey_300**, and **Psychology_Wellbeing_Study_300** datasets are specifically designed for pipeline validation and represent the best choices for demonstrating the framework's full capabilities. They provide:

- Authentic, realistic text with natural imperfections
- Clear, discoverable themes across multiple domains
- Built-in segmentation columns for demographic/group analysis
- Sufficient size (300 responses) for meaningful analysis
- Intentional preprocessing challenges to showcase robustness

For specialized demonstrations of specific features (multi-label, sentiment, large-scale), the benchmark datasets provide appropriate test cases.

---

*Report generated by multi-agent analysis assessing project features, dataset inventory, test patterns, and data quality.*
