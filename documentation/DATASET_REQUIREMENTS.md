# Dataset Requirements Guide

This document summarises the current dataset inventory, minimum requirements, and best-practice recommendations for achieving optimal results with the open-ended coding framework.

**Last Updated:** 2026-02-23
**Framework Version:** 1.4.0

---

## 1. Current Datasets

The project ships with **6 curated datasets** (6,000 rows total), each containing 1,000 rows expanded from original seeds using a variation engine (synonym substitution, phrase recombination, casual style markers). The expansion script is `scripts/expand_datasets.py`.

| Dataset | Rows | Response Column | Best Use Case |
|---------|------|-----------------|---------------|
| `Healthcare_Patient_Feedback_1000.csv` | 1,000 | `response` | Domain-specific patient experience patterns |
| `Market_Research_Survey_1000.csv` | 1,000 | `response` | Consumer insights and demographic analysis |
| `Psychology_Wellbeing_Study_1000.csv` | 1,000 | `response` | Rich emotional content, natural themes |
| `Remote_Work_Experiences_1000.csv` | 1,000 | `response` | Remote work feedback, 30+ themes |
| `cricket_responses_1000.csv` | 1,000 | `response` | Topic variety, 40+ natural cricket topics |
| `fashion_responses_1000.csv` | 1,000 | `response` | Theme diversity, 45+ fashion topics |

**Legacy datasets** (300/200 rows) are retained in the `data/` folder for quick testing but are not the primary analysis targets.

### Optimal Sampling

All interfaces (app.py, app_lite.py, both notebooks) provide an optimal sample size selector for target code counts:

| Target Codes | Sample Size | Responses per Code |
|---|---|---|
| 5 codes | 150 responses | 30 per code |
| 10 codes | 300 responses | 30 per code |
| 15 codes | 500 responses | 33 per code |
| 20 codes | 700 responses | 35 per code |

---

## 2. Minimum Dataset Requirements

### Hard Minimums

| Metric | Minimum | Notes |
|--------|---------|-------|
| Response count | 20 | Statistical validity requires at least 20-30 responses |
| Response length | 5 characters | Shorter responses are flagged for quality review |
| Words per response | 1 word | Single-word responses may lack analytic value |
| Analytic response ratio | 50% | Non-analytic responses are flagged but retained |
| Language | English | Current version supports English only |
| Encoding | UTF-8 | Required for all input files |

### Recommended Minimums (for robust analysis)

| Metric | Recommended | Project Standard |
|--------|-------------|------------------|
| Response count | 50+ | 300-1,000 |
| Response length | 10+ characters | 60-95 chars average |
| Words per response | 3+ | 10-20 typical |
| Analytic response ratio | 80%+ | 95%+ |

---

## 3. Dataset Size and Algorithm Selection

The framework supports 6 ML methods. Algorithm suitability varies by dataset size.

### Small Datasets (< 100 responses)

- **Prefer:** TF-IDF + K-Means, NMF, SVM Spectral
- **Avoid:** LDA (needs more data), LSTM (needs more data)
- **Recommended codes:** 3-5
- **Notes:** Simpler and faster methods; manual validation becomes more critical

### Medium Datasets (100-500 responses) — Optimal Range

- **Best methods:** All 6 methods work well (TF-IDF, LDA, NMF, LSTM, BERT, SVM)
- **Recommended codes:** 5-15 (varies by complexity)
- **Notes:** Good balance of statistical validity and efficiency. Use optimal sampling for best results.

### Large Datasets (500+ responses)

- **Prefer:** LDA and NMF for topic modeling; TF-IDF for speed
- **Also viable:** BERT, LSTM
- **Avoid:** SVM Spectral above 2,000 responses (O(n^3) cost)
- **Recommended codes:** 10-20+
- **Notes:** Better statistical power for validation; higher computational cost for deep learning methods

See `documentation/OPTIMAL_DATASET_SIZE.md` for detailed method-by-method guidance.

---

## 4. Recommended Code Counts by Dataset

| Dataset | Rows | Recommended Codes | Suggested Algorithm |
|---------|------|-------------------|---------------------|
| `Healthcare_Patient_Feedback_1000.csv` | 1,000 | 8-15 | TF-IDF + K-Means or LDA |
| `Market_Research_Survey_1000.csv` | 1,000 | 8-12 | TF-IDF + K-Means |
| `Psychology_Wellbeing_Study_1000.csv` | 1,000 | 10-15 | NMF or LDA |
| `Remote_Work_Experiences_1000.csv` | 1,000 | 10-15 | TF-IDF + K-Means or NMF |
| `cricket_responses_1000.csv` | 1,000 | 10-15 | TF-IDF + K-Means |
| `fashion_responses_1000.csv` | 1,000 | 10-15 | TF-IDF + K-Means |

---

## 5. Quality Benchmarks

### Output Quality Targets

| Metric | Target |
|--------|--------|
| Silhouette score | 0.3+ |
| Average confidence | 0.7+ |
| Coded response coverage | 95%+ |
| Uncoded responses | < 5% |

### Code Balance Rules

- No single code should dominate more than 50% of responses.
- 10-30% of responses should receive 2+ codes (realistic multi-coding).
- Every code should appear in at least 2-3% of responses.

### Response Quality Flags (from `src/data_loader.py`)

| Check | Threshold | Action |
|-------|-----------|--------|
| Minimum words | 3+ recommended | Flagged for review |
| Minimum characters | 5+ enforced | Flagged for review |
| Maximum repetition ratio | 0.7 (70%) | Flagged if exceeded |
| Minimum English word ratio | 0.3 (30%) | Flagged if exceeded |

All responses are retained and flagged — no automatic exclusion occurs.

---

## 6. Supported ML Methods

| Method | Minimum Responses | Optimal Range | Speed |
|--------|-------------------|---------------|-------|
| TF-IDF + K-Means | 50 | 200-500 | Fast |
| LDA | 100 | 500-2,000 | Moderate |
| NMF | 50 | 200-500 | Fast |
| LSTM + K-Means | 200 | 500-2,000 | Slow |
| BERT + K-Means | 50 | 200-500 | Moderate |
| SVM Spectral | 30 | 100-500 | Moderate |

---

## 7. Performance Expectations

| Dataset Size | Expected Processing Time (TF-IDF) | Deep Learning Methods |
|---|---|---|
| < 300 rows | 5-15 seconds | 30-120 seconds |
| 300-1,000 rows | 10-30 seconds | 1-5 minutes |
| 1,000+ rows | 30-60 seconds | 2-10 minutes |

---

## 8. Bringing Your Own Data

When preparing a custom dataset for this framework:

1. **Format:** CSV or Excel with a single text column containing open-ended responses.
2. **Minimum 50 responses** for meaningful results (20 absolute minimum).
3. **Average 10+ characters per response** to provide enough signal.
4. **80%+ analytic ratio** — most responses should contain substantive content.
5. **UTF-8 encoding** — ensure special characters are properly encoded.
6. **English language** — the current pipeline supports English only.
7. **Start with fewer codes** (5-7) and increase if the silhouette score is low or themes appear merged.
8. **Use the data template:** Available at `documentation/input_data_template.xlsx` or via the Streamlit UI.

---

*See also: [Input Data Specification](03_input_data_specification.md) | [Methods](METHODS.md) | [Optimal Dataset Size](OPTIMAL_DATASET_SIZE.md) | [Validation and Demonstration](06_validation_and_demonstration.md)*
