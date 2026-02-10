# Dataset Requirements Guide

This document summarises the current dataset inventory, minimum requirements, and best-practice recommendations for achieving optimal results with the open-ended coding framework.

---

## 1. Current Datasets

The project ships with **6 curated datasets** (1,500 rows total), split into two tiers.

### Tier 1 — Primary Analysis Datasets (300 rows each)

| Dataset | Rows | Avg Response Length | Best Use Case |
|---------|------|---------------------|---------------|
| `Psychology_Wellbeing_Study_300.csv` | 300 | 95 chars | Rich emotional content, natural themes |
| `Healthcare_Patient_Feedback_300.csv` | 300 | 69 chars | Domain-specific patient experience patterns |
| `Market_Research_Survey_300.csv` | 300 | 60 chars | Consumer insights and demographic analysis |

### Tier 2 — Demo / Quick-Test Datasets (200 rows each)

| Dataset | Rows | Avg Response Length | Best Use Case |
|---------|------|---------------------|---------------|
| `Remote_Work_Experiences_200.csv` | 200 | 77 chars | Quick demos, 30+ remote-work topics |
| `cricket_responses.csv` | 200 | 79 chars | Topic variety, 40+ natural cricket topics |
| `fashion_responses.csv` | 200 | 69 chars | Theme diversity, 45+ fashion topics |

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
| Response count | 50+ | 200-300 |
| Response length | 10+ characters | 60-95 chars average |
| Words per response | 3+ | 10-20 typical |
| Analytic response ratio | 80%+ | 95%+ |

---

## 3. Dataset Size and Algorithm Selection

The framework selects algorithms based on the number of responses available.

### Small Datasets (< 100 responses)

- **Prefer:** TF-IDF + K-Means, SVM Spectral
- **Avoid:** LDA (needs more data), LSTM (needs more data)
- **Recommended codes:** 3-5
- **Notes:** Simpler and faster methods; manual validation becomes more critical

### Medium Datasets (100-500 responses) — Optimal Range

- **Best methods:** TF-IDF + K-Means, LDA, BERT
- **Recommended codes:** 5-12 (varies by complexity)
- **Notes:** All algorithms work well; good balance of statistical validity and efficiency. The current project datasets (200-300 rows) sit in this range.

### Large Datasets (500+ responses)

- **Prefer:** LDA and topic modelling
- **Also viable:** BERT, LSTM
- **Recommended codes:** 10-15+
- **Notes:** Better statistical power for validation; higher computational cost

---

## 4. Recommended Code Counts by Dataset

| Dataset | Rows | Recommended Codes | Suggested Algorithm |
|---------|------|-------------------|---------------------|
| `Healthcare_Patient_Feedback_300.csv` | 300 | 6-8 | TF-IDF + K-Means |
| `Market_Research_Survey_300.csv` | 300 | 5-7 | TF-IDF + K-Means |
| `Psychology_Wellbeing_Study_300.csv` | 300 | 7-10 | NMF or LDA |
| `cricket_responses.csv` | 200 | 8-12 | TF-IDF + K-Means |
| `fashion_responses.csv` | 200 | 8-10 | TF-IDF + K-Means |
| `Remote_Work_Experiences_200.csv` | 200 | 6-8 | TF-IDF + K-Means |

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

## 6. Performance Expectations

| Dataset Size | Expected Processing Time |
|--------------|--------------------------|
| < 300 rows | 10-30 seconds |
| < 1,000 rows | < 30 seconds |
| Per-response granularity | < 50 ms per 1,000 responses |

---

## 7. Bringing Your Own Data

When preparing a custom dataset for this framework:

1. **Format:** CSV or Excel with a single text column containing open-ended responses.
2. **Minimum 50 responses** for meaningful results (20 absolute minimum).
3. **Average 10+ characters per response** to provide enough signal.
4. **80%+ analytic ratio** — most responses should contain substantive content.
5. **UTF-8 encoding** — ensure special characters are properly encoded.
6. **English language** — the current pipeline supports English only.
7. **Start with fewer codes** (5-7) and increase if the silhouette score is low or themes appear merged.

---

*See also: [Input Data Specification](03_input_data_specification.md) | [Methods](METHODS.md) | [Validation and Demonstration](06_validation_and_demonstration.md)*
