# Rigor Diagnostics Interpretation Guide

**Version:** 1.0
**Date:** 2025-12-25
**Module:** `src/rigor_diagnostics.py`

---

## Table of Contents

1. [Overview](#overview)
2. [Validity Dimensions](#validity-dimensions)
3. [Bias Detection Metrics](#bias-detection-metrics)
4. [Sanity Checks](#sanity-checks)
5. [Interpretation Guidelines](#interpretation-guidelines)
6. [Actionable Recommendations](#actionable-recommendations)
7. [Limitations & Caveats](#limitations--caveats)

---

## Overview

The Rigor Diagnostics module provides comprehensive methodological validity assessment for ML-assisted qualitative coding. This guide explains how to interpret each diagnostic metric and take appropriate action based on the results.

**Key Principles:**
- **Quantify uncertainty** - Don't suppress it
- **Surface warnings** - Make methodological concerns visible
- **Provide actionable recommendations** - Not just numbers
- **Maintain interpretability** - All metrics are transparent (not black-box)

---

## Validity Dimensions

The `assess_validity()` method evaluates 10 dimensions of methodological quality:

### 1. Coverage Ratio

**What it measures:** Percentage of responses that received at least one code assignment.

**Calculation:** `(coded_responses / total_responses) × 100`

**Interpretation:**
- **≥90%**: Excellent coverage - nearly all responses coded
- **80-89%**: Good coverage - acceptable for most analyses
- **70-79%**: Acceptable coverage - review uncoded responses
- **<70%**: Poor coverage - indicates model fit issues

**Example:**
```
Coverage Ratio: 85.3% - Good coverage
Coded: 853 responses
Uncoded: 147 responses
Total: 1000 responses
```

**What to do:**
- **High uncoded rate**: Consider lowering `min_confidence` threshold or increasing `n_codes`
- **Review uncoded responses** manually to identify if they represent:
  - True outliers/unique perspectives
  - Missing themes requiring new codes
  - Non-analytic content (e.g., "N/A", "No comment")

---

### 2. Code Utilization

**What it measures:** Percentage of codes that are actively used (assigned to at least one response).

**Calculation:** `(active_codes / total_codes) × 100`

**Interpretation:**
- **≥90%**: Excellent utilization - all codes meaningful
- **75-89%**: Good utilization - most codes used
- **50-74%**: Fair utilization - some codes underused
- **<50%**: Poor utilization - too many codes for dataset

**Example:**
```
Code Utilization: 80.0% - Good utilization
Active Codes: 8 out of 10
Underused: CODE_09, CODE_10
```

**What to do:**
- **Low utilization**: Reduce `n_codes` to eliminate unused clusters
- **Unused codes** suggest model created redundant or overly-specific categories
- Review underused codes to determine if they should be merged with similar codes

---

### 3. Theme Coherence

**What it measures:** Average semantic similarity within each code (how well responses within a code cluster together).

**Calculation:**
- With feature matrix: Average pairwise cosine similarity within each code
- Without feature matrix: Average confidence scores per code

**Interpretation:**
- **≥0.70**: High coherence - themes are well-defined
- **0.50-0.69**: Moderate coherence - themes have some consistency
- **<0.50**: Low coherence - themes may be poorly defined or overlapping

**Example:**
```
Theme Coherence: 0.68 - Moderate coherence
CODE_01: 0.72 (high)
CODE_02: 0.65 (moderate)
CODE_03: 0.42 (low - review this code)
```

**What to do:**
- **Low coherence**: Review code definitions for overlap or ambiguity
- **Specific codes with low coherence**: May need to be split or redefined
- Consider hierarchical coding if many codes show moderate coherence

---

### 4. Code Stability

**What it measures:** Consistency of code assignments when small perturbations are made to the data.

**Calculation:** `1 - (std(codes_per_response) / mean(codes_per_response))`

**Interpretation:**
- **≥0.80**: High stability - consistent assignments
- **0.60-0.79**: Moderate stability - some variation acceptable
- **<0.60**: Low stability - unreliable clustering

**Example:**
```
Code Stability: 0.74 - Moderate stability
Interpretation: Some variation in assignments
```

**What to do:**
- **Low stability**: May indicate:
  - Dataset too small for current `n_codes`
  - High noise in data
  - Need for different clustering method
- Run analysis with different `random_state` values to verify consistency

---

### 5. Thematic Saturation

**What it measures:** Whether you've captured all major themes or if important themes are missing.

**Assessment based on:**
- Uncoded percentage (>15% suggests missing themes)
- Unused codes (>20% suggests too many themes)

**Statuses:**
- **Under-saturated**: High uncoded rate suggests missing themes
- **Over-saturated**: Many unused codes suggest too many themes
- **Adequate**: Theme coverage appears appropriate

**Example:**
```
Thematic Saturation: Adequate
Confidence: Medium
Message: Theme coverage appears adequate
Uncoded: 12.3%
Unused codes: 1
```

**What to do:**
- **Under-saturated**: Increase `n_codes` to capture missing themes
- **Over-saturated**: Reduce `n_codes` to eliminate redundant categories
- **Adequate**: Proceed with validation of existing codes

---

### 6. Confidence Distribution

**What it measures:** Distribution of confidence scores across all code assignments.

**Key statistics:**
- Mean, median, standard deviation
- Percentiles (25th, 75th, 90th)
- Histogram shape

**Interpretation:**
- **Mean ≥0.70**: High confidence - strong assignments
- **Mean 0.50-0.69**: Moderate confidence - acceptable
- **Mean <0.50**: Low confidence - weak assignments, review model

**Example:**
```
Confidence Distribution:
- Mean: 0.68 - Moderate confidence
- Median: 0.72
- 25th percentile: 0.51
- 75th percentile: 0.83
- 90th percentile: 0.91
```

**What confidence means:**
- **≥0.90**: Very high certainty - strong thematic fit
- **0.70-0.89**: High certainty - clear code assignment
- **0.50-0.69**: Moderate certainty - acceptable but review edge cases
- **0.30-0.49**: Low certainty - near threshold, review recommended
- **<0.30**: Very low certainty - below default threshold

**What to do:**
- **75th percentile <0.50**: Indicates majority of assignments are uncertain
  - Check if `n_codes` is too high for dataset
  - Consider different ML method (e.g., LDA instead of K-means)
- **High standard deviation**: Suggests mixed quality - some codes well-defined, others not

---

### 7. Ambiguity Rate

**What it measures:** Percentage of responses that received multiple codes.

**Calculation:**
- Multi-coded: Responses with 2+ codes
- High ambiguity: Responses with 3+ codes

**Interpretation:**
- **Multi-code <10%**: Very low ambiguity - highly distinct themes
- **Multi-code 10-30%**: Low-moderate ambiguity - common for complex data
- **Multi-code 30-50%**: Moderate-high ambiguity - themes may overlap
- **Multi-code >50%**: High ambiguity - most responses span multiple themes

**Example:**
```
Ambiguity Rate:
- Multi-coded: 34.2% (342 responses)
- High ambiguity (3+ codes): 8.1% (81 responses)
Interpretation: Moderate ambiguity - common for complex data
```

**What to do:**
- **High ambiguity is not necessarily bad** - it may reflect genuine complexity
- **If problematic**: Review code definitions for overlap
- Consider hierarchical coding where some codes are sub-themes of others
- Multi-coding is expected when responses discuss multiple topics

---

### 8. Boundary Cases

**What it measures:** Number of responses near decision boundaries (confidence 0.3-0.5).

**Purpose:** Identify responses that barely met threshold - prime candidates for review.

**Interpretation:**
- **<10%**: Few boundary cases - most assignments clear
- **10-20%**: Moderate - acceptable level of uncertainty
- **>20%**: Many boundary cases - consider raising threshold or reviewing model

**Example:**
```
Boundary Cases: 124 responses (12.4%)
Note: Responses with max confidence between 0.3-0.5
```

**What to do:**
- Export boundary cases for manual review
- These responses are most likely to benefit from human validation
- May indicate need for additional codes or refinement of existing ones

---

### 9. Inter-Code Reliability

**What it measures:** Agreement between ML codes and human codes (if available).

**Metrics:**
- Cohen's Kappa (κ): Agreement adjusted for chance
- Percent agreement: Raw agreement rate

**Interpretation (Cohen's Kappa):**
- **κ >0.80**: Excellent agreement
- **κ 0.60-0.80**: Substantial agreement
- **κ 0.40-0.60**: Moderate agreement
- **κ <0.40**: Poor agreement - review ML assignments

**Example:**
```
Inter-Code Reliability:
Cohen's Kappa: 0.68 - Substantial agreement
Percent Agreement: 74.2%
```

**Note:** Requires human-coded sample for comparison. If not available, shows "Not assessed".

---

### 10. Random Seed Stability

**What it measures:** Sensitivity of results to `random_state` parameter.

**Purpose:** Ensure findings are reproducible, not artifacts of random initialization.

**How to assess:**
1. Run analysis with `random_state=42`
2. Re-run with `random_state=123, 456, 789`
3. Compare code labels and assignments

**What to do:**
- If results vary substantially: Dataset may be too small or noisy
- If results are stable: Good sign of robust clustering
- Always report `random_state` used in final analysis

---

## Bias Detection Metrics

The `detect_bias()` method identifies potential systematic biases:

### 1. Code Imbalance

**What it measures:** Distribution inequality across codes.

**Metrics:**
- **Imbalance Ratio**: Max count / Min count
- **Gini Coefficient**: 0 (perfect equality) to 1 (perfect inequality)

**Interpretation (Imbalance Ratio):**
- **<5:1**: Well-balanced distribution
- **5:1-10:1**: Moderate imbalance - acceptable variation
- **10:1-20:1**: High imbalance - some codes dominate
- **>20:1**: Severe imbalance - review code structure

**Example:**
```
Code Imbalance:
- Ratio: 12.5:1 - High imbalance
- Max code: CODE_01 (500 responses)
- Min code: CODE_08 (40 responses)
- Gini Coefficient: 0.42
```

**What to do:**
- **High imbalance** may indicate:
  - Dominant code should be split into sub-codes
  - Rare codes should be merged with similar themes
  - Natural distribution (some topics genuinely more common)

---

### 2. Demographic Representation

**What it measures:** Whether coding patterns differ across demographic groups.

**Test:** Chi-square test for independence between demographic groups and coding rates.

**Interpretation:**
- **p <0.05**: Significant difference - potential bias detected
- **p ≥0.05**: No significant difference - equitable coding

**Example:**
```
Demographic Representation:
- Gender: p=0.032 - Significant difference across groups
- Age: p=0.184 - No significant difference
- Department: p=0.006 - Significant difference across groups
```

**What to do if significant:**
1. **Investigate**: Are certain groups genuinely different in their responses?
2. **Review**: Is the ML model systematically under/over-coding certain groups?
3. **Consider**: Stratified analysis by demographic group
4. **Document**: Note in limitations section of report

**Important:** Statistical significance ≠ bias. Differences may reflect genuine variation in perspectives.

---

### 3. Positional Bias

**What it measures:** Whether early vs. late responses are coded differently.

**Calculation:** Compare average codes per response for first half vs. second half of dataset.

**Example:**
```
Positional Bias:
- First half avg: 1.82 codes/response
- Second half avg: 1.76 codes/response
- Difference: 0.06 - Not significant
```

**What to do if significant:**
- May indicate:
  - Respondent fatigue (later responses shorter/less detailed)
  - Data collection issues (survey modified mid-collection)
  - Random variation (check with larger sample)

---

## Sanity Checks

The `sanity_check()` method flags common methodological issues:

### 1. Long Labels

**Issue:** Code labels with >5 words are difficult to interpret and remember.

**Example:** "Remote Work Flexibility Benefits Including Schedule Control" (8 words)

**Better:** "Remote Work Flexibility" (3 words)

**What to do:** Simplify labels while preserving meaning. Use full keywords list for detail.

---

### 2. Code Imbalance

**Issue:** Max code count / Min code count >10:1

**Impact:** Dominant codes may overshadow smaller but important themes.

**What to do:** Review whether dominant code should be split, or rare codes merged.

---

### 3. Low Coverage

**Issue:** >20% of responses uncoded

**Impact:** Missing data reduces validity and completeness of analysis.

**What to do:**
1. Lower `min_confidence` (e.g., from 0.3 to 0.2)
2. Increase `n_codes` to capture more themes
3. Review uncoded responses for new themes

---

### 4. Low Confidence

**Issue:** 75th percentile of confidence <0.5

**Impact:** Most assignments are uncertain, reducing trust in results.

**What to do:**
1. Check if dataset is too small for current `n_codes`
2. Consider different ML method (LDA, NMF)
3. Increase human review sample size

---

### 5. Insufficient Data

**Issue:** Dataset <20 responses (or custom minimum)

**Impact:** Statistical methods unreliable with very small samples.

**What to do:**
1. Collect more data before running ML analysis
2. Use traditional qualitative methods instead
3. Treat results as exploratory only, not confirmatory

---

### 6. Unused Codes

**Issue:** Codes with zero assignments

**Impact:** Wasted computational resources, suggests over-clustering.

**What to do:** Reduce `n_codes` to eliminate empty clusters.

---

### 7. High Ambiguity

**Issue:** >30% of responses have 3+ codes

**Impact:** May indicate overlapping themes or poor code separation.

**What to do:**
1. Review code definitions for semantic overlap
2. Consider hierarchical coding structure
3. Verify multi-coding is appropriate for research question

---

## Interpretation Guidelines

### Overall Health Status

The system assigns an overall health status based on total issues detected:

- **Excellent** (0 issues): ✅ No issues detected. Analysis appears methodologically sound.
- **Good** (1-2 issues): 🟡 Minor issues detected. Review recommendations.
- **Fair** (3-4 issues): 🟠 Several issues detected. Address before finalizing.
- **Poor** (5+ issues): 🔴 Multiple issues detected. Major review recommended.

### Confidence in Results

Use this decision tree to assess confidence in your results:

```
                    START
                      ├─ Health Status: Excellent/Good?
                      │   ├─ YES → Coverage >80%?
                      │   │   ├─ YES → Avg Confidence >0.6?
                      │   │   │   ├─ YES → HIGH CONFIDENCE ✅
                      │   │   │   └─ NO → MODERATE CONFIDENCE 🟡
                      │   │   └─ NO → REVIEW NEEDED ⚠️
                      │   └─ NO → MAJOR REVIEW NEEDED 🔴
```

### When to Iterate

Re-run analysis with different parameters if:
1. **Health status**: Fair or Poor
2. **Coverage**: <80%
3. **Utilization**: <75%
4. **Avg Confidence**: <0.5
5. **>3 sanity check warnings**

### When Results Are Ready

Proceed to human validation when:
1. **Health status**: Excellent or Good
2. **Coverage**: ≥80%
3. **Coherence**: ≥0.5
4. **Utilization**: ≥75%
5. **≤2 sanity check warnings**

---

## Actionable Recommendations

The system generates specific, actionable recommendations. Here's what common recommendations mean:

### "Reduce min_confidence threshold"

**Context:** High uncoded rate (>20%)

**Action:** Change `min_confidence` from 0.3 to 0.2 or 0.25

**Trade-off:** More coverage but lower average confidence

**When appropriate:** When responses are genuinely ambiguous or cover multiple topics

---

### "Increase n_codes"

**Context:** High uncoded rate + under-saturation

**Action:** Increase `n_codes` from current value by 2-5

**Trade-off:** More granular themes but risk of over-clustering

**When appropriate:** When uncoded responses represent distinct themes not captured

---

### "Reduce n_codes"

**Context:** Low utilization (>20% unused codes)

**Action:** Decrease `n_codes` by 2-5

**Trade-off:** Fewer but more populated codes

**When appropriate:** When dataset is too small for current granularity

---

### "Review code definitions for overlap"

**Context:** Low coherence or high ambiguity

**Action:** Examine keywords and examples for similar codes

**Resolution:** Merge overlapping codes or refine definitions to be more distinct

---

### "Consider hierarchical coding"

**Context:** High multi-coding rate (>40%)

**Action:** Structure codes as parent-child relationships

**Example:**
```
Parent: Work-Life Balance
  ├─ Child: Remote Work
  ├─ Child: Flexible Hours
  └─ Child: Workload Management
```

---

### "Increase human review sample"

**Context:** Low confidence or many boundary cases

**Action:** Manually validate more responses, especially:
- Boundary cases (confidence 0.3-0.5)
- Multi-coded responses (3+ codes)
- Uncoded responses

**Target:** Review at least 10-20% of dataset

---

## Limitations & Caveats

### What These Metrics Cannot Do

1. **Replace human judgment** - Metrics guide, not dictate, decisions
2. **Detect all biases** - Only checks for specific patterns
3. **Ensure interpretive validity** - Cannot assess if codes match research questions
4. **Account for context** - Metrics are decontextualized, you know your data best

### Known Limitations

1. **Small datasets** (<50 responses):
   - Statistical metrics unreliable
   - Bootstrap stability not calculable
   - Use qualitative methods instead

2. **Feature matrix not available**:
   - Coherence based on confidence scores (proxy)
   - Cannot calculate true semantic similarity
   - Provide `feature_matrix` parameter for better metrics

3. **No human codes available**:
   - Inter-coder reliability not calculable
   - Consider coding a sample manually for validation

4. **Demographic data not provided**:
   - Bias detection limited to code distribution
   - Provide demographics for representation analysis

### Interpreting in Context

Always consider:
- **Research domain**: Some domains naturally have imbalanced themes
- **Data characteristics**: Short responses → lower coherence expected
- **Sample size**: Smaller samples → higher metric variance
- **Research goals**: Exploratory vs. confirmatory analysis

---

## Appendix: Metric Quick Reference

| Metric | Good | Acceptable | Concerning |
|--------|------|------------|------------|
| Coverage | ≥90% | 80-89% | <80% |
| Utilization | ≥90% | 75-89% | <75% |
| Coherence | ≥0.70 | 0.50-0.69 | <0.50 |
| Stability | ≥0.80 | 0.60-0.79 | <0.60 |
| Avg Confidence | ≥0.70 | 0.50-0.69 | <0.50 |
| Imbalance Ratio | <5:1 | 5:1-10:1 | >10:1 |
| Uncoded Rate | <10% | 10-20% | >20% |
| Multi-coding | 10-30% | 30-50% | >50% |
| Boundary Cases | <10% | 10-20% | >20% |

---

## Support & Further Reading

**For questions or issues with rigor diagnostics:**
- Review test cases: `tests/test_rigor_diagnostics.py`
- Consult orchestration plan: `ORCHESTRATION_PLAN.md` (Section 2.7)
- Check code documentation: `src/rigor_diagnostics.py`

**Methodological references:**
- Qualitative rigor: Lincoln & Guba (1985) - Credibility, Dependability, Confirmability
- ML clustering validation: Rousseeuw (1987) - Silhouette analysis
- Bias detection: Mehrabi et al. (2021) - A Survey on Bias and Fairness in ML

---

**Document Version:** 1.0
**Last Updated:** 2025-12-25
**Maintained by:** Evaluation & Validation Specialist (Agent-7)
