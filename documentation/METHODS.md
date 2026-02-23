# Methods Documentation

**Project:** Employee Satisfaction Survey Analysis
**Generated:** 2026-02-23
**Version:** 1.4

---

## 1. Data Preparation

### 1.1 Dataset Characteristics
- **Total responses collected:** 1,000 (per dataset)
- **Analytic sample size:** 1,000 responses (or optimal subsample)
- **Data collection period:** [User to specify]
- **Response format:** Open-ended text responses

### 1.2 Preprocessing Steps

The following preprocessing steps were applied:

- Null response removal: Applied
- Minimum response length: 5 characters
- Duplicate removal: Optional (user-configured)
- Text normalization: Lowercasing, punctuation removal, whitespace normalization

**Important Note:** All preprocessing decisions are logged and auditable. No responses
were silently excluded. 15 responses remained uncoded after
analysis and are flagged for human review.

---

## 2. Coding Approach

### 2.1 Method Selection

**Primary method:** TF-IDF with K-Means Clustering

This method was selected for the following characteristics:

- **Interpretability:** TF-IDF weights show which words matter most
- **Speed:** Fast computation even for large datasets
- **Transparency:** Clear connection between keywords and codes
- **Default choice:** Recommended for initial exploratory analysis

### 2.2 Representation Method

**Text representation:** TF-IDF (Term Frequency-Inverse Document Frequency) with bigrams

### 2.3 Code Discovery

- **Number of codes:** 10 codes
- **Code selection method:** Silhouette score optimization (automated)
- **Confidence threshold:** 0.30 (codes assigned only when probability ≥ 30%)

**Code label generation:** Code labels were auto-generated from the top 3 most
characteristic keywords per code. **These labels are suggestions only and require
human validation and refinement.**

### 2.4 Multi-Label Support

This analysis supports multi-label coding, where:
- Responses can receive 0, 1, or multiple codes
- Average codes per response: 1.37
- No response is forced to receive a code
- 15 responses received no codes (below confidence threshold)

---

## 3. Quality Assurance

### 3.1 Statistical Quality Metrics

- **Silhouette score:** 0.420 (range: -1 to 1, higher is better)
- **Average confidence:** 0.686
- **Confidence range:** 0.320 - 0.920

### 3.2 Coverage Assessment

- **Coded responses:** 285 (95.0%)
- **Uncoded responses:** 15 (5.0%)
- **Responses requiring human review:** [To be determined based on confidence thresholds]

### 3.3 Human Review Process

**Critical:** This system assists qualitative analysis but does not replace it.
Human review is required for:

1. **Validation of auto-generated code labels** - Keywords may not capture thematic meaning
2. **Review of low-confidence assignments** - Assignments below 0.5 confidence
3. **Examination of uncoded responses** - Responses that didn't fit discovered codes
4. **Interpretation of co-occurrence patterns** - Understanding why codes appear together
5. **Final code structure decisions** - Merging, splitting, or refining codes
6. **Contextualizing findings** - Placing results in broader research framework

---

## 4. Methodological Assumptions

### 4.1 Core Assumptions

The following assumptions underlie this analysis:

1. **Response Independence Assumption**
   - Each response is treated as an independent unit of analysis
   - No consideration for conversational context or respondent history
   - **Implication:** May miss relational or contextual meaning
   - **Mitigation:** Review responses in original context when interpreting

2. **Language Assumption**
   - English language processing (stop word removal: 'english')
   - Single-language dataset assumed
   - **Implication:** May not work for multilingual data or code-switching
   - **Mitigation:** Flag non-English responses for separate analysis

3. **Bag-of-Words Assumption**
   - Word order is ignored in representation
   - Only word frequency and co-occurrence matter
   - **Implication:** May miss meaning from word order, syntax, or grammar
   - **Mitigation:** Human review considers full sentence context

4. **Linear Separability Assumption** (for K-Means)
   - Themes can be separated in vector space
   - Clusters are roughly spherical and equal-sized
   - **Implication:** May not capture hierarchical or overlapping themes
   - **Mitigation:** Multi-label support allows some overlap

5. **Uniform Response Importance**
   - All responses weighted equally
   - No demographic or contextual weighting applied
   - **Implication:** May suppress minority perspectives if they're numerically small
   - **Mitigation:** Examine code distributions across demographic groups

6. **Keyword Representativeness Assumption**
   - Top keywords adequately represent theme meaning
   - Auto-generated labels are interpretable
   - **Implication:** Labels may be misleading or oversimplified
   - **Mitigation:** Human validation and label refinement required

7. **Confidence Score Interpretation**
   - Confidence scores reflect statistical probability, not truth
   - High confidence does not guarantee correct assignment
   - **Implication:** Overconfidence in high-scoring assignments
   - **Mitigation:** Sample validation across confidence levels

8. **Thematic Saturation Assumption**
   - Sufficient responses to discover major themes
   - Dataset size adequate for chosen number of codes
   - **Implication:** Rare themes may be missed with small datasets
   - **Mitigation:** Monitor code utilization and uncoded responses

### 4.2 Assumption Monitoring

**Researchers should validate these assumptions for their specific context.**
If assumptions are violated, results may not be valid.

---

## 5. Limitations

### 5.1 What This System Does

This system assists qualitative open coding by:
- Clustering responses based on word usage patterns
- Suggesting potential thematic codes with confidence scores
- Identifying co-occurrence patterns among codes
- Flagging responses that need human review
- Providing diagnostic metrics for quality assessment

### 5.2 What This System Cannot Do

**Language and Context:**
- Cannot understand sarcasm, irony, or non-literal language
- Cannot detect cultural nuances or context-dependent meanings
- Cannot handle multilingual responses or code-switching
- Cannot interpret emojis, images, or non-textual content

**Analytical Capabilities:**
- Cannot replace human qualitative judgment
- Cannot determine causal relationships
- Cannot assess validity of respondent claims
- Cannot detect contradictions or logical inconsistencies
- Cannot understand conversational context or dialogue structure

**Generalization:**
- Cannot generalize beyond the specific dataset analyzed
- Cannot account for sampling bias or non-response bias
- Cannot validate external validity of findings

**Quality Detection:**
- Cannot reliably detect low-quality or dishonest responses
- Cannot identify response patterns from bots or spam
- May misclassify short or ambiguous responses

### 5.3 Technical Limitations

**Method-Specific Constraints (tfidf_kmeans):**
- Assumes themes are separable in TF-IDF space
- Sensitive to outliers and extreme values
- May create unbalanced clusters with very different sizes
- Number of codes (k) must be specified in advance

**Dataset Size Constraints:**
- Minimum recommended: 50+ responses for 10 codes
- Optimal range: 300-500 responses for all 6 ML methods
- Current demo datasets: 1,000 rows each (use optimal sampling for target code counts)
- See `documentation/OPTIMAL_DATASET_SIZE.md` for detailed guidance

**Coverage Limitations:**
- 15 responses (5.0%) received no codes
- Low coverage may indicate:
  - Confidence threshold is too high
  - Number of codes is insufficient
  - Responses are highly heterogeneous
  - Some responses don't fit discovered themes

### 5.4 Known Biases and Constraints

**Algorithmic Bias:**
- System may reflect biases in preprocessing choices (e.g., English stop words)
- Keyword-based labeling may favor literal over interpretive themes
- Frequency-based approaches may overshadow minority perspectives

**Interpretive Constraints:**
- Auto-generated labels are based on word frequency, not semantic meaning
- Labels may not capture the "why" or underlying intent
- Decontextualized keywords can be misleading

**Reproducibility Constraints:**
- Results depend on random seed (K-Means initialization)
- Different preprocessing choices yield different codes
- Confidence thresholds are somewhat arbitrary

### 5.5 Recommendations for Mitigation

To address these limitations:

1. **Always validate auto-generated codes with human review**
2. **Compare multiple methods** (TF-IDF, LDA, NMF, BERT, LSTM, SVM) to check robustness
3. **Test sensitivity** to number of codes and confidence threshold
4. **Manually review uncoded and low-confidence responses**
5. **Consider qualitative coding of a subsample** for validation
6. **Document all human decisions** in code refinement process
7. **Report both ML and final human-validated results**

---

## 6. Ethical Considerations

### 6.1 Ethical Use of Automated Coding

**Responsibility:**
- Researchers remain fully responsible for interpretations and conclusions
- Automated coding does not absolve researchers from ethical obligations
- System outputs must be critically evaluated, not blindly accepted

**Fairness and Representation:**
- Algorithmic approaches may systematically favor majority perspectives
- Rare or minority voices may be underrepresented in discovered codes
- **Action Required:** Examine code distributions across demographic groups
- **Action Required:** Manually review responses from underrepresented groups

**Transparency Requirements:**
- All automated decisions must be documentable and explainable
- Confidence scores and rationales must accompany all outputs
- Human review process must be documented in audit trail
- Final publications should clearly distinguish ML-assisted from human-validated codes

**Limitations Disclosure:**
- Do not oversell capabilities of automated coding
- Acknowledge uncertainty and limitations prominently
- Avoid language suggesting "objectivity" or "ground truth"
- Be transparent about what the system cannot do

### 6.2 Data Privacy and Consent

**Participant Protection:**
- Original responses may contain sensitive or identifiable information
- Example quotes must be reviewed for anonymization before publication
- Data sharing must comply with consent and IRB requirements
- Consider privacy implications of exporting coded data

**Appropriate Use Cases:**
- ✅ Exploratory analysis to identify potential themes
- ✅ Initial coding to prioritize manual review
- ✅ Validation of human-generated codes
- ✅ Large-scale pattern detection with human oversight

**Inappropriate Use Cases:**
- ❌ High-stakes decisions without human validation
- ❌ Replacing required qualitative analysis in funded research
- ❌ Claiming "objective" or "unbiased" results
- ❌ Using without adequate methodological expertise

### 6.3 Bias Monitoring

**Recommended Checks:**

1. **Demographic Representation**
   - Are all demographic groups coded at similar rates?
   - Do any groups have disproportionately low confidence scores?
   - Are certain groups overrepresented in uncoded responses?

2. **Code Balance**
   - Are some codes vastly overused (>30% of responses)?
   - Are some codes underused (<1% of responses)?
   - Does code distribution reflect expected thematic diversity?

3. **Confidence Distribution**
   - Is confidence similar across demographic groups?
   - Are there systematic differences in confidence by response length?
   - Do certain themes get systematically higher/lower confidence?

**If bias is detected:**
- Document findings transparently
- Consider manual coding for affected groups
- Adjust confidence thresholds or number of codes
- Consult methodological literature on bias mitigation

### 6.4 Researcher Positionality

**Interpretive Authority:**
- Human researchers, not algorithms, have interpretive authority
- Researcher background, training, and perspective shape interpretation
- **Best Practice:** Include researcher positionality statement in publications
- **Best Practice:** Consider inter-coder reliability with diverse coders

### 6.5 Publication Ethics

When publishing results from ML-assisted coding:

1. **Clearly label** which codes are ML-generated vs. human-validated
2. **Report confidence scores** and coverage metrics
3. **Describe human review process** in detail
4. **Acknowledge limitations** of automated approach
5. **Share methodological details** sufficient for replication
6. **Avoid claiming objectivity** or superiority over human coding
7. **Consider sharing** de-identified data and code for reproducibility

---

**Ethical Bottom Line:** Automated coding is a tool to assist human judgment,
not replace it. Use responsibly, transparently, and with ongoing critical reflection.

---

## 7. Reproducibility Information

### 7.1 Software Environment

- Python version: 3.8+
- Key libraries: scikit-learn, pandas, numpy
- Random seed: 42 (for deterministic results)

### 7.2 Hyperparameters

**Core Parameters:**
- ML method: tfidf_kmeans
- Number of codes: 10
- Confidence threshold: 0.3
- Random seed: 42

**Vectorization:**
- Max features: 1000
- N-gram range: (1, 3) for TF-IDF, (1, 1) for LDA
- Min document frequency: 2
- Max document frequency: 0.8
- Stop words: English

**TF-IDF with K-Means Clustering Parameters:**
- K-Means n_init: 10
- K-Means algorithm: lloyd

### 7.3 Data Availability

- Original responses: [User to specify data sharing policy]
- Code assignments: Exportable with confidence scores
- Codebook: Includes keywords, frequencies, and example quotes
- Audit trail: All human review decisions logged

---

## 8. References

### Methodological Citations

- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium.

### Qualitative Research Framework

- Strauss, A., & Corbin, J. (1998). *Basics of qualitative research: Techniques and procedures for developing grounded theory* (2nd ed.). Sage.
- Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. *Qualitative Research in Psychology, 3*(2), 77-101.

### Computer-Assisted Qualitative Analysis

- Friese, S. (2019). *Qualitative data analysis with ATLAS.ti* (3rd ed.). Sage.

---

## 9. Transparency Statement

This methods section was auto-generated to ensure complete documentation of all
analytical decisions. The system:

- **Does:** Assist theme discovery through pattern recognition
- **Does:** Provide confidence scores for all assignments
- **Does:** Flag uncertain cases for human review
- **Does:** Log all parameters and decisions

- **Does NOT:** Replace human qualitative judgment
- **Does NOT:** Claim objectivity or "ground truth"
- **Does NOT:** Guarantee accuracy of auto-generated labels
- **Does NOT:** Eliminate researcher interpretation requirements

**Human researchers retain full responsibility for all interpretations and conclusions.**

---

*End of Methods Documentation*
