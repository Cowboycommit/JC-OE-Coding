"""
Methods Documentation Generator for ML Open-Ended Coding.

Auto-generates academic-style methods documentation with transparency about
assumptions, limitations, and ethical considerations. Designed to avoid
objectivity claims and ensure human judgment remains central.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter


class MethodsDocGenerator:
    """
    Generates comprehensive methods documentation for ML-assisted open coding.

    Produces:
    - Academic-style methods sections
    - Transparent assumption logs
    - Honest limitations documentation
    - Ethical considerations
    - Parameter logs for reproducibility

    Design Principles:
    - Transparency over marketing
    - Uncertainty surfaced, not suppressed
    - Human judgment acknowledged as central
    - No claims of objectivity or definitive accuracy
    """

    # Prohibited objectivity claims (for audit)
    PROHIBITED_PHRASES = [
        'objectively identifies',
        'accurately classifies',
        'ground truth',
        '100% accurate',
        'replaces human coding',
        'eliminates bias',
        'perfectly categorizes',
        'definitive themes',
        'true categories',
        'objective analysis'
    ]

    # Citation database for ML methods
    METHOD_CITATIONS = {
        'tfidf_kmeans': {
            'method': 'TF-IDF with K-Means Clustering',
            'citations': [
                'Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management.',
                'MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Proceedings of the Fifth Berkeley Symposium.'
            ]
        },
        'lda': {
            'method': 'Latent Dirichlet Allocation',
            'citations': [
                'Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. Journal of Machine Learning Research, 3, 993-1022.'
            ]
        },
        'nmf': {
            'method': 'Non-negative Matrix Factorization',
            'citations': [
                'Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. Nature, 401(6755), 788-791.'
            ]
        }
    }

    def __init__(self, project_name: str = "Open-Ended Coding Analysis"):
        """
        Initialize the methods documentation generator.

        Args:
            project_name: Name of the project for documentation
        """
        self.project_name = project_name
        self.timestamp = datetime.now()

    def generate_methods_section(
        self,
        coder: Any,
        results_df: pd.DataFrame,
        metrics: Dict[str, Any],
        preprocessing_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate comprehensive academic-style methods section.

        Args:
            coder: Fitted MLOpenCoder instance
            results_df: Results DataFrame with assignments
            metrics: Quality metrics dictionary
            preprocessing_params: Preprocessing parameters used

        Returns:
            Formatted methods section as markdown
        """
        methods = f"""# Methods Documentation

**Project:** {self.project_name}
**Generated:** {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Version:** 1.0

---

## 1. Data Preparation

### 1.1 Dataset Characteristics
- **Total responses collected:** {metrics.get('total_responses', len(results_df)):,}
- **Analytic sample size:** {len(results_df):,} responses
- **Data collection period:** [User to specify]
- **Response format:** Open-ended text responses

### 1.2 Preprocessing Steps

The following preprocessing steps were applied:

"""
        # Add preprocessing details
        if preprocessing_params:
            methods += self._format_preprocessing(preprocessing_params)
        else:
            methods += """- Null response removal: Applied
- Minimum response length: 5 characters
- Duplicate removal: Optional (user-configured)
- Text normalization: Lowercasing, punctuation removal, whitespace normalization

"""

        methods += f"""**Important Note:** All preprocessing decisions are logged and auditable. No responses
were silently excluded. {metrics.get('uncoded_count', 0)} responses remained uncoded after
analysis and are flagged for human review.

---

## 2. Coding Approach

### 2.1 Method Selection

**Primary method:** {self._format_method_name(metrics.get('method', 'tfidf_kmeans'))}

This method was selected for the following characteristics:
"""

        # Add method rationale
        methods += self._get_method_rationale(metrics.get('method', 'tfidf_kmeans'))

        methods += f"""
### 2.2 Representation Method

**Text representation:** {self._get_representation_description(metrics.get('method', 'tfidf_kmeans'))}

### 2.3 Code Discovery

- **Number of codes:** {metrics.get('n_codes', coder.n_codes)} codes
- **Code selection method:** {self._get_code_selection_method(metrics)}
- **Confidence threshold:** {coder.min_confidence:.2f} (codes assigned only when probability ≥ {coder.min_confidence:.0%})

**Code label generation:** Code labels were auto-generated from the top 3 most
characteristic keywords per code. **These labels are suggestions only and require
human validation and refinement.**

### 2.4 Multi-Label Support

This analysis supports multi-label coding, where:
- Responses can receive 0, 1, or multiple codes
- Average codes per response: {metrics.get('avg_codes_per_response', 0):.2f}
- No response is forced to receive a code
- {metrics.get('uncoded_count', 0):,} responses received no codes (below confidence threshold)

---

## 3. Quality Assurance

### 3.1 Statistical Quality Metrics

"""

        # Add quality metrics
        if 'silhouette_score' in metrics:
            methods += f"- **Silhouette score:** {metrics['silhouette_score']:.3f} (range: -1 to 1, higher is better)\n"
        if 'avg_confidence' in metrics:
            methods += f"- **Average confidence:** {metrics.get('avg_confidence', 0):.3f}\n"
            methods += f"- **Confidence range:** {metrics.get('min_confidence', 0):.3f} - {metrics.get('max_confidence', 0):.3f}\n"

        methods += f"""
### 3.2 Coverage Assessment

- **Coded responses:** {len(results_df) - metrics.get('uncoded_count', 0):,} ({metrics.get('coverage_pct', 0):.1f}%)
- **Uncoded responses:** {metrics.get('uncoded_count', 0):,} ({100 - metrics.get('coverage_pct', 100):.1f}%)
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

{self.document_assumptions(coder, results_df, metrics)}

---

## 5. Limitations

{self.generate_limitations(coder, results_df, metrics)}

---

## 6. Ethical Considerations

{self.generate_ethical_notes(metrics)}

---

## 7. Reproducibility Information

### 7.1 Software Environment

- Python version: 3.8+
- Key libraries: scikit-learn, pandas, numpy
- Random seed: 42 (for deterministic results)

### 7.2 Hyperparameters

"""

        # Log all hyperparameters
        methods += self._format_hyperparameters(coder, metrics)

        methods += f"""
### 7.3 Data Availability

- Original responses: [User to specify data sharing policy]
- Code assignments: Exportable with confidence scores
- Codebook: Includes keywords, frequencies, and example quotes
- Audit trail: All human review decisions logged

---

## 8. References

### Methodological Citations

{self._generate_citations(metrics.get('method', 'tfidf_kmeans'))}

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
- **Does NOT:** Claim objectivity or definitive accuracy
- **Does NOT:** Guarantee accuracy of auto-generated labels
- **Does NOT:** Eliminate researcher interpretation requirements

**Human researchers retain full responsibility for all interpretations and conclusions.**

---

*End of Methods Documentation*
"""

        return methods

    def document_assumptions(
        self,
        coder: Any,
        results_df: pd.DataFrame,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Document all assumptions made by the system.

        Returns:
            Formatted assumptions section
        """
        assumptions = """### 4.1 Core Assumptions

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
"""

        return assumptions

    def generate_limitations(
        self,
        coder: Any,
        results_df: pd.DataFrame,
        metrics: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive, honest limitations section.

        Returns:
            Formatted limitations section
        """
        limitations = """### 5.1 What This System Does

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

"""

        # Add specific technical limitations based on method
        method = metrics.get('method', 'tfidf_kmeans')

        limitations += f"""
**Method-Specific Constraints ({method}):**
"""

        if method == 'tfidf_kmeans':
            limitations += """- Assumes themes are separable in TF-IDF space
- Sensitive to outliers and extreme values
- May create unbalanced clusters with very different sizes
- Number of codes (k) must be specified in advance
"""
        elif method == 'lda':
            limitations += """- Assumes each document is a mixture of topics
- Sensitive to hyperparameter choices (alpha, beta)
- May produce non-interpretable topics with small datasets
- Computational complexity increases with dataset size
"""
        elif method == 'nmf':
            limitations += """- Assumes non-negative, additive representation
- Sensitive to initialization and local optima
- May produce sparse, hard-to-interpret factors
- Requires careful selection of number of components
"""

        limitations += f"""
**Dataset Size Constraints:**
- Minimum recommended: 50+ responses for {coder.n_codes} codes
- Current dataset: {len(results_df)} responses
"""

        if len(results_df) < 100:
            limitations += "- ⚠️ Dataset is relatively small; results may not be stable\n"

        limitations += f"""
**Coverage Limitations:**
- {metrics.get('uncoded_count', 0)} responses ({100 - metrics.get('coverage_pct', 100):.1f}%) received no codes
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
2. **Compare multiple methods** (TF-IDF, LDA, NMF) to check robustness
3. **Test sensitivity** to number of codes and confidence threshold
4. **Manually review uncoded and low-confidence responses**
5. **Consider qualitative coding of a subsample** for validation
6. **Document all human decisions** in code refinement process
7. **Report both ML and final human-validated results**
"""

        return limitations

    def generate_ethical_notes(self, metrics: Dict[str, Any]) -> str:
        """
        Generate ethical considerations section.

        Returns:
            Formatted ethical notes
        """
        ethical = """### 6.1 Ethical Use of Automated Coding

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
- Avoid language suggesting "objectivity" or "definitive accuracy"
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
6. **Avoid claiming objectivity** or definitive accuracy over human coding
7. **Consider sharing** de-identified data and code for reproducibility

---

**Ethical Bottom Line:** Automated coding is a tool to assist human judgment,
not replace it. Use responsibly, transparently, and with ongoing critical reflection.
"""

        return ethical

    def generate_bibtex_citations(self, method: str = 'tfidf_kmeans') -> str:
        """
        Generate BibTeX citations for ML methods used.

        Args:
            method: ML method name

        Returns:
            BibTeX formatted citations
        """
        bibtex = """# BibTeX Citations

## ML Methods

"""

        if method == 'tfidf_kmeans':
            bibtex += """@article{salton1988term,
  title={Term-weighting approaches in automatic text retrieval},
  author={Salton, Gerard and Buckley, Christopher},
  journal={Information Processing \\& Management},
  volume={24},
  number={5},
  pages={513--523},
  year={1988},
  publisher={Elsevier}
}

@inproceedings{macqueen1967methods,
  title={Some methods for classification and analysis of multivariate observations},
  author={MacQueen, James},
  booktitle={Proceedings of the fifth Berkeley symposium on mathematical statistics and probability},
  volume={1},
  number={14},
  pages={281--297},
  year={1967},
  organization={Oakland, CA, USA}
}
"""
        elif method == 'lda':
            bibtex += """@article{blei2003latent,
  title={Latent dirichlet allocation},
  author={Blei, David M and Ng, Andrew Y and Jordan, Michael I},
  journal={Journal of Machine Learning Research},
  volume={3},
  pages={993--1022},
  year={2003}
}
"""
        elif method == 'nmf':
            bibtex += """@article{lee1999learning,
  title={Learning the parts of objects by non-negative matrix factorization},
  author={Lee, Daniel D and Seung, H Sebastian},
  journal={Nature},
  volume={401},
  number={6755},
  pages={788--791},
  year={1999},
  publisher={Nature Publishing Group}
}
"""

        bibtex += """
## Qualitative Methods

@book{strauss1998basics,
  title={Basics of qualitative research: Techniques and procedures for developing grounded theory},
  author={Strauss, Anselm and Corbin, Juliet},
  year={1998},
  edition={2nd},
  publisher={Sage Publications}
}

@article{braun2006using,
  title={Using thematic analysis in psychology},
  author={Braun, Virginia and Clarke, Victoria},
  journal={Qualitative Research in Psychology},
  volume={3},
  number={2},
  pages={77--101},
  year={2006},
  publisher={Taylor \\& Francis}
}
"""

        return bibtex

    def audit_objectivity_claims(self, documentation: str) -> Tuple[bool, List[str]]:
        """
        Audit documentation for prohibited objectivity claims.

        Args:
            documentation: Documentation text to audit

        Returns:
            Tuple of (passed_audit, list_of_violations)
        """
        violations = []
        doc_lower = documentation.lower()

        for phrase in self.PROHIBITED_PHRASES:
            if phrase.lower() in doc_lower:
                # Find context
                idx = doc_lower.find(phrase.lower())
                context_start = max(0, idx - 50)
                context_end = min(len(documentation), idx + len(phrase) + 50)
                context = documentation[context_start:context_end].replace('\n', ' ')

                violations.append({
                    'phrase': phrase,
                    'context': context
                })

        passed = len(violations) == 0
        return passed, violations

    def generate_parameter_log(
        self,
        coder: Any,
        metrics: Dict[str, Any],
        preprocessing_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete parameter log for reproducibility.

        Args:
            coder: Fitted MLOpenCoder instance
            metrics: Quality metrics dictionary
            preprocessing_params: Preprocessing parameters

        Returns:
            Dictionary of all parameters
        """
        param_log = {
            'timestamp': self.timestamp.isoformat(),
            'project_name': self.project_name,
            'ml_method': {
                'name': metrics.get('method', 'tfidf_kmeans'),
                'n_codes': coder.n_codes,
                'min_confidence': coder.min_confidence,
                'random_seed': 42
            },
            'preprocessing': preprocessing_params or {
                'remove_nulls': True,
                'min_length': 5,
                'remove_duplicates': False,
                'stop_words': 'english'
            },
            'vectorization': {
                'max_features': 1000,
                'ngram_range': '(1, 2)' if metrics.get('method') == 'tfidf_kmeans' else '(1, 1)',
                'min_df': 2,
                'max_df': 0.8
            },
            'model_specific': self._get_model_specific_params(metrics.get('method', 'tfidf_kmeans')),
            'quality_metrics': {
                'total_responses': metrics.get('total_responses', 0),
                'coverage_pct': metrics.get('coverage_pct', 0),
                'avg_confidence': metrics.get('avg_confidence', 0),
                'silhouette_score': metrics.get('silhouette_score', None)
            }
        }

        return param_log

    # Helper methods

    def _format_method_name(self, method: str) -> str:
        """Format method name for display."""
        return self.METHOD_CITATIONS.get(method, {}).get('method', method.upper())

    def _get_method_rationale(self, method: str) -> str:
        """Get rationale for method selection."""
        rationales = {
            'tfidf_kmeans': """
- **Interpretability:** TF-IDF weights show which words matter most
- **Speed:** Fast computation even for large datasets
- **Transparency:** Clear connection between keywords and codes
- **Default choice:** Recommended for initial exploratory analysis
""",
            'lda': """
- **Probabilistic:** Soft assignments allow uncertainty quantification
- **Topic modeling:** Designed specifically for discovering latent themes
- **Mixed membership:** Responses can belong to multiple topics
- **Trade-off:** Less interpretable than TF-IDF, more complex
""",
            'nmf': """
- **Parts-based:** Discovers additive components of themes
- **Non-negativity:** Weights are always positive, easier to interpret
- **Sparsity:** Often produces clearer separation than LDA
- **Trade-off:** Sensitive to initialization, may need multiple runs
"""
        }
        return rationales.get(method, "- User-selected method\n")

    def _get_representation_description(self, method: str) -> str:
        """Get description of text representation."""
        if method == 'lda':
            return "Count Vectorization (bag-of-words)"
        else:
            return "TF-IDF (Term Frequency-Inverse Document Frequency) with bigrams"

    def _get_code_selection_method(self, metrics: Dict[str, Any]) -> str:
        """Determine how codes were selected."""
        if 'silhouette_score' in metrics:
            return "Silhouette score optimization (automated)"
        else:
            return "User-specified number of codes"

    def _format_preprocessing(self, params: Dict[str, Any]) -> str:
        """Format preprocessing parameters."""
        formatted = ""
        if params.get('remove_nulls'):
            formatted += "- Null response removal: Applied\n"
        if params.get('min_length'):
            formatted += f"- Minimum response length: {params['min_length']} characters\n"
        if params.get('remove_duplicates'):
            formatted += "- Duplicate removal: Applied\n"
        formatted += "- Text normalization: Lowercasing, punctuation removal, whitespace normalization\n\n"
        return formatted

    def _format_hyperparameters(self, coder: Any, metrics: Dict[str, Any]) -> str:
        """Format all hyperparameters."""
        params = f"""
**Core Parameters:**
- ML method: {metrics.get('method', 'tfidf_kmeans')}
- Number of codes: {coder.n_codes}
- Confidence threshold: {coder.min_confidence}
- Random seed: 42

**Vectorization:**
- Max features: 1000
- N-gram range: (1, 2) for TF-IDF, (1, 1) for LDA
- Min document frequency: 2
- Max document frequency: 0.8
- Stop words: English

"""

        # Add method-specific parameters
        method = metrics.get('method', 'tfidf_kmeans')
        params += f"**{self._format_method_name(method)} Parameters:**\n"
        params += self._get_model_specific_params(method)

        return params

    def _get_model_specific_params(self, method: str) -> str:
        """Get model-specific parameters."""
        if method == 'tfidf_kmeans':
            return "- K-Means n_init: 10\n- K-Means algorithm: lloyd\n"
        elif method == 'lda':
            return "- LDA max_iter: 20\n- LDA learning_method: batch\n"
        elif method == 'nmf':
            return "- NMF max_iter: 200\n- NMF init: random\n"
        return ""

    def _generate_citations(self, method: str) -> str:
        """Generate formatted citations."""
        if method in self.METHOD_CITATIONS:
            citations = self.METHOD_CITATIONS[method]['citations']
            return '\n'.join([f'- {cite}' for cite in citations])
        return f"- {method} (citations to be added)"


def export_methods_to_file(
    methods_doc: str,
    output_path: str = "METHODS.md"
) -> str:
    """
    Export methods documentation to markdown file.

    Args:
        methods_doc: Methods documentation string
        output_path: Output file path

    Returns:
        Path to exported file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(methods_doc)

    return output_path
