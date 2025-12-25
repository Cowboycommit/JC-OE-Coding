"""
Tests for Methods Documentation Generator.

Validates that methods documentation:
- Is complete and comprehensive
- Contains no objectivity claims
- Documents all assumptions
- Provides honest limitations
- Includes ethical considerations
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.methods_documentation import MethodsDocGenerator, export_methods_to_file


# Mock coder class for testing
class MockMLOpenCoder:
    """Mock coder for testing."""
    def __init__(self, n_codes=10, method='tfidf_kmeans', min_confidence=0.3):
        self.n_codes = n_codes
        self.method = method
        self.min_confidence = min_confidence
        self.codebook = {
            f'CODE_{i+1:02d}': {
                'label': f'Theme {i+1}',
                'keywords': ['keyword1', 'keyword2', 'keyword3'],
                'count': 10 + i,
                'examples': [],
                'avg_confidence': 0.5 + (i * 0.02)
            }
            for i in range(n_codes)
        }


@pytest.fixture
def sample_coder():
    """Create sample coder for testing."""
    return MockMLOpenCoder(n_codes=10, method='tfidf_kmeans', min_confidence=0.3)


@pytest.fixture
def sample_results_df():
    """Create sample results DataFrame."""
    return pd.DataFrame({
        'response_id': range(100),
        'response_text': [f'Sample response {i}' for i in range(100)],
        'assigned_codes': [['CODE_01', 'CODE_02'] if i % 2 == 0 else ['CODE_03'] for i in range(100)],
        'confidence_scores': [[0.8, 0.6] if i % 2 == 0 else [0.7] for i in range(100)],
        'num_codes': [2 if i % 2 == 0 else 1 for i in range(100)]
    })


@pytest.fixture
def sample_metrics():
    """Create sample metrics dictionary."""
    return {
        'total_responses': 100,
        'method': 'tfidf_kmeans',
        'n_codes': 10,
        'avg_codes_per_response': 1.5,
        'coverage_pct': 95.0,
        'uncoded_count': 5,
        'avg_confidence': 0.72,
        'min_confidence': 0.3,
        'max_confidence': 0.95,
        'silhouette_score': 0.42
    }


class TestMethodsDocGenerator:
    """Test suite for MethodsDocGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = MethodsDocGenerator(project_name="Test Project")
        assert generator.project_name == "Test Project"
        assert generator.timestamp is not None

    def test_prohibited_phrases_defined(self):
        """Test that prohibited phrases are defined."""
        generator = MethodsDocGenerator()
        assert len(generator.PROHIBITED_PHRASES) > 0
        assert 'objectively identifies' in generator.PROHIBITED_PHRASES
        assert 'ground truth' in generator.PROHIBITED_PHRASES
        assert 'replaces human coding' in generator.PROHIBITED_PHRASES

    def test_method_citations_exist(self):
        """Test that method citations are available."""
        generator = MethodsDocGenerator()
        assert 'tfidf_kmeans' in generator.METHOD_CITATIONS
        assert 'lda' in generator.METHOD_CITATIONS
        assert 'nmf' in generator.METHOD_CITATIONS

    def test_generate_methods_section(self, sample_coder, sample_results_df, sample_metrics):
        """Test full methods section generation."""
        generator = MethodsDocGenerator()
        methods = generator.generate_methods_section(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        assert isinstance(methods, str)
        assert len(methods) > 1000  # Should be comprehensive

        # Check for required sections
        assert '# Methods Documentation' in methods
        assert '## 1. Data Preparation' in methods
        assert '## 2. Coding Approach' in methods
        assert '## 3. Quality Assurance' in methods
        assert '## 4. Methodological Assumptions' in methods
        assert '## 5. Limitations' in methods
        assert '## 6. Ethical Considerations' in methods
        assert '## 7. Reproducibility Information' in methods
        assert '## 8. References' in methods
        assert '## 9. Transparency Statement' in methods

    def test_document_assumptions(self, sample_coder, sample_results_df, sample_metrics):
        """Test assumptions documentation."""
        generator = MethodsDocGenerator()
        assumptions = generator.document_assumptions(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        # Should contain at least 5 key assumptions (as per requirements)
        assumption_keywords = [
            'Response Independence',
            'Language Assumption',
            'Bag-of-Words',
            'Linear Separability',
            'Uniform Response Importance'
        ]

        for keyword in assumption_keywords:
            assert keyword in assumptions, f"Missing assumption: {keyword}"

        # Each assumption should have implications and mitigations
        assert 'Implication:' in assumptions
        assert 'Mitigation:' in assumptions

    def test_generate_limitations(self, sample_coder, sample_results_df, sample_metrics):
        """Test limitations generation."""
        generator = MethodsDocGenerator()
        limitations = generator.generate_limitations(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        # Check for required limitation categories
        assert 'What This System Does' in limitations
        assert 'What This System Cannot Do' in limitations
        assert 'Technical Limitations' in limitations
        assert 'Known Biases and Constraints' in limitations
        assert 'Recommendations for Mitigation' in limitations

        # Check for specific "cannot do" statements
        cannot_statements = [
            'cannot understand',
            'cannot replace',
            'cannot determine',
            'cannot generalize'
        ]
        limitations_lower = limitations.lower()
        found_cannot = sum(1 for stmt in cannot_statements if stmt in limitations_lower)
        assert found_cannot >= 3, "Should have multiple 'cannot' statements"

    def test_generate_ethical_notes(self, sample_metrics):
        """Test ethical considerations generation."""
        generator = MethodsDocGenerator()
        ethical = generator.generate_ethical_notes(sample_metrics)

        # Check for required ethical sections
        assert 'Ethical Use of Automated Coding' in ethical
        assert 'Data Privacy and Consent' in ethical
        assert 'Bias Monitoring' in ethical
        assert 'Researcher Positionality' in ethical
        assert 'Publication Ethics' in ethical

        # Check for specific ethical considerations
        assert 'Responsibility' in ethical
        assert 'Fairness and Representation' in ethical
        assert 'Transparency Requirements' in ethical
        assert 'Appropriate Use Cases' in ethical
        assert 'Inappropriate Use Cases' in ethical

    def test_audit_objectivity_claims_pass(self):
        """Test objectivity audit with clean documentation."""
        generator = MethodsDocGenerator()
        clean_doc = """
        This system assists qualitative analysis by suggesting potential themes.
        Results should be validated through human review. Confidence scores are
        estimates, not guarantees. The system surfaces patterns but cannot replace
        human qualitative judgment.
        """

        passed, violations = generator.audit_objectivity_claims(clean_doc)
        assert passed is True
        assert len(violations) == 0

    def test_audit_objectivity_claims_fail(self):
        """Test objectivity audit with problematic documentation."""
        generator = MethodsDocGenerator()
        problematic_doc = """
        This system objectively identifies themes in your data with 100% accuracy.
        It replaces human coding and eliminates bias, providing ground truth labels.
        """

        passed, violations = generator.audit_objectivity_claims(problematic_doc)
        assert passed is False
        assert len(violations) > 0

        # Check that violations are properly documented
        violation_phrases = [v['phrase'] for v in violations]
        assert 'objectively identifies' in violation_phrases
        assert '100% accurate' in violation_phrases

    def test_generate_bibtex_citations_tfidf(self):
        """Test BibTeX citation generation for TF-IDF."""
        generator = MethodsDocGenerator()
        bibtex = generator.generate_bibtex_citations('tfidf_kmeans')

        assert '@article{salton1988term' in bibtex
        assert '@inproceedings{macqueen1967' in bibtex
        assert 'Salton, Gerard' in bibtex
        assert 'MacQueen, James' in bibtex

    def test_generate_bibtex_citations_lda(self):
        """Test BibTeX citation generation for LDA."""
        generator = MethodsDocGenerator()
        bibtex = generator.generate_bibtex_citations('lda')

        assert '@article{blei2003latent' in bibtex
        assert 'Blei, David' in bibtex

    def test_generate_bibtex_citations_nmf(self):
        """Test BibTeX citation generation for NMF."""
        generator = MethodsDocGenerator()
        bibtex = generator.generate_bibtex_citations('nmf')

        assert '@article{lee1999learning' in bibtex
        assert 'Lee, Daniel' in bibtex

    def test_generate_parameter_log(self, sample_coder, sample_metrics):
        """Test parameter log generation."""
        generator = MethodsDocGenerator()
        param_log = generator.generate_parameter_log(
            sample_coder,
            sample_metrics,
            preprocessing_params={'remove_nulls': True, 'min_length': 5}
        )

        # Check structure
        assert 'timestamp' in param_log
        assert 'project_name' in param_log
        assert 'ml_method' in param_log
        assert 'preprocessing' in param_log
        assert 'vectorization' in param_log
        assert 'quality_metrics' in param_log

        # Check ML method details
        assert param_log['ml_method']['name'] == 'tfidf_kmeans'
        assert param_log['ml_method']['n_codes'] == 10
        assert param_log['ml_method']['min_confidence'] == 0.3
        assert param_log['ml_method']['random_seed'] == 42

    def test_export_methods_to_file(self, sample_coder, sample_results_df, sample_metrics):
        """Test exporting methods to file."""
        generator = MethodsDocGenerator()
        methods = generator.generate_methods_section(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_path = f.name

        try:
            export_path = export_methods_to_file(methods, temp_path)
            assert Path(export_path).exists()

            # Read back and verify
            with open(export_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert content == methods
        finally:
            # Cleanup
            Path(temp_path).unlink(missing_ok=True)

    def test_methods_includes_all_hyperparameters(self, sample_coder, sample_results_df, sample_metrics):
        """Test that all hyperparameters are documented."""
        generator = MethodsDocGenerator()
        methods = generator.generate_methods_section(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        # Check for hyperparameter documentation
        hyperparameter_keywords = [
            'n_codes',
            'confidence threshold',
            'random seed',
            'max features',
            'n-gram',
            'min document frequency',
            'max document frequency',
            'stop words'
        ]

        methods_lower = methods.lower()
        for keyword in hyperparameter_keywords:
            assert keyword in methods_lower, f"Missing hyperparameter: {keyword}"

    def test_no_objectivity_claims_in_generated_methods(self, sample_coder, sample_results_df, sample_metrics):
        """Test that generated methods contain no objectivity claims."""
        generator = MethodsDocGenerator()
        methods = generator.generate_methods_section(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        # Audit the generated methods
        passed, violations = generator.audit_objectivity_claims(methods)

        assert passed is True, f"Methods contain objectivity claims: {violations}"

    def test_assumptions_count(self, sample_coder, sample_results_df, sample_metrics):
        """Test that at least 5 key assumptions are documented."""
        generator = MethodsDocGenerator()
        assumptions = generator.document_assumptions(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        # Count numbered assumptions
        import re
        assumption_numbers = re.findall(r'^\d+\.\s+\*\*', assumptions, re.MULTILINE)
        assert len(assumption_numbers) >= 5, "Should document at least 5 key assumptions"

    def test_transparency_statement_present(self, sample_coder, sample_results_df, sample_metrics):
        """Test that transparency statement is included."""
        generator = MethodsDocGenerator()
        methods = generator.generate_methods_section(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        assert 'Transparency Statement' in methods
        assert 'Does:' in methods
        assert 'Does NOT:' in methods
        assert 'Human researchers retain full responsibility' in methods

    def test_human_review_prominence(self, sample_coder, sample_results_df, sample_metrics):
        """Test that human review is emphasized."""
        generator = MethodsDocGenerator()
        methods = generator.generate_methods_section(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        # Should mention human review multiple times
        human_review_count = methods.lower().count('human review')
        assert human_review_count >= 5, "Human review should be mentioned prominently"

        # Should explicitly list what requires human review
        assert 'Human review is required for:' in methods

    def test_different_methods_generate_different_content(self, sample_results_df):
        """Test that different ML methods generate appropriate method-specific content."""
        generator = MethodsDocGenerator()

        # Test TF-IDF
        coder_tfidf = MockMLOpenCoder(method='tfidf_kmeans')
        metrics_tfidf = {'method': 'tfidf_kmeans', 'n_codes': 10, 'total_responses': 100, 'coverage_pct': 95, 'uncoded_count': 5}
        methods_tfidf = generator.generate_methods_section(coder_tfidf, sample_results_df, metrics_tfidf)

        # Test LDA
        coder_lda = MockMLOpenCoder(method='lda')
        metrics_lda = {'method': 'lda', 'n_codes': 10, 'total_responses': 100, 'coverage_pct': 95, 'uncoded_count': 5}
        methods_lda = generator.generate_methods_section(coder_lda, sample_results_df, metrics_lda)

        # Should have different method-specific content
        assert 'K-Means' in methods_tfidf or 'TF-IDF' in methods_tfidf
        assert 'Latent Dirichlet' in methods_lda or 'LDA' in methods_lda
        assert methods_tfidf != methods_lda

    def test_coverage_reporting(self, sample_coder, sample_results_df, sample_metrics):
        """Test that coverage is properly reported."""
        generator = MethodsDocGenerator()
        methods = generator.generate_methods_section(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        # Should report both coded and uncoded
        assert 'Coded responses:' in methods
        assert 'Uncoded responses:' in methods
        assert '95.0%' in methods or '95%' in methods  # Coverage percentage
        assert '5' in methods  # Uncoded count

    def test_limitations_are_specific(self, sample_coder, sample_results_df, sample_metrics):
        """Test that limitations are specific, not generic."""
        generator = MethodsDocGenerator()
        limitations = generator.generate_limitations(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        # Should contain specific limitations, not just generic ones
        specific_keywords = [
            'sarcasm',
            'irony',
            'multilingual',
            'context',
            'generalize',
            'causal',
            'bias'
        ]

        limitations_lower = limitations.lower()
        found_specific = sum(1 for keyword in specific_keywords if keyword in limitations_lower)
        assert found_specific >= 5, "Should have specific, concrete limitations"

    def test_ethical_section_includes_action_items(self, sample_metrics):
        """Test that ethical section includes actionable recommendations."""
        generator = MethodsDocGenerator()
        ethical = generator.generate_ethical_notes(sample_metrics)

        # Should have action items
        assert 'Action Required:' in ethical or 'Recommended Checks:' in ethical
        assert 'Best Practice:' in ethical

        # Should distinguish appropriate vs inappropriate use
        assert 'Appropriate Use Cases' in ethical
        assert 'Inappropriate Use Cases' in ethical


class TestCompleteness:
    """Test suite for documentation completeness."""

    def test_all_required_sections_present(self, sample_coder, sample_results_df, sample_metrics):
        """Test that all 9 required sections are present."""
        generator = MethodsDocGenerator()
        methods = generator.generate_methods_section(
            sample_coder,
            sample_results_df,
            sample_metrics
        )

        required_sections = [
            '## 1. Data Preparation',
            '## 2. Coding Approach',
            '## 3. Quality Assurance',
            '## 4. Methodological Assumptions',
            '## 5. Limitations',
            '## 6. Ethical Considerations',
            '## 7. Reproducibility Information',
            '## 8. References',
            '## 9. Transparency Statement'
        ]

        for section in required_sections:
            assert section in methods, f"Missing required section: {section}"

    def test_minimum_length_requirements(self, sample_coder, sample_results_df, sample_metrics):
        """Test that documentation meets minimum length requirements."""
        generator = MethodsDocGenerator()

        # Test methods section
        methods = generator.generate_methods_section(sample_coder, sample_results_df, sample_metrics)
        assert len(methods) >= 5000, "Methods section should be comprehensive (≥5000 chars)"

        # Test assumptions
        assumptions = generator.document_assumptions(sample_coder, sample_results_df, sample_metrics)
        assert len(assumptions) >= 1000, "Assumptions should be detailed (≥1000 chars)"

        # Test limitations
        limitations = generator.generate_limitations(sample_coder, sample_results_df, sample_metrics)
        assert len(limitations) >= 1500, "Limitations should be comprehensive (≥1500 chars)"

        # Test ethical notes
        ethical = generator.generate_ethical_notes(sample_metrics)
        assert len(ethical) >= 2000, "Ethical considerations should be thorough (≥2000 chars)"


class TestObjectivityAudit:
    """Test suite for objectivity claims auditing."""

    def test_audit_catches_all_prohibited_phrases(self):
        """Test that audit catches all types of prohibited phrases."""
        generator = MethodsDocGenerator()

        for phrase in generator.PROHIBITED_PHRASES:
            test_doc = f"This system {phrase} the themes in your data."
            passed, violations = generator.audit_objectivity_claims(test_doc)

            assert passed is False, f"Failed to catch prohibited phrase: {phrase}"
            assert len(violations) > 0
            assert violations[0]['phrase'] == phrase

    def test_audit_provides_context(self):
        """Test that audit provides context for violations."""
        generator = MethodsDocGenerator()
        test_doc = "Our innovative system objectively identifies all themes with perfect accuracy."

        passed, violations = generator.audit_objectivity_claims(test_doc)
        assert passed is False
        assert len(violations) > 0
        assert 'context' in violations[0]
        assert len(violations[0]['context']) > 0

    def test_generated_docs_pass_audit(self, sample_coder, sample_results_df, sample_metrics):
        """Test that all generated documentation passes objectivity audit."""
        generator = MethodsDocGenerator()

        # Test methods section
        methods = generator.generate_methods_section(sample_coder, sample_results_df, sample_metrics)
        passed, violations = generator.audit_objectivity_claims(methods)
        assert passed is True, f"Methods section failed audit: {violations}"

        # Test assumptions
        assumptions = generator.document_assumptions(sample_coder, sample_results_df, sample_metrics)
        passed, violations = generator.audit_objectivity_claims(assumptions)
        assert passed is True, f"Assumptions section failed audit: {violations}"

        # Test limitations
        limitations = generator.generate_limitations(sample_coder, sample_results_df, sample_metrics)
        passed, violations = generator.audit_objectivity_claims(limitations)
        assert passed is True, f"Limitations section failed audit: {violations}"

        # Test ethical notes
        ethical = generator.generate_ethical_notes(sample_metrics)
        passed, violations = generator.audit_objectivity_claims(ethical)
        assert passed is True, f"Ethical section failed audit: {violations}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
