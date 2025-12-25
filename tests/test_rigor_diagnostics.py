"""
Unit tests for RigorDiagnostics class.
"""

import pytest
import pandas as pd
import numpy as np
from src.rigor_diagnostics import RigorDiagnostics


class TestRigorDiagnostics:
    """Test cases for RigorDiagnostics."""

    @pytest.fixture
    def diagnostics(self):
        """Create RigorDiagnostics instance."""
        return RigorDiagnostics()

    @pytest.fixture
    def mock_coder(self):
        """Create mock MLOpenCoder instance."""
        class MockCoder:
            def __init__(self):
                self.n_codes = 5
                self.codebook = {
                    'CODE_01': {
                        'label': 'Remote Work',
                        'keywords': ['remote', 'work', 'home'],
                        'count': 50,
                        'avg_confidence': 0.75,
                        'examples': []
                    },
                    'CODE_02': {
                        'label': 'Flexibility Benefits',
                        'keywords': ['flexible', 'schedule', 'balance'],
                        'count': 30,
                        'avg_confidence': 0.65,
                        'examples': []
                    },
                    'CODE_03': {
                        'label': 'Team Collaboration',
                        'keywords': ['team', 'collaborate', 'together'],
                        'count': 20,
                        'avg_confidence': 0.55,
                        'examples': []
                    },
                    'CODE_04': {
                        'label': 'Professional Development',
                        'keywords': ['learning', 'growth', 'development'],
                        'count': 10,
                        'avg_confidence': 0.45,
                        'examples': []
                    },
                    'CODE_05': {
                        'label': 'Company Culture',
                        'keywords': ['culture', 'values', 'environment'],
                        'count': 0,
                        'avg_confidence': 0.0,
                        'examples': []
                    }
                }
        return MockCoder()

    @pytest.fixture
    def sample_results(self):
        """Create sample results DataFrame."""
        return pd.DataFrame({
            'response_id': range(1, 101),
            'response_text': [f'Response {i}' for i in range(1, 101)],
            'assigned_codes': [
                ['CODE_01'] if i % 2 == 0 else ['CODE_01', 'CODE_02']
                if i % 3 == 0 else ['CODE_03']
                if i % 5 == 0 else []
                for i in range(1, 101)
            ],
            'confidence_scores': [
                [0.75] if i % 2 == 0 else [0.65, 0.55]
                if i % 3 == 0 else [0.45]
                if i % 5 == 0 else []
                for i in range(1, 101)
            ],
            'num_codes': [
                1 if i % 2 == 0 else 2
                if i % 3 == 0 else 1
                if i % 5 == 0 else 0
                for i in range(1, 101)
            ]
        })

    @pytest.fixture
    def demographics_df(self):
        """Create sample demographics DataFrame."""
        np.random.seed(42)
        return pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female', 'Other'], 100),
            'age_group': np.random.choice(['18-25', '26-35', '36-45', '46+'], 100),
            'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], 100)
        })

    # ============ Test Initialization ============

    def test_initialization(self, diagnostics):
        """Test RigorDiagnostics initialization."""
        assert diagnostics is not None
        assert hasattr(diagnostics, 'assess_validity')
        assert hasattr(diagnostics, 'detect_bias')
        assert hasattr(diagnostics, 'sanity_check')

    # ============ Test assess_validity() ============

    def test_assess_validity_basic(self, diagnostics, mock_coder, sample_results):
        """Test basic validity assessment."""
        validity = diagnostics.assess_validity(mock_coder, sample_results)

        assert isinstance(validity, dict)
        # Check all required dimensions present
        assert 'code_stability' in validity
        assert 'theme_coherence' in validity
        assert 'thematic_saturation' in validity
        assert 'coverage_ratio' in validity
        assert 'code_utilization' in validity
        assert 'confidence_distribution' in validity
        assert 'ambiguity_rate' in validity
        assert 'boundary_cases' in validity

    def test_assess_validity_coverage_ratio(self, diagnostics, mock_coder, sample_results):
        """Test coverage ratio calculation."""
        validity = diagnostics.assess_validity(mock_coder, sample_results)

        coverage = validity['coverage_ratio']
        assert 'coverage_percentage' in coverage
        assert 'coded_responses' in coverage
        assert 'uncoded_responses' in coverage
        assert 0 <= coverage['coverage_percentage'] <= 100

    def test_assess_validity_code_utilization(self, diagnostics, mock_coder, sample_results):
        """Test code utilization analysis."""
        validity = diagnostics.assess_validity(mock_coder, sample_results)

        utilization = validity['code_utilization']
        assert 'utilization_rate' in utilization
        assert 'active_codes' in utilization
        assert 'total_codes' in utilization
        assert 'underused_codes' in utilization
        assert 'overused_codes' in utilization

        # CODE_05 should be in underused (count=0)
        assert 'CODE_05' in utilization['underused_codes']

    def test_assess_validity_confidence_distribution(self, diagnostics, mock_coder, sample_results):
        """Test confidence distribution analysis."""
        validity = diagnostics.assess_validity(mock_coder, sample_results)

        conf_dist = validity['confidence_distribution']
        assert 'mean' in conf_dist
        assert 'median' in conf_dist
        assert 'std' in conf_dist
        assert 'percentiles' in conf_dist
        assert '25th' in conf_dist['percentiles']
        assert '75th' in conf_dist['percentiles']

    def test_assess_validity_ambiguity_rate(self, diagnostics, mock_coder, sample_results):
        """Test ambiguity rate calculation."""
        validity = diagnostics.assess_validity(mock_coder, sample_results)

        ambiguity = validity['ambiguity_rate']
        assert 'multi_code_percentage' in ambiguity
        assert 'high_ambiguity_percentage' in ambiguity
        assert 'multi_code_count' in ambiguity
        assert 0 <= ambiguity['multi_code_percentage'] <= 100

    def test_assess_validity_thematic_saturation(self, diagnostics, mock_coder, sample_results):
        """Test thematic saturation assessment."""
        validity = diagnostics.assess_validity(mock_coder, sample_results)

        saturation = validity['thematic_saturation']
        assert 'saturation_status' in saturation
        assert saturation['saturation_status'] in ['under_saturated', 'over_saturated', 'adequate']
        assert 'confidence' in saturation
        assert 'message' in saturation

    def test_assess_validity_boundary_cases(self, diagnostics, mock_coder, sample_results):
        """Test boundary case identification."""
        validity = diagnostics.assess_validity(mock_coder, sample_results)

        boundary = validity['boundary_cases']
        assert 'count' in boundary
        assert 'percentage' in boundary
        assert isinstance(boundary['count'], int)
        assert 0 <= boundary['percentage'] <= 100

    def test_assess_validity_with_human_codes(self, diagnostics, mock_coder, sample_results):
        """Test validity assessment with human codes."""
        human_codes = pd.DataFrame({
            'response_id': range(1, 101),
            'human_codes': [['CODE_01'] for _ in range(100)]
        })

        validity = diagnostics.assess_validity(
            mock_coder, sample_results, human_codes=human_codes
        )

        assert 'inter_code_reliability' in validity

    def test_assess_validity_with_feature_matrix(self, diagnostics, mock_coder, sample_results):
        """Test validity assessment with feature matrix."""
        # Create mock feature matrix
        from scipy.sparse import csr_matrix
        feature_matrix = csr_matrix(np.random.rand(100, 50))

        validity = diagnostics.assess_validity(
            mock_coder, sample_results, feature_matrix=feature_matrix
        )

        assert 'theme_coherence' in validity
        coherence = validity['theme_coherence']
        # Should have per-code coherence when feature matrix provided
        assert 'per_code_coherence' in coherence

    # ============ Test detect_bias() ============

    def test_detect_bias_basic(self, diagnostics, sample_results):
        """Test basic bias detection."""
        bias = diagnostics.detect_bias(sample_results)

        assert isinstance(bias, dict)
        assert 'demographic_representation' in bias
        assert 'code_imbalance' in bias
        assert 'warnings' in bias

    def test_detect_bias_code_imbalance(self, diagnostics, sample_results):
        """Test code imbalance detection."""
        bias = diagnostics.detect_bias(sample_results)

        imbalance = bias['code_imbalance']
        assert 'imbalance_ratio' in imbalance
        assert 'max_code_count' in imbalance
        assert 'min_code_count' in imbalance
        assert 'gini_coefficient' in imbalance
        assert 'interpretation' in imbalance

    def test_detect_bias_with_demographics(
        self, diagnostics, sample_results, demographics_df
    ):
        """Test bias detection with demographic data."""
        bias = diagnostics.detect_bias(
            sample_results,
            demographics=demographics_df,
            demographic_columns=['gender', 'age_group']
        )

        demo_rep = bias['demographic_representation']
        assert 'gender' in demo_rep
        assert 'age_group' in demo_rep
        assert demo_rep['gender']['chi_square_test'] is not None

    def test_detect_bias_systematic_patterns(self, diagnostics, sample_results):
        """Test systematic pattern detection."""
        bias = diagnostics.detect_bias(sample_results)

        patterns = bias['systematic_patterns']
        assert 'positional_bias' in patterns
        assert 'first_half_avg' in patterns['positional_bias']
        assert 'second_half_avg' in patterns['positional_bias']

    def test_detect_bias_empty_results(self, diagnostics):
        """Test bias detection with empty results."""
        empty_df = pd.DataFrame({
            'assigned_codes': [],
            'num_codes': []
        })

        bias = diagnostics.detect_bias(empty_df)
        assert isinstance(bias, dict)

    # ============ Test sanity_check() ============

    def test_sanity_check_basic(self, diagnostics, mock_coder, sample_results):
        """Test basic sanity checks."""
        sanity = diagnostics.sanity_check(mock_coder, sample_results)

        assert isinstance(sanity, dict)
        assert 'warnings' in sanity
        assert 'recommendations' in sanity
        assert 'issues' in sanity
        assert 'health_status' in sanity
        assert 'total_issues' in sanity

    def test_sanity_check_long_labels(self, diagnostics, sample_results):
        """Test detection of long labels."""
        class CoderWithLongLabels:
            codebook = {
                'CODE_01': {
                    'label': 'This Is A Very Long Label That Should Be Flagged',
                    'count': 10,
                    'avg_confidence': 0.7
                }
            }

        sanity = diagnostics.sanity_check(CoderWithLongLabels(), sample_results)

        assert 'long_labels' in sanity['issues']

    def test_sanity_check_code_imbalance(self, diagnostics, sample_results):
        """Test detection of code imbalance."""
        class ImbalancedCoder:
            codebook = {
                'CODE_01': {'label': 'Dominant', 'count': 100, 'avg_confidence': 0.7},
                'CODE_02': {'label': 'Rare', 'count': 5, 'avg_confidence': 0.6}
            }

        sanity = diagnostics.sanity_check(ImbalancedCoder(), sample_results)

        # Should detect 20:1 imbalance
        assert 'code_imbalance' in sanity['issues']

    def test_sanity_check_low_coverage(self, diagnostics, mock_coder):
        """Test detection of low coverage."""
        # Create results with 50% uncoded
        low_coverage_df = pd.DataFrame({
            'assigned_codes': [['CODE_01'] if i < 40 else [] for i in range(100)],
            'confidence_scores': [[0.7] if i < 40 else [] for i in range(100)],
            'num_codes': [1 if i < 40 else 0 for i in range(100)]
        })

        sanity = diagnostics.sanity_check(mock_coder, low_coverage_df)

        assert 'low_coverage' in sanity['issues']
        assert sanity['issues']['low_coverage'] > 20

    def test_sanity_check_low_confidence(self, diagnostics, mock_coder):
        """Test detection of low confidence."""
        # Create results with low confidence
        low_conf_df = pd.DataFrame({
            'assigned_codes': [['CODE_01'] for _ in range(100)],
            'confidence_scores': [[0.35] for _ in range(100)],
            'num_codes': [1 for _ in range(100)]
        })

        sanity = diagnostics.sanity_check(mock_coder, low_conf_df)

        assert 'low_confidence' in sanity['issues']

    def test_sanity_check_insufficient_data(self, diagnostics, mock_coder):
        """Test detection of insufficient data."""
        small_df = pd.DataFrame({
            'assigned_codes': [['CODE_01'] for _ in range(10)],
            'confidence_scores': [[0.7] for _ in range(10)],
            'num_codes': [1 for _ in range(10)]
        })

        sanity = diagnostics.sanity_check(mock_coder, small_df, min_responses=20)

        assert 'insufficient_data' in sanity['issues']

    def test_sanity_check_unused_codes(self, diagnostics, mock_coder, sample_results):
        """Test detection of unused codes."""
        sanity = diagnostics.sanity_check(mock_coder, sample_results)

        # CODE_05 has count=0
        assert 'unused_codes' in sanity['issues']
        assert 'CODE_05' in sanity['issues']['unused_codes']

    def test_sanity_check_high_ambiguity(self, diagnostics, mock_coder):
        """Test detection of high ambiguity."""
        # Create results with high ambiguity (3+ codes per response)
        high_ambig_df = pd.DataFrame({
            'assigned_codes': [['CODE_01', 'CODE_02', 'CODE_03'] for _ in range(100)],
            'confidence_scores': [[0.6, 0.5, 0.4] for _ in range(100)],
            'num_codes': [3 for _ in range(100)]
        })

        sanity = diagnostics.sanity_check(mock_coder, high_ambig_df)

        assert 'high_ambiguity' in sanity['issues']

    def test_sanity_check_health_status(self, diagnostics, mock_coder, sample_results):
        """Test health status calculation."""
        sanity = diagnostics.sanity_check(mock_coder, sample_results)

        health = sanity['health_status']
        assert 'status' in health
        assert health['status'] in ['excellent', 'good', 'fair', 'poor']
        assert 'message' in health
        assert 'issue_count' in health

    def test_sanity_check_excellent_health(self, diagnostics):
        """Test excellent health status."""
        # Create ideal coder and results
        class IdealCoder:
            codebook = {
                f'CODE_0{i}': {
                    'label': f'Code {i}',
                    'count': 20,
                    'avg_confidence': 0.75
                }
                for i in range(1, 6)
            }

        ideal_results = pd.DataFrame({
            'assigned_codes': [['CODE_01'] for _ in range(100)],
            'confidence_scores': [[0.75] for _ in range(100)],
            'num_codes': [1 for _ in range(100)]
        })

        sanity = diagnostics.sanity_check(IdealCoder(), ideal_results)

        # Should have 0 or very few issues
        assert sanity['total_issues'] <= 2

    # ============ Test generate_recommendations() ============

    def test_generate_recommendations(self, diagnostics, mock_coder, sample_results):
        """Test recommendation generation."""
        validity = diagnostics.assess_validity(mock_coder, sample_results)
        bias = diagnostics.detect_bias(sample_results)
        sanity = diagnostics.sanity_check(mock_coder, sample_results)

        recommendations = diagnostics.generate_recommendations(validity, bias, sanity)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Each recommendation should be a string
        assert all(isinstance(rec, str) for rec in recommendations)

    def test_generate_recommendations_low_coverage(self, diagnostics, mock_coder):
        """Test recommendations for low coverage."""
        low_cov_df = pd.DataFrame({
            'assigned_codes': [['CODE_01'] if i < 50 else [] for i in range(100)],
            'confidence_scores': [[0.7] if i < 50 else [] for i in range(100)],
            'num_codes': [1 if i < 50 else 0 for i in range(100)]
        })

        validity = diagnostics.assess_validity(mock_coder, low_cov_df)
        bias = diagnostics.detect_bias(low_cov_df)
        sanity = diagnostics.sanity_check(mock_coder, low_cov_df)

        recommendations = diagnostics.generate_recommendations(validity, bias, sanity)

        # Should mention coverage
        assert any('coverage' in rec.lower() for rec in recommendations)

    def test_generate_recommendations_excellent_results(self, diagnostics):
        """Test recommendations when results are excellent."""
        # Mock excellent validity/bias/sanity
        validity = {
            'coverage_ratio': {'coverage_percentage': 95},
            'code_utilization': {'utilization_rate': 90},
            'theme_coherence': {'average_coherence': 0.8}
        }
        bias = {
            'code_imbalance': {'imbalance_ratio': 3}
        }
        sanity = {
            'warnings': [],
            'recommendations': []
        }

        recommendations = diagnostics.generate_recommendations(validity, bias, sanity)

        # Should have positive message
        assert any('good' in rec.lower() or '✅' in rec for rec in recommendations)

    # ============ Test Helper Methods ============

    def test_calculate_gini_coefficient(self, diagnostics):
        """Test Gini coefficient calculation."""
        # Perfect equality
        equal_counts = [10, 10, 10, 10]
        gini_equal = diagnostics._calculate_gini(equal_counts)
        assert 0 <= gini_equal <= 0.1  # Should be near 0

        # Perfect inequality
        unequal_counts = [100, 1, 1, 1]
        gini_unequal = diagnostics._calculate_gini(unequal_counts)
        assert gini_unequal > 0.3  # Should be higher

    def test_interpret_methods(self, diagnostics):
        """Test interpretation helper methods."""
        # Test stability interpretation
        assert 'High' in diagnostics._interpret_stability(0.9)
        assert 'Low' in diagnostics._interpret_stability(0.3)

        # Test coherence interpretation
        assert 'High' in diagnostics._interpret_coherence(0.8)
        assert 'Low' in diagnostics._interpret_coherence(0.3)

        # Test coverage interpretation
        assert 'Excellent' in diagnostics._interpret_coverage(95)
        assert 'Poor' in diagnostics._interpret_coverage(50)

        # Test utilization interpretation
        assert 'Excellent' in diagnostics._interpret_utilization(95)
        assert 'Poor' in diagnostics._interpret_utilization(40)

    def test_edge_case_empty_results(self, diagnostics, mock_coder):
        """Test handling of completely empty results."""
        empty_df = pd.DataFrame({
            'assigned_codes': [],
            'confidence_scores': [],
            'num_codes': []
        })

        # Should not crash
        validity = diagnostics.assess_validity(mock_coder, empty_df)
        bias = diagnostics.detect_bias(empty_df)
        sanity = diagnostics.sanity_check(mock_coder, empty_df)

        assert isinstance(validity, dict)
        assert isinstance(bias, dict)
        assert isinstance(sanity, dict)

    def test_edge_case_all_uncoded(self, diagnostics, mock_coder):
        """Test handling of all uncoded responses."""
        uncoded_df = pd.DataFrame({
            'assigned_codes': [[] for _ in range(100)],
            'confidence_scores': [[] for _ in range(100)],
            'num_codes': [0 for _ in range(100)]
        })

        validity = diagnostics.assess_validity(mock_coder, uncoded_df)

        # Coverage should be 0%
        assert validity['coverage_ratio']['coverage_percentage'] == 0.0

    def test_edge_case_single_response(self, diagnostics, mock_coder):
        """Test handling of single response."""
        single_df = pd.DataFrame({
            'assigned_codes': [['CODE_01']],
            'confidence_scores': [[0.75]],
            'num_codes': [1]
        })

        # Should handle gracefully
        validity = diagnostics.assess_validity(mock_coder, single_df)
        assert isinstance(validity, dict)

    # ============ Integration Tests ============

    def test_full_diagnostic_workflow(self, diagnostics, mock_coder, sample_results):
        """Test complete diagnostic workflow."""
        # Run all diagnostics
        validity = diagnostics.assess_validity(mock_coder, sample_results)
        bias = diagnostics.detect_bias(sample_results)
        sanity = diagnostics.sanity_check(mock_coder, sample_results)
        recommendations = diagnostics.generate_recommendations(validity, bias, sanity)

        # Verify all outputs are valid
        assert isinstance(validity, dict)
        assert len(validity) >= 8  # At least 8 validity dimensions

        assert isinstance(bias, dict)
        assert 'code_imbalance' in bias

        assert isinstance(sanity, dict)
        assert 'health_status' in sanity

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_performance_requirement(self, diagnostics, mock_coder):
        """Test that diagnostics complete in <500ms."""
        import time

        # Create larger dataset
        large_df = pd.DataFrame({
            'assigned_codes': [['CODE_01', 'CODE_02'] for _ in range(1000)],
            'confidence_scores': [[0.7, 0.6] for _ in range(1000)],
            'num_codes': [2 for _ in range(1000)]
        })

        start = time.time()

        # Run all diagnostics
        validity = diagnostics.assess_validity(mock_coder, large_df)
        bias = diagnostics.detect_bias(large_df)
        sanity = diagnostics.sanity_check(mock_coder, large_df)
        recommendations = diagnostics.generate_recommendations(validity, bias, sanity)

        elapsed = (time.time() - start) * 1000  # Convert to ms

        # Should complete in under 500ms
        assert elapsed < 500, f"Diagnostics took {elapsed:.0f}ms, expected <500ms"
