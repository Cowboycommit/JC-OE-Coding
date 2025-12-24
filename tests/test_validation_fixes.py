"""
Test validation fixes for empty/under-sized datasets.

Tests the fixes for:
1. Empty dataset validation
2. Under-sized dataset validation
3. Division by zero guards
4. Invalid cluster count handling
"""

import pytest
import pandas as pd
import sys
sys.path.insert(0, '.')

from helpers.analysis import (
    run_ml_analysis,
    find_optimal_codes,
    calculate_metrics_summary,
)


class TestEmptyDatasetValidation:
    """Test that empty datasets are properly rejected."""

    def test_run_ml_analysis_empty_dataframe(self):
        """Should raise ValueError for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame is empty"):
            run_ml_analysis(df, text_column='text', n_codes=5)

    def test_run_ml_analysis_none_dataframe(self):
        """Should raise ValueError for None DataFrame."""
        with pytest.raises(ValueError, match="DataFrame is empty"):
            run_ml_analysis(None, text_column='text', n_codes=5)

    def test_find_optimal_codes_empty_dataframe(self):
        """Should raise ValueError for empty DataFrame."""
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame is empty"):
            find_optimal_codes(df, text_column='text')

    def test_find_optimal_codes_none_dataframe(self):
        """Should raise ValueError for None DataFrame."""
        with pytest.raises(ValueError, match="DataFrame is empty"):
            find_optimal_codes(None, text_column='text')


class TestUndersizedDatasetValidation:
    """Test that under-sized datasets are properly rejected."""

    def test_run_ml_analysis_too_few_rows(self):
        """Should raise ValueError when n_codes > number of rows."""
        df = pd.DataFrame({
            'text': ['response 1', 'response 2', 'response 3']
        })

        with pytest.raises(ValueError, match="Dataset too small for 10 codes"):
            run_ml_analysis(df, text_column='text', n_codes=10)

    def test_find_optimal_codes_too_few_samples(self):
        """Should raise ValueError for dataset with only 1 sample."""
        df = pd.DataFrame({
            'text': ['single response']
        })

        with pytest.raises(ValueError, match="Dataset too small for clustering"):
            find_optimal_codes(df, text_column='text', min_codes=3)

    def test_find_optimal_codes_min_exceeds_max_valid(self):
        """Should raise ValueError when min_codes exceeds what data can support."""
        df = pd.DataFrame({
            'text': ['short text', 'another text', 'third text']
        })

        # This should fail if min_codes is too high for the dataset
        with pytest.raises(ValueError, match="Dataset cannot support"):
            find_optimal_codes(df, text_column='text', min_codes=50)


class TestDivisionByZeroGuards:
    """Test that division by zero is prevented in metrics calculations."""

    def test_calculate_metrics_summary_empty_results(self):
        """Should return zeroed metrics for empty results without crashing."""
        class MockCoder:
            def __init__(self):
                self.codebook = {'CODE_01': {'count': 0}}

        coder = MockCoder()
        results_df = pd.DataFrame()

        # Should not raise ZeroDivisionError
        metrics = calculate_metrics_summary(coder, results_df)

        assert metrics['total_responses'] == 0
        assert metrics['avg_codes_per_response'] == 0.0
        assert metrics['coverage_pct'] == 0.0

    def test_calculate_metrics_summary_none_results(self):
        """Should handle None results without crashing."""
        class MockCoder:
            def __init__(self):
                self.codebook = {}

        coder = MockCoder()

        # Should not raise exception
        metrics = calculate_metrics_summary(coder, None)

        assert metrics['total_responses'] == 0


class TestSuccessfulValidation:
    """Test that valid datasets pass validation."""

    def test_run_ml_analysis_valid_dataset(self):
        """Should successfully run analysis on valid dataset."""
        df = pd.DataFrame({
            'text': [
                'This is a response about topic A',
                'Another response discussing topic B',
                'A third response about topic C',
                'More content about topic A and B',
                'Final response covering topic C',
                'Extra response for topic A',
                'Additional content about B',
                'More discussion on C',
                'Yet another response',
                'Last response here'
            ]
        })

        # Should complete without error
        coder, results_df, metrics = run_ml_analysis(
            df, text_column='text', n_codes=3
        )

        assert coder is not None
        assert results_df is not None
        assert len(results_df) == 10
        assert metrics['total_responses'] == 10

    def test_find_optimal_codes_valid_dataset(self):
        """Should successfully find optimal codes on valid dataset."""
        df = pd.DataFrame({
            'text': [
                'Response about machine learning and AI',
                'Data science and analytics discussion',
                'Cloud computing and infrastructure',
                'Software development practices',
                'Cybersecurity and privacy concerns',
                'DevOps and continuous integration',
                'Mobile app development',
                'Web development frameworks',
                'Database design patterns',
                'Agile project management'
            ]
        })

        # Should complete without error
        optimal_n, results = find_optimal_codes(
            df, text_column='text', min_codes=2, max_codes=5
        )

        assert optimal_n >= 2
        assert optimal_n <= 5
        assert 'silhouette_scores' in results
        assert 'optimal_n_codes' in results


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
