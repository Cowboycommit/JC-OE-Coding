"""
Unit tests for ThemeAnalyzer class.
"""

import pytest
import pandas as pd
from src.theme_analyzer import ThemeAnalyzer


class TestThemeAnalyzer:
    """Test cases for ThemeAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create ThemeAnalyzer instance."""
        return ThemeAnalyzer()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['sample1', 'sample2', 'sample3'],
            'codes': [['CODE1', 'CODE2'], ['CODE2'], ['CODE1', 'CODE3']]
        })

    def test_initialization(self, analyzer):
        """Test ThemeAnalyzer initialization."""
        assert len(analyzer.themes) == 0

    def test_define_theme(self, analyzer):
        """Test defining a theme."""
        analyzer.define_theme(
            'THEME1',
            'Test Theme',
            'Description',
            associated_codes=['CODE1', 'CODE2']
        )
        assert 'THEME1' in analyzer.themes
        assert analyzer.themes['THEME1']['name'] == 'Test Theme'
        assert len(analyzer.themes['THEME1']['codes']) == 2

    def test_identify_themes(self, analyzer, sample_df):
        """Test identifying themes in data."""
        analyzer.define_theme('THEME1', 'Theme 1', 'Desc', ['CODE1'])
        analyzer.define_theme('THEME2', 'Theme 2', 'Desc', ['CODE2'])

        df = analyzer.identify_themes(sample_df)
        assert 'themes' in df.columns
        assert isinstance(df.iloc[0]['themes'], list)

    def test_get_theme_responses(self, analyzer, sample_df):
        """Test getting responses for a theme."""
        analyzer.define_theme('THEME1', 'Theme 1', 'Desc', ['CODE1'])
        analyzer.identify_themes(sample_df)

        responses = analyzer.get_theme_responses('THEME1')
        assert isinstance(responses, list)

    def test_summary(self, analyzer):
        """Test generating theme summary."""
        analyzer.define_theme('THEME1', 'Theme 1', 'Desc', ['CODE1'])
        analyzer.define_theme('THEME2', 'Theme 2', 'Desc', ['CODE2'])

        summary = analyzer.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert 'Theme ID' in summary.columns

    def test_remove_theme(self, analyzer):
        """Test removing a theme."""
        analyzer.define_theme('THEME1', 'Theme 1', 'Desc')
        assert 'THEME1' in analyzer.themes

        analyzer.remove_theme('THEME1')
        assert 'THEME1' not in analyzer.themes

    def test_get_dominant_theme(self, analyzer, sample_df):
        """Test getting dominant theme."""
        analyzer.define_theme('THEME1', 'Theme 1', 'Desc', ['CODE1'])
        analyzer.define_theme('THEME2', 'Theme 2', 'Desc', ['CODE2'])

        df = analyzer.identify_themes(sample_df)
        dominant = analyzer.get_dominant_theme(df)

        assert isinstance(dominant, dict)

    def test_calculate_theme_coverage(self, analyzer, sample_df):
        """Test calculating theme coverage."""
        analyzer.define_theme('THEME1', 'Theme 1', 'Desc', ['CODE1'])
        df = analyzer.identify_themes(sample_df)

        coverage = analyzer.calculate_theme_coverage(df)
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 100

    def test_theme_co_occurrence(self, analyzer, sample_df):
        """Test theme co-occurrence matrix."""
        analyzer.define_theme('THEME1', 'Theme 1', 'Desc', ['CODE1'])
        analyzer.define_theme('THEME2', 'Theme 2', 'Desc', ['CODE2'])

        df = analyzer.identify_themes(sample_df)
        co_occur = analyzer.theme_co_occurrence(df)

        assert isinstance(co_occur, pd.DataFrame)
        assert co_occur.shape[0] == co_occur.shape[1]

    def test_len(self, analyzer):
        """Test __len__ method."""
        assert len(analyzer) == 0
        analyzer.define_theme('THEME1', 'Theme 1', 'Desc')
        assert len(analyzer) == 1
