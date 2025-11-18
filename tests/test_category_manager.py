"""
Unit tests for CategoryManager class.
"""

import pytest
import pandas as pd
from src.category_manager import CategoryManager


class TestCategoryManager:
    """Test cases for CategoryManager."""

    @pytest.fixture
    def manager(self):
        """Create CategoryManager instance."""
        return CategoryManager()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            'id': [1, 2, 3],
            'text': ['sample1', 'sample2', 'sample3'],
            'codes': [['CODE1', 'CODE2'], ['CODE2'], ['CODE1', 'CODE3']],
            'themes': [['THEME1'], ['THEME2'], ['THEME1', 'THEME2']]
        })

    def test_initialization(self, manager):
        """Test CategoryManager initialization."""
        assert len(manager.categories) == 0

    def test_create_category(self, manager):
        """Test creating a category."""
        manager.create_category(
            'CAT1',
            'Category 1',
            criteria={'codes_required': ['CODE1']},
            level=1
        )
        assert 'CAT1' in manager.categories
        assert manager.categories['CAT1']['name'] == 'Category 1'
        assert manager.categories['CAT1']['level'] == 1

    def test_categorize(self, manager, sample_df):
        """Test categorizing data."""
        manager.create_category(
            'CAT1',
            'Category 1',
            criteria={'codes_required': ['CODE1']}
        )
        df = manager.categorize(sample_df)

        assert 'categories' in df.columns
        assert isinstance(df.iloc[0]['categories'], list)

    def test_meets_criteria_codes_required(self, manager):
        """Test criteria checking for required codes."""
        row = pd.Series({'codes': ['CODE1', 'CODE2']})
        criteria = {'codes_required': ['CODE1']}

        assert manager._meets_criteria(row, criteria) is True

        criteria = {'codes_required': ['CODE3']}
        assert manager._meets_criteria(row, criteria) is False

    def test_meets_criteria_codes_all(self, manager):
        """Test criteria checking for all codes."""
        row = pd.Series({'codes': ['CODE1', 'CODE2']})
        criteria = {'codes_all': ['CODE1', 'CODE2']}

        assert manager._meets_criteria(row, criteria) is True

        criteria = {'codes_all': ['CODE1', 'CODE3']}
        assert manager._meets_criteria(row, criteria) is False

    def test_meets_criteria_min_codes(self, manager):
        """Test criteria checking for minimum codes."""
        row = pd.Series({'codes': ['CODE1', 'CODE2']})
        criteria = {'min_codes': 2}

        assert manager._meets_criteria(row, criteria) is True

        criteria = {'min_codes': 3}
        assert manager._meets_criteria(row, criteria) is False

    def test_get_category_responses(self, manager, sample_df):
        """Test getting responses for a category."""
        manager.create_category(
            'CAT1',
            'Category 1',
            criteria={'codes_required': ['CODE1']}
        )
        df = manager.categorize(sample_df)

        cat_responses = manager.get_category_responses(df, 'CAT1')
        assert isinstance(cat_responses, pd.DataFrame)

    def test_summary(self, manager):
        """Test generating category summary."""
        manager.create_category('CAT1', 'Category 1', {}, level=1)
        manager.create_category('CAT2', 'Category 2', {}, level=2)

        summary = manager.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert 'Category ID' in summary.columns

    def test_remove_category(self, manager):
        """Test removing a category."""
        manager.create_category('CAT1', 'Category 1', {})
        assert 'CAT1' in manager.categories

        manager.remove_category('CAT1')
        assert 'CAT1' not in manager.categories

    def test_get_categories_by_level(self, manager):
        """Test getting categories by level."""
        manager.create_category('CAT1', 'Category 1', {}, level=1)
        manager.create_category('CAT2', 'Category 2', {}, level=1)
        manager.create_category('CAT3', 'Category 3', {}, level=2)

        level1 = manager.get_categories_by_level(1)
        assert len(level1) == 2
        assert 'CAT1' in level1
        assert 'CAT2' in level1

    def test_calculate_coverage(self, manager, sample_df):
        """Test calculating category coverage."""
        manager.create_category(
            'CAT1',
            'Category 1',
            criteria={'codes_required': ['CODE1']}
        )
        df = manager.categorize(sample_df)

        coverage = manager.calculate_coverage(df)
        assert isinstance(coverage, float)
        assert 0 <= coverage <= 100

    def test_cross_tabulation(self, manager, sample_df):
        """Test cross-tabulation between categories."""
        manager.create_category('CAT1', 'Category 1', {'codes_required': ['CODE1']})
        manager.create_category('CAT2', 'Category 2', {'codes_required': ['CODE2']})

        df = manager.categorize(sample_df)
        crosstab = manager.cross_tabulation(df, 'CAT1', 'CAT2')

        assert isinstance(crosstab, pd.DataFrame)

    def test_len(self, manager):
        """Test __len__ method."""
        assert len(manager) == 0
        manager.create_category('CAT1', 'Category 1', {})
        assert len(manager) == 1
