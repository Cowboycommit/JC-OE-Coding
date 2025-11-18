"""
Unit tests for CodeFrame class.
"""

import pytest
import pandas as pd
from src.code_frame import CodeFrame


class TestCodeFrame:
    """Test cases for CodeFrame."""

    @pytest.fixture
    def code_frame(self):
        """Create CodeFrame instance."""
        return CodeFrame("Test Frame", "Test description")

    def test_initialization(self, code_frame):
        """Test CodeFrame initialization."""
        assert code_frame.name == "Test Frame"
        assert code_frame.description == "Test description"
        assert len(code_frame.codes) == 0

    def test_add_code(self, code_frame):
        """Test adding a code."""
        code_frame.add_code(
            'TEST_CODE',
            'Test Label',
            description='Test description',
            keywords=['test', 'sample']
        )
        assert 'TEST_CODE' in code_frame.codes
        assert code_frame.codes['TEST_CODE']['label'] == 'Test Label'
        assert len(code_frame.codes['TEST_CODE']['keywords']) == 2

    def test_add_code_with_parent(self, code_frame):
        """Test adding a code with parent."""
        code_frame.add_code('PARENT', 'Parent Code')
        code_frame.add_code('CHILD', 'Child Code', parent='PARENT')
        assert code_frame.codes['CHILD']['parent'] == 'PARENT'

    def test_apply_codes(self, code_frame):
        """Test applying codes to text."""
        code_frame.add_code('POSITIVE', 'Positive', keywords=['good', 'great'])
        code_frame.add_code('NEGATIVE', 'Negative', keywords=['bad', 'poor'])

        text = "This is a great example"
        matched = code_frame.apply_codes(text)
        assert 'POSITIVE' in matched
        assert 'NEGATIVE' not in matched

    def test_apply_codes_case_insensitive(self, code_frame):
        """Test case-insensitive code application."""
        code_frame.add_code('TEST', 'Test', keywords=['Example'])

        text = "This is an example"
        matched = code_frame.apply_codes(text, case_sensitive=False)
        assert 'TEST' in matched

    def test_apply_codes_empty_text(self, code_frame):
        """Test applying codes to empty text."""
        matched = code_frame.apply_codes("")
        assert matched == []

    def test_get_hierarchy(self, code_frame):
        """Test getting code hierarchy."""
        code_frame.add_code('PARENT', 'Parent')
        code_frame.add_code('CHILD1', 'Child 1', parent='PARENT')
        code_frame.add_code('CHILD2', 'Child 2', parent='PARENT')

        hierarchy = code_frame.get_hierarchy()
        assert 'PARENT' in hierarchy
        assert len(hierarchy['PARENT']) == 2
        assert 'CHILD1' in hierarchy['PARENT']

    def test_get_children(self, code_frame):
        """Test getting children of a code."""
        code_frame.add_code('PARENT', 'Parent')
        code_frame.add_code('CHILD', 'Child', parent='PARENT')

        children = code_frame.get_children('PARENT')
        assert 'CHILD' in children

    def test_get_parent(self, code_frame):
        """Test getting parent of a code."""
        code_frame.add_code('PARENT', 'Parent')
        code_frame.add_code('CHILD', 'Child', parent='PARENT')

        parent = code_frame.get_parent('CHILD')
        assert parent == 'PARENT'

    def test_summary(self, code_frame):
        """Test generating summary."""
        code_frame.add_code('CODE1', 'Code 1', keywords=['test'])
        code_frame.add_code('CODE2', 'Code 2', keywords=['sample'])

        summary = code_frame.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert 'Code ID' in summary.columns

    def test_reset_counts(self, code_frame):
        """Test resetting code counts."""
        code_frame.add_code('TEST', 'Test', keywords=['test'])
        code_frame.apply_codes("test text")
        assert code_frame.codes['TEST']['count'] > 0

        code_frame.reset_counts()
        assert code_frame.codes['TEST']['count'] == 0

    def test_remove_code(self, code_frame):
        """Test removing a code."""
        code_frame.add_code('TEST', 'Test')
        assert 'TEST' in code_frame.codes

        code_frame.remove_code('TEST')
        assert 'TEST' not in code_frame.codes

    def test_len(self, code_frame):
        """Test __len__ method."""
        assert len(code_frame) == 0
        code_frame.add_code('TEST', 'Test')
        assert len(code_frame) == 1
