"""
Unit tests for DataLoader class.
"""

import os
import pytest
import pandas as pd
import tempfile
from src.data_loader import DataLoader


class TestDataLoader:
    """Test cases for DataLoader."""

    @pytest.fixture
    def data_loader(self):
        """Create DataLoader instance."""
        return DataLoader()

    @pytest.fixture
    def sample_csv(self):
        """Create temporary CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,text,value\n")
            f.write("1,sample,100\n")
            f.write("2,test,200\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_load_csv_success(self, data_loader, sample_csv):
        """Test successful CSV loading."""
        df = data_loader.load_csv(sample_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['id', 'text', 'value']

    def test_load_csv_file_not_found(self, data_loader):
        """Test CSV loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            data_loader.load_csv('nonexistent.csv')

    def test_validate_dataframe_success(self, data_loader):
        """Test DataFrame validation."""
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        assert data_loader.validate_dataframe(df) is True

    def test_validate_dataframe_empty(self, data_loader):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            data_loader.validate_dataframe(df)

    def test_validate_dataframe_missing_columns(self, data_loader):
        """Test validation with missing required columns."""
        df = pd.DataFrame({'col1': [1, 2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            data_loader.validate_dataframe(df, required_columns=['col1', 'col2'])

    def test_load_csv_empty_file(self, data_loader):
        """Test loading empty CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(pd.errors.EmptyDataError):
                data_loader.load_csv(temp_path)
        finally:
            os.unlink(temp_path)
