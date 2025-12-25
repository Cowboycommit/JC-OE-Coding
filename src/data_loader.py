"""
Data loading utilities for Open-Ended Coding Analysis.

This module provides robust data loading from multiple sources including:
- CSV files
- Excel files
- SQLite databases
- PostgreSQL databases
- Content quality assessment and filtering
"""

import logging
import os
from typing import Union, Optional

import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from .content_quality import ContentQualityFilter


class DataLoader:
    """Handles data loading from various sources with error handling."""

    def __init__(self):
        """Initialize DataLoader with logging."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.quality_filter = None  # Lazily initialized when needed

    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            df = pd.read_csv(filepath, **kwargs)
            self.logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
            return df

        except pd.errors.EmptyDataError:
            self.logger.error(f"Empty file: {filepath}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading CSV {filepath}: {str(e)}")
            raise

    def load_excel(
        self, filepath: str, sheet_name: Union[str, int] = 0, **kwargs
    ) -> pd.DataFrame:
        """
        Load data from Excel file.

        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index
            **kwargs: Additional arguments for pd.read_excel

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")

            df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            self.logger.info(f"Successfully loaded {len(df)} rows from {filepath}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading Excel {filepath}: {str(e)}")
            raise

    def _validate_query(self, query: str) -> None:
        """
        Validate that a SQL query is safe for execution.

        Only SELECT queries are allowed. Destructive operations are rejected.

        Args:
            query: SQL query to validate

        Raises:
            ValueError: If query contains disallowed statements
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        normalized = query.strip().upper()

        # Check for destructive keywords
        disallowed = [
            "DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT",
            "ALTER", "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"
        ]
        for keyword in disallowed:
            # Check for keyword as a standalone word
            if f" {keyword} " in f" {normalized} " or normalized.startswith(f"{keyword} "):
                raise ValueError(
                    f"Query contains disallowed operation: {keyword}. "
                    "Only SELECT queries are permitted."
                )

        # Ensure query starts with SELECT or WITH (for CTEs)
        if not (normalized.startswith("SELECT") or normalized.startswith("WITH")):
            raise ValueError(
                "Query must start with SELECT or WITH. "
                "Only read operations are permitted."
            )

    def load_from_sqlite(self, db_path: str, query: str) -> pd.DataFrame:
        """
        Load data from SQLite database.

        Args:
            db_path: Path to SQLite database file
            query: SQL query to execute (SELECT statements only)

        Returns:
            DataFrame with query results

        Raises:
            FileNotFoundError: If database file doesn't exist
            ValueError: If query contains disallowed operations
            Exception: For database connection or query errors

        Security:
            Only SELECT queries are permitted. Destructive operations
            (DROP, DELETE, UPDATE, INSERT, etc.) are rejected.
        """
        self._validate_query(query)

        try:
            if not os.path.exists(db_path):
                raise FileNotFoundError(f"Database not found: {db_path}")

            with sqlite3.connect(db_path) as conn:
                df = pd.read_sql_query(query, conn)
            self.logger.info(f"Successfully loaded {len(df)} rows from SQLite database")
            return df

        except Exception as e:
            self.logger.error(f"Error loading from SQLite: {str(e)}")
            raise

    def load_from_postgres(self, connection_string: str, query: str) -> pd.DataFrame:
        """
        Load data from PostgreSQL database.

        Args:
            connection_string: PostgreSQL connection string
            query: SQL query to execute (SELECT statements only)

        Returns:
            DataFrame with query results

        Raises:
            ValueError: If query contains disallowed operations
            Exception: For database connection or query errors

        Security:
            Only SELECT queries are permitted. Destructive operations
            (DROP, DELETE, UPDATE, INSERT, etc.) are rejected.
        """
        self._validate_query(query)

        try:
            engine = create_engine(connection_string)
            try:
                df = pd.read_sql_query(query, engine)
                self.logger.info(
                    f"Successfully loaded {len(df)} rows from PostgreSQL database"
                )
                return df
            finally:
                engine.dispose()

        except Exception as e:
            self.logger.error(f"Error loading from PostgreSQL: {str(e)}")
            raise

    def load_json(
        self, filepath: str, lines: bool = False, orient: str = None, **kwargs
    ) -> pd.DataFrame:
        """
        Load data from JSON file.

        Args:
            filepath: Path to JSON file
            lines: Whether the file is in JSON lines format
            orient: Expected JSON orientation (passed to ``pandas.read_json``)
            **kwargs: Additional arguments for ``pandas.read_json``

        Returns:
            DataFrame with loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON cannot be decoded
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            df = pd.read_json(filepath, lines=lines, orient=orient, **kwargs)
            self.logger.info(
                "Successfully loaded %s rows from JSON file %s", len(df), filepath
            )
            return df
        except ValueError:
            self.logger.error("Invalid JSON content in %s", filepath)
            raise
        except Exception as e:
            self.logger.error(f"Error loading JSON {filepath}: {str(e)}")
            raise

    def validate_dataframe(
        self, df: pd.DataFrame, required_columns: list = None
    ) -> bool:
        """
        Validate DataFrame structure.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty")

        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        self.logger.info(f"DataFrame validated: {df.shape[0]} rows, {df.shape[1]} columns")
        return True

    def assess_content_quality(
        self,
        df: pd.DataFrame,
        text_column: str,
        min_words: int = 3,
        min_chars: int = 10,
        max_repetition_ratio: float = 0.7,
        min_english_word_ratio: float = 0.3
    ) -> pd.DataFrame:
        """
        Assess content quality for text responses in a DataFrame.

        This method adds quality assessment columns to the DataFrame WITHOUT
        removing any rows. All responses are retained and flagged for review.

        Args:
            df: DataFrame containing text responses
            text_column: Name of column containing text to assess
            min_words: Minimum number of words for analytic content (default: 3)
            min_chars: Minimum number of characters (default: 10)
            max_repetition_ratio: Maximum ratio of repeated words (default: 0.7)
            min_english_word_ratio: Minimum ratio of English words (default: 0.3)

        Returns:
            DataFrame with added quality assessment columns:
            - quality_is_analytic: bool
            - quality_confidence: float
            - quality_reason: str
            - quality_recommendation: str ('include', 'review', 'exclude')
            - quality_flags: list

        Notes:
            - NO responses are automatically excluded
            - All flagging is transparent and auditable
            - Use export_non_analytic_responses() to save flagged responses
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        # Initialize quality filter if not already done
        if self.quality_filter is None:
            self.quality_filter = ContentQualityFilter(
                min_words=min_words,
                min_chars=min_chars,
                max_repetition_ratio=max_repetition_ratio,
                min_english_word_ratio=min_english_word_ratio,
                logger=self.logger
            )

        # Assess each response
        self.logger.info(f"Assessing content quality for {len(df)} responses...")
        assessments = self.quality_filter.batch_assess(df[text_column].tolist())

        # Add assessment columns to DataFrame
        df['quality_is_analytic'] = [a['is_analytic'] for a in assessments]
        df['quality_confidence'] = [a['confidence'] for a in assessments]
        df['quality_reason'] = [a['reason'] for a in assessments]
        df['quality_recommendation'] = [a['recommendation'] for a in assessments]
        df['quality_flags'] = [a['flags'] for a in assessments]

        # Log summary statistics
        non_analytic_count = (~df['quality_is_analytic']).sum()
        exclude_count = (df['quality_recommendation'] == 'exclude').sum()
        review_count = (df['quality_recommendation'] == 'review').sum()

        self.logger.info(
            f"Quality assessment complete: "
            f"{non_analytic_count}/{len(df)} flagged as non-analytic "
            f"({exclude_count} exclude, {review_count} review)"
        )

        # Get flag statistics
        flag_stats = self.quality_filter.get_flag_statistics(assessments)
        if flag_stats:
            self.logger.info(f"Flag distribution: {flag_stats}")

        return df

    def export_non_analytic_responses(
        self,
        df: pd.DataFrame,
        output_path: str = 'non_analytic_responses.csv',
        include_review: bool = True
    ) -> int:
        """
        Export non-analytic responses to CSV for human review.

        Args:
            df: DataFrame with quality assessment columns
            output_path: Path to export CSV file
            include_review: Include responses marked for review (not just exclude)

        Returns:
            Number of responses exported

        Raises:
            ValueError: If quality assessment columns not found
        """
        required_cols = ['quality_is_analytic', 'quality_recommendation']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Quality assessment columns not found: {missing}. "
                "Run assess_content_quality() first."
            )

        # Filter responses based on recommendation
        if include_review:
            mask = df['quality_recommendation'].isin(['exclude', 'review'])
        else:
            mask = df['quality_recommendation'] == 'exclude'

        flagged_df = df[mask].copy()

        if len(flagged_df) == 0:
            self.logger.info("No non-analytic responses to export")
            return 0

        # Export to CSV
        flagged_df.to_csv(output_path, index=False)
        self.logger.info(
            f"Exported {len(flagged_df)} non-analytic responses to {output_path}"
        )

        return len(flagged_df)

    def get_quality_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for quality assessment.

        Args:
            df: DataFrame with quality assessment columns

        Returns:
            Dictionary with summary statistics

        Raises:
            ValueError: If quality assessment columns not found
        """
        required_cols = ['quality_is_analytic', 'quality_recommendation', 'quality_flags']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Quality assessment columns not found: {missing}. "
                "Run assess_content_quality() first."
            )

        total = len(df)
        analytic = df['quality_is_analytic'].sum()
        non_analytic = total - analytic

        return {
            'total_responses': total,
            'analytic_responses': int(analytic),
            'non_analytic_responses': int(non_analytic),
            'analytic_percentage': round(analytic / total * 100, 2) if total > 0 else 0,
            'recommendations': {
                'include': int((df['quality_recommendation'] == 'include').sum()),
                'review': int((df['quality_recommendation'] == 'review').sum()),
                'exclude': int((df['quality_recommendation'] == 'exclude').sum()),
            },
            'avg_confidence': round(df['quality_confidence'].mean(), 3),
            'flag_counts': self.quality_filter.get_flag_statistics(
                df[['quality_is_analytic', 'quality_confidence', 'quality_reason',
                    'quality_recommendation', 'quality_flags']].to_dict('records')
            ) if self.quality_filter else {}
        }
