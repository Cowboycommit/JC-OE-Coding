"""
Data loading utilities for Open-Ended Coding Analysis.

This module provides robust data loading from multiple sources including:
- CSV files
- Excel files
- SQLite databases
- PostgreSQL databases
"""

import logging
import os
from typing import Union

import pandas as pd
import sqlite3
from sqlalchemy import create_engine


class DataLoader:
    """Handles data loading from various sources with error handling."""

    def __init__(self):
        """Initialize DataLoader with logging."""
        self.logger = logging.getLogger(self.__class__.__name__)

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
