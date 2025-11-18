"""
Theme analysis for qualitative data.

This module identifies and analyzes recurring themes in coded qualitative data.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from collections import Counter


class ThemeAnalyzer:
    """Analyzes and identifies themes in qualitative data."""

    def __init__(self):
        """Initialize ThemeAnalyzer."""
        self.themes = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def define_theme(
        self,
        theme_id: str,
        name: str,
        description: str,
        associated_codes: Optional[List[str]] = None,
    ):
        """
        Define a theme.

        Args:
            theme_id: Unique identifier
            name: Theme name
            description: Detailed description
            associated_codes: List of code IDs associated with this theme
        """
        if theme_id in self.themes:
            self.logger.warning(f"Theme {theme_id} already exists. Overwriting.")

        self.themes[theme_id] = {
            "name": name,
            "description": description,
            "codes": associated_codes or [],
            "responses": [],
        }
        self.logger.info(f"Defined theme: {theme_id} - {name}")

    def remove_theme(self, theme_id: str):
        """
        Remove a theme.

        Args:
            theme_id: Theme identifier to remove
        """
        if theme_id in self.themes:
            del self.themes[theme_id]
            self.logger.info(f"Removed theme: {theme_id}")
        else:
            self.logger.warning(f"Theme {theme_id} not found")

    def identify_themes(
        self, df: pd.DataFrame, code_column: str = "codes"
    ) -> pd.DataFrame:
        """
        Identify themes in coded data.

        Args:
            df: DataFrame with coded responses
            code_column: Column name containing codes

        Returns:
            DataFrame with theme assignments
        """
        # Reset response lists
        for theme_info in self.themes.values():
            theme_info["responses"] = []

        theme_assignments = []

        for idx, row in df.iterrows():
            response_codes = set(row[code_column]) if row[code_column] else set()
            matched_themes = []

            for theme_id, theme_info in self.themes.items():
                theme_codes = set(theme_info["codes"])
                if response_codes & theme_codes:  # Intersection
                    matched_themes.append(theme_id)
                    self.themes[theme_id]["responses"].append(idx)

            theme_assignments.append(matched_themes)

        df["themes"] = theme_assignments
        self.logger.info(f"Identified themes in {len(df)} responses")
        return df

    def theme_co_occurrence(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate co-occurrence matrix of themes.

        Args:
            df: Optional DataFrame with themes column

        Returns:
            DataFrame with theme co-occurrence counts
        """
        theme_ids = list(self.themes.keys())
        n_themes = len(theme_ids)
        co_occurrence = np.zeros((n_themes, n_themes), dtype=int)

        if df is not None and "themes" in df.columns:
            for themes in df["themes"]:
                for i, theme1 in enumerate(theme_ids):
                    for j, theme2 in enumerate(theme_ids):
                        if theme1 in themes and theme2 in themes:
                            co_occurrence[i, j] += 1

        return pd.DataFrame(
            co_occurrence,
            index=[self.themes[t]["name"] for t in theme_ids],
            columns=[self.themes[t]["name"] for t in theme_ids],
        )

    def get_theme_responses(self, theme_id: str) -> List[int]:
        """
        Get response indices for a specific theme.

        Args:
            theme_id: Theme identifier

        Returns:
            List of response indices
        """
        if theme_id in self.themes:
            return self.themes[theme_id]["responses"]
        return []

    def summary(self) -> pd.DataFrame:
        """
        Generate theme summary statistics.

        Returns:
            DataFrame with theme statistics
        """
        summary_data = []
        for theme_id, theme_info in self.themes.items():
            summary_data.append(
                {
                    "Theme ID": theme_id,
                    "Name": theme_info["name"],
                    "Description": theme_info["description"],
                    "Associated Codes": len(theme_info["codes"]),
                    "Frequency": len(theme_info["responses"]),
                }
            )
        return pd.DataFrame(summary_data).sort_values("Frequency", ascending=False)

    def get_dominant_theme(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get the most frequently occurring theme.

        Args:
            df: DataFrame with themes column

        Returns:
            Dictionary with theme counts
        """
        if "themes" not in df.columns:
            return {}

        theme_counts = Counter()
        for themes in df["themes"]:
            for theme in themes:
                theme_counts[theme] += 1

        return dict(theme_counts)

    def calculate_theme_coverage(self, df: pd.DataFrame) -> float:
        """
        Calculate percentage of responses with at least one theme.

        Args:
            df: DataFrame with themes column

        Returns:
            Coverage percentage
        """
        if "themes" not in df.columns:
            return 0.0

        with_themes = df["themes"].apply(lambda x: len(x) > 0).sum()
        total = len(df)

        return (with_themes / total * 100) if total > 0 else 0.0

    def export_themes(self, filepath: str):
        """
        Export theme definitions to CSV file.

        Args:
            filepath: Output file path
        """
        theme_data = []
        for theme_id, theme_info in self.themes.items():
            theme_data.append(
                {
                    "Theme ID": theme_id,
                    "Name": theme_info["name"],
                    "Description": theme_info["description"],
                    "Associated Codes": ", ".join(theme_info["codes"]),
                    "Frequency": len(theme_info["responses"]),
                }
            )

        df = pd.DataFrame(theme_data)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Exported themes to {filepath}")

    def __len__(self) -> int:
        """Return number of themes."""
        return len(self.themes)

    def __repr__(self) -> str:
        """String representation of ThemeAnalyzer."""
        return f"ThemeAnalyzer(themes={len(self.themes)})"
