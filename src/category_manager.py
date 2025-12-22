"""
Category management for multi-level classification.

This module provides advanced categorization and classification of coded data.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd


class CategoryManager:
    """Manages multi-level categorization of qualitative data."""

    def __init__(self):
        """Initialize CategoryManager."""
        self.categories = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_category(
        self, category_id: str, name: str, criteria: Dict, level: int = 1
    ):
        """
        Create a category.

        Args:
            category_id: Unique identifier
            name: Category name
            criteria: Dictionary defining categorization criteria
            level: Hierarchical level (1 = top level)
        """
        if category_id in self.categories:
            self.logger.warning(f"Category {category_id} already exists. Overwriting.")

        self.categories[category_id] = {
            "name": name,
            "criteria": criteria,
            "level": level,
            "count": 0,
        }
        self.logger.info(f"Created category: {category_id} - {name} (Level {level})")

    def remove_category(self, category_id: str):
        """
        Remove a category.

        Args:
            category_id: Category identifier to remove
        """
        if category_id in self.categories:
            del self.categories[category_id]
            self.logger.info(f"Removed category: {category_id}")
        else:
            self.logger.warning(f"Category {category_id} not found")

    def categorize(
        self, df: pd.DataFrame, code_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply categorization to DataFrame.

        Args:
            df: DataFrame to categorize
            code_columns: List of code column names to consider

        Returns:
            DataFrame with category assignments
        """
        # Reset counts
        for cat_info in self.categories.values():
            cat_info["count"] = 0

        categories_assigned = []

        for idx, row in df.iterrows():
            assigned = []

            for cat_id, cat_info in self.categories.items():
                if self._meets_criteria(row, cat_info["criteria"]):
                    assigned.append(cat_id)
                    self.categories[cat_id]["count"] += 1

            categories_assigned.append(assigned)

        df["categories"] = categories_assigned
        self.logger.info(f"Categorized {len(df)} responses")
        return df

    def _meets_criteria(self, row: pd.Series, criteria: Dict) -> bool:
        """
        Check if a row meets category criteria.

        Args:
            row: DataFrame row
            criteria: Criteria dictionary

        Returns:
            True if criteria are met
        """
        for key, value in criteria.items():
            if key == "codes_required":
                # Check if any required codes are present
                if "codes" not in row or not any(
                    code in row.get("codes", []) for code in value
                ):
                    return False

            elif key == "codes_all":
                # Check if all codes are present
                if "codes" not in row or not all(
                    code in row.get("codes", []) for code in value
                ):
                    return False

            elif key == "themes_required":
                # Check if any required themes are present
                if "themes" not in row or not any(
                    theme in row.get("themes", []) for theme in value
                ):
                    return False

            elif key == "themes_all":
                # Check if all themes are present
                if "themes" not in row or not all(
                    theme in row.get("themes", []) for theme in value
                ):
                    return False

            elif key == "min_codes":
                # Check minimum number of codes
                if "codes" not in row or len(row.get("codes", [])) < value:
                    return False

            elif key == "max_codes":
                # Check maximum number of codes
                if "codes" not in row or len(row.get("codes", [])) > value:
                    return False

        return True

    def get_category_responses(
        self, df: pd.DataFrame, category_id: str
    ) -> pd.DataFrame:
        """
        Get all responses for a specific category.

        Args:
            df: DataFrame with categories
            category_id: Category identifier

        Returns:
            Filtered DataFrame
        """
        if "categories" not in df.columns:
            return pd.DataFrame()

        return df[df["categories"].apply(lambda x: category_id in x)]

    def summary(self) -> pd.DataFrame:
        """
        Generate category summary.

        Returns:
            DataFrame with category statistics
        """
        summary_data = []
        for cat_id, cat_info in self.categories.items():
            summary_data.append(
                {
                    "Category ID": cat_id,
                    "Name": cat_info["name"],
                    "Level": cat_info["level"],
                    "Count": cat_info["count"],
                }
            )
        return pd.DataFrame(summary_data).sort_values(["Level", "Count"], ascending=[True, False])

    def cross_tabulation(
        self, df: pd.DataFrame, category1: str, category2: str
    ) -> pd.DataFrame:
        """
        Create cross-tabulation between categories.

        Args:
            df: DataFrame with categories
            category1: First category ID
            category2: Second category ID

        Returns:
            Cross-tabulation DataFrame

        Raises:
            ValueError: If category IDs are not found or column names would conflict
        """
        if "categories" not in df.columns:
            return pd.DataFrame()

        # Validate category IDs exist
        if category1 not in self.categories:
            raise ValueError(f"Category '{category1}' not found in categories")
        if category2 not in self.categories:
            raise ValueError(f"Category '{category2}' not found in categories")

        # Check for column name conflicts
        col1 = f"has_{category1}"
        col2 = f"has_{category2}"
        if col1 in df.columns:
            raise ValueError(
                f"Column '{col1}' already exists in DataFrame. "
                "Rename or remove it before calling cross_tabulation."
            )
        if col2 in df.columns:
            raise ValueError(
                f"Column '{col2}' already exists in DataFrame. "
                "Rename or remove it before calling cross_tabulation."
            )

        # Work on a copy to avoid mutating the original DataFrame
        df_copy = df.copy()

        # Create binary indicators
        df_copy[col1] = df_copy["categories"].apply(
            lambda x: 1 if category1 in x else 0
        )
        df_copy[col2] = df_copy["categories"].apply(
            lambda x: 1 if category2 in x else 0
        )

        crosstab = pd.crosstab(
            df_copy[col1],
            df_copy[col2],
            rownames=[self.categories[category1]["name"]],
            colnames=[self.categories[category2]["name"]],
        )

        return crosstab

    def get_categories_by_level(self, level: int) -> List[str]:
        """
        Get all category IDs at a specific level.

        Args:
            level: Hierarchical level

        Returns:
            List of category IDs
        """
        return [
            cat_id
            for cat_id, cat_info in self.categories.items()
            if cat_info["level"] == level
        ]

    def calculate_coverage(self, df: pd.DataFrame) -> float:
        """
        Calculate percentage of responses with at least one category.

        Args:
            df: DataFrame with categories column

        Returns:
            Coverage percentage
        """
        if "categories" not in df.columns:
            return 0.0

        with_categories = df["categories"].apply(lambda x: len(x) > 0).sum()
        total = len(df)

        return (with_categories / total * 100) if total > 0 else 0.0

    def export_categories(self, filepath: str):
        """
        Export category definitions to CSV file.

        Args:
            filepath: Output file path
        """
        category_data = []
        for cat_id, cat_info in self.categories.items():
            category_data.append(
                {
                    "Category ID": cat_id,
                    "Name": cat_info["name"],
                    "Level": cat_info["level"],
                    "Criteria": str(cat_info["criteria"]),
                    "Count": cat_info["count"],
                }
            )

        df = pd.DataFrame(category_data)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Exported categories to {filepath}")

    def __len__(self) -> int:
        """Return number of categories."""
        return len(self.categories)

    def __repr__(self) -> str:
        """String representation of CategoryManager."""
        return f"CategoryManager(categories={len(self.categories)})"
