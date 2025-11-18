"""
Code frame management for qualitative data analysis.

This module provides systematic coding structures for categorizing qualitative data.
"""

import logging
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd


class CodeFrame:
    """Manages coding frames for qualitative analysis."""

    def __init__(self, name: str, description: str = ""):
        """
        Initialize CodeFrame.

        Args:
            name: Name of the code frame
            description: Description of the code frame
        """
        self.name = name
        self.description = description
        self.codes = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_code(
        self,
        code_id: str,
        label: str,
        description: str = "",
        keywords: Optional[List[str]] = None,
        parent: Optional[str] = None,
    ):
        """
        Add a code to the frame.

        Args:
            code_id: Unique identifier for the code
            label: Human-readable label
            description: Detailed description of the code
            keywords: List of keywords associated with this code
            parent: Parent code ID for hierarchical structures
        """
        if code_id in self.codes:
            self.logger.warning(f"Code {code_id} already exists. Overwriting.")

        self.codes[code_id] = {
            "label": label,
            "description": description,
            "keywords": keywords or [],
            "parent": parent,
            "count": 0,
        }
        self.logger.info(f"Added code: {code_id} - {label}")

    def remove_code(self, code_id: str):
        """
        Remove a code from the frame.

        Args:
            code_id: Code identifier to remove
        """
        if code_id in self.codes:
            del self.codes[code_id]
            self.logger.info(f"Removed code: {code_id}")
        else:
            self.logger.warning(f"Code {code_id} not found")

    def apply_codes(self, text: str, case_sensitive: bool = False) -> List[str]:
        """
        Apply codes to text based on keyword matching.

        Args:
            text: Text to code
            case_sensitive: Whether to use case-sensitive matching

        Returns:
            List of matching code IDs
        """
        if not text:
            return []

        if not case_sensitive:
            text = text.lower()

        matched_codes = []
        for code_id, code_info in self.codes.items():
            keywords = code_info["keywords"]
            if not case_sensitive:
                keywords = [k.lower() for k in keywords]

            for keyword in keywords:
                if keyword in text:
                    matched_codes.append(code_id)
                    self.codes[code_id]["count"] += 1
                    break

        return matched_codes

    def get_hierarchy(self) -> Dict:
        """
        Get hierarchical structure of codes.

        Returns:
            Dictionary mapping parents to child codes
        """
        hierarchy = defaultdict(list)
        for code_id, code_info in self.codes.items():
            parent = code_info.get("parent")
            if parent:
                hierarchy[parent].append(code_id)
            else:
                hierarchy["root"].append(code_id)
        return dict(hierarchy)

    def get_children(self, code_id: str) -> List[str]:
        """
        Get child codes of a given code.

        Args:
            code_id: Parent code ID

        Returns:
            List of child code IDs
        """
        hierarchy = self.get_hierarchy()
        return hierarchy.get(code_id, [])

    def get_parent(self, code_id: str) -> Optional[str]:
        """
        Get parent code of a given code.

        Args:
            code_id: Child code ID

        Returns:
            Parent code ID or None
        """
        if code_id in self.codes:
            return self.codes[code_id].get("parent")
        return None

    def summary(self) -> pd.DataFrame:
        """
        Generate summary statistics of code usage.

        Returns:
            DataFrame with code statistics
        """
        summary_data = []
        for code_id, code_info in self.codes.items():
            summary_data.append(
                {
                    "Code ID": code_id,
                    "Label": code_info["label"],
                    "Count": code_info["count"],
                    "Parent": code_info.get("parent", "None"),
                    "Keywords": len(code_info["keywords"]),
                }
            )
        return pd.DataFrame(summary_data).sort_values("Count", ascending=False)

    def reset_counts(self):
        """Reset all code counts to zero."""
        for code_info in self.codes.values():
            code_info["count"] = 0
        self.logger.info("Reset all code counts")

    def export_codebook(self, filepath: str):
        """
        Export codebook to CSV file.

        Args:
            filepath: Output file path
        """
        codebook_data = []
        for code_id, code_info in self.codes.items():
            codebook_data.append(
                {
                    "Code ID": code_id,
                    "Label": code_info["label"],
                    "Description": code_info["description"],
                    "Keywords": ", ".join(code_info["keywords"]),
                    "Parent": code_info.get("parent", ""),
                }
            )

        df = pd.DataFrame(codebook_data)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Exported codebook to {filepath}")

    def __len__(self) -> int:
        """Return number of codes in the frame."""
        return len(self.codes)

    def __repr__(self) -> str:
        """String representation of CodeFrame."""
        return f"CodeFrame(name='{self.name}', codes={len(self.codes)})"
