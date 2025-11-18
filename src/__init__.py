"""
Open-Ended Coding Analysis Framework

A comprehensive toolkit for qualitative data analysis including:
- Code frames for systematic categorization
- Theme identification and analysis
- Multi-level categorization
- Data loading from multiple sources
- Interactive visualizations
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import DataLoader
from .code_frame import CodeFrame
from .theme_analyzer import ThemeAnalyzer
from .category_manager import CategoryManager

__all__ = [
    "DataLoader",
    "CodeFrame",
    "ThemeAnalyzer",
    "CategoryManager",
]
