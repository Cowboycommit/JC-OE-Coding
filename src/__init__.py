"""
Open-Ended Coding Analysis Framework

A comprehensive toolkit for qualitative data analysis including:
- Code frames for systematic categorization
- Theme identification and analysis
- Multi-level categorization
- Data loading from multiple sources
- Interactive visualizations
- Semantic embeddings for text representation
- LLM-assisted cluster interpretation (optional)
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import DataLoader
from .code_frame import CodeFrame
from .theme_analyzer import ThemeAnalyzer
from .category_manager import CategoryManager

# Embedding classes are available but not auto-imported to avoid dependencies
# Import explicitly: from src.embeddings import SentenceBERTEmbedder, etc.

# LLM interpretation is available but not auto-imported to avoid API dependencies
# Import explicitly: from src.llm_interpretation import LLMClusterInterpreter, etc.

__all__ = [
    "DataLoader",
    "CodeFrame",
    "ThemeAnalyzer",
    "CategoryManager",
]
