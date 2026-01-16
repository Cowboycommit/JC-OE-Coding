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
- Gold standard text preprocessing and data cleaning
"""

__version__ = "1.1.0"
__author__ = "Your Name"

from .data_loader import DataLoader
from .code_frame import CodeFrame
from .theme_analyzer import ThemeAnalyzer
from .category_manager import CategoryManager

# Data cleaning and preprocessing
from .gold_standard_preprocessing import (
    GoldStandardTextProcessor,
    DataQualityMetrics,
    PreprocessingConfig,
    preprocess_dataframe,
    normalize_for_nlp,
    apply_gold_standard_normalization,
    create_processor_for_dataset,
)
from .text_preprocessor import (
    TextPreprocessor,
    TextPreprocessingError,
    DataCleaningPipeline,
)

# Embedding classes are available but not auto-imported to avoid dependencies
# Import explicitly: from src.embeddings import SentenceBERTEmbedder, etc.

# LLM interpretation is available but not auto-imported to avoid API dependencies
# Import explicitly: from src.llm_interpretation import LLMClusterInterpreter, etc.

__all__ = [
    # Core components
    "DataLoader",
    "CodeFrame",
    "ThemeAnalyzer",
    "CategoryManager",
    # Data cleaning and preprocessing
    "GoldStandardTextProcessor",
    "DataQualityMetrics",
    "PreprocessingConfig",
    "TextPreprocessor",
    "TextPreprocessingError",
    "DataCleaningPipeline",
    # Convenience functions
    "preprocess_dataframe",
    "normalize_for_nlp",
    "apply_gold_standard_normalization",
    "create_processor_for_dataset",
]
