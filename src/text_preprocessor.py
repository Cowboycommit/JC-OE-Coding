"""
Text Preprocessing Module for NLP Analysis.

Provides comprehensive text preprocessing utilities for topic modeling,
sentiment analysis, and other NLP tasks. Integrates with gold standard
preprocessing for consistent data cleaning.

This module is adapted from the JC-Text-Analysis-NLP project to provide
text preprocessing capabilities for the Open-Ended Coding Analysis framework.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from .gold_standard_preprocessing import (
    GoldStandardTextProcessor,
    DataQualityMetrics,
    PreprocessingConfig,
    normalize_for_nlp,
)

logger = logging.getLogger(__name__)

# Negation words to preserve during stopword removal for sentiment/topic analysis
# These words are important for understanding context and sentiment polarity
NEGATION_KEEP_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere",
    "cannot", "can't", "don't", "doesn't", "didn't", "won't", "wouldn't",
    "shouldn't", "couldn't", "hasn't", "haven't", "hadn't", "isn't", "aren't",
    "wasn't", "weren't"
}

# Default path for domain-specific stopwords file
# This file contains survey-specific stopwords that should be removed from survey text
DEFAULT_DOMAIN_STOPWORDS_PATH = Path(__file__).parent.parent / "data" / "stopwords_domain.txt"


def _download_nltk_resources():
    """Download required NLTK resources if not already present."""
    try:
        import nltk
        resources = [
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
        ]
        for path, name in resources:
            try:
                nltk.data.find(path)
            except LookupError:
                try:
                    nltk.download(name, quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download NLTK resource {name}: {e}")
    except ImportError:
        logger.warning("NLTK not installed. Some preprocessing features will be limited.")


class TextPreprocessingError(Exception):
    """Custom exception for text preprocessing errors."""
    pass


class TextPreprocessor:
    """
    Text preprocessing utilities with advanced features and gold standard normalization.

    This class provides comprehensive text preprocessing for NLP tasks including:
    - Gold standard normalization (Unicode, HTML, contractions, etc.)
    - Stopword removal (including domain-specific stopwords for survey data)
    - Lemmatization
    - Domain-specific cleaning
    - Long document handling
    - Language detection

    Example:
        >>> preprocessor = TextPreprocessor()
        >>> text = "I loooove this product!!! It's AMAZING"
        >>> clean_text = preprocessor.preprocess(text)
        >>> print(clean_text)
        'love product amazing'
    """

    @staticmethod
    def _load_domain_stopwords(
        file_path: Optional[Path] = None
    ) -> Set[str]:
        """
        Load domain-specific stopwords from a file.

        Args:
            file_path: Path to the domain stopwords file. If None, uses the default
                path (data/stopwords_domain.txt relative to the project root).

        Returns:
            Set of domain stopwords (lowercase). Returns empty set if file not found.
        """
        if file_path is None:
            file_path = DEFAULT_DOMAIN_STOPWORDS_PATH

        stopwords = set()
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word and not word.startswith('#'):  # Skip empty lines and comments
                            stopwords.add(word)
                logger.debug(f"Loaded {len(stopwords)} domain stopwords from {file_path}")
            else:
                logger.warning(
                    f"Domain stopwords file not found at {file_path}. "
                    "Continuing without domain stopwords."
                )
        except Exception as e:
            logger.warning(
                f"Error loading domain stopwords from {file_path}: {e}. "
                "Continuing without domain stopwords."
            )
        return stopwords

    def __init__(
        self,
        use_gold_standard: bool = True,
        normalize_unicode: bool = True,
        decode_html_entities: bool = True,
        expand_contractions: bool = True,
        normalize_elongations: bool = True,
        normalize_punctuation: bool = True,
        standardize_urls: bool = True,
        standardize_mentions: bool = True,
        process_hashtags: bool = True,
        expand_slang: bool = False,
        detect_spam: bool = False,
        detect_duplicates: bool = False,
        min_tokens: int = 3,
        max_tokens: int = 512,
        max_emoji_ratio: float = 0.7,
        max_char_repeat: int = 2,
        url_token: str = "<URL>",
        user_token: str = "<USER>",
        preserve_negations: bool = True,
        use_domain_stopwords: bool = True,
        domain_stopwords_path: Optional[Path] = None,
    ):
        """
        Initialize preprocessor with gold standard preprocessing support.

        Args:
            use_gold_standard: Enable gold standard preprocessing pipeline
            normalize_unicode: Apply NFKC Unicode normalization
            decode_html_entities: Decode HTML entities
            expand_contractions: Expand contractions (don't -> do not)
            normalize_elongations: Normalize character elongations
            normalize_punctuation: Normalize repeated punctuation
            standardize_urls: Replace URLs with token
            standardize_mentions: Replace @mentions with token
            process_hashtags: Remove # from hashtags
            expand_slang: Expand common slang terms
            detect_spam: Enable spam pattern detection
            detect_duplicates: Enable duplicate detection
            min_tokens: Minimum token threshold
            max_tokens: Maximum token threshold
            max_emoji_ratio: Maximum emoji-to-character ratio
            max_char_repeat: Maximum character repetitions
            url_token: Token to replace URLs
            user_token: Token to replace mentions
            preserve_negations: Preserve negation words (not, never, etc.) during
                stopword removal for better sentiment/topic analysis (default: True)
            use_domain_stopwords: Enable domain-specific stopwords for survey data
                (default: True). When enabled, loads additional stopwords from the
                domain stopwords file (e.g., 'response', 'survey', 'feedback').
            domain_stopwords_path: Custom path to domain stopwords file. If None,
                uses the default path (data/stopwords_domain.txt).
        """
        self.use_gold_standard = use_gold_standard
        self.preserve_negations = preserve_negations
        self.use_domain_stopwords = use_domain_stopwords

        self.gold_standard_config = {
            "normalize_unicode": normalize_unicode,
            "decode_html_entities": decode_html_entities,
            "expand_contractions": expand_contractions,
            "normalize_elongations": normalize_elongations,
            "normalize_punctuation": normalize_punctuation,
            "standardize_urls": standardize_urls,
            "standardize_mentions": standardize_mentions,
            "process_hashtags": process_hashtags,
            "expand_slang": expand_slang,
            "detect_spam": detect_spam,
            "detect_duplicates": detect_duplicates,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "max_emoji_ratio": max_emoji_ratio,
        }

        if use_gold_standard:
            self.gold_processor = GoldStandardTextProcessor(
                normalize_unicode=normalize_unicode,
                decode_html_entities=decode_html_entities,
                expand_contractions=expand_contractions,
                normalize_elongations=normalize_elongations,
                max_char_repeat=max_char_repeat,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                max_emoji_ratio=max_emoji_ratio,
                detect_spam=detect_spam,
                detect_duplicates=detect_duplicates,
                standardize_urls=standardize_urls,
                standardize_mentions=standardize_mentions,
                process_hashtags=process_hashtags,
                normalize_punctuation=normalize_punctuation,
                expand_slang=expand_slang,
                url_token=url_token,
                user_token=user_token,
            )
        else:
            self.gold_processor = None

        # Initialize NLTK resources
        _download_nltk_resources()

        try:
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.warning(f"Could not load NLTK resources: {e}")
            self.stop_words = set()
            self.lemmatizer = None

        # Load and merge domain stopwords if enabled
        self.domain_stopwords: Set[str] = set()
        if use_domain_stopwords:
            self.domain_stopwords = self._load_domain_stopwords(domain_stopwords_path)
            if self.domain_stopwords:
                self.stop_words = self.stop_words.union(self.domain_stopwords)
                logger.debug(
                    f"Merged {len(self.domain_stopwords)} domain stopwords. "
                    f"Total stopwords: {len(self.stop_words)}"
                )

    def get_quality_metrics(self) -> Optional[DataQualityMetrics]:
        """Get quality metrics from the gold standard processor."""
        if self.gold_processor:
            return self.gold_processor.get_metrics()
        return None

    def get_quality_report(self) -> str:
        """Generate a human-readable quality report."""
        if self.gold_processor:
            return self.gold_processor.get_report()
        return "Gold standard preprocessing not enabled."

    def reset_metrics(self) -> None:
        """Reset quality metrics for a new batch."""
        if self.gold_processor:
            self.gold_processor.reset_metrics()

    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            Detected language code (e.g., 'en', 'es', 'fr')

        Raises:
            TextPreprocessingError: If language detection fails
        """
        try:
            if not text or not text.strip():
                raise TextPreprocessingError("Cannot detect language of empty text")

            try:
                from langdetect import detect
                language = detect(text)
                logger.info(f"Detected language: {language}")
                return language
            except ImportError:
                logger.warning("langdetect not installed, defaulting to 'en'")
                return 'en'

        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            raise TextPreprocessingError(f"Language detection failed: {e}")

    def handle_long_documents(
        self,
        text: str,
        strategy: str = 'truncate',
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Union[str, List[str]]:
        """
        Handle long documents using different strategies.

        Args:
            text: Input text
            strategy: 'truncate', 'chunk', or 'summarize'
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            Processed text or list of chunks

        Raises:
            TextPreprocessingError: If processing fails
        """
        try:
            if not text or not text.strip():
                raise TextPreprocessingError("Cannot process empty text")

            if strategy not in ['truncate', 'chunk', 'summarize']:
                raise TextPreprocessingError(
                    f"Invalid strategy: {strategy}. Must be 'truncate', 'chunk', or 'summarize'"
                )

            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text)
            except Exception:
                tokens = text.split()

            logger.info(f"Document has {len(tokens)} tokens, using strategy: {strategy}")

            if strategy == 'truncate':
                truncated_tokens = tokens[:chunk_size]
                result = ' '.join(truncated_tokens)
                logger.debug(f"Truncated document to {len(truncated_tokens)} tokens")
                return result

            elif strategy == 'chunk':
                if chunk_size <= 0:
                    raise TextPreprocessingError("chunk_size must be positive")
                if chunk_overlap < 0 or chunk_overlap >= chunk_size:
                    raise TextPreprocessingError(
                        f"chunk_overlap must be >= 0 and < chunk_size ({chunk_size})"
                    )

                chunks = []
                start = 0

                while start < len(tokens):
                    end = start + chunk_size
                    chunk_tokens = tokens[start:end]
                    chunks.append(' '.join(chunk_tokens))
                    start += (chunk_size - chunk_overlap)

                logger.info(f"Split document into {len(chunks)} chunks")
                return chunks

            elif strategy == 'summarize':
                logger.warning("Summarize strategy not yet implemented, using truncate")
                truncated_tokens = tokens[:chunk_size]
                return ' '.join(truncated_tokens)

        except Exception as e:
            logger.error(f"Error handling long document: {e}")
            if isinstance(e, TextPreprocessingError):
                raise
            raise TextPreprocessingError(f"Failed to process long document: {e}")

    def clean_domain_specific(
        self,
        text: str,
        domain: str,
        preserve_terms: Optional[List[str]] = None
    ) -> str:
        """
        Clean text with domain-specific rules.

        Args:
            text: Input text
            domain: 'medical', 'legal', 'technical', or 'general'
            preserve_terms: Additional terms to preserve

        Returns:
            Cleaned text

        Raises:
            TextPreprocessingError: If cleaning fails
        """
        try:
            if not text or not text.strip():
                raise TextPreprocessingError("Cannot clean empty text")

            if domain not in ['medical', 'legal', 'technical', 'general']:
                raise TextPreprocessingError(
                    f"Invalid domain: {domain}. Must be 'medical', 'legal', 'technical', or 'general'"
                )

            domain_terms = {
                'medical': ['mg', 'ml', 'mcg', 'patient', 'diagnosis', 'treatment'],
                'legal': ['plaintiff', 'defendant', 'pursuant', 'hereby', 'whereas'],
                'technical': ['api', 'sdk', 'cli', 'gpu', 'cpu', 'ram']
            }

            terms_to_preserve = set()
            if domain in domain_terms:
                terms_to_preserve.update(domain_terms[domain])
            if preserve_terms:
                terms_to_preserve.update([t.lower() for t in preserve_terms])

            cleaned_text = text.lower()

            # Replace terms with placeholders
            term_placeholders = {}
            for i, term in enumerate(terms_to_preserve):
                placeholder = f"__TERM{i}__"
                term_placeholders[placeholder] = term
                cleaned_text = re.sub(
                    rf'\b{re.escape(term)}\b',
                    placeholder,
                    cleaned_text,
                    flags=re.IGNORECASE
                )

            # Remove non-alphabetic characters (except placeholders)
            cleaned_text = re.sub(r'[^a-zA-Z\s_]', '', cleaned_text)

            # Restore preserved terms
            for placeholder, term in term_placeholders.items():
                cleaned_text = cleaned_text.replace(placeholder, term)

            logger.debug(f"Domain-specific cleaning applied for domain: {domain}")
            return cleaned_text

        except Exception as e:
            logger.error(f"Error in domain-specific cleaning: {e}")
            if isinstance(e, TextPreprocessingError):
                raise
            raise TextPreprocessingError(f"Domain-specific cleaning failed: {e}")

    def preprocess(
        self,
        text: str,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        lowercase: bool = True,
        min_token_length: int = 2,
        track_metrics: bool = True
    ) -> str:
        """
        Preprocess text for NLP analysis with gold standard normalization.

        Args:
            text: Input text
            remove_stopwords: Remove common stopwords
            lemmatize: Apply lemmatization
            lowercase: Convert to lowercase
            min_token_length: Minimum token length to retain
            track_metrics: Track quality metrics

        Returns:
            Preprocessed text

        Raises:
            TextPreprocessingError: If preprocessing fails
        """
        try:
            if text is None:
                raise TextPreprocessingError("Text cannot be None")

            if not text.strip():
                logger.warning("Empty text provided to preprocess")
                return ""

            logger.debug(f"Preprocessing text of length {len(text)}")

            # Apply gold standard normalization first
            if self.use_gold_standard and self.gold_processor:
                text, _ = self.gold_processor.normalize(text)
                if track_metrics:
                    self.gold_processor.metrics.total_records += 1
                    token_count = self.gold_processor.count_tokens(text)
                    emoji_count = self.gold_processor.count_emojis(text)
                    self.gold_processor.metrics.record_text_stats(
                        text, token_count, emoji_count
                    )

            if lowercase:
                text = text.lower()

            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)

            # Tokenize
            try:
                from nltk.tokenize import word_tokenize
                tokens = word_tokenize(text)
            except Exception:
                tokens = text.split()

            # Remove stopwords
            if remove_stopwords and self.stop_words:
                if self.preserve_negations:
                    # Keep negation words even if they appear in stopwords
                    tokens = [t for t in tokens if t not in self.stop_words or t in NEGATION_KEEP_WORDS]
                else:
                    tokens = [t for t in tokens if t not in self.stop_words]

            # Lemmatize
            if lemmatize and self.lemmatizer:
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

            # Filter by token length
            tokens = [t for t in tokens if len(t) > min_token_length]

            result = ' '.join(tokens)
            logger.debug(f"Preprocessing complete: {len(tokens)} tokens retained")
            return result

        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            if isinstance(e, TextPreprocessingError):
                raise
            raise TextPreprocessingError(f"Preprocessing failed: {e}")

    def preprocess_batch(
        self,
        texts: List[str],
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        lowercase: bool = True,
        min_token_length: int = 2,
        track_metrics: bool = True,
        return_filtered: bool = False,
    ) -> List[str]:
        """
        Preprocess a batch of texts.

        Args:
            texts: List of input texts
            remove_stopwords: Remove common stopwords
            lemmatize: Apply lemmatization
            lowercase: Convert to lowercase
            min_token_length: Minimum token length
            track_metrics: Track quality metrics
            return_filtered: Include empty results

        Returns:
            List of preprocessed texts
        """
        results = []
        for text in texts:
            try:
                result = self.preprocess(
                    text,
                    remove_stopwords=remove_stopwords,
                    lemmatize=lemmatize,
                    lowercase=lowercase,
                    min_token_length=min_token_length,
                    track_metrics=track_metrics
                )
                if return_filtered or result:
                    results.append(result)
            except TextPreprocessingError:
                if return_filtered:
                    results.append("")
        return results

    def normalize_only(
        self,
        text: str,
        track_metrics: bool = False
    ) -> str:
        """
        Apply only gold standard normalization without tokenization/lemmatization.

        Useful when you want to clean text but preserve its original structure.

        Args:
            text: Input text
            track_metrics: Track quality metrics

        Returns:
            Normalized text
        """
        if self.use_gold_standard and self.gold_processor:
            result = self.gold_processor.process(text, track_metrics=track_metrics)
            return result if result else ""
        return normalize_for_nlp(text, preserve_case=True)


class DataCleaningPipeline:
    """
    High-level pipeline for cleaning and preprocessing data.

    Combines data loading, quality assessment, and preprocessing
    into a single workflow.

    Example:
        >>> pipeline = DataCleaningPipeline()
        >>> df = pipeline.clean_dataframe(df, text_column='response')
        >>> print(pipeline.get_summary())
    """

    def __init__(
        self,
        dataset_type: str = "survey",
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        lowercase: bool = True,
        preserve_negations: bool = True,
        use_domain_stopwords: bool = True,
    ):
        """
        Initialize the data cleaning pipeline.

        Args:
            dataset_type: 'survey', 'social_media', 'reviews', or 'news'
            remove_stopwords: Remove stopwords during preprocessing
            lemmatize: Apply lemmatization
            lowercase: Convert to lowercase
            preserve_negations: Preserve negation words during stopword removal
                for better sentiment/topic analysis (default: True)
            use_domain_stopwords: Enable domain-specific stopwords for survey data
                (default: True). Removes common survey terms like 'response',
                'survey', 'feedback', etc.
        """
        self.dataset_type = dataset_type
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.preserve_negations = preserve_negations
        self.use_domain_stopwords = use_domain_stopwords

        # Configure preprocessor based on dataset type
        config = self._get_config_for_dataset(dataset_type)
        config["preserve_negations"] = preserve_negations
        config["use_domain_stopwords"] = use_domain_stopwords
        self.preprocessor = TextPreprocessor(**config)

        self._summary = {}

    def _get_config_for_dataset(self, dataset_type: str) -> dict:
        """Get preprocessor configuration for dataset type."""
        # Base configuration for survey/general text
        survey_config = {
            "use_gold_standard": True,
            "expand_slang": False,
            "detect_spam": True,
            "min_tokens": 3,
        }
        configs = {
            "survey": survey_config,
            "general": survey_config,  # Alias for backward compatibility
            "social_media": {
                "use_gold_standard": True,
                "expand_slang": True,
                "standardize_urls": True,
                "standardize_mentions": True,
                "process_hashtags": True,
                "max_emoji_ratio": 0.5,
                "min_tokens": 2,
            },
            "reviews": {
                "use_gold_standard": True,
                "expand_contractions": True,
                "normalize_elongations": True,
                "detect_spam": True,
                "min_tokens": 5,
                "max_tokens": 1000,
            },
            "news": {
                "use_gold_standard": True,
                "expand_contractions": False,
                "normalize_elongations": False,
                "detect_spam": False,
                "min_tokens": 10,
                "max_tokens": 2000,
            },
        }
        return configs.get(dataset_type, configs["survey"])

    def clean_dataframe(
        self,
        df,
        text_column: str,
        output_column: Optional[str] = None,
        drop_filtered: bool = False,
    ):
        """
        Clean a DataFrame's text column.

        Args:
            df: Input DataFrame
            text_column: Name of column containing text
            output_column: Name for processed column (default: text_column + '_cleaned')
            drop_filtered: Remove rows with empty processed text

        Returns:
            DataFrame with cleaned text column added
        """
        import pandas as pd

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        if output_column is None:
            output_column = f"{text_column}_cleaned"

        # Reset metrics for fresh tracking
        self.preprocessor.reset_metrics()

        # Process each text
        processed = []
        for text in df[text_column]:
            try:
                result = self.preprocessor.preprocess(
                    str(text) if pd.notna(text) else "",
                    remove_stopwords=self.remove_stopwords,
                    lemmatize=self.lemmatize,
                    lowercase=self.lowercase,
                    track_metrics=True
                )
                processed.append(result if result else None)
            except TextPreprocessingError:
                processed.append(None)

        df = df.copy()
        df[output_column] = processed

        if drop_filtered:
            df = df.dropna(subset=[output_column])
            df = df[df[output_column].str.strip() != '']

        # Store summary
        metrics = self.preprocessor.get_quality_metrics()
        if metrics:
            self._summary = {
                "total_records": metrics.total_records,
                "valid_records": metrics.valid_records,
                "filtered_records": metrics.filtered_records,
                "valid_ratio": metrics.valid_ratio,
                "avg_text_length": metrics.avg_text_length,
                "avg_token_count": metrics.avg_token_count,
            }

        return df

    def clean_texts(
        self,
        texts: List[str],
    ) -> List[str]:
        """
        Clean a list of texts.

        Args:
            texts: List of input texts

        Returns:
            List of cleaned texts
        """
        self.preprocessor.reset_metrics()

        return self.preprocessor.preprocess_batch(
            texts,
            remove_stopwords=self.remove_stopwords,
            lemmatize=self.lemmatize,
            lowercase=self.lowercase,
            track_metrics=True,
            return_filtered=True
        )

    def get_summary(self) -> dict:
        """Get summary of last cleaning operation."""
        return self._summary

    def get_quality_report(self) -> str:
        """Get detailed quality report."""
        return self.preprocessor.get_quality_report()
