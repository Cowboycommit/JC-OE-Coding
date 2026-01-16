"""
VectorizerFactory for ensuring vectorization parity across clustering methods.

This module provides a factory that ensures identical tokenisation and filtering
across all clustering methods (KMeans, LDA, LSTM, BERT), with the ONLY difference being
term weighting:
- TF-IDF + KMeans: TF-IDF weighted matrix
- LDA: CountVectorizer with identical settings (counts instead of TF-IDF weights)
- LSTM + KMeans: LSTM embeddings (via embeddings.py), TF-IDF for term extraction
- BERT + KMeans: BERT embeddings (via embeddings.py), TF-IDF for term extraction

Key principles:
- All methods share the same vocabulary, stopwords, min_df/max_df, ngram_range for term extraction
- Preprocessing outputs are reused across methods where possible
- The only difference for LDA is term weighting (counts vs TF-IDF)
- LSTM and BERT use their own embedding methods for clustering
- Factory integrates with existing adaptive preprocessing

Classes:
    VectorizerFactory: Creates consistent vectorizers across methods
    VectorizerConfig: Configuration for vectorizer creation

Usage:
    >>> factory = VectorizerFactory()
    >>> vectorizer, matrix = factory.create_for_method('tfidf_kmeans', texts, config)
    >>> # For LDA, uses CountVectorizer with same settings
    >>> vectorizer, matrix = factory.create_for_method('lda', texts, config)
    >>> # For LSTM/BERT, returns TF-IDF for term extraction (embeddings handled separately)
    >>> vectorizer, matrix = factory.create_for_method('lstm_kmeans', texts, config)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

logger = logging.getLogger(__name__)


@dataclass
class VectorizerConfig:
    """
    Configuration for vectorizer creation.

    All settings are shared across TF-IDF and CountVectorizer to ensure
    identical tokenisation and vocabulary.

    Attributes:
        max_features: Maximum vocabulary size
        min_df: Minimum document frequency (int for count, float for ratio)
        max_df: Maximum document frequency (float ratio)
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction
        stop_words: Stop words to remove ('english', list, or None)
        sublinear_tf: Use sublinear TF scaling (TF-IDF only)
        lowercase: Convert all text to lowercase
        strip_accents: Remove accents from characters ('ascii', 'unicode', None)
        token_pattern: Regex pattern for tokenisation
    """
    max_features: int = 1000
    min_df: Union[int, float] = 2
    max_df: float = 0.8
    ngram_range: Tuple[int, int] = (1, 2)
    stop_words: Optional[Union[str, List[str]]] = 'english'
    sublinear_tf: bool = False
    lowercase: bool = True
    strip_accents: Optional[str] = None
    token_pattern: str = r'(?u)\b\w\w+\b'

    def to_base_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs shared by both TfidfVectorizer and CountVectorizer.

        Returns:
            Dictionary of kwargs for vectorizer construction
        """
        return {
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range,
            'stop_words': self.stop_words,
            'lowercase': self.lowercase,
            'strip_accents': self.strip_accents,
            'token_pattern': self.token_pattern
        }

    def to_tfidf_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs for TfidfVectorizer.

        Returns:
            Dictionary of kwargs for TfidfVectorizer
        """
        kwargs = self.to_base_kwargs()
        kwargs['sublinear_tf'] = self.sublinear_tf
        return kwargs

    def to_count_kwargs(self) -> Dict[str, Any]:
        """
        Convert to kwargs for CountVectorizer.

        Returns:
            Dictionary of kwargs for CountVectorizer (no TF-IDF specific params)
        """
        return self.to_base_kwargs()

    @classmethod
    def from_preprocessing_config(cls, config: Any) -> 'VectorizerConfig':
        """
        Create VectorizerConfig from PreprocessingConfig.

        This method bridges the adaptive preprocessing module with the vectorizer factory.

        Args:
            config: PreprocessingConfig from dataset_preprocessing module

        Returns:
            VectorizerConfig with equivalent settings
        """
        return cls(
            max_features=config.max_features,
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range=config.ngram_range,
            stop_words=config.stopwords_language if config.use_stopwords else None,
            sublinear_tf=config.sublinear_tf,
            lowercase=True,
            strip_accents=None
        )


class VectorizerFactory:
    """
    Factory for creating consistent vectorizers across clustering methods.

    This factory ensures that:
    1. All methods use identical tokenisation and filtering
    2. TF-IDF matrix is created for tfidf_kmeans, lstm_kmeans, bert_kmeans (for term extraction)
    3. LDA uses CountVectorizer with identical vocabulary settings
    4. The ONLY difference for LDA is term weighting (TF-IDF vs counts)

    Example:
        >>> factory = VectorizerFactory()
        >>> config = VectorizerConfig(max_features=1000, min_df=2)
        >>>
        >>> # TF-IDF based methods
        >>> tfidf_vec, tfidf_matrix = factory.create_for_method('tfidf_kmeans', texts, config)
        >>>
        >>> # LDA uses CountVectorizer with same settings
        >>> count_vec, count_matrix = factory.create_for_method('lda', texts, config)
        >>>
        >>> # LSTM/BERT use TF-IDF for term extraction (embeddings handled separately)
        >>> term_vec, term_matrix = factory.create_for_method('lstm_kmeans', texts, config)
    """

    def __init__(self):
        """Initialize the vectorizer factory."""
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._count_vectorizer = None
        self._count_matrix = None
        self._vocabulary = None
        self._config = None
        self._texts_hash = None

    def create_for_method(
        self,
        method: str,
        texts: List[str],
        config: Optional[VectorizerConfig] = None,
        force_refit: bool = False
    ) -> Tuple[Any, Any]:
        """
        Create vectorizer and matrix for the specified method.

        Args:
            method: Clustering method ('tfidf_kmeans', 'lda', 'lstm_kmeans', 'bert_kmeans', 'svm')
            texts: List of preprocessed text documents
            config: Optional vectorizer configuration (uses defaults if None)
            force_refit: If True, force refit even if cached

        Returns:
            Tuple of (vectorizer, feature_matrix)

        Raises:
            ValueError: If method is not supported

        Notes:
            - TF-IDF matrix is used for tfidf_kmeans and svm
            - Count matrix is created with identical settings for lda
            - LSTM and BERT methods use their respective embedders
            - Vocabulary is shared across TF-IDF methods
        """
        if config is None:
            config = VectorizerConfig()

        # Check if we need to refit
        texts_hash = hash(tuple(texts[:100]))  # Hash first 100 for efficiency
        needs_refit = (
            force_refit or
            self._config is None or
            self._texts_hash != texts_hash or
            self._config != config
        )

        if needs_refit:
            self._config = config
            self._texts_hash = texts_hash
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            self._count_vectorizer = None
            self._count_matrix = None
            self._vocabulary = None

        method = method.lower().strip()

        if method == 'tfidf_kmeans':
            return self._get_tfidf_vectorizer_and_matrix(texts, config)
        elif method == 'lda':
            return self._get_count_vectorizer_and_matrix(texts, config)
        elif method in ['lstm_kmeans', 'bert_kmeans']:
            # LSTM and BERT methods handle their own embeddings via embeddings.py
            # Return TF-IDF for term extraction (used in cluster interpretation)
            return self._get_tfidf_vectorizer_and_matrix(texts, config)
        elif method == 'svm':
            # SVM Spectral Clustering uses TF-IDF features
            return self._get_tfidf_vectorizer_and_matrix(texts, config)
        else:
            raise ValueError(
                f"Unsupported method: '{method}'. "
                f"Supported methods: 'tfidf_kmeans', 'lda', 'lstm_kmeans', 'bert_kmeans', 'svm'"
            )

    def _get_tfidf_vectorizer_and_matrix(
        self,
        texts: List[str],
        config: VectorizerConfig
    ) -> Tuple[TfidfVectorizer, csr_matrix]:
        """
        Get or create TF-IDF vectorizer and matrix.

        The matrix is cached for reuse between tfidf_kmeans, lstm_kmeans, and bert_kmeans methods.

        Args:
            texts: List of preprocessed text documents
            config: Vectorizer configuration

        Returns:
            Tuple of (TfidfVectorizer, sparse matrix)
        """
        if self._tfidf_vectorizer is not None and self._tfidf_matrix is not None:
            logger.debug("Reusing cached TF-IDF matrix")
            return self._tfidf_vectorizer, self._tfidf_matrix

        logger.info("Creating TF-IDF vectorizer and matrix")

        self._tfidf_vectorizer = TfidfVectorizer(**config.to_tfidf_kwargs())

        try:
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                raise ValueError(
                    "Empty vocabulary error: No valid terms found after text preprocessing. "
                    "This can happen when:\n"
                    "  1. The dataset is too small or contains only short responses\n"
                    "  2. All words are filtered by stop words\n"
                    "  3. The min_df/max_df thresholds are too restrictive\n"
                    "  4. Responses contain mostly non-text content\n\n"
                    "Suggestions:\n"
                    "  - Provide more diverse text responses\n"
                    "  - Try different stop words settings\n"
                    "  - Reduce min_df or increase max_df"
                ) from e
            raise

        # Store vocabulary for count vectorizer to reuse
        self._vocabulary = self._tfidf_vectorizer.vocabulary_

        logger.info(
            f"TF-IDF matrix created: {self._tfidf_matrix.shape[0]} docs x "
            f"{self._tfidf_matrix.shape[1]} features"
        )

        return self._tfidf_vectorizer, self._tfidf_matrix

    def _get_count_vectorizer_and_matrix(
        self,
        texts: List[str],
        config: VectorizerConfig
    ) -> Tuple[CountVectorizer, csr_matrix]:
        """
        Get or create CountVectorizer and matrix for LDA.

        Uses the same vocabulary as TF-IDF if available, ensuring
        identical tokenisation across methods.

        Args:
            texts: List of preprocessed text documents
            config: Vectorizer configuration

        Returns:
            Tuple of (CountVectorizer, sparse matrix)
        """
        if self._count_vectorizer is not None and self._count_matrix is not None:
            logger.debug("Reusing cached count matrix")
            return self._count_vectorizer, self._count_matrix

        logger.info("Creating CountVectorizer and matrix for LDA")

        count_kwargs = config.to_count_kwargs()

        # If we have a vocabulary from TF-IDF, use it to ensure identical tokenisation
        if self._vocabulary is not None:
            logger.debug("Reusing vocabulary from TF-IDF vectorizer")
            # When using vocabulary, we don't need max_features/min_df/max_df
            # as the vocabulary is already filtered
            count_kwargs_with_vocab = {
                'vocabulary': self._vocabulary,
                'lowercase': count_kwargs.get('lowercase', True),
                'strip_accents': count_kwargs.get('strip_accents', None),
                'token_pattern': count_kwargs.get('token_pattern', r'(?u)\b\w\w+\b'),
                'ngram_range': count_kwargs.get('ngram_range', (1, 2))
            }
            # Stop words are already applied in the vocabulary, but we keep
            # the setting for consistency in case of new terms
            if count_kwargs.get('stop_words'):
                count_kwargs_with_vocab['stop_words'] = count_kwargs['stop_words']

            self._count_vectorizer = CountVectorizer(**count_kwargs_with_vocab)
        else:
            # No vocabulary from TF-IDF, create fresh with same settings
            self._count_vectorizer = CountVectorizer(**count_kwargs)

        try:
            self._count_matrix = self._count_vectorizer.fit_transform(texts)
        except ValueError as e:
            if "empty vocabulary" in str(e).lower():
                raise ValueError(
                    "Empty vocabulary error for LDA: No valid terms found. "
                    "The dataset may be too small or overly filtered."
                ) from e
            raise

        # Store vocabulary if not already set
        if self._vocabulary is None:
            self._vocabulary = self._count_vectorizer.vocabulary_

        logger.info(
            f"Count matrix created: {self._count_matrix.shape[0]} docs x "
            f"{self._count_matrix.shape[1]} features"
        )

        return self._count_vectorizer, self._count_matrix

    def get_vocabulary(self) -> Optional[Dict[str, int]]:
        """
        Get the shared vocabulary.

        Returns:
            Vocabulary dictionary mapping terms to indices, or None if not fitted
        """
        return self._vocabulary

    def get_feature_names(self) -> Optional[np.ndarray]:
        """
        Get feature names from the fitted vectorizer.

        Returns:
            Array of feature names, or None if not fitted
        """
        if self._tfidf_vectorizer is not None:
            return self._tfidf_vectorizer.get_feature_names_out()
        elif self._count_vectorizer is not None:
            return self._count_vectorizer.get_feature_names_out()
        return None

    def clear_cache(self):
        """Clear all cached vectorizers and matrices."""
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None
        self._count_vectorizer = None
        self._count_matrix = None
        self._vocabulary = None
        self._config = None
        self._texts_hash = None
        logger.debug("Vectorizer factory cache cleared")

    def get_parity_report(self) -> Dict[str, Any]:
        """
        Generate a report showing vectorization parity across methods.

        Returns:
            Dictionary with parity information
        """
        report = {
            'tfidf_fitted': self._tfidf_vectorizer is not None,
            'count_fitted': self._count_vectorizer is not None,
            'vocabulary_shared': self._vocabulary is not None,
            'vocabulary_size': len(self._vocabulary) if self._vocabulary else 0
        }

        if self._tfidf_matrix is not None:
            report['tfidf_matrix_shape'] = self._tfidf_matrix.shape
            report['tfidf_nnz'] = self._tfidf_matrix.nnz

        if self._count_matrix is not None:
            report['count_matrix_shape'] = self._count_matrix.shape
            report['count_nnz'] = self._count_matrix.nnz

        # Check vocabulary consistency
        if self._tfidf_vectorizer is not None and self._count_vectorizer is not None:
            tfidf_vocab = set(self._tfidf_vectorizer.vocabulary_.keys())
            count_vocab = set(self._count_vectorizer.vocabulary_.keys())
            report['vocabulary_identical'] = tfidf_vocab == count_vocab
            report['vocab_intersection'] = len(tfidf_vocab & count_vocab)
            report['vocab_union'] = len(tfidf_vocab | count_vocab)

        return report


def create_vectorizer_for_method(
    method: str,
    texts: List[str],
    config: Optional[VectorizerConfig] = None,
    preprocessing_config: Any = None
) -> Tuple[Any, Any]:
    """
    Convenience function to create vectorizer for a specific method.

    This is a stateless alternative to the VectorizerFactory class.
    For batch processing multiple methods on the same data, use VectorizerFactory
    to benefit from caching.

    Args:
        method: Clustering method ('tfidf_kmeans', 'lda', 'lstm_kmeans', 'bert_kmeans', 'svm')
        texts: List of preprocessed text documents
        config: Optional VectorizerConfig (uses defaults if None)
        preprocessing_config: Optional PreprocessingConfig from adaptive preprocessing

    Returns:
        Tuple of (vectorizer, feature_matrix)

    Example:
        >>> vectorizer, matrix = create_vectorizer_for_method('lda', texts)
    """
    if config is None and preprocessing_config is not None:
        config = VectorizerConfig.from_preprocessing_config(preprocessing_config)

    factory = VectorizerFactory()
    return factory.create_for_method(method, texts, config)
