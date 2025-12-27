"""
Dataset-aware preprocessing for TF-IDF + KMeans clustering.

This module provides adaptive preprocessing that correctly handles different dataset types
(e.g., Reuters-21578 style multi-label documents, simple survey responses, 20 Newsgroups)
without degrading performance on simpler OE datasets.

Key principles:
- Adaptive preprocessing based on detected dataset characteristics
- No hard-coded dataset-specific logic
- Backward compatibility with existing simple OE workflows
- Clear separation between clustering (unsupervised) and evaluation (optional labels)

Classes:
    DatasetCharacteristicsDetector: Analyzes dataset to detect characteristics
    AdaptivePreprocessor: Applies conditional preprocessing based on dataset type
    MultiLabelHandler: Handles multi-label datasets without forcing single labels

Usage:
    # Detect characteristics
    detector = DatasetCharacteristicsDetector()
    characteristics = detector.analyze(texts, labels=None)

    # Get adaptive preprocessing config
    preprocessor = AdaptivePreprocessor()
    config = preprocessor.get_config(characteristics)

    # Create vectorizer with adaptive config
    vectorizer = preprocessor.create_vectorizer(config)
"""

import re
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetCharacteristics:
    """
    Detected characteristics of a text dataset.

    All characteristics are detected automatically without hard-coded
    dataset name checks. These characteristics drive adaptive preprocessing.

    Attributes:
        n_documents: Total number of documents
        median_doc_length: Median document length in words
        mean_doc_length: Mean document length in words
        std_doc_length: Standard deviation of document length
        doc_length_percentiles: Dictionary of percentile values (25, 50, 75, 90, 95)
        vocabulary_size: Estimated unique vocabulary size
        is_long_form: Whether documents are long-form (news articles, etc.)
        is_short_form: Whether documents are short-form (survey responses, tweets)
        has_labels: Whether label column is present
        is_multi_label: Whether labels contain multi-label indicators
        label_missing_ratio: Ratio of missing/unknown labels
        label_delimiter: Detected delimiter for multi-labels (if any)
        suggested_preprocessing: String describing suggested preprocessing level
        corpus_size_category: 'small', 'medium', or 'large' based on n_documents
    """
    n_documents: int = 0
    median_doc_length: float = 0.0
    mean_doc_length: float = 0.0
    std_doc_length: float = 0.0
    doc_length_percentiles: Dict[int, float] = field(default_factory=dict)
    vocabulary_size: int = 0
    is_long_form: bool = False
    is_short_form: bool = True
    has_labels: bool = False
    is_multi_label: bool = False
    label_missing_ratio: float = 0.0
    label_delimiter: Optional[str] = None
    suggested_preprocessing: str = 'standard'
    corpus_size_category: str = 'small'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            'n_documents': self.n_documents,
            'median_doc_length': round(self.median_doc_length, 2),
            'mean_doc_length': round(self.mean_doc_length, 2),
            'std_doc_length': round(self.std_doc_length, 2),
            'doc_length_percentiles': {k: round(v, 2) for k, v in self.doc_length_percentiles.items()},
            'vocabulary_size': self.vocabulary_size,
            'is_long_form': self.is_long_form,
            'is_short_form': self.is_short_form,
            'has_labels': self.has_labels,
            'is_multi_label': self.is_multi_label,
            'label_missing_ratio': round(self.label_missing_ratio, 3),
            'label_delimiter': self.label_delimiter,
            'suggested_preprocessing': self.suggested_preprocessing,
            'corpus_size_category': self.corpus_size_category
        }


@dataclass
class PreprocessingConfig:
    """
    Configuration for TF-IDF vectorization based on dataset characteristics.

    Attributes:
        max_features: Maximum number of features to extract
        min_df: Minimum document frequency (int for count, float for proportion)
        max_df: Maximum document frequency (float for proportion)
        ngram_range: Tuple of (min_n, max_n) for n-gram extraction
        sublinear_tf: Whether to apply sublinear TF scaling
        use_stopwords: Whether to remove stopwords
        stopwords_language: Language for stopwords if used
        lowercase: Whether to lowercase text
        strip_accents: Whether to strip accents from characters

        # Rationale for choices
        config_rationale: Human-readable explanation of config choices
    """
    max_features: int = 1000
    min_df: Union[int, float] = 2
    max_df: float = 0.8
    ngram_range: Tuple[int, int] = (1, 2)
    sublinear_tf: bool = False
    use_stopwords: bool = True
    stopwords_language: str = 'english'
    lowercase: bool = True
    strip_accents: Optional[str] = 'unicode'
    config_rationale: str = ''

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vectorizer kwargs."""
        config = {
            'max_features': self.max_features,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'ngram_range': self.ngram_range,
            'sublinear_tf': self.sublinear_tf,
            'lowercase': self.lowercase,
        }
        if self.use_stopwords:
            config['stop_words'] = self.stopwords_language
        if self.strip_accents:
            config['strip_accents'] = self.strip_accents
        return config


class DatasetCharacteristicsDetector:
    """
    Analyzes a text dataset to detect characteristics for adaptive preprocessing.

    This class examines document lengths, vocabulary, and label patterns to
    determine appropriate preprocessing strategies. It does NOT hard-code
    any dataset names or make assumptions about specific datasets.

    Thresholds are based on empirical observations across different text types:
    - Short OE responses: typically 5-50 words
    - Long-form articles: typically 100-500+ words
    - Multi-label indicators: comma, semicolon delimiters in label fields

    Example:
        >>> detector = DatasetCharacteristicsDetector()
        >>> characteristics = detector.analyze(texts, labels=labels_series)
        >>> print(characteristics.is_long_form)
        True
        >>> print(characteristics.suggested_preprocessing)
        'aggressive'
    """

    # Thresholds for classification (not hard-coded for specific datasets)
    LONG_FORM_MEDIAN_THRESHOLD = 100  # words
    SHORT_FORM_MEDIAN_THRESHOLD = 30  # words
    SMALL_CORPUS_THRESHOLD = 500      # documents
    MEDIUM_CORPUS_THRESHOLD = 5000    # documents
    MULTI_LABEL_DELIMITERS = [',', ';', '|']
    UNKNOWN_LABEL_PATTERNS = [
        r'^unknown$', r'^none$', r'^n/?a$', r'^\s*$', r'^unlabeled$',
        r'^unassigned$', r'^no\s*label$', r'^\?$'
    ]

    def __init__(
        self,
        long_form_threshold: int = 100,
        short_form_threshold: int = 30,
        small_corpus_threshold: int = 500,
        medium_corpus_threshold: int = 5000,
        high_missing_label_threshold: float = 0.3
    ):
        """
        Initialize the detector with configurable thresholds.

        Args:
            long_form_threshold: Median word count above which documents are long-form
            short_form_threshold: Median word count below which documents are short-form
            small_corpus_threshold: Document count below which corpus is 'small'
            medium_corpus_threshold: Document count above which corpus is 'large'
            high_missing_label_threshold: Ratio above which missing labels are significant
        """
        self.long_form_threshold = long_form_threshold
        self.short_form_threshold = short_form_threshold
        self.small_corpus_threshold = small_corpus_threshold
        self.medium_corpus_threshold = medium_corpus_threshold
        self.high_missing_label_threshold = high_missing_label_threshold

        # Compile unknown label patterns
        self.unknown_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.UNKNOWN_LABEL_PATTERNS
        ]

    def analyze(
        self,
        texts: List[str],
        labels: Optional[Union[List[str], pd.Series]] = None,
        sample_size: Optional[int] = None
    ) -> DatasetCharacteristics:
        """
        Analyze dataset to detect characteristics.

        Args:
            texts: List of document texts
            labels: Optional list/series of labels (for multi-label detection)
            sample_size: Optional sample size for large datasets (for efficiency)

        Returns:
            DatasetCharacteristics with detected properties

        Notes:
            - For large datasets (>10000 docs), analysis uses sampling for efficiency
            - Label analysis is optional and only affects evaluation mode
            - No labels are ever used for clustering/training
        """
        if not texts:
            logger.warning("Empty text list provided to analyze()")
            return DatasetCharacteristics()

        # Sample for efficiency on large datasets
        if sample_size and len(texts) > sample_size:
            np.random.seed(42)  # Reproducibility
            indices = np.random.choice(len(texts), sample_size, replace=False)
            sample_texts = [texts[i] for i in indices]
            sample_labels = [labels[i] for i in indices] if labels is not None else None
        else:
            sample_texts = texts
            sample_labels = labels

        characteristics = DatasetCharacteristics()
        characteristics.n_documents = len(texts)

        # Analyze document lengths
        self._analyze_document_lengths(sample_texts, characteristics)

        # Estimate vocabulary size
        self._estimate_vocabulary(sample_texts, characteristics)

        # Classify document type
        self._classify_document_type(characteristics)

        # Analyze labels if provided
        if sample_labels is not None:
            self._analyze_labels(sample_labels, characteristics)

        # Determine suggested preprocessing level
        self._determine_preprocessing_level(characteristics)

        # Determine corpus size category
        self._classify_corpus_size(characteristics)

        logger.info(f"Dataset analysis complete: {characteristics.to_dict()}")

        return characteristics

    def _analyze_document_lengths(
        self,
        texts: List[str],
        characteristics: DatasetCharacteristics
    ) -> None:
        """Analyze document length distribution."""
        # Count words per document
        word_counts = []
        for text in texts:
            if pd.isna(text) or not text:
                word_counts.append(0)
            else:
                # Simple word counting (split on whitespace)
                words = str(text).split()
                word_counts.append(len(words))

        word_counts = np.array(word_counts)

        characteristics.median_doc_length = float(np.median(word_counts))
        characteristics.mean_doc_length = float(np.mean(word_counts))
        characteristics.std_doc_length = float(np.std(word_counts))
        characteristics.doc_length_percentiles = {
            25: float(np.percentile(word_counts, 25)),
            50: float(np.percentile(word_counts, 50)),
            75: float(np.percentile(word_counts, 75)),
            90: float(np.percentile(word_counts, 90)),
            95: float(np.percentile(word_counts, 95))
        }

    def _estimate_vocabulary(
        self,
        texts: List[str],
        characteristics: DatasetCharacteristics
    ) -> None:
        """Estimate vocabulary size from sample."""
        all_words = set()
        for text in texts:
            if pd.isna(text) or not text:
                continue
            # Simple tokenization
            words = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
            all_words.update(words)

        characteristics.vocabulary_size = len(all_words)

    def _classify_document_type(
        self,
        characteristics: DatasetCharacteristics
    ) -> None:
        """Classify documents as long-form or short-form."""
        median = characteristics.median_doc_length

        if median >= self.long_form_threshold:
            characteristics.is_long_form = True
            characteristics.is_short_form = False
        elif median <= self.short_form_threshold:
            characteristics.is_long_form = False
            characteristics.is_short_form = True
        else:
            # Middle ground - neither clearly long nor short
            characteristics.is_long_form = False
            characteristics.is_short_form = False

    def _analyze_labels(
        self,
        labels: Union[List[str], pd.Series],
        characteristics: DatasetCharacteristics
    ) -> None:
        """Analyze label patterns for multi-label detection."""
        characteristics.has_labels = True

        # Convert to list if series
        if isinstance(labels, pd.Series):
            labels = labels.tolist()

        # Count missing/unknown labels
        missing_count = 0
        multi_label_indicators = Counter()

        for label in labels:
            # Check for missing
            if pd.isna(label) or label is None:
                missing_count += 1
                continue

            label_str = str(label).strip()

            # Check for unknown patterns
            if any(p.match(label_str) for p in self.unknown_patterns):
                missing_count += 1
                continue

            # Check for multi-label delimiters
            for delimiter in self.MULTI_LABEL_DELIMITERS:
                if delimiter in label_str:
                    multi_label_indicators[delimiter] += 1

        # Calculate missing ratio
        characteristics.label_missing_ratio = missing_count / len(labels) if labels else 0.0

        # Determine if multi-label (if >10% of labels have delimiters)
        if multi_label_indicators:
            most_common_delimiter = multi_label_indicators.most_common(1)[0]
            delimiter, count = most_common_delimiter
            ratio = count / (len(labels) - missing_count) if (len(labels) - missing_count) > 0 else 0

            if ratio > 0.1:  # >10% have this delimiter
                characteristics.is_multi_label = True
                characteristics.label_delimiter = delimiter

    def _determine_preprocessing_level(
        self,
        characteristics: DatasetCharacteristics
    ) -> None:
        """Determine suggested preprocessing level based on characteristics."""
        if characteristics.is_long_form:
            # Long-form documents (news articles, essays, etc.)
            # Need aggressive preprocessing to handle noise
            characteristics.suggested_preprocessing = 'aggressive'
        elif characteristics.is_short_form and characteristics.vocabulary_size < 500:
            # Very short responses with small vocabulary
            # Need minimal preprocessing to preserve signal
            characteristics.suggested_preprocessing = 'minimal'
        elif characteristics.is_short_form:
            # Short but diverse vocabulary
            characteristics.suggested_preprocessing = 'standard'
        else:
            # Middle ground
            characteristics.suggested_preprocessing = 'standard'

    def _classify_corpus_size(
        self,
        characteristics: DatasetCharacteristics
    ) -> None:
        """Classify corpus size category."""
        n = characteristics.n_documents

        if n < self.small_corpus_threshold:
            characteristics.corpus_size_category = 'small'
        elif n < self.medium_corpus_threshold:
            characteristics.corpus_size_category = 'medium'
        else:
            characteristics.corpus_size_category = 'large'


class AdaptivePreprocessor:
    """
    Generates adaptive TF-IDF preprocessing configuration based on dataset characteristics.

    This class takes detected dataset characteristics and produces an appropriate
    TF-IDF configuration. It ensures:
    - Long-form documents get stricter preprocessing (higher min_df, stopword removal)
    - Short-form responses get gentler preprocessing (lower min_df, preserved vocabulary)
    - Small corpora don't lose too much vocabulary
    - Bigrams are enabled only when beneficial

    The logic does NOT hard-code dataset names - it uses detected characteristics only.

    Example:
        >>> preprocessor = AdaptivePreprocessor()
        >>> config = preprocessor.get_config(characteristics)
        >>> vectorizer = TfidfVectorizer(**config.to_dict())
    """

    def __init__(
        self,
        default_max_features: int = 1000,
        default_stopwords: str = 'english'
    ):
        """
        Initialize adaptive preprocessor.

        Args:
            default_max_features: Default maximum features if not adapted
            default_stopwords: Default stopwords language
        """
        self.default_max_features = default_max_features
        self.default_stopwords = default_stopwords

    def get_config(
        self,
        characteristics: DatasetCharacteristics,
        override_config: Optional[Dict[str, Any]] = None
    ) -> PreprocessingConfig:
        """
        Generate preprocessing configuration based on dataset characteristics.

        Args:
            characteristics: Detected dataset characteristics
            override_config: Optional manual overrides for specific parameters

        Returns:
            PreprocessingConfig with adaptive settings

        Notes:
            The configuration logic follows these principles:

            For LONG-FORM documents (news, articles):
            - Higher min_df (10+) to remove rare noise terms
            - max_df ≤ 0.9 to remove overly common terms
            - sublinear_tf=True for better term weighting
            - Enable bigrams for phrase capture

            For SHORT-FORM documents (surveys, tweets):
            - Lower min_df (1-2) to preserve vocabulary
            - Avoid aggressive filtering
            - Bigrams only for larger corpora

            For SMALL corpora (<500 docs):
            - min_df=1 or 2 to preserve vocabulary
            - Smaller max_features to avoid noise

            These are heuristics, not hard rules. Users can override via override_config.
        """
        config = PreprocessingConfig()
        rationale_parts = []

        # Determine min_df based on corpus size and document type
        if characteristics.suggested_preprocessing == 'aggressive':
            # Long-form: stricter filtering
            if characteristics.corpus_size_category == 'large':
                config.min_df = 10
                rationale_parts.append("min_df=10 (large corpus, long-form docs)")
            elif characteristics.corpus_size_category == 'medium':
                config.min_df = 5
                rationale_parts.append("min_df=5 (medium corpus, long-form docs)")
            else:
                config.min_df = 3
                rationale_parts.append("min_df=3 (small corpus, long-form docs)")

            # Enable sublinear TF for long-form
            config.sublinear_tf = True
            rationale_parts.append("sublinear_tf=True (long-form content)")

        elif characteristics.suggested_preprocessing == 'minimal':
            # Very short, small vocabulary: preserve everything possible
            config.min_df = 1
            config.sublinear_tf = False
            rationale_parts.append("min_df=1 (minimal preprocessing for short responses)")

        else:  # 'standard'
            # Standard OE preprocessing
            if characteristics.corpus_size_category == 'small':
                config.min_df = 2
            else:
                config.min_df = 2
            rationale_parts.append("min_df=2 (standard preprocessing)")

        # Determine max_df based on document type
        if characteristics.is_long_form:
            config.max_df = 0.85  # Stricter for long-form
            rationale_parts.append("max_df=0.85 (long-form filtering)")
        else:
            config.max_df = 0.8  # Standard
            rationale_parts.append("max_df=0.8 (standard)")

        # Determine n-gram range
        # Always enable bigrams to capture meaningful phrases (e.g., "customer service", "too expensive")
        # This is especially important for focused survey responses where phrases carry more meaning
        if characteristics.is_long_form or characteristics.corpus_size_category == 'large':
            # Enable bigrams for long-form or large corpora
            config.ngram_range = (1, 2)
            rationale_parts.append("ngram_range=(1,2) (bigrams enabled)")
        elif characteristics.is_short_form and characteristics.corpus_size_category == 'small':
            # Enable bigrams for small short-form corpora to capture key phrases
            config.ngram_range = (1, 2)
            rationale_parts.append("ngram_range=(1,2) (bigrams for focused survey phrases)")
        else:
            # Default: include bigrams
            config.ngram_range = (1, 2)
            rationale_parts.append("ngram_range=(1,2) (default)")

        # Determine max_features based on vocabulary and corpus size
        if characteristics.vocabulary_size > 10000:
            config.max_features = min(2000, characteristics.vocabulary_size // 2)
            rationale_parts.append(f"max_features={config.max_features} (large vocabulary)")
        elif characteristics.vocabulary_size < 200:
            config.max_features = min(500, characteristics.vocabulary_size)
            rationale_parts.append(f"max_features={config.max_features} (small vocabulary)")
        else:
            config.max_features = self.default_max_features
            rationale_parts.append(f"max_features={config.max_features} (default)")

        # Stopwords: always use for long-form, optional for very short
        if characteristics.is_long_form:
            config.use_stopwords = True
            rationale_parts.append("stopwords=enabled (long-form)")
        elif characteristics.median_doc_length < 10:
            # Very short responses - stopwords might remove too much
            config.use_stopwords = False
            rationale_parts.append("stopwords=disabled (very short responses)")
        else:
            config.use_stopwords = True
            rationale_parts.append("stopwords=enabled (default)")

        config.stopwords_language = self.default_stopwords

        # Compile rationale
        config.config_rationale = "; ".join(rationale_parts)

        # Apply any manual overrides
        if override_config:
            for key, value in override_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            config.config_rationale += " [with manual overrides]"

        logger.info(f"Generated preprocessing config: {config.config_rationale}")

        return config

    def create_vectorizer(
        self,
        config: PreprocessingConfig,
        for_lda: bool = False
    ):
        """
        Create a vectorizer instance from configuration.

        Args:
            config: Preprocessing configuration
            for_lda: If True, creates CountVectorizer instead of TfidfVectorizer

        Returns:
            Configured TfidfVectorizer or CountVectorizer instance

        Notes:
            LDA requires CountVectorizer (non-negative counts).
            KMeans and NMF work with TfidfVectorizer.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

        kwargs = config.to_dict()

        # Remove sublinear_tf for CountVectorizer (not applicable)
        if for_lda:
            kwargs.pop('sublinear_tf', None)
            return CountVectorizer(**kwargs)
        else:
            return TfidfVectorizer(**kwargs)


class MultiLabelHandler:
    """
    Handles multi-label datasets for clustering and evaluation.

    Multi-label datasets (like Reuters-21578) have documents assigned to
    multiple categories. This class provides:

    1. Clustering mode: Labels are completely ignored for training
    2. Evaluation mode: Labels used only for post-hoc validation metrics

    CRITICAL: Labels are NEVER used in the clustering/training process.

    Example:
        >>> handler = MultiLabelHandler()
        >>>
        >>> # For clustering: just get texts, ignore labels
        >>> texts = handler.prepare_for_clustering(df, text_column='text')
        >>>
        >>> # For evaluation: compute metrics against true labels
        >>> metrics = handler.evaluate_clustering(
        ...     predicted_clusters=clusters,
        ...     true_labels=labels,
        ...     mode='overlap'  # handles multi-label
        ... )
    """

    def __init__(self, label_delimiter: Optional[str] = None):
        """
        Initialize multi-label handler.

        Args:
            label_delimiter: Delimiter used in label strings (e.g., ',', ';')
                            If None, will auto-detect
        """
        self.label_delimiter = label_delimiter
        self._detected_delimiter = None

    def prepare_for_clustering(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: Optional[str] = None
    ) -> Tuple[List[str], Optional[pd.DataFrame]]:
        """
        Prepare data for clustering (unsupervised training).

        Args:
            df: DataFrame with text data
            text_column: Name of column containing text
            label_column: Optional label column (will be separated, not used for training)

        Returns:
            Tuple of (texts_for_clustering, labels_df_for_later_evaluation)

        Notes:
            - Texts are returned for clustering (unsupervised)
            - Labels are separated and returned for optional post-hoc evaluation
            - Labels are NEVER used in the clustering process itself
        """
        texts = df[text_column].tolist()

        if label_column and label_column in df.columns:
            # Separate labels for later evaluation only
            labels_df = df[[text_column, label_column]].copy()
            labels_df['_original_index'] = df.index
            logger.info(
                f"Labels found in column '{label_column}'. "
                "Labels will be used for evaluation only, NOT for clustering."
            )
            return texts, labels_df
        else:
            return texts, None

    def parse_labels(
        self,
        label: str,
        delimiter: Optional[str] = None
    ) -> List[str]:
        """
        Parse a potentially multi-label string into individual labels.

        Args:
            label: Label string (possibly containing delimiters)
            delimiter: Delimiter to use (if None, uses detected or common delimiters)

        Returns:
            List of individual labels
        """
        if pd.isna(label) or label is None:
            return []

        label_str = str(label).strip()
        if not label_str:
            return []

        delimiter = delimiter or self.label_delimiter or self._detected_delimiter

        if delimiter and delimiter in label_str:
            labels = [l.strip() for l in label_str.split(delimiter)]
            return [l for l in labels if l]  # Remove empty strings
        else:
            return [label_str]

    def filter_single_label_documents(
        self,
        df: pd.DataFrame,
        label_column: str,
        delimiter: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter to only single-label documents for cleaner evaluation.

        Args:
            df: DataFrame with labels
            label_column: Column containing labels
            delimiter: Label delimiter

        Returns:
            DataFrame with only single-label documents

        Notes:
            This is useful for evaluation metrics that assume single labels.
            Multi-label documents are excluded from this filtered view.
        """
        delimiter = delimiter or self.label_delimiter

        def is_single_label(label):
            if pd.isna(label):
                return False
            label_str = str(label).strip()
            if not label_str:
                return False
            if delimiter and delimiter in label_str:
                return False
            return True

        mask = df[label_column].apply(is_single_label)
        filtered = df[mask].copy()

        logger.info(
            f"Filtered to {len(filtered)} single-label documents "
            f"(from {len(df)} total)"
        )

        return filtered

    def calculate_label_cluster_overlap(
        self,
        cluster_assignments: List[int],
        true_labels: List[str],
        delimiter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate overlap between clusters and true labels (multi-label aware).

        Args:
            cluster_assignments: List of cluster IDs per document
            true_labels: List of label strings (possibly multi-label)
            delimiter: Label delimiter

        Returns:
            Dictionary with overlap metrics:
                - cluster_label_matrix: Cluster-label co-occurrence matrix
                - cluster_purity: Purity score per cluster
                - overall_purity: Weighted average purity
                - label_coverage: How well clusters cover each label

        Notes:
            This evaluation happens AFTER clustering is complete.
            It is purely diagnostic and does not affect clustering.
        """
        delimiter = delimiter or self.label_delimiter

        # Parse all labels
        parsed_labels = [self.parse_labels(l, delimiter) for l in true_labels]

        # Get unique labels and clusters
        all_labels = set()
        for labels in parsed_labels:
            all_labels.update(labels)
        all_labels = sorted(all_labels)
        unique_clusters = sorted(set(cluster_assignments))

        if not all_labels:
            logger.warning("No valid labels found for evaluation")
            return {
                'cluster_label_matrix': None,
                'cluster_purity': {},
                'overall_purity': 0.0,
                'label_coverage': {}
            }

        # Build cluster-label co-occurrence matrix
        n_clusters = len(unique_clusters)
        n_labels = len(all_labels)
        label_to_idx = {l: i for i, l in enumerate(all_labels)}
        cluster_to_idx = {c: i for i, c in enumerate(unique_clusters)}

        matrix = np.zeros((n_clusters, n_labels), dtype=int)

        for cluster, labels in zip(cluster_assignments, parsed_labels):
            cluster_idx = cluster_to_idx[cluster]
            for label in labels:
                if label in label_to_idx:
                    label_idx = label_to_idx[label]
                    matrix[cluster_idx, label_idx] += 1

        # Calculate purity per cluster
        cluster_purity = {}
        cluster_sizes = []
        for cluster in unique_clusters:
            cluster_idx = cluster_to_idx[cluster]
            row = matrix[cluster_idx]
            total = row.sum()
            if total > 0:
                purity = row.max() / total
            else:
                purity = 0.0
            cluster_purity[cluster] = purity
            cluster_sizes.append(total)

        # Calculate overall purity (weighted by cluster size)
        total_docs = sum(cluster_sizes)
        if total_docs > 0:
            overall_purity = sum(
                cluster_purity[c] * size
                for c, size in zip(unique_clusters, cluster_sizes)
            ) / total_docs
        else:
            overall_purity = 0.0

        # Calculate label coverage
        label_coverage = {}
        for label in all_labels:
            label_idx = label_to_idx[label]
            col = matrix[:, label_idx]
            total = col.sum()
            if total > 0:
                # Coverage: what fraction of this label is in its dominant cluster
                coverage = col.max() / total
            else:
                coverage = 0.0
            label_coverage[label] = coverage

        return {
            'cluster_label_matrix': pd.DataFrame(
                matrix,
                index=[f"Cluster_{c}" for c in unique_clusters],
                columns=all_labels
            ),
            'cluster_purity': cluster_purity,
            'overall_purity': overall_purity,
            'label_coverage': label_coverage
        }


def get_adaptive_preprocessing(
    texts: List[str],
    labels: Optional[List[str]] = None,
    override_config: Optional[Dict[str, Any]] = None,
    sample_size: int = 5000
) -> Tuple[PreprocessingConfig, DatasetCharacteristics]:
    """
    Convenience function to get adaptive preprocessing configuration.

    This is the main entry point for adaptive preprocessing. It:
    1. Detects dataset characteristics
    2. Generates appropriate TF-IDF configuration
    3. Returns both for transparency

    Args:
        texts: List of document texts
        labels: Optional labels (for characteristic detection, NOT for training)
        override_config: Optional manual overrides
        sample_size: Sample size for large datasets

    Returns:
        Tuple of (PreprocessingConfig, DatasetCharacteristics)

    Example:
        >>> config, characteristics = get_adaptive_preprocessing(texts)
        >>> print(f"Detected: {characteristics.suggested_preprocessing}")
        >>> print(f"Using min_df={config.min_df}, ngrams={config.ngram_range}")
        >>>
        >>> vectorizer = TfidfVectorizer(**config.to_dict())
    """
    detector = DatasetCharacteristicsDetector()
    characteristics = detector.analyze(texts, labels, sample_size)

    preprocessor = AdaptivePreprocessor()
    config = preprocessor.get_config(characteristics, override_config)

    return config, characteristics
