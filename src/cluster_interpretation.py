"""
Cluster interpretation layer for TF-IDF + KMeans clustering.

This module provides MANDATORY interpretation of cluster results, ensuring that
cluster IDs are never exposed without human-readable explanations. This prevents
users from mistaking numeric cluster IDs for semantic labels.

Key principles:
- Every cluster MUST have a human-readable summary
- Cluster IDs are internal identifiers, NOT semantic labels
- Interpretation is based on top terms, not external labels
- Works for ALL datasets, not just specific ones

Classes:
    ClusterInterpreter: Generates human-readable cluster summaries
    ClusterSummary: Data class holding cluster interpretation

Usage:
    >>> interpreter = ClusterInterpreter()
    >>> summaries = interpreter.interpret_clusters(
    ...     vectorizer=fitted_vectorizer,
    ...     cluster_model=fitted_kmeans,
    ...     texts=original_texts,
    ...     cluster_assignments=labels
    ... )
    >>> for cluster_id, summary in summaries.items():
    ...     print(f"{cluster_id}: {summary.label}")
    ...     print(f"  Top terms: {summary.top_terms}")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import sentiment analysis (with graceful fallback)
try:
    from .sentiment_analysis import SurveySentimentAnalyzer, SentimentResult
    SENTIMENT_ANALYSIS_AVAILABLE = True
except ImportError:
    SENTIMENT_ANALYSIS_AVAILABLE = False
    logger.debug("Sentiment analysis module not available for cluster sentiment distribution")

# Import stopwords discovery utilities (with graceful fallback)
try:
    from .stopwords_discovery import (
        load_stopwords_from_file,
        load_keep_list,
        get_layered_stopwords,
        find_domain_stopword_candidates,
        StopwordDiscoveryReport,
    )
    STOPWORDS_DISCOVERY_AVAILABLE = True
except ImportError:
    STOPWORDS_DISCOVERY_AVAILABLE = False
    logger.debug("Stopwords discovery module not available")

# Default stopwords to exclude from topic labels (common non-descriptive words)
DEFAULT_LABEL_STOPWORDS: Set[str] = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
    'and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'both', 'either', 'neither',
    'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
    'of', 'in', 'to', 'on', 'at', 'by', 'with', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'any', 'if', 'because', 'until', 'while',
    'it', 'its', 'this', 'that', 'these', 'those', 'what', 'which', 'who',
    'whom', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
    'himself', 'she', 'her', 'hers', 'herself', 'they', 'them', 'their',
    'theirs', 'themselves', 'am', 'about', 'against', 'over', 'out', 'up',
    'down', 'off', 'on', 'over', 'under', 'again', 'get', 'got', 'getting',
    'really', 'actually', 'basically', 'simply', 'even', 'still', 'already',
    've', 'll', 're', 't', 's', 'd', 'm',  # Contractions
    # HTML entity artifacts
    'amp', 'nbsp', 'quot', 'lt', 'gt', 'apos', 'ndash', 'mdash',
    'rsquo', 'lsquo', 'rdquo', 'ldquo', 'hellip', 'bull', 'copy', 'reg', 'trade'
}

# Sentiment-bearing words that should be excluded from labels when a cluster
# has mixed sentiment (both positive and negative views on the same topic).
# These adjectives/adverbs inject a sentiment direction into what should be
# a neutral topic label, causing mismatch with representative quotes.
SENTIMENT_LABEL_WORDS: Set[str] = {
    # Negative sentiment words
    'poor', 'bad', 'terrible', 'horrible', 'awful', 'worst', 'worse',
    'lacking', 'inadequate', 'insufficient', 'inconsistent', 'unreliable',
    'disappointing', 'disappointed', 'frustrating', 'frustrated',
    'unacceptable', 'unsatisfactory', 'unhelpful', 'unprofessional',
    'rude', 'slow', 'delayed', 'broken', 'failed', 'failing', 'negative',
    'declining', 'deteriorating', 'problematic', 'difficult', 'confusing',
    'unclear', 'missing', 'neglected', 'ignored', 'forgotten',
    # Positive sentiment words
    'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'outstanding',
    'exceptional', 'perfect', 'best', 'superior', 'impressive',
    'satisfying', 'satisfied', 'pleasant', 'friendly', 'helpful',
    'professional', 'efficient', 'quick', 'fast', 'brilliant',
    'remarkable', 'superb', 'magnificent', 'positive', 'improving',
}

# For backward compatibility
LABEL_STOPWORDS = DEFAULT_LABEL_STOPWORDS


def detect_domain_stopwords(
    texts: List[str],
    min_doc_frequency: float = 0.7,
    max_words: int = 20
) -> Set[str]:
    """
    Detect domain-specific high-frequency words that should be treated as stopwords.

    These are words that appear in a high percentage of documents and are likely
    uninformative for distinguishing clusters in this specific domain.

    Args:
        texts: List of document texts
        min_doc_frequency: Minimum document frequency (0-1) to consider a word as domain stopword
        max_words: Maximum number of domain stopwords to detect

    Returns:
        Set of detected domain-specific stopwords
    """
    if not texts:
        return set()

    from collections import Counter

    # Count document frequency for each word
    word_doc_counts = Counter()
    n_docs = len(texts)

    for text in texts:
        if not text or pd.isna(text):
            continue
        # Get unique words in this document
        words = set(str(text).lower().split())
        for word in words:
            # Skip very short words and words that are already stopwords
            if len(word) > 2 and word not in DEFAULT_LABEL_STOPWORDS:
                word_doc_counts[word] += 1

    # Find words that appear in too many documents
    domain_stopwords = set()
    for word, count in word_doc_counts.most_common():
        doc_freq = count / n_docs
        if doc_freq >= min_doc_frequency:
            domain_stopwords.add(word)
            if len(domain_stopwords) >= max_words:
                break
        else:
            # Since most_common returns in descending order, we can stop early
            break

    if domain_stopwords:
        logger.info(f"Detected {len(domain_stopwords)} domain-specific stopwords: {domain_stopwords}")

    return domain_stopwords


@dataclass
class ClusterSummary:
    """
    Human-readable summary of a single cluster.

    IMPORTANT: cluster_id is an internal identifier, NOT a semantic label.
    The 'label' field provides a human-readable interpretation.

    Attributes:
        cluster_id: Internal cluster identifier (e.g., 0, 1, 2 or "CLUSTER_01")
        label: Human-readable label generated from top terms
        top_terms: List of most representative terms for this cluster
        term_weights: Weights/scores for each top term
        document_count: Number of documents in this cluster
        document_indices: Indices of documents assigned to this cluster
        representative_docs: Example documents from this cluster
        coherence_score: Optional semantic coherence score
        interpretation_notes: Additional notes about interpretation

        # Warning fields
        is_interpretable: Whether cluster has clear interpretation
        interpretation_confidence: Confidence in the interpretation (0-1)
        warnings: Any warnings about interpretation quality

        # LLM-enhanced fields
        llm_label: LLM-refined label (if available)
        llm_alternative_labels: Alternative label suggestions from LLM
        llm_description: Detailed description from LLM
        llm_reasoning: LLM's reasoning for the interpretation
        llm_source: Source of LLM interpretation ('api', 'local', or 'fallback')
    """
    cluster_id: Any
    label: str
    top_terms: List[str]
    term_weights: List[float] = field(default_factory=list)
    document_count: int = 0
    document_indices: List[int] = field(default_factory=list)
    representative_docs: List[Dict[str, Any]] = field(default_factory=list)
    coherence_score: Optional[float] = None
    interpretation_notes: str = ""

    # Interpretation quality indicators
    is_interpretable: bool = True
    interpretation_confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)

    # LLM-enhanced fields
    llm_label: Optional[str] = None
    llm_alternative_labels: List[str] = field(default_factory=list)
    llm_description: Optional[str] = None
    llm_reasoning: Optional[str] = None
    llm_source: Optional[str] = None

    # Sentiment distribution fields (for detecting mixed-sentiment clusters)
    sentiment_distribution: Optional[Dict[str, float]] = None  # {'positive': 0.4, 'negative': 0.3, 'neutral': 0.3}
    dominant_sentiment: Optional[str] = None  # 'positive', 'negative', 'neutral', or 'mixed'
    sentiment_coherence: Optional[float] = None  # 0-1 score, low = mixed sentiment
    has_mixed_sentiment: bool = False  # True if cluster contains conflicting sentiments

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'cluster_id': self.cluster_id,
            'label': self.label,
            'top_terms': self.top_terms,
            'term_weights': [round(w, 4) for w in self.term_weights],
            'document_count': self.document_count,
            'representative_docs': self.representative_docs,
            'coherence_score': round(self.coherence_score, 4) if self.coherence_score else None,
            'interpretation_notes': self.interpretation_notes,
            'is_interpretable': self.is_interpretable,
            'interpretation_confidence': round(self.interpretation_confidence, 3),
            'warnings': self.warnings,
        }
        # Include LLM-enhanced fields if available
        if self.llm_label:
            result['llm_label'] = self.llm_label
            result['llm_alternative_labels'] = self.llm_alternative_labels
            result['llm_description'] = self.llm_description
            result['llm_reasoning'] = self.llm_reasoning
            result['llm_source'] = self.llm_source
        # Include sentiment distribution fields if available
        if self.sentiment_distribution is not None:
            result['sentiment_distribution'] = {
                k: round(v, 3) for k, v in self.sentiment_distribution.items()
            }
            result['dominant_sentiment'] = self.dominant_sentiment
            result['sentiment_coherence'] = round(self.sentiment_coherence, 3) if self.sentiment_coherence else None
            result['has_mixed_sentiment'] = self.has_mixed_sentiment
        return result

    @property
    def display_label(self) -> str:
        """Get the best available label (LLM if available, otherwise term-based)."""
        return self.llm_label if self.llm_label else self.label

    def get_display_string(self, include_terms: bool = True, include_llm: bool = True) -> str:
        """Get a formatted display string for this cluster."""
        # Use LLM label if available
        display_label = self.display_label
        parts = [f"{self.cluster_id}: {display_label} ({self.document_count} docs)"]

        # Show original term-based label if LLM label is different
        if include_llm and self.llm_label and self.llm_label != self.label:
            parts.append(f"  Original label: {self.label}")

        if include_terms and self.top_terms:
            terms_str = ", ".join(self.top_terms[:5])
            parts.append(f"  Terms: {terms_str}")

        # Show LLM description if available
        if include_llm and self.llm_description:
            parts.append(f"  Description: {self.llm_description}")

        # Show alternative labels if available
        if include_llm and self.llm_alternative_labels:
            alts_str = ", ".join(self.llm_alternative_labels[:3])
            parts.append(f"  Alternatives: {alts_str}")

        # Show sentiment distribution if available
        if self.sentiment_distribution is not None:
            sent_parts = [f"{k}: {v:.0%}" for k, v in self.sentiment_distribution.items()]
            parts.append(f"  Sentiment: {', '.join(sent_parts)}")
            if self.has_mixed_sentiment:
                parts.append(f"  ⚠ MIXED SENTIMENT: This cluster contains both positive and negative views")

        if self.warnings:
            parts.append(f"  Warnings: {'; '.join(self.warnings)}")

        # Show LLM source
        if include_llm and self.llm_source:
            parts.append(f"  [LLM: {self.llm_source}]")

        return "\n".join(parts)


@dataclass
class ClusterInterpretationReport:
    """
    Complete interpretation report for all clusters.

    This report provides a comprehensive view of cluster interpretations
    and should be presented to users instead of raw cluster IDs.

    Attributes:
        summaries: Dictionary mapping cluster_id to ClusterSummary
        n_clusters: Total number of clusters
        n_documents: Total number of documents
        overall_interpretability: Average interpretability across clusters
        method_used: Description of clustering method
        warnings: Global warnings about interpretation
        llm_enhanced: Whether LLM enhancement was applied
        llm_backend: Backend used for LLM ('api', 'local', or None)
        overall_coherence: Average coherence score across clusters (if computed)
        tuning_recommendations: Automated recommendations for improving clusters
        domain_stopwords_used: Set of domain-specific stopwords detected
    """
    summaries: Dict[Any, ClusterSummary]
    n_clusters: int
    n_documents: int
    overall_interpretability: float
    method_used: str = ""
    warnings: List[str] = field(default_factory=list)
    llm_enhanced: bool = False
    llm_backend: Optional[str] = None
    overall_coherence: Optional[float] = None
    tuning_recommendations: List[str] = field(default_factory=list)
    domain_stopwords_used: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'n_clusters': self.n_clusters,
            'n_documents': self.n_documents,
            'overall_interpretability': round(self.overall_interpretability, 3),
            'method_used': self.method_used,
            'warnings': self.warnings,
            'llm_enhanced': self.llm_enhanced,
            'llm_backend': self.llm_backend,
            'cluster_summaries': {
                k: v.to_dict() for k, v in self.summaries.items()
            }
        }
        if self.overall_coherence is not None:
            result['overall_coherence'] = round(self.overall_coherence, 3)
        if self.tuning_recommendations:
            result['tuning_recommendations'] = self.tuning_recommendations
        if self.domain_stopwords_used:
            result['domain_stopwords_used'] = list(self.domain_stopwords_used)
        # Add sentiment analysis summary
        mixed_sentiment_clusters = [
            k for k, v in self.summaries.items() if v.has_mixed_sentiment
        ]
        if mixed_sentiment_clusters:
            result['mixed_sentiment_clusters'] = mixed_sentiment_clusters
            result['mixed_sentiment_count'] = len(mixed_sentiment_clusters)
        return result

    def get_display_report(self, always_show_terms: bool = True) -> str:
        """
        Generate a human-readable report.

        Args:
            always_show_terms: If True, always display top terms alongside labels.
                              This reduces over-reliance on short labels.

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 60,
            "CLUSTER INTERPRETATION REPORT",
            "=" * 60,
            "",
            f"Method: {self.method_used}",
            f"Total Documents: {self.n_documents}",
            f"Number of Clusters: {self.n_clusters}",
            f"Overall Interpretability: {self.overall_interpretability:.1%}",
        ]

        # Show coherence if available
        if self.overall_coherence is not None:
            lines.append(f"Overall Coherence: {self.overall_coherence:.2f}")

        # Show LLM enhancement status
        if self.llm_enhanced:
            lines.append(f"LLM Enhancement: Enabled ({self.llm_backend})")
        else:
            lines.append("LLM Enhancement: Disabled (using term-based labels)")

        # Show domain stopwords if any
        if self.domain_stopwords_used:
            lines.append(f"Domain Stopwords Detected: {', '.join(sorted(self.domain_stopwords_used)[:5])}" +
                        (f"... (+{len(self.domain_stopwords_used) - 5} more)" if len(self.domain_stopwords_used) > 5 else ""))

        lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        if self.tuning_recommendations:
            lines.append("TUNING RECOMMENDATIONS:")
            for rec in self.tuning_recommendations:
                lines.append(f"  - {rec}")
            lines.append("")

        lines.append("-" * 60)
        lines.append("CLUSTER DETAILS")
        lines.append("-" * 60)
        lines.append("")

        # Sort by document count
        sorted_summaries = sorted(
            self.summaries.values(),
            key=lambda s: s.document_count,
            reverse=True
        )

        for summary in sorted_summaries:
            # Always show both label and top terms to reduce over-reliance on labels
            lines.append(summary.get_display_string(
                include_terms=always_show_terms,
                include_llm=self.llm_enhanced
            ))
            lines.append("")

        lines.append("=" * 60)
        lines.append("NOTE: Cluster IDs are internal identifiers, not semantic labels.")
        lines.append("IMPORTANT: Always review top_terms alongside labels for full context.")
        if self.llm_enhanced:
            lines.append("Labels have been refined by LLM for improved readability.")
        else:
            lines.append("Labels are generated from top weighted terms/phrases.")
        lines.append("=" * 60)

        return "\n".join(lines)


class ClusterInterpreter:
    """
    Generates human-readable interpretations for clusters.

    This class is a MANDATORY component of the clustering pipeline.
    It ensures that cluster results are always presented with interpretable
    summaries, preventing users from mistaking cluster IDs for semantic labels.

    Key features:
    - Extracts top-N terms per cluster
    - Generates human-readable labels from terms
    - Provides confidence scores for interpretations
    - Includes representative documents per cluster
    - Warns when clusters have weak interpretations

    Example:
        >>> interpreter = ClusterInterpreter(n_top_terms=10)
        >>> report = interpreter.interpret_clusters(
        ...     vectorizer=fitted_tfidf,
        ...     cluster_model=fitted_kmeans,
        ...     texts=documents,
        ...     cluster_assignments=cluster_labels
        ... )
        >>> print(report.get_display_report())
    """

    def __init__(
        self,
        n_top_terms: int = 15,
        n_label_terms: int = 3,
        n_representative_docs: int = 10,  # Increased from 5 to provide more context for LLM
        min_term_weight_threshold: float = 0.005,
        low_interpretability_threshold: float = 0.3,
        custom_stopwords: Optional[Set[str]] = None,
        detect_domain_stopwords: bool = True,
        prefer_ngram_phrases: bool = True,
        min_coherence_threshold: float = 0.4,
        use_file_based_stopwords: bool = True,
        domain_stopwords_path: Optional[Union[str, Path]] = None,
        keep_list_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the cluster interpreter.

        Args:
            n_top_terms: Number of top terms to extract per cluster
            n_label_terms: Number of terms to use in generated labels
            n_representative_docs: Number of example documents to include (used by LLM for context)
            min_term_weight_threshold: Minimum weight for terms to be considered
            low_interpretability_threshold: Threshold below which to warn about interpretability
            custom_stopwords: Optional set of additional stopwords to exclude from labels
            detect_domain_stopwords: Whether to auto-detect domain-specific high-frequency stopwords
            prefer_ngram_phrases: Whether to prefer n-gram phrases over single words in labels
            min_coherence_threshold: Minimum coherence score below which to flag cluster issues
            use_file_based_stopwords: Whether to load domain stopwords from file
                                      (data/stopwords_domain.txt). Default True.
            domain_stopwords_path: Custom path to domain stopwords file.
                                   If None, uses data/stopwords_domain.txt
            keep_list_path: Custom path to keep-list file (semantic overrides).
                           If None, uses data/stopwords_keep.txt
        """
        self.n_top_terms = n_top_terms
        self.n_label_terms = n_label_terms
        self.n_representative_docs = n_representative_docs
        self.min_term_weight_threshold = min_term_weight_threshold
        self.low_interpretability_threshold = low_interpretability_threshold
        self.custom_stopwords = custom_stopwords or set()
        self.detect_domain_stopwords_flag = detect_domain_stopwords
        self.prefer_ngram_phrases = prefer_ngram_phrases
        self.min_coherence_threshold = min_coherence_threshold
        self.use_file_based_stopwords = use_file_based_stopwords
        self.domain_stopwords_path = domain_stopwords_path
        self.keep_list_path = keep_list_path
        self._domain_stopwords: Set[str] = set()
        self._file_based_stopwords: Set[str] = set()
        self._keep_list: Set[str] = set()
        self._texts_for_domain_detection: Optional[List[str]] = None
        self._stopwords_loaded: bool = False

    def _load_file_based_stopwords(self) -> None:
        """
        Load stopwords from files if enabled and not already loaded.

        This implements the layered governance model:
        1. Default label stopwords (always included)
        2. File-based domain stopwords (data/stopwords_domain.txt)
        3. Custom stopwords (passed to constructor)
        4. Dynamically detected domain stopwords
        5. Minus keep-list words (data/stopwords_keep.txt)
        """
        if self._stopwords_loaded:
            return

        if self.use_file_based_stopwords and STOPWORDS_DISCOVERY_AVAILABLE:
            try:
                # Load domain stopwords from file
                self._file_based_stopwords = load_stopwords_from_file(
                    self.domain_stopwords_path
                )
                if self._file_based_stopwords:
                    logger.info(
                        f"Loaded {len(self._file_based_stopwords)} domain stopwords from file"
                    )

                # Load keep-list
                self._keep_list = load_keep_list(self.keep_list_path)
                if self._keep_list:
                    logger.info(
                        f"Loaded {len(self._keep_list)} keep-list words from file"
                    )
            except Exception as e:
                logger.warning(f"Error loading file-based stopwords: {e}")

        self._stopwords_loaded = True

    def _get_combined_stopwords(self) -> Set[str]:
        """
        Get the combined set of all stopwords using the layered governance model.

        Layers (in order of application):
        1. Default label stopwords (common non-descriptive words)
        2. File-based domain stopwords (data/stopwords_domain.txt)
        3. Custom stopwords (passed to constructor)
        4. Dynamically detected domain stopwords (runtime detection)
        5. Minus keep-list words (data/stopwords_keep.txt - semantic overrides)

        Returns:
            Combined set of stopwords with keep-list exclusions applied
        """
        # Ensure file-based stopwords are loaded
        self._load_file_based_stopwords()

        # Layer 1: Default label stopwords
        combined = set(DEFAULT_LABEL_STOPWORDS)

        # Layer 2: File-based domain stopwords
        combined.update(self._file_based_stopwords)

        # Layer 3: Custom stopwords
        combined.update(self.custom_stopwords)

        # Layer 4: Dynamically detected domain stopwords
        combined.update(self._domain_stopwords)

        # Layer 5: Remove keep-list words (semantic overrides)
        combined -= self._keep_list

        return combined

    def _filter_stopwords_with_weights(
        self, terms: List[str], weights: List[float]
    ) -> Tuple[List[str], List[float]]:
        """
        Filter out stopwords from terms list, keeping weights in sync.

        Uses combined stopwords: default + custom + domain-specific.

        Args:
            terms: List of terms to filter
            weights: Corresponding weights for each term

        Returns:
            Tuple of (filtered_terms, filtered_weights) with stopwords removed
        """
        combined_stopwords = self._get_combined_stopwords()
        filtered_terms = []
        filtered_weights = []
        for term, weight in zip(terms, weights):
            if term.lower().strip() not in combined_stopwords:
                filtered_terms.append(term)
                filtered_weights.append(weight)
        return filtered_terms, filtered_weights

    def _filter_sentiment_terms_for_label(
        self,
        terms: List[str],
        weights: List[float]
    ) -> Tuple[List[str], List[float]]:
        """
        Remove sentiment-bearing words from label terms for mixed-sentiment clusters.

        When a cluster has mixed sentiment (both positive and negative views on
        the same topic), sentiment-bearing adjectives like "inconsistent" or
        "excellent" should not appear in the label because they create a
        misleading mismatch with representative quotes of the opposite sentiment.

        For multi-word terms (n-grams), individual sentiment words are removed
        from the phrase rather than dropping the entire term, so "inconsistent
        follow-up care" becomes "follow-up care".

        Args:
            terms: List of terms (may include n-grams)
            weights: Corresponding weights

        Returns:
            Tuple of (filtered_terms, filtered_weights) with sentiment words removed
        """
        filtered_terms = []
        filtered_weights = []
        for term, weight in zip(terms, weights):
            words = term.split()
            if len(words) > 1:
                # Multi-word term: strip sentiment words but keep the rest
                neutral_words = [
                    w for w in words if w.lower() not in SENTIMENT_LABEL_WORDS
                ]
                if neutral_words:
                    filtered_terms.append(" ".join(neutral_words))
                    filtered_weights.append(weight)
                # else: entire phrase was sentiment words, drop it
            else:
                # Single word: drop if it's a sentiment word
                if term.lower() not in SENTIMENT_LABEL_WORDS:
                    filtered_terms.append(term)
                    filtered_weights.append(weight)
        return filtered_terms, filtered_weights

    def _filter_label_terms(self, terms: List[str]) -> List[str]:
        """
        Filter terms for use in labels: remove duplicates and subset phrases.

        Note: Stopwords should already be filtered before calling this method.

        Args:
            terms: List of terms to filter

        Returns:
            Filtered list with no duplicates and no subset phrases
        """
        seen = set()
        filtered = []
        for term in terms:
            # Normalize for comparison
            term_lower = term.lower().strip()
            # Skip duplicates
            if term_lower not in seen:
                seen.add(term_lower)
                filtered.append(term)

        # Remove terms that are subsets of other terms
        # e.g., "Hard" is removed if "Hard To" exists
        filtered = self._remove_subset_terms(filtered)

        return filtered

    def _remove_subset_terms(self, terms: List[str]) -> List[str]:
        """
        Remove terms whose words are a subset of another term's words.

        For example, if we have ["Hard", "Hard To", "Stay"], "Hard" will be
        removed because all its words appear in "Hard To".

        Args:
            terms: List of terms to filter

        Returns:
            Filtered list with subset terms removed
        """
        if len(terms) <= 1:
            return terms

        # Convert each term to a set of words for comparison
        term_words = []
        for term in terms:
            words = set(term.lower().split())
            term_words.append((term, words))

        # Find terms that are subsets of other terms
        to_remove = set()
        for i, (term_i, words_i) in enumerate(term_words):
            for j, (term_j, words_j) in enumerate(term_words):
                if i != j:
                    # If term_i's words are a proper subset of term_j's words,
                    # mark term_i for removal
                    if words_i < words_j:  # proper subset check
                        to_remove.add(i)

        # Return terms that are not subsets
        return [term for i, (term, _) in enumerate(term_words) if i not in to_remove]

    def _generate_phrase_based_label(
        self,
        filtered_terms: List[str],
        filtered_weights: List[float],
        n_label_terms: int
    ) -> str:
        """
        Generate a label preferring n-gram phrases over single words.

        This method prioritizes multi-word phrases (2-3 words) because they
        are typically more semantically complete than single words.
        For example, "customer service" is more informative than just "customer".

        Strategy:
        1. Score each term: phrase_weight = base_weight * (1 + 0.3 * (n_words - 1))
           This gives a 30% bonus per additional word to prefer phrases
        2. Select top phrases that don't overlap too much in words
        3. Combine into a readable label

        Args:
            filtered_terms: List of terms (may include n-grams)
            filtered_weights: Corresponding weights
            n_label_terms: Target number of distinct concepts for label

        Returns:
            Generated label string
        """
        combined_stopwords = self._get_combined_stopwords()

        if not filtered_terms:
            return ""

        # Score each term, boosting n-grams
        scored_terms = []
        for term, weight in zip(filtered_terms, filtered_weights):
            # Clean the term
            term_words = [w for w in term.split() if w.lower() not in combined_stopwords]
            if not term_words:
                continue

            n_words = len(term_words)
            # Boost multi-word phrases: 30% bonus per additional word
            # This makes "customer service" (2 words) score 1.3x its base weight
            # and "easy to use" (3 words) score 1.6x its base weight
            phrase_boost = 1.0 + 0.3 * (n_words - 1)
            adjusted_weight = weight * phrase_boost

            scored_terms.append({
                'term': term,
                'words': term_words,
                'n_words': n_words,
                'base_weight': weight,
                'adjusted_weight': adjusted_weight
            })

        # Sort by adjusted weight
        scored_terms.sort(key=lambda x: x['adjusted_weight'], reverse=True)

        # Select terms that don't overlap too much
        selected_terms = []
        used_words = set()
        total_words = 0

        for item in scored_terms:
            term_words_lower = set(w.lower() for w in item['words'])

            # Check overlap with already selected terms
            overlap = len(term_words_lower & used_words)
            overlap_ratio = overlap / len(term_words_lower) if term_words_lower else 1.0

            # Accept if low overlap or if it's a longer phrase containing used words
            # (e.g., accept "customer service" even if "customer" was used)
            if overlap_ratio < 0.5 or (item['n_words'] > 1 and overlap < item['n_words']):
                selected_terms.append(item)
                used_words.update(term_words_lower)
                total_words += item['n_words']

                # Stop when we have enough concepts or words
                if len(selected_terms) >= n_label_terms or total_words >= n_label_terms + 2:
                    break

        if not selected_terms:
            # Fallback: just use top term
            if scored_terms:
                selected_terms = [scored_terms[0]]

        # Build the label
        label_parts = []
        for item in selected_terms:
            # Title case each word
            formatted = " ".join(w.title() for w in item['words'])
            label_parts.append(formatted)

        return " ".join(label_parts)

    def _generate_word_based_label(
        self,
        filtered_terms: List[str],
        filtered_weights: List[float],
        n_label_terms: int
    ) -> str:
        """
        Generate a label using the legacy word-by-word approach.

        This is the original approach: extract individual words from terms,
        sort by weight, and select top N words.

        Args:
            filtered_terms: List of terms
            filtered_weights: Corresponding weights
            n_label_terms: Number of words to use in label

        Returns:
            Generated label string
        """
        combined_stopwords = self._get_combined_stopwords()

        # Build label word-by-word with weight tracking for semantic coherence
        seen_words = set()
        word_weight_pairs = []
        for term, term_weight in zip(filtered_terms, filtered_weights):
            # Split term into individual words and filter
            for word in term.split():
                word_lower = word.lower().strip()
                # Skip stopwords and already-seen words
                if word_lower not in combined_stopwords and word_lower not in seen_words:
                    seen_words.add(word_lower)
                    word_weight_pairs.append((word, term_weight))

        # Sort by weight (highest first) for semantic coherence
        word_weight_pairs.sort(key=lambda x: x[1], reverse=True)

        # Take top N words by weight
        label_words = [w for w, _ in word_weight_pairs[:n_label_terms]]

        if label_words:
            return " ".join(word.title() for word in label_words)
        else:
            return ""

    def interpret_clusters(
        self,
        vectorizer,
        cluster_model,
        texts: List[str],
        cluster_assignments: List[int],
        feature_matrix=None,
        method_name: str = 'tfidf_kmeans'
    ) -> ClusterInterpretationReport:
        """
        Generate interpretation report for clustering results.

        This is the main entry point for cluster interpretation.
        It MUST be called after clustering to provide interpretable results.

        Args:
            vectorizer: Fitted TF-IDF or similar vectorizer with get_feature_names_out()
            cluster_model: Fitted clustering model (KMeans, LDA, NMF, etc.)
            texts: Original text documents
            cluster_assignments: Cluster assignment for each document
            feature_matrix: Optional precomputed feature matrix
            method_name: Name of clustering method for reporting

        Returns:
            ClusterInterpretationReport with all cluster summaries

        Raises:
            ValueError: If inputs are inconsistent

        Notes:
            - This method extracts top terms from cluster centroids or components
            - For KMeans: uses cluster_centers_
            - For LDA/NMF: uses components_
            - Labels are generated from top terms, NOT from external data
            - Supports phrase-based labels (prefer_ngram_phrases=True) for better semantic clarity
            - Detects domain-specific stopwords if detect_domain_stopwords=True
        """
        # Validate inputs
        if len(texts) != len(cluster_assignments):
            raise ValueError(
                f"Mismatch: {len(texts)} texts but {len(cluster_assignments)} assignments"
            )

        # Detect domain-specific stopwords if enabled
        if self.detect_domain_stopwords_flag and not self._domain_stopwords:
            self._domain_stopwords = detect_domain_stopwords(texts)
            if self._domain_stopwords:
                logger.info(f"Using domain stopwords: {self._domain_stopwords}")

        # Get feature names
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            # Fallback for embedders that use generic feature names
            feature_names = [f"feature_{i}" for i in range(
                cluster_model.cluster_centers_.shape[1] if hasattr(cluster_model, 'cluster_centers_')
                else cluster_model.components_.shape[1]
            )]

        # Determine cluster centers/components based on model type
        if hasattr(cluster_model, 'cluster_centers_'):
            # KMeans-style model
            centers = cluster_model.cluster_centers_
            unique_clusters = list(range(len(centers)))
        elif hasattr(cluster_model, 'components_'):
            # LDA/NMF-style model
            centers = cluster_model.components_
            unique_clusters = list(range(len(centers)))
        else:
            raise ValueError(
                f"Unknown model type: {type(cluster_model)}. "
                "Expected KMeans (cluster_centers_) or LDA/NMF (components_)."
            )

        # Group documents by cluster
        cluster_docs = {}
        for idx, cluster in enumerate(cluster_assignments):
            if cluster not in cluster_docs:
                cluster_docs[cluster] = []
            cluster_docs[cluster].append(idx)

        # Generate summaries for each cluster
        summaries = {}
        global_warnings = []
        interpretability_scores = []
        coherence_scores = []

        for cluster_idx in unique_clusters:
            # Get cluster center
            center = centers[cluster_idx]

            # Extract top terms
            top_indices = center.argsort()[-self.n_top_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            term_weights = [float(center[i]) for i in top_indices]

            # Filter by weight threshold
            filtered_terms = []
            filtered_weights = []
            for term, weight in zip(top_terms, term_weights):
                if weight >= self.min_term_weight_threshold:
                    filtered_terms.append(term)
                    filtered_weights.append(weight)

            # Filter stopwords from all terms (for keywords)
            filtered_terms, filtered_weights = self._filter_stopwords_with_weights(
                filtered_terms, filtered_weights
            )

            # Get document indices early - needed for sentiment analysis before label generation
            doc_indices = cluster_docs.get(cluster_idx, [])
            doc_count = len(doc_indices)

            # Analyze sentiment BEFORE label generation so we can produce
            # sentiment-neutral labels for mixed-sentiment clusters
            sentiment_result = self._analyze_cluster_sentiment(texts, doc_indices)

            # For mixed-sentiment clusters, remove sentiment-bearing words from
            # label terms to avoid misleading labels like "Inconsistent Follow-Up
            # Care" when the cluster contains both positive and negative quotes
            label_terms = filtered_terms
            label_weights = filtered_weights
            if sentiment_result['is_mixed']:
                label_terms, label_weights = self._filter_sentiment_terms_for_label(
                    filtered_terms, filtered_weights
                )
                # Fall back to unfiltered terms if filtering removed everything
                if not label_terms:
                    label_terms = filtered_terms
                    label_weights = filtered_weights

            # Generate label using phrase-based or word-based approach
            if self.prefer_ngram_phrases:
                label = self._generate_phrase_based_label(
                    label_terms, label_weights, self.n_label_terms
                )
            else:
                label = self._generate_word_based_label(
                    label_terms, label_weights, self.n_label_terms
                )

            if not label:
                label = f"Cluster {cluster_idx} (low confidence)"

            # Get representative documents
            representative_docs = self._get_representative_docs(
                texts, doc_indices, feature_matrix, center
            )

            # Calculate interpretation confidence
            if filtered_terms:
                # Confidence based on term weight variance and count
                weight_sum = sum(filtered_weights)
                confidence = min(1.0, weight_sum / 2.0)  # Normalize
            else:
                confidence = 0.0

            # Calculate coherence score for this cluster if we have feature matrix
            cluster_coherence = None
            if feature_matrix is not None and len(doc_indices) >= 2:
                cluster_coherence = self._calculate_cluster_coherence(
                    feature_matrix, doc_indices
                )
                coherence_scores.append(cluster_coherence)

            # Check for interpretation issues
            warnings = []
            is_interpretable = True

            if not filtered_terms:
                warnings.append("No terms above weight threshold")
                is_interpretable = False
            elif len(filtered_terms) < 2:
                # Reduced penalty threshold for survey-sized datasets
                warnings.append("Few distinctive terms")
                confidence *= 0.9
            if doc_count == 0:
                warnings.append("Empty cluster")
                is_interpretable = False
            elif doc_count < 3:
                # Reduced penalty threshold for survey-sized datasets (200-500 responses)
                warnings.append("Very small cluster (<3 docs)")
                confidence *= 0.95

            # Add coherence-based warnings
            if cluster_coherence is not None and cluster_coherence < self.min_coherence_threshold:
                warnings.append(f"Low coherence ({cluster_coherence:.2f}) - cluster may be poorly defined")
                confidence *= 0.85

            # Add warning for mixed-sentiment clusters
            if sentiment_result['is_mixed']:
                pos_pct = sentiment_result['distribution']['positive']
                neg_pct = sentiment_result['distribution']['negative']
                warnings.append(
                    f"Mixed sentiment ({pos_pct:.0%} positive, {neg_pct:.0%} negative) - "
                    "cluster contains conflicting views on same topic. "
                    "Consider reviewing representative quotes for label accuracy."
                )
                confidence *= 0.90  # Slight penalty for mixed sentiment

            interpretability_scores.append(confidence)

            # Create cluster ID string
            cluster_id_str = f"CLUSTER_{cluster_idx + 1:02d}"

            # Create summary
            summary = ClusterSummary(
                cluster_id=cluster_id_str,
                label=label,
                top_terms=filtered_terms,
                term_weights=filtered_weights,
                document_count=doc_count,
                document_indices=doc_indices,
                representative_docs=representative_docs,
                coherence_score=cluster_coherence,
                interpretation_confidence=confidence,
                is_interpretable=is_interpretable,
                warnings=warnings,
                sentiment_distribution=sentiment_result['distribution'],
                dominant_sentiment=sentiment_result['dominant'],
                sentiment_coherence=sentiment_result['coherence'],
                has_mixed_sentiment=sentiment_result['is_mixed']
            )

            summaries[cluster_id_str] = summary

        # Calculate overall interpretability
        overall_interpretability = np.mean(interpretability_scores) if interpretability_scores else 0.0

        # Calculate overall coherence
        overall_coherence = np.mean(coherence_scores) if coherence_scores else None

        # Add global warnings
        if overall_interpretability < self.low_interpretability_threshold:
            global_warnings.append(
                f"Low overall interpretability ({overall_interpretability:.1%}). "
                "Consider adjusting clustering parameters or preprocessing."
            )

        empty_clusters = sum(1 for s in summaries.values() if s.document_count == 0)
        if empty_clusters > 0:
            global_warnings.append(f"{empty_clusters} empty cluster(s) detected")

        # Add coherence-based recommendations
        if overall_coherence is not None and overall_coherence < self.min_coherence_threshold:
            global_warnings.append(
                f"Low overall coherence ({overall_coherence:.2f}). "
                "Recommended actions: reduce cluster count, adjust vectorizer parameters, "
                "or review preprocessing. Clusters may have overlapping themes."
            )

        # Count low-coherence clusters for additional recommendations
        low_coherence_count = sum(
            1 for s in summaries.values()
            if s.coherence_score is not None and s.coherence_score < self.min_coherence_threshold
        )
        if low_coherence_count > len(summaries) * 0.3:  # >30% of clusters have low coherence
            global_warnings.append(
                f"{low_coherence_count} clusters have low coherence. "
                "Consider: (1) reducing n_clusters, (2) using different preprocessing, "
                "(3) trying alternative clustering methods like LDA or NMF."
            )

        # Count mixed-sentiment clusters and add warning
        mixed_sentiment_clusters = [
            s.cluster_id for s in summaries.values() if s.has_mixed_sentiment
        ]
        if mixed_sentiment_clusters:
            cluster_list = ", ".join(mixed_sentiment_clusters[:5])
            if len(mixed_sentiment_clusters) > 5:
                cluster_list += f" and {len(mixed_sentiment_clusters) - 5} more"
            global_warnings.append(
                f"{len(mixed_sentiment_clusters)} cluster(s) have mixed sentiment "
                f"(both positive and negative views): {cluster_list}. "
                "These clusters group by topic, not sentiment. Review labels for accuracy."
            )

        # Generate tuning recommendations based on analysis
        tuning_recommendations = self._generate_tuning_recommendations(
            summaries=summaries,
            overall_interpretability=overall_interpretability,
            overall_coherence=overall_coherence,
            n_clusters=len(unique_clusters)
        )

        # Create report
        report = ClusterInterpretationReport(
            summaries=summaries,
            n_clusters=len(unique_clusters),
            n_documents=len(texts),
            overall_interpretability=overall_interpretability,
            method_used=method_name,
            warnings=global_warnings,
            overall_coherence=overall_coherence,
            tuning_recommendations=tuning_recommendations,
            domain_stopwords_used=self._domain_stopwords
        )

        coherence_str = f", coherence={overall_coherence:.2f}" if overall_coherence else ""
        logger.info(
            f"Cluster interpretation complete: {len(summaries)} clusters, "
            f"interpretability={overall_interpretability:.1%}{coherence_str}"
        )

        return report

    def _generate_tuning_recommendations(
        self,
        summaries: Dict[str, ClusterSummary],
        overall_interpretability: float,
        overall_coherence: Optional[float],
        n_clusters: int
    ) -> List[str]:
        """
        Generate automated tuning recommendations based on cluster analysis.

        Args:
            summaries: Dictionary of cluster summaries
            overall_interpretability: Average interpretability score
            overall_coherence: Average coherence score (or None)
            n_clusters: Number of clusters

        Returns:
            List of actionable recommendations
        """
        recommendations = []

        # Low interpretability recommendations
        if overall_interpretability < self.low_interpretability_threshold:
            recommendations.append(
                "Low interpretability detected. Try: (a) increase n_top_terms, "
                "(b) lower min_term_weight_threshold, (c) check text preprocessing"
            )

        # Low coherence recommendations
        if overall_coherence is not None and overall_coherence < self.min_coherence_threshold:
            if n_clusters > 5:
                recommendations.append(
                    f"Low coherence ({overall_coherence:.2f}). Consider reducing cluster count "
                    f"from {n_clusters} to {max(3, n_clusters - 2)}"
                )
            else:
                recommendations.append(
                    f"Low coherence ({overall_coherence:.2f}). Try: (a) different preprocessing, "
                    "(b) topic modeling (LDA/NMF) instead of KMeans"
                )

        # Check for empty or very small clusters
        empty_count = sum(1 for s in summaries.values() if s.document_count == 0)
        tiny_count = sum(1 for s in summaries.values() if 0 < s.document_count < 3)

        if empty_count > 0:
            recommendations.append(
                f"{empty_count} empty cluster(s). Reduce n_clusters parameter"
            )
        if tiny_count > n_clusters * 0.3:  # >30% tiny clusters
            recommendations.append(
                f"{tiny_count} very small clusters (<3 docs). Consider merging or reducing n_clusters"
            )

        # Check for low confidence clusters
        low_conf_count = sum(
            1 for s in summaries.values()
            if s.interpretation_confidence < self.low_interpretability_threshold
        )
        if low_conf_count > n_clusters * 0.3:
            recommendations.append(
                f"{low_conf_count} clusters have low confidence. Review vectorizer settings "
                "or increase n_top_terms for better term extraction"
            )

        # Check for clusters with warnings
        clusters_with_warnings = sum(1 for s in summaries.values() if s.warnings)
        if clusters_with_warnings > n_clusters * 0.5:
            recommendations.append(
                f"{clusters_with_warnings}/{n_clusters} clusters have warnings. "
                "Run with feature_matrix for coherence analysis"
            )

        # Check for mixed-sentiment clusters
        mixed_sentiment_count = sum(1 for s in summaries.values() if s.has_mixed_sentiment)
        if mixed_sentiment_count > 0:
            recommendations.append(
                f"{mixed_sentiment_count} cluster(s) have mixed sentiment (both positive and negative). "
                "This is expected behavior - clustering groups by topic, not sentiment. "
                "Options: (a) manually review and rename labels, "
                "(b) increase n_clusters to separate positive/negative views, "
                "(c) run sentiment-aware post-processing to split mixed clusters"
            )

        return recommendations

    def _get_representative_docs(
        self,
        texts: List[str],
        doc_indices: List[int],
        feature_matrix,
        center
    ) -> List[Dict[str, Any]]:
        """
        Get representative documents for a cluster.

        Selects documents that are closest to the cluster center
        (most representative of the cluster).
        """
        if not doc_indices:
            return []

        n_docs = min(self.n_representative_docs, len(doc_indices))

        if feature_matrix is not None:
            # Calculate distances to center
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                doc_vectors = feature_matrix[doc_indices]
                if hasattr(doc_vectors, 'toarray'):
                    doc_vectors = doc_vectors.toarray()

                center_reshaped = center.reshape(1, -1)
                similarities = cosine_similarity(doc_vectors, center_reshaped).flatten()

                # Get top N most similar
                top_indices = similarities.argsort()[-n_docs:][::-1]
                selected_indices = [doc_indices[i] for i in top_indices]
                selected_similarities = [float(similarities[i]) for i in top_indices]
            except Exception as e:
                logger.warning(f"Could not compute document similarities: {e}")
                # Fall back to random selection
                selected_indices = doc_indices[:n_docs]
                selected_similarities = [None] * n_docs
        else:
            # Random selection if no feature matrix
            selected_indices = doc_indices[:n_docs]
            selected_similarities = [None] * n_docs

        representative_docs = []
        for idx, sim in zip(selected_indices, selected_similarities):
            doc = {
                'index': idx,
                'text': str(texts[idx])[:500],  # Truncate for display
                'similarity': sim
            }
            representative_docs.append(doc)

        return representative_docs

    def _calculate_cluster_coherence(
        self,
        feature_matrix,
        doc_indices: List[int],
        max_pairs: int = 100
    ) -> float:
        """
        Calculate semantic coherence for a single cluster.

        Coherence is measured as the average pairwise cosine similarity
        between documents in the cluster. Higher coherence indicates
        that documents in the cluster are more similar to each other.

        Args:
            feature_matrix: TF-IDF or other feature matrix
            doc_indices: Indices of documents in this cluster
            max_pairs: Maximum number of pairs to sample (for efficiency)

        Returns:
            Coherence score between 0 and 1 (higher is better)
        """
        if len(doc_indices) < 2:
            return 1.0  # Single document clusters are perfectly coherent by definition

        try:
            from scipy.spatial.distance import cosine

            # Get vectors for this cluster
            vectors = feature_matrix[doc_indices]
            if hasattr(vectors, 'toarray'):
                vectors = vectors.toarray()

            n_docs = len(doc_indices)
            n_pairs = n_docs * (n_docs - 1) // 2

            # Sample pairs if too many - use efficient direct sampling
            # instead of generating all pairs first (which is O(n²) memory)
            if n_pairs > max_pairs:
                # Generate random pairs directly without creating full list
                import random
                sampled_pairs = set()
                while len(sampled_pairs) < max_pairs:
                    i = random.randint(0, n_docs - 1)
                    j = random.randint(0, n_docs - 1)
                    if i != j:
                        # Ensure consistent ordering (smaller index first)
                        pair = (min(i, j), max(i, j))
                        sampled_pairs.add(pair)
                sampled_pairs = list(sampled_pairs)
            else:
                sampled_pairs = [(i, j) for i in range(n_docs) for j in range(i + 1, n_docs)]

            # Calculate pairwise similarities
            similarities = []
            for i, j in sampled_pairs:
                sim = 1 - cosine(vectors[i], vectors[j])
                if not np.isnan(sim):
                    similarities.append(sim)

            if similarities:
                return float(np.mean(similarities))
            return 0.0

        except Exception as e:
            logger.warning(f"Could not calculate cluster coherence: {e}")
            return 0.0

    def _analyze_cluster_sentiment(
        self,
        texts: List[str],
        doc_indices: List[int],
        mixed_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Analyze sentiment distribution within a cluster.

        This helps detect clusters where the topic-based label may be misleading
        because the cluster contains both positive and negative sentiments about
        the same topic (e.g., "Declining Product Quality" containing both
        praise and criticism about quality).

        Args:
            texts: All original text documents
            doc_indices: Indices of documents in this cluster
            mixed_threshold: Minimum proportion to consider sentiment "present"
                            (default 0.25 = 25%). If both positive and negative
                            exceed this, cluster is flagged as mixed.

        Returns:
            Dictionary with:
                - distribution: {'positive': float, 'negative': float, 'neutral': float}
                - dominant: 'positive', 'negative', 'neutral', or 'mixed'
                - coherence: 0-1 score (higher = more uniform sentiment)
                - is_mixed: True if cluster has conflicting sentiments
        """
        if not SENTIMENT_ANALYSIS_AVAILABLE:
            logger.debug("Sentiment analysis not available for cluster sentiment distribution")
            return {
                'distribution': None,
                'dominant': None,
                'coherence': None,
                'is_mixed': False
            }

        if not doc_indices or len(doc_indices) < 2:
            return {
                'distribution': None,
                'dominant': None,
                'coherence': None,
                'is_mixed': False
            }

        try:
            # Get texts for this cluster
            cluster_texts = [texts[i] for i in doc_indices]

            # Use VADER for fast sentiment analysis (no GPU required)
            analyzer = SurveySentimentAnalyzer(method='vader')
            results = analyzer.analyze(cluster_texts)

            # Count sentiments
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            for result in results:
                sentiment_counts[result.label] += 1

            # Calculate distribution
            total = len(results)
            distribution = {
                k: v / total for k, v in sentiment_counts.items()
            }

            # Determine dominant sentiment
            pos_pct = distribution['positive']
            neg_pct = distribution['negative']
            neu_pct = distribution['neutral']

            # Check for mixed sentiment (both positive and negative above threshold)
            is_mixed = pos_pct >= mixed_threshold and neg_pct >= mixed_threshold

            if is_mixed:
                dominant = 'mixed'
            else:
                # Find the dominant sentiment
                max_sentiment = max(distribution, key=distribution.get)
                dominant = max_sentiment

            # Calculate sentiment coherence (how uniform is the sentiment?)
            # Uses entropy-based measure: coherence = 1 - normalized_entropy
            # Perfect uniformity (all same sentiment) = 1.0
            # Perfect chaos (33% each) = 0.0
            proportions = np.array([pos_pct, neg_pct, neu_pct])
            proportions = proportions[proportions > 0]  # Remove zeros for log
            if len(proportions) > 1:
                entropy = -np.sum(proportions * np.log2(proportions))
                max_entropy = np.log2(3)  # Maximum entropy for 3 categories
                coherence = 1.0 - (entropy / max_entropy)
            else:
                coherence = 1.0  # Only one sentiment present = perfect coherence

            return {
                'distribution': distribution,
                'dominant': dominant,
                'coherence': float(coherence),
                'is_mixed': is_mixed
            }

        except Exception as e:
            logger.warning(f"Could not analyze cluster sentiment: {e}")
            return {
                'distribution': None,
                'dominant': None,
                'coherence': None,
                'is_mixed': False
            }

    def apply_llm_enhancement(
        self,
        report: ClusterInterpretationReport,
        texts: List[str],
        cluster_assignments: List[int]
    ) -> ClusterInterpretationReport:
        """
        Apply LLM-based enhancement to cluster labels and descriptions.

        This method uses the LLM interpretation module to generate more
        human-readable labels, alternative suggestions, and detailed
        descriptions for each cluster.

        Args:
            report: Existing cluster interpretation report
            texts: Original text documents
            cluster_assignments: Cluster assignment for each document

        Returns:
            Enhanced ClusterInterpretationReport with LLM-generated content

        Notes:
            - Gracefully falls back to term-based labels if LLM unavailable
            - Uses Mistral API first, then local model, then fallback
            - Does not modify the original report, returns a new one
        """
        try:
            from src.llm_interpretation import LLMClusterInterpreter
        except ImportError:
            logger.warning("LLM interpretation module not available")
            return report

        # Initialize LLM interpreter
        llm_interpreter = LLMClusterInterpreter()

        if not llm_interpreter.is_available:
            logger.info("LLM interpretation not available, using term-based labels")
            return report

        logger.info(f"Applying LLM enhancement using {llm_interpreter.backend_type} backend")

        # Enhance all clusters
        enhanced_labels = llm_interpreter.enhance_all_clusters(
            cluster_summaries=report.summaries,
            texts=texts,
            cluster_assignments=cluster_assignments
        )

        # Update summaries with LLM-enhanced content
        for cluster_id, enhanced in enhanced_labels.items():
            if cluster_id in report.summaries:
                summary = report.summaries[cluster_id]
                summary.llm_label = enhanced.primary_label
                summary.llm_alternative_labels = enhanced.alternative_labels
                summary.llm_description = enhanced.description
                summary.llm_reasoning = enhanced.reasoning
                summary.llm_source = enhanced.source

        # Update report metadata
        report.llm_enhanced = True
        report.llm_backend = llm_interpreter.backend_type

        logger.info(f"LLM enhancement complete for {len(enhanced_labels)} clusters")
        return report

    def get_cluster_comparison(
        self,
        report: ClusterInterpretationReport
    ) -> pd.DataFrame:
        """
        Create a comparison table of all clusters.

        Args:
            report: Cluster interpretation report

        Returns:
            DataFrame with cluster comparison
        """
        rows = []
        for cluster_id, summary in report.summaries.items():
            row = {
                'Cluster ID': cluster_id,
                'Label': summary.display_label,  # Use LLM label if available
                'Documents': summary.document_count,
                'Top Terms': ', '.join(summary.top_terms[:5]),
                'Confidence': f"{summary.interpretation_confidence:.1%}",
                'Interpretable': 'Yes' if summary.is_interpretable else 'No',
                'Warnings': '; '.join(summary.warnings) if summary.warnings else ''
            }
            # Add LLM-specific columns if enhanced
            if report.llm_enhanced and summary.llm_label:
                row['Term-Based Label'] = summary.label
                row['LLM Description'] = summary.llm_description or ''
                row['Alternative Labels'] = ', '.join(summary.llm_alternative_labels[:3])
                row['LLM Source'] = summary.llm_source or ''
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values('Documents', ascending=False)
        return df

    def discover_domain_stopwords(
        self,
        texts: List[str],
        min_doc_frequency: float = 0.7,
        cluster_assignments: Optional[List[int]] = None
    ) -> Optional['StopwordDiscoveryReport']:
        """
        Discover domain stopword candidates from the corpus.

        This method analyzes the corpus to identify words that appear frequently
        across documents and may be candidates for domain-specific stopwords.

        The workflow for using this method:
        1. Run this method to get a discovery report
        2. Review the candidates in the report
        3. Add appropriate words to data/stopwords_domain.txt
        4. Add words to keep (semantic meaning) to data/stopwords_keep.txt
        5. Re-run your analysis pipeline

        Args:
            texts: List of document texts to analyze
            min_doc_frequency: Minimum document frequency (0-1) threshold.
                              Default 0.7 means words in 70%+ of documents.
            cluster_assignments: Optional cluster assignments for cluster-aware
                               discovery (identifies words across all clusters)

        Returns:
            StopwordDiscoveryReport with candidates and recommendations,
            or None if stopwords_discovery module is not available

        Example:
            >>> interpreter = ClusterInterpreter()
            >>> report = interpreter.discover_domain_stopwords(texts)
            >>> print(report.to_markdown())
            >>> # Review candidates, then add to data/stopwords_domain.txt
        """
        if not STOPWORDS_DISCOVERY_AVAILABLE:
            logger.warning(
                "Stopwords discovery module not available. "
                "Cannot run domain stopword discovery."
            )
            return None

        if cluster_assignments is not None:
            # Use cluster-aware discovery
            from .stopwords_discovery import discover_from_clusters
            report = discover_from_clusters(
                texts=texts,
                cluster_assignments=cluster_assignments,
                min_doc_frequency=min_doc_frequency
            )
        else:
            # Use basic discovery
            report = find_domain_stopword_candidates(
                texts=texts,
                min_doc_frequency=min_doc_frequency,
                compute_tfidf_variance=True
            )

        logger.info(
            f"Domain stopword discovery complete: {len(report.candidates)} candidates found"
        )
        return report

    def get_stopwords_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all stopwords being used.

        Returns:
            Dictionary with stopword counts by source and sample words
        """
        # Ensure stopwords are loaded
        self._load_file_based_stopwords()

        summary = {
            'default_label_stopwords': len(DEFAULT_LABEL_STOPWORDS),
            'file_based_domain_stopwords': len(self._file_based_stopwords),
            'custom_stopwords': len(self.custom_stopwords),
            'dynamic_domain_stopwords': len(self._domain_stopwords),
            'keep_list_words': len(self._keep_list),
            'total_combined': len(self._get_combined_stopwords()),
            'samples': {
                'file_based': list(self._file_based_stopwords)[:10],
                'dynamic': list(self._domain_stopwords)[:10],
                'keep_list': list(self._keep_list)[:10],
            }
        }
        return summary


class ClusterCodebook:
    """
    Generates a formal codebook from cluster interpretations.

    This class creates a structured codebook suitable for qualitative research,
    with clear documentation that cluster codes are derived from unsupervised
    clustering, not predefined categories.

    Example:
        >>> codebook = ClusterCodebook(report)
        >>> print(codebook.to_markdown())
    """

    def __init__(self, interpretation_report: ClusterInterpretationReport):
        """
        Initialize codebook from interpretation report.

        Args:
            interpretation_report: Complete cluster interpretation report
        """
        self.report = interpretation_report

    def to_dict(self) -> Dict[str, Any]:
        """Convert codebook to dictionary format."""
        codes = {}
        for cluster_id, summary in self.report.summaries.items():
            # Include up to 5 representative quotes (or all if fewer than 5)
            representative_docs = summary.representative_docs
            examples = [d['text'] for d in representative_docs[:5]]

            # Use LLM-enhanced content if available
            if summary.llm_label:
                name = summary.llm_label
                definition = summary.llm_description or f"Documents characterized by: {', '.join(summary.top_terms[:5])}"
            else:
                name = summary.label
                definition = f"Documents characterized by: {', '.join(summary.top_terms[:5])}"

            codes[cluster_id] = {
                'name': name,
                'term_based_name': summary.label,
                'definition': definition,
                'keywords': summary.top_terms,
                'n_documents': summary.document_count,
                'examples': examples,
                'confidence': summary.interpretation_confidence,
                'notes': summary.interpretation_notes,
                'warnings': summary.warnings,
                # LLM-enhanced fields
                'llm_label': summary.llm_label,
                'llm_alternative_labels': summary.llm_alternative_labels,
                'llm_description': summary.llm_description,
                'llm_reasoning': summary.llm_reasoning,
                'llm_source': summary.llm_source
            }

        disclaimer = (
            "These codes were generated through unsupervised machine learning (clustering). "
            "Cluster IDs are arbitrary identifiers. "
        )
        if self.report.llm_enhanced:
            disclaimer += (
                f"Labels have been refined using LLM ({self.report.llm_backend}) for improved readability. "
            )
        else:
            disclaimer += "Labels are derived from term frequencies. "
        disclaimer += "Human review is recommended."

        return {
            'method': self.report.method_used,
            'n_clusters': self.report.n_clusters,
            'n_documents': self.report.n_documents,
            'llm_enhanced': self.report.llm_enhanced,
            'llm_backend': self.report.llm_backend,
            'codes': codes,
            'disclaimer': disclaimer
        }

    def to_markdown(self) -> str:
        """Generate markdown format codebook."""
        lines = [
            "# Cluster Codebook",
            "",
            f"**Method**: {self.report.method_used}",
            f"**Documents**: {self.report.n_documents}",
            f"**Codes Discovered**: {self.report.n_clusters}",
        ]

        if self.report.llm_enhanced:
            lines.append(f"**LLM Enhancement**: Enabled ({self.report.llm_backend})")
            lines.append("")
            lines.append("> **Note**: These codes were generated through unsupervised clustering. ")
            lines.append("> Labels have been refined by LLM for improved human readability. ")
            lines.append("> Alternative labels and descriptions are provided. Human validation recommended.")
        else:
            lines.append("")
            lines.append("> **Note**: These codes were generated through unsupervised clustering. ")
            lines.append("> Cluster IDs are internal identifiers, not semantic categories. ")
            lines.append("> Labels are derived from term frequencies. Human validation recommended.")

        lines.extend(["", "---", ""])

        # Sort by document count
        sorted_summaries = sorted(
            self.report.summaries.items(),
            key=lambda x: x[1].document_count,
            reverse=True
        )

        for cluster_id, summary in sorted_summaries:
            # Use LLM label if available
            display_label = summary.display_label
            lines.append(f"## {cluster_id}: {display_label}")
            lines.append("")

            # Show original term-based label if different
            if summary.llm_label and summary.llm_label != summary.label:
                lines.append(f"*Term-based label: {summary.label}*")
                lines.append("")

            lines.append(f"**Documents**: {summary.document_count}")
            lines.append(f"**Confidence**: {summary.interpretation_confidence:.1%}")
            lines.append("")

            # Show LLM description if available
            if summary.llm_description:
                lines.append(f"**Description**: {summary.llm_description}")
                lines.append("")

            # Show alternative labels if available
            if summary.llm_alternative_labels:
                alts = ", ".join(summary.llm_alternative_labels[:3])
                lines.append(f"**Alternative Labels**: {alts}")
                lines.append("")

            lines.append("**Keywords**: " + ", ".join(summary.top_terms[:7]))
            lines.append("")

            if summary.representative_docs:
                lines.append("**Example Documents**:")
                # Show up to 5 representative quotes (or all if fewer than 5)
                for doc in summary.representative_docs[:5]:
                    text_preview = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                    lines.append(f"> {text_preview}")
                    lines.append("")

            if summary.warnings:
                lines.append("**Warnings**: " + "; ".join(summary.warnings))
                lines.append("")

            # Show LLM reasoning if available
            if summary.llm_reasoning:
                lines.append(f"*LLM Reasoning: {summary.llm_reasoning}*")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Generate CSV format for the codebook."""
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Header - include LLM fields if enhanced
        if self.report.llm_enhanced:
            writer.writerow([
                'Cluster ID', 'LLM Label', 'Term-Based Label', 'Description',
                'Alternative Labels', 'Documents', 'Keywords', 'Confidence',
                'Example 1', 'Example 2', 'Example 3', 'Example 4', 'Example 5',
                'LLM Source', 'Warnings'
            ])
        else:
            writer.writerow([
                'Cluster ID', 'Label', 'Documents', 'Keywords',
                'Confidence', 'Example 1', 'Example 2', 'Example 3',
                'Example 4', 'Example 5', 'Warnings'
            ])

        for cluster_id, summary in self.report.summaries.items():
            # Get up to 5 examples (or all if fewer than 5)
            examples = [d['text'][:200] for d in summary.representative_docs[:5]]
            while len(examples) < 5:
                examples.append('')

            if self.report.llm_enhanced:
                writer.writerow([
                    cluster_id,
                    summary.llm_label or summary.label,
                    summary.label,
                    summary.llm_description or '',
                    '; '.join(summary.llm_alternative_labels[:3]),
                    summary.document_count,
                    '; '.join(summary.top_terms[:5]),
                    f"{summary.interpretation_confidence:.1%}",
                    examples[0],
                    examples[1],
                    examples[2],
                    examples[3],
                    examples[4],
                    summary.llm_source or '',
                    '; '.join(summary.warnings)
                ])
            else:
                writer.writerow([
                    cluster_id,
                    summary.label,
                    summary.document_count,
                    '; '.join(summary.top_terms[:5]),
                    f"{summary.interpretation_confidence:.1%}",
                    examples[0],
                    examples[1],
                    examples[2],
                    examples[3],
                    examples[4],
                    '; '.join(summary.warnings)
                ])

        return output.getvalue()
