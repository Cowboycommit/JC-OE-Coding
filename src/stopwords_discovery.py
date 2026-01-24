"""
Domain Stopwords Discovery Module.

This module provides utilities for discovering, loading, and managing
domain-specific stopwords in a reproducible and auditable manner.

The workflow follows professional NLP best practices:
1. Run analysis with only general stopwords (NLTK + spaCy)
2. Discover domain stopword candidates from corpus statistics
3. Curate and persist stopwords to data/stopwords_domain.txt
4. Use keep-list (data/stopwords_keep.txt) for semantic overrides
5. Re-run analysis with cleaner topic separation

Key functions:
    - find_domain_stopword_candidates(): Discover stopword candidates from data
    - load_stopwords_from_file(): Load stopwords from text file
    - load_keep_list(): Load semantic override words
    - get_layered_stopwords(): Get combined stopwords with governance model
    - generate_discovery_report(): Generate human-readable discovery report

This creates a clean governance model:
    Layer               Source
    -----               ------
    General stopwords   NLTK + spaCy
    Domain stopwords    data/stopwords_domain.txt (your dataset statistics)
    Semantic overrides  data/stopwords_keep.txt (words to never remove)
"""

import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default paths relative to project root
DEFAULT_DOMAIN_STOPWORDS_PATH = "data/stopwords_domain.txt"
DEFAULT_KEEP_LIST_PATH = "data/stopwords_keep.txt"


@dataclass
class StopwordCandidate:
    """
    A candidate word for domain stopword status.

    Attributes:
        word: The candidate word
        document_frequency: Fraction of documents containing this word (0-1)
        document_count: Number of documents containing this word
        total_occurrences: Total count across all documents
        tfidf_variance: Variance of TF-IDF scores across documents (if computed)
        topic_coverage: Fraction of topics/clusters containing this word (if computed)
        is_recommended: Whether this word is recommended for removal
        reason: Human-readable reason for recommendation
    """
    word: str
    document_frequency: float
    document_count: int
    total_occurrences: int = 0
    tfidf_variance: Optional[float] = None
    topic_coverage: Optional[float] = None
    is_recommended: bool = True
    reason: str = ""


@dataclass
class StopwordDiscoveryReport:
    """
    Report from domain stopword discovery analysis.

    Attributes:
        candidates: List of stopword candidates sorted by document frequency
        total_documents: Total number of documents analyzed
        total_vocabulary: Total unique words in corpus
        min_doc_frequency_threshold: Threshold used for candidate selection
        keep_list_applied: Words protected by keep-list
        recommendations: Human-readable recommendations
        statistics: Additional corpus statistics
    """
    candidates: List[StopwordCandidate]
    total_documents: int
    total_vocabulary: int
    min_doc_frequency_threshold: float
    keep_list_applied: Set[str] = field(default_factory=set)
    recommendations: List[str] = field(default_factory=list)
    statistics: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert report to dictionary for serialization."""
        return {
            'total_documents': self.total_documents,
            'total_vocabulary': self.total_vocabulary,
            'min_doc_frequency_threshold': self.min_doc_frequency_threshold,
            'n_candidates': len(self.candidates),
            'keep_list_size': len(self.keep_list_applied),
            'candidates': [
                {
                    'word': c.word,
                    'document_frequency': round(c.document_frequency, 4),
                    'document_count': c.document_count,
                    'total_occurrences': c.total_occurrences,
                    'tfidf_variance': round(c.tfidf_variance, 6) if c.tfidf_variance else None,
                    'topic_coverage': round(c.topic_coverage, 4) if c.topic_coverage else None,
                    'is_recommended': c.is_recommended,
                    'reason': c.reason
                }
                for c in self.candidates
            ],
            'recommendations': self.recommendations,
            'statistics': self.statistics
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Domain Stopwords Discovery Report",
            "",
            "## Corpus Statistics",
            f"- **Total Documents**: {self.total_documents:,}",
            f"- **Total Vocabulary**: {self.total_vocabulary:,}",
            f"- **Document Frequency Threshold**: {self.min_doc_frequency_threshold:.0%}",
            f"- **Candidates Found**: {len(self.candidates)}",
            f"- **Keep-List Size**: {len(self.keep_list_applied)}",
            "",
        ]

        if self.recommendations:
            lines.append("## Recommendations")
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        lines.extend([
            "## Candidate Stopwords",
            "",
            "| Rank | Word | Doc Frequency | Doc Count | Occurrences | Recommended | Reason |",
            "|------|------|---------------|-----------|-------------|-------------|--------|",
        ])

        for i, c in enumerate(self.candidates[:50], 1):  # Top 50
            rec_str = "Yes" if c.is_recommended else "No"
            lines.append(
                f"| {i} | {c.word} | {c.document_frequency:.1%} | "
                f"{c.document_count:,} | {c.total_occurrences:,} | {rec_str} | {c.reason} |"
            )

        if len(self.candidates) > 50:
            lines.append(f"\n*...and {len(self.candidates) - 50} more candidates*")

        lines.extend([
            "",
            "## How to Use This Report",
            "",
            "1. Review the candidates above",
            "2. For words you want to remove, add them to `data/stopwords_domain.txt`",
            "3. For words you want to keep (semantic meaning), add to `data/stopwords_keep.txt`",
            "4. Re-run your analysis pipeline",
            "",
            "### Criteria for Adding to Domain Stopwords",
            "- Appears in >70% of documents",
            "- Does not differentiate clusters/topics",
            "- Is operational/procedural language",
            "- Does NOT carry semantic meaning",
            "",
            "### Words to Keep (Never Remove)",
            "- Negation words (not, no, never)",
            "- Sentiment-bearing words",
            "- Domain-specific but informative terms",
        ])

        return "\n".join(lines)


def _get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find project root by looking for common markers
    current = Path(__file__).resolve().parent

    for _ in range(5):  # Look up to 5 levels
        if (current / "data").exists() or (current / "src").exists():
            return current
        current = current.parent

    # Fallback to parent of src directory
    return Path(__file__).resolve().parent.parent


def load_stopwords_from_file(
    file_path: Optional[Union[str, Path]] = None,
    use_default: bool = True
) -> Set[str]:
    """
    Load stopwords from a text file.

    Args:
        file_path: Path to stopwords file. If None and use_default=True,
                   uses data/stopwords_domain.txt
        use_default: Whether to use default path if file_path is None

    Returns:
        Set of stopwords (lowercase, stripped)

    Notes:
        - Lines starting with # are treated as comments
        - Empty lines are ignored
        - All words are lowercased
        - Returns empty set if file doesn't exist (graceful degradation)
    """
    if file_path is None:
        if use_default:
            file_path = _get_project_root() / DEFAULT_DOMAIN_STOPWORDS_PATH
        else:
            return set()

    file_path = Path(file_path)

    if not file_path.exists():
        logger.debug(f"Stopwords file not found: {file_path}")
        return set()

    stopwords = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    stopwords.add(line.lower())

        logger.info(f"Loaded {len(stopwords)} stopwords from {file_path}")
    except Exception as e:
        logger.warning(f"Error loading stopwords from {file_path}: {e}")

    return stopwords


def load_keep_list(
    file_path: Optional[Union[str, Path]] = None,
    use_default: bool = True
) -> Set[str]:
    """
    Load keep-list (semantic override words) from a text file.

    These words will never be removed as stopwords, even if they
    appear frequently in the corpus.

    Args:
        file_path: Path to keep-list file. If None and use_default=True,
                   uses data/stopwords_keep.txt
        use_default: Whether to use default path if file_path is None

    Returns:
        Set of words to keep (lowercase, stripped)
    """
    if file_path is None:
        if use_default:
            file_path = _get_project_root() / DEFAULT_KEEP_LIST_PATH
        else:
            return set()

    file_path = Path(file_path)

    if not file_path.exists():
        logger.debug(f"Keep-list file not found: {file_path}")
        return set()

    keep_words = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    keep_words.add(line.lower())

        logger.info(f"Loaded {len(keep_words)} keep-list words from {file_path}")
    except Exception as e:
        logger.warning(f"Error loading keep-list from {file_path}: {e}")

    return keep_words


def get_nltk_stopwords() -> Set[str]:
    """
    Get NLTK English stopwords.

    Returns:
        Set of NLTK stopwords, or empty set if NLTK unavailable
    """
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words('english'))
    except Exception as e:
        logger.warning(f"Could not load NLTK stopwords: {e}")
        return set()


def get_layered_stopwords(
    include_nltk: bool = True,
    include_domain: bool = True,
    include_custom: Optional[Set[str]] = None,
    domain_stopwords_path: Optional[Union[str, Path]] = None,
    keep_list_path: Optional[Union[str, Path]] = None
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Get combined stopwords using the layered governance model.

    The layers are:
    1. NLTK general stopwords (if include_nltk=True)
    2. Domain stopwords from file (if include_domain=True)
    3. Custom stopwords (if provided)
    4. Minus keep-list words (semantic overrides)

    Args:
        include_nltk: Include NLTK English stopwords
        include_domain: Include domain stopwords from file
        include_custom: Optional set of additional custom stopwords
        domain_stopwords_path: Custom path to domain stopwords file
        keep_list_path: Custom path to keep-list file

    Returns:
        Tuple of (combined_stopwords, layer_counts) where layer_counts
        shows how many words came from each source
    """
    combined = set()
    layer_counts = {
        'nltk': 0,
        'domain': 0,
        'custom': 0,
        'keep_list_removed': 0
    }

    # Layer 1: NLTK stopwords
    if include_nltk:
        nltk_words = get_nltk_stopwords()
        combined.update(nltk_words)
        layer_counts['nltk'] = len(nltk_words)

    # Layer 2: Domain stopwords from file
    if include_domain:
        domain_words = load_stopwords_from_file(domain_stopwords_path)
        layer_counts['domain'] = len(domain_words)
        combined.update(domain_words)

    # Layer 3: Custom stopwords
    if include_custom:
        layer_counts['custom'] = len(include_custom)
        combined.update(include_custom)

    # Layer 4: Remove keep-list words (semantic overrides)
    keep_list = load_keep_list(keep_list_path)
    before_keep = len(combined)
    combined -= keep_list
    layer_counts['keep_list_removed'] = before_keep - len(combined)

    logger.info(
        f"Layered stopwords: {len(combined)} total "
        f"(NLTK: {layer_counts['nltk']}, Domain: {layer_counts['domain']}, "
        f"Custom: {layer_counts['custom']}, Keep-list removed: {layer_counts['keep_list_removed']})"
    )

    return combined, layer_counts


def tokenize_simple(text: str, min_length: int = 2) -> List[str]:
    """
    Simple tokenization for stopword discovery.

    Args:
        text: Input text
        min_length: Minimum token length

    Returns:
        List of lowercase tokens
    """
    if not text or pd.isna(text):
        return []

    # Simple word tokenization
    import re
    tokens = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
    return [t for t in tokens if len(t) >= min_length]


def find_domain_stopword_candidates(
    texts: List[str],
    min_doc_frequency: float = 0.7,
    max_candidates: int = 100,
    exclude_general_stopwords: bool = True,
    apply_keep_list: bool = True,
    compute_tfidf_variance: bool = True,
    tokenizer: Optional[callable] = None
) -> StopwordDiscoveryReport:
    """
    Discover domain stopword candidates from corpus statistics.

    This function analyzes document frequency and other statistics to
    identify words that are candidates for domain-specific stopwords.

    Args:
        texts: List of document texts
        min_doc_frequency: Minimum document frequency (0-1) to consider
                          a word as a stopword candidate. Default 0.7 means
                          words appearing in 70%+ of documents.
        max_candidates: Maximum number of candidates to return
        exclude_general_stopwords: Exclude NLTK stopwords from candidates
        apply_keep_list: Apply keep-list to filter candidates
        compute_tfidf_variance: Compute TF-IDF variance for each candidate
        tokenizer: Optional custom tokenizer function. If None, uses simple
                  word tokenization.

    Returns:
        StopwordDiscoveryReport with candidates and statistics

    Example:
        >>> texts = ["The patient reported feeling better after treatment.",
        ...          "Patient satisfaction with the treatment was high.",
        ...          "The treatment plan was adjusted for the patient."]
        >>> report = find_domain_stopword_candidates(texts, min_doc_frequency=0.6)
        >>> for c in report.candidates[:5]:
        ...     print(f"{c.word}: {c.document_frequency:.0%}")
        patient: 100%
        treatment: 100%
        the: 100%
    """
    if not texts:
        return StopwordDiscoveryReport(
            candidates=[],
            total_documents=0,
            total_vocabulary=0,
            min_doc_frequency_threshold=min_doc_frequency
        )

    # Use custom tokenizer or default
    tokenize = tokenizer or tokenize_simple

    # Get stopwords to exclude
    general_stopwords = get_nltk_stopwords() if exclude_general_stopwords else set()
    keep_list = load_keep_list() if apply_keep_list else set()

    # Count document frequency and total occurrences
    doc_count = len(texts)
    term_doc_freq = Counter()  # How many documents contain each term
    term_total_freq = Counter()  # Total occurrences across all documents
    doc_term_counts = []  # For TF-IDF variance calculation

    for text in texts:
        tokens = tokenize(text)
        unique_tokens = set(tokens)

        # Document frequency
        for token in unique_tokens:
            term_doc_freq[token] += 1

        # Total frequency
        term_total_freq.update(tokens)

        # Store per-document counts for variance calculation
        if compute_tfidf_variance:
            doc_term_counts.append(Counter(tokens))

    # Calculate vocabulary size
    total_vocabulary = len(term_doc_freq)

    # Find candidates meeting threshold
    candidates = []
    for term, df_count in term_doc_freq.items():
        doc_freq = df_count / doc_count

        if doc_freq >= min_doc_frequency:
            # Skip general stopwords (they're already handled)
            if term in general_stopwords:
                continue

            # Check if in keep-list
            is_in_keep_list = term in keep_list

            # Determine recommendation
            is_recommended = not is_in_keep_list
            reason = ""

            if is_in_keep_list:
                reason = "Protected by keep-list"
                is_recommended = False
            elif doc_freq >= 0.9:
                reason = "Very high frequency (>90%)"
            elif doc_freq >= 0.8:
                reason = "High frequency (>80%)"
            else:
                reason = f"Frequent ({doc_freq:.0%} of docs)"

            candidate = StopwordCandidate(
                word=term,
                document_frequency=doc_freq,
                document_count=df_count,
                total_occurrences=term_total_freq[term],
                is_recommended=is_recommended,
                reason=reason
            )
            candidates.append(candidate)

    # Compute TF-IDF variance if requested
    if compute_tfidf_variance and candidates and doc_term_counts:
        _compute_tfidf_variance(candidates, doc_term_counts, term_doc_freq, doc_count)

    # Sort by document frequency (highest first)
    candidates.sort(key=lambda x: (-x.document_frequency, -x.total_occurrences))

    # Limit candidates
    candidates = candidates[:max_candidates]

    # Generate recommendations
    recommendations = _generate_recommendations(
        candidates, doc_count, total_vocabulary, min_doc_frequency
    )

    # Compile statistics
    statistics = {
        'avg_doc_length': np.mean([len(tokenize(t)) for t in texts]),
        'high_freq_terms': len([c for c in candidates if c.document_frequency >= 0.9]),
        'protected_by_keep_list': len([c for c in candidates if not c.is_recommended]),
    }

    return StopwordDiscoveryReport(
        candidates=candidates,
        total_documents=doc_count,
        total_vocabulary=total_vocabulary,
        min_doc_frequency_threshold=min_doc_frequency,
        keep_list_applied=keep_list,
        recommendations=recommendations,
        statistics=statistics
    )


def _compute_tfidf_variance(
    candidates: List[StopwordCandidate],
    doc_term_counts: List[Counter],
    term_doc_freq: Counter,
    n_docs: int
) -> None:
    """
    Compute TF-IDF variance for each candidate.

    Low variance indicates the term has similar importance across all documents,
    making it a stronger stopword candidate.
    """
    # Compute IDF for each candidate term
    candidate_words = {c.word for c in candidates}
    idf = {}
    for term in candidate_words:
        df = term_doc_freq[term]
        idf[term] = np.log(n_docs / (df + 1)) + 1  # Smoothed IDF

    # Compute TF-IDF for each document
    tfidf_values = {term: [] for term in candidate_words}

    for doc_counts in doc_term_counts:
        doc_len = sum(doc_counts.values())
        if doc_len == 0:
            continue

        for term in candidate_words:
            tf = doc_counts.get(term, 0) / doc_len
            tfidf = tf * idf[term]
            tfidf_values[term].append(tfidf)

    # Compute variance for each candidate
    for candidate in candidates:
        values = tfidf_values.get(candidate.word, [])
        if values:
            candidate.tfidf_variance = float(np.var(values))


def _generate_recommendations(
    candidates: List[StopwordCandidate],
    doc_count: int,
    vocab_size: int,
    threshold: float
) -> List[str]:
    """Generate human-readable recommendations based on analysis."""
    recommendations = []

    # Count high-frequency candidates
    very_high_freq = [c for c in candidates if c.document_frequency >= 0.9]
    high_freq = [c for c in candidates if 0.8 <= c.document_frequency < 0.9]

    if very_high_freq:
        words = ", ".join(c.word for c in very_high_freq[:5])
        extra = f" (+{len(very_high_freq) - 5} more)" if len(very_high_freq) > 5 else ""
        recommendations.append(
            f"Found {len(very_high_freq)} words appearing in >90% of documents: {words}{extra}. "
            "These are strong candidates for domain stopwords."
        )

    if high_freq:
        words = ", ".join(c.word for c in high_freq[:5])
        recommendations.append(
            f"Found {len(high_freq)} words appearing in 80-90% of documents. "
            "Review these carefully - some may carry semantic meaning."
        )

    # Check for potential semantic words in candidates
    semantic_patterns = {'good', 'bad', 'great', 'poor', 'better', 'worse', 'best', 'worst'}
    semantic_candidates = [c for c in candidates if c.word in semantic_patterns]
    if semantic_candidates:
        words = ", ".join(c.word for c in semantic_candidates)
        recommendations.append(
            f"Warning: Found potential sentiment words in candidates: {words}. "
            "Consider adding these to the keep-list if sentiment analysis is important."
        )

    # General advice
    if len(candidates) > 20:
        recommendations.append(
            f"Found {len(candidates)} total candidates. Start with the top 10-15 and "
            "incrementally add more while monitoring cluster quality."
        )
    elif len(candidates) < 5:
        recommendations.append(
            f"Found only {len(candidates)} candidates. Your corpus may already be well-filtered, "
            "or try lowering the threshold to 0.6 for more candidates."
        )

    return recommendations


def discover_from_clusters(
    texts: List[str],
    cluster_assignments: List[int],
    min_doc_frequency: float = 0.7,
    min_cluster_coverage: float = 0.8
) -> StopwordDiscoveryReport:
    """
    Discover domain stopwords using cluster analysis.

    This method identifies words that appear across most clusters,
    suggesting they don't help differentiate topics.

    Args:
        texts: List of document texts
        cluster_assignments: Cluster assignment for each document
        min_doc_frequency: Minimum document frequency threshold
        min_cluster_coverage: Minimum fraction of clusters a word must
                             appear in to be considered a candidate

    Returns:
        StopwordDiscoveryReport with cluster-aware candidates
    """
    # First get basic candidates
    report = find_domain_stopword_candidates(
        texts,
        min_doc_frequency=min_doc_frequency,
        compute_tfidf_variance=True
    )

    if not report.candidates:
        return report

    # Analyze cluster coverage
    unique_clusters = set(cluster_assignments)
    n_clusters = len(unique_clusters)

    # Group texts by cluster
    cluster_texts = {c: [] for c in unique_clusters}
    for text, cluster in zip(texts, cluster_assignments):
        cluster_texts[cluster].append(text)

    # For each candidate, count how many clusters contain it
    for candidate in report.candidates:
        clusters_with_term = 0
        for cluster_id, cluster_docs in cluster_texts.items():
            # Check if term appears in any document in this cluster
            for doc in cluster_docs:
                if candidate.word in tokenize_simple(doc):
                    clusters_with_term += 1
                    break

        candidate.topic_coverage = clusters_with_term / n_clusters

        # Update recommendation based on cluster coverage
        if candidate.topic_coverage >= min_cluster_coverage:
            if candidate.is_recommended:
                candidate.reason += f"; appears in {candidate.topic_coverage:.0%} of clusters"
        else:
            candidate.is_recommended = False
            candidate.reason = f"Only in {candidate.topic_coverage:.0%} of clusters - may be discriminative"

    # Re-sort considering cluster coverage
    report.candidates.sort(
        key=lambda x: (-x.document_frequency * (x.topic_coverage or 1), -x.total_occurrences)
    )

    # Add cluster-specific recommendation
    report.recommendations.insert(0,
        f"Analyzed coverage across {n_clusters} clusters. "
        "Words appearing in most clusters are stronger stopword candidates."
    )

    return report


def save_candidates_to_file(
    candidates: List[StopwordCandidate],
    file_path: Union[str, Path],
    only_recommended: bool = True,
    include_header: bool = True
) -> int:
    """
    Save stopword candidates to a file.

    Args:
        candidates: List of StopwordCandidate objects
        file_path: Output file path
        only_recommended: Only save recommended candidates
        include_header: Include explanatory header in file

    Returns:
        Number of words saved
    """
    file_path = Path(file_path)

    words_to_save = [
        c.word for c in candidates
        if not only_recommended or c.is_recommended
    ]

    with open(file_path, 'w', encoding='utf-8') as f:
        if include_header:
            f.write("# Domain stopwords discovered from corpus analysis\n")
            f.write(f"# Generated with min_doc_frequency threshold\n")
            f.write(f"# Total candidates: {len(words_to_save)}\n")
            f.write("#\n")
            f.write("# Review each word before using in production!\n")
            f.write("# Remove words that carry semantic meaning.\n")
            f.write("#\n\n")

        for word in words_to_save:
            f.write(f"{word}\n")

    logger.info(f"Saved {len(words_to_save)} stopword candidates to {file_path}")
    return len(words_to_save)


def generate_discovery_report(
    texts: List[str],
    output_path: Optional[Union[str, Path]] = None,
    min_doc_frequency: float = 0.7,
    format: str = 'markdown'
) -> str:
    """
    Generate a comprehensive stopword discovery report.

    This is a convenience function that runs the discovery pipeline
    and generates a formatted report.

    Args:
        texts: List of document texts
        output_path: Optional path to save the report
        min_doc_frequency: Document frequency threshold
        format: Output format ('markdown', 'dict', 'json')

    Returns:
        Formatted report string
    """
    report = find_domain_stopword_candidates(
        texts,
        min_doc_frequency=min_doc_frequency,
        compute_tfidf_variance=True
    )

    if format == 'markdown':
        output = report.to_markdown()
    elif format == 'dict':
        import json
        output = json.dumps(report.to_dict(), indent=2)
    elif format == 'json':
        import json
        output = json.dumps(report.to_dict(), indent=2)
    else:
        output = report.to_markdown()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output)
        logger.info(f"Saved discovery report to {output_path}")

    return output
