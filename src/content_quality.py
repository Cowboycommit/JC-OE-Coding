"""
Content quality assessment for open-ended survey responses.

This module provides quality filtering capabilities to identify non-analytic
content in text responses, including gibberish, test responses, and other
low-quality content that may not be suitable for qualitative analysis.

Key features:
- Configurable quality thresholds
- Multiple quality indicators (word count, character count, repetition, etc.)
- Batch processing for efficiency
- Transparent flagging with reasons and recommendations
"""

import re
import logging
from typing import List, Dict, Any, Optional
from collections import Counter


class ContentQualityFilter:
    """
    Assesses content quality for open-ended text responses.

    This class provides methods to evaluate whether text responses contain
    meaningful, analytic content suitable for qualitative analysis.

    Design principles:
    - Transparent flagging: All assessments include reasons and recommendations
    - Conservative defaults: Prefer false negatives over false positives
    - Auditable: All decisions can be reviewed and overridden
    - No automatic exclusion: Flagging only, human review recommended

    Example usage:
        >>> filter = ContentQualityFilter(min_words=3, min_chars=10)
        >>> assessment = filter.assess("This is a thoughtful response about the topic.")
        >>> print(assessment['is_analytic'])  # True
        >>> print(assessment['recommendation'])  # 'include'
    """

    # Common English words for language detection
    COMMON_ENGLISH_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
        'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
        'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
        'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
        'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us',
        'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'did', 'does',
        'very', 'more', 'much', 'many', 'such', 'here', 'where', 'why', 'yes', 'no'
    }

    # Common non-analytic patterns
    NON_ANALYTIC_PATTERNS = [
        r'^n/?a$',                          # N/A, n/a, N/a
        r'^none\.?$',                       # none, None
        r'^nothing\.?$',                    # nothing
        r'^no\s*(comment|response|answer)?\.?$',  # no, no comment
        r'^idk\.?$',                        # idk
        r'^i\s*don\'?t\s*know\.?$',         # I don't know
        r'^test\.?$',                       # test
        r'^asdf+$',                         # asdf, asdfasdf
        r'^[a-z]+\1{2,}$',                  # repeated character patterns
        r'^\.+$',                           # just periods
        r'^-+$',                            # just dashes
        r'^\s*$',                           # whitespace only
    ]

    def __init__(
        self,
        min_words: int = 3,
        min_chars: int = 10,
        max_repetition_ratio: float = 0.7,
        min_english_word_ratio: float = 0.3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ContentQualityFilter.

        Args:
            min_words: Minimum number of words for analytic content (default: 3)
            min_chars: Minimum number of characters (default: 10)
            max_repetition_ratio: Maximum ratio of repeated words (default: 0.7)
            min_english_word_ratio: Minimum ratio of English words (default: 0.3)
            logger: Optional logger instance for logging messages
        """
        self.min_words = min_words
        self.min_chars = min_chars
        self.max_repetition_ratio = max_repetition_ratio
        self.min_english_word_ratio = min_english_word_ratio
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Compile non-analytic patterns
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.NON_ANALYTIC_PATTERNS
        ]

    def assess(self, text: str) -> Dict[str, Any]:
        """
        Assess the quality of a single text response.

        Args:
            text: The text response to assess

        Returns:
            Dictionary containing:
                - is_analytic: bool - Whether the content appears analytic
                - confidence: float - Confidence score (0.0 to 1.0)
                - reason: str - Human-readable explanation
                - recommendation: str - 'include', 'review', or 'exclude'
                - flags: list - List of quality flags triggered
        """
        flags = []
        confidence = 1.0

        # Handle None or empty
        if text is None or (isinstance(text, str) and not text.strip()):
            return {
                'is_analytic': False,
                'confidence': 1.0,
                'reason': 'Empty or null response',
                'recommendation': 'exclude',
                'flags': ['empty_response']
            }

        text = str(text).strip()

        # Check character length
        if len(text) < self.min_chars:
            flags.append('too_short_chars')
            confidence -= 0.3

        # Check word count
        words = self._tokenize(text)
        if len(words) < self.min_words:
            flags.append('too_few_words')
            confidence -= 0.3

        # Check for non-analytic patterns
        for pattern in self.compiled_patterns:
            if pattern.match(text):
                flags.append('non_analytic_pattern')
                confidence -= 0.5
                break

        # Check repetition ratio
        if words:
            repetition_ratio = self._calculate_repetition_ratio(words)
            if repetition_ratio > self.max_repetition_ratio:
                flags.append('high_repetition')
                confidence -= 0.2

        # Check English word ratio (for gibberish detection)
        if words:
            english_ratio = self._calculate_english_ratio(words)
            if english_ratio < self.min_english_word_ratio:
                flags.append('low_english_ratio')
                confidence -= 0.2

        # Determine final assessment
        confidence = max(0.0, min(1.0, confidence))
        is_analytic = len(flags) == 0

        # Determine recommendation
        if len(flags) == 0:
            recommendation = 'include'
            reason = 'Content appears analytic'
        elif confidence >= 0.5:
            recommendation = 'review'
            reason = f"Flagged for review: {', '.join(flags)}"
        else:
            recommendation = 'exclude'
            reason = f"Low quality indicators: {', '.join(flags)}"

        return {
            'is_analytic': is_analytic,
            'confidence': round(confidence, 3),
            'reason': reason,
            'recommendation': recommendation,
            'flags': flags
        }

    def batch_assess(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Assess quality for a batch of text responses.

        Args:
            texts: List of text responses to assess

        Returns:
            List of assessment dictionaries, one per input text
        """
        return [self.assess(text) for text in texts]

    def get_flag_statistics(self, assessments: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate statistics on quality flags from a batch of assessments.

        Args:
            assessments: List of assessment dictionaries from assess() or batch_assess()

        Returns:
            Dictionary mapping flag names to counts
        """
        flag_counts: Dict[str, int] = {}

        for assessment in assessments:
            flags = assessment.get('flags', [])
            for flag in flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        return flag_counts

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of lowercase words
        """
        # Simple word tokenization: split on non-word characters
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

    def _calculate_repetition_ratio(self, words: List[str]) -> float:
        """
        Calculate the ratio of repeated words in text.

        Args:
            words: List of words

        Returns:
            Ratio of repeated words (0.0 to 1.0)
        """
        if not words:
            return 0.0

        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)

        # Repetition ratio: 1 - (unique/total)
        # High ratio means more repetition
        return 1.0 - (unique_words / total_words)

    def _calculate_english_ratio(self, words: List[str]) -> float:
        """
        Calculate the ratio of common English words in text.

        Args:
            words: List of words

        Returns:
            Ratio of English words (0.0 to 1.0)
        """
        if not words:
            return 0.0

        english_count = sum(1 for word in words if word in self.COMMON_ENGLISH_WORDS)
        return english_count / len(words)
