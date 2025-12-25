"""
Content Quality Assessment Module for Open-Ended Coding Analysis.

This module provides transparent, auditable quality filtering that flags
(but never automatically excludes) low-signal responses for human review.

Key Principles:
- NEVER automatically exclude responses without user approval
- All flagging rules are explicit and documentable
- Provide confidence scores (0.0-1.0) for each assessment
- Allow manual override via configuration
- All responses are accounted for (flagged, not dropped)
"""

import re
import string
from typing import Dict, List, Tuple
import logging


class ContentQualityFilter:
    """
    Assesses content quality of text responses.

    Detects non-analytic content including:
    - Null/empty responses
    - Too-short responses (insufficient content)
    - Gibberish (keyboard walks, random characters)
    - Non-English content (when English expected)
    - Test responses (test data, placeholder text)
    - Non-responses (N/A, idk, no comment)
    - Excessive repetition
    """

    # Non-response patterns (case-insensitive)
    NON_RESPONSE_PATTERNS = [
        r'^\s*n\s*/\s*a\s*$',  # N / A with spaces
        r'^\s*n/?a\s*$',       # N/A without spaces
        r'^\s*na\s*$',
        r'^\s*none\s*$',
        r'^\s*nothing\s*$',
        r'^\s*idk\s*$',
        r'^\s*i don\'?t know\s*$',
        r'^\s*no comment\s*$',
        r'^\s*no response\s*$',
        r'^\s*skip\s*$',
        r'^\s*pass\s*$',
        r'^\s*-+\s*$',
        r'^\s*\.+\s*$',
        r'^\s*\?+\s*$',
    ]

    # Test response patterns (case-insensitive)
    TEST_PATTERNS = [
        r'^\s*test\s*$',
        r'^\s*testing\s*$',
        r'^\s*asdf',
        r'^\s*qwer',
        r'^\s*123+\s*$',
        r'^\s*abc+\s*$',
        r'^\s*xxx+\s*$',
        r'^\s*zzz+\s*$',
        r'^\s*sample\s*$',
        r'^\s*example\s*$',
        r'^\s*placeholder\s*$',
        r'^\s*lorem ipsum',
    ]

    # Keyboard walk patterns (gibberish detection)
    KEYBOARD_WALKS = [
        'qwerty', 'asdfgh', 'zxcvbn',
        'qwertyuiop', 'asdfghjkl', 'zxcvbnm',
        'poiuytrewq', 'lkjhgfdsa', 'mnbvcxz',
        '1234567890', '0987654321'
    ]

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
        # Additional common words
        'many', 'find', 'here', 'thing', 'give', 'more', 'very', 'where', 'much', 'through',
        'great', 'little', 'another', 'every', 'same', 'should', 'home', 'being', 'long', 'still',
        'between', 'both', 'few', 'may', 'own', 'such', 'too', 'under', 'might', 'place',
        'while', 'last', 'world', 'another', 'different', 'each', 'right', 'does', 'before', 'large',
        'must', 'big', 'high', 'something', 'seem', 'need', 'try', 'ask', 'three', 'system',
        'those', 'put', 'set', 'better', 'best', 'however', 'show', 'old', 'going', 'really',
        'word', 'words', 'response', 'content', 'text', 'yes', 'never', 'always', 'often', 'sometimes'
    }

    def __init__(
        self,
        min_words: int = 3,
        min_chars: int = 10,
        max_repetition_ratio: float = 0.7,
        min_english_word_ratio: float = 0.3,
        logger: logging.Logger = None
    ):
        """
        Initialize ContentQualityFilter.

        Args:
            min_words: Minimum number of words for analytic content (default: 3)
            min_chars: Minimum number of characters (default: 10)
            max_repetition_ratio: Maximum ratio of repeated words (default: 0.7)
            min_english_word_ratio: Minimum ratio of recognized English words (default: 0.3)
            logger: Optional logger instance
        """
        self.min_words = min_words
        self.min_chars = min_chars
        self.max_repetition_ratio = max_repetition_ratio
        self.min_english_word_ratio = min_english_word_ratio
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Compile regex patterns for performance
        self.non_response_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.NON_RESPONSE_PATTERNS
        ]
        self.test_regex = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.TEST_PATTERNS
        ]

    def assess_signal(self, text: str) -> Dict:
        """
        Assess the quality of a text response.

        Args:
            text: The text response to assess

        Returns:
            Dictionary with assessment results:
            {
                'is_analytic': bool,           # True if content appears analytic
                'confidence': float,           # 0.0-1.0 confidence score
                'reason': str,                 # Human-readable explanation
                'recommendation': str,         # 'include', 'review', or 'exclude'
                'flags': List[str]            # List of quality flags
            }
        """
        flags = []
        reasons = []
        confidence_scores = []

        # Check 1: Null or empty
        if text is None or not isinstance(text, str):
            return {
                'is_analytic': False,
                'confidence': 1.0,
                'reason': 'Response is null or not a string',
                'recommendation': 'exclude',
                'flags': ['null']
            }

        # Strip whitespace for analysis
        text_stripped = text.strip()

        if not text_stripped:
            return {
                'is_analytic': False,
                'confidence': 1.0,
                'reason': 'Response is empty (whitespace only)',
                'recommendation': 'exclude',
                'flags': ['empty']
            }

        # Check 2: Too short (character count)
        if len(text_stripped) < self.min_chars:
            flags.append('too_short_chars')
            reasons.append(f'Only {len(text_stripped)} characters (minimum: {self.min_chars})')
            confidence_scores.append(0.9)

        # Get word list for subsequent checks
        words = self._extract_words(text_stripped)
        word_count = len(words)

        # Check 3: Too short (word count)
        if word_count < self.min_words:
            flags.append('too_short_words')
            reasons.append(f'Only {word_count} words (minimum: {self.min_words})')
            confidence_scores.append(0.85)

        # Check 4: Non-response patterns
        non_response_match = self._check_non_response(text_stripped)
        if non_response_match:
            flags.append('non_response')
            reasons.append(f'Matches non-response pattern: "{non_response_match}"')
            confidence_scores.append(0.95)

        # Check 5: Test response patterns
        test_match = self._check_test_response(text_stripped)
        if test_match:
            flags.append('test_response')
            reasons.append(f'Matches test pattern: "{test_match}"')
            confidence_scores.append(0.9)

        # Check 6: Gibberish detection
        gibberish_score, gibberish_reason = self._check_gibberish(text_stripped, words)
        if gibberish_score > 0.5:
            flags.append('gibberish')
            reasons.append(gibberish_reason)
            confidence_scores.append(gibberish_score)

        # Check 7: Language detection (non-English)
        if word_count >= 3:  # Only check if enough words
            english_ratio = self._check_english(words)
            if english_ratio < self.min_english_word_ratio:
                flags.append('non_english')
                reasons.append(
                    f'Low English word ratio: {english_ratio:.2%} '
                    f'(minimum: {self.min_english_word_ratio:.2%})'
                )
                confidence_scores.append(0.7)

        # Check 8: Excessive repetition
        repetition_ratio = self._check_repetition(words)
        if repetition_ratio > self.max_repetition_ratio:
            flags.append('excessive_repetition')
            reasons.append(
                f'High repetition ratio: {repetition_ratio:.2%} '
                f'(maximum: {self.max_repetition_ratio:.2%})'
            )
            confidence_scores.append(0.75)

        # Check 9: All punctuation or numbers
        if self._is_all_punctuation_or_numbers(text_stripped):
            flags.append('no_alphabetic')
            reasons.append('Contains no alphabetic characters')
            confidence_scores.append(0.95)

        # Determine overall assessment
        if not flags:
            # No issues detected
            return {
                'is_analytic': True,
                'confidence': 1.0,
                'reason': 'No quality issues detected',
                'recommendation': 'include',
                'flags': []
            }

        # Calculate overall confidence (average of individual checks)
        overall_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5

        # Determine recommendation based on flags and confidence
        recommendation = self._determine_recommendation(flags, overall_confidence)

        # Combine reasons
        combined_reason = '; '.join(reasons)

        return {
            'is_analytic': False,
            'confidence': round(overall_confidence, 3),
            'reason': combined_reason,
            'recommendation': recommendation,
            'flags': flags
        }

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text (alphanumeric sequences)."""
        # Split on whitespace and punctuation, keep only alphanumeric
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w]

    def _check_non_response(self, text: str) -> str:
        """Check if text matches non-response patterns."""
        for pattern in self.non_response_regex:
            if pattern.match(text):
                return text.strip()[:50]  # Return matched text (max 50 chars)
        return None

    def _check_test_response(self, text: str) -> str:
        """Check if text matches test response patterns."""
        for pattern in self.test_regex:
            if pattern.search(text):
                return text.strip()[:50]
        return None

    def _check_gibberish(self, text: str, words: List[str]) -> Tuple[float, str]:
        """
        Detect gibberish using multiple heuristics.

        Returns:
            Tuple of (confidence_score, reason)
        """
        if not words:
            return 0.0, ""

        text_lower = text.lower()
        reasons = []
        scores = []

        # Heuristic 1: Keyboard walks
        for walk in self.KEYBOARD_WALKS:
            if walk in text_lower:
                scores.append(0.9)
                reasons.append(f'Contains keyboard walk: "{walk}"')
                break

        # Heuristic 2: High consonant-to-vowel ratio (gibberish often lacks vowels)
        vowel_ratio = self._calculate_vowel_ratio(text_lower)
        if vowel_ratio < 0.15 and len(text_lower) > 5:  # Less than 15% vowels
            scores.append(0.7)
            reasons.append(f'Very low vowel ratio: {vowel_ratio:.2%}')

        # Heuristic 3: Excessive character repetition (aaaaaaa, hhhhhh)
        if self._has_excessive_char_repetition(text_lower):
            scores.append(0.8)
            reasons.append('Contains excessive character repetition')

        # Heuristic 4: No recognizable words (very short words or gibberish)
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        if avg_word_length < 2.5 and len(words) > 2:
            scores.append(0.6)
            reasons.append(f'Very short average word length: {avg_word_length:.1f}')

        if not scores:
            return 0.0, ""

        # Return average score and combined reasons
        avg_score = sum(scores) / len(scores)
        combined_reason = '; '.join(reasons)
        return avg_score, combined_reason

    def _calculate_vowel_ratio(self, text: str) -> float:
        """Calculate ratio of vowels to total alphabetic characters."""
        vowels = 'aeiou'
        alpha_chars = [c for c in text.lower() if c.isalpha()]
        if not alpha_chars:
            return 0.0
        vowel_count = sum(1 for c in alpha_chars if c in vowels)
        return vowel_count / len(alpha_chars)

    def _has_excessive_char_repetition(self, text: str) -> bool:
        """Check for excessive repeated characters (e.g., 'aaaaaaa')."""
        # Look for 4+ consecutive identical characters
        pattern = re.compile(r'(.)\1{3,}')
        return bool(pattern.search(text))

    def _check_english(self, words: List[str]) -> float:
        """
        Calculate ratio of recognized English words.

        Returns:
            Float between 0.0 and 1.0 representing English word ratio
        """
        if not words:
            return 0.0

        english_count = sum(1 for word in words if word in self.COMMON_ENGLISH_WORDS)
        return english_count / len(words)

    def _check_repetition(self, words: List[str]) -> float:
        """Calculate ratio of repeated words to total words."""
        if not words or len(words) < 3:
            return 0.0  # Too short to judge repetition

        unique_words = len(set(words))
        total_words = len(words)

        # Repetition ratio = 1 - (unique/total)
        # High ratio means low diversity
        return 1.0 - (unique_words / total_words)

    def _is_all_punctuation_or_numbers(self, text: str) -> bool:
        """Check if text contains no alphabetic characters."""
        return not any(c.isalpha() for c in text)

    def _determine_recommendation(self, flags: List[str], confidence: float) -> str:
        """
        Determine recommendation based on flags and confidence.

        Args:
            flags: List of quality flags
            confidence: Overall confidence score

        Returns:
            'include', 'review', or 'exclude'
        """
        # Critical flags that suggest exclusion
        critical_flags = {'null', 'empty', 'non_response', 'no_alphabetic'}

        # Flags that suggest review
        review_flags = {'test_response', 'gibberish', 'excessive_repetition'}

        # Flags that are warnings but may be acceptable
        warning_flags = {'too_short_chars', 'too_short_words', 'non_english'}

        # Check for critical flags
        if any(flag in critical_flags for flag in flags):
            return 'exclude'

        # High confidence in non-analytic content
        if confidence > 0.85 and any(flag in review_flags for flag in flags):
            return 'exclude'

        # Medium-high confidence or review flags present
        if confidence > 0.6 or any(flag in review_flags for flag in flags):
            return 'review'

        # Low confidence or only warning flags
        if any(flag in warning_flags for flag in flags):
            return 'review'

        return 'include'

    def batch_assess(self, texts: List[str]) -> List[Dict]:
        """
        Assess multiple texts in batch.

        Args:
            texts: List of text responses

        Returns:
            List of assessment dictionaries
        """
        return [self.assess_signal(text) for text in texts]

    def get_flag_statistics(self, assessments: List[Dict]) -> Dict[str, int]:
        """
        Calculate statistics on flags across multiple assessments.

        Args:
            assessments: List of assessment results from assess_signal()

        Returns:
            Dictionary mapping flag names to counts
        """
        flag_counts = {}
        for assessment in assessments:
            for flag in assessment['flags']:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
        return flag_counts
