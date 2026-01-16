"""
Gold Standard Text Preprocessing Module.

Provides comprehensive text normalization, quality filtering, and metrics
reporting for NLP pipelines. Implements industry best practices for
text preprocessing with configurable options.

This module is adapted from the JC-Text-Analysis-NLP project to provide
consistent data cleaning capabilities for the Open-Ended Coding Analysis framework.
"""

import html
import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# COMPREHENSIVE DICTIONARIES
# =============================================================================

# 75+ contraction mappings
CONTRACTIONS: Dict[str, str] = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "ain't": "is not",
    "gonna": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "kinda": "kind of",
    "sorta": "sort of",
    "coulda": "could have",
    "woulda": "would have",
    "shoulda": "should have",
    "musta": "must have",
    "lemme": "let me",
    "gimme": "give me",
    "dunno": "do not know",
    "y'all": "you all",
    "c'mon": "come on",
    "ma'am": "madam",
    "o'clock": "of the clock",
    "'cause": "because",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "when's": "when is",
    "why'd": "why did",
    "why's": "why is",
}

# 50+ slang mappings (optional use)
SLANG_MAP: Dict[str, str] = {
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "lmfao": "laughing my fucking ass off",
    "rofl": "rolling on floor laughing",
    "brb": "be right back",
    "btw": "by the way",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "ngl": "not gonna lie",
    "idk": "i do not know",
    "omg": "oh my god",
    "omfg": "oh my fucking god",
    "wtf": "what the fuck",
    "wth": "what the heck",
    "smh": "shaking my head",
    "fyi": "for your information",
    "afaik": "as far as i know",
    "iirc": "if i recall correctly",
    "tldr": "too long did not read",
    "tl;dr": "too long did not read",
    "irl": "in real life",
    "ikr": "i know right",
    "nvm": "never mind",
    "ftw": "for the win",
    "fwiw": "for what it is worth",
    "ama": "ask me anything",
    "eli5": "explain like i am five",
    "diy": "do it yourself",
    "tmi": "too much information",
    "bff": "best friends forever",
    "jk": "just kidding",
    "np": "no problem",
    "thx": "thanks",
    "ty": "thank you",
    "yw": "you are welcome",
    "pls": "please",
    "plz": "please",
    "rn": "right now",
    "atm": "at the moment",
    "asap": "as soon as possible",
    "eta": "estimated time of arrival",
    "goat": "greatest of all time",
    "fomo": "fear of missing out",
    "yolo": "you only live once",
    "tbd": "to be determined",
    "tba": "to be announced",
    "nsfw": "not safe for work",
    "sfw": "safe for work",
    "afk": "away from keyboard",
    "gtg": "got to go",
    "g2g": "got to go",
    "cya": "see you",
    "cu": "see you",
}

# 13+ spam detection patterns
SPAM_PATTERNS: List[re.Pattern] = [
    re.compile(r"buy\s+now", re.IGNORECASE),
    re.compile(r"click\s+here", re.IGNORECASE),
    re.compile(r"limited\s+time\s+offer", re.IGNORECASE),
    re.compile(r"act\s+now", re.IGNORECASE),
    re.compile(r"free\s+(?:gift|money|trial)", re.IGNORECASE),
    re.compile(r"100%\s+(?:free|guaranteed)", re.IGNORECASE),
    re.compile(r"no\s+obligation", re.IGNORECASE),
    re.compile(r"winner\s+(?:selected|chosen)", re.IGNORECASE),
    re.compile(r"congratulations.*(?:won|winner)", re.IGNORECASE),
    re.compile(r"urgent.*(?:action|response)", re.IGNORECASE),
    re.compile(r"(?:earn|make)\s+\$?\d+.*(?:day|week|month)", re.IGNORECASE),
    re.compile(r"(?:credit|debit)\s+card\s+(?:number|info)", re.IGNORECASE),
    re.compile(r"(?:password|account)\s+(?:verify|confirm)", re.IGNORECASE),
    re.compile(r"nigerian?\s+prince", re.IGNORECASE),
    re.compile(r"(?:subscribe|unsubscribe)\s+(?:here|now)", re.IGNORECASE),
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Configuration for gold standard text preprocessing."""

    normalize_unicode: bool = True
    decode_html_entities: bool = True
    normalize_whitespace: bool = True
    standardize_urls: bool = True
    standardize_mentions: bool = True
    process_hashtags: bool = True
    expand_contractions: bool = True
    normalize_elongations: bool = True
    normalize_punctuation: bool = True
    expand_slang: bool = False

    max_char_repeat: int = 2

    min_tokens: int = 3
    max_tokens: int = 512
    max_emoji_ratio: float = 0.7
    detect_spam: bool = True
    detect_duplicates: bool = True
    filter_by_language: bool = False
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])

    url_token: str = "<URL>"
    user_token: str = "<USER>"

    verbose: bool = False


@dataclass
class DataQualityMetrics:
    """Tracks data quality metrics during preprocessing."""

    total_records: int = 0
    valid_records: int = 0
    filtered_records: int = 0

    filter_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    total_characters: int = 0
    total_tokens: int = 0
    total_emojis: int = 0
    text_lengths: List[int] = field(default_factory=list)
    token_counts: List[int] = field(default_factory=list)

    unicode_normalized: int = 0
    html_decoded: int = 0
    urls_replaced: int = 0
    mentions_replaced: int = 0
    hashtags_processed: int = 0
    contractions_expanded: int = 0
    elongations_normalized: int = 0
    punctuation_normalized: int = 0
    slang_expanded: int = 0

    seen_hashes: Set[str] = field(default_factory=set)
    duplicate_count: int = 0

    def add_filter_reason(self, reason: str) -> None:
        """Record a filter reason."""
        self.filter_reasons[reason] += 1
        self.filtered_records += 1

    def record_text_stats(self, text: str, token_count: int, emoji_count: int) -> None:
        """Record statistics for a valid text."""
        self.valid_records += 1
        self.total_characters += len(text)
        self.total_tokens += token_count
        self.total_emojis += emoji_count
        self.text_lengths.append(len(text))
        self.token_counts.append(token_count)

    def check_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate using MD5 hash."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.seen_hashes:
            self.duplicate_count += 1
            return True
        self.seen_hashes.add(text_hash)
        return False

    @property
    def avg_text_length(self) -> float:
        """Average text length in characters."""
        if not self.text_lengths:
            return 0.0
        return sum(self.text_lengths) / len(self.text_lengths)

    @property
    def avg_token_count(self) -> float:
        """Average token count per text."""
        if not self.token_counts:
            return 0.0
        return sum(self.token_counts) / len(self.token_counts)

    @property
    def avg_emoji_count(self) -> float:
        """Average emoji count per valid text."""
        if self.valid_records == 0:
            return 0.0
        return self.total_emojis / self.valid_records

    @property
    def valid_ratio(self) -> float:
        """Ratio of valid to total records."""
        if self.total_records == 0:
            return 0.0
        return self.valid_records / self.total_records

    def generate_report(self) -> str:
        """Generate a human-readable quality report."""
        lines = [
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            "",
            "RECORD COUNTS",
            "-" * 40,
            f"  Total records:     {self.total_records:,}",
            f"  Valid records:     {self.valid_records:,}",
            f"  Filtered records:  {self.filtered_records:,}",
            f"  Valid ratio:       {self.valid_ratio:.2%}",
            "",
        ]

        if self.filter_reasons:
            lines.extend([
                "FILTER BREAKDOWN",
                "-" * 40,
            ])
            for reason, count in sorted(self.filter_reasons.items(),
                                        key=lambda x: -x[1]):
                lines.append(f"  {reason}: {count:,}")
            lines.append("")

        lines.extend([
            "TEXT STATISTICS",
            "-" * 40,
            f"  Avg text length:   {self.avg_text_length:.1f} chars",
            f"  Avg token count:   {self.avg_token_count:.1f} tokens",
            f"  Avg emoji count:   {self.avg_emoji_count:.2f} emojis",
            f"  Total characters:  {self.total_characters:,}",
            f"  Total tokens:      {self.total_tokens:,}",
            "",
            "NORMALIZATION STATISTICS",
            "-" * 40,
            f"  Unicode normalized:      {self.unicode_normalized:,}",
            f"  HTML entities decoded:   {self.html_decoded:,}",
            f"  URLs replaced:           {self.urls_replaced:,}",
            f"  Mentions replaced:       {self.mentions_replaced:,}",
            f"  Hashtags processed:      {self.hashtags_processed:,}",
            f"  Contractions expanded:   {self.contractions_expanded:,}",
            f"  Elongations normalized:  {self.elongations_normalized:,}",
            f"  Punctuation normalized:  {self.punctuation_normalized:,}",
            f"  Slang expanded:          {self.slang_expanded:,}",
            "",
            "DUPLICATE DETECTION",
            "-" * 40,
            f"  Duplicates found:  {self.duplicate_count:,}",
            f"  Unique texts:      {len(self.seen_hashes):,}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "record_counts": {
                "total": self.total_records,
                "valid": self.valid_records,
                "filtered": self.filtered_records,
                "valid_ratio": self.valid_ratio,
            },
            "filter_breakdown": dict(self.filter_reasons),
            "text_statistics": {
                "avg_text_length": self.avg_text_length,
                "avg_token_count": self.avg_token_count,
                "avg_emoji_count": self.avg_emoji_count,
                "total_characters": self.total_characters,
                "total_tokens": self.total_tokens,
            },
            "normalization_statistics": {
                "unicode_normalized": self.unicode_normalized,
                "html_decoded": self.html_decoded,
                "urls_replaced": self.urls_replaced,
                "mentions_replaced": self.mentions_replaced,
                "hashtags_processed": self.hashtags_processed,
                "contractions_expanded": self.contractions_expanded,
                "elongations_normalized": self.elongations_normalized,
                "punctuation_normalized": self.punctuation_normalized,
                "slang_expanded": self.slang_expanded,
            },
            "duplicate_detection": {
                "duplicates_found": self.duplicate_count,
                "unique_texts": len(self.seen_hashes),
            },
        }


# =============================================================================
# GOLD STANDARD TEXT PROCESSOR
# =============================================================================

class GoldStandardTextProcessor:
    """
    Gold standard text preprocessing for NLP pipelines.

    Implements a comprehensive text normalization pipeline with configurable
    options for Unicode normalization, HTML entity decoding, contraction
    expansion, elongation normalization, and quality filtering.

    Example:
        >>> processor = GoldStandardTextProcessor()
        >>> text = "I loooove this product!!! It's AMAZING"
        >>> result = processor.process(text)
        >>> print(result)
        'I loove this product! It is AMAZING'
    """

    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+|'
        r'www\.[^\s<>"{}|\\^`\[\]]+|'
        r'[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:/[^\s]*)?'
    )
    MENTION_PATTERN = re.compile(r'@[\w]+')
    HASHTAG_PATTERN = re.compile(r'#([\w]+)')
    ELONGATION_PATTERN = re.compile(r'(.)\1{2,}')
    REPEATED_PUNCT_PATTERN = re.compile(r'([!?.])\1+')
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE
    )

    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        normalize_unicode: bool = True,
        decode_html_entities: bool = True,
        expand_contractions: bool = True,
        normalize_elongations: bool = True,
        max_char_repeat: int = 2,
        min_tokens: int = 3,
        max_tokens: int = 512,
        max_emoji_ratio: float = 0.7,
        detect_spam: bool = True,
        detect_duplicates: bool = True,
        standardize_urls: bool = True,
        standardize_mentions: bool = True,
        process_hashtags: bool = True,
        normalize_punctuation: bool = True,
        expand_slang: bool = False,
        url_token: str = "<URL>",
        user_token: str = "<USER>",
        verbose: bool = False,
    ):
        """
        Initialize the gold standard text processor.

        Args:
            config: PreprocessingConfig object (overrides other params if provided)
            normalize_unicode: Apply NFKC Unicode normalization
            decode_html_entities: Decode HTML entities (&amp; -> &)
            expand_contractions: Expand contractions (don't -> do not)
            normalize_elongations: Normalize character elongations (loooove -> loove)
            max_char_repeat: Maximum allowed character repetitions
            min_tokens: Minimum token threshold for valid text
            max_tokens: Maximum token threshold for valid text
            max_emoji_ratio: Maximum emoji-to-character ratio
            detect_spam: Enable spam pattern detection
            detect_duplicates: Enable duplicate detection via MD5 hash
            standardize_urls: Replace URLs with token
            standardize_mentions: Replace @mentions with token
            process_hashtags: Remove # from hashtags
            normalize_punctuation: Normalize repeated punctuation
            expand_slang: Expand common slang terms
            url_token: Token to replace URLs with
            user_token: Token to replace mentions with
            verbose: Enable verbose logging
        """
        if config is not None:
            self.config = config
        else:
            self.config = PreprocessingConfig(
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
                verbose=verbose,
            )

        self.metrics = DataQualityMetrics()

        contraction_keys = sorted(CONTRACTIONS.keys(), key=len, reverse=True)
        self._contraction_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in contraction_keys) + r')\b',
            re.IGNORECASE
        )

        if self.config.expand_slang:
            slang_keys = sorted(SLANG_MAP.keys(), key=len, reverse=True)
            self._slang_pattern = re.compile(
                r'\b(' + '|'.join(re.escape(k) for k in slang_keys) + r')\b',
                re.IGNORECASE
            )

    def reset_metrics(self) -> None:
        """Reset quality metrics for a new processing batch."""
        self.metrics = DataQualityMetrics()

    def normalize(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Apply the full normalization pipeline to text.

        Args:
            text: Input text to normalize

        Returns:
            Tuple of (normalized_text, normalization_counts)
        """
        if not text or not isinstance(text, str):
            return "", {}

        counts = defaultdict(int)

        if self.config.normalize_unicode:
            normalized = unicodedata.normalize('NFKC', text)
            if normalized != text:
                counts['unicode'] += 1
            text = normalized

        if self.config.decode_html_entities:
            decoded = html.unescape(text)
            if decoded != text:
                counts['html'] += 1
            text = decoded

        if self.config.normalize_whitespace:
            text = ' '.join(text.split())

        if self.config.standardize_urls:
            urls_found = len(self.URL_PATTERN.findall(text))
            if urls_found > 0:
                counts['urls'] = urls_found
                text = self.URL_PATTERN.sub(self.config.url_token, text)

        if self.config.standardize_mentions:
            mentions_found = len(self.MENTION_PATTERN.findall(text))
            if mentions_found > 0:
                counts['mentions'] = mentions_found
                text = self.MENTION_PATTERN.sub(self.config.user_token, text)

        if self.config.process_hashtags:
            hashtags_found = len(self.HASHTAG_PATTERN.findall(text))
            if hashtags_found > 0:
                counts['hashtags'] = hashtags_found
                text = self.HASHTAG_PATTERN.sub(r'\1', text)

        if self.config.expand_contractions:
            def replace_contraction(match):
                word = match.group(0)
                replacement = CONTRACTIONS.get(word.lower(), word)
                if word[0].isupper():
                    replacement = replacement.capitalize()
                counts['contractions'] += 1
                return replacement

            text = self._contraction_pattern.sub(replace_contraction, text)

        if self.config.normalize_elongations:
            def replace_elongation(match):
                char = match.group(1)
                counts['elongations'] += 1
                return char * self.config.max_char_repeat

            text = self.ELONGATION_PATTERN.sub(replace_elongation, text)

        if self.config.normalize_punctuation:
            punct_count = len(self.REPEATED_PUNCT_PATTERN.findall(text))
            if punct_count > 0:
                counts['punctuation'] = punct_count
                text = self.REPEATED_PUNCT_PATTERN.sub(r'\1', text)

        if self.config.expand_slang:
            def replace_slang(match):
                word = match.group(0)
                replacement = SLANG_MAP.get(word.lower(), word)
                counts['slang'] += 1
                return replacement

            text = self._slang_pattern.sub(replace_slang, text)

        return text, dict(counts)

    def count_tokens(self, text: str) -> int:
        """Count tokens using simple whitespace tokenization."""
        if not text:
            return 0
        return len(text.split())

    def count_emojis(self, text: str) -> int:
        """Count emojis in text."""
        if not text:
            return 0
        emojis = self.EMOJI_PATTERN.findall(text)
        return sum(len(e) for e in emojis)

    def detect_spam(self, text: str) -> bool:
        """Check if text matches spam patterns."""
        if not self.config.detect_spam:
            return False
        for pattern in SPAM_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def check_language(self, text: str) -> bool:
        """Check if text is in allowed languages."""
        if not self.config.filter_by_language:
            return True
        try:
            from langdetect import detect
            detected = detect(text)
            return detected in self.config.allowed_languages
        except Exception:
            return True

    def process(
        self,
        text: str,
        track_metrics: bool = True,
    ) -> Optional[str]:
        """
        Process a single text through the gold standard pipeline.

        Args:
            text: Input text to process
            track_metrics: Whether to update quality metrics

        Returns:
            Processed text, or None if filtered out
        """
        if track_metrics:
            self.metrics.total_records += 1

        if not text or not isinstance(text, str):
            if track_metrics:
                self.metrics.add_filter_reason("empty_or_invalid")
            return None

        text = text.strip()
        if not text:
            if track_metrics:
                self.metrics.add_filter_reason("empty_after_strip")
            return None

        normalized, counts = self.normalize(text)

        if track_metrics:
            self.metrics.unicode_normalized += counts.get('unicode', 0)
            self.metrics.html_decoded += counts.get('html', 0)
            self.metrics.urls_replaced += counts.get('urls', 0)
            self.metrics.mentions_replaced += counts.get('mentions', 0)
            self.metrics.hashtags_processed += counts.get('hashtags', 0)
            self.metrics.contractions_expanded += counts.get('contractions', 0)
            self.metrics.elongations_normalized += counts.get('elongations', 0)
            self.metrics.punctuation_normalized += counts.get('punctuation', 0)
            self.metrics.slang_expanded += counts.get('slang', 0)

        token_count = self.count_tokens(normalized)
        emoji_count = self.count_emojis(normalized)

        if token_count < self.config.min_tokens:
            if track_metrics:
                self.metrics.add_filter_reason("below_min_tokens")
            return None

        if token_count > self.config.max_tokens:
            if track_metrics:
                self.metrics.add_filter_reason("above_max_tokens")
            return None

        if len(normalized) > 0:
            emoji_ratio = emoji_count / len(normalized)
            if emoji_ratio > self.config.max_emoji_ratio:
                if track_metrics:
                    self.metrics.add_filter_reason("emoji_spam")
                return None

        if self.detect_spam(normalized):
            if track_metrics:
                self.metrics.add_filter_reason("spam_detected")
            return None

        if self.config.detect_duplicates:
            if track_metrics and self.metrics.check_duplicate(normalized):
                self.metrics.add_filter_reason("duplicate")
                return None

        if not self.check_language(normalized):
            if track_metrics:
                self.metrics.add_filter_reason("wrong_language")
            return None

        if track_metrics:
            self.metrics.record_text_stats(normalized, token_count, emoji_count)

        return normalized

    def process_batch(
        self,
        texts: List[str],
        track_metrics: bool = True,
        return_filtered: bool = False,
    ) -> List[Optional[str]]:
        """
        Process a batch of texts.

        Args:
            texts: List of input texts
            track_metrics: Whether to update quality metrics
            return_filtered: If True, include None for filtered texts

        Returns:
            List of processed texts (with or without None values)
        """
        results = []
        for text in texts:
            result = self.process(text, track_metrics=track_metrics)
            if return_filtered or result is not None:
                results.append(result)
        return results

    def get_metrics(self) -> DataQualityMetrics:
        """Get the current quality metrics."""
        return self.metrics

    def get_report(self) -> str:
        """Generate a human-readable quality report."""
        return self.metrics.generate_report()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_processor: Optional[GoldStandardTextProcessor] = None


def get_default_processor() -> GoldStandardTextProcessor:
    """Get or create the default processor instance."""
    global _default_processor
    if _default_processor is None:
        _default_processor = GoldStandardTextProcessor()
    return _default_processor


def apply_gold_standard_normalization(text: str) -> str:
    """
    Quick single-text normalization using default settings.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    processor = get_default_processor()
    result = processor.process(text, track_metrics=False)
    return result if result is not None else ""


def normalize_for_nlp(
    text: str,
    preserve_case: bool = False,
    remove_stopwords: bool = False,
) -> str:
    """
    NLP-safe normalization preserving linguistic features.

    This function applies gold standard normalization while preserving
    features important for NLP tasks like sentiment analysis and
    named entity recognition.

    Args:
        text: Input text to normalize
        preserve_case: If True, don't convert to lowercase
        remove_stopwords: If True, remove common stopwords

    Returns:
        NLP-normalized text
    """
    processor = GoldStandardTextProcessor(
        normalize_unicode=True,
        decode_html_entities=True,
        expand_contractions=True,
        normalize_elongations=True,
        max_char_repeat=2,
        min_tokens=1,
        max_tokens=1024,
        max_emoji_ratio=1.0,
        detect_spam=False,
        detect_duplicates=False,
        standardize_urls=True,
        standardize_mentions=True,
        process_hashtags=True,
        normalize_punctuation=True,
        expand_slang=False,
    )

    result = processor.process(text, track_metrics=False)

    if result is None:
        return ""

    if not preserve_case:
        result = result.lower()

    if remove_stopwords:
        stopwords = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'that', 'which', 'who', 'whom', 'this', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
        }
        tokens = result.split()
        result = ' '.join(t for t in tokens if t.lower() not in stopwords)

    return result


def create_processor_for_dataset(
    dataset_type: str = "general",
) -> GoldStandardTextProcessor:
    """
    Create a processor optimized for specific dataset types.

    Args:
        dataset_type: One of 'general', 'social_media', 'reviews', 'news'

    Returns:
        Configured GoldStandardTextProcessor instance
    """
    configs = {
        "general": PreprocessingConfig(),
        "social_media": PreprocessingConfig(
            expand_slang=True,
            standardize_urls=True,
            standardize_mentions=True,
            process_hashtags=True,
            max_emoji_ratio=0.5,
            min_tokens=2,
        ),
        "reviews": PreprocessingConfig(
            expand_contractions=True,
            normalize_elongations=True,
            detect_spam=True,
            min_tokens=5,
            max_tokens=1000,
        ),
        "news": PreprocessingConfig(
            expand_contractions=False,
            normalize_elongations=False,
            detect_spam=False,
            min_tokens=10,
            max_tokens=2000,
        ),
    }

    config = configs.get(dataset_type, configs["general"])
    return GoldStandardTextProcessor(config=config)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def preprocess_dataframe(
    df,
    text_column: str,
    output_column: Optional[str] = None,
    processor: Optional[GoldStandardTextProcessor] = None,
    drop_filtered: bool = True,
    inplace: bool = False,
) -> Tuple[Any, DataQualityMetrics]:
    """
    Apply gold standard preprocessing to a pandas DataFrame.

    Args:
        df: Input DataFrame
        text_column: Name of column containing text
        output_column: Name of output column (defaults to text_column + '_processed')
        processor: GoldStandardTextProcessor instance (creates default if None)
        drop_filtered: If True, drop rows that fail quality filters
        inplace: If True, modify DataFrame in place

    Returns:
        Tuple of (processed DataFrame, DataQualityMetrics)
    """
    import pandas as pd

    if not inplace:
        df = df.copy()

    if output_column is None:
        output_column = f"{text_column}_processed"

    if processor is None:
        processor = GoldStandardTextProcessor()
    else:
        processor.reset_metrics()

    df[output_column] = df[text_column].apply(
        lambda x: processor.process(x, track_metrics=True)
    )

    if drop_filtered:
        df = df.dropna(subset=[output_column])

    return df, processor.get_metrics()
