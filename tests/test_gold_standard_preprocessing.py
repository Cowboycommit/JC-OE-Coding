"""
Unit tests for Gold Standard Preprocessing module.
"""

import pytest
import pandas as pd
from src.gold_standard_preprocessing import (
    GoldStandardTextProcessor,
    DataQualityMetrics,
    PreprocessingConfig,
    preprocess_dataframe,
    normalize_for_nlp,
    apply_gold_standard_normalization,
    create_processor_for_dataset,
    CONTRACTIONS,
    SLANG_MAP,
)


class TestGoldStandardTextProcessor:
    """Test cases for GoldStandardTextProcessor."""

    @pytest.fixture
    def processor(self):
        """Create default processor instance."""
        return GoldStandardTextProcessor()

    @pytest.fixture
    def social_processor(self):
        """Create processor optimized for social media."""
        return GoldStandardTextProcessor(
            expand_slang=True,
            standardize_urls=True,
            standardize_mentions=True,
            max_emoji_ratio=0.5,
            min_tokens=2,
        )

    def test_basic_normalization(self, processor):
        """Test basic text normalization."""
        text = "Hello world, this is a test"
        result = processor.process(text)
        assert result is not None
        assert "hello" in result.lower() or "Hello" in result

    def test_contraction_expansion(self, processor):
        """Test contraction expansion."""
        text = "I don't think it's going to work"
        result, counts = processor.normalize(text)
        assert "do not" in result or "is" in result
        assert counts.get('contractions', 0) > 0

    def test_unicode_normalization(self, processor):
        """Test Unicode normalization."""
        text = "caf\u00e9 na\u00efve"  # cafe naive with accents
        result, counts = processor.normalize(text)
        assert result is not None
        # NFKC normalization should handle these characters

    def test_html_entity_decoding(self, processor):
        """Test HTML entity decoding."""
        text = "Tom &amp; Jerry &lt;3"
        result, counts = processor.normalize(text)
        assert "&amp;" not in result
        assert "&" in result or "Tom" in result

    def test_url_standardization(self, processor):
        """Test URL replacement."""
        text = "Check out https://example.com for more info"
        result, counts = processor.normalize(text)
        assert "<URL>" in result or "example.com" not in result
        assert counts.get('urls', 0) > 0

    def test_mention_standardization(self, social_processor):
        """Test @mention replacement."""
        text = "Thanks @user123 for the tip"
        result, counts = social_processor.normalize(text)
        assert "<USER>" in result or "@user123" not in result

    def test_hashtag_processing(self, processor):
        """Test hashtag processing."""
        text = "Love this #amazing product #bestever"
        result, counts = processor.normalize(text)
        assert "#" not in result or result.count("#") < text.count("#")

    def test_elongation_normalization(self, processor):
        """Test character elongation normalization."""
        text = "I loooooove this produuuuct"
        result, counts = processor.normalize(text)
        assert "ooooo" not in result
        assert "uuuu" not in result
        assert counts.get('elongations', 0) > 0

    def test_punctuation_normalization(self, processor):
        """Test repeated punctuation normalization."""
        text = "This is amazing!!! Really???"
        result, counts = processor.normalize(text)
        assert "!!!" not in result
        assert "???" not in result

    def test_slang_expansion(self, social_processor):
        """Test slang expansion."""
        text = "lol btw this is amazing tbh"
        result, counts = social_processor.normalize(text)
        # Check if at least some slang was expanded
        assert "laughing" in result.lower() or counts.get('slang', 0) > 0

    def test_min_tokens_filter(self):
        """Test minimum token filtering."""
        processor = GoldStandardTextProcessor(min_tokens=5)
        text = "Too short"  # Only 2 tokens
        result = processor.process(text)
        assert result is None

    def test_max_tokens_filter(self):
        """Test maximum token filtering."""
        processor = GoldStandardTextProcessor(max_tokens=5)
        text = "This is a very long sentence with many words"  # More than 5 tokens
        result = processor.process(text)
        assert result is None

    def test_spam_detection(self):
        """Test spam pattern detection."""
        processor = GoldStandardTextProcessor(detect_spam=True, min_tokens=1)
        spam_text = "Buy now! Limited time offer! Click here!"
        result = processor.process(spam_text)
        assert result is None

    def test_duplicate_detection(self):
        """Test duplicate detection."""
        processor = GoldStandardTextProcessor(detect_duplicates=True)
        text = "This is a sample text for testing"

        # First occurrence should pass
        result1 = processor.process(text)
        assert result1 is not None

        # Second occurrence should be filtered
        result2 = processor.process(text)
        assert result2 is None

        # Check metrics
        assert processor.metrics.duplicate_count == 1

    def test_empty_text_handling(self, processor):
        """Test handling of empty text."""
        assert processor.process("") is None
        assert processor.process("   ") is None
        assert processor.process(None) is None

    def test_emoji_ratio_filter(self):
        """Test emoji ratio filtering."""
        processor = GoldStandardTextProcessor(max_emoji_ratio=0.3, min_tokens=1)
        # Text with too many emojis relative to content (simplified test)
        # Note: actual emoji filtering depends on text content ratio

    def test_batch_processing(self, processor):
        """Test batch processing of texts."""
        texts = [
            "First sample text here",
            "Second sample text here",
            "Third sample text here",
        ]
        results = processor.process_batch(texts)
        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_batch_processing_with_filtering(self, processor):
        """Test batch processing with some texts filtered."""
        texts = [
            "Valid text sample here",
            "",  # Should be filtered
            "Another valid sample",
        ]
        results = processor.process_batch(texts, return_filtered=True)
        assert len(results) == 3
        assert results[1] is None

    def test_metrics_tracking(self, processor):
        """Test quality metrics tracking."""
        texts = [
            "This is the first sample text",
            "Here is another sample text",
        ]
        processor.reset_metrics()
        for text in texts:
            processor.process(text)

        metrics = processor.get_metrics()
        assert metrics.total_records == 2
        assert metrics.valid_records == 2

    def test_metrics_reset(self, processor):
        """Test metrics reset functionality."""
        processor.process("Sample text here")
        processor.reset_metrics()
        assert processor.metrics.total_records == 0
        assert processor.metrics.valid_records == 0


class TestDataQualityMetrics:
    """Test cases for DataQualityMetrics."""

    @pytest.fixture
    def metrics(self):
        """Create metrics instance."""
        return DataQualityMetrics()

    def test_record_text_stats(self, metrics):
        """Test recording text statistics."""
        metrics.record_text_stats("sample text", 2, 0)
        assert metrics.valid_records == 1
        assert metrics.total_tokens == 2

    def test_add_filter_reason(self, metrics):
        """Test adding filter reasons."""
        metrics.add_filter_reason("too_short")
        metrics.add_filter_reason("too_short")
        metrics.add_filter_reason("spam")

        assert metrics.filtered_records == 3
        assert metrics.filter_reasons["too_short"] == 2
        assert metrics.filter_reasons["spam"] == 1

    def test_check_duplicate(self, metrics):
        """Test duplicate detection."""
        text = "Sample text for duplicate test"
        assert metrics.check_duplicate(text) is False
        assert metrics.check_duplicate(text) is True
        assert metrics.duplicate_count == 1

    def test_avg_calculations(self, metrics):
        """Test average calculations."""
        metrics.record_text_stats("short", 1, 0)
        metrics.record_text_stats("medium text", 2, 0)
        metrics.record_text_stats("longer sample text", 3, 0)

        assert metrics.avg_token_count == 2.0
        assert metrics.valid_records == 3

    def test_valid_ratio(self, metrics):
        """Test valid ratio calculation."""
        metrics.total_records = 10
        metrics.valid_records = 8
        assert metrics.valid_ratio == 0.8

    def test_generate_report(self, metrics):
        """Test report generation."""
        metrics.total_records = 100
        metrics.valid_records = 90
        metrics.filtered_records = 10
        metrics.filter_reasons["spam"] = 5
        metrics.filter_reasons["too_short"] = 5

        report = metrics.generate_report()
        assert "DATA QUALITY REPORT" in report
        assert "100" in report
        assert "90" in report

    def test_to_dict(self, metrics):
        """Test conversion to dictionary."""
        metrics.total_records = 50
        metrics.valid_records = 45

        result = metrics.to_dict()
        assert result["record_counts"]["total"] == 50
        assert result["record_counts"]["valid"] == 45


class TestPreprocessingConfig:
    """Test cases for PreprocessingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessingConfig()
        assert config.normalize_unicode is True
        assert config.expand_contractions is True
        assert config.min_tokens == 3
        assert config.max_tokens == 512

    def test_custom_config(self):
        """Test custom configuration."""
        config = PreprocessingConfig(
            expand_slang=True,
            min_tokens=5,
            max_tokens=1000,
            detect_spam=False,
        )
        assert config.expand_slang is True
        assert config.min_tokens == 5
        assert config.max_tokens == 1000
        assert config.detect_spam is False


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_apply_gold_standard_normalization(self):
        """Test quick normalization function."""
        text = "I don't think it's working"
        result = apply_gold_standard_normalization(text)
        assert result is not None
        assert len(result) > 0

    def test_normalize_for_nlp(self):
        """Test NLP normalization function."""
        text = "This is a SAMPLE text for NLP"
        result = normalize_for_nlp(text, preserve_case=False)
        assert result == result.lower()

    def test_normalize_for_nlp_preserve_case(self):
        """Test NLP normalization with case preservation."""
        text = "This is a SAMPLE text"
        result = normalize_for_nlp(text, preserve_case=True)
        # Case should be preserved
        assert "SAMPLE" in result or "sample" not in result.lower()

    def test_normalize_for_nlp_remove_stopwords(self):
        """Test NLP normalization with stopword removal."""
        text = "This is a sample text for testing"
        result = normalize_for_nlp(text, remove_stopwords=True)
        assert "is" not in result.split()
        assert "a" not in result.split()

    def test_create_processor_for_dataset(self):
        """Test dataset-specific processor creation."""
        # Test different dataset types
        for dataset_type in ["general", "social_media", "reviews", "news"]:
            processor = create_processor_for_dataset(dataset_type)
            assert isinstance(processor, GoldStandardTextProcessor)


class TestPreprocessDataframe:
    """Test cases for preprocess_dataframe function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4],
            "text": [
                "This is the first sample text",
                "Here is the second sample",
                "Third text sample here",
                "",  # Empty text
            ],
        })

    def test_basic_preprocessing(self, sample_df):
        """Test basic DataFrame preprocessing."""
        result_df, metrics = preprocess_dataframe(
            sample_df,
            text_column="text",
            drop_filtered=True,
        )
        assert "text_processed" in result_df.columns
        assert len(result_df) < len(sample_df)  # Empty text filtered

    def test_custom_output_column(self, sample_df):
        """Test custom output column name."""
        result_df, metrics = preprocess_dataframe(
            sample_df,
            text_column="text",
            output_column="cleaned_text",
            drop_filtered=False,
        )
        assert "cleaned_text" in result_df.columns

    def test_keep_filtered(self, sample_df):
        """Test keeping filtered rows."""
        result_df, metrics = preprocess_dataframe(
            sample_df,
            text_column="text",
            drop_filtered=False,
        )
        assert len(result_df) == len(sample_df)

    def test_metrics_returned(self, sample_df):
        """Test that metrics are returned."""
        _, metrics = preprocess_dataframe(
            sample_df,
            text_column="text",
        )
        assert isinstance(metrics, DataQualityMetrics)
        assert metrics.total_records > 0


class TestDictionaries:
    """Test cases for preprocessing dictionaries."""

    def test_contractions_coverage(self):
        """Test that common contractions are covered."""
        common_contractions = ["don't", "can't", "won't", "isn't", "aren't"]
        for contraction in common_contractions:
            assert contraction in CONTRACTIONS

    def test_slang_coverage(self):
        """Test that common slang terms are covered."""
        common_slang = ["lol", "btw", "imo", "tbh", "idk"]
        for slang in common_slang:
            assert slang in SLANG_MAP
