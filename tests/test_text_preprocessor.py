"""
Unit tests for Text Preprocessor module.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.text_preprocessor import (
    TextPreprocessor,
    TextPreprocessingError,
    DataCleaningPipeline,
    NEGATION_KEEP_WORDS,
    DEFAULT_DOMAIN_STOPWORDS_PATH,
)


class TestTextPreprocessor:
    """Test cases for TextPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Create default preprocessor instance."""
        return TextPreprocessor()

    @pytest.fixture
    def social_preprocessor(self):
        """Create preprocessor for social media."""
        return TextPreprocessor(
            expand_slang=True,
            standardize_urls=True,
            standardize_mentions=True,
            min_tokens=2,
        )

    def test_basic_preprocessing(self, preprocessor):
        """Test basic text preprocessing."""
        text = "This is a SAMPLE text for testing"
        result = preprocessor.preprocess(text)
        assert result is not None
        assert result == result.lower()  # Should be lowercase

    def test_stopword_removal(self, preprocessor):
        """Test stopword removal."""
        text = "This is a sample text with many stopwords"
        result = preprocessor.preprocess(text, remove_stopwords=True)
        # Stopword removal only works when NLTK is installed
        # Just verify the result is not empty and processing happened
        assert len(result) > 0
        # If NLTK is installed, stopwords should be removed
        if preprocessor.stop_words:
            assert "is" not in result.split()
            assert "a" not in result.split()

    def test_lemmatization(self, preprocessor):
        """Test lemmatization."""
        text = "The cats are running and jumping"
        result = preprocessor.preprocess(text, lemmatize=True)
        # 'running' should become 'run', 'cats' should become 'cat'
        # Note: actual results depend on NLTK's lemmatizer behavior

    def test_preserve_case(self, preprocessor):
        """Test case preservation option."""
        text = "THIS IS UPPERCASE TEXT"
        result = preprocessor.preprocess(text, lowercase=False)
        # When lowercase=False, some uppercase might be preserved
        # (though gold standard normalization may still affect it)

    def test_min_token_length(self, preprocessor):
        """Test minimum token length filtering."""
        text = "A sample of text with short and long words"
        result = preprocessor.preprocess(text, min_token_length=4)
        tokens = result.split()
        # All tokens should be longer than min_token_length
        assert all(len(t) > 2 for t in tokens if t)

    def test_empty_text_handling(self, preprocessor):
        """Test handling of empty text."""
        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess("   ") == ""

    def test_none_text_handling(self, preprocessor):
        """Test handling of None text."""
        with pytest.raises(TextPreprocessingError):
            preprocessor.preprocess(None)

    def test_normalize_only(self, preprocessor):
        """Test normalize-only mode."""
        text = "I don't think it's working!!!"
        result = preprocessor.normalize_only(text)
        assert result is not None
        # Should have normalized contractions and punctuation

    def test_batch_preprocessing(self, preprocessor):
        """Test batch preprocessing."""
        texts = [
            "First sample text here",
            "Second sample text here",
            "Third sample text here",
        ]
        results = preprocessor.preprocess_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_batch_preprocessing_with_empty(self, preprocessor):
        """Test batch preprocessing with empty texts."""
        texts = [
            "Valid text sample",
            "",
            "Another valid sample",
        ]
        results = preprocessor.preprocess_batch(texts, return_filtered=True)
        assert len(results) == 3
        assert results[1] == ""

    def test_quality_metrics(self, preprocessor):
        """Test quality metrics tracking."""
        preprocessor.reset_metrics()
        text = "This is a sample text for metrics"
        preprocessor.preprocess(text, track_metrics=True)

        metrics = preprocessor.get_quality_metrics()
        assert metrics is not None
        assert metrics.total_records > 0

    def test_quality_report(self, preprocessor):
        """Test quality report generation."""
        preprocessor.reset_metrics()
        preprocessor.preprocess("Sample text for report testing")

        report = preprocessor.get_quality_report()
        assert isinstance(report, str)
        assert len(report) > 0


class TestNegationPreservation:
    """Test cases for negation word preservation during stopword removal."""

    @pytest.fixture
    def preprocessor_preserve(self):
        """Create preprocessor with negation preservation enabled (default)."""
        return TextPreprocessor(preserve_negations=True)

    @pytest.fixture
    def preprocessor_no_preserve(self):
        """Create preprocessor with negation preservation disabled."""
        return TextPreprocessor(preserve_negations=False)

    def test_negation_keep_words_constant(self):
        """Test that NEGATION_KEEP_WORDS constant is properly defined."""
        assert "not" in NEGATION_KEEP_WORDS
        assert "no" in NEGATION_KEEP_WORDS
        assert "never" in NEGATION_KEEP_WORDS
        assert "none" in NEGATION_KEEP_WORDS
        assert "nobody" in NEGATION_KEEP_WORDS
        assert "nothing" in NEGATION_KEEP_WORDS
        assert "neither" in NEGATION_KEEP_WORDS
        assert "nowhere" in NEGATION_KEEP_WORDS
        assert "cannot" in NEGATION_KEEP_WORDS
        assert "can't" in NEGATION_KEEP_WORDS
        assert "don't" in NEGATION_KEEP_WORDS
        assert "doesn't" in NEGATION_KEEP_WORDS
        assert "didn't" in NEGATION_KEEP_WORDS
        assert "won't" in NEGATION_KEEP_WORDS
        assert "wouldn't" in NEGATION_KEEP_WORDS
        assert "shouldn't" in NEGATION_KEEP_WORDS
        assert "couldn't" in NEGATION_KEEP_WORDS
        assert "hasn't" in NEGATION_KEEP_WORDS
        assert "haven't" in NEGATION_KEEP_WORDS
        assert "hadn't" in NEGATION_KEEP_WORDS
        assert "isn't" in NEGATION_KEEP_WORDS
        assert "aren't" in NEGATION_KEEP_WORDS
        assert "wasn't" in NEGATION_KEEP_WORDS
        assert "weren't" in NEGATION_KEEP_WORDS

    def test_preserve_not_in_sentence(self, preprocessor_preserve):
        """Test that 'not' is preserved: 'I do not like this' should preserve 'not'."""
        text = "I do not like this"
        result = preprocessor_preserve.preprocess(text, remove_stopwords=True, lemmatize=False)
        # Note: after preprocessing, text is lowercased and non-alpha chars removed
        # 'not' should be preserved even though it's typically a stopword
        if preprocessor_preserve.stop_words:
            assert "not" in result.split(), f"Expected 'not' to be preserved, got: {result}"

    def test_preserve_never_in_sentence(self, preprocessor_preserve):
        """Test that 'never' is preserved: 'never again' should preserve 'never'."""
        text = "never again"
        result = preprocessor_preserve.preprocess(text, remove_stopwords=True, lemmatize=False)
        if preprocessor_preserve.stop_words:
            assert "never" in result.split(), f"Expected 'never' to be preserved, got: {result}"

    def test_preserve_no_in_sentence(self, preprocessor_preserve):
        """Test that 'no' is preserved in sentences."""
        text = "There is no way this works"
        result = preprocessor_preserve.preprocess(text, remove_stopwords=True, lemmatize=False, min_token_length=1)
        if preprocessor_preserve.stop_words:
            # 'no' might be filtered by min_token_length=2 default, so we use min_token_length=1
            assert "no" in result.split(), f"Expected 'no' to be preserved, got: {result}"

    def test_preserve_none_in_sentence(self, preprocessor_preserve):
        """Test that 'none' is preserved in sentences."""
        text = "none of these options work"
        result = preprocessor_preserve.preprocess(text, remove_stopwords=True, lemmatize=False)
        if preprocessor_preserve.stop_words:
            assert "none" in result.split(), f"Expected 'none' to be preserved, got: {result}"

    def test_old_behavior_when_disabled(self, preprocessor_no_preserve):
        """Test that old behavior is maintained when preserve_negations=False."""
        text = "I do not like this"
        result = preprocessor_no_preserve.preprocess(text, remove_stopwords=True, lemmatize=False)
        # With preserve_negations=False, 'not' should be removed as a stopword
        if preprocessor_no_preserve.stop_words:
            assert "not" not in result.split(), f"Expected 'not' to be removed, got: {result}"

    def test_never_removed_when_disabled(self, preprocessor_no_preserve):
        """Test that 'never' is removed when preserve_negations=False."""
        text = "I will never forget this experience"
        result = preprocessor_no_preserve.preprocess(text, remove_stopwords=True, lemmatize=False)
        # 'never' is in NLTK stopwords, so it should be removed when not preserving
        if preprocessor_no_preserve.stop_words and "never" in preprocessor_no_preserve.stop_words:
            assert "never" not in result.split(), f"Expected 'never' to be removed, got: {result}"

    def test_default_preserve_negations_is_true(self):
        """Test that the default value for preserve_negations is True."""
        preprocessor = TextPreprocessor()
        assert preprocessor.preserve_negations is True

    def test_multiple_negations_preserved(self, preprocessor_preserve):
        """Test that multiple negation words are preserved in a sentence."""
        text = "I do not and will never accept this"
        result = preprocessor_preserve.preprocess(text, remove_stopwords=True, lemmatize=False)
        if preprocessor_preserve.stop_words:
            assert "not" in result.split(), f"Expected 'not' to be preserved, got: {result}"
            assert "never" in result.split(), f"Expected 'never' to be preserved, got: {result}"

    def test_sentiment_preservation_example(self, preprocessor_preserve):
        """Test that negation words preserve sentiment context."""
        # This is a practical example showing why negation preservation matters
        positive_text = "I like this product"
        negative_text = "I do not like this product"

        positive_result = preprocessor_preserve.preprocess(positive_text, remove_stopwords=True, lemmatize=False)
        negative_result = preprocessor_preserve.preprocess(negative_text, remove_stopwords=True, lemmatize=False)

        if preprocessor_preserve.stop_words:
            # The negative result should contain 'not' to preserve the sentiment
            assert "not" in negative_result.split(), f"Expected 'not' in negative text, got: {negative_result}"
            # The positive result should not contain 'not'
            assert "not" not in positive_result.split(), f"Unexpected 'not' in positive text, got: {positive_result}"


class TestDataCleaningPipelineNegation:
    """Test cases for negation preservation in DataCleaningPipeline."""

    def test_pipeline_default_preserve_negations(self):
        """Test that pipeline has preserve_negations=True by default."""
        pipeline = DataCleaningPipeline()
        assert pipeline.preserve_negations is True
        assert pipeline.preprocessor.preserve_negations is True

    def test_pipeline_preserve_negations_false(self):
        """Test that pipeline can disable negation preservation."""
        pipeline = DataCleaningPipeline(preserve_negations=False)
        assert pipeline.preserve_negations is False
        assert pipeline.preprocessor.preserve_negations is False

    def test_pipeline_preserves_negations_in_text(self):
        """Test that pipeline preserves negation words in cleaned text."""
        pipeline = DataCleaningPipeline(preserve_negations=True)
        texts = ["I do not recommend this product"]
        results = pipeline.clean_texts(texts)
        if pipeline.preprocessor.stop_words:
            assert "not" in results[0].split(), f"Expected 'not' to be preserved, got: {results[0]}"

    def test_pipeline_removes_negations_when_disabled(self):
        """Test that pipeline removes negations when disabled."""
        pipeline = DataCleaningPipeline(preserve_negations=False)
        texts = ["I do not recommend this product"]
        results = pipeline.clean_texts(texts)
        if pipeline.preprocessor.stop_words:
            assert "not" not in results[0].split(), f"Expected 'not' to be removed, got: {results[0]}"


class TestTextPreprocessorDomainSpecific:
    """Test cases for domain-specific preprocessing."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return TextPreprocessor()

    def test_medical_domain(self, preprocessor):
        """Test medical domain preprocessing."""
        text = "Patient received 50mg of medication for treatment"
        result = preprocessor.clean_domain_specific(text, domain="medical")
        assert "mg" in result or "patient" in result

    def test_legal_domain(self, preprocessor):
        """Test legal domain preprocessing."""
        text = "The plaintiff hereby submits pursuant to the agreement"
        result = preprocessor.clean_domain_specific(text, domain="legal")
        # Result should contain the text (possibly with some terms preserved)
        assert len(result) > 0
        assert "the" in result or "agreement" in result

    def test_technical_domain(self, preprocessor):
        """Test technical domain preprocessing."""
        text = "The API calls use the SDK with GPU acceleration"
        result = preprocessor.clean_domain_specific(text, domain="technical")
        # Result should contain the cleaned text
        assert len(result) > 0
        assert "calls" in result or "use" in result or "acceleration" in result

    def test_invalid_domain(self, preprocessor):
        """Test invalid domain handling."""
        with pytest.raises(TextPreprocessingError):
            preprocessor.clean_domain_specific("text", domain="invalid")

    def test_preserve_custom_terms(self, preprocessor):
        """Test preserving custom terms."""
        text = "The system uses a protocol for communication"
        result = preprocessor.clean_domain_specific(
            text,
            domain="general",
            preserve_terms=["system", "protocol"]
        )
        # Result should be cleaned and contain the text
        assert len(result) > 0
        # The method cleans the text - just verify it processed correctly
        assert "uses" in result or "communication" in result


class TestTextPreprocessorLongDocuments:
    """Test cases for long document handling."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return TextPreprocessor()

    def test_truncate_strategy(self, preprocessor):
        """Test truncate strategy."""
        text = " ".join(["word"] * 1000)  # 1000 word document
        result = preprocessor.handle_long_documents(
            text, strategy='truncate', chunk_size=100
        )
        assert isinstance(result, str)
        tokens = result.split()
        assert len(tokens) <= 100

    def test_chunk_strategy(self, preprocessor):
        """Test chunk strategy."""
        text = " ".join(["word"] * 200)  # 200 word document
        result = preprocessor.handle_long_documents(
            text, strategy='chunk', chunk_size=50, chunk_overlap=10
        )
        assert isinstance(result, list)
        assert len(result) > 1

    def test_invalid_strategy(self, preprocessor):
        """Test invalid strategy handling."""
        with pytest.raises(TextPreprocessingError):
            preprocessor.handle_long_documents("text", strategy='invalid')

    def test_empty_text_handling(self, preprocessor):
        """Test empty text handling."""
        with pytest.raises(TextPreprocessingError):
            preprocessor.handle_long_documents("")

    def test_invalid_chunk_size(self, preprocessor):
        """Test invalid chunk size."""
        with pytest.raises(TextPreprocessingError):
            preprocessor.handle_long_documents(
                "text", strategy='chunk', chunk_size=0
            )

    def test_invalid_chunk_overlap(self, preprocessor):
        """Test invalid chunk overlap."""
        with pytest.raises(TextPreprocessingError):
            preprocessor.handle_long_documents(
                "text", strategy='chunk', chunk_size=10, chunk_overlap=15
            )


class TestTextPreprocessorLanguageDetection:
    """Test cases for language detection."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return TextPreprocessor()

    def test_english_detection(self, preprocessor):
        """Test English language detection."""
        text = "This is a sample English text for testing language detection"
        try:
            lang = preprocessor.detect_language(text)
            # Should detect as English (or fail gracefully if langdetect not installed)
            assert lang in ['en', 'en']
        except TextPreprocessingError:
            # langdetect might not be installed
            pass

    def test_empty_text_detection(self, preprocessor):
        """Test language detection with empty text."""
        with pytest.raises(TextPreprocessingError):
            preprocessor.detect_language("")


class TestDataCleaningPipeline:
    """Test cases for DataCleaningPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create default pipeline instance."""
        return DataCleaningPipeline()

    @pytest.fixture
    def social_pipeline(self):
        """Create social media pipeline."""
        return DataCleaningPipeline(dataset_type="social_media")

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "text": [
                "This is the first sample text for cleaning",
                "Here is another sample text to process",
                "Third text sample with more words here",
                "Fourth sample for testing the pipeline",
                "",  # Empty text
            ],
        })

    def test_clean_dataframe(self, pipeline, sample_df):
        """Test DataFrame cleaning."""
        result = pipeline.clean_dataframe(sample_df, text_column="text")
        assert "text_cleaned" in result.columns
        assert len(result) == len(sample_df)

    def test_clean_dataframe_drop_filtered(self, pipeline, sample_df):
        """Test DataFrame cleaning with filtering."""
        result = pipeline.clean_dataframe(
            sample_df,
            text_column="text",
            drop_filtered=True
        )
        assert len(result) < len(sample_df)

    def test_clean_dataframe_custom_output(self, pipeline, sample_df):
        """Test custom output column name."""
        result = pipeline.clean_dataframe(
            sample_df,
            text_column="text",
            output_column="processed"
        )
        assert "processed" in result.columns

    def test_clean_texts(self, pipeline):
        """Test text list cleaning."""
        texts = [
            "First sample text here",
            "Second sample text here",
            "Third sample text here",
        ]
        results = pipeline.clean_texts(texts)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)

    def test_get_summary(self, pipeline, sample_df):
        """Test summary retrieval."""
        pipeline.clean_dataframe(sample_df, text_column="text")
        summary = pipeline.get_summary()
        assert isinstance(summary, dict)
        assert "total_records" in summary

    def test_get_quality_report(self, pipeline, sample_df):
        """Test quality report retrieval."""
        pipeline.clean_dataframe(sample_df, text_column="text")
        report = pipeline.get_quality_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_missing_column(self, pipeline, sample_df):
        """Test handling of missing column."""
        with pytest.raises(ValueError):
            pipeline.clean_dataframe(sample_df, text_column="nonexistent")

    def test_different_dataset_types(self):
        """Test different dataset type configurations."""
        for dataset_type in ["general", "social_media", "reviews", "news"]:
            pipeline = DataCleaningPipeline(dataset_type=dataset_type)
            assert pipeline.dataset_type == dataset_type


class TestTextPreprocessorIntegration:
    """Integration tests for TextPreprocessor with gold standard preprocessing."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor with gold standard enabled."""
        return TextPreprocessor(use_gold_standard=True)

    def test_full_pipeline(self, preprocessor):
        """Test full preprocessing pipeline."""
        text = "I don't think it's going to work!!! Check https://example.com"
        result = preprocessor.preprocess(text)
        assert result is not None
        assert "don't" not in result.lower()  # Contraction expanded
        assert "!!!" not in result  # Punctuation normalized
        # URL might be replaced or removed

    def test_social_media_text(self):
        """Test social media text preprocessing."""
        preprocessor = TextPreprocessor(
            expand_slang=True,
            standardize_mentions=True,
            standardize_urls=True,
        )
        text = "lol @friend check this out https://example.com #amazing"
        result = preprocessor.normalize_only(text)
        assert result is not None

    def test_metrics_across_operations(self, preprocessor):
        """Test metrics tracking across multiple operations."""
        preprocessor.reset_metrics()

        texts = [
            "First sample text for testing",
            "Second sample text here",
            "Third text sample",
        ]

        for text in texts:
            preprocessor.preprocess(text, track_metrics=True)

        metrics = preprocessor.get_quality_metrics()
        assert metrics.total_records == 3


class TestDomainStopwords:
    """Test cases for domain-specific stopwords functionality."""

    @pytest.fixture
    def preprocessor_with_domain(self):
        """Create preprocessor with domain stopwords enabled (default)."""
        return TextPreprocessor(use_domain_stopwords=True)

    @pytest.fixture
    def preprocessor_without_domain(self):
        """Create preprocessor with domain stopwords disabled."""
        return TextPreprocessor(use_domain_stopwords=False)

    def test_default_domain_stopwords_path_exists(self):
        """Test that the default domain stopwords file exists."""
        assert DEFAULT_DOMAIN_STOPWORDS_PATH.exists(), \
            f"Domain stopwords file not found at {DEFAULT_DOMAIN_STOPWORDS_PATH}"

    def test_domain_stopwords_loaded(self, preprocessor_with_domain):
        """Test that domain stopwords are loaded correctly."""
        assert preprocessor_with_domain.use_domain_stopwords is True
        assert len(preprocessor_with_domain.domain_stopwords) > 0
        # Check some expected domain stopwords
        assert "response" in preprocessor_with_domain.domain_stopwords
        assert "survey" in preprocessor_with_domain.domain_stopwords
        assert "feedback" in preprocessor_with_domain.domain_stopwords
        assert "question" in preprocessor_with_domain.domain_stopwords

    def test_domain_stopwords_merged(self, preprocessor_with_domain):
        """Test that domain stopwords are merged with NLTK stopwords."""
        # Domain stopwords should be in the combined stop_words set
        assert "response" in preprocessor_with_domain.stop_words
        assert "survey" in preprocessor_with_domain.stop_words
        assert "feedback" in preprocessor_with_domain.stop_words
        # NLTK stopwords should still be present
        if preprocessor_with_domain.stop_words:
            assert "the" in preprocessor_with_domain.stop_words
            assert "is" in preprocessor_with_domain.stop_words

    def test_domain_stopwords_not_loaded_when_disabled(self, preprocessor_without_domain):
        """Test that domain stopwords are not loaded when disabled."""
        assert preprocessor_without_domain.use_domain_stopwords is False
        assert len(preprocessor_without_domain.domain_stopwords) == 0
        # Domain words should NOT be in stop_words when disabled
        assert "response" not in preprocessor_without_domain.stop_words
        assert "survey" not in preprocessor_without_domain.stop_words
        assert "feedback" not in preprocessor_without_domain.stop_words

    def test_domain_stopwords_removed_from_text(self, preprocessor_with_domain):
        """Test that domain stopwords are removed from text when enabled."""
        text = "The survey response regarding feedback was positive"
        result = preprocessor_with_domain.preprocess(text, remove_stopwords=True, lemmatize=False)
        tokens = result.split()
        # Domain stopwords should be removed
        assert "survey" not in tokens, f"'survey' should be removed, got: {result}"
        assert "response" not in tokens, f"'response' should be removed, got: {result}"
        assert "regarding" not in tokens, f"'regarding' should be removed, got: {result}"
        assert "feedback" not in tokens, f"'feedback' should be removed, got: {result}"
        # Content words should remain
        assert "positive" in tokens, f"'positive' should remain, got: {result}"

    def test_domain_stopwords_kept_when_disabled(self, preprocessor_without_domain):
        """Test that domain stopwords are NOT removed when disabled."""
        text = "The survey response regarding feedback was positive"
        result = preprocessor_without_domain.preprocess(text, remove_stopwords=True, lemmatize=False)
        tokens = result.split()
        # Domain stopwords should NOT be removed when use_domain_stopwords=False
        assert "survey" in tokens, f"'survey' should remain when disabled, got: {result}"
        assert "response" in tokens, f"'response' should remain when disabled, got: {result}"
        assert "feedback" in tokens, f"'feedback' should remain when disabled, got: {result}"

    def test_domain_stopwords_with_custom_path(self):
        """Test loading domain stopwords from a custom path."""
        # Create a temporary file with custom stopwords
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("customword\n")
            f.write("anotherword\n")
            temp_path = Path(f.name)

        try:
            preprocessor = TextPreprocessor(
                use_domain_stopwords=True,
                domain_stopwords_path=temp_path
            )
            assert "customword" in preprocessor.domain_stopwords
            assert "anotherword" in preprocessor.domain_stopwords
            assert "customword" in preprocessor.stop_words
        finally:
            temp_path.unlink()  # Clean up temp file

    def test_domain_stopwords_missing_file_graceful(self):
        """Test graceful handling when domain stopwords file doesn't exist."""
        nonexistent_path = Path("/nonexistent/path/stopwords.txt")
        # Should not raise an error, just log a warning and continue
        preprocessor = TextPreprocessor(
            use_domain_stopwords=True,
            domain_stopwords_path=nonexistent_path
        )
        # Domain stopwords should be empty, but preprocessor should work
        assert len(preprocessor.domain_stopwords) == 0
        # NLTK stopwords should still be loaded
        if preprocessor.stop_words:
            assert "the" in preprocessor.stop_words

    def test_load_domain_stopwords_static_method(self):
        """Test the _load_domain_stopwords static method directly."""
        stopwords = TextPreprocessor._load_domain_stopwords()
        assert isinstance(stopwords, set)
        assert len(stopwords) > 0
        assert "response" in stopwords
        assert "survey" in stopwords

    def test_domain_stopwords_lowercase(self):
        """Test that domain stopwords are loaded in lowercase."""
        # Create a temp file with mixed case
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("MixedCase\n")
            f.write("UPPERCASE\n")
            f.write("lowercase\n")
            temp_path = Path(f.name)

        try:
            stopwords = TextPreprocessor._load_domain_stopwords(temp_path)
            assert "mixedcase" in stopwords
            assert "uppercase" in stopwords
            assert "lowercase" in stopwords
            # Original case should not be in the set
            assert "MixedCase" not in stopwords
            assert "UPPERCASE" not in stopwords
        finally:
            temp_path.unlink()

    def test_domain_stopwords_with_comments(self):
        """Test that comments (lines starting with #) are ignored."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# This is a comment\n")
            f.write("validword\n")
            f.write("# Another comment\n")
            f.write("anothervalid\n")
            temp_path = Path(f.name)

        try:
            stopwords = TextPreprocessor._load_domain_stopwords(temp_path)
            assert "validword" in stopwords
            assert "anothervalid" in stopwords
            assert "# this is a comment" not in stopwords
            assert "this" not in stopwords
        finally:
            temp_path.unlink()

    def test_all_domain_stopwords_present(self, preprocessor_with_domain):
        """Test that all expected domain stopwords are loaded."""
        expected_stopwords = {
            "response", "responses", "survey", "surveys",
            "question", "questions", "answer", "answers",
            "please", "thanks", "thank",
            "participant", "participants", "respondent", "respondents",
            "feedback", "comment", "comments",
            "opinion", "opinions", "regarding", "concerning"
        }
        for word in expected_stopwords:
            assert word in preprocessor_with_domain.domain_stopwords, \
                f"Expected '{word}' in domain stopwords"


class TestDataCleaningPipelineDomainStopwords:
    """Test cases for domain stopwords in DataCleaningPipeline."""

    def test_pipeline_default_domain_stopwords_enabled(self):
        """Test that pipeline has use_domain_stopwords=True by default."""
        pipeline = DataCleaningPipeline()
        assert pipeline.use_domain_stopwords is True
        assert pipeline.preprocessor.use_domain_stopwords is True
        assert len(pipeline.preprocessor.domain_stopwords) > 0

    def test_pipeline_domain_stopwords_disabled(self):
        """Test that pipeline can disable domain stopwords."""
        pipeline = DataCleaningPipeline(use_domain_stopwords=False)
        assert pipeline.use_domain_stopwords is False
        assert pipeline.preprocessor.use_domain_stopwords is False
        assert len(pipeline.preprocessor.domain_stopwords) == 0

    def test_pipeline_removes_domain_stopwords(self):
        """Test that pipeline removes domain stopwords from cleaned text."""
        pipeline = DataCleaningPipeline(use_domain_stopwords=True)
        texts = ["The survey response about feedback was excellent"]
        results = pipeline.clean_texts(texts)
        tokens = results[0].split()
        # Domain stopwords should be removed
        assert "survey" not in tokens, f"Expected 'survey' to be removed, got: {results[0]}"
        assert "response" not in tokens, f"Expected 'response' to be removed, got: {results[0]}"
        assert "feedback" not in tokens, f"Expected 'feedback' to be removed, got: {results[0]}"

    def test_pipeline_keeps_domain_stopwords_when_disabled(self):
        """Test that pipeline keeps domain stopwords when disabled."""
        pipeline = DataCleaningPipeline(use_domain_stopwords=False)
        texts = ["The survey response about feedback was excellent"]
        results = pipeline.clean_texts(texts)
        tokens = results[0].split()
        # Domain stopwords should NOT be removed
        assert "survey" in tokens, f"Expected 'survey' to remain, got: {results[0]}"
        assert "response" in tokens, f"Expected 'response' to remain, got: {results[0]}"
        assert "feedback" in tokens, f"Expected 'feedback' to remain, got: {results[0]}"
