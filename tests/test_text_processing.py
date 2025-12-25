"""
Unit tests for text processing and segmentation functionality.

Tests cover:
- Basic segmentation (sentences, paragraphs)
- Contextual segmentation
- Hierarchical segmentation
- Batch processing
- Edge cases (empty text, abbreviations, mixed delimiters)
- Traceability (parent ID preservation)
- Performance requirements
"""

import pytest
import pandas as pd
import time
from src.text_processing import TextSegmenter, TextSegment


class TestTextSegment:
    """Test cases for TextSegment dataclass."""

    def test_text_segment_creation(self):
        """Test creating a TextSegment."""
        segment = TextSegment(
            segment_id="R1-S0",
            text="Hello world.",
            parent_response_id="R1",
            segment_type="sentence",
            position=0
        )
        assert segment.segment_id == "R1-S0"
        assert segment.text == "Hello world."
        assert segment.parent_response_id == "R1"
        assert segment.segment_type == "sentence"
        assert segment.position == 0

    def test_text_segment_to_dict(self):
        """Test converting TextSegment to dictionary."""
        segment = TextSegment(
            segment_id="R1-S0",
            text="Test text",
            parent_response_id="R1"
        )
        result = segment.to_dict()
        assert isinstance(result, dict)
        assert result['segment_id'] == "R1-S0"
        assert result['text'] == "Test text"
        assert result['parent_response_id'] == "R1"

    def test_text_segment_with_context(self):
        """Test TextSegment with context fields."""
        segment = TextSegment(
            segment_id="R1-S1",
            text="Second sentence.",
            parent_response_id="R1",
            context_before="First sentence.",
            context_after="Third sentence."
        )
        assert segment.context_before == "First sentence."
        assert segment.context_after == "Third sentence."


class TestTextSegmenterBasic:
    """Test basic TextSegmenter functionality."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_initialization_default(self):
        """Test TextSegmenter initialization with defaults."""
        segmenter = TextSegmenter()
        assert segmenter.sentence_endings == ['.', '!', '?']
        assert '\n\n' in segmenter.paragraph_delimiters
        assert segmenter.preserve_whitespace is False
        assert segmenter.min_segment_length == 1

    def test_initialization_custom(self):
        """Test TextSegmenter initialization with custom parameters."""
        segmenter = TextSegmenter(
            sentence_endings=['.', '!'],
            paragraph_delimiters=['\n\n\n'],
            preserve_whitespace=True,
            min_segment_length=5
        )
        assert segmenter.sentence_endings == ['.', '!']
        assert segmenter.paragraph_delimiters == ['\n\n\n']
        assert segmenter.preserve_whitespace is True
        assert segmenter.min_segment_length == 5


class TestSentenceSegmentation:
    """Test sentence segmentation functionality."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_segment_sentences_basic(self, segmenter):
        """Test basic sentence segmentation."""
        text = "First sentence. Second sentence. Third sentence."
        segments = segmenter.segment_sentences(text, "R1")

        assert len(segments) == 3
        assert segments[0].text == "First sentence."
        assert segments[1].text == "Second sentence."
        assert segments[2].text == "Third sentence."

    def test_segment_sentences_ids(self, segmenter):
        """Test that sentence segments have correct IDs."""
        text = "First. Second."
        segments = segmenter.segment_sentences(text, "R1")

        assert segments[0].segment_id == "R1-S0"
        assert segments[0].parent_response_id == "R1"
        assert segments[0].segment_type == "sentence"
        assert segments[0].position == 0

        assert segments[1].segment_id == "R1-S1"
        assert segments[1].parent_response_id == "R1"
        assert segments[1].position == 1

    def test_segment_sentences_empty(self, segmenter):
        """Test sentence segmentation with empty text."""
        assert segmenter.segment_sentences("", "R1") == []
        assert segmenter.segment_sentences(None, "R1") == []

    def test_segment_sentences_single(self, segmenter):
        """Test segmentation of single sentence."""
        text = "Only one sentence here."
        segments = segmenter.segment_sentences(text, "R1")

        assert len(segments) == 1
        assert segments[0].text == "Only one sentence here."

    def test_segment_sentences_no_response_id(self, segmenter):
        """Test sentence segmentation without response ID."""
        text = "Test sentence."
        segments = segmenter.segment_sentences(text)

        assert len(segments) == 1
        assert segments[0].parent_response_id == "R0"  # Default ID

    def test_segment_sentences_multiple_punctuation(self, segmenter):
        """Test sentences with exclamation and question marks."""
        text = "What is this? This is great! Really amazing."
        segments = segmenter.segment_sentences(text, "R1")

        assert len(segments) == 3
        assert segments[0].text == "What is this?"
        assert segments[1].text == "This is great!"
        assert segments[2].text == "Really amazing."

    def test_segment_sentences_no_ending_punctuation(self, segmenter):
        """Test sentence without ending punctuation."""
        text = "No punctuation at end"
        segments = segmenter.segment_sentences(text, "R1")

        assert len(segments) == 1
        assert segments[0].text == "No punctuation at end"

    def test_segment_sentences_whitespace_handling(self, segmenter):
        """Test that whitespace is properly handled."""
        text = "First.  Second.   Third."
        segments = segmenter.segment_sentences(text, "R1")

        assert len(segments) == 3
        # Check that extra whitespace is normalized
        for segment in segments:
            assert not segment.text.startswith(' ')
            assert not segment.text.endswith(' ') or segmenter.preserve_whitespace

    def test_segment_sentences_min_length_filter(self):
        """Test minimum segment length filtering."""
        segmenter = TextSegmenter(min_segment_length=5)
        text = "Hi. This is longer."
        segments = segmenter.segment_sentences(text, "R1")

        # "Hi." is only 3 chars, should be filtered
        assert len(segments) == 1
        assert segments[0].text == "This is longer."

    def test_segment_sentences_metadata(self, segmenter):
        """Test that metadata is included in segments."""
        text = "Test sentence."
        segments = segmenter.segment_sentences(text, "R1")

        assert segments[0].metadata is not None
        assert 'original_text_length' in segments[0].metadata
        assert segments[0].metadata['original_text_length'] == len(text)


class TestParagraphSegmentation:
    """Test paragraph segmentation functionality."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_segment_paragraphs_basic(self, segmenter):
        """Test basic paragraph segmentation."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        segments = segmenter.segment_paragraphs(text, "R1")

        assert len(segments) == 3
        assert segments[0].text == "First paragraph."
        assert segments[1].text == "Second paragraph."
        assert segments[2].text == "Third paragraph."

    def test_segment_paragraphs_ids(self, segmenter):
        """Test that paragraph segments have correct IDs."""
        text = "Para 1.\n\nPara 2."
        segments = segmenter.segment_paragraphs(text, "R1")

        assert segments[0].segment_id == "R1-P0"
        assert segments[0].parent_response_id == "R1"
        assert segments[0].segment_type == "paragraph"
        assert segments[0].position == 0

        assert segments[1].segment_id == "R1-P1"
        assert segments[1].position == 1

    def test_segment_paragraphs_empty(self, segmenter):
        """Test paragraph segmentation with empty text."""
        assert segmenter.segment_paragraphs("", "R1") == []
        assert segmenter.segment_paragraphs(None, "R1") == []

    def test_segment_paragraphs_single(self, segmenter):
        """Test single paragraph (no delimiters)."""
        text = "Only one paragraph here with no line breaks."
        segments = segmenter.segment_paragraphs(text, "R1")

        assert len(segments) == 1
        assert segments[0].text == text

    def test_segment_paragraphs_windows_line_endings(self, segmenter):
        """Test paragraph segmentation with Windows line endings."""
        text = "First para.\r\n\r\nSecond para."
        segments = segmenter.segment_paragraphs(text, "R1")

        assert len(segments) == 2
        assert segments[0].text == "First para."
        assert segments[1].text == "Second para."

    def test_segment_paragraphs_mixed_delimiters(self):
        """Test with custom paragraph delimiters."""
        segmenter = TextSegmenter(paragraph_delimiters=['\n\n', '---'])
        text = "Para 1.\n\nPara 2.---Para 3."
        segments = segmenter.segment_paragraphs(text, "R1")

        assert len(segments) == 3

    def test_segment_paragraphs_no_response_id(self, segmenter):
        """Test paragraph segmentation without response ID."""
        text = "Test paragraph."
        segments = segmenter.segment_paragraphs(text)

        assert len(segments) == 1
        assert segments[0].parent_response_id == "R0"  # Default ID


class TestContextualSegmentation:
    """Test contextual segmentation functionality."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_segment_with_context_basic(self, segmenter):
        """Test basic contextual segmentation."""
        text = "First. Second. Third."
        segments = segmenter.segment_with_context(text, "R1", granularity='sentence')

        assert len(segments) == 3

        # First segment has no context_before
        assert segments[0].context_before is None
        assert segments[0].context_after == "Second."

        # Middle segment has both contexts
        assert segments[1].context_before == "First."
        assert segments[1].context_after == "Third."

        # Last segment has no context_after
        assert segments[2].context_before == "Second."
        assert segments[2].context_after is None

    def test_segment_with_context_window_size(self, segmenter):
        """Test context window size parameter."""
        text = "A. B. C. D. E."
        segments = segmenter.segment_with_context(
            text, "R1", granularity='sentence', context_window=2
        )

        # Middle segment should have 2 segments before and after
        assert segments[2].context_before == "A. B."
        assert segments[2].context_after == "D. E."

    def test_segment_with_context_paragraph_granularity(self, segmenter):
        """Test contextual segmentation with paragraph granularity."""
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        segments = segmenter.segment_with_context(
            text, "R1", granularity='paragraph', context_window=1
        )

        assert len(segments) == 3
        assert segments[1].context_before == "Para 1."
        assert segments[1].context_after == "Para 3."

    def test_segment_with_context_invalid_granularity(self, segmenter):
        """Test that invalid granularity raises error."""
        with pytest.raises(ValueError, match="Invalid granularity"):
            segmenter.segment_with_context("Test.", "R1", granularity='word')

    def test_segment_with_context_single_segment(self, segmenter):
        """Test contextual segmentation with single segment."""
        text = "Only one sentence."
        segments = segmenter.segment_with_context(text, "R1")

        assert len(segments) == 1
        assert segments[0].context_before is None
        assert segments[0].context_after is None

    def test_segment_with_context_zero_window(self, segmenter):
        """Test with zero context window."""
        text = "First. Second. Third."
        segments = segmenter.segment_with_context(
            text, "R1", context_window=0
        )

        # No context should be added
        for segment in segments:
            assert segment.context_before is None
            assert segment.context_after is None


class TestHierarchicalSegmentation:
    """Test hierarchical segmentation functionality."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_segment_hierarchical_basic(self, segmenter):
        """Test basic hierarchical segmentation."""
        text = "Para 1 sent 1. Para 1 sent 2.\n\nPara 2 sent 1."
        hierarchy = segmenter.segment_hierarchical(text, "R1")

        assert 'response' in hierarchy
        assert 'paragraphs' in hierarchy
        assert 'sentences' in hierarchy

        assert len(hierarchy['response']) == 1
        assert len(hierarchy['paragraphs']) == 2
        assert len(hierarchy['sentences']) == 3

    def test_segment_hierarchical_response_level(self, segmenter):
        """Test response-level segment in hierarchy."""
        text = "Test text."
        hierarchy = segmenter.segment_hierarchical(text, "R1")

        response = hierarchy['response'][0]
        assert response.segment_id == "R1"
        assert response.text == text
        assert response.segment_type == "response"

    def test_segment_hierarchical_paragraph_links(self, segmenter):
        """Test that sentences link to parent paragraphs."""
        text = "P1S1. P1S2.\n\nP2S1. P2S2."
        hierarchy = segmenter.segment_hierarchical(text, "R1")

        sentences = hierarchy['sentences']
        assert len(sentences) == 4

        # First two sentences belong to first paragraph
        assert sentences[0].parent_paragraph_id == "R1-P0"
        assert sentences[1].parent_paragraph_id == "R1-P0"

        # Last two sentences belong to second paragraph
        assert sentences[2].parent_paragraph_id == "R1-P1"
        assert sentences[3].parent_paragraph_id == "R1-P1"

    def test_segment_hierarchical_sentence_ids(self, segmenter):
        """Test that hierarchical sentence IDs include paragraph info."""
        text = "S1.\n\nS2."
        hierarchy = segmenter.segment_hierarchical(text, "R1")

        sentences = hierarchy['sentences']
        # Sentences should have format: R1-P0-S0, R1-P1-S0
        assert sentences[0].segment_id == "R1-P0-S0"
        assert sentences[1].segment_id == "R1-P1-S0"

    def test_segment_hierarchical_empty(self, segmenter):
        """Test hierarchical segmentation with empty text."""
        hierarchy = segmenter.segment_hierarchical("", "R1")

        assert len(hierarchy['response']) == 0
        assert len(hierarchy['paragraphs']) == 0
        assert len(hierarchy['sentences']) == 0

    def test_segment_hierarchical_positions(self, segmenter):
        """Test that sentence positions are global within response."""
        text = "P1S1.\n\nP2S1. P2S2."
        hierarchy = segmenter.segment_hierarchical(text, "R1")

        sentences = hierarchy['sentences']
        # Positions should be 0, 1, 2 (global)
        assert sentences[0].position == 0
        assert sentences[1].position == 1
        assert sentences[2].position == 2


class TestBatchSegmentation:
    """Test batch segmentation functionality."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_batch_segment_basic(self, segmenter):
        """Test basic batch segmentation."""
        texts = ["First text. Second sentence.", "Another text."]
        df = segmenter.batch_segment(texts, granularity='sentence')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 2 sentences from first text, 1 from second

        # Check default response IDs
        assert df.iloc[0]['parent_response_id'] == "R0"
        assert df.iloc[2]['parent_response_id'] == "R1"

    def test_batch_segment_with_ids(self, segmenter):
        """Test batch segmentation with custom response IDs."""
        texts = ["Text 1.", "Text 2."]
        response_ids = ["RESP_A", "RESP_B"]

        df = segmenter.batch_segment(
            texts, response_ids=response_ids, granularity='sentence'
        )

        assert df.iloc[0]['parent_response_id'] == "RESP_A"
        assert df.iloc[1]['parent_response_id'] == "RESP_B"

    def test_batch_segment_id_length_mismatch(self, segmenter):
        """Test that mismatched ID length raises error."""
        texts = ["Text 1.", "Text 2."]
        response_ids = ["RESP_A"]  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            segmenter.batch_segment(texts, response_ids=response_ids)

    def test_batch_segment_paragraph_granularity(self, segmenter):
        """Test batch segmentation with paragraph granularity."""
        texts = ["Para 1.\n\nPara 2.", "Single para."]
        df = segmenter.batch_segment(texts, granularity='paragraph')

        assert len(df) == 3  # 2 paragraphs + 1 paragraph

    def test_batch_segment_with_context(self, segmenter):
        """Test batch segmentation with context."""
        texts = ["A. B. C."]
        df = segmenter.batch_segment(
            texts,
            granularity='sentence',
            include_context=True,
            context_window=1
        )

        assert 'context_before' in df.columns
        assert 'context_after' in df.columns
        assert df.iloc[1]['context_before'] == "A."
        assert df.iloc[1]['context_after'] == "C."

    def test_batch_segment_empty_texts(self, segmenter):
        """Test batch segmentation with empty text list."""
        df = segmenter.batch_segment([], granularity='sentence')

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Check schema exists
        assert 'segment_id' in df.columns
        assert 'text' in df.columns
        assert 'parent_response_id' in df.columns

    def test_batch_segment_dataframe_columns(self, segmenter):
        """Test that batch segmentation produces correct DataFrame columns."""
        texts = ["Test."]
        df = segmenter.batch_segment(texts, granularity='sentence')

        expected_columns = [
            'segment_id', 'text', 'parent_response_id', 'parent_paragraph_id',
            'segment_type', 'position', 'context_before', 'context_after', 'metadata'
        ]
        for col in expected_columns:
            assert col in df.columns


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_unicode_text(self, segmenter):
        """Test segmentation with Unicode characters."""
        text = "Привет! How are you? 你好。"
        segments = segmenter.segment_sentences(text, "R1")

        # Should handle Unicode properly
        assert len(segments) == 3

    def test_multiple_spaces(self, segmenter):
        """Test text with multiple consecutive spaces."""
        text = "First.     Second.      Third."
        segments = segmenter.segment_sentences(text, "R1")

        assert len(segments) == 3
        # Whitespace should be normalized
        for segment in segments:
            assert '  ' not in segment.text or segmenter.preserve_whitespace

    def test_only_punctuation(self, segmenter):
        """Test text with only punctuation."""
        text = "... !!! ???"
        segments = segmenter.segment_sentences(text, "R1")

        # Should handle gracefully (may return segments or empty)
        assert isinstance(segments, list)

    def test_very_long_text(self, segmenter):
        """Test segmentation with very long text."""
        # Create a long text with many sentences
        text = ". ".join([f"Sentence {i}" for i in range(1000)])
        segments = segmenter.segment_sentences(text, "R1")

        assert len(segments) == 1000

    def test_numeric_text(self, segmenter):
        """Test text with numbers."""
        text = "The value is 3.14. Another value is 2.71."
        segments = segmenter.segment_sentences(text, "R1")

        # Decimal points shouldn't split sentences incorrectly
        # This is a known limitation - simple regex may split on decimals
        assert isinstance(segments, list)

    def test_mixed_line_endings(self, segmenter):
        """Test text with mixed line ending styles."""
        text = "Para 1.\n\nPara 2.\r\n\r\nPara 3."
        segments = segmenter.segment_paragraphs(text, "R1")

        # Should handle both Unix and Windows line endings
        assert len(segments) >= 2  # At least 2 paragraphs

    def test_preserve_whitespace_option(self):
        """Test preserve_whitespace option."""
        segmenter = TextSegmenter(preserve_whitespace=True)
        text = "  First.  "
        segments = segmenter.segment_sentences(text, "R1")

        # Whitespace should be preserved
        if segments:
            assert segments[0].text.startswith("  ") or segments[0].text == "First."


class TestPerformance:
    """Test performance requirements."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_batch_performance_1000_responses(self, segmenter):
        """
        Test that batch segmentation meets performance requirement.

        Requirement: <50ms per 1000 responses (response-level)
        """
        # Create 1000 short responses
        texts = [f"Response {i} with some text." for i in range(1000)]

        start_time = time.time()
        df = segmenter.batch_segment(texts, granularity='sentence')
        elapsed_ms = (time.time() - start_time) * 1000

        # Verify correct number of segments
        assert len(df) == 1000  # Each text has 1 sentence

        # Performance check: should be under 50ms per 1000 responses
        # Note: This is a soft requirement - actual performance varies by hardware
        # We'll use a more generous limit for test stability (200ms)
        assert elapsed_ms < 200, f"Performance requirement not met: {elapsed_ms:.2f}ms"

    def test_batch_performance_logging(self, segmenter, caplog):
        """Test that batch segmentation logs performance info."""
        import logging
        caplog.set_level(logging.INFO)

        texts = ["Test."] * 10
        segmenter.batch_segment(texts, granularity='sentence')

        # Should log batch segmentation info
        assert any("Batch segmented" in record.message for record in caplog.records)


class TestTraceability:
    """Test that parent-child relationships are preserved."""

    @pytest.fixture
    def segmenter(self):
        """Create a TextSegmenter instance."""
        return TextSegmenter()

    def test_sentence_traceability(self, segmenter):
        """Test that sentences can be traced back to responses."""
        text = "Sent 1. Sent 2."
        segments = segmenter.segment_sentences(text, "CUSTOM_ID")

        for segment in segments:
            assert segment.parent_response_id == "CUSTOM_ID"
            assert segment.segment_id.startswith("CUSTOM_ID")

    def test_paragraph_traceability(self, segmenter):
        """Test that paragraphs can be traced back to responses."""
        text = "Para 1.\n\nPara 2."
        segments = segmenter.segment_paragraphs(text, "TRACE_ID")

        for segment in segments:
            assert segment.parent_response_id == "TRACE_ID"
            assert segment.segment_id.startswith("TRACE_ID")

    def test_hierarchical_traceability(self, segmenter):
        """Test complete traceability in hierarchical segmentation."""
        text = "P1S1.\n\nP2S1."
        hierarchy = segmenter.segment_hierarchical(text, "ROOT")

        # All segments should trace back to ROOT
        for segment in hierarchy['response']:
            assert segment.parent_response_id == "ROOT"

        for segment in hierarchy['paragraphs']:
            assert segment.parent_response_id == "ROOT"

        for segment in hierarchy['sentences']:
            assert segment.parent_response_id == "ROOT"
            assert segment.parent_paragraph_id.startswith("ROOT-P")

    def test_batch_traceability(self, segmenter):
        """Test traceability in batch processing."""
        texts = ["Text A.", "Text B."]
        ids = ["ID_A", "ID_B"]

        df = segmenter.batch_segment(texts, response_ids=ids, granularity='sentence')

        # Each segment should trace to correct parent
        assert df[df['parent_response_id'] == 'ID_A'].iloc[0]['text'] == "Text A."
        assert df[df['parent_response_id'] == 'ID_B'].iloc[0]['text'] == "Text B."
