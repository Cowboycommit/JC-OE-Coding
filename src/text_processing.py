"""
Text segmentation and processing for multi-granularity analysis.

This module provides text segmentation capabilities to support analysis at different
granularities (response, paragraph, sentence levels) while maintaining traceability
to parent responses.

Key features:
- Sentence-level segmentation with configurable sentence boundary detection
- Paragraph-level segmentation with configurable paragraph delimiters
- Contextual segmentation that tracks parent-child relationships
- Preservation of response boundaries and metadata
- Performance optimized for large datasets (<50ms per 1000 responses)

Integration point: Used by helpers/analysis.py:MLOpenCoder for optional
multi-granularity text processing.

Note: Response-level analysis remains the default. Segmentation is opt-in only.
"""

import re
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """
    Represents a text segment with metadata and traceability.

    Attributes:
        segment_id: Unique identifier for this segment (e.g., "R1-P2-S3")
        text: The actual text content of the segment
        parent_response_id: ID of the parent response (for traceability)
        parent_paragraph_id: ID of parent paragraph (if applicable)
        segment_type: Type of segment ('response', 'paragraph', 'sentence')
        position: Position within parent (0-indexed)
        context_before: Text of preceding segment (if available)
        context_after: Text of following segment (if available)
        metadata: Additional metadata dictionary
    """
    segment_id: str
    text: str
    parent_response_id: str
    parent_paragraph_id: Optional[str] = None
    segment_type: str = 'response'
    position: int = 0
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for DataFrame conversion."""
        return {
            'segment_id': self.segment_id,
            'text': self.text,
            'parent_response_id': self.parent_response_id,
            'parent_paragraph_id': self.parent_paragraph_id,
            'segment_type': self.segment_type,
            'position': self.position,
            'context_before': self.context_before,
            'context_after': self.context_after,
            'metadata': self.metadata
        }


class TextSegmenter:
    """
    Multi-granularity text segmentation with traceability.

    This class provides methods to segment text at different granularities
    (sentences, paragraphs) while maintaining parent-child relationships
    and preserving response boundaries for qualitative analysis.

    Design principles:
    - Opt-in segmentation: Users must explicitly request segmentation
    - Backward compatible: Response-level remains default
    - Traceable: All segments linked to parent response IDs
    - Transparent: Segmentation rules are explicit and documented
    - Performant: Optimized for large-scale datasets

    Example usage:
        >>> segmenter = TextSegmenter()
        >>> segments = segmenter.segment_sentences("Hello world. How are you?")
        >>> print(len(segments))  # 2
        >>> print(segments[0].text)  # "Hello world."
    """

    def __init__(self,
                 sentence_endings: Optional[List[str]] = None,
                 paragraph_delimiters: Optional[List[str]] = None,
                 preserve_whitespace: bool = False,
                 min_segment_length: int = 1):
        """
        Initialize the TextSegmenter.

        Args:
            sentence_endings: List of sentence-ending punctuation marks.
                Default: ['.', '!', '?']
            paragraph_delimiters: List of paragraph delimiters.
                Default: ['\\n\\n', '\\r\\n\\r\\n']
            preserve_whitespace: Whether to preserve leading/trailing whitespace
                in segments. Default: False (strips whitespace)
            min_segment_length: Minimum character length for valid segments.
                Segments shorter than this are filtered out. Default: 1
        """
        self.sentence_endings = sentence_endings or ['.', '!', '?']
        self.paragraph_delimiters = paragraph_delimiters or ['\n\n', '\r\n\r\n']
        self.preserve_whitespace = preserve_whitespace
        self.min_segment_length = min_segment_length

        # Compile regex pattern for sentence segmentation
        # Handles common cases: "Dr. Smith", "U.S.A.", etc.
        endings_pattern = '|'.join(re.escape(e) for e in self.sentence_endings)
        self.sentence_pattern = re.compile(
            rf'([^{endings_pattern}]+[{endings_pattern}]+(?:\s+|$)|[^{endings_pattern}]+$)',
            re.MULTILINE
        )

    def segment_sentences(self,
                         text: str,
                         response_id: Optional[str] = None) -> List[TextSegment]:
        """
        Split text into sentence-level segments.

        Uses configurable sentence boundary detection to identify individual
        sentences. Handles common edge cases like abbreviations and ellipses.

        Args:
            text: Input text to segment
            response_id: Optional parent response ID for traceability.
                If not provided, generates a default ID.

        Returns:
            List of TextSegment objects, one per sentence.
            Empty list if text is empty or None.

        Example:
            >>> segmenter = TextSegmenter()
            >>> text = "I love remote work. It's very flexible."
            >>> segments = segmenter.segment_sentences(text, response_id="R1")
            >>> len(segments)
            2
            >>> segments[0].segment_id
            'R1-S0'
            >>> segments[0].text
            "I love remote work."
        """
        if not text or pd.isna(text):
            return []

        text = str(text)
        response_id = response_id or "R0"

        # Find sentence boundaries using regex
        raw_sentences = self.sentence_pattern.findall(text)

        # Clean and filter sentences
        sentences = []
        for sent in raw_sentences:
            cleaned = sent.strip() if not self.preserve_whitespace else sent
            if len(cleaned) >= self.min_segment_length:
                sentences.append(cleaned)

        # Create TextSegment objects
        segments = []
        for idx, sentence in enumerate(sentences):
            segment = TextSegment(
                segment_id=f"{response_id}-S{idx}",
                text=sentence,
                parent_response_id=response_id,
                segment_type='sentence',
                position=idx,
                metadata={'original_text_length': len(text)}
            )
            segments.append(segment)

        return segments

    def segment_paragraphs(self,
                          text: str,
                          response_id: Optional[str] = None) -> List[TextSegment]:
        """
        Split text into paragraph-level segments.

        Uses configurable paragraph delimiters (default: double newlines)
        to identify paragraph boundaries. Useful for long-form responses.

        Args:
            text: Input text to segment
            response_id: Optional parent response ID for traceability.
                If not provided, generates a default ID.

        Returns:
            List of TextSegment objects, one per paragraph.
            Empty list if text is empty or None.

        Example:
            >>> segmenter = TextSegmenter()
            >>> text = "First paragraph.\\n\\nSecond paragraph."
            >>> segments = segmenter.segment_paragraphs(text, response_id="R1")
            >>> len(segments)
            2
            >>> segments[0].segment_id
            'R1-P0'
        """
        if not text or pd.isna(text):
            return []

        text = str(text)
        response_id = response_id or "R0"

        # Split by paragraph delimiters
        # Create regex pattern that matches any of the delimiters
        delimiter_pattern = '|'.join(re.escape(d) for d in self.paragraph_delimiters)
        raw_paragraphs = re.split(delimiter_pattern, text)

        # Clean and filter paragraphs
        paragraphs = []
        for para in raw_paragraphs:
            cleaned = para.strip() if not self.preserve_whitespace else para
            if len(cleaned) >= self.min_segment_length:
                paragraphs.append(cleaned)

        # Create TextSegment objects
        segments = []
        for idx, paragraph in enumerate(paragraphs):
            segment = TextSegment(
                segment_id=f"{response_id}-P{idx}",
                text=paragraph,
                parent_response_id=response_id,
                segment_type='paragraph',
                position=idx,
                metadata={'original_text_length': len(text)}
            )
            segments.append(segment)

        return segments

    def segment_with_context(self,
                            text: str,
                            response_id: Optional[str] = None,
                            granularity: str = 'sentence',
                            context_window: int = 1) -> List[TextSegment]:
        """
        Segment text with contextual information (preceding/following segments).

        This method segments text at the specified granularity and adds context
        from surrounding segments. Useful for analyses that require understanding
        of how segments relate to their surrounding text.

        Args:
            text: Input text to segment
            response_id: Optional parent response ID for traceability
            granularity: Segmentation level ('sentence' or 'paragraph')
            context_window: Number of segments to include as context before/after.
                Default: 1 (immediate neighbors only)

        Returns:
            List of TextSegment objects with context_before and context_after
            populated from neighboring segments.

        Example:
            >>> segmenter = TextSegmenter()
            >>> text = "First. Second. Third."
            >>> segments = segmenter.segment_with_context(text, "R1", context_window=1)
            >>> segments[1].context_before
            "First."
            >>> segments[1].context_after
            "Third."

        Raises:
            ValueError: If granularity is not 'sentence' or 'paragraph'
        """
        if granularity not in ['sentence', 'paragraph']:
            raise ValueError(
                f"Invalid granularity: '{granularity}'. "
                "Must be 'sentence' or 'paragraph'."
            )

        # Get base segments
        if granularity == 'sentence':
            segments = self.segment_sentences(text, response_id)
        else:
            segments = self.segment_paragraphs(text, response_id)

        # Add context information
        for idx, segment in enumerate(segments):
            # Context before
            if idx > 0 and context_window > 0:
                start_idx = max(0, idx - context_window)
                context_segments = [s.text for s in segments[start_idx:idx]]
                segment.context_before = ' '.join(context_segments)

            # Context after
            if idx < len(segments) - 1 and context_window > 0:
                end_idx = min(len(segments), idx + context_window + 1)
                context_segments = [s.text for s in segments[idx+1:end_idx]]
                segment.context_after = ' '.join(context_segments)

        return segments

    def segment_hierarchical(self,
                           text: str,
                           response_id: Optional[str] = None) -> Dict[str, List[TextSegment]]:
        """
        Create hierarchical segmentation: Response → Paragraphs → Sentences.

        This method performs multi-level segmentation and preserves parent-child
        relationships across all levels. Useful for understanding how sentences
        relate to paragraphs and paragraphs to the full response.

        Args:
            text: Input text to segment
            response_id: Optional parent response ID for traceability

        Returns:
            Dictionary with keys:
                - 'response': Single TextSegment for full response
                - 'paragraphs': List of paragraph TextSegments
                - 'sentences': List of sentence TextSegments (across all paragraphs)

            Each sentence segment includes parent_paragraph_id to link to its
            containing paragraph.

        Example:
            >>> segmenter = TextSegmenter()
            >>> text = "Para 1 sent 1. Para 1 sent 2.\\n\\nPara 2 sent 1."
            >>> hierarchy = segmenter.segment_hierarchical(text, "R1")
            >>> len(hierarchy['paragraphs'])
            2
            >>> len(hierarchy['sentences'])
            3
            >>> hierarchy['sentences'][2].parent_paragraph_id
            'R1-P1'
        """
        if not text or pd.isna(text):
            return {
                'response': [],
                'paragraphs': [],
                'sentences': []
            }

        text = str(text)
        response_id = response_id or "R0"

        # Response-level segment
        response_segment = TextSegment(
            segment_id=response_id,
            text=text,
            parent_response_id=response_id,
            segment_type='response',
            position=0
        )

        # Paragraph-level segments
        paragraph_segments = self.segment_paragraphs(text, response_id)

        # Sentence-level segments (within each paragraph)
        all_sentence_segments = []
        for para_idx, para_segment in enumerate(paragraph_segments):
            para_id = para_segment.segment_id
            sentence_segments = self.segment_sentences(para_segment.text, response_id)

            # Update sentence segments with paragraph parent info
            for sent_idx, sent_segment in enumerate(sentence_segments):
                # Update segment ID to include paragraph
                sent_segment.segment_id = f"{para_id}-S{sent_idx}"
                sent_segment.parent_paragraph_id = para_id
                # Update position to be global sentence position
                sent_segment.position = len(all_sentence_segments)
                all_sentence_segments.append(sent_segment)

        return {
            'response': [response_segment],
            'paragraphs': paragraph_segments,
            'sentences': all_sentence_segments
        }

    def batch_segment(self,
                     texts: List[str],
                     response_ids: Optional[List[str]] = None,
                     granularity: str = 'sentence',
                     include_context: bool = False,
                     context_window: int = 1) -> pd.DataFrame:
        """
        Perform batch segmentation on multiple texts.

        Optimized for processing large datasets efficiently. Returns a DataFrame
        with all segments and their metadata for easy integration with existing
        analysis pipelines.

        Args:
            texts: List of text strings to segment
            response_ids: Optional list of response IDs (must match length of texts)
                If not provided, generates IDs as "R0", "R1", etc.
            granularity: Segmentation level ('sentence' or 'paragraph')
            include_context: Whether to include context_before/after
            context_window: Size of context window if include_context=True

        Returns:
            DataFrame with columns:
                - segment_id
                - text
                - parent_response_id
                - parent_paragraph_id (if applicable)
                - segment_type
                - position
                - context_before (if include_context=True)
                - context_after (if include_context=True)
                - metadata

        Raises:
            ValueError: If response_ids provided but length doesn't match texts

        Performance:
            Target: <50ms per 1000 responses (response-level)
            Actual performance depends on average text length and granularity.
        """
        if response_ids is not None and len(response_ids) != len(texts):
            raise ValueError(
                f"Length mismatch: {len(texts)} texts but {len(response_ids)} response_ids"
            )

        # Generate IDs if not provided
        if response_ids is None:
            response_ids = [f"R{i}" for i in range(len(texts))]

        # Process each text
        all_segments = []
        for text, resp_id in zip(texts, response_ids):
            if include_context:
                segments = self.segment_with_context(
                    text, resp_id, granularity, context_window
                )
            elif granularity == 'sentence':
                segments = self.segment_sentences(text, resp_id)
            else:
                segments = self.segment_paragraphs(text, resp_id)

            all_segments.extend(segments)

        # Convert to DataFrame
        if not all_segments:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'segment_id', 'text', 'parent_response_id', 'parent_paragraph_id',
                'segment_type', 'position', 'context_before', 'context_after', 'metadata'
            ])

        segment_dicts = [seg.to_dict() for seg in all_segments]
        df = pd.DataFrame(segment_dicts)

        logger.info(
            f"Batch segmented {len(texts)} texts into {len(all_segments)} "
            f"{granularity}-level segments"
        )

        return df
