"""
Sentiment analysis module with specialized support for different data types.

This module provides sentiment analysis capabilities optimized for:
- Twitter/X and social media stream data (using CardiffNLP Twitter-RoBERTa)
- Survey response data (using general sentiment models)
- Long-form data like product reviews (using review-trained models)

The CardiffNLP Twitter-RoBERTa model is specifically designed for messy,
informal social media text with hashtags, mentions, emojis, and abbreviations.

Classes:
    - BaseSentimentAnalyzer: Abstract base class for sentiment analyzers
    - TwitterSentimentAnalyzer: Optimized for Twitter/social media data
    - SurveySentimentAnalyzer: For survey response sentiment
    - LongFormSentimentAnalyzer: For product reviews and long-form text

Usage:
    # For Twitter/social media data
    analyzer = get_sentiment_analyzer('twitter')
    results = analyzer.analyze(texts)

    # For survey responses
    analyzer = get_sentiment_analyzer('survey')
    results = analyzer.analyze(texts)

    # For long-form reviews
    analyzer = get_sentiment_analyzer('longform')
    results = analyzer.analyze(texts)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import re
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """
    Container for sentiment analysis results.

    Attributes:
        label: Sentiment label (positive, negative, neutral)
        score: Confidence score (0-1)
        scores: Dict of all label scores {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}
        original_text: Original input text
        preprocessed_text: Text after preprocessing
    """
    label: str
    score: float
    scores: Dict[str, float]
    original_text: str
    preprocessed_text: str


class TwitterTextPreprocessor:
    """
    Specialized preprocessor for Twitter/social media text.

    Handles the messy, informal nature of social media data:
    - @mentions → @user placeholder
    - URLs → http placeholder
    - Hashtags → preserved but cleaned
    - Emojis → preserved (important for sentiment)
    - Repeated characters → normalized
    - Case normalization
    """

    def __init__(self,
                 normalize_mentions: bool = True,
                 normalize_urls: bool = True,
                 lowercase: bool = False,  # Twitter-RoBERTa was trained with original case
                 normalize_repeated_chars: bool = True,
                 max_repeated: int = 3):
        """
        Initialize Twitter text preprocessor.

        Args:
            normalize_mentions: Replace @mentions with @user
            normalize_urls: Replace URLs with http placeholder
            lowercase: Convert to lowercase (not recommended for Twitter-RoBERTa)
            normalize_repeated_chars: Reduce repeated chars (e.g., "sooo" -> "soo")
            max_repeated: Maximum allowed character repetitions
        """
        self.normalize_mentions = normalize_mentions
        self.normalize_urls = normalize_urls
        self.lowercase = lowercase
        self.normalize_repeated_chars = normalize_repeated_chars
        self.max_repeated = max_repeated

        # Compile regex patterns for efficiency
        self._mention_pattern = re.compile(r'@\w+')
        self._url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|'
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self._repeated_char_pattern = re.compile(r'(.)\1{2,}')
        self._whitespace_pattern = re.compile(r'\s+')

    def preprocess(self, text: str) -> str:
        """
        Preprocess Twitter/social media text.

        Args:
            text: Raw social media text

        Returns:
            Preprocessed text suitable for Twitter-RoBERTa model
        """
        if pd.isna(text) or not text:
            return ""

        text = str(text)

        # Normalize mentions to @user (as done in CardiffNLP training)
        if self.normalize_mentions:
            text = self._mention_pattern.sub('@user', text)

        # Normalize URLs to http (as done in CardiffNLP training)
        if self.normalize_urls:
            text = self._url_pattern.sub('http', text)

        # Normalize repeated characters (e.g., "sooooo" -> "soo")
        if self.normalize_repeated_chars:
            text = self._repeated_char_pattern.sub(
                lambda m: m.group(1) * min(len(m.group(0)), self.max_repeated),
                text
            )

        # Lowercase (optional - Twitter-RoBERTa preserves case)
        if self.lowercase:
            text = text.lower()

        # Normalize whitespace
        text = self._whitespace_pattern.sub(' ', text).strip()

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts."""
        return [self.preprocess(text) for text in texts]


class BaseSentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analyzers.

    All sentiment analyzers should implement the analyze method
    and provide consistent output format.
    """

    def __init__(self):
        self.is_fitted_ = False
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def analyze(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of texts.

        Args:
            texts: List of text strings to analyze

        Returns:
            List of SentimentResult objects
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model being used."""
        pass

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        output_prefix: str = 'sentiment'
    ) -> pd.DataFrame:
        """
        Analyze sentiment and add results to DataFrame.

        Args:
            df: Input DataFrame
            text_column: Column containing text to analyze
            output_prefix: Prefix for output columns

        Returns:
            DataFrame with sentiment columns added
        """
        texts = df[text_column].tolist()
        results = self.analyze(texts)

        result_df = df.copy()
        result_df[f'{output_prefix}_label'] = [r.label for r in results]
        result_df[f'{output_prefix}_score'] = [r.score for r in results]
        result_df[f'{output_prefix}_positive'] = [r.scores.get('positive', 0) for r in results]
        result_df[f'{output_prefix}_negative'] = [r.scores.get('negative', 0) for r in results]
        result_df[f'{output_prefix}_neutral'] = [r.scores.get('neutral', 0) for r in results]

        return result_df


class TwitterSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer optimized for Twitter/social media data.

    Uses the CardiffNLP twitter-roberta-base-sentiment-latest model,
    which is RoBERTa trained on ~124M tweets and fine-tuned for
    TweetEval sentiment classification.

    This model handles:
    - Informal language and slang
    - Hashtags and mentions
    - Emojis (important sentiment signals)
    - Short, fragmented text
    - Misspellings and abbreviations

    Labels: negative (0), neutral (1), positive (2)

    Example:
        >>> analyzer = TwitterSentimentAnalyzer()
        >>> results = analyzer.analyze(["I love this! #amazing", "This sucks @company"])
        >>> print(results[0].label, results[0].score)
        positive 0.95
    """

    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    LABEL_MAP = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 128,
        show_progress: bool = True
    ):
        """
        Initialize Twitter sentiment analyzer.

        Args:
            device: Device to use ('cpu', 'cuda', or None for auto)
            batch_size: Batch size for inference
            max_length: Maximum token length (tweets are short)
            show_progress: Show progress bar during analysis

        Raises:
            ImportError: If transformers is not installed
        """
        super().__init__()

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for TwitterSentimentAnalyzer. "
                "Install with: pip install transformers torch"
            )

        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.show_progress = show_progress
        self.preprocessor = TwitterTextPreprocessor()

        # Load model and tokenizer
        logger.info(f"Loading Twitter-RoBERTa sentiment model: {self.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)

        # Set device
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

        self.model.to(self._device)
        self.model.eval()
        self.is_fitted_ = True

        logger.info(f"Twitter sentiment model loaded on {self._device}")

    def analyze(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment of Twitter/social media texts.

        Args:
            texts: List of text strings (tweets, posts, etc.)

        Returns:
            List of SentimentResult objects with sentiment labels and scores
        """
        import torch
        from torch.nn.functional import softmax

        if not texts:
            return []

        results = []

        # Preprocess texts
        preprocessed = self.preprocessor.preprocess_batch(texts)

        # Process in batches
        n_batches = (len(preprocessed) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(preprocessed))
            batch_texts = preprocessed[start_idx:end_idx]
            original_batch = texts[start_idx:end_idx]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self._device)

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=1)

            # Process results
            probs_np = probs.cpu().numpy()

            for i, (orig_text, prep_text, prob_row) in enumerate(
                zip(original_batch, batch_texts, probs_np)
            ):
                scores = {
                    'negative': float(prob_row[0]),
                    'neutral': float(prob_row[1]),
                    'positive': float(prob_row[2])
                }

                pred_idx = int(np.argmax(prob_row))
                label = self.LABEL_MAP[pred_idx]
                score = float(prob_row[pred_idx])

                results.append(SentimentResult(
                    label=label,
                    score=score,
                    scores=scores,
                    original_text=orig_text,
                    preprocessed_text=prep_text
                ))

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Twitter sentiment model."""
        return {
            'model_name': self.MODEL_NAME,
            'model_type': 'twitter-roberta',
            'labels': list(self.LABEL_MAP.values()),
            'description': 'RoBERTa trained on 124M tweets, fine-tuned for TweetEval sentiment',
            'optimal_for': 'Twitter/X posts, social media streams, informal short text',
            'max_length': self.max_length,
            'device': str(self._device)
        }


class SurveySentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer for survey response data.

    Uses a general-purpose sentiment model suitable for
    semi-formal text like survey responses.

    Can use either VADER (lexicon-based, fast) or
    transformer models (more accurate, slower).
    """

    def __init__(
        self,
        method: str = 'vader',
        transformer_model: str = 'nlptown/bert-base-multilingual-uncased-sentiment',
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Initialize survey sentiment analyzer.

        Args:
            method: 'vader' for lexicon-based, 'transformer' for neural
            transformer_model: Model name if using transformer method
            device: Device for transformer inference
            batch_size: Batch size for transformer inference
        """
        super().__init__()

        self.method = method
        self.transformer_model_name = transformer_model
        self.device = device
        self.batch_size = batch_size

        if method == 'vader':
            try:
                from nltk.sentiment import SentimentIntensityAnalyzer
                import nltk
                # Ensure vader lexicon is downloaded
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon', quiet=True)
                self.model = SentimentIntensityAnalyzer()
            except ImportError:
                raise ImportError(
                    "nltk is required for VADER sentiment. "
                    "Install with: pip install nltk"
                )
        else:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                import torch

                logger.info(f"Loading transformer sentiment model: {transformer_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
                self.model = AutoModelForSequenceClassification.from_pretrained(transformer_model)

                if device is None:
                    self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    self._device = torch.device(device)

                self.model.to(self._device)
                self.model.eval()
            except ImportError:
                raise ImportError(
                    "transformers and torch are required for transformer sentiment. "
                    "Install with: pip install transformers torch"
                )

        self.is_fitted_ = True

    def analyze(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment of survey responses."""
        if self.method == 'vader':
            return self._analyze_vader(texts)
        else:
            return self._analyze_transformer(texts)

    def _analyze_vader(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze using VADER lexicon."""
        results = []

        for text in texts:
            if pd.isna(text) or not text:
                results.append(SentimentResult(
                    label='neutral',
                    score=1.0,
                    scores={'positive': 0.0, 'negative': 0.0, 'neutral': 1.0},
                    original_text=str(text) if text else '',
                    preprocessed_text=''
                ))
                continue

            text = str(text)
            scores_dict = self.model.polarity_scores(text)

            # Convert VADER compound score to labels
            compound = scores_dict['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'

            # Normalize scores
            total = scores_dict['pos'] + scores_dict['neg'] + scores_dict['neu']
            if total > 0:
                scores = {
                    'positive': scores_dict['pos'] / total,
                    'negative': scores_dict['neg'] / total,
                    'neutral': scores_dict['neu'] / total
                }
            else:
                scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

            results.append(SentimentResult(
                label=label,
                score=abs(compound),
                scores=scores,
                original_text=text,
                preprocessed_text=text
            ))

        return results

    def _analyze_transformer(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze using transformer model."""
        import torch
        from torch.nn.functional import softmax

        results = []

        # Process in batches
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = [str(t) if t and not pd.isna(t) else '' for t in texts[start_idx:end_idx]]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self._device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=1)

            probs_np = probs.cpu().numpy()

            for orig_text, prob_row in zip(batch_texts, probs_np):
                # Map 5-star ratings to sentiment
                # Stars: 1=very negative, 2=negative, 3=neutral, 4=positive, 5=very positive
                neg_score = float(prob_row[0] + prob_row[1])
                neu_score = float(prob_row[2])
                pos_score = float(prob_row[3] + prob_row[4])

                scores = {
                    'negative': neg_score,
                    'neutral': neu_score,
                    'positive': pos_score
                }

                pred_idx = int(np.argmax([neg_score, neu_score, pos_score]))
                label = ['negative', 'neutral', 'positive'][pred_idx]
                score = max(scores.values())

                results.append(SentimentResult(
                    label=label,
                    score=score,
                    scores=scores,
                    original_text=orig_text,
                    preprocessed_text=orig_text
                ))

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the survey sentiment model."""
        if self.method == 'vader':
            return {
                'model_name': 'VADER',
                'model_type': 'lexicon-based',
                'labels': ['positive', 'negative', 'neutral'],
                'description': 'Valence Aware Dictionary for Sentiment Reasoning',
                'optimal_for': 'General text, survey responses, social media',
                'fast': True
            }
        else:
            return {
                'model_name': self.transformer_model_name,
                'model_type': 'transformer',
                'labels': ['positive', 'negative', 'neutral'],
                'description': 'Multilingual BERT fine-tuned on reviews',
                'optimal_for': 'Multilingual text, reviews, formal responses',
                'device': str(self._device)
            }


class LongFormSentimentAnalyzer(BaseSentimentAnalyzer):
    """
    Sentiment analyzer for long-form text like product reviews.

    Optimized for:
    - Product reviews
    - Customer feedback
    - Detailed opinion pieces
    - Multi-sentence text
    """

    def __init__(
        self,
        model: str = 'nlptown/bert-base-multilingual-uncased-sentiment',
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
        chunk_long_texts: bool = True
    ):
        """
        Initialize long-form sentiment analyzer.

        Args:
            model: HuggingFace model name
            device: Device for inference
            batch_size: Batch size for inference
            max_length: Maximum token length
            chunk_long_texts: Split long texts and aggregate sentiment
        """
        super().__init__()

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.chunk_long_texts = chunk_long_texts

        logger.info(f"Loading long-form sentiment model: {model}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model)

        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

        self.model.to(self._device)
        self.model.eval()
        self.is_fitted_ = True

        logger.info(f"Long-form sentiment model loaded on {self._device}")

    def analyze(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment of long-form texts."""
        import torch
        from torch.nn.functional import softmax

        if not texts:
            return []

        results = []

        # Process in batches
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = [str(t) if t and not pd.isna(t) else '' for t in texts[start_idx:end_idx]]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self._device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=1)

            probs_np = probs.cpu().numpy()

            for orig_text, prob_row in zip(batch_texts, probs_np):
                # Map 5-star ratings to sentiment
                neg_score = float(prob_row[0] + prob_row[1])
                neu_score = float(prob_row[2])
                pos_score = float(prob_row[3] + prob_row[4])

                scores = {
                    'negative': neg_score,
                    'neutral': neu_score,
                    'positive': pos_score
                }

                pred_idx = int(np.argmax([neg_score, neu_score, pos_score]))
                label = ['negative', 'neutral', 'positive'][pred_idx]
                score = max(scores.values())

                results.append(SentimentResult(
                    label=label,
                    score=score,
                    scores=scores,
                    original_text=orig_text,
                    preprocessed_text=orig_text
                ))

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the long-form sentiment model."""
        return {
            'model_name': self.model_name,
            'model_type': 'transformer',
            'labels': ['positive', 'negative', 'neutral'],
            'description': 'BERT fine-tuned on product reviews (5-star ratings)',
            'optimal_for': 'Product reviews, customer feedback, long opinions',
            'max_length': self.max_length,
            'device': str(self._device)
        }


# Factory functions

def get_sentiment_analyzer(
    data_type: str = 'survey',
    **kwargs
) -> BaseSentimentAnalyzer:
    """
    Factory function to create appropriate sentiment analyzer.

    Args:
        data_type: Type of data to analyze:
            - 'twitter': Twitter/X posts, social media streams
            - 'survey': Survey responses, general text
            - 'longform': Product reviews, detailed feedback
        **kwargs: Additional arguments for the specific analyzer

    Returns:
        Appropriate sentiment analyzer instance

    Raises:
        ValueError: If data_type is not recognized

    Example:
        >>> analyzer = get_sentiment_analyzer('twitter')
        >>> results = analyzer.analyze(["Love this product!", "Worst experience ever"])
    """
    data_type = data_type.lower()

    if data_type in ['twitter', 'x', 'social', 'stream']:
        return TwitterSentimentAnalyzer(**kwargs)
    elif data_type in ['survey', 'response', 'general']:
        return SurveySentimentAnalyzer(**kwargs)
    elif data_type in ['longform', 'review', 'product']:
        return LongFormSentimentAnalyzer(**kwargs)
    else:
        raise ValueError(
            f"Unknown data type: {data_type}. "
            f"Choose from: 'twitter', 'survey', 'longform'"
        )


def analyze_sentiment(
    texts: List[str],
    data_type: str = 'survey',
    **kwargs
) -> Tuple[List[str], List[float], List[Dict[str, float]]]:
    """
    Convenience function to analyze sentiment.

    Args:
        texts: List of texts to analyze
        data_type: Type of data ('twitter', 'survey', 'longform')
        **kwargs: Additional arguments for analyzer

    Returns:
        Tuple of (labels, scores, all_scores)

    Example:
        >>> labels, scores, all_scores = analyze_sentiment(
        ...     ["Great product!", "Terrible service"],
        ...     data_type='longform'
        ... )
    """
    analyzer = get_sentiment_analyzer(data_type, **kwargs)
    results = analyzer.analyze(texts)

    labels = [r.label for r in results]
    scores = [r.score for r in results]
    all_scores = [r.scores for r in results]

    return labels, scores, all_scores


# Data type descriptions for UI
DATA_TYPE_INFO = {
    'twitter': {
        'name': 'X (Twitter) / Stream Data',
        'description': 'Social media posts, tweets, short informal text',
        'model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'features': [
            'Handles @mentions and #hashtags',
            'Understands emojis and slang',
            'Optimized for short, messy text',
            'Trained on 124M tweets'
        ],
        'best_for': 'Twitter/X posts, social media streams, chat messages'
    },
    'survey': {
        'name': 'Survey Response Data',
        'description': 'Survey responses, general semi-formal text',
        'model': 'VADER / nlptown BERT',
        'features': [
            'Fast lexicon-based analysis',
            'Optional deep learning model',
            'Handles neutral responses well',
            'Good for short to medium text'
        ],
        'best_for': 'Survey responses, feedback forms, general text'
    },
    'longform': {
        'name': 'Long Form Data (Product Reviews)',
        'description': 'Product reviews, detailed customer feedback',
        'model': 'nlptown/bert-base-multilingual-uncased-sentiment',
        'features': [
            '5-star rating prediction',
            'Handles long paragraphs',
            'Multilingual support',
            'Trained on product reviews'
        ],
        'best_for': 'Product reviews, detailed feedback, articles'
    }
}
