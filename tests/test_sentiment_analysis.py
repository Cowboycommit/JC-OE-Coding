"""
Tests for sentiment analysis module.

Tests the TwitterTextPreprocessor and sentiment analyzer classes.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd


class TestTwitterTextPreprocessor:
    """Tests for TwitterTextPreprocessor class."""

    def test_preprocessor_import(self):
        """Test that TwitterTextPreprocessor can be imported."""
        from src.sentiment_analysis import TwitterTextPreprocessor
        preprocessor = TwitterTextPreprocessor()
        assert preprocessor is not None

    def test_normalize_mentions(self):
        """Test that @mentions are normalized to @user."""
        from src.sentiment_analysis import TwitterTextPreprocessor
        preprocessor = TwitterTextPreprocessor(normalize_mentions=True)

        text = "Hey @JohnDoe and @JaneSmith, check this out!"
        result = preprocessor.preprocess(text)

        assert "@JohnDoe" not in result
        assert "@JaneSmith" not in result
        assert "@user" in result

    def test_normalize_urls(self):
        """Test that URLs are normalized to http placeholder."""
        from src.sentiment_analysis import TwitterTextPreprocessor
        preprocessor = TwitterTextPreprocessor(normalize_urls=True)

        text = "Check out https://example.com/page and www.test.com"
        result = preprocessor.preprocess(text)

        assert "https://example.com/page" not in result
        assert "www.test.com" not in result
        assert "http" in result

    def test_normalize_repeated_chars(self):
        """Test that repeated characters are normalized."""
        from src.sentiment_analysis import TwitterTextPreprocessor
        preprocessor = TwitterTextPreprocessor(normalize_repeated_chars=True, max_repeated=3)

        text = "This is sooooooo goooood!"
        result = preprocessor.preprocess(text)

        assert "sooooooo" not in result
        assert "goooood" not in result

    def test_empty_text(self):
        """Test that empty text returns empty string."""
        from src.sentiment_analysis import TwitterTextPreprocessor
        preprocessor = TwitterTextPreprocessor()

        assert preprocessor.preprocess("") == ""
        assert preprocessor.preprocess(None) == ""

    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        from src.sentiment_analysis import TwitterTextPreprocessor
        preprocessor = TwitterTextPreprocessor()

        texts = ["Hello @user!", "Check https://example.com", "Normal text"]
        results = preprocessor.preprocess_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_sentiment_result_creation(self):
        """Test creating a SentimentResult."""
        from src.sentiment_analysis import SentimentResult

        result = SentimentResult(
            label='positive',
            score=0.95,
            scores={'positive': 0.95, 'negative': 0.02, 'neutral': 0.03},
            original_text='I love this!',
            preprocessed_text='I love this!'
        )

        assert result.label == 'positive'
        assert result.score == 0.95
        assert result.scores['positive'] == 0.95


class TestDataTypeInfo:
    """Tests for DATA_TYPE_INFO constant."""

    def test_data_type_info_exists(self):
        """Test that DATA_TYPE_INFO is defined."""
        from src.sentiment_analysis import DATA_TYPE_INFO

        assert DATA_TYPE_INFO is not None
        assert isinstance(DATA_TYPE_INFO, dict)

    def test_twitter_data_type(self):
        """Test Twitter data type info."""
        from src.sentiment_analysis import DATA_TYPE_INFO

        assert 'twitter' in DATA_TYPE_INFO
        assert 'name' in DATA_TYPE_INFO['twitter']
        assert 'model' in DATA_TYPE_INFO['twitter']
        assert 'features' in DATA_TYPE_INFO['twitter']

    def test_survey_data_type(self):
        """Test Survey data type info."""
        from src.sentiment_analysis import DATA_TYPE_INFO

        assert 'survey' in DATA_TYPE_INFO

    def test_longform_data_type(self):
        """Test Long-form data type info."""
        from src.sentiment_analysis import DATA_TYPE_INFO

        assert 'longform' in DATA_TYPE_INFO


class TestGetSentimentAnalyzer:
    """Tests for get_sentiment_analyzer factory function."""

    def test_get_analyzer_twitter(self):
        """Test getting Twitter analyzer."""
        from src.sentiment_analysis import get_sentiment_analyzer

        # This will fail if transformers is not installed or model can't be downloaded
        try:
            analyzer = get_sentiment_analyzer('twitter')
            assert analyzer is not None
        except ImportError:
            pytest.skip("transformers not installed")
        except OSError:
            pytest.skip("Cannot download model (network/proxy restriction)")

    def test_get_analyzer_survey_vader(self):
        """Test getting Survey analyzer with VADER."""
        from src.sentiment_analysis import get_sentiment_analyzer

        try:
            analyzer = get_sentiment_analyzer('survey', method='vader')
            assert analyzer is not None
        except ImportError:
            pytest.skip("nltk not installed")

    def test_get_analyzer_invalid_type(self):
        """Test that invalid data type raises ValueError."""
        from src.sentiment_analysis import get_sentiment_analyzer

        with pytest.raises(ValueError):
            get_sentiment_analyzer('invalid_type')

    def test_get_analyzer_aliases(self):
        """Test that data type aliases work."""
        from src.sentiment_analysis import get_sentiment_analyzer

        # These should all map to valid analyzer types (but may fail due to missing deps)
        aliases = [
            ('x', 'twitter'),
            ('social', 'twitter'),
            ('stream', 'twitter'),
            ('response', 'survey'),
            ('general', 'survey'),
            ('review', 'longform'),
            ('product', 'longform')
        ]

        for alias, expected_type in aliases:
            try:
                analyzer = get_sentiment_analyzer(alias)
                # If it doesn't raise an exception, the alias worked
                assert True
            except ImportError:
                # Expected if dependencies aren't installed
                pass
            except OSError:
                # Expected if model can't be downloaded (network/proxy)
                pass
            except ValueError:
                # This would indicate the alias isn't recognized
                pytest.fail(f"Alias '{alias}' not recognized")


class TestSurveySentimentAnalyzerVADER:
    """Tests for SurveySentimentAnalyzer with VADER (no transformers required)."""

    @pytest.fixture
    def vader_analyzer(self):
        """Create a VADER-based analyzer."""
        try:
            from src.sentiment_analysis import SurveySentimentAnalyzer
            return SurveySentimentAnalyzer(method='vader')
        except ImportError:
            pytest.skip("nltk not installed")

    def test_analyze_positive(self, vader_analyzer):
        """Test analyzing positive sentiment."""
        results = vader_analyzer.analyze(["This is amazing! I love it so much!"])

        assert len(results) == 1
        assert results[0].label in ['positive', 'neutral', 'negative']
        assert 0 <= results[0].score <= 1

    def test_analyze_negative(self, vader_analyzer):
        """Test analyzing negative sentiment."""
        results = vader_analyzer.analyze(["This is terrible! I hate it!"])

        assert len(results) == 1
        assert results[0].label in ['positive', 'neutral', 'negative']

    def test_analyze_neutral(self, vader_analyzer):
        """Test analyzing neutral sentiment."""
        results = vader_analyzer.analyze(["The meeting is at 3pm."])

        assert len(results) == 1

    def test_analyze_empty_list(self, vader_analyzer):
        """Test analyzing empty list."""
        results = vader_analyzer.analyze([])
        assert len(results) == 0

    def test_analyze_batch(self, vader_analyzer):
        """Test batch analysis."""
        texts = [
            "Great product!",
            "Terrible service",
            "It's okay I guess",
            "No comment"
        ]

        results = vader_analyzer.analyze(texts)

        assert len(results) == len(texts)
        for result in results:
            assert result.label in ['positive', 'neutral', 'negative']
            assert 'positive' in result.scores
            assert 'negative' in result.scores
            assert 'neutral' in result.scores

    def test_model_info(self, vader_analyzer):
        """Test getting model info."""
        info = vader_analyzer.get_model_info()

        assert 'model_name' in info
        assert info['model_name'] == 'VADER'
        assert 'labels' in info


class TestAnalyzeDataframe:
    """Tests for analyze_dataframe method."""

    @pytest.fixture
    def vader_analyzer(self):
        """Create a VADER-based analyzer."""
        try:
            from src.sentiment_analysis import SurveySentimentAnalyzer
            return SurveySentimentAnalyzer(method='vader')
        except ImportError:
            pytest.skip("nltk not installed")

    def test_analyze_dataframe(self, vader_analyzer):
        """Test analyzing a DataFrame."""
        df = pd.DataFrame({
            'text': ['Great!', 'Terrible!', 'Okay'],
            'id': [1, 2, 3]
        })

        result_df = vader_analyzer.analyze_dataframe(df, 'text')

        assert 'sentiment_label' in result_df.columns
        assert 'sentiment_score' in result_df.columns
        assert 'sentiment_positive' in result_df.columns
        assert 'sentiment_negative' in result_df.columns
        assert 'sentiment_neutral' in result_df.columns
        assert len(result_df) == 3

    def test_analyze_dataframe_custom_prefix(self, vader_analyzer):
        """Test analyzing DataFrame with custom prefix."""
        df = pd.DataFrame({
            'text': ['Great!'],
            'id': [1]
        })

        result_df = vader_analyzer.analyze_dataframe(df, 'text', output_prefix='sent')

        assert 'sent_label' in result_df.columns
        assert 'sent_score' in result_df.columns


class TestAnalyzeSentimentFunction:
    """Tests for the convenience analyze_sentiment function."""

    def test_analyze_sentiment_survey(self):
        """Test analyze_sentiment with survey data."""
        try:
            from src.sentiment_analysis import analyze_sentiment

            labels, scores, all_scores = analyze_sentiment(
                ["Great product!", "Terrible service"],
                data_type='survey',
                method='vader'
            )

            assert len(labels) == 2
            assert len(scores) == 2
            assert len(all_scores) == 2
            assert all(label in ['positive', 'neutral', 'negative'] for label in labels)

        except ImportError:
            pytest.skip("nltk not installed")


class TestTwitterSentimentAnalyzerMocked:
    """Tests for TwitterSentimentAnalyzer using mocks (no actual model loading)."""

    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForSequenceClassification')
    def test_twitter_analyzer_init(self, mock_model_class, mock_tokenizer_class):
        """Test TwitterSentimentAnalyzer initialization with mocks."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from src.sentiment_analysis import TwitterSentimentAnalyzer

        # Setup mocks
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()

        analyzer = TwitterSentimentAnalyzer(device='cpu')

        assert analyzer.is_fitted_
        assert analyzer.MODEL_NAME == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        mock_model_class.from_pretrained.assert_called_once()
        mock_tokenizer_class.from_pretrained.assert_called_once()

    def test_twitter_model_info(self):
        """Test that Twitter analyzer has correct model info."""
        from src.sentiment_analysis import TwitterSentimentAnalyzer

        assert TwitterSentimentAnalyzer.MODEL_NAME == "cardiffnlp/twitter-roberta-base-sentiment-latest"
        assert TwitterSentimentAnalyzer.LABEL_MAP == {0: 'negative', 1: 'neutral', 2: 'positive'}
