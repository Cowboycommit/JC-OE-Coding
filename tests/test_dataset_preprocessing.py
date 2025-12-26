"""
Tests for dataset-aware preprocessing module.

These tests verify:
1. Dataset characteristics detection works correctly
2. Adaptive preprocessing generates appropriate configs
3. Multi-label handling works correctly
4. Backward compatibility is maintained
"""

import pytest
import numpy as np
import pandas as pd
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_preprocessing import (
    DatasetCharacteristicsDetector,
    DatasetCharacteristics,
    AdaptivePreprocessor,
    PreprocessingConfig,
    MultiLabelHandler,
    get_adaptive_preprocessing
)


class TestDatasetCharacteristicsDetector:
    """Tests for DatasetCharacteristicsDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a default detector instance."""
        return DatasetCharacteristicsDetector()

    @pytest.fixture
    def short_responses(self) -> List[str]:
        """Sample short OE survey responses."""
        return [
            "I like remote work",
            "Great product overall",
            "Could be better",
            "Very satisfied",
            "Nothing to add",
            "Good experience",
            "Needs improvement",
            "Love it",
            "Not sure about this",
            "Pretty good service"
        ]

    @pytest.fixture
    def long_form_documents(self) -> List[str]:
        """Sample long-form documents (news article style)."""
        base_doc = (
            "This is a longer document that simulates news article content. "
            "It contains multiple sentences and discusses various topics in depth. "
            "The document continues with more detailed information about the subject matter. "
            "Additional context is provided to give readers a comprehensive understanding. "
            "The article explores different perspectives and includes expert opinions. "
            "Statistical data and research findings are also presented for credibility. "
        )
        return [base_doc * 3 for _ in range(20)]  # ~150 words each

    @pytest.fixture
    def multi_label_data(self) -> List[str]:
        """Sample multi-label labels."""
        return [
            "business, technology",
            "sports",
            "politics, international",
            "technology, science",
            "entertainment",
            "business",
            "health, science",
            "sports, entertainment",
            "unknown",
            "politics"
        ]

    def test_analyze_empty_texts(self, detector):
        """Test handling of empty text list."""
        characteristics = detector.analyze([])
        assert characteristics.n_documents == 0

    def test_analyze_short_responses(self, detector, short_responses):
        """Test detection of short OE responses."""
        characteristics = detector.analyze(short_responses)

        assert characteristics.n_documents == 10
        assert characteristics.is_short_form is True
        assert characteristics.is_long_form is False
        assert characteristics.median_doc_length < 30
        assert characteristics.suggested_preprocessing in ['minimal', 'standard']

    def test_analyze_long_form_documents(self, detector, long_form_documents):
        """Test detection of long-form documents."""
        characteristics = detector.analyze(long_form_documents)

        assert characteristics.n_documents == 20
        assert characteristics.is_long_form is True
        assert characteristics.is_short_form is False
        assert characteristics.median_doc_length >= 100
        assert characteristics.suggested_preprocessing == 'aggressive'

    def test_detect_multi_label(self, detector, short_responses, multi_label_data):
        """Test multi-label detection in labels."""
        characteristics = detector.analyze(short_responses, labels=multi_label_data)

        assert characteristics.has_labels is True
        assert characteristics.is_multi_label is True
        assert characteristics.label_delimiter == ','

    def test_detect_single_label(self, detector, short_responses):
        """Test single-label detection."""
        single_labels = ["cat1", "cat2", "cat3", "cat1", "cat2",
                        "cat1", "cat3", "cat2", "cat1", "cat2"]
        characteristics = detector.analyze(short_responses, labels=single_labels)

        assert characteristics.has_labels is True
        assert characteristics.is_multi_label is False

    def test_missing_label_ratio(self, detector, short_responses):
        """Test missing label ratio calculation."""
        labels_with_missing = [
            "cat1", "unknown", "cat2", "n/a", "cat3",
            None, "cat1", "", "cat2", "cat3"
        ]
        characteristics = detector.analyze(short_responses, labels=labels_with_missing)

        assert characteristics.label_missing_ratio > 0.3  # 4/10 missing

    def test_corpus_size_classification(self, detector):
        """Test corpus size category detection."""
        # Small corpus
        small_texts = ["text"] * 100
        chars = detector.analyze(small_texts)
        assert chars.corpus_size_category == 'small'

        # Medium corpus
        medium_texts = ["text"] * 1000
        chars = detector.analyze(medium_texts)
        assert chars.corpus_size_category == 'medium'

        # Large corpus
        large_texts = ["text"] * 10000
        chars = detector.analyze(large_texts)
        assert chars.corpus_size_category == 'large'

    def test_to_dict(self, detector, short_responses):
        """Test dictionary serialization."""
        characteristics = detector.analyze(short_responses)
        result = characteristics.to_dict()

        assert isinstance(result, dict)
        assert 'n_documents' in result
        assert 'median_doc_length' in result
        assert 'is_long_form' in result
        assert 'suggested_preprocessing' in result


class TestAdaptivePreprocessor:
    """Tests for AdaptivePreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create a default preprocessor instance."""
        return AdaptivePreprocessor()

    def test_config_for_short_form(self, preprocessor):
        """Test config generation for short-form documents."""
        characteristics = DatasetCharacteristics(
            n_documents=100,
            median_doc_length=15,
            is_short_form=True,
            is_long_form=False,
            vocabulary_size=300,
            suggested_preprocessing='standard',
            corpus_size_category='small'
        )

        config = preprocessor.get_config(characteristics)

        # Short form should have lower min_df to preserve vocabulary
        assert config.min_df <= 2
        assert config.use_stopwords is True
        assert config.sublinear_tf is False

    def test_config_for_long_form(self, preprocessor):
        """Test config generation for long-form documents."""
        characteristics = DatasetCharacteristics(
            n_documents=1000,
            median_doc_length=250,
            is_short_form=False,
            is_long_form=True,
            vocabulary_size=10000,
            suggested_preprocessing='aggressive',
            corpus_size_category='medium'
        )

        config = preprocessor.get_config(characteristics)

        # Long form should have stricter preprocessing
        assert config.min_df >= 5
        assert config.sublinear_tf is True
        assert config.max_df <= 0.9
        assert config.ngram_range == (1, 2)

    def test_config_override(self, preprocessor):
        """Test manual config override."""
        characteristics = DatasetCharacteristics(
            suggested_preprocessing='standard',
            corpus_size_category='small'
        )

        config = preprocessor.get_config(
            characteristics,
            override_config={'min_df': 10, 'max_features': 500}
        )

        assert config.min_df == 10
        assert config.max_features == 500
        assert '[with manual overrides]' in config.config_rationale

    def test_create_tfidf_vectorizer(self, preprocessor):
        """Test TF-IDF vectorizer creation."""
        config = PreprocessingConfig(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            sublinear_tf=True,
            use_stopwords=True,
            stopwords_language='english'
        )

        vectorizer = preprocessor.create_vectorizer(config, for_lda=False)

        assert hasattr(vectorizer, 'fit_transform')
        assert vectorizer.max_features == 1000
        assert vectorizer.sublinear_tf is True

    def test_create_count_vectorizer_for_lda(self, preprocessor):
        """Test CountVectorizer creation for LDA."""
        config = PreprocessingConfig(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            sublinear_tf=True  # Should be ignored for LDA
        )

        vectorizer = preprocessor.create_vectorizer(config, for_lda=True)

        # Should be CountVectorizer, not TfidfVectorizer
        from sklearn.feature_extraction.text import CountVectorizer
        assert isinstance(vectorizer, CountVectorizer)

    def test_config_to_dict(self, preprocessor):
        """Test config dictionary conversion."""
        config = PreprocessingConfig(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2),
            use_stopwords=True,
            stopwords_language='english'
        )

        config_dict = config.to_dict()

        assert config_dict['max_features'] == 1000
        assert config_dict['min_df'] == 2
        assert config_dict['stop_words'] == 'english'


class TestMultiLabelHandler:
    """Tests for MultiLabelHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a default handler instance."""
        return MultiLabelHandler(label_delimiter=',')

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create sample DataFrame with multi-label data."""
        return pd.DataFrame({
            'text': [
                "Document about technology",
                "Sports news update",
                "Business and technology",
                "Entertainment news"
            ],
            'labels': [
                "technology, science",
                "sports",
                "business, technology",
                "entertainment"
            ]
        })

    def test_prepare_for_clustering(self, handler, sample_df):
        """Test preparing data for clustering."""
        texts, labels_df = handler.prepare_for_clustering(
            sample_df, 'text', 'labels'
        )

        assert len(texts) == 4
        assert labels_df is not None
        assert 'labels' in labels_df.columns

    def test_parse_multi_labels(self, handler):
        """Test parsing multi-label strings."""
        labels = handler.parse_labels("cat1, cat2, cat3")
        assert labels == ['cat1', 'cat2', 'cat3']

        single = handler.parse_labels("single_label")
        assert single == ['single_label']

        empty = handler.parse_labels("")
        assert empty == []

    def test_filter_single_label_documents(self, handler, sample_df):
        """Test filtering to single-label documents."""
        filtered = handler.filter_single_label_documents(sample_df, 'labels')

        # Only 'sports' and 'entertainment' are single-label
        assert len(filtered) == 2
        assert all(',' not in str(l) for l in filtered['labels'])

    def test_calculate_label_cluster_overlap(self, handler):
        """Test label-cluster overlap calculation."""
        clusters = [0, 0, 1, 1, 2, 2]
        labels = ["cat1", "cat1", "cat2", "cat2, cat3", "cat3", "cat3"]

        result = handler.calculate_label_cluster_overlap(clusters, labels)

        assert 'cluster_purity' in result
        assert 'overall_purity' in result
        assert 'label_coverage' in result
        assert result['overall_purity'] >= 0 and result['overall_purity'] <= 1


class TestGetAdaptivePreprocessing:
    """Tests for the convenience function."""

    def test_basic_usage(self):
        """Test basic function usage."""
        texts = ["Short response one", "Short response two"] * 50

        config, characteristics = get_adaptive_preprocessing(texts)

        assert isinstance(config, PreprocessingConfig)
        assert isinstance(characteristics, DatasetCharacteristics)

    def test_with_labels(self):
        """Test with labels for characteristic detection."""
        texts = ["Response"] * 100
        labels = ["cat1, cat2"] * 50 + ["cat3"] * 50

        config, characteristics = get_adaptive_preprocessing(texts, labels=labels)

        assert characteristics.has_labels is True
        assert characteristics.is_multi_label is True

    def test_with_override(self):
        """Test with manual override."""
        texts = ["Response"] * 100

        config, _ = get_adaptive_preprocessing(
            texts,
            override_config={'min_df': 10}
        )

        assert config.min_df == 10


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_default_preprocessing_for_simple_data(self):
        """Test that simple OE data gets appropriate preprocessing."""
        # Use slightly longer responses to test standard preprocessing
        texts = [
            "I really liked the product and would definitely buy it again soon",
            "The customer service was great and they resolved my issue quickly",
            "This could definitely be better in terms of quality and value",
            "I am very satisfied with my purchase and the overall experience",
            "Would definitely recommend this product to friends and family members"
        ] * 20  # 100 responses

        config, characteristics = get_adaptive_preprocessing(texts)

        # For simple OE responses, should use standard/minimal preprocessing
        assert characteristics.suggested_preprocessing in ['standard', 'minimal']
        assert config.min_df <= 2  # Not too aggressive
        # Stopwords may be enabled or disabled based on response length
        # The key is that we don't have aggressive filtering

    def test_works_without_labels(self):
        """Test that preprocessing works without any labels."""
        texts = ["Response text"] * 100

        config, characteristics = get_adaptive_preprocessing(texts, labels=None)

        assert characteristics.has_labels is False
        assert config is not None

    def test_empty_characteristics_on_empty_input(self):
        """Test graceful handling of empty input."""
        detector = DatasetCharacteristicsDetector()
        characteristics = detector.analyze([])

        assert characteristics.n_documents == 0
        assert characteristics.suggested_preprocessing == 'standard'


class TestIntegrationWithSklearn:
    """Integration tests with scikit-learn."""

    def test_vectorizer_produces_valid_matrix(self):
        """Test that created vectorizer produces valid feature matrix."""
        texts = [
            "This is a test document about machine learning",
            "Another document discussing natural language processing",
            "Machine learning is a subset of artificial intelligence",
            "NLP is used for text analysis and understanding",
            "Deep learning models can process text efficiently"
        ] * 20

        config, _ = get_adaptive_preprocessing(texts)
        preprocessor = AdaptivePreprocessor()
        vectorizer = preprocessor.create_vectorizer(config)

        matrix = vectorizer.fit_transform(texts)

        assert matrix.shape[0] == len(texts)
        assert matrix.shape[1] <= config.max_features
        assert matrix.nnz > 0  # Has non-zero entries

    def test_vectorizer_works_with_kmeans(self):
        """Test that created vectorizer works with KMeans."""
        from sklearn.cluster import KMeans

        texts = [
            "Technology and innovation are important",
            "Sports news and updates for fans",
            "Political developments in the region",
            "Entertainment and celebrity news",
            "Business and financial markets today"
        ] * 30

        config, _ = get_adaptive_preprocessing(texts)
        preprocessor = AdaptivePreprocessor()
        vectorizer = preprocessor.create_vectorizer(config)

        matrix = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)

        assert len(labels) == len(texts)
        assert len(set(labels)) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
