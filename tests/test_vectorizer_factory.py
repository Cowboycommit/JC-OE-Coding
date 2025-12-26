"""
Tests for vectorizer factory module.

These tests verify:
1. VectorizerFactory creates correct vectorizer types for each method
2. TF-IDF matrix is shared between tfidf_kmeans and nmf methods
3. CountVectorizer uses identical settings as TfidfVectorizer
4. Vocabulary is shared across all methods
5. Backward compatibility is maintained
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorizer_factory import (
    VectorizerFactory,
    VectorizerConfig,
    create_vectorizer_for_method
)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TestVectorizerConfig:
    """Tests for VectorizerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = VectorizerConfig()

        assert config.max_features == 1000
        assert config.min_df == 2
        assert config.max_df == 0.8
        assert config.ngram_range == (1, 2)
        assert config.stop_words == 'english'
        assert config.sublinear_tf is False
        assert config.lowercase is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = VectorizerConfig(
            max_features=500,
            min_df=5,
            max_df=0.9,
            ngram_range=(1, 3),
            stop_words=None,
            sublinear_tf=True
        )

        assert config.max_features == 500
        assert config.min_df == 5
        assert config.sublinear_tf is True

    def test_to_base_kwargs(self):
        """Test base kwargs extraction."""
        config = VectorizerConfig(max_features=500, min_df=3)
        kwargs = config.to_base_kwargs()

        assert kwargs['max_features'] == 500
        assert kwargs['min_df'] == 3
        assert 'sublinear_tf' not in kwargs  # TF-IDF specific

    def test_to_tfidf_kwargs(self):
        """Test TF-IDF specific kwargs."""
        config = VectorizerConfig(sublinear_tf=True)
        kwargs = config.to_tfidf_kwargs()

        assert kwargs['sublinear_tf'] is True
        assert kwargs['max_features'] == 1000

    def test_to_count_kwargs(self):
        """Test CountVectorizer kwargs (no TF-IDF params)."""
        config = VectorizerConfig(sublinear_tf=True)  # Should be ignored
        kwargs = config.to_count_kwargs()

        assert 'sublinear_tf' not in kwargs
        assert kwargs['max_features'] == 1000


class TestVectorizerFactory:
    """Tests for VectorizerFactory class."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "machine learning algorithms for prediction",
            "deep learning neural networks training",
            "artificial intelligence research papers",
            "sports news football soccer basketball",
            "athletic competition Olympic games",
            "team sports championship league",
            "political news government policy",
            "election campaign voting democracy",
            "congress parliament legislation bills"
        ] * 20  # 180 docs

    @pytest.fixture
    def factory(self):
        """Create factory instance."""
        return VectorizerFactory()

    @pytest.fixture
    def config(self):
        """Create default config."""
        return VectorizerConfig(
            max_features=100,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )

    def test_create_tfidf_kmeans(self, factory, sample_texts, config):
        """Test TF-IDF vectorizer creation for KMeans."""
        vectorizer, matrix = factory.create_for_method(
            'tfidf_kmeans', sample_texts, config
        )

        assert isinstance(vectorizer, TfidfVectorizer)
        assert matrix.shape[0] == len(sample_texts)
        assert matrix.shape[1] <= config.max_features

    def test_create_nmf(self, factory, sample_texts, config):
        """Test that NMF uses TF-IDF vectorizer."""
        vectorizer, matrix = factory.create_for_method(
            'nmf', sample_texts, config
        )

        assert isinstance(vectorizer, TfidfVectorizer)
        assert matrix.shape[0] == len(sample_texts)

    def test_create_lda(self, factory, sample_texts, config):
        """Test that LDA uses CountVectorizer."""
        vectorizer, matrix = factory.create_for_method(
            'lda', sample_texts, config
        )

        assert isinstance(vectorizer, CountVectorizer)
        assert matrix.shape[0] == len(sample_texts)

    def test_tfidf_matrix_reused_for_nmf(self, factory, sample_texts, config):
        """Test that TF-IDF matrix is cached and reused for NMF."""
        # Create for KMeans first
        vec1, mat1 = factory.create_for_method('tfidf_kmeans', sample_texts, config)

        # Create for NMF - should reuse
        vec2, mat2 = factory.create_for_method('nmf', sample_texts, config)

        # Same object should be returned
        assert vec1 is vec2
        assert mat1 is mat2

    def test_vocabulary_shared_across_methods(self, factory, sample_texts, config):
        """Test that vocabulary is identical across methods."""
        # Create TF-IDF first
        tfidf_vec, _ = factory.create_for_method('tfidf_kmeans', sample_texts, config)
        tfidf_vocab = set(tfidf_vec.vocabulary_.keys())

        # Create CountVectorizer for LDA
        count_vec, _ = factory.create_for_method('lda', sample_texts, config)
        count_vocab = set(count_vec.vocabulary_.keys())

        # Vocabularies should be identical
        assert tfidf_vocab == count_vocab

    def test_matrix_dimensions_match(self, factory, sample_texts, config):
        """Test that matrix dimensions are identical across methods."""
        _, tfidf_mat = factory.create_for_method('tfidf_kmeans', sample_texts, config)
        _, count_mat = factory.create_for_method('lda', sample_texts, config)

        # Same dimensions
        assert tfidf_mat.shape == count_mat.shape

    def test_clear_cache(self, factory, sample_texts, config):
        """Test cache clearing."""
        factory.create_for_method('tfidf_kmeans', sample_texts, config)
        factory.clear_cache()

        assert factory._tfidf_vectorizer is None
        assert factory._tfidf_matrix is None
        assert factory._vocabulary is None

    def test_force_refit(self, factory, sample_texts, config):
        """Test force refit creates new vectorizers."""
        vec1, mat1 = factory.create_for_method('tfidf_kmeans', sample_texts, config)
        vec2, mat2 = factory.create_for_method('tfidf_kmeans', sample_texts, config, force_refit=True)

        # New objects should be created
        assert vec1 is not vec2
        assert mat1 is not mat2

    def test_invalid_method_raises_error(self, factory, sample_texts, config):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError):
            factory.create_for_method('invalid_method', sample_texts, config)

    def test_get_parity_report(self, factory, sample_texts, config):
        """Test parity report generation."""
        factory.create_for_method('tfidf_kmeans', sample_texts, config)
        factory.create_for_method('lda', sample_texts, config)

        report = factory.get_parity_report()

        assert report['tfidf_fitted'] is True
        assert report['count_fitted'] is True
        assert report['vocabulary_shared'] is True
        assert report['vocabulary_identical'] is True

    def test_get_feature_names(self, factory, sample_texts, config):
        """Test getting feature names."""
        factory.create_for_method('tfidf_kmeans', sample_texts, config)
        feature_names = factory.get_feature_names()

        assert feature_names is not None
        assert len(feature_names) > 0

    def test_empty_texts_raises_error(self, factory, config):
        """Test that empty texts raise appropriate error."""
        with pytest.raises(ValueError):
            factory.create_for_method('tfidf_kmeans', [], config)


class TestMethodParity:
    """Tests to verify methodological parity across clustering methods."""

    @pytest.fixture
    def diverse_texts(self):
        """More diverse texts for comprehensive testing."""
        return [
            "technology innovation research development artificial intelligence",
            "machine learning deep neural networks training prediction",
            "sports news football soccer basketball team championship",
            "political government policy election democracy voting rights",
            "business economics market stock finance investment banking",
            "health medical research science biology genetics medicine",
            "entertainment movie film music celebrity arts culture",
            "travel tourism vacation destination hotel flight booking"
        ] * 30  # 240 docs

    def test_all_methods_use_same_preprocessing(self, diverse_texts):
        """Test that all methods use identical preprocessing."""
        factory = VectorizerFactory()
        config = VectorizerConfig(max_features=200, min_df=2, ngram_range=(1, 2))

        # Create all vectorizers
        tfidf_vec, _ = factory.create_for_method('tfidf_kmeans', diverse_texts, config)
        nmf_vec, _ = factory.create_for_method('nmf', diverse_texts, config)
        count_vec, _ = factory.create_for_method('lda', diverse_texts, config)

        # Get feature names
        tfidf_features = set(tfidf_vec.get_feature_names_out())
        nmf_features = set(nmf_vec.get_feature_names_out())
        count_features = set(count_vec.get_feature_names_out())

        # All should have identical features
        assert tfidf_features == nmf_features == count_features

    def test_only_weighting_differs(self, diverse_texts):
        """Test that only term weighting differs between TF-IDF and counts."""
        factory = VectorizerFactory()
        config = VectorizerConfig(max_features=100, min_df=2)

        _, tfidf_mat = factory.create_for_method('tfidf_kmeans', diverse_texts, config)
        _, count_mat = factory.create_for_method('lda', diverse_texts, config)

        # Same sparsity pattern (same non-zero positions)
        tfidf_nonzero = set(zip(*tfidf_mat.nonzero()))
        count_nonzero = set(zip(*count_mat.nonzero()))

        assert tfidf_nonzero == count_nonzero

        # But different values (TF-IDF vs raw counts)
        # TF-IDF values should be different from counts
        tfidf_values = tfidf_mat.data
        count_values = count_mat.data

        # At least some values should differ (TF-IDF applies weighting)
        assert not np.allclose(tfidf_values, count_values)

    def test_nmf_uses_exact_tfidf_matrix(self, diverse_texts):
        """Test that NMF uses exactly the same TF-IDF matrix as KMeans."""
        factory = VectorizerFactory()
        config = VectorizerConfig(max_features=100, min_df=2)

        _, kmeans_mat = factory.create_for_method('tfidf_kmeans', diverse_texts, config)
        _, nmf_mat = factory.create_for_method('nmf', diverse_texts, config)

        # Matrices should be identical
        assert np.allclose(kmeans_mat.toarray(), nmf_mat.toarray())

    def test_clustering_produces_valid_results(self, diverse_texts):
        """Test that all methods produce valid clustering results."""
        from sklearn.cluster import KMeans
        from sklearn.decomposition import LatentDirichletAllocation, NMF

        factory = VectorizerFactory()
        config = VectorizerConfig(max_features=100, min_df=2)

        # KMeans with TF-IDF
        _, tfidf_mat = factory.create_for_method('tfidf_kmeans', diverse_texts, config)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(tfidf_mat)

        # NMF with same TF-IDF
        _, nmf_mat = factory.create_for_method('nmf', diverse_texts, config)
        nmf = NMF(n_components=3, random_state=42, max_iter=100)
        nmf_topics = nmf.fit_transform(nmf_mat)
        nmf_labels = nmf_topics.argmax(axis=1)

        # LDA with counts
        _, count_mat = factory.create_for_method('lda', diverse_texts, config)
        lda = LatentDirichletAllocation(n_components=3, random_state=42, max_iter=10)
        lda_topics = lda.fit_transform(count_mat)
        lda_labels = lda_topics.argmax(axis=1)

        # All should produce valid labels
        assert len(kmeans_labels) == len(diverse_texts)
        assert len(nmf_labels) == len(diverse_texts)
        assert len(lda_labels) == len(diverse_texts)

        assert set(kmeans_labels).issubset({0, 1, 2})
        assert set(nmf_labels).issubset({0, 1, 2})
        assert set(lda_labels).issubset({0, 1, 2})


class TestConvenienceFunction:
    """Tests for the convenience function."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts with enough variety."""
        return [
            "machine learning algorithms for prediction",
            "deep learning neural networks training",
            "artificial intelligence research papers",
            "sports news football soccer basketball",
            "political news government policy"
        ] * 30  # 150 docs

    def test_basic_usage(self, sample_texts):
        """Test basic function usage."""
        vectorizer, matrix = create_vectorizer_for_method(
            'tfidf_kmeans', sample_texts
        )

        assert vectorizer is not None
        assert matrix is not None

    def test_with_config(self, sample_texts):
        """Test with custom config."""
        config = VectorizerConfig(max_features=50)
        vectorizer, matrix = create_vectorizer_for_method(
            'tfidf_kmeans', sample_texts, config=config
        )

        assert matrix.shape[1] <= 50


class TestIntegrationWithMLOpenCoder:
    """Integration tests with MLOpenCoder-style usage."""

    @pytest.fixture
    def diverse_texts(self):
        """Diverse texts for integration testing."""
        return [
            "machine learning prediction model algorithm",
            "deep neural network training optimization",
            "artificial intelligence research development",
            "football soccer basketball sports news",
            "athletic competition games tournament",
            "team championship league standings",
            "political news government legislation",
            "election democracy voting rights",
            "congress senate parliament debate"
        ] * 25

    def test_method_switching(self, diverse_texts):
        """Test switching between methods uses correct vectorizers."""
        factory = VectorizerFactory()
        config = VectorizerConfig(max_features=100, min_df=2)

        # Simulate what MLOpenCoder does
        methods = ['tfidf_kmeans', 'nmf', 'lda']
        vectorizers = {}
        matrices = {}

        for method in methods:
            vec, mat = factory.create_for_method(method, diverse_texts, config)
            vectorizers[method] = vec
            matrices[method] = mat

        # TF-IDF and NMF should share
        assert vectorizers['tfidf_kmeans'] is vectorizers['nmf']

        # LDA should use CountVectorizer
        assert isinstance(vectorizers['lda'], CountVectorizer)
        assert isinstance(vectorizers['tfidf_kmeans'], TfidfVectorizer)

    def test_parity_maintained_after_multiple_calls(self, diverse_texts):
        """Test parity is maintained across multiple factory calls."""
        config = VectorizerConfig(max_features=100, min_df=2)

        # Create multiple factories (simulating separate analysis runs)
        factories = [VectorizerFactory() for _ in range(3)]
        vocab_sizes = []

        for factory in factories:
            factory.create_for_method('tfidf_kmeans', diverse_texts, config)
            factory.create_for_method('lda', diverse_texts, config)
            vocab_sizes.append(len(factory.get_vocabulary()))

        # All should produce same vocabulary size
        assert len(set(vocab_sizes)) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
