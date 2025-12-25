"""
Tests for semantic embedding methods.

This module tests the various text embedding classes to ensure they:
1. Implement the correct interface (fit, transform, fit_transform)
2. Handle edge cases (empty texts, None values, short texts)
3. Produce vectors of the expected shape
4. Work offline without requiring API keys
5. Maintain backward compatibility with TF-IDF default
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import warnings

# Sample test data
SAMPLE_TEXTS = [
    "Remote work provides flexibility and autonomy",
    "Working from home improves work-life balance",
    "I love the flexibility of remote work",
    "Office collaboration is important for team dynamics",
    "In-person meetings foster better communication",
    "Remote work can feel isolating",
    "Hybrid work combines benefits of both models",
    "I prefer working in an office with my team",
    "Video calls are exhausting compared to face-to-face",
    "Remote work saves commute time and money"
]

SHORT_TEXTS = ["hi", "test", "ok"]
EMPTY_TEXTS = ["", "   ", None]
SINGLE_TEXT = ["This is a single text sample for testing"]


# Helper functions
def _is_sentence_transformers_available():
    """Check if sentence-transformers is installed."""
    try:
        import sentence_transformers
        return True
    except ImportError:
        return False


class TestWord2VecEmbedder:
    """Test Word2Vec embedding functionality."""

    def test_import(self):
        """Test that Word2VecEmbedder can be imported."""
        from src.embeddings import Word2VecEmbedder
        assert Word2VecEmbedder is not None

    def test_initialization(self):
        """Test Word2VecEmbedder initialization."""
        from src.embeddings import Word2VecEmbedder

        embedder = Word2VecEmbedder(vector_size=100, window=5, min_count=2)
        assert embedder.vector_size == 100
        assert embedder.window == 5
        assert embedder.min_count == 2
        assert embedder.is_fitted_ is False

    def test_fit_transform(self):
        """Test fit_transform produces correct shape."""
        from src.embeddings import Word2VecEmbedder

        embedder = Word2VecEmbedder(vector_size=50, min_count=1)
        vectors = embedder.fit_transform(SAMPLE_TEXTS)

        assert vectors.shape == (len(SAMPLE_TEXTS), 50)
        assert embedder.is_fitted_ is True

    def test_fit_then_transform(self):
        """Test separate fit and transform calls."""
        from src.embeddings import Word2VecEmbedder

        embedder = Word2VecEmbedder(vector_size=50, min_count=1)
        embedder.fit(SAMPLE_TEXTS)

        vectors = embedder.transform(SAMPLE_TEXTS[:3])
        assert vectors.shape == (3, 50)

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises ValueError."""
        from src.embeddings import Word2VecEmbedder

        embedder = Word2VecEmbedder(vector_size=50)

        with pytest.raises(ValueError, match="must be fitted"):
            embedder.transform(SAMPLE_TEXTS)

    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        from src.embeddings import Word2VecEmbedder

        embedder = Word2VecEmbedder(vector_size=50, min_count=1)
        texts = SAMPLE_TEXTS + ["", None]
        vectors = embedder.fit_transform(texts)

        # Should return zero vectors for empty texts
        assert vectors.shape == (len(texts), 50)
        assert np.allclose(vectors[-1], 0)  # Empty text -> zero vector

    def test_short_texts_warning(self):
        """Test that small vocabulary triggers warning."""
        from src.embeddings import Word2VecEmbedder

        embedder = Word2VecEmbedder(vector_size=50, min_count=5)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embedder.fit(SHORT_TEXTS)

            # Should warn about small vocabulary
            assert len(w) > 0
            assert "vocabulary" in str(w[0].message).lower()

    def test_get_feature_names_out(self):
        """Test feature name generation."""
        from src.embeddings import Word2VecEmbedder

        embedder = Word2VecEmbedder(vector_size=10, min_count=1)
        embedder.fit(SAMPLE_TEXTS)

        feature_names = embedder.get_feature_names_out()
        assert len(feature_names) == 10
        assert all(isinstance(name, str) for name in feature_names)


class TestFastTextEmbedder:
    """Test FastText embedding functionality."""

    def test_import(self):
        """Test that FastTextEmbedder can be imported."""
        from src.embeddings import FastTextEmbedder
        assert FastTextEmbedder is not None

    def test_initialization(self):
        """Test FastTextEmbedder initialization."""
        from src.embeddings import FastTextEmbedder

        embedder = FastTextEmbedder(vector_size=100, min_n=3, max_n=6)
        assert embedder.vector_size == 100
        assert embedder.min_n == 3
        assert embedder.max_n == 6
        assert embedder.is_fitted_ is False

    def test_fit_transform(self):
        """Test fit_transform produces correct shape."""
        from src.embeddings import FastTextEmbedder

        embedder = FastTextEmbedder(vector_size=50, min_count=1)
        vectors = embedder.fit_transform(SAMPLE_TEXTS)

        assert vectors.shape == (len(SAMPLE_TEXTS), 50)
        assert embedder.is_fitted_ is True

    def test_handles_oov_words(self):
        """Test that FastText can handle out-of-vocabulary words."""
        from src.embeddings import FastTextEmbedder

        embedder = FastTextEmbedder(vector_size=50, min_count=1)
        embedder.fit(SAMPLE_TEXTS)

        # New text with made-up word (should still work via subwords)
        new_texts = ["remoteify workfromhomeness"]
        vectors = embedder.transform(new_texts)

        assert vectors.shape == (1, 50)
        # Should not be all zeros (FastText can generate via subwords)
        assert not np.allclose(vectors[0], 0)

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises ValueError."""
        from src.embeddings import FastTextEmbedder

        embedder = FastTextEmbedder(vector_size=50)

        with pytest.raises(ValueError, match="must be fitted"):
            embedder.transform(SAMPLE_TEXTS)

    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        from src.embeddings import FastTextEmbedder

        embedder = FastTextEmbedder(vector_size=50, min_count=1)
        texts = SAMPLE_TEXTS + ["", None]
        vectors = embedder.fit_transform(texts)

        # Should return zero vectors for empty texts
        assert vectors.shape == (len(texts), 50)
        assert np.allclose(vectors[-1], 0)


class TestSentenceBERTEmbedder:
    """Test SentenceBERT embedding functionality."""

    def test_import(self):
        """Test that SentenceBERTEmbedder can be imported."""
        from src.embeddings import SentenceBERTEmbedder
        assert SentenceBERTEmbedder is not None

    def test_import_error_without_sentence_transformers(self):
        """Test that helpful error is raised if sentence-transformers not installed."""
        from src.embeddings import SentenceBERTEmbedder

        with patch.dict('sys.modules', {'sentence_transformers': None}):
            # This should raise ImportError with helpful message
            try:
                embedder = SentenceBERTEmbedder()
                embedder.fit(SAMPLE_TEXTS)
            except ImportError as e:
                assert 'sentence-transformers' in str(e)

    @pytest.mark.skipif(
        not _is_sentence_transformers_available(),
        reason="sentence-transformers not installed"
    )
    def test_fit_transform_with_real_model(self):
        """Test fit_transform with actual SentenceBERT model (if available)."""
        from src.embeddings import SentenceBERTEmbedder

        embedder = SentenceBERTEmbedder(model_name='all-MiniLM-L6-v2')
        vectors = embedder.fit_transform(SAMPLE_TEXTS[:3])  # Use subset to save time

        assert vectors.shape[0] == 3
        assert vectors.shape[1] == 384  # all-MiniLM-L6-v2 has 384 dimensions
        assert embedder.is_fitted_ is True

    @pytest.mark.skipif(
        not _is_sentence_transformers_available(),
        reason="sentence-transformers not installed"
    )
    def test_semantic_similarity(self):
        """Test that semantically similar texts have high cosine similarity."""
        from src.embeddings import SentenceBERTEmbedder
        from sklearn.metrics.pairwise import cosine_similarity

        embedder = SentenceBERTEmbedder(model_name='all-MiniLM-L6-v2')

        similar_texts = [
            "Remote work provides flexibility",
            "Working from home offers flexibility",
            "I love pizza"
        ]

        vectors = embedder.fit_transform(similar_texts)

        # First two texts should be more similar than first and third
        sim_0_1 = cosine_similarity([vectors[0]], [vectors[1]])[0, 0]
        sim_0_2 = cosine_similarity([vectors[0]], [vectors[2]])[0, 0]

        assert sim_0_1 > sim_0_2

    def test_initialization_parameters(self):
        """Test SentenceBERTEmbedder initialization parameters."""
        from src.embeddings import SentenceBERTEmbedder

        embedder = SentenceBERTEmbedder(
            model_name='all-MiniLM-L6-v2',
            device='cpu',
            batch_size=16,
            show_progress_bar=False
        )

        assert embedder.model_name == 'all-MiniLM-L6-v2'
        assert embedder.device == 'cpu'
        assert embedder.batch_size == 16
        assert embedder.show_progress_bar is False


class TestGetEmbedder:
    """Test the get_embedder factory function."""

    def test_get_embedder_tfidf(self):
        """Test get_embedder with tfidf returns None."""
        from src.embeddings import get_embedder

        result = get_embedder('tfidf')
        assert result is None  # TF-IDF handled separately

    def test_get_embedder_word2vec(self):
        """Test get_embedder with word2vec."""
        from src.embeddings import get_embedder, Word2VecEmbedder

        embedder = get_embedder('word2vec', vector_size=100)
        assert isinstance(embedder, Word2VecEmbedder)
        assert embedder.vector_size == 100

    def test_get_embedder_fasttext(self):
        """Test get_embedder with fasttext."""
        from src.embeddings import get_embedder, FastTextEmbedder

        embedder = get_embedder('fasttext', vector_size=50)
        assert isinstance(embedder, FastTextEmbedder)
        assert embedder.vector_size == 50

    def test_get_embedder_sbert(self):
        """Test get_embedder with sbert."""
        from src.embeddings import get_embedder, SentenceBERTEmbedder

        embedder = get_embedder('sbert', model_name='all-MiniLM-L6-v2')
        assert isinstance(embedder, SentenceBERTEmbedder)
        assert embedder.model_name == 'all-MiniLM-L6-v2'

    def test_get_embedder_invalid_type(self):
        """Test get_embedder with invalid representation type."""
        from src.embeddings import get_embedder

        with pytest.raises(ValueError, match="Unknown representation"):
            get_embedder('invalid_type')


class TestCompareEmbeddings:
    """Test the compare_embeddings utility function."""

    def test_compare_embeddings_basic(self):
        """Test basic comparison functionality."""
        from src.embeddings import compare_embeddings

        # Use only methods that don't require external dependencies
        results = compare_embeddings(
            SAMPLE_TEXTS,
            representations=['tfidf', 'word2vec', 'fasttext'],
            n_clusters=3
        )

        assert 'tfidf' in results
        assert 'word2vec' in results
        assert 'fasttext' in results

        # Check that metrics are present
        for method in ['tfidf', 'word2vec', 'fasttext']:
            if 'error' not in results[method]:
                assert 'fit_time' in results[method]
                assert 'transform_time' in results[method]
                assert 'total_time' in results[method]
                assert 'n_features' in results[method]

    def test_compare_embeddings_handles_errors(self):
        """Test that compare_embeddings handles errors gracefully."""
        from src.embeddings import compare_embeddings

        # Try with invalid representation (should skip it gracefully)
        results = compare_embeddings(
            SAMPLE_TEXTS,
            representations=['tfidf', 'invalid_method'],
            n_clusters=3
        )

        # Should still have results for valid methods
        assert 'tfidf' in results


class TestBackwardCompatibility:
    """Test backward compatibility with existing TF-IDF workflow."""

    def test_default_representation_is_tfidf(self):
        """Test that default representation is 'tfidf' for backward compatibility."""
        from helpers.analysis import run_ml_analysis
        import pandas as pd

        df = pd.DataFrame({'text': SAMPLE_TEXTS})

        # Should work without specifying representation (defaults to tfidf)
        coder, results, metrics = run_ml_analysis(
            df,
            text_column='text',
            n_codes=3,
            method='tfidf_kmeans'
        )

        assert metrics['representation'] == 'tfidf'
        assert coder is not None
        assert results is not None

    def test_tfidf_kmeans_still_works(self):
        """Test that original tfidf_kmeans method still works identically."""
        from helpers.analysis import run_ml_analysis
        import pandas as pd

        df = pd.DataFrame({'text': SAMPLE_TEXTS})

        coder, results, metrics = run_ml_analysis(
            df,
            text_column='text',
            n_codes=3,
            method='tfidf_kmeans',
            representation='tfidf'  # Explicitly use TF-IDF
        )

        # Should produce valid results
        assert len(results) == len(SAMPLE_TEXTS)
        assert 'assigned_codes' in results.columns
        assert 'confidence_scores' in results.columns
        assert metrics['representation'] == 'tfidf'


class TestIntegrationWithMLOpenCoder:
    """Test integration of embeddings with MLOpenCoder."""

    def test_word2vec_with_kmeans(self):
        """Test Word2Vec representation with K-Means clustering."""
        from helpers.analysis import run_ml_analysis
        import pandas as pd

        df = pd.DataFrame({'text': SAMPLE_TEXTS})

        coder, results, metrics = run_ml_analysis(
            df,
            text_column='text',
            n_codes=3,
            method='tfidf_kmeans',  # K-Means clustering
            representation='word2vec',  # Word2Vec embeddings
            embedding_kwargs={'vector_size': 50, 'min_count': 1}
        )

        assert metrics['representation'] == 'word2vec'
        assert len(results) == len(SAMPLE_TEXTS)
        assert coder.codebook is not None

    def test_fasttext_with_kmeans(self):
        """Test FastText representation with K-Means clustering."""
        from helpers.analysis import run_ml_analysis
        import pandas as pd

        df = pd.DataFrame({'text': SAMPLE_TEXTS})

        coder, results, metrics = run_ml_analysis(
            df,
            text_column='text',
            n_codes=3,
            method='tfidf_kmeans',
            representation='fasttext',
            embedding_kwargs={'vector_size': 50, 'min_count': 1}
        )

        assert metrics['representation'] == 'fasttext'
        assert len(results) == len(SAMPLE_TEXTS)

    @pytest.mark.skipif(
        not _is_sentence_transformers_available(),
        reason="sentence-transformers not installed"
    )
    def test_sbert_with_kmeans(self):
        """Test SentenceBERT representation with K-Means clustering."""
        from helpers.analysis import run_ml_analysis
        import pandas as pd

        df = pd.DataFrame({'text': SAMPLE_TEXTS[:5]})  # Small subset

        coder, results, metrics = run_ml_analysis(
            df,
            text_column='text',
            n_codes=2,
            method='tfidf_kmeans',
            representation='sbert',
            embedding_kwargs={'model_name': 'all-MiniLM-L6-v2'}
        )

        assert metrics['representation'] == 'sbert'
        assert len(results) == 5


# Parameterized tests
@pytest.mark.parametrize("representation,kwargs", [
    ('word2vec', {'vector_size': 50, 'min_count': 1}),
    ('fasttext', {'vector_size': 50, 'min_count': 1}),
])
def test_all_embedders_same_interface(representation, kwargs):
    """Test that all embedders follow the same interface."""
    from src.embeddings import get_embedder

    embedder = get_embedder(representation, **kwargs)

    # Should have fit, transform, fit_transform methods
    assert hasattr(embedder, 'fit')
    assert hasattr(embedder, 'transform')
    assert hasattr(embedder, 'fit_transform')
    assert hasattr(embedder, 'get_feature_names_out')

    # Test the interface works
    vectors = embedder.fit_transform(SAMPLE_TEXTS)
    assert vectors.shape[0] == len(SAMPLE_TEXTS)
    assert vectors.shape[1] == kwargs['vector_size']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
