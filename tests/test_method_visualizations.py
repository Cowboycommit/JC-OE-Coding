"""
Tests for method-specific visualizations module.

These tests verify:
1. MethodVisualizer works with all three methods (KMeans, NMF, LDA)
2. Visualizations are created correctly for each method type
3. Fallbacks work when optional libraries are unavailable
4. Recommendations are method-appropriate
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.method_visualizations import (
    MethodVisualizer,
    create_method_visualizations,
    PLOTLY_AVAILABLE,
    WORDCLOUD_AVAILABLE,
    SKLEARN_AVAILABLE
)


class MockCoder:
    """Mock MLOpenCoder for testing visualizations."""

    def __init__(self, method='tfidf_kmeans', n_codes=3):
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.cluster import KMeans
        from sklearn.decomposition import LatentDirichletAllocation, NMF

        self.method = method
        self.n_codes = n_codes

        # Sample texts
        self.texts = [
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

        # Preprocess
        import re
        processed = [re.sub(r'[^a-z\s]', ' ', t.lower()) for t in self.texts]

        # Create vectorizer and matrix
        if method == 'lda':
            self.vectorizer = CountVectorizer(max_features=100, min_df=2)
        else:
            self.vectorizer = TfidfVectorizer(max_features=100, min_df=2)

        self.feature_matrix = self.vectorizer.fit_transform(processed)

        # Create model
        if method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=n_codes, random_state=42, max_iter=10
            )
            doc_topics = self.model.fit_transform(self.feature_matrix)
            self.labels_ = doc_topics.argmax(axis=1)
        elif method == 'nmf':
            self.model = NMF(n_components=n_codes, random_state=42, max_iter=100)
            doc_topics = self.model.fit_transform(self.feature_matrix)
            self.labels_ = doc_topics.argmax(axis=1)
        else:  # tfidf_kmeans
            self.model = KMeans(n_clusters=n_codes, random_state=42, n_init=10)
            self.model.fit(self.feature_matrix)
            self.labels_ = self.model.labels_

        # Create codebook
        self.codebook = {
            f'CODE_{i+1:02d}': {
                'label': f'Code {i+1}',
                'count': int(sum(self.labels_ == i)),
                'keywords': ['term1', 'term2', 'term3']
            }
            for i in range(n_codes)
        }


@pytest.fixture
def kmeans_coder():
    """Create KMeans mock coder."""
    return MockCoder(method='tfidf_kmeans', n_codes=3)


@pytest.fixture
def nmf_coder():
    """Create NMF mock coder."""
    return MockCoder(method='nmf', n_codes=3)


@pytest.fixture
def lda_coder():
    """Create LDA mock coder."""
    return MockCoder(method='lda', n_codes=3)


@pytest.fixture
def sample_results_df(kmeans_coder):
    """Create sample results DataFrame."""
    coder = kmeans_coder
    return pd.DataFrame({
        'text': coder.texts,
        'assigned_codes': [[f'CODE_{l+1:02d}'] for l in coder.labels_],
        'confidence_scores': [[0.8] for _ in coder.labels_],
        'num_codes': [1 for _ in coder.labels_]
    })


class TestMethodVisualizer:
    """Tests for MethodVisualizer class."""

    def test_init_kmeans(self, kmeans_coder, sample_results_df):
        """Test initialization with KMeans model."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')

        assert viz.method == 'tfidf_kmeans'
        assert viz.vectorizer is not None
        assert viz.model is not None
        assert len(viz.assignments) == len(sample_results_df)

    def test_init_nmf(self, nmf_coder):
        """Test initialization with NMF model."""
        results_df = pd.DataFrame({
            'text': nmf_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in nmf_coder.labels_]
        })
        viz = MethodVisualizer(nmf_coder, results_df, 'text')

        assert viz.method == 'nmf'
        assert hasattr(viz, 'doc_topic_matrix')

    def test_init_lda(self, lda_coder):
        """Test initialization with LDA model."""
        results_df = pd.DataFrame({
            'text': lda_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in lda_coder.labels_]
        })
        viz = MethodVisualizer(lda_coder, results_df, 'text')

        assert viz.method == 'lda'
        assert hasattr(viz, 'doc_topic_matrix')


class TestClusterScatter:
    """Tests for cluster scatter plot."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE,
                        reason="Plotly or sklearn not available")
    def test_pca_scatter(self, kmeans_coder, sample_results_df):
        """Test PCA scatter plot creation."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_cluster_scatter(reduction_method='pca')

        assert fig is not None
        assert 'PCA' in fig.layout.title.text

    @pytest.mark.skipif(not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE,
                        reason="Plotly or sklearn not available")
    def test_tsne_scatter(self, kmeans_coder, sample_results_df):
        """Test t-SNE scatter plot creation."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_cluster_scatter(reduction_method='tsne')

        assert fig is not None
        assert 't-SNE' in fig.layout.title.text.upper() or 'TSNE' in fig.layout.title.text.upper()


class TestClusterNetwork:
    """Tests for cluster network diagram."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_network_spring_layout(self, kmeans_coder, sample_results_df):
        """Test network diagram with spring layout."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_cluster_network(layout='spring')

        assert fig is not None
        assert 'Network' in fig.layout.title.text

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_network_circular_layout(self, kmeans_coder, sample_results_df):
        """Test network diagram with circular layout."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_cluster_network(layout='circular')

        assert fig is not None
        assert 'Network' in fig.layout.title.text

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_network_kamada_kawai_layout(self, kmeans_coder, sample_results_df):
        """Test network diagram with Kamada-Kawai layout."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_cluster_network(layout='kamada_kawai')

        assert fig is not None
        assert 'Network' in fig.layout.title.text

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_network_nmf(self, nmf_coder):
        """Test network diagram for NMF model."""
        results_df = pd.DataFrame({
            'text': nmf_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in nmf_coder.labels_]
        })
        viz = MethodVisualizer(nmf_coder, results_df, 'text')
        fig = viz.create_cluster_network()

        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_network_lda(self, lda_coder):
        """Test network diagram for LDA model."""
        results_df = pd.DataFrame({
            'text': lda_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in lda_coder.labels_]
        })
        viz = MethodVisualizer(lda_coder, results_df, 'text')
        fig = viz.create_cluster_network()

        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_network_with_edge_labels(self, kmeans_coder, sample_results_df):
        """Test network diagram with edge labels shown."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_cluster_network(show_edge_labels=True)

        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_network_similarity_threshold(self, kmeans_coder, sample_results_df):
        """Test network diagram with different similarity thresholds."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')

        # Low threshold should show more edges
        fig_low = viz.create_cluster_network(similarity_threshold=0.0)
        # High threshold should show fewer edges
        fig_high = viz.create_cluster_network(similarity_threshold=0.9)

        assert fig_low is not None
        assert fig_high is not None


class TestSilhouettePlot:
    """Tests for silhouette plot."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE,
                        reason="Plotly or sklearn not available")
    def test_silhouette_kmeans(self, kmeans_coder, sample_results_df):
        """Test silhouette plot for KMeans."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_silhouette_plot()

        assert fig is not None
        assert 'Silhouette' in fig.layout.title.text

    def test_silhouette_not_for_lda(self, lda_coder):
        """Test that silhouette plot warns for non-KMeans."""
        results_df = pd.DataFrame({
            'text': lda_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in lda_coder.labels_]
        })
        viz = MethodVisualizer(lda_coder, results_df, 'text')
        fig = viz.create_silhouette_plot()

        # Should return None for LDA (no labels_ attribute on LDA model)
        assert fig is None


class TestTopicTermHeatmap:
    """Tests for topic-term heatmap."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_heatmap_nmf(self, nmf_coder):
        """Test heatmap for NMF."""
        results_df = pd.DataFrame({
            'text': nmf_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in nmf_coder.labels_]
        })
        viz = MethodVisualizer(nmf_coder, results_df, 'text')
        fig = viz.create_topic_term_heatmap(n_terms=10)

        assert fig is not None
        assert 'Topic-Term' in fig.layout.title.text

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_heatmap_lda(self, lda_coder):
        """Test heatmap for LDA."""
        results_df = pd.DataFrame({
            'text': lda_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in lda_coder.labels_]
        })
        viz = MethodVisualizer(lda_coder, results_df, 'text')
        fig = viz.create_topic_term_heatmap(n_terms=10)

        assert fig is not None

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_heatmap_kmeans(self, kmeans_coder, sample_results_df):
        """Test heatmap works for KMeans centroids too."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_topic_term_heatmap(n_terms=10)

        assert fig is not None


class TestTopicDistribution:
    """Tests for topic distribution chart."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_distribution_nmf(self, nmf_coder):
        """Test topic distribution for NMF."""
        results_df = pd.DataFrame({
            'text': nmf_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in nmf_coder.labels_]
        })
        viz = MethodVisualizer(nmf_coder, results_df, 'text')
        fig = viz.create_topic_distribution_chart()

        assert fig is not None
        assert 'Topic Distribution' in fig.layout.title.text

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_distribution_lda(self, lda_coder):
        """Test topic distribution for LDA."""
        results_df = pd.DataFrame({
            'text': lda_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in lda_coder.labels_]
        })
        viz = MethodVisualizer(lda_coder, results_df, 'text')
        fig = viz.create_topic_distribution_chart()

        assert fig is not None


class TestWordClouds:
    """Tests for word cloud visualizations."""

    @pytest.mark.skipif(not WORDCLOUD_AVAILABLE, reason="WordCloud not available")
    def test_single_cluster_wordcloud(self, kmeans_coder, sample_results_df):
        """Test single cluster word cloud."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_per_cluster_wordcloud(cluster_id=0)

        assert fig is not None

    @pytest.mark.skipif(not WORDCLOUD_AVAILABLE, reason="WordCloud not available")
    def test_all_cluster_wordclouds(self, kmeans_coder, sample_results_df):
        """Test grid of all cluster word clouds."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        fig = viz.create_all_cluster_wordclouds(max_words=20)

        assert fig is not None

    @pytest.mark.skipif(not WORDCLOUD_AVAILABLE, reason="WordCloud not available")
    def test_wordcloud_empty_cluster(self, kmeans_coder, sample_results_df):
        """Test word cloud handles empty cluster gracefully."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        # Try with invalid cluster ID
        fig = viz.create_per_cluster_wordcloud(cluster_id=999)

        assert fig is None


class TestRecommendations:
    """Tests for visualization recommendations."""

    def test_kmeans_recommendations(self, kmeans_coder, sample_results_df):
        """Test recommendations for KMeans."""
        viz = MethodVisualizer(kmeans_coder, sample_results_df, 'text')
        recs = viz.get_method_recommendations()

        assert recs['method'] == 'tfidf_kmeans'
        assert 'cluster_scatter' in recs['visualizations']
        assert 'silhouette_plot' in recs['visualizations']

    def test_lda_recommendations(self, lda_coder):
        """Test recommendations for LDA."""
        results_df = pd.DataFrame({
            'text': lda_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in lda_coder.labels_]
        })
        viz = MethodVisualizer(lda_coder, results_df, 'text')
        recs = viz.get_method_recommendations()

        assert recs['method'] == 'lda'
        assert 'topic_term_heatmap' in recs['visualizations']
        assert 'pyldavis' in recs['visualizations']

    def test_nmf_recommendations(self, nmf_coder):
        """Test recommendations for NMF."""
        results_df = pd.DataFrame({
            'text': nmf_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in nmf_coder.labels_]
        })
        viz = MethodVisualizer(nmf_coder, results_df, 'text')
        recs = viz.get_method_recommendations()

        assert recs['method'] == 'nmf'
        assert 'topic_term_heatmap' in recs['visualizations']
        assert 'topic_distribution' in recs['visualizations']


class TestConvenienceFunction:
    """Tests for create_method_visualizations convenience function."""

    def test_kmeans_visualizations(self, kmeans_coder, sample_results_df):
        """Test convenience function for KMeans."""
        figs = create_method_visualizations(kmeans_coder, sample_results_df, 'text')

        assert isinstance(figs, dict)
        # Should have at least some visualizations
        if PLOTLY_AVAILABLE and SKLEARN_AVAILABLE:
            assert 'cluster_scatter_pca' in figs or 'silhouette_plot' in figs

    def test_nmf_visualizations(self, nmf_coder):
        """Test convenience function for NMF."""
        results_df = pd.DataFrame({
            'text': nmf_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in nmf_coder.labels_]
        })
        figs = create_method_visualizations(nmf_coder, results_df, 'text')

        assert isinstance(figs, dict)
        if PLOTLY_AVAILABLE:
            assert 'topic_term_heatmap' in figs

    def test_lda_visualizations(self, lda_coder):
        """Test convenience function for LDA."""
        results_df = pd.DataFrame({
            'text': lda_coder.texts,
            'assigned_codes': [[f'CODE_{l+1:02d}'] for l in lda_coder.labels_]
        })
        figs = create_method_visualizations(lda_coder, results_df, 'text')

        assert isinstance(figs, dict)
        if PLOTLY_AVAILABLE:
            assert 'topic_term_heatmap' in figs


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
