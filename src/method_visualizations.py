"""
Method-specific visualizations for clustering/topic modeling.

This module provides visualizations tailored to each clustering method:
- TF-IDF + KMeans: Cluster separation plots, silhouette analysis
- NMF: Topic-term heatmaps, topic loadings
- LDA: Topic distributions, pyLDAvis integration

All visualizations use Plotly for interactivity and consistency with the existing UI.

Usage:
    >>> from src.method_visualizations import MethodVisualizer
    >>> viz = MethodVisualizer(coder, results_df, text_column='response')
    >>> fig = viz.create_cluster_scatter()  # PCA/t-SNE scatter
    >>> fig = viz.create_topic_term_heatmap()  # For NMF/LDA
    >>> fig = viz.create_per_cluster_wordcloud(cluster_id)  # Per-cluster word cloud
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # Placeholder for type hints
    px = None
    logger.warning("Plotly not available. Visualizations will be disabled.")

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    plt = None  # Placeholder for type hints
    logger.warning("WordCloud not available.")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_samples, silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn dimensionality reduction not available.")

try:
    import pyLDAvis
    import pyLDAvis.lda_model
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False
    logger.warning("pyLDAvis not available. LDA visualization will use fallback.")


class MethodVisualizer:
    """
    Creates method-specific visualizations for clustering/topic modeling results.

    Provides tailored visualizations based on the ML method used:
    - KMeans: Cluster scatter plots, silhouette analysis
    - NMF/LDA: Topic-term heatmaps, topic distributions
    - All: Per-cluster word clouds

    Example:
        >>> viz = MethodVisualizer(coder, results_df, 'response')
        >>> scatter_fig = viz.create_cluster_scatter(method='pca')
        >>> heatmap_fig = viz.create_topic_term_heatmap(n_terms=15)
    """

    def __init__(
        self,
        coder,
        results_df: pd.DataFrame,
        text_column: str,
        method: Optional[str] = None
    ):
        """
        Initialize the visualizer.

        Args:
            coder: Fitted MLOpenCoder instance with model and vectorizer
            results_df: Results DataFrame with assigned_codes column
            text_column: Name of the text column in results_df
            method: Override method detection ('tfidf_kmeans', 'nmf', 'lda')
        """
        self.coder = coder
        self.results_df = results_df
        self.text_column = text_column
        self.method = method or getattr(coder, 'method', 'tfidf_kmeans')

        # Extract key components
        self.vectorizer = coder.vectorizer
        self.model = coder.model
        self.feature_matrix = coder.feature_matrix
        self.codebook = coder.codebook

        # Get cluster/topic assignments
        if hasattr(self.model, 'labels_'):
            self.assignments = self.model.labels_
        elif hasattr(self.model, 'transform'):
            # LDA/NMF - get topic assignments
            doc_topics = self.model.transform(self.feature_matrix)
            self.assignments = doc_topics.argmax(axis=1)
            self.doc_topic_matrix = doc_topics
        else:
            # Fallback to code assignments
            self.assignments = np.array([
                int(codes[0].split('_')[1]) - 1 if codes else -1
                for codes in results_df['assigned_codes']
            ])

    def create_cluster_scatter(
        self,
        reduction_method: str = 'pca',
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42
    ) -> Optional[Any]:
        """
        Create 2D scatter plot of documents colored by cluster/topic.

        Best for: KMeans (shows cluster separation)
        Also works for: NMF, LDA (shows topic groupings)

        Args:
            reduction_method: 'pca' or 'tsne'
            n_components: Number of dimensions (2 or 3)
            perplexity: t-SNE perplexity (ignored for PCA)
            random_state: Random seed for reproducibility

        Returns:
            Plotly Figure or None if not available
        """
        if not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("Required libraries not available for scatter plot")
            return None

        # Get dense matrix for dimensionality reduction
        if hasattr(self.feature_matrix, 'toarray'):
            dense_matrix = self.feature_matrix.toarray()
        else:
            dense_matrix = np.array(self.feature_matrix)

        # Apply dimensionality reduction
        if reduction_method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced = reducer.fit_transform(dense_matrix)
            explained_var = reducer.explained_variance_ratio_
            axis_labels = [
                f'PC{i+1} ({explained_var[i]:.1%})'
                for i in range(n_components)
            ]
        else:  # t-SNE
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(perplexity, len(dense_matrix) - 1),
                random_state=random_state,
                n_iter=1000
            )
            reduced = reducer.fit_transform(dense_matrix)
            axis_labels = [f't-SNE {i+1}' for i in range(n_components)]

        # Create labels for coloring
        cluster_labels = [f'Cluster {i+1}' for i in self.assignments]

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': reduced[:, 0],
            'y': reduced[:, 1],
            'Cluster': cluster_labels,
            'Cluster_ID': self.assignments
        })

        # Add text preview for hover
        texts = self.results_df[self.text_column].astype(str).tolist()
        plot_df['Text'] = [t[:100] + '...' if len(t) > 100 else t for t in texts]

        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='Cluster',
            hover_data=['Text'],
            title=f'Document Clusters ({reduction_method.upper()})',
            labels={'x': axis_labels[0], 'y': axis_labels[1]}
        )

        fig.update_layout(
            legend_title='Cluster',
            hovermode='closest'
        )

        fig.update_traces(marker=dict(size=8, opacity=0.7))

        return fig

    def create_silhouette_plot(self) -> Optional[Any]:
        """
        Create silhouette plot showing cluster cohesion.

        Best for: KMeans (validates cluster quality)

        Returns:
            Plotly Figure or None if not available
        """
        if not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE:
            return None

        if not hasattr(self.model, 'labels_'):
            logger.warning("Silhouette plot requires KMeans model with labels_")
            return None

        # Get dense matrix
        if hasattr(self.feature_matrix, 'toarray'):
            dense_matrix = self.feature_matrix.toarray()
        else:
            dense_matrix = np.array(self.feature_matrix)

        labels = self.model.labels_
        n_clusters = len(set(labels))

        # Calculate silhouette scores
        silhouette_avg = silhouette_score(dense_matrix, labels)
        sample_silhouette_values = silhouette_samples(dense_matrix, labels)

        # Create figure
        fig = go.Figure()

        y_lower = 10
        for i in range(n_clusters):
            cluster_silhouette_values = sample_silhouette_values[labels == i]
            cluster_silhouette_values.sort()

            size_cluster_i = cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            fig.add_trace(go.Bar(
                x=cluster_silhouette_values,
                y=list(range(y_lower, y_upper)),
                orientation='h',
                name=f'Cluster {i+1}',
                showlegend=True
            ))

            y_lower = y_upper + 10

        # Add average line
        fig.add_vline(
            x=silhouette_avg,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Avg: {silhouette_avg:.3f}"
        )

        fig.update_layout(
            title=f'Silhouette Plot (Avg Score: {silhouette_avg:.3f})',
            xaxis_title='Silhouette Coefficient',
            yaxis_title='Cluster',
            showlegend=True,
            yaxis=dict(showticklabels=False)
        )

        return fig

    def create_topic_term_heatmap(
        self,
        n_terms: int = 15,
        normalize: bool = True
    ) -> Optional[Any]:
        """
        Create heatmap showing top terms per topic.

        Best for: NMF, LDA (shows topic composition)
        Also works for: KMeans (shows cluster centroids)

        Args:
            n_terms: Number of top terms to show per topic
            normalize: Normalize weights to 0-1 range

        Returns:
            Plotly Figure or None if not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Get topic/cluster weights
        if hasattr(self.model, 'components_'):
            # NMF or LDA
            components = self.model.components_
        elif hasattr(self.model, 'cluster_centers_'):
            # KMeans
            components = self.model.cluster_centers_
        else:
            logger.warning("Model has no components_ or cluster_centers_")
            return None

        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        n_topics = components.shape[0]

        # Get top terms for each topic
        all_terms = set()
        topic_term_weights = {}

        for topic_idx in range(n_topics):
            top_indices = components[topic_idx].argsort()[-n_terms:][::-1]
            for idx in top_indices:
                all_terms.add(feature_names[idx])
            topic_term_weights[topic_idx] = {
                feature_names[i]: components[topic_idx][i]
                for i in top_indices
            }

        # Create matrix for heatmap
        terms = sorted(all_terms)
        matrix = np.zeros((n_topics, len(terms)))

        for topic_idx in range(n_topics):
            for j, term in enumerate(terms):
                if term in topic_term_weights[topic_idx]:
                    matrix[topic_idx, j] = topic_term_weights[topic_idx][term]

        # Normalize if requested
        if normalize and matrix.max() > 0:
            matrix = matrix / matrix.max()

        # Create heatmap
        topic_labels = [f'Topic {i+1}' for i in range(n_topics)]

        fig = px.imshow(
            matrix,
            labels=dict(x='Terms', y='Topics', color='Weight'),
            x=terms,
            y=topic_labels,
            title=f'Topic-Term Heatmap (Top {n_terms} Terms)',
            color_continuous_scale='Viridis',
            aspect='auto'
        )

        fig.update_xaxes(tickangle=45)
        fig.update_layout(height=max(400, n_topics * 40))

        return fig

    def create_topic_distribution_chart(self) -> Optional[Any]:
        """
        Create stacked bar chart showing topic distribution per document.

        Best for: NMF, LDA (shows document composition)

        Returns:
            Plotly Figure or None if not available
        """
        if not PLOTLY_AVAILABLE:
            return None

        if not hasattr(self, 'doc_topic_matrix'):
            if hasattr(self.model, 'transform'):
                self.doc_topic_matrix = self.model.transform(self.feature_matrix)
            else:
                logger.warning("Topic distribution requires NMF/LDA model")
                return None

        # Sample documents if too many
        n_docs = min(50, len(self.doc_topic_matrix))
        sample_indices = np.linspace(0, len(self.doc_topic_matrix)-1, n_docs, dtype=int)

        doc_topics = self.doc_topic_matrix[sample_indices]
        n_topics = doc_topics.shape[1]

        # Create stacked bar data
        fig = go.Figure()

        for topic_idx in range(n_topics):
            fig.add_trace(go.Bar(
                name=f'Topic {topic_idx + 1}',
                x=[f'Doc {i+1}' for i in range(n_docs)],
                y=doc_topics[:, topic_idx],
            ))

        fig.update_layout(
            barmode='stack',
            title='Topic Distribution per Document (Sample)',
            xaxis_title='Document',
            yaxis_title='Topic Weight',
            legend_title='Topics'
        )

        return fig

    def create_per_cluster_wordcloud(
        self,
        cluster_id: int,
        max_words: int = 50,
        width: int = 800,
        height: int = 400
    ) -> Optional[Any]:
        """
        Create word cloud for a specific cluster/topic.

        Args:
            cluster_id: Cluster/topic index (0-based)
            max_words: Maximum words in cloud
            width: Image width
            height: Image height

        Returns:
            Matplotlib Figure or None if not available
        """
        if not WORDCLOUD_AVAILABLE:
            return None

        # Get documents in this cluster
        mask = self.assignments == cluster_id
        cluster_texts = self.results_df.loc[mask, self.text_column].tolist()

        if not cluster_texts:
            logger.warning(f"No documents in cluster {cluster_id}")
            return None

        # Combine and clean text
        combined_text = ' '.join(str(t) for t in cluster_texts)
        import re
        cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', combined_text.lower())

        # Generate word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            colormap='viridis',
            max_words=max_words,
            min_font_size=10,
            max_font_size=100
        ).generate(cleaned_text)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Cluster {cluster_id + 1} Word Cloud')

        plt.tight_layout()
        return fig

    def create_all_cluster_wordclouds(
        self,
        max_words: int = 30,
        cols: int = 3
    ) -> Optional[Any]:
        """
        Create grid of word clouds for all clusters.

        Args:
            max_words: Maximum words per cloud
            cols: Number of columns in grid

        Returns:
            Matplotlib Figure or None if not available
        """
        if not WORDCLOUD_AVAILABLE:
            return None

        n_clusters = len(set(self.assignments))
        rows = (n_clusters + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
        axes = axes.flatten() if n_clusters > 1 else [axes]

        import re

        for cluster_id in range(n_clusters):
            ax = axes[cluster_id]

            # Get documents in this cluster
            mask = self.assignments == cluster_id
            cluster_texts = self.results_df.loc[mask, self.text_column].tolist()

            if not cluster_texts:
                ax.text(0.5, 0.5, 'No documents', ha='center', va='center')
                ax.axis('off')
                continue

            # Generate word cloud
            combined_text = ' '.join(str(t) for t in cluster_texts)
            cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', combined_text.lower())

            try:
                wordcloud = WordCloud(
                    width=400,
                    height=200,
                    background_color='white',
                    colormap='viridis',
                    max_words=max_words
                ).generate(cleaned_text)

                ax.imshow(wordcloud, interpolation='bilinear')
            except ValueError:
                ax.text(0.5, 0.5, 'Insufficient text', ha='center', va='center')

            ax.axis('off')
            ax.set_title(f'Cluster {cluster_id + 1} ({sum(mask)} docs)')

        # Hide unused axes
        for idx in range(n_clusters, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def create_lda_visualization(self) -> Optional[str]:
        """
        Create pyLDAvis visualization for LDA model.

        Returns:
            HTML string for pyLDAvis or None if not available
        """
        if not PYLDAVIS_AVAILABLE:
            logger.warning("pyLDAvis not installed. Install with: pip install pyLDAvis")
            return None

        if self.method != 'lda':
            logger.warning("pyLDAvis is only available for LDA models")
            return None

        try:
            # Prepare pyLDAvis data
            if hasattr(self.feature_matrix, 'toarray'):
                doc_term_matrix = self.feature_matrix.toarray()
            else:
                doc_term_matrix = np.array(self.feature_matrix)

            # Get vocabulary
            vocab = self.vectorizer.get_feature_names_out()

            # Prepare the visualization
            vis_data = pyLDAvis.lda_model.prepare(
                self.model,
                self.feature_matrix,
                self.vectorizer
            )

            # Return HTML
            return pyLDAvis.prepared_data_to_html(vis_data)

        except Exception as e:
            logger.error(f"Failed to create pyLDAvis visualization: {e}")
            return None

    def get_method_recommendations(self) -> Dict[str, Any]:
        """
        Get recommended visualizations for the current method.

        Returns:
            Dictionary with recommended visualizations and their availability
        """
        recommendations = {
            'method': self.method,
            'visualizations': {}
        }

        if self.method == 'tfidf_kmeans':
            recommendations['visualizations'] = {
                'cluster_scatter': {
                    'available': PLOTLY_AVAILABLE and SKLEARN_AVAILABLE,
                    'priority': 'high',
                    'description': '2D scatter showing cluster separation'
                },
                'silhouette_plot': {
                    'available': PLOTLY_AVAILABLE and SKLEARN_AVAILABLE,
                    'priority': 'high',
                    'description': 'Cluster cohesion validation'
                },
                'per_cluster_wordclouds': {
                    'available': WORDCLOUD_AVAILABLE,
                    'priority': 'medium',
                    'description': 'Word clouds for each cluster'
                }
            }
        elif self.method in ['nmf', 'lda']:
            recommendations['visualizations'] = {
                'topic_term_heatmap': {
                    'available': PLOTLY_AVAILABLE,
                    'priority': 'high',
                    'description': 'Topic-term weight matrix'
                },
                'topic_distribution': {
                    'available': PLOTLY_AVAILABLE,
                    'priority': 'medium',
                    'description': 'Topic composition per document'
                },
                'per_cluster_wordclouds': {
                    'available': WORDCLOUD_AVAILABLE,
                    'priority': 'medium',
                    'description': 'Word clouds for each topic'
                }
            }

            if self.method == 'lda':
                recommendations['visualizations']['pyldavis'] = {
                    'available': PYLDAVIS_AVAILABLE,
                    'priority': 'high',
                    'description': 'Interactive LDA exploration'
                }

        return recommendations


def create_method_visualizations(
    coder,
    results_df: pd.DataFrame,
    text_column: str
) -> Dict[str, Any]:
    """
    Convenience function to create all appropriate visualizations.

    Args:
        coder: Fitted MLOpenCoder instance
        results_df: Results DataFrame
        text_column: Name of text column

    Returns:
        Dictionary of visualization name -> figure/html
    """
    viz = MethodVisualizer(coder, results_df, text_column)
    method = viz.method

    figures = {}

    # Common visualizations
    if WORDCLOUD_AVAILABLE:
        figures['cluster_wordclouds'] = viz.create_all_cluster_wordclouds()

    # Method-specific
    if method == 'tfidf_kmeans':
        if PLOTLY_AVAILABLE and SKLEARN_AVAILABLE:
            figures['cluster_scatter_pca'] = viz.create_cluster_scatter('pca')
            figures['cluster_scatter_tsne'] = viz.create_cluster_scatter('tsne')
            figures['silhouette_plot'] = viz.create_silhouette_plot()

    elif method in ['nmf', 'lda']:
        if PLOTLY_AVAILABLE:
            figures['topic_term_heatmap'] = viz.create_topic_term_heatmap()
            figures['topic_distribution'] = viz.create_topic_distribution_chart()

        if method == 'lda' and PYLDAVIS_AVAILABLE:
            figures['pyldavis_html'] = viz.create_lda_visualization()

    return figures
