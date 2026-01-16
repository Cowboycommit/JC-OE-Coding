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
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cosine, pdist, squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Cluster boundary overlays will be disabled.")

try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.warning("gensim not available. Semantic wordclouds will use fallback coloring.")

try:
    import pyLDAvis
    import pyLDAvis.lda_model
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False
    logger.warning("pyLDAvis not available. LDA visualization will use fallback.")


# =============================================================================
# VISUALIZATION-METHOD COMPATIBILITY MATRIX
# =============================================================================
# Defines which visualizations are available/recommended for each method
# =============================================================================

VISUALIZATION_COMPATIBILITY = {
    'cluster_scatter': {
        'compatible': ['tfidf_kmeans', 'nmf', 'lda'],
        'recommended': ['tfidf_kmeans'],
        'description': 'PCA/t-SNE scatter plot showing document clusters',
        'note': 'Best for KMeans - shows cluster separation. Works but less meaningful for topic models.'
    },
    'silhouette_plot': {
        'compatible': ['tfidf_kmeans'],
        'recommended': ['tfidf_kmeans'],
        'description': 'Silhouette analysis for cluster validation',
        'note': 'ONLY available for KMeans. Not applicable to topic models (NMF/LDA).'
    },
    'topic_term_heatmap': {
        'compatible': ['tfidf_kmeans', 'nmf', 'lda'],
        'recommended': ['nmf', 'lda'],
        'description': 'Heatmap showing top terms per topic/cluster',
        'note': 'Best for NMF/LDA topic interpretation. Also works for KMeans centroids.'
    },
    'topic_distribution': {
        'compatible': ['nmf', 'lda'],
        'recommended': ['nmf', 'lda'],
        'description': 'Stacked bar showing topic composition per document',
        'note': 'ONLY available for NMF/LDA. Not applicable to KMeans (hard assignments).'
    },
    'pyldavis': {
        'compatible': ['lda'],
        'recommended': ['lda'],
        'description': 'Interactive LDA topic exploration',
        'note': 'ONLY available for LDA method. Requires pyLDAvis package.'
    },
    'wordcloud': {
        'compatible': ['tfidf_kmeans', 'nmf', 'lda'],
        'recommended': ['tfidf_kmeans', 'nmf', 'lda'],
        'description': 'Word cloud for cluster/topic',
        'note': 'Available for all methods.'
    },
    'semantic_wordcloud': {
        'compatible': ['tfidf_kmeans', 'nmf', 'lda'],
        'recommended': ['tfidf_kmeans', 'nmf', 'lda'],
        'description': 'Semantic word cloud with color-coded word meanings',
        'note': 'Colors represent semantic similarity - similar colors mean similar meanings.'
    }
}


def get_visualization_availability(method: str) -> Dict[str, Dict[str, Any]]:
    """
    Get available visualizations for a specific method.

    Args:
        method: The ML method ('tfidf_kmeans', 'nmf', 'lda')

    Returns:
        Dictionary of visualization name -> availability info
    """
    result = {}
    for viz_name, info in VISUALIZATION_COMPATIBILITY.items():
        is_compatible = method in info['compatible']
        is_recommended = method in info['recommended']
        result[viz_name] = {
            'available': is_compatible,
            'recommended': is_recommended,
            'description': info['description'],
            'note': info['note']
        }
    return result


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

    def _get_topic_label(self, cluster_idx: int) -> str:
        """
        Get human-readable topic label for a cluster/topic index.

        Args:
            cluster_idx: Zero-based cluster/topic index

        Returns:
            Human-readable label from codebook, or fallback to "Topic N"
        """
        # Try to get label from codebook (CODE_01, CODE_02, etc.)
        code_id = f"CODE_{cluster_idx + 1:02d}"
        if self.codebook and code_id in self.codebook:
            label = self.codebook[code_id].get('label', '')
            if label:
                return label
        return f"Topic {cluster_idx + 1}"

    def create_cluster_scatter(
        self,
        reduction_method: str = 'pca',
        n_components: int = 2,
        perplexity: int = 30,
        random_state: int = 42,
        show_cluster_overlay: bool = True,
        overlay_opacity: float = 0.15
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
            show_cluster_overlay: Whether to show semi-transparent convex hull overlays
            overlay_opacity: Opacity of the cluster boundary overlays (0.0-1.0)

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
                max_iter=1000
            )
            reduced = reducer.fit_transform(dense_matrix)
            axis_labels = [f't-SNE {i+1}' for i in range(n_components)]

        # Create labels for coloring using topic labels from codebook
        cluster_labels = [self._get_topic_label(i) for i in self.assignments]

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

        # Create figure with graph_objects for more control
        fig = go.Figure()

        # Get unique clusters and assign colors
        unique_clusters = sorted(plot_df['Cluster_ID'].unique())
        n_clusters = len(unique_clusters)

        # Use Plotly's qualitative color palette
        colors = px.colors.qualitative.Plotly
        if n_clusters > len(colors):
            # Extend colors if needed
            colors = colors * (n_clusters // len(colors) + 1)

        # Add cluster boundary overlays first (so they appear behind points)
        if show_cluster_overlay and SCIPY_AVAILABLE:
            for i, cluster_id in enumerate(unique_clusters):
                cluster_mask = plot_df['Cluster_ID'] == cluster_id
                cluster_points = plot_df[cluster_mask][['x', 'y']].values

                # Need at least 3 points for convex hull
                if len(cluster_points) >= 3:
                    try:
                        hull = ConvexHull(cluster_points)
                        # Get hull vertices in order
                        hull_points = cluster_points[hull.vertices]
                        # Close the polygon by repeating the first point
                        hull_x = np.append(hull_points[:, 0], hull_points[0, 0])
                        hull_y = np.append(hull_points[:, 1], hull_points[0, 1])

                        fig.add_trace(go.Scatter(
                            x=hull_x,
                            y=hull_y,
                            fill='toself',
                            fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[i % len(colors)])) + [overlay_opacity])}',
                            line=dict(color=colors[i % len(colors)], width=2),
                            name=f'{self._get_topic_label(cluster_id)} boundary',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    except Exception as e:
                        # ConvexHull can fail for collinear points
                        logger.debug(f"Could not create convex hull for cluster {cluster_id}: {e}")

        # Add scatter points for each cluster
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = plot_df['Cluster_ID'] == cluster_id
            cluster_data = plot_df[cluster_mask]

            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                marker=dict(
                    size=8,
                    opacity=0.7,
                    color=colors[i % len(colors)]
                ),
                name=self._get_topic_label(cluster_id),
                text=cluster_data['Text'],
                hovertemplate='<b>%{text}</b><extra></extra>'
            ))

        fig.update_layout(
            title=f'Document Clusters ({reduction_method.upper()})',
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            legend_title='Topic',
            hovermode='closest'
        )

        return fig

    def create_silhouette_plot(self) -> Optional[Any]:
        """
        Create silhouette plot showing cluster cohesion.

        ⚠️  METHOD RESTRICTION: ONLY available for TF-IDF + KMeans.
            Not applicable to NMF or LDA (topic models use soft assignments).

        Returns:
            Plotly Figure or None if not available/compatible

        Raises:
            None - returns None with warning if method is incompatible
        """
        if not PLOTLY_AVAILABLE or not SKLEARN_AVAILABLE:
            return None

        # Method compatibility check
        if self.method != 'tfidf_kmeans':
            logger.warning(
                f"⚠️  Silhouette plot is ONLY available for TF-IDF + KMeans method. "
                f"Current method '{self.method}' uses soft topic assignments. "
                f"Consider using topic_term_heatmap or topic_distribution instead."
            )
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
                name=self._get_topic_label(i),
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
        normalize: bool = True,
        show_cluster_overlay: bool = True,
        overlay_opacity: float = 0.3
    ) -> Optional[Any]:
        """
        Create heatmap showing top terms per topic.

        Best for: NMF, LDA (shows topic composition)
        Also works for: KMeans (shows cluster centroids)

        Args:
            n_terms: Number of top terms to show per topic
            normalize: Normalize weights to 0-1 range
            show_cluster_overlay: Whether to show color-coded row boundary overlays
            overlay_opacity: Opacity of the row boundary overlays (0.0-1.0)

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

        # Create heatmap using topic labels from codebook
        topic_labels = [self._get_topic_label(i) for i in range(n_topics)]

        fig = go.Figure()

        # Add the main heatmap
        fig.add_trace(go.Heatmap(
            z=matrix,
            x=terms,
            y=topic_labels,
            colorscale='Viridis',
            colorbar=dict(title='Weight'),
            hovertemplate='Topic: %{y}<br>Term: %{x}<br>Weight: %{z:.3f}<extra></extra>'
        ))

        # Add color-coded row boundary overlays
        if show_cluster_overlay:
            colors = px.colors.qualitative.Plotly
            if n_topics > len(colors):
                colors = colors * (n_topics // len(colors) + 1)

            n_terms_total = len(terms)

            for topic_idx in range(n_topics):
                # Create a rectangle shape for each topic row
                # y coordinates: topic rows are at 0, 1, 2, ... so boundaries are at -0.5, 0.5, 1.5, etc.
                y0 = topic_idx - 0.5
                y1 = topic_idx + 0.5

                # Add left edge colored bar as cluster indicator
                fig.add_shape(
                    type='rect',
                    x0=-0.5,
                    x1=n_terms_total - 0.5,
                    y0=y0,
                    y1=y1,
                    line=dict(color=colors[topic_idx % len(colors)], width=3),
                    fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[topic_idx % len(colors)])) + [overlay_opacity])}',
                    layer='below'
                )

                # Add a colored indicator bar on the left side
                fig.add_annotation(
                    x=-1.5,
                    y=topic_idx,
                    text='▌',
                    font=dict(color=colors[topic_idx % len(colors)], size=20),
                    showarrow=False,
                    xanchor='right'
                )

        fig.update_layout(
            title=f'Topic-Term Heatmap (Top {n_terms} Terms)',
            xaxis_title='Terms',
            yaxis_title='Topics',
            height=max(400, n_topics * 50),
            xaxis=dict(tickangle=45)
        )

        return fig

    def create_topic_distribution_chart(self) -> Optional[Any]:
        """
        Create stacked bar chart showing topic distribution per document.

        ⚠️  METHOD RESTRICTION: ONLY available for NMF and LDA methods.
            Not applicable to KMeans (uses hard cluster assignments, not topic mixtures).

        Returns:
            Plotly Figure or None if not available/compatible

        Raises:
            None - returns None with warning if method is incompatible
        """
        if not PLOTLY_AVAILABLE:
            return None

        # Method compatibility check
        if self.method == 'tfidf_kmeans':
            logger.warning(
                f"⚠️  Topic distribution chart is ONLY available for NMF/LDA methods. "
                f"KMeans uses hard cluster assignments (each doc belongs to exactly one cluster). "
                f"Consider using cluster_scatter or silhouette_plot instead."
            )
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
                name=self._get_topic_label(topic_idx),
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
        # Use to_array() for numpy 2.0+ compatibility
        ax.imshow(wordcloud.to_array(), interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{self._get_topic_label(cluster_id)} Word Cloud')

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

                # Use to_array() for numpy 2.0+ compatibility
                ax.imshow(wordcloud.to_array(), interpolation='bilinear')
            except ValueError:
                ax.text(0.5, 0.5, 'Insufficient text', ha='center', va='center')

            ax.axis('off')
            ax.set_title(f'{self._get_topic_label(cluster_id)} ({sum(mask)} docs)')

        # Hide unused axes
        for idx in range(n_clusters, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

    def create_semantic_wordcloud(
        self,
        cluster_id: int,
        max_words: int = 50,
        width: int = 800,
        height: int = 400,
        colormap_name: Optional[str] = None
    ) -> Optional[Any]:
        """
        Create word cloud for a specific cluster/topic with semantic coloring.

        Words are sized by frequency and colored by semantic similarity.
        Similar colors indicate similar word meanings (based on word embeddings).

        Args:
            cluster_id: Cluster/topic index (0-based)
            max_words: Maximum words in cloud
            width: Image width
            height: Image height
            colormap_name: Optional colormap override. If None, auto-selects based on cluster_id

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

        # Get word frequencies
        words = cleaned_text.split()
        word_freq = Counter(words)

        # Remove common stopwords
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'and', 'but', 'or', 'for',
            'of', 'in', 'to', 'on', 'at', 'by', 'with', 'from', 'as', 'into',
            'it', 'its', 'this', 'that', 'these', 'those', 'what', 'which',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'they', 'them', 'their',
            've', 'll', 're', 't', 's', 'd', 'm', 'not', 'just', 'very', 'really',
            'about', 'get', 'got', 'so', 'too', 'also', 'been', 'being', 'if',
            'no', 'more', 'most', 'other', 'some', 'such', 'any', 'each', 'few',
            'all', 'both', 'only', 'own', 'same', 'than', 'then', 'now', 'here',
            'there', 'when', 'where', 'why', 'how', 'who', 'whom', 'which', 'am'
        }
        word_freq = {w: f for w, f in word_freq.items()
                     if w not in stopwords and len(w) > 2}

        if not word_freq:
            logger.warning(f"No valid words in cluster {cluster_id} after filtering")
            return None

        # Get top words by frequency
        top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_words])

        # Compute semantic colors
        word_colors = self._compute_semantic_colors(
            list(top_words.keys()),
            cluster_texts,
            cluster_id,
            colormap_name
        )

        # Create color function for WordCloud
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return word_colors.get(word.lower(), 'gray')

        # Generate word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color='white',
            max_words=max_words,
            min_font_size=10,
            max_font_size=100,
            color_func=color_func,
            prefer_horizontal=0.7
        ).generate_from_frequencies(top_words)

        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.imshow(wordcloud.to_array(), interpolation='bilinear')
        ax.axis('off')

        # Get cluster label if available
        code_id = f"CODE_{cluster_id + 1:02d}"
        if code_id in self.codebook:
            label = self.codebook[code_id].get('label', f'Cluster {cluster_id + 1}')
        else:
            label = f'Cluster {cluster_id + 1}'

        ax.set_title(f'{label}\n(colors show semantic similarity)', fontsize=12)

        plt.tight_layout()
        return fig

    def _compute_semantic_colors(
        self,
        words: List[str],
        texts: List[str],
        cluster_id: int,
        colormap_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Compute colors for words based on semantic similarity.

        Uses word embeddings to place semantically similar words
        at similar positions on a color gradient.

        Args:
            words: List of words to color
            texts: Cluster texts for training word embeddings
            cluster_id: Cluster ID (used to select colormap)
            colormap_name: Optional colormap override

        Returns:
            Dictionary mapping words to RGB color strings
        """
        import matplotlib.colors as mcolors

        # Define distinct colormaps for each cluster
        cluster_colormaps = [
            'Blues',      # Cluster 0 - Blue shades
            'Oranges',    # Cluster 1 - Orange shades
            'Greens',     # Cluster 2 - Green shades
            'Purples',    # Cluster 3 - Purple shades
            'Reds',       # Cluster 4 - Red shades
            'YlOrBr',     # Cluster 5 - Yellow-Orange-Brown
            'BuGn',       # Cluster 6 - Blue-Green
            'RdPu',       # Cluster 7 - Red-Purple
            'YlGn',       # Cluster 8 - Yellow-Green
            'OrRd',       # Cluster 9 - Orange-Red
            'PuBu',       # Cluster 10 - Purple-Blue
            'GnBu',       # Cluster 11 - Green-Blue
            'BuPu',       # Cluster 12 - Blue-Purple
            'PuRd',       # Cluster 13 - Purple-Red
            'YlGnBu',     # Cluster 14 - Yellow-Green-Blue
        ]

        # Select colormap
        if colormap_name:
            cmap = plt.cm.get_cmap(colormap_name)
        else:
            cmap_idx = cluster_id % len(cluster_colormaps)
            cmap = plt.cm.get_cmap(cluster_colormaps[cmap_idx])

        if len(words) <= 1:
            # Only one word - use middle color
            color = cmap(0.6)
            rgb = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            return {words[0]: rgb} if words else {}

        # Try to compute semantic embeddings
        word_positions = None

        if GENSIM_AVAILABLE:
            try:
                word_positions = self._get_semantic_positions(words, texts)
            except Exception as e:
                logger.debug(f"Semantic embedding failed: {e}, using fallback")

        # Fallback: use lexicographic ordering (words starting with similar letters get similar colors)
        if word_positions is None:
            # Sort words alphabetically and assign positions
            sorted_words = sorted(words)
            word_positions = {w: i / max(1, len(sorted_words) - 1) for i, w in enumerate(sorted_words)}

        # Map positions to colors (use range 0.2-0.9 for better visibility)
        word_colors = {}
        for word, pos in word_positions.items():
            # Scale position to colormap range (avoid very light colors)
            color_pos = 0.25 + pos * 0.65
            color = cmap(color_pos)
            rgb = f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
            word_colors[word] = rgb

        return word_colors

    def _get_semantic_positions(
        self,
        words: List[str],
        texts: List[str]
    ) -> Dict[str, float]:
        """
        Get 1D positions for words based on semantic similarity using embeddings.

        Uses Word2Vec trained on cluster texts to create embeddings,
        then reduces to 1D using PCA for color mapping.

        Args:
            words: List of words to position
            texts: Texts to train word embeddings on

        Returns:
            Dictionary mapping words to positions in [0, 1] range
        """
        if not GENSIM_AVAILABLE:
            return None

        # Tokenize all texts
        tokenized = [simple_preprocess(str(text)) for text in texts]

        # Flatten to get all tokens
        all_tokens = [token for tokens in tokenized for token in tokens]

        # Only proceed if we have enough tokens
        if len(all_tokens) < 20:
            return None

        # Train a small Word2Vec model on the cluster texts
        try:
            model = Word2Vec(
                sentences=tokenized,
                vector_size=50,  # Small dimension for speed
                window=5,
                min_count=1,  # Include all words
                workers=2,
                epochs=20,  # More epochs for small corpus
                seed=42
            )
        except Exception as e:
            logger.debug(f"Word2Vec training failed: {e}")
            return None

        # Get embeddings for words that are in vocabulary
        word_vectors = []
        valid_words = []

        for word in words:
            if word in model.wv:
                word_vectors.append(model.wv[word])
                valid_words.append(word)

        if len(valid_words) < 2:
            return None

        word_vectors = np.array(word_vectors)

        # Reduce to 1D using PCA
        if SKLEARN_AVAILABLE and len(valid_words) >= 2:
            try:
                pca = PCA(n_components=1, random_state=42)
                positions_1d = pca.fit_transform(word_vectors).flatten()

                # Normalize to [0, 1]
                min_pos = positions_1d.min()
                max_pos = positions_1d.max()
                if max_pos > min_pos:
                    positions_1d = (positions_1d - min_pos) / (max_pos - min_pos)
                else:
                    positions_1d = np.full_like(positions_1d, 0.5)

                return {word: pos for word, pos in zip(valid_words, positions_1d)}
            except Exception as e:
                logger.debug(f"PCA failed: {e}")
                return None

        return None

    def create_all_semantic_wordclouds(
        self,
        max_words: int = 40,
        cols: int = 3
    ) -> Optional[Any]:
        """
        Create grid of semantic word clouds for all clusters/topics.

        Each wordcloud has a unique color scheme, with word colors
        representing semantic similarity within that cluster.

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

        # Larger figure for semantic wordclouds
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
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

            # Combine and clean text
            combined_text = ' '.join(str(t) for t in cluster_texts)
            cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', combined_text.lower())

            # Get word frequencies
            words = cleaned_text.split()
            word_freq = Counter(words)

            # Remove common stopwords
            stopwords = {
                'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'and', 'but', 'or', 'for',
                'of', 'in', 'to', 'on', 'at', 'by', 'with', 'from', 'as', 'into',
                'it', 'its', 'this', 'that', 'these', 'those', 'what', 'which',
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'they', 'them', 'their',
                've', 'll', 're', 't', 's', 'd', 'm', 'not', 'just', 'very', 'really',
                'about', 'get', 'got', 'so', 'too', 'also', 'been', 'being', 'if',
                'no', 'more', 'most', 'other', 'some', 'such', 'any', 'each', 'few',
                'all', 'both', 'only', 'own', 'same', 'than', 'then', 'now', 'here',
                'there', 'when', 'where', 'why', 'how', 'who', 'whom', 'which', 'am'
            }
            word_freq = {w: f for w, f in word_freq.items()
                         if w not in stopwords and len(w) > 2}

            if not word_freq:
                ax.text(0.5, 0.5, 'Insufficient text', ha='center', va='center')
                ax.axis('off')
                continue

            # Get top words by frequency
            top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_words])

            # Compute semantic colors
            word_colors = self._compute_semantic_colors(
                list(top_words.keys()),
                cluster_texts,
                cluster_id
            )

            # Create color function for WordCloud
            def make_color_func(colors):
                def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                    return colors.get(word.lower(), 'gray')
                return color_func

            try:
                wordcloud = WordCloud(
                    width=500,
                    height=300,
                    background_color='white',
                    max_words=max_words,
                    min_font_size=8,
                    max_font_size=80,
                    color_func=make_color_func(word_colors),
                    prefer_horizontal=0.7
                ).generate_from_frequencies(top_words)

                ax.imshow(wordcloud.to_array(), interpolation='bilinear')
            except ValueError as e:
                ax.text(0.5, 0.5, 'Insufficient text', ha='center', va='center')

            ax.axis('off')

            # Get cluster label if available
            code_id = f"CODE_{cluster_id + 1:02d}"
            if code_id in self.codebook:
                label = self.codebook[code_id].get('label', f'Cluster {cluster_id + 1}')
            else:
                label = f'Cluster {cluster_id + 1}'

            ax.set_title(f'{label}\n({sum(mask)} docs)', fontsize=10)

        # Hide unused axes
        for idx in range(n_clusters, len(axes)):
            axes[idx].axis('off')

        # Add overall title
        fig.suptitle(
            'Semantic Word Clouds by Topic\n'
            '(Word size = frequency, Color shade = semantic similarity)',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )

        plt.tight_layout()
        return fig

    def create_lda_visualization(self) -> Optional[str]:
        """
        Create pyLDAvis visualization for LDA model.

        ⚠️  METHOD RESTRICTION: ONLY available for LDA method.
            Not applicable to KMeans or NMF. Requires pyLDAvis package.

        Returns:
            HTML string for pyLDAvis or None if not available/compatible

        Raises:
            None - returns None with warning if method is incompatible
        """
        if not PYLDAVIS_AVAILABLE:
            logger.warning(
                "⚠️  pyLDAvis not installed. Install with: pip install pyLDAvis"
            )
            return None

        if self.method != 'lda':
            logger.warning(
                f"⚠️  pyLDAvis is ONLY available for LDA method. "
                f"Current method '{self.method}' is not compatible. "
                f"pyLDAvis requires LDA's probabilistic topic model structure."
            )
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

        Returns a detailed breakdown of which visualizations are:
        - Available and recommended for the current method
        - Available but not optimal for the current method
        - Not available for the current method

        Returns:
            Dictionary with recommended visualizations and their availability
        """
        recommendations = {
            'method': self.method,
            'method_description': {
                'tfidf_kmeans': 'TF-IDF + KMeans (hard clustering)',
                'nmf': 'Non-negative Matrix Factorization (soft topic model)',
                'lda': 'Latent Dirichlet Allocation (probabilistic topic model)'
            }.get(self.method, self.method),
            'visualizations': {},
            'not_available': {},
            'notes': []
        }

        if self.method == 'tfidf_kmeans':
            recommendations['visualizations'] = {
                'cluster_scatter': {
                    'available': PLOTLY_AVAILABLE and SKLEARN_AVAILABLE,
                    'priority': 'high',
                    'description': '2D scatter showing cluster separation',
                    'note': 'Recommended - shows how well clusters are separated'
                },
                'silhouette_plot': {
                    'available': PLOTLY_AVAILABLE and SKLEARN_AVAILABLE,
                    'priority': 'high',
                    'description': 'Cluster cohesion validation',
                    'note': 'Recommended - validates cluster quality'
                },
                'topic_term_heatmap': {
                    'available': PLOTLY_AVAILABLE,
                    'priority': 'medium',
                    'description': 'Cluster centroid terms',
                    'note': 'Available - shows top terms per cluster centroid'
                },
                'per_cluster_wordclouds': {
                    'available': WORDCLOUD_AVAILABLE,
                    'priority': 'medium',
                    'description': 'Word clouds for each cluster',
                    'note': 'Available - visual summary of cluster content'
                }
            }
            recommendations['not_available'] = {
                'topic_distribution': 'Not applicable - KMeans uses hard assignments (1 cluster per doc)',
                'pyldavis': 'Not applicable - requires LDA probabilistic model'
            }
            recommendations['notes'] = [
                'KMeans assigns each document to exactly ONE cluster',
                'Use silhouette_plot to validate cluster quality',
                'Use cluster_scatter to visualize cluster separation'
            ]

        elif self.method == 'nmf':
            recommendations['visualizations'] = {
                'topic_term_heatmap': {
                    'available': PLOTLY_AVAILABLE,
                    'priority': 'high',
                    'description': 'Topic-term weight matrix',
                    'note': 'Recommended - shows topic composition'
                },
                'topic_distribution': {
                    'available': PLOTLY_AVAILABLE,
                    'priority': 'high',
                    'description': 'Topic composition per document',
                    'note': 'Recommended - shows document topic mixtures'
                },
                'cluster_scatter': {
                    'available': PLOTLY_AVAILABLE and SKLEARN_AVAILABLE,
                    'priority': 'low',
                    'description': '2D scatter (less meaningful for topics)',
                    'note': 'Available but less informative for topic models'
                },
                'per_cluster_wordclouds': {
                    'available': WORDCLOUD_AVAILABLE,
                    'priority': 'medium',
                    'description': 'Word clouds for each topic',
                    'note': 'Available - visual summary of topic content'
                }
            }
            recommendations['not_available'] = {
                'silhouette_plot': 'Not applicable - NMF uses soft topic assignments',
                'pyldavis': 'Not applicable - requires LDA model specifically'
            }
            recommendations['notes'] = [
                'NMF produces sparse, interpretable topic weights',
                'Documents can belong to multiple topics with different weights',
                'Use topic_term_heatmap to understand topic composition'
            ]

        elif self.method == 'lda':
            recommendations['visualizations'] = {
                'pyldavis': {
                    'available': PYLDAVIS_AVAILABLE,
                    'priority': 'high',
                    'description': 'Interactive LDA exploration',
                    'note': 'Highly recommended - industry standard for LDA'
                },
                'topic_term_heatmap': {
                    'available': PLOTLY_AVAILABLE,
                    'priority': 'high',
                    'description': 'Topic-term probability matrix',
                    'note': 'Recommended - shows word distributions per topic'
                },
                'topic_distribution': {
                    'available': PLOTLY_AVAILABLE,
                    'priority': 'high',
                    'description': 'Topic probability per document',
                    'note': 'Recommended - shows document topic mixtures'
                },
                'cluster_scatter': {
                    'available': PLOTLY_AVAILABLE and SKLEARN_AVAILABLE,
                    'priority': 'low',
                    'description': '2D scatter (less meaningful for topics)',
                    'note': 'Available but less informative for topic models'
                },
                'per_cluster_wordclouds': {
                    'available': WORDCLOUD_AVAILABLE,
                    'priority': 'medium',
                    'description': 'Word clouds for each topic',
                    'note': 'Available - visual summary of topic content'
                }
            }
            recommendations['not_available'] = {
                'silhouette_plot': 'Not applicable - LDA uses probabilistic topic assignments'
            }
            recommendations['notes'] = [
                'LDA models documents as probability distributions over topics',
                'Each topic is a probability distribution over words',
                'pyLDAvis provides the best interactive exploration (install with: pip install pyLDAvis)'
            ]

        return recommendations


def create_method_visualizations(
    coder,
    results_df: pd.DataFrame,
    text_column: str,
    include_semantic_wordclouds: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to create all appropriate visualizations.

    Args:
        coder: Fitted MLOpenCoder instance
        results_df: Results DataFrame
        text_column: Name of text column
        include_semantic_wordclouds: Whether to include semantic wordclouds (default True)

    Returns:
        Dictionary of visualization name -> figure/html
    """
    viz = MethodVisualizer(coder, results_df, text_column)
    method = viz.method

    figures = {}

    # Common visualizations
    if WORDCLOUD_AVAILABLE:
        figures['cluster_wordclouds'] = viz.create_all_cluster_wordclouds()

        # Add semantic wordclouds (color-coded by word meaning)
        if include_semantic_wordclouds:
            figures['semantic_wordclouds'] = viz.create_all_semantic_wordclouds()

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
