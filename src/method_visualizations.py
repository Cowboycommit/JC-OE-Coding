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

# Separate imports for wordcloud and matplotlib for better fallback handling
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None
    logger.warning("WordCloud not available. Please ensure the wordcloud package is installed.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    logger.warning("matplotlib not available. Will use PIL for wordcloud images.")

# PIL is bundled with wordcloud, use as fallback for image handling
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

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
# PIL-BASED WORD CLOUD GENERATOR (Fallback when wordcloud package unavailable)
# =============================================================================

class PILWordCloud:
    """
    Pure PIL-based word cloud generator.

    Used as a fallback when the 'wordcloud' package is not installed.
    Creates word clouds using PIL's ImageDraw for text rendering.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 400,
        background_color: str = 'white',
        max_words: int = 100,
        min_font_size: int = 10,
        max_font_size: int = 100,
        color_func: Optional[callable] = None,
        colormap: Optional[str] = None
    ):
        """
        Initialize PIL word cloud generator.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            background_color: Background color (name or hex)
            max_words: Maximum number of words to display
            min_font_size: Minimum font size in points
            max_font_size: Maximum font size in points
            color_func: Optional function to determine word color
            colormap: Colormap name for coloring words (viridis, etc.)
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.max_words = max_words
        self.min_font_size = min_font_size
        self.max_font_size = max_font_size
        self.color_func = color_func
        self.colormap = colormap or 'viridis'
        self._layout = []  # Stores (word, x, y, font_size, color)

    def generate(self, text: str) -> 'PILWordCloud':
        """
        Generate word cloud from text.

        Args:
            text: Input text to create word cloud from

        Returns:
            Self for chaining
        """
        from collections import Counter
        import re

        # Tokenize and count words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = Counter(words)

        # Remove common stopwords
        stopwords = {
            'the', 'and', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'but', 'for', 'with',
            'from', 'into', 'its', 'this', 'that', 'these', 'those', 'what',
            'which', 'who', 'whom', 'you', 'your', 'they', 'them', 'their',
            'not', 'just', 'very', 'really', 'about', 'get', 'got', 'also',
            'more', 'most', 'other', 'some', 'such', 'any', 'each', 'few',
            'all', 'both', 'only', 'own', 'same', 'than', 'then', 'now',
            'here', 'there', 'when', 'where', 'why', 'how'
        }
        word_freq = {w: f for w, f in word_freq.items() if w not in stopwords}

        return self.generate_from_frequencies(word_freq)

    def generate_from_frequencies(self, frequencies: Dict[str, int]) -> 'PILWordCloud':
        """
        Generate word cloud from word frequencies.

        Args:
            frequencies: Dictionary of word -> frequency

        Returns:
            Self for chaining
        """
        if not frequencies:
            self._layout = []
            return self

        # Sort by frequency and limit words
        sorted_words = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_words]

        if not sorted_words:
            self._layout = []
            return self

        # Calculate font sizes proportional to frequency
        max_freq = sorted_words[0][1]
        min_freq = sorted_words[-1][1] if len(sorted_words) > 1 else max_freq
        freq_range = max(max_freq - min_freq, 1)

        # Generate colors
        colors = self._generate_colors(len(sorted_words))

        # Calculate layout
        self._layout = []
        occupied = []  # List of bounding boxes (x1, y1, x2, y2)

        # Create a temporary image to measure text sizes
        from PIL import ImageDraw, ImageFont
        temp_img = Image.new('RGB', (self.width, self.height))
        temp_draw = ImageDraw.Draw(temp_img)

        # Try to load a font, fall back to default if not available
        font_cache = {}

        def get_font(size):
            if size not in font_cache:
                try:
                    # Try common fonts
                    for font_name in ['DejaVuSans.ttf', 'Arial.ttf', 'FreeSans.ttf', 'LiberationSans-Regular.ttf']:
                        try:
                            font_cache[size] = ImageFont.truetype(font_name, size)
                            break
                        except (OSError, IOError):
                            continue
                    else:
                        # Fall back to default font
                        font_cache[size] = ImageFont.load_default()
                except Exception:
                    font_cache[size] = ImageFont.load_default()
            return font_cache[size]

        # Place each word
        import random
        random.seed(42)  # Reproducible layout

        for idx, (word, freq) in enumerate(sorted_words):
            # Calculate font size
            if freq_range > 0:
                size_ratio = (freq - min_freq) / freq_range
            else:
                size_ratio = 1.0
            font_size = int(self.min_font_size + size_ratio * (self.max_font_size - self.min_font_size))
            font_size = max(self.min_font_size, min(self.max_font_size, font_size))

            font = get_font(font_size)

            # Get text bounding box
            try:
                bbox = temp_draw.textbbox((0, 0), word, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                # Older PIL versions
                text_width, text_height = temp_draw.textsize(word, font=font)

            # Add some padding
            text_width += 10
            text_height += 5

            # Get color for this word
            if self.color_func:
                color = self.color_func(word, font_size, None, None)
            else:
                color = colors[idx]

            # Try to find a position that doesn't overlap
            placed = False
            max_attempts = 100

            for attempt in range(max_attempts):
                # Generate random position
                x = random.randint(5, max(6, self.width - text_width - 5))
                y = random.randint(5, max(6, self.height - text_height - 5))

                # Check for overlaps
                new_box = (x, y, x + text_width, y + text_height)
                overlaps = False

                for ox1, oy1, ox2, oy2 in occupied:
                    if not (new_box[2] < ox1 or new_box[0] > ox2 or
                            new_box[3] < oy1 or new_box[1] > oy2):
                        overlaps = True
                        break

                if not overlaps:
                    self._layout.append((word, x, y, font_size, color))
                    occupied.append(new_box)
                    placed = True
                    break

            # If we couldn't place it after max attempts, skip it
            if not placed and len(occupied) < 10:
                # Force place first few words
                x = random.randint(5, max(6, self.width - text_width - 5))
                y = random.randint(5, max(6, self.height - text_height - 5))
                self._layout.append((word, x, y, font_size, color))
                occupied.append((x, y, x + text_width, y + text_height))

        return self

    def _generate_colors(self, n_colors: int) -> List[str]:
        """Generate colors based on colormap."""
        # Viridis-like color palette (for when matplotlib is not available)
        viridis_colors = [
            '#440154', '#481567', '#482677', '#453781', '#404788',
            '#39568c', '#33638d', '#2d708e', '#287d8e', '#238a8d',
            '#1f968b', '#20a387', '#29af7f', '#3cbb75', '#55c667',
            '#73d055', '#95d840', '#b8de29', '#dce319', '#fde725'
        ]

        # Generate evenly spaced colors from the palette
        colors = []
        for i in range(n_colors):
            idx = int(i * (len(viridis_colors) - 1) / max(n_colors - 1, 1))
            colors.append(viridis_colors[idx])

        return colors

    def to_image(self) -> 'Image.Image':
        """
        Render the word cloud to a PIL Image.

        Returns:
            PIL Image object
        """
        from PIL import ImageDraw, ImageFont

        # Create image
        img = Image.new('RGB', (self.width, self.height), self.background_color)
        draw = ImageDraw.Draw(img)

        # Font cache
        font_cache = {}

        def get_font(size):
            if size not in font_cache:
                try:
                    for font_name in ['DejaVuSans.ttf', 'Arial.ttf', 'FreeSans.ttf', 'LiberationSans-Regular.ttf']:
                        try:
                            font_cache[size] = ImageFont.truetype(font_name, size)
                            break
                        except (OSError, IOError):
                            continue
                    else:
                        font_cache[size] = ImageFont.load_default()
                except Exception:
                    font_cache[size] = ImageFont.load_default()
            return font_cache[size]

        # Draw each word
        for word, x, y, font_size, color in self._layout:
            font = get_font(font_size)
            draw.text((x, y), word, font=font, fill=color)

        return img

    def to_array(self) -> np.ndarray:
        """
        Convert word cloud to numpy array.

        Returns:
            Numpy array of the image
        """
        return np.array(self.to_image())


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
    },
    'cluster_network': {
        'compatible': ['tfidf_kmeans', 'nmf', 'lda'],
        'recommended': ['tfidf_kmeans', 'nmf', 'lda'],
        'description': 'Network diagram showing cluster relationships',
        'note': 'Nodes represent clusters, edges show inter-cluster similarity.'
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
        self.results_df = results_df.reset_index(drop=True)  # Ensure clean index for mask operations
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
                for codes in self.results_df['assigned_codes']
            ])

        # Validate assignments length matches DataFrame
        if len(self.assignments) != len(self.results_df):
            logger.warning(
                f"Assignments length ({len(self.assignments)}) != DataFrame length ({len(self.results_df)}). "
                "Using fallback from assigned_codes."
            )
            self.assignments = np.array([
                int(codes[0].split('_')[1]) - 1 if codes else -1
                for codes in self.results_df['assigned_codes']
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

    def create_cluster_network(
        self,
        similarity_threshold: float = 0.1,
        layout: str = 'spring',
        show_edge_labels: bool = False
    ) -> Optional[Any]:
        """
        Create network diagram showing cluster/topic relationships.

        Nodes represent clusters, sized by document count.
        Edges represent inter-cluster similarity based on centroid cosine similarity.

        Best for: All methods (KMeans, NMF, LDA)

        Args:
            similarity_threshold: Minimum similarity to show an edge (0.0-1.0)
            layout: Graph layout algorithm ('spring', 'circular', 'kamada_kawai')
            show_edge_labels: Whether to show similarity values on edges

        Returns:
            Plotly Figure or None if not available
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available for network diagram")
            return None

        # Get dense matrix
        if hasattr(self.feature_matrix, 'toarray'):
            dense_matrix = self.feature_matrix.toarray()
        else:
            dense_matrix = np.array(self.feature_matrix)

        # Get unique clusters
        unique_clusters = sorted(set(self.assignments))
        n_clusters = len(unique_clusters)

        if n_clusters < 2:
            logger.warning("Need at least 2 clusters for network diagram")
            return None

        # Calculate cluster centroids
        centroids = []
        cluster_sizes = []
        cluster_labels = []

        for cluster_id in unique_clusters:
            mask = self.assignments == cluster_id
            cluster_docs = dense_matrix[mask]
            centroid = cluster_docs.mean(axis=0)
            centroids.append(centroid)
            cluster_sizes.append(mask.sum())
            cluster_labels.append(self._get_topic_label(cluster_id))

        centroids = np.array(centroids)

        # Calculate pairwise similarities using cosine similarity
        # Cosine similarity = 1 - cosine distance
        if SCIPY_AVAILABLE:
            # Use scipy for efficient computation
            distances = pdist(centroids, metric='cosine')
            similarity_matrix = 1 - squareform(distances)
        else:
            # Manual computation
            similarity_matrix = np.zeros((n_clusters, n_clusters))
            for i in range(n_clusters):
                for j in range(n_clusters):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        norm_i = np.linalg.norm(centroids[i])
                        norm_j = np.linalg.norm(centroids[j])
                        if norm_i > 0 and norm_j > 0:
                            similarity_matrix[i, j] = np.dot(centroids[i], centroids[j]) / (norm_i * norm_j)

        # Create edges based on similarity threshold
        edges = []
        edge_weights = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                sim = similarity_matrix[i, j]
                if sim >= similarity_threshold:
                    edges.append((i, j))
                    edge_weights.append(sim)

        # Calculate node positions using simple layout algorithms
        if layout == 'circular':
            # Circular layout
            angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
            node_x = np.cos(angles)
            node_y = np.sin(angles)
        elif layout == 'kamada_kawai' and SCIPY_AVAILABLE:
            # Use spring layout with scipy optimization
            # Start with circular, then apply force-directed adjustment
            angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
            node_x = np.cos(angles).copy()
            node_y = np.sin(angles).copy()
            # Simple force-directed adjustment
            for _ in range(50):
                for i in range(n_clusters):
                    fx, fy = 0, 0
                    for j in range(n_clusters):
                        if i != j:
                            dx = node_x[i] - node_x[j]
                            dy = node_y[i] - node_y[j]
                            dist = max(np.sqrt(dx**2 + dy**2), 0.01)
                            # Repulsion
                            fx += dx / (dist ** 2) * 0.1
                            fy += dy / (dist ** 2) * 0.1
                            # Attraction for connected nodes
                            if (i, j) in edges or (j, i) in edges:
                                fx -= dx * similarity_matrix[i, j] * 0.05
                                fy -= dy * similarity_matrix[i, j] * 0.05
                    node_x[i] += fx * 0.1
                    node_y[i] += fy * 0.1
        else:  # spring layout (default)
            # Force-directed spring layout
            np.random.seed(42)
            node_x = np.random.rand(n_clusters) * 2 - 1
            node_y = np.random.rand(n_clusters) * 2 - 1

            for _ in range(100):
                for i in range(n_clusters):
                    fx, fy = 0, 0
                    for j in range(n_clusters):
                        if i != j:
                            dx = node_x[i] - node_x[j]
                            dy = node_y[i] - node_y[j]
                            dist = max(np.sqrt(dx**2 + dy**2), 0.01)
                            # Repulsion (all nodes repel each other)
                            repulsion = 0.5 / (dist ** 2)
                            fx += dx / dist * repulsion
                            fy += dy / dist * repulsion
                            # Attraction (connected nodes attract)
                            sim = similarity_matrix[i, j]
                            if sim >= similarity_threshold:
                                attraction = sim * dist * 0.5
                                fx -= dx / dist * attraction
                                fy -= dy / dist * attraction
                    node_x[i] += fx * 0.05
                    node_y[i] += fy * 0.05

        # Create Plotly figure
        fig = go.Figure()

        # Use color palette
        colors = px.colors.qualitative.Plotly
        if n_clusters > len(colors):
            colors = colors * (n_clusters // len(colors) + 1)

        # Add edges
        for idx, (i, j) in enumerate(edges):
            weight = edge_weights[idx]
            # Line width proportional to similarity
            line_width = 1 + weight * 5

            fig.add_trace(go.Scatter(
                x=[node_x[i], node_x[j], None],
                y=[node_y[i], node_y[j], None],
                mode='lines',
                line=dict(width=line_width, color='rgba(150, 150, 150, 0.6)'),
                hoverinfo='text',
                hovertext=f"Similarity: {weight:.2f}",
                showlegend=False
            ))

            # Add edge label if requested
            if show_edge_labels:
                mid_x = (node_x[i] + node_x[j]) / 2
                mid_y = (node_y[i] + node_y[j]) / 2
                fig.add_annotation(
                    x=mid_x, y=mid_y,
                    text=f"{weight:.2f}",
                    showarrow=False,
                    font=dict(size=9, color='gray')
                )

        # Add nodes
        # Scale node sizes based on document counts
        max_size = max(cluster_sizes)
        min_size = min(cluster_sizes)
        size_range = max_size - min_size if max_size > min_size else 1
        node_sizes = [20 + (s - min_size) / size_range * 40 for s in cluster_sizes]

        for i, (x, y, label, size, count) in enumerate(zip(
            node_x, node_y, cluster_labels, node_sizes, cluster_sizes
        )):
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=size,
                    color=colors[i % len(colors)],
                    line=dict(width=2, color='white')
                ),
                text=[label],
                textposition='top center',
                textfont=dict(size=10),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Documents: {count}<br>"
                    f"<extra></extra>"
                ),
                name=label,
                showlegend=True
            ))

        fig.update_layout(
            title='Cluster Network Diagram',
            showlegend=True,
            legend_title='Clusters',
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=''),
            plot_bgcolor='white'
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

        # Filter out terms (columns) that have zero co-occurrence with all topics
        # Keep all topics but drop terms where max value across all topics is 0
        term_max_values = matrix.max(axis=0)
        non_zero_term_mask = term_max_values > 0
        matrix = matrix[:, non_zero_term_mask]
        terms = [t for t, keep in zip(terms, non_zero_term_mask) if keep]

        # Normalize if requested
        if normalize and matrix.max() > 0:
            matrix = matrix / matrix.max()

        # Create heatmap using topic labels from codebook
        topic_labels = [self._get_topic_label(i) for i in range(n_topics)]

        fig = go.Figure()

        # Add the main heatmap - only show cells with score > 0
        # Create a masked version where zeros become None for display
        display_matrix = np.where(matrix > 0, matrix, np.nan)

        fig.add_trace(go.Heatmap(
            z=display_matrix,
            x=terms,
            y=topic_labels,
            colorscale='Viridis',
            colorbar=dict(title='Weight'),
            hovertemplate='Topic: %{y}<br>Term: %{x}<br>Weight: %{z:.3f}<extra></extra>',
            zmin=0
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
            Matplotlib Figure, PIL Image, or None if not available
        """
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

        # Use wordcloud package if available, otherwise fall back to PIL
        try:
            if WORDCLOUD_AVAILABLE:
                wordcloud = WordCloud(
                    width=width,
                    height=height,
                    background_color='white',
                    colormap='viridis',
                    max_words=max_words,
                    min_font_size=10,
                    max_font_size=100
                ).generate(cleaned_text)
            elif PIL_AVAILABLE:
                # Fall back to PIL-based word cloud
                wordcloud = PILWordCloud(
                    width=width,
                    height=height,
                    background_color='white',
                    max_words=max_words,
                    min_font_size=10,
                    max_font_size=100
                ).generate(cleaned_text)
            else:
                logger.warning("Neither wordcloud nor PIL available for word cloud generation")
                return None
        except ValueError as e:
            logger.warning(f"Word cloud generation failed for cluster {cluster_id}: {e}")
            return None

        # Use matplotlib if available, otherwise return PIL image directly
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            ax.imshow(wordcloud.to_image(), interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'{self._get_topic_label(cluster_id)} Word Cloud')
            plt.tight_layout()
            return fig
        else:
            # Return PIL image directly when matplotlib is not available
            return wordcloud.to_image()

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
            Matplotlib Figure, PIL Image, or None if not available
        """
        if not WORDCLOUD_AVAILABLE and not PIL_AVAILABLE:
            logger.warning("Neither wordcloud nor PIL available for word cloud generation")
            return None

        n_clusters = len(set(self.assignments))
        rows = (n_clusters + cols - 1) // cols

        import re

        def generate_wordcloud(cleaned_text, cell_width, cell_height):
            """Helper to generate wordcloud using available method."""
            if WORDCLOUD_AVAILABLE:
                return WordCloud(
                    width=cell_width,
                    height=cell_height,
                    background_color='white',
                    colormap='viridis',
                    max_words=max_words
                ).generate(cleaned_text)
            else:
                return PILWordCloud(
                    width=cell_width,
                    height=cell_height,
                    background_color='white',
                    max_words=max_words
                ).generate(cleaned_text)

        # Use matplotlib if available
        if MATPLOTLIB_AVAILABLE:
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
            # Handle axes array: single axis becomes [axis], 1D stays 1D, 2D gets flattened
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]

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
                    wordcloud = generate_wordcloud(cleaned_text, 400, 200)
                    # Use to_image() PIL method for numpy compatibility
                    ax.imshow(wordcloud.to_image(), interpolation='bilinear')
                except ValueError:
                    ax.text(0.5, 0.5, 'Insufficient text', ha='center', va='center')

                ax.axis('off')
                ax.set_title(f'{self._get_topic_label(cluster_id)} ({sum(mask)} docs)')

            # Hide unused axes
            for idx in range(n_clusters, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            return fig
        else:
            # PIL fallback: create a combined image grid
            wordcloud_images = []
            cell_width, cell_height = 400, 200

            for cluster_id in range(n_clusters):
                mask = self.assignments == cluster_id
                cluster_texts = self.results_df.loc[mask, self.text_column].tolist()

                if not cluster_texts:
                    # Create blank image for empty clusters
                    img = Image.new('RGB', (cell_width, cell_height), 'white')
                    wordcloud_images.append(img)
                    continue

                combined_text = ' '.join(str(t) for t in cluster_texts)
                cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', combined_text.lower())

                try:
                    wordcloud = generate_wordcloud(cleaned_text, cell_width, cell_height)
                    wordcloud_images.append(wordcloud.to_image())
                except ValueError:
                    img = Image.new('RGB', (cell_width, cell_height), 'white')
                    wordcloud_images.append(img)

            # Combine images into a grid
            grid_width = cols * cell_width
            grid_height = rows * cell_height
            grid_image = Image.new('RGB', (grid_width, grid_height), 'white')

            for idx, img in enumerate(wordcloud_images):
                row_idx = idx // cols
                col_idx = idx % cols
                grid_image.paste(img, (col_idx * cell_width, row_idx * cell_height))

            return grid_image

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
            Matplotlib Figure, PIL Image, or None if not available
        """
        if not WORDCLOUD_AVAILABLE and not PIL_AVAILABLE:
            logger.warning("Neither wordcloud nor PIL available for word cloud generation")
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

        # Compute semantic colors (uses matplotlib.colors internally, falls back gracefully)
        word_colors = self._compute_semantic_colors(
            list(top_words.keys()),
            cluster_texts,
            cluster_id,
            colormap_name
        )

        # Create color function for WordCloud
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return word_colors.get(word.lower(), 'gray')

        # Generate word cloud using available method
        try:
            if WORDCLOUD_AVAILABLE:
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
            else:
                # Use PILWordCloud fallback with color function
                wordcloud = PILWordCloud(
                    width=width,
                    height=height,
                    background_color='white',
                    max_words=max_words,
                    min_font_size=10,
                    max_font_size=100,
                    color_func=color_func
                ).generate_from_frequencies(top_words)
        except ValueError as e:
            logger.warning(f"Semantic word cloud generation failed for cluster {cluster_id}: {e}")
            return None

        # Use matplotlib if available, otherwise return PIL image directly
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            ax.imshow(wordcloud.to_image(), interpolation='bilinear')
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
        else:
            # Return PIL image directly when matplotlib is not available
            return wordcloud.to_image()

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
        # Define fallback color palettes when matplotlib is not available
        fallback_palettes = [
            # Blues (Cluster 0)
            [(198, 219, 239), (107, 174, 214), (33, 113, 181), (8, 69, 148)],
            # Oranges (Cluster 1)
            [(253, 208, 162), (253, 141, 60), (230, 85, 13), (166, 54, 3)],
            # Greens (Cluster 2)
            [(199, 233, 192), (116, 196, 118), (35, 139, 69), (0, 90, 50)],
            # Purples (Cluster 3)
            [(218, 218, 235), (158, 154, 200), (106, 81, 163), (63, 0, 125)],
            # Reds (Cluster 4)
            [(252, 187, 161), (251, 106, 74), (222, 45, 38), (165, 15, 21)],
            # YlOrBr (Cluster 5)
            [(255, 247, 188), (254, 196, 79), (217, 95, 14), (153, 52, 4)],
            # BuGn (Cluster 6)
            [(204, 236, 230), (102, 194, 164), (35, 139, 69), (0, 88, 36)],
            # RdPu (Cluster 7)
            [(253, 224, 221), (250, 159, 181), (197, 27, 138), (122, 1, 119)],
        ]

        if MATPLOTLIB_AVAILABLE:
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

            def get_color(pos):
                color = cmap(pos)
                return f'rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})'
        else:
            # Use fallback palette without matplotlib
            palette_idx = cluster_id % len(fallback_palettes)
            palette = fallback_palettes[palette_idx]

            def get_color(pos):
                # Interpolate within the palette
                idx = min(int(pos * (len(palette) - 1)), len(palette) - 2)
                frac = pos * (len(palette) - 1) - idx
                c1, c2 = palette[idx], palette[min(idx + 1, len(palette) - 1)]
                r = int(c1[0] + frac * (c2[0] - c1[0]))
                g = int(c1[1] + frac * (c2[1] - c1[1]))
                b = int(c1[2] + frac * (c2[2] - c1[2]))
                return f'rgb({r},{g},{b})'

        if len(words) <= 1:
            # Only one word - use middle color
            rgb = get_color(0.6)
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
            rgb = get_color(color_pos)
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
            Matplotlib Figure, PIL Image, or None if not available
        """
        if not WORDCLOUD_AVAILABLE and not PIL_AVAILABLE:
            logger.warning("Neither wordcloud nor PIL available for word cloud generation")
            return None

        n_clusters = len(set(self.assignments))
        rows = (n_clusters + cols - 1) // cols

        import re

        # Helper to generate wordcloud for a cluster
        def generate_cluster_wordcloud(cluster_id, cell_width=500, cell_height=300):
            mask = self.assignments == cluster_id
            cluster_texts = self.results_df.loc[mask, self.text_column].tolist()

            if not cluster_texts:
                return None, 0

            combined_text = ' '.join(str(t) for t in cluster_texts)
            cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', combined_text.lower())

            words = cleaned_text.split()
            word_freq = Counter(words)

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
                return None, sum(mask)

            top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_words])

            word_colors = self._compute_semantic_colors(
                list(top_words.keys()),
                cluster_texts,
                cluster_id
            )

            def make_color_func(colors):
                def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                    return colors.get(word.lower(), 'gray')
                return color_func

            try:
                if WORDCLOUD_AVAILABLE:
                    wordcloud = WordCloud(
                        width=cell_width,
                        height=cell_height,
                        background_color='white',
                        max_words=max_words,
                        min_font_size=8,
                        max_font_size=80,
                        color_func=make_color_func(word_colors),
                        prefer_horizontal=0.7
                    ).generate_from_frequencies(top_words)
                else:
                    wordcloud = PILWordCloud(
                        width=cell_width,
                        height=cell_height,
                        background_color='white',
                        max_words=max_words,
                        min_font_size=8,
                        max_font_size=80,
                        color_func=make_color_func(word_colors)
                    ).generate_from_frequencies(top_words)
                return wordcloud.to_image(), sum(mask)
            except ValueError:
                return None, sum(mask)

        if MATPLOTLIB_AVAILABLE:
            # Larger figure for semantic wordclouds
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
            # Handle axes array: single axis becomes [axis], 1D stays 1D, 2D gets flattened
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]

            for cluster_id in range(n_clusters):
                ax = axes[cluster_id]
                img, doc_count = generate_cluster_wordcloud(cluster_id)

                if img is None:
                    ax.text(0.5, 0.5, 'No documents' if doc_count == 0 else 'Insufficient text',
                            ha='center', va='center')
                else:
                    ax.imshow(img, interpolation='bilinear')

                ax.axis('off')

                code_id = f"CODE_{cluster_id + 1:02d}"
                if code_id in self.codebook:
                    label = self.codebook[code_id].get('label', f'Cluster {cluster_id + 1}')
                else:
                    label = f'Cluster {cluster_id + 1}'

                ax.set_title(f'{label}\n({doc_count} docs)', fontsize=10)

            for idx in range(n_clusters, len(axes)):
                axes[idx].axis('off')

            fig.suptitle(
                'Semantic Word Clouds by Topic\n'
                '(Word size = frequency, Color shade = semantic similarity)',
                fontsize=14,
                fontweight='bold',
                y=1.02
            )

            plt.tight_layout()
            return fig
        else:
            # PIL fallback: create a combined image grid
            cell_width, cell_height = 500, 300
            wordcloud_images = []

            for cluster_id in range(n_clusters):
                img, _ = generate_cluster_wordcloud(cluster_id, cell_width, cell_height)
                if img is None:
                    img = Image.new('RGB', (cell_width, cell_height), 'white')
                wordcloud_images.append(img)

            grid_width = cols * cell_width
            grid_height = rows * cell_height
            grid_image = Image.new('RGB', (grid_width, grid_height), 'white')

            for idx, img in enumerate(wordcloud_images):
                row_idx = idx // cols
                col_idx = idx % cols
                grid_image.paste(img, (col_idx * cell_width, row_idx * cell_height))

            return grid_image

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
