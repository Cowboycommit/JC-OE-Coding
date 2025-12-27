"""
Analysis helper functions for the Streamlit UI.

Provides functions for running ML analysis, generating insights,
and processing results.

Supports multi-granularity text processing (response, sentence, paragraph levels)
via optional integration with TextSegmenter. Response-level analysis remains
the default for backward compatibility.

NEW: Dataset-aware preprocessing for TF-IDF + KMeans clustering.
- Adaptive preprocessing based on detected dataset characteristics
- Mandatory cluster interpretation layer (prevents nonsensical codes)
- Strict separation between clustering (unsupervised) and evaluation (optional)
- Multi-label handling for Reuters-style datasets
- Full backward compatibility with simple OE datasets
"""

from collections import Counter
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.methods_documentation import MethodsDocGenerator, export_methods_to_file

# Import TextSegmenter for optional multi-granularity processing
try:
    from src.text_processing import TextSegmenter
    TEXT_SEGMENTER_AVAILABLE = True
except ImportError:
    TEXT_SEGMENTER_AVAILABLE = False
    logging.warning("TextSegmenter not available. Multi-granularity segmentation disabled.")

# Import dataset-aware preprocessing modules
try:
    from src.dataset_preprocessing import (
        DatasetCharacteristicsDetector,
        AdaptivePreprocessor,
        MultiLabelHandler,
        DatasetCharacteristics,
        PreprocessingConfig,
        get_adaptive_preprocessing
    )
    ADAPTIVE_PREPROCESSING_AVAILABLE = True
except ImportError:
    ADAPTIVE_PREPROCESSING_AVAILABLE = False
    logging.warning("Adaptive preprocessing not available.")

# Import cluster interpretation module
try:
    from src.cluster_interpretation import (
        ClusterInterpreter,
        ClusterInterpretationReport,
        ClusterSummary,
        ClusterCodebook
    )
    CLUSTER_INTERPRETATION_AVAILABLE = True
except ImportError:
    CLUSTER_INTERPRETATION_AVAILABLE = False
    logging.warning("Cluster interpretation not available.")

# Import cluster evaluation module
try:
    from src.cluster_evaluation import (
        ClusterEvaluator,
        EvaluationMetrics,
        evaluate_clusters_posthoc
    )
    CLUSTER_EVALUATION_AVAILABLE = True
except ImportError:
    CLUSTER_EVALUATION_AVAILABLE = False
    logging.warning("Cluster evaluation not available.")

# Import vectorizer factory for method parity
try:
    from src.vectorizer_factory import (
        VectorizerFactory,
        VectorizerConfig,
        create_vectorizer_for_method
    )
    VECTORIZER_FACTORY_AVAILABLE = True
except ImportError:
    VECTORIZER_FACTORY_AVAILABLE = False
    logging.warning("Vectorizer factory not available. Using legacy vectorization.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1
) -> Tuple[bool, str]:
    """
    Validate a DataFrame for analysis.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required

    Returns:
        Tuple of (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"

    if len(df) < min_rows:
        return False, f"DataFrame must have at least {min_rows} rows (found {len(df)})"

    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"

    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        return False, f"Columns with all null values: {', '.join(null_cols)}"

    return True, ""


def preprocess_responses(
    df: pd.DataFrame,
    text_column: str,
    remove_nulls: bool = True,
    remove_duplicates: bool = False,
    min_length: int = 5
) -> pd.DataFrame:
    """
    Preprocess response data for analysis.

    Args:
        df: DataFrame with responses
        text_column: Name of the text column
        remove_nulls: Remove rows with null responses
        remove_duplicates: Remove duplicate responses
        min_length: Minimum response length (characters)

    Returns:
        Preprocessed DataFrame
    """
    processed = df.copy()

    # Remove nulls
    if remove_nulls:
        initial_count = len(processed)
        processed = processed[processed[text_column].notna()]
        removed = initial_count - len(processed)
        if removed > 0:
            logger.info(f"Removed {removed} null responses")

    # Remove short responses
    if min_length > 0:
        initial_count = len(processed)
        processed = processed[processed[text_column].str.len() >= min_length]
        removed = initial_count - len(processed)
        if removed > 0:
            logger.info(f"Removed {removed} responses shorter than {min_length} characters")

    # Remove duplicates
    if remove_duplicates:
        initial_count = len(processed)
        processed = processed.drop_duplicates(subset=[text_column])
        removed = initial_count - len(processed)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate responses")

    return processed


def find_optimal_codes(
    df: pd.DataFrame,
    text_column: str,
    min_codes: int = 3,
    max_codes: int = 15,
    method: str = 'tfidf_kmeans',
    stop_words: str = 'english',
    progress_callback=None
) -> Tuple[int, Dict[str, Any]]:
    """
    Find the optimal number of codes using silhouette analysis.

    Args:
        df: DataFrame with responses
        text_column: Name of the text column
        min_codes: Minimum number of codes to test
        max_codes: Maximum number of codes to test
        method: ML method to use for testing
        stop_words: Stop words language
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (optimal_n_codes, analysis_results)

    Raises:
        ValueError: If dataset is too small for clustering
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    # Validate dataset is not empty
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty. Cannot perform code optimization.")

    responses = df[text_column].tolist()

    # Preprocess text with multilingual support
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Keep letters (including accented chars for Spanish/French/German) and whitespace
        # Pattern: a-z (basic Latin), ß (German eszett), à-ö and ø-ÿ (Latin-1 accented letters)
        # This preserves: é, è, ê, ë, à, â, ç, î, ï, ô, û, ù, ü, ÿ, ñ, á, í, ó, ú, ä, ö, ß, etc.
        text = re.sub(r'[^a-zßà-öø-ÿ\s]', ' ', text)
        text = ' '.join(text.split())
        return text

    processed = [preprocess_text(r) for r in responses]

    # Create feature matrix based on method
    if method == 'lda':
        vectorizer = CountVectorizer(
            max_features=1000, stop_words=stop_words, min_df=2, max_df=0.8
        )
    else:
        vectorizer = TfidfVectorizer(
            max_features=1000, stop_words=stop_words, min_df=2,
            max_df=0.8, ngram_range=(1, 2)
        )

    try:
        feature_matrix = vectorizer.fit_transform(processed)
    except ValueError as e:
        if "empty vocabulary" in str(e).lower():
            raise ValueError(
                "Empty vocabulary error: No valid terms found after text preprocessing. "
                "This can happen when:\n"
                "  1. The dataset is too small or contains only short responses\n"
                "  2. All words are filtered by stop words (try a different language)\n"
                "  3. The min_df/max_df thresholds are too restrictive\n"
                "  4. Responses contain mostly non-text content (numbers, symbols)\n\n"
                "Suggestions:\n"
                "  - Provide more diverse text responses\n"
                "  - Try a different stop words language setting\n"
                "  - Check that your data contains meaningful text content"
            ) from e
        raise

    # Calculate the maximum valid number of codes based on data constraints
    max_valid = min(len(df), feature_matrix.shape[1], max_codes)

    # Check if we have enough data for clustering
    if max_valid < 2:
        raise ValueError(
            f"Dataset too small for clustering. Need at least 2 samples and features, "
            f"but found {len(df)} samples and {feature_matrix.shape[1]} features after vectorization. "
            f"Try adding more diverse responses or reducing preprocessing constraints."
        )

    # Check if we can support the minimum requested codes
    if max_valid < min_codes:
        raise ValueError(
            f"Dataset cannot support {min_codes} codes. Maximum possible is {max_valid}. "
            f"Found {len(df)} samples and {feature_matrix.shape[1]} features. "
            f"Either reduce min_codes or provide a larger, more diverse dataset."
        )

    # Adjust max_codes to valid range
    max_codes = max_valid

    results = {
        'silhouette_scores': {},
        'calinski_scores': {},
        'tested_range': list(range(min_codes, max_codes + 1))
    }

    best_score = -1
    optimal_n = min_codes
    any_success = False  # Track whether any configuration succeeded

    total_iterations = max_codes - min_codes + 1

    for i, n in enumerate(range(min_codes, max_codes + 1)):
        if progress_callback:
            progress = (i + 1) / total_iterations
            progress_callback(progress, f"Testing {n} codes...")

        try:
            if method == 'tfidf_kmeans':
                model = KMeans(n_clusters=n, random_state=42, n_init=10)
                labels = model.fit_predict(feature_matrix)
            elif method == 'lda':
                model = LatentDirichletAllocation(n_components=n, random_state=42, max_iter=10)
                doc_topics = model.fit_transform(feature_matrix)
                labels = doc_topics.argmax(axis=1)
            else:  # nmf
                model = NMF(n_components=n, random_state=42, max_iter=100)
                doc_topics = model.fit_transform(feature_matrix)
                labels = doc_topics.argmax(axis=1)

            # Calculate silhouette score (only if we have more than 1 unique label)
            if len(set(labels)) > 1:
                sil_score = silhouette_score(feature_matrix, labels)
                cal_score = calinski_harabasz_score(feature_matrix.toarray(), labels)

                results['silhouette_scores'][n] = sil_score
                results['calinski_scores'][n] = cal_score
                any_success = True  # Mark that at least one configuration succeeded

                if sil_score > best_score:
                    best_score = sil_score
                    optimal_n = n
        except Exception as e:
            logger.warning(f"Could not evaluate {n} codes: {e}")
            continue

    # Validate that at least one configuration succeeded
    if not any_success:
        raise ValueError(
            f"Auto-optimization failed: No valid configurations found when testing {min_codes}-{max_codes} codes. "
            f"All attempts either failed or produced clusters with only one unique label. "
            f"This typically indicates:\n"
            f"  1. The dataset is too homogeneous (responses are too similar)\n"
            f"  2. The dataset is too small for meaningful clustering\n"
            f"  3. Text preprocessing removed too much content\n\n"
            f"Suggestions:\n"
            f"  - Provide a larger, more diverse dataset\n"
            f"  - Try reducing the number of codes\n"
            f"  - Check that responses contain varied content"
        )

    results['optimal_n_codes'] = optimal_n
    results['best_silhouette_score'] = best_score

    logger.info(f"Optimal number of codes: {optimal_n} (silhouette score: {best_score:.4f})")

    return optimal_n, results


def run_ml_analysis(
    df: pd.DataFrame,
    text_column: str,
    n_codes: int = 10,
    method: str = 'tfidf_kmeans',
    min_confidence: float = 0.2,
    representation: str = 'tfidf',
    embedding_kwargs: Optional[Dict[str, Any]] = None,
    progress_callback=None,
    use_adaptive_preprocessing: bool = True,
    preprocessing_override: Optional[Dict[str, Any]] = None,
    label_column: Optional[str] = None
) -> Tuple[Any, Any, Dict]:
    """
    Run ML-based open coding analysis.

    Args:
        df: DataFrame with responses
        text_column: Name of the text column
        n_codes: Number of codes to discover
        method: ML method ('tfidf_kmeans', 'lda', 'nmf')
        min_confidence: Minimum confidence threshold
        representation: Text representation method:
            - 'tfidf' (default): TF-IDF vectorization (bag-of-words, fast)
            - 'sbert': SentenceBERT embeddings (semantic, offline, slower)
            - 'word2vec': Word2Vec embeddings (semantic, trains on data)
            - 'fasttext': FastText embeddings (handles typos, trains on data)
        embedding_kwargs: Additional kwargs for embedding methods (e.g., model_name for SBERT)
        progress_callback: Optional callback for progress updates
        use_adaptive_preprocessing: If True (default), automatically detect dataset
            characteristics and apply appropriate preprocessing. This is transparent
            for simple OE datasets (uses standard settings) but adapts for
            long-form documents (news articles, etc.) to improve clustering quality.
        preprocessing_override: Optional dict to override specific TF-IDF parameters
            (e.g., {'min_df': 5, 'max_df': 0.9}). Useful for fine-tuning.
        label_column: Optional column name containing ground truth labels for
            POST-HOC evaluation only. CRITICAL: Labels are NEVER used for
            clustering/training. They are only used for diagnostic metrics
            (ARI, NMI, purity) after clustering is complete.

    Returns:
        Tuple of (coder, results, metrics)

    Raises:
        ValueError: If dataset is empty or too small for the requested number of codes

    Notes:
        - TF-IDF is the default for backward compatibility and speed
        - Semantic embeddings provide better understanding but are slower
        - All embedding methods work offline (no API keys required)
        - Which embedding was used is recorded in metrics['representation']
        - Adaptive preprocessing is transparent for simple OE datasets
        - Labels (if provided) are NEVER used for training, only for post-hoc evaluation

    Example:
        # Simple OE analysis (backward compatible)
        coder, results, metrics = run_ml_analysis(df, 'response', n_codes=5)

        # Reuters-style long-form documents with label evaluation
        coder, results, metrics = run_ml_analysis(
            df, 'text',
            n_codes=10,
            label_column='category'  # For post-hoc evaluation only
        )

        # Manual preprocessing override
        coder, results, metrics = run_ml_analysis(
            df, 'text',
            preprocessing_override={'min_df': 5, 'sublinear_tf': True}
        )
    """
    import sys
    sys.path.insert(0, '.')

    # Import from notebook (classes defined in cells)
    # For now, we'll recreate the classes here
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    import re
    from collections import Counter, defaultdict

    # Validate dataset before processing
    if df is None or len(df) == 0:
        raise ValueError("DataFrame is empty. Cannot perform analysis on empty dataset.")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    # Check that we have enough data for the requested number of codes
    if len(df) < n_codes:
        raise ValueError(
            f"Dataset too small for {n_codes} codes. "
            f"Need at least {n_codes} responses, but only have {len(df)}. "
            f"Either reduce the number of codes or provide more data."
        )

    # Simple MLOpenCoder class with dataset-aware preprocessing
    class MLOpenCoder:
        def __init__(
            self,
            n_codes=10,
            method='tfidf_kmeans',
            min_confidence=0.3,
            representation='tfidf',
            embedding_kwargs=None,
            use_adaptive_preprocessing=True,
            preprocessing_override=None,
            labels_for_evaluation=None
        ):
            """
            Initialize ML-based open coder.

            Args:
                n_codes: Number of codes to discover
                method: ML method ('tfidf_kmeans', 'lda', 'nmf')
                min_confidence: Minimum confidence threshold for code assignment
                representation: Text representation method:
                    - 'tfidf' (default): TF-IDF vectorization (bag-of-words)
                    - 'sbert': SentenceBERT embeddings (semantic, offline)
                    - 'word2vec': Word2Vec embeddings (semantic, trains on data)
                    - 'fasttext': FastText embeddings (handles typos, trains on data)
                embedding_kwargs: Additional kwargs for embedding methods
                use_adaptive_preprocessing: If True, auto-detect dataset characteristics
                    and apply appropriate preprocessing. Default: True for backward
                    compatibility (uses standard settings for simple datasets)
                preprocessing_override: Optional dict to override specific preprocessing
                    parameters (e.g., {'min_df': 5, 'max_df': 0.9})
                labels_for_evaluation: Optional labels for POST-HOC evaluation only.
                    CRITICAL: Labels are NEVER used for clustering, only for
                    diagnostic metrics after clustering is complete.
            """
            self.n_codes = n_codes
            self.method = method
            self.min_confidence = min_confidence
            self.representation = representation
            self.embedding_kwargs = embedding_kwargs or {}
            self.use_adaptive_preprocessing = use_adaptive_preprocessing
            self.preprocessing_override = preprocessing_override
            self.labels_for_evaluation = labels_for_evaluation
            self.vectorizer = None
            self.model = None
            self.codebook = {}
            self.code_assignments = None
            self.confidence_scores = None
            self.feature_matrix = None

            # New: Dataset characteristics and preprocessing config (detected during fit)
            self.dataset_characteristics = None
            self.preprocessing_config = None

            # New: Cluster interpretation report (generated after clustering)
            self.cluster_interpretation = None

            # New: Post-hoc evaluation metrics (if labels provided)
            self.evaluation_metrics = None

            # New: VectorizerFactory for method parity (NMF/LDA use same settings as KMeans)
            self._vectorizer_factory = None

            # Warn about computational costs for large datasets with embeddings
            if representation != 'tfidf':
                logger.info(
                    f"Using {representation} representation. "
                    "This may be slower than TF-IDF but provides better semantic understanding."
                )

        def preprocess_text(self, text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            # Keep letters (including accented chars for Spanish/French/German) and whitespace
            # Pattern: a-z (basic Latin), ß (German eszett), à-ö and ø-ÿ (Latin-1 accented letters)
            # This preserves: é, è, ê, ë, à, â, ç, î, ï, ô, û, ù, ü, ÿ, ñ, á, í, ó, ú, ä, ö, ß, etc.
            text = re.sub(r'[^a-zßà-öø-ÿ\s]', ' ', text)
            text = ' '.join(text.split())
            return text

        def fit(self, responses, stop_words='english'):
            processed = [self.preprocess_text(r) for r in responses]

            # Validate LDA/NMF compatibility with representation
            # LDA and NMF require non-negative count/frequency matrices (TF-IDF or CountVectorizer)
            # Semantic embeddings (SBERT, Word2Vec, FastText) produce dense vectors with negative values
            if self.method in ['lda', 'nmf'] and self.representation != 'tfidf':
                raise ValueError(
                    f"Method '{self.method.upper()}' is incompatible with '{self.representation}' representation. "
                    f"LDA and NMF require non-negative count/frequency matrices (bag-of-words). "
                    f"Semantic embeddings like SBERT, Word2Vec, and FastText produce vectors with negative values. "
                    f"Please use representation='tfidf' with LDA/NMF, or switch to method='tfidf_kmeans' "
                    f"for clustering with semantic embeddings."
                )

            # Choose vectorization method based on representation
            if self.representation == 'tfidf':
                # NEW: Adaptive preprocessing based on dataset characteristics
                if self.use_adaptive_preprocessing and ADAPTIVE_PREPROCESSING_AVAILABLE:
                    # Detect dataset characteristics
                    logger.info("Detecting dataset characteristics for adaptive preprocessing...")
                    config, characteristics = get_adaptive_preprocessing(
                        texts=responses,  # Use original responses for analysis
                        labels=self.labels_for_evaluation,  # Labels used ONLY for detection, not training
                        override_config=self.preprocessing_override
                    )
                    self.dataset_characteristics = characteristics
                    self.preprocessing_config = config

                    logger.info(
                        f"Dataset detected as: {characteristics.suggested_preprocessing} "
                        f"(median_len={characteristics.median_doc_length:.1f} words, "
                        f"n_docs={characteristics.n_documents}, "
                        f"is_long_form={characteristics.is_long_form})"
                    )
                    logger.info(f"Preprocessing config: {config.config_rationale}")

                    # NEW: Use VectorizerFactory for method parity
                    # This ensures NMF uses the same TF-IDF matrix as KMeans
                    # and LDA uses CountVectorizer with identical settings
                    if VECTORIZER_FACTORY_AVAILABLE:
                        vectorizer_config = VectorizerConfig.from_preprocessing_config(config)
                        factory = VectorizerFactory()
                        self.vectorizer, self.feature_matrix = factory.create_for_method(
                            self.method, processed, vectorizer_config
                        )
                        # Store factory for potential reuse
                        self._vectorizer_factory = factory
                        logger.info(
                            f"VectorizerFactory created {type(self.vectorizer).__name__} "
                            f"for method '{self.method}'"
                        )
                    else:
                        # Legacy fallback without factory
                        if self.method == 'lda':
                            vectorizer_kwargs = config.to_dict()
                            vectorizer_kwargs.pop('sublinear_tf', None)
                            self.vectorizer = CountVectorizer(**vectorizer_kwargs)
                        else:
                            self.vectorizer = TfidfVectorizer(**config.to_dict())
                        self.feature_matrix = self.vectorizer.fit_transform(processed)
                else:
                    # Fallback: Traditional fixed TF-IDF (backward compatible)
                    logger.info("Using standard preprocessing (adaptive disabled or unavailable)")

                    # Use VectorizerFactory even without adaptive preprocessing
                    if VECTORIZER_FACTORY_AVAILABLE:
                        default_config = VectorizerConfig(
                            max_features=1000,
                            stop_words=stop_words,
                            min_df=2,
                            max_df=0.8,
                            ngram_range=(1, 2)
                        )
                        factory = VectorizerFactory()
                        self.vectorizer, self.feature_matrix = factory.create_for_method(
                            self.method, processed, default_config
                        )
                        self._vectorizer_factory = factory
                        logger.info(
                            f"VectorizerFactory created {type(self.vectorizer).__name__} "
                            f"for method '{self.method}' (standard settings)"
                        )
                    else:
                        # Legacy fallback
                        if self.method == 'lda':
                            self.vectorizer = CountVectorizer(
                                max_features=1000, stop_words=stop_words, min_df=2, max_df=0.8
                            )
                        else:
                            self.vectorizer = TfidfVectorizer(
                                max_features=1000, stop_words=stop_words, min_df=2,
                                max_df=0.8, ngram_range=(1, 2)
                            )
                        try:
                            self.feature_matrix = self.vectorizer.fit_transform(processed)
                        except ValueError as e:
                            if "empty vocabulary" in str(e).lower():
                                raise ValueError(
                                    "Empty vocabulary error: No valid terms found after text preprocessing. "
                                    "This can happen when:\n"
                                    "  1. The dataset is too small or contains only short responses\n"
                                    "  2. All words are filtered by stop words (try a different language)\n"
                                    "  3. The min_df/max_df thresholds are too restrictive\n"
                                    "  4. Responses contain mostly non-text content (numbers, symbols)\n\n"
                                    "Suggestions:\n"
                                    "  - Provide more diverse text responses\n"
                                    "  - Try a different stop words language setting\n"
                                    "  - Check that your data contains meaningful text content"
                                ) from e
                            raise

            else:
                # Semantic embeddings (new functionality)
                from src.embeddings import get_embedder

                logger.info(f"Training {self.representation} embeddings...")
                self.vectorizer = get_embedder(self.representation, **self.embedding_kwargs)
                try:
                    self.feature_matrix = self.vectorizer.fit_transform(processed)
                except ValueError as e:
                    if "empty vocabulary" in str(e).lower():
                        raise ValueError(
                            "Empty vocabulary error: No valid terms found for embedding. "
                            "The dataset may be too small or contain only empty/whitespace responses. "
                            "Please provide more diverse text content."
                        ) from e
                    raise
                logger.info(
                    f"Embeddings created: shape={self.feature_matrix.shape}, "
                    f"features={self.feature_matrix.shape[1]}"
                )

            # Train clustering/topic model
            if self.method == 'lda':
                self.model = LatentDirichletAllocation(
                    n_components=self.n_codes, random_state=42, max_iter=20
                )
            elif self.method == 'nmf':
                self.model = NMF(n_components=self.n_codes, random_state=42, max_iter=200)
            else:  # tfidf_kmeans or any kmeans-based
                self.model = KMeans(n_clusters=self.n_codes, random_state=42, n_init=10)

            # Fit model and get document-topic distributions
            if self.method in ['lda', 'nmf']:
                doc_topic_matrix = self.model.fit_transform(self.feature_matrix)
            else:
                labels = self.model.fit_predict(self.feature_matrix)
                doc_topic_matrix = np.zeros((len(responses), self.n_codes))
                for i, label in enumerate(labels):
                    doc_topic_matrix[i, label] = 1.0

            self._generate_codebook()
            self._assign_codes(doc_topic_matrix, responses)

            # NEW: Mandatory cluster interpretation layer
            # This ensures cluster IDs are never exposed without human-readable explanations
            if CLUSTER_INTERPRETATION_AVAILABLE and self.representation == 'tfidf':
                try:
                    interpreter = ClusterInterpreter(
                        n_top_terms=15,
                        n_label_terms=3,
                        n_representative_docs=5,
                        min_term_weight_threshold=0.005
                    )
                    self.cluster_interpretation = interpreter.interpret_clusters(
                        vectorizer=self.vectorizer,
                        cluster_model=self.model,
                        texts=responses,
                        cluster_assignments=labels if hasattr(self.model, 'labels_') else
                            doc_topic_matrix.argmax(axis=1).tolist(),
                        feature_matrix=self.feature_matrix,
                        method_name=self.method
                    )
                    logger.info(
                        f"Cluster interpretation complete: "
                        f"{self.cluster_interpretation.n_clusters} clusters, "
                        f"interpretability={self.cluster_interpretation.overall_interpretability:.1%}"
                    )
                except Exception as e:
                    logger.warning(f"Cluster interpretation failed: {e}")
                    self.cluster_interpretation = None

            # NEW: Post-hoc evaluation with labels (if provided)
            # CRITICAL: Labels were NOT used for clustering, only for diagnostic metrics
            if self.labels_for_evaluation is not None and CLUSTER_EVALUATION_AVAILABLE:
                try:
                    cluster_labels = (
                        self.model.labels_.tolist() if hasattr(self.model, 'labels_') else
                        doc_topic_matrix.argmax(axis=1).tolist()
                    )
                    self.evaluation_metrics = evaluate_clusters_posthoc(
                        cluster_assignments=cluster_labels,
                        true_labels=self.labels_for_evaluation
                    )
                    if self.evaluation_metrics:
                        logger.info(
                            f"Post-hoc evaluation: ARI={self.evaluation_metrics.ari:.4f}, "
                            f"NMI={self.evaluation_metrics.nmi:.4f}, "
                            f"Purity={self.evaluation_metrics.purity:.4f}"
                        )
                except Exception as e:
                    logger.warning(f"Post-hoc evaluation failed: {e}")
                    self.evaluation_metrics = None

            return self

        def _generate_codebook(self, top_words=15):
            """Generate codebook with clean 3-word labels (stopwords/duplicates removed)."""
            # Stopwords to filter from labels
            stopwords = {
                'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'must', 'can', 'and', 'but', 'or', 'for',
                'of', 'in', 'to', 'on', 'at', 'by', 'with', 'from', 'as', 'into',
                'it', 'its', 'this', 'that', 'these', 'those', 'what', 'which',
                'i', 'me', 'my', 'we', 'our', 'you', 'your', 'they', 'them', 'their',
                've', 'll', 're', 't', 's', 'd', 'm', 'not', 'just', 'very', 'really',
                'about', 'get', 'got', 'so', 'too', 'also', 'been', 'being', 'if'
            }

            feature_names = self.vectorizer.get_feature_names_out()
            for code_idx in range(self.n_codes):
                code_id = f"CODE_{code_idx + 1:02d}"
                if self.method in ['lda', 'nmf']:
                    topic_weights = self.model.components_[code_idx]
                    top_indices = topic_weights.argsort()[-top_words:][::-1]
                else:
                    cluster_center = self.model.cluster_centers_[code_idx]
                    top_indices = cluster_center.argsort()[-top_words:][::-1]

                top_words_list = [feature_names[i] for i in top_indices]

                # Filter stopwords and duplicates for label
                seen = set()
                label_terms = []
                for word in top_words_list:
                    word_lower = word.lower().strip()
                    if word_lower not in stopwords and word_lower not in seen:
                        seen.add(word_lower)
                        label_terms.append(word)
                        if len(label_terms) >= 3:
                            break

                label = ' / '.join(term.title() for term in label_terms)

                self.codebook[code_id] = {
                    'label': label,
                    'keywords': top_words_list,
                    'count': 0,
                    'examples': [],
                    'avg_confidence': 0.0
                }

        def _assign_codes(self, doc_topic_matrix, responses):
            assignments = []
            confidences = []

            for doc_idx, topic_dist in enumerate(doc_topic_matrix):
                doc_codes = []
                doc_confidences = []

                for code_idx, confidence in enumerate(topic_dist):
                    if confidence >= self.min_confidence:
                        code_id = f"CODE_{code_idx + 1:02d}"
                        doc_codes.append(code_id)
                        doc_confidences.append(float(confidence))
                        self.codebook[code_id]['count'] += 1

                        if confidence > 0.6 and len(self.codebook[code_id]['examples']) < 10:
                            self.codebook[code_id]['examples'].append({
                                'text': str(responses[doc_idx]),
                                'confidence': float(confidence)
                            })

                assignments.append(doc_codes)
                confidences.append(doc_confidences)

            for doc_codes, doc_confs in zip(assignments, confidences):
                for code, conf in zip(doc_codes, doc_confs):
                    if self.codebook[code]['count'] > 0:
                        current_avg = self.codebook[code]['avg_confidence']
                        count = self.codebook[code]['count']
                        self.codebook[code]['avg_confidence'] = (
                            (current_avg * (count - 1) + conf) / count
                        )

            self.code_assignments = assignments
            self.confidence_scores = confidences

        def get_quality_metrics(self):
            metrics = {}

            # Guard against empty code_assignments
            if not self.code_assignments or len(self.code_assignments) == 0:
                metrics['total_assignments'] = 0
                metrics['avg_codes_per_response'] = 0.0
                metrics['coverage_pct'] = 0.0
                return metrics

            total_assignments = sum(len(codes) for codes in self.code_assignments)
            metrics['total_assignments'] = total_assignments
            metrics['avg_codes_per_response'] = total_assignments / len(self.code_assignments)

            coded_responses = sum(1 for codes in self.code_assignments if len(codes) > 0)
            metrics['coverage_pct'] = (coded_responses / len(self.code_assignments)) * 100

            all_confidences = [conf for confs in self.confidence_scores for conf in confs]
            if all_confidences:
                metrics['avg_confidence'] = np.mean(all_confidences)
                metrics['min_confidence'] = np.min(all_confidences)
                metrics['max_confidence'] = np.max(all_confidences)
                metrics['std_confidence'] = np.std(all_confidences)

            if self.feature_matrix is not None and hasattr(self.model, 'cluster_centers_'):
                labels = self.model.labels_
                if len(set(labels)) > 1:
                    metrics['silhouette_score'] = silhouette_score(self.feature_matrix, labels)

            # NEW: Include cluster interpretation metrics
            if self.cluster_interpretation is not None:
                metrics['cluster_interpretability'] = self.cluster_interpretation.overall_interpretability
                metrics['interpretation_warnings'] = len(self.cluster_interpretation.warnings)

            # NEW: Include post-hoc evaluation metrics (if labels were provided)
            if self.evaluation_metrics is not None:
                metrics['posthoc_ari'] = self.evaluation_metrics.ari
                metrics['posthoc_nmi'] = self.evaluation_metrics.nmi
                metrics['posthoc_purity'] = self.evaluation_metrics.purity

            # NEW: Include dataset characteristics (if adaptive preprocessing was used)
            if self.dataset_characteristics is not None:
                metrics['dataset_type'] = self.dataset_characteristics.suggested_preprocessing
                metrics['is_long_form'] = self.dataset_characteristics.is_long_form
                metrics['median_doc_length'] = self.dataset_characteristics.median_doc_length

            return metrics

        def get_cluster_interpretation_report(self) -> Optional[str]:
            """
            Get the human-readable cluster interpretation report.

            Returns:
                Formatted report string, or None if interpretation not available.

            Notes:
                This report should ALWAYS be presented to users alongside
                cluster assignments. Cluster IDs are internal identifiers,
                not semantic labels.
            """
            if self.cluster_interpretation is None:
                return None
            return self.cluster_interpretation.get_display_report()

        def get_cluster_codebook(self) -> Optional[Dict[str, Any]]:
            """
            Get the cluster codebook in dictionary format.

            Returns:
                Codebook dictionary with all cluster definitions, or None.
            """
            if self.cluster_interpretation is None:
                return None

            codebook = ClusterCodebook(self.cluster_interpretation)
            return codebook.to_dict()

        def get_evaluation_summary(self) -> Optional[str]:
            """
            Get the post-hoc evaluation summary.

            Returns:
                Evaluation summary string, or None if no labels were provided.

            Notes:
                This evaluation is purely diagnostic - labels were NOT used
                for clustering. Metrics measure agreement between discovered
                clusters and ground truth labels.
            """
            if self.evaluation_metrics is None:
                return None
            return self.evaluation_metrics.get_summary()

    # Run analysis with progress updates
    start_time = time.time()

    if progress_callback:
        progress_callback(0.1, "Initializing ML coder...")

    # Extract labels for post-hoc evaluation (if provided)
    # CRITICAL: Labels are NEVER used for clustering, only for diagnostic metrics
    labels_for_evaluation = None
    if label_column and label_column in df.columns:
        labels_for_evaluation = df[label_column].tolist()
        logger.info(
            f"Labels from column '{label_column}' will be used for POST-HOC "
            "evaluation only (NOT for clustering)"
        )

    coder = MLOpenCoder(
        n_codes=n_codes,
        method=method,
        min_confidence=min_confidence,
        representation=representation,
        embedding_kwargs=embedding_kwargs,
        use_adaptive_preprocessing=use_adaptive_preprocessing,
        preprocessing_override=preprocessing_override,
        labels_for_evaluation=labels_for_evaluation
    )

    if progress_callback:
        progress_callback(0.3, "Preprocessing text...")

    responses = df[text_column].tolist()

    if progress_callback:
        progress_callback(0.5, "Training ML model...")

    coder.fit(responses)

    if progress_callback:
        progress_callback(0.8, "Generating results...")

    # Create results
    results_df = df.copy()
    results_df['assigned_codes'] = coder.code_assignments
    results_df['confidence_scores'] = coder.confidence_scores
    results_df['num_codes'] = results_df['assigned_codes'].apply(len)

    # Calculate metrics
    metrics = coder.get_quality_metrics()
    metrics['execution_time'] = time.time() - start_time
    metrics['total_responses'] = len(df)
    metrics['method'] = method
    metrics['n_codes'] = n_codes
    metrics['representation'] = representation
    metrics['embedding_kwargs'] = embedding_kwargs or {}

    # NEW: Include adaptive preprocessing information
    metrics['use_adaptive_preprocessing'] = use_adaptive_preprocessing
    if coder.preprocessing_config is not None:
        metrics['preprocessing_config'] = coder.preprocessing_config.config_rationale
    if coder.dataset_characteristics is not None:
        metrics['dataset_characteristics'] = coder.dataset_characteristics.to_dict()

    # NEW: Include cluster interpretation availability
    metrics['has_cluster_interpretation'] = coder.cluster_interpretation is not None
    if coder.cluster_interpretation is not None:
        metrics['cluster_interpretation_warnings'] = coder.cluster_interpretation.warnings

    # NEW: Include post-hoc evaluation availability
    metrics['has_posthoc_evaluation'] = coder.evaluation_metrics is not None
    if label_column:
        metrics['label_column_used'] = label_column
        metrics['evaluation_note'] = "Labels used for POST-HOC evaluation only, NOT for clustering"

    if progress_callback:
        progress_callback(1.0, "Analysis complete!")

    logger.info(f"Analysis completed in {metrics['execution_time']:.2f} seconds")

    return coder, results_df, metrics


def calculate_metrics_summary(coder, results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics summary.

    Args:
        coder: Fitted MLOpenCoder instance
        results_df: Results DataFrame

    Returns:
        Dictionary of metrics
    """
    # Guard against empty results
    if results_df is None or len(results_df) == 0:
        return {
            'total_responses': 0,
            'total_codes': len(coder.codebook) if coder else 0,
            'active_codes': 0,
            'total_assignments': 0,
            'avg_codes_per_response': 0.0,
            'median_codes_per_response': 0.0,
            'max_codes_per_response': 0,
            'coverage_pct': 0.0,
            'uncoded_count': 0,
        }

    metrics = {
        'total_responses': len(results_df),
        'total_codes': len(coder.codebook),
        'active_codes': sum(1 for info in coder.codebook.values() if info['count'] > 0),
        'total_assignments': results_df['num_codes'].sum(),
        'avg_codes_per_response': results_df['num_codes'].mean(),
        'median_codes_per_response': results_df['num_codes'].median(),
        'max_codes_per_response': results_df['num_codes'].max(),
        'coverage_pct': (results_df['num_codes'] > 0).sum() / len(results_df) * 100,
        'uncoded_count': (results_df['num_codes'] == 0).sum(),
    }

    # Confidence metrics
    all_confidences = [
        conf for confs in results_df['confidence_scores'] for conf in confs
    ]
    if all_confidences:
        metrics['avg_confidence'] = np.mean(all_confidences)
        metrics['min_confidence'] = np.min(all_confidences)
        metrics['max_confidence'] = np.max(all_confidences)

    return metrics


def generate_insights(coder, results_df: pd.DataFrame, top_n: int = 5) -> List[str]:
    """
    Generate key insights from analysis results.

    Args:
        coder: Fitted MLOpenCoder instance
        results_df: Results DataFrame
        top_n: Number of top codes to analyze

    Returns:
        List of insight strings
    """
    insights = []

    # Get code frequencies
    code_counts = Counter()
    for codes in results_df['assigned_codes']:
        for code in codes:
            code_counts[code] += 1

    if code_counts:
        # Most common code
        top_code, top_count = code_counts.most_common(1)[0]
        top_label = coder.codebook[top_code]['label']
        top_pct = (top_count / len(results_df)) * 100
        insights.append(
            f"📊 **Dominant Theme**: '{top_label}' appears in {top_pct:.1f}% "
            f"of responses ({top_count:,} responses)"
        )

    # Coverage
    coded_count = (results_df['num_codes'] > 0).sum()
    coverage_pct = (coded_count / len(results_df)) * 100
    if coverage_pct < 80:
        uncoded = len(results_df) - coded_count
        insights.append(
            f"⚠️ **Coverage Note**: {uncoded:,} responses ({100-coverage_pct:.1f}%) "
            "were not assigned codes and may need manual review"
        )
    else:
        insights.append(
            f"✅ **High Coverage**: {coverage_pct:.1f}% of responses successfully coded"
        )

    # Multi-coding
    multi_coded = (results_df['num_codes'] > 1).sum()
    if multi_coded > 0:
        multi_pct = (multi_coded / len(results_df)) * 100
        insights.append(
            f"🔀 **Complex Responses**: {multi_coded:,} responses ({multi_pct:.1f}%) "
            "received multiple codes, indicating nuanced perspectives"
        )

    # Code diversity
    active_codes = sum(1 for info in coder.codebook.values() if info['count'] > 0)
    total_codes = len(coder.codebook)
    utilization = (active_codes / total_codes) * 100
    insights.append(
        f"📈 **Code Utilization**: {active_codes}/{total_codes} codes ({utilization:.1f}%) "
        "are actively used"
    )

    # Average confidence
    all_confidences = [
        conf for confs in results_df['confidence_scores'] for conf in confs
    ]
    if all_confidences:
        avg_conf = np.mean(all_confidences)
        if avg_conf >= 0.7:
            quality = "High"
            emoji = "🟢"
        elif avg_conf >= 0.5:
            quality = "Moderate"
            emoji = "🟡"
        else:
            quality = "Low"
            emoji = "🔴"

        insights.append(
            f"{emoji} **Confidence Level**: {quality} (avg: {avg_conf:.2f})"
        )

    return insights


def get_analysis_summary(coder, results_df: pd.DataFrame) -> str:
    """
    Generate a text summary of the analysis.

    Args:
        coder: Fitted MLOpenCoder instance
        results_df: Results DataFrame

    Returns:
        Summary text
    """
    metrics = calculate_metrics_summary(coder, results_df)
    insights = generate_insights(coder, results_df)

    summary = f"""
## Analysis Summary

**Dataset Overview**
- Total Responses: {metrics['total_responses']:,}
- Codes Discovered: {metrics['total_codes']}
- Active Codes: {metrics['active_codes']}

**Coding Statistics**
- Total Assignments: {metrics['total_assignments']:,}
- Average Codes per Response: {metrics['avg_codes_per_response']:.2f}
- Coverage: {metrics['coverage_pct']:.1f}%
- Uncoded Responses: {metrics['uncoded_count']:,}

**Quality Metrics**
- Average Confidence: {metrics.get('avg_confidence', 0):.2f}
- Confidence Range: {metrics.get('min_confidence', 0):.2f} - {metrics.get('max_confidence', 0):.2f}

---

### Key Insights

""" + "\n".join(insights)

    return summary


def get_top_codes(coder, n: int = 10, include_quotes: bool = True) -> pd.DataFrame:
    """
    Get top N codes by frequency with labels, keywords, and representative quotes.

    Args:
        coder: Fitted MLOpenCoder instance
        n: Number of codes to return
        include_quotes: Include representative quotes (default True)

    Returns:
        DataFrame with: Code, Label, Count, Keywords, Representative Quotes
    """
    code_data = []

    for code_id, info in coder.codebook.items():
        row = {
            'Code': code_id,
            'Label': info['label'],
            'Count': info['count'],
            'Avg Confidence': info['avg_confidence'],
            'Keywords': ', '.join(info['keywords'][:10])
        }

        if include_quotes:
            # Get top representative quote
            examples = info.get('examples', [])
            if examples:
                quote = examples[0].get('text', '')[:150]
                if len(examples[0].get('text', '')) > 150:
                    quote += '...'
                row['Representative Quote'] = quote
            else:
                row['Representative Quote'] = ''

        code_data.append(row)

    df = pd.DataFrame(code_data)
    df = df.sort_values('Count', ascending=False).head(n)

    return df


def get_code_summary_with_quotes(coder, n_quotes: int = 3) -> pd.DataFrame:
    """
    Get comprehensive code summary with representative quotes.

    Returns DataFrame with: Code, Label, Keywords, Representative Quotes

    Args:
        coder: Fitted MLOpenCoder instance
        n_quotes: Number of representative quotes per code (default 3)

    Returns:
        DataFrame with code summaries including representative quotes
    """
    code_data = []

    for code_id, info in coder.codebook.items():
        # Get representative quotes (examples stored during assignment)
        quotes = []
        for example in info.get('examples', [])[:n_quotes]:
            text = example.get('text', '')
            # Truncate long quotes
            if len(text) > 150:
                text = text[:150] + '...'
            quotes.append(text)

        code_data.append({
            'Code': code_id,
            'Label': info['label'],
            'Keywords': ', '.join(info['keywords'][:10]),
            'Count': info['count'],
            'Representative Quotes': ' | '.join(quotes) if quotes else '(no high-confidence examples)'
        })

    df = pd.DataFrame(code_data)
    df = df.sort_values('Count', ascending=False)

    return df


def get_cooccurrence_pairs(results_df: pd.DataFrame, min_count: int = 2) -> pd.DataFrame:
    """
    Get code pairs that frequently co-occur.

    Args:
        results_df: Results DataFrame
        min_count: Minimum co-occurrence count

    Returns:
        DataFrame with co-occurring pairs
    """
    pairs = Counter()

    for codes in results_df['assigned_codes']:
        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                pair = tuple(sorted([code1, code2]))
                pairs[pair] += 1

    pair_data = []
    for (code1, code2), count in pairs.most_common():
        if count >= min_count:
            pair_data.append({
                'Code 1': code1,
                'Code 2': code2,
                'Count': count,
                'Percentage': (count / len(results_df)) * 100
            })

    return pd.DataFrame(pair_data)


def get_qa_report(
    coder,
    results_df: pd.DataFrame,
    demographics: Optional[pd.DataFrame] = None,
    demographic_columns: Optional[List[str]] = None,
    include_rigor: bool = True
) -> str:
    """
    Generate comprehensive Quality Assurance (QA) report.

    This report includes basic quality metrics and advanced rigor diagnostics
    for methodological validity assessment.

    Args:
        coder: Fitted MLOpenCoder instance
        results_df: Results DataFrame with code assignments
        demographics: Optional demographics DataFrame for bias detection
        demographic_columns: Optional list of demographic columns to analyze
        include_rigor: Whether to include rigor diagnostics (default: True)

    Returns:
        Formatted QA report as string (Markdown format)
    """
    from src.rigor_diagnostics import RigorDiagnostics

    # Calculate basic metrics
    metrics = calculate_metrics_summary(coder, results_df)

    # Initialize report
    report = []
    report.append("# Quality Assurance Report")
    report.append("")
    report.append("---")
    report.append("")

    # Section 1: Basic Quality Metrics
    report.append("## 1. Basic Quality Metrics")
    report.append("")
    report.append(f"- **Total Responses**: {metrics['total_responses']:,}")
    report.append(f"- **Total Codes**: {metrics['total_codes']}")
    report.append(f"- **Active Codes**: {metrics['active_codes']} ({metrics['active_codes']/metrics['total_codes']*100:.1f}%)")
    report.append(f"- **Total Assignments**: {metrics['total_assignments']:,}")
    report.append(f"- **Coverage**: {metrics['coverage_pct']:.1f}% ({metrics['uncoded_count']:,} uncoded)")
    report.append(f"- **Avg Codes per Response**: {metrics['avg_codes_per_response']:.2f}")
    report.append("")

    if metrics.get('avg_confidence'):
        report.append(f"- **Average Confidence**: {metrics['avg_confidence']:.3f}")
        report.append(f"- **Confidence Range**: {metrics['min_confidence']:.3f} - {metrics['max_confidence']:.3f}")
        report.append("")

    # Section 2: Rigor Diagnostics (if requested)
    if include_rigor:
        diagnostics = RigorDiagnostics()

        # Run diagnostics
        validity = diagnostics.assess_validity(coder, results_df)
        bias = diagnostics.detect_bias(
            results_df,
            demographics=demographics,
            demographic_columns=demographic_columns
        )
        sanity = diagnostics.sanity_check(coder, results_df)
        recommendations = diagnostics.generate_recommendations(validity, bias, sanity)

        report.append("## 2. Rigor Diagnostics")
        report.append("")

        # 2.1 Sanity Check Summary
        report.append("### 2.1 Health Status")
        report.append("")
        health = sanity['health_status']
        report.append(f"**Status**: {health['message']}")
        report.append("")
        if sanity['total_issues'] > 0:
            report.append(f"**Issues Detected**: {sanity['total_issues']}")
            report.append("")

        # 2.2 Warnings
        if sanity['warnings']:
            report.append("### 2.2 Warnings")
            report.append("")
            for warning in sanity['warnings']:
                report.append(f"- {warning}")
            report.append("")

        # 2.3 Validity Dimensions
        report.append("### 2.3 Validity Assessment")
        report.append("")

        # Coverage
        coverage = validity['coverage_ratio']
        report.append(f"**Coverage Ratio**: {coverage['coverage_percentage']:.1f}% - {coverage['interpretation']}")

        # Code Utilization
        utilization = validity['code_utilization']
        report.append(f"**Code Utilization**: {utilization['utilization_rate']:.1f}% - {utilization['interpretation']}")

        # Theme Coherence
        coherence = validity['theme_coherence']
        if coherence['average_coherence'] is not None:
            report.append(f"**Theme Coherence**: {coherence['average_coherence']:.3f} - {coherence['interpretation']}")

        # Code Stability
        stability = validity['code_stability']
        if stability['stability_score'] is not None:
            report.append(f"**Code Stability**: {stability['stability_score']:.3f} - {stability['interpretation']}")

        # Thematic Saturation
        saturation = validity['thematic_saturation']
        report.append(f"**Thematic Saturation**: {saturation['saturation_status'].replace('_', ' ').title()} - {saturation['message']}")

        # Ambiguity Rate
        ambiguity = validity['ambiguity_rate']
        report.append(f"**Ambiguity Rate**: {ambiguity['multi_code_percentage']:.1f}% multi-coded - {ambiguity['interpretation']}")

        # Boundary Cases
        boundary = validity['boundary_cases']
        report.append(f"**Boundary Cases**: {boundary['count']} responses ({boundary['percentage']:.1f}%) near decision threshold")

        report.append("")

        # 2.4 Confidence Distribution
        report.append("### 2.4 Confidence Distribution")
        report.append("")
        conf_dist = validity['confidence_distribution']
        report.append(f"- **Mean**: {conf_dist['mean']:.3f}")
        report.append(f"- **Median**: {conf_dist['median']:.3f}")
        report.append(f"- **Std Dev**: {conf_dist['std']:.3f}")
        report.append(f"- **25th Percentile**: {conf_dist['percentiles']['25th']:.3f}")
        report.append(f"- **75th Percentile**: {conf_dist['percentiles']['75th']:.3f}")
        report.append(f"- **90th Percentile**: {conf_dist['percentiles']['90th']:.3f}")
        report.append(f"- **Interpretation**: {conf_dist['interpretation']}")
        report.append("")

        # 2.5 Bias Detection
        report.append("### 2.5 Bias Detection")
        report.append("")

        imbalance = bias['code_imbalance']
        report.append(f"**Code Imbalance Ratio**: {imbalance['imbalance_ratio']:.1f}:1 - {imbalance['interpretation']}")
        report.append(f"**Gini Coefficient**: {imbalance['gini_coefficient']:.3f} (0=perfect equality, 1=perfect inequality)")
        report.append("")

        # Demographic representation (if provided)
        if demographics is not None and demographic_columns:
            report.append("**Demographic Representation**:")
            demo_rep = bias['demographic_representation']
            for demo_col in demographic_columns:
                if demo_col in demo_rep:
                    chi_test = demo_rep[demo_col]['chi_square_test']
                    if 'p_value' in chi_test:
                        report.append(f"- {demo_col}: p={chi_test['p_value']:.4f} - {chi_test['interpretation']}")
            report.append("")

        # Systematic patterns
        patterns = bias['systematic_patterns']
        if 'positional_bias' in patterns:
            pos_bias = patterns['positional_bias']
            report.append(f"**Positional Bias**: First half avg={pos_bias['first_half_avg']:.2f}, "
                         f"Second half avg={pos_bias['second_half_avg']:.2f} "
                         f"({'Significant' if pos_bias['significant'] else 'Not significant'})")
            report.append("")

    # Section 3: Recommendations
    if include_rigor and recommendations:
        report.append("## 3. Recommendations")
        report.append("")
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec}")
        report.append("")

    # Section 4: Code Utilization Details
    report.append("## 4. Code Utilization Details")
    report.append("")

    if include_rigor:
        utilization = validity['code_utilization']
        if utilization['underused_codes']:
            report.append("**Underused Codes**:")
            for code in utilization['underused_codes'][:10]:
                code_info = coder.codebook.get(code, {})
                report.append(f"- {code}: '{code_info.get('label', 'N/A')}' (count={code_info.get('count', 0)})")
            report.append("")

        if utilization['overused_codes']:
            report.append("**Overused Codes**:")
            for code in utilization['overused_codes'][:5]:
                code_info = coder.codebook.get(code, {})
                report.append(f"- {code}: '{code_info.get('label', 'N/A')}' (count={code_info.get('count', 0)})")
            report.append("")

    # Section 5: Uncoded Responses
    if metrics['uncoded_count'] > 0:
        report.append("## 5. Uncoded Responses")
        report.append("")
        report.append(f"**Total Uncoded**: {metrics['uncoded_count']:,} ({100 - metrics['coverage_pct']:.1f}%)")
        report.append("")
        report.append("**Possible Reasons**:")
        report.append("- Response does not meet minimum confidence threshold")
        report.append("- Response content does not match any discovered themes")
        report.append("- Response may represent outlier or unique perspective")
        report.append("")
        report.append("**Recommendation**: Review uncoded responses manually to determine if:")
        report.append("1. Additional codes are needed")
        report.append("2. Confidence threshold should be lowered")
        report.append("3. Responses are truly non-analytic (e.g., 'N/A', 'No comment')")
        report.append("")

    # Footer
    report.append("---")
    report.append("")
    report.append("*This QA report was generated automatically. All metrics should be interpreted "
                 "in the context of your specific research questions and data characteristics.*")
    report.append("")
    report.append("*For interpretation guidance, see: documentation/RIGOR_DIAGNOSTICS_GUIDE.md*")

    return "\n".join(report)


def export_results_package(coder, results_df: pd.DataFrame, format: str = 'excel') -> bytes:
    """
    Export complete results package.

    Args:
        coder: Fitted MLOpenCoder instance
        results_df: Results DataFrame
        format: Export format ('excel', 'csv_zip')

    Returns:
        Bytes of exported data
    """
    from io import BytesIO

    if format == 'excel':
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Code assignments
            results_df[['assigned_codes', 'confidence_scores', 'num_codes']].to_excel(
                writer, sheet_name='Assignments', index=False
            )

            # Codebook
            codebook_data = []
            for code_id, info in coder.codebook.items():
                codebook_data.append({
                    'Code': code_id,
                    'Label': info['label'],
                    'Keywords': ', '.join(info['keywords']),
                    'Count': info['count'],
                    'Avg Confidence': info['avg_confidence']
                })
            pd.DataFrame(codebook_data).to_excel(
                writer, sheet_name='Codebook', index=False
            )

            # Frequency
            freq_df = get_top_codes(coder, n=100)
            freq_df.to_excel(writer, sheet_name='Frequencies', index=False)

            # Co-occurrences
            cooccur = get_cooccurrence_pairs(results_df)
            if not cooccur.empty:
                cooccur.to_excel(writer, sheet_name='Co-occurrences', index=False)

        return buffer.getvalue()

    else:
        raise ValueError(f"Unsupported format: {format}")


def generate_methods_documentation(
    coder,
    results_df: pd.DataFrame,
    metrics: Dict[str, Any],
    preprocessing_params: Optional[Dict[str, Any]] = None,
    project_name: str = "Open-Ended Coding Analysis",
    output_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive methods documentation.

    This function creates academic-style methods documentation including:
    - Data preparation details
    - Coding approach and methodology
    - Quality assurance metrics
    - Methodological assumptions
    - Honest limitations
    - Ethical considerations
    - Reproducibility information

    Args:
        coder: Fitted MLOpenCoder instance
        results_df: Results DataFrame with assignments
        metrics: Quality metrics dictionary
        preprocessing_params: Optional preprocessing parameters
        project_name: Name of the project
        output_path: Optional path to save METHODS.md file

    Returns:
        Methods documentation as markdown string

    Example:
        >>> methods_doc = generate_methods_documentation(
        ...     coder, results_df, metrics,
        ...     project_name="Survey Analysis 2024"
        ... )
        >>> # Optionally export to file
        >>> export_methods_to_file(methods_doc, "METHODS.md")
    """
    generator = MethodsDocGenerator(project_name=project_name)

    # Generate methods section
    methods = generator.generate_methods_section(
        coder,
        results_df,
        metrics,
        preprocessing_params
    )

    # Audit for objectivity claims
    passed, violations = generator.audit_objectivity_claims(methods)
    if not passed:
        logger.warning(
            f"Methods documentation contains {len(violations)} objectivity claim violations. "
            "Review and revise before publication."
        )
        for violation in violations:
            logger.warning(f"  - {violation['phrase']}: {violation['context']}")

    # Export to file if path provided
    if output_path:
        export_methods_to_file(methods, output_path)
        logger.info(f"Methods documentation exported to: {output_path}")

    return methods


def generate_executive_summary(
    coder,
    results_df: pd.DataFrame,
    metrics: Dict[str, Any],
    include_methods: bool = True
) -> str:
    """
    Generate executive summary with optional methods section.

    Produces a comprehensive summary including:
    - Key findings and insights
    - Quality metrics
    - Code frequency distribution
    - Coverage and confidence statistics
    - Optional: Full methods documentation

    Args:
        coder: Fitted MLOpenCoder instance
        results_df: Results DataFrame
        metrics: Quality metrics dictionary
        include_methods: Whether to include full methods section

    Returns:
        Executive summary as markdown string
    """
    summary = []

    summary.append("# Executive Summary: Open-Ended Coding Analysis")
    summary.append("")
    summary.append("---")
    summary.append("")

    # Key Findings
    summary.append("## Key Findings")
    summary.append("")
    insights = generate_insights(coder, results_df, top_n=5)
    for insight in insights:
        # Remove emojis for professional summary
        clean_insight = insight.encode('ascii', 'ignore').decode('ascii')
        summary.append(f"- {clean_insight}")
    summary.append("")

    # Analysis Overview
    summary.append("## Analysis Overview")
    summary.append("")
    summary.append(f"- **Total Responses**: {metrics.get('total_responses', len(results_df)):,}")
    summary.append(f"- **Codes Discovered**: {metrics.get('n_codes', coder.n_codes)}")
    summary.append(f"- **Active Codes**: {sum(1 for info in coder.codebook.values() if info['count'] > 0)}")
    summary.append(f"- **Coverage**: {metrics.get('coverage_pct', 0):.1f}%")
    summary.append(f"- **Average Confidence**: {metrics.get('avg_confidence', 0):.2f}")
    summary.append("")

    # Top Themes
    summary.append("## Top Themes")
    summary.append("")
    top_codes = get_top_codes(coder, n=10)
    for _, row in top_codes.iterrows():
        pct = (row['Count'] / len(results_df)) * 100
        summary.append(
            f"**{row['Label']}** ({row['Code']}): "
            f"{row['Count']:,} responses ({pct:.1f}%) | "
            f"Confidence: {row['Avg Confidence']:.2f}"
        )
        summary.append(f"  - Keywords: {row['Keywords']}")
        summary.append("")

    # Quality Indicators
    summary.append("## Quality Indicators")
    summary.append("")

    if 'silhouette_score' in metrics:
        score = metrics['silhouette_score']
        if score >= 0.5:
            quality = "Excellent"
        elif score >= 0.3:
            quality = "Good"
        elif score >= 0.1:
            quality = "Fair"
        else:
            quality = "Weak"
        summary.append(f"- **Clustering Quality**: {quality} (Silhouette: {score:.3f})")

    summary.append(f"- **Multi-Coding Rate**: {(results_df['num_codes'] > 1).sum() / len(results_df) * 100:.1f}%")
    summary.append(f"- **Uncoded Responses**: {metrics.get('uncoded_count', 0):,}")
    summary.append("")

    # Recommendations
    summary.append("## Recommendations")
    summary.append("")

    uncoded_pct = 100 - metrics.get('coverage_pct', 100)
    if uncoded_pct > 20:
        summary.append("- **High uncoded rate** ({:.1f}%): Consider lowering confidence threshold or adding more codes".format(uncoded_pct))

    if metrics.get('avg_confidence', 1.0) < 0.5:
        summary.append("- **Low average confidence**: Review model fit and consider alternative ML methods")

    active_codes = sum(1 for info in coder.codebook.values() if info['count'] > 0)
    if active_codes < coder.n_codes * 0.7:
        summary.append(f"- **Low code utilization** ({active_codes}/{coder.n_codes}): Consider reducing number of codes")

    multi_coded = (results_df['num_codes'] > 1).sum()
    if multi_coded / len(results_df) > 0.5:
        summary.append("- **High multi-coding rate**: Responses are nuanced; ensure codes are distinct")

    if not any([uncoded_pct > 20, metrics.get('avg_confidence', 1.0) < 0.5, active_codes < coder.n_codes * 0.7]):
        summary.append("- Analysis quality is strong; proceed with human validation")

    summary.append("")
    summary.append("**Next Steps:**")
    summary.append("1. Review representative quotes for each code")
    summary.append("2. Validate auto-generated code labels")
    summary.append("3. Examine uncoded and low-confidence responses")
    summary.append("4. Refine code structure (merge/split as needed)")
    summary.append("5. Document human review decisions in audit trail")
    summary.append("")

    # Include methods section if requested
    if include_methods:
        summary.append("---")
        summary.append("")
        methods_doc = generate_methods_documentation(
            coder,
            results_df,
            metrics,
            project_name="Open-Ended Coding Analysis"
        )
        summary.append(methods_doc)

    return "\n".join(summary)
