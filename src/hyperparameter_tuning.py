"""
Hyperparameter Tuning Module for ML-Based Open Coding.

This module provides automatic hyperparameter optimization using Optuna
for all clustering methods in the system:
- TF-IDF + K-Means
- LDA (Latent Dirichlet Allocation)
- LSTM + K-Means
- BERT + K-Means
- SVM (Spectral Clustering)

Features:
- Bayesian optimization via Optuna
- Method-specific search spaces
- Multi-objective optimization (silhouette, coherence, etc.)
- Cross-validation support
- Progress callbacks for UI integration
- Caching for expensive embedding computations

Usage:
    from src.hyperparameter_tuning import HyperparameterTuner, TuningConfig

    tuner = HyperparameterTuner(
        method='tfidf_kmeans',
        n_trials=50,
        optimization_metric='silhouette'
    )
    best_params = tuner.tune(texts)
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd

# Suppress Optuna logs during optimization
logging.getLogger("optuna").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


class OptimizationMetric(Enum):
    """Available optimization metrics."""
    SILHOUETTE = "silhouette"
    CALINSKI_HARABASZ = "calinski_harabasz"
    DAVIES_BOULDIN = "davies_bouldin"  # Lower is better
    COMBINED = "combined"  # Weighted combination


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning.

    Attributes:
        n_trials: Number of optimization trials (default: 50)
        timeout: Maximum optimization time in seconds (default: 300)
        optimization_metric: Metric to optimize (default: silhouette)
        n_jobs: Number of parallel jobs (-1 for all cores, default: 1)
        random_state: Random seed for reproducibility
        early_stopping_rounds: Stop if no improvement after N trials
        cv_folds: Number of cross-validation folds (0 for no CV)
        tune_n_codes: Whether to tune number of clusters (default: True)
        min_codes: Minimum number of clusters to test
        max_codes: Maximum number of clusters to test
        tune_vectorizer: Whether to tune vectorizer params (default: True)
        tune_model: Whether to tune model-specific params (default: True)
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
    """
    n_trials: int = 50
    timeout: Optional[int] = 300
    optimization_metric: OptimizationMetric = OptimizationMetric.SILHOUETTE
    n_jobs: int = 1
    random_state: int = 42
    early_stopping_rounds: int = 10
    cv_folds: int = 0
    tune_n_codes: bool = True
    min_codes: int = 3
    max_codes: int = 15
    tune_vectorizer: bool = True
    tune_model: bool = True
    verbose: int = 1


@dataclass
class TuningResult:
    """Results from hyperparameter tuning.

    Attributes:
        best_params: Best hyperparameter configuration
        best_score: Best optimization metric value
        optimization_history: List of (trial_number, score, params) tuples
        n_trials_completed: Total number of trials run
        optimization_time: Total optimization time in seconds
        method: The ML method that was tuned
        metric_name: Name of the optimization metric used
        all_metrics: Dictionary of all computed metrics for best params
    """
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Tuple[int, float, Dict[str, Any]]]
    n_trials_completed: int
    optimization_time: float
    method: str
    metric_name: str
    all_metrics: Dict[str, float] = field(default_factory=dict)

    def get_summary(self) -> str:
        """Get a human-readable summary of tuning results."""
        lines = [
            "=" * 60,
            "HYPERPARAMETER TUNING RESULTS",
            "=" * 60,
            f"Method: {self.method}",
            f"Optimization Metric: {self.metric_name}",
            f"Best Score: {self.best_score:.4f}",
            f"Trials Completed: {self.n_trials_completed}",
            f"Optimization Time: {self.optimization_time:.1f}s",
            "",
            "Best Parameters:",
        ]
        for param, value in sorted(self.best_params.items()):
            lines.append(f"  {param}: {value}")

        if self.all_metrics:
            lines.append("")
            lines.append("All Metrics (best params):")
            for metric, value in sorted(self.all_metrics.items()):
                lines.append(f"  {metric}: {value:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)


class SearchSpaceDefinition:
    """Defines hyperparameter search spaces for each ML method."""

    @staticmethod
    def get_vectorizer_space(trial, method: str) -> Dict[str, Any]:
        """Get search space for TF-IDF/CountVectorizer parameters."""
        params = {}

        # Common vectorizer parameters
        params['max_features'] = trial.suggest_int('max_features', 500, 5000, step=500)
        params['min_df'] = trial.suggest_int('min_df', 1, 10)
        params['max_df'] = trial.suggest_float('max_df', 0.5, 0.95, step=0.05)

        # N-gram range
        ngram_max = trial.suggest_int('ngram_max', 1, 3)
        params['ngram_range'] = (1, ngram_max)

        # TF-IDF specific
        if method not in ['lda']:
            params['sublinear_tf'] = trial.suggest_categorical('sublinear_tf', [True, False])

        return params

    @staticmethod
    def get_kmeans_space(trial) -> Dict[str, Any]:
        """Get search space for K-Means parameters."""
        params = {}
        params['n_init'] = trial.suggest_int('kmeans_n_init', 5, 20)
        params['max_iter'] = trial.suggest_int('kmeans_max_iter', 100, 500, step=50)
        params['algorithm'] = trial.suggest_categorical('kmeans_algorithm', ['lloyd', 'elkan'])
        return params

    @staticmethod
    def get_lda_space(trial) -> Dict[str, Any]:
        """Get search space for LDA parameters."""
        params = {}
        params['max_iter'] = trial.suggest_int('lda_max_iter', 10, 50, step=5)
        params['learning_method'] = trial.suggest_categorical(
            'lda_learning_method', ['batch', 'online']
        )
        params['learning_decay'] = trial.suggest_float('lda_learning_decay', 0.5, 0.9, step=0.1)

        # Document-topic and topic-word priors
        params['doc_topic_prior'] = trial.suggest_float('lda_alpha', 0.01, 1.0, log=True)
        params['topic_word_prior'] = trial.suggest_float('lda_beta', 0.01, 1.0, log=True)

        return params

    @staticmethod
    def get_spectral_space(trial) -> Dict[str, Any]:
        """Get search space for Spectral Clustering (SVM) parameters."""
        params = {}
        params['n_init'] = trial.suggest_int('spectral_n_init', 5, 20)
        params['affinity'] = trial.suggest_categorical(
            'spectral_affinity', ['rbf', 'nearest_neighbors']
        )

        if params['affinity'] == 'rbf':
            params['gamma'] = trial.suggest_float('spectral_gamma', 0.01, 10.0, log=True)
        else:
            params['n_neighbors'] = trial.suggest_int('spectral_n_neighbors', 5, 30)

        params['assign_labels'] = trial.suggest_categorical(
            'spectral_assign_labels', ['kmeans', 'discretize']
        )

        return params

    @staticmethod
    def get_lstm_space(trial) -> Dict[str, Any]:
        """Get search space for LSTM embedding parameters."""
        params = {}
        params['embedding_dim'] = trial.suggest_int('lstm_embedding_dim', 50, 200, step=25)
        params['lstm_units'] = trial.suggest_int('lstm_units', 64, 256, step=32)
        params['max_sequence_length'] = trial.suggest_int('lstm_max_seq_length', 50, 200, step=25)
        params['epochs'] = trial.suggest_int('lstm_epochs', 5, 20)
        params['batch_size'] = trial.suggest_categorical('lstm_batch_size', [16, 32, 64])
        return params

    @staticmethod
    def get_bert_space(trial) -> Dict[str, Any]:
        """Get search space for BERT embedding parameters."""
        params = {}
        # Different BERT model variants
        params['model_name'] = trial.suggest_categorical(
            'bert_model_name',
            [
                'all-MiniLM-L6-v2',      # Fast, good quality
                'all-mpnet-base-v2',      # Higher quality, slower
                'paraphrase-MiniLM-L6-v2' # Good for paraphrase detection
            ]
        )
        params['batch_size'] = trial.suggest_categorical('bert_batch_size', [16, 32, 64])
        return params


class HyperparameterTuner:
    """
    Automatic hyperparameter optimization for ML-based open coding.

    Uses Optuna's Bayesian optimization to find optimal hyperparameters
    for any clustering method in the system.

    Example:
        >>> tuner = HyperparameterTuner(method='tfidf_kmeans')
        >>> result = tuner.tune(texts, n_codes_hint=10)
        >>> print(result.get_summary())
        >>>
        >>> # Use best params with MLOpenCoder
        >>> from helpers.analysis import run_ml_analysis
        >>> coder, results, metrics = run_ml_analysis(
        ...     df, text_column,
        ...     n_codes=result.best_params['n_codes'],
        ...     preprocessing_override=result.best_params.get('vectorizer_params'),
        ...     embedding_kwargs=result.best_params.get('embedding_params')
        ... )
    """

    # Supported methods
    SUPPORTED_METHODS = ['tfidf_kmeans', 'lda', 'lstm_kmeans', 'bert_kmeans', 'svm']

    def __init__(
        self,
        method: str = 'tfidf_kmeans',
        config: Optional[TuningConfig] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            method: ML method to tune. One of:
                - 'tfidf_kmeans': TF-IDF + K-Means
                - 'lda': Latent Dirichlet Allocation
                - 'lstm_kmeans': LSTM embeddings + K-Means
                - 'bert_kmeans': BERT embeddings + K-Means
                - 'svm': SVM-based Spectral Clustering
            config: Tuning configuration (uses defaults if None)
            progress_callback: Optional callback for progress updates
                signature: callback(progress: float, message: str)
        """
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                f"Supported methods: {self.SUPPORTED_METHODS}"
            )

        self.method = method
        self.config = config or TuningConfig()
        self.progress_callback = progress_callback

        # Lazy imports to avoid loading heavy libraries unnecessarily
        self._optuna = None
        self._study = None

        # Cache for expensive computations (embeddings)
        self._embedding_cache = {}

    def _import_optuna(self):
        """Lazy import of Optuna."""
        if self._optuna is None:
            try:
                import optuna
                self._optuna = optuna
                # Configure Optuna logging
                optuna.logging.set_verbosity(
                    optuna.logging.WARNING if self.config.verbose < 2
                    else optuna.logging.INFO
                )
            except ImportError:
                raise ImportError(
                    "Optuna is required for hyperparameter tuning. "
                    "Install with: pip install optuna"
                )
        return self._optuna

    def _report_progress(self, progress: float, message: str):
        """Report progress to callback if available."""
        if self.progress_callback:
            self.progress_callback(progress, message)
        if self.config.verbose >= 1:
            logger.info(f"[{progress*100:.0f}%] {message}")

    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts for clustering."""
        import re

        processed = []
        for text in texts:
            if pd.isna(text):
                processed.append("")
                continue
            text = str(text).lower()
            # Keep letters (including accented chars) and whitespace
            text = re.sub(r'[^a-zßà-öø-ÿ\s]', ' ', text)
            text = ' '.join(text.split())
            processed.append(text)
        return processed

    def _compute_metrics(
        self,
        feature_matrix,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute clustering quality metrics."""
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score
        )

        metrics = {}

        # Need at least 2 clusters
        n_unique = len(set(labels))
        if n_unique < 2:
            return {'silhouette': -1.0, 'calinski_harabasz': 0.0, 'davies_bouldin': float('inf')}

        # Convert sparse to dense if needed
        if hasattr(feature_matrix, 'toarray'):
            feature_dense = feature_matrix.toarray()
        else:
            feature_dense = feature_matrix

        try:
            metrics['silhouette'] = silhouette_score(feature_matrix, labels)
        except Exception:
            metrics['silhouette'] = -1.0

        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(feature_dense, labels)
        except Exception:
            metrics['calinski_harabasz'] = 0.0

        try:
            metrics['davies_bouldin'] = davies_bouldin_score(feature_dense, labels)
        except Exception:
            metrics['davies_bouldin'] = float('inf')

        # Combined score (normalized)
        # Silhouette: [-1, 1] -> normalize to [0, 1]
        # Davies-Bouldin: lower is better, invert and normalize
        sil_norm = (metrics['silhouette'] + 1) / 2
        db_norm = 1 / (1 + metrics['davies_bouldin'])  # Inverse, bounded
        ch_norm = min(metrics['calinski_harabasz'] / 1000, 1.0)  # Rough normalization

        metrics['combined'] = 0.5 * sil_norm + 0.3 * db_norm + 0.2 * ch_norm

        return metrics

    def _get_score(self, metrics: Dict[str, float]) -> float:
        """Get the optimization score based on configured metric."""
        metric_name = self.config.optimization_metric.value

        if metric_name == 'davies_bouldin':
            # Lower is better, so negate
            return -metrics.get('davies_bouldin', float('inf'))

        return metrics.get(metric_name, -float('inf'))

    def _create_objective(self, texts: List[str], n_codes_hint: Optional[int] = None):
        """Create the Optuna objective function."""
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.cluster import KMeans, SpectralClustering
        from sklearn.decomposition import LatentDirichletAllocation

        processed_texts = self._preprocess_texts(texts)
        n_samples = len(processed_texts)

        # Determine n_codes range
        if self.config.tune_n_codes:
            min_codes = max(2, self.config.min_codes)
            max_codes = min(self.config.max_codes, n_samples - 1)
        else:
            # Use hint or default
            fixed_codes = n_codes_hint or 10
            min_codes = max_codes = fixed_codes

        def objective(trial):
            try:
                params = {}

                # 1. Number of clusters
                if self.config.tune_n_codes:
                    n_codes = trial.suggest_int('n_codes', min_codes, max_codes)
                else:
                    n_codes = min_codes
                params['n_codes'] = n_codes

                # 2. Vectorizer parameters (for TF-IDF based methods)
                vectorizer_params = {}
                if self.config.tune_vectorizer and self.method in ['tfidf_kmeans', 'lda', 'svm']:
                    vectorizer_params = SearchSpaceDefinition.get_vectorizer_space(
                        trial, self.method
                    )
                    params['vectorizer_params'] = vectorizer_params

                # 3. Model-specific parameters
                model_params = {}
                embedding_params = {}

                if self.config.tune_model:
                    if self.method == 'tfidf_kmeans':
                        model_params = SearchSpaceDefinition.get_kmeans_space(trial)
                    elif self.method == 'lda':
                        model_params = SearchSpaceDefinition.get_lda_space(trial)
                    elif self.method == 'svm':
                        model_params = SearchSpaceDefinition.get_spectral_space(trial)
                    elif self.method == 'lstm_kmeans':
                        embedding_params = SearchSpaceDefinition.get_lstm_space(trial)
                        model_params = SearchSpaceDefinition.get_kmeans_space(trial)
                    elif self.method == 'bert_kmeans':
                        embedding_params = SearchSpaceDefinition.get_bert_space(trial)
                        model_params = SearchSpaceDefinition.get_kmeans_space(trial)

                    params['model_params'] = model_params
                    if embedding_params:
                        params['embedding_params'] = embedding_params

                # 4. Create feature matrix
                if self.method in ['tfidf_kmeans', 'svm']:
                    vec_kwargs = {
                        'max_features': vectorizer_params.get('max_features', 1000),
                        'min_df': vectorizer_params.get('min_df', 2),
                        'max_df': vectorizer_params.get('max_df', 0.8),
                        'ngram_range': vectorizer_params.get('ngram_range', (1, 2)),
                        'stop_words': 'english',
                        'sublinear_tf': vectorizer_params.get('sublinear_tf', False)
                    }
                    vectorizer = TfidfVectorizer(**vec_kwargs)
                    feature_matrix = vectorizer.fit_transform(processed_texts)

                elif self.method == 'lda':
                    vec_kwargs = {
                        'max_features': vectorizer_params.get('max_features', 1000),
                        'min_df': vectorizer_params.get('min_df', 2),
                        'max_df': vectorizer_params.get('max_df', 0.8),
                        'ngram_range': vectorizer_params.get('ngram_range', (1, 2)),
                        'stop_words': 'english'
                    }
                    vectorizer = CountVectorizer(**vec_kwargs)
                    feature_matrix = vectorizer.fit_transform(processed_texts)

                elif self.method in ['lstm_kmeans', 'bert_kmeans']:
                    # Use cached embeddings if available with same params
                    cache_key = f"{self.method}_{hash(str(embedding_params))}"

                    if cache_key in self._embedding_cache:
                        feature_matrix = self._embedding_cache[cache_key]
                    else:
                        from src.embeddings import get_embedder

                        embed_type = 'lstm' if self.method == 'lstm_kmeans' else 'bert'
                        embedder = get_embedder(embed_type, **embedding_params)
                        feature_matrix = embedder.fit_transform(processed_texts)

                        # Cache embeddings (limit cache size)
                        if len(self._embedding_cache) > 5:
                            self._embedding_cache.clear()
                        self._embedding_cache[cache_key] = feature_matrix

                # 5. Train clustering model
                if self.method == 'lda':
                    lda_kwargs = {
                        'n_components': n_codes,
                        'random_state': self.config.random_state,
                        'max_iter': model_params.get('max_iter', 20),
                        'learning_method': model_params.get('learning_method', 'batch'),
                        'learning_decay': model_params.get('learning_decay', 0.7),
                        'doc_topic_prior': model_params.get('doc_topic_prior'),
                        'topic_word_prior': model_params.get('topic_word_prior')
                    }
                    model = LatentDirichletAllocation(**lda_kwargs)
                    doc_topics = model.fit_transform(feature_matrix)
                    labels = doc_topics.argmax(axis=1)

                elif self.method == 'svm':
                    # Prepare spectral clustering kwargs
                    spectral_kwargs = {
                        'n_clusters': n_codes,
                        'random_state': self.config.random_state,
                        'n_init': model_params.get('n_init', 10),
                        'affinity': model_params.get('affinity', 'rbf'),
                        'assign_labels': model_params.get('assign_labels', 'kmeans')
                    }
                    if spectral_kwargs['affinity'] == 'rbf':
                        spectral_kwargs['gamma'] = model_params.get('gamma', 1.0)
                    elif spectral_kwargs['affinity'] == 'nearest_neighbors':
                        spectral_kwargs['n_neighbors'] = model_params.get('n_neighbors', 10)

                    model = SpectralClustering(**spectral_kwargs)
                    if hasattr(feature_matrix, 'toarray'):
                        labels = model.fit_predict(feature_matrix.toarray())
                    else:
                        labels = model.fit_predict(feature_matrix)

                else:  # tfidf_kmeans, lstm_kmeans, bert_kmeans
                    kmeans_kwargs = {
                        'n_clusters': n_codes,
                        'random_state': self.config.random_state,
                        'n_init': model_params.get('n_init', 10),
                        'max_iter': model_params.get('max_iter', 300),
                        'algorithm': model_params.get('algorithm', 'lloyd')
                    }
                    model = KMeans(**kmeans_kwargs)
                    labels = model.fit_predict(feature_matrix)

                # 6. Compute metrics
                metrics = self._compute_metrics(feature_matrix, labels)
                score = self._get_score(metrics)

                # Store all metrics as trial user attributes
                for metric_name, metric_value in metrics.items():
                    trial.set_user_attr(metric_name, metric_value)

                return score

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('-inf')

        return objective

    def tune(
        self,
        texts: Union[List[str], pd.Series],
        n_codes_hint: Optional[int] = None
    ) -> TuningResult:
        """
        Run hyperparameter optimization.

        Args:
            texts: List or Series of text responses to cluster
            n_codes_hint: Hint for number of clusters (used if tune_n_codes=False)

        Returns:
            TuningResult with best parameters and optimization history
        """
        import time

        optuna = self._import_optuna()

        # Convert to list if Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()

        # Filter empty texts
        texts = [t for t in texts if t and str(t).strip()]

        if len(texts) < 10:
            raise ValueError("Need at least 10 non-empty texts for tuning")

        self._report_progress(0.0, f"Starting hyperparameter tuning for {self.method}")

        # Create study
        sampler = optuna.samplers.TPESampler(seed=self.config.random_state)

        # Early stopping callback
        early_stop_callback = None
        if self.config.early_stopping_rounds > 0:
            early_stop_callback = optuna.callbacks.EarlyStoppingCallback(
                self.config.early_stopping_rounds,
                direction="maximize"
            )

        self._study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"{self.method}_tuning"
        )

        # Create objective
        objective = self._create_objective(texts, n_codes_hint)

        # Progress callback for Optuna
        optimization_history = []
        start_time = time.time()

        def optuna_callback(study, trial):
            progress = (trial.number + 1) / self.config.n_trials
            best_value = study.best_value if study.best_trial else 0
            self._report_progress(
                progress,
                f"Trial {trial.number + 1}/{self.config.n_trials} - "
                f"Best: {best_value:.4f}"
            )
            optimization_history.append((
                trial.number,
                trial.value if trial.value is not None else float('-inf'),
                trial.params.copy()
            ))

        callbacks = [optuna_callback]
        if early_stop_callback:
            callbacks.append(early_stop_callback)

        # Run optimization
        try:
            self._study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs,
                callbacks=callbacks,
                show_progress_bar=self.config.verbose >= 2
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

        optimization_time = time.time() - start_time

        # Get best trial
        best_trial = self._study.best_trial
        best_params = best_trial.params.copy()

        # Reconstruct structured params
        structured_params = {'n_codes': best_params.get('n_codes', 10)}

        # Extract vectorizer params
        vectorizer_keys = ['max_features', 'min_df', 'max_df', 'ngram_max', 'sublinear_tf']
        vectorizer_params = {}
        for key in vectorizer_keys:
            if key in best_params:
                if key == 'ngram_max':
                    vectorizer_params['ngram_range'] = (1, best_params[key])
                else:
                    vectorizer_params[key] = best_params[key]
        if vectorizer_params:
            structured_params['vectorizer_params'] = vectorizer_params

        # Extract model params
        model_params = {}
        embedding_params = {}
        for key, value in best_params.items():
            if key.startswith('kmeans_'):
                model_params[key.replace('kmeans_', '')] = value
            elif key.startswith('lda_'):
                param_name = key.replace('lda_', '')
                if param_name == 'alpha':
                    model_params['doc_topic_prior'] = value
                elif param_name == 'beta':
                    model_params['topic_word_prior'] = value
                else:
                    model_params[param_name] = value
            elif key.startswith('spectral_'):
                model_params[key.replace('spectral_', '')] = value
            elif key.startswith('lstm_'):
                embedding_params[key.replace('lstm_', '')] = value
            elif key.startswith('bert_'):
                embedding_params[key.replace('bert_', '')] = value

        if model_params:
            structured_params['model_params'] = model_params
        if embedding_params:
            structured_params['embedding_params'] = embedding_params

        # Get all metrics from best trial
        all_metrics = {}
        for key in ['silhouette', 'calinski_harabasz', 'davies_bouldin', 'combined']:
            if key in best_trial.user_attrs:
                all_metrics[key] = best_trial.user_attrs[key]

        self._report_progress(1.0, "Hyperparameter tuning complete!")

        return TuningResult(
            best_params=structured_params,
            best_score=best_trial.value,
            optimization_history=optimization_history,
            n_trials_completed=len(self._study.trials),
            optimization_time=optimization_time,
            method=self.method,
            metric_name=self.config.optimization_metric.value,
            all_metrics=all_metrics
        )

    def get_study(self):
        """Get the underlying Optuna study for advanced analysis."""
        return self._study

    def get_importance(self) -> Dict[str, float]:
        """Get parameter importance scores."""
        if self._study is None:
            return {}

        try:
            optuna = self._import_optuna()
            importance = optuna.importance.get_param_importances(self._study)
            return dict(importance)
        except Exception as e:
            logger.warning(f"Could not compute parameter importance: {e}")
            return {}


def tune_hyperparameters(
    texts: Union[List[str], pd.Series],
    method: str = 'tfidf_kmeans',
    n_trials: int = 50,
    timeout: Optional[int] = 300,
    optimization_metric: str = 'silhouette',
    n_codes_hint: Optional[int] = None,
    tune_n_codes: bool = True,
    min_codes: int = 3,
    max_codes: int = 15,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    verbose: int = 1
) -> TuningResult:
    """
    Convenience function for hyperparameter tuning.

    This is the main entry point for hyperparameter optimization.

    Args:
        texts: Text responses to cluster
        method: ML method ('tfidf_kmeans', 'lda', 'lstm_kmeans', 'bert_kmeans', 'svm')
        n_trials: Number of optimization trials
        timeout: Maximum time in seconds (None for no limit)
        optimization_metric: Metric to optimize ('silhouette', 'calinski_harabasz',
            'davies_bouldin', 'combined')
        n_codes_hint: Hint for number of clusters
        tune_n_codes: Whether to tune the number of clusters
        min_codes: Minimum clusters to test
        max_codes: Maximum clusters to test
        progress_callback: Optional progress callback
        verbose: Verbosity level

    Returns:
        TuningResult with best parameters and optimization history

    Example:
        >>> result = tune_hyperparameters(
        ...     texts=df['response'],
        ...     method='tfidf_kmeans',
        ...     n_trials=30,
        ...     optimization_metric='silhouette'
        ... )
        >>> print(f"Best n_codes: {result.best_params['n_codes']}")
        >>> print(f"Best score: {result.best_score:.4f}")
    """
    # Convert metric string to enum
    try:
        metric_enum = OptimizationMetric(optimization_metric)
    except ValueError:
        raise ValueError(
            f"Invalid optimization_metric: {optimization_metric}. "
            f"Valid options: {[m.value for m in OptimizationMetric]}"
        )

    config = TuningConfig(
        n_trials=n_trials,
        timeout=timeout,
        optimization_metric=metric_enum,
        tune_n_codes=tune_n_codes,
        min_codes=min_codes,
        max_codes=max_codes,
        verbose=verbose
    )

    tuner = HyperparameterTuner(
        method=method,
        config=config,
        progress_callback=progress_callback
    )

    return tuner.tune(texts, n_codes_hint=n_codes_hint)


def get_default_params(method: str) -> Dict[str, Any]:
    """
    Get default hyperparameters for a method.

    These are reasonable defaults that work well across most datasets.
    Use hyperparameter tuning for optimal results on specific data.

    Args:
        method: ML method name

    Returns:
        Dictionary of default parameters
    """
    defaults = {
        'tfidf_kmeans': {
            'n_codes': 10,
            'vectorizer_params': {
                'max_features': 1000,
                'min_df': 2,
                'max_df': 0.8,
                'ngram_range': (1, 2),
                'sublinear_tf': False
            },
            'model_params': {
                'n_init': 10,
                'max_iter': 300,
                'algorithm': 'lloyd'
            }
        },
        'lda': {
            'n_codes': 10,
            'vectorizer_params': {
                'max_features': 1000,
                'min_df': 2,
                'max_df': 0.8,
                'ngram_range': (1, 2)
            },
            'model_params': {
                'max_iter': 20,
                'learning_method': 'batch',
                'learning_decay': 0.7
            }
        },
        'svm': {
            'n_codes': 10,
            'vectorizer_params': {
                'max_features': 1000,
                'min_df': 2,
                'max_df': 0.8,
                'ngram_range': (1, 2),
                'sublinear_tf': False
            },
            'model_params': {
                'n_init': 10,
                'affinity': 'rbf',
                'assign_labels': 'kmeans'
            }
        },
        'lstm_kmeans': {
            'n_codes': 10,
            'embedding_params': {
                'embedding_dim': 100,
                'lstm_units': 128,
                'max_sequence_length': 100,
                'epochs': 10,
                'batch_size': 32
            },
            'model_params': {
                'n_init': 10,
                'max_iter': 300,
                'algorithm': 'lloyd'
            }
        },
        'bert_kmeans': {
            'n_codes': 10,
            'embedding_params': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32
            },
            'model_params': {
                'n_init': 10,
                'max_iter': 300,
                'algorithm': 'lloyd'
            }
        }
    }

    if method not in defaults:
        raise ValueError(f"Unknown method: {method}")

    return defaults[method]
