"""
Semantic embedding methods for ML open-ended coding analysis.

This module provides alternative text representation methods to complement
the default TF-IDF approach. All methods work offline without requiring API keys.

Classes:
    - BaseEmbedder: Abstract base class defining the embedding interface
    - SentenceBERTEmbedder: Sentence-level semantic embeddings (offline)
    - Word2VecEmbedder: Average word vectors using Word2Vec
    - FastTextEmbedder: Subword-aware embeddings using FastText

Usage:
    # Using SentenceBERT (best for semantic similarity)
    embedder = SentenceBERTEmbedder(model_name='all-MiniLM-L6-v2')
    vectors = embedder.fit_transform(texts)

    # Using Word2Vec (faster, good for keyword-based themes)
    embedder = Word2VecEmbedder(vector_size=100, min_count=2)
    vectors = embedder.fit_transform(texts)

    # Using FastText (best for misspellings and rare words)
    embedder = FastTextEmbedder(vector_size=100, min_count=2)
    vectors = embedder.fit_transform(texts)

Trade-offs:
    - TF-IDF: Fast, interpretable, no dependencies, but ignores semantics
    - SentenceBERT: Best semantic quality, slower, requires sentence-transformers
    - Word2Vec: Good balance, handles synonyms, requires training data
    - FastText: Handles typos/rare words via subwords, similar to Word2Vec
"""

import os

# Fix Keras 3 compatibility issue with HuggingFace Transformers
# Keras 3 is not yet supported by transformers, so we configure TensorFlow
# to use the backwards-compatible tf-keras package instead.
# See: https://github.com/huggingface/transformers/issues/27850
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
import warnings
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """
    Abstract base class for text embeddings.

    Provides a scikit-learn-compatible interface for text vectorization.
    All embedders should implement fit(), transform(), and fit_transform().
    """

    def __init__(self):
        self.is_fitted_ = False
        self.feature_names_ = None

    @abstractmethod
    def fit(self, texts: List[str]) -> 'BaseEmbedder':
        """
        Fit the embedder on a collection of texts.

        Args:
            texts: List of text strings to fit on

        Returns:
            Self (fitted embedder)
        """
        pass

    @abstractmethod
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into embedding vectors.

        Args:
            texts: List of text strings to transform

        Returns:
            Array of shape (n_samples, n_features)
        """
        pass

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the embedder and transform texts in one step.

        Args:
            texts: List of text strings

        Returns:
            Array of shape (n_samples, n_features)
        """
        return self.fit(texts).transform(texts)

    def get_feature_names_out(self) -> np.ndarray:
        """
        Get feature names for compatibility with scikit-learn.

        Returns:
            Array of feature names
        """
        if self.feature_names_ is None:
            # Generate generic feature names
            n_features = self._get_n_features()
            self.feature_names_ = np.array([f"embedding_{i}" for i in range(n_features)])
        return self.feature_names_

    @abstractmethod
    def _get_n_features(self) -> int:
        """Get the number of features in the embedding."""
        pass


class SentenceBERTEmbedder(BaseEmbedder):
    """
    Sentence-level semantic embeddings using sentence-transformers.

    This embedder uses pre-trained transformer models to create dense
    semantic representations of text. It captures meaning better than
    TF-IDF but is computationally more expensive.

    Attributes:
        model_name: Name of the sentence-transformer model
        model: Loaded sentence-transformer model
        embedding_dim: Dimension of the embedding vectors

    Recommended models:
        - 'all-MiniLM-L6-v2': Fast, good quality (384 dimensions)
        - 'all-mpnet-base-v2': Best quality, slower (768 dimensions)
        - 'paraphrase-MiniLM-L6-v2': Good for paraphrase detection

    Performance:
        - Speed: Moderate to slow (depends on model size)
        - Quality: Excellent semantic understanding
        - Memory: Moderate (model size ~80-400 MB)
        - Offline: Yes (downloads model on first use, then cached)

    Example:
        >>> embedder = SentenceBERTEmbedder(model_name='all-MiniLM-L6-v2')
        >>> vectors = embedder.fit_transform(["remote work", "working from home"])
        >>> # These will have high cosine similarity due to semantic meaning
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):
        """
        Initialize SentenceBERT embedder.

        Args:
            model_name: Pre-trained model name from sentence-transformers
            device: Device to use ('cpu' or 'cuda'). Auto-detects if None
            batch_size: Batch size for encoding (larger = faster but more memory)
            show_progress_bar: Whether to show progress during encoding

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        super().__init__()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceBERTEmbedder. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.model = None
        self.embedding_dim = None

    def fit(self, texts: List[str]) -> 'SentenceBERTEmbedder':
        """
        Fit the SentenceBERT embedder (loads the model).

        Args:
            texts: List of texts (used to verify model works)

        Returns:
            Self (fitted embedder)
        """
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading SentenceBERT model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Get embedding dimension from a test encoding
        test_embedding = self.model.encode(["test"], show_progress_bar=False)
        self.embedding_dim = test_embedding.shape[1]

        self.is_fitted_ = True
        logger.info(f"SentenceBERT model loaded. Embedding dimension: {self.embedding_dim}")

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into SentenceBERT embeddings.

        Args:
            texts: List of text strings

        Returns:
            Array of shape (n_samples, embedding_dim)

        Raises:
            ValueError: If embedder is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("SentenceBERTEmbedder must be fitted before transform")

        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        # Convert to strings and handle None/NaN
        texts = [str(t) if pd.notna(t) else "" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True
        )

        return embeddings

    def _get_n_features(self) -> int:
        """Get the number of features (embedding dimension)."""
        if self.embedding_dim is None:
            raise ValueError("Model not fitted yet")
        return self.embedding_dim


class Word2VecEmbedder(BaseEmbedder):
    """
    Word-level embeddings using Word2Vec, averaged per document.

    This embedder trains a Word2Vec model on the corpus and represents
    each document as the average of its word vectors. It's faster than
    SentenceBERT and handles synonyms better than TF-IDF.

    Attributes:
        vector_size: Dimension of word vectors
        window: Context window size for training
        min_count: Minimum word frequency to include
        model: Trained gensim Word2Vec model

    Performance:
        - Speed: Fast (after initial training)
        - Quality: Good for synonym/semantic similarity
        - Memory: Low (small model size)
        - Offline: Yes (trains on your data)

    Trade-offs:
        - Requires sufficient training data (ideally 1000+ responses)
        - Averages word vectors (loses word order information)
        - May not work well with very short texts

    Example:
        >>> embedder = Word2VecEmbedder(vector_size=100, window=5)
        >>> vectors = embedder.fit_transform(texts)
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        sg: int = 0  # 0 = CBOW, 1 = Skip-gram
    ):
        """
        Initialize Word2Vec embedder.

        Args:
            vector_size: Dimension of word vectors (50-300 typical)
            window: Context window size (5-10 typical)
            min_count: Minimum word frequency (2-5 typical)
            workers: Number of worker threads for training
            epochs: Number of training epochs
            sg: Training algorithm (0=CBOW, 1=Skip-gram)

        Raises:
            ImportError: If gensim is not installed
        """
        super().__init__()

        try:
            from gensim.models import Word2Vec
        except ImportError:
            raise ImportError(
                "gensim is required for Word2VecEmbedder. "
                "Install with: pip install gensim"
            )

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.sg = sg
        self.model = None

    def fit(self, texts: List[str]) -> 'Word2VecEmbedder':
        """
        Fit Word2Vec model on the corpus.

        Args:
            texts: List of text strings

        Returns:
            Self (fitted embedder)
        """
        from gensim.models import Word2Vec
        from gensim.utils import simple_preprocess

        # Tokenize texts, treating None and empty strings as empty token lists
        tokenized = []
        for text in texts:
            if text is None or (isinstance(text, str) and not text.strip()):
                tokenized.append([])
            else:
                tokenized.append(simple_preprocess(str(text)))

        # Filter out empty token lists for training (Word2Vec can't train on empty sentences)
        training_sentences = [tokens for tokens in tokenized if tokens]

        logger.info(f"Training Word2Vec model (vector_size={self.vector_size}, epochs={self.epochs})")

        if not training_sentences:
            raise ValueError(
                "No valid tokens found after preprocessing. "
                "All texts are empty or too short for Word2Vec training."
            )

        # Build vocabulary first to check if any words pass min_count
        self.model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg
        )
        self.model.build_vocab(training_sentences)

        vocab_size = len(self.model.wv)

        if vocab_size == 0:
            # No words passed min_count threshold - train with min_count=1 as fallback
            warnings.warn(
                f"No words met min_count={self.min_count}. "
                f"Falling back to min_count=1 to build vocabulary.",
                UserWarning
            )
            self.model = Word2Vec(
                vector_size=self.vector_size,
                window=self.window,
                min_count=1,
                workers=self.workers,
                sg=self.sg
            )
            self.model.build_vocab(training_sentences)
            vocab_size = len(self.model.wv)

        # Train the model
        self.model.train(
            training_sentences,
            total_examples=self.model.corpus_count,
            epochs=self.epochs
        )

        self.is_fitted_ = True
        logger.info(f"Word2Vec model trained. Vocabulary size: {vocab_size}")

        if vocab_size < 50:
            warnings.warn(
                f"Small vocabulary size ({vocab_size} words). "
                "Consider lowering min_count or providing more diverse text data.",
                UserWarning
            )

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into Word2Vec embeddings (averaged word vectors).

        Args:
            texts: List of text strings

        Returns:
            Array of shape (n_samples, vector_size)

        Raises:
            ValueError: If embedder is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("Word2VecEmbedder must be fitted before transform")

        from gensim.utils import simple_preprocess

        embeddings = []

        for text in texts:
            if text is None or (isinstance(text, str) and not text.strip()):
                # None or empty text -> zero vector
                embeddings.append(np.zeros(self.vector_size))
                continue

            tokens = simple_preprocess(str(text))

            # Get vectors for words in vocabulary
            word_vectors = [
                self.model.wv[word]
                for word in tokens
                if word in self.model.wv
            ]

            if word_vectors:
                # Average word vectors
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # No words in vocabulary - use zero vector
                doc_vector = np.zeros(self.vector_size)

            embeddings.append(doc_vector)

        return np.array(embeddings)

    def _get_n_features(self) -> int:
        """Get the number of features (vector size)."""
        return self.vector_size


class LSTMEmbedder(BaseEmbedder):
    """
    LSTM-based text embeddings using a sequence autoencoder.

    This embedder trains an LSTM autoencoder on the corpus and uses the
    encoder's output as document representations. It captures sequential
    patterns and context in text that bag-of-words methods miss.

    Attributes:
        embedding_dim: Dimension of word embeddings
        lstm_units: Number of LSTM units (output dimension)
        max_sequence_length: Maximum sequence length for padding
        model: Trained Keras LSTM autoencoder

    Performance:
        - Speed: Moderate (requires training)
        - Quality: Good for capturing sequential patterns
        - Memory: Moderate (depends on vocabulary size)
        - Offline: Yes (trains on your data)

    Trade-offs:
        - Requires TensorFlow/Keras
        - Needs sufficient training data (ideally 500+ responses)
        - Captures word order (unlike bag-of-words methods)
        - Better for longer texts with sequential structure

    Example:
        >>> embedder = LSTMEmbedder(lstm_units=128, embedding_dim=100)
        >>> vectors = embedder.fit_transform(texts)
    """

    def __init__(
        self,
        embedding_dim: int = 100,
        lstm_units: int = 128,
        max_sequence_length: int = 100,
        max_vocab_size: int = 10000,
        epochs: int = 10,
        batch_size: int = 32
    ):
        """
        Initialize LSTM embedder.

        Args:
            embedding_dim: Dimension of word embeddings (50-200 typical)
            lstm_units: Number of LSTM units (64-256 typical)
            max_sequence_length: Maximum sequence length (100-500 typical)
            max_vocab_size: Maximum vocabulary size
            epochs: Number of training epochs
            batch_size: Batch size for training

        Raises:
            ImportError: If TensorFlow/Keras is not installed
        """
        super().__init__()

        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError(
                "TensorFlow is required for LSTMEmbedder. "
                "Install with: pip install tensorflow"
            )

        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_sequence_length = max_sequence_length
        self.max_vocab_size = max_vocab_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.encoder = None
        self.tokenizer = None

    def fit(self, texts: List[str]) -> 'LSTMEmbedder':
        """
        Fit LSTM autoencoder on the corpus.

        Args:
            texts: List of text strings

        Returns:
            Self (fitted embedder)
        """
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.layers import (
            Input, Embedding, LSTM, Dense, RepeatVector, TimeDistributed
        )
        from tensorflow.keras.models import Model

        logger.info(f"Training LSTM embedder (units={self.lstm_units}, epochs={self.epochs})")

        # Tokenize texts
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        vocab_size = min(len(self.tokenizer.word_index) + 1, self.max_vocab_size)
        logger.info(f"Vocabulary size: {vocab_size}")

        # Build LSTM autoencoder
        # Encoder
        encoder_input = Input(shape=(self.max_sequence_length,))
        x = Embedding(vocab_size, self.embedding_dim, mask_zero=True)(encoder_input)
        x = LSTM(self.lstm_units, return_sequences=False)(x)
        encoder_output = Dense(self.lstm_units, activation='relu')(x)

        # Decoder
        x = RepeatVector(self.max_sequence_length)(encoder_output)
        x = LSTM(self.lstm_units, return_sequences=True)(x)
        decoder_output = TimeDistributed(Dense(vocab_size, activation='softmax'))(x)

        # Full autoencoder
        self.model = Model(encoder_input, decoder_output)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

        # Encoder-only model for embeddings
        self.encoder = Model(encoder_input, encoder_output)

        # Train autoencoder
        target = padded_sequences.reshape((*padded_sequences.shape, 1))

        # Suppress verbose output
        self.model.fit(
            padded_sequences,
            target,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0
        )

        self.is_fitted_ = True
        logger.info(f"LSTM embedder trained. Output dimension: {self.lstm_units}")

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into LSTM embeddings.

        Args:
            texts: List of text strings

        Returns:
            Array of shape (n_samples, lstm_units)

        Raises:
            ValueError: If embedder is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("LSTMEmbedder must be fitted before transform")

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Convert to strings and handle None/NaN
        texts = [str(t) if pd.notna(t) else "" for t in texts]

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )

        embeddings = self.encoder.predict(padded_sequences, verbose=0)
        return embeddings

    def _get_n_features(self) -> int:
        """Get the number of features (LSTM units)."""
        return self.lstm_units


class FastTextEmbedder(BaseEmbedder):
    """
    Subword-aware embeddings using FastText, averaged per document.

    FastText extends Word2Vec by representing words as bags of character
    n-grams. This allows it to generate vectors for out-of-vocabulary words
    and handle misspellings better than Word2Vec.

    Attributes:
        vector_size: Dimension of word vectors
        window: Context window size for training
        min_count: Minimum word frequency to include
        model: Trained gensim FastText model

    Performance:
        - Speed: Similar to Word2Vec
        - Quality: Better than Word2Vec for rare/misspelled words
        - Memory: Slightly higher than Word2Vec
        - Offline: Yes (trains on your data)

    Advantages over Word2Vec:
        - Handles typos and misspellings (via subword info)
        - Can generate vectors for unseen words
        - Better for morphologically rich languages

    Example:
        >>> embedder = FastTextEmbedder(vector_size=100)
        >>> vectors = embedder.fit_transform(texts)
        >>> # Can handle "workfromhome" even if only trained on "work" and "home"
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        epochs: int = 10,
        min_n: int = 3,  # Min character n-gram length
        max_n: int = 6   # Max character n-gram length
    ):
        """
        Initialize FastText embedder.

        Args:
            vector_size: Dimension of word vectors (50-300 typical)
            window: Context window size (5-10 typical)
            min_count: Minimum word frequency (2-5 typical)
            workers: Number of worker threads for training
            epochs: Number of training epochs
            min_n: Minimum character n-gram length (3-5 typical)
            max_n: Maximum character n-gram length (5-6 typical)

        Raises:
            ImportError: If gensim is not installed
        """
        super().__init__()

        try:
            from gensim.models import FastText
        except ImportError:
            raise ImportError(
                "gensim is required for FastTextEmbedder. "
                "Install with: pip install gensim"
            )

        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.min_n = min_n
        self.max_n = max_n
        self.model = None

    def fit(self, texts: List[str]) -> 'FastTextEmbedder':
        """
        Fit FastText model on the corpus.

        Args:
            texts: List of text strings

        Returns:
            Self (fitted embedder)
        """
        from gensim.models import FastText
        from gensim.utils import simple_preprocess

        # Tokenize texts, treating None and empty strings as empty token lists
        tokenized = []
        for text in texts:
            if text is None or (isinstance(text, str) and not text.strip()):
                tokenized.append([])
            else:
                tokenized.append(simple_preprocess(str(text)))

        # Filter out empty token lists for training
        training_sentences = [tokens for tokens in tokenized if tokens]

        logger.info(f"Training FastText model (vector_size={self.vector_size}, epochs={self.epochs})")

        if not training_sentences:
            raise ValueError(
                "No valid tokens found after preprocessing. "
                "All texts are empty or too short for FastText training."
            )

        # Train FastText model
        self.model = FastText(
            sentences=training_sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=self.epochs,
            min_n=self.min_n,
            max_n=self.max_n
        )

        self.is_fitted_ = True
        vocab_size = len(self.model.wv)
        logger.info(f"FastText model trained. Vocabulary size: {vocab_size}")

        if vocab_size < 50:
            warnings.warn(
                f"Small vocabulary size ({vocab_size} words). "
                "Consider lowering min_count or providing more diverse text data.",
                UserWarning
            )

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into FastText embeddings (averaged word vectors).

        Args:
            texts: List of text strings

        Returns:
            Array of shape (n_samples, vector_size)

        Raises:
            ValueError: If embedder is not fitted
        """
        if not self.is_fitted_:
            raise ValueError("FastTextEmbedder must be fitted before transform")

        from gensim.utils import simple_preprocess

        embeddings = []

        for text in texts:
            if text is None or (isinstance(text, str) and not text.strip()):
                # None or empty text -> zero vector
                embeddings.append(np.zeros(self.vector_size))
                continue

            tokens = simple_preprocess(str(text))

            if tokens:
                # FastText can handle OOV words via subword info
                word_vectors = [self.model.wv[word] for word in tokens]
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # Empty text - use zero vector
                doc_vector = np.zeros(self.vector_size)

            embeddings.append(doc_vector)

        return np.array(embeddings)

    def _get_n_features(self) -> int:
        """Get the number of features (vector size)."""
        return self.vector_size


def get_embedder(
    representation: str = 'tfidf',
    **kwargs
) -> Union[BaseEmbedder, Any]:
    """
    Factory function to create embedders based on representation type.

    Args:
        representation: Type of representation to use:
            - 'tfidf': TF-IDF vectorization (returns None, handled separately)
            - 'sbert': SentenceBERT embeddings
            - 'word2vec': Word2Vec embeddings
            - 'fasttext': FastText embeddings
        **kwargs: Additional arguments passed to the embedder constructor

    Returns:
        Embedder instance or None (for TF-IDF)

    Raises:
        ValueError: If representation type is not recognized

    Example:
        >>> embedder = get_embedder('sbert', model_name='all-MiniLM-L6-v2')
        >>> vectors = embedder.fit_transform(texts)
    """
    if representation == 'tfidf':
        # TF-IDF handled by existing code
        return None
    elif representation == 'sbert':
        return SentenceBERTEmbedder(**kwargs)
    elif representation == 'bert':
        # BERT is an alias for SentenceBERT (uses BERT-based transformers)
        return SentenceBERTEmbedder(**kwargs)
    elif representation == 'lstm':
        return LSTMEmbedder(**kwargs)
    elif representation == 'word2vec':
        return Word2VecEmbedder(**kwargs)
    elif representation == 'fasttext':
        return FastTextEmbedder(**kwargs)
    else:
        raise ValueError(
            f"Unknown representation: {representation}. "
            f"Choose from: 'tfidf', 'sbert', 'bert', 'lstm', 'word2vec', 'fasttext'"
        )


def compare_embeddings(
    texts: List[str],
    representations: List[str] = ['tfidf', 'sbert', 'word2vec', 'fasttext'],
    n_clusters: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Compare different embedding methods on the same dataset.

    This function trains each embedding method and evaluates clustering quality
    to help you choose the best representation for your data.

    Args:
        texts: List of text strings
        representations: List of embedding methods to compare
        n_clusters: Number of clusters for evaluation

    Returns:
        Dictionary mapping representation names to metrics:
            - 'fit_time': Time to fit the model (seconds)
            - 'transform_time': Time to transform texts (seconds)
            - 'silhouette_score': Clustering quality metric
            - 'n_features': Number of features in embedding

    Example:
        >>> results = compare_embeddings(texts, representations=['tfidf', 'sbert'])
        >>> for method, metrics in results.items():
        ...     print(f"{method}: silhouette={metrics['silhouette_score']:.3f}")
    """
    import time
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    results = {}

    for rep in representations:
        logger.info(f"Evaluating {rep}...")

        try:
            # Create embedder
            if rep == 'tfidf':
                embedder = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            else:
                embedder = get_embedder(rep)

            # Time fitting
            start = time.time()
            embedder.fit(texts)
            fit_time = time.time() - start

            # Time transformation
            start = time.time()
            vectors = embedder.transform(texts)
            transform_time = time.time() - start

            # Evaluate clustering
            if vectors.shape[0] >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(vectors)
                sil_score = silhouette_score(vectors, labels)
            else:
                sil_score = None

            results[rep] = {
                'fit_time': fit_time,
                'transform_time': transform_time,
                'total_time': fit_time + transform_time,
                'silhouette_score': sil_score,
                'n_features': vectors.shape[1],
                'memory_mb': vectors.nbytes / (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Failed to evaluate {rep}: {e}")
            results[rep] = {'error': str(e)}

    return results
