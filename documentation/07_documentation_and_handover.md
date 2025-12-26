# Documentation and Handover

**Framework**: Open-Ended Coding Analysis Framework
**Document Version**: 1.0
**Last Updated**: 2025-12-26
**Purpose**: Comprehensive methodology documentation, developer handover, and maintenance guide

---

## Executive Overview

This document provides complete methodology documentation and handover information for the Open-Ended Coding Analysis Framework. It serves as the definitive reference for understanding how each ML technique works, how to use the framework programmatically, and how to maintain and extend it over time.

**Intended Audiences:**
- **Analysts/Clients**: Understand methods, interpret results, answer common questions
- **Developers**: API reference, extension points, architecture details
- **Maintainers**: Long-term support, dependency management, troubleshooting

---

## Table of Contents

1. [Methodology Documentation](#1-methodology-documentation-per-technique)
2. [Developer Notes](#2-developer-notes)
3. [Analyst/Client Guidance](#3-analyst-client-guidance)
4. [Handover Checklist](#4-handover-checklist)
5. [Long-term Maintenance](#5-long-term-maintenance-notes)

---

## 1. Methodology Documentation Per Technique

### 1.1 TF-IDF + K-Means Clustering

#### Objectives

**Primary Goal**: Automatically discover distinct themes in qualitative data by grouping semantically similar responses.

**Use Cases**:
- Exploratory analysis of survey responses
- Initial theme discovery in large datasets
- Identifying major categories in customer feedback
- Baseline for comparison with other methods

**When to Use**:
- Dataset size: 50-10,000+ responses
- When themes are expected to be distinct (non-overlapping)
- Fast turnaround required
- Interpretability is paramount

#### Assumptions

1. **Distributional Hypothesis**: Words that appear in similar contexts have similar meanings
2. **Cluster Validity**: Responses can be meaningfully grouped into K distinct clusters
3. **TF-IDF Relevance**: Word importance based on frequency and rarity is a valid signal
4. **Independence**: Features (words) are treated as independent (bag-of-words model)
5. **Spherical Clusters**: K-Means assumes clusters are roughly spherical and equally sized

#### Step-by-Step Process

**Step 1: Text Preprocessing**
```python
def preprocess_text(text):
    """
    Clean and normalize text for analysis.

    Operations:
    - Convert to lowercase
    - Remove special characters (keep alphanumeric and spaces)
    - Remove extra whitespace
    - Optional: lemmatization, stopword removal
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join(text.split())
    return text
```

**Step 2: TF-IDF Vectorization**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=1000,        # Top 1000 most important words
    stop_words='english',     # Remove common words
    min_df=2,                 # Word must appear in at least 2 documents
    max_df=0.8,              # Ignore words in >80% of documents
    ngram_range=(1, 2)       # Unigrams and bigrams
)

feature_matrix = vectorizer.fit_transform(preprocessed_texts)
# Shape: (n_documents, n_features)
```

**TF-IDF Formula**:
```
TF-IDF(word, document) = TF(word, document) × IDF(word)

where:
TF(word, document) = count of word in document / total words in document
IDF(word) = log(total documents / documents containing word)
```

**Step 3: K-Means Clustering**
```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=10,           # Number of themes to discover
    random_state=42,         # Reproducibility
    n_init=10,              # Number of initializations
    max_iter=300            # Maximum iterations
)

cluster_labels = kmeans.fit_predict(feature_matrix)
# cluster_labels[i] = cluster ID for document i
```

**K-Means Algorithm**:
1. Initialize K cluster centroids randomly
2. Assign each document to nearest centroid (Euclidean distance)
3. Update centroids as mean of assigned documents
4. Repeat steps 2-3 until convergence or max iterations

**Step 4: Code Extraction**
```python
def extract_codes(kmeans, vectorizer, top_n_keywords=10):
    """Extract interpretable codes from clusters."""
    feature_names = vectorizer.get_feature_names_out()
    codes = {}

    for cluster_id in range(kmeans.n_clusters):
        # Get cluster centroid
        centroid = kmeans.cluster_centers_[cluster_id]

        # Get top keywords by TF-IDF weight
        top_indices = centroid.argsort()[-top_n_keywords:][::-1]
        keywords = [feature_names[i] for i in top_indices]

        # Generate label from top 3 keywords
        label = '_'.join(keywords[:3])

        codes[f'CODE_{cluster_id+1:02d}'] = {
            'label': label,
            'keywords': keywords,
            'centroid': centroid
        }

    return codes
```

**Step 5: Assignment with Confidence**
```python
def assign_codes_with_confidence(kmeans, feature_matrix, min_confidence=0.3):
    """
    Assign codes to documents with confidence scores.

    Confidence = 1 - (distance to assigned cluster / max distance)
    """
    from sklearn.metrics.pairwise import euclidean_distances

    assignments = []
    confidences = []

    for i, doc_vector in enumerate(feature_matrix):
        # Calculate distances to all clusters
        distances = euclidean_distances(
            doc_vector.reshape(1, -1),
            kmeans.cluster_centers_
        )[0]

        # Assign to closest cluster
        assigned_cluster = distances.argmin()

        # Calculate confidence score
        max_dist = distances.max()
        min_dist = distances.min()
        confidence = 1 - (min_dist / max_dist) if max_dist > 0 else 1.0

        # Only assign if confidence exceeds threshold
        if confidence >= min_confidence:
            assignments.append([f'CODE_{assigned_cluster+1:02d}'])
            confidences.append([confidence])
        else:
            assignments.append([])
            confidences.append([])

    return assignments, confidences
```

#### Limitations

1. **Fixed K**: Requires pre-specifying number of clusters (can use elbow method or silhouette analysis)
2. **Spherical Assumption**: May not capture elongated or irregularly-shaped clusters
3. **Sensitivity to Initialization**: Different random seeds can produce slightly different results
4. **Hard Assignment**: Each document assigned to exactly one cluster (no overlap)
5. **Euclidean Distance**: May not be ideal for high-dimensional sparse text data
6. **Stopwords**: Removal may lose important context in some domains
7. **No Semantic Understanding**: Doesn't understand synonyms, polysemy, or context

#### Evaluation Metrics

**Silhouette Score** (Range: -1 to 1, higher is better):
```python
from sklearn.metrics import silhouette_score

silhouette = silhouette_score(feature_matrix, cluster_labels)
# > 0.5: Strong clustering
# 0.25-0.5: Moderate clustering
# < 0.25: Weak clustering
```

**Calinski-Harabasz Score** (Higher is better):
```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(
    feature_matrix.toarray(),
    cluster_labels
)
# Measures ratio of between-cluster to within-cluster variance
```

**Davies-Bouldin Index** (Lower is better):
```python
from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(
    feature_matrix.toarray(),
    cluster_labels
)
# < 1.0: Good separation
# 1.0-2.0: Moderate separation
```

#### Key Parameters and Their Effects

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `n_clusters` | Number of themes | 3-15 for most datasets; use silhouette analysis to optimize |
| `max_features` | Vocabulary size | 500-2000; lower for small datasets, higher for large/diverse |
| `min_df` | Min document frequency | 2-5; higher reduces noise, lower captures rare themes |
| `max_df` | Max document frequency | 0.7-0.9; excludes overly common words |
| `ngram_range` | Word sequences | (1,1) for unigrams, (1,2) adds bigrams (more context, higher dimensionality) |
| `min_confidence` | Assignment threshold | 0.2-0.4; lower includes more assignments, higher is more selective |
| `random_state` | Reproducibility | Set to fixed value (e.g., 42) for consistent results |

#### References

- **TF-IDF**: Salton, G., & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." Information Processing & Management.
- **K-Means**: MacQueen, J. (1967). "Some methods for classification and analysis of multivariate observations." Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability.
- **Silhouette Coefficient**: Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis." Journal of Computational and Applied Mathematics.

---

### 1.2 Latent Dirichlet Allocation (LDA)

#### Objectives

**Primary Goal**: Discover latent topics in a collection of documents, where each document is a mixture of topics and each topic is a distribution over words.

**Use Cases**:
- When responses can belong to multiple themes simultaneously
- Discovering overlapping or related themes
- Probabilistic topic modeling for nuanced analysis
- Comparing topic prevalence across subgroups

**When to Use**:
- Dataset size: 100+ responses (preferably 500+)
- When multi-topic assignments are expected
- When probabilistic interpretation is valuable
- Longer, more complex text documents

#### Assumptions

1. **Bag of Words**: Word order doesn't matter, only occurrence
2. **Exchangeability**: Order of documents and words within documents doesn't matter
3. **Fixed Topics**: Number of topics K is known in advance
4. **Dirichlet Priors**: Document-topic and topic-word distributions follow Dirichlet distributions
5. **Discrete Topics**: Topics are discrete, interpretable themes (not continuous)

#### Step-by-Step Process

**Step 1: Text Preprocessing** (same as TF-IDF)

**Step 2: Document-Term Matrix (Count Vectorization)**
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(
    max_features=1000,
    stop_words='english',
    min_df=2,
    max_df=0.8
)

doc_term_matrix = vectorizer.fit_transform(preprocessed_texts)
# Counts of each word in each document (not TF-IDF)
```

**Step 3: LDA Model Training**
```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(
    n_components=10,         # Number of topics
    random_state=42,
    max_iter=20,            # Number of EM iterations
    learning_method='batch', # 'batch' or 'online'
    n_jobs=-1               # Parallel processing
)

doc_topic_matrix = lda.fit_transform(doc_term_matrix)
# Shape: (n_documents, n_topics)
# doc_topic_matrix[i, j] = probability of topic j in document i
```

**LDA Generative Process** (Conceptual):
```
For each document d:
    1. Draw topic distribution θ_d ~ Dirichlet(α)

    For each word position n in document d:
        2. Draw topic z_n ~ Multinomial(θ_d)
        3. Draw word w_n ~ Multinomial(φ_z_n)

where:
- α: Document-topic Dirichlet prior
- φ_k: Topic-word distribution for topic k
```

**Step 4: Topic Extraction**
```python
def extract_topics(lda, vectorizer, top_n_words=10):
    """Extract topics from trained LDA model."""
    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for topic_idx, topic_dist in enumerate(lda.components_):
        # Get top words for this topic
        top_indices = topic_dist.argsort()[-top_n_words:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        weights = topic_dist[top_indices]

        topics[f'TOPIC_{topic_idx+1:02d}'] = {
            'keywords': keywords,
            'weights': weights,
            'label': '_'.join(keywords[:3])
        }

    return topics
```

**Step 5: Multi-Topic Assignment**
```python
def assign_topics_probabilistic(doc_topic_matrix, min_prob=0.2):
    """
    Assign topics to documents with probability thresholds.

    A document can belong to multiple topics.
    """
    assignments = []
    confidences = []

    for doc_topics in doc_topic_matrix:
        # Find topics above threshold
        topic_probs = doc_topics / doc_topics.sum()  # Normalize

        doc_assignments = []
        doc_confidences = []

        for topic_idx, prob in enumerate(topic_probs):
            if prob >= min_prob:
                doc_assignments.append(f'TOPIC_{topic_idx+1:02d}')
                doc_confidences.append(float(prob))

        assignments.append(doc_assignments)
        confidences.append(doc_confidences)

    return assignments, confidences
```

#### Limitations

1. **Requires Many Documents**: Works best with hundreds to thousands of documents
2. **Hyperparameter Sensitivity**: α and η priors significantly impact results
3. **No Ground Truth**: Topics are latent; validation requires human judgment
4. **Computational Cost**: EM algorithm is slower than K-Means
5. **Non-Deterministic**: Different runs produce different (but similar) results
6. **Interpretability**: Topics may not always be semantically coherent
7. **Fixed K**: Number of topics must be specified in advance
8. **Perplexity vs. Coherence**: Low perplexity doesn't guarantee interpretable topics

#### Evaluation Metrics

**Perplexity** (Lower is better):
```python
perplexity = lda.perplexity(doc_term_matrix)
# Measures how well model predicts held-out data
# Lower = better fit (but may overfit)
```

**Topic Coherence** (Human evaluation or automated):
```python
# PMI-based coherence (requires gensim)
from gensim.models.coherencemodel import CoherenceModel

# Convert to gensim format
coherence_model = CoherenceModel(
    topics=topics,
    texts=tokenized_texts,
    coherence='c_v'  # or 'u_mass', 'c_npmi'
)
coherence_score = coherence_model.get_coherence()
# Higher = more coherent topics
```

#### Key Parameters and Their Effects

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `n_components` | Number of topics | 5-20 for most datasets; optimize using coherence |
| `max_iter` | EM iterations | 20-50; more iterations = better fit but slower |
| `doc_topic_prior` (α) | Document-topic smoothing | Default: 1/K; lower = sparser topics |
| `topic_word_prior` (η) | Topic-word smoothing | Default: 1/K; lower = more focused topics |
| `learning_method` | Batch vs. online | 'batch' for small datasets, 'online' for large |
| `min_prob` | Topic assignment threshold | 0.15-0.25 for multi-topic, 0.3+ for primary topic only |

#### References

- **LDA**: Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). "Latent Dirichlet Allocation." Journal of Machine Learning Research.
- **Topic Coherence**: Röder, M., Both, A., & Hinneburg, A. (2015). "Exploring the Space of Topic Coherence Measures." WSDM.
- **Perplexity**: Wallach, H. M., et al. (2009). "Evaluation methods for topic models." ICML.

---

### 1.3 Non-negative Matrix Factorization (NMF)

#### Objectives

**Primary Goal**: Decompose the document-term matrix into non-negative document-topic and topic-word matrices, producing sparse and interpretable topics.

**Use Cases**:
- Discovering distinct, non-overlapping themes
- When sparsity and interpretability are priorities
- Deterministic topic modeling (same results every run)
- Parts-based representation of text data

**When to Use**:
- Dataset size: 50+ responses
- When themes are expected to be distinct
- Reproducibility is critical
- Faster alternative to LDA with comparable results

#### Assumptions

1. **Non-Negativity**: All values (word counts, topic weights) are ≥ 0
2. **Additive Parts**: Documents are additive combinations of topics
3. **Sparsity**: Topics and assignments are sparse (few non-zero values)
4. **Linear Combination**: Document vectors are weighted sums of topic vectors
5. **No Probabilistic Interpretation**: Unlike LDA, not a generative probabilistic model

#### Step-by-Step Process

**Step 1: Text Preprocessing** (same as others)

**Step 2: TF-IDF Vectorization** (NMF works with TF-IDF or raw counts)
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    min_df=2,
    max_df=0.8
)

tfidf_matrix = vectorizer.fit_transform(preprocessed_texts)
```

**Step 3: NMF Decomposition**
```python
from sklearn.decomposition import NMF

nmf = NMF(
    n_components=10,         # Number of topics
    init='nndsvda',         # Initialization method
    random_state=42,
    max_iter=200,
    alpha=0.1,              # Regularization
    l1_ratio=0.5            # L1 vs L2 regularization
)

W = nmf.fit_transform(tfidf_matrix)  # Document-topic matrix
H = nmf.components_                   # Topic-word matrix

# tfidf_matrix ≈ W × H (low-rank approximation)
```

**NMF Optimization**:
```
Minimize: ||X - WH||² + α × l1_ratio × ||W||₁ + α × (1 - l1_ratio) × ||W||₂²

where:
- X: Original document-term matrix
- W: Document-topic matrix (n_documents × n_topics)
- H: Topic-word matrix (n_topics × n_words)
- α: Regularization strength
- l1_ratio: Balance between L1 (sparsity) and L2 (smoothness)
```

**Step 4: Topic Extraction**
```python
def extract_nmf_topics(nmf, vectorizer, top_n_words=10):
    """Extract topics from NMF components."""
    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for topic_idx, topic_weights in enumerate(nmf.components_):
        # Get top words by weight
        top_indices = topic_weights.argsort()[-top_n_words:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        weights = topic_weights[top_indices]

        # Calculate sparsity
        sparsity = (topic_weights == 0).sum() / len(topic_weights)

        topics[f'COMPONENT_{topic_idx+1:02d}'] = {
            'keywords': keywords,
            'weights': weights,
            'sparsity': sparsity,
            'label': '_'.join(keywords[:3])
        }

    return topics
```

**Step 5: Sparse Assignment**
```python
def assign_nmf_components(W, min_weight=0.25):
    """
    Assign components to documents based on weights.

    Typically results in 1-2 strong components per document.
    """
    assignments = []
    confidences = []

    for doc_weights in W:
        # Normalize weights
        total_weight = doc_weights.sum()
        if total_weight > 0:
            normalized = doc_weights / total_weight
        else:
            normalized = doc_weights

        doc_assignments = []
        doc_confidences = []

        for comp_idx, weight in enumerate(normalized):
            if weight >= min_weight:
                doc_assignments.append(f'COMPONENT_{comp_idx+1:02d}')
                doc_confidences.append(float(weight))

        assignments.append(doc_assignments)
        confidences.append(doc_confidences)

    return assignments, confidences
```

#### Limitations

1. **Local Minima**: Optimization may converge to local minima (less of an issue with good initialization)
2. **No Probabilistic Interpretation**: Weights don't have clear probabilistic meaning
3. **Hyperparameter Tuning**: Requires tuning α and l1_ratio for optimal results
4. **Fixed K**: Number of components must be specified
5. **Dense Input**: Works best with TF-IDF; less effective with raw counts
6. **Initialization Sensitivity**: Different initializations can affect results
7. **No Standard Evaluation Metric**: No equivalent to perplexity for LDA

#### Evaluation Metrics

**Reconstruction Error** (Lower is better):
```python
reconstruction_error = nmf.reconstruction_err_
# Measures ||X - WH||²
# Lower = better approximation
```

**Sparsity**:
```python
def calculate_sparsity(matrix):
    """Calculate sparsity (proportion of zeros)."""
    return (matrix == 0).sum() / matrix.size

W_sparsity = calculate_sparsity(W)
H_sparsity = calculate_sparsity(H)
# Higher sparsity = more interpretable
```

#### Key Parameters and Their Effects

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `n_components` | Number of topics | 5-15 for most datasets |
| `init` | Initialization | 'nndsvda' (deterministic), 'nndsvd' (faster), 'random' |
| `alpha` | Regularization | 0.0-0.5; higher = more sparse, lower = better fit |
| `l1_ratio` | L1 vs L2 regularization | 0.5-1.0; higher = sparser (L1), lower = smoother (L2) |
| `max_iter` | Iterations | 200-500; more for better convergence |
| `min_weight` | Assignment threshold | 0.2-0.4; higher = fewer, stronger assignments |

#### References

- **NMF**: Lee, D. D., & Seung, H. S. (1999). "Learning the parts of objects by non-negative matrix factorization." Nature.
- **NMF for Text**: Xu, W., Liu, X., & Gong, Y. (2003). "Document clustering based on non-negative matrix factorization." SIGIR.
- **Sparse NMF**: Hoyer, P. O. (2004). "Non-negative matrix factorization with sparseness constraints." Journal of Machine Learning Research.

---

### 1.4 Embeddings-Based Clustering (Optional)

#### Objectives

**Primary Goal**: Leverage pre-trained semantic embeddings (e.g., SentenceTransformers, word2vec) to capture semantic similarity beyond keyword matching.

**Use Cases**:
- Short texts where keywords are insufficient
- When semantic similarity matters more than exact keyword overlap
- Cross-lingual or multilingual analysis
- Capturing paraphrases and synonyms

**When to Use**:
- Dataset size: Any size (even small datasets benefit)
- When responses use varied vocabulary for similar concepts
- GPU available for faster embedding generation
- Higher quality needed than TF-IDF

#### Step-by-Step Process

**Step 1: Generate Embeddings**
```python
from sentence_transformers import SentenceTransformers

# Load pre-trained model
model = SentenceTransformers('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(preprocessed_texts, show_progress_bar=True)
# Shape: (n_documents, embedding_dim) e.g., (150, 384)
```

**Step 2: Clustering on Embeddings**
```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# K-Means on embeddings
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# Or DBSCAN for automatic cluster count
dbscan = DBSCAN(eps=0.5, min_samples=5)
cluster_labels = dbscan.fit_predict(embeddings)
```

**Step 3: Extract Representative Keywords**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_cluster_keywords(texts, labels, n_keywords=10):
    """Extract keywords for each cluster using TF-IDF."""
    unique_labels = set(labels)
    cluster_keywords = {}

    for label in unique_labels:
        if label == -1:  # Noise cluster in DBSCAN
            continue

        # Get texts in this cluster
        cluster_texts = [texts[i] for i in range(len(texts)) if labels[i] == label]

        # Vectorize and get top keywords
        vectorizer = TfidfVectorizer(max_features=n_keywords, stop_words='english')
        tfidf = vectorizer.fit_transform(cluster_texts)
        keywords = vectorizer.get_feature_names_out()

        cluster_keywords[label] = keywords.tolist()

    return cluster_keywords
```

#### Key Parameters and Their Effects

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `model` | Embedding quality | 'all-MiniLM-L6-v2' (fast), 'all-mpnet-base-v2' (quality) |
| `clustering_method` | Algorithm | K-Means (fast), DBSCAN (auto K), Agglomerative (hierarchical) |
| `eps` (DBSCAN) | Neighborhood size | 0.3-0.7; tune based on embedding distances |
| `min_samples` (DBSCAN) | Min cluster size | 3-10; higher = fewer, larger clusters |

#### Limitations

1. **Computational Cost**: Embedding generation can be slow without GPU
2. **Model Dependency**: Quality depends on pre-trained model
3. **Black Box**: Harder to interpret than keyword-based methods
4. **Memory**: Large embedding matrices for big datasets
5. **Domain Mismatch**: Pre-trained models may not capture domain-specific semantics

---

## 2. Developer Notes

### 2.1 API Reference and Key Classes

#### DataLoader Class

**Purpose**: Load data from multiple sources with robust error handling.

**Location**: `src/data_loader.py`

**Key Methods**:

```python
from src.data_loader import DataLoader

loader = DataLoader()

# Load from CSV
df = loader.load_csv('data/responses.csv')

# Load from Excel
df = loader.load_excel('data/responses.xlsx', sheet_name='Sheet1')

# Load from database
df = loader.load_from_sqlite('database.db', 'SELECT * FROM responses')
df = loader.load_from_postgres('postgresql://user:pass@host/db', 'SELECT * FROM responses')

# Load from JSON
df = loader.load_json('data/responses.json', lines=True)

# Validate DataFrame
loader.validate_dataframe(df, required_columns=['response_id', 'text'])

# Assess content quality
df = loader.assess_content_quality(
    df,
    text_column='text',
    min_words=3,
    min_chars=10,
    max_repetition_ratio=0.7,
    min_english_word_ratio=0.3
)

# Export flagged responses
loader.export_non_analytic_responses(df, 'non_analytic.csv')

# Get quality summary
summary = loader.get_quality_summary(df)
```

**Important Notes**:
- SQL queries are validated to prevent destructive operations (only SELECT allowed)
- Quality assessment NEVER deletes data, only flags for review
- All file operations include error handling with informative messages

---

#### CodeFrame Class

**Purpose**: Manage coding frames for qualitative analysis.

**Location**: `src/code_frame.py`

**Key Methods**:

```python
from src.code_frame import CodeFrame

# Create code frame
frame = CodeFrame(name='Customer Feedback', description='Product feedback codes')

# Add codes
frame.add_code(
    code_id='QUALITY',
    label='Product Quality',
    description='Mentions of product quality, durability, craftsmanship',
    keywords=['quality', 'durable', 'well-made', 'craftsmanship'],
    parent=None  # Top-level code
)

frame.add_code(
    code_id='QUALITY_POSITIVE',
    label='Positive Quality Feedback',
    description='Praise for product quality',
    keywords=['excellent quality', 'high quality', 'well-made'],
    parent='QUALITY'  # Child of QUALITY
)

# Apply codes to text
matched_codes = frame.apply_codes(
    text="This product has excellent quality and durability",
    case_sensitive=False,
    update_counts=True
)

# Get hierarchy
hierarchy = frame.get_hierarchy()
# Returns: {'root': ['QUALITY'], 'QUALITY': ['QUALITY_POSITIVE']}

# Get summary statistics
summary = frame.summary()  # Returns DataFrame

# Export codebook
frame.export_codebook('codebook.csv')

# Reset counts for reprocessing
frame.reset_counts()
```

**Important Notes**:
- Codes are applied via keyword matching (simple but effective)
- Hierarchical structures supported for nested coding schemes
- Counts automatically updated when `update_counts=True`

---

#### ThemeAnalyzer Class

**Purpose**: Analyze themes and patterns in coded data.

**Location**: `src/theme_analyzer.py`

**Key Methods**:

```python
from src.theme_analyzer import ThemeAnalyzer

analyzer = ThemeAnalyzer()

# Define themes (higher-level than codes)
analyzer.add_theme(
    theme_id='USER_EXPERIENCE',
    name='User Experience',
    description='Overall experience using the product',
    associated_codes=['EASE_OF_USE', 'LEARNING_CURVE', 'INTERFACE']
)

# Assign themes to responses
df_with_themes = analyzer.assign_themes(coded_df, code_column='assigned_codes')

# Get theme summary
theme_summary = analyzer.get_theme_summary(df_with_themes)

# Calculate theme co-occurrence
cooccurrence = analyzer.calculate_theme_cooccurrence(df_with_themes)

# Get dominant theme per document
dominant = analyzer.get_dominant_theme(df_with_themes)
```

---

#### CategoryManager Class

**Purpose**: Manage hierarchical category structures.

**Location**: `src/category_manager.py`

**Key Methods**:

```python
from src.category_manager import CategoryManager

manager = CategoryManager()

# Add categories
manager.add_category('PRODUCT', 'Product Features')
manager.add_category('PRODUCT_QUALITY', 'Quality', parent='PRODUCT')
manager.add_category('PRODUCT_DESIGN', 'Design', parent='PRODUCT')

# Get all descendants
descendants = manager.get_descendants('PRODUCT')
# Returns: ['PRODUCT_QUALITY', 'PRODUCT_DESIGN']

# Validate category structure
is_valid = manager.validate()

# Export category tree
manager.export_tree('category_tree.json')
```

---

#### ContentQualityFilter Class

**Purpose**: Filter out non-analytic responses.

**Location**: `src/content_quality.py`

**Key Methods**:

```python
from src.content_quality import ContentQualityFilter

filter = ContentQualityFilter(
    min_words=3,
    min_chars=10,
    max_repetition_ratio=0.7,
    min_english_word_ratio=0.3
)

# Assess single response
assessment = filter.assess('This is a test response.')
# Returns: {
#     'is_analytic': True,
#     'confidence': 0.95,
#     'reason': 'Valid analytic response',
#     'recommendation': 'include',
#     'flags': []
# }

# Batch assess
responses = ['Response 1', 'test', 'asdfasdf', ...]
assessments = filter.batch_assess(responses)

# Get flag statistics
stats = filter.get_flag_statistics(assessments)
```

**Quality Flags**:
- `null_empty`: Null or empty string
- `too_short`: Below minimum word/character count
- `test_response`: Test entries (e.g., "test", "asdf")
- `excessive_repetition`: Too many repeated words
- `non_english`: Too few English words
- `gibberish`: Random character patterns

---

### 2.2 Key Parameters and Their Effects

#### Global Parameters

```python
# Typical configuration for most analyses
CONFIG = {
    # Data loading
    'encoding': 'utf-8',
    'delimiter': ',',  # CSV delimiter

    # Text preprocessing
    'lowercase': True,
    'remove_punctuation': True,
    'remove_stopwords': True,
    'stopwords_language': 'english',
    'lemmatize': False,  # Set to True for better normalization

    # Vectorization
    'max_features': 1000,  # 500-2000 depending on dataset size
    'min_df': 2,          # Minimum document frequency
    'max_df': 0.8,        # Maximum document frequency
    'ngram_range': (1, 2), # Unigrams and bigrams

    # Clustering/Topic Modeling
    'n_clusters': 10,      # Number of themes
    'random_state': 42,    # For reproducibility

    # Assignment
    'min_confidence': 0.3,  # Confidence threshold

    # Quality filtering
    'min_words': 3,
    'min_chars': 10,
    'max_repetition_ratio': 0.7,
}
```

#### Method-Specific Parameters

**TF-IDF + K-Means**:
```python
TFIDF_KMEANS_PARAMS = {
    'tfidf': {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.8,
        'ngram_range': (1, 2),
        'norm': 'l2',  # L2 normalization
    },
    'kmeans': {
        'n_clusters': 10,
        'n_init': 10,  # Number of initializations
        'max_iter': 300,
        'random_state': 42,
    }
}
```

**LDA**:
```python
LDA_PARAMS = {
    'vectorizer': {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.8,
    },
    'lda': {
        'n_components': 10,
        'max_iter': 20,
        'learning_method': 'batch',  # or 'online' for large datasets
        'random_state': 42,
        'doc_topic_prior': None,  # Auto: 1/n_components
        'topic_word_prior': None,  # Auto: 1/n_components
    }
}
```

**NMF**:
```python
NMF_PARAMS = {
    'tfidf': {
        'max_features': 1000,
        'min_df': 2,
        'max_df': 0.8,
    },
    'nmf': {
        'n_components': 10,
        'init': 'nndsvda',  # Deterministic initialization
        'max_iter': 200,
        'random_state': 42,
        'alpha': 0.1,       # Regularization
        'l1_ratio': 0.5,    # L1/L2 balance
    }
}
```

---

### 2.3 Extension Points

#### Adding a New ML Algorithm

**Step 1**: Create new method in `helpers/analysis.py`:

```python
def run_custom_algorithm(df, text_column, n_codes, **kwargs):
    """
    Run custom ML algorithm for coding.

    Args:
        df: DataFrame with text data
        text_column: Column name containing text
        n_codes: Number of codes to discover
        **kwargs: Algorithm-specific parameters

    Returns:
        dict: Results with keys:
            - 'coded_df': DataFrame with code assignments
            - 'codebook': Dictionary of codes
            - 'metrics': Quality metrics
            - 'method': 'custom_algorithm'
    """
    # Preprocessing
    texts = df[text_column].fillna('').tolist()

    # Your algorithm here
    # ...

    # Return standardized format
    return {
        'coded_df': df_with_codes,
        'codebook': codebook,
        'metrics': {
            'coverage': coverage_pct,
            'avg_confidence': avg_conf,
            'n_codes': n_codes,
        },
        'method': 'custom_algorithm'
    }
```

**Step 2**: Register in Streamlit UI (`app.py`):

```python
# In page_configuration()
method = st.selectbox(
    "ML Algorithm",
    options=['tfidf_kmeans', 'lda', 'nmf', 'custom_algorithm'],
    format_func=lambda x: {
        'tfidf_kmeans': 'TF-IDF + K-Means',
        'lda': 'Latent Dirichlet Allocation',
        'nmf': 'Non-negative Matrix Factorization',
        'custom_algorithm': 'My Custom Algorithm'  # Add here
    }[x]
)
```

**Step 3**: Add tests (`tests/test_custom.py`):

```python
def test_custom_algorithm():
    from helpers.analysis import run_custom_algorithm

    df = pd.DataFrame({'text': ['Sample text 1', 'Sample text 2', ...]})
    results = run_custom_algorithm(df, 'text', n_codes=5)

    assert 'coded_df' in results
    assert 'codebook' in results
    assert len(results['codebook']) == 5
```

---

#### Adding a New Visualization

**Step 1**: Create function in `helpers/visualization.py` (or new file):

```python
def plot_custom_visualization(results, **kwargs):
    """
    Create custom visualization of coding results.

    Args:
        results: Results dictionary from analysis
        **kwargs: Plot-specific parameters

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    # Extract data
    coded_df = results['coded_df']
    codebook = results['codebook']

    # Create plot
    fig = go.Figure(...)

    fig.update_layout(
        title="Custom Visualization",
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        height=600
    )

    return fig
```

**Step 2**: Add to Streamlit UI:

```python
# In page_visualizations()
with tab_custom:
    st.markdown("### Custom Visualization")
    fig = plot_custom_visualization(results)
    st.plotly_chart(fig, use_container_width=True)
```

---

#### Adding a New Export Format

**Step 1**: Create export function:

```python
def export_to_custom_format(results, output_path):
    """
    Export results to custom format.

    Args:
        results: Results dictionary
        output_path: Path to save file

    Returns:
        str: Path to saved file
    """
    # Extract data
    coded_df = results['coded_df']
    codebook = results['codebook']

    # Format and save
    # ... your custom export logic ...

    return output_path
```

**Step 2**: Register in export UI:

```python
# In page_export_results()
if st.button("Export to Custom Format"):
    output_path = export_to_custom_format(results, 'output/custom.xyz')
    st.success(f"Exported to {output_path}")

    # Provide download
    with open(output_path, 'rb') as f:
        st.download_button(
            label="Download Custom File",
            data=f,
            file_name="results.xyz",
            mime="application/octet-stream"
        )
```

---

## 3. Analyst/Client Guidance

### 3.1 How to Run Analyses (Step-by-Step)

#### Option 1: Streamlit Web Interface (Recommended for Non-Programmers)

**Step 1: Launch Application**
```bash
cd /home/user/JC-OE-Coding
streamlit run app.py
```
Browser will open automatically to http://localhost:8501

**Step 2: Upload Your Data**
- Click "📤 Data Upload" in sidebar
- Option A: Select a sample dataset from dropdown
- Option B: Upload your own CSV/Excel file
- Preview data to verify correct loading

**Step 3: Configure Analysis**
- Click "⚙️ Configuration"
- Select text column containing responses
- Choose number of codes (start with 8-12)
- Select ML algorithm:
  - **TF-IDF + K-Means**: Recommended for first-time users
  - **LDA**: For overlapping themes
  - **NMF**: For distinct, sparse themes
- Set confidence threshold (0.3 is a good starting point)

**Step 4: Run Analysis**
- Click "🚀 Run Analysis"
- Review configuration summary
- Click "🚀 Start Analysis" button
- Wait for completion (typically 5-30 seconds)

**Step 5: Review Results**
- Click "📊 Results Overview" to see:
  - Key metrics (coverage, codes found, confidence)
  - Top codes with keywords and examples
  - Code assignments table
- Click "📈 Visualizations" to explore:
  - Frequency charts
  - Co-occurrence heatmaps
  - Network diagrams
  - Distribution plots

**Step 6: Export Results**
- Click "💾 Export Results"
- Download Excel package (includes all results)
- Download individual components (CSV, codebook, etc.)

---

#### Option 2: Jupyter Notebook (For Programmers/Researchers)

**Step 1: Open Notebook**
```bash
jupyter notebook ml_open_coding_analysis.ipynb
```

**Step 2: Run Setup Cells** (Cells 1-4)
- Imports libraries
- Configures environment
- Defines helper functions

**Step 3: Load Your Data** (Cell 15)
```python
df = pd.read_csv('path/to/your/data.csv')
print(f"Loaded {len(df)} responses")
df.head()
```

**Step 4: Configure and Run** (Cell 17)
```python
coder = MLOpenCoder(
    n_codes=10,
    method='tfidf_kmeans',  # or 'lda', 'nmf'
    min_confidence=0.3
)

coder.fit(df['your_text_column'])
```

**Step 5: Generate Results** (Cell 19)
```python
results = OpenCodingResults(df, coder, response_col='your_text_column')
```

**Step 6: Review Outputs** (Cells 21-62)
- Execute cells sequentially
- Each cell generates one of the 15 essential outputs
- Review tables, charts, and summaries

**Step 7: Export** (Cells 54-55)
```python
exporter = ResultsExporter(results, output_dir='output')
exporter.export_all()
exporter.export_excel('my_results.xlsx')
```

---

### 3.2 How to Interpret Results

#### Understanding Metrics

**Coverage (%)**:
- **What it is**: Percentage of responses that received at least one code
- **Good**: > 80%
- **Acceptable**: 70-80%
- **Poor**: < 70% (may need to adjust parameters or review data quality)
- **Action**: If low, try lowering min_confidence or increasing n_codes

**Average Confidence**:
- **What it is**: Mean confidence score across all assignments
- **Good**: > 0.55
- **Acceptable**: 0.45-0.55
- **Poor**: < 0.45 (assignments may be weak)
- **Action**: If low, consider reducing n_codes or reviewing algorithm choice

**Silhouette Score** (TF-IDF + K-Means only):
- **What it is**: Measures cluster separation quality (-1 to 1)
- **Excellent**: > 0.5
- **Good**: 0.3-0.5
- **Acceptable**: 0.2-0.3
- **Poor**: < 0.2 (clusters overlap significantly)
- **Action**: If low, try different n_codes or use LDA/NMF

**Number of Codes**:
- **Too many**: Codes overlap, difficult to interpret
- **Too few**: Important themes missed, low granularity
- **Rule of thumb**: 5-15 codes for most datasets
- **Action**: Use silhouette analysis or coherence scores to optimize

---

#### Reading the Codebook

**Code Structure**:
```
CODE_01: quality_product_value
- Keywords: quality, product, value, worth, price, excellent
- Count: 32 responses (21.3%)
- Avg Confidence: 0.72
- Examples:
  1. "The quality of this product is excellent for the price" [0.85]
  2. "Great value, high quality materials" [0.78]
  3. "Quality could be better but decent for price point" [0.65]
```

**Interpretation**:
- **Label**: Auto-generated from top 3 keywords (can be manually relabeled)
- **Keywords**: Most characteristic words for this theme
- **Count**: How many responses assigned to this code
- **Percentage**: Proportion of total responses
- **Avg Confidence**: How strongly responses match this code
- **Examples**: Representative quotes with confidence scores

**Action Items**:
1. Review top 5-10 codes (capture majority of responses)
2. Read example quotes to validate code interpretation
3. Manually relabel codes with meaningful names
4. Look for overlap or redundancy between codes
5. Identify codes that should be merged or split

---

#### Understanding Visualizations

**Frequency Chart**:
- Shows distribution of codes across responses
- **Bars**: Height = number of responses
- **Colors**: Confidence levels (darker = higher confidence)
- **Look for**: Dominant themes, balanced distribution

**Co-occurrence Heatmap**:
- Shows which codes appear together in responses
- **Cells**: Color intensity = frequency of co-occurrence
- **Diagonal**: Self-occurrences (how often code appears)
- **Look for**: Related codes, complementary themes

**Network Diagram**:
- Visual representation of code relationships
- **Nodes**: Codes (size = frequency)
- **Edges**: Co-occurrence (thickness = frequency)
- **Look for**: Clusters of related codes, isolated codes

**Distribution Histogram**:
- Shows how many codes per response
- **X-axis**: Number of codes
- **Y-axis**: Number of responses
- **Look for**: Most responses have 1-3 codes (typical)

---

### 3.3 Common Questions and Answers

**Q: Why do some responses have no codes assigned?**

A: Responses may be uncoded for several reasons:
1. **Low confidence**: No code exceeded the confidence threshold
2. **Outliers**: Response is unlike any cluster/topic
3. **Poor quality**: Very short, vague, or off-topic responses
4. **Insufficient data**: Not enough signal for algorithm to assign

**Action**: Review uncoded responses manually. They may represent:
- Edge cases requiring manual coding
- New themes not captured by current analysis
- Data quality issues (test responses, gibberish)

---

**Q: How do I choose the right number of codes?**

A: Several approaches:
1. **Domain knowledge**: Based on your understanding of the data
2. **Silhouette analysis**: Run with different K values, plot silhouette scores
3. **Elbow method**: Plot inertia/reconstruction error vs. K, look for "elbow"
4. **Coherence scores**: For LDA, maximize topic coherence
5. **Iterative refinement**: Start with 8-12, adjust based on results

**Rule of thumb**: 5-15 codes for most datasets

---

**Q: Which ML method should I use?**

A: Decision guide:

**Use TF-IDF + K-Means when**:
- First time analyzing this data
- Need fast results for exploration
- Themes expected to be distinct
- Interpretability is priority
- Baseline for comparison

**Use LDA when**:
- Responses likely have multiple themes
- Want probabilistic interpretation
- Willing to trade speed for nuance
- Comparing topic prevalence across groups

**Use NMF when**:
- Need deterministic, reproducible results
- Themes expected to be sparse and non-overlapping
- Want faster alternative to LDA
- Interpretability via sparsity important

**Best practice**: Run all three methods and compare results for triangulation

---

**Q: How do I validate the results?**

A: Multi-step validation process:

1. **Manual Review** (30 responses):
   - Randomly sample coded responses
   - Verify code assignments make sense
   - Calculate agreement rate

2. **Expert Review** (3-5 experts):
   - Have domain experts review codebook
   - Assess keyword relevance and label accuracy
   - Gather feedback on interpretability

3. **Compare Methods**:
   - Run multiple algorithms
   - Check for consistency in major themes
   - Investigate discrepancies

4. **Statistical Validation**:
   - Check silhouette score (K-Means)
   - Review perplexity (LDA)
   - Assess coherence scores

5. **Iterative Refinement**:
   - Adjust parameters based on results
   - Merge redundant codes
   - Split overly broad codes
   - Re-run and validate improvements

---

**Q: Can I combine automated coding with manual coding?**

A: **Yes! Recommended approach**:

1. **Start Automated**: Use ML to discover initial themes
2. **Review & Refine**: Manually review top codes and examples
3. **Create Codebook**: Use ML-discovered codes as starting point
4. **Manual Coding**: Code subset manually for validation
5. **Inter-rater Reliability**: Calculate Cohen's kappa or similar
6. **Hybrid Approach**: Use ML for bulk coding, manual for edge cases
7. **Iterative**: Refine codebook based on manual review, re-run ML

**Benefits**:
- ML provides speed and consistency
- Human judgment adds nuance and context
- Combined approach balances efficiency and quality

---

**Q: What if codes don't make sense?**

A: Troubleshooting steps:

1. **Check data quality**:
   - Review sample responses
   - Look for test entries, duplicates, gibberish
   - Run quality filtering

2. **Adjust preprocessing**:
   - Try lemmatization
   - Adjust stopwords
   - Add domain-specific stopwords

3. **Tune parameters**:
   - Increase/decrease n_codes
   - Adjust confidence threshold
   - Try different algorithm

4. **Review keywords**:
   - Are top keywords meaningful?
   - Do examples match keywords?
   - Is vocabulary too generic?

5. **Consider domain-specific approach**:
   - Add custom stopwords
   - Use domain-specific embeddings
   - Pre-seed with known themes

---

**Q: How do I report results to stakeholders?**

A: Use the **Executive Summary** (auto-generated):

**Included automatically**:
1. Overview statistics (total responses, themes found, coverage)
2. Top 5 themes with descriptions and frequencies
3. Key insights (dominant themes, co-occurrences)
4. Recommendations (focus areas, further investigation)

**Additional reporting**:
1. **Visualizations**: Include frequency chart, co-occurrence heatmap
2. **Example quotes**: Provide 2-3 representative quotes per top theme
3. **Methodology**: Brief description of ML method used
4. **Caveats**: Note limitations (automated, requires validation, etc.)
5. **Next steps**: Recommend manual review, deep dives, etc.

**Template** (in executive_summary.md):
```markdown
## Executive Summary: [Dataset Name]

### Overview
- Analyzed: [N] responses
- Discovered: [K] themes
- Coverage: [X]% of responses coded
- Method: [TF-IDF + K-Means / LDA / NMF]

### Top Themes
1. **[Theme Name]** ([X]% of responses)
   - Description: [Auto-generated or manual]
   - Example: "[Representative quote]"

2. **[Theme Name]** ([X]% of responses)
   ...

### Key Insights
- [Insight 1: e.g., "Quality mentioned in 35% of responses"]
- [Insight 2: e.g., "Price and quality frequently co-occur"]
- [Insight 3: e.g., "Service complaints concentrated in Region A"]

### Recommendations
1. [Actionable recommendation based on findings]
2. [Further analysis suggestions]
3. [Areas for manual deep dive]
```

---

## 4. Handover Checklist

Use this checklist when handing over the framework to a new team member or client.

### 4.1 Documentation Review

- [ ] **README.md**: Reviewed and updated
- [ ] **Installation guide**: Tested on clean environment
- [ ] **API documentation**: Complete for all core classes
- [ ] **Methodology docs**: This file reviewed and accurate
- [ ] **Sample datasets**: All samples load correctly
- [ ] **Video walkthroughs**: Available or storyboard provided

### 4.2 Codebase Handover

- [ ] **Repository access**: Granted to new maintainers
- [ ] **Code structure**: Explained (src/, tests/, helpers/, etc.)
- [ ] **Key modules**: Walked through (data_loader, code_frame, etc.)
- [ ] **Extension points**: Documented and demonstrated
- [ ] **Test suite**: All tests passing
- [ ] **CI/CD pipeline**: Configured (if applicable)

### 4.3 Data and Outputs

- [ ] **Sample datasets**: Provided in `data/` folder
- [ ] **Expected outputs**: Examples shared for reference
- [ ] **Output formats**: Explained (CSV, Excel, JSON, visualizations)
- [ ] **Quality benchmarks**: Standards documented

### 4.4 Operational Readiness

- [ ] **Environment setup**: Dependencies installed and tested
- [ ] **Streamlit app**: Launches successfully
- [ ] **Jupyter notebooks**: All cells execute without errors
- [ ] **Database connections**: Tested (if used)
- [ ] **Export functionality**: Verified

### 4.5 Knowledge Transfer

- [ ] **Demo session**: Live demonstration conducted
- [ ] **Q&A session**: Common questions answered
- [ ] **Troubleshooting guide**: Common issues documented
- [ ] **Contact information**: Support contacts provided
- [ ] **Escalation process**: Defined for critical issues

### 4.6 Future Roadmap

- [ ] **Planned features**: Documented
- [ ] **Known limitations**: Listed
- [ ] **Enhancement requests**: Logged
- [ ] **Technical debt**: Identified and prioritized

---

## 5. Long-term Maintenance Notes

### 5.1 Dependencies Management

#### Current Dependencies (requirements.txt)

**Core Libraries**:
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
```

**NLP & ML**:
```
nltk>=3.8.0
gensim>=4.3.0
sentence-transformers>=2.2.0  # Optional
```

**Visualization**:
```
plotly>=5.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.0  # Optional
```

**Web & UI**:
```
streamlit>=1.28.0
```

**Data & Export**:
```
openpyxl>=3.1.0
xlrd>=2.0.0
python-docx>=1.0.0
```

**Database** (Optional):
```
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
```

**Testing & Development**:
```
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
```

#### Dependency Update Strategy

**Monthly**: Check for security updates
```bash
pip list --outdated
pip-audit  # Check for known vulnerabilities
```

**Quarterly**: Update minor versions
```bash
# Test in isolated environment first
pip install --upgrade pip
pip install --upgrade -r requirements.txt
pytest tests/  # Verify no breakage
```

**Annually**: Major version updates
- Review changelogs for breaking changes
- Update code for deprecated functions
- Run full test suite
- Update documentation

**Pinning Strategy**:
- Pin major+minor versions (e.g., `pandas>=2.0.0,<3.0.0`)
- Allow patch updates automatically
- Use `requirements-lock.txt` for exact reproducibility

---

### 5.2 Testing Strategy

#### Test Coverage Goals

- **Unit tests**: ≥ 80% coverage for src/
- **Integration tests**: All major workflows
- **Regression tests**: Previous bugs don't reoccur
- **Performance tests**: Execution time benchmarks

#### Running Tests

```bash
# Quick test (no coverage)
pytest tests/ -v

# Full test with coverage
pytest tests/ --cov=src --cov-report=html --cov-report=term

# Specific module
pytest tests/test_data_loader.py -v

# Parallel execution (faster)
pytest tests/ -n auto  # Requires pytest-xdist
```

#### Adding Tests for New Features

**Template**:
```python
# tests/test_new_feature.py
import pytest
from src.new_module import NewClass

@pytest.fixture
def sample_data():
    """Fixture providing test data."""
    return pd.DataFrame({'text': ['Sample 1', 'Sample 2']})

def test_new_feature_basic(sample_data):
    """Test basic functionality of new feature."""
    obj = NewClass()
    result = obj.process(sample_data)

    assert result is not None
    assert len(result) == len(sample_data)

def test_new_feature_edge_case():
    """Test edge case handling."""
    obj = NewClass()

    with pytest.raises(ValueError):
        obj.process(None)  # Should raise error

def test_new_feature_performance(sample_data):
    """Test performance requirements."""
    import time

    obj = NewClass()
    start = time.time()
    obj.process(sample_data)
    duration = time.time() - start

    assert duration < 1.0  # Should complete in < 1 second
```

---

### 5.3 Known Issues and Workarounds

#### Issue 1: NLTK Data Downloads

**Problem**: First run requires NLTK data downloads, may fail in restricted environments.

**Workaround**:
```python
# In app.py or notebook, add at top:
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
```

**Long-term fix**: Pre-package NLTK data or provide alternative download method.

---

#### Issue 2: Streamlit Rerun Overhead

**Problem**: Streamlit reruns entire script on every interaction, slow for large datasets.

**Workaround**:
```python
# Use session state for caching
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if st.button("Run Analysis"):
    results = run_analysis(...)  # Expensive operation
    st.session_state.analysis_results = results  # Cache in session
```

**Long-term fix**: Migrate to more stateful framework (Dash, FastAPI + React) if needed.

---

#### Issue 3: Large Dataset Memory Issues

**Problem**: Datasets > 100K responses may exceed available RAM.

**Workaround**:
```python
# Process in batches
def batch_process(df, batch_size=10000):
    results = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        batch_result = analyze(batch)
        results.append(batch_result)
    return pd.concat(results)
```

**Long-term fix**: Implement out-of-core processing with Dask or chunked reading.

---

#### Issue 4: Inconsistent Results Across Runs (LDA)

**Problem**: LDA produces different results each run despite same random seed.

**Root cause**: EM algorithm inherent variability, multi-threading randomness.

**Workaround**:
```python
# Run multiple times and ensemble
results = []
for seed in [42, 43, 44, 45, 46]:
    lda = LatentDirichletAllocation(n_components=10, random_state=seed)
    result = lda.fit_transform(doc_term_matrix)
    results.append(result)

# Average results or select most stable
final_result = np.mean(results, axis=0)
```

**Long-term fix**: Provide option to run ensemble automatically, report stability metric.

---

### 5.4 Performance Optimization Tips

#### For Large Datasets (>10K responses)

1. **Reduce max_features**: Lower from 1000 to 500
2. **Increase min_df**: Filter rare words (min_df=5 or higher)
3. **Use TF-IDF + K-Means**: Fastest algorithm
4. **Batch processing**: Process in chunks
5. **Parallel processing**: Use n_jobs=-1 where available
6. **Sparse matrices**: Keep matrices sparse (don't use .toarray() unless necessary)

#### For Real-time Analysis

1. **Pre-compute models**: Train model offline, load for inference
2. **Cache results**: Use session state or Redis
3. **Async processing**: Move ML operations to background jobs
4. **Progressive updates**: Show partial results while processing

#### Memory Optimization

```python
# Use sparse matrices
from scipy.sparse import csr_matrix

# Avoid creating large dense arrays
# Bad:
dense_matrix = feature_matrix.toarray()  # Memory explosion!

# Good:
# Operate directly on sparse matrix
distances = euclidean_distances(feature_matrix, cluster_centers)

# Clean up large objects
import gc
del large_matrix
gc.collect()
```

---

### 5.5 Scaling Considerations

#### When to Scale Up

**Indicators**:
- Datasets consistently > 50K responses
- Analysis takes > 5 minutes
- Memory usage > 8GB RAM
- Multiple concurrent users

**Scaling Strategies**:

1. **Vertical Scaling** (easier):
   - Upgrade to machine with more RAM (16GB → 32GB)
   - Use faster CPU (more cores for parallel processing)
   - Add GPU for embeddings (optional)

2. **Horizontal Scaling** (more complex):
   - Distributed processing with Dask or Spark
   - Cloud deployment (AWS Lambda, Google Cloud Functions)
   - Queue-based architecture (Celery + Redis)

3. **Algorithmic Optimization**:
   - Migrate to polars for faster data processing
   - Use cuML (RAPIDS) for GPU-accelerated clustering
   - Implement incremental learning for online updates

#### Cloud Deployment Considerations

**AWS**:
- **EC2**: Full control, suitable for Streamlit deployment
- **Lambda**: Serverless, good for API endpoints
- **SageMaker**: Managed ML infrastructure (overkill for this use case)

**Google Cloud**:
- **Cloud Run**: Containerized Streamlit app
- **App Engine**: Managed app hosting
- **Vertex AI**: Managed notebooks

**Azure**:
- **App Service**: Web app hosting
- **Container Instances**: Docker deployment
- **Machine Learning**: Managed ML platform

**Recommendation**: Start with simple EC2/Cloud Run deployment, scale as needed.

---

## Document Status

**Status**: Complete
**Version**: 1.0
**Last Updated**: 2025-12-26
**Maintained By**: Framework Development Team

**Related Documents**:
- [06_validation_and_demonstration.md](./06_validation_and_demonstration.md) - Validation procedures
- [01_open_source_tools_review.md](./01_open_source_tools_review.md) - Tools selection rationale
- [README.md](../README.md) - Quick start guide

**Review Schedule**: Quarterly or after major updates

---

**End of Documentation and Handover Guide**
