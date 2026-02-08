# Optimal Dataset Size Guide

This document provides evidence-based recommendations for dataset sizes that ensure
all ML methods in this project train and predict with good accuracy.

---

## Executive Summary

| Dataset Size | Suitability | Recommended Methods |
|---|---|---|
| < 50 | Not recommended | Manual qualitative coding preferred |
| 50-99 | Minimal viable | TF-IDF + K-Means only |
| 100-299 | Good for most methods | TF-IDF, NMF, SVM Spectral |
| **300-500** | **Optimal range** | **All 6 methods perform well** |
| 500-1,000 | Strong | All methods; LDA and BERT excel here |
| 1,000-5,000 | Excellent | All methods; LDA and deep embeddings shine |
| 5,000+ | Enterprise scale | All methods; prefer TF-IDF/NMF for speed |

**The recommended optimal dataset size is 300-500 responses.** This range provides
sufficient data for all six ML methods to produce meaningful, stable clusters while
remaining computationally efficient.

---

## Method-by-Method Optimal Sizes

### 1. TF-IDF + K-Means (Default Method)

| Parameter | Constraint | Impact on Minimum Size |
|---|---|---|
| `min_df=2` | Each term must appear in 2+ documents | Needs 50+ docs for adequate vocabulary |
| `max_df=0.8` | Terms in >80% of docs are filtered | Needs 20+ docs to differentiate common/rare |
| `max_features=1000` | Vocabulary capped at 1,000 terms | More data = richer vocabulary selection |
| `ngram_range=(1,3)` | Unigrams through trigrams | Trigrams need 100+ docs to be meaningful |
| `n_init=10` | 10 K-Means initializations | Stable across sizes, but more data = more consistent |

- **Minimum:** 50 responses
- **Recommended:** 150+ responses
- **Optimal:** 200-500 responses
- **Strengths:** Most robust to small datasets; fast; interpretable
- **Failure mode:** Below 30 responses, `min_df=2` can produce an empty vocabulary

### 2. LDA (Latent Dirichlet Allocation)

LDA is a probabilistic generative model that discovers latent topics. It models each
document as a mixture of topics and each topic as a distribution over words.

- **Minimum:** 100 responses
- **Recommended:** 300+ responses
- **Optimal:** 500-2,000 responses
- **Strengths:** Excels with larger datasets and longer texts; probabilistic topic assignments
- **Failure mode:** Below 100 responses, topic distributions are unstable and topics
  become incoherent. LDA needs enough co-occurrence data to estimate the document-topic
  and topic-word distributions reliably.
- **Rule of thumb:** At least 50 responses per requested topic (e.g., 10 topics needs 500+ responses)

### 3. NMF (Non-negative Matrix Factorization)

NMF decomposes the TF-IDF matrix into two non-negative matrices, yielding additive,
parts-based topic representations.

- **Minimum:** 50 responses
- **Recommended:** 150+ responses
- **Optimal:** 200-500 responses
- **Strengths:** Comparable to TF-IDF+K-Means in small-data tolerance; produces coherent,
  non-overlapping topics; deterministic
- **Failure mode:** With very sparse matrices (few docs, many features), factorization
  can produce degenerate solutions

### 4. LSTM Embeddings + K-Means

The LSTM autoencoder trains on the corpus itself, learning sequential patterns
before passing embeddings to K-Means.

| Parameter | Constraint | Impact on Minimum Size |
|---|---|---|
| `sequence_length=100` | Max 100 tokens per response | Needs varied-length text |
| `embedding_dim=100` | 100-dimensional embeddings | Needs enough data to learn meaningful representations |
| `epochs=10` | 10 training iterations | Underfits on small datasets |
| `dropout=0.2` | 20% dropout rate | Regularization assumes enough training samples |

- **Minimum:** 200 responses
- **Recommended:** 500+ responses
- **Optimal:** 500-2,000 responses
- **Strengths:** Captures sequential/narrative patterns; good for longer texts
- **Failure mode:** Below 200 responses, the autoencoder cannot learn meaningful
  representations and embeddings become noise. The model has thousands of trainable
  parameters and needs sufficient data to generalize.
- **Note:** Training time increases linearly with dataset size

### 5. BERT Embeddings + K-Means

Uses a pre-trained Sentence-BERT model (`all-MiniLM-L6-v2`) which generates 384-dimensional
semantic embeddings. No training on the corpus is required.

- **Minimum:** 50 responses
- **Recommended:** 100+ responses
- **Optimal:** 200-500 responses
- **Strengths:** Best semantic understanding; handles synonyms and paraphrases;
  pre-trained (no corpus-specific training needed); works well even at 50 responses
- **Failure mode:** K-Means can struggle in high-dimensional space (384 dims) with
  very few samples. Below 50 responses, cluster centroids are unreliable.
- **Note:** Inference is slower than TF-IDF but dataset size has minimal impact on
  embedding quality since the model is pre-trained

### 6. SVM Spectral Clustering

Uses Spectral Clustering with an RBF kernel, which constructs a similarity graph
and partitions it using eigenvectors of the graph Laplacian.

- **Minimum:** 30 responses
- **Recommended:** 100-300 responses
- **Optimal:** 100-500 responses
- **Strengths:** Best for non-linear cluster boundaries; explicitly designed for
  smaller datasets in this project's auto-selection logic
- **Failure mode:** Computational cost is O(n^3) for eigendecomposition, making it
  impractical above 2,000 responses. Below 30 responses, the similarity graph
  is too sparse for meaningful partitioning.
- **Note:** The project's auto-method selector penalizes this method for large datasets

---

## Supporting Embedding Methods

These embedding methods feed into K-Means clustering and have their own data requirements:

### Word2Vec Embedder

- **Minimum corpus:** 200 responses (needs enough context windows)
- **`min_count=2`:** Words appearing once are excluded
- **Warning trigger:** Vocabulary < 50 unique words
- **Optimal:** 500+ responses with diverse vocabulary
- **Key limitation:** Trains on the corpus, so small datasets produce poor word vectors

### FastText Embedder

- **Minimum corpus:** 150 responses (subword information helps)
- **`min_count=2`:** Same as Word2Vec
- **Optimal:** 300+ responses
- **Advantage over Word2Vec:** Handles typos and rare words via subword decomposition,
  so it degrades more gracefully on smaller datasets

---

## Dataset Size vs. Number of Codes

The number of themes/codes you request must be supported by the data. The project
enforces: `max_codes = min(n_samples, n_features, requested_max)`.

| Desired Codes | Minimum Responses | Recommended Responses | Responses per Code |
|---|---|---|---|
| 3-5 | 30 | 75-100 | 15-25 |
| 6-10 | 60 | 150-300 | 15-30 |
| 10-15 | 100 | 300-500 | 20-35 |
| 15-20 | 200 | 500-1,000 | 25-50 |
| 20-30 | 300 | 1,000+ | 30-50+ |

**Rule of thumb:** Maintain at least 15-30 responses per requested code for stable clusters.

---

## Quality Thresholds by Dataset Size

Expected clustering quality metrics at different dataset sizes (based on the project's
evaluation framework):

| Metric | 50 responses | 100 responses | 300 responses | 500+ responses |
|---|---|---|---|---|
| Silhouette Score | 0.15-0.30 | 0.20-0.40 | 0.25-0.50 | 0.30-0.55 |
| Coverage (%) | 60-75% | 70-85% | 80-92% | 85-95% |
| Theme Coherence | Low-Medium | Medium | Medium-High | High |
| Code Stability | Unstable | Moderate | Stable | Very Stable |
| Bootstrap Consistency | Low | Moderate | High | Very High |

The project's rigor diagnostics target:
- Coverage >= 80% (achievable at 200+ responses)
- Code Utilization >= 75% (achievable at 150+ responses)
- Theme Coherence >= 0.50 (achievable at 300+ responses)

---

## Recommendations by Use Case

### Quick Exploratory Analysis
- **Size:** 100-200 responses
- **Method:** TF-IDF + K-Means or BERT + K-Means
- **Codes:** 5-8

### Standard Research Analysis
- **Size:** 300-500 responses
- **Method:** Auto-select (all methods viable)
- **Codes:** 8-15

### Publication-Quality Analysis
- **Size:** 500-1,000+ responses
- **Method:** Run multiple methods and compare (triangulation)
- **Codes:** 10-20
- **Additional:** Enable rigor diagnostics, bootstrap stability testing

### Large-Scale Survey Analysis
- **Size:** 1,000-10,000+ responses
- **Method:** TF-IDF + K-Means or NMF (for speed); LDA (for depth)
- **Codes:** 15-30
- **Note:** Avoid SVM Spectral above 2,000 responses (O(n^3) cost)

---

## Practical Constraints from Project Configuration

These are the hard constraints enforced by the codebase:

| Constraint | Value | Source |
|---|---|---|
| Absolute minimum responses | 5 (after null filtering) | Data loader validation |
| Sanity check minimum | 20 responses | `rigor_diagnostics.py` |
| Statistical validity minimum | 20-30 responses | `03_input_data_specification.md` |
| Recommended minimum | 50+ responses | Documentation |
| min_df default | 2 documents | `vectorizer_factory.py` |
| max_df default | 0.8 (80%) | `vectorizer_factory.py` |
| Word2Vec/FastText min_count | 2 occurrences | `embeddings.py` |
| Vocabulary warning threshold | < 50 unique words | `embeddings.py` |
| Max features (vocabulary) | 1,000 terms | `vectorizer_factory.py` |
| Auto-select size boundaries | < 100 / 100-500 / >= 500 | `helpers/analysis.py` |

---

## Summary

For this project to deliver good accuracy across **all six ML methods**:

- **Minimum viable:** 200 responses (covers all methods except LDA may underperform)
- **Optimal sweet spot:** **300-500 responses** (all methods perform well, including
  LSTM and LDA; computation remains fast)
- **Best accuracy ceiling:** 500-1,000 responses (diminishing returns beyond this
  for most methods; LDA continues to improve)

The current demo datasets (200-300 responses each) are well-sized for demonstrating
most methods. For production use targeting all six methods at good accuracy, scaling
to **300-500 responses** is the recommended target.
