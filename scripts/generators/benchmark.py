"""Generate benchmark standards documentation from codebase analysis."""

import re
from pathlib import Path
from typing import List, Dict, Any, Set
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import ProjectConfig
from utils.document import (
    create_document, set_document_defaults, add_heading,
    add_paragraph, add_bullet_point, add_table, save_document
)
from utils.introspection import discover_project, get_dependencies


# Comprehensive technique database with gold standards and references
TECHNIQUE_DATABASE = {
    # Text Vectorization
    'tfidf': {
        'name': 'TF-IDF (Term Frequency-Inverse Document Frequency)',
        'category': 'Text Vectorization',
        'description': 'Numerical statistic reflecting word importance in a document relative to a corpus.',
        'gold_standard': 'Scikit-learn implementation using sublinear TF scaling and L2 normalization.',
        'expected_output': 'Sparse matrix of shape (n_documents, n_features) with normalized term weights.',
        'quality_metrics': [
            'Sparsity ratio typically 95-99% for text data',
            'Feature values between 0 and 1 after normalization',
            'Vocabulary size depends on max_features parameter',
        ],
        'references': [
            {'title': 'A Statistical Interpretation of Term Specificity', 'authors': 'Spärck Jones, K.', 'year': '1972', 'venue': 'Journal of Documentation'},
            {'title': 'Term-Weighting Approaches in Automatic Text Retrieval', 'authors': 'Salton, G., Buckley, C.', 'year': '1988', 'venue': 'Information Processing & Management'},
        ],
        'implementations': ['scikit-learn TfidfVectorizer', 'Gensim TfidfModel'],
        'patterns': [r'tfidf', r'TfidfVectorizer', r'TfidfTransformer', r'tf-idf', r'tf_idf'],
    },
    'countvectorizer': {
        'name': 'Count Vectorizer (Bag of Words)',
        'category': 'Text Vectorization',
        'description': 'Converts text documents to a matrix of token counts.',
        'gold_standard': 'Scikit-learn CountVectorizer with configurable n-gram ranges.',
        'expected_output': 'Sparse matrix of shape (n_documents, n_features) with integer counts.',
        'quality_metrics': [
            'Non-negative integer values',
            'Row sums equal total tokens per document',
        ],
        'references': [
            {'title': 'A Vector Space Model for Automatic Indexing', 'authors': 'Salton, G., Wong, A., Yang, C.S.', 'year': '1975', 'venue': 'Communications of the ACM'},
        ],
        'implementations': ['scikit-learn CountVectorizer'],
        'patterns': [r'CountVectorizer', r'bag.of.words', r'bow', r'count_vectorizer'],
    },
    'word2vec': {
        'name': 'Word2Vec',
        'category': 'Word Embeddings',
        'description': 'Neural network-based word embeddings capturing semantic relationships.',
        'gold_standard': 'Gensim implementation with configurable CBOW or Skip-gram architecture.',
        'expected_output': 'Dense word vectors of configurable dimensionality (typically 100-300).',
        'quality_metrics': [
            'Cosine similarity captures semantic relationships',
            'Analogies test (king - man + woman ≈ queen)',
            'Word similarity correlation with human judgments',
        ],
        'references': [
            {'title': 'Efficient Estimation of Word Representations in Vector Space', 'authors': 'Mikolov, T., et al.', 'year': '2013', 'venue': 'arXiv:1301.3781'},
            {'title': 'Distributed Representations of Words and Phrases', 'authors': 'Mikolov, T., et al.', 'year': '2013', 'venue': 'NIPS'},
        ],
        'implementations': ['Gensim Word2Vec', 'TensorFlow', 'PyTorch'],
        'patterns': [r'word2vec', r'Word2Vec', r'w2v'],
    },
    # Clustering
    'kmeans': {
        'name': 'K-Means Clustering',
        'category': 'Clustering',
        'description': 'Partitioning algorithm that divides data into k clusters by minimizing within-cluster variance.',
        'gold_standard': 'Scikit-learn KMeans with k-means++ initialization.',
        'expected_output': 'Cluster labels (0 to k-1) for each sample, cluster centroids.',
        'quality_metrics': [
            'Silhouette Score: -1 to 1 (higher is better, >0.5 is good)',
            'Inertia: Sum of squared distances to centroids (lower is better)',
            'Calinski-Harabasz Index: Higher values indicate better clustering',
            'Davies-Bouldin Index: Lower values indicate better clustering',
        ],
        'references': [
            {'title': 'Some Methods for Classification and Analysis of Multivariate Observations', 'authors': 'MacQueen, J.', 'year': '1967', 'venue': 'Berkeley Symposium on Mathematical Statistics and Probability'},
            {'title': 'k-means++: The Advantages of Careful Seeding', 'authors': 'Arthur, D., Vassilvitskii, S.', 'year': '2007', 'venue': 'SODA'},
        ],
        'implementations': ['scikit-learn KMeans', 'MiniBatchKMeans'],
        'patterns': [r'kmeans', r'KMeans', r'k-means', r'k_means'],
    },
    'hierarchical': {
        'name': 'Hierarchical/Agglomerative Clustering',
        'category': 'Clustering',
        'description': 'Bottom-up clustering that builds a hierarchy of clusters.',
        'gold_standard': 'Scikit-learn AgglomerativeClustering with Ward linkage.',
        'expected_output': 'Cluster labels and dendrogram showing merge hierarchy.',
        'quality_metrics': [
            'Cophenetic correlation coefficient (>0.7 indicates good hierarchy)',
            'Silhouette Score for cut-off cluster assignments',
        ],
        'references': [
            {'title': 'Hierarchical Grouping to Optimize an Objective Function', 'authors': 'Ward, J.H.', 'year': '1963', 'venue': 'Journal of the American Statistical Association'},
        ],
        'implementations': ['scikit-learn AgglomerativeClustering', 'scipy.cluster.hierarchy'],
        'patterns': [r'hierarchical', r'agglomerative', r'AgglomerativeClustering', r'dendrogram', r'linkage'],
    },
    'dbscan': {
        'name': 'DBSCAN (Density-Based Spatial Clustering)',
        'category': 'Clustering',
        'description': 'Density-based clustering that finds arbitrarily shaped clusters and identifies outliers.',
        'gold_standard': 'Scikit-learn DBSCAN with epsilon and min_samples tuning.',
        'expected_output': 'Cluster labels (-1 for noise points).',
        'quality_metrics': [
            'Silhouette Score (excluding noise points)',
            'Percentage of points classified as noise',
            'Number of clusters discovered',
        ],
        'references': [
            {'title': 'A Density-Based Algorithm for Discovering Clusters', 'authors': 'Ester, M., et al.', 'year': '1996', 'venue': 'KDD'},
        ],
        'implementations': ['scikit-learn DBSCAN', 'HDBSCAN'],
        'patterns': [r'dbscan', r'DBSCAN', r'density.based'],
    },
    # Topic Modeling
    'lda': {
        'name': 'Latent Dirichlet Allocation (LDA)',
        'category': 'Topic Modeling',
        'description': 'Generative probabilistic model for discovering abstract topics in document collections.',
        'gold_standard': 'Gensim LdaModel or scikit-learn LatentDirichletAllocation.',
        'expected_output': 'Document-topic distributions and topic-word distributions.',
        'quality_metrics': [
            'Coherence Score (C_v): 0.4-0.7 is typical, higher is better',
            'Perplexity: Lower is better (but can overfit)',
            'Topic Diversity: Unique words across top-N words per topic',
            'Human interpretability of top words per topic',
        ],
        'references': [
            {'title': 'Latent Dirichlet Allocation', 'authors': 'Blei, D.M., Ng, A.Y., Jordan, M.I.', 'year': '2003', 'venue': 'Journal of Machine Learning Research'},
            {'title': 'Online Learning for Latent Dirichlet Allocation', 'authors': 'Hoffman, M., Blei, D.M., Bach, F.', 'year': '2010', 'venue': 'NIPS'},
        ],
        'implementations': ['Gensim LdaModel', 'scikit-learn LatentDirichletAllocation', 'MALLET'],
        'patterns': [r'\blda\b', r'LDA', r'LatentDirichletAllocation', r'latent.dirichlet'],
    },
    'nmf': {
        'name': 'Non-negative Matrix Factorization (NMF)',
        'category': 'Topic Modeling',
        'description': 'Matrix decomposition technique producing interpretable, additive topic representations.',
        'gold_standard': 'Scikit-learn NMF with Frobenius norm or KL divergence.',
        'expected_output': 'Document-topic matrix (W) and topic-term matrix (H).',
        'quality_metrics': [
            'Reconstruction error (Frobenius norm)',
            'Topic coherence scores',
            'Sparsity of resulting matrices',
        ],
        'references': [
            {'title': 'Learning the Parts of Objects by Non-negative Matrix Factorization', 'authors': 'Lee, D.D., Seung, H.S.', 'year': '1999', 'venue': 'Nature'},
            {'title': 'Algorithms for Non-negative Matrix Factorization', 'authors': 'Lee, D.D., Seung, H.S.', 'year': '2001', 'venue': 'NIPS'},
        ],
        'implementations': ['scikit-learn NMF'],
        'patterns': [r'\bnmf\b', r'NMF', r'NonNegativeMatrixFactorization', r'non.negative.matrix'],
    },
    'lsa': {
        'name': 'Latent Semantic Analysis (LSA/LSI)',
        'category': 'Topic Modeling',
        'description': 'Dimensionality reduction using truncated SVD on term-document matrices.',
        'gold_standard': 'Scikit-learn TruncatedSVD or Gensim LsiModel.',
        'expected_output': 'Reduced-dimension document representations.',
        'quality_metrics': [
            'Explained variance ratio',
            'Semantic similarity preservation',
        ],
        'references': [
            {'title': 'Indexing by Latent Semantic Analysis', 'authors': 'Deerwester, S., et al.', 'year': '1990', 'venue': 'Journal of the American Society for Information Science'},
        ],
        'implementations': ['scikit-learn TruncatedSVD', 'Gensim LsiModel'],
        'patterns': [r'\blsa\b', r'\blsi\b', r'LSA', r'LSI', r'TruncatedSVD', r'latent.semantic'],
    },
    # Classification
    'naive_bayes': {
        'name': 'Naive Bayes Classifier',
        'category': 'Classification',
        'description': 'Probabilistic classifier based on Bayes theorem with feature independence assumption.',
        'gold_standard': 'Scikit-learn MultinomialNB for text classification.',
        'expected_output': 'Class labels and probability estimates.',
        'quality_metrics': [
            'Accuracy, Precision, Recall, F1-Score',
            'ROC-AUC for binary classification',
            'Confusion matrix',
        ],
        'references': [
            {'title': 'A Comparison of Event Models for Naive Bayes Text Classification', 'authors': 'McCallum, A., Nigam, K.', 'year': '1998', 'venue': 'AAAI Workshop'},
        ],
        'implementations': ['scikit-learn MultinomialNB', 'GaussianNB', 'BernoulliNB'],
        'patterns': [r'naive.bayes', r'NaiveBayes', r'MultinomialNB', r'GaussianNB'],
    },
    'svm': {
        'name': 'Support Vector Machine (SVM)',
        'category': 'Classification',
        'description': 'Maximum-margin classifier using kernel functions.',
        'gold_standard': 'Scikit-learn SVC with RBF or linear kernel.',
        'expected_output': 'Class labels, decision function values.',
        'quality_metrics': [
            'Accuracy, Precision, Recall, F1-Score',
            'Margin width (larger is better)',
            'Number of support vectors (fewer indicates better generalization)',
        ],
        'references': [
            {'title': 'A Training Algorithm for Optimal Margin Classifiers', 'authors': 'Boser, B., Guyon, I., Vapnik, V.', 'year': '1992', 'venue': 'COLT'},
            {'title': 'Support-Vector Networks', 'authors': 'Cortes, C., Vapnik, V.', 'year': '1995', 'venue': 'Machine Learning'},
        ],
        'implementations': ['scikit-learn SVC', 'LinearSVC', 'SGDClassifier'],
        'patterns': [r'\bsvm\b', r'SVM', r'SVC', r'support.vector', r'LinearSVC'],
    },
    'random_forest': {
        'name': 'Random Forest',
        'category': 'Classification/Regression',
        'description': 'Ensemble of decision trees using bagging and feature randomization.',
        'gold_standard': 'Scikit-learn RandomForestClassifier with 100+ trees.',
        'expected_output': 'Class labels, probability estimates, feature importances.',
        'quality_metrics': [
            'Out-of-bag (OOB) error estimate',
            'Feature importance rankings',
            'Accuracy, Precision, Recall, F1-Score',
        ],
        'references': [
            {'title': 'Random Forests', 'authors': 'Breiman, L.', 'year': '2001', 'venue': 'Machine Learning'},
        ],
        'implementations': ['scikit-learn RandomForestClassifier', 'RandomForestRegressor'],
        'patterns': [r'random.forest', r'RandomForest', r'RandomForestClassifier'],
    },
    # NLP Preprocessing
    'tokenization': {
        'name': 'Tokenization',
        'category': 'NLP Preprocessing',
        'description': 'Splitting text into individual tokens (words, subwords, or characters).',
        'gold_standard': 'NLTK word_tokenize or spaCy tokenizer.',
        'expected_output': 'List of tokens preserving meaningful units.',
        'quality_metrics': [
            'Handles punctuation correctly',
            'Preserves contractions appropriately',
            'Handles special characters and numbers',
        ],
        'references': [
            {'title': 'Natural Language Processing with Python', 'authors': 'Bird, S., Klein, E., Loper, E.', 'year': '2009', 'venue': "O'Reilly Media"},
        ],
        'implementations': ['NLTK word_tokenize', 'spaCy', 'transformers tokenizers'],
        'patterns': [r'tokeniz', r'word_tokenize', r'sent_tokenize', r'\.tokenize'],
    },
    'stemming': {
        'name': 'Stemming',
        'category': 'NLP Preprocessing',
        'description': 'Reducing words to their stem/root form using rule-based algorithms.',
        'gold_standard': 'NLTK PorterStemmer or SnowballStemmer.',
        'expected_output': 'Stemmed tokens (may not be valid words).',
        'quality_metrics': [
            'Consistency of stemming (same stem for word variants)',
            'Vocabulary reduction ratio',
        ],
        'references': [
            {'title': 'An Algorithm for Suffix Stripping', 'authors': 'Porter, M.F.', 'year': '1980', 'venue': 'Program'},
        ],
        'implementations': ['NLTK PorterStemmer', 'SnowballStemmer', 'LancasterStemmer'],
        'patterns': [r'stemm', r'PorterStemmer', r'SnowballStemmer', r'\.stem\('],
    },
    'lemmatization': {
        'name': 'Lemmatization',
        'category': 'NLP Preprocessing',
        'description': 'Reducing words to their dictionary form (lemma) using morphological analysis.',
        'gold_standard': 'spaCy lemmatizer or NLTK WordNetLemmatizer.',
        'expected_output': 'Lemmatized tokens (valid dictionary words).',
        'quality_metrics': [
            'Accuracy of lemma identification',
            'Preservation of word meaning',
        ],
        'references': [
            {'title': 'WordNet: A Lexical Database for English', 'authors': 'Miller, G.A.', 'year': '1995', 'venue': 'Communications of the ACM'},
        ],
        'implementations': ['NLTK WordNetLemmatizer', 'spaCy'],
        'patterns': [r'lemmatiz', r'WordNetLemmatizer', r'\.lemma'],
    },
    'stopwords': {
        'name': 'Stop Words Removal',
        'category': 'NLP Preprocessing',
        'description': 'Filtering out common words that carry little semantic meaning.',
        'gold_standard': 'NLTK or spaCy stop word lists, customized per domain.',
        'expected_output': 'Token list without stop words.',
        'quality_metrics': [
            'Appropriate stop word list for domain',
            'Preservation of meaningful short words',
        ],
        'references': [
            {'title': 'Introduction to Information Retrieval', 'authors': 'Manning, C.D., Raghavan, P., Schütze, H.', 'year': '2008', 'venue': 'Cambridge University Press'},
        ],
        'implementations': ['NLTK stopwords', 'spaCy stop words', 'scikit-learn stop_words'],
        'patterns': [r'stopword', r'stop_word', r'STOPWORDS', r'is_stop'],
    },
    # Dimensionality Reduction
    'pca': {
        'name': 'Principal Component Analysis (PCA)',
        'category': 'Dimensionality Reduction',
        'description': 'Linear dimensionality reduction using orthogonal transformation.',
        'gold_standard': 'Scikit-learn PCA with automatic component selection.',
        'expected_output': 'Transformed data in reduced dimensions, explained variance ratios.',
        'quality_metrics': [
            'Cumulative explained variance (aim for 80-95%)',
            'Scree plot elbow identification',
        ],
        'references': [
            {'title': 'Principal Component Analysis', 'authors': 'Jolliffe, I.T.', 'year': '2002', 'venue': 'Springer'},
        ],
        'implementations': ['scikit-learn PCA', 'IncrementalPCA'],
        'patterns': [r'\bpca\b', r'PCA', r'principal.component'],
    },
    'tsne': {
        'name': 't-SNE (t-distributed Stochastic Neighbor Embedding)',
        'category': 'Dimensionality Reduction',
        'description': 'Non-linear dimensionality reduction for visualization.',
        'gold_standard': 'Scikit-learn TSNE with perplexity tuning.',
        'expected_output': '2D or 3D embeddings for visualization.',
        'quality_metrics': [
            'Preservation of local structure',
            'Cluster separation in visualization',
            'KL divergence (lower is better)',
        ],
        'references': [
            {'title': 'Visualizing Data using t-SNE', 'authors': 'van der Maaten, L., Hinton, G.', 'year': '2008', 'venue': 'JMLR'},
        ],
        'implementations': ['scikit-learn TSNE', 'openTSNE'],
        'patterns': [r'tsne', r'TSNE', r't-sne', r't_sne'],
    },
    'umap': {
        'name': 'UMAP (Uniform Manifold Approximation and Projection)',
        'category': 'Dimensionality Reduction',
        'description': 'Non-linear dimensionality reduction preserving global and local structure.',
        'gold_standard': 'umap-learn library with n_neighbors and min_dist tuning.',
        'expected_output': 'Low-dimensional embeddings.',
        'quality_metrics': [
            'Preservation of global structure (better than t-SNE)',
            'Trustworthiness and continuity metrics',
        ],
        'references': [
            {'title': 'UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction', 'authors': 'McInnes, L., Healy, J., Melville, J.', 'year': '2018', 'venue': 'arXiv:1802.03426'},
        ],
        'implementations': ['umap-learn'],
        'patterns': [r'umap', r'UMAP'],
    },
    # Evaluation Metrics
    'silhouette': {
        'name': 'Silhouette Score',
        'category': 'Evaluation Metrics',
        'description': 'Measures how similar objects are to their own cluster vs other clusters.',
        'gold_standard': 'Scikit-learn silhouette_score.',
        'expected_output': 'Score between -1 and 1.',
        'quality_metrics': [
            '>0.7: Strong structure',
            '0.5-0.7: Reasonable structure',
            '0.25-0.5: Weak structure',
            '<0.25: No substantial structure',
        ],
        'references': [
            {'title': 'Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis', 'authors': 'Rousseeuw, P.J.', 'year': '1987', 'venue': 'Computational and Applied Mathematics'},
        ],
        'implementations': ['scikit-learn silhouette_score', 'silhouette_samples'],
        'patterns': [r'silhouette', r'silhouette_score'],
    },
    'coherence': {
        'name': 'Topic Coherence',
        'category': 'Evaluation Metrics',
        'description': 'Measures semantic interpretability of discovered topics.',
        'gold_standard': 'Gensim CoherenceModel with C_v measure.',
        'expected_output': 'Coherence score (higher is better).',
        'quality_metrics': [
            'C_v: 0.4-0.7 typical range',
            'C_umass: Negative values, closer to 0 is better',
        ],
        'references': [
            {'title': 'Exploring the Space of Topic Coherence Measures', 'authors': 'Röder, M., Both, A., Hinneburg, A.', 'year': '2015', 'venue': 'WSDM'},
        ],
        'implementations': ['Gensim CoherenceModel'],
        'patterns': [r'coherence', r'CoherenceModel', r'c_v', r'u_mass'],
    },
}


def detect_techniques(project_root: str, source_dirs: List[str]) -> List[Dict[str, Any]]:
    """Detect ML/NLP techniques used in the codebase.

    Args:
        project_root: Path to project root
        source_dirs: Source directories to scan

    Returns:
        List of detected technique info dictionaries
    """
    root = Path(project_root)
    detected = {}

    for source_dir in source_dirs:
        source_path = root / source_dir
        if not source_path.exists():
            continue

        for py_file in source_path.rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                for tech_id, tech_info in TECHNIQUE_DATABASE.items():
                    if tech_id in detected:
                        continue

                    for pattern in tech_info['patterns']:
                        if re.search(pattern, content, re.IGNORECASE):
                            detected[tech_id] = {
                                **tech_info,
                                'id': tech_id,
                                'found_in': str(py_file.relative_to(root)),
                            }
                            break

            except (IOError, UnicodeDecodeError):
                continue

    return list(detected.values())


def generate_benchmark_standards(config: ProjectConfig) -> str:
    """Generate benchmark standards documentation.

    Args:
        config: Project configuration

    Returns:
        Path to generated document
    """
    doc = create_document(config.formatting.margin_inches)
    set_document_defaults(doc, config.formatting.font_name, config.formatting.font_size_body)

    fmt = config.formatting

    # Detect techniques in codebase
    techniques = detect_techniques(config.project_root, config.source_dirs)

    # Title
    add_heading(doc, f"{config.framework_name} - Benchmark Standards", level=0,
                font_name=fmt.font_name)

    add_paragraph(doc, f"Version: {config.version}", font_name=fmt.font_name,
                  font_size=fmt.font_size_body, italic=True)

    # 1. Introduction
    add_heading(doc, "1. Introduction", level=1, font_name=fmt.font_name)

    intro_text = config.custom_sections.get('benchmark_intro') or (
        f"This document establishes benchmark standards for techniques used in {config.framework_name}. "
        f"Each technique is documented with gold-standard outputs, quality metrics, and authoritative references. "
        f"These standards ensure reproducibility and enable quality assessment of analysis results."
    )
    add_paragraph(doc, intro_text, font_name=fmt.font_name, font_size=fmt.font_size_body)

    # 2. Techniques Overview
    add_heading(doc, "2. Detected Techniques Overview", level=1, font_name=fmt.font_name)

    if techniques:
        add_paragraph(doc, f"The following {len(techniques)} techniques were detected in the codebase:",
                      font_name=fmt.font_name, font_size=fmt.font_size_body)

        # Group by category
        by_category = {}
        for tech in techniques:
            cat = tech['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(tech)

        overview_rows = []
        for category, techs in sorted(by_category.items()):
            for tech in techs:
                overview_rows.append([tech['name'], category, tech['found_in']])

        add_table(doc, ["Technique", "Category", "Found In"], overview_rows,
                  header_bg=fmt.table_header_bg, font_name=fmt.font_name)
    else:
        add_paragraph(doc, "No specific ML/NLP techniques were detected in the codebase. "
                      "This may indicate the project uses custom implementations or the source directories "
                      "are not correctly configured.",
                      font_name=fmt.font_name, font_size=fmt.font_size_body)

    # 3. Detailed Benchmark Standards (by category)
    add_heading(doc, "3. Detailed Benchmark Standards", level=1, font_name=fmt.font_name)

    section_num = 1
    for category, techs in sorted(by_category.items()) if techniques else []:
        add_heading(doc, f"3.{section_num} {category}", level=2, font_name=fmt.font_name)

        for tech in techs:
            add_heading(doc, tech['name'], level=3, font_name=fmt.font_name)

            # Description
            add_paragraph(doc, tech['description'], font_name=fmt.font_name,
                          font_size=fmt.font_size_body)

            # Gold Standard
            add_paragraph(doc, "Gold Standard Implementation:", font_name=fmt.font_name,
                          font_size=fmt.font_size_body, bold=True)
            add_paragraph(doc, tech['gold_standard'], font_name=fmt.font_name,
                          font_size=fmt.font_size_body)

            # Expected Output
            add_paragraph(doc, "Expected Output:", font_name=fmt.font_name,
                          font_size=fmt.font_size_body, bold=True)
            add_paragraph(doc, tech['expected_output'], font_name=fmt.font_name,
                          font_size=fmt.font_size_body)

            # Quality Metrics
            add_paragraph(doc, "Quality Metrics:", font_name=fmt.font_name,
                          font_size=fmt.font_size_body, bold=True)
            for metric in tech['quality_metrics']:
                add_bullet_point(doc, metric, font_name=fmt.font_name)

            # Reference Implementations
            add_paragraph(doc, "Reference Implementations:", font_name=fmt.font_name,
                          font_size=fmt.font_size_body, bold=True)
            for impl in tech['implementations']:
                add_bullet_point(doc, impl, font_name=fmt.font_name)

            # Academic References
            if tech.get('references'):
                add_paragraph(doc, "Authoritative References:", font_name=fmt.font_name,
                              font_size=fmt.font_size_body, bold=True)

                ref_rows = []
                for ref in tech['references']:
                    ref_rows.append([
                        ref['authors'],
                        ref['title'],
                        f"{ref['venue']} ({ref['year']})"
                    ])

                add_table(doc, ["Authors", "Title", "Publication"], ref_rows,
                          header_bg=fmt.table_header_bg, font_name=fmt.font_name)

        section_num += 1

    # 4. Validation Guidelines
    add_heading(doc, "4. Validation Guidelines", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "To ensure results meet benchmark standards, follow these validation steps:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    validation_steps = [
        "Compare outputs against expected formats documented above",
        "Verify quality metrics fall within acceptable ranges",
        "Cross-validate with multiple random seeds for reproducibility",
        "Test on known benchmark datasets when available",
        "Document any deviations from gold-standard implementations",
    ]
    for step in validation_steps:
        add_bullet_point(doc, step, font_name=fmt.font_name)

    # 5. Benchmark Datasets
    add_heading(doc, "5. Recommended Benchmark Datasets", level=1, font_name=fmt.font_name)

    benchmark_datasets = [
        ["20 Newsgroups", "Text Classification", "~20,000 newsgroup posts", "scikit-learn"],
        ["IMDB Reviews", "Sentiment Analysis", "50,000 movie reviews", "TensorFlow Datasets"],
        ["Reuters-21578", "Multi-label Classification", "~10,000 news articles", "NLTK"],
        ["Wikipedia", "Topic Modeling", "Large-scale corpus", "Gensim"],
        ["UCI ML Repository", "Various", "Multiple datasets", "uci.edu"],
    ]

    add_table(doc, ["Dataset", "Task", "Size", "Source"], benchmark_datasets,
              header_bg=fmt.table_header_bg, font_name=fmt.font_name)

    # 6. References Summary
    add_heading(doc, "6. References Summary", level=1, font_name=fmt.font_name)

    add_paragraph(doc, "Complete bibliography of authoritative sources cited in this document:",
                  font_name=fmt.font_name, font_size=fmt.font_size_body)

    # Collect all unique references
    all_refs = []
    seen_titles = set()
    for tech in techniques:
        for ref in tech.get('references', []):
            if ref['title'] not in seen_titles:
                all_refs.append(ref)
                seen_titles.add(ref['title'])

    # Sort by year, then author
    all_refs.sort(key=lambda x: (x['year'], x['authors']))

    for ref in all_refs:
        citation = f"{ref['authors']} ({ref['year']}). {ref['title']}. {ref['venue']}."
        add_bullet_point(doc, citation, font_name=fmt.font_name)

    # Save document
    output_path = Path(config.project_root) / config.docs_dir / "Benchmark_Standards.docx"
    save_document(doc, str(output_path))

    return str(output_path)
