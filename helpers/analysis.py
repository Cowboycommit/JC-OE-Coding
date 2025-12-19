"""
Analysis helper functions for the Streamlit UI.

Provides functions for running ML analysis, generating insights,
and processing results.
"""

from collections import Counter

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
import time
import logging

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
    """
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    responses = df[text_column].tolist()

    # Preprocess text
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
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

    feature_matrix = vectorizer.fit_transform(processed)

    # Limit max_codes based on data size
    max_codes = min(max_codes, len(df) - 1, feature_matrix.shape[1] - 1)
    if max_codes < min_codes:
        max_codes = min_codes

    results = {
        'silhouette_scores': {},
        'calinski_scores': {},
        'tested_range': list(range(min_codes, max_codes + 1))
    }

    best_score = -1
    optimal_n = min_codes

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

                if sil_score > best_score:
                    best_score = sil_score
                    optimal_n = n
        except Exception as e:
            logger.warning(f"Could not evaluate {n} codes: {e}")
            continue

    results['optimal_n_codes'] = optimal_n
    results['best_silhouette_score'] = best_score

    logger.info(f"Optimal number of codes: {optimal_n} (silhouette score: {best_score:.4f})")

    return optimal_n, results


def run_ml_analysis(
    df: pd.DataFrame,
    text_column: str,
    n_codes: int = 10,
    method: str = 'tfidf_kmeans',
    min_confidence: float = 0.3,
    progress_callback=None
) -> Tuple[Any, Any, Dict]:
    """
    Run ML-based open coding analysis.

    Args:
        df: DataFrame with responses
        text_column: Name of the text column
        n_codes: Number of codes to discover
        method: ML method ('tfidf_kmeans', 'lda', 'nmf')
        min_confidence: Minimum confidence threshold
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (coder, results, metrics)
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

    # Simple MLOpenCoder class
    class MLOpenCoder:
        def __init__(self, n_codes=10, method='tfidf_kmeans', min_confidence=0.3):
            self.n_codes = n_codes
            self.method = method
            self.min_confidence = min_confidence
            self.vectorizer = None
            self.model = None
            self.codebook = {}
            self.code_assignments = None
            self.confidence_scores = None
            self.feature_matrix = None

        def preprocess_text(self, text):
            if pd.isna(text):
                return ""
            text = str(text).lower()
            text = re.sub(r'[^a-z\\s]', ' ', text)
            text = ' '.join(text.split())
            return text

        def fit(self, responses, stop_words='english'):
            processed = [self.preprocess_text(r) for r in responses]

            if self.method == 'lda':
                self.vectorizer = CountVectorizer(
                    max_features=1000, stop_words=stop_words, min_df=2, max_df=0.8
                )
                self.feature_matrix = self.vectorizer.fit_transform(processed)
                self.model = LatentDirichletAllocation(
                    n_components=self.n_codes, random_state=42, max_iter=20
                )
            elif self.method == 'nmf':
                self.vectorizer = TfidfVectorizer(
                    max_features=1000, stop_words=stop_words, min_df=2, max_df=0.8
                )
                self.feature_matrix = self.vectorizer.fit_transform(processed)
                self.model = NMF(n_components=self.n_codes, random_state=42, max_iter=200)
            else:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000, stop_words=stop_words, min_df=2,
                    max_df=0.8, ngram_range=(1, 2)
                )
                self.feature_matrix = self.vectorizer.fit_transform(processed)
                self.model = KMeans(n_clusters=self.n_codes, random_state=42, n_init=10)

            if self.method in ['lda', 'nmf']:
                doc_topic_matrix = self.model.fit_transform(self.feature_matrix)
            else:
                labels = self.model.fit_predict(self.feature_matrix)
                doc_topic_matrix = np.zeros((len(responses), self.n_codes))
                for i, label in enumerate(labels):
                    doc_topic_matrix[i, label] = 1.0

            self._generate_codebook()
            self._assign_codes(doc_topic_matrix, responses)
            return self

        def _generate_codebook(self, top_words=10):
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
                label = ' '.join(top_words_list[:3]).title()

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

            return metrics

    # Run analysis with progress updates
    start_time = time.time()

    if progress_callback:
        progress_callback(0.1, "Initializing ML coder...")

    coder = MLOpenCoder(n_codes=n_codes, method=method, min_confidence=min_confidence)

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


def get_top_codes(coder, n: int = 10) -> pd.DataFrame:
    """
    Get top N codes by frequency.

    Args:
        coder: Fitted MLOpenCoder instance
        n: Number of codes to return

    Returns:
        DataFrame with top codes
    """
    code_data = []

    for code_id, info in coder.codebook.items():
        code_data.append({
            'Code': code_id,
            'Label': info['label'],
            'Count': info['count'],
            'Avg Confidence': info['avg_confidence'],
            'Keywords': ', '.join(info['keywords'][:5])
        })

    df = pd.DataFrame(code_data)
    df = df.sort_values('Count', ascending=False).head(n)

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
