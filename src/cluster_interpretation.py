"""
Cluster interpretation layer for TF-IDF + KMeans clustering.

This module provides MANDATORY interpretation of cluster results, ensuring that
cluster IDs are never exposed without human-readable explanations. This prevents
users from mistaking numeric cluster IDs for semantic labels.

Key principles:
- Every cluster MUST have a human-readable summary
- Cluster IDs are internal identifiers, NOT semantic labels
- Interpretation is based on top terms, not external labels
- Works for ALL datasets, not just specific ones

Classes:
    ClusterInterpreter: Generates human-readable cluster summaries
    ClusterSummary: Data class holding cluster interpretation

Usage:
    >>> interpreter = ClusterInterpreter()
    >>> summaries = interpreter.interpret_clusters(
    ...     vectorizer=fitted_vectorizer,
    ...     cluster_model=fitted_kmeans,
    ...     texts=original_texts,
    ...     cluster_assignments=labels
    ... )
    >>> for cluster_id, summary in summaries.items():
    ...     print(f"{cluster_id}: {summary.label}")
    ...     print(f"  Top terms: {summary.top_terms}")
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ClusterSummary:
    """
    Human-readable summary of a single cluster.

    IMPORTANT: cluster_id is an internal identifier, NOT a semantic label.
    The 'label' field provides a human-readable interpretation.

    Attributes:
        cluster_id: Internal cluster identifier (e.g., 0, 1, 2 or "CLUSTER_01")
        label: Human-readable label generated from top terms
        top_terms: List of most representative terms for this cluster
        term_weights: Weights/scores for each top term
        document_count: Number of documents in this cluster
        document_indices: Indices of documents assigned to this cluster
        representative_docs: Example documents from this cluster
        coherence_score: Optional semantic coherence score
        interpretation_notes: Additional notes about interpretation

        # Warning fields
        is_interpretable: Whether cluster has clear interpretation
        interpretation_confidence: Confidence in the interpretation (0-1)
        warnings: Any warnings about interpretation quality
    """
    cluster_id: Any
    label: str
    top_terms: List[str]
    term_weights: List[float] = field(default_factory=list)
    document_count: int = 0
    document_indices: List[int] = field(default_factory=list)
    representative_docs: List[Dict[str, Any]] = field(default_factory=list)
    coherence_score: Optional[float] = None
    interpretation_notes: str = ""

    # Interpretation quality indicators
    is_interpretable: bool = True
    interpretation_confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'label': self.label,
            'top_terms': self.top_terms,
            'term_weights': [round(w, 4) for w in self.term_weights],
            'document_count': self.document_count,
            'representative_docs': self.representative_docs,
            'coherence_score': round(self.coherence_score, 4) if self.coherence_score else None,
            'interpretation_notes': self.interpretation_notes,
            'is_interpretable': self.is_interpretable,
            'interpretation_confidence': round(self.interpretation_confidence, 3),
            'warnings': self.warnings
        }

    def get_display_string(self, include_terms: bool = True) -> str:
        """Get a formatted display string for this cluster."""
        parts = [f"{self.cluster_id}: {self.label} ({self.document_count} docs)"]
        if include_terms and self.top_terms:
            terms_str = ", ".join(self.top_terms[:5])
            parts.append(f"  Terms: {terms_str}")
        if self.warnings:
            parts.append(f"  Warnings: {'; '.join(self.warnings)}")
        return "\n".join(parts)


@dataclass
class ClusterInterpretationReport:
    """
    Complete interpretation report for all clusters.

    This report provides a comprehensive view of cluster interpretations
    and should be presented to users instead of raw cluster IDs.

    Attributes:
        summaries: Dictionary mapping cluster_id to ClusterSummary
        n_clusters: Total number of clusters
        n_documents: Total number of documents
        overall_interpretability: Average interpretability across clusters
        method_used: Description of clustering method
        warnings: Global warnings about interpretation
    """
    summaries: Dict[Any, ClusterSummary]
    n_clusters: int
    n_documents: int
    overall_interpretability: float
    method_used: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'n_clusters': self.n_clusters,
            'n_documents': self.n_documents,
            'overall_interpretability': round(self.overall_interpretability, 3),
            'method_used': self.method_used,
            'warnings': self.warnings,
            'cluster_summaries': {
                k: v.to_dict() for k, v in self.summaries.items()
            }
        }

    def get_display_report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 60,
            "CLUSTER INTERPRETATION REPORT",
            "=" * 60,
            "",
            f"Method: {self.method_used}",
            f"Total Documents: {self.n_documents}",
            f"Number of Clusters: {self.n_clusters}",
            f"Overall Interpretability: {self.overall_interpretability:.1%}",
            ""
        ]

        if self.warnings:
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        lines.append("-" * 60)
        lines.append("CLUSTER DETAILS")
        lines.append("-" * 60)
        lines.append("")

        # Sort by document count
        sorted_summaries = sorted(
            self.summaries.values(),
            key=lambda s: s.document_count,
            reverse=True
        )

        for summary in sorted_summaries:
            lines.append(summary.get_display_string(include_terms=True))
            lines.append("")

        lines.append("=" * 60)
        lines.append("NOTE: Cluster IDs are internal identifiers, not semantic labels.")
        lines.append("Use the 'label' and 'top_terms' for interpretation.")
        lines.append("=" * 60)

        return "\n".join(lines)


class ClusterInterpreter:
    """
    Generates human-readable interpretations for clusters.

    This class is a MANDATORY component of the clustering pipeline.
    It ensures that cluster results are always presented with interpretable
    summaries, preventing users from mistaking cluster IDs for semantic labels.

    Key features:
    - Extracts top-N terms per cluster
    - Generates human-readable labels from terms
    - Provides confidence scores for interpretations
    - Includes representative documents per cluster
    - Warns when clusters have weak interpretations

    Example:
        >>> interpreter = ClusterInterpreter(n_top_terms=10)
        >>> report = interpreter.interpret_clusters(
        ...     vectorizer=fitted_tfidf,
        ...     cluster_model=fitted_kmeans,
        ...     texts=documents,
        ...     cluster_assignments=cluster_labels
        ... )
        >>> print(report.get_display_report())
    """

    def __init__(
        self,
        n_top_terms: int = 15,
        n_label_terms: int = 5,
        n_representative_docs: int = 5,
        min_term_weight_threshold: float = 0.005,
        low_interpretability_threshold: float = 0.3
    ):
        """
        Initialize the cluster interpreter.

        Args:
            n_top_terms: Number of top terms to extract per cluster
            n_label_terms: Number of terms to use in generated labels
            n_representative_docs: Number of example documents to include
            min_term_weight_threshold: Minimum weight for terms to be considered
            low_interpretability_threshold: Threshold below which to warn about interpretability
        """
        self.n_top_terms = n_top_terms
        self.n_label_terms = n_label_terms
        self.n_representative_docs = n_representative_docs
        self.min_term_weight_threshold = min_term_weight_threshold
        self.low_interpretability_threshold = low_interpretability_threshold

    def interpret_clusters(
        self,
        vectorizer,
        cluster_model,
        texts: List[str],
        cluster_assignments: List[int],
        feature_matrix=None,
        method_name: str = 'tfidf_kmeans'
    ) -> ClusterInterpretationReport:
        """
        Generate interpretation report for clustering results.

        This is the main entry point for cluster interpretation.
        It MUST be called after clustering to provide interpretable results.

        Args:
            vectorizer: Fitted TF-IDF or similar vectorizer with get_feature_names_out()
            cluster_model: Fitted clustering model (KMeans, LDA, NMF, etc.)
            texts: Original text documents
            cluster_assignments: Cluster assignment for each document
            feature_matrix: Optional precomputed feature matrix
            method_name: Name of clustering method for reporting

        Returns:
            ClusterInterpretationReport with all cluster summaries

        Raises:
            ValueError: If inputs are inconsistent

        Notes:
            - This method extracts top terms from cluster centroids or components
            - For KMeans: uses cluster_centers_
            - For LDA/NMF: uses components_
            - Labels are generated from top terms, NOT from external data
        """
        # Validate inputs
        if len(texts) != len(cluster_assignments):
            raise ValueError(
                f"Mismatch: {len(texts)} texts but {len(cluster_assignments)} assignments"
            )

        # Get feature names
        try:
            feature_names = vectorizer.get_feature_names_out()
        except AttributeError:
            # Fallback for embedders that use generic feature names
            feature_names = [f"feature_{i}" for i in range(
                cluster_model.cluster_centers_.shape[1] if hasattr(cluster_model, 'cluster_centers_')
                else cluster_model.components_.shape[1]
            )]

        # Determine cluster centers/components based on model type
        if hasattr(cluster_model, 'cluster_centers_'):
            # KMeans-style model
            centers = cluster_model.cluster_centers_
            unique_clusters = list(range(len(centers)))
        elif hasattr(cluster_model, 'components_'):
            # LDA/NMF-style model
            centers = cluster_model.components_
            unique_clusters = list(range(len(centers)))
        else:
            raise ValueError(
                f"Unknown model type: {type(cluster_model)}. "
                "Expected KMeans (cluster_centers_) or LDA/NMF (components_)."
            )

        # Group documents by cluster
        cluster_docs = {}
        for idx, cluster in enumerate(cluster_assignments):
            if cluster not in cluster_docs:
                cluster_docs[cluster] = []
            cluster_docs[cluster].append(idx)

        # Generate summaries for each cluster
        summaries = {}
        global_warnings = []
        interpretability_scores = []

        for cluster_idx in unique_clusters:
            # Get cluster center
            center = centers[cluster_idx]

            # Extract top terms
            top_indices = center.argsort()[-self.n_top_terms:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
            term_weights = [float(center[i]) for i in top_indices]

            # Filter by weight threshold
            filtered_terms = []
            filtered_weights = []
            for term, weight in zip(top_terms, term_weights):
                if weight >= self.min_term_weight_threshold:
                    filtered_terms.append(term)
                    filtered_weights.append(weight)

            # Generate label from top terms
            label_terms = filtered_terms[:self.n_label_terms]
            if label_terms:
                label = " / ".join(term.title() for term in label_terms)
            else:
                label = f"Cluster {cluster_idx} (low confidence)"

            # Get document count and indices
            doc_indices = cluster_docs.get(cluster_idx, [])
            doc_count = len(doc_indices)

            # Get representative documents
            representative_docs = self._get_representative_docs(
                texts, doc_indices, feature_matrix, center
            )

            # Calculate interpretation confidence
            if filtered_terms:
                # Confidence based on term weight variance and count
                weight_sum = sum(filtered_weights)
                confidence = min(1.0, weight_sum / 2.0)  # Normalize
            else:
                confidence = 0.0

            # Check for interpretation issues
            warnings = []
            is_interpretable = True

            if not filtered_terms:
                warnings.append("No terms above weight threshold")
                is_interpretable = False
            elif len(filtered_terms) < 2:
                # Reduced penalty threshold for survey-sized datasets
                warnings.append("Few distinctive terms")
                confidence *= 0.9
            if doc_count == 0:
                warnings.append("Empty cluster")
                is_interpretable = False
            elif doc_count < 3:
                # Reduced penalty threshold for survey-sized datasets (200-500 responses)
                warnings.append("Very small cluster (<3 docs)")
                confidence *= 0.95

            interpretability_scores.append(confidence)

            # Create cluster ID string
            cluster_id_str = f"CLUSTER_{cluster_idx + 1:02d}"

            # Create summary
            summary = ClusterSummary(
                cluster_id=cluster_id_str,
                label=label,
                top_terms=filtered_terms,
                term_weights=filtered_weights,
                document_count=doc_count,
                document_indices=doc_indices,
                representative_docs=representative_docs,
                interpretation_confidence=confidence,
                is_interpretable=is_interpretable,
                warnings=warnings
            )

            summaries[cluster_id_str] = summary

        # Calculate overall interpretability
        overall_interpretability = np.mean(interpretability_scores) if interpretability_scores else 0.0

        # Add global warnings
        if overall_interpretability < self.low_interpretability_threshold:
            global_warnings.append(
                f"Low overall interpretability ({overall_interpretability:.1%}). "
                "Consider adjusting clustering parameters or preprocessing."
            )

        empty_clusters = sum(1 for s in summaries.values() if s.document_count == 0)
        if empty_clusters > 0:
            global_warnings.append(f"{empty_clusters} empty cluster(s) detected")

        # Create report
        report = ClusterInterpretationReport(
            summaries=summaries,
            n_clusters=len(unique_clusters),
            n_documents=len(texts),
            overall_interpretability=overall_interpretability,
            method_used=method_name,
            warnings=global_warnings
        )

        logger.info(
            f"Cluster interpretation complete: {len(summaries)} clusters, "
            f"interpretability={overall_interpretability:.1%}"
        )

        return report

    def _get_representative_docs(
        self,
        texts: List[str],
        doc_indices: List[int],
        feature_matrix,
        center
    ) -> List[Dict[str, Any]]:
        """
        Get representative documents for a cluster.

        Selects documents that are closest to the cluster center
        (most representative of the cluster).
        """
        if not doc_indices:
            return []

        n_docs = min(self.n_representative_docs, len(doc_indices))

        if feature_matrix is not None:
            # Calculate distances to center
            try:
                from sklearn.metrics.pairwise import cosine_similarity

                doc_vectors = feature_matrix[doc_indices]
                if hasattr(doc_vectors, 'toarray'):
                    doc_vectors = doc_vectors.toarray()

                center_reshaped = center.reshape(1, -1)
                similarities = cosine_similarity(doc_vectors, center_reshaped).flatten()

                # Get top N most similar
                top_indices = similarities.argsort()[-n_docs:][::-1]
                selected_indices = [doc_indices[i] for i in top_indices]
                selected_similarities = [float(similarities[i]) for i in top_indices]
            except Exception as e:
                logger.warning(f"Could not compute document similarities: {e}")
                # Fall back to random selection
                selected_indices = doc_indices[:n_docs]
                selected_similarities = [None] * n_docs
        else:
            # Random selection if no feature matrix
            selected_indices = doc_indices[:n_docs]
            selected_similarities = [None] * n_docs

        representative_docs = []
        for idx, sim in zip(selected_indices, selected_similarities):
            doc = {
                'index': idx,
                'text': str(texts[idx])[:500],  # Truncate for display
                'similarity': sim
            }
            representative_docs.append(doc)

        return representative_docs

    def get_cluster_comparison(
        self,
        report: ClusterInterpretationReport
    ) -> pd.DataFrame:
        """
        Create a comparison table of all clusters.

        Args:
            report: Cluster interpretation report

        Returns:
            DataFrame with cluster comparison
        """
        rows = []
        for cluster_id, summary in report.summaries.items():
            rows.append({
                'Cluster ID': cluster_id,
                'Label': summary.label,
                'Documents': summary.document_count,
                'Top Terms': ', '.join(summary.top_terms[:5]),
                'Confidence': f"{summary.interpretation_confidence:.1%}",
                'Interpretable': 'Yes' if summary.is_interpretable else 'No',
                'Warnings': '; '.join(summary.warnings) if summary.warnings else ''
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('Documents', ascending=False)
        return df


class ClusterCodebook:
    """
    Generates a formal codebook from cluster interpretations.

    This class creates a structured codebook suitable for qualitative research,
    with clear documentation that cluster codes are derived from unsupervised
    clustering, not predefined categories.

    Example:
        >>> codebook = ClusterCodebook(report)
        >>> print(codebook.to_markdown())
    """

    def __init__(self, interpretation_report: ClusterInterpretationReport):
        """
        Initialize codebook from interpretation report.

        Args:
            interpretation_report: Complete cluster interpretation report
        """
        self.report = interpretation_report

    def to_dict(self) -> Dict[str, Any]:
        """Convert codebook to dictionary format."""
        codes = {}
        for cluster_id, summary in self.report.summaries.items():
            codes[cluster_id] = {
                'name': summary.label,
                'definition': f"Documents characterized by: {', '.join(summary.top_terms[:5])}",
                'keywords': summary.top_terms,
                'n_documents': summary.document_count,
                'examples': [d['text'] for d in summary.representative_docs[:3]],
                'confidence': summary.interpretation_confidence,
                'notes': summary.interpretation_notes,
                'warnings': summary.warnings
            }
        return {
            'method': self.report.method_used,
            'n_clusters': self.report.n_clusters,
            'n_documents': self.report.n_documents,
            'codes': codes,
            'disclaimer': (
                "These codes were generated through unsupervised machine learning (clustering). "
                "Cluster IDs are arbitrary identifiers. Labels are derived from term frequencies "
                "and do not represent predefined categories. Human review is recommended."
            )
        }

    def to_markdown(self) -> str:
        """Generate markdown format codebook."""
        lines = [
            "# Cluster Codebook",
            "",
            f"**Method**: {self.report.method_used}",
            f"**Documents**: {self.report.n_documents}",
            f"**Codes Discovered**: {self.report.n_clusters}",
            "",
            "> **Note**: These codes were generated through unsupervised clustering. ",
            "> Cluster IDs are internal identifiers, not semantic categories. ",
            "> Labels are derived from term frequencies. Human validation recommended.",
            "",
            "---",
            ""
        ]

        # Sort by document count
        sorted_summaries = sorted(
            self.report.summaries.items(),
            key=lambda x: x[1].document_count,
            reverse=True
        )

        for cluster_id, summary in sorted_summaries:
            lines.append(f"## {cluster_id}: {summary.label}")
            lines.append("")
            lines.append(f"**Documents**: {summary.document_count}")
            lines.append(f"**Confidence**: {summary.interpretation_confidence:.1%}")
            lines.append("")
            lines.append("**Keywords**: " + ", ".join(summary.top_terms[:7]))
            lines.append("")

            if summary.representative_docs:
                lines.append("**Example Documents**:")
                for doc in summary.representative_docs[:3]:
                    text_preview = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                    lines.append(f"> {text_preview}")
                    lines.append("")

            if summary.warnings:
                lines.append("**Warnings**: " + "; ".join(summary.warnings))
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """Generate CSV format for the codebook."""
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'Cluster ID', 'Label', 'Documents', 'Keywords',
            'Confidence', 'Example 1', 'Example 2', 'Warnings'
        ])

        for cluster_id, summary in self.report.summaries.items():
            examples = [d['text'][:200] for d in summary.representative_docs[:2]]
            while len(examples) < 2:
                examples.append('')

            writer.writerow([
                cluster_id,
                summary.label,
                summary.document_count,
                '; '.join(summary.top_terms[:5]),
                f"{summary.interpretation_confidence:.1%}",
                examples[0],
                examples[1],
                '; '.join(summary.warnings)
            ])

        return output.getvalue()
