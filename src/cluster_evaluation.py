"""
Post-hoc cluster evaluation for optional label-based validation.

This module provides evaluation metrics for clustering results when ground truth
labels are available. It enforces a STRICT separation between:

1. UNSUPERVISED TRAINING: Clustering uses ONLY text features, NEVER labels
2. POST-HOC EVALUATION: Labels used ONLY for diagnostic metrics after clustering

CRITICAL PRINCIPLE:
Labels are NEVER used during clustering. They are only used afterward to
validate/diagnose the quality of discovered clusters.

Metrics provided:
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Cluster purity
- Homogeneity, completeness, V-measure
- Multi-label overlap metrics

Usage:
    # AFTER clustering is complete
    evaluator = ClusterEvaluator()
    metrics = evaluator.evaluate(
        cluster_assignments=predicted_clusters,
        true_labels=ground_truth_labels,
        mode='standard'  # or 'multi_label'
    )
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Container for cluster evaluation metrics.

    All metrics are computed POST-HOC after clustering is complete.
    They measure agreement between discovered clusters and ground truth labels.

    Attributes:
        ari: Adjusted Rand Index (-1 to 1, 1 = perfect agreement)
        nmi: Normalized Mutual Information (0 to 1, 1 = perfect agreement)
        purity: Cluster purity (0 to 1, 1 = each cluster has single label)
        homogeneity: Homogeneity score (0 to 1)
        completeness: Completeness score (0 to 1)
        v_measure: V-measure (harmonic mean of homogeneity and completeness)

        # Additional diagnostics
        n_clusters: Number of clusters
        n_labels: Number of unique true labels
        label_to_cluster_mapping: Most common cluster for each label
        cluster_to_label_mapping: Most common label for each cluster

        # Metadata
        evaluation_mode: 'standard' or 'multi_label'
        warnings: Any warnings generated during evaluation
        notes: Interpretive notes about the metrics
    """
    ari: float = 0.0
    nmi: float = 0.0
    purity: float = 0.0
    homogeneity: float = 0.0
    completeness: float = 0.0
    v_measure: float = 0.0

    n_clusters: int = 0
    n_labels: int = 0
    label_to_cluster_mapping: Dict[str, int] = field(default_factory=dict)
    cluster_to_label_mapping: Dict[int, str] = field(default_factory=dict)

    evaluation_mode: str = 'standard'
    warnings: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ari': round(self.ari, 4),
            'nmi': round(self.nmi, 4),
            'purity': round(self.purity, 4),
            'homogeneity': round(self.homogeneity, 4),
            'completeness': round(self.completeness, 4),
            'v_measure': round(self.v_measure, 4),
            'n_clusters': self.n_clusters,
            'n_labels': self.n_labels,
            'evaluation_mode': self.evaluation_mode,
            'warnings': self.warnings,
            'notes': self.notes
        }

    def get_summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            "CLUSTER EVALUATION METRICS (POST-HOC)",
            "=" * 50,
            "",
            "NOTE: These metrics compare discovered clusters to ground",
            "truth labels AFTER clustering. Labels were NOT used for training.",
            "",
            f"Evaluation Mode: {self.evaluation_mode}",
            f"Clusters: {self.n_clusters}, True Labels: {self.n_labels}",
            "",
            "METRICS:",
            f"  Adjusted Rand Index (ARI): {self.ari:.4f}",
            f"  Normalized Mutual Information (NMI): {self.nmi:.4f}",
            f"  Cluster Purity: {self.purity:.4f}",
            f"  Homogeneity: {self.homogeneity:.4f}",
            f"  Completeness: {self.completeness:.4f}",
            f"  V-Measure: {self.v_measure:.4f}",
            "",
        ]

        # Interpretation
        if self.ari > 0.7:
            lines.append("INTERPRETATION: Strong agreement between clusters and labels")
        elif self.ari > 0.4:
            lines.append("INTERPRETATION: Moderate agreement between clusters and labels")
        elif self.ari > 0.1:
            lines.append("INTERPRETATION: Weak agreement between clusters and labels")
        else:
            lines.append("INTERPRETATION: Little to no agreement between clusters and labels")

        if self.warnings:
            lines.append("")
            lines.append("WARNINGS:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)


class ClusterEvaluator:
    """
    Evaluates clustering results against ground truth labels (POST-HOC ONLY).

    This class provides metrics to assess how well discovered clusters
    align with known categories. It is designed for DIAGNOSTIC purposes
    after clustering is complete.

    CRITICAL: This evaluator does NOT influence clustering. It only measures
    agreement between clusters and labels after the fact.

    Supported evaluation modes:
    - 'standard': Assumes single label per document
    - 'multi_label': Handles documents with multiple labels
    - 'single_label_only': Filters to single-label documents before evaluation

    Example:
        >>> # AFTER clustering
        >>> evaluator = ClusterEvaluator()
        >>> metrics = evaluator.evaluate(
        ...     cluster_assignments=kmeans.labels_,
        ...     true_labels=ground_truth
        ... )
        >>> print(metrics.get_summary())
    """

    VALID_MODES = ['standard', 'multi_label', 'single_label_only']

    def __init__(
        self,
        multi_label_delimiter: Optional[str] = None,
        unknown_label_patterns: Optional[List[str]] = None
    ):
        """
        Initialize the evaluator.

        Args:
            multi_label_delimiter: Delimiter for multi-label strings (e.g., ',')
            unknown_label_patterns: Patterns to treat as missing labels
        """
        self.multi_label_delimiter = multi_label_delimiter
        self.unknown_label_patterns = unknown_label_patterns or [
            'unknown', 'unlabeled', 'n/a', 'none', ''
        ]

    def evaluate(
        self,
        cluster_assignments: Union[List[int], np.ndarray],
        true_labels: Union[List[str], pd.Series],
        mode: str = 'standard'
    ) -> EvaluationMetrics:
        """
        Evaluate clustering against ground truth labels.

        Args:
            cluster_assignments: Cluster ID for each document
            true_labels: Ground truth label(s) for each document
            mode: Evaluation mode:
                - 'standard': Single label per document
                - 'multi_label': Multiple labels per document
                - 'single_label_only': Filter to single-label docs first

        Returns:
            EvaluationMetrics with all computed metrics

        Raises:
            ValueError: If mode is invalid or inputs are mismatched

        Notes:
            - This is a POST-HOC evaluation only
            - Labels were NOT used during clustering
            - Metrics measure agreement, not clustering quality per se
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Use one of {self.VALID_MODES}")

        # Convert inputs
        if isinstance(cluster_assignments, np.ndarray):
            cluster_assignments = cluster_assignments.tolist()
        if isinstance(true_labels, pd.Series):
            true_labels = true_labels.tolist()

        # Validate lengths
        if len(cluster_assignments) != len(true_labels):
            raise ValueError(
                f"Length mismatch: {len(cluster_assignments)} clusters, "
                f"{len(true_labels)} labels"
            )

        logger.info(
            f"Evaluating clustering (mode={mode}): "
            f"{len(cluster_assignments)} documents"
        )

        # Filter out unknown labels
        valid_indices = []
        for i, label in enumerate(true_labels):
            if not self._is_unknown_label(label):
                valid_indices.append(i)

        if len(valid_indices) < len(cluster_assignments):
            logger.info(
                f"Filtered {len(cluster_assignments) - len(valid_indices)} "
                "documents with unknown labels"
            )

        if not valid_indices:
            warnings.warn("No valid labels found for evaluation")
            return EvaluationMetrics(
                warnings=["No valid labels found"],
                notes="Evaluation skipped - no valid ground truth labels"
            )

        # Apply filtering
        filtered_clusters = [cluster_assignments[i] for i in valid_indices]
        filtered_labels = [true_labels[i] for i in valid_indices]

        # Handle mode-specific processing
        if mode == 'multi_label':
            return self._evaluate_multi_label(filtered_clusters, filtered_labels)
        elif mode == 'single_label_only':
            # Further filter to single-label documents
            single_indices = []
            for i, label in enumerate(filtered_labels):
                if not self._is_multi_label(label):
                    single_indices.append(i)

            if not single_indices:
                return EvaluationMetrics(
                    warnings=["No single-label documents found"],
                    notes="Evaluation skipped - all documents are multi-label"
                )

            filtered_clusters = [filtered_clusters[i] for i in single_indices]
            filtered_labels = [filtered_labels[i] for i in single_indices]

            logger.info(f"Filtered to {len(filtered_clusters)} single-label documents")

        return self._evaluate_standard(filtered_clusters, filtered_labels)

    def _evaluate_standard(
        self,
        cluster_assignments: List[int],
        true_labels: List[str]
    ) -> EvaluationMetrics:
        """Evaluate with standard single-label metrics."""
        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score,
            homogeneity_completeness_v_measure
        )

        # Convert labels to numeric for sklearn
        unique_labels = sorted(set(true_labels))
        label_to_int = {l: i for i, l in enumerate(unique_labels)}
        numeric_labels = [label_to_int[l] for l in true_labels]

        # Compute metrics
        ari = adjusted_rand_score(numeric_labels, cluster_assignments)
        nmi = normalized_mutual_info_score(numeric_labels, cluster_assignments)
        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
            numeric_labels, cluster_assignments
        )

        # Compute purity
        purity = self._compute_purity(cluster_assignments, true_labels)

        # Compute mappings
        label_to_cluster = self._compute_label_to_cluster_mapping(
            cluster_assignments, true_labels
        )
        cluster_to_label = self._compute_cluster_to_label_mapping(
            cluster_assignments, true_labels
        )

        # Generate notes
        n_clusters = len(set(cluster_assignments))
        n_labels = len(unique_labels)

        notes = []
        if n_clusters > n_labels * 2:
            notes.append(f"Many more clusters ({n_clusters}) than labels ({n_labels})")
        elif n_clusters < n_labels // 2:
            notes.append(f"Fewer clusters ({n_clusters}) than labels ({n_labels})")

        return EvaluationMetrics(
            ari=float(ari),
            nmi=float(nmi),
            purity=purity,
            homogeneity=float(homogeneity),
            completeness=float(completeness),
            v_measure=float(v_measure),
            n_clusters=n_clusters,
            n_labels=n_labels,
            label_to_cluster_mapping=label_to_cluster,
            cluster_to_label_mapping=cluster_to_label,
            evaluation_mode='standard',
            notes="; ".join(notes) if notes else "Standard evaluation completed"
        )

    def _evaluate_multi_label(
        self,
        cluster_assignments: List[int],
        true_labels: List[str]
    ) -> EvaluationMetrics:
        """
        Evaluate with multi-label aware metrics.

        For multi-label evaluation, we use overlap-based metrics
        since standard metrics assume single labels.
        """
        # Parse multi-labels
        parsed_labels = [self._parse_multi_label(l) for l in true_labels]

        # Get all unique labels
        all_labels = set()
        for labels in parsed_labels:
            all_labels.update(labels)
        all_labels = sorted(all_labels)

        n_clusters = len(set(cluster_assignments))
        n_labels = len(all_labels)

        # For multi-label, we compute purity using the most common label per cluster
        # and overlap-based metrics

        # Cluster purity: for each cluster, what fraction have the most common label
        cluster_docs = {}
        for i, cluster in enumerate(cluster_assignments):
            if cluster not in cluster_docs:
                cluster_docs[cluster] = []
            cluster_docs[cluster].append(parsed_labels[i])

        cluster_purities = []
        cluster_to_label = {}

        for cluster, doc_labels_list in cluster_docs.items():
            # Flatten all labels in this cluster
            all_cluster_labels = []
            for labels in doc_labels_list:
                all_cluster_labels.extend(labels)

            if all_cluster_labels:
                label_counts = Counter(all_cluster_labels)
                most_common_label, most_common_count = label_counts.most_common(1)[0]

                # Purity: fraction of docs that have the most common label
                n_with_common = sum(
                    1 for labels in doc_labels_list if most_common_label in labels
                )
                purity = n_with_common / len(doc_labels_list)
                cluster_purities.append(purity)
                cluster_to_label[cluster] = most_common_label
            else:
                cluster_purities.append(0.0)
                cluster_to_label[cluster] = "unknown"

        overall_purity = np.mean(cluster_purities) if cluster_purities else 0.0

        # For multi-label, we compute a modified NMI using label overlap
        # This is an approximation - we use the primary label
        primary_labels = [labels[0] if labels else "unknown" for labels in parsed_labels]
        unique_primary = sorted(set(primary_labels))
        label_to_int = {l: i for i, l in enumerate(unique_primary)}
        numeric_labels = [label_to_int[l] for l in primary_labels]

        from sklearn.metrics import (
            adjusted_rand_score,
            normalized_mutual_info_score
        )

        ari = adjusted_rand_score(numeric_labels, cluster_assignments)
        nmi = normalized_mutual_info_score(numeric_labels, cluster_assignments)

        # Label to cluster mapping
        label_to_cluster = {}
        for label in all_labels:
            # Find most common cluster for this label
            clusters_with_label = []
            for i, labels in enumerate(parsed_labels):
                if label in labels:
                    clusters_with_label.append(cluster_assignments[i])
            if clusters_with_label:
                most_common = Counter(clusters_with_label).most_common(1)[0][0]
                label_to_cluster[label] = most_common

        return EvaluationMetrics(
            ari=float(ari),
            nmi=float(nmi),
            purity=overall_purity,
            homogeneity=0.0,  # Not directly applicable to multi-label
            completeness=0.0,
            v_measure=0.0,
            n_clusters=n_clusters,
            n_labels=n_labels,
            label_to_cluster_mapping=label_to_cluster,
            cluster_to_label_mapping=cluster_to_label,
            evaluation_mode='multi_label',
            warnings=["Using primary label for ARI/NMI (multi-label approximation)"],
            notes="Multi-label evaluation uses overlap-based purity and primary-label metrics"
        )

    def _compute_purity(
        self,
        cluster_assignments: List[int],
        true_labels: List[str]
    ) -> float:
        """Compute cluster purity."""
        cluster_label_counts = {}

        for cluster, label in zip(cluster_assignments, true_labels):
            if cluster not in cluster_label_counts:
                cluster_label_counts[cluster] = Counter()
            cluster_label_counts[cluster][label] += 1

        total_correct = 0
        for cluster, counts in cluster_label_counts.items():
            most_common_count = counts.most_common(1)[0][1]
            total_correct += most_common_count

        return total_correct / len(cluster_assignments)

    def _compute_label_to_cluster_mapping(
        self,
        cluster_assignments: List[int],
        true_labels: List[str]
    ) -> Dict[str, int]:
        """Find most common cluster for each label."""
        label_cluster_counts = {}

        for cluster, label in zip(cluster_assignments, true_labels):
            if label not in label_cluster_counts:
                label_cluster_counts[label] = Counter()
            label_cluster_counts[label][cluster] += 1

        mapping = {}
        for label, counts in label_cluster_counts.items():
            mapping[label] = counts.most_common(1)[0][0]

        return mapping

    def _compute_cluster_to_label_mapping(
        self,
        cluster_assignments: List[int],
        true_labels: List[str]
    ) -> Dict[int, str]:
        """Find most common label for each cluster."""
        cluster_label_counts = {}

        for cluster, label in zip(cluster_assignments, true_labels):
            if cluster not in cluster_label_counts:
                cluster_label_counts[cluster] = Counter()
            cluster_label_counts[cluster][label] += 1

        mapping = {}
        for cluster, counts in cluster_label_counts.items():
            mapping[cluster] = counts.most_common(1)[0][0]

        return mapping

    def _is_unknown_label(self, label) -> bool:
        """Check if label is unknown/missing."""
        if pd.isna(label) or label is None:
            return True
        label_str = str(label).strip().lower()
        return label_str in self.unknown_label_patterns

    def _is_multi_label(self, label) -> bool:
        """Check if label contains multiple labels."""
        if pd.isna(label) or label is None:
            return False
        if self.multi_label_delimiter:
            return self.multi_label_delimiter in str(label)
        # Check common delimiters
        label_str = str(label)
        return any(d in label_str for d in [',', ';', '|'])

    def _parse_multi_label(self, label) -> List[str]:
        """Parse a multi-label string into individual labels."""
        if pd.isna(label) or label is None:
            return []

        label_str = str(label).strip()
        if not label_str:
            return []

        # Try delimiter
        delimiter = self.multi_label_delimiter
        if not delimiter:
            # Auto-detect
            for d in [',', ';', '|']:
                if d in label_str:
                    delimiter = d
                    break

        if delimiter:
            labels = [l.strip() for l in label_str.split(delimiter)]
            return [l for l in labels if l and l.lower() not in self.unknown_label_patterns]
        else:
            return [label_str]


def evaluate_clusters_posthoc(
    cluster_assignments: Union[List[int], np.ndarray],
    true_labels: Optional[Union[List[str], pd.Series]] = None,
    multi_label_delimiter: Optional[str] = None
) -> Optional[EvaluationMetrics]:
    """
    Convenience function for post-hoc cluster evaluation.

    This function evaluates clustering results against ground truth labels
    AFTER clustering is complete. It is purely diagnostic.

    Args:
        cluster_assignments: Cluster ID for each document
        true_labels: Optional ground truth labels (if None, returns None)
        multi_label_delimiter: Delimiter for multi-label strings

    Returns:
        EvaluationMetrics if labels provided, None otherwise

    Example:
        >>> metrics = evaluate_clusters_posthoc(
        ...     cluster_assignments=kmeans.labels_,
        ...     true_labels=df['category']
        ... )
        >>> if metrics:
        ...     print(f"ARI: {metrics.ari:.4f}")
    """
    if true_labels is None:
        logger.info("No labels provided - skipping post-hoc evaluation")
        return None

    evaluator = ClusterEvaluator(multi_label_delimiter=multi_label_delimiter)

    # Auto-detect multi-label
    if isinstance(true_labels, pd.Series):
        labels_list = true_labels.tolist()
    else:
        labels_list = list(true_labels)

    has_multi_label = any(
        d in str(l) for l in labels_list
        for d in [',', ';', '|']
        if not pd.isna(l)
    )

    mode = 'multi_label' if has_multi_label else 'standard'
    logger.info(f"Auto-detected evaluation mode: {mode}")

    return evaluator.evaluate(cluster_assignments, labels_list, mode=mode)
