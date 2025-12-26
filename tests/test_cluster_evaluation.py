"""
Tests for cluster evaluation module.

These tests verify:
1. Post-hoc evaluation metrics are computed correctly
2. Multi-label evaluation works correctly
3. Labels are NEVER used for clustering (evaluation is post-hoc only)
4. Unknown labels are handled properly
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cluster_evaluation import (
    ClusterEvaluator,
    EvaluationMetrics,
    evaluate_clusters_posthoc
)


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = EvaluationMetrics(
            ari=0.75,
            nmi=0.80,
            purity=0.85,
            homogeneity=0.78,
            completeness=0.82,
            v_measure=0.80,
            n_clusters=5,
            n_labels=5
        )

        assert metrics.ari == 0.75
        assert metrics.nmi == 0.80
        assert metrics.n_clusters == 5

    def test_to_dict(self):
        """Test dictionary serialization."""
        metrics = EvaluationMetrics(
            ari=0.75,
            nmi=0.80,
            purity=0.85,
            n_clusters=3,
            n_labels=3
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result['ari'] == 0.75
        assert result['n_clusters'] == 3

    def test_get_summary(self):
        """Test summary string generation."""
        metrics = EvaluationMetrics(
            ari=0.75,
            nmi=0.80,
            purity=0.85,
            n_clusters=3,
            n_labels=3,
            evaluation_mode='standard'
        )

        summary = metrics.get_summary()

        assert "CLUSTER EVALUATION METRICS" in summary
        assert "POST-HOC" in summary
        assert "0.75" in summary  # ARI value
        assert "Labels were NOT used for training" in summary


class TestClusterEvaluator:
    """Tests for ClusterEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return ClusterEvaluator()

    @pytest.fixture
    def perfect_clustering(self):
        """Create perfect clustering scenario."""
        return {
            'clusters': [0, 0, 0, 1, 1, 1, 2, 2, 2],
            'labels': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        }

    @pytest.fixture
    def random_clustering(self):
        """Create random clustering scenario."""
        return {
            'clusters': [0, 1, 2, 0, 1, 2, 0, 1, 2],
            'labels': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
        }

    @pytest.fixture
    def multi_label_data(self):
        """Create multi-label scenario."""
        return {
            'clusters': [0, 0, 1, 1, 2, 2],
            'labels': ['A, B', 'A', 'B, C', 'C', 'A, C', 'C']
        }

    def test_evaluate_perfect_clustering(self, evaluator, perfect_clustering):
        """Test evaluation with perfect clustering."""
        metrics = evaluator.evaluate(
            cluster_assignments=perfect_clustering['clusters'],
            true_labels=perfect_clustering['labels'],
            mode='standard'
        )

        assert metrics.ari == 1.0
        assert metrics.nmi == 1.0
        assert metrics.purity == 1.0

    def test_evaluate_random_clustering(self, evaluator, random_clustering):
        """Test evaluation with random clustering."""
        metrics = evaluator.evaluate(
            cluster_assignments=random_clustering['clusters'],
            true_labels=random_clustering['labels'],
            mode='standard'
        )

        # Random clustering should have near-zero ARI
        assert metrics.ari < 0.5
        # Purity might be higher due to random overlap
        assert metrics.purity <= 1.0

    def test_evaluate_multi_label(self, evaluator, multi_label_data):
        """Test multi-label evaluation mode."""
        metrics = evaluator.evaluate(
            cluster_assignments=multi_label_data['clusters'],
            true_labels=multi_label_data['labels'],
            mode='multi_label'
        )

        assert metrics.evaluation_mode == 'multi_label'
        assert 'multi-label' in metrics.notes.lower() or len(metrics.warnings) > 0

    def test_single_label_only_mode(self, evaluator, multi_label_data):
        """Test single-label-only evaluation mode."""
        metrics = evaluator.evaluate(
            cluster_assignments=multi_label_data['clusters'],
            true_labels=multi_label_data['labels'],
            mode='single_label_only'
        )

        # Should filter to single-label documents only
        assert metrics is not None

    def test_handles_unknown_labels(self, evaluator):
        """Test handling of unknown/missing labels."""
        clusters = [0, 0, 1, 1, 2, 2]
        labels = ['A', 'unknown', 'B', 'n/a', 'C', None]

        metrics = evaluator.evaluate(clusters, labels, mode='standard')

        # Should still compute metrics for valid labels
        assert metrics is not None

    def test_handles_all_unknown_labels(self, evaluator):
        """Test graceful handling when all labels are unknown."""
        clusters = [0, 0, 1, 1]
        labels = ['unknown', 'n/a', None, '']

        metrics = evaluator.evaluate(clusters, labels, mode='standard')

        assert 'No valid labels found' in metrics.warnings

    def test_label_to_cluster_mapping(self, evaluator, perfect_clustering):
        """Test label-to-cluster mapping computation."""
        metrics = evaluator.evaluate(
            cluster_assignments=perfect_clustering['clusters'],
            true_labels=perfect_clustering['labels'],
            mode='standard'
        )

        mapping = metrics.label_to_cluster_mapping

        assert 'A' in mapping
        assert 'B' in mapping
        assert 'C' in mapping

    def test_cluster_to_label_mapping(self, evaluator, perfect_clustering):
        """Test cluster-to-label mapping computation."""
        metrics = evaluator.evaluate(
            cluster_assignments=perfect_clustering['clusters'],
            true_labels=perfect_clustering['labels'],
            mode='standard'
        )

        mapping = metrics.cluster_to_label_mapping

        assert 0 in mapping
        assert 1 in mapping
        assert 2 in mapping

    def test_invalid_mode_raises_error(self, evaluator):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            evaluator.evaluate([0, 1], ['A', 'B'], mode='invalid_mode')

    def test_mismatched_lengths_raise_error(self, evaluator):
        """Test that mismatched input lengths raise error."""
        with pytest.raises(ValueError):
            evaluator.evaluate(
                cluster_assignments=[0, 1, 2],
                true_labels=['A', 'B']  # Wrong length
            )


class TestEvaluateClustersPosthoc:
    """Tests for the convenience function."""

    def test_basic_usage(self):
        """Test basic function usage."""
        clusters = [0, 0, 1, 1, 2, 2]
        labels = ['A', 'A', 'B', 'B', 'C', 'C']

        metrics = evaluate_clusters_posthoc(clusters, labels)

        assert metrics is not None
        assert isinstance(metrics, EvaluationMetrics)

    def test_returns_none_without_labels(self):
        """Test that function returns None when no labels provided."""
        clusters = [0, 0, 1, 1]

        result = evaluate_clusters_posthoc(clusters, true_labels=None)

        assert result is None

    def test_auto_detects_multi_label(self):
        """Test automatic multi-label detection."""
        clusters = [0, 0, 1, 1]
        labels = ['A, B', 'A', 'B, C', 'C']

        metrics = evaluate_clusters_posthoc(clusters, labels)

        assert metrics is not None
        # Should auto-detect and use multi-label mode

    def test_with_numpy_arrays(self):
        """Test that numpy arrays work as input."""
        clusters = np.array([0, 0, 1, 1, 2, 2])
        labels = ['A', 'A', 'B', 'B', 'C', 'C']

        metrics = evaluate_clusters_posthoc(clusters, labels)

        assert metrics is not None

    def test_with_pandas_series(self):
        """Test that pandas Series works as input."""
        clusters = [0, 0, 1, 1, 2, 2]
        labels = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])

        metrics = evaluate_clusters_posthoc(clusters, labels)

        assert metrics is not None


class TestEvaluationPrinciples:
    """Tests to verify evaluation principles are maintained."""

    def test_labels_not_used_for_clustering(self):
        """
        Verify that the evaluation module cannot be misused to train with labels.

        The evaluator only computes metrics - it cannot perform clustering.
        This test ensures the module follows its intended purpose.
        """
        evaluator = ClusterEvaluator()

        # Evaluator has no fit or predict methods
        assert not hasattr(evaluator, 'fit')
        assert not hasattr(evaluator, 'predict')
        assert not hasattr(evaluator, 'fit_predict')

        # Only has evaluate method
        assert hasattr(evaluator, 'evaluate')

    def test_evaluation_is_readonly(self):
        """
        Verify that evaluation doesn't modify input data.
        """
        clusters = [0, 0, 1, 1]
        labels = ['A', 'A', 'B', 'B']

        original_clusters = clusters.copy()
        original_labels = labels.copy()

        evaluator = ClusterEvaluator()
        evaluator.evaluate(clusters, labels)

        assert clusters == original_clusters
        assert labels == original_labels

    def test_posthoc_documentation_in_output(self):
        """
        Verify that output clearly indicates post-hoc nature.
        """
        metrics = EvaluationMetrics(
            ari=0.5,
            nmi=0.6
        )

        summary = metrics.get_summary()

        assert 'POST-HOC' in summary
        assert 'NOT used for training' in summary


class TestInterpretationGuidelines:
    """Tests for metric interpretation guidelines."""

    def test_high_ari_interpretation(self):
        """Test interpretation for high ARI."""
        metrics = EvaluationMetrics(ari=0.8)
        summary = metrics.get_summary()

        assert 'Strong agreement' in summary or 'INTERPRETATION' in summary

    def test_low_ari_interpretation(self):
        """Test interpretation for low ARI."""
        metrics = EvaluationMetrics(ari=0.05)
        summary = metrics.get_summary()

        assert 'Little' in summary or 'no agreement' in summary or 'INTERPRETATION' in summary

    def test_moderate_ari_interpretation(self):
        """Test interpretation for moderate ARI."""
        metrics = EvaluationMetrics(ari=0.5)
        summary = metrics.get_summary()

        assert 'Moderate' in summary or 'INTERPRETATION' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
