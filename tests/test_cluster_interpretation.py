"""
Tests for cluster interpretation module.

These tests verify:
1. ClusterInterpreter generates human-readable summaries
2. ClusterSummary contains required information
3. ClusterCodebook produces valid output formats
4. Interpretation works with different clustering methods
"""

import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cluster_interpretation import (
    ClusterInterpreter,
    ClusterInterpretationReport,
    ClusterSummary,
    ClusterCodebook
)


class TestClusterSummary:
    """Tests for ClusterSummary dataclass."""

    def test_basic_creation(self):
        """Test basic summary creation."""
        summary = ClusterSummary(
            cluster_id="CLUSTER_01",
            label="Technology / Innovation / Research",
            top_terms=["technology", "innovation", "research", "development"],
            term_weights=[0.8, 0.6, 0.5, 0.4],
            document_count=50,
            interpretation_confidence=0.85
        )

        assert summary.cluster_id == "CLUSTER_01"
        assert "technology" in summary.top_terms
        assert summary.document_count == 50

    def test_to_dict(self):
        """Test dictionary serialization."""
        summary = ClusterSummary(
            cluster_id="CLUSTER_01",
            label="Test Label",
            top_terms=["term1", "term2"],
            document_count=10
        )

        result = summary.to_dict()

        assert isinstance(result, dict)
        assert result['cluster_id'] == "CLUSTER_01"
        assert result['label'] == "Test Label"
        assert result['document_count'] == 10

    def test_display_string(self):
        """Test display string generation."""
        summary = ClusterSummary(
            cluster_id="CLUSTER_01",
            label="Technology",
            top_terms=["tech", "innovation", "digital", "software", "hardware"],
            document_count=25
        )

        display = summary.get_display_string()

        assert "CLUSTER_01" in display
        assert "Technology" in display
        assert "25 docs" in display
        assert "tech" in display

    def test_warnings_in_display(self):
        """Test warnings appear in display string."""
        summary = ClusterSummary(
            cluster_id="CLUSTER_01",
            label="Weak Cluster",
            top_terms=["term"],
            document_count=2,
            warnings=["Very small cluster (<5 docs)", "Few distinctive terms"]
        )

        display = summary.get_display_string()

        assert "Warnings:" in display


class TestClusterInterpreter:
    """Tests for ClusterInterpreter class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        texts = [
            "machine learning algorithms for prediction",
            "deep learning neural networks training",
            "artificial intelligence research papers",
            "sports news football soccer basketball",
            "athletic competition Olympic games",
            "team sports championship league",
            "political news government policy",
            "election campaign voting democracy",
            "congress parliament legislation bills"
        ] * 10

        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)

        return {
            'texts': texts,
            'vectorizer': vectorizer,
            'model': kmeans,
            'labels': labels.tolist(),
            'matrix': matrix
        }

    @pytest.fixture
    def interpreter(self):
        """Create interpreter instance."""
        return ClusterInterpreter(
            n_top_terms=10,
            n_label_terms=3,
            n_representative_docs=3
        )

    def test_interpret_clusters(self, interpreter, sample_data):
        """Test basic cluster interpretation."""
        report = interpreter.interpret_clusters(
            vectorizer=sample_data['vectorizer'],
            cluster_model=sample_data['model'],
            texts=sample_data['texts'],
            cluster_assignments=sample_data['labels'],
            feature_matrix=sample_data['matrix'],
            method_name='tfidf_kmeans'
        )

        assert isinstance(report, ClusterInterpretationReport)
        assert report.n_clusters == 3
        assert report.n_documents == len(sample_data['texts'])
        assert len(report.summaries) == 3

    def test_summaries_have_labels(self, interpreter, sample_data):
        """Test that all clusters have human-readable labels."""
        report = interpreter.interpret_clusters(
            vectorizer=sample_data['vectorizer'],
            cluster_model=sample_data['model'],
            texts=sample_data['texts'],
            cluster_assignments=sample_data['labels'],
            feature_matrix=sample_data['matrix']
        )

        for cluster_id, summary in report.summaries.items():
            assert summary.label is not None
            assert len(summary.label) > 0
            assert summary.label != cluster_id  # Label should differ from ID

    def test_summaries_have_top_terms(self, interpreter, sample_data):
        """Test that all clusters have top terms."""
        report = interpreter.interpret_clusters(
            vectorizer=sample_data['vectorizer'],
            cluster_model=sample_data['model'],
            texts=sample_data['texts'],
            cluster_assignments=sample_data['labels'],
            feature_matrix=sample_data['matrix']
        )

        for cluster_id, summary in report.summaries.items():
            assert len(summary.top_terms) > 0
            assert all(isinstance(t, str) for t in summary.top_terms)

    def test_representative_docs_included(self, interpreter, sample_data):
        """Test that representative documents are included."""
        report = interpreter.interpret_clusters(
            vectorizer=sample_data['vectorizer'],
            cluster_model=sample_data['model'],
            texts=sample_data['texts'],
            cluster_assignments=sample_data['labels'],
            feature_matrix=sample_data['matrix']
        )

        for cluster_id, summary in report.summaries.items():
            if summary.document_count > 0:
                assert len(summary.representative_docs) > 0
                for doc in summary.representative_docs:
                    assert 'text' in doc
                    assert 'index' in doc

    def test_interpretability_score(self, interpreter, sample_data):
        """Test interpretability scores are computed."""
        report = interpreter.interpret_clusters(
            vectorizer=sample_data['vectorizer'],
            cluster_model=sample_data['model'],
            texts=sample_data['texts'],
            cluster_assignments=sample_data['labels'],
            feature_matrix=sample_data['matrix']
        )

        assert report.overall_interpretability >= 0
        assert report.overall_interpretability <= 1

        for summary in report.summaries.values():
            assert summary.interpretation_confidence >= 0
            assert summary.interpretation_confidence <= 1

    def test_display_report(self, interpreter, sample_data):
        """Test display report generation."""
        report = interpreter.interpret_clusters(
            vectorizer=sample_data['vectorizer'],
            cluster_model=sample_data['model'],
            texts=sample_data['texts'],
            cluster_assignments=sample_data['labels'],
            feature_matrix=sample_data['matrix']
        )

        display = report.get_display_report()

        assert "CLUSTER INTERPRETATION REPORT" in display
        assert "CLUSTER DETAILS" in display
        assert "NOTE: Cluster IDs are internal identifiers" in display

    def test_to_dict(self, interpreter, sample_data):
        """Test dictionary serialization of report."""
        report = interpreter.interpret_clusters(
            vectorizer=sample_data['vectorizer'],
            cluster_model=sample_data['model'],
            texts=sample_data['texts'],
            cluster_assignments=sample_data['labels'],
            feature_matrix=sample_data['matrix']
        )

        result = report.to_dict()

        assert isinstance(result, dict)
        assert 'n_clusters' in result
        assert 'cluster_summaries' in result
        assert len(result['cluster_summaries']) == 3

    def test_cluster_comparison_table(self, interpreter, sample_data):
        """Test cluster comparison table generation."""
        report = interpreter.interpret_clusters(
            vectorizer=sample_data['vectorizer'],
            cluster_model=sample_data['model'],
            texts=sample_data['texts'],
            cluster_assignments=sample_data['labels'],
            feature_matrix=sample_data['matrix']
        )

        df = interpreter.get_cluster_comparison(report)

        assert isinstance(df, pd.DataFrame)
        assert 'Cluster ID' in df.columns
        assert 'Label' in df.columns
        assert 'Documents' in df.columns
        assert len(df) == 3

    def test_mismatched_inputs_raise_error(self, interpreter):
        """Test that mismatched inputs raise ValueError."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        texts = ["doc1", "doc2", "doc3"]
        wrong_assignments = [0, 1]  # Wrong length

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(matrix)

        with pytest.raises(ValueError):
            interpreter.interpret_clusters(
                vectorizer=vectorizer,
                cluster_model=kmeans,
                texts=texts,
                cluster_assignments=wrong_assignments,
                feature_matrix=matrix
            )


class TestClusterCodebook:
    """Tests for ClusterCodebook class."""

    @pytest.fixture
    def sample_report(self):
        """Create sample interpretation report."""
        summaries = {
            "CLUSTER_01": ClusterSummary(
                cluster_id="CLUSTER_01",
                label="Technology / AI",
                top_terms=["technology", "ai", "machine", "learning", "data"],
                term_weights=[0.8, 0.7, 0.6, 0.5, 0.4],
                document_count=30,
                representative_docs=[
                    {"text": "AI and machine learning advances", "index": 0, "similarity": 0.9}
                ],
                interpretation_confidence=0.85
            ),
            "CLUSTER_02": ClusterSummary(
                cluster_id="CLUSTER_02",
                label="Sports / Athletics",
                top_terms=["sports", "team", "game", "player", "score"],
                term_weights=[0.75, 0.65, 0.55, 0.45, 0.35],
                document_count=25,
                representative_docs=[
                    {"text": "Team wins championship game", "index": 1, "similarity": 0.88}
                ],
                interpretation_confidence=0.80
            )
        }

        return ClusterInterpretationReport(
            summaries=summaries,
            n_clusters=2,
            n_documents=55,
            overall_interpretability=0.825,
            method_used='tfidf_kmeans'
        )

    def test_to_dict(self, sample_report):
        """Test dictionary output."""
        codebook = ClusterCodebook(sample_report)
        result = codebook.to_dict()

        assert 'method' in result
        assert 'n_clusters' in result
        assert 'codes' in result
        assert 'disclaimer' in result
        assert len(result['codes']) == 2

    def test_to_markdown(self, sample_report):
        """Test markdown output."""
        codebook = ClusterCodebook(sample_report)
        md = codebook.to_markdown()

        assert "# Cluster Codebook" in md
        assert "CLUSTER_01" in md
        assert "Technology / AI" in md
        assert "Keywords" in md
        assert "> **Note**:" in md

    def test_to_csv(self, sample_report):
        """Test CSV output."""
        codebook = ClusterCodebook(sample_report)
        csv = codebook.to_csv()

        assert "Cluster ID" in csv
        assert "CLUSTER_01" in csv
        assert "Technology / AI" in csv

    def test_disclaimer_included(self, sample_report):
        """Test that disclaimer about unsupervised nature is included."""
        codebook = ClusterCodebook(sample_report)
        result = codebook.to_dict()

        assert 'disclaimer' in result
        assert 'unsupervised' in result['disclaimer'].lower()


class TestInterpretationWithDifferentMethods:
    """Tests for interpretation with different clustering methods."""

    @pytest.fixture
    def texts(self):
        """Sample texts for testing."""
        return [
            "machine learning prediction model",
            "deep neural network training",
            "artificial intelligence research",
            "football soccer basketball sports",
            "athletic competition games",
            "team championship league"
        ] * 20

    def test_with_lda(self, texts):
        """Test interpretation works with LDA."""
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation

        vectorizer = CountVectorizer(max_features=100)
        matrix = vectorizer.fit_transform(texts)

        lda = LatentDirichletAllocation(n_components=2, random_state=42, max_iter=10)
        doc_topics = lda.fit_transform(matrix)
        labels = doc_topics.argmax(axis=1).tolist()

        interpreter = ClusterInterpreter()
        report = interpreter.interpret_clusters(
            vectorizer=vectorizer,
            cluster_model=lda,
            texts=texts,
            cluster_assignments=labels,
            feature_matrix=matrix,
            method_name='lda'
        )

        assert report.n_clusters == 2
        assert all(s.label for s in report.summaries.values())

    def test_with_nmf(self, texts):
        """Test interpretation works with NMF."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import NMF

        vectorizer = TfidfVectorizer(max_features=100)
        matrix = vectorizer.fit_transform(texts)

        nmf = NMF(n_components=2, random_state=42, max_iter=100)
        doc_topics = nmf.fit_transform(matrix)
        labels = doc_topics.argmax(axis=1).tolist()

        interpreter = ClusterInterpreter()
        report = interpreter.interpret_clusters(
            vectorizer=vectorizer,
            cluster_model=nmf,
            texts=texts,
            cluster_assignments=labels,
            feature_matrix=matrix,
            method_name='nmf'
        )

        assert report.n_clusters == 2
        assert all(s.label for s in report.summaries.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
