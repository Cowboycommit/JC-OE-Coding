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
    ClusterCodebook,
    detect_domain_stopwords,
    DEFAULT_LABEL_STOPWORDS
)


class TestClusterSummary:
    """Tests for ClusterSummary dataclass."""

    def test_basic_creation(self):
        """Test basic summary creation."""
        summary = ClusterSummary(
            cluster_id="CLUSTER_01",
            label="Technology Innovation Research",
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
                label="Technology AI",
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
                label="Sports Athletics",
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
        assert "Technology AI" in md
        assert "Keywords" in md
        assert "> **Note**:" in md

    def test_to_csv(self, sample_report):
        """Test CSV output."""
        codebook = ClusterCodebook(sample_report)
        csv = codebook.to_csv()

        assert "Cluster ID" in csv
        assert "CLUSTER_01" in csv
        assert "Technology AI" in csv

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


class TestDomainStopwords:
    """Tests for domain-specific stopword detection."""

    def test_detect_domain_stopwords_high_frequency(self):
        """Test that high-frequency words are detected as domain stopwords."""
        # Create texts where 'survey' appears in >70% of documents
        texts = [
            "survey response quality good",
            "survey feedback excellent",
            "survey results show improvement",
            "survey data analysis complete",
            "survey participants satisfied",
            "survey methodology sound",
            "survey questions clear",
            "survey findings important",
            "unrelated topic here",  # No survey
            "different subject matter"  # No survey
        ]

        domain_stopwords = detect_domain_stopwords(texts, min_doc_frequency=0.7)

        # 'survey' appears in 8/10 = 80% of documents, should be detected
        assert "survey" in domain_stopwords

    def test_detect_domain_stopwords_empty_texts(self):
        """Test that empty texts return empty set."""
        domain_stopwords = detect_domain_stopwords([])
        assert domain_stopwords == set()

    def test_detect_domain_stopwords_excludes_default_stopwords(self):
        """Test that default stopwords are not included."""
        texts = ["the product is good"] * 10

        domain_stopwords = detect_domain_stopwords(texts)

        assert "the" not in domain_stopwords
        assert "is" not in domain_stopwords

    def test_detect_domain_stopwords_max_words_limit(self):
        """Test that max_words parameter limits results."""
        texts = [f"word{i} common text here" for i in range(20)]
        texts.extend(["common text here"] * 50)

        domain_stopwords = detect_domain_stopwords(texts, max_words=3)

        assert len(domain_stopwords) <= 3


class TestPhraseBasedLabels:
    """Tests for phrase-based label generation."""

    @pytest.fixture
    def interpreter_with_phrases(self):
        """Create interpreter with phrase preference enabled."""
        return ClusterInterpreter(
            n_top_terms=10,
            n_label_terms=3,
            prefer_ngram_phrases=True
        )

    @pytest.fixture
    def interpreter_without_phrases(self):
        """Create interpreter with phrase preference disabled."""
        return ClusterInterpreter(
            n_top_terms=10,
            n_label_terms=3,
            prefer_ngram_phrases=False
        )

    def test_phrase_based_label_prefers_ngrams(self, interpreter_with_phrases):
        """Test that phrase-based labels prefer n-grams."""
        filtered_terms = ["customer service", "customer", "service", "support", "help"]
        filtered_weights = [0.8, 0.6, 0.5, 0.4, 0.3]

        label = interpreter_with_phrases._generate_phrase_based_label(
            filtered_terms, filtered_weights, n_label_terms=3
        )

        # Should prefer "customer service" over individual words
        assert "Customer Service" in label

    def test_word_based_label_uses_individual_words(self, interpreter_without_phrases):
        """Test that word-based labels use individual words."""
        filtered_terms = ["customer service", "support", "help"]
        filtered_weights = [0.8, 0.5, 0.4]

        label = interpreter_without_phrases._generate_word_based_label(
            filtered_terms, filtered_weights, n_label_terms=3
        )

        # Word-based should break up phrases
        assert label  # Should have a label
        words = label.lower().split()
        assert len(words) == 3

    def test_phrase_based_avoids_full_overlap(self, interpreter_with_phrases):
        """Test that phrase-based labels avoid fully overlapping phrases."""
        # Use terms where one is a subset of another
        filtered_terms = ["customer", "customer service", "support team"]
        filtered_weights = [0.5, 0.8, 0.6]  # "customer service" should be preferred

        label = interpreter_with_phrases._generate_phrase_based_label(
            filtered_terms, filtered_weights, n_label_terms=2
        )

        # "customer service" should be selected, and "customer" alone should be skipped
        # due to full overlap
        assert "Customer Service" in label
        # If only "Customer Service" is selected, "customer" should not appear separately
        # The label should have distinct phrases
        assert label  # Should have a non-empty label


class TestCoherenceCalculation:
    """Tests for cluster coherence calculation."""

    @pytest.fixture
    def sample_data_with_matrix(self):
        """Create sample data with feature matrix."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        texts = [
            # Cluster 1: Very coherent (similar texts)
            "machine learning algorithms prediction model",
            "machine learning model training data",
            "machine learning neural network deep",
            # Cluster 2: Less coherent (diverse texts)
            "sports news football",
            "political election campaign",
            "weather forecast rain",
        ] * 5

        vectorizer = TfidfVectorizer(max_features=50)
        matrix = vectorizer.fit_transform(texts)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)

        return {
            'texts': texts,
            'vectorizer': vectorizer,
            'model': kmeans,
            'labels': labels.tolist(),
            'matrix': matrix
        }

    def test_coherence_calculated_with_feature_matrix(self, sample_data_with_matrix):
        """Test that coherence is calculated when feature matrix provided."""
        interpreter = ClusterInterpreter()

        report = interpreter.interpret_clusters(
            vectorizer=sample_data_with_matrix['vectorizer'],
            cluster_model=sample_data_with_matrix['model'],
            texts=sample_data_with_matrix['texts'],
            cluster_assignments=sample_data_with_matrix['labels'],
            feature_matrix=sample_data_with_matrix['matrix']
        )

        # Check that coherence scores are computed
        for summary in report.summaries.values():
            if summary.document_count >= 2:
                assert summary.coherence_score is not None
                assert 0 <= summary.coherence_score <= 1

        # Overall coherence should be computed
        assert report.overall_coherence is not None

    def test_coherence_not_calculated_without_matrix(self):
        """Test that coherence is None when no feature matrix provided."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        texts = ["text one", "text two", "text three"] * 10
        vectorizer = TfidfVectorizer(max_features=50)
        matrix = vectorizer.fit_transform(texts)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)

        interpreter = ClusterInterpreter()
        report = interpreter.interpret_clusters(
            vectorizer=vectorizer,
            cluster_model=kmeans,
            texts=texts,
            cluster_assignments=labels.tolist(),
            feature_matrix=None  # No matrix provided
        )

        # Coherence should be None
        for summary in report.summaries.values():
            assert summary.coherence_score is None


class TestTuningRecommendations:
    """Tests for automated tuning recommendations."""

    def test_recommendations_for_low_interpretability(self):
        """Test that low interpretability triggers recommendations."""
        interpreter = ClusterInterpreter(low_interpretability_threshold=0.5)

        # Create mock summaries with low confidence
        summaries = {
            "CLUSTER_01": ClusterSummary(
                cluster_id="CLUSTER_01",
                label="Test",
                top_terms=["term"],
                document_count=10,
                interpretation_confidence=0.2
            ),
            "CLUSTER_02": ClusterSummary(
                cluster_id="CLUSTER_02",
                label="Test2",
                top_terms=["term2"],
                document_count=10,
                interpretation_confidence=0.2
            )
        }

        recommendations = interpreter._generate_tuning_recommendations(
            summaries=summaries,
            overall_interpretability=0.2,
            overall_coherence=None,
            n_clusters=2
        )

        assert len(recommendations) > 0
        assert any("interpretability" in r.lower() for r in recommendations)

    def test_recommendations_for_low_coherence(self):
        """Test that low coherence triggers recommendations."""
        interpreter = ClusterInterpreter(min_coherence_threshold=0.5)

        summaries = {
            "CLUSTER_01": ClusterSummary(
                cluster_id="CLUSTER_01",
                label="Test",
                top_terms=["term"],
                document_count=10,
                interpretation_confidence=0.8,
                coherence_score=0.2
            )
        }

        recommendations = interpreter._generate_tuning_recommendations(
            summaries=summaries,
            overall_interpretability=0.8,
            overall_coherence=0.2,
            n_clusters=6
        )

        assert len(recommendations) > 0
        assert any("coherence" in r.lower() for r in recommendations)

    def test_recommendations_for_empty_clusters(self):
        """Test that empty clusters trigger recommendations."""
        interpreter = ClusterInterpreter()

        summaries = {
            "CLUSTER_01": ClusterSummary(
                cluster_id="CLUSTER_01",
                label="Test",
                top_terms=["term"],
                document_count=0,  # Empty
                interpretation_confidence=0.0
            ),
            "CLUSTER_02": ClusterSummary(
                cluster_id="CLUSTER_02",
                label="Test2",
                top_terms=["term2"],
                document_count=10,
                interpretation_confidence=0.8
            )
        }

        recommendations = interpreter._generate_tuning_recommendations(
            summaries=summaries,
            overall_interpretability=0.4,
            overall_coherence=None,
            n_clusters=2
        )

        assert any("empty" in r.lower() for r in recommendations)


class TestCustomStopwords:
    """Tests for custom stopword functionality."""

    def test_custom_stopwords_excluded_from_labels(self):
        """Test that custom stopwords are excluded from labels."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        # Create texts with a domain-specific word that should be excluded
        texts = [
            "acme product quality good",
            "acme service excellent",
            "acme delivery fast",
        ] * 10

        vectorizer = TfidfVectorizer(max_features=50)
        matrix = vectorizer.fit_transform(texts)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(matrix)

        # Use "acme" as custom stopword
        interpreter = ClusterInterpreter(
            custom_stopwords={"acme"},
            detect_domain_stopwords=False  # Disable auto-detection
        )

        report = interpreter.interpret_clusters(
            vectorizer=vectorizer,
            cluster_model=kmeans,
            texts=texts,
            cluster_assignments=labels.tolist(),
            feature_matrix=matrix
        )

        # Labels should not contain "acme"
        for summary in report.summaries.values():
            assert "acme" not in summary.label.lower()

    def test_combined_stopwords(self):
        """Test that default, custom, and domain stopwords are combined."""
        interpreter = ClusterInterpreter(
            custom_stopwords={"custom1", "custom2"}
        )
        interpreter._domain_stopwords = {"domain1", "domain2"}

        combined = interpreter._get_combined_stopwords()

        # Should have default stopwords
        assert "the" in combined
        assert "is" in combined
        # Should have custom stopwords
        assert "custom1" in combined
        assert "custom2" in combined
        # Should have domain stopwords
        assert "domain1" in combined
        assert "domain2" in combined


class TestReportWithNewFields:
    """Tests for ClusterInterpretationReport with new fields."""

    def test_report_includes_coherence(self):
        """Test that report includes overall coherence."""
        summaries = {
            "CLUSTER_01": ClusterSummary(
                cluster_id="CLUSTER_01",
                label="Test",
                top_terms=["term"],
                document_count=10,
                coherence_score=0.75
            )
        }

        report = ClusterInterpretationReport(
            summaries=summaries,
            n_clusters=1,
            n_documents=10,
            overall_interpretability=0.8,
            overall_coherence=0.75,
            tuning_recommendations=["Test recommendation"],
            domain_stopwords_used={"domain1", "domain2"}
        )

        result = report.to_dict()

        assert "overall_coherence" in result
        assert result["overall_coherence"] == 0.75
        assert "tuning_recommendations" in result
        assert "domain_stopwords_used" in result

    def test_display_report_shows_coherence(self):
        """Test that display report includes coherence information."""
        summaries = {
            "CLUSTER_01": ClusterSummary(
                cluster_id="CLUSTER_01",
                label="Test Label",
                top_terms=["term1", "term2"],
                document_count=10,
                coherence_score=0.65
            )
        }

        report = ClusterInterpretationReport(
            summaries=summaries,
            n_clusters=1,
            n_documents=10,
            overall_interpretability=0.8,
            overall_coherence=0.65,
            tuning_recommendations=["Reduce cluster count"]
        )

        display = report.get_display_report()

        assert "Coherence" in display
        assert "0.65" in display
        assert "TUNING RECOMMENDATIONS" in display

    def test_display_report_always_shows_terms(self):
        """Test that display report shows top terms by default."""
        summaries = {
            "CLUSTER_01": ClusterSummary(
                cluster_id="CLUSTER_01",
                label="Short Label",
                top_terms=["important_term", "another_term", "third_term"],
                document_count=10
            )
        }

        report = ClusterInterpretationReport(
            summaries=summaries,
            n_clusters=1,
            n_documents=10,
            overall_interpretability=0.8
        )

        display = report.get_display_report(always_show_terms=True)

        # Terms should be visible in display
        assert "important_term" in display
        assert "Always review top_terms" in display


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
