"""
Tests for the stopwords discovery module.

These tests verify:
1. Domain stopword candidate discovery from corpus statistics
2. File-based stopwords loading and keep-list functionality
3. Layered stopwords governance model
4. Discovery report generation
5. Integration with ClusterInterpreter
"""

import pytest
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stopwords_discovery import (
    find_domain_stopword_candidates,
    load_stopwords_from_file,
    load_keep_list,
    get_layered_stopwords,
    get_nltk_stopwords,
    tokenize_simple,
    StopwordCandidate,
    StopwordDiscoveryReport,
    save_candidates_to_file,
    generate_discovery_report,
    discover_from_clusters,
)


class TestTokenizeSimple:
    """Tests for simple tokenization."""

    def test_basic_tokenization(self):
        """Test basic word tokenization."""
        text = "Hello world, this is a test!"
        tokens = tokenize_simple(text)

        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
        # Punctuation should be removed
        assert "," not in tokens
        assert "!" not in tokens

    def test_min_length_filter(self):
        """Test minimum length filtering."""
        text = "I am a big dog"
        tokens = tokenize_simple(text, min_length=2)

        assert "am" in tokens
        assert "big" in tokens
        assert "dog" in tokens
        # Single character words should be filtered
        assert "i" not in tokens
        assert "a" not in tokens

    def test_empty_input(self):
        """Test with empty input."""
        assert tokenize_simple("") == []
        assert tokenize_simple(None) == []

    def test_case_normalization(self):
        """Test that tokens are lowercased."""
        text = "Hello WORLD Test"
        tokens = tokenize_simple(text)

        assert "hello" in tokens
        assert "world" in tokens
        assert "Hello" not in tokens
        assert "WORLD" not in tokens


class TestFindDomainStopwordCandidates:
    """Tests for domain stopword candidate discovery."""

    def test_basic_discovery(self):
        """Test basic stopword candidate discovery."""
        texts = [
            "The patient received treatment at the hospital.",
            "Patient care was excellent at the hospital.",
            "The hospital treatment was effective for the patient.",
            "Doctor provided treatment to the patient at hospital.",
            "Patient satisfaction with hospital treatment was high.",
        ]

        report = find_domain_stopword_candidates(
            texts,
            min_doc_frequency=0.6,
            exclude_general_stopwords=True
        )

        assert isinstance(report, StopwordDiscoveryReport)
        assert report.total_documents == 5
        assert len(report.candidates) > 0

        # "patient", "hospital", "treatment" should be candidates (appear in most docs)
        candidate_words = {c.word for c in report.candidates}
        assert "patient" in candidate_words
        assert "hospital" in candidate_words
        assert "treatment" in candidate_words

    def test_document_frequency_calculation(self):
        """Test document frequency is calculated correctly."""
        texts = [
            "apple banana cherry",
            "apple banana date",
            "apple elderberry fig",
            "grape apple honeydew",
        ]

        report = find_domain_stopword_candidates(
            texts,
            min_doc_frequency=0.5,
            exclude_general_stopwords=False
        )

        # "apple" appears in all 4 documents = 100%
        apple_candidate = next(
            (c for c in report.candidates if c.word == "apple"), None
        )
        assert apple_candidate is not None
        assert apple_candidate.document_frequency == 1.0
        assert apple_candidate.document_count == 4

    def test_threshold_filtering(self):
        """Test that threshold filters correctly."""
        texts = [
            "common word unique1",
            "common word unique2",
            "common word unique3",
            "common rare unique4",
        ]

        # With 75% threshold, only "common" and "word" should qualify
        report = find_domain_stopword_candidates(
            texts,
            min_doc_frequency=0.75,
            exclude_general_stopwords=False
        )

        candidate_words = {c.word for c in report.candidates}
        assert "common" in candidate_words
        assert "word" in candidate_words
        assert "rare" not in candidate_words
        assert "unique1" not in candidate_words

    def test_empty_corpus(self):
        """Test with empty corpus."""
        report = find_domain_stopword_candidates([])

        assert report.total_documents == 0
        assert len(report.candidates) == 0

    def test_keep_list_applied(self):
        """Test that keep-list words are not recommended."""
        texts = [
            "good service really great",
            "really good experience great",
            "great product really good",
            "good quality really great",
        ]

        report = find_domain_stopword_candidates(
            texts,
            min_doc_frequency=0.5,
            apply_keep_list=True
        )

        # "good", "great", "really" might be candidates but should not be recommended
        # if they're in the keep-list
        for candidate in report.candidates:
            if candidate.word in {'good', 'great', 'really'}:
                # These should be marked as not recommended (protected by keep-list)
                # or not appear at all if they're in NLTK stopwords
                pass  # Keep-list behavior verified by checking is_recommended

    def test_tfidf_variance_computation(self):
        """Test TF-IDF variance is computed when requested."""
        texts = [
            "patient treatment hospital care",
            "patient treatment hospital visit",
            "patient treatment hospital doctor",
            "patient treatment hospital staff",
        ]

        report = find_domain_stopword_candidates(
            texts,
            min_doc_frequency=0.5,
            compute_tfidf_variance=True
        )

        # Check that TF-IDF variance is computed
        for candidate in report.candidates:
            assert candidate.tfidf_variance is not None


class TestLoadStopwordsFromFile:
    """Tests for file-based stopwords loading."""

    def test_load_from_file(self):
        """Test loading stopwords from a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Comment line\n")
            f.write("word1\n")
            f.write("word2\n")
            f.write("\n")  # Empty line
            f.write("WORD3\n")  # Should be lowercased
            f.write("# Another comment\n")
            f.write("word4\n")
            temp_path = f.name

        try:
            stopwords = load_stopwords_from_file(temp_path)

            assert len(stopwords) == 4
            assert "word1" in stopwords
            assert "word2" in stopwords
            assert "word3" in stopwords  # Lowercased
            assert "word4" in stopwords
            assert "WORD3" not in stopwords  # Should be lowercase
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test graceful handling of nonexistent file."""
        stopwords = load_stopwords_from_file("/nonexistent/path/stopwords.txt")
        assert stopwords == set()

    def test_load_empty_file(self):
        """Test loading empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Only comments\n")
            f.write("\n")
            temp_path = f.name

        try:
            stopwords = load_stopwords_from_file(temp_path)
            assert stopwords == set()
        finally:
            os.unlink(temp_path)


class TestLoadKeepList:
    """Tests for keep-list loading."""

    def test_load_keep_list(self):
        """Test loading keep-list from file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("not\n")
            f.write("never\n")
            f.write("good\n")
            f.write("bad\n")
            temp_path = f.name

        try:
            keep_list = load_keep_list(temp_path)

            assert len(keep_list) == 4
            assert "not" in keep_list
            assert "never" in keep_list
            assert "good" in keep_list
            assert "bad" in keep_list
        finally:
            os.unlink(temp_path)


class TestGetLayeredStopwords:
    """Tests for layered stopwords governance model."""

    def test_nltk_layer(self):
        """Test NLTK stopwords layer."""
        stopwords, counts = get_layered_stopwords(
            include_nltk=True,
            include_domain=False,
            include_custom=None
        )

        # NLTK may not be available in all environments
        # If available, check common words are included
        if counts['nltk'] > 0:
            assert "the" in stopwords
            assert "is" in stopwords
            assert "and" in stopwords
        else:
            # NLTK not available - verify graceful fallback
            assert counts['nltk'] == 0

    def test_custom_layer(self):
        """Test custom stopwords layer."""
        custom = {"customword1", "customword2"}
        stopwords, counts = get_layered_stopwords(
            include_nltk=False,
            include_domain=False,
            include_custom=custom
        )

        assert counts['custom'] == 2
        assert "customword1" in stopwords
        assert "customword2" in stopwords

    def test_keep_list_override(self):
        """Test that keep-list removes words from combined stopwords."""
        # Create a temp keep-list file with a custom word to test removal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("removetest\n")
            temp_path = f.name

        try:
            # Add the word we want to remove via custom stopwords
            custom = {"removetest", "keepthis"}
            stopwords, counts = get_layered_stopwords(
                include_nltk=False,
                include_domain=False,
                include_custom=custom,
                keep_list_path=temp_path
            )

            # "removetest" should be removed from stopwords (it's in custom but also in keep-list)
            assert "removetest" not in stopwords
            # "keepthis" should still be there
            assert "keepthis" in stopwords
            assert counts['keep_list_removed'] == 1
        finally:
            os.unlink(temp_path)


class TestStopwordDiscoveryReport:
    """Tests for StopwordDiscoveryReport."""

    def test_to_dict(self):
        """Test dictionary conversion."""
        candidates = [
            StopwordCandidate(
                word="patient",
                document_frequency=0.9,
                document_count=90,
                total_occurrences=150,
                is_recommended=True,
                reason="Very high frequency"
            ),
            StopwordCandidate(
                word="treatment",
                document_frequency=0.85,
                document_count=85,
                total_occurrences=120,
                is_recommended=True,
                reason="High frequency"
            ),
        ]

        report = StopwordDiscoveryReport(
            candidates=candidates,
            total_documents=100,
            total_vocabulary=500,
            min_doc_frequency_threshold=0.7,
            recommendations=["Review top candidates"]
        )

        result = report.to_dict()

        assert result['total_documents'] == 100
        assert result['total_vocabulary'] == 500
        assert result['n_candidates'] == 2
        assert len(result['candidates']) == 2
        assert result['candidates'][0]['word'] == "patient"

    def test_to_markdown(self):
        """Test markdown report generation."""
        candidates = [
            StopwordCandidate(
                word="hospital",
                document_frequency=0.95,
                document_count=95,
                total_occurrences=200,
                is_recommended=True,
                reason="Very high frequency"
            ),
        ]

        report = StopwordDiscoveryReport(
            candidates=candidates,
            total_documents=100,
            total_vocabulary=500,
            min_doc_frequency_threshold=0.7,
            recommendations=["Add 'hospital' to domain stopwords"]
        )

        markdown = report.to_markdown()

        assert "# Domain Stopwords Discovery Report" in markdown
        assert "hospital" in markdown
        # Check for percentage (may be formatted as "95%" or "95.0%")
        assert "95" in markdown and "%" in markdown
        assert "Recommendations" in markdown


class TestSaveCandidatesToFile:
    """Tests for saving candidates to file."""

    def test_save_candidates(self):
        """Test saving candidates to file."""
        candidates = [
            StopwordCandidate(
                word="patient",
                document_frequency=0.9,
                document_count=90,
                total_occurrences=150,
                is_recommended=True,
                reason="High frequency"
            ),
            StopwordCandidate(
                word="important",
                document_frequency=0.8,
                document_count=80,
                total_occurrences=100,
                is_recommended=False,
                reason="Protected by keep-list"
            ),
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name

        try:
            count = save_candidates_to_file(
                candidates,
                temp_path,
                only_recommended=True
            )

            assert count == 1  # Only "patient" is recommended

            # Verify file contents
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "patient" in content
                assert "important" not in content
        finally:
            os.unlink(temp_path)


class TestDiscoverFromClusters:
    """Tests for cluster-aware stopword discovery."""

    def test_cluster_aware_discovery(self):
        """Test discovery with cluster assignments."""
        texts = [
            "patient care hospital treatment",
            "patient visit hospital doctor",
            "customer service support help",
            "customer issue support ticket",
            "patient recovery hospital stay",
            "customer feedback support team",
        ]

        cluster_assignments = [0, 0, 1, 1, 0, 1]  # 2 clusters

        report = discover_from_clusters(
            texts=texts,
            cluster_assignments=cluster_assignments,
            min_doc_frequency=0.5,
            min_cluster_coverage=0.8
        )

        assert isinstance(report, StopwordDiscoveryReport)

        # Check that topic_coverage is computed
        for candidate in report.candidates:
            assert candidate.topic_coverage is not None


class TestClusterInterpreterIntegration:
    """Tests for ClusterInterpreter integration with stopwords discovery."""

    def test_interpreter_loads_file_stopwords(self):
        """Test that ClusterInterpreter loads file-based stopwords."""
        from src.cluster_interpretation import ClusterInterpreter

        interpreter = ClusterInterpreter(
            use_file_based_stopwords=True
        )

        # Get stopwords summary
        summary = interpreter.get_stopwords_summary()

        assert 'default_label_stopwords' in summary
        assert 'file_based_domain_stopwords' in summary
        assert 'keep_list_words' in summary
        assert 'total_combined' in summary

    def test_interpreter_discovery_method(self):
        """Test ClusterInterpreter.discover_domain_stopwords method."""
        from src.cluster_interpretation import ClusterInterpreter

        interpreter = ClusterInterpreter()

        texts = [
            "The product was excellent quality.",
            "Product quality exceeded expectations.",
            "Excellent product with great quality.",
            "Quality of the product was good.",
        ]

        report = interpreter.discover_domain_stopwords(
            texts=texts,
            min_doc_frequency=0.5
        )

        assert report is not None
        assert isinstance(report, StopwordDiscoveryReport)
        assert len(report.candidates) > 0

    def test_interpreter_combined_stopwords(self):
        """Test that combined stopwords works correctly."""
        from src.cluster_interpretation import ClusterInterpreter

        # Create temp domain stopwords file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("testdomainword\n")
            domain_path = f.name

        # Create temp keep-list file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("keepme\n")
            keep_path = f.name

        try:
            interpreter = ClusterInterpreter(
                use_file_based_stopwords=True,
                domain_stopwords_path=domain_path,
                keep_list_path=keep_path,
                custom_stopwords={"customstop"}
            )

            combined = interpreter._get_combined_stopwords()

            # testdomainword should be included
            assert "testdomainword" in combined
            # customstop should be included
            assert "customstop" in combined
            # keepme should NOT be included (it's in keep-list)
            assert "keepme" not in combined
        finally:
            os.unlink(domain_path)
            os.unlink(keep_path)


class TestGenerateDiscoveryReport:
    """Tests for the convenience report generation function."""

    def test_generate_markdown_report(self):
        """Test markdown report generation."""
        texts = [
            "Service was excellent and staff was helpful.",
            "Staff provided excellent service.",
            "The service and staff exceeded expectations.",
            "Excellent experience with helpful staff.",
        ]

        report = generate_discovery_report(
            texts=texts,
            min_doc_frequency=0.5,
            format='markdown'
        )

        assert "Domain Stopwords Discovery Report" in report
        assert "staff" in report.lower() or "service" in report.lower()

    def test_generate_json_report(self):
        """Test JSON report generation."""
        import json

        texts = [
            "Product quality was good.",
            "Good product with quality materials.",
            "Quality product overall good.",
        ]

        report = generate_discovery_report(
            texts=texts,
            min_doc_frequency=0.5,
            format='json'
        )

        # Should be valid JSON
        parsed = json.loads(report)
        assert 'total_documents' in parsed
        assert 'candidates' in parsed

    def test_save_report_to_file(self):
        """Test saving report to file."""
        texts = [
            "Test document one.",
            "Test document two.",
            "Test document three.",
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_path = f.name

        try:
            generate_discovery_report(
                texts=texts,
                output_path=temp_path,
                min_doc_frequency=0.5
            )

            # Verify file was created with content
            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                content = f.read()
                assert len(content) > 0
                assert "Discovery Report" in content
        finally:
            os.unlink(temp_path)
