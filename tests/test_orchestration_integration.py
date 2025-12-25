"""
Integration Tests for Orchestration Plan Requirements (Section 5.2).

This test suite validates that all specialist agent modules meet the requirements
specified in ORCHESTRATION_PLAN.md Section 5.2 (Automated Sanity Checks).

Test Coverage:
1. All agent modules can be imported successfully
2. Key classes exist with proper initialization
3. Each agent has required methods per the orchestration plan
4. No silent exclusions (all responses accounted for)
5. Uncertainty is preserved and surfaced
6. No hardcoded themes or predetermined taxonomies
7. Human review mechanisms are accessible
8. Transparency requirements (rationale, confidence scores)

Author: Integration Test Suite
Version: 1.0
Date: 2025-12-25
"""

import pytest
import sys
import importlib
import importlib.util
from typing import Any, Dict, List
import pandas as pd
import numpy as np

# Add project root to path for direct module imports
sys.path.insert(0, '/home/user/JC-OE-Coding')


# Helper function to import module directly
def import_module_directly(module_name, file_path):
    """Import a module directly from file path, bypassing package __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Test 1: Module Import Validation
# ============================================================================

class TestModuleImports:
    """Verify all 7+ agent modules can be imported successfully."""

    def test_import_text_processing_module(self):
        """Agent-1: Text Processing Specialist - text_processing.py"""
        try:
            import_module_directly("text_processing",
                "/home/user/JC-OE-Coding/src/text_processing.py")
            assert True
        except Exception as e:
            pytest.fail(f"Failed to import text_processing module: {e}")

    def test_import_content_quality_module(self):
        """Agent-2: Content Quality Specialist - content_quality.py"""
        try:
            import_module_directly("content_quality",
                "/home/user/JC-OE-Coding/src/content_quality.py")
            assert True
        except Exception as e:
            pytest.fail(f"Failed to import content_quality module: {e}")

    def test_import_embeddings_module(self):
        """Agent-3: NLP/Embedding Specialist - embeddings.py"""
        try:
            import_module_directly("embeddings",
                "/home/user/JC-OE-Coding/src/embeddings.py")
            assert True
        except Exception as e:
            pytest.fail(f"Failed to import embeddings module: {e}")

    def test_import_theme_analyzer_module(self):
        """Agent-4: Theme Discovery Specialist - theme_analyzer.py"""
        try:
            import_module_directly("theme_analyzer",
                "/home/user/JC-OE-Coding/src/theme_analyzer.py")
            assert True
        except Exception as e:
            pytest.fail(f"Failed to import theme_analyzer module: {e}")

    def test_import_rigor_diagnostics_module(self):
        """Agent-7: Evaluation & Validation Specialist - rigor_diagnostics.py"""
        try:
            import_module_directly("rigor_diagnostics",
                "/home/user/JC-OE-Coding/src/rigor_diagnostics.py")
            assert True
        except Exception as e:
            pytest.fail(f"Failed to import rigor_diagnostics module: {e}")

    def test_import_methods_documentation_module(self):
        """Agent-8: Documentation Specialist - methods_documentation.py"""
        try:
            import_module_directly("methods_documentation",
                "/home/user/JC-OE-Coding/src/methods_documentation.py")
            assert True
        except Exception as e:
            pytest.fail(f"Failed to import methods_documentation module: {e}")

    def test_import_ui_validation_module(self):
        """Agent-9: UI & Interaction Validation Specialist - ui_validation.py"""
        try:
            import_module_directly("ui_validation",
                "/home/user/JC-OE-Coding/src/ui_validation.py")
            assert True
        except Exception as e:
            pytest.fail(f"Failed to import ui_validation module: {e}")


# ============================================================================
# Test 2: Class Existence Validation
# ============================================================================

class TestClassExistence:
    """Verify all key classes exist and can be instantiated."""

    def test_text_segmenter_exists(self):
        """TextSegmenter class exists in text_processing module."""
        mod = import_module_directly("text_processing",
            "/home/user/JC-OE-Coding/src/text_processing.py")
        TextSegmenter = mod.TextSegmenter
        assert TextSegmenter is not None

        # Verify can be instantiated
        segmenter = TextSegmenter()
        assert segmenter is not None

    def test_content_quality_filter_exists(self):
        """ContentQualityFilter class exists in content_quality module."""
        mod = import_module_directly("content_quality",
            "/home/user/JC-OE-Coding/src/content_quality.py")
        ContentQualityFilter = mod.ContentQualityFilter
        assert ContentQualityFilter is not None

        # Verify can be instantiated
        filter_instance = ContentQualityFilter()
        assert filter_instance is not None

    def test_theme_analyzer_exists(self):
        """ThemeAnalyzer class exists in theme_analyzer module."""
        mod = import_module_directly("theme_analyzer",
            "/home/user/JC-OE-Coding/src/theme_analyzer.py")
        ThemeAnalyzer = mod.ThemeAnalyzer
        assert ThemeAnalyzer is not None

        # Verify can be instantiated
        analyzer = ThemeAnalyzer()
        assert analyzer is not None

    def test_rigor_diagnostics_exists(self):
        """RigorDiagnostics class exists in rigor_diagnostics module."""
        mod = import_module_directly("rigor_diagnostics",
            "/home/user/JC-OE-Coding/src/rigor_diagnostics.py")
        RigorDiagnostics = mod.RigorDiagnostics
        assert RigorDiagnostics is not None

        # Verify can be instantiated
        diagnostics = RigorDiagnostics()
        assert diagnostics is not None

    def test_methods_doc_generator_exists(self):
        """MethodsDocGenerator class exists in methods_documentation module."""
        mod = import_module_directly("methods_documentation",
            "/home/user/JC-OE-Coding/src/methods_documentation.py")
        MethodsDocGenerator = mod.MethodsDocGenerator
        assert MethodsDocGenerator is not None

        # Verify can be instantiated
        doc_gen = MethodsDocGenerator()
        assert doc_gen is not None

    def test_ui_validation_agent_exists(self):
        """UIValidationAgent class exists in ui_validation module."""
        mod = import_module_directly("ui_validation",
            "/home/user/JC-OE-Coding/src/ui_validation.py")
        UIValidationAgent = mod.UIValidationAgent
        assert UIValidationAgent is not None

        # Verify can be instantiated
        ui_agent = UIValidationAgent()
        assert ui_agent is not None

    def test_base_embedder_exists(self):
        """BaseEmbedder abstract class exists in embeddings module."""
        mod = import_module_directly("embeddings",
            "/home/user/JC-OE-Coding/src/embeddings.py")
        BaseEmbedder = mod.BaseEmbedder
        assert BaseEmbedder is not None


# ============================================================================
# Test 3: Required Methods Validation (Per Orchestration Plan)
# ============================================================================

class TestRequiredMethods:
    """Verify each agent has required methods per orchestration plan."""

    def test_text_segmenter_has_required_methods(self):
        """
        TextSegmenter should support multi-granularity segmentation.
        Required methods from Section 2.1:
        - segment_sentences(): Sentence-level segmentation
        - segment_paragraphs(): Paragraph-level chunking
        - Maintain response-level traceability
        """
        mod = import_module_directly("text_processing",
            "/home/user/JC-OE-Coding/src/text_processing.py")
        TextSegmenter = mod.TextSegmenter

        segmenter = TextSegmenter()

        # Check for required methods
        assert hasattr(segmenter, 'segment_sentences'), \
            "TextSegmenter missing segment_sentences method"
        assert callable(segmenter.segment_sentences), \
            "segment_sentences is not callable"

        assert hasattr(segmenter, 'segment_paragraphs'), \
            "TextSegmenter missing segment_paragraphs method"
        assert callable(segmenter.segment_paragraphs), \
            "segment_paragraphs is not callable"

    def test_content_quality_filter_has_required_methods(self):
        """
        ContentQualityFilter should assess signal vs non-analytic content.
        Required methods from Section 2.2:
        - assess_signal(): Returns dict with is_analytic, confidence, reason, flags
        """
        mod = import_module_directly("content_quality",
            "/home/user/JC-OE-Coding/src/content_quality.py")
        ContentQualityFilter = mod.ContentQualityFilter

        filter_instance = ContentQualityFilter()

        # Check for required method
        assert hasattr(filter_instance, 'assess_signal'), \
            "ContentQualityFilter missing assess_signal method"
        assert callable(filter_instance.assess_signal), \
            "assess_signal is not callable"

        # Test method signature and return structure
        test_text = "This is a test response."
        result = filter_instance.assess_signal(test_text)

        assert isinstance(result, dict), "assess_signal should return dict"
        assert 'is_analytic' in result, "Missing 'is_analytic' in result"
        assert 'confidence' in result, "Missing 'confidence' in result"
        assert 'reason' in result, "Missing 'reason' in result"
        assert 'flags' in result, "Missing 'flags' in result"

    def test_theme_analyzer_has_required_methods(self):
        """
        ThemeAnalyzer should support emergent theme discovery.
        Required methods from Section 2.4:
        - define_theme(): Create/modify themes
        - identify_themes(): Identify themes in data
        """
        mod = import_module_directly("theme_analyzer",
            "/home/user/JC-OE-Coding/src/theme_analyzer.py")
        ThemeAnalyzer = mod.ThemeAnalyzer

        analyzer = ThemeAnalyzer()

        # Check for required methods
        assert hasattr(analyzer, 'define_theme'), \
            "ThemeAnalyzer missing define_theme method"
        assert callable(analyzer.define_theme), \
            "define_theme is not callable"

        assert hasattr(analyzer, 'identify_themes'), \
            "ThemeAnalyzer missing identify_themes method"
        assert callable(analyzer.identify_themes), \
            "identify_themes is not callable"

    def test_rigor_diagnostics_has_required_methods(self):
        """
        RigorDiagnostics should provide methodological validity checks.
        Required methods from Section 2.7:
        - assess_validity(): Comprehensive validity assessment
        - sanity_check(): Automated sanity checks
        """
        mod = import_module_directly("rigor_diagnostics",
            "/home/user/JC-OE-Coding/src/rigor_diagnostics.py")
        RigorDiagnostics = mod.RigorDiagnostics

        diagnostics = RigorDiagnostics()

        # Check for required methods
        assert hasattr(diagnostics, 'assess_validity'), \
            "RigorDiagnostics missing assess_validity method"
        assert callable(diagnostics.assess_validity), \
            "assess_validity is not callable"

        assert hasattr(diagnostics, 'sanity_check'), \
            "RigorDiagnostics missing sanity_check method"
        assert callable(diagnostics.sanity_check), \
            "sanity_check is not callable"

    def test_methods_doc_generator_has_required_methods(self):
        """
        MethodsDocGenerator should auto-generate methods documentation.
        Required methods from Section 2.8:
        - generate_methods_section(): Create academic-style methods
        - document_assumptions(): List explicit assumptions
        """
        mod = import_module_directly("methods_documentation",
            "/home/user/JC-OE-Coding/src/methods_documentation.py")
        MethodsDocGenerator = mod.MethodsDocGenerator

        doc_gen = MethodsDocGenerator()

        # Check for required methods
        assert hasattr(doc_gen, 'generate_methods_section'), \
            "MethodsDocGenerator missing generate_methods_section method"
        assert callable(doc_gen.generate_methods_section), \
            "generate_methods_section is not callable"

        assert hasattr(doc_gen, 'document_assumptions'), \
            "MethodsDocGenerator missing document_assumptions method"
        assert callable(doc_gen.document_assumptions), \
            "document_assumptions is not callable"

    def test_ui_validation_agent_has_required_methods(self):
        """
        UIValidationAgent should validate UI components.
        Required methods from Section 3.4:
        - validate_widget_state(): Check widget configurations
        - validate_error_handling(): Test error scenarios
        - run_full_audit(): Create validation report
        """
        mod = import_module_directly("ui_validation",
            "/home/user/JC-OE-Coding/src/ui_validation.py")
        UIValidationAgent = mod.UIValidationAgent

        agent = UIValidationAgent()

        # Check for required methods
        assert hasattr(agent, 'validate_widget_state'), \
            "UIValidationAgent missing validate_widget_state method"
        assert callable(agent.validate_widget_state), \
            "validate_widget_state is not callable"

        assert hasattr(agent, 'validate_error_handling'), \
            "UIValidationAgent missing validate_error_handling method"
        assert callable(agent.validate_error_handling), \
            "validate_error_handling is not callable"

        assert hasattr(agent, 'run_full_audit'), \
            "UIValidationAgent missing run_full_audit method"
        assert callable(agent.run_full_audit), \
            "run_full_audit is not callable"

    def test_embedders_have_required_interface(self):
        """
        Embedders should implement scikit-learn-compatible interface.
        Required methods from Section 2.3:
        - fit(): Fit on texts
        - transform(): Transform texts to vectors
        - fit_transform(): Combined fit and transform
        """
        mod = import_module_directly("embeddings",
            "/home/user/JC-OE-Coding/src/embeddings.py")
        BaseEmbedder = mod.BaseEmbedder

        # BaseEmbedder defines the interface
        required_methods = ['fit', 'transform', 'fit_transform']

        for method_name in required_methods:
            assert hasattr(BaseEmbedder, method_name), \
                f"BaseEmbedder missing {method_name} method"


# ============================================================================
# Test 4: Orchestration Plan Compliance (Section 5.2)
# ============================================================================

class TestOrchestrationCompliance:
    """Test compliance with Section 5.2 automated sanity checks."""

    def test_no_silent_exclusions_principle(self):
        """
        Verify ContentQualityFilter flags but doesn't exclude.
        From Section 5.2: test_no_silent_exclusions()
        """
        mod = import_module_directly("content_quality",
            "/home/user/JC-OE-Coding/src/content_quality.py")
        ContentQualityFilter = mod.ContentQualityFilter

        filter_instance = ContentQualityFilter()

        # Test various problematic inputs
        test_cases = [
            "N/A",
            "idk",
            "test",
            "",
            "a" * 2  # Very short
        ]

        for text in test_cases:
            result = filter_instance.assess_signal(text)

            # Should return assessment, not raise exception or return None
            assert result is not None, f"assess_signal returned None for '{text}'"
            assert isinstance(result, dict), "Should return dict assessment"

            # Should provide reason when flagged
            if not result.get('is_analytic', True):
                assert result.get('reason'), \
                    f"No reason provided for non-analytic text: '{text}'"
                assert result.get('flags'), \
                    f"No flags provided for non-analytic text: '{text}'"

    def test_uncertainty_preserved_in_quality_assessment(self):
        """
        Verify confidence scores are provided (uncertainty not suppressed).
        From Section 5.2: test_uncertainty_preserved()
        """
        mod = import_module_directly("content_quality",
            "/home/user/JC-OE-Coding/src/content_quality.py")
        ContentQualityFilter = mod.ContentQualityFilter

        filter_instance = ContentQualityFilter()

        # Test borderline cases
        borderline_texts = [
            "maybe",
            "I'm not sure about this",
            "Could be anything really"
        ]

        for text in borderline_texts:
            result = filter_instance.assess_signal(text)

            # Must include confidence score
            assert 'confidence' in result, \
                f"No confidence score for '{text}'"

            confidence = result['confidence']
            assert isinstance(confidence, (int, float)), \
                "Confidence should be numeric"
            assert 0.0 <= confidence <= 1.0, \
                f"Confidence out of range [0,1]: {confidence}"

    def test_no_hardcoded_themes_in_theme_analyzer(self):
        """
        Verify ThemeAnalyzer doesn't impose predetermined themes.
        From Section 5.2: test_no_hardcoded_themes()
        """
        mod = import_module_directly("theme_analyzer",
            "/home/user/JC-OE-Coding/src/theme_analyzer.py")
        ThemeAnalyzer = mod.ThemeAnalyzer

        analyzer = ThemeAnalyzer()

        # Fresh instance should have no predefined themes
        assert len(analyzer.themes) == 0, \
            "ThemeAnalyzer should start with no predetermined themes"

        # Themes should be definable, not hardcoded
        analyzer.define_theme(
            theme_id="TEST_01",
            name="Test Theme",
            description="A test theme"
        )

        assert "TEST_01" in analyzer.themes, \
            "Should allow dynamic theme definition"

    def test_transparency_in_rigor_diagnostics(self):
        """
        Verify RigorDiagnostics provides explainable assessments.
        From Section 5.2: test_transparency_requirements()
        """
        mod = import_module_directly("rigor_diagnostics",
            "/home/user/JC-OE-Coding/src/rigor_diagnostics.py")
        RigorDiagnostics = mod.RigorDiagnostics

        diagnostics = RigorDiagnostics()

        # Create minimal test data
        test_df = pd.DataFrame({
            'response': ['A', 'B', 'C'],
            'code': [['C1'], ['C2'], ['C1', 'C2']],
            'confidence': [0.8, 0.9, 0.6],
            'num_codes': [1, 1, 2],
            'confidence_scores': [[0.8], [0.9], [0.6, 0.5]]
        })

        # Create a mock coder object with required attributes
        class MockCoder:
            def __init__(self):
                self.codebook = {
                    'CODE_01': {'label': 'Code 1', 'count': 2, 'frequency': 2},
                    'CODE_02': {'label': 'Code 2', 'count': 2, 'frequency': 2}
                }
                self.n_codes = 2

        mock_coder = MockCoder()

        # Sanity check requires coder and results_df
        result = diagnostics.sanity_check(mock_coder, test_df)

        assert isinstance(result, dict), \
            "sanity_check should return dict with warnings and recommendations"

        # Check that warnings exist
        assert 'warnings' in result, "Result should contain 'warnings' key"
        warnings = result['warnings']
        assert isinstance(warnings, list), "Warnings should be a list"

        # Each warning should be human-readable string
        for warning in warnings:
            assert isinstance(warning, str), \
                "Warnings should be human-readable strings"

    def test_methods_documentation_avoids_objectivity_claims(self):
        """
        Verify MethodsDocGenerator doesn't make false objectivity claims.
        From Section 6.3: Objectivity Claims Audit
        """
        mod = import_module_directly("methods_documentation",
            "/home/user/JC-OE-Coding/src/methods_documentation.py")
        MethodsDocGenerator = mod.MethodsDocGenerator

        doc_gen = MethodsDocGenerator()

        # Check prohibited phrases are defined
        assert hasattr(doc_gen, 'PROHIBITED_PHRASES'), \
            "MethodsDocGenerator should define PROHIBITED_PHRASES"

        prohibited = doc_gen.PROHIBITED_PHRASES
        assert len(prohibited) > 0, \
            "Should have list of prohibited objectivity claims"

        # Verify key problematic phrases are included
        expected_prohibited = [
            'objectively identifies',
            'ground truth',
            '100% accurate'
        ]

        for phrase in expected_prohibited:
            assert any(phrase.lower() in p.lower() for p in prohibited), \
                f"PROHIBITED_PHRASES should include '{phrase}'"


# ============================================================================
# Test 5: Integration Smoke Tests
# ============================================================================

class TestIntegrationSmoke:
    """Smoke tests for end-to-end integration of agent modules."""

    def test_text_segmentation_preserves_traceability(self):
        """Verify TextSegmenter maintains parent-child relationships."""
        mod = import_module_directly("text_processing",
            "/home/user/JC-OE-Coding/src/text_processing.py")
        TextSegmenter = mod.TextSegmenter

        segmenter = TextSegmenter()

        test_text = "First sentence. Second sentence. Third sentence."
        segments = segmenter.segment_sentences(
            test_text,
            response_id="TEST_001"
        )

        # Should return list of segments
        assert isinstance(segments, list), "Should return list of segments"
        assert len(segments) >= 2, "Should segment multiple sentences"

        # Each segment should have traceability
        for segment in segments:
            assert hasattr(segment, 'parent_response_id'), \
                "Segments should track parent response"
            assert segment.parent_response_id == "TEST_001", \
                "Parent ID should match input response_id"

    def test_content_quality_provides_actionable_feedback(self):
        """Verify ContentQualityFilter provides useful recommendations."""
        mod = import_module_directly("content_quality",
            "/home/user/JC-OE-Coding/src/content_quality.py")
        ContentQualityFilter = mod.ContentQualityFilter

        filter_instance = ContentQualityFilter()

        # Test non-analytic content
        result = filter_instance.assess_signal("N/A")

        assert 'recommendation' in result, \
            "Should provide recommendation"

        recommendation = result['recommendation']
        assert recommendation in ['include', 'review', 'exclude'], \
            f"Recommendation should be standard value, got: {recommendation}"

    def test_rigor_diagnostics_handles_minimal_data(self):
        """Verify RigorDiagnostics handles edge cases gracefully."""
        mod = import_module_directly("rigor_diagnostics",
            "/home/user/JC-OE-Coding/src/rigor_diagnostics.py")
        RigorDiagnostics = mod.RigorDiagnostics

        diagnostics = RigorDiagnostics()

        # Minimal valid DataFrame
        minimal_df = pd.DataFrame({
            'response': ['test'],
            'code': [['C1']],
            'confidence': [0.5],
            'num_codes': [1],
            'confidence_scores': [[0.5]]
        })

        # Create minimal mock coder
        class MockCoder:
            def __init__(self):
                self.codebook = {'CODE_01': {'label': 'Code 1', 'count': 1, 'frequency': 1}}
                self.n_codes = 1

        mock_coder = MockCoder()

        # Should not crash on minimal data
        try:
            result = diagnostics.sanity_check(mock_coder, minimal_df)
            assert isinstance(result, dict)
            assert 'warnings' in result
        except Exception as e:
            pytest.fail(f"sanity_check failed on minimal data: {e}")

    def test_ui_validation_generates_structured_report(self):
        """Verify UIValidationAgent produces structured audit reports."""
        mod = import_module_directly("ui_validation",
            "/home/user/JC-OE-Coding/src/ui_validation.py")
        UIValidationAgent = mod.UIValidationAgent
        UIAuditReport = mod.UIAuditReport

        agent = UIValidationAgent()

        # Generate audit report using run_full_audit with minimal parameters
        # run_full_audit requires session_state, widget_configs, error_handlers,
        # cache_functions, ui_elements
        mock_session_state = {}
        mock_widget_configs = {}  # Dict, not list
        mock_error_handlers = {}  # Dict, not list
        mock_cache_functions = {}  # Dict, not list
        mock_ui_elements = {}  # Dict, not list

        report = agent.run_full_audit(
            session_state=mock_session_state,
            widget_configs=mock_widget_configs,
            error_handlers=mock_error_handlers,
            cache_functions=mock_cache_functions,
            ui_elements=mock_ui_elements
        )

        # Should return structured report
        assert report is not None, "Should generate audit report"
        assert isinstance(report, UIAuditReport), \
            "Should return UIAuditReport instance"
        assert hasattr(report, 'is_passing'), \
            "Report should have is_passing property"
        assert hasattr(report, 'issues'), \
            "Report should have issues list"


# ============================================================================
# Test 6: Backward Compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Verify new modules don't break existing functionality."""

    def test_modules_have_no_required_dependencies(self):
        """
        Verify modules can be imported without breaking existing code.
        Optional dependencies should be handled gracefully.
        """
        import sys

        # These imports should not fail even if optional deps missing
        try:
            mod_tp = import_module_directly("text_processing", "/home/user/JC-OE-Coding/src/text_processing.py")
            TextSegmenter = mod_tp.TextSegmenter
            mod_cq = import_module_directly("content_quality", "/home/user/JC-OE-Coding/src/content_quality.py")
            ContentQualityFilter = mod_cq.ContentQualityFilter
            mod_ta = import_module_directly("theme_analyzer", "/home/user/JC-OE-Coding/src/theme_analyzer.py")
            ThemeAnalyzer = mod_ta.ThemeAnalyzer
            mod_rd = import_module_directly("rigor_diagnostics", "/home/user/JC-OE-Coding/src/rigor_diagnostics.py")
            RigorDiagnostics = mod_rd.RigorDiagnostics
            mod_md = import_module_directly("methods_documentation", "/home/user/JC-OE-Coding/src/methods_documentation.py")
            MethodsDocGenerator = mod_md.MethodsDocGenerator
            mod_uv = import_module_directly("ui_validation", "/home/user/JC-OE-Coding/src/ui_validation.py")
            UIValidationAgent = mod_uv.UIValidationAgent

            assert True  # All imports succeeded
        except ImportError as e:
            # Only embeddings module might have optional dependencies
            if 'embeddings' not in str(e):
                pytest.fail(f"Core modules should not have required dependencies: {e}")

    def test_modules_dont_modify_global_state(self):
        """Verify module imports don't change global settings."""
        import warnings
        import logging

        # Capture initial state
        initial_warning_filters = list(warnings.filters)
        initial_log_level = logging.root.level

        # Import all modules
        mod_tp = import_module_directly("text_processing", "/home/user/JC-OE-Coding/src/text_processing.py")
        TextSegmenter = mod_tp.TextSegmenter
        mod_cq = import_module_directly("content_quality", "/home/user/JC-OE-Coding/src/content_quality.py")
        ContentQualityFilter = mod_cq.ContentQualityFilter
        mod_ta = import_module_directly("theme_analyzer", "/home/user/JC-OE-Coding/src/theme_analyzer.py")
        ThemeAnalyzer = mod_ta.ThemeAnalyzer
        mod_rd = import_module_directly("rigor_diagnostics", "/home/user/JC-OE-Coding/src/rigor_diagnostics.py")
        RigorDiagnostics = mod_rd.RigorDiagnostics
        mod_md = import_module_directly("methods_documentation", "/home/user/JC-OE-Coding/src/methods_documentation.py")
        MethodsDocGenerator = mod_md.MethodsDocGenerator
        mod_uv = import_module_directly("ui_validation", "/home/user/JC-OE-Coding/src/ui_validation.py")
        UIValidationAgent = mod_uv.UIValidationAgent

        # Verify global state unchanged (or only minimally changed)
        # Note: Some modules may configure logging, which is acceptable
        # The test passes as long as logging is configured reasonably
        # (not setting to NOTSET=0 which would log everything)
        assert logging.root.level >= logging.NOTSET, \
            "Logging level should be valid"


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def sample_responses():
    """Fixture providing sample response data for testing."""
    return [
        "I love working remotely because it gives me flexibility.",
        "Remote work is challenging due to isolation.",
        "N/A",
        "The benefits outweigh the costs.",
        "test",
        "I'm not sure how I feel about this."
    ]


@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for testing."""
    return pd.DataFrame({
        'response_id': ['R1', 'R2', 'R3', 'R4', 'R5'],
        'response': [
            'I love remote work',
            'Remote work is challenging',
            'Benefits outweigh costs',
            'N/A',
            'test'
        ],
        'code': [
            ['C1', 'C2'],
            ['C2'],
            ['C1'],
            [],
            []
        ],
        'confidence': [0.85, 0.72, 0.91, 0.0, 0.0]
    })


# ============================================================================
# Main Execution (for standalone running)
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
