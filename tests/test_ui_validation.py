"""
Test suite for UI Validation Agent (Agent-9)

Tests UI widget validation, error handling verification,
state stability checks, UX auditing, and UI-backend contracts.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path for direct import (avoids importing entire src package)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ui_validation import (
    UIValidationAgent,
    ValidationSeverity,
    ValidationCategory,
    ValidationIssue,
    UIAuditReport,
    create_ui_validation_agent,
    quick_validate_widget,
    quick_validate_error_handling,
    generate_ui_audit_checklist,
)


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_issue_creation(self):
        issue = ValidationIssue(
            category=ValidationCategory.WIDGET_STATE,
            severity=ValidationSeverity.HIGH,
            component="selectbox:text_column",
            description="Required field not selected",
            location="Configuration page",
            recommendation="Select a text column"
        )
        assert issue.severity == ValidationSeverity.HIGH
        assert issue.category == ValidationCategory.WIDGET_STATE
        assert "selectbox" in issue.component

    def test_issue_to_dict(self):
        issue = ValidationIssue(
            category=ValidationCategory.ERROR_HANDLING,
            severity=ValidationSeverity.CRITICAL,
            component="error:empty_dataset",
            description="App crashes on empty dataset",
            location="Analysis page",
            recommendation="Add validation",
            code_snippet="if df.empty: st.error('No data')"
        )
        d = issue.to_dict()
        assert d['category'] == 'error_handling'
        assert d['severity'] == 'critical'
        assert d['code_snippet'] is not None


class TestUIAuditReport:
    """Tests for UIAuditReport."""

    def test_empty_report(self):
        report = UIAuditReport()
        assert len(report.issues) == 0
        assert report.is_passing
        assert report.critical_count == 0

    def test_add_issues(self):
        report = UIAuditReport()
        report.add_issue(ValidationIssue(
            category=ValidationCategory.WIDGET_STATE,
            severity=ValidationSeverity.HIGH,
            component="test",
            description="Test issue",
            location="Test",
            recommendation="Fix it"
        ))
        report.add_issue(ValidationIssue(
            category=ValidationCategory.ERROR_HANDLING,
            severity=ValidationSeverity.LOW,
            component="test2",
            description="Minor issue",
            location="Test",
            recommendation="Consider fixing"
        ))
        assert len(report.issues) == 2
        assert report.high_count == 1
        assert report.low_count == 1
        assert not report.is_passing  # Has HIGH issue

    def test_critical_fails_report(self):
        report = UIAuditReport()
        report.add_issue(ValidationIssue(
            category=ValidationCategory.WIDGET_STATE,
            severity=ValidationSeverity.CRITICAL,
            component="test",
            description="Critical issue",
            location="Test",
            recommendation="Fix immediately"
        ))
        assert report.critical_count == 1
        assert not report.is_passing

    def test_low_issues_pass(self):
        report = UIAuditReport()
        report.add_issue(ValidationIssue(
            category=ValidationCategory.UX_INTERPRETABILITY,
            severity=ValidationSeverity.LOW,
            component="test",
            description="Minor issue",
            location="Test",
            recommendation="Nice to fix"
        ))
        report.add_issue(ValidationIssue(
            category=ValidationCategory.UX_INTERPRETABILITY,
            severity=ValidationSeverity.INFO,
            component="test2",
            description="Info only",
            location="Test",
            recommendation="FYI"
        ))
        assert report.is_passing  # LOW and INFO don't fail

    def test_get_issues_by_category(self):
        report = UIAuditReport()
        report.add_issue(ValidationIssue(
            category=ValidationCategory.WIDGET_STATE,
            severity=ValidationSeverity.MEDIUM,
            component="widget1",
            description="Widget issue",
            location="Test",
            recommendation="Fix"
        ))
        report.add_issue(ValidationIssue(
            category=ValidationCategory.ERROR_HANDLING,
            severity=ValidationSeverity.MEDIUM,
            component="error1",
            description="Error issue",
            location="Test",
            recommendation="Fix"
        ))
        report.add_issue(ValidationIssue(
            category=ValidationCategory.WIDGET_STATE,
            severity=ValidationSeverity.LOW,
            component="widget2",
            description="Another widget issue",
            location="Test",
            recommendation="Fix"
        ))

        widget_issues = report.get_issues_by_category(ValidationCategory.WIDGET_STATE)
        assert len(widget_issues) == 2

        error_issues = report.get_issues_by_category(ValidationCategory.ERROR_HANDLING)
        assert len(error_issues) == 1

    def test_to_dataframe(self):
        report = UIAuditReport()
        report.add_issue(ValidationIssue(
            category=ValidationCategory.WIDGET_STATE,
            severity=ValidationSeverity.HIGH,
            component="test",
            description="Test",
            location="Test",
            recommendation="Fix"
        ))
        df = report.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'category' in df.columns
        assert 'severity' in df.columns

    def test_get_summary(self):
        report = UIAuditReport()
        report.widgets_checked = 10
        report.error_scenarios_tested = 5
        report.add_issue(ValidationIssue(
            category=ValidationCategory.WIDGET_STATE,
            severity=ValidationSeverity.MEDIUM,
            component="test",
            description="Test",
            location="Test",
            recommendation="Fix"
        ))

        summary = report.get_summary()
        assert summary['total_issues'] == 1
        assert summary['medium'] == 1
        assert summary['widgets_checked'] == 10
        assert summary['error_scenarios_tested'] == 5
        assert summary['is_passing']

    def test_generate_markdown_report(self):
        report = UIAuditReport()
        report.widgets_checked = 5
        report.add_issue(ValidationIssue(
            category=ValidationCategory.ERROR_HANDLING,
            severity=ValidationSeverity.HIGH,
            component="error:crash",
            description="App crashes on error",
            location="Analysis",
            recommendation="Add try-except"
        ))

        markdown = report.generate_markdown_report()
        assert "# UI Validation Audit Report" in markdown
        assert "High:" in markdown
        assert "error:crash" in markdown
        assert "FAIL" in markdown  # Has HIGH issue


class TestUIValidationAgentWidgets:
    """Tests for widget state validation."""

    @pytest.fixture
    def agent(self):
        return UIValidationAgent()

    def test_validate_required_widget_missing(self, agent):
        issue = agent.validate_widget_state(
            widget_type="selectbox",
            widget_key="text_column",
            current_value=None,
            required=True
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.HIGH
        assert "required" in issue.description.lower()

    def test_validate_required_widget_present(self, agent):
        issue = agent.validate_widget_state(
            widget_type="selectbox",
            widget_key="text_column",
            current_value="response",
            required=True
        )
        assert issue is None

    def test_validate_invalid_value(self, agent):
        issue = agent.validate_widget_state(
            widget_type="selectbox",
            widget_key="method",
            current_value="invalid_method",
            allowed_values=["tfidf_kmeans", "lda", "nmf"]
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.CRITICAL

    def test_validate_valid_value(self, agent):
        issue = agent.validate_widget_state(
            widget_type="selectbox",
            widget_key="method",
            current_value="lda",
            allowed_values=["tfidf_kmeans", "lda", "nmf"]
        )
        assert issue is None

    def test_slider_in_range(self, agent):
        issue = agent.validate_slider_range(
            slider_key="n_codes",
            current_value=10,
            min_value=3,
            max_value=30
        )
        assert issue is None

    def test_slider_out_of_range(self, agent):
        issue = agent.validate_slider_range(
            slider_key="n_codes",
            current_value=50,
            min_value=3,
            max_value=30
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.CRITICAL

    def test_slider_below_logical_min(self, agent):
        issue = agent.validate_slider_range(
            slider_key="n_codes",
            current_value=2,
            min_value=1,
            max_value=30,
            logical_min=3
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.MEDIUM

    def test_widget_combination_invalid(self, agent):
        widgets = {
            'method': 'lda',
            'representation': 'sbert'  # Invalid combination
        }
        invalid_combos = [
            {'method': 'lda', 'representation': 'sbert'}
        ]
        issue = agent.check_widget_combination_validity(
            widgets=widgets,
            invalid_combinations=invalid_combos,
            blocked=False
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.HIGH

    def test_widget_combination_blocked(self, agent):
        widgets = {
            'method': 'lda',
            'representation': 'sbert'
        }
        invalid_combos = [
            {'method': 'lda', 'representation': 'sbert'}
        ]
        issue = agent.check_widget_combination_validity(
            widgets=widgets,
            invalid_combinations=invalid_combos,
            blocked=True
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.INFO  # Correctly blocked

    def test_widget_combination_valid(self, agent):
        widgets = {
            'method': 'tfidf_kmeans',
            'representation': 'tfidf'
        }
        invalid_combos = [
            {'method': 'lda', 'representation': 'sbert'}
        ]
        issue = agent.check_widget_combination_validity(
            widgets=widgets,
            invalid_combinations=invalid_combos,
            blocked=False
        )
        assert issue is None


class TestUIValidationAgentErrorHandling:
    """Tests for error handling validation."""

    @pytest.fixture
    def agent(self):
        return UIValidationAgent()

    def test_app_crash_detected(self, agent):
        issue = agent.validate_error_handling(
            scenario="empty_dataset",
            error_caught=False,
            user_message_shown=False,
            app_crashed=True
        )
        assert issue.severity == ValidationSeverity.CRITICAL

    def test_error_not_caught(self, agent):
        issue = agent.validate_error_handling(
            scenario="non_numeric_column",
            error_caught=False,
            user_message_shown=False,
            app_crashed=False
        )
        assert issue.severity == ValidationSeverity.HIGH

    def test_no_user_message(self, agent):
        issue = agent.validate_error_handling(
            scenario="missing_values",
            error_caught=True,
            user_message_shown=False,
            app_crashed=False
        )
        assert issue.severity == ValidationSeverity.MEDIUM

    def test_non_actionable_message(self, agent):
        issue = agent.validate_error_handling(
            scenario="insufficient_samples",
            error_caught=True,
            user_message_shown=True,
            app_crashed=False,
            message_actionable=False
        )
        assert issue.severity == ValidationSeverity.LOW

    def test_proper_error_handling(self, agent):
        issue = agent.validate_error_handling(
            scenario="single_variable",
            error_caught=True,
            user_message_shown=True,
            app_crashed=False,
            message_actionable=True
        )
        assert issue.severity == ValidationSeverity.INFO


class TestUIValidationAgentStateStability:
    """Tests for state stability validation."""

    @pytest.fixture
    def agent(self):
        return UIValidationAgent()

    def test_state_unexpectedly_changed(self, agent):
        issue = agent.validate_state_stability(
            state_key="uploaded_df",
            initial_value=pd.DataFrame({'a': [1, 2]}),
            value_after_rerun=None,
            expected_persistence=True
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.HIGH

    def test_state_correctly_persisted(self, agent):
        df = pd.DataFrame({'a': [1, 2]})
        issue = agent.validate_state_stability(
            state_key="uploaded_df",
            initial_value=df,
            value_after_rerun=df,
            expected_persistence=True
        )
        assert issue is None

    def test_state_should_reset(self, agent):
        issue = agent.validate_state_stability(
            state_key="analysis_results",
            initial_value={'codes': [1, 2]},
            value_after_rerun={'codes': [1, 2]},
            expected_persistence=False
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.MEDIUM

    def test_cache_expensive_function_not_cached(self, agent):
        issue = agent.validate_cache_usage(
            function_name="run_ml_analysis",
            is_cached=False,
            is_expensive=True
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.MEDIUM

    def test_cache_with_side_effects(self, agent):
        issue = agent.validate_cache_usage(
            function_name="update_session_state",
            is_cached=True,
            cache_type="cache_data",
            has_side_effects=True
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.HIGH


class TestUIValidationAgentUX:
    """Tests for UX and interpretability validation."""

    @pytest.fixture
    def agent(self):
        return UIValidationAgent()

    def test_missing_label(self, agent):
        issue = agent.validate_ux_interpretability(
            element_type="chart",
            element_id="frequency_bar",
            has_label=False,
            label_explains_meaning=False
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.MEDIUM

    def test_label_without_meaning(self, agent):
        issue = agent.validate_ux_interpretability(
            element_type="metric",
            element_id="silhouette_score",
            has_label=True,
            label_explains_meaning=False
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.LOW

    def test_control_without_tooltip(self, agent):
        issue = agent.validate_ux_interpretability(
            element_type="slider",
            element_id="confidence_threshold",
            has_label=True,
            label_explains_meaning=True,
            has_tooltip=False
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.INFO

    def test_proper_ux(self, agent):
        issue = agent.validate_ux_interpretability(
            element_type="chart",
            element_id="frequency_bar",
            has_label=True,
            label_explains_meaning=True,
            has_tooltip=True
        )
        assert issue is None


class TestUIValidationAgentContract:
    """Tests for UI-backend contract validation."""

    @pytest.fixture
    def agent(self):
        return UIValidationAgent()

    def test_wrong_type(self, agent):
        issue = agent.validate_ui_backend_contract(
            parameter_name="n_codes",
            ui_value="10",  # String instead of int
            expected_type=int,
            backend_validates=False
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.CRITICAL

    def test_wrong_type_with_backend_validation(self, agent):
        issue = agent.validate_ui_backend_contract(
            parameter_name="n_codes",
            ui_value="10",
            expected_type=int,
            backend_validates=True
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.MEDIUM

    def test_out_of_range(self, agent):
        issue = agent.validate_ui_backend_contract(
            parameter_name="min_confidence",
            ui_value=1.5,  # > 1.0
            expected_type=float,
            valid_range=(0.0, 1.0),
            backend_validates=False
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.HIGH

    def test_backend_trusts_blindly(self, agent):
        issue = agent.validate_ui_backend_contract(
            parameter_name="text_column",
            ui_value="response",
            expected_type=str,
            backend_validates=False
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.LOW

    def test_proper_contract(self, agent):
        issue = agent.validate_ui_backend_contract(
            parameter_name="n_codes",
            ui_value=10,
            expected_type=int,
            valid_range=(3, 30),
            backend_validates=True
        )
        assert issue is None


class TestUIValidationAgentDataFrame:
    """Tests for DataFrame display validation."""

    @pytest.fixture
    def agent(self):
        return UIValidationAgent()

    def test_null_dataframe_not_handled(self, agent):
        issues = agent.audit_dataframe_display(
            df=None,
            display_id="results_table",
            handles_empty=False
        )
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.HIGH

    def test_empty_dataframe_not_handled(self, agent):
        issues = agent.audit_dataframe_display(
            df=pd.DataFrame(),
            display_id="results_table",
            handles_empty=False
        )
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.MEDIUM

    def test_large_dataframe_no_pagination(self, agent):
        issues = agent.audit_dataframe_display(
            df=pd.DataFrame({'a': range(500)}),
            display_id="results_table",
            has_pagination=False,
            handles_empty=True
        )
        assert len(issues) == 1
        assert issues[0].severity == ValidationSeverity.LOW

    def test_proper_dataframe_handling(self, agent):
        issues = agent.audit_dataframe_display(
            df=pd.DataFrame({'a': range(50)}),
            display_id="results_table",
            has_pagination=False,
            row_limit=100,
            handles_empty=True
        )
        assert len(issues) == 0


class TestUIValidationAgentFullAudit:
    """Tests for full audit functionality."""

    @pytest.fixture
    def agent(self):
        return UIValidationAgent()

    def test_full_audit_basic(self, agent):
        session_state = {
            'text_column': 'response',
            'method': 'tfidf_kmeans'
        }
        widget_configs = {
            'text_column': {
                'type': 'selectbox',
                'required': True,
                'allowed_values': ['response', 'feedback', 'comment']
            },
            'method': {
                'type': 'selectbox',
                'required': False,
                'allowed_values': ['tfidf_kmeans', 'lda', 'nmf']
            }
        }
        ui_elements = [
            {
                'type': 'chart',
                'id': 'frequency_bar',
                'has_label': True,
                'explains_meaning': True
            }
        ]

        report = agent.run_full_audit(
            session_state=session_state,
            widget_configs=widget_configs,
            error_handlers={},
            cache_functions=[],
            ui_elements=ui_elements
        )

        assert report.widgets_checked == 2
        assert report.ux_checks_performed == 1

    def test_full_audit_with_issues(self, agent):
        session_state = {
            'text_column': None,  # Missing required
            'method': 'invalid'   # Invalid value
        }
        widget_configs = {
            'text_column': {
                'type': 'selectbox',
                'required': True
            },
            'method': {
                'type': 'selectbox',
                'allowed_values': ['tfidf_kmeans', 'lda', 'nmf']
            }
        }

        report = agent.run_full_audit(
            session_state=session_state,
            widget_configs=widget_configs,
            error_handlers={},
            cache_functions=[],
            ui_elements=[]
        )

        assert len(report.issues) == 2
        assert not report.is_passing


class TestConvenienceFunctions:
    """Tests for convenience/factory functions."""

    def test_create_agent(self):
        agent = create_ui_validation_agent()
        assert isinstance(agent, UIValidationAgent)

    def test_quick_validate_widget(self):
        issue = quick_validate_widget(
            widget_type="selectbox",
            widget_key="test",
            current_value=None,
            required=True
        )
        assert issue is not None
        assert issue.severity == ValidationSeverity.HIGH

    def test_quick_validate_error_handling(self):
        issue = quick_validate_error_handling(
            scenario="test_error",
            error_caught=True,
            user_message_shown=True,
            app_crashed=False,
            message_actionable=True
        )
        assert issue.severity == ValidationSeverity.INFO

    def test_generate_checklist(self):
        checklist = generate_ui_audit_checklist()
        assert isinstance(checklist, list)
        assert len(checklist) == 5  # 5 categories

        categories = [item['category'] for item in checklist]
        assert "Widget State & Interaction" in categories
        assert "Error Handling & User Feedback" in categories


class TestUIValidationAgentRecommendations:
    """Tests for recommendation generation."""

    @pytest.fixture
    def agent(self):
        return UIValidationAgent()

    def test_generate_failure_modes(self, agent):
        modes = agent.generate_failure_modes()
        assert isinstance(modes, list)
        assert len(modes) > 0

        # Check structure
        for mode in modes:
            assert 'mode' in mode
            assert 'symptom' in mode
            assert 'fix' in mode

        # Check specific failure modes
        mode_names = [m['mode'] for m in modes]
        assert 'Empty Dataset Upload' in mode_names
        assert 'Session State Lost on Rerun' in mode_names

    def test_generate_widget_redesign_recommendations(self, agent):
        recommendations = agent.generate_widget_redesign_recommendations()
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Check structure
        for rec in recommendations:
            assert 'widget' in rec
            assert 'current' in rec
            assert 'recommendation' in rec


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def agent(self):
        return UIValidationAgent()

    def test_numpy_array_comparison(self, agent):
        arr = np.array([1, 2, 3])
        issue = agent.validate_state_stability(
            state_key="embeddings",
            initial_value=arr,
            value_after_rerun=arr.copy(),
            expected_persistence=True
        )
        assert issue is None  # Arrays are equal

    def test_numpy_array_changed(self, agent):
        issue = agent.validate_state_stability(
            state_key="embeddings",
            initial_value=np.array([1, 2, 3]),
            value_after_rerun=np.array([1, 2, 4]),
            expected_persistence=True
        )
        assert issue is not None

    def test_dataframe_comparison(self, agent):
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        issue = agent.validate_state_stability(
            state_key="data",
            initial_value=df1,
            value_after_rerun=df2,
            expected_persistence=True
        )
        assert issue is None  # DataFrames are equal

    def test_empty_widget_configs(self, agent):
        report = agent.run_full_audit(
            session_state={},
            widget_configs={},
            error_handlers={},
            cache_functions=[],
            ui_elements=[]
        )
        assert report.widgets_checked == 0
        assert len(report.issues) == 0

    def test_slider_edge_at_boundary(self, agent):
        # Value exactly at boundary should be valid
        issue = agent.validate_slider_range(
            slider_key="n_codes",
            current_value=3,
            min_value=3,
            max_value=30
        )
        assert issue is None

        issue = agent.validate_slider_range(
            slider_key="n_codes",
            current_value=30,
            min_value=3,
            max_value=30
        )
        assert issue is None
