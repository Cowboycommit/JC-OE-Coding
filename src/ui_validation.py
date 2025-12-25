"""
UI & Interaction Validation Agent (Agent-9)
Role: Front-End Robustness & Usability Auditor for Streamlit

This agent validates Streamlit UI components to ensure:
- Safe default states and proper error handling
- Robust interaction flows and state stability
- Clear user feedback and interpretability
- Proper UI-backend contract validation

Author: Agent-9 (UI Validation Specialist)
Version: 1.0
Date: 2025-12-25
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np


class ValidationSeverity(Enum):
    """Severity levels for UI validation issues."""
    CRITICAL = "critical"  # App crash or data loss
    HIGH = "high"          # Functionality broken
    MEDIUM = "medium"      # Poor UX but functional
    LOW = "low"            # Minor cosmetic/usability issue
    INFO = "info"          # Recommendation, not a bug


class ValidationCategory(Enum):
    """Categories of UI validation checks."""
    WIDGET_STATE = "widget_state"
    ERROR_HANDLING = "error_handling"
    STATE_STABILITY = "state_stability"
    UX_INTERPRETABILITY = "ux_interpretability"
    UI_BACKEND_CONTRACT = "ui_backend_contract"


@dataclass
class ValidationIssue:
    """Represents a single UI validation issue."""
    category: ValidationCategory
    severity: ValidationSeverity
    component: str
    description: str
    location: str
    recommendation: str
    code_snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'component': self.component,
            'description': self.description,
            'location': self.location,
            'recommendation': self.recommendation,
            'code_snippet': self.code_snippet
        }


@dataclass
class UIAuditReport:
    """Complete UI validation audit report."""
    issues: List[ValidationIssue] = field(default_factory=list)
    widgets_checked: int = 0
    error_scenarios_tested: int = 0
    state_transitions_verified: int = 0
    ux_checks_performed: int = 0
    contract_validations: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.HIGH)

    @property
    def medium_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.MEDIUM)

    @property
    def low_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.LOW)

    @property
    def is_passing(self) -> bool:
        """Returns True if no critical or high severity issues."""
        return self.critical_count == 0 and self.high_count == 0

    def add_issue(self, issue: ValidationIssue):
        self.issues.append(issue)

    def get_issues_by_category(self, category: ValidationCategory) -> List[ValidationIssue]:
        return [i for i in self.issues if i.category == category]

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == severity]

    def to_dataframe(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=['category', 'severity', 'component',
                                         'description', 'location', 'recommendation'])
        return pd.DataFrame([i.to_dict() for i in self.issues])

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_issues': len(self.issues),
            'critical': self.critical_count,
            'high': self.high_count,
            'medium': self.medium_count,
            'low': self.low_count,
            'info': sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO),
            'is_passing': self.is_passing,
            'widgets_checked': self.widgets_checked,
            'error_scenarios_tested': self.error_scenarios_tested,
            'state_transitions_verified': self.state_transitions_verified,
            'ux_checks_performed': self.ux_checks_performed,
            'contract_validations': self.contract_validations
        }

    def generate_markdown_report(self) -> str:
        """Generate a markdown-formatted audit report."""
        lines = [
            "# UI Validation Audit Report",
            "",
            "## Summary",
            f"- **Total Issues:** {len(self.issues)}",
            f"- **Critical:** {self.critical_count}",
            f"- **High:** {self.high_count}",
            f"- **Medium:** {self.medium_count}",
            f"- **Low:** {self.low_count}",
            f"- **Pass Status:** {'✅ PASS' if self.is_passing else '❌ FAIL'}",
            "",
            "## Checks Performed",
            f"- Widgets Checked: {self.widgets_checked}",
            f"- Error Scenarios Tested: {self.error_scenarios_tested}",
            f"- State Transitions Verified: {self.state_transitions_verified}",
            f"- UX Checks: {self.ux_checks_performed}",
            f"- Contract Validations: {self.contract_validations}",
            ""
        ]

        # Group issues by category
        for category in ValidationCategory:
            category_issues = self.get_issues_by_category(category)
            if category_issues:
                lines.append(f"## {category.value.replace('_', ' ').title()}")
                lines.append("")
                for issue in category_issues:
                    severity_icon = {
                        ValidationSeverity.CRITICAL: "🔴",
                        ValidationSeverity.HIGH: "🟠",
                        ValidationSeverity.MEDIUM: "🟡",
                        ValidationSeverity.LOW: "🔵",
                        ValidationSeverity.INFO: "ℹ️"
                    }.get(issue.severity, "⚪")

                    lines.append(f"### {severity_icon} [{issue.severity.value.upper()}] {issue.component}")
                    lines.append(f"**Location:** `{issue.location}`")
                    lines.append(f"**Description:** {issue.description}")
                    lines.append(f"**Recommendation:** {issue.recommendation}")
                    if issue.code_snippet:
                        lines.append(f"```python\n{issue.code_snippet}\n```")
                    lines.append("")

        return "\n".join(lines)


class UIValidationAgent:
    """
    UI & Interaction Validation Agent for Streamlit applications.

    Responsibilities:
    1. UI State & Interaction Flow validation
    2. Error Handling & User Feedback verification
    3. Rerun & State Stability checks
    4. UX & Interpretability auditing
    5. UI-Backend Contract validation
    """

    def __init__(self):
        self.report = UIAuditReport()

        # Define safe default patterns for common widgets
        self.safe_selectbox_defaults = {
            'text_column': None,  # Should require explicit selection
            'method': 'tfidf_kmeans',  # Safe default
            'stop_words': 'english',  # Safe default
        }

        # Define required validation patterns
        self.required_error_handlers = [
            'empty_dataset',
            'non_numeric_columns',
            'single_variable',
            'missing_values',
            'insufficient_samples'
        ]

    def validate_widget_state(
        self,
        widget_type: str,
        widget_key: str,
        current_value: Any,
        allowed_values: Optional[List[Any]] = None,
        required: bool = False,
        default_value: Any = None
    ) -> Optional[ValidationIssue]:
        """
        Validate a Streamlit widget's state.

        Args:
            widget_type: Type of widget (selectbox, slider, checkbox, etc.)
            widget_key: Unique key/identifier for the widget
            current_value: Current value of the widget
            allowed_values: List of valid values (for selectbox/radio)
            required: Whether a non-default selection is required
            default_value: The default value for this widget

        Returns:
            ValidationIssue if problem found, None otherwise
        """
        self.report.widgets_checked += 1

        # Check for None in required fields
        if required and current_value is None:
            return ValidationIssue(
                category=ValidationCategory.WIDGET_STATE,
                severity=ValidationSeverity.HIGH,
                component=f"{widget_type}:{widget_key}",
                description=f"Required widget '{widget_key}' has no value selected",
                location=f"Widget: {widget_key}",
                recommendation=f"Add validation to require selection before proceeding"
            )

        # Check value against allowed values
        if allowed_values and current_value not in allowed_values:
            return ValidationIssue(
                category=ValidationCategory.WIDGET_STATE,
                severity=ValidationSeverity.CRITICAL,
                component=f"{widget_type}:{widget_key}",
                description=f"Widget '{widget_key}' has invalid value: {current_value}",
                location=f"Widget: {widget_key}",
                recommendation=f"Ensure value is one of: {allowed_values}"
            )

        return None

    def validate_slider_range(
        self,
        slider_key: str,
        current_value: Union[int, float],
        min_value: Union[int, float],
        max_value: Union[int, float],
        logical_min: Optional[Union[int, float]] = None,
        logical_max: Optional[Union[int, float]] = None
    ) -> Optional[ValidationIssue]:
        """
        Validate slider value is within logical bounds.

        Args:
            slider_key: Unique key for the slider
            current_value: Current slider value
            min_value: Slider minimum
            max_value: Slider maximum
            logical_min: Logical minimum (may differ from slider min)
            logical_max: Logical maximum (may differ from slider max)
        """
        self.report.widgets_checked += 1

        # Check basic range
        if current_value < min_value or current_value > max_value:
            return ValidationIssue(
                category=ValidationCategory.WIDGET_STATE,
                severity=ValidationSeverity.CRITICAL,
                component=f"slider:{slider_key}",
                description=f"Slider '{slider_key}' value {current_value} outside range [{min_value}, {max_value}]",
                location=f"Slider: {slider_key}",
                recommendation="Ensure slider bounds are enforced"
            )

        # Check logical bounds
        if logical_min is not None and current_value < logical_min:
            return ValidationIssue(
                category=ValidationCategory.WIDGET_STATE,
                severity=ValidationSeverity.MEDIUM,
                component=f"slider:{slider_key}",
                description=f"Slider '{slider_key}' value {current_value} below logical minimum {logical_min}",
                location=f"Slider: {slider_key}",
                recommendation=f"Consider adjusting slider minimum to {logical_min}"
            )

        if logical_max is not None and current_value > logical_max:
            return ValidationIssue(
                category=ValidationCategory.WIDGET_STATE,
                severity=ValidationSeverity.MEDIUM,
                component=f"slider:{slider_key}",
                description=f"Slider '{slider_key}' value {current_value} above logical maximum {logical_max}",
                location=f"Slider: {slider_key}",
                recommendation=f"Consider adjusting slider maximum to {logical_max}"
            )

        return None

    def validate_error_handling(
        self,
        scenario: str,
        error_caught: bool,
        user_message_shown: bool,
        app_crashed: bool,
        message_actionable: bool = False
    ) -> ValidationIssue:
        """
        Validate error handling for an edge case scenario.

        Args:
            scenario: Description of the error scenario tested
            error_caught: Whether the error was caught by exception handling
            user_message_shown: Whether a user-friendly message was displayed
            app_crashed: Whether the app crashed or showed a traceback
            message_actionable: Whether the message tells user what to do
        """
        self.report.error_scenarios_tested += 1

        if app_crashed:
            return ValidationIssue(
                category=ValidationCategory.ERROR_HANDLING,
                severity=ValidationSeverity.CRITICAL,
                component=f"error_handler:{scenario}",
                description=f"App crashed or showed traceback for scenario: {scenario}",
                location=f"Error scenario: {scenario}",
                recommendation="Add try-except block with user-friendly error message"
            )

        if not error_caught:
            return ValidationIssue(
                category=ValidationCategory.ERROR_HANDLING,
                severity=ValidationSeverity.HIGH,
                component=f"error_handler:{scenario}",
                description=f"Error not caught for scenario: {scenario}",
                location=f"Error scenario: {scenario}",
                recommendation="Add exception handling for this edge case"
            )

        if not user_message_shown:
            return ValidationIssue(
                category=ValidationCategory.ERROR_HANDLING,
                severity=ValidationSeverity.MEDIUM,
                component=f"error_handler:{scenario}",
                description=f"No user message shown for scenario: {scenario}",
                location=f"Error scenario: {scenario}",
                recommendation="Add st.error() or st.warning() with explanation"
            )

        if not message_actionable:
            return ValidationIssue(
                category=ValidationCategory.ERROR_HANDLING,
                severity=ValidationSeverity.LOW,
                component=f"error_handler:{scenario}",
                description=f"Error message not actionable for scenario: {scenario}",
                location=f"Error scenario: {scenario}",
                recommendation="Include specific next steps in error message"
            )

        # All checks passed - return info-level success
        return ValidationIssue(
            category=ValidationCategory.ERROR_HANDLING,
            severity=ValidationSeverity.INFO,
            component=f"error_handler:{scenario}",
            description=f"Error handling working correctly for: {scenario}",
            location=f"Error scenario: {scenario}",
            recommendation="No action needed"
        )

    def validate_state_stability(
        self,
        state_key: str,
        initial_value: Any,
        value_after_rerun: Any,
        expected_persistence: bool = True
    ) -> Optional[ValidationIssue]:
        """
        Validate session state stability across reruns.

        Args:
            state_key: Session state key being validated
            initial_value: Value before rerun/widget change
            value_after_rerun: Value after rerun/widget change
            expected_persistence: Whether value should persist
        """
        self.report.state_transitions_verified += 1

        values_match = self._values_equal(initial_value, value_after_rerun)

        if expected_persistence and not values_match:
            return ValidationIssue(
                category=ValidationCategory.STATE_STABILITY,
                severity=ValidationSeverity.HIGH,
                component=f"session_state:{state_key}",
                description=f"State '{state_key}' unexpectedly changed on rerun",
                location=f"st.session_state.{state_key}",
                recommendation="Use st.session_state initialization pattern or st.cache_data"
            )

        if not expected_persistence and values_match:
            return ValidationIssue(
                category=ValidationCategory.STATE_STABILITY,
                severity=ValidationSeverity.MEDIUM,
                component=f"session_state:{state_key}",
                description=f"State '{state_key}' persisted when it should reset",
                location=f"st.session_state.{state_key}",
                recommendation="Clear state explicitly when context changes"
            )

        return None

    def _values_equal(self, v1: Any, v2: Any) -> bool:
        """Compare two values for equality, handling DataFrames and arrays."""
        # Handle None cases first
        if v1 is None and v2 is None:
            return True
        if v1 is None or v2 is None:
            return False
        # Handle DataFrames
        if isinstance(v1, pd.DataFrame) and isinstance(v2, pd.DataFrame):
            return v1.equals(v2)
        # Handle numpy arrays
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            return np.array_equal(v1, v2)
        # Handle mixed types (one is DataFrame/array, other is not)
        if isinstance(v1, (pd.DataFrame, np.ndarray)) or isinstance(v2, (pd.DataFrame, np.ndarray)):
            return False
        # Standard equality for other types
        try:
            return v1 == v2
        except (ValueError, TypeError):
            return False

    def validate_cache_usage(
        self,
        function_name: str,
        is_cached: bool,
        cache_type: Optional[str] = None,  # 'cache_data' or 'cache_resource'
        has_side_effects: bool = False,
        is_expensive: bool = True
    ) -> Optional[ValidationIssue]:
        """
        Validate caching strategy for functions.

        Args:
            function_name: Name of the function
            is_cached: Whether function uses caching
            cache_type: Type of cache decorator used
            has_side_effects: Whether function has side effects
            is_expensive: Whether function is computationally expensive
        """
        self.report.state_transitions_verified += 1

        if is_expensive and not is_cached:
            return ValidationIssue(
                category=ValidationCategory.STATE_STABILITY,
                severity=ValidationSeverity.MEDIUM,
                component=f"cache:{function_name}",
                description=f"Expensive function '{function_name}' not cached",
                location=f"Function: {function_name}",
                recommendation="Add @st.cache_data or @st.cache_resource decorator"
            )

        if is_cached and has_side_effects and cache_type == 'cache_data':
            return ValidationIssue(
                category=ValidationCategory.STATE_STABILITY,
                severity=ValidationSeverity.HIGH,
                component=f"cache:{function_name}",
                description=f"Function '{function_name}' with side effects uses @st.cache_data",
                location=f"Function: {function_name}",
                recommendation="Use @st.cache_resource for functions with side effects, or remove caching"
            )

        return None

    def validate_ux_interpretability(
        self,
        element_type: str,
        element_id: str,
        has_label: bool,
        label_explains_meaning: bool,
        has_tooltip: bool = False,
        has_warning_for_misuse: bool = False
    ) -> Optional[ValidationIssue]:
        """
        Validate UX and interpretability of UI elements.

        Args:
            element_type: Type of element (chart, metric, control, etc.)
            element_id: Identifier for the element
            has_label: Whether element has a label
            label_explains_meaning: Whether label explains statistical meaning
            has_tooltip: Whether tooltip/help text is present
            has_warning_for_misuse: Whether warnings present for potential misuse
        """
        self.report.ux_checks_performed += 1

        if not has_label:
            return ValidationIssue(
                category=ValidationCategory.UX_INTERPRETABILITY,
                severity=ValidationSeverity.MEDIUM,
                component=f"{element_type}:{element_id}",
                description=f"Element '{element_id}' missing label",
                location=f"UI element: {element_id}",
                recommendation="Add descriptive label to clarify purpose"
            )

        if element_type in ['chart', 'metric', 'statistic'] and not label_explains_meaning:
            return ValidationIssue(
                category=ValidationCategory.UX_INTERPRETABILITY,
                severity=ValidationSeverity.LOW,
                component=f"{element_type}:{element_id}",
                description=f"Label for '{element_id}' doesn't explain statistical meaning",
                location=f"UI element: {element_id}",
                recommendation="Add explanation of what the metric means and how to interpret it"
            )

        if element_type in ['slider', 'selectbox'] and not has_tooltip:
            return ValidationIssue(
                category=ValidationCategory.UX_INTERPRETABILITY,
                severity=ValidationSeverity.INFO,
                component=f"{element_type}:{element_id}",
                description=f"Control '{element_id}' lacks tooltip/help text",
                location=f"UI element: {element_id}",
                recommendation="Add help parameter to explain control purpose"
            )

        return None

    def validate_ui_backend_contract(
        self,
        parameter_name: str,
        ui_value: Any,
        expected_type: type,
        valid_range: Optional[Tuple[Any, Any]] = None,
        backend_validates: bool = False
    ) -> Optional[ValidationIssue]:
        """
        Validate that UI passes correct data types and ranges to backend.

        Args:
            parameter_name: Name of the parameter being passed
            ui_value: Value from UI widget
            expected_type: Expected type for backend
            valid_range: Valid range (min, max) if applicable
            backend_validates: Whether backend defensively validates
        """
        self.report.contract_validations += 1

        # Type check
        if not isinstance(ui_value, expected_type):
            severity = ValidationSeverity.CRITICAL if not backend_validates else ValidationSeverity.MEDIUM
            return ValidationIssue(
                category=ValidationCategory.UI_BACKEND_CONTRACT,
                severity=severity,
                component=f"contract:{parameter_name}",
                description=f"UI passes {type(ui_value).__name__} but backend expects {expected_type.__name__}",
                location=f"Parameter: {parameter_name}",
                recommendation="Add type conversion in UI or validation in backend"
            )

        # Range check
        if valid_range is not None:
            min_val, max_val = valid_range
            if ui_value < min_val or ui_value > max_val:
                severity = ValidationSeverity.HIGH if not backend_validates else ValidationSeverity.MEDIUM
                return ValidationIssue(
                    category=ValidationCategory.UI_BACKEND_CONTRACT,
                    severity=severity,
                    component=f"contract:{parameter_name}",
                    description=f"UI value {ui_value} outside valid range [{min_val}, {max_val}]",
                    location=f"Parameter: {parameter_name}",
                    recommendation="Enforce range in UI slider/input or add backend guard"
                )

        if not backend_validates:
            return ValidationIssue(
                category=ValidationCategory.UI_BACKEND_CONTRACT,
                severity=ValidationSeverity.LOW,
                component=f"contract:{parameter_name}",
                description=f"Backend trusts UI value for '{parameter_name}' without validation",
                location=f"Parameter: {parameter_name}",
                recommendation="Add defensive validation in backend function"
            )

        return None

    def check_widget_combination_validity(
        self,
        widgets: Dict[str, Any],
        invalid_combinations: List[Dict[str, Any]],
        blocked: bool = False
    ) -> Optional[ValidationIssue]:
        """
        Check if current widget combination is valid.

        Args:
            widgets: Dictionary of widget_key -> current_value
            invalid_combinations: List of invalid combination patterns
            blocked: Whether invalid combinations are blocked in UI
        """
        self.report.widgets_checked += 1

        for invalid_combo in invalid_combinations:
            is_match = all(
                widgets.get(key) == value
                for key, value in invalid_combo.items()
            )
            if is_match:
                if not blocked:
                    return ValidationIssue(
                        category=ValidationCategory.WIDGET_STATE,
                        severity=ValidationSeverity.HIGH,
                        component="widget_combination",
                        description=f"Invalid widget combination allowed: {invalid_combo}",
                        location="Widget interaction",
                        recommendation="Add conditional logic to block this combination"
                    )
                else:
                    return ValidationIssue(
                        category=ValidationCategory.WIDGET_STATE,
                        severity=ValidationSeverity.INFO,
                        component="widget_combination",
                        description=f"Invalid combination correctly blocked: {invalid_combo}",
                        location="Widget interaction",
                        recommendation="No action needed"
                    )

        return None

    def audit_dataframe_display(
        self,
        df: pd.DataFrame,
        display_id: str,
        has_pagination: bool = False,
        row_limit: Optional[int] = None,
        handles_empty: bool = False
    ) -> List[ValidationIssue]:
        """
        Audit DataFrame display in UI.

        Args:
            df: DataFrame being displayed
            display_id: Identifier for this display
            has_pagination: Whether pagination is implemented
            row_limit: Maximum rows displayed
            handles_empty: Whether empty DataFrame case is handled
        """
        issues = []
        self.report.ux_checks_performed += 1

        if df is None:
            if not handles_empty:
                issues.append(ValidationIssue(
                    category=ValidationCategory.ERROR_HANDLING,
                    severity=ValidationSeverity.HIGH,
                    component=f"dataframe:{display_id}",
                    description=f"DataFrame display '{display_id}' doesn't handle None case",
                    location=f"DataFrame: {display_id}",
                    recommendation="Add check: if df is not None and not df.empty"
                ))
            return issues

        if df.empty:
            if not handles_empty:
                issues.append(ValidationIssue(
                    category=ValidationCategory.ERROR_HANDLING,
                    severity=ValidationSeverity.MEDIUM,
                    component=f"dataframe:{display_id}",
                    description=f"DataFrame display '{display_id}' shows empty table without message",
                    location=f"DataFrame: {display_id}",
                    recommendation="Show informative message when DataFrame is empty"
                ))
            return issues

        # Large DataFrame without pagination
        if len(df) > 100 and not has_pagination and row_limit is None:
            issues.append(ValidationIssue(
                category=ValidationCategory.UX_INTERPRETABILITY,
                severity=ValidationSeverity.LOW,
                component=f"dataframe:{display_id}",
                description=f"Large DataFrame ({len(df)} rows) displayed without pagination",
                location=f"DataFrame: {display_id}",
                recommendation="Consider pagination or limiting displayed rows"
            ))

        return issues

    def run_full_audit(
        self,
        session_state: Dict[str, Any],
        widget_configs: Dict[str, Dict[str, Any]],
        error_handlers: Dict[str, Callable],
        cache_functions: List[str],
        ui_elements: List[Dict[str, Any]]
    ) -> UIAuditReport:
        """
        Run complete UI validation audit.

        Args:
            session_state: Current session state dictionary
            widget_configs: Configuration for each widget
            error_handlers: Error handling functions to test
            cache_functions: List of cached function names
            ui_elements: List of UI elements with metadata

        Returns:
            Complete UIAuditReport
        """
        self.report = UIAuditReport()  # Reset report

        # 1. Validate widget states
        for widget_key, config in widget_configs.items():
            issue = self.validate_widget_state(
                widget_type=config.get('type', 'unknown'),
                widget_key=widget_key,
                current_value=session_state.get(widget_key),
                allowed_values=config.get('allowed_values'),
                required=config.get('required', False),
                default_value=config.get('default')
            )
            if issue:
                self.report.add_issue(issue)

        # 2. Validate UI elements
        for element in ui_elements:
            issue = self.validate_ux_interpretability(
                element_type=element.get('type', 'unknown'),
                element_id=element.get('id', 'unknown'),
                has_label=element.get('has_label', False),
                label_explains_meaning=element.get('explains_meaning', False),
                has_tooltip=element.get('has_tooltip', False),
                has_warning_for_misuse=element.get('has_warning', False)
            )
            if issue:
                self.report.add_issue(issue)

        return self.report

    def generate_audit_report(
        self,
        session_state: Dict[str, Any],
        widget_configs: Dict[str, Dict[str, Any]],
        error_handlers: Dict[str, Callable],
        cache_functions: List[str],
        ui_elements: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a markdown-formatted UI audit report.

        This method wraps run_full_audit() and returns a formatted markdown report.

        Args:
            session_state: Current session state dictionary
            widget_configs: Configuration for each widget
            error_handlers: Error handling functions to test
            cache_functions: List of cached function names
            ui_elements: List of UI elements with metadata

        Returns:
            Markdown-formatted audit report string
        """
        audit_report = self.run_full_audit(
            session_state=session_state,
            widget_configs=widget_configs,
            error_handlers=error_handlers,
            cache_functions=cache_functions,
            ui_elements=ui_elements
        )
        return audit_report.generate_markdown_report()

    def generate_checklist(self) -> List[Dict[str, Any]]:
        """Generate a UI audit checklist based on best practices."""
        return [
            {
                "category": "Widget State & Interaction",
                "checks": [
                    "All selectboxes have safe default values",
                    "Required selections are enforced before proceeding",
                    "Invalid widget combinations are blocked",
                    "Slider ranges match logical constraints",
                    "File uploaders validate file types",
                ]
            },
            {
                "category": "Error Handling & User Feedback",
                "checks": [
                    "Empty dataset case shows helpful message",
                    "Non-numeric column errors are caught",
                    "Single-variable input case is handled",
                    "Missing values don't cause crashes",
                    "All errors show actionable messages",
                    "No raw tracebacks shown to users",
                ]
            },
            {
                "category": "State Stability",
                "checks": [
                    "Session state persists across reruns",
                    "Widget changes don't lose user data",
                    "File re-uploads are handled correctly",
                    "Method switching preserves relevant state",
                    "Expensive functions are cached appropriately",
                    "No unintended recomputation loops",
                ]
            },
            {
                "category": "UX & Interpretability",
                "checks": [
                    "All charts have clear titles and labels",
                    "Statistical metrics include explanations",
                    "Warnings present for potential misuse",
                    "Controls have tooltips/help text",
                    "No ambiguous or misleading controls",
                    "Screens are not overloaded with information",
                ]
            },
            {
                "category": "UI-Backend Contract",
                "checks": [
                    "UI passes correct data types to backend",
                    "Parameter ranges are enforced",
                    "Backend defensively validates all inputs",
                    "Backend never trusts UI state blindly",
                    "Type conversions are explicit",
                ]
            }
        ]

    def generate_failure_modes(self) -> List[Dict[str, str]]:
        """Generate list of potential failure modes and their fixes."""
        return [
            {
                "mode": "Empty Dataset Upload",
                "symptom": "App crashes or shows confusing error",
                "fix": "Add early validation: if df is None or df.empty: st.error(...); return",
            },
            {
                "mode": "Non-Numeric Column Selected for Numeric Operation",
                "symptom": "TypeError or ValueError in analysis",
                "fix": "Filter column options to only show valid types",
            },
            {
                "mode": "n_codes > n_samples",
                "symptom": "Clustering fails silently or crashes",
                "fix": "Dynamically set slider max based on dataset size",
            },
            {
                "mode": "Missing Text Column Selection",
                "symptom": "KeyError when accessing DataFrame",
                "fix": "Require column selection before enabling analysis",
            },
            {
                "mode": "Session State Lost on Rerun",
                "symptom": "User loses progress, config resets",
                "fix": "Initialize state with: if key not in st.session_state: st.session_state[key] = default",
            },
            {
                "mode": "Large Dataset Causes Timeout",
                "symptom": "App hangs or times out",
                "fix": "Add progress indicators and use st.cache_data for expensive operations",
            },
            {
                "mode": "Invalid Method-Parameter Combination",
                "symptom": "Unexpected behavior or wrong results",
                "fix": "Conditionally show/hide parameters based on selected method",
            },
            {
                "mode": "Division by Zero in Metrics",
                "symptom": "NaN or Inf values in output",
                "fix": "Add guard clauses: if denominator == 0: return 0 or handle appropriately",
            },
        ]

    def generate_widget_redesign_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations for widget redesign."""
        return [
            {
                "widget": "Text Column Selector",
                "current": "All columns shown",
                "recommendation": "Filter to only text/object columns; show preview of first value",
            },
            {
                "widget": "Number of Codes Slider",
                "current": "Fixed range 3-30",
                "recommendation": "Dynamic max based on min(30, n_samples/2); show optimal value indicator",
            },
            {
                "widget": "Confidence Threshold Slider",
                "current": "0.1-0.9 range",
                "recommendation": "Add visual indicator of how many responses would be uncoded at each threshold",
            },
            {
                "widget": "Method Selector",
                "current": "Simple selectbox",
                "recommendation": "Add short description of each method; show recommended method for data size",
            },
            {
                "widget": "File Uploader",
                "current": "Basic upload",
                "recommendation": "Show file size limit; validate immediately on upload; preview first rows",
            },
        ]


def create_ui_validation_agent() -> UIValidationAgent:
    """Factory function to create a UI Validation Agent instance."""
    return UIValidationAgent()


# Convenience functions for quick validation
def quick_validate_widget(
    widget_type: str,
    widget_key: str,
    current_value: Any,
    **kwargs
) -> Optional[ValidationIssue]:
    """Quick validation of a single widget."""
    agent = UIValidationAgent()
    return agent.validate_widget_state(
        widget_type=widget_type,
        widget_key=widget_key,
        current_value=current_value,
        **kwargs
    )


def quick_validate_error_handling(scenario: str, **kwargs) -> ValidationIssue:
    """Quick validation of error handling for a scenario."""
    agent = UIValidationAgent()
    return agent.validate_error_handling(scenario=scenario, **kwargs)


def generate_ui_audit_checklist() -> List[Dict[str, Any]]:
    """Generate a standalone UI audit checklist."""
    agent = UIValidationAgent()
    return agent.generate_checklist()
