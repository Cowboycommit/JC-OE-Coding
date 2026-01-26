"""
Shared UI utilities for Streamlit applications.

Provides common functions, patterns, and utilities used across
app.py, app_lite.py, and other Streamlit interfaces to eliminate
code duplication and ensure consistency.

This module contains:
- Session state management utilities
- Prerequisite checking decorators and functions
- Progress mapping and stage tracking
- Error recovery UI components
- Code label extraction utilities
- Stage metadata rendering helpers
- Common visualization helpers
"""

import streamlit as st
from typing import Dict, Any, Optional, List, Callable, Tuple
import functools


# =============================================================================
# CODE LABEL UTILITIES
# =============================================================================

def get_code_label(code_info: Dict[str, Any]) -> str:
    """
    Extract the display label for a code, preferring LLM-generated labels.

    This standardizes the pattern of getting the best available label:
    - First tries 'llm_label' (AI-refined label)
    - Falls back to 'label' (auto-generated from keywords)

    Args:
        code_info: Dictionary containing code metadata with 'label' and
                   optionally 'llm_label' keys

    Returns:
        The best available label string

    Example:
        >>> code_info = {'label': 'work flexibility', 'llm_label': 'Work-Life Balance'}
        >>> get_code_label(code_info)
        'Work-Life Balance'
    """
    return code_info.get('llm_label', code_info.get('label', 'Unknown'))


def get_code_labels_mapping(codebook: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    Create a mapping from code IDs to display labels.

    Args:
        codebook: Dictionary mapping code IDs to code info dicts

    Returns:
        Dictionary mapping code IDs to their display labels

    Example:
        >>> codebook = {
        ...     'CODE_01': {'label': 'flexibility', 'llm_label': 'Work Flexibility'},
        ...     'CODE_02': {'label': 'communication'}
        ... }
        >>> get_code_labels_mapping(codebook)
        {'CODE_01': 'Work Flexibility', 'CODE_02': 'communication'}
    """
    return {
        code_id: get_code_label(info)
        for code_id, info in codebook.items()
    }


# =============================================================================
# PROGRESS MAPPING UTILITIES
# =============================================================================

def map_progress_to_stage(progress: float, num_stages: int = 5) -> int:
    """
    Map a progress value (0-1) to a stage index.

    This standardizes the progress-to-stage mapping logic used during
    analysis execution to show which stage is currently active.

    Args:
        progress: Float value from 0.0 to 1.0 representing completion
        num_stages: Total number of stages (default 5)

    Returns:
        Stage index (0-based) corresponding to the progress

    Example:
        >>> map_progress_to_stage(0.25)  # 25% complete
        0  # Still in first stage
        >>> map_progress_to_stage(0.75)  # 75% complete
        2  # In third stage
    """
    if num_stages == 5:
        # Standard 5-stage mapping used in analysis
        if progress < 0.3:
            return 0  # Data Preparation
        elif progress < 0.5:
            return 1  # Feature Extraction
        elif progress < 0.8:
            return 2  # Clustering/Modeling
        elif progress < 0.95:
            return 3  # Code Labeling
        else:
            return 4  # Generating Insights
    else:
        # Generic mapping for other stage counts
        return min(int(progress * num_stages), num_stages - 1)


def create_adjusted_progress_callback(
    base_update_fn: Callable[[float, str, int], None],
    progress_offset: float = 0.0,
    progress_scale: float = 1.0
) -> Callable[[float, str], None]:
    """
    Create an adjusted progress callback that maps progress to stages.

    This eliminates the duplicate adjusted_progress function definitions
    in app.py by providing a factory function.

    Args:
        base_update_fn: The underlying update function that takes
                        (progress, message, stage_idx)
        progress_offset: Value to add to progress (for multi-phase operations)
        progress_scale: Value to multiply progress by

    Returns:
        A callback function that takes (progress, message) and automatically
        maps to the correct stage

    Example:
        >>> def update_ui(p, msg, stage):
        ...     progress_bar.progress(p)
        ...     status.text(msg)
        >>> callback = create_adjusted_progress_callback(update_ui, 0.4, 0.6)
        >>> callback(0.5, "Processing...")  # Maps to stage 1, progress 0.7
    """
    def adjusted_callback(progress: float, message: str):
        adjusted_progress = progress_offset + (progress * progress_scale)
        stage = map_progress_to_stage(progress)
        base_update_fn(adjusted_progress, message, stage)

    return adjusted_callback


# =============================================================================
# PREREQUISITE CHECKING
# =============================================================================

def check_prerequisite(
    condition: bool,
    warning_message: str,
    return_on_fail: bool = True
) -> bool:
    """
    Check a prerequisite condition and show warning if not met.

    Args:
        condition: Boolean condition that should be True
        warning_message: Message to show if condition is False
        return_on_fail: If True, signals caller should return early

    Returns:
        True if condition is met, False otherwise

    Example:
        >>> if not check_prerequisite(
        ...     st.session_state.uploaded_df is not None,
        ...     "Please upload data first"
        ... ):
        ...     return
    """
    if not condition:
        st.warning(f"⚠️ {warning_message}")
        return False
    return True


def check_data_uploaded() -> bool:
    """Check if data has been uploaded to session state."""
    return check_prerequisite(
        st.session_state.get('uploaded_df') is not None or
        st.session_state.get('raw_df') is not None,
        "Please upload data first"
    )


def check_config_set() -> bool:
    """Check if analysis configuration has been set."""
    return check_prerequisite(
        'config' in st.session_state,
        "Please configure the analysis first"
    )


def check_analysis_complete() -> bool:
    """Check if analysis has been completed."""
    return check_prerequisite(
        st.session_state.get('analysis_complete', False) or
        st.session_state.get('stage_4_complete', False),
        "Please run the analysis first"
    )


# =============================================================================
# ERROR RECOVERY UI COMPONENTS
# =============================================================================

def render_retry_buttons(
    config: Dict[str, Any],
    auto_optimal: bool = False,
    key_prefix: str = ""
) -> None:
    """
    Render standardized retry/recovery buttons for analysis errors.

    This eliminates the duplicate retry button code blocks in app.py.

    Args:
        config: Current analysis configuration dictionary
        auto_optimal: Whether auto-optimization is enabled
        key_prefix: Prefix for button keys to avoid conflicts

    Example:
        >>> render_retry_buttons(st.session_state.config, auto_optimal=False)
    """
    st.markdown("#### Quick Fixes")
    col1, col2, col3 = st.columns(3)

    with col1:
        key = f"{key_prefix}_fewer_codes" if key_prefix else "retry_fewer_codes"
        if st.button("Try with fewer codes", key=key, use_container_width=True):
            if not auto_optimal and config.get('n_codes', 0) > 2:
                st.session_state.config['n_codes'] = max(2, config['n_codes'] - 2)
                st.success(f"Reduced codes to {st.session_state.config['n_codes']}. Click 'Start Analysis' again.")
                st.rerun()
            else:
                st.warning("Cannot reduce codes further or auto-optimization is enabled.")

    with col2:
        key = f"{key_prefix}_lower_conf" if key_prefix else "retry_lower_conf"
        if st.button("Try with lower confidence", key=key, use_container_width=True):
            if config.get('min_confidence', 0) > 0.1:
                st.session_state.config['min_confidence'] = max(0.1, config['min_confidence'] - 0.1)
                st.success(f"Reduced confidence to {st.session_state.config['min_confidence']:.2f}. Click 'Start Analysis' again.")
                st.rerun()
            else:
                st.warning("Confidence already at minimum (0.1).")

    with col3:
        key = f"{key_prefix}_reset" if key_prefix else "retry_reset"
        if st.button("Reset and try again", key=key, use_container_width=True):
            st.session_state.analysis_complete = False
            for k in ['coder', 'results_df', 'metrics', 'viz_data']:
                st.session_state.pop(k, None)
            st.success("Analysis state reset. Click 'Start Analysis' again.")
            st.rerun()


# =============================================================================
# SENTIMENT DISPLAY UTILITIES
# =============================================================================

def format_sentiment_distribution(
    sentiment_dist: Dict[str, int],
    format_type: str = 'html'
) -> str:
    """
    Format sentiment distribution for display.

    Args:
        sentiment_dist: Dictionary with 'positive', 'neutral', 'negative' counts
        format_type: 'html' for HTML display, 'text' for plain text

    Returns:
        Formatted string representation

    Example:
        >>> dist = {'positive': 50, 'neutral': 30, 'negative': 20}
        >>> format_sentiment_distribution(dist, 'text')
        '50 positive, 30 neutral, 20 negative'
    """
    positive = sentiment_dist.get('positive', 0)
    neutral = sentiment_dist.get('neutral', 0)
    negative = sentiment_dist.get('negative', 0)

    if format_type == 'html':
        return f"😊 {positive} positive, 😐 {neutral} neutral, 😞 {negative} negative"
    else:
        return f"{positive} positive, {neutral} neutral, {negative} negative"


def render_sentiment_metrics(
    metrics: Dict[str, Any],
    show_model: bool = True
) -> None:
    """
    Render sentiment analysis metrics in columns.

    Args:
        metrics: Metrics dictionary containing sentiment information
        show_model: Whether to show the model name
    """
    sentiment_dist = metrics.get('sentiment_distribution', {})

    if show_model:
        col1, col2, col3, col4 = st.columns(4)
    else:
        col1, col2, col3 = st.columns(3)
        col4 = None

    with col1:
        st.metric("😊 Positive", sentiment_dist.get('positive', 0))
    with col2:
        st.metric("😐 Neutral", sentiment_dist.get('neutral', 0))
    with col3:
        st.metric("😞 Negative", sentiment_dist.get('negative', 0))

    if col4:
        with col4:
            st.metric("Avg Confidence", f"{metrics.get('sentiment_avg_confidence', 0):.2f}")


# =============================================================================
# STAGE METADATA RENDERING (for app_lite.py)
# =============================================================================

def render_stage_metadata(stage_info: Dict[str, Any]) -> None:
    """
    Render stage metadata block (purpose, module, inputs, outputs, mistakes).

    This eliminates the repetitive stage rendering pattern in app_lite.py.

    Args:
        stage_info: Dictionary containing stage metadata with keys:
                   'purpose', 'module', 'function', 'inputs', 'outputs', 'mistakes'

    Example:
        >>> stage = PIPELINE_STAGES[0]
        >>> render_stage_metadata(stage)
    """
    st.markdown(f"**Purpose**: {stage_info['purpose']}")
    st.markdown(f"**Responsible Module**: `{stage_info['module']}`")
    st.markdown(f"**Functions**: `{stage_info['function']}`")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Required Inputs**:")
        for inp in stage_info.get("inputs", []):
            st.markdown(f"- {inp}")
    with col2:
        st.markdown("**Outputs/Artifacts**:")
        for out in stage_info.get("outputs", []):
            st.markdown(f"- {out}")

    st.markdown("**Common Engineering Mistakes**:")
    for mistake in stage_info.get("mistakes", []):
        st.markdown(f"- {mistake}")


def render_stage_expander(
    stage_info: Dict[str, Any],
    status_badge: str,
    expanded: bool,
    content_fn: Optional[Callable[[], None]] = None
) -> None:
    """
    Render a complete stage expander with metadata and optional content.

    Args:
        stage_info: Stage metadata dictionary
        status_badge: Status string like "[COMPLETE]" or "[READY]"
        expanded: Whether the expander should be expanded
        content_fn: Optional function to render additional content
    """
    with st.expander(
        f"Stage {stage_info['number']}: {stage_info['name']} {status_badge}",
        expanded=expanded
    ):
        render_stage_metadata(stage_info)
        st.markdown("---")
        if content_fn:
            content_fn()


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def render_chart_explanation(
    what_shows: str,
    how_interpret: str,
    key_things: List[str],
    expanded: bool = False
) -> None:
    """
    Render a standardized chart explanation expander.

    This provides consistent "What am I seeing?" explanations across
    visualization tabs.

    Args:
        what_shows: Description of what the chart displays
        how_interpret: Instructions for interpretation
        key_things: List of key things to look for
        expanded: Whether the expander should be expanded by default
    """
    with st.expander("What am I seeing?", expanded=expanded):
        st.markdown(f"**What this shows:** {what_shows}")
        st.markdown(f"**How to interpret:** {how_interpret}")
        st.markdown("**Key things to look for:**")
        for item in key_things:
            st.markdown(f"- {item}")


def render_metrics_row(
    metrics_dict: Dict[str, Tuple[str, Any]],
    num_columns: int = 4
) -> None:
    """
    Render a row of metrics in columns.

    Args:
        metrics_dict: Dictionary mapping metric labels to (format, value) tuples
        num_columns: Number of columns to use

    Example:
        >>> render_metrics_row({
        ...     "Responses": ("", 100),
        ...     "Coverage": ("{:.1f}%", 95.5),
        ...     "Avg Confidence": ("{:.2f}", 0.85)
        ... })
    """
    cols = st.columns(num_columns)
    items = list(metrics_dict.items())

    for i, (label, (fmt, value)) in enumerate(items):
        col_idx = i % num_columns
        with cols[col_idx]:
            if fmt:
                st.metric(label, fmt.format(value))
            else:
                st.metric(label, value)


# =============================================================================
# SESSION STATE UTILITIES
# =============================================================================

def get_session_value(key: str, default: Any = None) -> Any:
    """
    Safely get a value from session state with a default.

    Args:
        key: Session state key
        default: Default value if key doesn't exist

    Returns:
        Value from session state or default
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """
    Set a value in session state.

    Args:
        key: Session state key
        value: Value to set
    """
    st.session_state[key] = value


def clear_session_keys(*keys: str) -> None:
    """
    Clear multiple keys from session state.

    Args:
        *keys: Variable number of keys to clear
    """
    for key in keys:
        st.session_state.pop(key, None)


def reset_downstream_state(from_stage: int, max_stages: int = 7) -> None:
    """
    Reset all stage completion flags and artifacts downstream from a given stage.

    This is a generalized version of reset_downstream_stages from app_lite.py.

    Args:
        from_stage: Stage number to reset from (exclusive)
        max_stages: Maximum number of stages in the pipeline
    """
    # Reset completion flags
    for i in range(from_stage + 1, max_stages + 1):
        st.session_state[f"stage_{i}_complete"] = False

    # Clear artifacts based on stage
    if from_stage < 4:
        clear_session_keys('coder', 'results_df', 'metrics')
    if from_stage < 5:
        clear_session_keys('qa_report')
    if from_stage < 6:
        clear_session_keys('top_codes_df', 'cooccurrence_df', 'viz_data')
    if from_stage < 7:
        clear_session_keys('excel_bytes', 'methods_doc', 'executive_summary')


# =============================================================================
# STAGE STATUS UTILITIES
# =============================================================================

def get_stage_status(stage_num: int) -> str:
    """
    Get the status of a pipeline stage.

    Args:
        stage_num: Stage number (1-based)

    Returns:
        Status string: 'COMPLETE', 'READY', or 'BLOCKED'
    """
    if get_session_value(f"stage_{stage_num}_complete", False):
        return "COMPLETE"

    # Check if prior stages are complete
    for i in range(1, stage_num):
        if not get_session_value(f"stage_{i}_complete", False):
            return "BLOCKED"

    return "READY"


def render_status_badge(status: str) -> str:
    """
    Get a text badge for a status.

    Args:
        status: Status string ('COMPLETE', 'READY', 'BLOCKED', 'RUNNING')

    Returns:
        Badge string like '[COMPLETE]'
    """
    badges = {
        "COMPLETE": "[COMPLETE]",
        "READY": "[READY]",
        "BLOCKED": "[BLOCKED]",
        "RUNNING": "[RUNNING...]",
    }
    return badges.get(status, "[UNKNOWN]")
