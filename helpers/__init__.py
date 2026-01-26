"""
Helper modules for Open-Ended Coding Analysis.

This package provides utility functions for formatting, analysis,
visualization, and UI components in the Streamlit applications.

Modules:
    - formatting: Data formatting and styling utilities
    - analysis: ML analysis orchestration functions
    - ui_utils: Shared UI components and utilities (NEW)
"""

from .formatting import *
from .analysis import *
from .ui_utils import *

__all__ = [
    # Formatting utilities
    'format_dataframe_for_display',
    'format_number',
    'format_percentage',
    'format_confidence_score',
    'create_download_link',
    'highlight_confidence_scores',
    'style_frequency_table',
    # Analysis functions
    'run_ml_analysis',
    'get_analysis_summary',
    'validate_dataframe',
    'preprocess_responses',
    'calculate_metrics_summary',
    'generate_insights',
    # UI utilities (NEW)
    'get_code_label',
    'get_code_labels_mapping',
    'map_progress_to_stage',
    'create_adjusted_progress_callback',
    'check_prerequisite',
    'check_data_uploaded',
    'check_config_set',
    'check_analysis_complete',
    'render_retry_buttons',
    'format_sentiment_distribution',
    'render_sentiment_metrics',
    'render_stage_metadata',
    'render_stage_expander',
    'render_chart_explanation',
    'render_metrics_row',
    'get_session_value',
    'set_session_value',
    'clear_session_keys',
    'reset_downstream_state',
    'get_stage_status',
    'render_status_badge',
]

__version__ = '1.1.0'
