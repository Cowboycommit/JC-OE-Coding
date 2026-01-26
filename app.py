"""
Streamlit UI for ML-Based Open Coding Analysis

A comprehensive web interface for automatic qualitative data analysis
using machine learning algorithms.
"""

import os

# Fix Keras 3 compatibility issue with HuggingFace Transformers
# Must be set BEFORE importing tensorflow or transformers
# Keras 3 is not yet supported by transformers, so we configure TensorFlow
# to use the backwards-compatible tf-keras package instead.
# See: https://github.com/huggingface/transformers/issues/27850
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import re

# Word cloud support (optional but recommended)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# PIL for fallback wordcloud rendering
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Method visualizations (includes semantic wordclouds and PILWordCloud fallback)
try:
    from src.method_visualizations import MethodVisualizer, PILWordCloud
    METHOD_VISUALIZER_AVAILABLE = True
except ImportError:
    METHOD_VISUALIZER_AVAILABLE = False
    MethodVisualizer = None
    PILWordCloud = None

# Import helper modules
from helpers.formatting import (
    format_number,
    format_percentage,
    format_confidence_score,
    format_dataframe_for_display,
    highlight_confidence_scores,
    style_frequency_table,
    truncate_text,
    create_badge,
    format_duration
)
from helpers.analysis import (
    validate_dataframe,
    preprocess_responses,
    run_ml_analysis,
    find_optimal_codes,
    calculate_metrics_summary,
    generate_insights,
    get_analysis_summary,
    get_top_codes,
    get_cooccurrence_pairs,
    export_results_package
)
from itertools import combinations

# Import sentiment analysis module
try:
    from src.sentiment_analysis import (
        get_sentiment_analyzer,
        TwitterSentimentAnalyzer,
        SurveySentimentAnalyzer,
        LongFormSentimentAnalyzer,
        DATA_TYPE_INFO
    )
    SENTIMENT_ANALYSIS_AVAILABLE = True
except ImportError:
    SENTIMENT_ANALYSIS_AVAILABLE = False
    DATA_TYPE_INFO = {}
    import logging
    logging.warning("Sentiment analysis module not available. Install transformers and torch.")

# Import text preprocessing modules
try:
    from src.gold_standard_preprocessing import (
        GoldStandardTextProcessor,
        DataQualityMetrics,
        PreprocessingConfig,
        normalize_for_nlp,
        preprocess_dataframe,
        create_processor_for_dataset
    )
    from src.text_preprocessor import (
        TextPreprocessor,
        DataCleaningPipeline,
        TextPreprocessingError
    )
    TEXT_PREPROCESSOR_AVAILABLE = True
except ImportError:
    TEXT_PREPROCESSOR_AVAILABLE = False
    import logging
    logging.warning("Text preprocessing modules not available.")


# Cache sentiment analyzer to avoid reloading models on every analysis run
# This prevents the hang that occurs when transformer models are loaded from scratch
@st.cache_resource
def get_cached_sentiment_analyzer(data_type: str):
    """
    Get a cached sentiment analyzer instance.

    Uses Streamlit's cache_resource to ensure the model is only loaded once
    per session, preventing long load times on subsequent analyses.
    """
    return get_sentiment_analyzer(data_type)


# ============================================================================
# VISUALIZATION PRE-COMPUTATION SYSTEM
# ============================================================================
# All visualization data is pre-computed once after analysis completes.
# This eliminates re-computation on every tab switch or interaction.
# ============================================================================


def precompute_all_visualizations(coder, results_df):
    """
    Pre-compute ALL visualization data once after analysis.
    Stores everything in session state for instant access.

    This is the key to preventing UI hangs - we compute everything
    upfront instead of on-demand when users switch tabs.
    """
    viz_data = {}

    # 1. Code frequency data (Tab 1)
    top_codes_data = []
    for code_id, info in sorted(coder.codebook.items(), key=lambda x: x[1]['count'], reverse=True)[:15]:
        top_codes_data.append({
            'Code': code_id,
            'Label': info.get('llm_label', info['label']),  # Prefer LLM-generated label
            'Count': info['count'],
            'Avg Confidence': info['avg_confidence'],
            'Keywords': ', '.join(info['keywords'][:5])
        })
    viz_data['top_codes_df'] = pd.DataFrame(top_codes_data)

    # 2. Co-occurrence matrix (Tab 2)
    codes = list(coder.codebook.keys())
    n = len(codes)
    code_to_idx = {code: i for i, code in enumerate(codes)}
    cooccur = np.zeros((n, n))

    for assigned_codes in results_df['assigned_codes']:
        for code1, code2 in combinations(assigned_codes, 2):
            if code1 in code_to_idx and code2 in code_to_idx:
                i, j = code_to_idx[code1], code_to_idx[code2]
                cooccur[i, j] += 1
                cooccur[j, i] += 1
        for code in assigned_codes:
            if code in code_to_idx:
                cooccur[code_to_idx[code], code_to_idx[code]] += 1

    viz_data['cooccurrence_matrix'] = cooccur
    viz_data['cooccurrence_labels'] = [coder.codebook[c].get('llm_label', coder.codebook[c]['label']) for c in codes]  # Prefer LLM labels
    viz_data['cooccurrence_codes'] = codes

    # 3. Co-occurrence pairs
    pairs_df = get_cooccurrence_pairs(results_df, min_count=2)
    viz_data['cooccurrence_pairs'] = pairs_df

    # 4. Distribution stats (Tab 3)
    viz_data['num_codes_data'] = results_df['num_codes'].tolist()
    viz_data['num_codes_stats'] = {
        'mean': float(results_df['num_codes'].mean()),
        'median': float(results_df['num_codes'].median()),
        'max': int(results_df['num_codes'].max())
    }

    # 5. Confidence scores (Tab 4)
    all_confidences = [conf for confs in results_df['confidence_scores'] for conf in confs]
    viz_data['all_confidences'] = all_confidences
    if all_confidences:
        viz_data['confidence_stats'] = {
            'mean': float(np.mean(all_confidences)),
            'median': float(np.median(all_confidences)),
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences))
        }
    else:
        viz_data['confidence_stats'] = None

    # 6. Quotes data (Tab 5) - Pre-build for each code
    text_col = [col for col in results_df.columns if col not in ['assigned_codes', 'confidence_scores', 'num_codes', 'themes']][0]
    viz_data['text_column'] = text_col

    # Pre-build text lookup for fast quote filtering
    viz_data['text_to_row'] = dict(zip(results_df[text_col], results_df.to_dict('records')))

    # Pre-sort codes by frequency for dropdown
    codes_sorted = sorted(
        coder.codebook.keys(),
        key=lambda x: coder.codebook[x]['count'],
        reverse=True
    )
    codes_with_examples = [c for c in codes_sorted if coder.codebook[c]['examples']]
    viz_data['codes_with_examples'] = codes_with_examples

    # Pre-format code options for dropdown (avoid computation on render)
    viz_data['code_options'] = {
        c: f"{coder.codebook[c].get('llm_label', coder.codebook[c]['label'])} ({coder.codebook[c]['count']} occurrences)"  # Prefer LLM labels
        for c in codes_with_examples
    }

    # 7. Word Cloud data (pre-compute word frequencies)
    text_col = viz_data['text_column']
    all_text = ' '.join(results_df[text_col].astype(str).tolist())
    # Clean text for word cloud
    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', all_text.lower())
    cleaned_text = ' '.join(cleaned_text.split())  # Normalize whitespace
    viz_data['wordcloud_text'] = cleaned_text
    # Wordcloud is available if either wordcloud package OR PIL fallback is available
    viz_data['wordcloud_available'] = (WORDCLOUD_AVAILABLE or (PIL_AVAILABLE and PILWordCloud is not None)) and len(cleaned_text) > 0

    # 8. Sunburst chart data (hierarchical code structure)
    sunburst_data = []
    # Add codes with their relationships
    for code_id, info in coder.codebook.items():
        if info['count'] > 0:  # Only include active codes
            sunburst_data.append({
                'id': code_id,
                'label': info.get('llm_label', info['label']),  # Prefer LLM-generated label
                'parent': 'All Codes',
                'value': info['count'],
                'confidence': info['avg_confidence']
            })
    # Add root node
    total_count = sum(item['value'] for item in sunburst_data)
    viz_data['sunburst_data'] = sunburst_data
    viz_data['sunburst_total'] = total_count

    # 9. Scatter plot data (frequency vs confidence) - uses top_codes_df already computed
    # Already have top_codes_df with 'Count' and 'Avg Confidence' columns

    return viz_data


def ensure_viz_data_ready():
    """Ensure visualization data is pre-computed. Call at start of viz page."""
    if 'viz_data' not in st.session_state or st.session_state.viz_data is None:
        if st.session_state.analysis_complete:
            st.session_state.viz_data = precompute_all_visualizations(
                st.session_state.coder,
                st.session_state.results_df
            )
            return True
        return False
    return True


# Page configuration
st.set_page_config(
    page_title="ML Open Coding Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Color Palette:
# Primary: #1f77b4 (Medium blue)
# Background: #ffffff (White)
# Secondary Background: #f0f2f6 (Light gray)
# Text: #262730 (Dark charcoal)
# Font: sans-serif
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #262730;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-family: sans-serif;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #262730;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #262730;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: #262730;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: #ffffff;
        font-weight: 600;
        font-family: sans-serif;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        color: #ffffff;
    }
    /* Ensure text uses the correct color */
    p, li, span {
        color: #262730;
    }
    /* Stat chips styling */
    .stat-chip {
        display: inline-block;
        background: linear-gradient(135deg, #1f77b4 0%, #155a8a 100%);
        color: #ffffff;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        font-family: sans-serif;
    }
    .stat-chip:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    /* Stepper navigation styles */
    .stepper-container {
        padding: 10px 0;
    }
    .stepper-item {
        display: flex;
        align-items: center;
        padding: 8px 5px;
        margin: 4px 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .stepper-item.active {
        background-color: #e8f4f8;
        border-left: 3px solid #1f77b4;
    }
    .stepper-item.disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }
    .stepper-item.completed .step-number {
        background-color: #28a745;
    }
    .step-number {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background-color: #1f77b4;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .step-number.disabled {
        background-color: #ccc;
    }
    .step-label {
        font-size: 14px;
        color: #262730;
    }
    .step-label.disabled {
        color: #999;
    }
    .workflow-caption {
        font-size: 11px;
        color: #666;
        padding: 5px 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 10px;
    }
    /* Validation badge styles */
    .validation-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
    .badge-valid {
        background-color: #d4edda;
        color: #155724;
    }
    .badge-invalid {
        background-color: #f8d7da;
        color: #721c24;
    }
    .badge-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    /* Stage checklist styles */
    .stage-item {
        padding: 6px 0;
        font-size: 14px;
    }
    .stage-pending {
        color: #999;
    }
    .stage-active {
        color: #1f77b4;
        font-weight: 600;
    }
    .stage-complete {
        color: #28a745;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'coder' not in st.session_state:
        st.session_state.coder = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None


def reset_analysis():
    """Reset all analysis data and return to initial state."""
    st.session_state.analysis_complete = False
    st.session_state.coder = None
    st.session_state.results_df = None
    st.session_state.metrics = None
    st.session_state.uploaded_df = None
    st.session_state.viz_data = None  # Clear pre-computed visualization data
    if 'config' in st.session_state:
        del st.session_state.config
    if 'text_column' in st.session_state:
        del st.session_state.text_column


def set_navigation_page(page):
    """Callback to set navigation page."""
    st.session_state.navigation_page = page


def render_next_button(next_page):
    """Render a 'Next Step' button that navigates to the next page."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.button(
            f"➡️ Next: {next_page}",
            use_container_width=True,
            type="primary",
            on_click=set_navigation_page,
            args=(next_page,)
        )


def render_stepper_navigation():
    """Render a numbered stepper navigation with disabled states based on workflow progress."""
    # Define pages with their prerequisites
    pages = [
        {"name": "📤 Data Upload", "step": 1, "requires": None},
        {"name": "🔧 Text Processor", "step": 2, "requires": "data"},
        {"name": "⚙️ Configuration", "step": 3, "requires": "data"},
        {"name": "🚀 Run Analysis", "step": 4, "requires": "config"},
        {"name": "📊 Results Overview", "step": 5, "requires": "analysis"},
        {"name": "📈 Visualizations", "step": 6, "requires": "analysis"},
        {"name": "💾 Export Results", "step": 7, "requires": "analysis"},
        {"name": "ℹ️ About", "step": 8, "requires": None},
    ]

    # Determine which steps are accessible
    has_data = st.session_state.uploaded_df is not None
    has_config = 'config' in st.session_state
    has_analysis = st.session_state.analysis_complete

    # Get current page
    current_page = st.session_state.get('navigation_page', "📤 Data Upload")

    # Workflow caption
    st.markdown('<div class="workflow-caption">📍 Upload → Configure → Run → Review → Export</div>', unsafe_allow_html=True)

    st.markdown('<div class="stepper-container">', unsafe_allow_html=True)

    for page in pages:
        # Determine if step is accessible
        is_accessible = True
        if page["requires"] == "data" and not has_data:
            is_accessible = False
        elif page["requires"] == "config" and not has_config:
            is_accessible = False
        elif page["requires"] == "analysis" and not has_analysis:
            is_accessible = False

        # Determine if step is completed
        is_completed = False
        if page["step"] == 1 and has_data:
            is_completed = True
        elif page["step"] == 2 and has_data:  # Text Processor (optional but accessible)
            is_completed = False  # Optional step, not required for completion
        elif page["step"] == 3 and has_config:
            is_completed = True
        elif page["step"] == 4 and has_analysis:
            is_completed = True

        is_active = current_page == page["name"]

        # Build step HTML
        item_class = "stepper-item"
        if is_active:
            item_class += " active"
        if not is_accessible:
            item_class += " disabled"
        if is_completed:
            item_class += " completed"

        number_class = "step-number"
        if not is_accessible:
            number_class += " disabled"

        label_class = "step-label"
        if not is_accessible:
            label_class += " disabled"

        # Display indicator
        if is_completed and not is_active:
            indicator = "✓"
        else:
            indicator = str(page["step"])

        # Create clickable button for accessible steps
        if is_accessible:
            if st.button(
                f"{indicator}  {page['name']}",
                key=f"nav_{page['step']}",
                use_container_width=True,
                type="secondary" if not is_active else "primary"
            ):
                st.session_state.navigation_page = page["name"]
                st.rerun()
        else:
            # Show disabled step as text
            st.markdown(f"""
            <div class="{item_class}">
                <span class="{number_class}">{indicator}</span>
                <span class="{label_class}">{page['name']}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    return current_page


def get_stage_checklist_html(stages, current_stage_idx):
    """Generate HTML for the analysis stage checklist.

    Args:
        stages: List of stage names
        current_stage_idx: Index of current stage (0-based), -1 for all complete
    """
    html = '<div class="stage-checklist">'
    for i, stage in enumerate(stages):
        if current_stage_idx == -1 or i < current_stage_idx:
            # Completed
            html += f'<div class="stage-item stage-complete">✅ {stage}</div>'
        elif i == current_stage_idx:
            # Active
            html += f'<div class="stage-item stage-active">⏳ {stage}...</div>'
        else:
            # Pending
            html += f'<div class="stage-item stage-pending">⬜ {stage}</div>'
    html += '</div>'
    return html


def render_chart_controls(chart_name, fig, data_df=None, explanation=None):
    """
    Render consistent controls for charts including explanations and download buttons.

    Args:
        chart_name: Name of the chart (used for filenames)
        fig: Plotly figure object
        data_df: Optional pandas DataFrame with underlying data for CSV download
        explanation: Dict with 'what', 'how', and 'look_for' keys for the explainer
    """
    # Add "What am I seeing?" expander if explanation provided
    if explanation:
        with st.expander("ℹ️ What am I seeing?", expanded=False):
            if 'what' in explanation:
                st.markdown(f"**What this shows:** {explanation['what']}")
            if 'how' in explanation:
                st.markdown(f"**How to interpret:** {explanation['how']}")
            if 'look_for' in explanation:
                st.markdown(f"**Key things to look for:** {explanation['look_for']}")

    # Download buttons
    col1, col2 = st.columns(2)

    with col1:
        # Download PNG
        try:
            # Convert plotly figure to PNG bytes
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            st.download_button(
                label="📥 Download PNG",
                data=img_bytes,
                file_name=f"{chart_name.lower().replace(' ', '_')}.png",
                mime="image/png",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"PNG download requires kaleido: pip install kaleido")

    with col2:
        # Download CSV (if data provided)
        if data_df is not None:
            csv_buffer = BytesIO()
            data_df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{chart_name.lower().replace(' ', '_')}_data.csv",
                mime="text/csv",
                use_container_width=True
            )


def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<div class="main-header">📊 ML Open Coding Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Welcome!</strong> This tool uses machine learning to automatically discover themes
    and code qualitative data. Upload your responses and let the algorithms find patterns.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        # Reset button at the top
        if st.button("🔄 Reset Analysis", use_container_width=True, help="Clear all data and start over"):
            reset_analysis()
            st.rerun()

        st.markdown("---")
        st.markdown("### 🎯 Navigation")

        # Use stepper navigation with disabled states
        page = render_stepper_navigation()

        st.markdown("---")
        st.markdown("### 📋 Quick Stats")
        if st.session_state.uploaded_df is not None:
            st.metric("Responses Loaded", f"{len(st.session_state.uploaded_df):,}")

        if st.session_state.analysis_complete:
            st.metric("Analysis Status", "✅ Complete")
            st.metric("Codes Found", st.session_state.metrics.get('n_codes', 0))
        else:
            st.metric("Analysis Status", "⏸️ Pending")


    # Page routing
    if page == "📤 Data Upload":
        page_data_upload()
    elif page == "🔧 Text Processor":
        page_text_processor()
    elif page == "⚙️ Configuration":
        page_configuration()
    elif page == "🚀 Run Analysis":
        page_run_analysis()
    elif page == "📊 Results Overview":
        page_results_overview()
    elif page == "📈 Visualizations":
        page_visualizations()
    elif page == "💾 Export Results":
        page_export_results()
    elif page == "ℹ️ About":
        page_about()


def page_data_upload():
    """Data upload page."""
    st.markdown('<h2 class="sub-header">📂 Load Sample Data</h2>', unsafe_allow_html=True)

    st.markdown("""
    Try the tool with sample data or upload your own qualitative responses.
    """)

    # Sample data option with dropdown
    st.markdown("### 📊 Select a Sample Dataset")

    # Define available sample datasets with metadata
    sample_datasets = {
        # Primary demo datasets - qualitative survey data
        "Healthcare Patient Feedback": {
            "path": "data/Healthcare_Patient_Feedback_300.csv",
            "description": "Patient feedback across hospital departments (Emergency, Cardiology, etc.). Rich qualitative data ideal for thematic analysis and department-based segmentation.",
            "text_column": "response",
            "type": "Healthcare"
        },
        "Market Research Survey": {
            "path": "data/Market_Research_Survey_300.csv",
            "description": "Consumer insights survey with demographic segmentation. Open-ended responses about products, services, and customer experiences.",
            "text_column": "response",
            "type": "Market Research"
        },
        "Psychology Wellbeing Study": {
            "path": "data/Psychology_Wellbeing_Study_300.csv",
            "description": "Wellbeing and mental health study responses. Complex emotional themes around burnout, work-life balance, and personal challenges.",
            "text_column": "response",
            "type": "Psychology"
        },
        # Sentiment analysis benchmark datasets
        "SST-2 (Binary Sentiment)": {
            "path": "data/SST-2 Sentiment Dataset.csv",
            "description": "Stanford Sentiment Treebank - Binary classification. Expert-labeled movie review sentences with positive/negative sentiment labels. Industry standard benchmark.",
            "text_column": "text",
            "type": "Sentiment",
            "labels": "positive, negative"
        },
        "SST-5 (Fine-grained Sentiment)": {
            "path": "data/SST-5 Sentiment Dataset.csv",
            "description": "Stanford Sentiment Treebank - 5-class sentiment. Fine-grained labels from very negative to very positive. Tests nuanced sentiment detection.",
            "text_column": "text",
            "type": "Sentiment",
            "labels": "very negative, negative, neutral, positive, very positive"
        },
        "Twitter Sentiment (SemEval)": {
            "path": "data/SemEval Twitter Sentiment.csv",
            "description": "SemEval Twitter sentiment benchmark. Professionally annotated tweets with 3-class sentiment. Tests short, informal text analysis.",
            "text_column": "text",
            "type": "Sentiment",
            "labels": "positive, neutral, negative"
        },
        "GoEmotions (Multi-label)": {
            "path": "data/GoEmotions Multi-Label.csv",
            "description": "Google's emotion dataset with 27 emotion categories. Multi-label annotations from Reddit comments. Tests fine-grained emotion detection.",
            "text_column": "text",
            "type": "Emotion",
            "labels": "admiration, amusement, anger, joy, sadness, fear, surprise, + 21 more"
        },
        "AG News Classification": {
            "path": "data/AG News Classification.csv",
            "description": "News article classification benchmark. Clean, human-curated news snippets in 4 categories. Tests topic/domain classification.",
            "text_column": "text",
            "type": "Topic",
            "labels": "World, Sports, Business, Sci/Tech"
        },
        "SNIPS Intent Classification": {
            "path": "data/SNIPS Intent Classification.csv",
            "description": "Voice assistant intent dataset. Human-annotated queries across 7 intent categories. Tests command/intent understanding.",
            "text_column": "text",
            "type": "Intent",
            "labels": "PlayMusic, BookRestaurant, GetWeather, + 4 more"
        },
    }

    # Dropdown to select sample dataset
    selected_dataset = st.selectbox(
        "Choose a sample dataset:",
        options=list(sample_datasets.keys()),
        help="Select from pre-loaded sample datasets to explore the tool"
    )

    # Show dataset description
    dataset_info = sample_datasets[selected_dataset]
    st.info(f"**{dataset_info['type']}** · {dataset_info['description']}")
    if "labels" in dataset_info:
        st.caption(f"Labels: {dataset_info['labels']}")

    # Load button
    if st.button("Load Selected Dataset", use_container_width=True):
        try:
            dataset_path = dataset_info["path"]
            df = pd.read_csv(dataset_path)

            st.session_state.uploaded_df = df
            st.success(f"✅ {selected_dataset} loaded successfully! ({len(df)} responses)")
            st.info(f"👉 **Next step:** Go to '⚙️ Configuration' and select **'{dataset_info['text_column']}'** as your text column.")
            st.rerun()
        except FileNotFoundError:
            st.error(f"❌ Dataset file not found: {dataset_path}")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")

    # Upload your data section
    st.markdown("---")
    st.markdown("### 📤 Or Upload Your Data")

    # Template download section
    st.markdown("#### 📥 Download Data Template")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        Need help formatting your data? Download our Excel template with instructions,
        examples, and proper column formatting.
        """)

    with col2:
        # Load template file for download
        template_path = "documentation/input_data_template.xlsx"
        try:
            with open(template_path, "rb") as template_file:
                template_bytes = template_file.read()

            st.download_button(
                label="📥 Download Template",
                data=template_bytes,
                file_name="data_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="Download Excel template with formatting and instructions"
            )
        except FileNotFoundError:
            st.warning("Template file not found. Please contact administrator.")

    st.markdown("")  # Add spacing

    # File requirements info box - ABOVE the uploader
    st.markdown("""
    <div class="info-box">
    <strong>📋 File Requirements:</strong><br>
    • <strong>Accepted formats:</strong> CSV, Excel (.xlsx, .xls)<br>
    • <strong>Required:</strong> At least one text column with responses<br>
    • <strong>Recommended:</strong> 20+ responses for reliable analysis
    </div>
    """, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file with response data"
    )

    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.uploaded_df = df

            # Validate data and show badge
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            has_text_cols = len(text_columns) > 0
            has_enough_rows = len(df) >= 20

            # Show validation badge
            if has_text_cols and has_enough_rows:
                badge_html = '<span class="validation-badge badge-valid">✅ Valid</span>'
            elif has_text_cols and not has_enough_rows:
                badge_html = '<span class="validation-badge badge-warning">⚠️ Low sample size</span>'
            else:
                badge_html = '<span class="validation-badge badge-invalid">❌ No text columns</span>'

            # Show success message with badge
            st.markdown(f"""
            <div class="success-box">
            ✅ <strong>File uploaded successfully!</strong> {badge_html}<br>
            Loaded {len(df):,} rows and {len(df.columns)} columns
            </div>
            """, unsafe_allow_html=True)

            # Inline text column selector - so users don't have to go to Configuration
            if has_text_cols:
                st.markdown("### 📝 Select Text Column")
                st.markdown("Choose the column containing responses to analyze:")

                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_column = st.selectbox(
                        "Text column:",
                        text_columns,
                        index=0,
                        key="upload_text_column",
                        help="Select the column with text responses for coding"
                    )
                    st.session_state.text_column = selected_column

                with col2:
                    # Show sample count for selected column
                    non_null_count = df[selected_column].dropna().shape[0]
                    st.metric("Valid responses", f"{non_null_count:,}")

                # Show sample of selected column
                st.caption(f"**Sample from '{selected_column}':**")
                sample_texts = df[selected_column].dropna().head(3).tolist()
                for i, text in enumerate(sample_texts, 1):
                    st.text(f"{i}. {truncate_text(str(text), 100)}")

                st.success(f"✅ Column '{selected_column}' selected! You can proceed to Configuration or customize settings there.")
            else:
                st.error("❌ No text columns found. Please upload a file with at least one text column.")

            # Display data preview
            st.markdown("### 🔍 Preview Data")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))

            # Show dataframe
            st.dataframe(
                format_dataframe_for_display(df, max_rows=10),
                use_container_width=True,
                height=300
            )

            # Column info - collapsed by default
            with st.expander("📋 Column Information", expanded=False):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null Count': df.isnull().sum().values,
                    'Unique Values': df.nunique().values
                })
                st.dataframe(col_info, use_container_width=True)

            # Data preprocessing options - collapsed by default
            with st.expander("🔧 Advanced Preprocessing Options", expanded=False):
                with st.form("preprocessing_form"):
                    col1, col2 = st.columns(2)

                    with col1:
                        remove_nulls = st.checkbox("Remove null responses", value=True)
                        remove_duplicates = st.checkbox("Remove duplicate responses", value=False)

                    with col2:
                        min_length = st.number_input(
                            "Minimum response length (characters)",
                            min_value=0,
                            value=5,
                            step=1
                        )

                    if st.form_submit_button("Apply Preprocessing", use_container_width=True):
                        text_column = st.session_state.get('text_column', df.columns[0])

                        if text_column in df.columns:
                            processed_df = preprocess_responses(
                                df,
                                text_column,
                                remove_nulls=remove_nulls,
                                remove_duplicates=remove_duplicates,
                                min_length=min_length
                            )

                            st.session_state.uploaded_df = processed_df

                            st.success(f"✅ Preprocessed! Went from {len(df):,} to {len(processed_df):,} responses")
                        else:
                            st.error("Please select a text column first")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

    # Next button - show when data is loaded
    if st.session_state.uploaded_df is not None:
        render_next_button("🔧 Text Processor")


def page_text_processor():
    """
    Text Processor page - comprehensive text preprocessing with quality metrics.

    Features:
    - GoldStandardTextProcessor: Unicode normalization, HTML decoding, contraction expansion,
      URL/mention/hashtag standardization, elongation normalization, punctuation normalization,
      slang expansion, spam detection, duplicate detection, quality filtering
    - TextPreprocessor: NLTK-based stopword removal, lemmatization, domain-specific cleaning,
      long document handling, language detection
    - DataCleaningPipeline: Batch DataFrame processing with dataset type presets
    - DataQualityMetrics: Detailed preprocessing statistics with human-readable reports
    """
    st.markdown('<h2 class="sub-header">🔧 Text Processor</h2>', unsafe_allow_html=True)

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section")
        return

    if not TEXT_PREPROCESSOR_AVAILABLE:
        st.error("❌ Text preprocessing modules not available. Please check your installation.")
        return

    df = st.session_state.uploaded_df.copy()

    st.markdown("""
    <div class="info-box">
    <strong>Text Preprocessing</strong> cleans and normalizes your text data before analysis.<br><br>
    <strong>When to use:</strong> Preprocessing is <em>optional but recommended</em> for:
    <ul style="margin: 5px 0 0 15px;">
    <li><strong>Social media data</strong> — handles slang, hashtags, emojis, URLs</li>
    <li><strong>Noisy survey data</strong> — removes spam, very short responses, duplicates</li>
    <li><strong>Sentiment analysis</strong> — preserves negation words for accuracy</li>
    <li><strong>Topic modeling</strong> — removes domain-specific noise words</li>
    </ul><br>
    <strong>Skip if:</strong> Your data is already clean and well-formatted.
    </div>
    """, unsafe_allow_html=True)

    # Get text columns
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    if not text_columns:
        st.error("❌ No text columns found in the data")
        return

    # Text column selector
    st.markdown("### 📝 Select Text Column")
    selected_column = st.selectbox(
        "Choose the column to preprocess:",
        text_columns,
        index=text_columns.index(st.session_state.get('text_column', text_columns[0]))
            if st.session_state.get('text_column') in text_columns else 0,
        key="preprocess_text_column"
    )

    # Show sample of original text
    st.markdown("#### Original Text Sample")
    sample_texts = df[selected_column].dropna().head(3).tolist()
    for i, text in enumerate(sample_texts, 1):
        st.text(f"{i}. {truncate_text(str(text), 150)}")

    st.markdown("---")

    # ==========================================================================
    # Quick Dataset Presets Section Header
    # ==========================================================================
    st.markdown("### Quick Dataset Presets")
    st.markdown("""
    Select a preset optimized for your data type. Each preset configures preprocessing
    options and sets the appropriate response type for sentiment analysis. Use the tabs
    below to customize your preprocessing workflow.
    """)

    # Verbose explanation of tab options
    st.info("""
    **🎯 Quick Presets** — One-click preprocessing for your data type. Each preset:
    - Configures optimal text normalization settings
    - Sets the appropriate sentiment analysis model
    - Includes negation preservation and domain stopword options
    - Best choice for most users - start here!

    **⚙️ Social Media Options** — Fine-grained control for Twitter/X data (requires Social Media preset):
    - URL, @mention, and #hashtag handling
    - Slang expansion (lol → laughing out loud)
    - Emoji ratio filtering
    - Elongation normalization (soooo → so)

    **🔬 Advanced Processing** — Expert NLTK-based processing for special requirements:
    - Custom stopword removal with negation preservation
    - Lemmatization (running → run) vs stemming options
    - Domain-specific term preservation (medical, legal, technical)
    - Token limits for long documents (truncate or chunk)

    **📊 Quality Report** — After preprocessing, view detailed statistics:
    - Records filtered and why (too short, spam, duplicates)
    - Token count distributions
    - Before/after text comparisons
    - Downloadable quality report
    """)

    # Create tabs for different processing options
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Quick Presets",
        "⚙️ Social Media Options",
        "🔬 Advanced Processing",
        "📊 Quality Report"
    ])

    # ==========================================================================
    # TAB 1: Quick Presets (DataCleaningPipeline)
    # ==========================================================================
    with tab1:

        # Preset options with data_type mapping for sentiment analysis
        preset_options = {
            "survey": {
                "name": "📋 Survey Responses",
                "description": "Optimized for survey feedback, open-ended questions, and form responses",
                "features": ["Contraction expansion", "Spam detection", "Domain stopwords", "Min 3 tokens"],
                "remove_stopwords": True,
                "lemmatize": True,
                "lowercase": True,
                "data_type": "survey",
                "sentiment_model": "VADER (rule-based)"
            },
            "social_media": {
                "name": "🐦 Social Media (Twitter/X)",
                "description": "Optimized for short, informal text with slang and emojis",
                "features": ["Slang expansion", "URL/mention handling", "Hashtag processing", "Emoji tolerance", "Min 2 tokens"],
                "remove_stopwords": True,
                "lemmatize": True,
                "lowercase": True,
                "data_type": "twitter",
                "sentiment_model": "Twitter-RoBERTa (transformer)"
            },
            "reviews": {
                "name": "⭐ Long-form Reviews",
                "description": "For customer reviews with elongated expressions",
                "features": ["Elongation normalization", "Contraction expansion", "Spam detection", "Min 5 tokens"],
                "remove_stopwords": True,
                "lemmatize": True,
                "lowercase": True,
                "data_type": "longform",
                "sentiment_model": "Review-BERT (transformer)"
            },
            "news": {
                "name": "📰 News Articles",
                "description": "Minimal preprocessing for formal, well-written text",
                "features": ["Minimal normalization", "No spam detection", "Min 10 tokens", "Max 2000 tokens"],
                "remove_stopwords": False,
                "lemmatize": False,
                "lowercase": True,
                "data_type": "news",
                "sentiment_model": "Review-BERT (transformer)"
            }
        }

        selected_preset = st.radio(
            "Select a preset:",
            options=list(preset_options.keys()),
            format_func=lambda x: preset_options[x]["name"],
            horizontal=True,
            key="dataset_preset"
        )

        # Show preset details with sentiment model info
        preset = preset_options[selected_preset]
        st.markdown(f"""
        <div class="info-box" style="padding: 15px;">
        <strong>{preset['name']}</strong><br>
        <em>{preset['description']}</em><br><br>
        <strong>Processing Features:</strong>
        <ul style="margin: 5px 0 0 15px;">
        {''.join([f"<li>{f}</li>" for f in preset['features']])}
        </ul>
        <br>
        <strong>Sentiment Model:</strong> {preset['sentiment_model']}
        </div>
        """, unsafe_allow_html=True)

        # Option to drop filtered rows
        preset_drop_filtered = st.checkbox("Drop filtered rows", value=False, key="preset_drop")

        # Enhanced preprocessing options
        with st.expander("🔧 Enhanced Preprocessing Options", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                preset_preserve_negations = st.checkbox("Preserve negation words", value=True,
                    help="Keep 'not', 'no', 'never' etc. for better sentiment/topic analysis",
                    key="preset_preserve_negations")
            with col2:
                preset_domain_stopwords = st.checkbox("Use domain stopwords", value=True,
                    help="Remove survey-specific words like 'response', 'survey', 'participant'",
                    key="preset_domain_stopwords")

        if st.button("🚀 Apply Preset", use_container_width=True, key="apply_preset_btn"):
            with st.spinner(f"Applying {preset['name']} preprocessing..."):
                try:
                    # Create pipeline with selected preset (using preset-specific settings)
                    pipeline = DataCleaningPipeline(
                        dataset_type=selected_preset,
                        remove_stopwords=preset['remove_stopwords'],
                        lemmatize=preset['lemmatize'],
                        lowercase=preset['lowercase'],
                        preserve_negations=preset_preserve_negations,
                        use_domain_stopwords=preset_domain_stopwords
                    )

                    # Process the data
                    processed_df = pipeline.clean_dataframe(
                        df,
                        text_column=selected_column,
                        output_column=f"{selected_column}_processed",
                        drop_filtered=preset_drop_filtered
                    )

                    # Store results
                    st.session_state.uploaded_df = processed_df
                    st.session_state.preprocessing_report = pipeline.get_quality_report()
                    st.session_state.preprocessing_summary = pipeline.get_summary()

                    # Set data_type from preset for sentiment analysis
                    st.session_state.data_type = preset.get('data_type', 'survey')

                    st.success(f"✅ Preprocessing complete! Processed {len(processed_df):,} rows.")
                    st.info(f"📊 Response type set to **{preset['name']}** - Sentiment model: {preset['sentiment_model']}")

                    # Show before/after comparison
                    st.markdown("#### Before/After Comparison")
                    comparison_df = processed_df[[selected_column, f"{selected_column}_processed"]].head(5)
                    comparison_df.columns = ["Original", "Processed"]
                    st.dataframe(comparison_df, use_container_width=True)

                    # Show summary metrics
                    summary = pipeline.get_summary()
                    if summary:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Valid Records", f"{summary.get('valid_records', 0):,}")
                        col2.metric("Avg Token Count", f"{summary.get('avg_token_count', 0):.1f}")
                        col3.metric("Valid Ratio", f"{summary.get('valid_ratio', 0):.1%}")

                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {str(e)}")

    # ==========================================================================
    # TAB 2: Gold Standard Options (Social Media Only)
    # ==========================================================================
    with tab2:
        # Check if social media preset is selected
        current_preset = st.session_state.get('dataset_preset', 'survey')

        if current_preset != 'social_media':
            st.markdown("### Gold Standard Text Processor")
            st.info("""
            **This tab is designed for Social Media data only.**

            The Gold Standard Text Processor provides specialized options for handling
            social media content including:
            - URL and @mention standardization
            - Hashtag processing
            - Slang expansion (lol → laughing out loud)
            - Emoji ratio filtering

            To use these options, please select **🐦 Social Media (Twitter/X)**
            in the Quick Presets tab first.
            """)
        else:
            st.markdown("### Gold Standard Text Processor")
            st.markdown("""
            Configure individual preprocessing steps for fine-grained control.
            The Gold Standard processor implements industry best practices for text normalization.
            """)

            # Normalization options
            st.markdown("#### Normalization Options")
            col1, col2 = st.columns(2)

            with col1:
                gs_unicode = st.checkbox("Unicode normalization (NFKC)", value=True,
                    help="Normalize Unicode characters to standard form", key="gs_unicode")
                gs_html = st.checkbox("Decode HTML entities", value=True,
                    help="Convert &amp; to &, &lt; to <, etc.", key="gs_html")
                gs_contractions = st.checkbox("Expand contractions", value=True,
                    help="Convert don't to do not, I'm to I am, etc.", key="gs_contractions")
                gs_elongation = st.checkbox("Normalize elongations", value=True,
                    help="Convert loooove to loove, etc.", key="gs_elongation")
                gs_punctuation = st.checkbox("Normalize punctuation", value=True,
                    help="Convert !!! to !, etc.", key="gs_punctuation")

            with col2:
                gs_urls = st.checkbox("Standardize URLs", value=True,
                    help="Replace URLs with <URL> token", key="gs_urls")
                gs_mentions = st.checkbox("Standardize @mentions", value=True,
                    help="Replace @user with <USER> token", key="gs_mentions")
                gs_hashtags = st.checkbox("Process hashtags", value=True,
                    help="Remove # from hashtags", key="gs_hashtags")
                gs_slang = st.checkbox("Expand slang", value=False,
                    help="Convert lol to laughing out loud, brb to be right back, etc.", key="gs_slang")

            # Quality filtering options
            st.markdown("#### Quality Filtering")
            col1, col2, col3 = st.columns(3)

            with col1:
                gs_min_tokens = st.number_input("Min tokens", min_value=1, max_value=50, value=3,
                    help="Minimum number of tokens required", key="gs_min_tokens")
                gs_max_tokens = st.number_input("Max tokens", min_value=50, max_value=5000, value=512,
                    help="Maximum number of tokens allowed", key="gs_max_tokens")

            with col2:
                gs_max_emoji_ratio = st.slider("Max emoji ratio", min_value=0.0, max_value=1.0, value=0.7,
                    help="Maximum ratio of emojis to characters", key="gs_emoji_ratio")
                gs_max_char_repeat = st.number_input("Max char repeat", min_value=1, max_value=5, value=2,
                    help="Maximum character repetitions (e.g., 2 keeps 'oo' in 'loooove')", key="gs_char_repeat")

            with col3:
                gs_spam = st.checkbox("Detect spam", value=True,
                    help="Filter out spam-like text patterns", key="gs_spam")
                gs_duplicates = st.checkbox("Detect duplicates", value=True,
                    help="Filter out duplicate texts using MD5 hash", key="gs_duplicates")

            # Custom tokens
            with st.expander("🏷️ Custom Replacement Tokens"):
                col1, col2 = st.columns(2)
                with col1:
                    gs_url_token = st.text_input("URL replacement token", value="<URL>", key="gs_url_token")
                with col2:
                    gs_user_token = st.text_input("User replacement token", value="<USER>", key="gs_user_token")

            if st.button("🚀 Apply Gold Standard Processing", use_container_width=True, key="apply_gs_btn"):
                with st.spinner("Applying Gold Standard preprocessing..."):
                    try:
                        # Create config
                        config = PreprocessingConfig(
                            normalize_unicode=gs_unicode,
                            decode_html_entities=gs_html,
                            expand_contractions=gs_contractions,
                            normalize_elongations=gs_elongation,
                            normalize_punctuation=gs_punctuation,
                            standardize_urls=gs_urls,
                            standardize_mentions=gs_mentions,
                            process_hashtags=gs_hashtags,
                            expand_slang=gs_slang,
                            min_tokens=gs_min_tokens,
                            max_tokens=gs_max_tokens,
                            max_emoji_ratio=gs_max_emoji_ratio,
                            max_char_repeat=gs_max_char_repeat,
                            detect_spam=gs_spam,
                            detect_duplicates=gs_duplicates,
                            url_token=gs_url_token,
                            user_token=gs_user_token
                        )

                        # Create processor
                        processor = GoldStandardTextProcessor(config=config)

                        # Process data
                        processed_df, metrics = preprocess_dataframe(
                            df,
                            text_column=selected_column,
                            output_column=f"{selected_column}_processed",
                            processor=processor,
                            drop_filtered=False
                        )

                        # Store results
                        st.session_state.uploaded_df = processed_df
                        st.session_state.preprocessing_metrics = metrics
                        st.session_state.preprocessing_report = metrics.generate_report()

                        st.success(f"✅ Gold Standard preprocessing complete!")

                        # Show metrics
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Records", f"{metrics.total_records:,}")
                        col2.metric("Valid Records", f"{metrics.valid_records:,}")
                        col3.metric("Filtered", f"{metrics.filtered_records:,}")
                        col4.metric("Valid Ratio", f"{metrics.valid_ratio:.1%}")

                        # Show normalization counts
                        st.markdown("#### Normalization Statistics")
                        norm_data = {
                            "Operation": ["Unicode", "HTML", "URLs", "Mentions", "Hashtags",
                                         "Contractions", "Elongations", "Punctuation", "Slang"],
                            "Count": [metrics.unicode_normalized, metrics.html_decoded,
                                     metrics.urls_replaced, metrics.mentions_replaced,
                                     metrics.hashtags_processed, metrics.contractions_expanded,
                                     metrics.elongations_normalized, metrics.punctuation_normalized,
                                     metrics.slang_expanded]
                        }
                        st.dataframe(pd.DataFrame(norm_data), use_container_width=True)

                        # Show before/after comparison
                        st.markdown("#### Before/After Comparison")
                        comparison_df = processed_df[[selected_column, f"{selected_column}_processed"]].dropna().head(5)
                        comparison_df.columns = ["Original", "Processed"]
                        st.dataframe(comparison_df, use_container_width=True)

                    except Exception as e:
                        st.error(f"❌ Preprocessing failed: {str(e)}")

    # ==========================================================================
    # TAB 3: Advanced Processing (TextPreprocessor)
    # ==========================================================================
    with tab3:
        st.markdown("### Advanced Text Processing")
        st.markdown("""
        Additional processing options using NLTK including stopword removal,
        lemmatization, domain-specific cleaning, and long document handling.
        """)

        # NLTK options
        st.markdown("#### NLTK Processing")
        col1, col2 = st.columns(2)

        with col1:
            adv_stopwords = st.checkbox("Remove stopwords", value=True,
                help="Remove common English stopwords", key="adv_stopwords")
            adv_lemmatize = st.checkbox("Lemmatize", value=True,
                help="Reduce words to their base form (running -> run)", key="adv_lemmatize")
            adv_lowercase = st.checkbox("Convert to lowercase", value=True, key="adv_lowercase")

        with col2:
            adv_min_token_len = st.number_input("Min token length", min_value=1, max_value=10, value=2,
                help="Minimum character length for tokens", key="adv_min_token_len")

        # Enhanced preprocessing options
        st.markdown("#### Enhanced Preprocessing")
        col1, col2 = st.columns(2)

        with col1:
            adv_preserve_negations = st.checkbox("Preserve negation words", value=True,
                help="Keep words like 'not', 'no', 'never' that are critical for sentiment and topic coherence",
                key="adv_preserve_negations")

        with col2:
            adv_domain_stopwords = st.checkbox("Use domain stopwords", value=True,
                help="Remove survey-specific words like 'response', 'survey', 'participant' that add noise",
                key="adv_domain_stopwords")

        # Token count filtering
        st.markdown("#### Token Count Filtering")
        col1, col2 = st.columns(2)
        with col1:
            adv_min_tokens = st.number_input("Min tokens per response", min_value=1, max_value=50, value=3,
                help="Responses with fewer tokens will be filtered out", key="adv_min_tokens")
        with col2:
            adv_max_tokens = st.number_input("Max tokens per response", min_value=50, max_value=5000, value=512,
                help="Responses exceeding this limit will be filtered out", key="adv_max_tokens")

        # Domain-specific cleaning
        st.markdown("#### Domain-Specific Cleaning")
        domain_options = {
            "general": "General (no special handling)",
            "medical": "Medical (preserve mg, ml, patient, diagnosis, etc.)",
            "legal": "Legal (preserve plaintiff, defendant, pursuant, etc.)",
            "technical": "Technical (preserve api, sdk, cpu, gpu, etc.)"
        }

        adv_domain = st.selectbox(
            "Select domain:",
            options=list(domain_options.keys()),
            format_func=lambda x: domain_options[x],
            key="adv_domain"
        )

        # Preserve custom terms
        adv_preserve_terms = st.text_input(
            "Additional terms to preserve (comma-separated):",
            placeholder="e.g., COVID-19, AI, ML",
            key="adv_preserve_terms"
        )

        # Long document handling
        st.markdown("#### Long Document Handling")
        adv_long_doc_strategy = st.selectbox(
            "Strategy for long documents:",
            options=["none", "truncate", "chunk"],
            format_func=lambda x: {
                "none": "None (process as-is)",
                "truncate": "Truncate to max tokens",
                "chunk": "Split into chunks"
            }[x],
            key="adv_long_doc"
        )

        if adv_long_doc_strategy in ["truncate", "chunk"]:
            col1, col2 = st.columns(2)
            with col1:
                adv_chunk_size = st.number_input("Chunk/max size (tokens)", min_value=50, max_value=2000, value=512,
                    key="adv_chunk_size")
            with col2:
                if adv_long_doc_strategy == "chunk":
                    adv_chunk_overlap = st.number_input("Chunk overlap (tokens)", min_value=0, max_value=200, value=50,
                        key="adv_chunk_overlap")

        if st.button("🚀 Apply Advanced Processing", use_container_width=True, key="apply_adv_btn"):
            with st.spinner("Applying advanced preprocessing..."):
                try:
                    # Create TextPreprocessor with token limits and enhanced options
                    preprocessor = TextPreprocessor(
                        use_gold_standard=True,
                        expand_slang=False,
                        detect_spam=True,
                        detect_duplicates=True,
                        min_tokens=adv_min_tokens,
                        max_tokens=adv_max_tokens,
                        preserve_negations=adv_preserve_negations,
                        use_domain_stopwords=adv_domain_stopwords
                    )

                    # Parse preserve terms
                    preserve_terms = None
                    if adv_preserve_terms.strip():
                        preserve_terms = [t.strip() for t in adv_preserve_terms.split(",")]

                    # Process each text
                    processed_texts = []
                    for text in df[selected_column]:
                        if pd.isna(text):
                            processed_texts.append(None)
                            continue

                        text = str(text)

                        # Handle long documents first if needed
                        if adv_long_doc_strategy == "truncate":
                            text = preprocessor.handle_long_documents(
                                text, strategy="truncate", chunk_size=adv_chunk_size
                            )
                        elif adv_long_doc_strategy == "chunk":
                            chunks = preprocessor.handle_long_documents(
                                text, strategy="chunk",
                                chunk_size=adv_chunk_size,
                                chunk_overlap=adv_chunk_overlap
                            )
                            text = chunks[0] if isinstance(chunks, list) and chunks else text

                        # Apply domain-specific cleaning if not general
                        if adv_domain != "general":
                            text = preprocessor.clean_domain_specific(
                                text, domain=adv_domain, preserve_terms=preserve_terms
                            )

                        # Apply main preprocessing
                        try:
                            result = preprocessor.preprocess(
                                text,
                                remove_stopwords=adv_stopwords,
                                lemmatize=adv_lemmatize,
                                lowercase=adv_lowercase,
                                min_token_length=adv_min_token_len,
                                track_metrics=True
                            )
                            processed_texts.append(result)
                        except TextPreprocessingError:
                            processed_texts.append(None)

                    # Update dataframe
                    processed_df = df.copy()
                    processed_df[f"{selected_column}_processed"] = processed_texts

                    # Store results
                    st.session_state.uploaded_df = processed_df
                    st.session_state.preprocessing_report = preprocessor.get_quality_report()

                    st.success(f"✅ Advanced preprocessing complete!")

                    # Show quality metrics
                    metrics = preprocessor.get_quality_metrics()
                    if metrics:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Records", f"{metrics.total_records:,}")
                        col2.metric("Valid Records", f"{metrics.valid_records:,}")
                        col3.metric("Avg Token Count", f"{metrics.avg_token_count:.1f}")

                    # Show before/after comparison
                    st.markdown("#### Before/After Comparison")
                    comparison_df = processed_df[[selected_column, f"{selected_column}_processed"]].dropna().head(5)
                    comparison_df.columns = ["Original", "Processed"]
                    st.dataframe(comparison_df, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Preprocessing failed: {str(e)}")

    # ==========================================================================
    # TAB 4: Quality Report
    # ==========================================================================
    with tab4:
        st.markdown("### Data Quality Report")

        if 'preprocessing_report' in st.session_state and st.session_state.preprocessing_report:
            # Display the report
            st.code(st.session_state.preprocessing_report, language=None)

            # Download button
            st.download_button(
                "📥 Download Quality Report",
                data=st.session_state.preprocessing_report,
                file_name="data_quality_report.txt",
                mime="text/plain",
                use_container_width=True
            )

            # Show metrics visualization if available
            if 'preprocessing_metrics' in st.session_state:
                metrics = st.session_state.preprocessing_metrics

                st.markdown("---")
                st.markdown("#### Visual Summary")

                # Filter reasons chart
                if metrics.filter_reasons:
                    st.markdown("##### Filter Reasons")
                    filter_df = pd.DataFrame([
                        {"Reason": k, "Count": v}
                        for k, v in sorted(metrics.filter_reasons.items(), key=lambda x: -x[1])
                    ])
                    fig = px.bar(filter_df, x="Reason", y="Count", title="Records Filtered by Reason")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                # Text length distribution
                if metrics.text_lengths:
                    st.markdown("##### Text Length Distribution")
                    fig = px.histogram(x=metrics.text_lengths, nbins=30,
                                      title="Distribution of Text Lengths (characters)",
                                      labels={"x": "Text Length", "y": "Count"})
                    st.plotly_chart(fig, use_container_width=True)

                # Token count distribution
                if metrics.token_counts:
                    st.markdown("##### Token Count Distribution")
                    fig = px.histogram(x=metrics.token_counts, nbins=30,
                                      title="Distribution of Token Counts",
                                      labels={"x": "Token Count", "y": "Count"})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No preprocessing has been run yet. Use one of the preprocessing tabs to generate a quality report.")

            # Show convenience functions
            st.markdown("---")
            st.markdown("### Convenience Functions")
            st.markdown("""
            The text processing module also provides these convenience functions:

            - **`normalize_for_nlp(text)`**: Quick NLP-safe normalization
            - **`preprocess_dataframe(df, text_column)`**: Apply preprocessing to a DataFrame
            - **`create_processor_for_dataset(dataset_type)`**: Get a pre-configured processor

            These are used internally by the preset options above.
            """)

    # ==========================================================================
    # Show processed column info and next steps
    # ==========================================================================
    st.markdown("---")

    # Check if processed column exists
    if f"{selected_column}_processed" in st.session_state.uploaded_df.columns:
        st.markdown("### ✅ Processed Data Ready")

        processed_col = f"{selected_column}_processed"
        valid_count = st.session_state.uploaded_df[processed_col].notna().sum()
        total_count = len(st.session_state.uploaded_df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Processed Column", processed_col)
        col2.metric("Valid Entries", f"{valid_count:,}")
        col3.metric("Coverage", f"{valid_count/total_count:.1%}")

        st.info(f"""
        💡 **Tip:** In the Configuration page, you can now select **'{processed_col}'**
        as your text column to run analysis on the preprocessed data.
        """)

    # Next button
    render_next_button("⚙️ Configuration")


def page_configuration():
    """Configuration page."""
    st.markdown('<h2 class="sub-header">⚙️ Analysis Configuration</h2>', unsafe_allow_html=True)

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload data first in the 'Data Upload' section")
        return

    df = st.session_state.uploaded_df

    st.markdown("### 📝 Select Text Column")

    # Column selector
    text_columns = df.select_dtypes(include=['object']).columns.tolist()

    if not text_columns:
        st.error("❌ No text columns found in the data")
        return

    selected_column = st.selectbox(
        "Choose the column containing responses:",
        text_columns,
        index=0
    )

    st.session_state.text_column = selected_column

    # Show sample responses
    st.markdown("#### Sample Responses")
    sample_responses = df[selected_column].dropna().head(5)
    for i, response in enumerate(sample_responses, 1):
        st.text(f"{i}. {truncate_text(str(response), 150)}")

    st.markdown("---")

    # ML Configuration
    st.markdown("### 🤖 ML Algorithm Settings")

    # Show text column badge
    if 'text_column' in st.session_state and st.session_state.text_column:
        st.markdown(f"""
        <div class="info-box" style="padding: 10px; margin-bottom: 15px;">
        📝 <strong>Text column:</strong> {st.session_state.text_column}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box" style="padding: 10px; margin-bottom: 15px;">
        ⚠️ <strong>Select text column in Data Upload first</strong>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        auto_optimal_codes = st.checkbox(
            "Auto-select optimal number of codes",
            value=False,
            help="Let the algorithm automatically determine the optimal number of codes based on your data using silhouette analysis"
        )

        n_codes = st.slider(
            "Number of codes to discover",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            help="How many themes/codes should the algorithm discover?",
            disabled=auto_optimal_codes
        )

        if auto_optimal_codes:
            st.markdown("""
            <div class="info-box" style="padding: 10px;">
            🔍 <strong>Auto-optimization enabled:</strong>
            <ul style="margin: 5px 0 0 0;">
            <li>Adds ~2-5 min to test 3-15 code configurations</li>
            <li>Uses silhouette analysis to find optimal separation</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        method = st.selectbox(
            "ML Algorithm",
            options=['tfidf_kmeans', 'lda', 'lstm_kmeans', 'bert_kmeans', 'svm'],
            index=0,
            format_func=lambda x: {
                'tfidf_kmeans': 'TF-IDF + K-Means (Fast, Recommended)',
                'lda': 'Latent Dirichlet Allocation (Topic Modeling)',
                'lstm_kmeans': 'LSTM + K-Means (Sequential Patterns)',
                'bert_kmeans': 'BERT + K-Means (Semantic Understanding)',
                'svm': 'SVM Spectral Clustering (Kernel-based)'
            }[x],
            help="Choose the machine learning algorithm"
        )

        # Algorithm descriptions with runtime hints and pros/cons
        algorithm_info = {
            'tfidf_kmeans': {
                'description': "**TF-IDF + K-Means** converts text into numerical features based on word importance, then groups similar responses together.",
                'runtime': "⚡ **Fast** (~5-10s for 1000 responses)",
                'good_for': "Good for distinct, separable themes",
                'watch_out': "Watch out for overlapping topics"
            },
            'lda': {
                'description': "**Latent Dirichlet Allocation** is a probabilistic model that discovers hidden topics in text. Each response can belong to multiple topics.",
                'runtime': "🐢 **Moderate** (~30-60s for 1000 responses)",
                'good_for': "Good for discovering overlapping themes",
                'watch_out': "Watch out for slower performance with large datasets"
            },
            'lstm_kmeans': {
                'description': "**LSTM + K-Means** uses a recurrent neural network to capture sequential patterns in text, then clusters the learned representations.",
                'runtime': "🐢 **Slow** (~2-5 min for 1000 responses)",
                'good_for': "Good for capturing word order and context",
                'watch_out': "Watch out for longer training time; requires TensorFlow"
            },
            'bert_kmeans': {
                'description': "**BERT + K-Means** uses transformer-based embeddings to capture deep semantic meaning, then clusters semantically similar responses.",
                'runtime': "🐢 **Moderate** (~1-2 min for 1000 responses)",
                'good_for': "Good for semantic similarity and nuanced meanings",
                'watch_out': "Watch out for requiring sentence-transformers package"
            },
            'svm': {
                'description': "**SVM Spectral Clustering** uses kernel methods (like SVM) to find non-linear cluster boundaries in the feature space.",
                'runtime': "🐢 **Moderate** (~30-60s for 1000 responses)",
                'good_for': "Good for finding complex, non-linear cluster boundaries",
                'watch_out': "Watch out for memory usage with large datasets"
            }
        }
        
        algo = algorithm_info[method]
        st.markdown(f"""
        <div class="info-box" style="padding: 10px;">
        {algo['description']}<br><br>
        {algo['runtime']}<br>
        ✅ {algo['good_for']}<br>
        ⚠️ {algo['watch_out']}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        min_confidence = st.slider(
            "Minimum confidence threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Minimum confidence score for code assignment (lower = more codes per response)"
        )

        # Language is hardcoded to English (non-English text is filtered out)
        stop_words = 'english'

    # ==========================================================================
    # Data Type & Sentiment Analysis Configuration
    # ==========================================================================
    st.markdown("---")
    st.markdown("### 📊 Response Type & Sentiment Analysis")

    # Data type selection with clear descriptions (matches Text Processor presets)
    data_type_options = {
        'survey': {
            'name': 'Survey Responses',
            'description': 'Survey feedback, open-ended questions, form responses',
            'model': 'VADER (rule-based, fast)',
            'icon': '📋'
        },
        'twitter': {
            'name': 'Social Media (Twitter/X)',
            'description': 'Tweets, social posts, short informal text with slang/emojis',
            'model': 'Twitter-RoBERTa (transformer-based)',
            'icon': '🐦'
        },
        'longform': {
            'name': 'Long-form Reviews',
            'description': 'Customer reviews, detailed feedback, elongated expressions',
            'model': 'Review-BERT (transformer-based)',
            'icon': '⭐'
        },
        'news': {
            'name': 'News Articles',
            'description': 'Formal, well-written articles and news content',
            'model': 'Review-BERT (transformer-based)',
            'icon': '📰'
        }
    }

    # Get current data_type from session state, defaulting to 'survey'
    current_data_type = st.session_state.get('data_type', 'survey')
    # Ensure it's a valid option
    if current_data_type not in data_type_options:
        current_data_type = 'survey'

    # Response type radio button
    selected_data_type = st.radio(
        "Select your response type:",
        options=list(data_type_options.keys()),
        format_func=lambda x: f"{data_type_options[x]['icon']} {data_type_options[x]['name']}",
        index=list(data_type_options.keys()).index(current_data_type),
        horizontal=True,
        key="config_data_type",
        help="This affects sentiment analysis model selection and text processing behavior"
    )

    # Update session state immediately when selection changes
    st.session_state.data_type = selected_data_type

    # Show selected type details
    selected_type_info = data_type_options[selected_data_type]
    st.markdown(f"""
    <div class="info-box" style="padding: 12px; margin: 10px 0;">
    <strong>{selected_type_info['icon']} {selected_type_info['name']}</strong><br>
    <em>{selected_type_info['description']}</em><br><br>
    <strong>Sentiment Model:</strong> {selected_type_info['model']}
    </div>
    """, unsafe_allow_html=True)

    # Enable sentiment analysis checkbox
    # Use a single session state key to avoid conflicts between widget key and manual state
    if SENTIMENT_ANALYSIS_AVAILABLE:
        # Initialize if not set
        if 'enable_sentiment' not in st.session_state:
            st.session_state.enable_sentiment = False

        enable_sentiment = st.checkbox(
            "Enable Sentiment Analysis",
            value=st.session_state.enable_sentiment,
            help="Run sentiment analysis to detect positive, neutral, or negative sentiment in each response"
        )
        # Update session state when checkbox changes
        st.session_state.enable_sentiment = enable_sentiment

        if enable_sentiment:
            st.success(f"✅ Sentiment analysis will use **{selected_type_info['model']}** for {selected_type_info['name'].lower()}")
    else:
        st.markdown("""
        <div class="warning-box" style="padding: 10px; font-size: 0.9em;">
        ⚠️ Sentiment analysis not available.<br>
        <code>pip install transformers torch</code>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.enable_sentiment = False
        enable_sentiment = False

    # Save configuration with the selected data_type
    st.session_state.config = {
        'text_column': selected_column,
        'n_codes': n_codes,
        'auto_optimal_codes': auto_optimal_codes,
        'method': method,
        'min_confidence': min_confidence,
        'stop_words': stop_words,
        'data_type': selected_data_type,
        'enable_sentiment': enable_sentiment
    }

    # Show configuration summary
    st.markdown("---")
    st.markdown("### 📋 Configuration Summary")

    config_col1, config_col2, config_col3, config_col4 = st.columns(4)

    with config_col1:
        st.metric("Responses", f"{len(df):,}")
    with config_col2:
        st.metric("Codes to Find", "Auto" if auto_optimal_codes else n_codes)
    with config_col3:
        st.metric("Algorithm", method.upper())
    with config_col4:
        st.metric("Response Type", selected_type_info['name'].split()[0])

    st.success("✅ Configuration saved! Go to 'Run Analysis' to start.")

    # Next button - show when config is saved
    if 'config' in st.session_state:
        render_next_button("🚀 Run Analysis")


def update_stage_checklist(stages, current_stage):
    """
    Generate HTML for stage checklist display.

    Args:
        stages: List of stage names
        current_stage: Index of current stage (0-based), or -1 for all incomplete

    Returns:
        HTML string for the checklist
    """
    stage_icons = {
        'incomplete': '⬜',
        'complete': '✅',
        'current': '🔄'
    }

    html = '<div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0;">'
    html += '<h4 style="margin-top: 0; color: #1f77b4;">Analysis Progress</h4>'
    html += '<ul style="list-style-type: none; padding-left: 0; margin: 10px 0;">'

    for idx, stage in enumerate(stages):
        if idx < current_stage:
            icon = stage_icons['complete']
            style = 'color: #28a745; font-weight: bold;'
        elif idx == current_stage:
            icon = stage_icons['current']
            style = 'color: #1f77b4; font-weight: bold;'
        else:
            icon = stage_icons['incomplete']
            style = 'color: #999;'

        html += f'<li style="padding: 5px 0; {style}">{icon} {stage}</li>'

    html += '</ul></div>'
    return html


def page_run_analysis():
    """Run analysis page."""
    st.markdown('<h2 class="sub-header">🚀 Run ML Analysis</h2>', unsafe_allow_html=True)

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload data first")
        return

    if 'config' not in st.session_state:
        st.warning("⚠️ Please configure the analysis first")
        return

    config = st.session_state.config
    df = st.session_state.uploaded_df

    # Display configuration
    st.markdown("### 📋 Ready to Analyze")

    auto_optimal = config.get('auto_optimal_codes', False)

    # Check if sentiment analysis is enabled
    enable_sentiment = config.get('enable_sentiment', False)
    data_type = config.get('data_type', 'survey')

    # Map data_type to user-friendly labels
    data_type_display = {
        'survey': 'Survey',
        'twitter': 'Social Media',
        'longform': 'Reviews',
        'news': 'News'
    }
    data_type_label = data_type_display.get(data_type, data_type.title())

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Responses", f"{len(df):,}")
    with col2:
        st.metric("Text Column", config['text_column'])
    with col3:
        st.metric("Codes", "Auto" if auto_optimal else config['n_codes'])
    with col4:
        st.metric("Method", config['method'].upper())
    with col5:
        st.metric("Response Type", data_type_label)

    if auto_optimal:
        st.info("🔍 The algorithm will automatically determine the optimal number of codes before running the analysis.")

    if enable_sentiment and SENTIMENT_ANALYSIS_AVAILABLE:
        data_type_models = {'twitter': 'Twitter-RoBERTa', 'survey': 'VADER', 'longform': 'Review-BERT'}
        model_label = data_type_models.get(data_type, 'Standard')
        st.success(f"📊 **Sentiment Analysis Enabled** - Using {model_label} model for {data_type_label} responses")

    st.markdown("---")

    # Run button
    if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
        # Define analysis stages (include sentiment if enabled)
        stages = [
            "Data Preparation",
            "Feature Extraction",
            "Clustering/Modeling",
            "Code Labeling",
            "Generating Insights"
        ]
        if enable_sentiment and SENTIMENT_ANALYSIS_AVAILABLE:
            stages.append("Sentiment Analysis")

        # Progress tracking UI elements
        progress_bar = st.progress(0)
        status_text = st.empty()
        stage_checklist = st.empty()

        # Initialize checklist
        stage_checklist.markdown(update_stage_checklist(stages, -1), unsafe_allow_html=True)

        def update_progress(progress, message, stage_idx=-1):
            progress_bar.progress(progress)
            status_text.text(message)
            if stage_idx >= 0:
                stage_checklist.markdown(update_stage_checklist(stages, stage_idx), unsafe_allow_html=True)

        try:
            # Run analysis
            start_time = time.time()

            # Determine number of codes
            n_codes = config['n_codes']

            if auto_optimal:
                update_progress(0.1, "🔍 Finding optimal number of codes...", 0)

                try:
                    optimal_n, optimal_results = find_optimal_codes(
                        df=df,
                        text_column=config['text_column'],
                        method=config['method'],
                        stop_words=config.get('stop_words', 'english'),
                        progress_callback=lambda p, m: update_progress(0.1 + p * 0.3, m, 0)
                    )
                    n_codes = optimal_n

                    # Show which representation was used for optimization
                    rep_used = optimal_results.get('representation_used', 'tfidf')
                    rep_display = rep_used.upper() if rep_used in ('bert', 'lstm') else 'TF-IDF'
                    st.success(f"✨ Optimal number of codes determined: **{optimal_n}** (silhouette score: {optimal_results['best_silhouette_score']:.4f}, using {rep_display} features)")
                except ValueError as ve:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Cannot find optimal codes: {str(ve)}")
                    st.info("💡 **Suggestion:** Try uploading a larger dataset with more diverse responses, or disable auto-optimization and manually select a smaller number of codes.")
                    raise

                # Adjust progress for main analysis
                def adjusted_progress(p, m):
                    # Map progress 0-1 to stages 0-4 and progress 0.4-1.0
                    if p < 0.3:
                        stage = 0
                    elif p < 0.5:
                        stage = 1
                    elif p < 0.8:
                        stage = 2
                    elif p < 0.95:
                        stage = 3
                    else:
                        stage = 4
                    update_progress(0.4 + p * 0.6, m, stage)
            else:
                # Map progress 0-1 to stages 0-4
                def adjusted_progress(p, m):
                    if p < 0.3:
                        stage = 0
                    elif p < 0.5:
                        stage = 1
                    elif p < 0.8:
                        stage = 2
                    elif p < 0.95:
                        stage = 3
                    else:
                        stage = 4
                    update_progress(p, m, stage)

            coder, results_df, metrics = run_ml_analysis(
                df=df,
                text_column=config['text_column'],
                n_codes=n_codes,
                method=config['method'],
                min_confidence=config['min_confidence'],
                progress_callback=adjusted_progress
            )

            # Store optimization info if auto-detection was used
            if auto_optimal:
                metrics['auto_optimal'] = True
                metrics['optimal_analysis'] = optimal_results

            # Run sentiment analysis if enabled
            sentiment_results = None
            if enable_sentiment and SENTIMENT_ANALYSIS_AVAILABLE:
                try:
                    update_progress(0.92, "Running sentiment analysis...", len(stages) - 1)
                    status_text.text(f"🔄 Loading {data_type} sentiment model...")

                    # Get the appropriate analyzer for the data type (cached to avoid reload)
                    analyzer = get_cached_sentiment_analyzer(data_type)

                    # Run sentiment analysis
                    texts = df[config['text_column']].tolist()
                    sentiment_results = analyzer.analyze(texts)

                    # Add sentiment columns to results_df
                    results_df['sentiment_label'] = [r.label for r in sentiment_results]
                    results_df['sentiment_score'] = [r.score for r in sentiment_results]
                    results_df['sentiment_positive'] = [r.scores.get('positive', 0) for r in sentiment_results]
                    results_df['sentiment_negative'] = [r.scores.get('negative', 0) for r in sentiment_results]
                    results_df['sentiment_neutral'] = [r.scores.get('neutral', 0) for r in sentiment_results]

                    # Store sentiment metrics
                    sentiment_counts = pd.Series([r.label for r in sentiment_results]).value_counts()
                    metrics['sentiment_enabled'] = True
                    metrics['sentiment_model'] = analyzer.get_model_info()['model_name']
                    metrics['sentiment_distribution'] = sentiment_counts.to_dict()
                    metrics['sentiment_avg_confidence'] = float(np.mean([r.score for r in sentiment_results]))

                    status_text.text("✅ Sentiment analysis complete!")

                except Exception as e:
                    st.warning(f"⚠️ Sentiment analysis failed: {str(e)}. Continuing without sentiment.")
                    metrics['sentiment_enabled'] = False
                    metrics['sentiment_error'] = str(e)
            else:
                metrics['sentiment_enabled'] = False

            # Save to session state
            st.session_state.coder = coder
            st.session_state.results_df = results_df
            st.session_state.metrics = metrics
            st.session_state.analysis_complete = True
            st.session_state.sentiment_results = sentiment_results

            # Pre-compute all visualization data upfront (prevents viz page hangs)
            st.session_state.viz_data = precompute_all_visualizations(coder, results_df)

            # Mark all stages complete
            stage_checklist.markdown(update_stage_checklist(stages, len(stages)), unsafe_allow_html=True)

            # Clear progress
            progress_bar.empty()
            status_text.empty()

            # Show success
            execution_time = time.time() - start_time

            # Build success message with optional sentiment info
            sentiment_info = ""
            if metrics.get('sentiment_enabled', False):
                sentiment_dist = metrics.get('sentiment_distribution', {})
                sentiment_info = f"""<hr style="margin: 10px 0;">
<p><strong>Sentiment Analysis:</strong> ✅ Complete</p>
<p><strong>Model:</strong> {metrics.get('sentiment_model', 'N/A')}</p>
<p><strong>Distribution:</strong> 😊 {sentiment_dist.get('positive', 0)} positive, 😐 {sentiment_dist.get('neutral', 0)} neutral, 😞 {sentiment_dist.get('negative', 0)} negative</p>"""

            st.markdown(f"""
            <div class="success-box">
            <h3>✅ Analysis Complete!</h3>
            <p><strong>Execution Time:</strong> {format_duration(execution_time)}</p>
            <p><strong>Codes Found:</strong> {metrics['n_codes']}</p>
            <p><strong>Total Assignments:</strong> {metrics.get('total_assignments', 0):,}</p>
            <p><strong>Coverage:</strong> {metrics.get('coverage_pct', 0):.1f}%</p>
            {sentiment_info}
            </div>
            """, unsafe_allow_html=True)

            # Show quick insights
            st.markdown("### 🔍 Quick Insights")
            insights = generate_insights(coder, results_df)
            for insight in insights:
                st.markdown(insight)

            # Show sentiment summary if enabled
            if metrics.get('sentiment_enabled', False):
                st.markdown("### 📊 Sentiment Summary")
                sentiment_dist = metrics.get('sentiment_distribution', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("😊 Positive", sentiment_dist.get('positive', 0))
                with col2:
                    st.metric("😐 Neutral", sentiment_dist.get('neutral', 0))
                with col3:
                    st.metric("😞 Negative", sentiment_dist.get('negative', 0))
                with col4:
                    st.metric("Avg Confidence", f"{metrics.get('sentiment_avg_confidence', 0):.2f}")

            st.info("👉 Go to 'Results Overview' to see detailed results")

        except ValueError as ve:
            # ValueError indicates dataset validation issues - already handled above or here
            if "Cannot find optimal codes" not in str(ve):
                progress_bar.empty()
                status_text.empty()
                stage_checklist.empty()
                st.error(f"❌ Dataset validation error: {str(ve)}")
                st.info("💡 **Suggestions:**\n- Ensure your dataset has enough responses\n- Check that responses aren't too short or too similar\n- Try reducing the number of codes requested\n- Review your preprocessing settings")

                # Add retry buttons
                st.markdown("#### 🔄 Quick Fixes")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Try with fewer codes", use_container_width=True):
                        if not auto_optimal and config['n_codes'] > 2:
                            st.session_state.config['n_codes'] = max(2, config['n_codes'] - 2)
                            st.success(f"✅ Reduced codes to {st.session_state.config['n_codes']}. Click 'Start Analysis' again.")
                            st.rerun()
                        else:
                            st.warning("Cannot reduce codes further or auto-optimization is enabled.")

                with col2:
                    if st.button("Try with lower confidence", use_container_width=True):
                        if config['min_confidence'] > 0.1:
                            st.session_state.config['min_confidence'] = max(0.1, config['min_confidence'] - 0.1)
                            st.success(f"✅ Reduced confidence to {st.session_state.config['min_confidence']:.2f}. Click 'Start Analysis' again.")
                            st.rerun()
                        else:
                            st.warning("Confidence already at minimum (0.1).")

                with col3:
                    if st.button("Reset and try again", use_container_width=True):
                        st.session_state.analysis_complete = False
                        st.session_state.pop('coder', None)
                        st.session_state.pop('results_df', None)
                        st.session_state.pop('metrics', None)
                        st.success("✅ Analysis state reset. Click 'Start Analysis' again.")
                        st.rerun()
            else:
                # For optimal codes finding error, clear UI elements
                stage_checklist.empty()

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            stage_checklist.empty()
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)

            # Add retry buttons for general errors too
            st.markdown("#### 🔄 Quick Fixes")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Try with fewer codes", key="error_fewer_codes", use_container_width=True):
                    if not auto_optimal and config['n_codes'] > 2:
                        st.session_state.config['n_codes'] = max(2, config['n_codes'] - 2)
                        st.success(f"✅ Reduced codes to {st.session_state.config['n_codes']}. Click 'Start Analysis' again.")
                        st.rerun()
                    else:
                        st.warning("Cannot reduce codes further or auto-optimization is enabled.")

            with col2:
                if st.button("Try with lower confidence", key="error_lower_conf", use_container_width=True):
                    if config['min_confidence'] > 0.1:
                        st.session_state.config['min_confidence'] = max(0.1, config['min_confidence'] - 0.1)
                        st.success(f"✅ Reduced confidence to {st.session_state.config['min_confidence']:.2f}. Click 'Start Analysis' again.")
                        st.rerun()
                    else:
                        st.warning("Confidence already at minimum (0.1).")

            with col3:
                if st.button("Reset and try again", key="error_reset", use_container_width=True):
                    st.session_state.analysis_complete = False
                    st.session_state.pop('coder', None)
                    st.session_state.pop('results_df', None)
                    st.session_state.pop('metrics', None)
                    st.success("✅ Analysis state reset. Click 'Start Analysis' again.")
                    st.rerun()

    # Next button - show when analysis is complete
    if st.session_state.analysis_complete:
        render_next_button("📊 Results Overview")


def page_results_overview():
    """Results overview page."""
    st.markdown('<h2 class="sub-header">📊 Results Overview</h2>', unsafe_allow_html=True)

    if not st.session_state.analysis_complete:
        st.warning("⚠️ Please run the analysis first")
        return

    coder = st.session_state.coder
    results_df = st.session_state.results_df
    metrics = st.session_state.metrics

    # Stat chips - compact metric display
    st.markdown("### 📈 Key Metrics")

    stat_chips_html = f"""
    <div style="margin: 10px 0 20px 0;">
        <span class="stat-chip">📊 {metrics.get('total_responses', 0):,} Responses</span>
        <span class="stat-chip">🏷️ {metrics.get('n_codes', 0)} Codes</span>
        <span class="stat-chip">📈 {metrics.get('avg_codes_per_response', 0):.2f} Avg/Response</span>
        <span class="stat-chip">✅ {metrics.get('coverage_pct', 0):.1f}% Coverage</span>
    </div>
    """
    st.markdown(stat_chips_html, unsafe_allow_html=True)

    # Download All button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        try:
            excel_data = export_results_package(coder, results_df, format='excel')
            st.download_button(
                label="📥 Download All",
                data=excel_data,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Export error: {str(e)}")

    # Tabs for Extracted Codes, Assignments, and Sentiment (if enabled)
    st.markdown("---")

    # Check if sentiment analysis was enabled
    sentiment_enabled = metrics.get('sentiment_enabled', False)

    if sentiment_enabled:
        tab1, tab2, tab3 = st.tabs(["Extracted Codes", "Label and Sentiment Assignments", "😊 Sentiment"])
    else:
        tab1, tab2 = st.tabs(["Extracted Codes", "Label and Sentiment Assignments"])

    with tab1:
        # All codes ordered by code ID (CODE_01, CODE_02, etc.)
        all_codes_df = get_top_codes(coder, n=None, sort_by='code')

        # Display as styled table with all codes
        st.dataframe(
            style_frequency_table(all_codes_df),
            use_container_width=True,
            height=500
        )

    with tab2:
        # Label and Sentiment Assignments
        # Prepare assignments dataframe
        assignments_df = results_df.copy()
        total_responses = len(assignments_df)

        # Calculate prevalence for each code
        code_counts = {}
        for codes in assignments_df['assigned_codes']:
            if codes:
                for code in codes:
                    code_counts[code] = code_counts.get(code, 0) + 1

        # Create prevalence lookup
        def get_prevalence(codes):
            if not codes or len(codes) == 0:
                return "0.0%"
            # Use the first code's prevalence for display
            first_code = codes[0]
            count = code_counts.get(first_code, 0)
            prevalence = (count / total_responses) * 100 if total_responses > 0 else 0
            return f"{prevalence:.1f}%"

        # Build display columns: Response, Label, Sentiment (if available), Prevalence
        text_col = st.session_state.config['text_column']
        display_cols = [text_col, 'assigned_codes']

        # Check if sentiment data is available
        has_sentiment = 'sentiment_label' in assignments_df.columns

        if has_sentiment:
            display_cols.append('sentiment_label')

        # Select sample
        sample_size = min(20, len(assignments_df))
        sample_df = assignments_df[display_cols].head(sample_size).copy()

        # Format for display
        display_df = sample_df.copy()

        # Format assigned_codes as Label (lookup actual labels from codebook)
        display_df['Label'] = display_df['assigned_codes'].apply(
            lambda codes: ', '.join([coder.codebook[code]['label'] for code in codes if code in coder.codebook]) if codes else 'None'
        )

        # Add Prevalence column
        display_df['Prevalence'] = sample_df['assigned_codes'].apply(get_prevalence)

        # Rename columns for display
        display_df = display_df.rename(columns={text_col: 'Response'})

        if has_sentiment:
            display_df = display_df.rename(columns={'sentiment_label': 'Sentiment'})
            # Capitalize sentiment values
            display_df['Sentiment'] = display_df['Sentiment'].apply(
                lambda x: x.capitalize() if isinstance(x, str) else 'N/A'
            )
            final_cols = ['Response', 'Label', 'Sentiment', 'Prevalence']
        else:
            final_cols = ['Response', 'Label', 'Prevalence']

        display_df = display_df[final_cols]

        if len(display_df) > 0:
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.info("No assignments to display")

    # Sentiment tab (only if enabled)
    if sentiment_enabled:
        with tab3:
            st.markdown("### 😊 Sentiment Analysis Results")

            # Show model info
            model_name = metrics.get('sentiment_model', 'Unknown')
            st.markdown(f"""
            <div class="info-box" style="padding: 10px; margin-bottom: 15px;">
            <strong>Model Used:</strong> <code>{model_name}</code><br>
            <strong>Avg Confidence:</strong> {metrics.get('sentiment_avg_confidence', 0):.2f}
            </div>
            """, unsafe_allow_html=True)

            # Sentiment distribution chart
            sentiment_dist = metrics.get('sentiment_distribution', {})
            if sentiment_dist:
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Create pie chart
                    labels = list(sentiment_dist.keys())
                    values = list(sentiment_dist.values())
                    colors = {'positive': '#28a745', 'neutral': '#6c757d', 'negative': '#dc3545'}

                    fig = px.pie(
                        values=values,
                        names=labels,
                        title='Sentiment Distribution',
                        color=labels,
                        color_discrete_map=colors
                    )
                    fig.update_traces(textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Show metrics
                    total = sum(sentiment_dist.values())
                    st.metric("😊 Positive", f"{sentiment_dist.get('positive', 0)} ({sentiment_dist.get('positive', 0)/total*100:.1f}%)" if total > 0 else "0")
                    st.metric("😐 Neutral", f"{sentiment_dist.get('neutral', 0)} ({sentiment_dist.get('neutral', 0)/total*100:.1f}%)" if total > 0 else "0")
                    st.metric("😞 Negative", f"{sentiment_dist.get('negative', 0)} ({sentiment_dist.get('negative', 0)/total*100:.1f}%)" if total > 0 else "0")

            # Sample responses by sentiment
            st.markdown("#### Sample Responses by Sentiment")

            sentiment_filter = st.selectbox(
                "Filter by sentiment:",
                options=['All', 'Positive', 'Neutral', 'Negative'],
                index=0
            )

            # Filter and display
            sentiment_df = results_df.copy()
            if sentiment_filter != 'All':
                sentiment_df = sentiment_df[sentiment_df['sentiment_label'] == sentiment_filter.lower()]

            # Select columns to display
            text_col = st.session_state.config['text_column']
            display_cols_sentiment = [text_col, 'sentiment_label', 'sentiment_score']
            if 'assigned_codes' in sentiment_df.columns:
                display_cols_sentiment.append('assigned_codes')

            sample_sentiment_df = sentiment_df[display_cols_sentiment].head(20).copy()

            # Format assigned_codes if present
            if 'assigned_codes' in sample_sentiment_df.columns:
                sample_sentiment_df['assigned_codes'] = sample_sentiment_df['assigned_codes'].apply(
                    lambda x: ', '.join(x) if x else 'None'
                )

            # Format sentiment score
            sample_sentiment_df['sentiment_score'] = sample_sentiment_df['sentiment_score'].apply(
                lambda x: f"{x:.3f}"
            )

            st.dataframe(sample_sentiment_df, use_container_width=True, height=400)

            # Download sentiment results
            sentiment_csv = results_df[[text_col, 'sentiment_label', 'sentiment_score',
                                        'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']].to_csv(index=False).encode()
            st.download_button(
                label="📥 Download Sentiment Results (CSV)",
                data=sentiment_csv,
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

    # Word Cloud Visualization (directly after results table, before codebook)
    st.markdown("---")
    st.markdown("### ☁️ Word Cloud")

    # Ensure visualization data is pre-computed
    wordcloud_generated = False
    if ensure_viz_data_ready():
        viz_data = st.session_state.viz_data

        if viz_data.get('wordcloud_available', False):
            wordcloud_text = viz_data['wordcloud_text']

            # Generate word cloud
            wordcloud = None
            try:
                if WORDCLOUD_AVAILABLE:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='viridis',
                        max_words=100,
                        min_font_size=10,
                        max_font_size=100
                    ).generate(wordcloud_text)
                elif PILWordCloud is not None:
                    wordcloud = PILWordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        max_words=100,
                        min_font_size=10,
                        max_font_size=100
                    ).generate(wordcloud_text)
            except ValueError:
                # WordCloud raises ValueError if text is empty or only stopwords
                st.info("📝 Not enough text content to generate word cloud. The text may contain only common words (stopwords) that are filtered out.")
                wordcloud = None

            if wordcloud is not None:
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud.to_image(), interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout()

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)  # Clean up to prevent memory leaks

                st.caption("Word cloud generated from all response text - larger words appear more frequently")
                wordcloud_generated = True
            elif wordcloud_text.strip():
                st.warning("⚠️ Word cloud generation failed. PIL library may not be available.")

    # Fallback: try to generate wordcloud directly from results_df if viz_data wasn't available
    if not wordcloud_generated and (WORDCLOUD_AVAILABLE or (PIL_AVAILABLE and PILWordCloud is not None)):
        try:
            results_df = st.session_state.results_df
            # Find text column
            text_col = [col for col in results_df.columns if col not in ['assigned_codes', 'confidence_scores', 'num_codes', 'themes', 'sentiment_label', 'sentiment_score', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']][0]
            all_text = ' '.join(results_df[text_col].dropna().astype(str).tolist())
            cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', all_text.lower())
            cleaned_text = ' '.join(cleaned_text.split())

            if cleaned_text.strip():
                wordcloud = None
                if WORDCLOUD_AVAILABLE:
                    wordcloud = WordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        colormap='viridis',
                        max_words=100,
                        min_font_size=10,
                        max_font_size=100
                    ).generate(cleaned_text)
                elif PILWordCloud is not None:
                    wordcloud = PILWordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        max_words=100,
                        min_font_size=10,
                        max_font_size=100
                    ).generate(cleaned_text)

                if wordcloud is not None:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud.to_image(), interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout()

                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    st.caption("Word cloud generated from all response text - larger words appear more frequently")
                    wordcloud_generated = True
        except Exception as e:
            st.warning(f"⚠️ Word cloud generation failed: {str(e)}")

    # Final fallback: pure matplotlib word frequency chart (no wordcloud/PIL needed)
    if not wordcloud_generated and 'results_df' in st.session_state:
        try:
            results_df = st.session_state.results_df

            # Find text column - try multiple approaches
            exclude_cols = {'assigned_codes', 'confidence_scores', 'num_codes', 'themes',
                           'sentiment_label', 'sentiment_score', 'sentiment_positive',
                           'sentiment_negative', 'sentiment_neutral', 'index'}
            text_candidates = [col for col in results_df.columns if col.lower() not in exclude_cols]

            # Prefer columns with common text column names
            text_col = None
            for preferred in ['response', 'text', 'content', 'answer', 'comment', 'feedback', 'message']:
                for col in text_candidates:
                    if preferred in col.lower():
                        text_col = col
                        break
                if text_col:
                    break

            # Fall back to first non-excluded column
            if not text_col and text_candidates:
                text_col = text_candidates[0]

            if text_col is None:
                st.warning(f"⚠️ Could not find text column. Available columns: {list(results_df.columns)}")
            else:
                all_text = ' '.join(results_df[text_col].dropna().astype(str).tolist())
                cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', all_text.lower())
                words = cleaned_text.split()

                # Very minimal stopwords - ONLY articles/prepositions/pronouns
                # NO content words since data is likely lemmatized/preprocessed
                stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'if', 'it', 'its', 'this', 'that', 'i', 'you', 'we', 'they', 'my', 'your', 'he', 'she', 'him', 'her', 'me', 'them', 'us', 'is', 'was', 'are', 'were', 'be', 'been', 'am', 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'from', 'than', 'so', 'not', 'no', 'nan', 'none', 'also', 'very', 'just', 'only', 'even', 'still', 'already', 'yet'}
                word_counts = {}
                for word in words:
                    if len(word) > 1 and word not in stopwords:  # Allow 2-char words
                        word_counts[word] = word_counts.get(word, 0) + 1

                if word_counts:
                    # Get top 30 words
                    top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:30]
                    words_list = [w[0] for w in top_words]
                    counts_list = [w[1] for w in top_words]

                    # Create horizontal bar chart
                    fig, ax = plt.subplots(figsize=(10, 8))
                    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(words_list)))
                    ax.barh(range(len(words_list)), counts_list, color=colors)
                    ax.set_yticks(range(len(words_list)))
                    ax.set_yticklabels(words_list)
                    ax.invert_yaxis()  # Largest at top
                    ax.set_xlabel('Frequency')
                    ax.set_title('Top 30 Most Frequent Words')
                    plt.tight_layout()

                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    st.caption("Word frequency chart - showing the most common words in response text")
                    wordcloud_generated = True
                else:
                    st.info(f"📝 No words found after filtering. Text column: '{text_col}', Total words before filter: {len(words)}")
        except Exception as e:
            st.error(f"⚠️ Word frequency visualization failed: {type(e).__name__}: {str(e)}")

    if not wordcloud_generated:
        if 'results_df' not in st.session_state:
            st.info("📝 Word visualization not available. Please run an analysis first.")
        else:
            st.info("📝 Word visualization could not be generated. Check error messages above.")

    # Detailed codebook
    st.markdown("---")
    st.markdown("### 📖 Complete Codebook")

    # Show warning if any codes have mixed sentiment
    if hasattr(coder, 'cluster_interpretation') and coder.cluster_interpretation is not None:
        mixed_clusters = [
            s.cluster_id.replace('CLUSTER_', 'CODE_')
            for s in coder.cluster_interpretation.summaries.values()
            if s.has_mixed_sentiment
        ]
        if mixed_clusters:
            with st.expander(f"⚠️ {len(mixed_clusters)} code(s) have mixed sentiment", expanded=False):
                st.markdown("""
                **What this means:** These codes contain both positive and negative views about the same topic.
                This happens because clustering groups responses by *topic similarity*, not by sentiment.

                **Example:** A code labeled "Product Quality" might contain both "quality is excellent!"
                and "quality has declined" because both discuss *quality*.

                **What to do:**
                - Review the representative quotes to ensure the label accurately represents the mix
                - Consider renaming ambiguous labels (e.g., "Product Quality" instead of "Declining Quality")
                - Increase the number of clusters to potentially separate positive/negative views
                """)
                st.markdown(f"**Affected codes:** {', '.join(mixed_clusters)}")

    for code_id, info in sorted(coder.codebook.items(), key=lambda x: x[0]):  # Sort by code ID (CODE_01, CODE_02, etc.)
        if info['count'] > 0:  # Only show active codes
            # Use LLM-generated label if available
            display_label = info.get('llm_label', info['label'])
            with st.expander(f"**{code_id}**: {display_label} ({info['count']} responses)"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Show LLM description if available
                    if info.get('llm_description'):
                        st.markdown(f"**Description:** {info['llm_description']}")

                    # Show alternative labels if available
                    if info.get('alternative_labels'):
                        st.markdown(f"**Alternative Labels:** {', '.join(info['alternative_labels'])}")

                    st.markdown(f"**Keywords:** {', '.join(info['keywords'][:10])}")

                    # Show examples (5 representative quotes or all if fewer)
                    if info['examples']:
                        st.markdown("**Example Responses:**")
                        examples = info['examples'][:5]  # Show up to 5 quotes
                        for i, example in enumerate(examples, 1):
                            st.text(f"{i}. {truncate_text(example['text'], 100)} [{example['confidence']:.2f}]")

                with col2:
                    st.metric("Count", f"{info['count']:,}")
                    st.metric("Avg Confidence", f"{info['avg_confidence']:.2f}")

                    # Show sentiment distribution from cluster interpretation if available
                    if hasattr(coder, 'cluster_interpretation') and coder.cluster_interpretation is not None:
                        # Map CODE_XX to CLUSTER_XX
                        cluster_id = code_id.replace('CODE_', 'CLUSTER_')
                        if cluster_id in coder.cluster_interpretation.summaries:
                            summary = coder.cluster_interpretation.summaries[cluster_id]
                            if summary.sentiment_distribution is not None:
                                st.markdown("**Sentiment Mix:**")
                                pos_pct = summary.sentiment_distribution.get('positive', 0)
                                neg_pct = summary.sentiment_distribution.get('negative', 0)
                                neu_pct = summary.sentiment_distribution.get('neutral', 0)
                                st.caption(f"👍 {pos_pct:.0%} · 👎 {neg_pct:.0%} · 😐 {neu_pct:.0%}")
                                if summary.has_mixed_sentiment:
                                    st.warning("⚠️ Mixed sentiment: contains both positive and negative views on this topic")

    # Next button - always show on this page if analysis is complete
    render_next_button("📈 Visualizations")


def page_visualizations():
    """
    Visualizations page - completely revised for performance.

    Architecture:
    - All data is pre-computed once after analysis (stored in viz_data)
    - No heavy computations happen during tab switches
    - Minimal interactive elements to reduce re-renders
    - Simplified Quotes tab with pagination instead of complex filters
    """
    st.markdown('<h2 class="sub-header">📈 Visualizations</h2>', unsafe_allow_html=True)

    if not st.session_state.analysis_complete:
        st.warning("⚠️ Please run the analysis first")
        return

    # Ensure visualization data is pre-computed
    if not ensure_viz_data_ready():
        st.error("Unable to load visualization data. Please re-run analysis.")
        return

    # Get pre-computed data (instant access, no computation)
    viz_data = st.session_state.viz_data
    coder = st.session_state.coder

    # 7-tab layout with visualizations (word cloud, sunburst, network)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Frequency",
        "🔥 Heatmap",
        "📉 Stats",
        "💬 Quotes",
        "☁️ Word Cloud",
        "🌞 Sunburst",
        "🔗 Network"
    ])

    # =========================================================================
    # TAB 1: Code Frequency (uses pre-computed data)
    # =========================================================================
    with tab1:
        st.markdown("### Code Frequency Distribution")

        with st.expander("ℹ️ What am I seeing?", expanded=False):
            st.markdown("""
            **What this shows:** Bar chart of the top 15 most frequent codes.

            **How to interpret:** Taller bars = more responses. Color = confidence.
            """)

        # Use pre-computed DataFrame (instant)
        top_codes_df = viz_data['top_codes_df']

        fig = px.bar(
            top_codes_df,
            x='Label',
            y='Count',
            color='Avg Confidence',
            title='Top 15 Code Frequencies',
            color_continuous_scale='Viridis',
            text='Count'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(xaxis_tickangle=-45, height=500)

        st.plotly_chart(fig, use_container_width=True, key="freq_chart")

        # Simple download
        csv_data = top_codes_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Download Data (CSV)",
            data=csv_data,
            file_name="code_frequency.csv",
            mime="text/csv"
        )

    # =========================================================================
    # TAB 2: Co-occurrence Heatmap (uses pre-computed matrix)
    # =========================================================================
    with tab2:
        st.markdown("### Co-occurrence Heatmap")

        with st.expander("ℹ️ What am I seeing?", expanded=False):
            st.markdown("""
            **What this shows:** Which codes appear together in the same response.

            **Interpretation:** Darker cells = more frequent co-occurrence.
            """)

        # Use pre-computed matrix and labels (instant)
        cooccur = viz_data['cooccurrence_matrix']
        labels = viz_data['cooccurrence_labels']

        fig = px.imshow(
            cooccur,
            labels=dict(color="Co-occurrences"),
            x=labels,
            y=labels,
            title="Code Co-occurrence Matrix",
            color_continuous_scale='YlOrRd',
            aspect='auto'
        )
        fig.update_layout(height=600)

        st.plotly_chart(fig, use_container_width=True, key="heatmap_chart")

        # Top pairs table (pre-computed)
        st.markdown("#### Top Co-occurring Pairs")
        pairs_df = viz_data['cooccurrence_pairs']
        if not pairs_df.empty:
            st.dataframe(pairs_df.head(10), use_container_width=True)
        else:
            st.info("No significant co-occurrences found")

    # =========================================================================
    # TAB 3: Distribution Stats (uses pre-computed stats)
    # =========================================================================
    with tab3:
        st.markdown("### Distribution Statistics")

        # Codes per response histogram
        st.markdown("#### Codes per Response")
        num_codes_data = viz_data['num_codes_data']
        stats = viz_data['num_codes_stats']

        fig1 = px.histogram(
            x=num_codes_data,
            title='Distribution of Codes per Response',
            labels={'x': 'Number of Codes', 'y': 'Frequency'},
            nbins=max(stats['max'], 5)
        )
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True, key="dist_chart")

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", f"{stats['mean']:.2f}")
        col2.metric("Median", f"{stats['median']:.0f}")
        col3.metric("Max", f"{stats['max']}")

        st.markdown("---")

        # Confidence distribution
        st.markdown("#### Confidence Scores")
        all_confidences = viz_data['all_confidences']
        conf_stats = viz_data['confidence_stats']

        if all_confidences and conf_stats:
            fig2 = px.histogram(
                x=all_confidences,
                nbins=30,
                title='Distribution of Confidence Scores',
                labels={'x': 'Confidence Score', 'y': 'Frequency'}
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True, key="conf_chart")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{conf_stats['mean']:.3f}")
            col2.metric("Median", f"{conf_stats['median']:.3f}")
            col3.metric("Min", f"{conf_stats['min']:.3f}")
            col4.metric("Max", f"{conf_stats['max']:.3f}")
        else:
            st.info("No confidence scores available")

    # =========================================================================
    # TAB 4: Quotes (simplified - no complex filtering)
    # =========================================================================
    with tab4:
        st.markdown("### Representative Quotes")

        # Use pre-computed code list
        codes_with_examples = viz_data['codes_with_examples']

        if not codes_with_examples:
            st.info("No example quotes available.")
        else:
            # Simple dropdown using pre-formatted options
            selected_code = st.selectbox(
                "Select a code:",
                options=codes_with_examples,
                format_func=lambda x: viz_data['code_options'].get(x, x),
                key="quote_code_select"
            )

            if selected_code:
                code_info = coder.codebook[selected_code]
                display_label = code_info.get('llm_label', code_info['label'])  # Prefer LLM label

                # Code summary
                st.markdown(f"**{display_label}**")
                if code_info.get('llm_description'):
                    st.caption(code_info['llm_description'])
                st.caption(f"Keywords: {', '.join(code_info['keywords'][:8])}")
                st.caption(f"Count: {code_info['count']} | Avg Confidence: {code_info['avg_confidence']:.2f}")

                st.markdown("---")

                # Get examples sorted by confidence
                examples = sorted(code_info['examples'], key=lambda x: x['confidence'], reverse=True)

                # Simple pagination instead of complex filters
                total_quotes = len(examples)
                quotes_per_page = 5

                if total_quotes > quotes_per_page:
                    page = st.number_input(
                        f"Page (1-{(total_quotes + quotes_per_page - 1) // quotes_per_page})",
                        min_value=1,
                        max_value=(total_quotes + quotes_per_page - 1) // quotes_per_page,
                        value=1,
                        key="quote_page"
                    )
                    start_idx = (page - 1) * quotes_per_page
                    end_idx = min(start_idx + quotes_per_page, total_quotes)
                    display_examples = examples[start_idx:end_idx]
                    st.caption(f"Showing quotes {start_idx + 1}-{end_idx} of {total_quotes}")
                else:
                    display_examples = examples
                    st.caption(f"Showing all {total_quotes} quotes")

                # Display quotes (simple layout)
                for i, ex in enumerate(display_examples, 1):
                    conf_color = "🟢" if ex['confidence'] >= 0.7 else "🟡" if ex['confidence'] >= 0.4 else "🔴"
                    st.markdown(f"**{conf_color} Quote** (confidence: {ex['confidence']:.2f})")
                    st.markdown(f"> {ex['text']}")
                    st.markdown("---")

                # Simple export button
                if examples:
                    export_text = f"Code: {display_label}\n\n"
                    for i, ex in enumerate(examples, 1):
                        export_text += f"Quote {i} (confidence: {ex['confidence']:.3f}):\n{ex['text']}\n\n"

                    st.download_button(
                        "📥 Export All Quotes",
                        data=export_text,
                        file_name=f"quotes_{selected_code}.txt",
                        mime="text/plain"
                    )

    # =========================================================================
    # TAB 5: Word Cloud (uses pre-computed text)
    # =========================================================================
    with tab5:
        st.markdown("### Word Cloud Visualizations")

        # Create sub-tabs for different wordcloud types
        wc_tab1, wc_tab2, wc_tab3 = st.tabs([
            "📊 Overall Word Cloud",
            "🎨 Semantic Word Clouds by Topic",
            "📝 Topic Word Clouds (Simple)"
        ])

        # Sub-tab 1: Overall Word Cloud
        with wc_tab1:
            st.markdown("#### Overall Word Cloud")

            with st.expander("ℹ️ What am I seeing?", expanded=False):
                st.markdown("""
                **What this shows:** Visual representation of the most frequent words across ALL responses.

                **How to interpret:** Larger words = more frequent. Colors are for visual distinction only.
                """)

            if viz_data.get('wordcloud_available', False):
                wordcloud_text = viz_data['wordcloud_text']

                # Generate word cloud (lightweight - just rendering pre-cleaned text)
                # Use wordcloud package if available, otherwise fall back to PIL
                wordcloud = None
                try:
                    if WORDCLOUD_AVAILABLE:
                        wordcloud = WordCloud(
                            width=800,
                            height=400,
                            background_color='white',
                            colormap='viridis',
                            max_words=100,
                            min_font_size=10,
                            max_font_size=100
                        ).generate(wordcloud_text)
                    elif PILWordCloud is not None:
                        wordcloud = PILWordCloud(
                            width=800,
                            height=400,
                            background_color='white',
                            max_words=100,
                            min_font_size=10,
                            max_font_size=100
                        ).generate(wordcloud_text)
                except ValueError as e:
                    # WordCloud raises ValueError if text is empty or only stopwords
                    st.info(f"📝 Not enough text content to generate word cloud. The text may contain only common words (stopwords) that are filtered out.")
                    wordcloud = None

                if wordcloud is not None:
                    # Create matplotlib figure
                    fig, ax = plt.subplots(figsize=(12, 6))
                    # Use to_image() PIL method for numpy compatibility
                    ax.imshow(wordcloud.to_image(), interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout()

                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)  # Clean up to prevent memory leaks

                    st.caption("Word cloud generated from all response text")
                elif wordcloud_text.strip():
                    # Text exists but wordcloud failed for other reason
                    st.warning("⚠️ Word cloud generation failed. PIL library may not be available.")
            elif PIL_AVAILABLE and PILWordCloud is not None:
                # Fallback: generate word cloud even if viz_data didn't have it pre-computed
                try:
                    results_df = st.session_state.results_df
                    text_column = viz_data.get('text_column', 'response')
                    all_text = ' '.join(results_df[text_column].dropna().astype(str).tolist())
                    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', all_text.lower())

                    wordcloud = PILWordCloud(
                        width=800,
                        height=400,
                        background_color='white',
                        max_words=100,
                        min_font_size=10,
                        max_font_size=100
                    ).generate(cleaned_text)

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.imshow(wordcloud.to_image(), interpolation='bilinear')
                    ax.axis('off')
                    plt.tight_layout()

                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    st.caption("Word cloud generated from all response text (using PIL fallback)")
                except Exception as e:
                    st.warning(f"⚠️ Word cloud generation failed: {str(e)}")
            else:
                st.warning("⚠️ Word cloud not available. Please ensure the `wordcloud` or `PIL` package is installed.")

        # Sub-tab 2: Semantic Word Clouds by Topic
        with wc_tab2:
            st.markdown("#### Semantic Word Clouds by Topic")

            with st.expander("ℹ️ What am I seeing?", expanded=False):
                st.markdown("""
                **What this shows:** Word clouds for each discovered topic/cluster with **semantic coloring**.

                **How to interpret:**
                - **Word SIZE** = frequency (larger = more common in that topic)
                - **Word COLOR** = semantic similarity (words with similar colors have similar meanings)
                - Each topic has its own **unique color scheme** (Blues, Oranges, Greens, etc.)
                - Within each topic, similar shades indicate semantically related words

                **Example:** In a topic about "work", words like "job", "career", and "employment"
                would have similar color shades because they're semantically related.
                """)

            if WORDCLOUD_AVAILABLE or PIL_AVAILABLE:
                try:
                    results_df = st.session_state.results_df
                    text_column = viz_data.get('text_column', 'response')

                    # Create visualizer
                    visualizer = MethodVisualizer(coder, results_df, text_column)

                    # Generate semantic wordclouds
                    with st.spinner("Generating semantic word clouds (analyzing word meanings)..."):
                        semantic_fig = visualizer.create_all_semantic_wordclouds(
                            max_words=40,
                            cols=3
                        )

                    if semantic_fig:
                        st.pyplot(semantic_fig, use_container_width=True)
                        plt.close(semantic_fig)

                        st.caption(
                            "💡 **Color Legend:** Words with similar colors within each topic "
                            "have similar semantic meanings. Different topics use different color schemes."
                        )

                        # Option to view individual topic wordclouds
                        st.markdown("---")
                        st.markdown("##### View Individual Topic")

                        n_clusters = len(set(visualizer.assignments))
                        topic_options = {}
                        for i in range(n_clusters):
                            code_id = f"CODE_{i + 1:02d}"
                            if code_id in coder.codebook:
                                label = coder.codebook[code_id].get('label', f'Topic {i + 1}')
                            else:
                                label = f'Topic {i + 1}'
                            topic_options[f"{label} (Topic {i + 1})"] = i

                        selected_topic = st.selectbox(
                            "Select a topic for detailed view:",
                            options=list(topic_options.keys()),
                            key="semantic_wc_topic_select"
                        )

                        if selected_topic:
                            topic_id = topic_options[selected_topic]
                            individual_fig = visualizer.create_semantic_wordcloud(
                                cluster_id=topic_id,
                                max_words=60,
                                width=1000,
                                height=500
                            )
                            if individual_fig:
                                st.pyplot(individual_fig, use_container_width=True)
                                plt.close(individual_fig)
                    else:
                        st.info("Unable to generate semantic wordclouds. This may happen with very small datasets.")
                except Exception as e:
                    st.error(f"Error generating semantic wordclouds: {str(e)}")
                    st.info("Falling back to simple wordclouds in the next tab.")
            else:
                st.warning(
                    "⚠️ Semantic word clouds require additional packages. "
                    "Please ensure `wordcloud` or `PIL` and `gensim` are installed."
                )

        # Sub-tab 3: Simple Topic Word Clouds (fallback)
        with wc_tab3:
            st.markdown("#### Simple Topic Word Clouds")

            with st.expander("ℹ️ What am I seeing?", expanded=False):
                st.markdown("""
                **What this shows:** Word clouds for each topic using standard viridis coloring.

                **How to interpret:** Larger words = more frequent within that topic.
                Colors are random and do not indicate meaning.
                """)

            if WORDCLOUD_AVAILABLE or PIL_AVAILABLE:
                try:
                    results_df = st.session_state.results_df
                    text_column = viz_data.get('text_column', 'response')

                    visualizer = MethodVisualizer(coder, results_df, text_column)
                    simple_fig = visualizer.create_all_cluster_wordclouds(
                        max_words=30,
                        cols=3
                    )

                    if simple_fig:
                        st.pyplot(simple_fig, use_container_width=True)
                        plt.close(simple_fig)
                    else:
                        st.info("Unable to generate wordclouds.")
                except Exception as e:
                    st.error(f"Error generating wordclouds: {str(e)}")
            else:
                st.warning("⚠️ Word clouds not available. Please ensure `wordcloud` or `PIL` is installed.")

    # =========================================================================
    # TAB 6: Sunburst Chart (hierarchical code structure)
    # =========================================================================
    with tab6:
        st.markdown("### Code Hierarchy Sunburst")

        with st.expander("ℹ️ What am I seeing?", expanded=False):
            st.markdown("""
            **What this shows:** Hierarchical view of all codes as a sunburst chart.

            **How to interpret:** Arc size = code frequency. Click to zoom into sections.
            """)

        sunburst_data = viz_data.get('sunburst_data', [])

        if sunburst_data:
            # Build sunburst DataFrame
            sunburst_df = pd.DataFrame(sunburst_data)

            # Add root node
            root_df = pd.DataFrame([{
                'id': 'All Codes',
                'label': 'All Codes',
                'parent': '',
                'value': viz_data['sunburst_total'],
                'confidence': 0
            }])
            sunburst_df = pd.concat([root_df, sunburst_df], ignore_index=True)

            fig = px.sunburst(
                sunburst_df,
                ids='id',
                names='label',
                parents='parent',
                values='value',
                color='confidence',
                color_continuous_scale='Viridis',
                title='Code Distribution Hierarchy'
            )
            fig.update_layout(height=600)
            fig.update_traces(
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Confidence: %{color:.2f}<extra></extra>'
            )

            st.plotly_chart(fig, use_container_width=True, key="sunburst_chart")

            # Summary stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Codes", len(sunburst_data))
            col2.metric("Total Assignments", viz_data['sunburst_total'])
            avg_conf = np.mean([d['confidence'] for d in sunburst_data]) if sunburst_data else 0
            col3.metric("Avg Confidence", f"{avg_conf:.2f}")
        else:
            st.info("No code data available for sunburst chart.")

    # =========================================================================
    # TAB 7: Cluster Network Diagram
    # =========================================================================
    with tab7:
        st.markdown("### Cluster Network Diagram")

        with st.expander("ℹ️ What am I seeing?", expanded=False):
            st.markdown("""
            **What this shows:** Network diagram showing relationships between clusters/topics.

            **How to interpret:**
            - **Nodes:** Each node represents a cluster/topic. Larger nodes = more documents.
            - **Edges:** Lines connect similar clusters. Thicker lines = higher similarity.
            - **Layout:** Similar clusters are positioned closer together.
            """)

        if METHOD_VISUALIZER_AVAILABLE:
            try:
                results_df = st.session_state.results_df
                text_column = viz_data.get('text_column', 'response')

                visualizer = MethodVisualizer(coder, results_df, text_column)

                # Layout selection
                col1, col2 = st.columns([3, 1])
                with col2:
                    network_layout = st.selectbox(
                        "Layout Algorithm",
                        options=['spring', 'circular', 'kamada_kawai'],
                        format_func=lambda x: {
                            'spring': 'Spring (Force-directed)',
                            'circular': 'Circular',
                            'kamada_kawai': 'Kamada-Kawai'
                        }[x],
                        key="network_layout_select"
                    )

                with st.spinner("Generating network diagram..."):
                    network_fig = visualizer.create_cluster_network(layout=network_layout)

                if network_fig is not None:
                    st.plotly_chart(network_fig, use_container_width=True, key="network_chart")
                    st.caption(
                        "Node size represents document count. "
                        "Edge thickness indicates inter-cluster similarity based on centroid cosine similarity."
                    )
                else:
                    st.warning("Could not generate network diagram. Need at least 2 clusters.")
            except Exception as e:
                st.error(f"Error generating network diagram: {str(e)}")
        else:
            st.warning("Network diagram requires the method visualizations module.")

    # Next button
    render_next_button("💾 Export Results")


def page_export_results():
    """Export results page."""
    st.markdown('<h2 class="sub-header">💾 Export Results</h2>', unsafe_allow_html=True)

    if not st.session_state.analysis_complete:
        st.warning("⚠️ Please run the analysis first")
        return

    coder = st.session_state.coder
    results_df = st.session_state.results_df

    st.markdown("### 📦 Available Exports")

    # Export options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Complete Results Package")
        st.markdown("""
        Includes:
        - Code assignments
        - Complete codebook
        - Frequency tables
        - Co-occurrence analysis
        """)

        if st.button("📥 Download Excel Package", use_container_width=True):
            try:
                excel_data = export_results_package(coder, results_df, format='excel')

                st.download_button(
                    label="Download",
                    data=excel_data,
                    file_name=f"coding_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.success("✅ Excel package ready for download!")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

    with col2:
        st.markdown("#### 📋 Individual Components")

        # Code assignments
        assignments_csv = results_df.to_csv(index=False).encode()
        st.download_button(
            label="📄 Code Assignments (CSV)",
            data=assignments_csv,
            file_name="code_assignments.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Codebook with representative quotes and LLM-enhanced labels
        codebook_data = []
        for code_id, info in coder.codebook.items():
            # Get up to 5 representative quotes (or all if fewer)
            examples = info.get('examples', [])[:5]
            quotes = [ex['text'][:200] for ex in examples]
            # Pad with empty strings if fewer than 5 quotes
            while len(quotes) < 5:
                quotes.append('')

            # Get sentiment distribution if available
            sentiment_positive = ''
            sentiment_negative = ''
            sentiment_neutral = ''
            has_mixed_sentiment = ''
            if hasattr(coder, 'cluster_interpretation') and coder.cluster_interpretation is not None:
                cluster_id = code_id.replace('CODE_', 'CLUSTER_')
                if cluster_id in coder.cluster_interpretation.summaries:
                    summary = coder.cluster_interpretation.summaries[cluster_id]
                    if summary.sentiment_distribution is not None:
                        sentiment_positive = f"{summary.sentiment_distribution.get('positive', 0):.1%}"
                        sentiment_negative = f"{summary.sentiment_distribution.get('negative', 0):.1%}"
                        sentiment_neutral = f"{summary.sentiment_distribution.get('neutral', 0):.1%}"
                        has_mixed_sentiment = 'Yes' if summary.has_mixed_sentiment else 'No'

            row = {
                'Code': code_id,
                'Label': info.get('llm_label', info['label']),  # Use LLM label if available
                'Term-Based Label': info['label'],
                'Alternative Labels': ', '.join(info.get('alternative_labels', [])[:3]),
                'LLM Description': info.get('llm_description', ''),
                'Keywords': ', '.join(info['keywords']),
                'Count': info['count'],
                'Avg Confidence': info['avg_confidence'],
                'Sentiment Positive': sentiment_positive,
                'Sentiment Negative': sentiment_negative,
                'Sentiment Neutral': sentiment_neutral,
                'Mixed Sentiment': has_mixed_sentiment,
                'Quote 1': quotes[0],
                'Quote 2': quotes[1],
                'Quote 3': quotes[2],
                'Quote 4': quotes[3],
                'Quote 5': quotes[4]
            }
            codebook_data.append(row)
        codebook_df = pd.DataFrame(codebook_data)
        codebook_csv = codebook_df.to_csv(index=False).encode()

        st.download_button(
            label="📖 Codebook (CSV)",
            data=codebook_csv,
            file_name="codebook.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Generate summary report
    st.markdown("---")
    st.markdown("### 📝 Generate Summary Report")

    if st.button("Generate Executive Summary", use_container_width=True):
        try:
            summary = get_analysis_summary(coder, results_df)

            st.success("✅ Executive summary generated successfully!")

            st.markdown(summary)

            # Download as markdown
            summary_bytes = summary.encode()
            st.download_button(
                label="Download Summary (Markdown)",
                data=summary_bytes,
                file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"❌ Failed to generate summary: {str(e)}")


def page_about():
    """About page."""
    st.markdown('<h2 class="sub-header">ℹ️ About</h2>', unsafe_allow_html=True)

    st.markdown("""
    ## ML-Based Open Coding Analysis

    This tool provides **automatic qualitative data analysis** using machine learning algorithms.

    ### 🎯 Core Features

    - **Automatic Theme Discovery**: ML algorithms discover themes and patterns in your data
    - **Multiple ML Algorithms**: TF-IDF+K-Means, LDA, LSTM, BERT, and SVM-based clustering
    - **Confidence Scoring**: Every code assignment includes a confidence score
    - **LLM-Enhanced Labels**: Auto-generated code labels refined by language models
    - **15 Essential Outputs**: Complete analysis package for researchers
    - **Interactive Visualizations**: Explore data with charts, word clouds, and network diagrams
    - **Multiple Export Formats**: Download results in CSV, Excel, or Markdown

    ### 🔧 Text Processor

    The **Text Processor** provides comprehensive preprocessing before analysis:

    - **Quick Presets**: One-click configurations for different data types:
      - Survey Responses (feedback, open-ended questions, forms)
      - Social Media (Twitter/X with slang, emojis, hashtags)
      - Long-form Reviews (elongated expressions, informal language)
      - News Articles (formal, minimal preprocessing)

    - **Enhanced Preprocessing Options**:
      - **Negation Preservation**: Keeps words like "not", "never" for accurate sentiment/topic analysis
      - **Domain Stopwords**: Removes survey-specific noise words like "response", "survey", "participant"

    - **Gold Standard Processing**: Fine-grained control including:
      - Unicode normalization, HTML entity decoding
      - URL/mention/hashtag standardization
      - Contraction and slang expansion
      - Elongation normalization ("loooove" → "love")
      - Spam and duplicate detection

    - **Advanced NLTK Processing**: Stemming, lemmatization, custom stopwords, domain-specific cleaning

    ### 📊 Sentiment Analysis

    Integrated sentiment analysis with data-type-specific models:

    - **Survey Responses**: VADER (rule-based, fast)
    - **Social Media (Twitter/X)**: Twitter-RoBERTa (transformer-based, handles slang/emojis)
    - **Long-form Reviews**: Review-BERT (transformer-based, nuanced sentiment)

    Results include positive/neutral/negative classification with confidence scores.

    ### 🤖 Supported ML Algorithms

    | Algorithm | Description | Best For |
    |-----------|-------------|----------|
    | **TF-IDF + K-Means** | Fast bag-of-words clustering | General use, quick exploration |
    | **LDA** | Probabilistic topic modeling | Overlapping themes, academic research |
    | **LSTM + K-Means** | Sequential pattern recognition | Order-dependent text, narratives |
    | **BERT + K-Means** | Semantic embedding clustering | Nuanced meaning, synonyms |
    | **SVM Spectral** | Kernel-based clustering | Complex, non-linear boundaries |

    ### 📈 Visualizations

    - **Code Frequency Charts**: Bar charts showing theme prevalence
    - **Co-occurrence Heatmaps**: Which codes appear together
    - **Word Clouds**: Overall and per-topic semantic word clouds
    - **Sunburst Charts**: Hierarchical code distribution
    - **Network Diagrams**: Cluster relationships and similarity
    - **Confidence Distributions**: Quality assessment histograms

    ### 📊 15 Essential Outputs

    1. Code Assignments with confidence scores
    2. Auto-generated Codebook with LLM-enhanced labels
    3. Code Frequency Tables
    4. Quality Metrics (silhouette score, coverage)
    5. Binary Matrix for statistical analysis
    6. Representative Quotes per code
    7. Co-Occurrence Analysis
    8. Descriptive Statistics
    9. Segmentation Analysis
    10. QA Report with validity assessment
    11. Interactive Visualizations
    12. Multiple Export Formats (Excel, CSV, Markdown)
    13. Method Documentation (reproducibility)
    14. Uncoded Response Detection
    15. Executive Summary

    ### 🚀 Getting Started

    1. **Upload Data**: Load CSV/Excel or select a sample dataset
    2. **Preprocess (Optional)**: Use Text Processor for data cleaning
    3. **Configure**: Choose algorithm, code count, and enable sentiment analysis
    4. **Run Analysis**: ML discovers themes with progress tracking
    5. **Explore Results**: View codes, insights, sentiment, and visualizations
    6. **Export**: Download complete results package

    ### 📋 Data Requirements

    - **Format**: CSV or Excel (.xlsx, .xls)
    - **Required**: At least one text column with responses
    - **Recommended**: 20+ responses for reliable analysis
    - **Language**: English (non-English text is filtered)
    """)


if __name__ == "__main__":
    main()
