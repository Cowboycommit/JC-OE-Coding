"""
Streamlit UI for ML-Based Open Coding Analysis

A comprehensive web interface for automatic qualitative data analysis
using machine learning algorithms.
"""

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
import networkx as nx


# Cached computation functions to prevent UI hangs on visualization page
# Using hash_funcs to properly cache DataFrame-derived computations

def _get_data_hash(results_df):
    """Generate a hash key for caching based on DataFrame content."""
    # Use shape and first/last values as a quick hash proxy
    return (len(results_df), tuple(results_df.columns), id(results_df))


@st.cache_data(show_spinner=False)
def compute_cooccurrence_matrix(data_hash, _results_df, codes_list):
    """Compute co-occurrence matrix with caching."""
    n = len(codes_list)
    code_to_idx = {code: i for i, code in enumerate(codes_list)}
    cooccur = np.zeros((n, n))

    for assigned_codes in _results_df['assigned_codes']:
        for code1, code2 in combinations(assigned_codes, 2):
            if code1 in code_to_idx and code2 in code_to_idx:
                i, j = code_to_idx[code1], code_to_idx[code2]
                cooccur[i, j] += 1
                cooccur[j, i] += 1

        for code in assigned_codes:
            if code in code_to_idx:
                cooccur[code_to_idx[code], code_to_idx[code]] += 1

    return cooccur


@st.cache_data(show_spinner=False)
def compute_edge_weights(data_hash, _results_df):
    """Compute network edge weights with caching."""
    edge_weights = {}
    for assigned_codes in _results_df['assigned_codes']:
        for i, code1 in enumerate(assigned_codes):
            for code2 in assigned_codes[i+1:]:
                edge = tuple(sorted([code1, code2]))
                edge_weights[edge] = edge_weights.get(edge, 0) + 1
    return edge_weights


@st.cache_data(show_spinner=False)
def compute_network_layout(nodes_tuple, edges_tuple):
    """Compute spring layout with caching - this is very expensive."""
    G = nx.Graph()
    G.add_nodes_from(nodes_tuple)
    G.add_edges_from(edges_tuple)

    if len(G.nodes()) > 0 and len(G.edges()) > 0:
        return nx.spring_layout(G, k=2, iterations=50, seed=42)
    return {}


@st.cache_data(show_spinner=False)
def get_text_column(columns_tuple):
    """Get text column name with caching."""
    return [col for col in columns_tuple if col not in ['assigned_codes', 'confidence_scores', 'num_codes', 'themes']][0]


@st.cache_data(show_spinner=False)
def build_text_to_row_lookup(data_hash, _results_df, text_col):
    """Build text-to-row lookup dictionary with caching (replaces slow iterrows)."""
    # Use vectorized approach instead of iterrows
    return dict(zip(_results_df[text_col], _results_df.to_dict('records')))


@st.cache_data(show_spinner=False)
def compute_code_frequency(data_hash, _results_df, _codebook):
    """Compute code frequency for theme prevalence fallback."""
    code_freq = {}
    for codes in _results_df['assigned_codes']:
        for code in codes:
            if code in _codebook:
                label = _codebook[code]['label']
                code_freq[label] = code_freq.get(label, 0) + 1
    return code_freq


@st.cache_data(show_spinner=False)
def compute_all_confidences(data_hash, _results_df):
    """Flatten all confidence scores with caching."""
    return [conf for confs in _results_df['confidence_scores'] for conf in confs]


@st.cache_data(show_spinner=False)
def build_feature_name_index(_feature_names_tuple):
    """Build feature name to index mapping for O(1) lookups."""
    return {name: idx for idx, name in enumerate(_feature_names_tuple)}


@st.cache_data(show_spinner=False)
def compute_coherence_metrics(model_id, _feature_matrix, _labels):
    """Compute coherence metrics with caching - these are expensive O(n²) operations."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    metrics = {}
    try:
        if len(set(_labels)) > 1:
            # Convert sparse matrix to dense if needed for some metrics
            dense_matrix = _feature_matrix.toarray() if hasattr(_feature_matrix, 'toarray') else _feature_matrix

            metrics['Silhouette Score'] = silhouette_score(_feature_matrix, _labels)
            metrics['Calinski-Harabasz Score'] = calinski_harabasz_score(dense_matrix, _labels)
            metrics['Davies-Bouldin Score'] = davies_bouldin_score(dense_matrix, _labels)
    except Exception:
        pass
    return metrics


@st.cache_data(show_spinner=False)
def get_cluster_labels(model_id, _model, _feature_matrix):
    """Get cluster labels from model with caching."""
    if hasattr(_model, 'labels_'):
        return _model.labels_
    elif hasattr(_model, 'components_'):  # LDA/NMF
        doc_topic_matrix = _model.transform(_feature_matrix)
        return doc_topic_matrix.argmax(axis=1)
    else:
        return _model.predict(_feature_matrix)


@st.cache_data(show_spinner=False)
def get_top_codes_cached(codebook_hash, _coder, n):
    """Cached wrapper for get_top_codes to avoid recalculation on every render."""
    return get_top_codes(_coder, n=n)


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
        {"name": "⚙️ Configuration", "step": 2, "requires": "data"},
        {"name": "🚀 Run Analysis", "step": 3, "requires": "config"},
        {"name": "📊 Results Overview", "step": 4, "requires": "analysis"},
        {"name": "📈 Visualizations", "step": 5, "requires": "analysis"},
        {"name": "💾 Export Results", "step": 6, "requires": "analysis"},
        {"name": "ℹ️ About", "step": 7, "requires": None},
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
        elif page["step"] == 2 and has_config:
            is_completed = True
        elif page["step"] == 3 and has_analysis:
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

    # Define available sample datasets
    sample_datasets = {
        "Remote Work Experiences": "data/sample_responses.csv",
        "Fashion Industry Perspectives": "data/fashion_responses.csv",
        "Cricket Commentary": "data/cricket_responses.csv",
        "Cultural Commentary": "data/cultural_commentary_responses.csv",
        "Consumer Perspectives": "data/consumer_perspectives_responses.csv",
        "Industry Professional Responses": "data/industry_professional_responses.csv"
    }

    # Dropdown to select sample dataset
    selected_dataset = st.selectbox(
        "Choose a sample dataset:",
        options=list(sample_datasets.keys()),
        help="Select from pre-loaded sample datasets to explore the tool"
    )

    # Load button
    if st.button("Load Selected Dataset", use_container_width=True):
        try:
            dataset_path = sample_datasets[selected_dataset]
            df = pd.read_csv(dataset_path)

            st.session_state.uploaded_df = df
            st.success("Selected data loaded")
            st.success(f"✅ {selected_dataset} loaded successfully! ({len(df)} responses)")
            st.info("👉 **Next step:** Go to '⚙️ Configuration' in the sidebar to select your text column and set up analysis parameters.")
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
            options=['tfidf_kmeans', 'lda', 'nmf'],
            index=0,
            format_func=lambda x: {
                'tfidf_kmeans': 'TF-IDF + K-Means (Fast, Recommended)',
                'lda': 'Latent Dirichlet Allocation (Topic Modeling)',
                'nmf': 'Non-negative Matrix Factorization'
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
            'nmf': {
                'description': "**Non-negative Matrix Factorization** decomposes text into parts-based representations. Produces sparse, interpretable results.",
                'runtime': "⚡ **Fast** (~5-15s for 1000 responses)",
                'good_for': "Good for sparse, interpretable results",
                'watch_out': "Watch out for sensitivity to n_codes parameter"
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

        # Advanced options
        with st.expander("🔧 Advanced Options"):
            stop_words = st.selectbox(
                "Stop words language",
                options=['english', 'spanish', 'french', 'german'],
                index=0
            )

    # Save configuration
    st.session_state.config = {
        'text_column': selected_column,
        'n_codes': n_codes,
        'auto_optimal_codes': auto_optimal_codes,
        'method': method,
        'min_confidence': min_confidence,
        'stop_words': stop_words
    }

    # Show configuration summary
    st.markdown("---")
    st.markdown("### 📋 Configuration Summary")

    config_col1, config_col2, config_col3 = st.columns(3)

    with config_col1:
        st.metric("Responses", f"{len(df):,}")
    with config_col2:
        st.metric("Codes to Find", "Auto" if auto_optimal_codes else n_codes)
    with config_col3:
        st.metric("Algorithm", method.upper())

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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Responses", f"{len(df):,}")
    with col2:
        st.metric("Text Column", config['text_column'])
    with col3:
        st.metric("Codes", "Auto" if auto_optimal else config['n_codes'])
    with col4:
        st.metric("Method", config['method'].upper())

    if auto_optimal:
        st.info("🔍 The algorithm will automatically determine the optimal number of codes before running the analysis.")

    st.markdown("---")

    # Run button
    if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
        # Define analysis stages
        stages = [
            "Data Preparation",
            "Feature Extraction",
            "Clustering/Modeling",
            "Code Labeling",
            "Generating Insights"
        ]

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

                    st.success(f"✨ Optimal number of codes determined: **{optimal_n}** (silhouette score: {optimal_results['best_silhouette_score']:.4f})")
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

            # Save to session state
            st.session_state.coder = coder
            st.session_state.results_df = results_df
            st.session_state.metrics = metrics
            st.session_state.analysis_complete = True

            # Mark all stages complete
            stage_checklist.markdown(update_stage_checklist(stages, len(stages)), unsafe_allow_html=True)

            # Clear progress
            progress_bar.empty()
            status_text.empty()

            # Show success
            execution_time = time.time() - start_time

            st.markdown(f"""
            <div class="success-box">
            <h3>✅ Analysis Complete!</h3>
            <p><strong>Execution Time:</strong> {format_duration(execution_time)}</p>
            <p><strong>Codes Found:</strong> {metrics['n_codes']}</p>
            <p><strong>Total Assignments:</strong> {metrics.get('total_assignments', 0):,}</p>
            <p><strong>Coverage:</strong> {metrics.get('coverage_pct', 0):.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Show quick insights
            st.markdown("### 🔍 Quick Insights")
            insights = generate_insights(coder, results_df)
            for insight in insights:
                st.markdown(insight)

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

    # Show previous results if available
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.markdown("""
        <div class="success-box">
        <h3>✅ Previous Analysis Available</h3>
        <p>Your analysis results are ready! Navigate to <strong>"📊 Results Overview"</strong> in the sidebar to view them.</p>
        </div>
        """, unsafe_allow_html=True)

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

    # Tabs for Insights, Top Codes, and Assignments
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["💡 Insights", "🏆 Top Codes", "📋 Assignments"])

    with tab1:
        # Key insights
        insights = generate_insights(coder, results_df)
        for insight in insights:
            st.markdown(insight)

    with tab2:
        # Top codes (using cached version)
        codebook_hash = len(coder.codebook)  # Simple hash based on codebook size
        top_codes_df = get_top_codes_cached(codebook_hash, coder, n=10)

        # Display as styled table
        st.dataframe(
            style_frequency_table(top_codes_df),
            use_container_width=True,
            height=400
        )

    with tab3:
        # Code assignments with uncertainty filter
        st.markdown("#### Sample Code Assignments")

        # Add checkbox for filtering uncertain rows
        show_uncertain_only = st.checkbox(
            "Show only uncertain rows (low confidence)",
            value=False,
            help="Filter to show only rows where maximum confidence < 0.5"
        )

        # Prepare assignments dataframe with confidence
        assignments_df = results_df.copy()

        # Calculate max confidence for each row
        if 'confidence_scores' in assignments_df.columns:
            assignments_df['max_confidence'] = assignments_df['confidence_scores'].apply(
                lambda x: max(x) if x and len(x) > 0 else 0.0
            )
        else:
            assignments_df['max_confidence'] = 1.0

        # Sort by confidence (ascending - lowest confidence first)
        assignments_df = assignments_df.sort_values('max_confidence', ascending=True)

        # Apply filter if checkbox is selected
        if show_uncertain_only:
            assignments_df = assignments_df[assignments_df['max_confidence'] < 0.5]

        # Select columns and limit rows
        sample_size = min(20, len(assignments_df))
        display_cols = [
            st.session_state.config['text_column'],
            'assigned_codes',
            'num_codes',
            'max_confidence'
        ]
        sample_df = assignments_df[display_cols].head(sample_size)

        # Format for display
        display_df = sample_df.copy()
        display_df['assigned_codes'] = display_df['assigned_codes'].apply(
            lambda x: ', '.join(x) if x else 'None'
        )
        display_df['max_confidence'] = display_df['max_confidence'].apply(
            lambda x: f"{x:.3f}"
        )

        if len(display_df) > 0:
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.info("No uncertain rows found (all rows have confidence ≥ 0.5)")

    # Detailed codebook
    st.markdown("---")
    st.markdown("### 📖 Complete Codebook")

    for code_id, info in sorted(coder.codebook.items(), key=lambda x: x[1]['count'], reverse=True):
        if info['count'] > 0:  # Only show active codes
            with st.expander(f"**{code_id}**: {info['label']} ({info['count']} responses)"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**Keywords:** {', '.join(info['keywords'][:10])}")

                    # Show examples
                    if info['examples']:
                        st.markdown("**Example Responses:**")
                        for i, example in enumerate(info['examples'][:3], 1):
                            st.text(f"{i}. {truncate_text(example['text'], 100)} [{example['confidence']:.2f}]")

                with col2:
                    st.metric("Count", f"{info['count']:,}")
                    st.metric("Avg Confidence", f"{info['avg_confidence']:.2f}")

    # Next button - always show on this page if analysis is complete
    render_next_button("📈 Visualizations")


def page_visualizations():
    """Visualizations page."""
    st.markdown('<h2 class="sub-header">📈 Visualizations</h2>', unsafe_allow_html=True)

    if not st.session_state.analysis_complete:
        st.warning("⚠️ Please run the analysis first")
        return

    coder = st.session_state.coder
    results_df = st.session_state.results_df

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Frequency",
        "🔥 Heatmap",
        "🕸️ Network",
        "📉 Distribution",
        "🎯 Confidence",
        "🏷️ Topic-Term",
        "📊 Theme Prevalence",
        "🗺️ 2D Embedding",
        "🔬 Coherence",
        "💬 Quotes"
    ])

    with tab1:
        st.markdown("### Code Frequency Distribution")

        # Explanation expander
        with st.expander("ℹ️ What am I seeing?", expanded=False):
            st.markdown("""
            **What this shows:** Bar chart of the top 15 most frequent codes discovered in your data.

            **How to interpret:** Taller bars = more responses assigned to that code. Color intensity shows average confidence.

            **Key things to look for:**
            - Dominant codes that capture a large portion of responses
            - Codes with high confidence (darker colors) vs low confidence
            - Even vs uneven distribution across codes
            """)

        codebook_hash = len(coder.codebook)
        top_codes_df = get_top_codes_cached(codebook_hash, coder, n=15)

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

        st.plotly_chart(fig, use_container_width=True)

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            csv_data = top_codes_df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="code_frequency.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            try:
                img_bytes = fig.to_image(format="png", width=1200, height=600)
                st.download_button(
                    label="📥 Download PNG",
                    data=img_bytes,
                    file_name="code_frequency.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception:
                st.caption("PNG export requires kaleido: `pip install kaleido`")

    with tab2:
        st.markdown("### Co-occurrence Heatmap")

        # Explanation and legend
        with st.expander("ℹ️ What am I seeing?", expanded=False):
            st.markdown("""
            **What this shows:** Matrix showing how often codes appear together in the same response.

            **How to interpret:**
            - **Rows/Columns:** Each row and column represents a discovered code
            - **Cell color:** Intensity indicates co-occurrence count (darker = more frequent)
            - **Diagonal:** Shows total count for each code (self-occurrence)

            **Key things to look for:**
            - Hot spots (dark cells) show codes that frequently appear together
            - Patterns may reveal thematic clusters or related concepts
            - Isolated codes (light row/column) may be distinct themes
            """)

        # Build co-occurrence matrix using cached function
        codes = list(coder.codebook.keys())
        data_hash = _get_data_hash(results_df)
        cooccur = compute_cooccurrence_matrix(data_hash, results_df, codes)

        labels = [coder.codebook[c]['label'] for c in codes]

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

        st.plotly_chart(fig, use_container_width=True)

        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            cooccur_df = pd.DataFrame(cooccur, index=labels, columns=labels)
            csv_data = cooccur_df.to_csv().encode()
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="cooccurrence_matrix.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            try:
                img_bytes = fig.to_image(format="png", width=1200, height=800)
                st.download_button(
                    label="📥 Download PNG",
                    data=img_bytes,
                    file_name="cooccurrence_heatmap.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception:
                st.caption("PNG export requires kaleido")

        # Co-occurrence pairs table
        st.markdown("#### Top Co-occurring Pairs")
        pairs_df = get_cooccurrence_pairs(results_df, min_count=2)
        if not pairs_df.empty:
            st.dataframe(pairs_df.head(10), use_container_width=True)
        else:
            st.info("No significant co-occurrences found")

    with tab3:
        st.markdown("### Code Network Diagram")

        # Explanation expander
        with st.expander("ℹ️ What am I seeing?", expanded=False):
            st.markdown("""
            **What this shows:** Interactive network graph where codes are nodes and connections show co-occurrence.

            **How to interpret:**
            - **Nodes (circles):** Each node represents a code. Size = frequency, color = confidence.
            - **Edges (lines):** Connections between codes that appear together in responses.
            - **Clustering:** Codes that frequently co-occur will cluster together.

            **Key things to look for:**
            - Central nodes with many connections (hub themes)
            - Isolated nodes (distinct, standalone themes)
            - Dense clusters (related concepts that often appear together)
            """)

        try:
            # Build network graph from co-occurrences
            codes = [c for c in coder.codebook.keys() if coder.codebook[c]['count'] > 0]

            # Create graph
            G = nx.Graph()

            # Add nodes with attributes
            for code in codes:
                G.add_node(
                    code,
                    label=coder.codebook[code]['label'],
                    count=coder.codebook[code]['count'],
                    confidence=coder.codebook[code]['avg_confidence']
                )

            # Compute edge weights using cached function
            data_hash = _get_data_hash(results_df)
            edge_weights = compute_edge_weights(data_hash, results_df)

            # Preset buttons for threshold
            st.markdown("**Network density presets:**")
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            max_weight = max(edge_weights.values()) if edge_weights else 5

            # Calculate preset values
            sparse_val = max(1, int(max_weight * 0.6))
            medium_val = max(1, int(max_weight * 0.3))
            dense_val = 1

            with preset_col1:
                if st.button("🔵 Sparse", help=f"Threshold: {sparse_val}+ (fewer connections)", use_container_width=True):
                    st.session_state.network_threshold = sparse_val
            with preset_col2:
                if st.button("🟢 Medium", help=f"Threshold: {medium_val}+ (balanced)", use_container_width=True):
                    st.session_state.network_threshold = medium_val
            with preset_col3:
                if st.button("🟠 Dense", help=f"Threshold: {dense_val}+ (all connections)", use_container_width=True):
                    st.session_state.network_threshold = dense_val

            # Get threshold from session state or default
            default_threshold = st.session_state.get('network_threshold', 2)

            min_cooccurrence = st.slider(
                "Minimum co-occurrence threshold",
                min_value=1,
                max_value=max_weight,
                value=min(default_threshold, max_weight),
                help="Only show connections between codes that appear together this many times"
            )

            # Filter edges by threshold
            filtered_edges = [(code1, code2, weight) for (code1, code2), weight in edge_weights.items()
                              if weight >= min_cooccurrence and code1 in G and code2 in G]

            for code1, code2, weight in filtered_edges:
                G.add_edge(code1, code2, weight=weight)

            if len(G.nodes()) > 0 and len(G.edges()) > 0:
                # Use cached spring layout for positioning
                nodes_tuple = tuple(sorted(G.nodes()))
                edges_tuple = tuple(sorted((e[0], e[1]) for e in G.edges()))
                pos = compute_network_layout(nodes_tuple, edges_tuple)

                # Create edge trace
                edge_x = []
                edge_y = []
                edge_weights_list = []

                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights_list.append(edge[2]['weight'])

                edge_trace = go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    line=dict(width=1.5, color='#888'),
                    hoverinfo='none',
                    mode='lines',
                    showlegend=False
                )

                # Create node trace
                node_x = []
                node_y = []
                node_text = []
                node_size = []
                node_color = []

                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)

                    label = G.nodes[node]['label']
                    count = G.nodes[node]['count']
                    confidence = G.nodes[node]['confidence']

                    node_text.append(f"{label}<br>Count: {count}<br>Confidence: {confidence:.2f}")
                    node_size.append(np.sqrt(count) * 10)
                    node_color.append(confidence)

                node_trace = go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=[G.nodes[node]['label'] for node in G.nodes()],
                    textposition="top center",
                    textfont=dict(size=10),
                    hovertext=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='Viridis',
                        size=node_size,
                        color=node_color,
                        colorbar=dict(
                            title="Avg<br>Confidence",
                            thickness=15,
                            xanchor='left',
                            titleside='right'
                        ),
                        line=dict(width=2, color='white')
                    ),
                    showlegend=False
                )

                # Create figure
                fig = go.Figure(data=[edge_trace, node_trace])

                fig.update_layout(
                    title="Code Co-occurrence Network<br><sub>Node size = frequency, Color = confidence, Edges = co-occurrence</sub>",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20, l=5, r=5, t=60),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=600,
                    plot_bgcolor='white'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Network statistics
                st.markdown("#### Network Statistics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Nodes (Codes)", len(G.nodes()))
                with col2:
                    st.metric("Edges (Connections)", len(G.edges()))
                with col3:
                    if len(G.nodes()) > 0:
                        density = nx.density(G)
                        st.metric("Network Density", f"{density:.3f}")
                with col4:
                    if len(G.edges()) > 0:
                        avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
                        st.metric("Avg Connections", f"{avg_degree:.1f}")

            elif len(G.nodes()) > 0:
                st.warning(f"⚠️ No co-occurrences found with threshold of {min_cooccurrence}. Try lowering the threshold.")
            else:
                st.info("No active codes to visualize")

        except ImportError:
            st.error("❌ NetworkX library not installed. Please install it with: pip install networkx")

            # Fallback: Simple scatter plot
            st.markdown("### Alternative View: Code Distribution")
            codebook_hash = len(coder.codebook)
            top_codes_df = get_top_codes_cached(codebook_hash, coder, n=20)

            fig = px.scatter(
                top_codes_df,
                x='Count',
                y='Avg Confidence',
                size='Count',
                color='Avg Confidence',
                hover_data=['Label', 'Keywords'],
                title='Code Distribution: Count vs Confidence',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)

            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Distribution of Codes per Response")

        fig = px.histogram(
            results_df,
            x='num_codes',
            title='Distribution of Codes per Response',
            labels={'num_codes': 'Number of Codes', 'count': 'Frequency'},
            nbins=max(results_df['num_codes'].max(), 5)
        )
        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{results_df['num_codes'].mean():.2f}")
        with col2:
            st.metric("Median", f"{results_df['num_codes'].median():.0f}")
        with col3:
            st.metric("Max", f"{results_df['num_codes'].max():.0f}")

    with tab5:
        st.markdown("### Confidence Score Distribution")

        # Use cached function to compute all confidences
        data_hash = _get_data_hash(results_df)
        all_confidences = compute_all_confidences(data_hash, results_df)

        if all_confidences:
            fig = px.histogram(
                x=all_confidences,
                nbins=30,
                title='Distribution of Confidence Scores',
                labels={'x': 'Confidence Score', 'y': 'Frequency'}
            )
            fig.update_layout(height=400)

            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{np.mean(all_confidences):.3f}")
            with col2:
                st.metric("Median", f"{np.median(all_confidences):.3f}")
            with col3:
                st.metric("Min", f"{np.min(all_confidences):.3f}")
            with col4:
                st.metric("Max", f"{np.max(all_confidences):.3f}")
        else:
            st.info("No confidence scores available")

    with tab6:
        st.markdown("### Topic-Term Importance Plot")

        # Get top terms for each topic/code
        codes = list(coder.codebook.keys())

        # Select which codes to display
        n_codes_to_show = st.slider(
            "Number of topics to display",
            min_value=3,
            max_value=min(len(codes), 15),
            value=min(len(codes), 8),
            help="Select how many topics to show"
        )

        n_terms_per_code = st.slider(
            "Number of terms per topic",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of top terms to show for each topic"
        )

        # Get top codes by frequency (cached)
        codebook_hash = len(coder.codebook)
        top_codes_df = get_top_codes_cached(codebook_hash, coder, n=n_codes_to_show)
        selected_codes = [code for code in codes if coder.codebook[code]['label'] in top_codes_df['Label'].values][:n_codes_to_show]

        # Pre-build feature name index for O(1) lookups (critical performance fix)
        feature_names = coder.vectorizer.get_feature_names_out()
        feature_name_index = build_feature_name_index(tuple(feature_names))

        # Create heatmap data for topic-term importance
        term_data = []
        for code in selected_codes:
            keywords = coder.codebook[code]['keywords'][:n_terms_per_code]
            label = coder.codebook[code]['label']

            # Get term weights from the model
            if hasattr(coder.model, 'components_'):  # LDA or NMF
                code_idx = int(code.split('_')[1]) - 1
                topic_weights = coder.model.components_[code_idx]

                for term in keywords:
                    if term in feature_name_index:
                        term_idx = feature_name_index[term]
                        weight = topic_weights[term_idx]
                        term_data.append({
                            'Topic': label,
                            'Term': term,
                            'Importance': weight
                        })
            else:  # K-Means
                code_idx = int(code.split('_')[1]) - 1
                cluster_center = coder.model.cluster_centers_[code_idx]

                for term in keywords:
                    if term in feature_name_index:
                        term_idx = feature_name_index[term]
                        weight = cluster_center[term_idx]
                        term_data.append({
                            'Topic': label,
                            'Term': term,
                            'Importance': weight
                        })

        if term_data:
            term_df = pd.DataFrame(term_data)

            # Create pivot table for heatmap
            pivot_df = term_df.pivot(index='Term', columns='Topic', values='Importance')

            fig = px.imshow(
                pivot_df,
                labels=dict(color="Importance"),
                title="Topic-Term Importance Matrix",
                color_continuous_scale='RdYlBu_r',
                aspect='auto'
            )
            fig.update_layout(height=max(400, len(pivot_df) * 20))

            st.plotly_chart(fig, use_container_width=True)

            # Also create a bar chart for top terms in selected topic
            st.markdown("#### Top Terms by Topic")
            selected_topic = st.selectbox(
                "Select a topic to see top terms",
                options=term_df['Topic'].unique()
            )

            topic_terms = term_df[term_df['Topic'] == selected_topic].sort_values('Importance', ascending=False).head(15)

            fig2 = px.bar(
                topic_terms,
                x='Importance',
                y='Term',
                orientation='h',
                title=f'Top Terms for: {selected_topic}',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig2.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Term importance data not available for the current model")

    with tab7:
        st.markdown("### Theme Prevalence Distribution")

        # Check if themes exist in results
        has_themes = 'themes' in results_df.columns if results_df is not None else False

        if has_themes and hasattr(st.session_state, 'theme_analyzer') and st.session_state.theme_analyzer:
            theme_analyzer = st.session_state.theme_analyzer

            # Get theme counts
            theme_counts = theme_analyzer.get_dominant_theme(results_df)

            if theme_counts:
                # Create DataFrame for visualization
                theme_df = pd.DataFrame([
                    {
                        'Theme': theme_analyzer.themes[theme_id]['name'],
                        'Count': count,
                        'Percentage': (count / len(results_df)) * 100
                    }
                    for theme_id, count in theme_counts.items()
                ]).sort_values('Count', ascending=False)

                # Bar chart
                fig = px.bar(
                    theme_df,
                    x='Theme',
                    y='Count',
                    title='Theme Prevalence Distribution',
                    color='Percentage',
                    color_continuous_scale='Viridis',
                    text='Count',
                    labels={'Count': 'Frequency', 'Percentage': 'Coverage %'}
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Pie chart
                fig2 = px.pie(
                    theme_df,
                    values='Count',
                    names='Theme',
                    title='Theme Distribution (Proportional)',
                    hole=0.3
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Statistics
                st.markdown("#### Theme Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Themes", len(theme_counts))
                with col2:
                    coverage = theme_analyzer.calculate_theme_coverage(results_df)
                    st.metric("Coverage", f"{coverage:.1f}%")
                with col3:
                    dominant_theme = theme_df.iloc[0]['Theme']
                    st.metric("Dominant Theme", dominant_theme)

                # Display theme table
                st.markdown("#### Theme Details")
                st.dataframe(theme_df, use_container_width=True)
            else:
                st.warning("No themes found in the data. Please run theme analysis first.")
        else:
            # Use codes as themes fallback
            st.info("Theme analysis not performed. Showing code distribution instead.")

            # Get code frequencies using cached function
            data_hash = _get_data_hash(results_df)
            code_freq = compute_code_frequency(data_hash, results_df, coder.codebook)

            if code_freq:
                freq_df = pd.DataFrame([
                    {'Code': code, 'Count': count, 'Percentage': (count / len(results_df)) * 100}
                    for code, count in code_freq.items()
                ]).sort_values('Count', ascending=False)

                fig = px.bar(
                    freq_df,
                    x='Code',
                    y='Count',
                    title='Code Prevalence Distribution',
                    color='Percentage',
                    color_continuous_scale='Viridis',
                    text='Count'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_tickangle=-45, height=500)
                st.plotly_chart(fig, use_container_width=True)

    with tab8:
        st.markdown("### 2D Embedding (UMAP/t-SNE) Colored by Theme")

        # Select embedding method
        embedding_method = st.radio(
            "Select embedding method",
            options=['UMAP', 't-SNE'],
            horizontal=True,
            help="UMAP is generally faster and preserves global structure better"
        )

        if coder.feature_matrix is not None:
            try:
                # Parameters for embeddings
                col1, col2 = st.columns(2)
                with col1:
                    n_neighbors = st.slider(
                        "N neighbors (UMAP only)",
                        min_value=5,
                        max_value=50,
                        value=15,
                        help="Controls local vs global structure"
                    ) if embedding_method == 'UMAP' else 15
                with col2:
                    perplexity = st.slider(
                        "Perplexity (t-SNE only)",
                        min_value=5,
                        max_value=50,
                        value=30,
                        help="Balance between local and global structure"
                    ) if embedding_method == 't-SNE' else 30

                # Generate embeddings
                if st.button(f"Generate {embedding_method} Embedding", type="primary"):
                    with st.spinner(f"Computing {embedding_method} embedding..."):
                        if embedding_method == 'UMAP':
                            try:
                                from umap import UMAP
                                reducer = UMAP(n_neighbors=n_neighbors, n_components=2, random_state=42)
                                embedding = reducer.fit_transform(coder.feature_matrix.toarray() if hasattr(coder.feature_matrix, 'toarray') else coder.feature_matrix)
                                st.session_state['embedding_2d'] = embedding
                                st.session_state['embedding_method'] = 'UMAP'
                            except ImportError:
                                st.error("UMAP not installed. Please install with: pip install umap-learn")
                                st.stop()
                        else:  # t-SNE
                            from sklearn.manifold import TSNE
                            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                            embedding = reducer.fit_transform(coder.feature_matrix.toarray() if hasattr(coder.feature_matrix, 'toarray') else coder.feature_matrix)
                            st.session_state['embedding_2d'] = embedding
                            st.session_state['embedding_method'] = 't-SNE'

                # Display embedding if available
                if 'embedding_2d' in st.session_state:
                    embedding = st.session_state['embedding_2d']
                    method_name = st.session_state.get('embedding_method', embedding_method)

                    # Prepare data for plotting
                    embed_df = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])

                    # Add code/theme information
                    embed_df['Primary Code'] = [
                        coder.codebook[codes[0]]['label'] if codes else 'Uncoded'
                        for codes in results_df['assigned_codes']
                    ]

                    embed_df['Confidence'] = [
                        confs[0] if confs else 0.0
                        for confs in results_df['confidence_scores']
                    ]

                    # Add themes if available
                    if 'themes' in results_df.columns:
                        embed_df['Theme'] = [
                            ', '.join(themes) if themes else 'No Theme'
                            for themes in results_df['themes']
                        ]
                        color_by = st.radio(
                            "Color by",
                            options=['Primary Code', 'Theme', 'Confidence'],
                            horizontal=True
                        )
                    else:
                        color_by = st.radio(
                            "Color by",
                            options=['Primary Code', 'Confidence'],
                            horizontal=True
                        )

                    # Create scatter plot
                    fig = px.scatter(
                        embed_df,
                        x='Dim1',
                        y='Dim2',
                        color=color_by,
                        title=f'{method_name} 2D Embedding of Documents',
                        hover_data={'Dim1': ':.2f', 'Dim2': ':.2f', 'Confidence': ':.2f'},
                        color_continuous_scale='Viridis' if color_by == 'Confidence' else None,
                        opacity=0.7
                    )
                    fig.update_layout(height=600)
                    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))

                    st.plotly_chart(fig, use_container_width=True)

                    # Add description
                    st.info(f"""
                    **About this visualization:**
                    - Each point represents a document/response
                    - Similar documents are positioned closer together
                    - Colors show the {color_by.lower()}
                    - {method_name} reduces high-dimensional text data to 2D while preserving structure
                    """)
                else:
                    st.info(f"Click the button above to generate {embedding_method} embedding")

            except Exception as e:
                st.error(f"Error generating embedding: {str(e)}")
        else:
            st.warning("Feature matrix not available. Please run ML analysis first.")

    with tab9:
        st.markdown("### Topic Coherence Diagnostic")

        if coder.feature_matrix is not None and hasattr(coder, 'model'):
            try:
                # Use cached functions for expensive coherence calculations
                # Create a model ID for cache key based on model type and shape
                model_id = (type(coder.model).__name__, coder.feature_matrix.shape)

                # Get cluster labels using cached function
                labels = get_cluster_labels(model_id, coder.model, coder.feature_matrix)

                # Calculate metrics using cached function
                metrics = compute_coherence_metrics(model_id, coder.feature_matrix, labels)

                # Display metrics
                st.markdown("#### Clustering Quality Metrics")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if 'Silhouette Score' in metrics:
                        score = metrics['Silhouette Score']
                        st.metric(
                            "Silhouette Score",
                            f"{score:.3f}",
                            help="Range: [-1, 1]. Higher is better. >0.5 is good."
                        )
                        if score > 0.5:
                            st.success("Excellent separation")
                        elif score > 0.25:
                            st.info("Moderate separation")
                        else:
                            st.warning("Poor separation")

                with col2:
                    if 'Calinski-Harabasz Score' in metrics:
                        score = metrics['Calinski-Harabasz Score']
                        st.metric(
                            "Calinski-Harabasz",
                            f"{score:.1f}",
                            help="Higher is better. Measures cluster density and separation."
                        )

                with col3:
                    if 'Davies-Bouldin Score' in metrics:
                        score = metrics['Davies-Bouldin Score']
                        st.metric(
                            "Davies-Bouldin",
                            f"{score:.3f}",
                            help="Range: [0, ∞]. Lower is better. <1.0 is good."
                        )
                        if score < 1.0:
                            st.success("Good separation")
                        elif score < 2.0:
                            st.info("Moderate separation")
                        else:
                            st.warning("Poor separation")

                # Per-topic coherence
                st.markdown("#### Per-Topic Statistics")

                topic_stats = []
                for code in coder.codebook.keys():
                    code_idx = int(code.split('_')[1]) - 1
                    topic_docs = (labels == code_idx).sum()

                    topic_stats.append({
                        'Topic': coder.codebook[code]['label'],
                        'Documents': topic_docs,
                        'Percentage': f"{(topic_docs / len(labels) * 100):.1f}%",
                        'Avg Confidence': coder.codebook[code]['avg_confidence']
                    })

                stats_df = pd.DataFrame(topic_stats).sort_values('Documents', ascending=False)

                # Visualize topic sizes
                fig = px.bar(
                    stats_df,
                    x='Topic',
                    y='Documents',
                    title='Documents per Topic',
                    color='Avg Confidence',
                    color_continuous_scale='Viridis',
                    text='Documents'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_tickangle=-45, height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Display table
                st.dataframe(stats_df, use_container_width=True)

                # Interpretation guide
                with st.expander("📚 How to interpret these metrics"):
                    st.markdown("""
                    **Silhouette Score** (-1 to 1):
                    - Measures how similar documents are to their own cluster vs other clusters
                    - > 0.7: Strong, well-separated clusters
                    - 0.5-0.7: Reasonable structure
                    - 0.25-0.5: Weak structure, overlapping clusters
                    - < 0.25: Poor clustering, consider different parameters

                    **Calinski-Harabasz Score** (0 to ∞):
                    - Ratio of between-cluster to within-cluster dispersion
                    - Higher values indicate better-defined clusters
                    - No fixed threshold, compare across different n_codes

                    **Davies-Bouldin Score** (0 to ∞):
                    - Average similarity between each cluster and its most similar cluster
                    - Lower is better
                    - < 1.0: Good separation
                    - 1.0-2.0: Moderate separation
                    - > 2.0: Poor separation, clusters may be too similar
                    """)

            except Exception as e:
                st.error(f"Error calculating coherence: {str(e)}")
        else:
            st.warning("Model data not available. Please run ML analysis first.")

    with tab10:
        st.markdown("### Representative Quotes per Theme")

        # Select display mode
        display_mode = st.radio(
            "Display by",
            options=['Code', 'Theme'] if 'themes' in results_df.columns else ['Code'],
            horizontal=True
        )

        if display_mode == 'Code':
            # Get codes sorted by frequency
            codes_sorted = sorted(
                coder.codebook.keys(),
                key=lambda x: coder.codebook[x]['count'],
                reverse=True
            )

            # Filter out codes with no examples
            codes_with_examples = [c for c in codes_sorted if coder.codebook[c]['examples']]

            if codes_with_examples:
                selected_code = st.selectbox(
                    "Select a code to see representative quotes",
                    options=codes_with_examples,
                    format_func=lambda x: f"{coder.codebook[x]['label']} ({coder.codebook[x]['count']} occurrences)"
                )

                code_info = coder.codebook[selected_code]

                st.markdown(f"#### {code_info['label']}")
                st.markdown(f"**Keywords:** {', '.join(code_info['keywords'][:10])}")
                st.markdown(f"**Frequency:** {code_info['count']} | **Avg Confidence:** {code_info['avg_confidence']:.3f}")

                st.markdown("---")

                # Filters and controls section
                st.markdown("#### Filters & Export")

                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    # Confidence range filter
                    confidence_range = st.slider(
                        "Confidence Range",
                        min_value=0.0,
                        max_value=1.0,
                        value=(0.0, 1.0),
                        step=0.05,
                        help="Filter quotes by confidence score range"
                    )

                with col2:
                    # Maximum text length filter
                    max_length = st.number_input(
                        "Max Text Length (chars)",
                        min_value=0,
                        max_value=10000,
                        value=0,
                        step=50,
                        help="Filter quotes by maximum character length (0 = no limit)"
                    )

                with col3:
                    # Focus on edge cases button
                    if st.button("🔍 Focus on Edge Cases", help="Show quotes with confidence 0.3-0.5 (uncertain assignments for QA review)"):
                        confidence_range = (0.3, 0.5)
                        st.rerun()

                # Multiple codes filter
                show_multi_codes_only = st.checkbox(
                    "Show only examples with multiple codes",
                    value=False,
                    help="Filter to show only responses that have multiple codes assigned"
                )

                st.markdown("---")
                st.markdown("#### Representative Quotes")

                # Sort examples by confidence
                examples = sorted(code_info['examples'], key=lambda x: x['confidence'], reverse=True)

                # Pre-compute text column and lookup dictionary for efficient filtering
                text_col = get_text_column(tuple(results_df.columns))
                # Use cached lookup function (much faster than iterrows)
                data_hash = _get_data_hash(results_df)
                text_to_row = build_text_to_row_lookup(data_hash, results_df, text_col)

                # Apply filters
                filtered_examples = []
                for example in examples:
                    # Confidence filter
                    if not (confidence_range[0] <= example['confidence'] <= confidence_range[1]):
                        continue

                    # Text length filter
                    if max_length > 0 and len(example['text']) > max_length:
                        continue

                    # Multiple codes filter
                    if show_multi_codes_only:
                        # Use lookup dictionary instead of DataFrame filtering
                        row = text_to_row.get(example['text'])
                        if row is not None and row['num_codes'] <= 1:
                            continue

                    filtered_examples.append(example)

                # Display count of filtered examples
                st.caption(f"Showing {len(filtered_examples)} of {len(examples)} total quotes")

                # Export buttons
                if filtered_examples:
                    col_exp1, col_exp2 = st.columns(2)

                    with col_exp1:
                        # Prepare clipboard text
                        clipboard_text = f"Code: {code_info['label']}\n"
                        clipboard_text += f"Keywords: {', '.join(code_info['keywords'][:10])}\n"
                        clipboard_text += f"Showing {len(filtered_examples)} quotes\n\n"
                        clipboard_text += "=" * 80 + "\n\n"

                        for i, ex in enumerate(filtered_examples, 1):
                            clipboard_text += f"Quote {i} (Confidence: {ex['confidence']:.3f}):\n{ex['text']}\n\n"

                        st.download_button(
                            label="📋 Copy All Quotes to Clipboard",
                            data=clipboard_text,
                            file_name=f"quotes_{selected_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            help="Download all filtered quotes as a text file"
                        )

                    with col_exp2:
                        # Prepare CSV export using pre-computed lookup
                        csv_data = []

                        for ex in filtered_examples:
                            # Use lookup dictionary instead of DataFrame filtering
                            row = text_to_row.get(ex['text'])
                            other_codes = ""
                            num_codes = 1

                            if row is not None:
                                row_codes = row['assigned_codes']
                                num_codes = len(row_codes)
                                other_code_labels = [coder.codebook[c]['label'] for c in row_codes if c != selected_code and c in coder.codebook]
                                other_codes = ', '.join(other_code_labels)

                            csv_data.append({
                                'Code': code_info['label'],
                                'Confidence': ex['confidence'],
                                'Text': ex['text'],
                                'Text_Length': len(ex['text']),
                                'Num_Codes': num_codes,
                                'Other_Codes': other_codes
                            })

                        csv_df = pd.DataFrame(csv_data)
                        csv_buffer = BytesIO()
                        csv_df.to_csv(csv_buffer, index=False)
                        csv_buffer.seek(0)

                        st.download_button(
                            label="📥 Export as CSV",
                            data=csv_buffer,
                            file_name=f"quotes_{selected_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download all filtered quotes as a CSV file"
                        )

                st.markdown("---")

                # Number of quotes slider
                n_quotes = st.slider(
                    "Number of quotes to display",
                    min_value=1,
                    max_value=min(len(filtered_examples), 20) if filtered_examples else 1,
                    value=min(len(filtered_examples), 5) if filtered_examples else 1
                )

                # Display filtered quotes
                if filtered_examples:
                    # Use pre-computed text_to_row lookup for efficient code lookup

                    for i, example in enumerate(filtered_examples[:n_quotes], 1):
                        with st.container():
                            # Use lookup dictionary instead of DataFrame filtering
                            row = text_to_row.get(example['text'])
                            has_multiple_codes = False
                            other_code_labels = []
                            row_codes = []

                            if row is not None:
                                row_codes = row['assigned_codes']
                                has_multiple_codes = len(row_codes) > 1
                                other_code_labels = [coder.codebook[c]['label'] for c in row_codes if c != selected_code and c in coder.codebook]

                            col1, col2 = st.columns([4, 1])
                            with col1:
                                # Show badge if multiple codes
                                if has_multiple_codes:
                                    badge = create_badge(f'{len(row_codes)} codes', 'info')
                                    st.markdown(f"**Quote {i}:** {badge}")
                                else:
                                    st.markdown(f"**Quote {i}:**")
                                st.markdown(f'> {example["text"]}')

                                # Show other assigned codes if any
                                if other_code_labels:
                                    st.caption(f"**Also assigned to:** {', '.join(other_code_labels)}")
                            with col2:
                                st.metric("Confidence", f"{example['confidence']:.2f}")
                            st.markdown("---")
                else:
                    st.info("No quotes match the selected filters. Try adjusting the filter criteria.")
            else:
                st.info("No examples available. Try lowering the min_confidence parameter or ensure high-confidence assignments exist.")

        else:  # Theme mode
            if hasattr(st.session_state, 'theme_analyzer') and st.session_state.theme_analyzer:
                theme_analyzer = st.session_state.theme_analyzer

                if theme_analyzer.themes:
                    theme_ids = list(theme_analyzer.themes.keys())

                    selected_theme = st.selectbox(
                        "Select a theme to see representative quotes",
                        options=theme_ids,
                        format_func=lambda x: f"{theme_analyzer.themes[x]['name']} ({len(theme_analyzer.themes[x]['responses'])} responses)"
                    )

                    theme_info = theme_analyzer.themes[selected_theme]

                    st.markdown(f"#### {theme_info['name']}")
                    st.markdown(f"**Description:** {theme_info['description']}")
                    st.markdown(f"**Associated Codes:** {', '.join(theme_info['codes'])}")
                    st.markdown(f"**Frequency:** {len(theme_info['responses'])}")

                    st.markdown("---")

                    # Filters and controls section
                    st.markdown("#### Filters & Export")

                    col1, col2, col3 = st.columns([2, 2, 1])

                    with col1:
                        # Confidence range filter (for theme mode, filter by average confidence)
                        confidence_range_theme = st.slider(
                            "Confidence Range",
                            min_value=0.0,
                            max_value=1.0,
                            value=(0.0, 1.0),
                            step=0.05,
                            help="Filter quotes by average confidence score range",
                            key="theme_confidence_range"
                        )

                    with col2:
                        # Maximum text length filter
                        max_length_theme = st.number_input(
                            "Max Text Length (chars)",
                            min_value=0,
                            max_value=10000,
                            value=0,
                            step=50,
                            help="Filter quotes by maximum character length (0 = no limit)",
                            key="theme_max_length"
                        )

                    with col3:
                        # Focus on edge cases button
                        if st.button("🔍 Focus on Edge Cases", help="Show quotes with confidence 0.3-0.5 (uncertain assignments for QA review)", key="theme_edge_cases"):
                            confidence_range_theme = (0.3, 0.5)
                            st.rerun()

                    # Multiple codes filter
                    show_multi_codes_only_theme = st.checkbox(
                        "Show only examples with multiple codes",
                        value=False,
                        help="Filter to show only responses that have multiple codes assigned",
                        key="theme_multi_codes"
                    )

                    st.markdown("---")
                    st.markdown("#### Representative Quotes")

                    # Get responses for this theme
                    theme_responses = theme_info['responses']

                    if theme_responses:
                        # Get text column name using cached function
                        text_col = get_text_column(tuple(results_df.columns))

                        # Apply filters
                        filtered_responses = []
                        for resp_idx in theme_responses:
                            row = results_df.iloc[resp_idx]

                            # Calculate average confidence for this response
                            avg_conf = np.mean(row['confidence_scores']) if 'confidence_scores' in row and len(row['confidence_scores']) > 0 else 0.5

                            # Confidence filter
                            if not (confidence_range_theme[0] <= avg_conf <= confidence_range_theme[1]):
                                continue

                            # Text length filter
                            if max_length_theme > 0 and len(row[text_col]) > max_length_theme:
                                continue

                            # Multiple codes filter
                            if show_multi_codes_only_theme and row['num_codes'] <= 1:
                                continue

                            filtered_responses.append(resp_idx)

                        # Display count of filtered responses
                        st.caption(f"Showing {len(filtered_responses)} of {len(theme_responses)} total quotes")

                        # Export buttons
                        if filtered_responses:
                            col_exp1, col_exp2 = st.columns(2)

                            with col_exp1:
                                # Prepare clipboard text
                                clipboard_text = f"Theme: {theme_info['name']}\n"
                                clipboard_text += f"Description: {theme_info['description']}\n"
                                clipboard_text += f"Associated Codes: {', '.join(theme_info['codes'])}\n"
                                clipboard_text += f"Showing {len(filtered_responses)} quotes\n\n"
                                clipboard_text += "=" * 80 + "\n\n"

                                for i, resp_idx in enumerate(filtered_responses, 1):
                                    row = results_df.iloc[resp_idx]
                                    clipboard_text += f"Quote {i}:\n{row[text_col]}\n"
                                    code_labels = [coder.codebook[c]['label'] for c in row['assigned_codes'] if c in coder.codebook]
                                    clipboard_text += f"Codes: {', '.join(code_labels)}\n\n"

                                st.download_button(
                                    label="📋 Copy All Quotes to Clipboard",
                                    data=clipboard_text,
                                    file_name=f"quotes_theme_{selected_theme}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    help="Download all filtered quotes as a text file",
                                    key="theme_clipboard"
                                )

                            with col_exp2:
                                # Prepare CSV export
                                csv_data = []
                                for resp_idx in filtered_responses:
                                    row = results_df.iloc[resp_idx]
                                    code_labels = [coder.codebook[c]['label'] for c in row['assigned_codes'] if c in coder.codebook]
                                    avg_conf = np.mean(row['confidence_scores']) if 'confidence_scores' in row and len(row['confidence_scores']) > 0 else 0.5

                                    csv_data.append({
                                        'Theme': theme_info['name'],
                                        'Text': row[text_col],
                                        'Text_Length': len(row[text_col]),
                                        'Num_Codes': row['num_codes'],
                                        'Assigned_Codes': ', '.join(code_labels),
                                        'Avg_Confidence': avg_conf
                                    })

                                csv_df = pd.DataFrame(csv_data)
                                csv_buffer = BytesIO()
                                csv_df.to_csv(csv_buffer, index=False)
                                csv_buffer.seek(0)

                                st.download_button(
                                    label="📥 Export as CSV",
                                    data=csv_buffer,
                                    file_name=f"quotes_theme_{selected_theme}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="Download all filtered quotes as a CSV file",
                                    key="theme_csv"
                                )

                        st.markdown("---")

                        # Number of quotes slider
                        n_quotes = st.slider(
                            "Number of quotes to display",
                            min_value=1,
                            max_value=min(len(filtered_responses), 20) if filtered_responses else 1,
                            value=min(len(filtered_responses), 5) if filtered_responses else 1
                        )

                        # Display filtered quotes
                        if filtered_responses:
                            for i, resp_idx in enumerate(filtered_responses[:n_quotes], 1):
                                with st.container():
                                    row = results_df.iloc[resp_idx]
                                    has_multiple_codes = row['num_codes'] > 1

                                    # Show badge if multiple codes
                                    if has_multiple_codes:
                                        badge = create_badge(f'{row["num_codes"]} codes', 'info')
                                        st.markdown(f"**Quote {i}:** {badge}")
                                    else:
                                        st.markdown(f"**Quote {i}:**")

                                    st.markdown(f'> {row[text_col]}')

                                    # Show assigned codes for this response
                                    codes = row['assigned_codes']
                                    code_labels = [coder.codebook[c]['label'] for c in codes if c in coder.codebook]
                                    st.caption(f"**Codes:** {', '.join(code_labels)}")
                                    st.markdown("---")
                        else:
                            st.info("No quotes match the selected filters. Try adjusting the filter criteria.")
                    else:
                        st.info("No responses found for this theme.")
                else:
                    st.warning("No themes defined. Please run theme analysis first.")
            else:
                st.warning("Theme analyzer not available. Showing code-based view instead.")
                st.info("Switch to 'Code' display mode above to see representative quotes by code.")

    # Next button - always show on this page if analysis is complete
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

        # Codebook
        codebook_data = []
        for code_id, info in coder.codebook.items():
            codebook_data.append({
                'Code': code_id,
                'Label': info['label'],
                'Keywords': ', '.join(info['keywords']),
                'Count': info['count'],
                'Avg Confidence': info['avg_confidence']
            })
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

    ### 🎯 Features

    - **Automatic Theme Discovery**: Let ML algorithms find themes in your data
    - **Multiple Algorithms**: Choose from TF-IDF+K-Means, LDA, or NMF
    - **Confidence Scoring**: Every code assignment includes a confidence score
    - **15 Essential Outputs**: Complete analysis package for researchers
    - **Interactive Visualizations**: Explore your data with interactive charts
    - **Multiple Export Formats**: Download results in CSV, Excel, or JSON

    ### 🤖 Supported Algorithms

    #### TF-IDF + K-Means (Recommended)
    - Fast and interpretable clustering
    - Good for well-separated themes
    - Best for exploratory analysis

    #### Latent Dirichlet Allocation (LDA)
    - Probabilistic topic modeling
    - Handles overlapping themes well
    - Good for document collections

    #### Non-negative Matrix Factorization (NMF)
    - Parts-based decomposition
    - Produces sparse, interpretable results
    - Good for distinct themes

    ### 📊 15 Essential Outputs

    1. Code Assignments with confidence scores
    2. Auto-generated Codebook
    3. Code Frequency Tables
    4. Quality Metrics
    5. Binary Matrix for statistical analysis
    6. Representative Quotes
    7. Co-Occurrence Analysis
    8. Descriptive Statistics
    9. Segmentation Analysis
    10. QA Report
    11. Interactive Visualizations
    12. Multiple Export Formats
    13. Method Documentation
    14. Uncoded Response Detection
    15. Executive Summary

    ### 🚀 Getting Started

    1. **Upload Data**: CSV or Excel file with text responses
    2. **Configure**: Choose algorithm and parameters
    3. **Run Analysis**: Let ML discover themes
    4. **Explore Results**: View codes, insights, and visualizations
    5. **Export**: Download complete results package
    """)


if __name__ == "__main__":
    main()
