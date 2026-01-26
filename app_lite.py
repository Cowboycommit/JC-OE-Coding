"""
Lite / Engineering View - ML Open Coding Analytics Pipeline

PURPOSE:
    This UI exists to teach how the system works, not to hide complexity.
    It is an engineering and onboarding aid that visually documents and
    enforces the correct end-to-end execution order of the analytics pipeline.

DESIGN PHILOSOPHY:
    - One page only, vertical top-to-bottom flow
    - No advanced widgets or nested state logic
    - No business logic implemented in the UI
    - UI calls existing pipeline functions, never re-implements them
    - Clarity and correctness over polish

PIPELINE STAGES:
    1. Dataset Ingestion - Load CSV/Excel/JSON data
    2. Data Validation & Text Preprocessing - Validate structure, apply text cleaning
       (includes negation preservation, domain stopwords, quality filtering)
    3. Method Eligibility Checks - Verify ML method compatibility with data
    4. Model Execution - Run clustering/topic modeling (TF-IDF, LDA, LSTM, BERT, SVM)
    5. Diagnostics & Assumptions - Assess validity, generate QA report
    6. Visualization Generation - Charts, word clouds, network diagrams
    7. Export & Reporting - Excel package, methods documentation, executive summary

SUPPORTED ML METHODS:
    - tfidf_kmeans: TF-IDF + K-Means (fast, bag-of-words)
    - lda: Latent Dirichlet Allocation (topic modeling)
    - lstm_kmeans: LSTM + K-Means (sequential patterns)
    - bert_kmeans: BERT + K-Means (semantic understanding)
    - svm: SVM Spectral Clustering (kernel-based)

KEY FEATURES:
    - LLM-enhanced code labels and descriptions
    - Optional sentiment analysis (Twitter-RoBERTa, VADER, Review-BERT)
    - Semantic word clouds with meaning-based coloring
    - Auto-selection of optimal code count via silhouette analysis

AUTHOR: Engineering Team
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional imports for enhanced visualizations
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Optional imports for advanced visualizations
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_samples, silhouette_score
    SKLEARN_VIZ_AVAILABLE = True
except ImportError:
    SKLEARN_VIZ_AVAILABLE = False

try:
    import pyLDAvis
    import pyLDAvis.lda_model
    PYLDAVIS_AVAILABLE = True
except ImportError:
    PYLDAVIS_AVAILABLE = False

# Word cloud support (optional)
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

# -----------------------------------------------------------------------------
# PATH SETUP
# Ensure src/ is importable for pipeline modules
# -----------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

# -----------------------------------------------------------------------------
# IMPORT PIPELINE MODULES
# These are the ONLY modules that contain analysis logic.
# The UI NEVER implements analysis - it only orchestrates calls.
# -----------------------------------------------------------------------------
from src.data_loader import DataLoader
from src.rigor_diagnostics import RigorDiagnostics
from src.method_visualizations import (
    MethodVisualizer,
    get_visualization_availability,
    PILWordCloud,
)
from helpers.analysis import (
    validate_dataframe,
    preprocess_responses,
    run_ml_analysis,
    calculate_metrics_summary,
    generate_insights,
    get_top_codes,
    get_cooccurrence_pairs,
    get_qa_report,
    export_results_package,
    generate_methods_documentation,
    generate_executive_summary,
    find_optimal_codes,
)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"


def get_available_datasets() -> Dict[str, str]:
    """
    Dynamically scan the data folder for available datasets.

    Returns:
        Dictionary mapping filename to display name.
    """
    datasets = {}

    if not DATA_DIR.exists():
        return datasets

    # Scan for CSV, Excel, and JSON files
    for ext in ['*.csv', '*.xlsx', '*.xls', '*.json']:
        for filepath in DATA_DIR.glob(ext):
            filename = filepath.name
            # Skip README and hidden files
            if filename.startswith('.') or filename.upper() == 'README.MD':
                continue

            # Generate display name from filename
            # Remove extension and convert underscores to spaces
            base_name = filepath.stem
            display_name = base_name.replace('_', ' ').title()

            # Add size indicator for large files (> 10MB)
            file_size = filepath.stat().st_size
            if file_size > 10 * 1024 * 1024:
                display_name += " (Large)"

            datasets[filename] = display_name

    return datasets

PIPELINE_STAGES = [
    {
        "number": 1,
        "name": "Dataset Ingestion",
        "purpose": "Load raw data from file (CSV/Excel/JSON) into memory",
        "inputs": ["File path or uploaded file"],
        "module": "src.data_loader.DataLoader",
        "function": "load_csv(), load_excel(), load_json()",
        "outputs": ["Raw pandas DataFrame"],
        "mistakes": [
            "Loading data directly in UI code instead of using DataLoader",
            "Not handling encoding issues (UTF-8 vs Latin-1)",
            "Ignoring empty files or malformed data",
        ],
    },
    {
        "number": 2,
        "name": "Data Validation & Text Preprocessing",
        "purpose": "Validate structure and apply text preprocessing with quality filtering",
        "inputs": [
            "Raw DataFrame",
            "Text column name",
            "Preprocessing options (preset type, negation preservation, domain stopwords)",
        ],
        "module": "helpers.analysis, src.text_preprocessor, src.gold_standard_preprocessing",
        "function": "validate_dataframe(), preprocess_responses(), DataCleaningPipeline.clean_dataframe()",
        "outputs": [
            "Validated DataFrame with preprocessed text column",
            "Quality metrics (valid ratio, token counts, filter reasons)",
        ],
        "mistakes": [
            "Skipping validation before analysis",
            "Not preserving negation words for sentiment analysis",
            "Using generic stopwords instead of domain-specific ones",
            "Not documenting preprocessing decisions for reproducibility",
        ],
    },
    {
        "number": 3,
        "name": "Method Eligibility Checks",
        "purpose": "Verify that selected ML method is compatible with data",
        "inputs": ["DataFrame", "Method selection", "Number of codes"],
        "module": "helpers.analysis (within run_ml_analysis)",
        "function": "MLOpenCoder._validate_method_compatibility()",
        "outputs": ["Eligibility confirmation or error"],
        "mistakes": [
            "Using LDA with semantic embeddings (incompatible)",
            "Requesting more codes than samples",
            "Not checking vocabulary size after vectorization",
        ],
    },
    {
        "number": 4,
        "name": "Model Execution",
        "purpose": "Run ML clustering/topic modeling to discover codes",
        "inputs": ["Preprocessed responses", "n_codes", "method", "representation"],
        "module": "helpers.analysis",
        "function": "run_ml_analysis()",
        "outputs": ["MLOpenCoder object", "Results DataFrame", "Metrics dict"],
        "mistakes": [
            "Running analysis in UI thread without progress feedback",
            "Not storing results in session state",
            "Re-running analysis unnecessarily on page refresh",
        ],
    },
    {
        "number": 5,
        "name": "Diagnostics & Assumptions",
        "purpose": "Assess methodological validity and detect potential bias",
        "inputs": ["MLOpenCoder", "Results DataFrame"],
        "module": "src.rigor_diagnostics, helpers.analysis",
        "function": "get_qa_report(), RigorDiagnostics methods",
        "outputs": ["QA Report (Markdown)", "Validity metrics", "Recommendations"],
        "mistakes": [
            "Skipping diagnostics before reporting results",
            "Not reviewing uncoded responses",
            "Ignoring low confidence warnings",
        ],
    },
    {
        "number": 6,
        "name": "Visualization Generation",
        "purpose": "Create visual representations of analysis and sentiment results",
        "inputs": ["MLOpenCoder", "Results DataFrame", "Metrics", "Sentiment results (optional)"],
        "module": "helpers.analysis (data prep), src.method_visualizations, UI (rendering only)",
        "function": "get_top_codes(), get_cooccurrence_pairs(), MethodVisualizer.create_*()",
        "outputs": [
            "Code frequency bar charts",
            "Co-occurrence heatmaps and network diagrams",
            "Sunburst hierarchical charts",
            "Overall and per-topic word clouds (with semantic coloring)",
            "Cluster scatter plots (PCA/t-SNE)",
            "Silhouette analysis plots (K-Means methods)",
            "Topic-term heatmaps",
            "Topic distribution charts (LDA)",
            "Sentiment distribution pie charts (if enabled)",
            "Confidence score histograms",
        ],
        "mistakes": [
            "Computing chart data in Streamlit callbacks (use precompute_all_visualizations)",
            "Not caching visualization data in session state",
            "Putting Plotly/chart logic in analysis modules instead of UI",
            "Showing incompatible visualizations for methods (e.g., pyLDAvis for K-Means)",
        ],
    },
    {
        "number": 7,
        "name": "Export & Reporting",
        "purpose": "Package results for external consumption",
        "inputs": ["MLOpenCoder", "Results DataFrame", "Metrics"],
        "module": "helpers.analysis",
        "function": "export_results_package(), generate_methods_documentation()",
        "outputs": ["Excel file (bytes)", "Methods documentation (Markdown)"],
        "mistakes": [
            "Generating exports on every page load",
            "Not including codebook in exports",
            "Missing reproducibility information in methods section",
        ],
    },
]

ML_METHODS = {
    "tfidf_kmeans": "TF-IDF + K-Means (Fast, bag-of-words)",
    "lda": "LDA - Latent Dirichlet Allocation (Topic modeling)",
    "lstm_kmeans": "LSTM + K-Means (Sequential patterns)",
    "bert_kmeans": "BERT + K-Means (Semantic understanding)",
    "svm": "SVM Spectral Clustering (Kernel-based)",
}

REPRESENTATIONS = {
    "tfidf": "TF-IDF (Default, fast, bag-of-words)",
    "sbert": "SentenceBERT (Semantic, offline)",
    "lstm": "LSTM (Sequential patterns)",
    "bert": "BERT (Semantic, same as SentenceBERT)",
    "word2vec": "Word2Vec (Trains on your data)",
    "fasttext": "FastText (Handles typos)",
}


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# All pipeline state is stored here. Never in local variables.
# -----------------------------------------------------------------------------
def init_session_state():
    """
    Initialize session state for pipeline tracking.

    WHY HERE: Session state must be initialized once at app start.
    This ensures stage completion tracking persists across reruns.
    """
    defaults = {
        # Stage completion flags
        "stage_1_complete": False,
        "stage_2_complete": False,
        "stage_3_complete": False,
        "stage_4_complete": False,
        "stage_5_complete": False,
        "stage_6_complete": False,
        "stage_7_complete": False,
        # Data artifacts
        "raw_df": None,
        "validated_df": None,
        "text_column": None,
        "method": "tfidf_kmeans",
        "representation": "tfidf",
        "n_codes": 10,
        "min_confidence": 0.3,
        "stop_words": "english",
        # Analysis artifacts
        "coder": None,
        "results_df": None,
        "metrics": None,
        # Diagnostics artifacts
        "qa_report": None,
        # Visualization artifacts
        "top_codes_df": None,
        "cooccurrence_df": None,
        # Export artifacts
        "excel_bytes": None,
        "methods_doc": None,
        "executive_summary": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# Pure functions for status display. No business logic.
# -----------------------------------------------------------------------------
def get_stage_status(stage_num: int) -> str:
    """Return status indicator for a pipeline stage."""
    if st.session_state.get(f"stage_{stage_num}_complete", False):
        return "COMPLETE"
    # Check if prior stages are complete
    for i in range(1, stage_num):
        if not st.session_state.get(f"stage_{i}_complete", False):
            return "BLOCKED"
    return "READY"


def render_status_badge(status: str) -> str:
    """Return a text badge for status."""
    badges = {
        "COMPLETE": "[COMPLETE]",
        "READY": "[READY]",
        "BLOCKED": "[BLOCKED]",
        "RUNNING": "[RUNNING...]",
    }
    return badges.get(status, "[UNKNOWN]")


def reset_downstream_stages(from_stage: int):
    """
    Reset all stages after the given stage number.

    WHY: When upstream data changes, downstream results are invalid.
    This enforces pipeline integrity.
    """
    for i in range(from_stage + 1, 8):
        st.session_state[f"stage_{i}_complete"] = False

    # Clear downstream artifacts based on stage
    if from_stage < 4:
        st.session_state["coder"] = None
        st.session_state["results_df"] = None
        st.session_state["metrics"] = None
    if from_stage < 5:
        st.session_state["qa_report"] = None
    if from_stage < 6:
        st.session_state["top_codes_df"] = None
        st.session_state["cooccurrence_df"] = None
    if from_stage < 7:
        st.session_state["excel_bytes"] = None
        st.session_state["methods_doc"] = None
        st.session_state["executive_summary"] = None


# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    """
    Main application entry point.

    STRUCTURE:
    1. Pipeline Overview (what this tool does)
    2. Stage-by-Stage Execution Blocks
    3. Execution Controls (buttons)
    4. Artifacts Panel (outputs)
    5. Separation of Concerns Callouts
    """
    # Initialize state
    init_session_state()

    # Page config
    st.set_page_config(
        page_title="ML Open Coding - Lite/Engineering View",
        page_icon="🔧",
        layout="wide",
    )

    # ==========================================================================
    # SECTION: HEADER & PIPELINE OVERVIEW
    # ==========================================================================
    st.title("ML Open Coding Analytics Pipeline")
    st.markdown("### Lite / Engineering View")

    st.markdown("---")

    st.markdown("""
    **Purpose**: This UI documents and enforces the correct execution order of
    the ML-based open coding analytics pipeline. It is designed for engineering
    onboarding and debugging, not for end-user interaction.

    **Full Analysis Lifecycle**:

    1. **Dataset Ingestion** - Load data from CSV/Excel/JSON files
    2. **Data Validation & Text Preprocessing** - Validate structure, apply text cleaning
       (negation preservation, domain stopwords, quality filtering)
    3. **Method Eligibility Checks** - Verify ML method compatibility with data characteristics
    4. **Model Execution** - Run clustering/topic modeling (TF-IDF, LDA, LSTM, BERT, SVM)
    5. **Diagnostics & Assumptions** - Assess validity, generate QA report, detect bias
    6. **Visualization Generation** - Charts, word clouds, network diagrams, sentiment plots
    7. **Export & Reporting** - Excel package, methods documentation, executive summary

    Each stage must complete before the next can begin. This prevents
    mixing UI logic with analytics logic and ensures reproducibility.
    """)

    st.markdown("---")

    # ==========================================================================
    # SECTION: STAGE-BY-STAGE EXECUTION BLOCKS
    # ==========================================================================
    st.header("Pipeline Stages")

    # --------------------------------------------------------------------------
    # STAGE 1: Dataset Ingestion
    # --------------------------------------------------------------------------
    stage_1 = PIPELINE_STAGES[0]
    with st.expander(
        f"Stage {stage_1['number']}: {stage_1['name']} {render_status_badge(get_stage_status(1))}",
        expanded=not st.session_state["stage_1_complete"],
    ):
        st.markdown(f"**Purpose**: {stage_1['purpose']}")
        st.markdown(f"**Responsible Module**: `{stage_1['module']}`")
        st.markdown(f"**Functions**: `{stage_1['function']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Required Inputs**:")
            for inp in stage_1["inputs"]:
                st.markdown(f"- {inp}")
        with col2:
            st.markdown("**Outputs/Artifacts**:")
            for out in stage_1["outputs"]:
                st.markdown(f"- {out}")

        st.markdown("**Common Engineering Mistakes**:")
        for mistake in stage_1["mistakes"]:
            st.markdown(f"- {mistake}")

        st.markdown("---")

        # Execution control - Dataset selection
        st.markdown("**Select a sample dataset or upload your own:**")

        # Dynamically scan data folder for available datasets
        available_datasets = get_available_datasets()

        if not available_datasets:
            st.warning(f"No datasets found in {DATA_DIR}. Please add CSV, Excel, or JSON files.")
            dataset_options = ["-- No datasets available --"]
            selected_dataset = st.selectbox(
                "Sample Dataset",
                options=dataset_options,
                key="dataset_select",
            )
        else:
            dataset_options = ["-- Select a sample dataset --"] + list(available_datasets.keys())
            selected_dataset = st.selectbox(
                "Sample Dataset",
                options=dataset_options,
                format_func=lambda x: available_datasets.get(x, x),
                key="dataset_select",
            )

        if st.button("Execute Stage 1: Load Data", key="btn_stage_1"):
            if selected_dataset not in ["-- Select a sample dataset --", "-- No datasets available --"]:
                try:
                    # Load from project's data directory
                    filepath = DATA_DIR / selected_dataset
                    loader = DataLoader()

                    # Handle different file types
                    if selected_dataset.endswith('.csv'):
                        df = loader.load_csv(str(filepath))
                    elif selected_dataset.endswith(('.xlsx', '.xls')):
                        df = loader.load_excel(str(filepath))
                    elif selected_dataset.endswith('.json'):
                        df = loader.load_json(str(filepath))
                    else:
                        df = loader.load_csv(str(filepath))  # Default to CSV

                    st.session_state["raw_df"] = df
                    st.session_state["stage_1_complete"] = True
                    reset_downstream_stages(1)

                    st.success(f"Loaded '{selected_dataset}': {len(df)} rows, {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Failed to load data: {e}")
            else:
                st.warning("Please select a dataset first")

        # Show artifact if complete
        if st.session_state["stage_1_complete"] and st.session_state["raw_df"] is not None:
            df = st.session_state["raw_df"]
            st.markdown("**Dataset Summary**:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

            st.markdown("**Columns**: " + ", ".join(f"`{c}`" for c in df.columns))
            st.markdown("**Preview (first 5 rows)**:")
            st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------------------------------
    # STAGE 2: Data Validation & Typing
    # --------------------------------------------------------------------------
    stage_2 = PIPELINE_STAGES[1]
    with st.expander(
        f"Stage {stage_2['number']}: {stage_2['name']} {render_status_badge(get_stage_status(2))}",
        expanded=st.session_state["stage_1_complete"] and not st.session_state["stage_2_complete"],
    ):
        st.markdown(f"**Purpose**: {stage_2['purpose']}")
        st.markdown(f"**Responsible Module**: `{stage_2['module']}`")
        st.markdown(f"**Functions**: `{stage_2['function']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Required Inputs**:")
            for inp in stage_2["inputs"]:
                st.markdown(f"- {inp}")
        with col2:
            st.markdown("**Outputs/Artifacts**:")
            for out in stage_2["outputs"]:
                st.markdown(f"- {out}")

        st.markdown("**Common Engineering Mistakes**:")
        for mistake in stage_2["mistakes"]:
            st.markdown(f"- {mistake}")

        st.markdown("---")

        # Controls
        status_2 = get_stage_status(2)
        if status_2 == "BLOCKED":
            st.info("Complete Stage 1 first")
        else:
            if st.session_state["raw_df"] is not None:
                columns = st.session_state["raw_df"].columns.tolist()

                # Auto-detect text column (prefer 'response' if available)
                default_idx = columns.index("response") if "response" in columns else 0

                text_col = st.selectbox(
                    "Select text column for analysis",
                    options=columns,
                    index=default_idx,
                    key="text_col_select",
                )

                # Data cleaning options
                col1, col2, col3 = st.columns(3)
                with col1:
                    remove_nulls = st.checkbox("Remove null responses", value=True)
                with col2:
                    remove_duplicates = st.checkbox("Remove duplicates", value=False)
                with col3:
                    min_length = st.number_input(
                        "Min response length",
                        min_value=0,
                        max_value=100,
                        value=5,
                        help="Responses shorter than this will be removed"
                    )

                if st.button("Execute Stage 2: Validate & Preprocess", key="btn_stage_2"):
                    try:
                        df = st.session_state["raw_df"]

                        # WHY: We call validate_dataframe from helpers.analysis
                        # This is the canonical validation function
                        is_valid, error_msg = validate_dataframe(
                            df, required_columns=[text_col], min_rows=1
                        )

                        if not is_valid:
                            st.error(f"Validation failed: {error_msg}")
                        else:
                            # WHY: We call preprocess_responses from helpers.analysis
                            # This is the canonical preprocessing function
                            processed_df = preprocess_responses(
                                df,
                                text_column=text_col,
                                remove_nulls=remove_nulls,
                                remove_duplicates=remove_duplicates,
                                min_length=min_length,
                            )

                            st.session_state["validated_df"] = processed_df
                            st.session_state["text_column"] = text_col
                            st.session_state["stage_2_complete"] = True
                            reset_downstream_stages(2)

                            st.success(
                                f"Validation complete. {len(processed_df)} responses ready."
                            )
                    except Exception as e:
                        st.error(f"Validation failed: {e}")

        # Show artifact
        if st.session_state["stage_2_complete"]:
            vdf = st.session_state["validated_df"]
            text_col = st.session_state["text_column"]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Responses Ready", len(vdf))
            with col2:
                st.metric("Text Column", text_col)

            st.markdown("**Sample Responses (first 3)**:")
            for i, text in enumerate(vdf[text_col].head(3).tolist(), 1):
                st.markdown(f"{i}. _{str(text)[:150]}{'...' if len(str(text)) > 150 else ''}_")

    # --------------------------------------------------------------------------
    # STAGE 3: Method Eligibility Checks
    # --------------------------------------------------------------------------
    stage_3 = PIPELINE_STAGES[2]
    with st.expander(
        f"Stage {stage_3['number']}: {stage_3['name']} {render_status_badge(get_stage_status(3))}",
        expanded=st.session_state["stage_2_complete"] and not st.session_state["stage_3_complete"],
    ):
        st.markdown(f"**Purpose**: {stage_3['purpose']}")
        st.markdown(f"**Responsible Module**: `{stage_3['module']}`")
        st.markdown(f"**Functions**: `{stage_3['function']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Required Inputs**:")
            for inp in stage_3["inputs"]:
                st.markdown(f"- {inp}")
        with col2:
            st.markdown("**Outputs/Artifacts**:")
            for out in stage_3["outputs"]:
                st.markdown(f"- {out}")

        st.markdown("**Common Engineering Mistakes**:")
        for mistake in stage_3["mistakes"]:
            st.markdown(f"- {mistake}")

        st.markdown("---")

        status_3 = get_stage_status(3)
        if status_3 == "BLOCKED":
            st.info("Complete Stage 2 first")
        else:
            # Auto-selection options
            st.markdown("**Auto-Selection Options**")
            st.markdown("Enable these to automatically select optimal settings based on data characteristics:")

            auto_col1, auto_col2 = st.columns(2)
            with auto_col1:
                auto_method = st.checkbox(
                    "Auto-select best method",
                    value=False,
                    key="auto_method_checkbox",
                    help="Automatically select the best ML method based on dataset size and text characteristics"
                )
            with auto_col2:
                auto_n_codes = st.checkbox(
                    "Auto-select number of codes",
                    value=False,
                    key="auto_n_codes_checkbox",
                    help="Use silhouette analysis to find the optimal number of codes"
                )

            st.markdown("---")

            # Manual configuration options
            col1, col2 = st.columns(2)

            with col1:
                if auto_method:
                    st.info("Method will be auto-selected based on data characteristics")
                    method = None  # Will be determined during execution
                else:
                    method = st.selectbox(
                        "ML Method",
                        options=list(ML_METHODS.keys()),
                        format_func=lambda x: ML_METHODS[x],
                        key="method_select",
                    )

                if auto_n_codes:
                    st.info("Number of codes will be optimized using silhouette analysis")
                    n_codes = None  # Will be determined during execution
                else:
                    n_codes = st.slider(
                        "Number of codes",
                        min_value=3,
                        max_value=20,
                        value=8,
                        key="n_codes_slider",
                    )

            with col2:
                # Set representation based on method
                if method == "lda":
                    representation = "tfidf"
                    st.info("Using TF-IDF (required for LDA)")
                elif method == "lstm_kmeans":
                    representation = "lstm"
                    st.info("Using LSTM embeddings")
                elif method == "bert_kmeans":
                    representation = "bert"
                    st.info("Using BERT embeddings")
                elif method is None:
                    representation = "tfidf"
                    st.info("Using TF-IDF (default for auto-selection)")
                else:
                    representation = st.selectbox(
                        "Text Representation",
                        options=list(REPRESENTATIONS.keys()),
                        format_func=lambda x: REPRESENTATIONS[x],
                        key="repr_select",
                    )

                min_confidence = st.slider(
                    "Min confidence",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.3,
                    step=0.1,
                    key="confidence_slider",
                )

            # Language is hardcoded to English
            stop_words = "english"

            if st.button("Execute Stage 3: Check Eligibility", key="btn_stage_3"):
                try:
                    validated_df = st.session_state["validated_df"]
                    text_col = st.session_state["text_column"]
                    n_samples = len(validated_df)

                    # Auto-select method based on data characteristics
                    if auto_method:
                        with st.spinner("Analyzing data characteristics to select best method..."):
                            # Use TF-IDF + KMeans as default - best for most cases
                            # For larger datasets, consider LDA for topic modeling
                            if n_samples > 500:
                                selected_method = "lda"
                                st.info(f"Auto-selected LDA for larger dataset ({n_samples} samples)")
                            else:
                                selected_method = "tfidf_kmeans"
                                st.info(f"Auto-selected TF-IDF + K-Means for dataset ({n_samples} samples)")
                    else:
                        selected_method = method

                    # Auto-select number of codes using silhouette analysis
                    if auto_n_codes:
                        with st.spinner("Running silhouette analysis to find optimal number of codes..."):
                            try:
                                optimal_n, analysis_results = find_optimal_codes(
                                    df=validated_df,
                                    text_column=text_col,
                                    min_codes=3,
                                    max_codes=min(15, n_samples - 1),
                                    method=selected_method,
                                    stop_words=stop_words
                                )
                                selected_n_codes = optimal_n
                                st.success(
                                    f"Optimal number of codes: {optimal_n} "
                                    f"(silhouette score: {analysis_results['best_silhouette_score']:.4f})"
                                )
                            except ValueError as e:
                                st.error(f"Could not auto-select number of codes: {e}")
                                return
                    else:
                        selected_n_codes = n_codes

                    # Check n_codes vs dataset size
                    if selected_n_codes and selected_n_codes > n_samples:
                        st.error(
                            f"Cannot request {selected_n_codes} codes with only {n_samples} samples. "
                            f"Reduce n_codes to at most {n_samples}."
                        )
                    else:
                        st.session_state["method"] = selected_method
                        st.session_state["representation"] = representation
                        st.session_state["n_codes"] = selected_n_codes
                        st.session_state["min_confidence"] = min_confidence
                        st.session_state["stop_words"] = stop_words
                        st.session_state["auto_method"] = auto_method
                        st.session_state["auto_n_codes"] = auto_n_codes
                        st.session_state["stage_3_complete"] = True
                        reset_downstream_stages(3)

                        st.success(f"Configuration set: {ML_METHODS[selected_method]}, {selected_n_codes} codes")

                except Exception as e:
                    st.error(f"Configuration failed: {e}")

        if st.session_state["stage_3_complete"]:
            st.markdown(f"**Method**: `{st.session_state['method']}`")
            st.markdown(f"**Representation**: `{st.session_state['representation']}`")
            st.markdown(f"**N Codes**: `{st.session_state['n_codes']}`")
            st.markdown(f"**Min Confidence**: `{st.session_state['min_confidence']}`")
            st.markdown(f"**Stop Words**: `{st.session_state['stop_words']}`")

    # --------------------------------------------------------------------------
    # STAGE 4: Model Execution
    # --------------------------------------------------------------------------
    stage_4 = PIPELINE_STAGES[3]
    with st.expander(
        f"Stage {stage_4['number']}: {stage_4['name']} {render_status_badge(get_stage_status(4))}",
        expanded=st.session_state["stage_3_complete"] and not st.session_state["stage_4_complete"],
    ):
        st.markdown(f"**Purpose**: {stage_4['purpose']}")
        st.markdown(f"**Responsible Module**: `{stage_4['module']}`")
        st.markdown(f"**Functions**: `{stage_4['function']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Required Inputs**:")
            for inp in stage_4["inputs"]:
                st.markdown(f"- {inp}")
        with col2:
            st.markdown("**Outputs/Artifacts**:")
            for out in stage_4["outputs"]:
                st.markdown(f"- {out}")

        st.markdown("**Common Engineering Mistakes**:")
        for mistake in stage_4["mistakes"]:
            st.markdown(f"- {mistake}")

        st.markdown("---")

        status_4 = get_stage_status(4)
        if status_4 == "BLOCKED":
            st.info("Complete Stage 3 first")
        else:
            if st.button("Execute Stage 4: Run ML Analysis", key="btn_stage_4"):
                try:
                    # WHY: We call run_ml_analysis from helpers.analysis
                    # This is THE entry point for all ML analysis
                    # NEVER implement ML logic in the UI

                    with st.spinner("Running ML analysis... This may take a moment."):
                        coder, results_df, metrics = run_ml_analysis(
                            df=st.session_state["validated_df"],
                            text_column=st.session_state["text_column"],
                            n_codes=st.session_state["n_codes"],
                            method=st.session_state["method"],
                            min_confidence=st.session_state["min_confidence"],
                            representation=st.session_state["representation"],
                        )

                    st.session_state["coder"] = coder
                    st.session_state["results_df"] = results_df
                    st.session_state["metrics"] = metrics
                    st.session_state["stage_4_complete"] = True
                    reset_downstream_stages(4)

                    st.success(
                        f"Analysis complete. "
                        f"Discovered {metrics.get('n_codes', 'N/A')} codes, "
                        f"coverage: {metrics.get('coverage_pct', 0):.1f}%"
                    )
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        if st.session_state["stage_4_complete"] and st.session_state["metrics"]:
            metrics = st.session_state["metrics"]
            coder = st.session_state["coder"]

            # Metrics in columns
            st.markdown("**Metrics Summary**:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Responses", metrics.get('total_responses', 'N/A'))
            with col2:
                st.metric("Coverage", f"{metrics.get('coverage_pct', 0):.1f}%")
            with col3:
                st.metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.2f}")
            with col4:
                st.metric("Time", f"{metrics.get('execution_time', 0):.1f}s")

            # Codebook preview - sorted by code ID (CODE_01, CODE_02, etc.)
            st.markdown("**Discovered Codebook**:")
            codebook_data = []
            total_responses = metrics.get('total_responses', len(coder.code_assignments) if coder.code_assignments else 1)
            for code_id, info in sorted(coder.codebook.items(), key=lambda x: x[0]):
                # Extract at least 3 sample texts from examples (original, not preprocessed)
                examples = info.get('examples', [])
                sample_texts = []
                for ex in examples[:3]:
                    text = ex.get('text', '')
                    # Truncate long texts for display but keep original wording
                    if len(text) > 100:
                        text = text[:100] + "..."
                    sample_texts.append(f'"{text}"')
                # Pad with empty quotes if fewer than 3 samples
                while len(sample_texts) < 3:
                    sample_texts.append('""')
                sample_text_display = " | ".join(sample_texts)

                # Calculate % of total docs
                pct_of_total = (info['count'] / total_responses * 100) if total_responses > 0 else 0

                codebook_data.append({
                    "Code": code_id,
                    "Label": info.get('llm_label', info['label']),  # Use LLM label if available
                    "Alternative Labels": ", ".join(info.get('alternative_labels', [])[:3]),
                    "Keywords": ", ".join(info['keywords'][:5]),
                    "Sample Text": sample_text_display,
                    "Count": info['count'],
                    "% of Total": f"{pct_of_total:.1f}%",
                    "Confidence": f"{info['avg_confidence']:.2f}"
                })
            st.dataframe(pd.DataFrame(codebook_data), use_container_width=True)

    # --------------------------------------------------------------------------
    # STAGE 5: Diagnostics & Assumptions
    # --------------------------------------------------------------------------
    stage_5 = PIPELINE_STAGES[4]
    with st.expander(
        f"Stage {stage_5['number']}: {stage_5['name']} {render_status_badge(get_stage_status(5))}",
        expanded=st.session_state["stage_4_complete"] and not st.session_state["stage_5_complete"],
    ):
        st.markdown(f"**Purpose**: {stage_5['purpose']}")
        st.markdown(f"**Responsible Module**: `{stage_5['module']}`")
        st.markdown(f"**Functions**: `{stage_5['function']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Required Inputs**:")
            for inp in stage_5["inputs"]:
                st.markdown(f"- {inp}")
        with col2:
            st.markdown("**Outputs/Artifacts**:")
            for out in stage_5["outputs"]:
                st.markdown(f"- {out}")

        st.markdown("**Common Engineering Mistakes**:")
        for mistake in stage_5["mistakes"]:
            st.markdown(f"- {mistake}")

        st.markdown("---")

        status_5 = get_stage_status(5)
        if status_5 == "BLOCKED":
            st.info("Complete Stage 4 first")
        else:
            if st.button("Execute Stage 5: Run Diagnostics", key="btn_stage_5"):
                try:
                    # WHY: We call get_qa_report from helpers.analysis
                    # This function internally uses RigorDiagnostics
                    # NEVER implement diagnostics logic in the UI

                    with st.spinner("Running diagnostics..."):
                        qa_report = get_qa_report(
                            coder=st.session_state["coder"],
                            results_df=st.session_state["results_df"],
                            include_rigor=True,
                        )

                    st.session_state["qa_report"] = qa_report
                    st.session_state["stage_5_complete"] = True
                    reset_downstream_stages(5)

                    st.success("Diagnostics complete. QA Report generated.")
                except Exception as e:
                    st.error(f"Diagnostics failed: {e}")

        if st.session_state["stage_5_complete"] and st.session_state["qa_report"]:
            st.markdown("**QA Report**:")
            # Render full report with markdown formatting
            st.markdown(st.session_state["qa_report"])

    # --------------------------------------------------------------------------
    # STAGE 6: Visualization Generation
    # --------------------------------------------------------------------------
    stage_6 = PIPELINE_STAGES[5]
    with st.expander(
        f"Stage {stage_6['number']}: {stage_6['name']} {render_status_badge(get_stage_status(6))}",
        expanded=st.session_state["stage_5_complete"] and not st.session_state["stage_6_complete"],
    ):
        st.markdown(f"**Purpose**: {stage_6['purpose']}")
        st.markdown(f"**Responsible Module**: `{stage_6['module']}`")
        st.markdown(f"**Functions**: `{stage_6['function']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Required Inputs**:")
            for inp in stage_6["inputs"]:
                st.markdown(f"- {inp}")
        with col2:
            st.markdown("**Outputs/Artifacts**:")
            for out in stage_6["outputs"]:
                st.markdown(f"- {out}")

        st.markdown("**Common Engineering Mistakes**:")
        for mistake in stage_6["mistakes"]:
            st.markdown(f"- {mistake}")

        st.markdown("---")

        status_6 = get_stage_status(6)
        if status_6 == "BLOCKED":
            st.info("Complete Stage 5 first")
        else:
            if st.button("Execute Stage 6: Prepare Visualization Data", key="btn_stage_6"):
                try:
                    # WHY: We call get_top_codes and get_cooccurrence_pairs
                    # from helpers.analysis. These prepare DATA for visualization.
                    # Actual chart rendering would happen in a separate UI layer.

                    top_codes_df = get_top_codes(
                        coder=st.session_state["coder"], n=20
                    )
                    cooccurrence_df = get_cooccurrence_pairs(
                        results_df=st.session_state["results_df"], min_count=2
                    )

                    st.session_state["top_codes_df"] = top_codes_df
                    st.session_state["cooccurrence_df"] = cooccurrence_df
                    st.session_state["stage_6_complete"] = True
                    reset_downstream_stages(6)

                    st.success("Visualization data prepared.")
                except Exception as e:
                    st.error(f"Visualization prep failed: {e}")

        if st.session_state["stage_6_complete"]:
            coder = st.session_state["coder"]
            results_df = st.session_state["results_df"]
            top_codes_df = st.session_state["top_codes_df"]

            # Generate and show insights
            insights = generate_insights(coder, results_df)
            st.markdown("**Key Insights**:")
            for insight in insights:
                st.markdown(insight)

            st.markdown("---")

            # === VISUALIZATION 1: Code Frequency Bar Chart ===
            st.markdown("**Code Frequency Distribution**:")
            if top_codes_df is not None and not top_codes_df.empty:
                chart_data = top_codes_df.set_index('Label')['Count'].head(15)
                st.bar_chart(chart_data)

            # === VISUALIZATION 2: Code Co-occurrence Heatmap ===
            st.markdown("**Code Co-occurrence Matrix**:")
            cooccurrence_df = st.session_state["cooccurrence_df"]
            if cooccurrence_df is not None and not cooccurrence_df.empty:
                # Build matrix for heatmap using code labels instead of code IDs
                codes = list(coder.codebook.keys())
                # Create mapping from code ID to label (prefer LLM labels)
                code_to_label = {code_id: info.get('llm_label', info['label']) for code_id, info in coder.codebook.items()}
                labels = [code_to_label[code] for code in codes]

                # Use labels for both index and columns
                matrix = pd.DataFrame(0, index=labels, columns=labels)
                for _, row in cooccurrence_df.iterrows():
                    c1, c2 = row['Code 1'], row['Code 2']
                    # Convert code IDs to labels
                    label1 = code_to_label.get(c1, c1)
                    label2 = code_to_label.get(c2, c2)
                    if label1 in matrix.index and label2 in matrix.columns:
                        matrix.loc[label1, label2] = row['Count']
                        matrix.loc[label2, label1] = row['Count']
                # Show as heatmap (using dataframe with background gradient)
                st.dataframe(
                    matrix.style.background_gradient(cmap='Blues', axis=None),
                    use_container_width=True
                )
            else:
                st.markdown("*No co-occurrence pairs detected*")

            # === VISUALIZATION 5: Word Frequency Chart ===
            st.markdown("**Top Words**:")
            if PLOTLY_AVAILABLE:
                text_col = st.session_state["text_column"]
                all_text = ' '.join(results_df[text_col].astype(str).tolist())
                if all_text.strip():
                    try:
                        # Simple word frequency using Counter
                        from collections import Counter
                        import re as re_module
                        # Tokenize and clean
                        words = re_module.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
                        # Remove common stop words
                        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has', 'have', 'been', 'were', 'they', 'their', 'what', 'when', 'where', 'who', 'will', 'with', 'would', 'there', 'this', 'that', 'from', 'which', 'more', 'some', 'than', 'into', 'other', 'about', 'these', 'just', 'also', 'very', 'being', 'because'}
                        words = [w for w in words if w not in stop_words]
                        word_counts = Counter(words).most_common(30)

                        if word_counts:
                            word_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
                            fig = px.bar(
                                word_df,
                                x='Count',
                                y='Word',
                                orientation='h',
                                title='Top 30 Words',
                                color='Count',
                                color_continuous_scale='viridis'
                            )
                            fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.markdown("*No words found after filtering*")
                    except Exception as e:
                        st.warning(f"Could not generate word frequency chart: {e}")
                else:
                    st.markdown("*No text content available*")
            else:
                st.info("Word frequency chart requires `plotly`. Install with: `pip install plotly`")

            # === VISUALIZATION 6: Code Network Diagram ===
            st.markdown("**Code Co-occurrence Network**:")
            if NETWORKX_AVAILABLE and cooccurrence_df is not None and not cooccurrence_df.empty:
                try:
                    # Build network graph
                    G = nx.Graph()
                    code_to_label = {code_id: info.get('llm_label', info['label']) for code_id, info in coder.codebook.items()}  # Prefer LLM labels

                    # Add nodes with size based on count
                    for code_id, info in coder.codebook.items():
                        if info['count'] > 0:
                            G.add_node(
                                code_to_label[code_id],
                                count=info['count']
                            )

                    # Add edges from co-occurrences
                    for _, row in cooccurrence_df.iterrows():
                        label1 = code_to_label.get(row['Code 1'], row['Code 1'])
                        label2 = code_to_label.get(row['Code 2'], row['Code 2'])
                        if label1 in G.nodes and label2 in G.nodes:
                            G.add_edge(label1, label2, weight=row['Count'])

                    if len(G.nodes) > 0 and len(G.edges) > 0:
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(12, 8))

                        # Layout
                        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

                        # Node sizes based on count
                        node_sizes = [G.nodes[node].get('count', 1) * 100 + 300 for node in G.nodes]

                        # Edge widths based on weight
                        edge_weights = [G.edges[edge]['weight'] for edge in G.edges]
                        max_weight = max(edge_weights) if edge_weights else 1
                        edge_widths = [2 + (w / max_weight) * 4 for w in edge_weights]

                        # Draw network
                        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                                               edgecolors='darkblue', linewidths=2, ax=ax)
                        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                                               edge_color='gray', ax=ax)
                        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

                        ax.set_title("Code Co-occurrence Network", fontsize=14, fontweight='bold')
                        ax.axis('off')
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.markdown("*Not enough co-occurrences to create network diagram*")
                except Exception as e:
                    st.warning(f"Could not generate network diagram: {e}")
            elif not NETWORKX_AVAILABLE:
                st.info("Network diagram requires the `networkx` package. Install with: `pip install networkx`")
            else:
                st.markdown("*No co-occurrence pairs for network visualization*")

            # === VISUALIZATION 7: Theme Distribution Pie Chart ===
            st.markdown("**Theme Distribution**:")
            if top_codes_df is not None and not top_codes_df.empty:
                # Get top 10 codes for pie chart
                top_10 = top_codes_df.head(10)
                if not top_10.empty and top_10['Count'].sum() > 0:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_10)))

                    wedges, texts, autotexts = ax.pie(
                        top_10['Count'],
                        labels=top_10['Label'],
                        autopct='%1.1f%%',
                        colors=colors,
                        pctdistance=0.75,
                        labeldistance=1.1
                    )

                    # Style the labels
                    for text in texts:
                        text.set_fontsize(9)
                    for autotext in autotexts:
                        autotext.set_fontsize(8)
                        autotext.set_color('white')

                    ax.set_title("Top 10 Themes by Frequency", fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.markdown("*No data available for pie chart*")

            st.markdown("---")

            # =================================================================
            # ADVANCED METHOD-SPECIFIC VISUALIZATIONS
            # These visualizations are from src.method_visualizations and
            # provide deeper insights based on the ML method used.
            # =================================================================
            st.markdown("### Advanced Analytics Visualizations")

            # Get method-specific recommendations
            method = st.session_state.get("method", "tfidf_kmeans")
            viz_availability = get_visualization_availability(method)

            # Show method context
            method_names = {
                'tfidf_kmeans': 'TF-IDF + K-Means (Hard Clustering)',
                'lda': 'LDA - Latent Dirichlet Allocation (Topic Model)',
                'lstm_kmeans': 'LSTM + K-Means (Sequential Patterns)',
                'bert_kmeans': 'BERT + K-Means (Semantic Understanding)',
                'svm': 'SVM Spectral Clustering (Kernel-based)'
            }
            st.info(f"**Current Method**: {method_names.get(method, method)} - Visualizations below are tailored to this method.")

            # Create MethodVisualizer instance
            try:
                visualizer = MethodVisualizer(
                    coder=coder,
                    results_df=results_df,
                    text_column=st.session_state["text_column"],
                    method=method
                )

                # === VISUALIZATION: Cluster Visualization (Scatter or Network) ===
                st.markdown("**Cluster/Topic Visualization**:")
                if PLOTLY_AVAILABLE and SKLEARN_VIZ_AVAILABLE:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col2:
                        viz_type = st.selectbox(
                            "Visualization Type",
                            options=['scatter', 'network'],
                            format_func=lambda x: 'Scatter Plot' if x == 'scatter' else 'Network Diagram',
                            key="cluster_viz_type"
                        )
                    with col3:
                        if viz_type == 'scatter':
                            reduction_method = st.selectbox(
                                "Reduction Method",
                                options=['pca', 'tsne'],
                                format_func=lambda x: 'PCA (Fast)' if x == 'pca' else 't-SNE (Better separation)',
                                key="scatter_reduction_method"
                            )
                        else:
                            network_layout = st.selectbox(
                                "Layout",
                                options=['spring', 'circular', 'kamada_kawai'],
                                format_func=lambda x: {'spring': 'Spring (Force-directed)', 'circular': 'Circular', 'kamada_kawai': 'Kamada-Kawai'}[x],
                                key="network_layout"
                            )

                    if viz_type == 'scatter':
                        scatter_fig = visualizer.create_cluster_scatter(reduction_method=reduction_method)
                        if scatter_fig is not None:
                            st.plotly_chart(scatter_fig, use_container_width=True)
                            if method == 'tfidf_kmeans':
                                st.caption("Shows cluster separation - well-separated clusters indicate good clustering quality.")
                            else:
                                st.caption("Shows document groupings by dominant topic. Less meaningful for topic models than for clustering.")
                        else:
                            st.warning("Could not generate cluster scatter plot.")
                    else:  # network diagram
                        network_fig = visualizer.create_cluster_network(layout=network_layout)
                        if network_fig is not None:
                            st.plotly_chart(network_fig, use_container_width=True)
                            st.caption("Network diagram showing cluster relationships. Node size represents document count, edges show inter-cluster similarity.")
                        else:
                            st.warning("Could not generate network diagram. Need at least 2 clusters.")
                else:
                    st.info("Cluster visualization requires `plotly` and `scikit-learn`. Install with: `pip install plotly scikit-learn`")

                # === VISUALIZATION: Silhouette Plot (Hard clustering methods) ===
                if method in ['tfidf_kmeans', 'lstm_kmeans', 'bert_kmeans', 'svm']:
                    st.markdown("**Silhouette Analysis (Cluster Quality)**:")
                    if PLOTLY_AVAILABLE and SKLEARN_VIZ_AVAILABLE:
                        silhouette_fig = visualizer.create_silhouette_plot()
                        if silhouette_fig is not None:
                            st.plotly_chart(silhouette_fig, use_container_width=True)
                            st.caption("Silhouette coefficients measure cluster cohesion. Values close to 1 indicate well-clustered points; values near 0 or negative indicate overlapping clusters.")
                        else:
                            st.warning("Could not generate silhouette plot.")
                    else:
                        st.info("Silhouette plot requires `plotly` and `scikit-learn`.")
                else:
                    with st.expander("Why no Silhouette Plot?"):
                        st.markdown(f"""
                        **Silhouette analysis is only available for hard clustering methods.**

                        Your current method ({method_names.get(method, method)}) uses **soft topic assignments**
                        where documents can belong to multiple topics with different weights.

                        Silhouette scores require hard cluster assignments (each document in exactly one cluster),
                        which is how K-Means and SVM clustering work.
                        """)

                # === VISUALIZATION: Topic-Term Heatmap ===
                st.markdown("**Topic-Term Heatmap**:")
                if PLOTLY_AVAILABLE:
                    col1, col2 = st.columns([3, 1])
                    with col2:
                        n_terms = st.slider("Terms per topic", min_value=5, max_value=25, value=15, key="heatmap_n_terms")

                    heatmap_fig = visualizer.create_topic_term_heatmap(n_terms=n_terms)
                    if heatmap_fig is not None:
                        st.plotly_chart(heatmap_fig, use_container_width=True)
                        if method == 'lda':
                            st.caption("Shows the weight/probability of each term within each topic. Higher values indicate more characteristic terms.")
                        else:
                            st.caption("Shows cluster centroid weights. Higher values indicate terms more central to the cluster definition.")
                    else:
                        st.warning("Could not generate topic-term heatmap.")
                else:
                    st.info("Topic-term heatmap requires `plotly`. Install with: `pip install plotly`")

                # === VISUALIZATION: Topic Distribution (LDA only) ===
                if method == 'lda':
                    st.markdown("**Topic Distribution per Document**:")
                    if PLOTLY_AVAILABLE:
                        topic_dist_fig = visualizer.create_topic_distribution_chart()
                        if topic_dist_fig is not None:
                            st.plotly_chart(topic_dist_fig, use_container_width=True)
                            st.caption("Shows the topic composition of sampled documents. Each bar represents a document, with colors showing topic weights.")
                        else:
                            st.warning("Could not generate topic distribution chart.")
                    else:
                        st.info("Topic distribution chart requires `plotly`.")
                else:
                    with st.expander("Why no Topic Distribution Chart?"):
                        st.markdown("""
                        **Topic distribution is only available for LDA topic models.**

                        K-Means based methods (TF-IDF, LSTM, BERT) use **hard cluster assignments** where each document belongs to exactly one cluster.
                        There are no "topic weights" to visualize - each document simply belongs to its assigned cluster.

                        For clustering methods, the cluster scatter plot and silhouette analysis provide better insights.
                        """)

                # === VISUALIZATION: pyLDAvis (LDA only) ===
                if method == 'lda':
                    st.markdown("**Interactive LDA Visualization (pyLDAvis)**:")
                    if PYLDAVIS_AVAILABLE:
                        with st.spinner("Generating interactive LDA visualization..."):
                            lda_html = visualizer.create_lda_visualization()
                        if lda_html is not None:
                            import streamlit.components.v1 as components
                            components.html(lda_html, height=800, scrolling=True)
                            st.caption("Interactive exploration of LDA topics. Click on topics to see their top terms; adjust relevance slider to balance frequency vs. distinctiveness.")
                        else:
                            st.warning("Could not generate pyLDAvis visualization.")
                    else:
                        st.info("Interactive LDA visualization requires `pyLDAvis`. Install with: `pip install pyLDAvis`")
                else:
                    with st.expander("Why no pyLDAvis?"):
                        st.markdown("""
                        **pyLDAvis is specifically designed for LDA models.**

                        pyLDAvis relies on LDA's probabilistic structure (Dirichlet priors, document-topic distributions
                        as probabilities summing to 1).

                        K-Means based methods (TF-IDF, LSTM, BERT) produce hard cluster assignments rather than
                        probabilistic topic distributions.

                        The topic-term heatmap and cluster scatter plot provide similar insights for these methods.
                        """)

                # === VISUALIZATION: Per-Cluster Word Clouds ===
                st.markdown("**Per-Cluster/Topic Word Clouds**:")
                if WORDCLOUD_AVAILABLE or PIL_AVAILABLE:
                    try:
                        # Use semantic wordclouds with color-coded meanings
                        with st.spinner("Generating semantic word clouds..."):
                            semantic_fig = visualizer.create_all_semantic_wordclouds(
                                max_words=40,
                                cols=3
                            )

                        if semantic_fig is not None:
                            st.pyplot(semantic_fig, use_container_width=True)
                            plt.close(semantic_fig)
                            st.caption(
                                "Word clouds for each topic/cluster. Word SIZE indicates frequency; "
                                "word COLOR indicates semantic similarity (similar colors = similar meanings)."
                            )

                            # Option to view individual topic wordclouds
                            with st.expander("View Individual Topic Word Cloud"):
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
                            # Fallback to simple wordclouds
                            simple_fig = visualizer.create_all_cluster_wordclouds(
                                max_words=30,
                                cols=3
                            )
                            if simple_fig is not None:
                                st.pyplot(simple_fig, use_container_width=True)
                                plt.close(simple_fig)
                                st.caption("Word clouds for each topic/cluster. Larger words appear more frequently.")
                            else:
                                st.info("Unable to generate word clouds for this dataset.")
                    except Exception as e:
                        st.warning(f"Could not generate word clouds: {str(e)}")
                else:
                    st.info("Word clouds require the `wordcloud` or `PIL` package. Install with: `pip install wordcloud` or `pip install Pillow`")

                # === Method Recommendations Summary ===
                with st.expander("Visualization Recommendations for Your Method"):
                    recommendations = visualizer.get_method_recommendations()

                    st.markdown(f"**Method**: {recommendations['method_description']}")
                    st.markdown("---")

                    st.markdown("**Available Visualizations:**")
                    for viz_name, viz_info in recommendations['visualizations'].items():
                        status = "Available" if viz_info['available'] else "Not installed"
                        priority = viz_info['priority'].upper()
                        st.markdown(f"- **{viz_name}** [{priority}] - {viz_info['description']}")
                        st.markdown(f"  - _{viz_info['note']}_")

                    if recommendations['not_available']:
                        st.markdown("**Not Available for This Method:**")
                        for viz_name, reason in recommendations['not_available'].items():
                            st.markdown(f"- **{viz_name}**: {reason}")

                    st.markdown("**Notes:**")
                    for note in recommendations['notes']:
                        st.markdown(f"- {note}")

            except Exception as e:
                st.error(f"Error creating advanced visualizations: {e}")
                import traceback
                st.code(traceback.format_exc())

            st.markdown("---")

            # === Data Tables ===
            st.markdown("**Top Codes Table**:")
            if top_codes_df is not None:
                st.dataframe(top_codes_df, use_container_width=True)

            st.markdown("**Co-occurrence Pairs Table**:")
            if cooccurrence_df is not None and not cooccurrence_df.empty:
                # Create a copy with labels instead of code IDs
                display_df = cooccurrence_df.copy()
                code_to_label = {code_id: info.get('llm_label', info['label']) for code_id, info in coder.codebook.items()}  # Prefer LLM labels

                # Replace code IDs with labels
                display_df['Code 1 Label'] = display_df['Code 1'].map(code_to_label)
                display_df['Code 2 Label'] = display_df['Code 2'].map(code_to_label)

                # Reorder columns to show labels prominently
                display_cols = ['Code 1 Label', 'Code 2 Label', 'Count', 'Percentage']
                # Only include columns that exist
                display_cols = [col for col in display_cols if col in display_df.columns]
                st.dataframe(display_df[display_cols].head(15), use_container_width=True)
            else:
                st.markdown("*No co-occurrence pairs with min_count >= 2*")

    # --------------------------------------------------------------------------
    # STAGE 7: Export & Reporting
    # --------------------------------------------------------------------------
    stage_7 = PIPELINE_STAGES[6]
    with st.expander(
        f"Stage {stage_7['number']}: {stage_7['name']} {render_status_badge(get_stage_status(7))}",
        expanded=st.session_state["stage_6_complete"] and not st.session_state["stage_7_complete"],
    ):
        st.markdown(f"**Purpose**: {stage_7['purpose']}")
        st.markdown(f"**Responsible Module**: `{stage_7['module']}`")
        st.markdown(f"**Functions**: `{stage_7['function']}`")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Required Inputs**:")
            for inp in stage_7["inputs"]:
                st.markdown(f"- {inp}")
        with col2:
            st.markdown("**Outputs/Artifacts**:")
            for out in stage_7["outputs"]:
                st.markdown(f"- {out}")

        st.markdown("**Common Engineering Mistakes**:")
        for mistake in stage_7["mistakes"]:
            st.markdown(f"- {mistake}")

        st.markdown("---")

        status_7 = get_stage_status(7)
        if status_7 == "BLOCKED":
            st.info("Complete Stage 6 first")
        else:
            if st.button("Execute Stage 7: Generate Exports", key="btn_stage_7"):
                try:
                    # WHY: We call export_results_package and generate_methods_documentation
                    # from helpers.analysis. Export logic NEVER belongs in UI.

                    with st.spinner("Generating exports..."):
                        excel_bytes = export_results_package(
                            coder=st.session_state["coder"],
                            results_df=st.session_state["results_df"],
                            format="excel",
                        )

                        methods_doc = generate_methods_documentation(
                            coder=st.session_state["coder"],
                            results_df=st.session_state["results_df"],
                            metrics=st.session_state["metrics"],
                        )

                        exec_summary = generate_executive_summary(
                            coder=st.session_state["coder"],
                            results_df=st.session_state["results_df"],
                            metrics=st.session_state["metrics"],
                            include_methods=False,
                        )

                    st.session_state["excel_bytes"] = excel_bytes
                    st.session_state["methods_doc"] = methods_doc
                    st.session_state["executive_summary"] = exec_summary
                    st.session_state["stage_7_complete"] = True

                    st.success("All exports generated. Pipeline complete!")
                except Exception as e:
                    st.error(f"Export generation failed: {e}")

        if st.session_state["stage_7_complete"]:
            st.markdown("**Available Downloads**:")

            if st.session_state["excel_bytes"]:
                st.download_button(
                    label="Download Excel Package",
                    data=st.session_state["excel_bytes"],
                    file_name="analysis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            if st.session_state["methods_doc"]:
                st.download_button(
                    label="Download Methods Documentation (MD)",
                    data=st.session_state["methods_doc"],
                    file_name="METHODS.md",
                    mime="text/markdown",
                )

            if st.session_state["executive_summary"]:
                st.download_button(
                    label="Download Executive Summary (MD)",
                    data=st.session_state["executive_summary"],
                    file_name="EXECUTIVE_SUMMARY.md",
                    mime="text/markdown",
                )

    # ==========================================================================
    # SECTION: ARTIFACTS & OUTPUTS PANEL
    # ==========================================================================
    st.markdown("---")
    st.header("Artifacts & Outputs Summary")

    st.markdown("""
    This panel shows the current state of all pipeline artifacts.
    All values are **read-only** displays of session state.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Data Artifacts**")
        st.markdown(f"- Raw DataFrame: {'Present' if st.session_state['raw_df'] is not None else 'None'}")
        st.markdown(f"- Validated DataFrame: {'Present' if st.session_state['validated_df'] is not None else 'None'}")
        st.markdown(f"- Text Column: `{st.session_state['text_column'] or 'Not set'}`")

    with col2:
        st.markdown("**Analysis Artifacts**")
        st.markdown(f"- MLOpenCoder: {'Present' if st.session_state['coder'] is not None else 'None'}")
        st.markdown(f"- Results DataFrame: {'Present' if st.session_state['results_df'] is not None else 'None'}")
        st.markdown(f"- Metrics Dict: {'Present' if st.session_state['metrics'] is not None else 'None'}")

    with col3:
        st.markdown("**Export Artifacts**")
        st.markdown(f"- QA Report: {'Present' if st.session_state['qa_report'] else 'None'}")
        st.markdown(f"- Excel Bytes: {'Present' if st.session_state['excel_bytes'] else 'None'}")
        st.markdown(f"- Methods Doc: {'Present' if st.session_state['methods_doc'] else 'None'}")

    # ==========================================================================
    # SECTION: SEPARATION OF CONCERNS CALLOUTS
    # ==========================================================================
    st.markdown("---")
    st.header("Separation of Concerns")

    st.markdown("""
    **This section documents what belongs where. Violating these boundaries
    leads to unmaintainable code.**
    """)

    with st.expander("What MUST NEVER be done in Streamlit UI code", expanded=False):
        st.markdown("""
        - **Data cleaning/preprocessing logic** - Use `helpers.analysis.preprocess_responses()`
        - **ML model training** - Use `helpers.analysis.run_ml_analysis()`
        - **Vectorization/embedding computation** - Use `src.embeddings` via `run_ml_analysis()`
        - **Statistical calculations** - Use `src.rigor_diagnostics.RigorDiagnostics`
        - **File format handling (Excel sheets)** - Use `helpers.analysis.export_results_package()`
        - **Text generation (methods docs)** - Use `src.methods_documentation`
        - **Direct sklearn/numpy operations** - Wrap in pipeline functions

        **Why**: UI code runs on every widget interaction. Expensive operations
        in callbacks cause lag and duplicate computation.
        """)

    with st.expander("What belongs in the Pipeline Layer (helpers/, src/)", expanded=False):
        st.markdown("""
        - All ML model initialization and training
        - Text preprocessing (tokenization, normalization)
        - Feature extraction (TF-IDF, embeddings)
        - Clustering/topic modeling algorithms
        - Quality metrics calculation
        - Validity assessment and bias detection
        - Codebook generation and code assignment
        - Co-occurrence matrix computation
        - Any operation that takes > 100ms

        **Why**: Pipeline code is testable, reusable, and cacheable.
        It can be called from CLI, notebooks, or any UI.
        """)

    with st.expander("What belongs in Exports/Reporting Layer", expanded=False):
        st.markdown("""
        - Excel workbook generation with multiple sheets
        - Markdown documentation generation
        - PDF report generation (if added)
        - JSON export for API consumption
        - Codebook export in various formats

        **Why**: Export logic is decoupled from analysis. Same analysis
        can produce different export formats without re-running models.
        """)

    with st.expander("What Streamlit UI code SHOULD do", expanded=False):
        st.markdown("""
        - Display data and results (st.dataframe, st.markdown)
        - Collect user input (st.selectbox, st.slider, st.button)
        - Manage session state for pipeline tracking
        - Provide progress feedback (st.spinner, st.progress)
        - Offer download buttons for generated exports
        - Handle navigation between conceptual stages

        **Why**: UI is a thin layer that orchestrates pipeline calls
        and displays results. It should be replaceable without losing analysis logic.
        """)

    # ==========================================================================
    # SECTION: RESET BUTTON
    # ==========================================================================
    st.markdown("---")

    if st.button("Reset Pipeline (Clear All State)", key="btn_reset"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
