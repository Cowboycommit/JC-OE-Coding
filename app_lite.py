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
    1. Dataset Ingestion
    2. Data Validation & Typing
    3. Method Eligibility Checks
    4. Model Execution
    5. Diagnostics & Assumptions
    6. Visualization Generation
    7. Export & Reporting

AUTHOR: Engineering Team
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np

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
)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# Available sample datasets in the project
SAMPLE_DATASETS = {
    "sample_responses.csv": "Sample Responses (General)",
    "cricket_responses.csv": "Cricket Responses",
    "fashion_responses.csv": "Fashion Responses",
    "consumer_perspectives_responses.csv": "Consumer Perspectives",
    "cultural_commentary_responses.csv": "Cultural Commentary",
    "industry_professional_responses.csv": "Industry Professional",
    "20_newsgroups.csv": "20 Newsgroups (Large)",
    "reuters21578.csv": "Reuters (Large)",
}

DATA_DIR = Path(__file__).parent / "data"

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
        "name": "Data Validation & Typing",
        "purpose": "Validate data structure and preprocess text responses",
        "inputs": ["Raw DataFrame", "Text column name", "Preprocessing options"],
        "module": "helpers.analysis",
        "function": "validate_dataframe(), preprocess_responses()",
        "outputs": ["Validated DataFrame", "Preprocessed responses"],
        "mistakes": [
            "Skipping validation before analysis",
            "Not removing null/empty responses",
            "Not documenting preprocessing decisions",
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
            "Using LDA/NMF with semantic embeddings (incompatible)",
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
        "purpose": "Create visual representations of analysis results",
        "inputs": ["MLOpenCoder", "Results DataFrame", "Metrics"],
        "module": "helpers.analysis (data prep), UI (rendering only)",
        "function": "get_top_codes(), get_cooccurrence_pairs()",
        "outputs": ["Frequency tables", "Co-occurrence data", "Chart data"],
        "mistakes": [
            "Computing chart data in Streamlit callbacks",
            "Not caching visualization data",
            "Putting Plotly/chart logic in analysis modules",
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
    "nmf": "NMF - Non-negative Matrix Factorization (Topic modeling)",
}

REPRESENTATIONS = {
    "tfidf": "TF-IDF (Default, fast, bag-of-words)",
    "sbert": "SentenceBERT (Semantic, offline)",
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

    1. **Dataset Ingestion** - Load data from file into memory
    2. **Data Validation & Typing** - Validate structure, preprocess text
    3. **Method Eligibility Checks** - Verify ML method compatibility
    4. **Model Execution** - Run clustering/topic modeling
    5. **Diagnostics & Assumptions** - Assess validity, detect bias
    6. **Visualization Generation** - Prepare chart data
    7. **Export & Reporting** - Package results for consumption

    Each stage must complete before the next can begin. This prevents
    mixing UI logic with analytics logic.
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

        dataset_options = ["-- Select a sample dataset --"] + list(SAMPLE_DATASETS.keys())
        selected_dataset = st.selectbox(
            "Sample Dataset",
            options=dataset_options,
            format_func=lambda x: SAMPLE_DATASETS.get(x, x),
            key="dataset_select",
        )

        if st.button("Execute Stage 1: Load Data", key="btn_stage_1"):
            if selected_dataset != "-- Select a sample dataset --":
                try:
                    # Load from project's data directory
                    filepath = DATA_DIR / selected_dataset
                    loader = DataLoader()
                    df = loader.load_csv(str(filepath))

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
            st.markdown("**Artifact Preview (first 5 rows)**:")
            st.dataframe(st.session_state["raw_df"].head(), use_container_width=True)

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

                # Simplified preprocessing with sensible defaults
                col1, col2 = st.columns(2)
                with col1:
                    remove_nulls = st.checkbox("Remove null responses", value=True)
                with col2:
                    remove_duplicates = st.checkbox("Remove duplicates", value=False)

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
                                min_length=5,  # Sensible default
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
            st.markdown(f"**Text Column**: `{st.session_state['text_column']}`")
            st.markdown(f"**Rows after preprocessing**: {len(st.session_state['validated_df'])}")

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
            # Simplified configuration with sensible defaults
            col1, col2 = st.columns(2)

            with col1:
                method = st.selectbox(
                    "ML Method",
                    options=list(ML_METHODS.keys()),
                    format_func=lambda x: ML_METHODS[x],
                    key="method_select",
                )

                n_codes = st.slider(
                    "Number of codes",
                    min_value=3,
                    max_value=20,
                    value=8,
                    key="n_codes_slider",
                )

            with col2:
                # Only show TF-IDF for LDA/NMF compatibility
                if method in ["lda", "nmf"]:
                    representation = "tfidf"
                    st.info("Using TF-IDF (required for LDA/NMF)")
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

            if st.button("Execute Stage 3: Check Eligibility", key="btn_stage_3"):
                # Check n_codes vs dataset size
                n_samples = len(st.session_state["validated_df"])
                if n_codes > n_samples:
                    st.error(
                        f"Cannot request {n_codes} codes with only {n_samples} samples. "
                        f"Reduce n_codes to at most {n_samples}."
                    )
                else:
                    st.session_state["method"] = method
                    st.session_state["representation"] = representation
                    st.session_state["n_codes"] = n_codes
                    st.session_state["min_confidence"] = min_confidence
                    st.session_state["stage_3_complete"] = True
                    reset_downstream_stages(3)

                    st.success(f"Configuration set: {ML_METHODS[method]}, {n_codes} codes")

        if st.session_state["stage_3_complete"]:
            st.markdown(f"**Method**: `{st.session_state['method']}`")
            st.markdown(f"**Representation**: `{st.session_state['representation']}`")
            st.markdown(f"**N Codes**: `{st.session_state['n_codes']}`")
            st.markdown(f"**Min Confidence**: `{st.session_state['min_confidence']}`")

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
            st.markdown("**Metrics Summary**:")
            st.markdown(f"- Total Responses: {metrics.get('total_responses', 'N/A')}")
            st.markdown(f"- Total Assignments: {metrics.get('total_assignments', 'N/A')}")
            st.markdown(f"- Coverage: {metrics.get('coverage_pct', 0):.1f}%")
            st.markdown(f"- Avg Confidence: {metrics.get('avg_confidence', 0):.3f}")
            st.markdown(f"- Execution Time: {metrics.get('execution_time', 0):.2f}s")

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
            st.markdown("**QA Report Preview (first 1000 chars)**:")
            st.text(st.session_state["qa_report"][:1000] + "...")

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
            st.markdown("**Top Codes (Data Only)**:")
            if st.session_state["top_codes_df"] is not None:
                st.dataframe(st.session_state["top_codes_df"], use_container_width=True)

            st.markdown("**Co-occurrence Pairs (Data Only)**:")
            if st.session_state["cooccurrence_df"] is not None and not st.session_state["cooccurrence_df"].empty:
                st.dataframe(st.session_state["cooccurrence_df"].head(10), use_container_width=True)
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
