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
    calculate_metrics_summary,
    generate_insights,
    get_analysis_summary,
    get_top_codes,
    get_cooccurrence_pairs,
    export_results_package
)

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
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: left;
        margin-bottom: 2rem;
        font-family: sans-serif;
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


def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">📊 ML-Based Open Coding Analysis</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>Welcome!</strong> This tool uses machine learning to automatically discover themes
    and code qualitative data. Upload your responses and let the algorithms find patterns.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Open+Coding", use_container_width=True)
        st.markdown("### 🎯 Navigation")

        page = st.radio(
            "Choose a section:",
            [
                "📤 Data Upload",
                "⚙️ Configuration",
                "🚀 Run Analysis",
                "📊 Results Overview",
                "📈 Visualizations",
                "💾 Export Results",
                "ℹ️ About"
            ],
            label_visibility="collapsed"
        )

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

    # Sample data option
    if st.button("Load Sample Data", use_container_width=True):
        # Create sample data
        sample_data = {
            'response': [
                "I love the flexibility of remote work and the better work-life balance it provides.",
                "Communication challenges and feeling isolated are major issues with remote work.",
                "Remote work has improved my productivity significantly due to fewer distractions.",
                "I miss the social interaction and collaboration from the office environment.",
                "The flexibility to work from anywhere is the best part of remote work.",
                "Video call fatigue and technology issues make remote work challenging.",
                "I appreciate being able to spend more time with family while working remotely.",
                "It's difficult to separate work and personal life when working from home.",
                "Remote work has eliminated my commute and reduced stress levels.",
                "I struggle with motivation and staying focused when working remotely."
            ] * 5  # Repeat for more data
        }

        st.session_state.uploaded_df = pd.DataFrame(sample_data)
        st.success("✅ Sample data loaded! Go to Configuration to continue.")
        st.rerun()

    # Upload your data section
    st.markdown("---")
    st.markdown("### 📤 Or Upload Your Data")

    st.markdown("""
    Upload a CSV or Excel file containing your qualitative responses.
    Your file should have at least one column with text responses.
    """)

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

            # Show success message
            st.markdown(f"""
            <div class="success-box">
            ✅ <strong>File uploaded successfully!</strong><br>
            Loaded {len(df):,} rows and {len(df.columns)} columns
            </div>
            """, unsafe_allow_html=True)

            # Display column selector
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

            # Column info
            with st.expander("📋 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.astype(str),
                    'Non-Null': df.count().values,
                    'Null Count': df.isnull().sum().values,
                    'Unique Values': df.nunique().values
                })
                st.dataframe(col_info, use_container_width=True)

            # Data preprocessing options
            st.markdown("### 🔧 Preprocessing Options")

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
                        st.error("Please select a text column in the Configuration page first")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")


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

    col1, col2 = st.columns(2)

    with col1:
        n_codes = st.slider(
            "Number of codes to discover",
            min_value=3,
            max_value=30,
            value=10,
            step=1,
            help="How many themes/codes should the algorithm discover?"
        )

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
        st.metric("Codes to Find", n_codes)
    with config_col3:
        st.metric("Algorithm", method.upper())

    st.success("✅ Configuration saved! Go to 'Run Analysis' to start.")


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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Responses", f"{len(df):,}")
    with col2:
        st.metric("Text Column", config['text_column'])
    with col3:
        st.metric("Codes", config['n_codes'])
    with col4:
        st.metric("Method", config['method'].upper())

    st.markdown("---")

    # Run button
    if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)

        try:
            # Run analysis
            start_time = time.time()

            coder, results_df, metrics = run_ml_analysis(
                df=df,
                text_column=config['text_column'],
                n_codes=config['n_codes'],
                method=config['method'],
                min_confidence=config['min_confidence'],
                progress_callback=update_progress
            )

            # Save to session state
            st.session_state.coder = coder
            st.session_state.results_df = results_df
            st.session_state.metrics = metrics
            st.session_state.analysis_complete = True

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

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)

    # Show previous results if available
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.markdown("### ✅ Previous Analysis Available")

        col1, col2 = st.columns(2)
        with col1:
            st.info("Analysis results are ready to view")
        with col2:
            if st.button("View Results", use_container_width=True):
                st.session_state.current_page = "📊 Results Overview"
                st.rerun()


def page_results_overview():
    """Results overview page."""
    st.markdown('<h2 class="sub-header">📊 Results Overview</h2>', unsafe_allow_html=True)

    if not st.session_state.analysis_complete:
        st.warning("⚠️ Please run the analysis first")
        return

    coder = st.session_state.coder
    results_df = st.session_state.results_df
    metrics = st.session_state.metrics

    # Metrics overview
    st.markdown("### 📈 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Responses",
            f"{metrics.get('total_responses', 0):,}"
        )

    with col2:
        st.metric(
            "Codes Found",
            metrics.get('n_codes', 0),
            delta=f"{metrics.get('active_codes', 0)} active" if 'active_codes' in metrics else None
        )

    with col3:
        st.metric(
            "Avg Codes/Response",
            f"{metrics.get('avg_codes_per_response', 0):.2f}"
        )

    with col4:
        st.metric(
            "Coverage",
            f"{metrics.get('coverage_pct', 0):.1f}%"
        )

    # Key insights
    st.markdown("---")
    st.markdown("### 💡 Key Insights")

    insights = generate_insights(coder, results_df)
    for insight in insights:
        st.markdown(insight)

    # Top codes
    st.markdown("---")
    st.markdown("### 🏆 Top Codes")

    top_codes_df = get_top_codes(coder, n=10)

    # Display as styled table
    st.dataframe(
        style_frequency_table(top_codes_df),
        use_container_width=True,
        height=400
    )

    # Code assignments sample
    st.markdown("---")
    st.markdown("### 📋 Sample Code Assignments")

    sample_size = min(10, len(results_df))
    sample_df = results_df[[
        st.session_state.config['text_column'],
        'assigned_codes',
        'num_codes'
    ]].head(sample_size)

    # Format for display
    display_df = sample_df.copy()
    display_df['assigned_codes'] = display_df['assigned_codes'].apply(
        lambda x: ', '.join(x) if x else 'None'
    )

    st.dataframe(display_df, use_container_width=True, height=400)

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


def page_visualizations():
    """Visualizations page."""
    st.markdown('<h2 class="sub-header">📈 Visualizations</h2>', unsafe_allow_html=True)

    if not st.session_state.analysis_complete:
        st.warning("⚠️ Please run the analysis first")
        return

    coder = st.session_state.coder
    results_df = st.session_state.results_df

    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Frequency",
        "🔥 Heatmap",
        "🕸️ Network",
        "📉 Distribution",
        "🎯 Confidence"
    ])

    with tab1:
        st.markdown("### Code Frequency Distribution")

        top_codes_df = get_top_codes(coder, n=15)

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

    with tab2:
        st.markdown("### Co-occurrence Heatmap")

        # Build co-occurrence matrix
        codes = list(coder.codebook.keys())
        n = len(codes)
        cooccur = np.zeros((n, n))

        for assigned_codes in results_df['assigned_codes']:
            for i, code1 in enumerate(codes):
                for j, code2 in enumerate(codes):
                    if code1 in assigned_codes and code2 in assigned_codes:
                        cooccur[i, j] += 1

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

        # Co-occurrence pairs table
        st.markdown("#### Top Co-occurring Pairs")
        pairs_df = get_cooccurrence_pairs(results_df, min_count=2)
        if not pairs_df.empty:
            st.dataframe(pairs_df.head(10), use_container_width=True)
        else:
            st.info("No significant co-occurrences found")

    with tab3:
        st.markdown("### Code Network Diagram")
        st.info("Network visualization requires networkx library")

        # Simple scatter plot as alternative
        top_codes_df = get_top_codes(coder, n=20)

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

        all_confidences = [
            conf for confs in results_df['confidence_scores'] for conf in confs
        ]

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
        summary = get_analysis_summary(coder, results_df)

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
