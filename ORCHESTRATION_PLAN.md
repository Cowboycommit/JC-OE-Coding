# ML Open-Ended Coding Project: Orchestration Plan

**Version:** 1.0
**Date:** 2025-12-25
**Branch:** `claude/orchestration-plan-ml-T8ZOr`
**Orchestrator Role:** Methodological Lead & Systems Integrator

---

## Executive Summary

This orchestration plan establishes a framework for coordinating specialist agents to enhance the ML open-ended coding project while maintaining methodological rigor, qualitative validity, and human-centered analysis. The plan ensures that:

- **Analytic integrity** is preserved (no regression of existing functionality)
- **Uncertainty is surfaced**, not suppressed
- **Human judgment** remains central and auditable
- **Multi-label coding** and emergent themes are supported
- **Confidence and rationale** accompany all automated outputs

---

## 1. PROJECT INSPECTION

### 1.1 Current State Assessment

**Unit of Analysis:**
- **Response-level only** - Complete survey responses as atomic units
- No sentence, turn, or paragraph-level segmentation
- Responses treated as independent, single-perspective statements

**Current Representation Methods:**

1. **TF-IDF Vectorization (Primary)**
   - Bigrams (1,2), max_features=1000
   - Bag-of-words approach, no semantic embeddings
   - English-only stop word removal

2. **Count Vectorization**
   - Used for LDA topic modeling
   - Same preprocessing as TF-IDF

3. **No Advanced Embeddings**
   - No word2vec, GloVe, FastText
   - No BERT, transformers, or contextual embeddings
   - No sentence-level semantic representations

**Existing Clustering/Topic Modeling:**

1. **TF-IDF + K-Means** (Default, recommended)
   - Distance-based cluster assignment
   - Probabilistic confidence scores (min_confidence=0.3)
   - Multi-label support (responses can receive 0 to N codes)

2. **Latent Dirichlet Allocation (LDA)**
   - Probabilistic topic distributions
   - Document-topic soft assignments

3. **Non-negative Matrix Factorization (NMF)**
   - Parts-based decomposition
   - Feature-topic matrices

**Quality Metrics:**
- Silhouette score (cluster cohesion)
- Calinski-Harabasz score (cluster separation)
- Davies-Bouldin score (cluster similarity)

**Implicit Assumptions About "Themes" or "Labels":**

⚠️ **CRITICAL FINDINGS:**

1. **Auto-Generated Code Labels**
   ```python
   # Code labels generated from top 3 keywords
   top_words_list[:3].title()
   # Example: "Remote Work Flexibility"
   ```
   - **Risk:** Labels may not reflect true thematic meaning
   - **Risk:** Keyword-based naming lacks interpretive nuance
   - **Risk:** May create false sense of "ground truth"

2. **Hardcoded Code IDs**
   - Format: `CODE_01, CODE_02, ..., CODE_N`
   - **Risk:** Implies fixed, predetermined taxonomy
   - **Risk:** Not emergent from data

3. **Fixed Confidence Threshold**
   - Default: 0.3 (30% probability)
   - **Risk:** Arbitrary cutoff may vary by context
   - **Risk:** Not adaptive to data characteristics

4. **English-Only Assumption**
   - Hardcoded `stop_words='english'`
   - **Risk:** Excludes multilingual data
   - **Risk:** Cultural bias in preprocessing

5. **Response Independence Assumption**
   - No consideration for conversational context
   - No tracking of respondent-level patterns
   - **Risk:** Misses relational or contextual meaning

6. **Uniform Importance Weighting**
   - All responses treated equally
   - No demographic or contextual weighting
   - **Risk:** May suppress minority perspectives

### 1.2 Signal vs Non-Analytic Content Detection

**Currently Implemented:**

1. **Null Response Removal**
   - `remove_nulls=True` (default)
   - Drops empty, NaN, or None responses

2. **Minimum Length Filtering**
   - `min_length=5` characters
   - **Gap:** Arbitrary threshold, not content-aware

3. **Duplicate Detection**
   - `remove_duplicates=False` (default, opt-in)
   - **Gap:** No fuzzy matching for near-duplicates

**NOT Currently Implemented:**

❌ **Non-Analytic Content Detection:**
- No identification of "I don't know", "N/A", "No comment"
- No detection of gibberish or random keystrokes
- No filtering of test responses ("test", "asdf")
- No spam or bot detection
- No language detection for non-English responses

❌ **Signal Quality Assessment:**
- No measurement of response informativeness
- No detection of copy-paste answers
- No identification of sarcasm or non-literal language
- No sentiment-adjusted filtering

**Integration Opportunity:** Build explicit non-analytic content classifier to flag (not remove) low-signal responses for human review.

---

## 2. INTEGRATION POINTS

### 2.1 Text Segmentation & Unit Handling

**Current State:**
- Response-level only (atomic unit)
- No sub-response segmentation

**Integration Point:** `helpers/analysis.py:MLOpenCoder.preprocess_text()`

**Enhancement Opportunities:**

1. **Multi-Granularity Analysis**
   - Add optional sentence-level segmentation
   - Support paragraph-level chunking for long responses
   - Enable turn-level analysis for conversational data
   - Maintain response-level as default for backward compatibility

2. **Contextual Windowing**
   - Track preceding/following sentences for context
   - Preserve response boundaries in outputs
   - Link segments to parent response for traceability

3. **Hierarchical Unit Management**
   ```
   Response → Paragraphs → Sentences → Phrases
   └─ Preserve parent-child relationships
   └─ Aggregate codes upward (sentence → response)
   └─ Explicit handling of ambiguity across units
   ```

**Delegation:** Assign to **Text Processing Specialist Agent**

---

### 2.2 Signal vs Non-Analytic Content Detection

**Integration Point:** `src/data_loader.py:DataLoader.load_data()` + new `ContentQualityFilter` module

**Enhancement Strategy:**

1. **Create New Module:** `src/content_quality.py`
   ```python
   class ContentQualityFilter:
       def assess_signal(self, text: str) -> dict:
           return {
               'is_analytic': bool,           # True/False
               'confidence': float,           # 0.0-1.0
               'reason': str,                 # Human-readable explanation
               'recommendation': str,         # 'include', 'review', 'exclude'
               'flags': List[str]            # ['too_short', 'non_english', etc.]
           }
   ```

2. **Detection Rules (Transparent & Auditable):**
   - Minimum meaningful word count (not just character count)
   - Non-response patterns: regex for "n/a", "idk", "no comment"
   - Language detection (flag non-English, don't auto-exclude)
   - Gibberish detection (high consonant ratio, keyboard walks)
   - Test response patterns: "test", "asdf", "123", etc.

3. **Integration into Pipeline:**
   ```python
   # In DataLoader.load_data()
   quality_filter = ContentQualityFilter()
   df['quality_assessment'] = df['response'].apply(quality_filter.assess_signal)
   df['is_analytic'] = df['quality_assessment'].apply(lambda x: x['is_analytic'])
   df['quality_flags'] = df['quality_assessment'].apply(lambda x: x['flags'])

   # Export for human review
   df[df['is_analytic'] == False].to_csv('non_analytic_responses.csv')
   ```

4. **No Silent Drops:**
   - Never automatically exclude responses
   - Flag for human review
   - Document all filtering decisions
   - Allow override in configuration

**Delegation:** Assign to **Content Quality Specialist Agent**

---

### 2.3 Semantic Representation / Embedding Generation

**Current State:**
- TF-IDF only (bag-of-words)
- No semantic embeddings

**Integration Point:** `helpers/analysis.py:MLOpenCoder.__init__()` + new embedding options

**Enhancement Strategy:**

1. **Add Embedding Methods (Optional, Not Default):**
   ```python
   representation_methods = {
       'tfidf': TfidfVectorizer,          # Current default
       'word2vec': Word2VecEmbeddings,    # NEW: Average word vectors
       'sentence_bert': SentenceBERT,     # NEW: Sentence transformers
       'openai': OpenAIEmbeddings,        # NEW: API-based (opt-in)
       'fasttext': FastTextEmbeddings,    # NEW: Subword embeddings
   }
   ```

2. **Backward Compatibility:**
   - Keep `method='tfidf_kmeans'` as default
   - Add new methods: `'sbert_kmeans'`, `'word2vec_kmeans'`, etc.
   - Document trade-offs (speed vs. semantic richness)

3. **Embedding Quality Metrics:**
   - Semantic coherence of clusters (beyond statistical metrics)
   - Human evaluation of representative quotes
   - Comparison across embedding methods

4. **Integration into Existing Pipeline:**
   ```python
   class MLOpenCoder:
       def __init__(self, ..., representation='tfidf', embedding_model=None):
           if representation == 'tfidf':
               self.vectorizer = TfidfVectorizer(...)
           elif representation == 'sbert':
               self.vectorizer = SentenceTransformerEmbedder(model=embedding_model)
           # ... etc.
   ```

5. **Transparency Requirements:**
   - Document which embedding method used
   - Include in method_documentation output
   - Warn about API costs for OpenAI embeddings
   - Provide offline alternatives (SBERT, FastText)

**Delegation:** Assign to **NLP/Embedding Specialist Agent**

---

### 2.4 Theme Discovery & Clustering

**Current State:**
- Fixed K-Means/LDA/NMF with predetermined `n_codes`
- Auto-generated labels from top keywords
- Optimal code selection via silhouette analysis

**Integration Point:** `helpers/analysis.py:MLOpenCoder.fit()` + enhanced theme discovery

**Enhancement Strategy:**

1. **Emergent Theme Discovery (Not Predetermined):**
   ```python
   class EmergentThemeDiscovery:
       def discover_themes(self, data, min_themes=3, max_themes=15):
           # Hierarchical clustering to find natural groupings
           # DBSCAN for density-based discovery (outlier-aware)
           # Topic modeling with automatic topic count selection
           return {
               'recommended_n_themes': int,
               'theme_hierarchy': dict,      # Parent-child relationships
               'outlier_responses': list,    # Responses that don't fit
               'rationale': str              # Why this number of themes?
           }
   ```

2. **Hierarchical Theme Structure:**
   - Support multi-level themes (parent codes with child sub-codes)
   - Allow themes to merge or split based on data
   - Track theme evolution across iterations

3. **Label Generation Improvements:**
   - Move from keyword-only to interpretive labels
   - Suggest labels, require human validation
   - Support manual label override
   - Track label history (renamed, merged, split)

4. **Uncertainty Handling:**
   - Identify ambiguous boundaries between themes
   - Flag responses that could belong to multiple themes
   - Provide confidence intervals for theme assignments
   - Surface "uncategorizable" responses (don't force fit)

5. **Integration:**
   ```python
   # In MLOpenCoder.fit()
   if auto_discover_themes:
       discovery = EmergentThemeDiscovery()
       result = discovery.discover_themes(processed_data)

       # Present to user for validation
       st.info(f"Discovered {result['recommended_n_themes']} themes")
       st.write(f"Rationale: {result['rationale']}")

       # Allow override
       n_codes = st.number_input("Adjust if needed",
                                  value=result['recommended_n_themes'])
   ```

**Delegation:** Assign to **Theme Discovery Specialist Agent**

---

### 2.5 Multi-Label Coding Assignment

**Current State:**
- ✅ Multi-label support exists (responses can receive 0 to N codes)
- ✅ Confidence scores tracked per code
- ✅ Ambiguous response identification (3+ codes)

**Integration Point:** `helpers/analysis.py:MLOpenCoder._assign_codes()`

**Enhancement Strategy:**

1. **Preserve & Enhance Existing Multi-Label Logic:**
   - Keep current probabilistic assignment
   - Add explicit "No Code Assigned" handling (not just 0 codes)
   - Track reason for no assignment (low signal vs. low confidence vs. out-of-scope)

2. **Ambiguity Surfacing (Not Suppression):**
   ```python
   def assess_ambiguity(self, response_codes: List[dict]) -> dict:
       return {
           'is_ambiguous': bool,                  # Multiple high-confidence codes
           'ambiguity_type': str,                 # 'multi_faceted', 'boundary_case', 'conflicting'
           'code_combinations': List[tuple],      # Which codes co-occur?
           'confidence_spread': float,            # Entropy of confidence scores
           'human_review_priority': str          # 'high', 'medium', 'low'
       }
   ```

3. **Code Co-Occurrence Analysis (Already Implemented):**
   - ✅ Keep existing co-occurrence heatmap (optimized 38.6x speedup)
   - ✅ Keep network analysis
   - Enhance: Add temporal co-occurrence (if timestamp available)
   - Enhance: Add demographic co-occurrence patterns

4. **No Forced Coverage:**
   - Never force every response to receive a code
   - Explicitly track "Uncoded" as valid state (already implemented ✅)
   - Export uncoded responses for review (already implemented ✅)
   - Document why responses remained uncoded

5. **Integration Enhancement:**
   ```python
   # In MLOpenCoder._assign_codes()
   for response in responses:
       assigned_codes = []
       for code, confidence in code_probabilities:
           if confidence >= min_confidence:
               assigned_codes.append({
                   'code_id': code,
                   'confidence': confidence,
                   'reason': self._explain_assignment(response, code)  # NEW
               })

       if not assigned_codes:
           ambiguity = self.assess_ambiguity(code_probabilities)
           assigned_codes.append({
               'code_id': 'UNCODED',
               'reason': ambiguity['reason'],  # NEW: Why uncoded?
               'review_priority': ambiguity['human_review_priority']
           })
   ```

**Delegation:** Assign to **Multi-Label Logic Specialist Agent**

---

### 2.6 Human-in-the-Loop Review Mechanisms

**Current State:**
- ✅ Low-confidence response flagging (threshold=0.5)
- ✅ Uncoded response export
- ✅ Ambiguous response identification (3+ codes)
- ✅ QA report with quality metrics
- ✅ Representative quote display

**Integration Point:** New module `src/human_review_workflow.py`

**Enhancement Strategy:**

1. **Create Structured Review Interface:**
   ```python
   class HumanReviewWorkflow:
       def generate_review_queue(self, results: OpenCodingResults) -> pd.DataFrame:
           """Prioritize responses for human review"""
           return pd.DataFrame({
               'response_id': ...,
               'response_text': ...,
               'ml_assigned_codes': ...,
               'confidence_scores': ...,
               'review_reason': ...,        # Why flagged for review?
               'priority': ...,             # 1 (high) to 5 (low)
               'suggested_action': ...,     # 'validate', 'reassign', 'create_new_code'
           })

       def accept_human_feedback(self, review_data: pd.DataFrame):
           """Incorporate human decisions back into model"""
           # Track human vs. ML disagreement
           # Update confidence thresholds adaptively
           # Identify systematic ML errors
   ```

2. **Active Learning Loop:**
   - Prioritize uncertain cases for review (not random sampling)
   - Learn from human corrections to improve future assignments
   - Track inter-coder reliability (human vs. ML)
   - Surface patterns in disagreement

3. **Review Types:**
   - **Validation Review:** Confirm ML assignments (high confidence)
   - **Correction Review:** Reassign codes (medium confidence)
   - **Discovery Review:** Create new codes for uncoded responses
   - **Merge/Split Review:** Refine code structure based on examples

4. **Audit Trail:**
   ```python
   class ReviewAudit:
       def log_decision(self, response_id, ml_code, human_code, reason, reviewer):
           """Track all human decisions for reproducibility"""
           # Export to: 'human_review_log.csv'
           # Include: timestamp, reviewer_id, change_type, rationale
   ```

5. **Integration into Streamlit App:**
   - Add new page: "Human Review Queue"
   - Display flagged responses with ML suggestions
   - Allow inline code assignment/override
   - Track review progress (X of Y reviewed)
   - Export final human-validated dataset

**Delegation:** Assign to **Human-AI Collaboration Specialist Agent**

---

### 2.7 Evaluation & Rigor Diagnostics

**Current State:**
- ✅ Silhouette score (cluster quality)
- ✅ Calinski-Harabasz score
- ✅ Davies-Bouldin score
- ✅ QA report with counts

**Integration Point:** New module `src/rigor_diagnostics.py`

**Enhancement Strategy:**

1. **Methodological Validity Checks:**
   ```python
   class RigorDiagnostics:
       def assess_validity(self, results: OpenCodingResults) -> dict:
           return {
               # Internal Consistency
               'inter_code_reliability': float,      # Cohen's kappa if human codes available
               'code_stability': float,              # Consistency across bootstrap samples
               'theme_coherence': float,             # Semantic similarity within codes

               # Coverage & Completeness
               'thematic_saturation': bool,          # Are we missing themes?
               'coverage_ratio': float,              # % of responses coded
               'code_utilization': dict,             # Which codes are underused?

               # Uncertainty Quantification
               'confidence_distribution': array,     # Histogram of confidence scores
               'ambiguity_rate': float,              # % of multi-coded responses
               'boundary_cases': int,                # Responses near decision boundaries

               # Bias Detection
               'demographic_representation': dict,   # Are all groups coded equally?
               'code_imbalance': dict,               # Are some codes overused?

               # Reproducibility
               'random_seed_stability': float,       # Does random_state matter?
               'parameter_sensitivity': dict,        # How sensitive to hyperparameters?
           }
   ```

2. **Qualitative Rigor Criteria:**
   - **Credibility:** Do representative quotes match code labels?
   - **Transferability:** Are code definitions clear enough for reuse?
   - **Dependability:** Is the coding process documented and auditable?
   - **Confirmability:** Can results be traced back to raw data?

3. **Automated Sanity Checks:**
   ```python
   def sanity_check(self, results):
       warnings = []

       # Check 1: Are code labels meaningful?
       if any(code.label.count(' ') > 5 for code in results.codes):
           warnings.append("Some code labels are too long (>5 words)")

       # Check 2: Are codes balanced?
       if max(results.frequencies) / min(results.frequencies) > 10:
           warnings.append("Code imbalance detected (10:1 ratio)")

       # Check 3: Is coverage adequate?
       if results.uncoded_count / results.total_responses > 0.2:
           warnings.append("20%+ responses uncoded - consider more codes or lower threshold")

       # Check 4: Is confidence distribution healthy?
       if np.percentile(results.confidences, 75) < 0.5:
           warnings.append("75% of assignments have <0.5 confidence - review model fit")

       return warnings
   ```

4. **Integration into QA Report:**
   - Expand existing `get_qa_report()` method
   - Add rigor diagnostics section
   - Include recommendations for improvement
   - Flag methodological concerns prominently

**Delegation:** Assign to **Evaluation & Validation Specialist Agent**

---

### 2.8 Documentation & Methods Notes

**Current State:**
- ✅ Comprehensive README (625 lines)
- ✅ 5 Word documents (methodology, specs, standards)
- ✅ Code-level docstrings
- ✅ Notebook markdown cells

**Integration Point:** Enhanced `scripts/documentation_generator.py`

**Enhancement Strategy:**

1. **Auto-Generated Methods Documentation:**
   ```python
   class MethodsDocGenerator:
       def generate_methods_section(self, results: OpenCodingResults) -> str:
           """Generate academic-style methods section"""
           return f"""
           ## Methods

           ### Data Preparation
           - Dataset: {results.total_responses} responses
           - Preprocessing: {self._describe_preprocessing()}
           - Quality filtering: {self._describe_quality_checks()}
           - Final analytic sample: {results.coded_responses} responses

           ### Coding Approach
           - Method: {results.method} ({self._cite_method()})
           - Representation: {results.representation_method}
           - Number of codes: {results.n_codes} (selected via {results.code_selection_method})
           - Confidence threshold: {results.min_confidence}
           - Multi-label support: Yes (avg {results.avg_codes_per_response} codes/response)

           ### Quality Assurance
           - Silhouette score: {results.silhouette_score:.3f}
           - Human review: {results.human_reviewed_count} responses validated
           - Inter-coder reliability: {results.inter_coder_kappa:.3f} (Cohen's κ)

           ### Limitations
           - {self._generate_limitations()}

           ### Ethical Considerations
           - {self._generate_ethical_notes()}
           """
   ```

2. **Transparent Parameter Logging:**
   - Log all hyperparameters used
   - Document any manual overrides
   - Track random seeds for reproducibility
   - Export complete configuration as YAML/JSON

3. **Assumption Documentation:**
   ```python
   def document_assumptions(self) -> List[str]:
       return [
           "Assumption 1: Responses are independent (no conversational context)",
           "Assumption 2: English language (stop word removal)",
           "Assumption 3: Bag-of-words representation (word order ignored)",
           "Assumption 4: Linear separability of themes (K-means assumption)",
           "Assumption 5: Uniform response importance (no weighting)",
           # ... etc.
       ]
   ```

4. **Limitations Section (Auto-Generated):**
   - What the system does
   - What the system cannot do
   - Where human judgment is required
   - Known biases or constraints
   - Generalizability constraints

5. **Integration:**
   - Add to existing `ExecutiveSummaryGenerator`
   - Export as standalone `METHODS.md` file
   - Include in Streamlit "About" page
   - Generate BibTeX citations for ML methods used

**Delegation:** Assign to **Documentation Specialist Agent**

---

## 3. DELEGATION & COORDINATION

### 3.1 Specialist Agent Definitions

| Agent ID | Role | Primary Responsibility | Key Constraint |
|----------|------|------------------------|----------------|
| **Agent-1** | Text Processing Specialist | Multi-granularity segmentation (sentence, paragraph) | Preserve response-level traceability |
| **Agent-2** | Content Quality Specialist | Signal vs. non-analytic detection | Flag only, never auto-exclude |
| **Agent-3** | NLP/Embedding Specialist | Add semantic representation methods (SBERT, Word2Vec) | Keep TF-IDF as default for backward compatibility |
| **Agent-4** | Theme Discovery Specialist | Emergent theme discovery, hierarchical structures | No predetermined taxonomies |
| **Agent-5** | Multi-Label Logic Specialist | Enhance ambiguity handling and co-occurrence | Preserve existing multi-label support |
| **Agent-6** | Human-AI Collaboration Specialist | Review workflows, active learning | Human judgment is final authority |
| **Agent-7** | Evaluation & Validation Specialist | Rigor diagnostics, bias detection | Quantify uncertainty, don't suppress it |
| **Agent-8** | Documentation Specialist | Auto-generate methods sections, assumption logs | Transparent about what system can/cannot do |
| **Agent-9** | UI & Interaction Validation Specialist | Front-end robustness, usability auditing (Streamlit) | Coordinate with Agent-0 for UI-backend contract validation |

### 3.2 Task Scoping Template

For each agent, provide:

```yaml
agent_id: Agent-X
task_title: "Brief descriptive title"
context: "What exists now, what needs enhancement"
objective: "Specific, measurable outcome"
constraints:
  - "Do NOT impose hard labels prematurely"
  - "Do NOT collapse uncertainty into false certainty"
  - "Surface confidence and rationale with outputs"
deliverables:
  - "Code module: src/module_name.py"
  - "Tests: tests/test_module_name.py"
  - "Documentation: Updated README section"
  - "Integration: Modified helpers/analysis.py (lines X-Y)"
success_criteria:
  - "Backward compatibility maintained (all existing tests pass)"
  - "New functionality opt-in (default behavior unchanged)"
  - "Human review mechanism included"
dependencies:
  - "Requires completion of Agent-Y task"
  - "Provides input to Agent-Z task"
```

### 3.3 Example Delegation (Agent-2: Content Quality)

```yaml
agent_id: Agent-2
task_title: "Build Non-Analytic Content Detection System"

context: |
  Currently, the system only filters by:
  - Null removal (remove_nulls=True)
  - Minimum length (min_length=5 chars)
  - Optional duplicate removal

  No detection of non-responses like "N/A", "idk", gibberish, test responses.

objective: |
  Create a transparent, auditable content quality filter that flags (not excludes)
  low-signal responses for human review. Flag reasons must be human-readable.

constraints:
  - NEVER automatically exclude responses without user approval
  - All flagging rules must be explicit and documentable
  - Provide confidence scores for each quality assessment
  - Allow manual override of any flag
  - Export flagged responses to separate CSV for review

deliverables:
  - Code module: src/content_quality.py (ContentQualityFilter class)
  - Tests: tests/test_content_quality.py (edge cases: multilingual, sarcasm)
  - Integration: Modified src/data_loader.py (add quality_assessment column)
  - Documentation: README section on "Data Quality Filtering"
  - Export: New output file "non_analytic_responses.csv"

success_criteria:
  - Detects at least 5 types of non-analytic content (null, too_short, gibberish, non_english, test_response)
  - Provides human-readable reason for each flag
  - 100% of original responses retained in output (flagged, not dropped)
  - Flags are auditable (traceable to specific rules)
  - Performance: <100ms per 1000 responses

dependencies:
  - None (can start immediately)
  - Provides input to Agent-6 (human review queue prioritization)

testing_requirements:
  - Test with multilingual responses (should flag, not error)
  - Test with sarcasm/irony (expected to miss, document limitation)
  - Test with edge case: "N/A - see above" (context-dependent, needs human review)
  - Test with profanity (should NOT auto-flag as non-analytic)
```

### 3.4 Agent-9: UI & Interaction Validation Specialist

```yaml
agent_id: Agent-9
task_title: "UI & Interaction Validation Agent (Streamlit)"
role: "Front-End Robustness & Usability Auditor"

context: |
  The Streamlit application (app.py) has 7 pages with extensive widgets:
  - Selectboxes (text column, method, stop words)
  - Sliders (n_codes 3-30, min_confidence 0.1-0.9)
  - File uploaders (CSV/Excel)
  - Buttons (Start Analysis, Reset, Export)
  - Checkboxes (preprocessing options)

  Current validation exists in helpers/analysis.py (validate_dataframe, etc.)
  but UI-specific edge cases need dedicated validation.

objective: |
  Build a UI validation agent that audits Streamlit components for:
  1. Safe default states and required selection enforcement
  2. Robust error handling with actionable user messages
  3. State stability across reruns and widget changes
  4. Clear UX with proper labels and interpretability
  5. Correct UI-backend contract (types, ranges, validation)

responsibilities:
  ui_state_interaction:
    - Inspect all Streamlit widgets (selectboxes, sliders, uploaders, buttons)
    - Verify default states are safe and don't cause errors
    - Ensure required selections are enforced before proceeding
    - Block invalid widget combinations

  error_handling:
    - Trigger edge cases: empty datasets, non-numeric columns, single-variable inputs
    - Confirm errors are caught and don't crash the app
    - Verify user sees actionable messages (not raw tracebacks)

  state_stability:
    - Verify behavior under repeated widget changes
    - Test file re-uploads and method switching (Pearson ↔ Spearman)
    - Check cached functions (st.cache_data, st.cache_resource)
    - Prevent unintended recomputation loops and duplicated outputs

  ux_interpretability:
    - Confirm labels explain statistical meaning
    - Verify visuals are clearly titled and annotated
    - Flag ambiguous controls and misleading defaults
    - Check for warnings where misuse is likely

  ui_backend_contract:
    - Ensure UI passes correct data types
    - Validate parameter ranges before backend calls
    - Ensure backend defensively validates UI inputs
    - Backend never trusts UI state blindly

constraints:
  - Agent-9 does NOT do core logic testing (delegates to Agent-0)
  - Agent-9 focuses only on Streamlit UI components
  - Must coordinate with Agent-0 for UI-backend integration
  - All recommendations must be actionable with code examples

deliverables:
  - Code module: src/ui_validation.py (UIValidationAgent class)
  - Tests: tests/test_ui_validation.py (comprehensive test suite)
  - UI audit checklist (5 categories, 25+ checks)
  - List of failure modes & fixes (8+ common scenarios)
  - Widget redesign recommendations

success_criteria:
  - Validates 5 categories: widget_state, error_handling, state_stability, ux_interpretability, ui_backend_contract
  - Generates markdown audit reports with severity levels (critical, high, medium, low, info)
  - Passing report = no critical or high severity issues
  - Performance: <100ms per widget validation
  - 90%+ test coverage for new module

dependencies:
  - Works after core logic is stable (Agent-1 through Agent-8)
  - Provides input to Agent-0 for integration fixes

agent_0_interaction:
  - Agent-0 does NOT do UI testing
  - Agent-0 delegates UI inspection to Agent-9
  - Agent-0 integrates fixes where backend changes are required
  - Agent-0 ensures UI constraints align with statistical logic
  - Agent-0 ensures UI does not expose invalid analytical paths

agent_0_responsibility_update: |
  Add to Agent-0's responsibility list:
  "Coordinate UI-level validation (e.g. Streamlit) to ensure usability,
  error handling, and safe interaction with analytical logic."
```

### 3.5 Coordination Protocols

**Before Starting Work:**
1. Agent reads full task specification
2. Agent reviews relevant existing code (specified file paths)
3. Agent asks clarifying questions (via orchestrator) if ambiguous
4. Orchestrator confirms no conflicts with other agents' work

**During Work:**
1. Agent commits work to feature branch: `agent-{ID}-{task-slug}`
2. Agent runs existing tests to ensure no regression
3. Agent documents any deviations from spec (with rationale)
4. Agent surfaces blockers immediately (don't proceed silently)

**After Work:**
1. Agent submits deliverables checklist
2. Agent provides integration instructions (which files changed, how to test)
3. Orchestrator reviews for:
   - Methodological validity
   - No premature hard labeling
   - Uncertainty surfaced, not suppressed
   - Human review included
4. Orchestrator integrates if approved, or provides feedback for revision

**Conflict Resolution:**
- If two agents modify same file section → orchestrator merges manually
- If two approaches contradict → orchestrator chooses more conservative (preserves uncertainty)
- If agent cannot complete task → escalate to orchestrator, don't force incomplete solution

---

## 4. INTEGRATION & CONFLICT RESOLUTION

### 4.1 Integration Strategy

**Principle: Incremental, Opt-In Enhancements**

1. **Backward Compatibility First**
   - All existing functionality must continue working
   - Default parameters unchanged
   - New features opt-in via configuration flags
   - Example: `use_embeddings=False` (default), set to `True` to enable

2. **Modular Integration**
   - New capabilities in separate modules (src/content_quality.py, src/rigor_diagnostics.py)
   - Existing modules minimally modified (add calls to new modules)
   - Clear separation of concerns

3. **Configuration Management**
   ```python
   # NEW: config/orchestration_config.yaml
   quality_filtering:
     enabled: false  # Opt-in
     auto_exclude: false  # NEVER true
     flag_types: [null, too_short, gibberish, non_english, test_response]

   embeddings:
     enabled: false  # Keep TF-IDF default
     method: 'sbert'
     model: 'all-MiniLM-L6-v2'

   human_review:
     enabled: true  # Always enabled
     review_queue_size: 50
     priority_threshold: 0.5

   rigor_diagnostics:
     enabled: true
     include_bias_detection: true
     bootstrap_iterations: 100
   ```

4. **Integration Testing**
   - Run full test suite after each agent integration
   - Add integration tests: `tests/test_orchestration_integration.py`
   - Test with sample datasets (remote work, cricket, fashion)
   - Verify outputs match expected format

### 4.2 Conflict Resolution Priorities

**When conflicts arise, prioritize in this order:**

1. **Methodological Validity** (highest priority)
   - Does it preserve qualitative rigor?
   - Does it surface uncertainty?
   - Does it allow human judgment?

2. **Interpretability**
   - Can users understand what the system did?
   - Are decisions transparent and auditable?
   - Is rationale provided for automated choices?

3. **Reproducibility**
   - Can results be replicated with same inputs?
   - Are random seeds documented?
   - Are all parameters logged?

4. **Performance**
   - Is it fast enough for real-world use?
   - Does it scale to 10,000+ responses?

5. **User Experience**
   - Is it easy to use?
   - Are error messages helpful?

**Example Conflict Resolution:**

*Scenario:* Agent-3 (Embeddings) proposes making SBERT the default method because it's "more accurate."

*Conflict:* This violates backward compatibility and interpretability (TF-IDF is more transparent).

*Resolution:*
- **Decision:** Keep TF-IDF as default
- **Rationale:** Methodological validity requires transparency > statistical performance
- **Compromise:** Add SBERT as opt-in, document trade-offs clearly
- **Documentation:** "TF-IDF prioritizes interpretability (which words matter?). SBERT prioritizes semantic similarity (meaning over keywords). Choose based on your research goals."

### 4.3 Preventing Silent Issues

**Guardrails Against Common Pitfalls:**

1. **No Silent Drops**
   ```python
   # BAD (silent exclusion)
   df = df[df['quality'] == 'high']

   # GOOD (explicit tracking)
   df['excluded'] = df['quality'] != 'high'
   df['exclusion_reason'] = df.apply(lambda x: x['quality_flags'] if x['excluded'] else None)
   excluded_df = df[df['excluded']]
   excluded_df.to_csv('excluded_responses_for_review.csv')
   ```

2. **No Forced Coverage**
   ```python
   # BAD (forcing code assignment)
   if len(assigned_codes) == 0:
       assigned_codes = [get_closest_code()]  # Force assignment

   # GOOD (explicit uncoded state)
   if len(assigned_codes) == 0:
       assigned_codes = [{
           'code_id': 'UNCODED',
           'reason': 'No codes met confidence threshold',
           'recommendation': 'Human review or lower threshold'
       }]
   ```

3. **No Confidence Suppression**
   ```python
   # BAD (only show final label)
   return code_label

   # GOOD (show label + confidence + rationale)
   return {
       'code_label': code_label,
       'confidence': confidence_score,
       'rationale': top_keywords,
       'alternative_codes': other_possible_codes  # Show runner-ups
   }
   ```

4. **No Misleading Objectivity Claims**
   ```markdown
   <!-- BAD -->
   "This system objectively identifies themes in your data."

   <!-- GOOD -->
   "This system assists theme identification by clustering responses based on word usage patterns. Results should be validated through human review. The system surfaces patterns but cannot replace qualitative judgment."
   ```

---

## 5. QUALITY ASSURANCE & SANITY CHECKS

### 5.1 Pre-Integration Checklist

Before integrating any agent's work, verify:

- [ ] **Supports Emergent Themes**
  - No hardcoded taxonomies
  - Themes can be discovered from data, not imposed
  - System allows for "Other" or "Uncategorized"

- [ ] **Allows Multi-Label Coding**
  - Responses can receive 0, 1, or many codes
  - No forced single-label classification
  - Co-occurrence patterns tracked

- [ ] **Explicitly Handles "Unclear / Needs Review"**
  - `UNCODED` is a valid state (not error)
  - Low-confidence responses flagged for review
  - Ambiguous responses (multiple high-confidence codes) surfaced
  - Reason provided for each edge case

- [ ] **Exposes Confidence Scores**
  - Every code assignment includes confidence (0.0-1.0)
  - Confidence calculation is transparent (not black box)
  - Confidence distribution visualized for users

- [ ] **Provides Rationale or Diagnostics**
  - Why was this code assigned? (top keywords, nearest neighbors)
  - Why was this response uncoded? (no keywords match, low signal)
  - Why is this response ambiguous? (multiple themes present)

- [ ] **Human Review Remains Central**
  - System never claims to replace human judgment
  - Review mechanisms are prominent, not hidden
  - Review queue is prioritized (most uncertain first)

- [ ] **Human Review is Traceable**
  - All human decisions logged (who, when, what, why)
  - Changes from ML assignments tracked
  - Audit trail exportable for reproducibility

- [ ] **Human Review is Auditable**
  - Can reproduce final codes from raw data + audit log
  - Inter-coder reliability metrics available
  - Disagreements between human and ML documented

### 5.2 Automated Sanity Checks

Create test suite: `tests/test_orchestration_sanity.py`

```python
def test_no_silent_exclusions():
    """Verify no responses are dropped without documentation"""
    results = run_full_pipeline(sample_data)

    original_count = len(sample_data)
    coded_count = len(results.coded_responses)
    uncoded_count = len(results.uncoded_responses)
    excluded_count = len(results.excluded_responses)

    # All responses accounted for
    assert coded_count + uncoded_count + excluded_count == original_count

    # Exclusions documented
    for response in results.excluded_responses:
        assert response['exclusion_reason'] is not None
        assert response['exclusion_reason'] != ''

def test_uncertainty_preserved():
    """Verify uncertainty is surfaced, not suppressed"""
    results = run_full_pipeline(sample_data)

    # Low confidence responses flagged
    low_conf = [r for r in results.coded_responses if r['confidence'] < 0.5]
    assert len(low_conf) > 0  # We expect some low confidence
    assert all(r['flagged_for_review'] for r in low_conf)

    # Ambiguous responses identified
    ambiguous = [r for r in results.coded_responses if len(r['codes']) >= 3]
    assert all(r['ambiguity_noted'] for r in ambiguous)

def test_no_hardcoded_themes():
    """Verify code labels are data-driven, not predetermined"""
    results = run_full_pipeline(sample_data)

    # Code labels should vary with data
    results_1 = run_full_pipeline(dataset_1)
    results_2 = run_full_pipeline(dataset_2)  # Different topic

    labels_1 = {code['label'] for code in results_1.codebook}
    labels_2 = {code['label'] for code in results_2.codebook}

    # At least 50% different (not same taxonomy)
    overlap = len(labels_1 & labels_2) / max(len(labels_1), len(labels_2))
    assert overlap < 0.5, "Code labels appear hardcoded (too similar across datasets)"

def test_human_review_prominence():
    """Verify human review mechanisms are accessible"""
    results = run_full_pipeline(sample_data)

    # Review outputs generated
    assert 'review_queue.csv' in results.output_files
    assert 'low_confidence_responses.csv' in results.output_files
    assert 'uncoded_responses.csv' in results.output_files

    # Review queue prioritized
    review_queue = results.review_queue
    confidences = [r['confidence'] for r in review_queue]

    # Lower confidence should appear first (prioritized)
    assert confidences == sorted(confidences)

def test_transparency_requirements():
    """Verify all automated decisions are explainable"""
    results = run_full_pipeline(sample_data)

    for response in results.coded_responses:
        for code_assignment in response['codes']:
            # Must include rationale
            assert 'rationale' in code_assignment
            assert code_assignment['rationale'] is not None

            # Rationale must be human-readable
            assert len(code_assignment['rationale']) > 10  # Not just "keyword"
            assert not code_assignment['rationale'].startswith('ERROR')

def test_backward_compatibility():
    """Verify existing functionality unchanged with default settings"""
    # Run with original parameters
    original_results = run_legacy_pipeline(sample_data)

    # Run with orchestration enhancements (default settings)
    enhanced_results = run_full_pipeline(sample_data, use_defaults=True)

    # Core outputs should match
    assert len(original_results.coded_responses) == len(enhanced_results.coded_responses)
    assert original_results.n_codes == enhanced_results.n_codes

    # Enhanced version has additional outputs (not fewer)
    assert len(enhanced_results.output_files) >= len(original_results.output_files)
```

### 5.3 Manual Review Checklist (Orchestrator Responsibility)

After integration, manually verify:

1. **Documentation Review**
   - [ ] README updated with new features
   - [ ] Limitations clearly stated
   - [ ] Examples show both success and edge cases
   - [ ] No claims of "objectivity" or "ground truth"

2. **UI/UX Review (Streamlit App)**
   - [ ] Human review page is prominent (not buried)
   - [ ] Confidence scores visible on results page
   - [ ] "Uncoded" responses shown, not hidden
   - [ ] Export includes audit trail

3. **Output Review**
   - [ ] Run on all 3 sample datasets (remote work, cricket, fashion)
   - [ ] Verify code labels make sense (not gibberish)
   - [ ] Check representative quotes match code labels
   - [ ] Confirm co-occurrence patterns are interpretable

4. **Edge Case Testing**
   - [ ] Test with 100% non-analytic responses (should flag all)
   - [ ] Test with 2 responses (should warn "too small")
   - [ ] Test with highly ambiguous data (should surface uncertainty)
   - [ ] Test with perfectly separable data (should show high confidence)

---

## 6. FINAL VERIFICATION

### 6.1 Regression Testing

**Objective:** Ensure no loss of existing analytical functionality

**Test Suite:** `tests/test_regression.py`

```python
def test_original_15_outputs_intact():
    """Verify all 15 essential outputs still generated"""
    results = run_full_pipeline(sample_data)

    required_outputs = [
        'code_assignments',
        'codebook',
        'frequency_table',
        'quality_metrics',
        'binary_matrix',
        'representative_quotes',
        'cooccurrence_matrix',
        'descriptive_stats',
        'segmentation_analysis',
        'qa_report',
        'visualizations',
        'export_formats',
        'method_documentation',
        'uncoded_responses',
        'executive_summary'
    ]

    for output in required_outputs:
        assert output in results.outputs, f"Missing original output: {output}"

def test_output_format_unchanged():
    """Verify output formats match original specification"""
    results = run_full_pipeline(sample_data)

    # Code assignments should have same columns
    code_assignments = results.outputs['code_assignments']
    required_columns = ['response_id', 'response_text', 'code_id', 'confidence']
    assert all(col in code_assignments.columns for col in required_columns)

    # Codebook should have same structure
    codebook = results.outputs['codebook']
    required_fields = ['code_id', 'label', 'keywords', 'frequency', 'examples']
    assert all(field in codebook.columns for field in required_fields)

def test_performance_no_regression():
    """Verify processing speed not significantly degraded"""
    import time

    # Baseline (original system)
    start = time.time()
    _ = run_legacy_pipeline(large_dataset)  # 10,000 responses
    baseline_time = time.time() - start

    # Enhanced system (with default settings)
    start = time.time()
    _ = run_full_pipeline(large_dataset, use_defaults=True)
    enhanced_time = time.time() - start

    # Should not be >2x slower
    assert enhanced_time < baseline_time * 2, f"Performance regression: {enhanced_time/baseline_time:.2f}x slower"
```

### 6.2 No Hard-Coding Verification

**Objective:** Confirm no predetermined themes or false certainty

**Manual Audit:**

1. **Code Review: Search for Hardcoded Labels**
   ```bash
   # Search for suspicious patterns
   grep -r "THEME_" src/ helpers/  # Should find NONE
   grep -r "hardcoded_codes" src/ helpers/  # Should find NONE
   grep -r "predefined_taxonomy" src/ helpers/  # Should find NONE
   ```

2. **Configuration Review**
   ```python
   # Check config files for hardcoded categories
   # BAD: preset_themes: ['positive', 'negative', 'neutral']
   # GOOD: n_codes: 10  # Number only, not labels
   ```

3. **Test with Multiple Datasets**
   - Run on Dataset A (remote work)
   - Run on Dataset B (cricket)
   - Run on Dataset C (fashion)
   - **Verify:** Code labels differ across datasets (not same taxonomy)

4. **Inspect Confidence Score Logic**
   ```python
   # Ensure confidence scores are probabilistic, not binary
   # BAD: confidence = 1.0 if match else 0.0
   # GOOD: confidence = probability_distribution[code_idx]
   ```

### 6.3 Objectivity Claims Audit

**Objective:** Ensure no misleading claims about accuracy or objectivity

**Documentation Scan:**

Search all docs for problematic language:

**Prohibited Phrases:**
- ❌ "Objectively identifies themes"
- ❌ "Accurately classifies responses"
- ❌ "Ground truth labels"
- ❌ "100% accurate"
- ❌ "Replaces human coding"
- ❌ "Eliminates bias"

**Required Disclaimers:**
- ✅ "Assists qualitative analysis, does not replace it"
- ✅ "Results should be validated through human review"
- ✅ "Confidence scores are estimates, not guarantees"
- ✅ "System surfaces patterns; interpretation requires judgment"
- ✅ "Edge cases and outliers may require manual review"
- ✅ "Use as exploratory tool alongside traditional methods"

**Verification Script:**

```python
def audit_objectivity_claims():
    """Scan all documentation for misleading language"""
    docs = [
        'README.md',
        'documentation/*.docx',
        'src/**/*.py',  # Docstrings
        'app.py',  # Streamlit text
    ]

    prohibited = [
        'objectively identifies',
        'accurately classifies',
        'ground truth',
        '100% accurate',
        'replaces human coding',
        'eliminates bias'
    ]

    for doc in docs:
        content = read_file(doc)
        for phrase in prohibited:
            if phrase.lower() in content.lower():
                raise ValueError(f"Prohibited phrase '{phrase}' found in {doc}")

    print("✅ Objectivity claims audit passed")
```

### 6.4 Final Documentation Requirements

**Objective:** Ensure transparency about capabilities and limitations

**Required Sections in README:**

1. **What This System Does**
   - Assists thematic coding by clustering responses based on word patterns
   - Generates suggested codes with confidence scores
   - Flags responses requiring human review
   - Provides diagnostic metrics for quality assessment

2. **What This System Cannot Do**
   - Cannot replace human qualitative judgment
   - Cannot understand context, irony, or cultural nuance
   - Cannot determine causal relationships
   - Cannot generalize beyond the specific dataset
   - Cannot detect sarcasm or non-literal language

3. **Where Human Judgment is Required**
   - Validating auto-generated code labels
   - Reviewing low-confidence assignments
   - Interpreting co-occurrence patterns
   - Deciding on final code structure (merge/split)
   - Assessing representativeness of quotes
   - Contextualizing findings in research framework

4. **Known Limitations**
   - English-only language processing
   - Response-level granularity only (no sentence segmentation by default)
   - Bag-of-words assumption (word order ignored)
   - Requires minimum 20-30 responses for meaningful patterns
   - Performance degrades with highly heterogeneous data
   - Cannot handle multimodal data (images, audio)

5. **Ethical Considerations**
   - Algorithmic bias may reflect biases in training data or method design
   - Demographic subgroups may be coded differently (monitor with rigor diagnostics)
   - Automated coding should not be used for high-stakes decisions without human validation
   - Researchers remain responsible for interpretations and conclusions

### 6.5 Final Sign-Off Criteria

The orchestration integration is complete when:

- ✅ All 9 specialist agents have delivered and integrated (including Agent-9: UI Validation)
- ✅ All existing tests pass (zero regression)
- ✅ New integration tests pass (orchestration features work)
- ✅ Manual review checklist completed (no hardcoded themes, no false certainty)
- ✅ Documentation updated (what it does, cannot do, where human judgment needed)
- ✅ Sample run on 3 datasets shows:
  - Different code labels per dataset (emergent, not hardcoded)
  - Confidence scores surfaced in outputs
  - Review queue generated and prioritized
  - Uncoded responses explicitly tracked
  - Audit trail exportable
- ✅ Stakeholder review (if applicable): Research team validates that system supports, not replaces, qualitative analysis
- ✅ Ethics review (if applicable): No claims of objectivity, bias monitoring in place

**Final Output:** Orchestrated ML Open-Ended Coding System v2.0

---

## 7. APPENDIX

### 7.1 Orchestration Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ ORCHESTRATOR (Agent-0)                                      │
│ - Inspects existing system                                  │
│ - Identifies integration points                             │
│ - Delegates tasks to specialists                            │
│ - Integrates outputs                                        │
│ - Validates quality                                         │
│ - Coordinates UI-level validation (via Agent-9)             │
└─────────────┬───────────────────────────────────────────────┘
              │
              ├─> [Agent-1: Text Processing] ──────────┐
              │    - Multi-granularity segmentation    │
              │                                         │
              ├─> [Agent-2: Content Quality] ──────────┤
              │    - Non-analytic detection            │
              │                                         │
              ├─> [Agent-3: Embeddings] ───────────────┤
              │    - Semantic representations          │
              │                                         │
              ├─> [Agent-4: Theme Discovery] ──────────┤
              │    - Emergent themes, hierarchies      │
              │                                         ├──> INTEGRATION
              ├─> [Agent-5: Multi-Label Logic] ────────┤    (Orchestrator)
              │    - Ambiguity handling                │         │
              │                                         │         ▼
              ├─> [Agent-6: Human-AI Collaboration] ───┤    ┌─────────┐
              │    - Review workflows                  │    │ Testing │
              │                                         │    │ QA      │
              ├─> [Agent-7: Validation] ───────────────┤    │ Sanity  │
              │    - Rigor diagnostics                 │    └────┬────┘
              │                                         │         │
              ├─> [Agent-8: Documentation] ────────────┤         ▼
              │    - Methods, assumptions, limits      │    ┌──────────────┐
              │                                         │    │ Final System │
              └─> [Agent-9: UI Validation] ────────────┘    │ v2.0         │
                   - Streamlit robustness & UX             └──────────────┘
                   - Error handling verification
                   - State stability checks
```

### 7.2 Risk Mitigation Strategies

| Risk | Mitigation Strategy |
|------|---------------------|
| **Agent introduces hardcoded themes** | Pre-integration review scans for hardcoded labels; automated tests verify themes vary across datasets |
| **Agent suppresses uncertainty** | Mandatory confidence score exposure; sanity tests verify low-confidence responses flagged |
| **Agent silently drops responses** | Automated test verifies all responses accounted for (coded + uncoded + excluded); exclusions documented |
| **Integration breaks existing functionality** | Regression test suite runs after each integration; backward compatibility enforced |
| **Performance degradation** | Performance benchmarks run on 10k response dataset; must be <2x slower than baseline |
| **Misleading documentation** | Objectivity claims audit scans all docs; prohibited phrases trigger errors |
| **Loss of human review centrality** | Manual UI review verifies review page prominent; audit trail exportable |
| **Conflicting agent implementations** | Orchestrator resolves conflicts using priority framework (validity > interpretability > reproducibility) |

### 7.3 Success Metrics

**System-Level:**
- ✅ 100% of original 15 outputs still generated
- ✅ 0 regressions in existing test suite
- ✅ <2x performance degradation on large datasets
- ✅ >90% code coverage for new modules

**Methodological:**
- ✅ Code labels vary across datasets (not hardcoded taxonomy)
- ✅ Confidence scores provided for 100% of assignments
- ✅ Uncoded responses explicitly tracked (not hidden)
- ✅ Human review queue generated for every run

**Documentation:**
- ✅ "What it cannot do" section present in README
- ✅ Zero occurrences of prohibited objectivity claims
- ✅ Methods section auto-generated with parameters logged
- ✅ Ethical considerations documented

**User Experience:**
- ✅ Human review page accessible within 2 clicks
- ✅ Audit trail exportable as CSV
- ✅ Error messages include next steps (not just "failed")
- ✅ Configuration options documented with examples

### 7.4 Glossary

- **Emergent Themes:** Themes discovered from the data, not predetermined
- **Multi-Label Coding:** Assigning multiple codes to a single response
- **Confidence Score:** Probabilistic estimate (0.0-1.0) of code assignment accuracy
- **Ambiguous Response:** Response with multiple high-confidence codes (>0.5)
- **Uncoded Response:** Response with zero codes meeting confidence threshold
- **Signal vs. Non-Analytic:** Distinction between substantive content and non-responses ("N/A", "idk")
- **Audit Trail:** Log of all automated and human coding decisions for reproducibility
- **Rigor Diagnostics:** Quantitative metrics assessing methodological validity
- **Active Learning:** Prioritizing uncertain cases for human review to improve model

### 7.5 References & Citations

**ML Methods:**
- K-Means Clustering: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.
- Latent Dirichlet Allocation: Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation. JMLR.
- TF-IDF: Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. IPM.

**Qualitative Methods:**
- Open Coding: Strauss, A., & Corbin, J. (1998). Basics of qualitative research. Sage.
- Thematic Analysis: Braun, V., & Clarke, V. (2006). Using thematic analysis in psychology. QRP.

**Mixed Methods:**
- Computer-Assisted Qualitative Analysis: Friese, S. (2019). Qualitative data analysis with ATLAS.ti. Sage.

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-25 | Orchestrator Agent-0 | Initial orchestration plan created |

**Next Review:** After Agent-1 through Agent-8 complete initial deliverables

**Approval Required From:**
- [ ] Research Lead (methodological validity)
- [ ] Technical Lead (integration feasibility)
- [ ] Ethics Committee (if applicable)

---

**END OF ORCHESTRATION PLAN**
