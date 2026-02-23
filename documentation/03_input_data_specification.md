# Input Data Specification

**Version:** 1.0
**Date:** 2026-02-23
**Module:** `src/data_loader.py`
**Framework:** Open-Ended Coding Analysis Framework

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Requirements](#dataset-requirements)
3. [Required Columns](#required-columns)
4. [Optional Columns](#optional-columns)
5. [Supported File Formats](#supported-file-formats)
6. [Data Type Specifications](#data-type-specifications)
7. [Constraints and Assumptions](#constraints-and-assumptions)
8. [ID, Weight, and Time Field Requirements](#id-weight-and-time-field-requirements)
9. [Sample Data Schema](#sample-data-schema)
10. [Cross-References](#cross-references)

---

## Overview

The Open-Ended Coding Analysis Framework accepts data from multiple sources for qualitative analysis of text responses. This document specifies the input data requirements, supported formats, and structural expectations.

**Key Design Principles:**
- **Flexible column naming** - Framework adapts to common variations
- **Multiple data sources** - Support for CSV, Excel, and JSON
- **Minimal requirements** - Only `response` column is required
- **Quality validation** - Automatic content assessment with transparent flagging
- **UTF-8 encoding** - Universal text support

**Target Use Cases:**
- Survey open-ended responses
- Customer feedback analysis
- Interview transcripts
- Focus group data
- Social media comments
- Support ticket analysis

---

## Dataset Requirements

### Minimum Dataset Size

| Metric | Minimum | Recommended | Notes |
|--------|---------|-------------|-------|
| **Response Count** | 20 | 300-1,000 | Statistical validity requires 20-30 minimum; 300-1,000 responses optimal for all 6 ML methods (current demo datasets are 1,000 rows each) (see `OPTIMAL_DATASET_SIZE.md`) |
| **Response Length** | 5 characters | 10+ characters | Shorter responses flagged for quality review |
| **Word Count per Response** | 1 word | 3+ words | Single-word responses may lack analytic value |
| **Analytic Response Ratio** | 50% | 80%+ | Non-analytic responses flagged but retained |

### Language Requirements

- **Primary Language:** English only (current version)
- **Character Encoding:** UTF-8 required
- **Special Characters:** Supported (emojis, accented characters, etc.)
- **Mixed Languages:** Flagged for review but not excluded

---

## Required Columns

The framework requires exactly **one column** containing text responses for analysis.

| Column Name | Type | Required | Description | Validation Rules |
|-------------|------|----------|-------------|------------------|
| `response` | Text/String | **YES** | Open-ended text response to analyze | • Non-null<br>• Minimum 5 characters recommended<br>• UTF-8 encoded |

**Alternative Column Names (automatically detected):**
- `response`
- `text`
- `comment`
- `answer`
- `feedback`
- `open_ended`

**Example:**
```csv
response
"I love the flexibility of remote work"
"Better work-life balance is crucial"
"Communication challenges with team members"
```

---

## Optional Columns

These columns enhance analysis but are not required. The framework automatically detects and uses them when present.

| Column Name | Type | Required | Description | Example Values |
|-------------|------|----------|-------------|----------------|
| `id` | Integer or String | No | Unique identifier for each response | `1`, `2`, `3` or `"RESP001"` |
| `respondent_id` | String | No | Identifier for individual respondent (for multi-response datasets) | `"R001"`, `"R002"` |
| `timestamp` | DateTime or String | No | When response was submitted | `"2024-01-01"`, `"2024-01-01 14:30:00"` |
| `topic` | String/Categorical | No | Topic or category of the response | `"fashion"`, `"technology"` |
| `demographic_*` | Various | No | Demographic variables for stratification | `age`, `gender`, `location`, `department` |
| `weight` | Float | No | Statistical weight for sampling adjustments | `1.0`, `1.5`, `0.8` |
| `source` | String | No | Source of the response | `"survey"`, `"interview"`, `"social_media"` |

### Demographic Columns

Any additional columns in your dataset are preserved and can be used for:
- **Stratified analysis** - Compare themes across demographic groups
- **Filtering** - Subset data by attributes
- **Reporting** - Include context in outputs

**Common Demographic Variables:**
```csv
id,response,age_group,gender,department,tenure_years
1,"Great work environment",25-34,Female,Engineering,3
2,"Need better communication",35-44,Male,Sales,7
```

---

## Supported File Formats

### 1. CSV (Comma-Separated Values)

**File Extensions:** `.csv`

**Specifications:**
- **Delimiter:** Comma (`,`) - standard; semicolon (`;`) supported via parameters
- **Text Qualifier:** Double quotes (`"`) for fields containing commas
- **Encoding:** UTF-8 (recommended), UTF-8-BOM, ISO-8859-1
- **Line Endings:** CRLF (`\r\n`) or LF (`\n`)
- **Header Row:** Required (first row must contain column names)

**Loading Example:**
```python
from src.data_loader import DataLoader

loader = DataLoader()
df = loader.load_csv('data/responses.csv')
```

**With Custom Delimiter:**
```python
df = loader.load_csv('data/responses.csv', sep=';', encoding='utf-8')
```

**Best Practices:**
- Use UTF-8 encoding to preserve special characters
- Enclose text fields in double quotes if they contain delimiters
- Remove or escape newlines within responses
- Use consistent date formats

---

### 2. Excel (.xlsx, .xls)

**File Extensions:** `.xlsx` (Excel 2007+), `.xls` (Excel 97-2003)

**Specifications:**
- **Sheet Name:** First sheet by default (or specify sheet name/index)
- **Header Row:** First row (must contain column names)
- **Empty Rows:** Automatically skipped
- **Formulas:** Evaluated to values on load
- **Formatting:** Stripped (only values retained)

**Loading Example:**
```python
# Load first sheet
df = loader.load_excel('data/responses.xlsx')

# Load specific sheet by name
df = loader.load_excel('data/responses.xlsx', sheet_name='Survey_Results')

# Load specific sheet by index (0-based)
df = loader.load_excel('data/responses.xlsx', sheet_name=1)
```

**Best Practices:**
- Place data in the first sheet or specify sheet name
- Use first row for column headers
- Avoid merged cells in data area
- Remove charts, pivot tables, and formatting before export

---

### 3. JSON (JavaScript Object Notation)

**File Extensions:** `.json`, `.jsonl` (JSON Lines)

**Specifications:**
- **Format:** Array of objects (records) or JSON Lines (one record per line)
- **Encoding:** UTF-8
- **Structure:** Each object represents one response record

**Standard JSON (Array of Objects):**
```json
[
  {
    "id": 1,
    "response": "I love the flexibility of remote work",
    "respondent_id": "R001",
    "timestamp": "2024-01-01"
  },
  {
    "id": 2,
    "response": "Better work-life balance is crucial",
    "respondent_id": "R002",
    "timestamp": "2024-01-02"
  }
]
```

**JSON Lines Format (one object per line):**
```json
{"id": 1, "response": "I love the flexibility of remote work", "respondent_id": "R001"}
{"id": 2, "response": "Better work-life balance is crucial", "respondent_id": "R002"}
```

**Loading Example:**
```python
# Standard JSON (array of objects)
df = loader.load_json('data/responses.json')

# JSON Lines format
df = loader.load_json('data/responses.jsonl', lines=True)
```

---

## Data Type Specifications

The framework automatically infers data types but expects the following:

| Column Type | Expected Format | Examples | Notes |
|-------------|----------------|----------|-------|
| **Text/String** | UTF-8 text | `"This is a response"` | Primary response column |
| **Integer** | Whole numbers | `1`, `42`, `-5` | IDs, counts, categorical codes |
| **Float** | Decimal numbers | `1.5`, `3.14`, `-0.5` | Weights, scores, ratios |
| **DateTime** | ISO 8601 or common formats | `"2024-01-01"`, `"2024-01-01 14:30:00"` | Timestamps, dates |
| **Boolean** | True/False or 0/1 | `True`, `False`, `0`, `1` | Flags, binary variables |
| **Categorical** | String labels | `"Male"`, `"Female"`, `"Tech"` | Groups, categories |

### Date/Time Format Specifications

**Supported Formats (auto-detected):**
- ISO 8601: `YYYY-MM-DD`, `YYYY-MM-DD HH:MM:SS`
- US Format: `MM/DD/YYYY`, `MM-DD-YYYY`
- European Format: `DD/MM/YYYY`, `DD-MM-YYYY`
- Timestamps: Unix epoch, ISO 8601 with timezone

**Recommended Format:** ISO 8601 (`YYYY-MM-DD HH:MM:SS`)

**Example:**
```csv
id,response,timestamp
1,"Great experience","2024-01-01 09:30:00"
2,"Needs improvement","2024-01-02 14:15:00"
```

For detailed formatting rules, see [Data Formatting Rules](04_data_formatting_rules.md).

---

## Constraints and Assumptions

### 1. Minimum Response Count

**Constraint:** At least **20-30 responses** required for statistical validity.

**Rationale:**
- Qualitative analysis requires sufficient data for pattern detection
- Machine learning models need minimum sample size
- Theme emergence requires adequate coverage

**Recommendation:** 300-1,000 responses optimal for robust analysis across all 6 ML methods (current demo datasets are 1,000 rows each) (see `OPTIMAL_DATASET_SIZE.md`)

---

### 2. Minimum Response Length

**Constraint:** Minimum **5 characters** per response (enforced by quality filter).

**What Happens:**
- Responses < 5 characters flagged as `too_short_chars`
- Responses < 3 words flagged as `too_short_words`
- Flagged responses retained but marked for review

**Examples:**
- ✅ `"Great!"` (6 characters, 1 word) - **Flagged but retained**
- ✅ `"I love this"` (11 characters, 3 words) - **Passes all checks**
- ⚠️ `"OK"` (2 characters) - **Flagged for review**
- ⚠️ `"N/A"` (3 characters) - **Flagged as non-response**

---

### 3. English Language Only

**Constraint:** Current version processes **English text only**.

**Detection Method:**
- Compares words against common English word dictionary
- Requires ≥30% recognized English words (configurable)
- Responses below threshold flagged as `non_english`

**What Happens to Non-English Responses:**
- Flagged but retained in dataset
- Quality assessment adds `non_english` flag
- Recommendation: `review` (manual inspection suggested)

**Future Versions:** Multilingual support planned

---

### 4. UTF-8 Encoding

**Constraint:** All text files must use **UTF-8 encoding**.

**Why UTF-8:**
- Supports international characters (é, ñ, ü, etc.)
- Compatible with emojis (😊, 👍, etc.)
- Universal standard for text data

**How to Ensure UTF-8:**
- **CSV Export:** Select "UTF-8" encoding option
- **Excel:** Save as "CSV UTF-8 (Comma delimited)"
- **Database Export:** Specify UTF-8 in export settings

**Troubleshooting:** If special characters appear garbled, re-save file as UTF-8.

---

### 5. Content Quality Assumptions

The framework assumes:
- **Authentic responses:** Not test data or placeholder text
- **Single language:** Primarily English (mixed languages flagged)
- **Meaningful content:** Not gibberish or keyboard walks
- **Unique responses:** Not excessive copy-paste duplication

**Quality Filtering:**
- See `src/content_quality.py` for detailed filtering logic
- All quality checks transparent and auditable
- No automatic exclusion - flagging only

---

## ID, Weight, and Time Field Requirements

### ID Fields

**Purpose:** Unique identification of responses and respondents

| Field | Format | Required | Description |
|-------|--------|----------|-------------|
| `id` | Integer or String | No | Unique response identifier |
| `respondent_id` | String | No | Identifier for respondent (links multiple responses from same person) |

**Example - Single Response per Respondent:**
```csv
id,response,respondent_id
1,"First response",R001
2,"Second response",R002
```

**Example - Multiple Responses per Respondent:**
```csv
id,response,respondent_id,question_number
1,"Answer to Q1",R001,1
2,"Answer to Q2",R001,2
3,"Answer to Q1",R002,1
```

---

### Weight Fields

**Purpose:** Statistical weighting for survey sampling adjustments

| Field | Format | Required | Description |
|-------|--------|----------|-------------|
| `weight` | Float | No | Statistical weight (default: 1.0 if absent) |

**When to Use Weights:**
- Survey oversampling/undersampling corrections
- Post-stratification adjustments
- Importance weighting

**Example:**
```csv
id,response,weight
1,"Response from oversampled group",0.5
2,"Response from undersampled group",2.0
3,"Response from balanced group",1.0
```

**Note:** If weight column absent, all responses weighted equally (1.0).

---

### Time/Date Fields

**Purpose:** Temporal analysis and tracking

| Field | Format | Required | Description |
|-------|--------|----------|-------------|
| `timestamp` | DateTime or String | No | When response was recorded |
| `date` | Date | No | Date only (alternative to timestamp) |

**Recommended Format:** ISO 8601 (`YYYY-MM-DD HH:MM:SS`)

**Alternative Accepted Formats:**
- `YYYY-MM-DD`
- `MM/DD/YYYY HH:MM:SS`
- `DD/MM/YYYY`

**Example:**
```csv
id,response,timestamp
1,"Morning feedback","2024-01-01 09:15:00"
2,"Afternoon feedback","2024-01-01 15:30:00"
3,"Evening feedback","2024-01-01 20:45:00"
```

**Use Cases:**
- Trend analysis over time
- Before/after comparisons
- Seasonal patterns

**Timezone Handling:** If timezone not specified, assumed to be UTC or local time (document your convention).

---

## Sample Data Schema

Based on actual sample files in the `/home/user/JC-OE-Coding/data/` directory:

### Basic Schema (Minimal Required)

**File:** `Remote_Work_Experiences_1000.csv`

```csv
id,response,respondent_id,timestamp
1,"I love the flexibility of remote work",R001,2024-01-01
2,"Better work-life balance is crucial",R002,2024-01-02
3,"Communication challenges with team members",R003,2024-01-03
```

**Columns:**
- `id` - Sequential integer identifier
- `response` - Text response (required)
- `respondent_id` - Respondent identifier
- `timestamp` - Date (YYYY-MM-DD format)

---

### Extended Schema (With Topic and Demographics)

**File:** `consumer_perspectives_responses.csv`

```csv
id,response,respondent_id,timestamp,topic
1,"I finally switched to sustainable fashion brands and my wallet is crying",R001,2024-05-01,fashion
2,"These new AI assistants are genuinely creeping me out with how much they know",R002,2024-05-01,technology
3,"Vinyl records sound so much warmer than streaming services ever could",R003,2024-05-01,music
```

**Columns:**
- `id` - Sequential integer identifier
- `response` - Text response (required)
- `respondent_id` - Respondent identifier
- `timestamp` - Date (YYYY-MM-DD format)
- `topic` - Categorical variable for response topic/category

---

### Full Schema Example (All Optional Fields)

```csv
id,response,respondent_id,timestamp,topic,age_group,gender,location,weight,source
1,"Remote work has been life-changing",R001,2024-01-01 09:00:00,work,25-34,Female,Urban,1.0,survey
2,"I miss office collaboration",R002,2024-01-01 10:30:00,work,35-44,Male,Suburban,1.2,interview
3,"Flexible schedule is amazing",R003,2024-01-01 14:15:00,work,45-54,Non-binary,Rural,0.8,survey
```

**All Columns:**
- **Required:** `response`
- **Identifiers:** `id`, `respondent_id`
- **Temporal:** `timestamp`
- **Categorical:** `topic`, `age_group`, `gender`, `location`, `source`
- **Numerical:** `weight`

---

## Cross-References

**Related Documentation:**
- [Data Formatting Rules](04_data_formatting_rules.md) - Detailed formatting specifications
- [Input Data Template](/home/user/JC-OE-Coding/documentation/input_data_template.xlsx) - Excel template with proper structure
- `src/data_loader.py` - DataLoader implementation
- `src/content_quality.py` - Content quality filtering logic

**Framework Documentation:**
- [Rigor Diagnostics Guide](RIGOR_DIAGNOSTICS_GUIDE.md) - Quality assessment metrics
- [Embedding Methods](EMBEDDING_METHODS.md) - Text vectorization approaches

---

## Quick Start Checklist

Before loading your data, verify:

- [ ] **Required column present:** Dataset includes `response` column (or alternative name)
- [ ] **Minimum responses:** At least 20-30 responses in dataset
- [ ] **UTF-8 encoding:** File saved with UTF-8 encoding
- [ ] **Header row:** First row contains column names
- [ ] **No empty responses:** Null/empty responses handled appropriately
- [ ] **Date format:** Timestamps use consistent format (ISO 8601 recommended)
- [ ] **File format:** Using supported format (CSV, Excel, JSON)
- [ ] **Special characters:** Text properly escaped/quoted if contains commas or quotes

**Next Steps:**
1. Review [Data Formatting Rules](04_data_formatting_rules.md)
2. Download [Input Data Template](/home/user/JC-OE-Coding/documentation/input_data_template.xlsx)
3. Load your data using `DataLoader` class
4. Run content quality assessment
5. Proceed with analysis

---

**Document Revision History:**

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-02-23 | Initial release | Agent-D (Data Contract) |

---

**For Questions or Issues:**
- Review code documentation in `src/data_loader.py`
- Check sample data files in `data/` directory
- Consult [Data Formatting Rules](04_data_formatting_rules.md) for specific formatting questions
