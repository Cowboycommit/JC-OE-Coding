# Data Formatting Rules

**Version:** 1.0
**Date:** 2025-12-25
**Module:** `src/data_loader.py`, `src/content_quality.py`
**Framework:** Open-Ended Coding Analysis Framework

---

## Table of Contents

1. [Overview](#overview)
2. [Naming Conventions](#naming-conventions)
3. [Character Encoding Rules](#character-encoding-rules)
4. [Date and Time Formats](#date-and-time-formats)
5. [Missing Values Policy](#missing-values-policy)
6. [Categorical Encoding](#categorical-encoding)
7. [Text Content Rules](#text-content-rules)
8. [CSV Formatting Specifications](#csv-formatting-specifications)
9. [Excel Formatting Specifications](#excel-formatting-specifications)
10. [JSON Formatting Specifications](#json-formatting-specifications)
11. [Template and Schema Reference](#template-and-schema-reference)
12. [Minimal Sample Dataset](#minimal-sample-dataset)
13. [Validation Rules](#validation-rules)
14. [Cross-References](#cross-references)

---

## Overview

This document specifies the detailed formatting rules for input data in the Open-Ended Coding Analysis Framework. Following these rules ensures successful data loading, accurate analysis, and proper quality assessment.

**Formatting Philosophy:**
- **Standardization** - Consistent formatting enables reliable processing
- **Flexibility** - Framework accommodates common variations
- **Validation** - Automatic checking with clear error messages
- **Transparency** - All formatting rules explicit and documentable

**Who Should Use This Document:**
- Data preparers creating input files
- Survey administrators exporting data
- Developers integrating with the framework
- Analysts troubleshooting loading issues

---

## Naming Conventions

### Column Names

**General Rules:**
- Use lowercase with underscores: `respondent_id`, `age_group`
- Avoid spaces (use underscores instead): `created_date` not `Created Date`
- Keep names concise but descriptive: `timestamp` not `t` or `response_submission_timestamp`
- Use alphanumeric characters only (no special symbols except underscore)

**Case Sensitivity:**
- Column names are **case-insensitive** during loading
- Framework automatically normalizes: `Response` = `response` = `RESPONSE`
- Recommended: Use lowercase for consistency

**Reserved Column Names (automatically detected):**

| Purpose | Primary Name | Alternative Names |
|---------|--------------|-------------------|
| Response text | `response` | `text`, `comment`, `answer`, `feedback`, `open_ended` |
| Response ID | `id` | `response_id`, `record_id` |
| Respondent ID | `respondent_id` | `user_id`, `participant_id`, `subject_id` |
| Timestamp | `timestamp` | `date`, `datetime`, `created_at`, `submitted_at` |
| Topic/Category | `topic` | `category`, `theme`, `question`, `prompt` |
| Weight | `weight` | `sample_weight`, `wt` |

**Examples - Good Column Names:**
```csv
✅ id,response,respondent_id,timestamp,age_group,gender
✅ response_id,text,user_id,date,topic,location
✅ record_id,feedback,participant_id,created_at,department
```

**Examples - Problematic Column Names:**
```csv
❌ ID#,Response Text,User-ID,Date/Time,Age Group,Gender (M/F)
   (Contains special characters, spaces)

✅ id,response,user_id,datetime,age_group,gender
   (Corrected version)
```

---

### File Names

**Recommended Pattern:**
```
{dataset_name}_{optional_descriptor}.{extension}
```

**Examples:**
- `survey_responses.csv`
- `customer_feedback_2024.xlsx`
- `employee_comments_q1.json`
- `interview_transcripts_tech.csv`

**Best Practices:**
- Use lowercase
- Separate words with underscores
- Include descriptive information (date, source, topic)
- Avoid spaces and special characters
- Use standard extensions (`.csv`, `.xlsx`, `.json`, `.db`)

---

## Character Encoding Rules

### Required Encoding: UTF-8

**All text files must use UTF-8 encoding** to ensure proper handling of:
- International characters (é, ñ, ü, ç, etc.)
- Emoji characters (😊, 👍, 🎉, etc.)
- Special symbols (©, ®, €, £, etc.)
- Smart quotes (" " ' ')
- Em dashes (—) and en dashes (–)

---

### How to Ensure UTF-8 Encoding

**CSV Files:**

**Microsoft Excel (Windows):**
1. Save As → More Options
2. Tools → Web Options → Encoding
3. Select "Unicode (UTF-8)"
4. Save

**Excel (Mac):**
1. File → Save As
2. File Format: "CSV UTF-8 (Comma delimited) (.csv)"

**Google Sheets:**
1. File → Download → Comma-separated values (.csv)
2. Automatically saved as UTF-8

**Text Editors:**
- **VS Code:** Bottom right corner shows encoding; select UTF-8
- **Notepad++:** Encoding → Convert to UTF-8
- **Sublime Text:** File → Save with Encoding → UTF-8

---

### Verifying Encoding

**Command Line (Linux/Mac):**
```bash
file -I your_file.csv
# Should show: text/plain; charset=utf-8
```

**Python:**
```python
import chardet

with open('your_file.csv', 'rb') as f:
    result = chardet.detect(f.read())
    print(result['encoding'])  # Should show: 'utf-8'
```

---

### Common Encoding Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Wrong encoding** | Characters like `Ã©` instead of `é` | Re-save as UTF-8 |
| **Excel corruption** | Emoji replaced with `?` | Use "CSV UTF-8" format, not "CSV" |
| **Smart quotes broken** | `â€œ` instead of `"` | Ensure UTF-8, not Windows-1252 |
| **Accents garbled** | `Ã±` instead of `ñ` | Convert file to UTF-8 |

---

## Date and Time Formats

### Recommended Format: ISO 8601

**Standard:** `YYYY-MM-DD HH:MM:SS`

**Examples:**
- `2024-01-01` (date only)
- `2024-01-01 09:30:00` (date and time)
- `2024-01-01T09:30:00` (ISO 8601 with T separator)
- `2024-01-01 09:30:00-05:00` (with timezone offset)

**Why ISO 8601:**
- Unambiguous (no month/day confusion)
- Sorts correctly as text
- International standard
- Machine-readable

---

### Supported Formats (Auto-Detected)

| Format | Example | Notes |
|--------|---------|-------|
| **ISO 8601** | `2024-01-01`, `2024-01-01 14:30:00` | ✅ Recommended |
| **US Format** | `01/01/2024`, `01-01-2024` | Month first (MM/DD/YYYY) |
| **European Format** | `01/01/2024`, `01.01.2024` | Day first (DD/MM/YYYY) |
| **Timestamps** | `1704110400` | Unix epoch (seconds since 1970-01-01) |
| **ISO with TZ** | `2024-01-01T14:30:00Z` | With timezone (Z = UTC) |

**Warning:** Ambiguous dates like `01/02/2024` may be interpreted differently.
→ **Use ISO 8601 to avoid ambiguity**

---

### Timezone Handling

**Default Assumption:** UTC or local time (document your convention)

**Best Practices:**
1. **Include timezone offset** if data spans multiple zones:
   ```
   2024-01-01 09:00:00-05:00  (US Eastern)
   2024-01-01 14:00:00+00:00  (UTC)
   ```

2. **Convert to UTC** before export if possible

3. **Document timezone** in metadata or filename:
   ```
   survey_responses_utc.csv
   ```

---

### Date-Only vs. DateTime

**Date Only (YYYY-MM-DD):**
```csv
id,response,date
1,"Response text","2024-01-01"
```
- Use when time of day not relevant
- Simpler format
- Easier to read

**DateTime (YYYY-MM-DD HH:MM:SS):**
```csv
id,response,timestamp
1,"Response text","2024-01-01 14:30:00"
```
- Use when time of day matters
- Enables temporal analysis
- Supports high-resolution time series

---

### Invalid Date Examples

| Invalid | Reason | Valid Alternative |
|---------|--------|-------------------|
| `1/1/24` | Ambiguous, 2-digit year | `2024-01-01` |
| `Jan 1, 2024` | Text month name | `2024-01-01` |
| `01-01-24` | 2-digit year ambiguous | `2024-01-01` |
| `2024/1/1` | Inconsistent zero-padding | `2024-01-01` |
| `1st January 2024` | Ordinal, text month | `2024-01-01` |

---

## Missing Values Policy

### How Missing Values Are Handled

The framework distinguishes between different types of missing values and handles them accordingly.

| Representation | Type | Handling | Recommendation |
|----------------|------|----------|----------------|
| **Empty cell** | Truly missing | Treated as `NaN` (pandas null) | ✅ Preferred for missing data |
| **`NULL`** | Database null | Treated as `NaN` | ✅ Acceptable |
| **Empty string `""`** | Zero-length text | Flagged by quality filter as "empty" | ⚠️ Use empty cell instead |
| **`"N/A"`** | Explicit non-response | Flagged as "non_response" | ⚠️ Use empty cell or actual response |
| **`"NA"`** | Explicit missing | Flagged as "non_response" | ⚠️ Use empty cell |
| **`"-"`** | Dash placeholder | Flagged as "non_response" | ⚠️ Use empty cell |
| **`"None"`** | Text "None" | Flagged as "non_response" | ⚠️ Use empty cell |
| **Whitespace `"   "`** | Spaces only | Flagged as "empty" | ❌ Invalid; use empty cell |

---

### Missing Value Examples

**❌ Problematic - Using Text Placeholders:**
```csv
id,response,age_group,gender
1,"Great product",25-34,Female
2,"N/A",35-44,N/A
3,"-",N/A,Male
```
→ `"N/A"` and `"-"` flagged as non-responses by quality filter

**✅ Correct - Using Empty Cells:**
```csv
id,response,age_group,gender
1,"Great product",25-34,Female
2,,35-44,
3,,",Male
```
→ Empty cells properly represent missing data

---

### Response Column Missing Values

**Important:** Missing values in the `response` column are handled specially:

1. **Empty/Null Responses:**
   - Flagged by quality filter
   - Assigned quality flags: `null` or `empty`
   - Recommendation: `exclude`
   - **Retained in dataset** (not automatically removed)

2. **Non-Response Patterns:**
   - Text like `"N/A"`, `"no comment"`, `"idk"`, `"skip"`
   - Flagged as `non_response`
   - Recommendation: `exclude`
   - **Retained but marked for review**

**Example Quality Assessment:**
```python
# Response: ""
{
    'is_analytic': False,
    'confidence': 1.0,
    'reason': 'Response is empty (whitespace only)',
    'recommendation': 'exclude',
    'flags': ['empty']
}

# Response: "N/A"
{
    'is_analytic': False,
    'confidence': 0.95,
    'reason': 'Matches non-response pattern: "N/A"',
    'recommendation': 'exclude',
    'flags': ['non_response']
}
```

---

### Optional Column Missing Values

**Missing values in optional columns are acceptable:**
- Demographic variables can be partially complete
- Weights default to 1.0 if missing
- Timestamps can be absent (no temporal analysis)

**Example - Partial Demographics:**
```csv
id,response,age_group,gender,department
1,"Great service",25-34,Female,Sales
2,"Needs improvement",,Male,
3,"Very satisfied",45-54,,Engineering
```
→ Missing demographics do not prevent analysis

---

## Categorical Encoding

### Text Labels (Recommended)

**Use descriptive text labels** for categorical variables instead of numeric codes.

**✅ Good - Text Labels:**
```csv
id,response,gender,satisfaction,department
1,"Great experience",Female,Very Satisfied,Engineering
2,"Could be better",Male,Neutral,Sales
3,"Excellent service",Non-binary,Satisfied,Marketing
```

**❌ Avoid - Numeric Codes:**
```csv
id,response,gender,satisfaction,department
1,"Great experience",1,5,3
2,"Could be better",2,3,1
3,"Excellent service",3,4,2
```
→ Requires separate codebook; unclear without documentation

---

### Standard Categorical Formats

| Variable Type | Recommended Values | Notes |
|---------------|-------------------|-------|
| **Gender** | `Male`, `Female`, `Non-binary`, `Prefer not to say` | Inclusive options |
| **Yes/No** | `Yes`, `No` | Avoid `Y`/`N`, `1`/`0` for clarity |
| **Agreement Scale** | `Strongly Disagree`, `Disagree`, `Neutral`, `Agree`, `Strongly Agree` | Full text preferred |
| **Satisfaction** | `Very Dissatisfied`, `Dissatisfied`, `Neutral`, `Satisfied`, `Very Satisfied` | 5-point standard |
| **Frequency** | `Never`, `Rarely`, `Sometimes`, `Often`, `Always` | Clear progression |
| **Age Groups** | `18-24`, `25-34`, `35-44`, `45-54`, `55-64`, `65+` | Non-overlapping ranges |

---

### Case Sensitivity

**Categorical values are case-sensitive** in pandas:
- `"Male"` ≠ `"male"` ≠ `"MALE"`

**Best Practice:** Use consistent capitalization throughout dataset

**✅ Consistent Capitalization:**
```csv
gender
Male
Female
Male
Non-binary
Female
```

**❌ Inconsistent Capitalization:**
```csv
gender
Male
female
MALE
Non-Binary
Female
```

**Solution:** Standardize case before export or use post-processing normalization.

---

### Boolean Values

**Accepted Formats:**

| Format | True Values | False Values |
|--------|-------------|--------------|
| **Boolean** | `True`, `true` | `False`, `false` |
| **Integer** | `1` | `0` |
| **String** | `"Yes"`, `"yes"`, `"Y"` | `"No"`, `"no"`, `"N"` |

**Recommended:** Use `True`/`False` or `1`/`0` for clarity

**Example:**
```csv
id,response,opted_in,verified
1,"Feedback text",True,1
2,"More feedback",False,0
```

---

## Text Content Rules

### Response Text Requirements

**Required Characteristics:**
- Non-empty (minimum 1 character)
- Readable text (not gibberish)
- Primarily English (current version)
- Authentic content (not test data)

**Quality Filtering Criteria (from `content_quality.py`):**

| Criterion | Threshold | Flag if Violated |
|-----------|-----------|------------------|
| **Minimum characters** | 10 characters | `too_short_chars` |
| **Minimum words** | 3 words | `too_short_words` |
| **English word ratio** | ≥30% recognized English words | `non_english` |
| **Repetition ratio** | ≤70% repeated words | `excessive_repetition` |
| **Vowel ratio** | ≥15% vowels (for gibberish detection) | `gibberish` |

---

### Accepted Text Patterns

**✅ Valid Responses:**
```
"I love the flexibility of remote work"
"Better work-life balance is crucial"
"Communication challenges with team members"
"Increased productivity at home"
"Great! 👍"  (includes emoji)
"C'est très bien"  (non-English, flagged but retained)
```

**⚠️ Flagged Responses (retained but marked):**
```
"OK"  (too short)
"N/A"  (non-response pattern)
"test"  (test response pattern)
"asdfghjkl"  (keyboard walk / gibberish)
"good good good good good"  (excessive repetition)
```

---

### Special Characters in Text

**Supported Special Characters:**
- **Punctuation:** `. , ! ? ; : " ' - ( )`
- **Symbols:** `@ # $ % & * + = / \`
- **Emoji:** 😊 👍 🎉 ❤️ (requires UTF-8)
- **Accented characters:** é è ê ñ ü ç (requires UTF-8)
- **Smart quotes:** " " ' ' (requires UTF-8)
- **Dashes:** — (em dash), – (en dash)

**Character Escaping in CSV:**
- Enclose text containing commas in double quotes
- Escape double quotes by doubling them: `""`

**Examples:**
```csv
response
"I love the product, it's great!"
"She said, ""This is amazing!"""
"Cost: $50 (includes tax)"
```

---

### Line Breaks in Text

**Handling Multi-Line Responses:**

CSV requires special handling for line breaks within responses.

**✅ Correct - Quoted Multi-Line Text:**
```csv
id,response
1,"First line
Second line
Third line"
2,"Another response"
```

**❌ Problematic - Unquoted Line Breaks:**
```csv
id,response
1,First line
Second line  ← Interpreted as new row!
```

**Best Practice:**
- Enclose responses with line breaks in double quotes
- Or replace line breaks with space/semicolon before export

---

### Reserved Patterns (Flagged by Quality Filter)

**Non-Response Patterns:**
```
N/A, n/a, NA, na, none, nothing, idk, i don't know,
no comment, no response, skip, pass, ---, ..., ???
```

**Test Response Patterns:**
```
test, testing, asdf, qwer, 123, abc, xxx, zzz,
sample, example, placeholder, lorem ipsum
```

**Keyboard Walks (Gibberish):**
```
qwerty, asdfgh, zxcvbn, qwertyuiop, asdfghjkl,
1234567890, 0987654321
```

**Why Flagged:**
- Non-analytic content
- Likely not genuine responses
- Low information value

**What Happens:**
- Response retained in dataset
- Quality flags added
- Recommendation: `exclude` or `review`
- User decides whether to include in analysis

---

## CSV Formatting Specifications

### Standard CSV Format

**Specifications:**
- **Delimiter:** Comma (`,`)
- **Text Qualifier:** Double quote (`"`)
- **Line Ending:** CRLF (`\r\n`) or LF (`\n`)
- **Encoding:** UTF-8
- **Header:** Required (first row)

---

### CSV Best Practices

**1. Always Include Header Row:**
```csv
id,response,timestamp
1,"Response text","2024-01-01"
```

**2. Quote Text Fields Containing Delimiters:**
```csv
id,response
1,"Text with comma, needs quotes"
2,"Text without comma"
```

**3. Escape Double Quotes by Doubling:**
```csv
id,response
1,"She said, ""Hello!"""
```
→ Renders as: `She said, "Hello!"`

**4. Handle Line Breaks with Quotes:**
```csv
id,response
1,"Line 1
Line 2"
```

**5. Use Consistent Delimiters:**
- Standard: comma (`,`)
- Alternative: semicolon (`;`) - specify when loading
- Avoid tab (`\t`) unless necessary

---

### Alternative Delimiters

**Semicolon-Delimited (common in European locales):**
```csv
id;response;timestamp
1;"Response text";"2024-01-01"
```

**Loading with Custom Delimiter:**
```python
df = loader.load_csv('data.csv', sep=';')
```

---

### CSV Example

**File: `survey_responses.csv`**
```csv
id,response,respondent_id,timestamp,age_group,gender
1,"I love the flexibility of remote work",R001,2024-01-01,25-34,Female
2,"Better work-life balance is crucial",R002,2024-01-02,35-44,Male
3,"Communication challenges with team members",R003,2024-01-03,25-34,Female
4,"Increased productivity at home",R004,2024-01-04,45-54,Male
5,"Missing social interactions with colleagues",R005,2024-01-05,25-34,Non-binary
```

---

## Excel Formatting Specifications

### Excel Workbook Structure

**Requirements:**
- Header row in first row (row 1)
- Data starts in second row (row 2)
- No merged cells in data area
- No formulas (values only, or formulas evaluated on load)
- No empty rows within data

---

### Sheet Organization

**Single Sheet (Recommended):**
- Place all data in first sheet
- Name sheet descriptively: `"Survey_Responses"`

**Multiple Sheets:**
- Specify sheet name when loading:
  ```python
  df = loader.load_excel('data.xlsx', sheet_name='Survey_Responses')
  ```

---

### Excel Best Practices

**1. Use First Row for Headers:**
```
Row 1: id | response | timestamp
Row 2: 1  | "Text"   | 2024-01-01
```

**2. Format Dates as ISO 8601:**
- Select column → Format Cells → Custom → `YYYY-MM-DD`

**3. Avoid Formatting:**
- No bold/italic (formatting stripped on load)
- No colors (not preserved)
- No borders (not relevant)

**4. Remove Non-Data Elements:**
- Delete charts, images, pivot tables
- Remove extra sheets if not needed

**5. Save Correctly:**
- **For UTF-8 CSV:** File → Save As → CSV UTF-8 (Comma delimited)
- **For Excel:** Use `.xlsx` (not `.xls` for modern format)

---

### Excel to CSV Conversion

**Method 1: Save As CSV UTF-8**
1. File → Save As
2. Format: "CSV UTF-8 (Comma delimited) (.csv)"
3. Save

**Method 2: Google Sheets**
1. Upload Excel file to Google Sheets
2. File → Download → Comma-separated values (.csv)
3. Automatically UTF-8 encoded

**Why Convert:**
- CSV more portable
- Smaller file size
- Easier version control
- Universal compatibility

---

## JSON Formatting Specifications

### JSON Array of Objects

**Standard format** for tabular data:

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

**Loading:**
```python
df = loader.load_json('data.json')
```

---

### JSON Lines (JSONL)

**One object per line** (newline-delimited JSON):

```json
{"id": 1, "response": "I love the flexibility of remote work", "respondent_id": "R001", "timestamp": "2024-01-01"}
{"id": 2, "response": "Better work-life balance is crucial", "respondent_id": "R002", "timestamp": "2024-01-02"}
```

**Loading:**
```python
df = loader.load_json('data.jsonl', lines=True)
```

**When to Use JSON Lines:**
- Large datasets (streaming)
- Appending records incrementally
- Log file format

---

### JSON Best Practices

1. **Use consistent key names** across all objects
2. **Include all fields** even if null: `"age_group": null`
3. **Use UTF-8 encoding** (default for JSON)
4. **Escape special characters:**
   - Quotes: `\"`
   - Newlines: `\n`
   - Backslash: `\\`
5. **Validate JSON** before loading (use online validator)

---

## Template and Schema Reference

### Input Data Template

**Location:** `/home/user/JC-OE-Coding/documentation/input_data_template.xlsx`

**Contents:**
- Pre-formatted Excel template with proper column structure
- Example data demonstrating format
- Data validation rules
- Instructions sheet

**How to Use:**
1. Download template from repository
2. Replace example data with your responses
3. Maintain column structure
4. Save as CSV UTF-8 or Excel (.xlsx)
5. Load into framework

**Template Structure:**

| id | response | respondent_id | timestamp | topic | age_group | gender |
|----|----------|---------------|-----------|-------|-----------|--------|
| 1 | [Your response text here] | R001 | 2024-01-01 | [Optional topic] | 25-34 | Female |
| 2 | [Your response text here] | R002 | 2024-01-01 | [Optional topic] | 35-44 | Male |

---

### Schema Validation

**Automatic Validation on Load:**
- Column name detection
- Data type inference
- Encoding verification
- Response count check
- Required column presence

**Manual Validation:**
```python
from src.data_loader import DataLoader

loader = DataLoader()
df = loader.load_csv('your_file.csv')

# Validate structure
loader.validate_dataframe(df, required_columns=['response'])

# Assess content quality
df = loader.assess_content_quality(df, text_column='response')

# Get quality summary
summary = loader.get_quality_summary(df)
print(summary)
```

---

## Minimal Sample Dataset

### Basic CSV Example

**Minimal viable dataset** (3 columns, 5 rows):

```csv
id,response,timestamp
1,"I love the flexibility of remote work",2024-01-01
2,"Better work-life balance is crucial",2024-01-02
3,"Communication challenges with team members",2024-01-03
4,"Increased productivity at home",2024-01-04
5,"Missing social interactions with colleagues",2024-01-05
```

**This dataset includes:**
- ✅ Required `response` column
- ✅ Optional `id` and `timestamp` columns
- ✅ UTF-8 encoding
- ✅ Proper CSV format
- ✅ Sufficient responses (5, though 20+ recommended)

---

### Extended Example with Demographics

```csv
id,response,respondent_id,timestamp,topic,age_group,gender,location
1,"Remote work has been life-changing",R001,2024-01-01,work,25-34,Female,Urban
2,"I miss office collaboration",R002,2024-01-02,work,35-44,Male,Suburban
3,"Flexible schedule is amazing",R003,2024-01-03,work,25-34,Non-binary,Rural
4,"Technology issues are frustrating",R004,2024-01-04,work,45-54,Female,Urban
5,"Cost savings from not commuting",R005,2024-01-05,work,35-44,Male,Suburban
```

**Additional columns enable:**
- Demographic stratification
- Topic-based analysis
- Temporal trends
- Geographic comparisons

---

## Validation Rules

### Automatic Validation Checks

When data is loaded, the framework automatically validates:

| Check | Description | Action if Failed |
|-------|-------------|------------------|
| **File exists** | File path is valid | Raise `FileNotFoundError` |
| **File readable** | File can be opened | Raise `IOError` |
| **Encoding valid** | UTF-8 or compatible | Attempt auto-detection; warn if issues |
| **Header present** | First row contains column names | Raise `ValueError` |
| **Response column** | Required `response` column exists | Raise `ValueError` with suggestion |
| **Non-empty** | Dataset contains at least 1 row | Raise `pd.errors.EmptyDataError` |
| **Data types** | Columns parseable to expected types | Auto-convert; warn on failures |

---

### Content Quality Validation

**Automatic quality assessment** via `ContentQualityFilter`:

| Check | Criteria | Flag | Recommendation |
|-------|----------|------|----------------|
| **Null/Empty** | Response is null or empty string | `null`, `empty` | `exclude` |
| **Too Short (chars)** | < 10 characters | `too_short_chars` | `review` |
| **Too Short (words)** | < 3 words | `too_short_words` | `review` |
| **Non-Response** | Matches patterns like "N/A", "idk" | `non_response` | `exclude` |
| **Test Response** | Matches patterns like "test", "asdf" | `test_response` | `exclude` |
| **Gibberish** | Keyboard walks, low vowel ratio | `gibberish` | `exclude` or `review` |
| **Non-English** | < 30% recognized English words | `non_english` | `review` |
| **Excessive Repetition** | > 70% repeated words | `excessive_repetition` | `review` |
| **No Alphabetic** | Only punctuation/numbers | `no_alphabetic` | `exclude` |

**Example Quality Assessment:**
```python
df = loader.assess_content_quality(
    df,
    text_column='response',
    min_words=3,
    min_chars=10,
    max_repetition_ratio=0.7,
    min_english_word_ratio=0.3
)

# Columns added:
# - quality_is_analytic (bool)
# - quality_confidence (float 0-1)
# - quality_reason (str)
# - quality_recommendation ('include', 'review', 'exclude')
# - quality_flags (list)
```

---

### Manual Validation Steps

**Before Loading Data:**

1. **Open file in text editor** - verify encoding (UTF-8), check for special characters
2. **Check delimiter** - ensure commas (or specified delimiter) used consistently
3. **Review header row** - confirm column names match expected format
4. **Scan first/last rows** - check data populated correctly
5. **Count rows** - ensure meets minimum (20-30 responses)

**After Loading Data:**

```python
# Check loaded data
print(df.shape)  # (rows, columns)
print(df.columns.tolist())  # Column names
print(df.head())  # First 5 rows
print(df.info())  # Data types
print(df['response'].isna().sum())  # Count missing responses

# Validate structure
loader.validate_dataframe(df, required_columns=['response'])

# Assess quality
df = loader.assess_content_quality(df, text_column='response')
summary = loader.get_quality_summary(df)
print(f"Analytic responses: {summary['analytic_percentage']}%")
```

---

### Common Validation Errors

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `FileNotFoundError` | Incorrect file path | Verify path, use absolute path |
| `EmptyDataError` | File is empty | Check file has data rows |
| `Missing required columns: {'response'}` | No `response` column | Rename column or add `response` column |
| `UnicodeDecodeError` | Wrong encoding | Re-save as UTF-8 |
| `ParserError: Expected N fields, saw M` | Inconsistent column count | Check for unquoted commas in text |
| `ValueError: Query contains disallowed operation` | SQL query uses DROP/UPDATE | Use SELECT-only queries |

---

## Cross-References

**Related Documentation:**
- [Input Data Specification](03_input_data_specification.md) - Data requirements and structure
- [Input Data Template](/home/user/JC-OE-Coding/documentation/input_data_template.xlsx) - Excel template
- [Rigor Diagnostics Guide](RIGOR_DIAGNOSTICS_GUIDE.md) - Quality assessment metrics

**Source Code:**
- `src/data_loader.py` - Data loading implementation
- `src/content_quality.py` - Content quality filtering logic

**Sample Data:**
- `/home/user/JC-OE-Coding/data/Remote_Work_Experiences_200.csv`
- `/home/user/JC-OE-Coding/data/consumer_perspectives_responses.csv`

---

## Quick Reference: Formatting Checklist

Before loading your data, verify:

- [ ] **Encoding:** File saved as UTF-8
- [ ] **Header:** First row contains column names
- [ ] **Response column:** Present and named `response` (or similar)
- [ ] **Delimiters:** Commas (or consistent delimiter throughout)
- [ ] **Text quoting:** Fields with commas/quotes properly escaped
- [ ] **Date format:** ISO 8601 (YYYY-MM-DD) preferred
- [ ] **Missing values:** Empty cells (not "N/A" text)
- [ ] **Categorical values:** Consistent case/spelling
- [ ] **Minimum rows:** At least 20-30 responses
- [ ] **No empty responses:** All response cells have content (or properly empty)
- [ ] **Valid content:** No test data, placeholders, or gibberish
- [ ] **Column names:** Lowercase with underscores, no special characters

**Next Steps:**
1. Download [Input Data Template](/home/user/JC-OE-Coding/documentation/input_data_template.xlsx)
2. Review [Input Data Specification](03_input_data_specification.md)
3. Load data and run quality validation
4. Review flagged responses and decide on inclusion

---

**Document Revision History:**

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-25 | Initial release | Agent-D (Data Contract) |

---

**For Questions or Issues:**
- Consult source code: `src/data_loader.py`, `src/content_quality.py`
- Review sample data files in `data/` directory
- Check [Input Data Specification](03_input_data_specification.md) for requirements
