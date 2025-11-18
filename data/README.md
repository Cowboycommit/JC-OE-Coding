# Data Directory

This directory contains sample data files for the Open-Ended Coding Analysis framework.

## Files

### sample_responses.csv

Sample dataset of remote work experiences with 30 responses.

**Columns:**
- `id`: Unique response identifier
- `response`: Text response from participant
- `respondent_id`: Unique participant identifier
- `timestamp`: Response submission date

**Use Case:**
This sample data demonstrates coding, theme identification, and categorization of qualitative responses about remote work experiences.

## Adding Your Own Data

### CSV Format

Your CSV file should include at minimum:
- One column with text responses
- Optional: respondent identifiers, timestamps, demographics

Example:
```csv
id,response,metadata
1,"Your text response here",optional_field
2,"Another response",optional_field
```

### Excel Format

Excel files are also supported. Ensure your data is in a worksheet with:
- Header row with column names
- Text response column
- Any additional metadata columns

### Database Format

For SQLite or PostgreSQL databases:
1. Create a table with response data
2. Use SQL queries to load data in the notebook
3. Example: `SELECT response_text, user_id FROM survey_responses`

## Data Privacy

**Important:** This directory is for demonstration purposes. Always:
- Remove personally identifiable information (PII) before analysis
- Follow your institution's IRB guidelines
- Ensure participant consent for data analysis
- Keep sensitive data secure and encrypted

## Sample Data Description

The provided sample data represents fictional responses to a survey question:
*"How has remote work affected your professional experience?"*

Responses cover various themes:
- Work-life balance
- Productivity
- Social connections
- Technology challenges
- Flexibility and autonomy

This diversity makes it ideal for demonstrating the coding framework's capabilities.
