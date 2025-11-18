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

### trump_responses.csv

Dataset analyzing public opinions and perspectives on political leadership with 30 responses.

**Columns:**
- `id`: Unique response identifier
- `response`: Text response from participant
- `respondent_id`: Unique participant identifier
- `timestamp`: Response submission date
- `topic`: Topic category tag

**Use Case:**
Demonstrates qualitative coding of political discourse, policy analysis, and public opinion research. Covers topics including:
- Policy impacts (trade, immigration, healthcare)
- Media coverage and communication styles
- Foreign policy and international relations
- Economic policies and reforms
- Political rhetoric and voter engagement

### epstein_case_responses.csv

Dataset examining institutional accountability and justice system issues with 30 responses.

**Columns:**
- `id`: Unique response identifier
- `response`: Text response from participant
- `respondent_id`: Unique participant identifier
- `timestamp`: Response submission date
- `topic`: Topic category tag

**Use Case:**
Demonstrates qualitative analysis of systemic issues, institutional oversight, and justice system accountability. Covers themes including:
- Institutional failures and oversight
- Justice system transparency
- Media coverage and investigative journalism
- Legal reforms and accountability
- Victim advocacy and support systems

**Note:** This dataset focuses on systemic and institutional analysis for educational purposes.

### cricket_responses.csv

Dataset analyzing cricket enthusiasts' perspectives with 30 responses.

**Columns:**
- `id`: Unique response identifier
- `response`: Text response from participant
- `respondent_id`: Unique participant identifier
- `timestamp`: Response submission date
- `topic`: Topic category tag

**Use Case:**
Demonstrates qualitative coding of sports-related discourse and fan perspectives. Covers topics including:
- Different cricket formats (Test, T20, ODI)
- Commercialization and leagues (IPL, Big Bash)
- Technology in cricket (DRS)
- Women's cricket and inclusivity
- Rivalries and traditions
- Community and cultural aspects

### fashion_responses.csv

Dataset examining fashion industry perspectives and consumer attitudes with 30 responses.

**Columns:**
- `id`: Unique response identifier
- `response`: Text response from participant
- `respondent_id`: Unique participant identifier
- `timestamp`: Response submission date
- `topic`: Topic category tag

**Use Case:**
Demonstrates qualitative analysis of consumer behavior, industry trends, and social issues in fashion. Covers themes including:
- Sustainability and ethical fashion
- Fast fashion vs. sustainable alternatives
- Body positivity and inclusivity
- Personal style and self-expression
- Fashion industry trends and innovation
- Cultural and social aspects of fashion

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

The provided sample datasets represent diverse research topics suitable for qualitative analysis:

### Available Datasets:

1. **Remote Work (sample_responses.csv)**
   - Survey question: *"How has remote work affected your professional experience?"*
   - Themes: work-life balance, productivity, social connections, technology

2. **Political Leadership (trump_responses.csv)**
   - Public opinion on political leadership and policy impacts
   - Themes: policy analysis, media coverage, governance, voter perspectives

3. **Justice System (epstein_case_responses.csv)**
   - Institutional accountability and justice system analysis
   - Themes: oversight, transparency, legal reform, accountability

4. **Sports Culture (cricket_responses.csv)**
   - Cricket enthusiasts' perspectives on the sport
   - Themes: formats, commercialization, technology, tradition, community

5. **Fashion Industry (fashion_responses.csv)**
   - Consumer attitudes and industry perspectives
   - Themes: sustainability, inclusivity, trends, ethics, self-expression

### Using These Datasets

Each dataset contains 30 diverse responses designed to demonstrate:
- Code frame development and application
- Theme identification and analysis
- Multi-level categorization
- Co-occurrence patterns
- Visualization techniques

Choose the dataset that best fits your research interests or analysis needs. All datasets follow the same structure for easy comparison and learning.
