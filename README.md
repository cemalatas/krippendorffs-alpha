# Krippendorff's Alpha Calculator

A comprehensive Streamlit web application for computing intercoder reliability statistics, with a focus on Krippendorff's Alpha. This tool serves as both a calculator and an educational platform, featuring AI-powered explanations via Claude.

## Location

```
/Users/cemalatas/Desktop/vscode/kalpha_streamlit/
```

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/cemalatas/Desktop/vscode/kalpha_streamlit
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py --server.headless true
```

### 3. Open in Browser

Navigate to: **http://localhost:8501**

---

## What This App Does

This application calculates intercoder reliability statistics for content analysis research. It helps researchers determine how consistently multiple coders (raters) categorize the same content.

### Primary Statistics Computed

| Statistic | Description |
|-----------|-------------|
| **Krippendorff's Alpha** | The gold standard for intercoder reliability. Handles any number of coders, missing data, and all measurement levels |
| **Per-Variable Alpha** | Individual reliability scores for each coded variable |
| **Pairwise Alpha** | Agreement between each pair of coders |
| **Cohen's Kappa** | Chance-corrected pairwise agreement (2 coders) |
| **Fleiss' Kappa** | Multi-coder generalization of Cohen's Kappa |
| **Percent Agreement** | Simple proportion of matching codes (baseline) |

### Key Features

- **7-Step Wizard Interface**: Guided workflow from data upload to report generation
- **AI-Powered Explanations**: Claude Opus 4.5 provides contextual explanations and interpretations
- **Per-Row Disagreement Detection**: Identifies exactly where coders disagree
- **Coder Impact Analysis**: Shows how each coder affects overall reliability
- **Interactive Visualizations**: Heatmaps, bar charts, and confusion matrices
- **Multiple Export Formats**: CSV, XLSX, Markdown, DOCX, PDF

---

## App Structure

### Pages (7-Step Workflow)

| Step | Page | Description |
|------|------|-------------|
| 1 | **Welcome** | Enter Anthropic API key for AI features |
| 2 | **Data Upload** | Upload coder files (XLSX/CSV) |
| 3 | **Configure Variables** | Select unit ID column and coded variables |
| 4 | **Run Analysis** | Compute all reliability statistics |
| 5 | **Disagreements** | Explore per-row disagreements with filters |
| 6 | **Results** | Full dashboard with all statistics |
| 7 | **Report** | Generate and export final report |

### Project Structure

```
kalpha_streamlit/
├── app.py                      # Main entry point & sidebar chat
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── core/                       # Core computation modules
│   ├── models.py               # Data classes (CoderData, AlphaResult, etc.)
│   ├── krippendorff.py         # Alpha calculations
│   ├── additional_stats.py     # Kappa, percent agreement
│   ├── disagreement_analyzer.py # Per-row disagreement detection
│   ├── data_transformer.py     # Data loading and transformation
│   └── llm_client.py           # Claude API integration
│
├── visualization/              # Chart generation
│   └── charts.py               # Plotly visualizations
│
├── pages/                      # Streamlit multipage app
│   ├── 1_welcome.py
│   ├── 2_data_upload.py
│   ├── 3_variable_config.py
│   ├── 4_analysis.py
│   ├── 5_disagreements.py
│   ├── 6_results.py
│   └── 7_report.py
│
└── test_data/                  # Sample data files
    ├── ICR_Duygu.xlsx
    ├── ICR_ELIF.xlsx
    ├── ICR_ERIN.xlsx
    └── ICR_NAZLI.xlsx
```

---

## Supported Data Formats

### 1. Multiple Files (One Per Coder) - Recommended

Each coder has their own XLSX/CSV file with identical structure:
- Rows = units (articles, cases, etc.)
- Columns = metadata + coded variables

### 2. Single File with Coder Column

One file where each row is a coding instance:
- `unit_id` column: identifies the unit
- `coder_id` column: identifies who coded it
- Other columns: coded variables

### 3. Long Format

Tidy data format with columns:
- `unit_id`, `coder`, `variable`, `value`

---

## Measurement Levels

The app supports all four measurement levels for Krippendorff's Alpha:

| Level | Description | Example |
|-------|-------------|---------|
| **Nominal** | Categories with no inherent order | positive/negative/neutral |
| **Ordinal** | Ordered categories, unequal intervals | 1-3 rating scale |
| **Interval** | Equal intervals, no true zero | Temperature in Celsius |
| **Ratio** | Equal intervals with true zero | Counts, percentages |

Most content analysis variables are **nominal** or **ordinal**.

---

## AI Features (Requires Anthropic API Key)

The app integrates Claude Opus 4.5 (`claude-opus-4-5-20251101`) for:

1. **Data Structure Analysis**: AI analyzes your uploaded data and suggests variable configurations
2. **Results Interpretation**: Contextual explanations of reliability statistics
3. **Disagreement Analysis**: AI explains why specific disagreements might have occurred
4. **Report Generation**: AI-written executive summaries based on Krippendorff's methodology
5. **Sidebar Chat**: Ask questions about your results or intercoder reliability concepts

**Note**: The API key is stored in session state only (not saved to disk).

---

## Interpreting Results

### Krippendorff's Alpha Thresholds

| Alpha Value | Interpretation |
|-------------|----------------|
| α ≥ 0.80 | **Acceptable** - Good reliability |
| 0.67 ≤ α < 0.80 | **Tentative** - Use with caution |
| α < 0.67 | **Insufficient** - Not reliable |

### Understanding "Overall Alpha"

The **Overall K's Alpha** displayed is the **mean of all per-variable alphas**, not a single combined calculation. To compare with tools like ReCal:
- Use the per-variable alpha values
- The "Verify Data Before Analysis" section shows individual variable alphas with 6 decimal places

---

## Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.0.0
krippendorff>=0.8.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
plotly>=5.18.0
anthropic>=0.40.0
python-docx>=1.1.0
pdfplumber>=0.10.0
reportlab>=4.0.0
```

---

## Comparison with ReCal

This app uses the same underlying `krippendorff` Python library. To verify calculations match ReCal:

1. Go to **Step 4: Run Analysis**
2. Expand **"Verify Data Before Analysis"**
3. Select a single variable
4. Compare the displayed alpha (6 decimals) with ReCal's output

---

## Troubleshooting

### "Please enter your API key" error
The AI features require an Anthropic API key. Enter it on the Welcome page or skip AI features.

### Calculation discrepancy with ReCal
- Ensure you're comparing the same variable (not "Overall Alpha" which is a mean)
- Verify the correct columns are selected in Step 3
- Check measurement level matches (nominal vs ordinal)

### Streamlit asks for email on startup
Run with the headless flag:
```bash
streamlit run app.py --server.headless true
```

---

## Based On

- **Computation Core**: Adapted from `/Users/cemalatas/Desktop/vscode/okul/intercoder_rel/icr_krippendorff_compute.py`
- **Reference**: [ReCal](https://github.com/dfreelon/recal) by Deen Freelon
- **Methodology**: Krippendorff, K. (2004). *Content Analysis: An Introduction to Its Methodology*

---

## Author

Created for intercoder reliability analysis in content analysis research.
