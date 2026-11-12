# E-Commerce Customer & Sales Analytics Copilot

An enterprise-ready analytics engineering and business intelligence platform for e-commerce intelligence. This solution combines automated data validation, customer RFM segmentation, statistical hypothesis testing (A/B testing), unsupervised anomaly detection (Isolation Forest), an interactive Streamlit application, dimensional star-schema modeling, and privacy-guarded AI insights.

---

## Table of Contents
- [Executive Overview](#executive-overview)
- [System Architecture](#system-architecture)
- [Core Analytical Modules](#core-analytical-modules)
  - [1. Data Validation & Quality Gates](#1-data-validation--quality-gates)
  - [2. Exploratory Data Analysis & KPI Engine](#2-exploratory-data-analysis--kpi-engine)
  - [3. RFM Customer Segmentation](#3-rfm-customer-segmentation)
  - [4. Statistical Hypothesis Testing (A/B Testing)](#4-statistical-hypothesis-testing-ab-testing)
  - [5. Machine Learning Anomaly Detection](#5-machine-learning-anomaly-detection)
  - [6. Guarded AI Insight Synthesis](#6-guarded-ai-insight-synthesis)
  - [7. Enterprise SQL & Dimensional Modeling](#7-enterprise-sql--dimensional-modeling)
- [Interactive Streamlit Dashboard](#interactive-streamlit-dashboard)
- [Key Business Insights & Recommendations](#key-business-insights--recommendations)
- [Repository Structure](#repository-structure)
- [Installation & Quickstart Guide](#installation--quickstart-guide)
- [Automated Testing & CI/CD](#automated-testing--cicd)
- [Data Integrity Notice](#data-integrity-notice)

---

## Executive Overview

E-commerce leadership teams often face fragmented visibility across transaction streams, customer lifecycles, and digital experiments. This platform establishes a reliable, reproducible analytics workflow:

- **Volume Ingested**: 15,000 transaction records, 2,500 customer profiles, and 40,000 web sessions.
- **Total Portfolio Revenue**: $76,055,193.85 across 610 operating days ($5,070 AOV).
- **Primary Methodologies**: Welch's two-sample t-test, Pearson's Chi-square test, Bonferroni correction, Cohen's d effect size, RFM quintile ranking, and scikit-learn Isolation Forest anomaly isolation.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        INGESTION & RAW STORAGE                         │
│  data/raw/orders.csv       │  data/raw/customers.csv                   │
│  data/raw/sessions.csv     │  data/raw/products.csv                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               DATA INTEGRITY & VALIDATION (00_data_validation.py)      │
│  • Schema conformity & type enforcement                                │
│  • Null / missing value thresholds (<5%)                               │
│  • Foreign-key referential integrity check                             │
│  • Value domain & range checks (discount %: 0-1, rating: 1-5)          │
│  OUTPUT: data/processed/data_validation_report.json                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Validation Passed)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              EXPLORATORY DATA ANALYSIS ENGINE (01_eda.py)              │
│  • Relational merge: Orders + Customers + Products                     │
│  • Daily KPI aggregation (revenue, order counts, returns)              │
│  • Monthly cohort rollups & growth velocities                          │
│  OUTPUT: daily_kpis.csv, monthly_kpis.csv                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐        ┌──────────────────┐
│ RFM SEGMENTATION │       │   A/B TESTING    │        │ ANOMALY DETECTOR │
│ (02_rfm_seg.py)  │       │ (03_ab_test.py)  │        │ (04_anomaly.py)  │
├──────────────────┤       ├──────────────────┤        ├──────────────────┤
│ • Recency (days) │       │ • Welch's t-test │        │ • Isolation      │
│ • Frequency (qty)│       │   (Revenue)      │          Forest algorithm │
│ • Monetary ($)   │       │ • Chi-Square     │        │ • 7d/30d moving  │
│ • 5-tier scoring │       │   (Return Rate)  │          window features  │
│ OUTPUT:          │       │ • Bonferroni     │        │ • 3% anomaly     │
│ rfm_segments.csv │       │   adjustment     │          contamination    │
│                  │       │ OUTPUT:          │        │ OUTPUT: daily_   │
│                  │       │ ab_test_results  │        │ kpis_anomalies   │
└────────┬─────────┘       └─────────┬────────┘        └────────┬─────────┘
         │                           │                          │
         └───────────────────────────┼──────────────────────────┘
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      CONSUMPTION & DECISION LAYER                      │
│                                                                        │
│  📊 Streamlit App (dashboard.py)         🤖 Guarded AI Engine (ai/)    │
│  • Executive KPI Overview                • Strict PII sanitization     │
│  • RFM Customer Distributions            • Dynamic prompt templates    │
│  • Statistical A/B Experiment View       • Zero-hallucination metric   │
│  • Time-Series Anomaly Investigation       constraints                  │
│  • Data Quality Audit Metrics            📈 Dimensional Star Schema    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Core Analytical Modules

### 1. Data Validation & Quality Gates
- **Script**: `python/00_data_validation.py`
- Enforces strict constraints prior to analytics execution:
  - Missing value tolerance: `<5%` allowed per column.
  - Referential integrity: Identifies and flags orphaned records between `orders.csv` and `customers.csv`.
  - Domain constraints: Discounts bounded in `[0.0, 1.0]`, review ratings in `[1, 5]`, shipping days in `[1, 30]`.
- Output: Structured JSON report in `data/processed/data_validation_report.json`.

### 2. Exploratory Data Analysis & KPI Engine
- **Script**: `python/01_eda.py`
- Computes baseline financial and operational metrics across 610 operating days:
  - Net Revenue, Gross Margin, Average Order Value (AOV).
  - Monthly active customer trajectories and order volume trends.
  - Output tables: `daily_kpis.csv` and `monthly_kpis.csv`.

### 3. RFM Customer Segmentation
- **Script**: `python/02_rfm_segmentation.py`
- Ranks each customer on a 1–5 quintile scale across three behavioral dimensions:
  - **Recency ($R$)**: Days since latest completed order.
  - **Frequency ($F$)**: Total completed lifetime orders.
  - **Monetary ($M$)**: Cumulative revenue generated.
- Segment Definitions:
  - **Champions** ($R \ge 4, F \ge 4$): 448 customers (17.9%) — High engagement, highest LTV.
  - **Loyal** ($R \ge 3, F \ge 3$): 459 customers (18.4%) — Steady repeat purchase patterns.
  - **At Risk** ($R \le 2, F \ge 3$): 356 customers (14.2%) — High historic spend, lapsed recency.
  - **New / Promising** ($R \ge 4, F \le 2$): 277 customers (11.1%) — High recent intent, single purchase.
  - **Regular**: 960 customers (38.4%) — Baseline customer cohort.

### 4. Statistical Hypothesis Testing (A/B Testing)
- **Script**: `python/03_ab_testing.py`
- Tests checkout experiment groups (Control: 7,620 orders vs. Variant: 7,380 orders):
  - **Revenue Evaluation**: Welch's two-sample t-test (two-sided, unequal variance).
  - **Return Rate Evaluation**: Pearson's Chi-square test of independence.
  - **Multiple Comparison Correction**: Bonferroni adjustment ($\alpha = 0.05 / 2 = 0.025$).
  - **Standardized Effect Size**: Cohen's $d$ calculation.

### 5. Machine Learning Anomaly Detection
- **Script**: `python/04_anomaly_detection.py`
- Uses scikit-learn's `IsolationForest` to identify irregular operational events:
  - Evaluates multi-dimensional feature space: Revenue, Order Volume, Return Rate, plus 7-day and 30-day rolling averages.
  - Contamination parameter calibrated to 3.0%.
  - Flags 19 anomalous operational days across 610 tracked dates.

### 6. Guarded AI Insight Synthesis
- **Module**: `ai/insight_generator.py`
- Synthesizes executive briefings with safety guardrails:
  - **PII Scrubbing**: Strips customer names, email addresses, and phone numbers.
  - **Schema Constraints**: Ensures only verified aggregate figures enter the prompt template (`ai/prompt_template.md`).
  - **Zero-Hallucination Guard**: Rejects prompts containing negative revenue or out-of-bound return rates.

### 7. Enterprise SQL & Dimensional Modeling
- **Location**: `sql/`
- Implements an enterprise star schema (`DimCustomer`, `DimProduct`, `FactOrder`):
  - Foreign key constraints and non-clustered composite indexes (`IX_FactOrder_OrderDate_Customer`).
  - Window functions (`ROW_NUMBER`, `DENSE_RANK`, `LAG`) for Month-over-Month growth calculations.
  - Production-ready DAX measures (`powerbi/dax_measures.md`) for Power BI deployment (MTD, Rolling 30D, AOV, Return Rate).

---

## Interactive Streamlit Dashboard

Launch the web application via `streamlit run dashboard.py` to access 5 interactive pages:

| Page | Description | Key Visualizations |
|---|---|---|
| **📈 Overview** | Executive summary of sales performance | Metric cards, monthly revenue spline charts, order volume bar charts, active customer trends |
| **👥 Customer Segmentation** | RFM tier breakdown | Interactive donut chart, customer counts, box plots of monetary spend and frequency by segment |
| **🧪 A/B Testing** | Hypothesis test conclusions | Control vs. Variant metric comparison cards, Cohen's d effect indicators, p-value readouts |
| **🚨 Anomalies** | Isolation Forest timeline analysis | Scatter plot of daily revenue highlighting normal vs. anomalous dates, detailed anomaly data grid |
| **📋 Data Quality** | Health and integrity scorecards | Record counts, missingness status, referential integrity badge, validation issue summary |

---

## Key Business Insights & Recommendations

### 1. A/B Testing Conclusion
- **Mean Order Revenue**: Control = $5,107.17 vs. Variant = $5,032.33 (-1.47% observed delta; $p = 0.4408$).
- **Return Rate**: Control = 9.53% vs. Variant = 9.13% (-0.39% points observed delta; $p = 0.4221$).
- **Decision**: **Fail to reject the null hypothesis ($H_0$)** for both metrics. Neither revenue nor return differences are statistically significant ($p > 0.05$). **Recommendation:** Do not invest engineering resources in rolling out the variant checkout design without a larger test or revised value proposition.

### 2. Customer Retention Strategy
- **Champions & Loyal Customers** represent 36.3% of users and drive the majority of margin. Implement a tiered loyalty reward program to preserve retention.
- **356 At-Risk Customers** (14.2%) show historical high order frequency but no purchases in the last 60+ days. Recommended automated win-back re-engagement campaign.

### 3. Anomaly Patterns
- Out of 610 days, 19 days (3.11%) were detected as operational anomalies.
- Top revenue spikes (e.g., $288,994 on 2026-07-16) coincided with promotional periods that also exhibited elevated return volume. Operational monitoring should review inventory buffers prior to scheduled promotions.

---

## Repository Structure

```text
├── .github/workflows/
│   └── data-quality.yml          # GitHub Actions CI for automated data testing
├── .streamlit/
│   └── config.toml               # Streamlit application configurations
├── ai/
│   ├── insight_generator.py      # PII-sanitized LLM prompt generator
│   ├── prompt_template.md        # Prompt contract for executive summaries
│   └── sample_metrics.json       # Structured KPI test payload
├── config.yml                    # Central configuration for pipeline thresholds
├── dashboard.py                  # Multi-page Streamlit dashboard
├── data/
│   ├── processed/                # Validated and modeled outputs
│   │   ├── ab_test_results.csv   # Statistical test metrics
│   │   ├── daily_kpis.csv        # Aggregated daily operational metrics
│   │   ├── daily_kpis_with_anomalies.csv # Time-series with anomaly flags
│   │   ├── data_validation_report.json   # Quality audit output
│   │   ├── monthly_kpis.csv      # Monthly executive metrics
│   │   └── rfm_segments.csv      # Customer segment assignments
│   └── raw/                      # Raw event tables (orders, customers, products, sessions)
├── docs/
│   ├── ARCHITECTURE.md           # In-depth system architectural specifications
│   ├── ab_test_results.md        # Detailed statistical test methodology
│   ├── anomaly_analysis.md       # Isolation Forest analysis and anomaly list
│   ├── data_dictionary.md        # Column definitions and data types
│   └── interview_story.md        # STAR method portfolio breakdown
├── logs/                         # Execution logs from pipeline runs
├── powerbi/
│   ├── dashboard_spec.md         # Canvas layout and visual mockups
│   └── dax_measures.md           # Production DAX expressions
├── python/
│   ├── 00_data_validation.py     # Schema and referential integrity validator
│   ├── 01_eda.py                 # KPI aggregation pipeline
│   ├── 02_rfm_segmentation.py    # RFM classification engine
│   ├── 03_ab_testing.py          # Welch's t-test and Chi-square evaluation
│   ├── 04_anomaly_detection.py   # Isolation Forest model execution
│   ├── 05_data_profiling.py      # Statistical distribution analysis
│   └── run_pipeline.py           # Master CLI orchestrator
├── sql/
│   ├── 01_schema.sql             # DDL star-schema definitions
│   ├── 02_kpi_queries.sql        # Window functions and KPI queries
│   ├── 03_customer_analytics.sql # Cohort retention SQL
│   └── 04_data_quality.sql       # Database constraint verification
├── tests/
│   └── test_data_validation.py   # Pytest test cases for data gates
└── requirements.txt              # Project dependencies
```

---

## Installation & Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14
- Git

### 2. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/NETIZEN-11/ai-emcomerce-data-analysisi.git
cd ai-emcomerce-data-analysisi

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Run Analytics Pipeline
Execute all processing steps in dependency order:
```bash
python python/run_pipeline.py
```
*Options:*
- `python python/run_pipeline.py --dry-run` : Validate configuration without running scripts.
- `python python/run_pipeline.py --skip-validation` : Bypass initial data validation.

### 4. Start the Interactive Dashboard
Launch the Streamlit web application:
```bash
streamlit run dashboard.py
```
Open your browser to `http://localhost:8501`.

---

## Automated Testing & CI/CD

Unit tests validate data integrity rules, boundary conditions, and schema specifications:

```bash
pytest tests/
```

- **Test Suite**: `tests/test_data_validation.py`
- **Coverage**: 13 automated test cases verifying schema conformity, missing value limits, referential integrity, and outlier detection logic.
- **CI Workflow**: `.github/workflows/data-quality.yml` triggers test runs on commits to `main`.

---

## Data Integrity Notice

All transaction records, session logs, and customer profiles in this repository are **100% synthetically generated** for educational and demonstration purposes. No confidential, proprietary, or customer-identifiable information is utilized.
