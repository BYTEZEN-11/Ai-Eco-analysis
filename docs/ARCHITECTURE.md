# Data Analytics Project Architecture

## Overview

This is a production-ready end-to-end data analytics platform for e-commerce customer insights. The architecture follows a modular, scalable design with clear separation of concerns across data ingestion, validation, analysis, and export layers.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ DATA SOURCES (Raw CSV Files)                                │
├─────────────────────────────────────────────────────────────┤
│ orders.csv (15K rows) | customers.csv (2.5K rows) |         │
│ products.csv | sessions.csv (40K rows)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ DATA VALIDATION LAYER (00_data_validation.py)               │
├─────────────────────────────────────────────────────────────┤
│ • Schema validation                                         │
│ • Null/missing value detection                              │
│ • Referential integrity checks                              │
│ • Outlier flagging                                          │
│ • Data quality scoring                                      │
│ OUTPUT: data_validation_report.json                         │
└──────────────────────────┬──────────────────────────────────┘
                           │ (if valid)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ EXPLORATORY DATA ANALYSIS (01_eda.py)                       │
├─────────────────────────────────────────────────────────────┤
│ • Load & merge customer/order data                          │
│ • Compute executive KPIs                                    │
│ • Generate daily aggregates (for anomaly detection)         │
│ • Generate monthly aggregates (for trends)                  │
│ OUTPUT: daily_kpis.csv, monthly_kpis.csv                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ RFM SEGMENTATION │ │ A/B TESTING      │ │ PROFILING LAYER  │
│ (02_rfm_seg.py)  │ │ (03_ab_test.py)  │ │ (05_profiling.py)│
├──────────────────┤ ├──────────────────┤ ├──────────────────┤
│ • Compute RFM    │ │ • T-test revenue │ │ • Distributions  │
│ • Quintile score │ │ • Chi-sq returns │ │ • Correlations   │
│ • Assign segment │ │ • Effect sizes   │ │ • Outliers       │
│ • Export CSV     │ │ • Export CSV     │ │ • Temporal trend │
└──────────────────┘ └──────────────────┘ └──────────────────┘
          │                ▼                 │
          │       ab_test_results.csv        │
          │                                  │
          └────────────────┬─────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────┐
        │ ANOMALY DETECTION               │
        │ (04_anomaly_detection.py)       │
        ├─────────────────────────────────┤
        │ • Isolation Forest algorithm    │
        │ • Feature scaling               │
        │ • Rolling averages (7d, 30d)    │
        │ • Anomaly scoring               │
        │ OUTPUT: daily_kpis_with_        │
        │         anomalies.csv           │
        └─────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PROCESSED DATA LAYER (/data/processed/)                     │
├─────────────────────────────────────────────────────────────┤
│ • monthly_kpis.csv (trends, seasonality)                    │
│ • daily_kpis.csv (time-series, anomaly baseline)            │
│ • daily_kpis_with_anomalies.csv (flagged anomalies)         │
│ • rfm_segments.csv (customer segmentation)                  │
│ • ab_test_results.csv (hypothesis test outcomes)            │
│ • data_profile.json (statistical profiles)                  │
│ • correlation_matrix.csv (feature relationships)            │
│ • data_validation_report.json (quality metrics)             │
└─────────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
    PowerBI           Python/R           SQL Database
    Dashboard        Analytics          Data Warehouse
    & Reports        Applications        Integration
```

## Data Pipeline Execution

### Phase 1: Data Validation
**Script:** `python/00_data_validation.py`

**Purpose:** Pre-analysis data quality gate

**Key Checks:**
- Schema validation (required columns, data types)
- Null/missing value detection (fails if >5% missing)
- Duplicate detection (warns if duplicates found)
- Referential integrity (all customer_ids in customers table)
- Value range validation (discount 0-1, rating 1-5, etc.)
- Statistical outlier flagging (IQR method)

**Output:**
- `data/processed/data_validation_report.json` - Detailed validation results
- Exit code: 0 (pass), 1 (warnings), 2 (critical failure)

**Configuration:** `config.yml` → `analysis.validation`

---

### Phase 2: Exploratory Data Analysis (EDA)
**Script:** `python/01_eda.py`

**Purpose:** Generate aggregate KPIs for downstream analysis

**Key Outputs:**
- Daily KPIs (610 days): revenue, order count, customer count, return rate, shipping days, rating
- Monthly KPIs (21 months): same metrics + AOV (average order value)
- Executive summary statistics

**Critical:** This script creates `daily_kpis.csv`, which is required input for anomaly detection. Missing this file blocks Phase 4.

**Calculation Logic:**
```python
# Daily aggregation by order_date
Daily KPIs = {
    'revenue': SUM(net_revenue) per day,
    'orders': COUNT(DISTINCT order_id) per day,
    'customers': COUNT(DISTINCT customer_id) per day,
    'return_rate': AVG(returned) per day,  # 0-1
    'avg_shipping_days': AVG(shipping_days) per day,
    'avg_rating': AVG(rating) per day,
}

# Monthly aggregation
Monthly KPIs = {
    'revenue': SUM(net_revenue) per month,
    'orders': COUNT(DISTINCT order_id) per month,
    'customers': COUNT(DISTINCT customer_id) per month,
    'aov': revenue / orders,
}
```

---

### Phase 3: RFM Customer Segmentation
**Script:** `python/02_rfm_segmentation.py`

**Purpose:** Segment 2,500 customers into 5 actionable groups

**RFM Methodology:**
1. **Recency (R):** Days since last purchase (lower = better)
   - Formula: `(snapshot_date - MAX(order_date))`
   - Scored 5→1 (recent = high, inactive = low)

2. **Frequency (F):** Number of purchases (higher = better)
   - Formula: `COUNT(DISTINCT order_id)`
   - Scored 1→5 (quintiles)

3. **Monetary (M):** Total lifetime spending (higher = better)
   - Formula: `SUM(net_revenue)`
   - Scored 1→5 (quintiles)

**Segmentation Rules:**
```
If R ≥ 4 AND F ≥ 4         → Champions (20.8%)
If R ≥ 3 AND F ≥ 3         → Loyal (20.7%)
If R ≤ 2 AND F ≥ 3         → At Risk (18.5%)
If R ≥ 4 AND F ≤ 2         → New / Promising (11.1%)
Else                        → Regular (28.9%)
```

**Business Implications:**
- **Champions:** VIP treatment, loyalty rewards, personalized offers
- **Loyal:** Consistent engagement, maintain satisfaction
- **At Risk:** Win-back campaigns, special incentives
- **New/Promising:** Nurture path, growth potential
- **Regular:** Standard offerings, potential upsell

**Configuration:** `config.yml` → `analysis.rfm`

---

### Phase 4: Anomaly Detection
**Script:** `python/04_anomaly_detection.py`

**Purpose:** Identify unusual operational days for investigation

**Algorithm:** Isolation Forest (unsupervised)
- **Features:** revenue, orders, return_rate
- **Contamination:** 3% (expects ~18-19 anomalies in 610 days)
- **Random state:** 42 (reproducible)

**Output:**
- `daily_kpis_with_anomalies.csv` - Original daily KPIs + 3 new columns:
  - `rolling_7d_revenue`: 7-day moving average
  - `rolling_30d_revenue`: 30-day moving average
  - `anomaly_flag`: -1 (anomaly) or 1 (normal)
  - `is_anomaly`: binary 1/0 for easier filtering
  - `anomaly_score`: lower = more anomalous

**Interpretation:**
- Detected 19 anomalies (3.1% of 610 days)
- Average anomaly revenue: $152,976 (vs overall $104,900)
- Average anomaly orders: 26 (vs overall 25)
- Average anomaly return rate: 12.6% (vs overall 9.3%)

**Note:** Anomalies include both positive spikes (promotions) and negative (issues). Requires business context to interpret.

**Configuration:** `config.yml` → `analysis.anomaly`

---

### Phase 5: A/B Testing Analysis
**Script:** `python/03_ab_testing.py`

**Purpose:** Determine if variant outperforms control in test

**Statistical Tests:**

1. **Revenue Comparison** (Welch's t-test)
   - Tests: Control mean revenue vs Variant mean revenue
   - Assumptions: Independent samples, large sample size (robust to non-normality)
   - Output: t-statistic, p-value, Cohen's d (effect size), 95% CI

2. **Return Rate Comparison** (Chi-square test)
   - Tests: Control return rate vs Variant return rate
   - Contingency table: 2×2 (group × returned)
   - Output: Chi-square statistic, p-value, proportions

**Results:**
- Control revenue: $5,107.17 (7,620 orders)
- Variant revenue: $5,032.33 (7,380 orders)
- Difference: -$74.84 (-1.47%, Cohen's d=0.0126)
- p-value: 0.44078 (NOT statistically significant at α=0.05)

- Control return rate: 9.53%
- Variant return rate: 9.13%
- Difference: -0.39 percentage points
- p-value: 0.42206 (NOT statistically significant)

**Bonferroni Correction:**
- Running 2 tests → adjusted α = 0.05/2 = 0.025
- Both tests still fail to reach adjusted significance threshold

**Conclusion:** Variant does NOT significantly outperform Control on either metric.

**Configuration:** `config.yml` → `analysis.ab_test`

---

### Phase 6: Advanced Data Profiling
**Script:** `python/05_data_profiling.py`

**Purpose:** Generate comprehensive statistical profiles for descriptive reporting

**Outputs:**

1. **data_profile.json**
   - Numeric column stats (mean, std, skewness, kurtosis, outliers, normality)
   - Categorical column distributions
   - Distribution shapes and tail behaviors
   - Temporal analysis (date range, trends)
   - Data quality scores (completeness, uniqueness, validity)

2. **correlation_matrix.csv**
   - Pearson correlations between numeric features
   - Identifies multi-collinearity

**Data Quality Score:** 
- Composite of:
  - Completeness (40%): % non-null values
  - Uniqueness (30%): % non-duplicate rows
  - Validity (30%): % non-outlier values
- Rating: Excellent (≥90), Good (≥80), Fair (≥70), Poor (<70)

---

## Configuration Management

**Location:** `config.yml` (YAML format)

**Structure:**
```yaml
data:
  raw_dir: data/raw                    # Input data location
  processed_dir: data/processed        # Output data location
  backup_dir: data/backups            # Backup location
  logs_dir: logs                       # Log output location

analysis:
  rfm:
    snapshot_offset_days: 1            # Days after max order date
    segment_rules:
      champions: {r_score_min: 4, f_score_min: 4}
      loyal: {r_score_min: 3, f_score_min: 3}
      at_risk: {r_score_max: 2, f_score_min: 3}
      new_promising: {r_score_min: 4, f_score_max: 2}
  
  anomaly:
    contamination_rate: 0.03           # Expected % anomalies
    rolling_windows: [7, 30]           # Rolling average days
    features: [revenue, orders, return_rate]
    random_state: 42                   # Reproducibility

  ab_test:
    alpha: 0.05                        # Significance level
    correction: bonferroni             # Multiple test correction
    min_sample_size: 100               # Minimum per group

  validation:
    discount_pct_min: 0.0
    discount_pct_max: 1.0
    quantity_min: 1
    rating_min: 1
    rating_max: 5

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/analytics.log
```

**Usage:**
All analysis scripts read `config.yml` at startup. To tune parameters, edit this file before running scripts.

---

## Logging and Monitoring

**Log Files Created:**
- `logs/analytics.log` - Pipeline master log
- `logs/eda.log` - EDA execution log
- `logs/rfm_segmentation.log` - RFM analysis log
- `logs/ab_testing.log` - A/B test log
- `logs/anomaly_detection.log` - Anomaly detection log
- `logs/data_validation.log` - Data validation log
- `logs/data_profiling.log` - Profiling analysis log
- `logs/pipeline.log` - Orchestrator execution log
- `logs/pipeline_results.json` - Structured pipeline results

**Log Level:** INFO (shows progress), DEBUG (detailed diagnostics)

**Monitoring:**
- Each script logs start/end timestamps
- Exit codes indicate success (0), warnings (1), failure (2)
- JSON reports document all key metrics and decisions

---

## Error Handling & Recovery

**Data Validation Failures:**
- If data fails validation (missing files, schema errors), pipeline stops immediately (exit code 2)
- Check `data_validation_report.json` for details

**Partial Failures:**
- If one analysis script fails but others succeed, overall pipeline returns exit code 2
- Use `--continue-on-error` flag to bypass and continue (not recommended for production)
- Check individual log files for root cause

**Recovery:**
1. Review error logs: `tail logs/*.log`
2. Check data quality: `python python/00_data_validation.py`
3. Fix data or configuration issues
4. Re-run pipeline from beginning

---

## Performance Characteristics

**Dataset Size:** 15,000 orders, 2,500 customers, 610 days
**Pipeline Runtime:** ~15-20 seconds total
**Memory Usage:** ~200MB peak (during analysis)
**Disk Usage:** ~5MB for all processed data + logs

**Scalability:**
- **100K orders:** ~2 minutes (linear scaling)
- **1M orders:** ~20 minutes
- **Bottleneck:** Anomaly detection (Isolation Forest training, O(n log n))

**Optimization Opportunities:**
- Batch processing for large datasets
- Parallel processing of independent analyses
- Incremental anomaly detection
- Caching of correlation matrices

---

## Security & Privacy

**Data Handling:**
- Raw data never modified in place (read-only)
- Backups created before major transformations
- All backups in `data/backups/` for audit trail

**PII Protection:**
- Aggregation gate before LLM processing (no customer-level data to AI)
- Blocked keys in insight generator (customer_id, email, phone, address)
- Analysis works only with aggregate metrics

**Credentials:**
- No API keys or passwords hardcoded
- Environment variables for sensitive config
- `.gitignore` excludes credential files

---

## Quality Assurance

**Test Coverage:**
- Unit tests for data validation (13 tests)
- Integration tests for pipeline (end-to-end workflow)
- Data quality tests (schema, referential integrity, value ranges)

**Validation Gates:**
1. **Data Validation** - Pre-analysis checks (schema, nulls, duplicates, ranges)
2. **Logical Validation** - Post-analysis consistency checks (totals, summaries)
3. **Statistical Validation** - Test assumptions and result interpretation

**Audit Trail:**
- All processing logged with timestamps
- CSV outputs include metadata (data range, record count)
- Reproducible with fixed random seeds (e.g., Isolation Forest seed=42)

---

## Deployment

### Development Environment
- **OS:** Windows 10/11, macOS, Linux
- **Python:** 3.9+
- **Key Packages:** pandas, numpy, scipy, scikit-learn, pyyaml

### Production Environment
- **Container:** Docker (optional)
- **Scheduling:** Airflow, cron, or Windows Task Scheduler
- **Monitoring:** Log aggregation, alerting on failures
- **Data Store:** SQL Server, PostgreSQL, or S3-compatible storage

### Deployment Steps
1. `pip install -r requirements.txt`
2. Update `config.yml` for production paths/settings
3. `python python/run_pipeline.py` (orchestrates all phases)
4. Schedule as cron job or Airflow DAG for daily/weekly runs

---

## Future Enhancements

1. **Real-time Analytics:** Stream processing for live dashboards
2. **Predictive Models:** Customer churn, LTV forecasting, demand prediction
3. **Cohort Analysis:** Retention curves by acquisition cohort
4. **Funnel Analysis:** Conversion rates by stage
5. **Experimentation Framework:** Multi-armed bandits, sequential testing
6. **Data Lineage:** Automated tracking of data origins and transformations
7. **Distributed Processing:** Spark/Dask for massive datasets
8. **API Layer:** REST endpoints for programmatic access
9. **Self-Service BI:** User-friendly query builder for analysts
10. **Alert System:** Anomaly notifications, SLA monitoring

