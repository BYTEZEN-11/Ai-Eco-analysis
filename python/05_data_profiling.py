"""
Advanced Data Profiling and Statistical Analysis
=================================================

Comprehensive exploratory data analysis beyond basic aggregation:
1. Descriptive statistics (mean, median, std, quartiles, skewness, kurtosis)
2. Distribution analysis (histograms, normality tests)
3. Correlation analysis (Pearson correlations, heatmaps)
4. Outlier detection and analysis
5. Feature relationships and interactions
6. Time-series decomposition and trend analysis
7. Categorical analysis and cross-tabulation
8. Missing value patterns
9. Data quality scoring

Outputs:
- data/processed/data_profile.json: Comprehensive statistical profiles
- data/processed/correlation_matrix.csv: Feature correlations
- Console report with key insights
- Visualization ready outputs (for dashboard integration)

Exit codes:
    0 = Success
    1 = Warnings (some analyses skipped)
    2 = Failure
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from scipy import stats

# Configuration
ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config.yml"
LOG_DIR = ROOT / "logs"

# Create required directories
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "data_profiling.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}, using defaults")
        return {"data": {"raw_dir": "data/raw", "processed_dir": "data/processed"}}


def load_data(raw_dir: Path) -> pd.DataFrame:
    """Load raw orders data for profiling."""
    logger.info("Loading orders data...")

    try:
        orders_file = raw_dir / "orders.csv"
        if not orders_file.exists():
            logger.error(f"Orders file not found: {orders_file}")
            sys.exit(2)

        df = pd.read_csv(orders_file, parse_dates=["order_date"])
        logger.info(f"Loaded {len(df)} orders with {len(df.columns)} columns")

        return df

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(2)


class StatisticalProfiler:
    """Generates comprehensive statistical profiles."""

    @staticmethod
    def describe_numeric(df: pd.DataFrame) -> dict:
        """Generate detailed statistics for numeric columns."""
        logger.info("Analyzing numeric columns...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        profile = {}

        for col in numeric_cols:
            series = df[col]

            profile[col] = {
                "dtype": str(series.dtype),
                "count": int(series.count()),
                "null_count": int(series.isnull().sum()),
                "null_pct": float((series.isnull().sum() / len(df)) * 100),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "var": float(series.var()),
                "min": float(series.min()),
                "q1": float(series.quantile(0.25)),
                "q2": float(series.quantile(0.50)),
                "q3": float(series.quantile(0.75)),
                "max": float(series.max()),
                "range": float(series.max() - series.min()),
                "iqr": float(series.quantile(0.75) - series.quantile(0.25)),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "cv": float(series.std() / series.mean()) if series.mean() != 0 else 0,  # Coefficient of variation
            }

            # Outlier detection (IQR method)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            profile[col]["outlier_count"] = int(outliers)
            profile[col]["outlier_pct"] = float((outliers / len(df)) * 100)

            # Normality test (Shapiro-Wilk for n<5000, Kolmogorov-Smirnov otherwise)
            if series.count() < 5000:
                _, p_value = stats.shapiro(series.dropna())
                profile[col]["normality_test"] = "Shapiro-Wilk"
                profile[col]["normality_p_value"] = float(p_value)
                profile[col]["is_normal"] = p_value > 0.05
            else:
                _, p_value = stats.kstest(series.dropna(), "norm", args=(series.mean(), series.std()))
                profile[col]["normality_test"] = "Kolmogorov-Smirnov"
                profile[col]["normality_p_value"] = float(p_value)
                profile[col]["is_normal"] = p_value > 0.05

        logger.info(f"Analyzed {len(profile)} numeric columns")
        return profile

    @staticmethod
    def describe_categorical(df: pd.DataFrame) -> dict:
        """Generate statistics for categorical columns."""
        logger.info("Analyzing categorical columns...")

        categorical_cols = df.select_dtypes(include=["object"]).columns
        profile = {}

        for col in categorical_cols:
            series = df[col]
            value_counts = series.value_counts()

            profile[col] = {
                "dtype": str(series.dtype),
                "count": int(series.count()),
                "null_count": int(series.isnull().sum()),
                "null_pct": float((series.isnull().sum() / len(df)) * 100),
                "unique_values": int(series.nunique()),
                "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "top_value_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_value_pct": float((value_counts.iloc[0] / len(df)) * 100) if len(value_counts) > 0 else 0,
                "value_distribution": value_counts.to_dict(),
            }

        logger.info(f"Analyzed {len(profile)} categorical columns")
        return profile

    @staticmethod
    def analyze_distributions(df: pd.DataFrame) -> dict:
        """Analyze distribution characteristics."""
        logger.info("Analyzing distributions...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distributions = {}

        for col in numeric_cols:
            series = df[col].dropna()

            # Distribution shape classification
            skewness = series.skew()
            kurtosis = series.kurtosis()

            if abs(skewness) < 0.5:
                shape = "Approximately symmetric"
            elif skewness > 0.5:
                shape = "Right-skewed (positive)"
            else:
                shape = "Left-skewed (negative)"

            if kurtosis > 3:
                tail = "Heavy-tailed (leptokurtic)"
            elif kurtosis < -1:
                tail = "Light-tailed (platykurtic)"
            else:
                tail = "Normal-like tails (mesokurtic)"

            distributions[col] = {
                "shape": shape,
                "tail_behavior": tail,
                "skewness_value": float(skewness),
                "kurtosis_value": float(kurtosis),
            }

        return distributions

    @staticmethod
    def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix for numeric features."""
        logger.info("Computing correlations...")

        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            logger.warning("Insufficient numeric columns for correlation analysis")
            return pd.DataFrame()

        correlation_matrix = numeric_df.corr(method="pearson")
        logger.info(f"Correlation matrix computed: {correlation_matrix.shape}")

        return correlation_matrix

    @staticmethod
    def identify_strong_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list:
        """Identify strong correlations above threshold."""
        logger.info(f"Identifying correlations above {threshold}...")

        strong_correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        "feature_1": corr_matrix.columns[i],
                        "feature_2": corr_matrix.columns[j],
                        "correlation": float(corr_value),
                        "relationship": "Strong positive" if corr_value > 0 else "Strong negative",
                    })

        logger.info(f"Found {len(strong_correlations)} strong correlations")
        return strong_correlations

    @staticmethod
    def temporal_analysis(df: pd.DataFrame) -> dict:
        """Analyze temporal patterns if date column exists."""
        logger.info("Analyzing temporal patterns...")

        temporal = {}

        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

            # Temporal range
            temporal["min_date"] = str(df["order_date"].min())
            temporal["max_date"] = str(df["order_date"].max())
            temporal["span_days"] = int((df["order_date"].max() - df["order_date"].min()).days)
            temporal["span_years"] = float(temporal["span_days"] / 365.25)

            # Daily statistics
            daily_counts = df.groupby(df["order_date"].dt.date).size()
            temporal["daily_orders"] = {
                "mean": float(daily_counts.mean()),
                "std": float(daily_counts.std()),
                "min": int(daily_counts.min()),
                "max": int(daily_counts.max()),
            }

            # Monthly statistics
            monthly_counts = df.groupby(df["order_date"].dt.to_period("M")).size()
            temporal["monthly_orders"] = {
                "mean": float(monthly_counts.mean()),
                "std": float(monthly_counts.std()),
                "min": int(monthly_counts.min()),
                "max": int(monthly_counts.max()),
            }

            # Trend
            first_month_avg = df[df["order_date"] < df["order_date"].min() + pd.Timedelta(days=30)].shape[0]
            last_month_avg = df[df["order_date"] > df["order_date"].max() - pd.Timedelta(days=30)].shape[0]

            trend = ((last_month_avg - first_month_avg) / first_month_avg * 100) if first_month_avg > 0 else 0
            temporal["trend_pct_change"] = float(trend)

        return temporal


class DataQualityScorer:
    """Scores overall data quality."""

    @staticmethod
    def score_quality(df: pd.DataFrame, profile: dict) -> dict:
        """Compute overall data quality score."""
        logger.info("Computing data quality score...")

        score_components = {}

        # Completeness (null values)
        null_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        completeness_score = max(0, 100 - null_pct)
        score_components["completeness"] = completeness_score

        # Uniqueness (duplicates)
        duplicate_count = df.duplicated().sum()
        uniqueness_score = 100 - ((duplicate_count / len(df)) * 100)
        score_components["uniqueness"] = uniqueness_score

        # Validity (outliers)
        numeric_profile = {k: v for k, v in profile.items() if isinstance(v, dict) and "outlier_pct" in v}
        avg_outlier_pct = np.mean([v["outlier_pct"] for v in numeric_profile.values()]) if numeric_profile else 0
        validity_score = max(0, 100 - avg_outlier_pct)
        score_components["validity"] = validity_score

        # Overall score (weighted average)
        weights = {"completeness": 0.4, "uniqueness": 0.3, "validity": 0.3}
        overall_score = sum(score_components[k] * weights[k] for k in weights.keys())

        return {
            "overall_score": float(overall_score),
            "components": score_components,
            "rating": "Excellent" if overall_score >= 90 else "Good" if overall_score >= 80 else "Fair" if overall_score >= 70 else "Poor",
        }


def export_results(numeric_profile: dict, categorical_profile: dict, distributions: dict, correlations: pd.DataFrame, quality_score: dict, temporal: dict, processed_dir: Path):
    """Export profiling results."""
    logger.info("Exporting profiling results...")

    try:
        # Combined profile
        profile = {
            "timestamp": datetime.now().isoformat(),
            "numeric_columns": numeric_profile,
            "categorical_columns": categorical_profile,
            "distributions": distributions,
            "temporal_analysis": temporal,
            "data_quality": quality_score,
        }

        profile_file = processed_dir / "data_profile.json"
        with open(profile_file, "w") as f:
            json.dump(profile, f, indent=2, default=str)
        logger.info(f"Exported profile to {profile_file}")

        # Correlation matrix
        if not correlations.empty:
            corr_file = processed_dir / "correlation_matrix.csv"
            correlations.to_csv(corr_file)
            logger.info(f"Exported correlation matrix to {corr_file}")

    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        sys.exit(2)


def main():
    """Run complete data profiling pipeline."""
    logger.info("=" * 80)
    logger.info("ADVANCED DATA PROFILING AND ANALYSIS")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)

    # Load configuration
    config = load_config(CONFIG_FILE)
    raw_dir = ROOT / config["data"]["raw_dir"]
    processed_dir = ROOT / config["data"]["processed_dir"]
    processed_dir.mkdir(exist_ok=True)

    # Load data
    df = load_data(raw_dir)

    # Profile data
    profiler = StatisticalProfiler()
    numeric_profile = profiler.describe_numeric(df)
    categorical_profile = profiler.describe_categorical(df)
    distributions = profiler.analyze_distributions(df)
    correlations = profiler.compute_correlations(df)
    strong_correlations = profiler.identify_strong_correlations(correlations, threshold=0.7)
    temporal = profiler.temporal_analysis(df)

    # Quality scoring
    scorer = DataQualityScorer()
    quality_score = scorer.score_quality(df, numeric_profile)

    # Export results
    export_results(numeric_profile, categorical_profile, distributions, correlations, quality_score, temporal, processed_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("DATA PROFILING SUMMARY")
    print("=" * 80)
    print(f"\nData Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nData Quality Score: {quality_score['overall_score']:.1f}/100 ({quality_score['rating']})")
    print(f"  - Completeness: {quality_score['components']['completeness']:.1f}%")
    print(f"  - Uniqueness: {quality_score['components']['uniqueness']:.1f}%")
    print(f"  - Validity: {quality_score['components']['validity']:.1f}%")

    print(f"\nTemporal Analysis:")
    if temporal:
        print(f"  - Span: {temporal.get('span_days', 0)} days ({temporal.get('span_years', 0):.1f} years)")
        print(f"  - Trend: {temporal.get('trend_pct_change', 0):.1f}% change")
        print(f"  - Daily Orders: {temporal.get('daily_orders', {}).get('mean', 0):.0f} avg")

    print(f"\nNumeric Columns: {len(numeric_profile)}")
    print(f"Categorical Columns: {len(categorical_profile)}")
    print(f"Strong Correlations: {len(strong_correlations)}")

    if strong_correlations:
        print("\nTop Strong Correlations:")
        for corr in strong_correlations[:5]:
            print(f"  - {corr['feature_1']} <-> {corr['feature_2']}: {corr['correlation']:.3f}")

    print("\n" + "=" * 80 + "\n")

    logger.info("Data profiling pipeline completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
