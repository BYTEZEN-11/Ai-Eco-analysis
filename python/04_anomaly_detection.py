"""
Anomaly Detection in Daily KPIs
================================

Detects unusual operational patterns in daily KPI metrics using Isolation Forest.

Features analyzed:
- Daily revenue
- Daily order count
- Daily return rate

Methodology:
- Unsupervised Isolation Forest algorithm (does not assume normal distribution)
- Contamination parameter set to expected anomaly rate (~3%)
- Rolling 7-day and 30-day averages for trend context
- Binary flag (is_anomaly) for downstream analysis

Outputs:
- data/processed/daily_kpis_with_anomalies.csv: Daily metrics + anomaly flags
- Console report with anomaly summary

Exit codes:
    0 = Success
    1 = Warnings
    2 = Failure
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
from sklearn.ensemble import IsolationForest

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
        logging.FileHandler(LOG_DIR / "anomaly_detection.log"),
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
        return {
            "analysis": {
                "anomaly": {
                    "contamination_rate": 0.03,
                    "rolling_windows": [7, 30],
                    "features": ["revenue", "orders", "return_rate"],
                    "random_state": 42,
                }
            }
        }


def load_daily_kpis(processed_dir: Path) -> pd.DataFrame:
    """Load daily KPIs with error handling."""
    logger.info("Loading daily KPIs...")
    
    try:
        kpis_file = processed_dir / "daily_kpis.csv"
        if not kpis_file.exists():
            logger.error(f"Daily KPIs file not found: {kpis_file}")
            logger.info("Run 01_eda.py first to generate daily_kpis.csv")
            sys.exit(2)
        
        df = pd.read_csv(kpis_file, parse_dates=['order_date'])
        logger.info(f"Loaded {len(df)} daily records")
        
        # Validate required columns
        required_cols = ['order_date', 'revenue', 'orders', 'return_rate']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            sys.exit(2)
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to load daily KPIs: {e}")
        sys.exit(2)


def validate_data(df: pd.DataFrame) -> bool:
    """Validate data for anomaly detection."""
    logger.info("Validating data...")
    
    # Check for null values
    null_counts = df[['revenue', 'orders', 'return_rate']].isnull().sum()
    if null_counts.sum() > 0:
        logger.error(f"Found null values: {null_counts}")
        return False
    
    # Check for negative values
    if (df['revenue'] < 0).any():
        logger.error("Found negative revenue values")
        return False
    
    if (df['orders'] < 0).any():
        logger.error("Found negative order counts")
        return False
    
    # Check for invalid return rates
    if ((df['return_rate'] < 0) | (df['return_rate'] > 1)).any():
        logger.error("Found invalid return_rate values (should be 0-1)")
        return False
    
    logger.info("Data validation passed")
    return True


def compute_rolling_averages(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Compute rolling averages for trend context."""
    logger.info(f"Computing rolling averages with windows: {windows}")
    
    for window in windows:
        col_name = f'rolling_{window}d_revenue'
        df[col_name] = df['revenue'].rolling(
            window=window,
            min_periods=1,  # Allow partial windows at start
            center=False
        ).mean()
        
        logger.debug(f"  Computed {col_name}")
    
    return df


def detect_anomalies(df: pd.DataFrame, features: list, contamination: float, random_state: int) -> pd.DataFrame:
    """Apply Isolation Forest to detect anomalies."""
    logger.info(f"Detecting anomalies with features: {features}")
    logger.info(f"  Contamination rate: {contamination} ({contamination*100:.1f}%)")
    logger.info(f"  Random state: {random_state}")
    
    # Prepare feature matrix
    X = df[features].values
    
    # Normalize features for fair comparison (Isolation Forest can be scale-sensitive)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.debug(f"Feature matrix shape: {X_scaled.shape}")
    logger.debug(f"Feature statistics:\n{pd.DataFrame(X_scaled, columns=features).describe()}")
    
    # Apply Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
        max_samples='auto',
        verbose=0
    )
    
    # Get predictions (-1 = anomaly, 1 = normal)
    predictions = model.fit_predict(X_scaled)
    
    # Get anomaly scores (lower = more anomalous)
    anomaly_scores = model.score_samples(X_scaled)
    
    # Add to dataframe
    df['anomaly_flag'] = predictions
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly'] = (predictions == -1).astype(int)
    
    # Log statistics
    n_anomalies = (df['is_anomaly'] == 1).sum()
    pct_anomalies = (n_anomalies / len(df)) * 100
    
    logger.info(f"Detected {n_anomalies} anomalies ({pct_anomalies:.2f}%)")
    
    return df


def analyze_anomalies(df: pd.DataFrame) -> dict:
    """Analyze detected anomalies."""
    logger.info("Analyzing anomalies...")
    
    anomalies = df[df['is_anomaly'] == 1]
    
    if len(anomalies) == 0:
        logger.info("No anomalies detected")
        return {}
    
    analysis = {
        'total_anomalies': len(anomalies),
        'date_range': f"{anomalies['order_date'].min().date()} to {anomalies['order_date'].max().date()}",
        'revenue_anomalies': {
            'mean': round(anomalies['revenue'].mean(), 2),
            'min': round(anomalies['revenue'].min(), 2),
            'max': round(anomalies['revenue'].max(), 2),
            'std': round(anomalies['revenue'].std(), 2),
        },
        'order_anomalies': {
            'mean': round(anomalies['orders'].mean(), 2),
            'min': int(anomalies['orders'].min()),
            'max': int(anomalies['orders'].max()),
        },
        'return_rate_anomalies': {
            'mean': round(anomalies['return_rate'].mean(), 4),
            'min': round(anomalies['return_rate'].min(), 4),
            'max': round(anomalies['return_rate'].max(), 4),
        },
    }
    
    logger.info(f"Anomaly summary:")
    logger.info(f"  Revenue: ${analysis['revenue_anomalies']['mean']:.2f} avg")
    logger.info(f"  Orders: {analysis['order_anomalies']['mean']:.0f} avg")
    logger.info(f"  Return Rate: {analysis['return_rate_anomalies']['mean']:.2%} avg")
    
    return analysis


def export_results(df: pd.DataFrame, processed_dir: Path):
    """Export results to CSV."""
    logger.info("Exporting results...")
    
    try:
        output_file = processed_dir / "daily_kpis_with_anomalies.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Exported results to {output_file}")
    
    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        sys.exit(2)


def main():
    """Run complete anomaly detection pipeline."""
    logger.info("=" * 80)
    logger.info("ANOMALY DETECTION PIPELINE")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    processed_dir = ROOT / config['data']['processed_dir']
    processed_dir.mkdir(exist_ok=True)
    
    anomaly_config = config['analysis']['anomaly']
    
    # Load and validate data
    df = load_daily_kpis(processed_dir)
    if not validate_data(df):
        sys.exit(2)
    
    # Compute rolling averages
    df = compute_rolling_averages(df, anomaly_config['rolling_windows'])
    
    # Detect anomalies
    df = detect_anomalies(
        df,
        features=anomaly_config['features'],
        contamination=anomaly_config['contamination_rate'],
        random_state=anomaly_config['random_state']
    )
    
    # Analyze results
    analysis = analyze_anomalies(df)
    
    # Export
    export_results(df, processed_dir)
    
    # Print summary
    if analysis:
        print("\n" + "=" * 80)
        print("ANOMALY DETECTION SUMMARY")
        print("=" * 80)
        print(f"Total Anomalies: {analysis['total_anomalies']}")
        print(f"Date Range: {analysis['date_range']}")
        print(f"Revenue (anomalies): ${analysis['revenue_anomalies']['mean']:.2f} avg")
        print(f"Orders (anomalies): {analysis['order_anomalies']['mean']:.0f} avg")
        print(f"Return Rate (anomalies): {analysis['return_rate_anomalies']['mean']:.2%} avg")
        print("=" * 80 + "\n")
    
    logger.info("Anomaly detection pipeline completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
