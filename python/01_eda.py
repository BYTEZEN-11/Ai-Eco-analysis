"""
Exploratory Data Analysis (EDA)
================================

This module performs initial exploratory analysis on the e-commerce dataset:
1. Loads and validates raw data
2. Merges orders with customer information
3. Generates aggregate KPIs
4. Computes daily and monthly metrics
5. Exports processed data for downstream analysis

Outputs:
- data/processed/monthly_kpis.csv: Aggregated metrics by month
- data/processed/daily_kpis.csv: Aggregated metrics by day
- Console report with key statistics

Exit codes:
    0 = Success
    1 = Warnings (data issues but recoverable)
    2 = Failure (data missing or invalid)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yaml

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
        logging.FileHandler(LOG_DIR / "eda.log"),
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


def load_data(raw_dir: Path) -> tuple:
    """Load raw data files with error handling."""
    logger.info("Loading raw data files...")
    
    try:
        orders_file = raw_dir / "orders.csv"
        customers_file = raw_dir / "customers.csv"
        
        if not orders_file.exists():
            logger.error(f"Orders file not found: {orders_file}")
            sys.exit(2)
        if not customers_file.exists():
            logger.error(f"Customers file not found: {customers_file}")
            sys.exit(2)
        
        orders = pd.read_csv(orders_file, parse_dates=['order_date'])
        customers = pd.read_csv(customers_file)
        
        logger.info(f"Loaded orders: {len(orders)} rows, {len(orders.columns)} columns")
        logger.info(f"Loaded customers: {len(customers)} rows, {len(customers.columns)} columns")
        
        return orders, customers
    
    except pd.errors.ParserError as e:
        logger.error(f"Failed to parse data file: {e}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        sys.exit(2)


def validate_data(orders: pd.DataFrame, customers: pd.DataFrame) -> bool:
    """Validate loaded data integrity."""
    logger.info("Validating data...")
    
    # Check required columns
    required_orders_cols = ['order_id', 'order_date', 'customer_id', 'net_revenue', 'returned']
    required_customers_cols = ['customer_id']
    
    missing_orders = set(required_orders_cols) - set(orders.columns)
    missing_customers = set(required_customers_cols) - set(customers.columns)
    
    if missing_orders or missing_customers:
        logger.error(f"Missing required columns: orders={missing_orders}, customers={missing_customers}")
        return False
    
    # Check for null values
    null_orders = orders[required_orders_cols].isnull().sum().sum()
    null_customers = customers[required_customers_cols].isnull().sum().sum()
    
    if null_orders > 0:
        logger.warning(f"Found {null_orders} null values in required orders columns")
    if null_customers > 0:
        logger.error(f"Found {null_customers} null values in customer_id")
        return False
    
    # Check for negative revenue
    negative_revenue = (orders['net_revenue'] < 0).sum()
    if negative_revenue > 0:
        logger.warning(f"Found {negative_revenue} orders with negative revenue")
    
    # Check referential integrity
    orphaned = ~orders['customer_id'].isin(customers['customer_id'])
    if orphaned.any():
        orphan_count = orphaned.sum()
        logger.warning(f"Found {orphan_count} orders from non-existent customers ({orphan_count/len(orders)*100:.2f}%)")
        logger.info("These orders will be retained in merge (left join)")
    
    logger.info("Data validation completed")
    return True


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute executive summary KPIs."""
    logger.info("Computing KPIs...")
    
    kpis = {
        'Total Rows': len(df),
        'Total Revenue': round(df['net_revenue'].sum(), 2),
        'Total Orders': df['order_id'].nunique(),
        'Total Customers': df['customer_id'].nunique(),
        'Return Rate': round(df['returned'].mean(), 4),
        'Average Order Value': round(df['net_revenue'].sum() / df['order_id'].nunique(), 2),
        'Date Range': f"{df['order_date'].min().date()} to {df['order_date'].max().date()}",
    }
    
    return kpis


def generate_monthly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Generate monthly aggregate KPIs."""
    logger.info("Generating monthly KPIs...")
    
    monthly = (df
        .assign(month=df['order_date'].dt.to_period('M').astype(str))
        .groupby('month', as_index=False)
        .agg({
            'net_revenue': 'sum',
            'order_id': 'nunique',
            'customer_id': 'nunique',
            'returned': ['sum', 'mean'],  # count and rate
        }))
    
    # Flatten multi-level columns
    monthly.columns = ['month', 'revenue', 'orders', 'customers', 'returns_count', 'return_rate']
    monthly['aov'] = monthly['revenue'] / monthly['orders']
    monthly['aov'] = monthly['aov'].round(2)
    monthly['return_rate'] = monthly['return_rate'].round(4)
    
    logger.info(f"Generated {len(monthly)} monthly records")
    return monthly


def generate_daily_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Generate daily aggregate KPIs for anomaly detection."""
    logger.info("Generating daily KPIs...")
    
    daily = (df
        .groupby(df['order_date'].dt.date, as_index=False)
        .agg({
            'net_revenue': 'sum',
            'order_id': 'nunique',
            'customer_id': 'nunique',
            'returned': ['sum', 'mean'],
            'shipping_days': 'mean',
            'rating': 'mean',
        }))
    
    # Flatten multi-level columns
    daily.columns = ['order_date', 'revenue', 'orders', 'customers', 'returns_count', 'return_rate', 'avg_shipping_days', 'avg_rating']
    daily['order_date'] = pd.to_datetime(daily['order_date'])
    daily['return_rate'] = daily['return_rate'].round(4)
    daily['avg_shipping_days'] = daily['avg_shipping_days'].round(2)
    daily['avg_rating'] = daily['avg_rating'].round(2)
    
    logger.info(f"Generated {len(daily)} daily records")
    return daily


def export_data(monthly: pd.DataFrame, daily: pd.DataFrame, processed_dir: Path):
    """Export processed data to CSV files."""
    logger.info("Exporting processed data...")
    
    try:
        monthly_file = processed_dir / "monthly_kpis.csv"
        daily_file = processed_dir / "daily_kpis.csv"
        
        monthly.to_csv(monthly_file, index=False)
        daily.to_csv(daily_file, index=False)
        
        logger.info(f"Exported monthly KPIs to {monthly_file}")
        logger.info(f"Exported daily KPIs to {daily_file}")
    
    except Exception as e:
        logger.error(f"Failed to export data: {e}")
        sys.exit(2)


def print_report(kpis: dict):
    """Print summary report to console."""
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 80)
    for key, value in kpis.items():
        print(f"{key:.<50} {value}")
    print("=" * 80 + "\n")


def main():
    """Run complete EDA pipeline."""
    logger.info("=" * 80)
    logger.info("EXPLORATORY DATA ANALYSIS PIPELINE")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    raw_dir = ROOT / config['data']['raw_dir']
    processed_dir = ROOT / config['data']['processed_dir']
    processed_dir.mkdir(exist_ok=True)
    
    # Load and validate data
    orders, customers = load_data(raw_dir)
    if not validate_data(orders, customers):
        logger.error("Data validation failed")
        sys.exit(2)
    
    # Merge datasets
    logger.info("Merging orders with customer information...")
    df = orders.merge(customers, on='customer_id', how='left')
    logger.info(f"Merged dataset: {len(df)} rows")
    
    # Compute KPIs
    kpis = compute_kpis(df)
    print_report(kpis)
    
    # Generate and export aggregates
    monthly = generate_monthly_kpis(df)
    daily = generate_daily_kpis(df)
    export_data(monthly, daily, processed_dir)
    
    logger.info("EDA pipeline completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
