"""
RFM (Recency, Frequency, Monetary) Customer Segmentation
=========================================================

Segments customers into actionable groups based on:
- Recency: How recently they made a purchase (lower is better)
- Frequency: How many purchases they made (higher is better)
- Monetary: Total spending (higher is better)

Each metric is scored 1-5 (quintiles), then segmentation rules are applied.

Segment Definitions:
- Champions: Recent, frequent, high-value customers (R≥4, F≥4)
- Loyal: Consistent, regular customers (R≥3, F≥3)
- At Risk: Were frequent/valuable but haven't recently purchased (R≤2, F≥3)
- New/Promising: Recent but not yet frequent (R≥4, F≤2)
- Regular: All other customers

Outputs:
- data/processed/rfm_segments.csv: Full RFM scores and segment assignments

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
        logging.FileHandler(LOG_DIR / "rfm_segmentation.log"),
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
        return {"analysis": {"rfm": {"snapshot_offset_days": 1}}}


def load_orders(raw_dir: Path) -> pd.DataFrame:
    """Load orders data with error handling."""
    logger.info("Loading orders data...")
    
    try:
        orders_file = raw_dir / "orders.csv"
        if not orders_file.exists():
            logger.error(f"Orders file not found: {orders_file}")
            sys.exit(2)
        
        df = pd.read_csv(orders_file, parse_dates=['order_date'])
        logger.info(f"Loaded {len(df)} orders")
        
        # Validate required columns
        required_cols = ['customer_id', 'order_id', 'order_date', 'net_revenue']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            sys.exit(2)
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.sum() > 0:
            logger.error(f"Found null values in required columns: {null_counts}")
            sys.exit(2)
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to load orders: {e}")
        sys.exit(2)


def compute_rfm(df: pd.DataFrame, snapshot_offset_days: int = 1) -> pd.DataFrame:
    """Compute RFM metrics for each customer."""
    logger.info("Computing RFM metrics...")
    
    # Define snapshot date (one day after max order date)
    snapshot_date = df['order_date'].max() + pd.Timedelta(days=snapshot_offset_days)
    logger.info(f"Snapshot date: {snapshot_date.date()}")
    
    # Compute RFM
    rfm = df.groupby('customer_id').agg({
        'order_date': lambda x: (snapshot_date - x.max()).days,  # Recency
        'order_id': 'nunique',  # Frequency
        'net_revenue': 'sum',  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    logger.info(f"Computed RFM for {len(rfm)} customers")
    logger.info(f"  Recency: {rfm['recency'].min()}-{rfm['recency'].max()} days")
    logger.info(f"  Frequency: {rfm['frequency'].min()}-{rfm['frequency'].max()} orders")
    logger.info(f"  Monetary: ${rfm['monetary'].min():.2f}-${rfm['monetary'].max():.2f}")
    
    return rfm


def compute_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Compute RFM quintile scores (1-5 scale)."""
    logger.info("Computing RFM scores (quintiles)...")
    
    # Recency: Lower is better (recent = higher score)
    # Using rank with 'first' method handles ties consistently
    rfm['r_score'] = pd.qcut(
        rfm['recency'].rank(method='first'),
        q=5,
        labels=[5, 4, 3, 2, 1],  # Invert so lower recency = higher score
        duplicates='drop'
    ).astype(int)
    
    # Frequency: Higher is better
    rfm['f_score'] = pd.qcut(
        rfm['frequency'].rank(method='first'),
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(int)
    
    # Monetary: Higher is better
    rfm['m_score'] = pd.qcut(
        rfm['monetary'].rank(method='first'),
        q=5,
        labels=[1, 2, 3, 4, 5],
        duplicates='drop'
    ).astype(int)
    
    logger.info("RFM scores computed")
    return rfm


def assign_segment(row: pd.Series, rules: dict) -> str:
    """Assign segment based on RFM scores and business rules."""
    r = row['r_score']
    f = row['f_score']
    m = row['m_score']
    
    # Champions: Recent, frequent, high-value
    if r >= rules['champions']['r_score_min'] and f >= rules['champions']['f_score_min']:
        return 'Champions'
    
    # Loyal: Regular customers, good engagement
    if r >= rules['loyal']['r_score_min'] and f >= rules['loyal']['f_score_min']:
        return 'Loyal'
    
    # At Risk: Were engaged but haven't purchased recently
    if r <= rules['at_risk']['r_score_max'] and f >= rules['at_risk']['f_score_min']:
        return 'At Risk'
    
    # New/Promising: Recent but not yet frequent
    if r >= rules['new_promising']['r_score_min'] and f <= rules['new_promising']['f_score_max']:
        return 'New / Promising'
    
    # Regular: Everything else (catch-all)
    return 'Regular'


def segment_customers(rfm: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply segmentation rules to assign customers to segments."""
    logger.info("Segmenting customers...")
    
    rules = config['analysis']['rfm']['segment_rules']
    
    rfm['segment'] = rfm.apply(
        lambda row: assign_segment(row, rules),
        axis=1
    )
    
    # Log segment distribution
    segment_dist = rfm['segment'].value_counts().to_dict()
    logger.info("Segment distribution:")
    for segment, count in segment_dist.items():
        pct = (count / len(rfm)) * 100
        logger.info(f"  {segment:.<30} {count:>5} ({pct:>5.1f}%)")
    
    return rfm


def export_segments(rfm: pd.DataFrame, processed_dir: Path):
    """Export RFM segments to CSV."""
    logger.info("Exporting RFM segments...")
    
    try:
        output_file = processed_dir / "rfm_segments.csv"
        rfm.to_csv(output_file, index=False)
        logger.info(f"Exported RFM segments to {output_file}")
    
    except Exception as e:
        logger.error(f"Failed to export segments: {e}")
        sys.exit(2)


def main():
    """Run complete RFM segmentation pipeline."""
    logger.info("=" * 80)
    logger.info("RFM CUSTOMER SEGMENTATION PIPELINE")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    raw_dir = ROOT / config['data']['raw_dir']
    processed_dir = ROOT / config['data']['processed_dir']
    processed_dir.mkdir(exist_ok=True)
    
    snapshot_offset = config['analysis']['rfm']['snapshot_offset_days']
    
    # Load orders
    orders = load_orders(raw_dir)
    
    # Compute RFM
    rfm = compute_rfm(orders, snapshot_offset)
    
    # Score each dimension
    rfm = compute_scores(rfm)
    
    # Assign segments
    rfm = segment_customers(rfm, config)
    
    # Export
    export_segments(rfm, processed_dir)
    
    logger.info("RFM segmentation pipeline completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
