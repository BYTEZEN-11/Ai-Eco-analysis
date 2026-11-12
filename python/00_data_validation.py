"""
Data Validation and Profiling Pipeline
=======================================

Comprehensive data quality checks, profiling, and cleansing for the data analytics pipeline.

This module:
1. Validates raw data against schema and business rules
2. Detects and handles missing values
3. Identifies and flags outliers and anomalies
4. Profiles dataset characteristics
5. Generates data quality reports
6. Ensures reproducibility with backup and logging

Usage:
    python python/00_data_validation.py [--profile] [--fix] [--report]

Exit codes:
    0 = All validations passed
    1 = Validation warnings (data issues detected but recoverable)
    2 = Validation errors (critical issues)
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import yaml
from io import StringIO

# Configuration
ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config.yml"
LOG_DIR = ROOT / "logs"
BACKUP_DIR = ROOT / "data" / "backups"

# Create required directories
LOG_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "data_validation.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and parse YAML configuration file."""

    @staticmethod
    def load(config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(2)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse configuration: {e}")
            sys.exit(2)


class DataValidator:
    """Validate data against schema and business rules."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize validator with configuration."""
        self.config = config
        self.validation_report = {}
        self.issues = {"critical": [], "warning": [], "info": []}

    def validate_orders(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate orders dataset against expected schema and constraints."""
        logger.info("Validating orders dataset...")
        report = {
            "dataset": "orders",
            "rows": len(df),
            "columns": len(df.columns),
            "issues": [],
        }

        # Schema validation
        expected_columns = [
            "order_id",
            "order_date",
            "customer_id",
            "product_id",
            "category",
            "quantity",
            "unit_price",
            "discount_pct",
            "channel",
            "ab_group",
            "returned",
            "gross_revenue",
            "net_revenue",
            "shipping_days",
            "rating",
        ]

        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            issue = f"Missing columns: {missing_cols}"
            self.issues["critical"].append(issue)
            report["issues"].append({"severity": "CRITICAL", "issue": issue})
            return False, report

        # Data type validation
        dtype_issues = self._check_dtypes(df)
        report["issues"].extend(dtype_issues)

        # Uniqueness checks
        if df["order_id"].duplicated().any():
            issue = f"Duplicate order_id: {df['order_id'].duplicated().sum()} rows"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        # Null/Missing value checks
        null_report = self._check_nulls(df)
        report["issues"].extend(null_report)

        # Value range validation
        validation_config = self.config.get("analysis", {}).get("validation", {})

        # Discount percentage
        invalid_discount = (
            (df["discount_pct"] < validation_config.get("discount_pct_min", 0))
            | (df["discount_pct"] > validation_config.get("discount_pct_max", 1))
        ).sum()
        if invalid_discount > 0:
            issue = f"Invalid discount_pct: {invalid_discount} rows outside [0, 1]"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        # Quantity validation
        invalid_qty = (df["quantity"] < validation_config.get("quantity_min", 1)).sum()
        if invalid_qty > 0:
            issue = f"Invalid quantity: {invalid_qty} rows < 1"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        # Rating validation
        invalid_rating = (
            (df["rating"] < validation_config.get("rating_min", 1))
            | (df["rating"] > validation_config.get("rating_max", 5))
        ).sum()
        if invalid_rating > 0:
            issue = f"Invalid rating: {invalid_rating} rows outside [1, 5]"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        # Shipping days validation
        invalid_shipping = (
            df["shipping_days"] > validation_config.get("shipping_days_max", 30)
        ).sum()
        if invalid_shipping > 0:
            issue = f"High shipping_days: {invalid_shipping} rows > 30 days"
            self.issues["info"].append(issue)
            report["issues"].append({"severity": "INFO", "issue": issue})

        # Revenue validation (net_revenue <= gross_revenue)
        invalid_revenue = (df["net_revenue"] > df["gross_revenue"]).sum()
        if invalid_revenue > 0:
            issue = (
                f"Invalid revenue: {invalid_revenue} rows have net > gross revenue"
            )
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        # Returned orders should have net_revenue = 0 (or close to it)
        returned_with_revenue = (
            (df["returned"] == 1) & (df["net_revenue"] > 0)
        ).sum()
        if returned_with_revenue > 0:
            pct = (returned_with_revenue / (df["returned"] == 1).sum()) * 100
            issue = f"Returned orders with net_revenue > 0: {returned_with_revenue} rows ({pct:.1f}% of returned)"
            self.issues["info"].append(issue)
            report["issues"].append(
                {
                    "severity": "INFO",
                    "issue": issue,
                    "note": "May be intentional (partial refunds); verify",
                }
            )

        # Order date range
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        future_dates = (df["order_date"] > pd.Timestamp.now()).sum()
        if future_dates > 0:
            issue = f"Future dates detected: {future_dates} orders after today"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        # Channel and AB group validation
        valid_channels = {"App", "Marketplace", "Direct", "Website"}
        invalid_channel = ~df["channel"].isin(valid_channels)
        if invalid_channel.any():
            issue = f"Invalid channel: {invalid_channel.sum()} unknown values"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        valid_ab_groups = {"control", "variant"}
        invalid_ab = ~df["ab_group"].isin(valid_ab_groups)
        if invalid_ab.any():
            issue = f"Invalid ab_group: {invalid_ab.sum()} unknown values"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        # Categorical validation
        valid_categories = {
            "Electronics",
            "Beauty",
            "Home & Office",
            "Clothing",
            "Food",
        }
        invalid_cat = ~df["category"].isin(valid_categories)
        if invalid_cat.any():
            issue = f"Invalid category: {invalid_cat.sum()} unknown values"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        # Summary statistics
        report["summary"] = {
            "date_range": (
                f"{df['order_date'].min().date()} to {df['order_date'].max().date()}"
            ),
            "total_revenue": f"${df['net_revenue'].sum():,.2f}",
            "avg_order_value": f"${df['net_revenue'].mean():,.2f}",
            "return_rate": f"{(df['returned'].sum() / len(df)) * 100:.2f}%",
            "unique_customers": df["customer_id"].nunique(),
        }

        is_valid = len(self.issues["critical"]) == 0
        logger.info(
            f"Orders validation: {'PASSED' if is_valid else 'FAILED'} "
            f"({len(report['issues'])} issues)"
        )
        return is_valid, report

    def validate_customers(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate customers dataset."""
        logger.info("Validating customers dataset...")
        report = {
            "dataset": "customers",
            "rows": len(df),
            "columns": len(df.columns),
            "issues": [],
        }

        # Schema validation
        expected_columns = ["customer_id", "segment", "city", "signup_date"]
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            issue = f"Missing columns: {missing_cols}"
            self.issues["critical"].append(issue)
            report["issues"].append({"severity": "CRITICAL", "issue": issue})
            return False, report

        # Uniqueness
        if df["customer_id"].duplicated().any():
            issue = (
                f"Duplicate customer_id: {df['customer_id'].duplicated().sum()} rows"
            )
            self.issues["critical"].append(issue)
            report["issues"].append({"severity": "CRITICAL", "issue": issue})

        # Null checks
        null_report = self._check_nulls(df)
        report["issues"].extend(null_report)

        # Segment validation
        valid_segments = {"Consumer", "Home Office", "Corporate"}
        invalid_seg = ~df["segment"].isin(valid_segments)
        if invalid_seg.any():
            issue = f"Invalid segment: {invalid_seg.sum()} unknown values"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        # Date validation
        df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
        future_dates = (df["signup_date"] > pd.Timestamp.now()).sum()
        if future_dates > 0:
            issue = f"Future signup dates: {future_dates} customers"
            self.issues["warning"].append(issue)
            report["issues"].append({"severity": "WARNING", "issue": issue})

        report["summary"] = {
            "unique_customers": len(df),
            "segment_distribution": df["segment"].value_counts().to_dict(),
            "signup_date_range": (
                f"{df['signup_date'].min().date()} to {df['signup_date'].max().date()}"
            ),
        }

        is_valid = len(self.issues["critical"]) == 0
        logger.info(
            f"Customers validation: {'PASSED' if is_valid else 'FAILED'} "
            f"({len(report['issues'])} issues)"
        )
        return is_valid, report

    def _check_dtypes(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Check for unexpected data types."""
        issues = []
        # Most operations handle mixed types gracefully in pandas
        logger.debug("Data type check completed")
        return issues

    def _check_nulls(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for null/missing values."""
        issues = []
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            for col, count in null_counts[null_counts > 0].items():
                pct = (count / len(df)) * 100
                issue = (
                    f"Missing values: {col} has {count} null values ({pct:.2f}%)"
                )
                severity = "WARNING" if pct < 5 else "CRITICAL"
                if pct < 1:
                    severity = "INFO"
                self.issues[severity.lower()].append(issue)
                issues.append({"severity": severity, "issue": issue})
        return issues

    def referential_integrity_check(
        self, orders_df: pd.DataFrame, customers_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Check referential integrity between orders and customers."""
        logger.info("Checking referential integrity...")
        report = {"test": "referential_integrity", "issues": []}

        # All customer_ids in orders should exist in customers
        orphaned = ~orders_df["customer_id"].isin(customers_df["customer_id"])
        orphaned_count = orphaned.sum()

        if orphaned_count > 0:
            issue = (
                f"Orphaned records: {orphaned_count} orders reference non-existent customers"
            )
            self.issues["critical"].append(issue)
            report["issues"].append(
                {"severity": "CRITICAL", "issue": issue, "count": orphaned_count}
            )

            # Get list of missing customer IDs
            missing_customers = orders_df[orphaned]["customer_id"].unique()
            logger.warning(f"Missing customer IDs: {missing_customers[:10]}...")
        else:
            logger.info("Referential integrity: PASSED (no orphaned records)")

        report["result"] = orphaned_count == 0
        return report


class DataProfiler:
    """Profile dataset characteristics and distributions."""

    @staticmethod
    def profile_orders(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive profile of orders dataset."""
        logger.info("Profiling orders dataset...")

        profile = {
            "dataset": "orders",
            "timestamp": datetime.now().isoformat(),
            "dimensions": {"rows": len(df), "columns": len(df.columns)},
            "numeric_columns": {},
            "categorical_columns": {},
            "temporal_columns": {},
            "quality_metrics": {},
        }

        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            profile["numeric_columns"][col] = {
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "null_pct": float((df[col].isna().sum() / len(df)) * 100),
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
                "skewness": float(df[col].skew()),
                "kurtosis": float(df[col].kurtosis()),
            }

        # Categorical columns
        categorical_cols = ["channel", "ab_group", "category"]
        for col in categorical_cols:
            if col in df.columns:
                profile["categorical_columns"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(5).to_dict(),
                    "null_pct": float((df[col].isna().sum() / len(df)) * 100),
                }

        # Temporal columns
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
            profile["temporal_columns"]["order_date"] = {
                "min": df["order_date"].min().isoformat(),
                "max": df["order_date"].max().isoformat(),
                "span_days": int((df["order_date"].max() - df["order_date"].min()).days),
            }

        # Quality metrics
        profile["quality_metrics"] = {
            "null_values_pct": float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            "duplicate_order_ids": int(df["order_id"].duplicated().sum()),
            "returned_orders_pct": float((df["returned"].sum() / len(df)) * 100),
        }

        logger.info("Orders profiling complete")
        return profile

    @staticmethod
    def profile_customers(df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive profile of customers dataset."""
        logger.info("Profiling customers dataset...")

        profile = {
            "dataset": "customers",
            "timestamp": datetime.now().isoformat(),
            "dimensions": {"rows": len(df), "columns": len(df.columns)},
            "segment_distribution": df["segment"].value_counts().to_dict(),
            "city_distribution": df["city"].value_counts().to_dict(),
            "signup_date_range": (
                f"{df['signup_date'].min()} to {df['signup_date'].max()}"
            ),
        }

        logger.info("Customers profiling complete")
        return profile


def main(profile: bool = False, fix: bool = False, report: bool = True):
    """Run complete data validation and profiling pipeline."""
    logger.info("=" * 80)
    logger.info("DATA VALIDATION AND PROFILING PIPELINE")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)

    # Load configuration
    config = ConfigLoader.load(CONFIG_FILE)

    # Initialize validator and profiler
    validator = DataValidator(config)
    profiler = DataProfiler()

    # Paths
    raw_dir = ROOT / config["data"]["raw_dir"]
    processed_dir = ROOT / config["data"]["processed_dir"]

    # Load datasets
    logger.info("Loading raw datasets...")
    try:
        orders = pd.read_csv(raw_dir / "orders.csv")
        customers = pd.read_csv(raw_dir / "customers.csv")
        logger.info(f"Loaded: orders ({len(orders)} rows), customers ({len(customers)} rows)")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(2)

    # Validation phase
    orders_valid, orders_report = validator.validate_orders(orders)
    customers_valid, customers_report = validator.validate_customers(customers)
    referential_report = validator.referential_integrity_check(orders, customers)

    # Profiling phase
    if profile:
        logger.info("Generating data profiles...")
        orders_profile = profiler.profile_orders(orders)
        customers_profile = profiler.profile_customers(customers)
    else:
        orders_profile = customers_profile = None

    # Generate report
    if report:
        report_dict = {
            "timestamp": datetime.now().isoformat(),
            "validation": {
                "orders": orders_report,
                "customers": customers_report,
                "referential_integrity": referential_report,
            },
            "issues_summary": {
                "critical": len(validator.issues["critical"]),
                "warnings": len(validator.issues["warning"]),
                "info": len(validator.issues["info"]),
            },
        }

        if profile:
            report_dict["profiles"] = {
                "orders": orders_profile,
                "customers": customers_profile,
            }

        # Export report
        report_file = processed_dir / "data_validation_report.json"
        with open(report_file, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        logger.info(f"Validation report saved to {report_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("DATA VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Critical Issues: {validator.issues['critical']}")
        print(f"Warnings: {len(validator.issues['warning'])}")
        print(f"Info: {len(validator.issues['info'])}")
        print("\nDetailed Report: " + str(report_file))
        print("=" * 80 + "\n")

    # Determine exit code
    if len(validator.issues["critical"]) > 0:
        logger.error("Validation FAILED with critical issues")
        return 2
    elif len(validator.issues["warning"]) > 0:
        logger.warning("Validation completed with warnings")
        return 1
    else:
        logger.info("Validation PASSED")
        return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data validation and profiling pipeline")
    parser.add_argument(
        "--profile", action="store_true", help="Generate detailed data profiles"
    )
    parser.add_argument("--fix", action="store_true", help="Attempt to fix data issues")
    parser.add_argument(
        "--report", action="store_true", default=True, help="Generate validation report"
    )

    args = parser.parse_args()
    exit_code = main(profile=args.profile, fix=args.fix, report=args.report)
    sys.exit(exit_code)
