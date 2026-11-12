"""
Unit Tests for Data Validation Module
======================================

Tests for python/00_data_validation.py functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "python"))

# Import the modules
import importlib.util
spec = importlib.util.spec_from_file_location("data_validation", parent_dir / "python" / "00_data_validation.py")
data_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_validation)

DataValidator = data_validation.DataValidator
DataProfiler = data_validation.DataProfiler
ConfigLoader = data_validation.ConfigLoader


class TestDataValidator:
    """Test data validation functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "analysis": {
                "validation": {
                    "discount_pct_min": 0.0,
                    "discount_pct_max": 1.0,
                    "quantity_min": 1,
                    "rating_min": 1,
                    "rating_max": 5,
                    "shipping_days_max": 30,
                }
            }
        }

    @pytest.fixture
    def sample_orders(self):
        """Create sample orders dataframe."""
        return pd.DataFrame({
            "order_id": ["O001", "O002", "O003"],
            "order_date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "customer_id": ["C001", "C002", "C003"],
            "product_id": ["P001", "P002", "P003"],
            "category": ["Electronics", "Beauty", "Home & Office"],
            "quantity": [1, 2, 1],
            "unit_price": [1000, 500, 2000],
            "discount_pct": [0.1, 0.05, 0.2],
            "channel": ["App", "Web", "App"],
            "ab_group": ["control", "variant", "control"],
            "returned": [0, 0, 1],
            "gross_revenue": [1000, 1000, 2000],
            "net_revenue": [900.0, 950.0, 0.0],
            "shipping_days": [2, 3, 5],
            "rating": [5, 4, 3],
        })

    @pytest.fixture
    def sample_customers(self):
        """Create sample customers dataframe."""
        return pd.DataFrame({
            "customer_id": ["C001", "C002", "C003"],
            "segment": ["Consumer", "Home Office", "Corporate"],
            "city": ["Mumbai", "Bangalore", "Delhi"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        })

    def test_validate_orders_valid(self, sample_config, sample_orders):
        """Test validation of valid orders data."""
        validator = DataValidator(sample_config)
        is_valid, report = validator.validate_orders(sample_orders)

        assert is_valid is True
        assert report["dataset"] == "orders"
        assert report["rows"] == 3

    def test_validate_orders_missing_columns(self, sample_config, sample_orders):
        """Test validation with missing columns."""
        sample_orders = sample_orders.drop("order_id", axis=1)
        validator = DataValidator(sample_config)
        is_valid, report = validator.validate_orders(sample_orders)

        assert is_valid is False
        assert len(validator.issues["critical"]) > 0

    def test_validate_orders_duplicate_ids(self, sample_config, sample_orders):
        """Test detection of duplicate order IDs."""
        sample_orders.loc[1, "order_id"] = "O001"  # Create duplicate
        validator = DataValidator(sample_config)
        is_valid, report = validator.validate_orders(sample_orders)

        assert len(validator.issues["warning"]) > 0
        warning_found = any("Duplicate" in issue for issue in validator.issues["warning"])
        assert warning_found

    def test_validate_orders_invalid_discount(self, sample_config, sample_orders):
        """Test detection of invalid discount percentages."""
        sample_orders.loc[0, "discount_pct"] = 1.5  # Invalid discount > 1
        validator = DataValidator(sample_config)
        is_valid, report = validator.validate_orders(sample_orders)

        assert len(validator.issues["warning"]) > 0

    def test_validate_customers_valid(self, sample_config, sample_customers):
        """Test validation of valid customers data."""
        validator = DataValidator(sample_config)
        is_valid, report = validator.validate_customers(sample_customers)

        assert is_valid is True
        assert report["dataset"] == "customers"
        assert report["rows"] == 3

    def test_validate_customers_duplicate_ids(self, sample_config, sample_customers):
        """Test detection of duplicate customer IDs."""
        sample_customers.loc[1, "customer_id"] = "C001"  # Create duplicate
        validator = DataValidator(sample_config)
        is_valid, report = validator.validate_customers(sample_customers)

        assert is_valid is False
        assert len(validator.issues["critical"]) > 0

    def test_referential_integrity_valid(self, sample_config, sample_orders, sample_customers):
        """Test referential integrity check with valid data."""
        validator = DataValidator(sample_config)
        report = validator.referential_integrity_check(sample_orders, sample_customers)

        assert bool(report["result"]) is True

    def test_referential_integrity_orphaned_records(self, sample_config, sample_orders, sample_customers):
        """Test detection of orphaned records."""
        sample_orders.loc[0, "customer_id"] = "C999"  # Non-existent customer
        validator = DataValidator(sample_config)
        report = validator.referential_integrity_check(sample_orders, sample_customers)

        assert bool(report["result"]) is False
        assert len(validator.issues["critical"]) > 0


class TestDataProfiler:
    """Test data profiling functionality."""

    @pytest.fixture
    def sample_orders(self):
        """Create sample orders dataframe."""
        return pd.DataFrame({
            "order_id": [f"O{i:03d}" for i in range(100)],
            "order_date": pd.date_range("2025-01-01", periods=100),
            "customer_id": [f"C{i%20:03d}" for i in range(100)],
            "product_id": [f"P{i%10:03d}" for i in range(100)],
            "category": np.random.choice(["Electronics", "Beauty", "Home & Office"], 100),
            "quantity": np.random.randint(1, 5, 100),
            "unit_price": np.random.randint(500, 5000, 100),
            "discount_pct": np.random.uniform(0, 0.3, 100),
            "channel": np.random.choice(["App", "Web", "Marketplace"], 100),
            "ab_group": np.random.choice(["control", "variant"], 100),
            "returned": np.random.choice([0, 1], 100),
            "gross_revenue": np.random.randint(500, 10000, 100),
            "net_revenue": np.random.uniform(400, 9000, 100),
            "shipping_days": np.random.randint(1, 10, 100),
            "rating": np.random.randint(1, 6, 100),
        })

    def test_profile_orders_numeric(self, sample_orders):
        """Test numeric column profiling."""
        profile = DataProfiler.profile_orders(sample_orders)

        assert profile["dataset"] == "orders"
        assert profile["dimensions"]["rows"] == 100
        assert len(profile["numeric_columns"]) > 0
        
        # Check specific metric
        assert "quantity" in profile["numeric_columns"]
        assert profile["numeric_columns"]["quantity"]["mean"] > 0

    def test_profile_orders_categorical(self, sample_orders):
        """Test categorical column profiling."""
        profile = DataProfiler.profile_orders(sample_orders)

        assert len(profile["categorical_columns"]) > 0
        assert "channel" in profile["categorical_columns"]

    def test_profile_quality_metrics(self, sample_orders):
        """Test quality metric calculation."""
        profile = DataProfiler.profile_orders(sample_orders)

        assert "quality_metrics" in profile
        assert profile["quality_metrics"]["null_values_pct"] >= 0
        assert profile["quality_metrics"]["returned_orders_pct"] >= 0


class TestStatisticalAnalysis:
    """Test statistical analysis functions."""

    def test_correlation_matrix(self):
        """Test correlation matrix computation."""
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.randn(100),
        })

        corr = df.corr()

        # Check symmetry
        assert (corr == corr.T).all().all()

        # Check diagonal is 1
        assert np.allclose(np.diag(corr), 1.0)

    def test_outlier_detection(self):
        """Test IQR-based outlier detection."""
        data = pd.Series(list(range(100)) + [500])  # One extreme outlier

        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((data < lower_bound) | (data > upper_bound)).sum()

        assert outliers >= 1  # Should detect the 500 as outlier


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
