"""
A/B Testing Analysis
====================

Conducts statistical hypothesis tests on A/B test results:
1. Two-sample t-test on revenue (Welch's t-test, unequal variances)
2. Chi-square test on return rates
3. Computes effect sizes (Cohen's d)
4. Computes confidence intervals for estimates
5. Applies multiple comparison correction (Bonferroni)

Key Assumptions:
- Samples are independent (no customer appears in both groups)
- Revenue data can have outliers (t-test is robust for large samples)
- Return rate is binary (0/1)

Outputs:
- data/processed/ab_test_results.csv: Statistical test results and estimates
- Console report with interpretation

Exit codes:
    0 = Success
    1 = Warnings (test completed but with caveats)
    2 = Failure (insufficient data or invalid test)
"""

import sys
import logging
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
        logging.FileHandler(LOG_DIR / "ab_testing.log"),
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
        return {"analysis": {"ab_test": {"alpha": 0.05, "correction": "bonferroni"}}}


def load_orders(raw_dir: Path) -> pd.DataFrame:
    """Load orders data with error handling."""
    logger.info("Loading orders data...")
    
    try:
        orders_file = raw_dir / "orders.csv"
        if not orders_file.exists():
            logger.error(f"Orders file not found: {orders_file}")
            sys.exit(2)
        
        df = pd.read_csv(orders_file)
        logger.info(f"Loaded {len(df)} orders")
        
        # Validate required columns
        required_cols = ['ab_group', 'net_revenue', 'returned']
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


def validate_ab_groups(df: pd.DataFrame) -> bool:
    """Validate A/B group assignments."""
    logger.info("Validating A/B groups...")
    
    # Check groups exist
    unique_groups = df['ab_group'].unique()
    logger.info(f"Found groups: {unique_groups}")
    
    if len(unique_groups) != 2 or not all(g in ['control', 'variant'] for g in unique_groups):
        logger.error(f"Invalid ab_group values: {unique_groups}")
        return False
    
    # Check group sizes
    group_counts = df['ab_group'].value_counts()
    logger.info(f"Control group: {group_counts['control']} orders")
    logger.info(f"Variant group: {group_counts['variant']} orders")
    
    # Warn if severely imbalanced
    min_size = min(group_counts)
    if min_size < 100:
        logger.warning(f"Small sample size detected: {min_size} orders in smallest group")
        logger.warning("Test results may have low statistical power")
    
    # Check for imbalance
    ratio = max(group_counts) / min(group_counts)
    if ratio > 1.2:
        logger.warning(f"Imbalanced groups: {ratio:.2f}x size difference")
    
    return True


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Compute Cohen's d effect size for two samples."""
    nx = len(x)
    ny = len(y)
    
    dof = nx + ny - 2
    return (x.mean() - y.mean()) / np.sqrt(((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / dof)


def compute_ci(mean: float, std: float, n: int, ci: float = 0.95) -> tuple:
    """Compute confidence interval for mean."""
    alpha = 1 - ci
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    se = std / np.sqrt(n)
    margin = t_crit * se
    return (mean - margin, mean + margin)


def test_revenue(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Test for difference in mean revenue between groups."""
    logger.info("Testing revenue difference...")
    
    control = df[df['ab_group'] == 'control']['net_revenue']
    variant = df[df['ab_group'] == 'variant']['net_revenue']
    
    # Remove outliers using IQR method for summary stats
    def remove_outliers(series):
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        return series[(series >= Q1 - 1.5 * IQR) & (series <= Q3 + 1.5 * IQR)]
    
    control_clean = remove_outliers(control)
    variant_clean = remove_outliers(variant)
    
    # Perform t-test (Welch's t-test, unequal variances)
    t_stat, p_value = stats.ttest_ind(control, variant, equal_var=False)
    
    # Compute effect size
    effect_size = cohens_d(control, variant)
    
    # Compute confidence intervals
    control_ci = compute_ci(control.mean(), control.std(), len(control))
    variant_ci = compute_ci(variant.mean(), variant.std(), len(variant))
    
    result = {
        'test': 'Revenue (Welch\'s t-test)',
        'control_mean': round(control.mean(), 2),
        'control_median': round(control.median(), 2),
        'control_n': len(control),
        'control_std': round(control.std(), 2),
        'control_ci_lower': round(control_ci[0], 2),
        'control_ci_upper': round(control_ci[1], 2),
        'variant_mean': round(variant.mean(), 2),
        'variant_median': round(variant.median(), 2),
        'variant_n': len(variant),
        'variant_std': round(variant.std(), 2),
        'variant_ci_lower': round(variant_ci[0], 2),
        'variant_ci_upper': round(variant_ci[1], 2),
        'difference': round(variant.mean() - control.mean(), 2),
        'pct_difference': round((variant.mean() - control.mean()) / control.mean() * 100, 2),
        't_statistic': round(t_stat, 4),
        'p_value': round(p_value, 5),
        'significant_at_alpha': 'Yes' if p_value < alpha else 'No',
        'cohens_d': round(effect_size, 4),
        'effect_size': interpret_cohens_d(effect_size),
        'samples_removed_as_outliers': len(control) - len(control_clean) + len(variant) - len(variant_clean),
    }
    
    logger.info(f"  Control mean: ${result['control_mean']}")
    logger.info(f"  Variant mean: ${result['variant_mean']}")
    logger.info(f"  Difference: ${result['difference']} ({result['pct_difference']}%)")
    logger.info(f"  p-value: {result['p_value']}")
    logger.info(f"  Cohen's d: {result['cohens_d']} ({result['effect_size']})")
    
    return result


def test_return_rate(df: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Test for difference in return rate between groups."""
    logger.info("Testing return rate difference...")
    
    control = df[df['ab_group'] == 'control']['returned']
    variant = df[df['ab_group'] == 'variant']['returned']
    
    # Create contingency table
    contingency = pd.crosstab(df['ab_group'], df['returned'])
    
    # Perform chi-square test
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Compute proportions and CIs
    control_rate = control.mean()
    variant_rate = variant.mean()
    
    # Wilson score CI for proportions (more accurate than normal approximation)
    def wilson_ci(successes, n, ci=0.95):
        z = stats.norm.ppf(1 - (1 - ci) / 2)
        p_hat = successes / n
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        adjustment = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
        return (center - adjustment, center + adjustment)
    
    control_ci = wilson_ci(control.sum(), len(control))
    variant_ci = wilson_ci(variant.sum(), len(variant))
    
    result = {
        'test': 'Return Rate (Chi-square test)',
        'control_returns': int(control.sum()),
        'control_n': len(control),
        'control_rate': round(control_rate, 4),
        'control_pct': round(control_rate * 100, 2),
        'control_ci_lower': round(control_ci[0] * 100, 2),
        'control_ci_upper': round(control_ci[1] * 100, 2),
        'variant_returns': int(variant.sum()),
        'variant_n': len(variant),
        'variant_rate': round(variant_rate, 4),
        'variant_pct': round(variant_rate * 100, 2),
        'variant_ci_lower': round(variant_ci[0] * 100, 2),
        'variant_ci_upper': round(variant_ci[1] * 100, 2),
        'absolute_difference_pct': round((variant_rate - control_rate) * 100, 2),
        'relative_lift': round((variant_rate - control_rate) / control_rate * 100, 2) if control_rate > 0 else 0,
        'chi2_statistic': round(chi2_stat, 4),
        'p_value': round(p_value, 5),
        'significant_at_alpha': 'Yes' if p_value < alpha else 'No',
    }
    
    logger.info(f"  Control return rate: {result['control_pct']}%")
    logger.info(f"  Variant return rate: {result['variant_pct']}%")
    logger.info(f"  Absolute difference: {result['absolute_difference_pct']}%")
    logger.info(f"  p-value: {result['p_value']}")
    
    return result


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def apply_multiple_comparison_correction(results: list, correction: str = "bonferroni", original_alpha: float = 0.05) -> list:
    """Apply multiple comparison correction to p-values."""
    logger.info(f"Applying {correction} correction for {len(results)} tests...")
    
    if correction == "bonferroni":
        adjusted_alpha = original_alpha / len(results)
        logger.info(f"Bonferroni adjusted alpha: {original_alpha} / {len(results)} = {adjusted_alpha:.5f}")
        
        for result in results:
            if 'p_value' in result:
                result['adjusted_alpha'] = round(adjusted_alpha, 5)
                result['significant_at_adjusted_alpha'] = 'Yes' if result['p_value'] < adjusted_alpha else 'No'
    
    return results


def export_results(results: list, processed_dir: Path):
    """Export test results to CSV."""
    logger.info("Exporting test results...")
    
    try:
        output_file = processed_dir / "ab_test_results.csv"
        
        # Flatten results for CSV export
        flattened = []
        for result in results:
            flattened.append(result)
        
        # Convert to DataFrame and export
        df_results = pd.DataFrame(flattened)
        df_results.to_csv(output_file, index=False)
        
        logger.info(f"Exported A/B test results to {output_file}")
    
    except Exception as e:
        logger.error(f"Failed to export results: {e}")
        sys.exit(2)


def main():
    """Run complete A/B testing pipeline."""
    logger.info("=" * 80)
    logger.info("A/B TESTING ANALYSIS PIPELINE")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(CONFIG_FILE)
    raw_dir = ROOT / config['data']['raw_dir']
    processed_dir = ROOT / config['data']['processed_dir']
    processed_dir.mkdir(exist_ok=True)
    
    alpha = config['analysis']['ab_test']['alpha']
    correction = config['analysis']['ab_test'].get('correction', 'bonferroni')
    
    # Load and validate data
    orders = load_orders(raw_dir)
    if not validate_ab_groups(orders):
        sys.exit(2)
    
    # Perform tests
    results = []
    results.append(test_revenue(orders, alpha))
    results.append(test_return_rate(orders, alpha))
    
    # Apply multiple comparison correction
    results = apply_multiple_comparison_correction(results, correction, alpha)
    
    # Export
    export_results(results, processed_dir)
    
    # Print summary
    print("\n" + "=" * 80)
    print("A/B TEST RESULTS SUMMARY")
    print("=" * 80)
    for result in results:
        print(f"\nTest: {result['test']}")
        if result['test'].startswith('Revenue'):
            print(f"  Control Mean: ${result['control_mean']:,.2f} ({result['control_n']} orders)")
            print(f"  Variant Mean: ${result['variant_mean']:,.2f} ({result['variant_n']} orders)")
            print(f"  Difference: ${result['difference']:,.2f} ({result['pct_difference']}%)")
            print(f"  Cohen's d: {result['cohens_d']} ({result['effect_size']})")
        else:
            print(f"  Control Rate: {result['control_pct']}% ({result['control_returns']}/{result['control_n']})")
            print(f"  Variant Rate: {result['variant_pct']}% ({result['variant_returns']}/{result['variant_n']})")
            print(f"  Difference: {result['absolute_difference_pct']}% points")
        
        print(f"  p-value: {result['p_value']}")
        print(f"  Significant (alpha={alpha})?: {result['significant_at_alpha']}")
        if 'significant_at_adjusted_alpha' in result:
            print(f"  Significant (Bonferroni corrected alpha={result['adjusted_alpha']})?: {result['significant_at_adjusted_alpha']}")
    print("=" * 80 + "\n")
    
    logger.info("A/B testing pipeline completed successfully")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
